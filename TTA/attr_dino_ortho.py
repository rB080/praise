# attr_dinotxt_ortho_i2i.py
# ACS-Ortho Test-Time Adaptation for I2I ReID using Dino.txt backbone.
#
# Assumptions:
#   from backbones.dino.dinotxt import *
#   model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l()
#   model.encode_image(images) -> (B,d)
#   model.encode_text(tokens)  -> (B,d)
#
# Tokenizer:
#   import open_clip
#   tokenizer = open_clip.get_tokenizer("ViT-B-16")

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import open_clip
from backbones.dino.dinotxt import dinov2_vitl14_reg4_dinotxt_tet1280d20h24l


# ============================================================
# 1) Atomic attribute prompts (ACS bank)
# ============================================================
_prompts = {
    "age": ["young", "adult", "elderly"],
    "gender": ["man", "woman"],
    "hair": ["short hair", "long hair", "bald head"],
    "ubcl": ["long sleeved clothing", "short sleeved clothing"],
    "ubcc": ["black top", "blue top", "brown top", "green top", "gray top",
             "orange top", "pink top", "purple top", "red top", "white top", "yellow top"],
    "lbcl": ["long pants", "shorts"],
    "lbcc": ["black pants", "blue pants", "brown pants", "green pants", "gray pants",
             "orange pants", "pink pants", "purple pants", "red pants", "white pants", "yellow pants"],
    "lbct": ["trousers or shorts", "dress skirt"],
    "accessories": ["a backpack", "a bag", "glasses", "a hat"],
}

_prompt_templates = {
    "age": "A photo of a {} person.",
    "gender": "A photo of a {}.",
    "hair": "A photo of a person with {}.",
    "ubcl": "A photo of a person wearing {}.",
    "ubcc": "A photo of a person wearing {}.",
    "lbcl": "A photo of a person wearing {}.",
    "lbcc": "A photo of a person wearing {}.",
    "lbct": "A photo of a person wearing {}.",
    "accessories": "A photo of a person with {}.",
}

_prompt_set: Dict[str, List[str]] = {k: [_prompt_templates[k].format(p) for p in _prompts[k]] for k in _prompts.keys()}
_prompt_set["accessories"] = _prompt_set["accessories"] + ["A photo of a person without any accessories."]


# ============================================================
# 2) Utilities
# ============================================================
def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


def topk_rowwise(p: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    kk = min(k, p.shape[1])
    return torch.topk(p, k=kk, dim=1, largest=True)


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, min=eps)
    return -(p * torch.log(p)).sum(dim=1).mean()


def disable_dropout(module: nn.Module) -> int:
    dropout_types = (
        nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
        nn.AlphaDropout, nn.FeatureAlphaDropout
    )
    cnt = 0
    for m in module.modules():
        if isinstance(m, dropout_types):
            m.eval()
            m.training = False
            cnt += 1
    return cnt


def _name_suggests_text(name: str) -> bool:
    n = name.lower()
    return ("text" in n) or ("txt" in n) or ("lang" in n)


def _name_suggests_image(name: str) -> bool:
    n = name.lower()
    return (("image" in n) or ("img" in n) or ("visual" in n) or ("vision" in n) or ("patch" in n) or ("backbone" in n)) and (not _name_suggests_text(n))


# ============================================================
# 3) Dino.txt ACS-Ortho I2I TTA
# ============================================================
class ATTR_TTA(nn.Module):
    """
    I2I ACS-Ortho TTA for Dino.txt.

    Public API (like your other i2i adapters):
      init_model()
      tokenize_text()
      precompute_attr_bank()
      initialize(gallery_feats)
      reset()
      forward(imgs, G_feats, hybrid_eval=...)
    """

    def __init__(
        self,
        steps: int = 1,
        topk: int = 5,
        id_temp: float = 0.6,
        device: str = "cuda:0",
        lr: float = 5e-4,
        episodic: bool = True,

        # ACS
        use_acs: bool = True,
        acs_groups: Optional[List[str]] = None,
        acs_kmax: int = 10,
        acs_alpha: float = 0.4,
        acs_conf_thresh: float = 0.1,
        acs_use_residual: bool = False,
        neutral_text: str = "A photo of a person.",
        zscore_text_logits: bool = False,

        # tokenizer
        tokenizer_name: str = "ViT-B-16",

        bank_chunksize: int = 64,
    ):
        super().__init__()
        self.episodic = bool(episodic)
        self.steps = int(steps)
        self.topk = int(topk)
        self.id_temp = float(id_temp)
        self.device = str(device)
        self.lr = float(lr)

        self.use_acs = bool(use_acs)
        self.acs_groups = acs_groups if acs_groups is not None else ["gender", "hair", "ubcc", "lbcc", "accessories"]
        self.acs_kmax = int(acs_kmax)
        self.acs_alpha = float(acs_alpha)
        self.acs_conf_thresh = float(acs_conf_thresh)
        self.acs_use_residual = bool(acs_use_residual)
        self.neutral_text = str(neutral_text)
        self.zscore_text_logits = bool(zscore_text_logits)

        self.tokenizer_name = str(tokenizer_name)
        self.bank_chunksize = int(bank_chunksize)
        self.tokenizer = open_clip.get_tokenizer(self.tokenizer_name)

        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model_state = None
        self.optimizer_state = None

        self.attr_prompt_feats: Dict[str, torch.Tensor] = {}
        self.attr_dirs: Dict[str, torch.Tensor] = {}
        self.attr_strings: Dict[str, List[str]] = {}
        self.neutral_text_feat: Optional[torch.Tensor] = None

    def init_model(self):
        self.model = dinov2_vitl14_reg4_dinotxt_tet1280d20h24l(load_pretrained=True).to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        n_drop = disable_dropout(self.model)
        if n_drop:
            print(f"[Dino.txt I2I] Disabled dropout in {n_drop} modules.")

    def tokenize_text(self, texts: List[str]) -> torch.Tensor:
        return self.tokenizer(texts)

    def collect_params(self) -> Tuple[List[nn.Parameter], List[str]]:
        assert self.model is not None, "Call init_model() first."
        self.model.requires_grad_(False)

        params: List[nn.Parameter] = []
        names: List[str] = []

        # Heuristic: include norms that look like image/visual modules; exclude text
        candidates = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                if _name_suggests_text(nm):
                    continue
                if _name_suggests_image(nm):
                    candidates.append((nm, m))

        # Fallback: if can't separate branches, adapt all norms except explicit "text"
        if len(candidates) == 0:
            for nm, m in self.model.named_modules():
                if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)) and (not _name_suggests_text(nm)):
                    candidates.append((nm, m))

        for nm, m in candidates:
            for np, p in m.named_parameters(recurse=False):
                if np in ["weight", "bias"]:
                    p.requires_grad_(True)
                    params.append(p)
                    names.append(f"{nm}.{np}")

            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

        return params, names

    def copy_model_and_optimizer(self):
        assert self.model is not None and self.optimizer is not None
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def load_model_and_optimizer(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("reset called before initialize()")
        assert self.model is not None and self.optimizer is not None
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def initialize(self, gallery_feats: Optional[torch.Tensor] = None, **kwargs):
        assert self.model is not None, "Call init_model() first."
        params, _ = self.collect_params()
        print(f"[Dino.txt I2I] Optimizing {len(params)} norm affine params.")
        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        self.copy_model_and_optimizer()

        if self.use_acs and len(self.attr_dirs) == 0:
            self.precompute_attr_bank(chunksize=self.bank_chunksize)

    def reset(self):
        self.load_model_and_optimizer()

    @torch.no_grad()
    def precompute_attr_bank(self, prompt_set: Dict[str, List[str]] = _prompt_set, chunksize: int = 64, verbose: bool = True):
        assert self.model is not None, "Call init_model() first."
        self.model.eval()
        device = torch.device(self.device)

        t0 = self.tokenize_text([self.neutral_text]).to(device)
        tf0 = self.model.encode_text(t0)
        tf0 = l2_normalize(tf0.to(torch.float32)).squeeze(0)
        self.neutral_text_feat = tf0

        for group in self.acs_groups:
            prompts = prompt_set[group]
            feats = []
            it = range(0, len(prompts), chunksize)
            if verbose:
                it = tqdm(it, desc=f"[Dino.txt] Text bank: {group}")
            for i in it:
                chunk = prompts[i:i + chunksize]
                ti = self.tokenize_text(chunk).to(device)
                tf = self.model.encode_text(ti)
                feats.append(tf.detach().to(torch.float32))

            tf = torch.cat(feats, dim=0)
            tf = l2_normalize(tf)
            dirs = l2_normalize(tf - tf0.unsqueeze(0))

            self.attr_prompt_feats[group] = tf
            self.attr_dirs[group] = dirs
            self.attr_strings[group] = prompts

        if verbose:
            print("[Dino.txt I2I] Built ACS groups:", list(self.attr_dirs.keys()))

    @torch.no_grad()
    def predict_attrs_for_batch(self, x_norm: torch.Tensor) -> List[List[Tuple[str, int, float]]]:
        B, _ = x_norm.shape
        selected: List[List[Tuple[str, int, float]]] = [[] for _ in range(B)]

        for group in self.acs_groups:
            tf = self.attr_prompt_feats[group].to(x_norm.device)
            logits = x_norm @ tf.t()

            if self.zscore_text_logits:
                mu = logits.mean(dim=0, keepdim=True)
                sd = logits.std(dim=0, keepdim=True) + 1e-6
                logits = (logits - mu) / sd

            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
            for b in range(B):
                c = float(conf[b].item())
                if c >= self.acs_conf_thresh:
                    selected[b].append((group, int(idx[b].item()), c))

        for b in range(B):
            selected[b] = sorted(selected[b], key=lambda t: t[2], reverse=True)[: self.acs_kmax]
        return selected

    def acs_similarity(self, x_norm: torch.Tensor, G_norm: torch.Tensor, selected: List[List[Tuple[str, int, float]]], eps: float = 1e-8) -> torch.Tensor:
        device = x_norm.device
        dtype = x_norm.dtype
        B, _ = x_norm.shape
        Ng = G_norm.shape[0]
        out = torch.empty((B, Ng), device=device, dtype=dtype)

        for b in range(B):
            items = selected[b]
            S_raw = (x_norm[b:b+1] @ G_norm.t())

            if len(items) == 0:
                out[b] = S_raw.squeeze(0)
                continue

            dirs, weights = [], []
            for (group, idx, conf) in items:
                dirs.append(self.attr_dirs[group][idx].to(device=device, dtype=dtype))
                weights.append(conf)

            U = torch.stack(dirs, dim=1)
            w = torch.tensor(weights, device=device, dtype=dtype)
            w = w / (w.sum() + eps)
            U = U * torch.sqrt(w).unsqueeze(0)

            Q, _ = torch.linalg.qr(U, mode="reduced")

            cq = (x_norm[b:b+1] @ Q)
            cG = (G_norm @ Q)
            cq = l2_normalize(cq, eps=eps)
            cG = l2_normalize(cG, eps=eps)
            S_sub = (cq @ cG.t())

            if self.acs_use_residual:
                S = S_raw - ((x_norm[b:b+1] @ Q) @ (G_norm @ Q).t())
            else:
                S = S_raw + self.acs_alpha * S_sub

            out[b] = S.squeeze(0)

        return out

    def encode_images(self, x: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        return self.model.encode_image(x)

    def forward(self, x: torch.Tensor, G: torch.Tensor, **kwargs):
        if self.episodic:
            self.reset()

        out = None
        for _ in range(self.steps):
            out = forward_and_adapt(self, x, G, **kwargs)
        return out


@torch.enable_grad()
def forward_and_adapt(adapter: ATTR_TTA, x: torch.Tensor, G: torch.Tensor, **kwargs):
    assert adapter.model is not None and adapter.optimizer is not None, "Call init_model()+initialize() first."

    adapter.model.train()

    x_feat = adapter.encode_images(x.to(adapter.device))
    G = G.to(x_feat.device)

    x_norm = l2_normalize(x_feat.to(torch.float32))
    G_norm = l2_normalize(G.to(torch.float32))

    if adapter.use_acs:
        if len(adapter.attr_dirs) == 0:
            raise RuntimeError("ACS enabled but prompt bank missing. Call precompute_attr_bank() once.")
        selected = adapter.predict_attrs_for_batch(x_norm.detach())
        S = adapter.acs_similarity(x_norm, G_norm, selected)
    else:
        S = x_norm @ G_norm.t()

    p = torch.softmax(S / adapter.id_temp, dim=1)
    vals, _ = topk_rowwise(p, adapter.topk)
    vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-12)
    loss = entropy_from_probs(vals)

    loss.backward()
    adapter.optimizer.step()
    adapter.optimizer.zero_grad(set_to_none=True)

    if kwargs.get("hybrid_eval", False):
        return p.detach(), x_feat.detach(), float(loss.item())
    return x_feat.detach(), float(loss.item())
