# attr_clipreid_ortho_t2i_refresh.py
# T2I ATTR_TTA with ACS (orthogonal subspace) for CLIPReID backbone.
# Text encoder: CLIP model used by CLIPReID (self.clip.encode_text)
# Gallery: precomputed image embeddings (Ng,d) from CLIPReID image encoder proj[:,0]
# Adaptation: LayerNorm affine params in CLIP *text* branch only (training-time prompts drift -> optional refresh each step)

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# -----------------------------
# Project-specific imports (CLIPReID)
# -----------------------------
from backbones.CLIPReID.make_model_clipreid import make_model, load_clip_to_cpu
from backbones.CLIPReID.config import cfg
from backbones.CLIPReID.clip.clip import tokenize as clip_tokenize

from utils.utils import Test_parse_args


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


def disable_dropout(module: nn.Module) -> None:
    dropout_types = (
        nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
        nn.AlphaDropout, nn.FeatureAlphaDropout
    )
    for m in module.modules():
        if isinstance(m, dropout_types):
            m.eval()
            m.training = False


# ============================================================
# 3) T2I ATTR_TTA with ACS + bank refresh (CLIPReID)
# ============================================================
class ATTR_TTA(nn.Module):
    def __init__(
        self,
        steps: int = 1,
        topk: int = 5,
        id_temp: float = 0.6,
        device: str = "cuda:0",
        lr: float = 5e-4,
        episodic: bool = True,

        # -------- ACS knobs --------
        use_acs: bool = True,
        acs_groups: Optional[List[str]] = None,
        acs_kmax: int = 8,
        acs_alpha: float = 0.1,
        acs_conf_thresh: float = 0.0,
        acs_use_residual: bool = False,
        neutral_text: str = "A photo of a person.",

        # ---- crucial for T2I if adapting text encoder ----
        refresh_attr_bank_each_step: bool = False,
        refresh_chunksize: int = 64,

        # Optional: zscore prompt logits for selection
        zscore_text_logits: bool = False,

        # CLIPReID config/model
        cfg_path: str = "backbones/CLIPReID/configs/person/vit_clipreid_market.yml",
        num_class: int = 1041,
        camera_num: int = 6,
        view_num: int = 1,
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

        self.refresh_attr_bank_each_step = bool(refresh_attr_bank_each_step)
        self.refresh_chunksize = int(refresh_chunksize)

        self.cfg_path = str(cfg_path)
        self.num_class = int(num_class)
        self.camera_num = int(camera_num)
        self.view_num = int(view_num)

        self.args = Test_parse_args(notebook=True)

        self.model: Optional[nn.Module] = None   # CLIPReID reid model (not adapted here)
        self.clip: Optional[nn.Module] = None    # CLIP model (text branch adapted)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.clip_state = None
        self.optimizer_state = None

        self.attr_prompt_feats: Dict[str, torch.Tensor] = {}
        self.attr_dirs: Dict[str, torch.Tensor] = {}
        self.attr_strings: Dict[str, List[str]] = {}
        self.neutral_text_feat: Optional[torch.Tensor] = None

    # -----------------------------
    # Init
    # -----------------------------
    def _init_clip_model(self) -> nn.Module:
        h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(cfg.MODEL.NAME, h_resolution, w_resolution, vision_stride_size)
        clip_model.eval()
        return clip_model.to(self.device)

    def init_model(self, model_path: Optional[str] = None):
        cfg.merge_from_file(self.cfg_path)
        cfg.freeze()

        model = make_model(cfg, num_class=self.num_class, camera_num=self.camera_num, view_num=self.view_num)
        if model_path is not None:
            model.load_param_finetune(model_path)
        else:
            model.load_param(self.args.model_path)
        self.model = model.to(self.device)

        self.clip = self._init_clip_model()

        # freeze CLIPReID model; we adapt CLIP text LN only
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        disable_dropout(self.model)
        disable_dropout(self.clip)

    # -----------------------------
    # Tokenization
    # -----------------------------
    def tokenize(self, texts: List[str]) -> torch.Tensor:
        return clip_tokenize(texts, context_length=77, truncate=True)

    # -----------------------------
    # Param collection (CLIP TEXT LayerNorm only)
    # -----------------------------
    def collect_params(self) -> Tuple[List[nn.Parameter], List[str]]:
        assert self.clip is not None, "Call init_model() first."

        self.clip.requires_grad_(False)

        params: List[nn.Parameter] = []
        names: List[str] = []
        for nm, m in self.clip.named_modules():
            if isinstance(m, nn.LayerNorm):
                if nm.startswith("visual"):
                    continue
                for np, p in m.named_parameters(recurse=False):
                    if np in ["weight", "bias"]:
                        p.requires_grad_(True)
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def copy_model_and_optimizer(self):
        assert self.clip is not None and self.optimizer is not None
        self.clip_state = deepcopy(self.clip.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def load_model_and_optimizer(self):
        if self.clip_state is None or self.optimizer_state is None:
            raise RuntimeError("reset called before initialize()")
        assert self.clip is not None and self.optimizer is not None
        self.clip.load_state_dict(self.clip_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def initialize(self, gallery_feats: torch.Tensor):
        params, _ = self.collect_params()
        print(f"Optimizing {len(params)} params (CLIP TEXT LayerNorm affine only).")
        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, weight_decay=1e-4)
        self.copy_model_and_optimizer()

        if self.use_acs and len(self.attr_dirs) == 0:
            self.precompute_attr_bank()

    def reset(self):
        self.load_model_and_optimizer()

    # ============================================================
    # ACS bank builders (using CURRENT CLIP text encoder)
    # ============================================================
    @torch.no_grad()
    def precompute_attr_bank(self, prompt_set: Dict[str, List[str]] = _prompt_set, chunksize: int = 64, verbose: bool = True):
        self._build_attr_bank(prompt_set=prompt_set, chunksize=chunksize, show_pbar=verbose)

    @torch.no_grad()
    def refresh_attr_bank(self, prompt_set: Dict[str, List[str]] = _prompt_set, chunksize: Optional[int] = None):
        if chunksize is None:
            chunksize = self.refresh_chunksize
        self._build_attr_bank(prompt_set=prompt_set, chunksize=chunksize, show_pbar=False)

    @torch.no_grad()
    def _build_attr_bank(self, prompt_set: Dict[str, List[str]], chunksize: int, show_pbar: bool):
        assert self.clip is not None, "Call init_model() first."
        self.clip.eval()
        device = torch.device(self.device)

        t0 = self.tokenize([self.neutral_text]).to(device)
        tf0 = self.clip.encode_text(t0)
        tf0 = l2_normalize(tf0.to(torch.float32)).squeeze(0)
        self.neutral_text_feat = tf0

        for group in self.acs_groups:
            prompts = prompt_set[group]
            feats = []
            it = range(0, len(prompts), chunksize)
            if show_pbar:
                it = tqdm(it, desc=f"Text bank: {group}")
            for i in it:
                chunk = prompts[i:i + chunksize]
                ti = self.tokenize(chunk).to(device)
                tf = self.clip.encode_text(ti)
                feats.append(tf.detach().to(torch.float32))

            tf = torch.cat(feats, dim=0)
            tf = l2_normalize(tf)
            dirs = l2_normalize(tf - tf0.unsqueeze(0))

            self.attr_prompt_feats[group] = tf
            self.attr_dirs[group] = dirs
            self.attr_strings[group] = prompts

        if show_pbar:
            print("Built ACS attribute bank groups:", list(self.attr_dirs.keys()))

    # ============================================================
    # ACS selection (on text queries)
    # ============================================================
    def predict_attrs_for_batch(self, q_norm: torch.Tensor) -> List[List[Tuple[str, int, float]]]:
        B, _ = q_norm.shape
        selected: List[List[Tuple[str, int, float]]] = [[] for _ in range(B)]

        for group in self.acs_groups:
            tf = self.attr_prompt_feats[group].to(q_norm.device)
            logits = q_norm @ tf.t()

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
        return selected

    # ============================================================
    # ACS similarity (text query vs image gallery)
    # ============================================================
    def acs_similarity(self, q_norm: torch.Tensor, G_norm: torch.Tensor, selected: List[List[Tuple[str, int, float]]], eps: float = 1e-8) -> torch.Tensor:
        device = q_norm.device
        dtype = q_norm.dtype
        B, _ = q_norm.shape
        Ng = G_norm.shape[0]
        out = torch.empty((B, Ng), device=device, dtype=dtype)

        for b in range(B):
            items = sorted(selected[b], key=lambda t: t[2], reverse=True)[: self.acs_kmax]
            S_raw = (q_norm[b:b+1] @ G_norm.t())

            if len(items) == 0:
                out[b] = S_raw.squeeze(0)
                continue

            dirs, weights = [], []
            for (group, idx, conf) in items:
                dirs.append(self.attr_dirs[group][idx].to(device=device, dtype=dtype))
                weights.append(conf)

            U = torch.stack(dirs, dim=1)  # (d,k)
            w = torch.tensor(weights, device=device, dtype=dtype)
            w = w / (w.sum() + eps)
            U = U * torch.sqrt(w).unsqueeze(0)

            Q, _ = torch.linalg.qr(U, mode="reduced")

            cq = (q_norm[b:b+1] @ Q)
            cG = (G_norm @ Q)
            cq = l2_normalize(cq, eps=eps)
            cG = l2_normalize(cG, eps=eps)
            S_sub = (cq @ cG.t())

            if self.acs_use_residual:
                S = S_raw - ((q_norm[b:b+1] @ Q) @ (G_norm @ Q).t())
            else:
                S = S_raw + self.acs_alpha * S_sub

            out[b] = S.squeeze(0)

        return out

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, captions: List[str], G: torch.Tensor, **kwargs):
        if self.episodic:
            self.reset()

        out = None
        for _ in range(self.steps):
            out = forward_and_adapt(self, captions, G, **kwargs)
        return out


@torch.enable_grad()
def forward_and_adapt(adapter: ATTR_TTA, captions: List[str], G: torch.Tensor, **kwargs):
    assert adapter.clip is not None and adapter.optimizer is not None, "Call init_model()+initialize() first."
    adapter.clip.train()
    device = torch.device(adapter.device)

    ti = adapter.tokenize(list(captions)).to(device)
    q_feat = adapter.clip.encode_text(ti)

    G = G.to(q_feat.device)
    q_norm = l2_normalize(q_feat.to(torch.float32))
    G_norm = l2_normalize(G.to(torch.float32))

    if adapter.use_acs and adapter.refresh_attr_bank_each_step:
        adapter.refresh_attr_bank()

    if adapter.use_acs:
        selected = adapter.predict_attrs_for_batch(q_norm)
        S = adapter.acs_similarity(q_norm, G_norm, selected)
    else:
        S = q_norm @ G_norm.t()

    p = torch.softmax(S / adapter.id_temp, dim=1)
    vals, _ = topk_rowwise(p, adapter.topk)
    vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-12)
    loss = entropy_from_probs(vals)

    loss.backward()
    adapter.optimizer.step()
    adapter.optimizer.zero_grad(set_to_none=True)

    if kwargs.get("hybrid_eval", False):
        return p.detach(), q_feat.detach(), float(loss.item())
    return q_feat.detach(), float(loss.item())
