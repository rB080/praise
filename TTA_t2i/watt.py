
"""
attr_plip_watt_t2i.py

WATT-style Test-Time Adaptation (TTA) for Text-to-Image (T2I) ReID using PLIP.

Key idea (adapted from WATT: Weight Average Test-Time Adaptation of CLIP):
  - create multiple "caption prompt templates" around each query caption
  - adapt ONLY lightweight normalization parameters (LayerNorm/BatchNorm affine) in the TEXT encoder
  - for each template, run a few gradient steps to minimize an unsupervised objective on the
    similarity distribution over the fixed gallery (viewing gallery instances as proxy classes)
  - weight-average the adapted parameters across templates (optionally repeated M times)
  - (optional) at inference, ensemble text embeddings across templates

This file is designed to be drop-in compatible with your t2i runner skeleton:
  - exposes ATTR_TTA with init_model(), tokenize(), initialize(), forward()
  - supports hybrid_eval=True to return (preds/probs, text_feats, loss)
  - optionally supports ACS (attribute-conditioned subspace) similarity

NOTE:
  - Since we adapt the text encoder, ACS prompt banks are recomputed per batch by default.
    You can disable that with refresh_attr_bank=False, but then ACS directions may drift.

"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer

# ---- project imports (keep your repo paths) ----
from backbones.PLIP.PLIPmodel import Create_PLIP_Model
from utils.utils import Test_parse_args


# ============================================================
# 1) Atomic attribute prompts (ACS basis)
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

_prompt_set = {
    "age": [_prompt_templates["age"].format(p) for p in _prompts["age"]],
    "gender": [_prompt_templates["gender"].format(p) for p in _prompts["gender"]],
    "hair": [_prompt_templates["hair"].format(p) for p in _prompts["hair"]],
    "ubcl": [_prompt_templates["ubcl"].format(p) for p in _prompts["ubcl"]],
    "ubcc": [_prompt_templates["ubcc"].format(p) for p in _prompts["ubcc"]],
    "lbcl": [_prompt_templates["lbcl"].format(p) for p in _prompts["lbcl"]],
    "lbcc": [_prompt_templates["lbcc"].format(p) for p in _prompts["lbcc"]],
    "lbct": [_prompt_templates["lbct"].format(p) for p in _prompts["lbct"]],
    "accessories": [_prompt_templates["accessories"].format(p) for p in _prompts["accessories"]] +
                   ["A photo of a person without any accessories."],
}


# ============================================================
# 2) Utilities
# ============================================================
def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


def topk_rowwise(p: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(p, k=k, dim=1, largest=True)


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, min=eps)
    return -(p * torch.log(p)).sum(dim=1).mean()


def soft_ce(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Cross-entropy with soft targets.
    logits: (B, K)
    targets: (B, K) rows sum to 1
    """
    logp = F.log_softmax(logits, dim=-1)
    loss = -(targets * logp).sum(dim=1)
    if reduction == "mean":
        return loss.mean()
    return loss


def disable_dropout(module: nn.Module) -> int:
    dropout_types = (
        nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
        nn.AlphaDropout, nn.FeatureAlphaDropout
    )
    n = 0
    for m in module.modules():
        if isinstance(m, dropout_types):
            m.eval()
            m.training = False
            n += 1
    return n


# ============================================================
# 3) WATT-T2I adapter for PLIP
# ============================================================
class ATTR_TTA(nn.Module):
    """
    T2I WATT-style adapter:
      - adapt the TEXT encoder (norm affine params) using unlabeled captions
      - use the gallery as a proxy "class set" (instances)
      - weight-average updated params across multiple caption prompt templates
    """
    DEFAULT_CAPTION_TEMPLATES = [
        "{}",                          # raw caption
        "A photo of a person. {}",     # light prefix
        "A photo of a person, {}.",    # comma version
        "This person: {}.",            # colon version
        "A person who {}.",            # relative clause version
    ]

    def __init__(
        self,
        steps: int = 2,            # mapped to WATT inner steps (L)
        topk: int = 50,
        id_temp: float = 1.0,
        device: str = "cuda:0",
        lr: float = 1e-3,
        episodic: bool = True,

        # -------- WATT knobs --------
        watt_m: int = 1,                       # outer repeats (M)
        watt_type: str = "parallel",           # "parallel" or "sequential"
        caption_templates: Optional[List[str]] = None,
        text_ensemble_eval: bool = True,
        reset_opt_on_average: bool = True,

        # -------- loss knobs --------
        use_transductive: bool = True,
        tta_tau: float = 0.01,                 # for transductive soft targets

        # -------- ACS knobs (optional) --------
        use_acs: bool = False,
        acs_groups: Optional[List[str]] = None,
        acs_kmax: int = 8,
        acs_alpha: float = 0.4,
        acs_conf_thresh: float = 0.0,
        acs_use_residual: bool = False,
        neutral_text: str = "A photo of a person.",
        zscore_text_logits: bool = False,
        refresh_attr_bank: bool = True,        # important when adapting text encoder
    ):
        super().__init__()
        self.episodic = episodic
        self.steps = int(steps)
        self.topk = int(topk)
        self.id_temp = float(id_temp)
        self.device = device
        self.lr = float(lr)

        # WATT
        self.watt_m = int(watt_m)
        self.watt_type = watt_type
        assert self.watt_type in ["parallel", "sequential"]
        self.caption_templates = caption_templates or list(self.DEFAULT_CAPTION_TEMPLATES)
        self.text_ensemble_eval = bool(text_ensemble_eval)
        self.reset_opt_on_average = bool(reset_opt_on_average)

        # loss
        self.use_transductive = bool(use_transductive)
        self.tta_tau = float(tta_tau)

        # ACS
        self.use_acs = bool(use_acs)
        self.acs_groups = acs_groups if acs_groups is not None else ["gender", "hair", "ubcc", "lbcc", "accessories"]
        self.acs_kmax = int(acs_kmax)
        self.acs_alpha = float(acs_alpha)
        self.acs_conf_thresh = float(acs_conf_thresh)
        self.acs_use_residual = bool(acs_use_residual)
        self.neutral_text = neutral_text
        self.zscore_text_logits = bool(zscore_text_logits)
        self.refresh_attr_bank = bool(refresh_attr_bank)

        # internal
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model_state = None
        self.optimizer_state = None

        self.TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
        self.args = Test_parse_args(notebook=True)

        # ACS banks (computed from CURRENT text encoder by default)
        self.attr_prompt_feats: Dict[str, torch.Tensor] = {}
        self.attr_dirs: Dict[str, torch.Tensor] = {}
        self.attr_strings: Dict[str, List[str]] = {}
        self.neutral_text_feat: Optional[torch.Tensor] = None

        # optional cached gallery feats
        self._G_cache: Optional[torch.Tensor] = None
        self._G_cache_norm: Optional[torch.Tensor] = None

    # -----------------------------
    # 3.1 Model init + tokenizer
    # -----------------------------
    def init_model(self):
        model = Create_PLIP_Model(self.args).to(self.device)
        checkpoint = torch.load(self.args.model_path, map_location="cpu")
        model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
        model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"], strict=False)
        self.model = model.to(self.device)

    def tokenize(self, captions: List[str], max_len: int = 250) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.TOKENIZER(
            captions,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        return result["input_ids"], result["attention_mask"]

    # -----------------------------
    # 3.2 Param collection / reset
    # -----------------------------
    def collect_params(self, disable_dropouts: bool = True) -> Tuple[List[nn.Parameter], List[str]]:
        """
        Collect only TEXT-encoder normalization affine params (LN/BN weight,bias).
        """
        assert self.model is not None, "Call init_model() first."
        params: List[nn.Parameter] = []
        names: List[str] = []

        # Freeze everything by default
        self.model.requires_grad_(False)

        for nm, m in self.model.named_modules():
            if nm.split(".")[0] != "text_encoder":
                continue
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                for np_, p in m.named_parameters(recurse=False):
                    if np_ in ["weight", "bias"]:
                        p.requires_grad_(True)
                        params.append(p)
                        names.append(f"{nm}.{np_}")

        if disable_dropouts:
            n = disable_dropout(self.model.text_encoder)
            if n > 0:
                print(f"[WATT-T2I] Disabled dropout in {n} text modules.")

        return params, names

    def copy_model_and_optimizer(self):
        assert self.model is not None and self.optimizer is not None
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def load_model_and_optimizer(self):
        assert self.model is not None and self.optimizer is not None
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("reset called before initialize()")
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def initialize(self, gallery_feats: Optional[torch.Tensor] = None):
        """
        Set optimizer over TEXT norm affine params only, and snapshot initial states.
        Optionally caches gallery feats for speed.
        """
        assert self.model is not None, "Call init_model() first."
        params, names = self.collect_params()
        print(f"[WATT-T2I] Optimizing {len(params)} params (TEXT norm affine only).")
        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        self.copy_model_and_optimizer()

        if gallery_feats is not None:
            self.cache_gallery(gallery_feats)

        if self.use_acs:
            print("[WATT-T2I] ACS enabled. Will (re)build attr bank from the current text encoder as needed.")

    def cache_gallery(self, G: torch.Tensor):
        self._G_cache = G.detach()
        self._G_cache_norm = l2_normalize(self._G_cache)

    def reset(self):
        self.load_model_and_optimizer()
        if self.use_acs and self.refresh_attr_bank:
            # rebuild bank after reset to ensure it matches current weights
            self.precompute_attr_bank()

    # ============================================================
    # 4) ACS bank
    # ============================================================
    @torch.no_grad()
    def precompute_attr_bank(self, prompt_set: Dict[str, List[str]] = _prompt_set, chunksize: int = 64):
        """
        Recomputes ACS prompt embeddings and directions using CURRENT text encoder weights.
        """
        assert self.model is not None, "Call init_model() first."
        self.model.eval()
        device = torch.device(self.device)

        self.attr_prompt_feats = {}
        self.attr_dirs = {}
        self.attr_strings = {}

        # neutral
        ti, tm = self.tokenize([self.neutral_text])
        tf0, _ = self.model.text_encoder(ti.to(device), tm.to(device))
        tf0 = l2_normalize(tf0.squeeze(0))
        self.neutral_text_feat = tf0

        for group in self.acs_groups:
            prompts = prompt_set[group]
            feats = []
            for i in range(0, len(prompts), chunksize):
                chunk = prompts[i:i + chunksize]
                ti, tm = self.tokenize(chunk)
                tf, _ = self.model.text_encoder(ti.to(device), tm.to(device))
                feats.append(tf.detach())
            tf = torch.cat(feats, dim=0)       # (nvals, d)
            tf = l2_normalize(tf)
            dirs = l2_normalize(tf - tf0.unsqueeze(0))
            self.attr_prompt_feats[group] = tf
            self.attr_dirs[group] = dirs
            self.attr_strings[group] = prompts

    # ============================================================
    # 5) ACS attribute selection (from TEXT query embeddings)
    # ============================================================
    def predict_attrs_for_batch(self, x_norm: torch.Tensor) -> List[List[Tuple[str, int, float]]]:
        """
        For each query caption embedding, pick the top-1 attribute per group.
        """
        B, _ = x_norm.shape
        selected: List[List[Tuple[str, int, float]]] = [[] for _ in range(B)]

        # bank must exist
        if len(self.attr_prompt_feats) == 0 or len(self.attr_dirs) == 0:
            self.precompute_attr_bank()

        for group in self.acs_groups:
            tf = self.attr_prompt_feats[group]  # (nvals, d)
            logits = x_norm @ tf.t()            # (B, nvals)

            if self.zscore_text_logits:
                mu = logits.mean(dim=0, keepdim=True)
                sd = logits.std(dim=0, keepdim=True) + 1e-6
                logits = (logits - mu) / sd

            probs = logits.softmax(dim=1)
            conf, idx = probs.max(dim=1)        # (B,), (B,)

            for b in range(B):
                c = float(conf[b].item())
                if c >= self.acs_conf_thresh:
                    selected[b].append((group, int(idx[b].item()), c))

        return selected

    def acs_similarity(self, x_norm: torch.Tensor, G_norm: torch.Tensor, selected: List[List[Tuple[str, int, float]]], eps: float = 1e-8) -> torch.Tensor:
        """
        Same ACS similarity as your i2i implementation, but used for text queries.
        """
        device = x_norm.device
        dtype = x_norm.dtype
        B, d = x_norm.shape
        Ng = G_norm.shape[0]
        out = torch.empty((B, Ng), device=device, dtype=dtype)

        for b in range(B):
            items = sorted(selected[b], key=lambda t: t[2], reverse=True)[: self.acs_kmax]
            if len(items) == 0:
                out[b] = (x_norm[b:b+1] @ G_norm.t()).squeeze(0)
                continue

            dirs = []
            weights = []
            for (group, idx, conf) in items:
                dirs.append(self.attr_dirs[group][idx])
                weights.append(conf)

            U = torch.stack(dirs, dim=1)  # (d, k)
            w = torch.tensor(weights, device=device, dtype=dtype)
            w = w / (w.sum() + eps)
            U = U * torch.sqrt(w).unsqueeze(0)

            Q, _ = torch.linalg.qr(U, mode="reduced")  # (d, k)

            S_raw = (x_norm[b:b+1] @ G_norm.t())       # (1, Ng)

            cq = (x_norm[b:b+1] @ Q)                   # (1, k)
            cG = (G_norm @ Q)                          # (Ng, k)
            cq = l2_normalize(cq, eps=eps)
            cG = l2_normalize(cG, eps=eps)
            S_sub = (cq @ cG.t())                      # (1, Ng)

            if self.acs_use_residual:
                S = S_raw - ((x_norm[b:b+1] @ Q) @ (G_norm @ Q).t())
            else:
                if self.acs_alpha < 1.0:
                    S = S_raw + self.acs_alpha * S_sub
                else:
                    S = S_sub

            out[b] = S.squeeze(0)

        return out

    # ============================================================
    # 6) Text encoding helpers
    # ============================================================
    def _apply_caption_template(self, captions: List[str], template: str) -> List[str]:
        if "{}" not in template:
            raise ValueError(f"Template must contain '{{}}': got: {template}")
        # defensively avoid doubled punctuation
        out = []
        for c in captions:
            c2 = c.strip()
            out.append(template.format(c2))
        return out

    def encode_text(self, captions: List[str], template: str = "{}", max_len: int = 250) -> torch.Tensor:
        """
        No-grad text encoding for inference.
        Returns (B,d) normalized.
        """
        assert self.model is not None
        device = torch.device(self.device)
        texts = self._apply_caption_template(captions, template)
        ti, tm = self.tokenize(texts, max_len=max_len)
        tf, _ = self.model.text_encoder(ti.to(device), tm.to(device))
        tf = l2_normalize(tf)
        return tf

    def encode_text_ensemble(self, captions: List[str], templates: List[str], max_len: int = 250) -> torch.Tensor:
        with torch.no_grad():
            feats = []
            for t in templates:
                feats.append(self.encode_text(captions, template=t, max_len=max_len))
            tf = torch.stack(feats, dim=0).mean(dim=0)
            return l2_normalize(tf)

    # ============================================================
    # 7) WATT: extract/apply/average adaptable parameters
    # ============================================================
    def _iter_adaptable_named_params(self):
        """
        Iterate over adaptable params (requires_grad=True) with their *parameter* names.
        """
        assert self.model is not None
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                yield name, p

    def _snapshot_adaptable_params(self, to_cpu: bool = True) -> Dict[str, torch.Tensor]:
        snap = {}
        for name, p in self._iter_adaptable_named_params():
            t = p.detach().clone()
            if to_cpu:
                t = t.cpu()
            snap[name] = t
        return snap

    def _load_adaptable_params(self, state: Dict[str, torch.Tensor]):
        assert self.model is not None
        with torch.no_grad():
            for name, p in self._iter_adaptable_named_params():
                if name in state:
                    p.copy_(state[name].to(p.device))

    @staticmethod
    def _average_param_states(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        assert len(states) > 0
        K = len(states)
        out: Dict[str, torch.Tensor] = OrderedDict()
        keys = list(states[0].keys())
        for k in keys:
            out[k] = sum(s[k] for s in states) / K
        return out

    # ============================================================
    # 8) One adaptation step for ONE template
    # ============================================================
    @torch.enable_grad()
    def _adapt_step(self, captions: List[str], G: torch.Tensor, template: str) -> Tuple[torch.Tensor, float]:
        """
        Returns:
          tf_norm: (B,d) current text features (normalized), detached
          loss_value: float
        """
        assert self.model is not None and self.optimizer is not None
        self.model.train()

        device = torch.device(self.device)
        texts = self._apply_caption_template(captions, template)
        ti, tm = self.tokenize(texts)
        tf, _ = self.model.text_encoder(ti.to(device), tm.to(device))   # (B,d)
        tf_norm = l2_normalize(tf)

        # gallery
        if self._G_cache_norm is not None and (G is self._G_cache or torch.equal(G, self._G_cache)):
            G_norm = self._G_cache_norm.to(device)
        else:
            G_norm = l2_normalize(G.to(device))

        # similarity
        if self.use_acs:
            if self.refresh_attr_bank:
                self.precompute_attr_bank()
            selected = self.predict_attrs_for_batch(tf_norm.detach())  # selection doesn't need grad
            S = self.acs_similarity(tf_norm, G_norm, selected)
        else:
            S = tf_norm @ G_norm.t()

        # prob over gallery (proxy classes)
        p = torch.softmax(S / self.id_temp, dim=1)

        # loss
        if self.use_transductive and tf_norm.shape[0] > 1:
            # pseudo-visual "labels" from top-1 gallery retrieval
            idx1 = torch.argmax(p.detach(), dim=1)         # (B,)
            G_sel = G_norm[idx1]                           # (B,d)
            Sv = G_sel @ G_sel.t()                         # (B,B)
            St = tf_norm @ tf_norm.t()                     # (B,B)

            targets = torch.softmax(((Sv + St) / 2.0) / self.tta_tau, dim=1)  # (B,B)
            logits = (tf_norm @ G_sel.t()) / self.tta_tau                      # (B,B)
            loss = soft_ce(logits, targets, reduction="mean")
        else:
            vals, _ = topk_rowwise(p, min(self.topk, p.shape[1]))
            vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-12)
            loss = entropy_from_probs(vals)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return tf_norm.detach(), float(loss.item())

    # ============================================================
    # 9) Forward = WATT loop + final embedding
    # ============================================================
    def forward(self, captions: List[str], G: torch.Tensor, hybrid_eval: bool = False):
        """
        captions: list[str] of length B
        G: gallery feats (Ng,d)
        Returns:
          if hybrid_eval: (p, text_feats, loss)
          else: (text_feats, loss)
        """
        if self.episodic:
            self.reset()

        # cache gallery (optional)
        if self._G_cache is None:
            self.cache_gallery(G)

        last_loss = 0.0
        avg_state: Optional[Dict[str, torch.Tensor]] = None

        for m in range(self.watt_m):
            all_states: List[Dict[str, torch.Tensor]] = []

            if self.watt_type == "sequential":
                # init weights for this outer round
                if m == 0:
                    self.load_model_and_optimizer()
                else:
                    assert avg_state is not None
                    self._load_adaptable_params(avg_state)
                    if self.reset_opt_on_average:
                        self.optimizer.load_state_dict(self.optimizer_state)

            for template in self.caption_templates:
                if self.watt_type == "parallel":
                    if m == 0:
                        self.load_model_and_optimizer()
                    else:
                        assert avg_state is not None
                        self._load_adaptable_params(avg_state)
                        if self.reset_opt_on_average:
                            self.optimizer.load_state_dict(self.optimizer_state)

                for _ in range(self.steps):
                    _, last_loss = self._adapt_step(captions, G, template)

                all_states.append(self._snapshot_adaptable_params(to_cpu=True))

            avg_state = self._average_param_states(all_states)
            self._load_adaptable_params(avg_state)
            if self.reset_opt_on_average:
                self.optimizer.load_state_dict(self.optimizer_state)

        # final text feature (optionally ensemble templates)
        if self.text_ensemble_eval:
            tf_final = self.encode_text_ensemble(captions, self.caption_templates)
        else:
            tf_final = self.encode_text(captions, template="{}")

        if not hybrid_eval:
            return tf_final, last_loss

        # hybrid: also return probs over gallery for analysis
        with torch.no_grad():
            G_norm = l2_normalize(G.to(tf_final.device))
            if self.use_acs:
                if self.refresh_attr_bank:
                    self.precompute_attr_bank()
                selected = self.predict_attrs_for_batch(tf_final)
                S = self.acs_similarity(tf_final, G_norm, selected)
            else:
                S = tf_final @ G_norm.t()
            p = torch.softmax(S / self.id_temp, dim=1)
        return p.detach(), tf_final.detach(), last_loss
