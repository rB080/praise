"""
attr_plip_cliptta_t2i.py

CLIPTTA-style Test-Time Adaptation (TTA) for Text-to-Image (T2I) ReID using PLIP.

Goal:
  Adapt the PLIP TEXT encoder at test time using a CLIP-style contrastive loss (same spirit as CLIPTTA),
  but for retrieval settings where:
      query   = captions
      gallery = images (precomputed embeddings, fixed during TTA)

Core idea (T2I adaptation):
  1) Encode a batch of captions -> text features (B,d)
  2) Retrieve pseudo-positive image features from the gallery for each caption
     (top-1 or weighted top-k from text-to-gallery similarity)
  3) Build in-batch contrastive logits (B x B) between text features and pseudo-positive image features
  4) Minimize CLIP-like loss (or entropy / transductive variants)

This file is designed to be drop-in compatible with your T2I runner skeleton:
  - exposes ATTR_TTA with init_model(), tokenize(), initialize(), forward()
  - supports hybrid_eval=True to return (preds/probs, text_feats, loss)
  - optionally supports ACS (attribute-conditioned subspace) similarity (for pseudo-positive selection and hybrid eval)

Notes:
  - In pure retrieval, there are no "class names"; we treat the gallery as a pool of instance prototypes.
  - Batch size 1 cannot form in-batch negatives for contrastive; we automatically fall back to entropy over gallery.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer

# ---- project imports (keep your repo paths) ----
from backbones.PLIP.PLIPmodel import Create_PLIP_Model
from utils.utils import Test_parse_args


# ============================================================
# 1) Atomic attribute prompts (ACS basis)  [same as your ACS code]
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
# 2) Utils
# ============================================================
def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


def disable_dropout(module: nn.Module) -> int:
    """
    Force dropout layers into eval mode, even if parent model is train().
    Returns number of dropout modules disabled.
    """
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


def soft_ce(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Soft cross-entropy with targets as probabilities (N,C).
    """
    logp = F.log_softmax(logits, dim=-1)
    loss = -(targets * logp).sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def softmax_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(torch.clamp(p, min=1e-12))).sum(dim=-1)


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, min=eps)
    return -(p * torch.log(p)).sum(dim=-1).mean()


# ============================================================
# 3) Optional memory (caption -> pseudo-positive)
# ============================================================
@dataclass
class _MemItem:
    caption: str
    pos_idx: int
    score: float


class CaptionMemory:
    """
    Very lightweight FIFO memory.
    Stores caption strings + pseudo-positive gallery index + confidence score.
    At sampling time, captions are re-encoded with the current text encoder.
    """
    def __init__(self, capacity: int = 512, sample_size: int = 64):
        self.capacity = int(capacity)
        self.sample_size = int(sample_size)
        self.items: List[_MemItem] = []

    def reset(self):
        self.items = []

    def update(self, captions: List[str], pos_idx: torch.Tensor, scores: torch.Tensor):
        pos_idx = pos_idx.detach().cpu().tolist()
        scores = scores.detach().cpu().tolist()
        for c, i, s in zip(captions, pos_idx, scores):
            self.items.append(_MemItem(caption=str(c), pos_idx=int(i), score=float(s)))
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

    def sample(self) -> List[_MemItem]:
        if len(self.items) == 0:
            return []
        n = min(self.sample_size, len(self.items))
        # bias towards higher score samples
        scores = torch.tensor([it.score for it in self.items])
        probs = scores - scores.min()
        probs = probs + 1e-6
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, num_samples=n, replacement=False).tolist()
        return [self.items[i] for i in idx]


# ============================================================
# 4) Main T2I CLIPTTA-style class
# ============================================================
class ATTR_TTA(nn.Module):
    """
    T2I CLIPTTA-style adapter for PLIP:
      - adapts TEXT encoder norm affine params at test time
      - builds pseudo-positive image features from the fixed gallery
      - minimizes a CLIP-like contrastive loss (or alternatives)
    """
    DEFAULT_CAPTION_TEMPLATES = [
        "{}",                      # raw caption
        "A photo of a person. {}", # light prefix
        "This person: {}",         # alternate phrasing
    ]

    def __init__(
        self,
        steps: int = 1,
        topk: int = 50,                 # for entropy fallback / tent
        id_temp: float = 1.0,           # softmax temperature for gallery distribution
        contrastive_temp: float = 0.07, # tau for CLIP-style contrastive logits (B x B)

        device: str = "cuda:0",
        lr: float = 1e-3,
        episodic: bool = True,

        # --- CLIPTTA-style switches ---
        use_softmax_entropy: bool = False,  # use entropy on BxB logits instead of one-hot CE
        use_tent: bool = False,             # entropy minimization over gallery distribution
        use_clipartt: bool = False,         # transductive soft-target loss (CLIPArTT-like)
        clipartt_temp: float = 0.01,        # temp for transductive target softmax
        pos_topk: int = 1,                  # pseudo-positive from top1 (1) or weighted top-k (>1)
        caption_templates: Optional[List[str]] = None,
        shuffle_templates: bool = True,

        # --- memory (optional) ---
        use_memory: bool = False,
        memory_capacity: int = 512,
        memory_sample_size: int = 64,

        # --- ACS knobs (optional) ---
        use_acs: bool = False,
        acs_groups: Optional[List[str]] = None,
        acs_kmax: int = 10,
        acs_alpha: float = 0.4,
        acs_conf_thresh: float = 0.1,
        acs_use_residual: bool = False,
        neutral_text: str = "A photo of a person.",
        refresh_attr_bank: bool = True,
        zscore_text_logits: bool = False,
    ):
        super().__init__()
        self.steps = int(steps)
        self.topk = int(topk)
        self.id_temp = float(id_temp)
        self.contrastive_temp = float(contrastive_temp)
        self.device = str(device)
        self.lr = float(lr)
        self.episodic = bool(episodic)

        self.use_softmax_entropy = bool(use_softmax_entropy)
        self.use_tent = bool(use_tent)
        self.use_clipartt = bool(use_clipartt)
        self.clipartt_temp = float(clipartt_temp)
        self.pos_topk = int(pos_topk)

        self.caption_templates = caption_templates if caption_templates is not None else list(self.DEFAULT_CAPTION_TEMPLATES)
        self.shuffle_templates = bool(shuffle_templates)

        # memory
        self.use_memory = bool(use_memory)
        self.memory = CaptionMemory(capacity=memory_capacity, sample_size=memory_sample_size) if self.use_memory else None

        # ACS
        self.use_acs = bool(use_acs)
        self.acs_groups = acs_groups if acs_groups is not None else ["gender", "hair", "ubcc", "lbcc", "accessories"]
        self.acs_kmax = int(acs_kmax)
        self.acs_alpha = float(acs_alpha)
        self.acs_conf_thresh = float(acs_conf_thresh)
        self.acs_use_residual = bool(acs_use_residual)
        self.neutral_text = str(neutral_text)
        self.refresh_attr_bank = bool(refresh_attr_bank)
        self.zscore_text_logits = bool(zscore_text_logits)

        # internal
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model_state = None
        self.optimizer_state = None

        # Tokenizer
        self.TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
        self.args = Test_parse_args(notebook=True)

        # ACS prompt bank (computed on demand)
        self.attr_prompt_feats: Dict[str, torch.Tensor] = {}
        self.attr_dirs: Dict[str, torch.Tensor] = {}
        self.neutral_text_feat: Optional[torch.Tensor] = None

        # gallery cache
        self._G_cache: Optional[torch.Tensor] = None
        self._G_cache_norm: Optional[torch.Tensor] = None

    # -----------------------------
    # 4.1 Model init + tokenizer
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
    # 4.2 Param collection / reset
    # -----------------------------
    def collect_params(self, disable_dropouts: bool = True) -> Tuple[List[nn.Parameter], List[str]]:
        """
        Collect only TEXT-encoder normalization affine params (LN/BN/GN weight,bias).
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
                print(f"[CLIPTTA-T2I] Disabled dropout in {n} text modules.")

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
        params, _ = self.collect_params()
        print(f"[CLIPTTA-T2I] Optimizing {len(params)} params (TEXT norm affine only).")
        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        self.copy_model_and_optimizer()

        if gallery_feats is not None:
            self.cache_gallery(gallery_feats)

        if self.use_acs:
            print("[CLIPTTA-T2I] ACS enabled. Will (re)build attr bank from the current text encoder as needed.")

    def cache_gallery(self, G: torch.Tensor):
        self._G_cache = G.detach()
        self._G_cache_norm = l2_normalize(G.detach())

    def reset(self):
        self.load_model_and_optimizer()
        if self.use_memory and self.memory is not None:
            self.memory.reset()

    # -----------------------------
    # 4.3 Text encoding helpers
    # -----------------------------
    def _apply_caption_template(self, captions: List[str], template: str) -> List[str]:
        if "{}" not in template:
            return [template + " " + c for c in captions]
        return [template.format(c) for c in captions]

    @torch.no_grad()
    def encode_text(self, captions: List[str], template: str = "{}") -> torch.Tensor:
        assert self.model is not None
        device = torch.device(self.device)
        texts = self._apply_caption_template(captions, template)
        ti, tm = self.tokenize(texts)
        tf, _ = self.model.text_encoder(ti.to(device), tm.to(device))
        return l2_normalize(tf)

    # ============================================================
    # 5) ACS bank (optional)
    # ============================================================
    @torch.no_grad()
    def precompute_attr_bank(self, prompt_set: Dict[str, List[str]] = _prompt_set, chunksize: int = 64):
        """
        Build prompt embeddings and direction vectors from the *current* text encoder.
        """
        assert self.model is not None
        self.model.eval()
        device = torch.device(self.device)

        # neutral
        ti, tm = self.tokenize([self.neutral_text])
        tf0, _ = self.model.text_encoder(ti.to(device), tm.to(device))
        tf0 = l2_normalize(tf0.squeeze(0))
        self.neutral_text_feat = tf0

        self.attr_prompt_feats = {}
        self.attr_dirs = {}
        for group in self.acs_groups:
            prompts = prompt_set[group]
            feats = []
            for i in tqdm(range(0, len(prompts), chunksize), desc=f"[ACS bank] {group}"):
                chunk = prompts[i:i + chunksize]
                ti, tm = self.tokenize(chunk)
                tf, _ = self.model.text_encoder(ti.to(device), tm.to(device))
                feats.append(tf.detach())
            tf = torch.cat(feats, dim=0)  # (nvals,d)
            tf = l2_normalize(tf)
            dirs = l2_normalize(tf - tf0.unsqueeze(0))
            self.attr_prompt_feats[group] = tf
            self.attr_dirs[group] = dirs

    def predict_attrs_for_batch(self, x_norm: torch.Tensor) -> List[List[Tuple[str, int, float]]]:
        """
        x_norm: (B,d) normalized text features
        returns selected[b] = [(group, idx, conf), ...]
        """
        B, _ = x_norm.shape
        selected: List[List[Tuple[str, int, float]]] = [[] for _ in range(B)]

        for group in self.acs_groups:
            tf = self.attr_prompt_feats[group]  # (nvals,d)
            logits = x_norm @ tf.t()            # (B,nvals)
            if self.zscore_text_logits:
                mu = logits.mean(dim=0, keepdim=True)
                sd = logits.std(dim=0, keepdim=True) + 1e-6
                logits = (logits - mu) / sd
            probs = logits.softmax(dim=1)
            conf, idx = probs.max(dim=1)
            for b in range(B):
                c = float(conf[b].item())
                if c >= self.acs_conf_thresh:
                    selected[b].append((group, int(idx[b].item()), c))
        return selected

    def acs_similarity(self, x_norm: torch.Tensor, G_norm: torch.Tensor, selected: List[List[Tuple[str, int, float]]], eps: float = 1e-8) -> torch.Tensor:
        """
        Compute ACS similarity per query by projecting into a per-query attribute subspace.
        Treat selection + basis as stop-gradient (stable for TTA).
        """
        device = x_norm.device
        dtype = x_norm.dtype
        B, _ = x_norm.shape
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

            U = torch.stack(dirs, dim=1)  # (d,k)
            w = torch.tensor(weights, device=device, dtype=dtype)
            w = w / (w.sum() + eps)
            U = U * torch.sqrt(w).unsqueeze(0)

            Q, _ = torch.linalg.qr(U, mode="reduced")  # (d,k)

            S_raw = (x_norm[b:b+1] @ G_norm.t())  # (1,Ng)

            cq = (x_norm[b:b+1] @ Q)   # (1,k)
            cG = (G_norm @ Q)          # (Ng,k)

            cq = l2_normalize(cq, eps=eps)
            cG = l2_normalize(cG, eps=eps)
            S_sub = (cq @ cG.t())      # (1,Ng)

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
    # 6) Adaptation step
    # ============================================================
    def _compute_similarity(self, tf_norm: torch.Tensor, G_norm: torch.Tensor) -> torch.Tensor:
        """
        Similarity used for pseudo-positive selection (and hybrid eval).
        """
        if not self.use_acs:
            return tf_norm @ G_norm.t()
        if self.refresh_attr_bank or len(self.attr_dirs) == 0:
            self.precompute_attr_bank()
        selected = self.predict_attrs_for_batch(tf_norm.detach())
        return self.acs_similarity(tf_norm, G_norm, selected)

    def _pseudo_positive(self, S: torch.Tensor, G_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select pseudo-positive image prototypes from gallery.
        Returns:
            pos_feats: (B,d)
            scores: (B,) confidence proxy
        """
        B, Ng = S.shape
        if self.pos_topk <= 1 or Ng == 1:
            scores, idx = S.max(dim=1)  # (B,)
            pos = G_norm[idx]           # (B,d)
            return pos, scores

        k = min(self.pos_topk, Ng)
        vals, idx = torch.topk(S, k=k, dim=1, largest=True)   # (B,k)
        w = F.softmax(vals, dim=1)                            # (B,k)
        pos = torch.einsum("bk,bkd->bd", w, G_norm[idx])      # weighted mean
        scores = vals[:, 0]
        pos = l2_normalize(pos)
        return pos, scores

    @torch.enable_grad()
    def _adapt_step(self, captions: List[str], G: torch.Tensor, template: str) -> float:
        """
        One gradient step.
        """
        assert self.model is not None and self.optimizer is not None
        self.model.train()

        device = torch.device(self.device)

        # text
        texts = self._apply_caption_template(captions, template)
        ti, tm = self.tokenize(texts)
        tf, _ = self.model.text_encoder(ti.to(device), tm.to(device))
        tf_norm = l2_normalize(tf)

        # gallery (cached)
        if self._G_cache_norm is not None and (G is self._G_cache or torch.equal(G, self._G_cache)):
            G_norm = self._G_cache_norm.to(device)
        else:
            G_norm = l2_normalize(G.to(device))

        # similarity + pseudo positives
        S = self._compute_similarity(tf_norm, G_norm)                 # (B,Ng)
        pos, scores = self._pseudo_positive(S.detach(), G_norm)       # stop-grad selection

        # optional memory update
        if self.use_memory and self.memory is not None:
            # store only from the first step to reduce duplication
            self.memory.update(captions, pos_idx=torch.argmax(S.detach(), dim=1), scores=scores)

        # ----- loss branches (mirrors CLIPTTA structure, adapted to retrieval) -----
        B = tf_norm.shape[0]

        # (A) TENT-style entropy on gallery distribution
        if self.use_tent or B == 1:
            p = torch.softmax(S / self.id_temp, dim=1)               # (B,Ng)
            # focus on top-k to reduce tail noise
            k = min(self.topk, p.shape[1])
            vals, _ = torch.topk(p, k=k, dim=1, largest=True)
            vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-12)
            loss = entropy_from_probs(vals)

        else:
            # build CLIP-style BxB logits from pseudo positives
            logits_t = (tf_norm @ pos.t()) / self.contrastive_temp    # (B,B)
            logits_i = (pos @ tf_norm.t()) / self.contrastive_temp    # (B,B)

            if self.use_clipartt:
                # transductive soft targets (CLIPArTT-like)
                Sv = pos @ pos.t()               # (B,B)
                St = tf_norm @ tf_norm.t()       # (B,B)
                targets = torch.softmax(((Sv + St) / 2.0) / self.clipartt_temp, dim=1)
                loss = soft_ce(logits_t, targets, reduction="mean")
            else:
                if self.use_softmax_entropy:
                    loss = (softmax_entropy_from_logits(logits_t).mean() + softmax_entropy_from_logits(logits_i).mean()) / 2.0
                else:
                    targets = torch.eye(B, device=logits_t.device, dtype=logits_t.dtype)
                    loss = (soft_ce(logits_t, targets, reduction="mean") + soft_ce(logits_i, targets, reduction="mean")) / 2.0

        # memory replay (optional): add extra CLIP loss on sampled memory items
        if (not self.use_tent) and (self.use_memory and self.memory is not None):
            items = self.memory.sample()
            if len(items) > 0:
                mem_caps = [it.caption for it in items]
                mem_pos = torch.tensor([it.pos_idx for it in items], device=device, dtype=torch.long)
                mem_pos = G_norm[mem_pos]  # (Bm,d)

                mem_tf = self.encode_text(mem_caps, template="{}").to(device)  # (Bm,d)

                Bm = mem_tf.shape[0]
                if Bm > 1:
                    logits_mt = (mem_tf @ mem_pos.t()) / self.contrastive_temp
                    logits_mi = (mem_pos @ mem_tf.t()) / self.contrastive_temp
                    targets_m = torch.eye(Bm, device=device, dtype=logits_mt.dtype)
                    loss_mem = (soft_ce(logits_mt, targets_m, reduction="mean") + soft_ce(logits_mi, targets_m, reduction="mean")) / 2.0
                    loss = loss + 0.5 * loss_mem  # mild weight

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    # ============================================================
    # 7) Forward
    # ============================================================
    def forward(self, captions: List[str], G: torch.Tensor, hybrid_eval: bool = False):
        """
        captions: list[str] length B
        G: gallery feats (Ng,d)
        Returns:
          if hybrid_eval: (p, text_feats, loss)
          else: (text_feats, loss)
        """
        if self.episodic:
            self.reset()

        if self._G_cache is None:
            self.cache_gallery(G)

        # adapt
        last_loss = 0.0
        templates = list(self.caption_templates)
        if self.shuffle_templates and len(templates) > 1:
            # re-shuffle each call for fairness
            import random as _r
            _r.shuffle(templates)

        for step in range(self.steps):
            template = templates[step % len(templates)]
            last_loss = self._adapt_step(captions, G, template)

        # final text features (use the first template by default for eval)
        tf_final = self.encode_text(captions, template="{}")

        if not hybrid_eval:
            return tf_final, last_loss

        # hybrid: return probs over gallery (for analysis / debug)
        with torch.no_grad():
            G_norm = l2_normalize(G.to(tf_final.device))
            S = self._compute_similarity(tf_final, G_norm)
            p = torch.softmax(S / self.id_temp, dim=1)
        return p.detach(), tf_final.detach(), last_loss
