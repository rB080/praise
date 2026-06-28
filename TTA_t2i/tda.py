"""
attr_plip_tda_t2i.py

Training-free Dynamic Adapter (TDA) adapted for Text-to-Image (T2I) ReID using PLIP.

Setting:
  - Query: captions (text)
  - Gallery: images (precomputed image embeddings)

Core idea (TDA / Tip-Adapter style):
  - Maintain a POSITIVE cache of confident pseudo-labels (gallery indices)
  - Maintain a NEGATIVE cache of moderately-uncertain samples to suppress likely distractors
  - At inference, correct base logits with cache-based logit adjustments:
        logits = logits_base + pos_correction - neg_suppression

This is *training-free* (no gradients). Caches accumulate across test batches unless episodic=True.

API matches your T2I runner:
  init_model()
  tokenize(captions)
  initialize(gallery_feats)
  forward(captions, G_feats, hybrid_eval=True/False) -> (preds, feats, 0.0) or (feats, 0.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

# Project-specific imports (keep as in your repo)
from backbones.PLIP.PLIPmodel import Create_PLIP_Model
from utils.utils import Test_parse_args


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


@torch.no_grad()
def topk_entropy_from_logits(logits: torch.Tensor, k: int = 128, temp: float = 1.0) -> torch.Tensor:
    """
    Approximate predictive entropy using only top-k gallery indices.
    Returns normalized entropy in [0,1] per sample.
    """
    B, C = logits.shape
    kk = min(k, C)
    vals, _ = torch.topk(logits / temp, k=kk, dim=1, largest=True)
    p = torch.softmax(vals, dim=1)
    p = torch.clamp(p, min=1e-12)
    H = -(p * torch.log(p)).sum(dim=1)  # (B,)
    Hn = H / torch.log(torch.tensor(float(kk), device=logits.device))
    return Hn


@dataclass
class CacheEntryPos:
    key: torch.Tensor  # (d,) normalized query feature (caption)
    label: int         # gallery index (0..Ng-1)
    entropy: float     # normalized entropy (0..1)


@dataclass
class CacheEntryNeg:
    key: torch.Tensor      # (d,) normalized query feature (caption)
    neg_idx: torch.Tensor  # (m,) gallery indices to suppress
    entropy: float         # normalized entropy (0..1)


class ATTR_TTA(nn.Module):
    """
    TDA for T2I ReID (caption query, image gallery) with PLIP.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        backbone: str = "plip",

        # Base similarity temperature (applied as division in logits)
        id_temp: float = 10.0,

        # Cache weighting (Tip-Adapter / TDA style)
        tda_alpha_pos: float = 1.0,
        tda_beta_pos: float = 5.0,
        tda_alpha_neg: float = 1.0,
        tda_beta_neg: float = 5.0,

        # Cache sizes
        pos_cache_size: int = 512,
        neg_cache_size: int = 512,

        # Uncertainty gates (entropy in [0,1])
        tau_pos: float = 0.25,
        tau_neg_lo: float = 0.35,
        tau_neg_hi: float = 0.75,

        # How many gallery indices to suppress per negative entry
        neg_topm: int = 10,

        # Entropy computed on top-k for speed
        entropy_topk: int = 128,

        # Episodic resets cache each forward (typically False for TDA)
        episodic: bool = False,
    ):
        super().__init__()
        self.device = str(device)
        self.backbone = str(backbone)
        self.id_temp = float(id_temp)

        self.tda_alpha_pos = float(tda_alpha_pos)
        self.tda_beta_pos = float(tda_beta_pos)
        self.tda_alpha_neg = float(tda_alpha_neg)
        self.tda_beta_neg = float(tda_beta_neg)

        self.pos_cache_size = int(pos_cache_size)
        self.neg_cache_size = int(neg_cache_size)

        self.tau_pos = float(tau_pos)
        self.tau_neg_lo = float(tau_neg_lo)
        self.tau_neg_hi = float(tau_neg_hi)

        self.neg_topm = int(neg_topm)
        self.entropy_topk = int(entropy_topk)
        self.episodic = bool(episodic)

        self.model: Optional[nn.Module] = None
        self.args = Test_parse_args(notebook=True)
        self.TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

        self.G_norm: Optional[torch.Tensor] = None  # (Ng,d)

        self.pos_cache: List[CacheEntryPos] = []
        self.neg_cache: List[CacheEntryNeg] = []

    # -----------------------------
    # Model init + tokenization
    # -----------------------------
    def init_model(self):
        if self.backbone != "plip":
            raise NotImplementedError("attr_plip_tda_t2i currently supports backbone='plip' only.")

        model = Create_PLIP_Model(self.args).to(self.device)
        checkpoint = torch.load(self.args.model_path, map_location="cpu")
        model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
        model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"], strict=False)
        self.model = model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def tokenize(self, captions: List[str], max_len: int = 250) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.TOKENIZER(
            captions,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        return result["input_ids"], result["attention_mask"]

    @torch.no_grad()
    def encode_text(self, captions: List[str]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Call init_model() first.")
        ti, tm = self.tokenize(list(captions))
        tf, _ = self.model.text_encoder(ti.to(self.device), tm.to(self.device))
        return tf

    # -----------------------------
    # Cache lifecycle
    # -----------------------------
    def reset(self):
        self.pos_cache.clear()
        self.neg_cache.clear()

    def initialize(self, gallery_feats: torch.Tensor, **kwargs):
        self.G_norm = l2_normalize(gallery_feats.to(self.device).to(torch.float32))
        self.reset()

    # -----------------------------
    # Cache scoring A(sim)
    # -----------------------------
    @staticmethod
    def _A(sim: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        return alpha * torch.exp(-beta * (1.0 - sim))

    @torch.no_grad()
    def _enqueue_pos(self, q: torch.Tensor, label: int, entropy: float):
        entry = CacheEntryPos(key=q.detach().cpu(), label=int(label), entropy=float(entropy))
        if len(self.pos_cache) < self.pos_cache_size:
            self.pos_cache.append(entry)
            return
        worst_i = max(range(len(self.pos_cache)), key=lambda i: self.pos_cache[i].entropy)
        if entry.entropy < self.pos_cache[worst_i].entropy:
            self.pos_cache[worst_i] = entry

    @torch.no_grad()
    def _enqueue_neg(self, q: torch.Tensor, neg_idx: torch.Tensor, entropy: float):
        entry = CacheEntryNeg(key=q.detach().cpu(), neg_idx=neg_idx.detach().cpu().to(torch.long), entropy=float(entropy))
        if len(self.neg_cache) < self.neg_cache_size:
            self.neg_cache.append(entry)
            return
        worst_i = max(range(len(self.neg_cache)), key=lambda i: self.neg_cache[i].entropy)
        if entry.entropy < self.neg_cache[worst_i].entropy:
            self.neg_cache[worst_i] = entry

    # -----------------------------
    # Forward
    # -----------------------------
    @torch.no_grad()
    def forward(self, captions: List[str], G_feats: torch.Tensor, hybrid_eval: bool = False, **kwargs):
        if self.episodic:
            self.reset()

        if self.G_norm is None:
            self.G_norm = l2_normalize(G_feats.to(self.device).to(torch.float32))

        feats = self.encode_text(list(captions))              # (B,d)
        q = l2_normalize(feats.to(torch.float32))             # (B,d)
        G = self.G_norm                                       # (Ng,d)

        logits_base = (q @ G.t()) / self.id_temp              # (B,Ng)
        Hn = topk_entropy_from_logits(logits_base, k=self.entropy_topk, temp=1.0)

        p_base = torch.softmax(logits_base, dim=1)
        y_hat = torch.argmax(p_base, dim=1)

        neg_m = min(self.neg_topm, p_base.shape[1])
        topm_idx = torch.topk(p_base, k=neg_m, dim=1, largest=True).indices

        for b in range(q.shape[0]):
            ent = float(Hn[b].item())
            if ent <= self.tau_pos:
                self._enqueue_pos(q[b], int(y_hat[b].item()), ent)
            elif (self.tau_neg_lo <= ent <= self.tau_neg_hi):
                self._enqueue_neg(q[b], topm_idx[b], ent)

        logits = logits_base.clone()

        # positive correction
        if len(self.pos_cache) > 0 and self.tda_alpha_pos != 0.0:
            Kp = torch.stack([e.key for e in self.pos_cache], dim=0).to(self.device)  # (Mp,d)
            Lp = torch.tensor([e.label for e in self.pos_cache], device=self.device, dtype=torch.long)
            sim_p = q @ Kp.t()
            w_p = self._A(sim_p, self.tda_alpha_pos, self.tda_beta_pos)
            idx = Lp.unsqueeze(0).expand(q.shape[0], -1)
            logits.scatter_add_(1, idx, w_p.to(logits.dtype))

        # negative suppression
        if len(self.neg_cache) > 0 and self.tda_alpha_neg != 0.0:
            Kn = torch.stack([e.key for e in self.neg_cache], dim=0).to(self.device)
            sim_n = q @ Kn.t()
            w_n = self._A(sim_n, self.tda_alpha_neg, self.tda_beta_neg)
            for m, e in enumerate(self.neg_cache):
                idx_m = e.neg_idx.to(self.device).unsqueeze(0).expand(q.shape[0], -1)
                logits.scatter_add_(1, idx_m, -w_n[:, m].unsqueeze(1).expand_as(idx_m).to(logits.dtype))

        dummy_loss = 0.0
        if hybrid_eval:
            return logits.detach(), feats.detach(), dummy_loss
        return feats.detach(), dummy_loss
