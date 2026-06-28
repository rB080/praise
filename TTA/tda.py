from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Project-specific imports (keep as in your repo)
# -----------------------------
from backbones.PLIP.PLIPmodel import Create_PLIP_Model
from utils.utils import Test_parse_args


# -----------------------------
# Small utilities
# -----------------------------
def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


@torch.no_grad()
def topk_entropy_from_logits(logits: torch.Tensor, k: int = 128, temp: float = 1.0) -> torch.Tensor:
    """
    Approximate predictive entropy using only top-k classes (gallery indices).
    This keeps runtime reasonable for large Ng.
    Returns: (B,) entropy values.
    """
    B, C = logits.shape
    kk = min(k, C)

    vals, idx = torch.topk(logits / temp, k=kk, dim=1, largest=True)
    p = torch.softmax(vals, dim=1)  # (B, kk)
    p = torch.clamp(p, min=1e-12)
    H = -(p * torch.log(p)).sum(dim=1)  # (B,)
    # Normalize to [0,1] scale (max entropy is log(kk))
    Hn = H / torch.log(torch.tensor(float(kk), device=logits.device))
    return Hn


@dataclass
class CacheEntryPos:
    key: torch.Tensor          # (d,) normalized query feature
    label: int                 # gallery index (0..Ng-1)
    entropy: float             # normalized entropy (0..1)


@dataclass
class CacheEntryNeg:
    key: torch.Tensor          # (d,) normalized query feature
    neg_idx: torch.Tensor      # (m,) gallery indices to suppress
    entropy: float             # normalized entropy (0..1)


class TDA_REID(nn.Module):
    """
    ReID-oriented Training-free Dynamic Adapter (TDA).

    - No gradient updates (training-free).
    - Maintains positive and negative caches keyed by test features.
    - Produces adapted similarity logits over gallery indices.

    Paper idea (ported):
      logits' = logits_base + P_pos + P_neg   (classification version)  :contentReference[oaicite:3]{index=3}
    For ReID we interpret negative cache as *suppression* (subtract logits on likely distractors),
    which is the natural retrieval analogue.

    API matches your runner:
      init_model()
      initialize(gallery_feats)
      forward(imgs, G_feats, hybrid_eval=True/False) -> (preds, feats, loss) or (feats, loss)
    """

    def __init__(
        self,
        device: str = "cuda:0",
        backbone: str = "plip",          # "plip" or "clipreid" (extend if needed)

        # Base similarity temperature (like your other TTAs)
        id_temp: float = 10.0,

        # --- TDA cache weighting (alpha, beta) ---
        # In TDA/Tip-Adapter style, cache weight is exp(-beta*(1 - sim)) scaled by alpha. :contentReference[oaicite:4]{index=4}
        tda_alpha_pos: float = 1.0,
        tda_beta_pos: float = 5.0,
        tda_alpha_neg: float = 1.0,
        tda_beta_neg: float = 5.0,

        # --- Cache sizes (global priority queues) ---
        pos_cache_size: int = 512,
        neg_cache_size: int = 512,

        # --- Uncertainty gates (normalized entropy in [0,1]) ---
        # TDA uses uncertainty thresholds to decide whether to enqueue positives vs negatives. :contentReference[oaicite:5]{index=5}
        tau_pos: float = 0.25,   # confident => positive cache
        tau_neg_lo: float = 0.35,  # moderately uncertain => negative cache
        tau_neg_hi: float = 0.75,

        # How many gallery indices to suppress per negative entry
        neg_topm: int = 10,

        # Entropy computed on top-k classes for speed
        entropy_topk: int = 128,

        # Episodic resets cache each forward (usually False for dataset-wide TDA)
        episodic: bool = False,
    ):
        super().__init__()
        self.device = device
        self.backbone = backbone
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

        self.model = None
        self.args = Test_parse_args(notebook=True)

        # Cached gallery feats (normalized)
        self.G_norm: Optional[torch.Tensor] = None  # (Ng,d)

        # Dynamic caches
        self.pos_cache: List[CacheEntryPos] = []
        self.neg_cache: List[CacheEntryNeg] = []

    # -----------------------------
    # Model init
    # -----------------------------
    def init_model(self):
        """
        Mirrors your PLIP loading style.
        Extend here if you want clipreid/qwen, etc.
        """
        if self.backbone != "plip":
            raise NotImplementedError("TDA_ReID.init_model currently implemented for backbone='plip' only.")

        model = Create_PLIP_Model(self.args).to(self.device)
        checkpoint = torch.load(self.args.model_path, map_location="cpu")
        model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
        model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"], strict=False)
        self.model = model.to(self.device)

    @torch.no_grad()
    def encode_images(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Call init_model() first.")
        if self.backbone == "plip":
            return self.model.get_image_embeddings(imgs)
        raise NotImplementedError

    # -----------------------------
    # Cache lifecycle
    # -----------------------------
    def reset(self):
        self.pos_cache.clear()
        self.neg_cache.clear()

    def initialize(self, gallery_feats: torch.Tensor, **kwargs):
        """
        Called once per dataset (recommended).
        gallery_feats: (Ng,d) precomputed.
        """
        self.G_norm = l2_normalize(gallery_feats.to(self.device).to(torch.float32))
        self.reset()

    # -----------------------------
    # Cache scoring (A(sim))
    # -----------------------------
    @staticmethod
    def _A(sim: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        """
        sim: (B,M) cosine similarities in [-1,1].
        A(sim) = alpha * exp(-beta * (1 - sim))  (Tip-Adapter / TDA-style). :contentReference[oaicite:6]{index=6}
        """
        return alpha * torch.exp(-beta * (1.0 - sim))

    @torch.no_grad()
    def _enqueue_pos(self, q: torch.Tensor, label: int, entropy: float):
        """
        Global priority queue: keep lowest-entropy (most reliable) positives.
        """
        entry = CacheEntryPos(key=q.detach(), label=int(label), entropy=float(entropy))

        if len(self.pos_cache) < self.pos_cache_size:
            self.pos_cache.append(entry)
            return

        # Replace worst (highest entropy) if new is better
        worst_i = max(range(len(self.pos_cache)), key=lambda i: self.pos_cache[i].entropy)
        if entry.entropy < self.pos_cache[worst_i].entropy:
            self.pos_cache[worst_i] = entry

    @torch.no_grad()
    def _enqueue_neg(self, q: torch.Tensor, neg_idx: torch.Tensor, entropy: float):
        """
        Global priority queue for negatives: keep most "usefully uncertain" samples.
        """
        entry = CacheEntryNeg(key=q.detach(), neg_idx=neg_idx.detach().to(torch.long), entropy=float(entropy))

        if len(self.neg_cache) < self.neg_cache_size:
            self.neg_cache.append(entry)
            return

        worst_i = max(range(len(self.neg_cache)), key=lambda i: self.neg_cache[i].entropy)
        # For negatives you can keep *moderate entropy* entries; we still use "replace-worst" heuristic.
        if entry.entropy < self.neg_cache[worst_i].entropy:
            self.neg_cache[worst_i] = entry

    # -----------------------------
    # Main forward
    # -----------------------------
    @torch.no_grad()
    def forward(self, imgs: torch.Tensor, G_feats: torch.Tensor, hybrid_eval: bool = False, **kwargs):
        """
        Returns:
          - if hybrid_eval: (preds, feats, loss) where preds is (B,Ng) similarity logits
          - else: (feats, loss) (feats unchanged; loss is dummy 0.0)
        """
        if self.episodic:
            self.reset()

        if self.G_norm is None:
            # allow passing G each time if user doesn't call initialize()
            self.G_norm = l2_normalize(G_feats.to(self.device).to(torch.float32))

        # 1) Encode query
        feats = self.encode_images(imgs.to(self.device))
        q = l2_normalize(feats.to(torch.float32))        # (B,d)
        G = self.G_norm                                 # (Ng,d)

        # 2) Base logits over gallery indices
        logits_base = (q @ G.t()) / self.id_temp         # (B,Ng)

        # 3) Uncertainty (entropy) on top-k mass, normalized
        Hn = topk_entropy_from_logits(logits_base, k=self.entropy_topk, temp=1.0)  # (B,)

        # 4) Update caches (paper uses uncertainty thresholds for pos/neg enqueue). :contentReference[oaicite:7]{index=7}
        # Pseudo-label = argmax gallery index
        p_base = torch.softmax(logits_base, dim=1)       # (B,Ng)
        y_hat = torch.argmax(p_base, dim=1)              # (B,)

        # Negative indices: top-m most likely distractors
        neg_m = min(self.neg_topm, p_base.shape[1])
        topm_idx = torch.topk(p_base, k=neg_m, dim=1, largest=True).indices  # (B,neg_m)

        for b in range(q.shape[0]):
            ent = float(Hn[b].item())
            if ent <= self.tau_pos:
                self._enqueue_pos(q[b], int(y_hat[b].item()), ent)
            elif (self.tau_neg_lo <= ent <= self.tau_neg_hi):
                self._enqueue_neg(q[b], topm_idx[b], ent)

        # 5) Cache predictions (logit correction)
        logits = logits_base.clone()

        # 5.1 Positive cache: add weight to the cached label index
        if len(self.pos_cache) > 0 and self.tda_alpha_pos != 0.0:
            Kp = torch.stack([e.key for e in self.pos_cache], dim=0)             # (Mp,d)
            Lp = torch.tensor([e.label for e in self.pos_cache], device=q.device, dtype=torch.long)  # (Mp,)
            sim_p = q @ Kp.t()                                                   # (B,Mp)
            w_p = self._A(sim_p, self.tda_alpha_pos, self.tda_beta_pos)          # (B,Mp)

            idx = Lp.unsqueeze(0).expand(q.shape[0], -1)                         # (B,Mp)
            logits.scatter_add_(1, idx, w_p)

        # 5.2 Negative cache: suppress top-m indices for cached uncertain samples
        if len(self.neg_cache) > 0 and self.tda_alpha_neg != 0.0:
            Kn = torch.stack([e.key for e in self.neg_cache], dim=0)             # (Mn,d)
            sim_n = q @ Kn.t()                                                   # (B,Mn)
            w_n = self._A(sim_n, self.tda_alpha_neg, self.tda_beta_neg)          # (B,Mn)

            # subtract weights at neg indices (retrieval analogue of negative cache)
            for m, e in enumerate(self.neg_cache):
                # (B,neg_m)
                idx_m = e.neg_idx.unsqueeze(0).expand(q.shape[0], -1)
                logits.scatter_add_(1, idx_m, -w_n[:, m].unsqueeze(1).expand_as(idx_m).to(logits.dtype))

        # Note: feats are unchanged (training-free). Return logits for hybrid_eval.
        dummy_loss = 0.0
        if hybrid_eval:
            # Return similarity logits; in your eval use DISTMAT = -preds (monotone for ranking).
            return logits.detach(), feats.detach(), dummy_loss
        return feats.detach(), dummy_loss
