"""
attr_plip_tcr_t2i.py

TCR (Test-time adaptation for Cross-modal Retrieval with Query Shift) adapted to:
  - Text-to-Image (T2I) retrieval in ReID benchmarks (RSTPReID / CUHK-PEDES)
  - Backbone: PLIP

Paper key pieces implemented:
  1) Query prediction refinement:
     - For each text query, pick its nearest-neighbor image from gallery (candidate selection).
     - Build refined predictions over the *candidate set* (B candidates for B queries), not the full gallery.

  2) Constraint estimation via a queue:
     - Compute SI criterion (Eq. 6) and keep the lowest-SI pairs.
     - Use the queue to estimate:
        - ΔS (source-like modality gap constraint, Eq. 7)
        - Em (self-adaptive entropy threshold, Eq. 8)

  3) Joint objective:
     - LNA  (noise-robust adaptation, Eq. 11)  -> weighted entropy using Em
     - LMU  (intra-modality uniformity learning, Eq. 9)
     - LMG  (inter-modality gap learning, Eq. 10) -> rectifies ΔT to ΔS

Adaptation details:
  - Update only LayerNorm affine params inside PLIP text_encoder (query modality for T2I).
  - AdamW optimizer.

API matches your T2I runners:
  - init_model()
  - tokenize(captions)
  - initialize(gallery_feats=...)
  - forward(captions, G_feats, hybrid_eval=False) -> (text_feats, loss) or (preds, text_feats, loss)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

from backbones.PLIP.PLIPmodel import Create_PLIP_Model
from utils.utils import Test_parse_args


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


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


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, min=eps)
    return -(p * torch.log(p)).sum(dim=-1)


@dataclass
class QueueItem:
    zq: torch.Tensor
    zg: torch.Tensor
    ent: float
    si: float


class ATTR_TTA(nn.Module):
    def __init__(
        self,
        steps: int = 1,
        device: str = "cuda:0",
        lr: float = 3e-5,
        weight_decay: float = 0.0,
        episodic: bool = False,

        temperature: float = 0.02,
        t_uniform: float = 10.0,

        con_ratio: float = 0.3,
        max_queue_size: int = 64,
        max_queue_updates: int = 10,

        w_na: float = 1.0,
        w_mu: float = 1.0,
        w_mg: float = 1.0,

        id_temp: float = 1.0,
    ):
        super().__init__()
        self.steps = int(steps)
        self.device = str(device)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.episodic = bool(episodic)

        self.temperature = float(temperature)
        self.t_uniform = float(t_uniform)

        self.con_ratio = float(con_ratio)
        self.max_queue_size = int(max_queue_size)
        self.max_queue_updates = int(max_queue_updates)

        self.w_na = float(w_na)
        self.w_mu = float(w_mu)
        self.w_mg = float(w_mg)
        self.id_temp = float(id_temp)

        self.args = Test_parse_args(notebook=True)
        self.TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model: Optional[nn.Module] = None

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model_state = None
        self.opt_state = None

        self._G_norm: Optional[torch.Tensor] = None

        self.queue: List[QueueItem] = []
        self.queue_updates_done: int = 0

    # -----------------------------
    # init / tokenize
    # -----------------------------
    def init_model(self):
        model = Create_PLIP_Model(self.args).to(self.device)
        checkpoint = torch.load(self.args.model_path, map_location="cpu")
        model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
        model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"], strict=False)
        self.model = model.to(self.device)

        n = disable_dropout(self.model.text_encoder)
        if n > 0:
            print(f"[TCR-T2I] Disabled dropout in {n} text modules.")

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
    # parameter selection
    # -----------------------------
    @staticmethod
    def _set_ln_grads_text(text_encoder: nn.Module) -> None:
        text_encoder.requires_grad_(False)
        for m in text_encoder.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters(recurse=False):
                    p.requires_grad_(True)

    @staticmethod
    def _collect_ln_params_text(text_encoder: nn.Module):
        params, names = [], []
        for nm, m in text_encoder.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np_, p in m.named_parameters(recurse=False):
                    if np_ in ["weight", "bias"]:
                        params.append(p)
                        names.append(f"text_encoder.{nm}.{np_}")
        return params, names

    def initialize(self, gallery_feats: Optional[torch.Tensor] = None):
        assert self.model is not None, "Call init_model() first."
        self._set_ln_grads_text(self.model.text_encoder)
        params, _ = self._collect_ln_params_text(self.model.text_encoder)
        print(f"[TCR-T2I] Optimizing {len(params)} LN affine params in text_encoder.")

        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        self.model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.opt_state = {k: v for k, v in self.optimizer.state_dict().items()}

        if gallery_feats is not None:
            self.cache_gallery(gallery_feats)

        self.queue = []
        self.queue_updates_done = 0

    def cache_gallery(self, G_feats: torch.Tensor):
        self._G_norm = l2_normalize(G_feats.to(self.device).to(torch.float32)).detach()

    def reset(self):
        assert self.model is not None and self.optimizer is not None
        if self.model_state is None or self.opt_state is None:
            raise RuntimeError("reset called before initialize()")
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.opt_state)
        self.queue = []
        self.queue_updates_done = 0

    # -----------------------------
    # Embedding helpers
    # -----------------------------
    def encode_text(self, captions: List[str]) -> torch.Tensor:
        assert self.model is not None
        ti, tm = self.tokenize(captions)
        tf, _ = self.model.text_encoder(ti.to(self.device), tm.to(self.device))
        return tf

    # -----------------------------
    # TCR math pieces
    # -----------------------------
    @staticmethod
    def _compute_SI(zq: torch.Tensor, zg: torch.Tensor) -> torch.Tensor:
        zq_c = zq.mean(dim=0, keepdim=True)
        zg_c = zg.mean(dim=0, keepdim=True)
        term1 = 2.0 * torch.norm(zq - zg, dim=1)
        term2 = torch.norm(zq - zq_c, dim=1) + torch.norm(zg - zg_c, dim=1)
        return term1 - term2

    def _update_queue(self, zq: torch.Tensor, zg: torch.Tensor, ent: torch.Tensor):
        if self.queue_updates_done >= self.max_queue_updates:
            return

        B = zq.shape[0]
        M = max(1, int(round(self.con_ratio * B)))

        SI = self._compute_SI(zq.detach(), zg.detach())
        idx = torch.topk(SI, k=M, largest=False).indices

        for i in idx.tolist():
            self.queue.append(
                QueueItem(
                    zq=zq[i].detach().cpu(),
                    zg=zg[i].detach().cpu(),
                    ent=float(ent[i].detach().cpu().item()),
                    si=float(SI[i].detach().cpu().item()),
                )
            )

        self.queue.sort(key=lambda it: it.si)
        if len(self.queue) > self.max_queue_size:
            self.queue = self.queue[: self.max_queue_size]

        self.queue_updates_done += 1

    def _get_constraints(self, fallback_zq: torch.Tensor, fallback_zg: torch.Tensor, fallback_ent: torch.Tensor) -> Tuple[float, float]:
        eps = 1e-12
        if len(self.queue) > 0:
            zq = torch.stack([it.zq for it in self.queue], dim=0).to(fallback_zq.device)
            zg = torch.stack([it.zg for it in self.queue], dim=0).to(fallback_zq.device)
            ent = torch.tensor([it.ent for it in self.queue], device=fallback_zq.device, dtype=fallback_ent.dtype)
            delta_s = torch.norm(zq.mean(dim=0) - zg.mean(dim=0)).item()
            e_m = max(float(ent.max().item()), eps)
            return delta_s, e_m

        B = fallback_zq.shape[0]
        M = max(1, int(round(self.con_ratio * B)))
        SI = self._compute_SI(fallback_zq.detach(), fallback_zg.detach())
        idx = torch.topk(SI, k=M, largest=False).indices
        zq_m = fallback_zq[idx]
        zg_m = fallback_zg[idx]
        ent_m = fallback_ent[idx]
        delta_s = torch.norm(zq_m.mean(dim=0) - zg_m.mean(dim=0)).item()
        e_m = max(float(ent_m.max().item()), eps)
        return delta_s, e_m

    # -----------------------------
    # Losses
    # -----------------------------
    @staticmethod
    def loss_uniformity(zq: torch.Tensor, t: float) -> torch.Tensor:
        zc = zq.mean(dim=0, keepdim=True)
        dist = torch.norm(zq - zc, dim=1)
        return torch.exp(-dist / max(t, 1e-6)).mean()

    @staticmethod
    def loss_gap(zq: torch.Tensor, zg: torch.Tensor, delta_s: float) -> torch.Tensor:
        zq_c = zq.mean(dim=0)
        zg_c = zg.mean(dim=0)
        delta_t = torch.norm(zq_c - zg_c)
        return (delta_t - float(delta_s)) ** 2

    @staticmethod
    def loss_noise_robust(ent: torch.Tensor, e_m: float) -> torch.Tensor:
        e_m = float(max(e_m, 1e-12))
        S = torch.clamp(1.0 - (ent / e_m), min=0.0)
        mask = S > 0
        if mask.sum() == 0:
            return ent.new_tensor(0.0)
        return (S[mask] * ent[mask]).sum() / (mask.sum().to(ent.dtype) + 1e-12)

    # -----------------------------
    # Adapt step
    # -----------------------------
    @torch.enable_grad()
    def _forward_and_adapt(self, captions: List[str], G_norm: torch.Tensor, step: int):
        assert self.model is not None and self.optimizer is not None

        tf = self.encode_text(captions)
        zq = l2_normalize(tf.to(torch.float32))     # (B,d)

        sim_matrix = zq @ G_norm.t()                # (B,Ng)
        nn_idx = sim_matrix.argmax(dim=1)
        zg = G_norm[nn_idx]                         # (B,d)

        sim_inter = (zq @ zg.t()) / max(self.temperature, 1e-6)  # (B,B)
        p_hat = torch.softmax(sim_inter, dim=1)
        ent_det = entropy_from_probs(p_hat).detach()

        if step == 0:
            self._update_queue(zq, zg, ent_det)

        delta_s, e_m = self._get_constraints(zq.detach(), zg.detach(), ent_det)

        ent = entropy_from_probs(p_hat)
        LNA = self.loss_noise_robust(ent, e_m)
        LMU = self.loss_uniformity(zq, self.t_uniform)
        LMG = self.loss_gap(zq, zg, delta_s)
        loss = self.w_na * LNA + self.w_mu * LMU + self.w_mg * LMG

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return zq.detach(), sim_matrix.detach(), float(loss.item())

    # -----------------------------
    # Public forward
    # -----------------------------
    def forward(self, captions: List[str], G_feats: torch.Tensor, hybrid_eval: bool = False, **kwargs):
        assert self.model is not None and self.optimizer is not None, "Call init_model() + initialize() first."

        if self.episodic:
            self.reset()

        if self._G_norm is None:
            self.cache_gallery(G_feats)
        G_norm = self._G_norm

        last_loss = 0.0
        for s in range(self.steps):
            _, _, last_loss = self._forward_and_adapt(list(captions), G_norm, step=s)

        with torch.no_grad():
            tf = self.encode_text(list(captions))
            zq = l2_normalize(tf.to(torch.float32))

            if hybrid_eval:
                sim = zq @ G_norm.t()
                preds = torch.softmax(sim / max(self.id_temp, 1e-6), dim=1)
                return preds.detach().cpu(), zq.detach().cpu(), last_loss

        return zq.detach().cpu(), last_loss
