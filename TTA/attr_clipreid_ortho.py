from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

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
# 1) Atomic attribute prompts (USE THESE FOR ACS)
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

_prompt_set: Dict[str, List[str]] = {
    k: [_prompt_templates[k].format(p) for p in _prompts[k]]
    for k in _prompts.keys()
}
_prompt_set["accessories"] = _prompt_set["accessories"] + ["A photo of a person without any accessories."]


# ============================================================
# 2) Helpers
# ============================================================
def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, min=eps)
    return -(p * torch.log(p)).sum(dim=-1).mean()


def topk_rowwise(p: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    p: (B,N)
    returns: (vals, idx) where vals is (B,k)
    """
    kk = min(k, p.shape[1])
    vals, idx = torch.topk(p, k=kk, dim=1, largest=True)
    return vals, idx


def disable_dropout(module: nn.Module) -> int:
    """
    Force dropout layers into eval mode.
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


# ============================================================
# 3) CLIPReID ACS-Ortho TTA
# ============================================================
class ATTR_TTA(nn.Module):
    """
    ACS-Ortho style test-time adaptation for I2I ReID using CLIPReID backbone.

    Matches the *structure* of attr_plip_ortho.py:
      - precompute atomic prompt bank (text embeddings)
      - for each target batch, select confident attributes per sample
      - compute ACS similarity (orthogonal subspace projection) between query and gallery
      - minimize top-k entropy over gallery distribution
      - update only BN/LN affine params in the *visual* branch (TENT-style)

    IMPORTANT:
      - Text bank embeddings are computed using the underlying CLIP text encoder (self.clip.encode_text),
        matching your PRAISE script.
      - Image embeddings use CLIPReID's image_encoder API:
            _, image_features, image_features_proj = model.image_encoder(x)
            feats = image_features_proj[:, 0]
    """

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
        acs_kmax: int = 10,
        acs_alpha: float = 0.4,
        acs_conf_thresh: float = 0.1,
        acs_use_residual: bool = False,
        neutral_text: str = "A photo of a person.",

        # z-score the prompt logits (stabilizes across datasets)
        zscore_text_logits: bool = False,

        # CLIPReID cfg/model
        cfg_path: str = "backbones/CLIPReID/configs/person/vit_clipreid_market.yml",
        num_class: int = 751,
        camera_num: int = 6,
        view_num: int = 1,
    ):
        super().__init__()
        self.steps = int(steps)
        self.topk = int(topk)
        self.id_temp = float(id_temp)
        self.device = str(device)
        self.lr = float(lr)
        self.episodic = bool(episodic)

        self.use_acs = bool(use_acs)
        self.acs_groups = acs_groups if acs_groups is not None else ["gender", "hair", "ubcc", "lbcc", "accessories"]
        self.acs_kmax = int(acs_kmax)
        self.acs_alpha = float(acs_alpha)
        self.acs_conf_thresh = float(acs_conf_thresh)
        self.acs_use_residual = bool(acs_use_residual)
        self.neutral_text = str(neutral_text)
        self.zscore_text_logits = bool(zscore_text_logits)

        self.cfg_path = str(cfg_path)
        self.num_class = int(num_class)
        self.camera_num = int(camera_num)
        self.view_num = int(view_num)

        self.args = Test_parse_args(notebook=True)

        self.model: Optional[nn.Module] = None   # CLIPReID reid model
        self.clip: Optional[nn.Module] = None    # underlying CLIP (text encoder used for prompts)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model_state = None
        self.optimizer_state = None

        # ACS bank buffers
        self.attr_prompt_feats: Dict[str, torch.Tensor] = {}
        self.attr_dirs: Dict[str, torch.Tensor] = {}
        self.attr_strings: Dict[str, List[str]] = {}
        self.neutral_text_feat: Optional[torch.Tensor] = None

    # -----------------------------
    # Init CLIP (text encoder) same as PRAISE script
    # -----------------------------
    def _init_clip_model(self) -> nn.Module:
        h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(cfg.MODEL.NAME, h_resolution, w_resolution, vision_stride_size)
        clip_model.eval()
        return clip_model.to(self.device)

    def init_model(self, model_path: Optional[str] = None):
        """
        Loads CLIPReID model + CLIP text encoder used for prompt embeddings.
        """
        # breakpoint()
        cfg.merge_from_file(self.cfg_path)
        cfg.freeze()

        model = make_model(cfg, num_class=self.num_class, camera_num=self.camera_num, view_num=self.view_num)
        if model_path is not None:
            model.load_param_finetune(model_path)
        else:
            model.load_param(self.args.model_path)

        self.model = model.to(self.device)
        self.clip = self._init_clip_model()

        # Dropout off helps stability for TTA
        disable_dropout(self.model)

    # -----------------------------
    # CLIP tokenization (77)
    # -----------------------------
    def tokenize(self, texts: List[str]) -> torch.Tensor:
        return clip_tokenize(texts, context_length=77, truncate=True)

    # -----------------------------
    # Parameter selection (BN/LN affine, visual only)
    # -----------------------------
    @staticmethod
    def collect_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
        params: List[nn.Parameter] = []
        names: List[str] = []

        # Freeze everything first
        model.requires_grad_(False)

        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                for np, p in m.named_parameters(recurse=False):
                    if np in ["weight", "bias"]:
                        # skip any explicit text branch if it exists
                        if nm.split(".")[0] == "text_encoder":
                            continue
                        p.requires_grad_(True)
                        params.append(p)
                        names.append(f"{nm}.{np}")

                # For BN layers: force batch stats (TENT-style)
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

        return params, names

    def initialize(self, gallery_feats: Optional[torch.Tensor] = None, **kwargs):
        """
        Build optimizer and snapshot initial states.
        """
        assert self.model is not None, "Call init_model() first."
        params, _ = self.collect_params(self.model)
        print(f"[CLIPReID-ACS] Optimizing {len(params)} params (BN/LN affine only).")
        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset without saved model/optimizer state (call initialize())")
        assert self.model is not None and self.optimizer is not None
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    # -----------------------------
    # Text bank (ACS prompts)
    # -----------------------------
    @torch.no_grad()
    def precompute_attr_bank(
        self,
        prompt_set: Dict[str, List[str]] = _prompt_set,
        chunksize: int = 64,
    ):
        """
        Creates:
            self.attr_prompt_feats[group]: (nvals, d) normalized prompt embeddings
            self.attr_dirs[group]: (nvals, d) normalized direction vectors (prompt - neutral)
        Uses CLIP text encoder: self.clip.encode_text
        """
        assert self.clip is not None, "Call init_model() first."

        device = torch.device(self.device)

        # neutral embedding
        t0 = self.tokenize([self.neutral_text]).to(device)
        tf0 = self.clip.encode_text(t0)
        tf0 = l2_normalize(tf0.to(torch.float32)).squeeze(0)
        self.neutral_text_feat = tf0

        self.attr_prompt_feats = {}
        self.attr_dirs = {}
        self.attr_strings = {}

        for group in self.acs_groups:
            prompts = prompt_set[group]
            feats = []
            for i in tqdm(range(0, len(prompts), chunksize), desc=f"[CLIPReID] Text bank: {group}"):
                chunk = prompts[i:i + chunksize]
                ti = self.tokenize(chunk).to(device)
                tf = self.clip.encode_text(ti)  # (chunk,d)
                feats.append(tf.detach().to(torch.float32))
            tf = torch.cat(feats, dim=0)     # (nvals,d)
            tf = l2_normalize(tf)

            dirs = l2_normalize(tf - tf0.unsqueeze(0))

            self.attr_prompt_feats[group] = tf
            self.attr_dirs[group] = dirs
            self.attr_strings[group] = prompts

        print("[CLIPReID] Built ACS attribute bank groups:", list(self.attr_dirs.keys()))

    # -----------------------------
    # Attribute selection
    # -----------------------------
    @torch.no_grad()
    def predict_attrs_for_batch(self, x_norm: torch.Tensor) -> List[List[Tuple[str, int, float]]]:
        """
        x_norm: (B,d) normalized image features
        Returns list per sample: [(group, idx, conf), ...] filtered by conf_thresh
        """
        B, _ = x_norm.shape
        selected: List[List[Tuple[str, int, float]]] = [[] for _ in range(B)]

        for group in self.acs_groups:
            tf = self.attr_prompt_feats[group].to(x_norm.device)  # (nvals,d)
            logits = x_norm @ tf.t()                               # (B,nvals)

            if self.zscore_text_logits:
                mu = logits.mean(dim=0, keepdim=True)
                sd = logits.std(dim=0, keepdim=True) + 1e-6
                logits = (logits - mu) / sd

            probs = logits.softmax(dim=1)
            conf, idx = probs.max(dim=1)  # (B,)
            for b in range(B):
                c = float(conf[b].item())
                if c >= self.acs_conf_thresh:
                    selected[b].append((group, int(idx[b].item()), c))

        # keep only top acs_kmax by confidence
        for b in range(B):
            selected[b] = sorted(selected[b], key=lambda t: t[2], reverse=True)[: self.acs_kmax]

        return selected

    # -----------------------------
    # ACS similarity (orthogonal subspace)
    # -----------------------------
    def acs_similarity(
        self,
        x_norm: torch.Tensor,
        G_norm: torch.Tensor,
        selected: List[List[Tuple[str, int, float]]],
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Computes per-query ACS similarity: either projected-subspace sim or blend with raw sim.
        x_norm: (B,d), G_norm: (Ng,d)
        returns: (B,Ng)
        """
        device = x_norm.device
        dtype = x_norm.dtype

        B, d = x_norm.shape
        Ng = G_norm.shape[0]
        out = torch.empty((B, Ng), device=device, dtype=dtype)

        for b in range(B):
            items = selected[b]
            if len(items) == 0:
                out[b] = (x_norm[b:b+1] @ G_norm.t()).squeeze(0)
                continue

            # build weighted basis U (d,k)
            dirs = []
            weights = []
            for (group, idx, conf) in items:
                dirs.append(self.attr_dirs[group][idx].to(device=device, dtype=dtype))
                weights.append(conf)

            U = torch.stack(dirs, dim=1)  # (d,k)
            w = torch.tensor(weights, device=device, dtype=dtype)
            w = w / (w.sum() + eps)
            U = U * torch.sqrt(w).unsqueeze(0)  # weight columns

            # orthonormal basis Q via QR
            Q, _ = torch.linalg.qr(U, mode="reduced")  # (d,k)

            # raw similarity
            S_raw = x_norm[b:b+1] @ G_norm.t()  # (1,Ng)

            # projected coefficients
            cq = x_norm[b:b+1] @ Q              # (1,k)
            cG = G_norm @ Q                     # (Ng,k)

            cq = l2_normalize(cq, eps=eps)
            cG = l2_normalize(cG, eps=eps)
            S_sub = cq @ cG.t()                 # (1,Ng)

            if self.acs_use_residual:
                # subtract subspace component from raw
                # (xQ)(GQ)^T is the subspace correlation term
                S = S_raw - (cq @ cG.t())
            else:
                if self.acs_alpha < 1.0:
                    S = (1.0 - self.acs_alpha) * S_raw + self.acs_alpha * S_sub
                else:
                    S = S_sub

            out[b] = S.squeeze(0)

        return out

    # -----------------------------
    # Image encoding (CLIPReID API)
    # -----------------------------
    def encode_images(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W) -> (B,d)
        """
        assert self.model is not None, "Call init_model() first."
        _, image_features, image_features_proj = self.model.image_encoder(x)
        feats = image_features_proj[:, 0]
        return feats

    # ============================================================
    # Forward + adapt loop
    # ============================================================
    def forward(self, x: torch.Tensor, G: torch.Tensor, **kwargs):
        """
        x: query batch images (B,C,H,W)
        G: gallery_feats tensor (Ng, d) already precomputed
        """
        if self.episodic:
            self.reset()

        out = None
        for _ in range(self.steps):
            out = forward_and_adapt(self, x, G, **kwargs)
        return out


# ============================================================
# Adaptation step
# ============================================================
@torch.enable_grad()
def forward_and_adapt(adapter: ATTR_TTA, x: torch.Tensor, G: torch.Tensor, **kwargs):
    """
    Performs one TTA step:
      - compute similarity (ACS or raw)
      - entropy minimization on top-k gallery probs
      - update BN/LN affine params
    """
    assert adapter.model is not None and adapter.optimizer is not None, "Call init_model()+initialize() first."

    adapter.model.train()

    # Encode images with grad
    x_feat = adapter.encode_images(x.to(adapter.device))  # (B,d)

    # Ensure gallery on same device
    G = G.to(x_feat.device)

    # Normalize
    x_norm = l2_normalize(x_feat.to(torch.float32))
    G_norm = l2_normalize(G.to(torch.float32))

    # Similarity matrix
    if adapter.use_acs:
        if len(adapter.attr_prompt_feats) == 0:
            raise RuntimeError("ACS enabled but attr bank missing. Call precompute_attr_bank() once before testing.")
        selected = adapter.predict_attrs_for_batch(x_norm.detach())  # selection is stop-grad
        S = adapter.acs_similarity(x_norm, G_norm, selected)
    else:
        S = x_norm @ G_norm.t()

    # Probabilities over gallery
    p = torch.softmax(S / adapter.id_temp, dim=1)

    # Top-k entropy
    vals, _ = topk_rowwise(p, adapter.topk)
    vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-12)
    loss = entropy_from_probs(vals)

    loss.backward()
    adapter.optimizer.step()
    adapter.optimizer.zero_grad(set_to_none=True)

    if kwargs.get("hybrid_eval", False):
        return p.detach(), x_feat.detach(), float(loss.item())
    return x_feat.detach(), float(loss.item())
