# attr_plip_ortho_t2i_refresh.py
# Full copy-paste script: T2I ATTR_TTA with ACS, and (crucially) refresh of neutral + prompt bank each step/episode
# Works with PLIP Create_PLIP_Model + BertTokenizer like your current code.

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer

from backbones.PLIP.PLIPmodel import Create_PLIP_Model
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
    "accessories": "A photo of a person with {}."
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
                   ["A photo of a person without any accessories."]
}


# ============================================================
# 2) Utilities
# ============================================================
def make_nonsense_prompt_set(prompt_set: Dict[str, List[str]]) -> Dict[str, List[str]]:
    # keep structure identical, but destroy semantics
    out = {}
    for g, lst in prompt_set.items():
        out[g] = [s.replace("person", "blorpt").replace("photo", "snarp") + " qxjv" for s in lst]
    return out

def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


def topk_rowwise(p: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(p, k=k, dim=1, largest=True)


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, min=eps)
    return -(p * torch.log(p)).sum(dim=1).mean()


def disable_dropout(module: nn.Module) -> None:
    dropout_types = (
        nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
        nn.AlphaDropout, nn.FeatureAlphaDropout
    )
    count = 0
    for m in module.modules():
        if isinstance(m, dropout_types):
            m.eval()
            m.training = False
            count += 1
    if count > 0:
        print(f"Disabled dropout in {count} modules.")


# ============================================================
# 3) T2I ATTR_TTA with ACS + bank refresh
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
        acs_alpha: float = 0.1,          # IMPORTANT: for T2I start small; additive by default below
        acs_conf_thresh: float = 0.0,
        acs_use_residual: bool = False,
        neutral_text: str = "A photo of a person.",

        # ---- crucial for T2I if adapting text encoder ----
        refresh_attr_bank_each_step: bool = False,
        refresh_chunksize: int = 64,

        # Optional: zscore prompt logits for selection
        zscore_text_logits: bool = False,
        max_len: int = 250,

        # Prompt Ablations
        prompt_ablation_mode: str = "none",
        prompt_ablation_seed: int = 0,
    ):
        super().__init__()
        self.episodic = episodic
        self.steps = steps
        self.topk = topk
        self.id_temp = id_temp
        self.device = device
        self.lr = lr
        self.max_len = max_len

        # ACS config
        self.use_acs = use_acs
        self.acs_groups = acs_groups if acs_groups is not None else ["gender", "hair", "ubcc", "lbcc", "accessories"]
        self.acs_kmax = acs_kmax
        self.acs_alpha = acs_alpha
        self.acs_conf_thresh = acs_conf_thresh
        self.acs_use_residual = acs_use_residual
        self.neutral_text = neutral_text
        self.zscore_text_logits = zscore_text_logits

        self.refresh_attr_bank_each_step = refresh_attr_bank_each_step
        self.refresh_chunksize = refresh_chunksize

        # Internal
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model_state = None
        self.optimizer_state = None

        self.TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
        self.args = Test_parse_args(notebook=True)

        # Attr bank holders
        self.attr_prompt_feats: Dict[str, torch.Tensor] = {}
        self.attr_dirs: Dict[str, torch.Tensor] = {}
        self.attr_strings: Dict[str, List[str]] = {}
        self.neutral_text_feat: Optional[torch.Tensor] = None

        # Prompt Ablations
        self.prompt_ablation_mode = prompt_ablation_mode
        self.prompt_ablation_seed = prompt_ablation_seed
        self.fixed_attr_choice = {
            "gender": 0,       # index into _prompt_set["gender"]
            "hair": 0,         # index into _prompt_set["hair"]
            "ubcc": 0,         # index into _prompt_set["ubcc"]
            "lbcc": 0,         # index into _prompt_set["lbcc"]
            "accessories": 0,  # index into _prompt_set["accessories"]
        }
        self._fixed_dirs_cache = None
        self._fixed_tf_cache = None

    # -----------------------------
    # Model init + tokenizer
    # -----------------------------
    def init_model(self):
        model = Create_PLIP_Model(self.args).to(self.device)
        checkpoint = torch.load(self.args.model_path, map_location="cpu")
        model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
        model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"], strict=False)
        self.model = model.to(self.device)

    def tokenize(self, captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.TOKENIZER(
            captions,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        return result["input_ids"], result["attention_mask"]

    # -----------------------------
    # Param collection / reset
    # -----------------------------
    def collect_params(self, disable_dropouts: bool = True) -> Tuple[List[nn.Parameter], List[str]]:
        """
        T2I here: gallery image feats are frozen (precomputed),
        so we adapt text_encoder LN/BN affine params.
        """
        assert self.model is not None, "Call init_model() first."

        params: List[nn.Parameter] = []
        names: List[str] = []
        # self.model.eval()
        self.model.requires_grad_(False)
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                for np, p in m.named_parameters(recurse=False):
                    if np in ["weight", "bias"]:
                        # skip image encoder norms
                        if nm.split(".")[0] == "image_encoder":
                            continue
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        p.requires_grad = True

        if disable_dropouts:
            disable_dropout(self.model)
        return params, names

    def copy_model_and_optimizer(self):
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def load_model_and_optimizer(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("reset called before initialize()")
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def initialize(self, gallery_feats: torch.Tensor):
        params, names = self.collect_params()
        print(f"Optimizing {len(params)} params (LN/BN affine; image_encoder excluded).")
        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, weight_decay=1e-4)
        self.copy_model_and_optimizer()

        if self.use_acs:
            # Build initial bank once (will refresh later if enabled)
            if len(self.attr_dirs) == 0:
                self.precompute_attr_bank()

    def reset(self):
        self.load_model_and_optimizer()

    # ============================================================
    # ACS bank builders
    # ============================================================
    @torch.no_grad()
    def precompute_attr_bank(
        self,
        prompt_set: Dict[str, List[str]] = _prompt_set,
        chunksize: int = 64,
        verbose: bool = True,
    ):
        """
        One-time build (also usable as refresh).
        """
        self._build_attr_bank(prompt_set=prompt_set, chunksize=chunksize, show_pbar=verbose)

    @torch.no_grad()
    def refresh_attr_bank(
        self,
        prompt_set: Dict[str, List[str]] = _prompt_set,
        chunksize: Optional[int] = None,
    ):
        """
        Refresh using CURRENT text encoder parameters.
        Call this if you adapt text encoder.
        """
        if chunksize is None:
            chunksize = self.refresh_chunksize
        self._build_attr_bank(prompt_set=prompt_set, chunksize=chunksize, show_pbar=False)

    @torch.no_grad()
    def _build_attr_bank(
        self,
        prompt_set: Dict[str, List[str]],
        chunksize: int,
        show_pbar: bool,
    ):
        assert self.model is not None, "Call init_model() first."

        # Deterministic prompt encoding (dropout already disabled)
        self.model.eval()
        device = torch.device(self.device)

        # neutral embedding
        ti, tm = self.tokenize([self.neutral_text])
        tf0, _ = self.model.text_encoder(ti.to(device), tm.to(device))
        tf0 = l2_normalize(tf0.squeeze(0))
        self.neutral_text_feat = tf0

        for group in self.acs_groups:
            prompts = prompt_set[group]

            # Ablation mode select
            # --- ablation: nonsense prompts ---
            if self.prompt_ablation_mode == "nonsense":
                prompt_set = make_nonsense_prompt_set(prompt_set)

            feats = []
            it = range(0, len(prompts), chunksize)
            if show_pbar:
                it = tqdm(it, desc=f"Text bank: {group}")
            for i in it:
                chunk = prompts[i:i + chunksize]
                ti, tm = self.tokenize(chunk)
                tf, _ = self.model.text_encoder(ti.to(device), tm.to(device))
                feats.append(tf.detach())
            tf = torch.cat(feats, dim=0)  # (nvals,d)
            tf = l2_normalize(tf)
            dirs = l2_normalize(tf - tf0.unsqueeze(0))

            self.attr_prompt_feats[group] = tf
            self.attr_dirs[group] = dirs
            self.attr_strings[group] = prompts

        # --- ablation: fixed subspace selection (constant across queries) ---
        if self.prompt_ablation_mode == "fixed":
            # simplest fixed: pick index 0 per group (or you can pick the most "neutral-far" direction)
            self.fixed_attr_choice = {g: 0 for g in self.acs_groups}

            # cache fixed dirs/tf for faster per-query use
            dirs = [self.attr_dirs[g][self.fixed_attr_choice[g]] for g in self.acs_groups]
            self._fixed_dirs_cache = torch.stack(dirs, dim=1)  # (d, k_fixed)

            tfs = [self.attr_prompt_feats[g][self.fixed_attr_choice[g]] for g in self.acs_groups]
            self._fixed_tf_cache = torch.stack(tfs, dim=0)     # (k_fixed, d)

        if show_pbar:
            print("Built ACS attribute bank groups:", list(self.attr_dirs.keys()))

    # ============================================================
    # ACS selection
    # ============================================================
    # def predict_attrs_for_batch(self, q_norm: torch.Tensor) -> List[List[Tuple[str, int, float]]]:
    #     """
    #     Top-1 value per group (you can extend to top-m if desired).
    #     """
    #     B, _ = q_norm.shape
    #     selected: List[List[Tuple[str, int, float]]] = [[] for _ in range(B)]

    #     for group in self.acs_groups:
    #         tf = self.attr_prompt_feats[group]  # (nvals,d)
    #         logits = q_norm @ tf.t()            # (B,nvals)

    #         if self.zscore_text_logits:
    #             mu = logits.mean(dim=0, keepdim=True)
    #             sd = logits.std(dim=0, keepdim=True) + 1e-6
    #             logits = (logits - mu) / sd

    #         probs = torch.softmax(logits, dim=1)
    #         conf, idx = probs.max(dim=1)

    #         for b in range(B):
    #             c = float(conf[b].item())
    #             if c >= self.acs_conf_thresh:
    #                 selected[b].append((group, int(idx[b].item()), c))

    #     return selected

    def predict_attrs_for_batch(self, q_norm: torch.Tensor) -> List[List[Tuple[str, int, float]]]:
        B, _ = q_norm.shape
        selected: List[List[Tuple[str, int, float]]] = [[] for _ in range(B)]

        mode = self.prompt_ablation_mode
        device = q_norm.device

        # --- "shuffle": pick attrs using a permuted batch of queries ---
        if mode == "shuffle" and B > 1:
            # deterministic-ish shuffle: roll by 1 (cheap, stable)
            q_for_select = torch.roll(q_norm, shifts=1, dims=0)
        else:
            q_for_select = q_norm

        # --- "fixed": same attr indices for every query ---
        if mode == "fixed" and len(self.fixed_attr_choice) > 0:
            for b in range(B):
                for g in self.acs_groups:
                    idx = int(self.fixed_attr_choice[g])
                    selected[b].append((g, idx, 1.0))
            return selected

        # --- "random": random attr indices (no semantics), constant per forward call ---
        if mode == "random":
            gen = torch.Generator(device=device)
            gen.manual_seed(self.prompt_ablation_seed)
            for b in range(B):
                for g in self.acs_groups:
                    nvals = self.attr_prompt_feats[g].shape[0]
                    idx = int(torch.randint(low=0, high=nvals, size=(1,), generator=gen, device=device).item())
                    selected[b].append((g, idx, 1.0))
            return selected

        # --- normal / nonsense / score_fusion path uses normal argmax selection ---
        for group in self.acs_groups:
            tf = self.attr_prompt_feats[group]   # (nvals, d)
            logits = q_for_select @ tf.t()       # (B, nvals)

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
    # ACS similarity
    # ============================================================
    def acs_similarity(
        self,
        q_norm: torch.Tensor,     # (B,d) text queries
        G_norm: torch.Tensor,     # (Ng,d) image gallery
        selected: List[List[Tuple[str, int, float]]],
        eps: float = 1e-8,
    ) -> torch.Tensor:
        device = q_norm.device
        dtype = q_norm.dtype
        B, d = q_norm.shape
        Ng = G_norm.shape[0]
        out = torch.empty((B, Ng), device=device, dtype=dtype)

        for b in range(B):
            items = sorted(selected[b], key=lambda t: t[2], reverse=True)[: self.acs_kmax]

            S_raw = (q_norm[b:b+1] @ G_norm.t())  # (1,Ng)

            if len(items) == 0:
                out[b] = S_raw.squeeze(0)
                continue

            dirs = []
            weights = []
            for (group, idx, conf) in items:
                dirs.append(self.attr_dirs[group][idx])  # (d,)
                weights.append(conf)

            U = torch.stack(dirs, dim=1)  # (d,k)
            w = torch.tensor(weights, device=device, dtype=dtype)
            w = w / (w.sum() + eps)
            U = U * torch.sqrt(w).unsqueeze(0)

            Q, _ = torch.linalg.qr(U, mode="reduced")  # (d,k)

            cq = (q_norm[b:b+1] @ Q)                   # (1,k)
            cG = (G_norm @ Q)                          # (Ng,k)
            cq = l2_normalize(cq, eps=eps)
            cG = l2_normalize(cG, eps=eps)
            S_sub = (cq @ cG.t())                      # (1,Ng)

            if self.acs_use_residual:
                # remove attribute-subspace contribution (rarely helpful for T2I)
                S = S_raw - ((q_norm[b:b+1] @ Q) @ (G_norm @ Q).t())
            else:
                # IMPORTANT: T2I safer as ADDITIVE (keep raw caption signal)
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


# ============================================================
# Adaptation step (T2I)
# ============================================================
@torch.enable_grad()
def forward_and_adapt(adapter: ATTR_TTA, captions: List[str], G: torch.Tensor, **kwargs):
    adapter.model.train()

    # Encode text query
    ti, tm = adapter.tokenize(captions)
    q_feat, _ = adapter.model.text_encoder(ti.to(adapter.device), tm.to(adapter.device))

    # Normalize & align
    G = G.to(q_feat.device)
    q_norm = l2_normalize(q_feat)
    G_norm = l2_normalize(G)

    # Refresh bank each step if requested (keeps neutral/prompts aligned with current text encoder)
    if adapter.use_acs and adapter.refresh_attr_bank_each_step:
        adapter.refresh_attr_bank()

    # Similarity
    if adapter.use_acs:
        selected = adapter.predict_attrs_for_batch(q_norm)
        S = adapter.acs_similarity(q_norm, G_norm, selected)
    else:
        S = q_norm @ G_norm.t()

    # Entropy-min over top-k
    p = torch.softmax(S / adapter.id_temp, dim=1)
    vals, _ = topk_rowwise(p, adapter.topk)
    vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-12)
    loss = entropy_from_probs(vals)

    loss.backward()
    adapter.optimizer.step()
    adapter.optimizer.zero_grad()

    if kwargs.get("hybrid_eval", False):
        return p.detach(), q_feat.detach(), float(loss.item())
    else:
        return q_feat.detach(), float(loss.item())
