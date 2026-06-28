# attr_plip_ortho_t2i_refresh.py
# Updated copy-paste script:
# 1) Adds ACS "kernel" mode that matches your original T2I: A_att = (q@Z^T) @ (G@Z^T)^T
# 2) Keeps existing "subspace" (QR) ACS mode
# 3) Lets you choose acs_mode in __init__ (default: "kernel" to reproduce old behavior)
# 4) Optional: refresh banks each step (works for both modes)
# 5) Uses additive mixing with S_raw by default (safer); you can set acs_additive=False to mimic "pure att"
#
# Works with PLIP Create_PLIP_Model + BertTokenizer.

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


def build_combinatorial_prompts() -> List[str]:
    """
    Reproduces the old combinatorial prompt list used in your original script.
    Roughly ~4k prompts: gender × hair variants × clothing type × colors × shoes × (bottom colors).
    """
    prompt_list: List[str] = []
    colours = ["black", "blue", "brown", "green", "gray", "orange", "pink", "purple", "red", "white", "yellow"]

    for g, pronoun in {"man": "He", "woman": "She"}.items():
        for h in ["", "with long hair", "with hair", "with short hair"]:
            for l in ["sp", "d", "sk"]:  # sp=shirt+pants, d=dress, sk=top+skirt
                for c in colours:
                    if l != "d":
                        for cc in colours:
                            for shoe in ["dark coloured shoes", "light coloured shoes"]:
                                prompt_list.append(
                                    f"A {g} {h} wearing a {c} top. {pronoun} is also wearing "
                                    f"{cc} {'pants' if l == 'sp' else 'skirt'} and {shoe}."
                                )
                    else:
                        for shoe in ["dark coloured shoes", "light coloured shoes"]:
                            prompt_list.append(
                                f"A {g} {h} wearing a {c} dress. {pronoun} is also wearing {shoe}."
                            )
    return prompt_list




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
# 3) T2I ATTR_TTA with ACS (subspace or kernel) + optional refresh
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
        acs_mode: str = "subspace",        # "kernel" = original T2I; "subspace" = QR ACS
        acs_groups: Optional[List[str]] = None,
        acs_kmax: int = 8,
        acs_alpha: float = 0.2,          # scale for ACS term when additive
        acs_conf_thresh: float = 0.0,
        acs_use_residual: bool = False,
        neutral_text: str = "A photo of a person.",

        # if True: S = S_raw + alpha * S_acs (recommended)
        # if False: S = S_acs (closer to "use only attribute similarity")
        acs_additive: bool = True,

        # ---- refresh bank each step (needed if adapting text encoder) ----
        refresh_attr_bank_each_step: bool = False,
        refresh_chunksize: int = 64,

        # Optional: zscore prompt logits for selection (subspace selection uses this)
        zscore_text_logits: bool = False,
        max_len: int = 250,

        # ---- Combinatorial kernel bank (old behavior) ----
        use_combinatorial_kernel_bank: bool = True,
        combinatorial_neutral_center: bool = False,  # match old behavior if you subtracted neutral
        combinatorial_chunksize: int = 64,

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
        self.acs_mode = acs_mode.lower().strip()
        assert self.acs_mode in ["kernel", "subspace"], "acs_mode must be 'kernel' or 'subspace'"

        self.acs_groups = acs_groups if acs_groups is not None else ["gender", "hair", "ubcc", "lbcc", "accessories"]
        self.acs_kmax = acs_kmax
        self.acs_alpha = acs_alpha
        self.acs_conf_thresh = acs_conf_thresh
        self.acs_use_residual = acs_use_residual
        self.neutral_text = neutral_text
        self.acs_additive = acs_additive

        self.refresh_attr_bank_each_step = refresh_attr_bank_each_step
        self.refresh_chunksize = refresh_chunksize
        self.zscore_text_logits = zscore_text_logits

        # Internal
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model_state = None
        self.optimizer_state = None

        self.TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
        self.args = Test_parse_args(notebook=True)

        # Combinatorial bank settings
        self.use_combinatorial_kernel_bank = use_combinatorial_kernel_bank
        self.combinatorial_neutral_center = combinatorial_neutral_center
        self.combinatorial_chunksize = combinatorial_chunksize


        # Attr bank holders
        # For "subspace": group-wise prompt feats + dirs
        self.attr_prompt_feats: Dict[str, torch.Tensor] = {}
        self.attr_dirs: Dict[str, torch.Tensor] = {}
        self.attr_strings: Dict[str, List[str]] = {}
        self.neutral_text_feat: Optional[torch.Tensor] = None

        # For "kernel": full prompt bank concatenated across selected groups
        self.kernel_Z: Optional[torch.Tensor] = None  # (P,d) normalized prompt feats (optionally neutral-centered)
        self.kernel_prompts: List[str] = []

    # -----------------------------
    # Model init + tokenizer
    # -----------------------------
    def init_model(self):
        # breakpoint()
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
        so we adapt text_encoder LN/BN affine params (image encoder excluded).
        """
        assert self.model is not None, "Call init_model() first."

        params: List[nn.Parameter] = []
        names: List[str] = []

        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                for np, p in m.named_parameters(recurse=False):
                    if np in ["weight", "bias"]:
                        # skip image encoder norms
                        if nm.split(".")[0] == "image_encoder":
                            continue
                        params.append(p)
                        names.append(f"{nm}.{np}")

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
        params, _ = self.collect_params()
        print(f"Optimizing {len(params)} params (LN/BN affine; image_encoder excluded).")
        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, weight_decay=1e-4)
        self.copy_model_and_optimizer()

        if self.use_acs:
            # Build initial bank once (and refresh later if enabled)
            self.precompute_attr_bank()

    def reset(self):
        self.load_model_and_optimizer()

    # ============================================================
    # ACS bank builders
    # - builds both group banks (for subspace) and kernel_Z (for kernel)
    # ============================================================
    @torch.no_grad()
    def precompute_attr_bank(
        self,
        prompt_set: Dict[str, List[str]] = _prompt_set,
        chunksize: int = 64,
        verbose: bool = True,
    ):
        self._build_attr_bank(prompt_set=prompt_set, chunksize=chunksize, show_pbar=verbose)

    @torch.no_grad()
    def refresh_attr_bank(
        self,
        prompt_set: Dict[str, List[str]] = _prompt_set,
        chunksize: Optional[int] = None,
    ):
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
        self.model.eval()
        device = torch.device(self.device)

        # --- neutral embedding (for optional centering) ---
        ti, tm = self.tokenize([self.neutral_text])
        tf0, _ = self.model.text_encoder(ti.to(device), tm.to(device))
        tf0 = l2_normalize(tf0.squeeze(0))
        self.neutral_text_feat = tf0

        # ============================================================
        # Option A: Combinatorial kernel bank (old behavior)
        # ============================================================
        if self.use_combinatorial_kernel_bank:
            prompts = build_combinatorial_prompts()
            feats = []

            it = range(0, len(prompts), self.combinatorial_chunksize)
            if show_pbar:
                it = tqdm(it, desc=f"Text bank: combinatorial ({len(prompts)})")

            for i in it:
                chunk = prompts[i:i + self.combinatorial_chunksize]
                ti, tm = self.tokenize(chunk)
                tf, _ = self.model.text_encoder(ti.to(device), tm.to(device))
                feats.append(tf.detach())

            tf = torch.cat(feats, dim=0)           # (P,d)
            tf = l2_normalize(tf)

            if self.combinatorial_neutral_center:
                Z = l2_normalize(tf - tf0.unsqueeze(0))
            else:
                Z = tf

            self.kernel_Z = l2_normalize(Z)
            self.kernel_prompts = prompts

            # optional: clear atomic banks (to avoid accidental usage / memory)
            self.attr_prompt_feats = {}
            self.attr_dirs = {}
            self.attr_strings = {}

            if show_pbar:
                print(f"Built COMBINATORIAL kernel Z: {tuple(self.kernel_Z.shape)}")
            return

        # ============================================================
        # Option B: Atomic group banks (current behavior)
        # ============================================================
        kernel_feats = []
        kernel_prompts = []

        for group in self.acs_groups:
            prompts = prompt_set[group]
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

            # neutral-centered directions
            dirs = l2_normalize(tf - tf0.unsqueeze(0))

            self.attr_prompt_feats[group] = tf
            self.attr_dirs[group] = dirs
            self.attr_strings[group] = prompts

            # kernel uses dirs by default (existing behavior)
            kernel_feats.append(dirs)
            kernel_prompts.extend(prompts)

        Z = torch.cat(kernel_feats, dim=0)  # (P,d)
        self.kernel_Z = l2_normalize(Z)
        self.kernel_prompts = kernel_prompts

        if show_pbar:
            print("Built ACS attribute bank groups:", list(self.attr_dirs.keys()))
            print(f"Kernel bank Z shape: {tuple(self.kernel_Z.shape)}")

    # ============================================================
    # ACS selection (for subspace mode)
    # ============================================================
    def predict_attrs_for_batch(self, q_norm: torch.Tensor) -> List[List[Tuple[str, int, float]]]:
        B, _ = q_norm.shape
        selected: List[List[Tuple[str, int, float]]] = [[] for _ in range(B)]

        for group in self.acs_groups:
            tf = self.attr_prompt_feats[group]  # (nvals,d)
            logits = q_norm @ tf.t()            # (B,nvals)

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
    # ACS similarity: subspace mode (QR)
    # ============================================================
    def acs_similarity_subspace(
        self,
        q_norm: torch.Tensor,
        G_norm: torch.Tensor,
        selected: List[List[Tuple[str, int, float]]],
        eps: float = 1e-8,
    ) -> torch.Tensor:
        device = q_norm.device
        dtype = q_norm.dtype
        B, _ = q_norm.shape
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
                S_acs = S_raw - ((q_norm[b:b+1] @ Q) @ (G_norm @ Q).t())
            else:
                S_acs = S_sub

            if self.acs_additive:
                out[b] = (S_raw + self.acs_alpha * S_acs).squeeze(0)
            else:
                out[b] = S_acs.squeeze(0)

        return out

    # ============================================================
    # ACS similarity: kernel mode (ORIGINAL T2I SPECIAL CASE)
    # S_att = (q @ Z^T) @ (G @ Z^T)^T
    # ============================================================
    def acs_similarity_kernel(self, q_norm: torch.Tensor, G_norm: torch.Tensor) -> torch.Tensor:
        assert self.kernel_Z is not None, "Kernel bank not built. Call precompute_attr_bank()."
        Z = self.kernel_Z.to(device=q_norm.device, dtype=q_norm.dtype)  # (P,d)

        S_raw = q_norm @ G_norm.t()          # (B,Ng)
        Ab = q_norm @ Z.t()                  # (B,P)
        Ag = G_norm @ Z.t()                  # (Ng,P)
        S_att = Ab @ Ag.t()                  # (B,Ng)  (old behavior)

        if self.acs_additive:
            return S_raw + self.acs_alpha * S_att
        else:
            return S_att


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

    # Refresh bank each step if requested (needed if adapting text encoder and using prompt banks)
    if adapter.use_acs and adapter.refresh_attr_bank_each_step:
        adapter.refresh_attr_bank()

    # Similarity
    if adapter.use_acs:
        if adapter.acs_mode == "kernel":
            S = adapter.acs_similarity_kernel(q_norm, G_norm)
        else:
            selected = adapter.predict_attrs_for_batch(q_norm)
            S = adapter.acs_similarity_subspace(q_norm, G_norm, selected)
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
