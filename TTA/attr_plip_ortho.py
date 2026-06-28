from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer

# -----------------------------
# Project-specific imports
# -----------------------------
# NOTE: keep these if they exist in your repo; otherwise adjust.
from backbones.PLIP.PLIPmodel import Create_PLIP_Model
from utils.utils import Test_parse_args  # your existing arg parser


# ============================================================
# 1) Atomic attribute prompts (USE THESE FOR ACS)
#    Avoid the combinatorial prompt_list for ACS basis.
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
    """
    Returns (values, indices) for top-k per row.
    """
    return torch.topk(p, k=k, dim=1, largest=True)


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12, nll: bool = False) -> torch.Tensor:
    """
    Proper entropy: -sum p log p.
    Expects p rows sum to 1.
    """
    p = torch.clamp(p, min=eps)
    if not nll: return -(p * torch.log(p)).sum(dim=1).mean()
    else: return -(torch.log(p)).sum(dim=1).mean()


def disable_dropout(module):
    import torch.nn as nn
    dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                     nn.AlphaDropout, nn.FeatureAlphaDropout)
    count_dropouts_closed = 0
    for m in module.modules():
        if isinstance(m, dropout_types):
            m.eval()  # forces dropout off for that module
            m.training = False  # also set training=False to be safe
            count_dropouts_closed += 1
    print(f"Disabled dropout in {count_dropouts_closed} modules.")



# ============================================================
# 3) Main class with ACS integrated
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
        acs_groups: Optional[List[str]] = None,   # which attribute groups to use
        acs_kmax: int = 8,                        # how many attribute directions per query
        acs_alpha: float = 1.0,                   # 1.0 = use subspace-only; <1 = blend with raw dot
        acs_conf_thresh: float = 0.0,             # ignore very low-conf attrs
        acs_use_residual: bool = False,           # True = remove subspace (nuisance removal)
        neutral_text: str = "A photo of a person.",

        # Prompt Ablations
        prompt_ablation_mode: str = "none",
        prompt_ablation_seed: int = 0,

        # Optional: enable zscore for prompt logits before attribute selection
        zscore_text_logits: bool = False,
    ):
        super().__init__()
        self.episodic = episodic
        self.steps = steps
        self.topk = topk
        self.id_temp = id_temp
        self.device = device
        self.lr = lr

        # ACS config
        self.use_acs = use_acs
        self.acs_groups = acs_groups if acs_groups is not None else ["gender", "hair", "ubcc", "lbcc", "accessories"]
        self.acs_kmax = acs_kmax
        self.acs_alpha = acs_alpha
        self.acs_conf_thresh = acs_conf_thresh
        self.acs_use_residual = acs_use_residual
        self.neutral_text = neutral_text
        self.zscore_text_logits = zscore_text_logits

        # Internal
        self.model = None
        self.optimizer = None
        self.model_state = None
        self.optimizer_state = None

        # Tokenizer
        self.TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

        # Your args
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
    # 3.1 Model init + tokenizer
    # -----------------------------
    def init_model(self):
        model = Create_PLIP_Model(self.args).to(self.device)
        checkpoint = torch.load(self.args.model_path, map_location="cpu")
        model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
        model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"], strict=False)
        self.model = model.to(self.device)

    def tokenize(self, captions: List[str], max_len: int = 250) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids: (B, L)
            attention_mask: (B, L)
        """
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
    def collect_params(self, disable_dropouts=True) -> Tuple[List[nn.Parameter], List[str]]:
        """
        Collect only LN/BN affine params (weight, bias), excluding text_encoder.
        """
        params: List[nn.Parameter] = []
        names: List[str] = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                for np, p in m.named_parameters(recurse=False):
                    if np in ["weight", "bias"]:
                        if nm.split(".")[0] == "text_encoder":
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

    def initialize(self, gallery_feats: torch.Tensor, set_learnable_text=False):
        """
        Set optimizer over LN/BN params only.
        """
        params, names = self.collect_params()
        print(f"Optimizing {len(params)} params (LN/BN affine only).")
        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, weight_decay=1e-4)
        self.copy_model_and_optimizer()

        # Cache nothing else here; you pass gallery feats into forward().
        # But we do ensure attr bank exists if ACS is enabled.
        if self.use_acs and (len(self.attr_dirs) == 0):
            print("ACS enabled but attr bank is empty. Call precompute_attr_bank() once.")

    def reset(self):
        self.load_model_and_optimizer()

    # ============================================================
    # 4) ACS: build an attribute bank once
    # ============================================================
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
        """
        assert self.model is not None, "Call init_model() first."
        self.model.eval()

        device = torch.device(self.device)

        # neutral text embedding
        ti, tm = self.tokenize([self.neutral_text])
        tf0, _ = self.model.text_encoder(ti.to(device), tm.to(device))
        tf0 = l2_normalize(tf0.squeeze(0))
        self.neutral_text_feat = tf0

        for group in self.acs_groups:
            # breakpoint()
            prompts = prompt_set[group]
            # Ablation mode select
            # --- ablation: nonsense prompts ---
            if self.prompt_ablation_mode == "nonsense":
                prompt_set = make_nonsense_prompt_set(prompt_set)
            feats = []
            for i in tqdm(range(0, len(prompts), chunksize), desc=f"Text bank: {group}"):
                chunk = prompts[i:i + chunksize]
                ti, tm = self.tokenize(chunk)
                tf, _ = self.model.text_encoder(ti.to(device), tm.to(device))
                feats.append(tf.detach())
            tf = torch.cat(feats, dim=0)              # (nvals, d)
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

        print("Built ACS attribute bank groups:", list(self.attr_dirs.keys()))

    # ============================================================
    # 5) ACS: per-query attribute selection
    # ============================================================
    # def predict_attrs_for_batch(self, x_norm: torch.Tensor) -> List[List[Tuple[str, int, float]]]:
    #     """
    #     For each query in batch:
    #         pick top-1 value per group by prompt similarity.
    #     Returns:
    #         selected[b] = [(group, idx, conf), ...]
    #     """
    #     B, d = x_norm.shape
    #     selected: List[List[Tuple[str, int, float]]] = [[] for _ in range(B)]

    #     for group in self.acs_groups:
    #         tf = self.attr_prompt_feats[group]  # (nvals, d)
    #         logits = x_norm @ tf.t()            # (B, nvals)

    #         if self.zscore_text_logits:
    #             mu = logits.mean(dim=0, keepdim=True)
    #             sd = logits.std(dim=0, keepdim=True) + 1e-6
    #             logits = (logits - mu) / sd

    #         probs = logits.softmax(dim=1)
    #         conf, idx = probs.max(dim=1)        # (B,)

    #         for b in range(B):
    #             c = conf[b].item()
    #             if c >= self.acs_conf_thresh:
    #                 selected[b].append((group, int(idx[b].item()), float(c)))

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
            # breakpoint()
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
    # 6) ACS: per-query subspace similarity
    # ============================================================
    def acs_similarity(
        self,
        x_norm: torch.Tensor,     # (B, d)
        G_norm: torch.Tensor,     # (Ng, d)
        selected: List[List[Tuple[str, int, float]]],
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        For each query b:
          - stack selected direction vectors into U (d,k)
          - QR -> Q (d,k) orthonormal
          - compute similarity using only coordinates in subspace:
                c_q = x @ Q  (1,k)
                c_G = G @ Q  (Ng,k)
                S_sub = cosine(c_q, c_G)
          - optionally:
                blend with raw dot: S = (1-a)*S_raw + a*S_sub
                or remove subspace: S = S_raw - ((x@Q) @ (G@Q)^T)
        Returns:
            S: (B, Ng)
        """
        device = x_norm.device
        dtype = x_norm.dtype
        B, d = x_norm.shape
        Ng = G_norm.shape[0]
        mode = self.prompt_ablation_mode

        out = torch.empty((B, Ng), device=device, dtype=dtype)

        for b in range(B):
            items = sorted(selected[b], key=lambda t: t[2], reverse=True)[: self.acs_kmax]
            
            # raw dot
            S_raw = (x_norm[b:b+1] @ G_norm.t())       # (1, Ng)

            # Ablation mode
            if mode == "score_fusion":
                # Build selected prompt embeddings (not dirs)
                tfs = []
                weights = []
                for (group, idx, conf) in items:
                    tfs.append(self.attr_prompt_feats[group][idx])  # (d,)
                    weights.append(conf)
                T = torch.stack(tfs, dim=0)  # (k, d)

                w = torch.tensor(weights, device=device, dtype=dtype)
                w = w / (w.sum() + eps)
                # prompt-score vectors
                q_scores = (x_norm[b:b+1] @ T.t()) * torch.sqrt(w).unsqueeze(0)  # (1,k)
                g_scores = (G_norm @ T.t()) * torch.sqrt(w).unsqueeze(0)         # (Ng,k)

                # cosine in prompt-score space
                q_scores = l2_normalize(q_scores, eps=eps)
                g_scores = l2_normalize(g_scores, eps=eps)
                S_ps = (q_scores @ g_scores.t())                                 # (1,Ng)

                # Blend with raw dot to match your other ablations
                S = S_raw + self.acs_alpha * S_ps
                out[b] = S.squeeze(0)
                continue
            if mode == "fixed" and self._fixed_dirs_cache is not None:
                U = self._fixed_dirs_cache  # (d, k_fixed)
                Q, _ = torch.linalg.qr(U, mode="reduced")
                cq = (x_norm[b:b+1] @ Q)
                cG = (G_norm @ Q)
                cq = l2_normalize(cq, eps=eps)
                cG = l2_normalize(cG, eps=eps)
                S_sub = (cq @ cG.t())
                out[b] = (S_raw + self.acs_alpha * S_sub).squeeze(0)
                continue
            ######################

            # breakpoint()
            # Fallback if no attrs found
            if len(items) == 0:
                out[b] = (x_norm[b:b+1] @ G_norm.t()).squeeze(0)
                continue

            dirs = []
            weights = []
            for (group, idx, conf) in items:
                dirs.append(self.attr_dirs[group][idx])  # (d,)
                weights.append(conf)

            U = torch.stack(dirs, dim=1)  # (d, k)
            w = torch.tensor(weights, device=device, dtype=dtype)
            w = w / (w.sum() + eps)
            U = U * torch.sqrt(w).unsqueeze(0)

            # Orthonormal basis
            Q, _ = torch.linalg.qr(U, mode="reduced")  # (d, k)
            # breakpoint()
            

            # coordinates in subspace
            cq = (x_norm[b:b+1] @ Q)                   # (1, k)
            cG = (G_norm @ Q)                          # (Ng, k)

            # cosine inside subspace
            cq = l2_normalize(cq, eps=eps)
            cG = l2_normalize(cG, eps=eps)
            S_sub = (cq @ cG.t())                      # (1, Ng)

            if self.acs_use_residual:
                # residual after removing attribute subspace
                # since Q is orthonormal: (x@Q)(G@Q)^T equals dot inside that subspace
                S = S_raw - ((x_norm[b:b+1] @ Q) @ (G_norm @ Q).t())
            else:
                if self.acs_alpha < 1.0:
                    # S = (1.0 - self.acs_alpha) * S_raw + self.acs_alpha * S_sub
                    S = S_raw + self.acs_alpha * S_sub
                else:
                    S = S_sub

            out[b] = S.squeeze(0)

        return out

    # ============================================================
    # 7) Forward + adapt loop
    # ============================================================
    def forward(self, x, G: torch.Tensor, **kwargs):
        """
        x: query batch (tensor or list depending on your model API)
        G: gallery_feats tensor (Ng, d) already precomputed
        """
        if self.episodic:
            self.reset()

        out = None
        for _ in range(self.steps):
            out = forward_and_adapt(self, x, G, **kwargs)
        return out


# ============================================================
# 8) Adaptation step (replaces the A_id + A_att policy)
# ============================================================
@torch.enable_grad()
def forward_and_adapt(adapter: ATTR_TTA, x, G: torch.Tensor, **kwargs):
    """
    Performs one TTA step: compute similarity -> entropy loss -> update LN/BN.
    """
    adapter.model.train()

    # ---- Encode images ----
    # Keep your original logic: adjust if your PLIP wrapper differs.
    if not isinstance(x, (list, tuple)):
        # If x is image tensor
        x_feat = adapter.model.get_image_embeddings(x)
    else:
        # If x is list of paths / PIL
        x_feat = adapter.model.encode_image(x)

    # Ensure gallery on same device
    G = G.to(x_feat.device)

    # Normalize
    x_norm = l2_normalize(x_feat)
    G_norm = l2_normalize(G)

    # ---- Similarity matrix to optimize ----
    if adapter.use_acs:
        # 1) pick attributes from atomic prompt bank
        selected = adapter.predict_attrs_for_batch(x_norm)

        # 2) compute ACS similarity
        S = adapter.acs_similarity(x_norm, G_norm, selected)
    else:
        # fallback to plain dot similarity
        S = x_norm @ G_norm.t()
    # breakpoint()
    # ---- Convert to probability over gallery and do entropy-min ----
    # softmax over gallery
    p = torch.softmax(S / adapter.id_temp, dim=1)

    # Top-k entropy (renormalize top-k)
    vals, idx = topk_rowwise(p, adapter.topk)
    vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-12)
    loss = entropy_from_probs(vals)

    loss.backward()
    adapter.optimizer.step()
    adapter.optimizer.zero_grad()

    # return features + loss (match your API)
    if kwargs.get("hybrid_eval", False):
        # return p.detach(), x_feat.detach(), float(loss.item())
        return S.detach(), x_feat.detach(), float(loss.item())
    else:
        return x_feat.detach(), float(loss.item())

