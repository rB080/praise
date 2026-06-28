"""
ATTR_TTA with a single toggle to switch similarity computation:

  sim_mode = "acs"    -> ACS subspace similarity (atomic attribute prompts)
  sim_mode = "legacy" -> old A_id + alpha * A_att (prompt-bank fusion)
  sim_mode = "plain"  -> cosine similarity only

Also includes adapt_branch:
  adapt_branch = "image" -> adapt image_encoder LN/BN (skip text_encoder)
  adapt_branch = "text"  -> adapt text_encoder LN/BN (skip image_encoder)

Keep your repo-specific imports (Create_PLIP_Model, Test_parse_args).
"""

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
from backbones.PLIP.PLIPmodel import Create_PLIP_Model
from utils.utils import Test_parse_args


# ============================================================
# 1) Atomic attribute prompts (ACS uses these)
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

_prompt_set: Dict[str, List[str]] = {
    "age": [_prompt_templates["age"].format(p) for p in _prompts["age"]],
    "gender": [_prompt_templates["gender"].format(p) for p in _prompts["gender"]],
    "hair": [_prompt_templates["hair"].format(p) for p in _prompts["hair"]],
    "ubcl": [_prompt_templates["ubcl"].format(p) for p in _prompts["ubcl"]],
    "ubcc": [_prompt_templates["ubcc"].format(p) for p in _prompts["ubcc"]],
    "lbcl": [_prompt_templates["lbcl"].format(p) for p in _prompts["lbcl"]],
    "lbcc": [_prompt_templates["lbcc"].format(p) for p in _prompts["lbcc"]],
    "lbct": [_prompt_templates["lbct"].format(p) for p in _prompts["lbct"]],
    "accessories": (
        [_prompt_templates["accessories"].format(p) for p in _prompts["accessories"]]
        + ["A photo of a person without any accessories."]
    )
}


# ============================================================
# 2) Optional: build the old combinatorial prompt list (Legacy can use this)
# ============================================================
def build_combinatorial_prompt_list() -> List[str]:
    prompt_list: List[str] = []
    colours = ["black", "blue", "brown", "green", "gray", "orange", "pink", "purple", "red", "white", "yellow"]
    for g, pronoun in {"man": "He", "woman": "She"}.items():
        for h in ["", "with long hair", "with hair", "with short hair"]:
            for l in ["sp", "d", "sk"]:
                for c in colours:
                    if l != "d":
                        for cc in colours:
                            for ccc in ["dark coloured shoes", "light coloured shoes"]:
                                prompt_list.append(
                                    f"A {g} {h} wearing a {c} top. {pronoun} is also wearing "
                                    f"{cc} {'pants' if l == 'sp' else 'skirt'} and {ccc}."
                                )
                    else:
                        for ccc in ["dark coloured shoes", "light coloured shoes"]:
                            prompt_list.append(
                                f"A {g} {h} wearing a {c} dress. {pronoun} is also wearing {ccc}."
                            )
    return prompt_list


# ============================================================
# 3) Utilities
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
    print(f"Disabled dropout in {count} modules.")


# ============================================================
# 4) Main class
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

        # ---- Core toggle ----
        sim_mode: str = "acs",  # "acs" | "legacy" | "plain"

        # ---- Which branch to adapt (important for true revertibility) ----
        adapt_branch: str = "image",  # "image" | "text"

        # ---- Legacy knobs ----
        alpha: float = 0.2,
        att_temp: float = 0.7,
        zscore_text: bool = False,

        # ---- ACS knobs ----
        use_acs: bool = True,
        acs_groups: Optional[List[str]] = None,
        acs_kmax: int = 8,
        acs_alpha: float = 1.0,        # blend knob (see acs_similarity)
        acs_conf_thresh: float = 0.0,
        acs_use_residual: bool = False,
        neutral_text: str = "A photo of a person.",
        zscore_text_logits: bool = False,
    ):
        super().__init__()
        assert sim_mode in ["acs", "legacy", "plain"], f"bad sim_mode: {sim_mode}"
        assert adapt_branch in ["image", "text"], f"bad adapt_branch: {adapt_branch}"

        self.episodic = episodic
        self.steps = steps
        self.topk = topk
        self.id_temp = id_temp
        self.device = device
        self.lr = lr

        self.sim_mode = sim_mode
        self.adapt_branch = adapt_branch

        # Legacy
        self.alpha = alpha
        self.att_temp = att_temp
        self.zscore_text = zscore_text

        # ACS
        self.acs_groups = acs_groups if acs_groups is not None else ["gender", "hair", "ubcc", "lbcc", "accessories"]
        self.acs_kmax = acs_kmax
        self.acs_alpha = acs_alpha
        self.acs_conf_thresh = acs_conf_thresh
        self.acs_use_residual = acs_use_residual
        self.neutral_text = neutral_text
        self.zscore_text_logits = zscore_text_logits

        # Internal
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.model_state = None
        self.optimizer_state = None

        # Tokenizer
        self.TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

        # Args
        self.args = Test_parse_args(notebook=True)

        # ACS banks
        self.attr_prompt_feats: Dict[str, torch.Tensor] = {}
        self.attr_dirs: Dict[str, torch.Tensor] = {}
        self.attr_strings: Dict[str, List[str]] = {}
        self.neutral_text_feat: Optional[torch.Tensor] = None

        # Legacy bank
        self.preset_text_features: Optional[torch.Tensor] = None  # (P,d), usually stored on CPU
        self.prompt_list: Optional[List[str]] = None

    # -----------------------------
    # Model init + tokenizer
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
    # Param collection / reset
    # -----------------------------
    def collect_params(self, disable_dropouts: bool = True) -> Tuple[List[nn.Parameter], List[str]]:
        """
        Collect only LN/BN affine params (weight, bias).
        adapt_branch:
          - "image": skip text_encoder modules
          - "text" : skip image_encoder modules
        """
        assert self.model is not None
        params: List[nn.Parameter] = []
        names: List[str] = []

        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                # Skip by branch
                root = nm.split(".")[0]
                if self.adapt_branch == "image" and root == "text_encoder":
                    continue
                if self.adapt_branch == "text" and root == "image_encoder":
                    continue

                for np, p in m.named_parameters(recurse=False):
                    if np in ["weight", "bias"]:
                        params.append(p)
                        names.append(f"{nm}.{np}")

        if disable_dropouts:
            disable_dropout(self.model)

        return params, names

    def copy_model_and_optimizer(self):
        assert self.model is not None and self.optimizer is not None
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def load_model_and_optimizer(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("reset called before initialize()")
        assert self.model is not None and self.optimizer is not None
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def initialize(self, gallery_feats: torch.Tensor):
        """
        Set optimizer over LN/BN params only.
        """
        assert self.model is not None, "Call init_model() first."
        params, names = self.collect_params()
        print(f"Optimizing {len(params)} params (LN/BN affine only), adapt_branch={self.adapt_branch}, sim_mode={self.sim_mode}.")
        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, weight_decay=1e-4)
        self.copy_model_and_optimizer()

        # Warnings to avoid silent "it doesn't work" states
        if self.sim_mode == "acs" and len(self.attr_dirs) == 0:
            print("WARNING: sim_mode='acs' but ACS bank is empty. Call precompute_attr_bank() once.")
        if self.sim_mode == "legacy" and self.preset_text_features is None:
            print("WARNING: sim_mode='legacy' but legacy text bank is empty. Call precompute_legacy_text_bank() once.")

    def reset(self):
        self.load_model_and_optimizer()

    # ============================================================
    # ACS bank: build once
    # ============================================================
    @torch.no_grad()
    def precompute_attr_bank(
        self,
        prompt_set: Dict[str, List[str]] = _prompt_set,
        chunksize: int = 64,
    ):
        """
        Creates:
            attr_prompt_feats[group]: (nvals, d) normalized prompt embeddings
            attr_dirs[group]: (nvals, d) normalized direction vectors (prompt - neutral)
        """
        assert self.model is not None, "Call init_model() first."
        self.model.eval()
        device = torch.device(self.device)

        # neutral text embedding
        ti, tm = self.tokenize([self.neutral_text])
        tf0, _ = self.model.text_encoder(ti.to(device), tm.to(device))
        tf0 = l2_normalize(tf0.squeeze(0))
        self.neutral_text_feat = tf0

        self.attr_prompt_feats.clear()
        self.attr_dirs.clear()
        self.attr_strings.clear()

        for group in self.acs_groups:
            prompts = prompt_set[group]
            feats = []
            for i in tqdm(range(0, len(prompts), chunksize), desc=f"ACS text bank: {group}"):
                chunk = prompts[i:i + chunksize]
                ti, tm = self.tokenize(chunk)
                tf, _ = self.model.text_encoder(ti.to(device), tm.to(device))
                feats.append(tf.detach())
            tf = torch.cat(feats, dim=0)          # (nvals, d)
            tf = l2_normalize(tf)
            dirs = l2_normalize(tf - tf0.unsqueeze(0))

            self.attr_prompt_feats[group] = tf
            self.attr_dirs[group] = dirs
            self.attr_strings[group] = prompts

        print("Built ACS attribute bank groups:", list(self.attr_dirs.keys()))

    # ============================================================
    # Legacy bank: build once
    # ============================================================
    @torch.no_grad()
    def precompute_legacy_text_bank(
        self,
        prompt_list: List[str],
        chunksize: int = 64,
        neutral_text: Optional[str] = None,
        store_on_cpu: bool = True,
    ):
        """
        Builds preset_text_features: (P,d) normalized prompt embeddings.
        If neutral_text is given, centers by (prompt - neutral) then renormalizes.
        """
        assert self.model is not None, "Call init_model() first."
        self.model.eval()
        device = torch.device(self.device)

        feats = []
        for i in tqdm(range(0, len(prompt_list), chunksize), desc="Legacy text bank"):
            chunk = prompt_list[i:i + chunksize]
            ti, tm = self.tokenize(chunk)
            tf, _ = self.model.text_encoder(ti.to(device), tm.to(device))
            feats.append(tf.detach())

        Zt = torch.cat(feats, dim=0)  # (P,d)
        Zt = l2_normalize(Zt)

        if neutral_text is not None:
            ti, tm = self.tokenize([neutral_text])
            tf0, _ = self.model.text_encoder(ti.to(device), tm.to(device))
            tf0 = l2_normalize(tf0.squeeze(0))
            Zt = l2_normalize(Zt - tf0.unsqueeze(0))
            print("Legacy bank: using neutral centering:", neutral_text)

        if store_on_cpu:
            Zt = Zt.cpu()

        self.preset_text_features = Zt
        self.prompt_list = prompt_list
        print("Built legacy text bank:", self.preset_text_features.shape)

    # ============================================================
    # ACS: attribute selection
    # ============================================================
    def predict_attrs_for_batch(self, x_norm: torch.Tensor) -> List[List[Tuple[str, int, float]]]:
        """
        For each query in batch:
            pick top-1 value per group by prompt similarity.
        Returns:
            selected[b] = [(group, idx, conf), ...]
        """
        B, _ = x_norm.shape
        selected: List[List[Tuple[str, int, float]]] = [[] for _ in range(B)]

        for group in self.acs_groups:
            tf = self.attr_prompt_feats[group]  # (nvals, d)
            logits = x_norm @ tf.t()            # (B, nvals)

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

    # ============================================================
    # ACS: subspace similarity
    # ============================================================
    def acs_similarity(
        self,
        x_norm: torch.Tensor,     # (B, d)
        G_norm: torch.Tensor,     # (Ng, d)
        selected: List[List[Tuple[str, int, float]]],
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Per query:
          - build Q from selected attribute directions
          - S_sub = cosine in subspace coords
          - optionally residual or blend
        Returns:
            S: (B, Ng)
        """
        device = x_norm.device
        dtype = x_norm.dtype
        B, _ = x_norm.shape
        Ng = G_norm.shape[0]
        out = torch.empty((B, Ng), device=device, dtype=dtype)

        for b in range(B):
            items = sorted(selected[b], key=lambda t: t[2], reverse=True)[: self.acs_kmax]

            # Fallback
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

            Q, _ = torch.linalg.qr(U, mode="reduced")  # (d, k)

            S_raw = (x_norm[b:b+1] @ G_norm.t())       # (1, Ng)

            cq = (x_norm[b:b+1] @ Q)                   # (1, k)
            cG = (G_norm @ Q)                          # (Ng, k)

            cq = l2_normalize(cq, eps=eps)
            cG = l2_normalize(cG, eps=eps)
            S_sub = (cq @ cG.t())                      # (1, Ng)

            if self.acs_use_residual:
                # remove subspace contribution
                S = S_raw - ((x_norm[b:b+1] @ Q) @ (G_norm @ Q).t())
            else:
                # If you truly want blending, use convex combo.
                # Your earlier code used S_raw + alpha*S_sub; keep that if desired.
                if self.acs_alpha < 1.0:
                    S = (1.0 - self.acs_alpha) * S_raw + self.acs_alpha * S_sub
                else:
                    S = S_sub

            out[b] = S.squeeze(0)

        return out

    # ============================================================
    # Legacy similarity (old A_id + alpha*A_att)
    # ============================================================
    def legacy_similarity(self, x_norm: torch.Tensor, G_norm: torch.Tensor) -> torch.Tensor:
        """
        Matches your old behavior:
          A_id  = x @ G^T
          Ab    = x @ Zt^T
          Ag    = G @ Zt^T
          A_att = Ab @ Ag^T
          P     = softmax(A_id/t_id) + alpha*softmax(A_att/t_att)
          Return S = log(P) so the rest of the pipeline (softmax over gallery) still works.
        """
        if self.preset_text_features is None:
            raise RuntimeError("Legacy mode requires preset_text_features. Call precompute_legacy_text_bank() once.")

        Zt = self.preset_text_features.to(device=x_norm.device, dtype=x_norm.dtype)  # (P,d)

        A_id = x_norm @ G_norm.t()             # (B,Ng)
        Ab_logits = x_norm @ Zt.t()            # (B,P)
        Ag_logits = G_norm @ Zt.t()            # (Ng,P)

        if self.zscore_text:
            mu = Ag_logits.mean(dim=0, keepdim=True)
            sd = Ag_logits.std(dim=0, keepdim=True) + 1e-6
            Ab_logits = (Ab_logits - mu) / sd
            Ag_logits = (Ag_logits - mu) / sd

        A_att = Ab_logits @ Ag_logits.t()      # (B,Ng)

        P_id = torch.softmax(A_id / self.id_temp, dim=1)
        P_att = torch.softmax(A_att / self.att_temp, dim=1)

        P = P_id + self.alpha * P_att          # (B,Ng)
        S = torch.log(P + 1e-12)
        return S

    # ============================================================
    # Forward + adapt loop
    # ============================================================
    def forward(self, x, G: torch.Tensor, **kwargs):
        if self.episodic:
            self.reset()

        out = None
        for _ in range(self.steps):
            out = forward_and_adapt(self, x, G, **kwargs)
        return out


# ============================================================
# 5) Adapt step
# ============================================================
@torch.enable_grad()
def forward_and_adapt(adapter: ATTR_TTA, x, G: torch.Tensor, **kwargs):
    """
    Performs one TTA step:
      - encode image features (your PLIP wrapper must support get_image_embeddings or encode_image)
      - compute similarity S depending on sim_mode
      - entropy-min over top-k gallery probs
      - update LN/BN affine params
    """
    assert adapter.model is not None
    assert adapter.optimizer is not None

    adapter.model.train()

    # ---- Encode text ----
    x, xmask = adapter.tokenize(x)
    x, _ = adapter.model.text_encoder(x.to(adapter.device), xmask.to(adapter.device))
    x_feat = x

    # Ensure gallery on same device
    G = G.to(x_feat.device)

    # Normalize
    x_norm = l2_normalize(x_feat)
    G_norm = l2_normalize(G)

    # ---- Similarity matrix ----
    if adapter.sim_mode == "acs":
        selected = adapter.predict_attrs_for_batch(x_norm)
        S = adapter.acs_similarity(x_norm, G_norm, selected)
    elif adapter.sim_mode == "legacy":
        S = adapter.legacy_similarity(x_norm, G_norm)
    else:  # "plain"
        S = x_norm @ G_norm.t()

    # ---- Convert to probability over gallery and do entropy-min ----
    p = torch.softmax(S / adapter.id_temp, dim=1)

    vals, idx = topk_rowwise(p, adapter.topk)
    vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-12)
    loss = entropy_from_probs(vals)

    loss.backward()
    adapter.optimizer.step()
    adapter.optimizer.zero_grad()

    # return features + loss (match your API)
    if kwargs.get("hybrid_eval", False):
        return p.detach(), x_feat.detach(), float(loss.item())
    else:
        return x_feat.detach(), float(loss.item())


# ============================================================
# 6) Example usage (commented)
# ============================================================
# tta = ATTR_TTA(sim_mode="acs", adapt_branch="image", steps=1, topk=5, id_temp=0.6, device="cuda:0")
# tta.init_model()
# tta.precompute_attr_bank()  # needed for ACS
# tta.initialize(gallery_feats)
#
# # Legacy:
# tta = ATTR_TTA(sim_mode="legacy", adapt_branch="text", alpha=0.2, att_temp=0.7, zscore_text=True, device="cuda:0")
# tta.init_model()
# legacy_prompts = build_combinatorial_prompt_list()
# tta.precompute_legacy_text_bank(legacy_prompts, neutral_text="A photo of a person.")
# tta.initialize(gallery_feats)
