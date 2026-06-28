"""
attr_plip_tpt_t2i.py

TPT (Test-Time Prompt Tuning) adapted for Text-to-Image (T2I) ReID using PLIP.

Setting:
  - Query: captions (text)
  - Gallery: images (precomputed image embeddings)

What gets adapted:
  - A small low-rank prompt adapter that perturbs an attribute prompt bank.
  - PLIP encoders are frozen.

TPT objective (text-side views):
  - Create M "views" of each caption via random token masking and/or template wrappers
  - Compute posterior q^{(m)}(prompt | caption_view)
  - Minimize marginal entropy H( mean_m q^{(m)} ) (optionally marginal - conditional)

Retrieval scoring:
  S = S_id + att_weight * S_att
    - S_id: cosine similarity between caption embedding and image embedding
    - S_att: similarity between attribute posteriors q_text and q_image
            where q_image is computed from image->prompt posterior under tuned prompt bank

Designed for your T2I runner style:
  - class name ATTR_TTA
  - exposes init_model(), tokenize(), precompute_prompt_bank(), initialize(), forward(captions, G, hybrid_eval=...)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer

# Project imports (keep as in your repo)
from backbones.PLIP.PLIPmodel import Create_PLIP_Model
from utils.utils import Test_parse_args


# ------------------------------------------------------------
# Atomic prompt bank (same idea as ACS/TPT baselines)
# ------------------------------------------------------------
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

_prompt_set = {k: [_prompt_templates[k].format(p) for p in _prompts.keys() for _ in [0]] for k in []}  # dummy to keep linters happy
_prompt_set = {k: [_prompt_templates[k].format(p) for p in _prompts[k]] for k in _prompts.keys()}
_prompt_set["accessories"] = _prompt_set["accessories"] + ["A photo of a person without any accessories."]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Shannon entropy per row, averaged over batch.
    p: (B, K) probabilities.
    """
    p = torch.clamp(p, min=eps)
    return -(p * torch.log(p)).sum(dim=1).mean()


# ------------------------------------------------------------
# Prompt adapter (low-rank additive)
#   Z_t = normalize(Z0 + A @ B)
#   A: (L, r), B: (r, d)
# ------------------------------------------------------------
class LowRankPromptAdapter(nn.Module):
    def __init__(self, num_prompts: int, dim: int, rank: int = 4, init_scale: float = 1e-3):
        super().__init__()
        self.A = nn.Parameter(torch.randn(num_prompts, rank) * init_scale)
        self.B = nn.Parameter(torch.randn(rank, dim) * init_scale)

    def forward(self, Z0: torch.Tensor) -> torch.Tensor:
        delta = self.A @ self.B  # (L, d)
        return Z0 + delta


# ============================================================
# TPT for T2I ReID (caption query, image gallery)
# ============================================================
class ATTR_TTA(nn.Module):
    """
    ReID-oriented TPT baseline adapted to T2I:
      - Freeze PLIP encoders
      - Tune low-rank prompt adapter on attribute prompt bank at test time
      - Use TPT marginal entropy across *text views* (caption perturbations)

    Forward returns:
      - (text_feats, loss) by default
      - (S_or_P, text_feats, loss) if hybrid_eval=True
    """

    DEFAULT_TEXT_TEMPLATES = [
        "{}",                      # raw caption
        "A photo of a person. {}", # mild prefix
        "This person: {}",
    ]

    def __init__(
        self,
        steps: int = 10,
        device: str = "cuda:0",
        lr: float = 1e-3,
        episodic: bool = True,

        # TPT knobs
        num_views: int = 16,
        view_select_frac: float = 0.5,
        text_temp: float = 0.01,
        tpt_loss: str = "marginal_entropy",

        # Text-view augmentation knobs
        view_mode: str = "mask",            # "mask", "templates", "mix"
        mask_prob: float = 0.15,
        text_templates: Optional[List[str]] = None,

        # Retrieval scoring knobs
        id_temp: float = 2.0,
        att_weight: float = 0.3,
        gallery_chunk: int = 8192,

        # Prompt bank knobs
        prompt_groups: Optional[List[str]] = None,
        neutral_text: str = "A photo of a person.",

        # Prompt adapter
        prompt_adapter_rank: int = 4,

        # hybrid output
        return_probs: bool = True,
    ):
        super().__init__()
        self.steps = int(steps)
        self.device = str(device)
        self.lr = float(lr)
        self.episodic = bool(episodic)

        self.num_views = int(num_views)
        self.view_select_frac = float(view_select_frac)
        self.text_temp = float(text_temp)
        self.tpt_loss = str(tpt_loss)

        self.view_mode = str(view_mode)
        assert self.view_mode in ["mask", "templates", "mix"], "view_mode must be one of: mask/templates/mix"
        self.mask_prob = float(mask_prob)
        self.text_templates = list(text_templates) if text_templates is not None else list(self.DEFAULT_TEXT_TEMPLATES)

        self.id_temp = float(id_temp)
        self.att_weight = float(att_weight)
        self.gallery_chunk = int(gallery_chunk)
        self.return_probs = bool(return_probs)

        self.prompt_groups = prompt_groups if prompt_groups is not None else ["gender", "hair", "ubcc", "lbcc", "accessories"]
        self.neutral_text = str(neutral_text)
        self.prompt_adapter_rank = int(prompt_adapter_rank)

        # Model + tokenizer
        self.args = Test_parse_args(notebook=True)
        self.TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model: Optional[nn.Module] = None

        # Prompt bank buffers
        self.prompts: List[str] = []
        self.Z0: Optional[torch.Tensor] = None  # (L, d) centered base prompt bank
        self.neutral_feat: Optional[torch.Tensor] = None

        # Prompt adapter (learned at test time)
        self.adapter: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._adapter_state = None
        self._opt_state = None

    # -----------------------------
    # Model init
    # -----------------------------
    def init_model(self):
        model = Create_PLIP_Model(self.args).to(self.device)
        checkpoint = torch.load(self.args.model_path, map_location="cpu")
        model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
        model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"], strict=False)
        self.model = model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad_(False)

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
    # Prompt bank precompute
    # -----------------------------
    @torch.no_grad()
    def precompute_prompt_bank(self, prompt_set: Dict[str, List[str]] = _prompt_set, chunksize: int = 64):
        assert self.model is not None, "Call init_model() first."
        was_training = self.model.training
        self.model.eval()

        prompts: List[str] = []
        for g in self.prompt_groups:
            prompts.extend(prompt_set[g])
        self.prompts = prompts

        device = torch.device(self.device)

        ti, tm = self.tokenize([self.neutral_text])
        tf0, _ = self.model.text_encoder(ti.to(device), tm.to(device))
        tf0 = l2_normalize(tf0.squeeze(0))
        self.neutral_feat = tf0

        feats = []
        for i in tqdm(range(0, len(prompts), chunksize), desc="TPT prompt bank (T2I)"):
            chunk = prompts[i:i + chunksize]
            ti, tm = self.tokenize(chunk)
            tf, _ = self.model.text_encoder(ti.to(device), tm.to(device))
            feats.append(tf.detach())
        Z = torch.cat(feats, dim=0)  # (L, d)
        Z = l2_normalize(Z)
        Z = l2_normalize(Z - tf0.unsqueeze(0))  # center
        self.Z0 = Z.detach().clone()

        L, d = self.Z0.shape
        self.adapter = LowRankPromptAdapter(L, d, rank=self.prompt_adapter_rank).to(device)
        print(f"[TPT-T2I] Prompt bank built: L={L}, d={d}, groups={self.prompt_groups}")

        if was_training:
            self.model.train()

    # -----------------------------
    # Optimizer setup + episodic reset
    # -----------------------------
    def initialize(self):
        assert self.adapter is not None, "Call precompute_prompt_bank() first."
        self.optimizer = torch.optim.Adam(params=list(self.adapter.parameters()), lr=self.lr, weight_decay=0.0)
        self._adapter_state = deepcopy(self.adapter.state_dict())
        self._opt_state = deepcopy(self.optimizer.state_dict())

    def reset(self):
        if self._adapter_state is None or self._opt_state is None:
            raise RuntimeError("reset called before initialize()")
        assert self.adapter is not None and self.optimizer is not None
        self.adapter.load_state_dict(self._adapter_state, strict=True)
        self.optimizer.load_state_dict(self._opt_state)

    # -----------------------------
    # Tuned prompt bank
    # -----------------------------
    def tuned_prompt_bank(self) -> torch.Tensor:
        assert self.adapter is not None and self.Z0 is not None
        Z = self.adapter(self.Z0)
        return l2_normalize(Z)

    # -----------------------------
    # Text encoding (frozen)
    # -----------------------------
    @torch.no_grad()
    def encode_text_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        tf, _ = self.model.text_encoder(input_ids, attention_mask)
        return tf

    @torch.no_grad()
    def encode_text(self, captions: List[str]) -> torch.Tensor:
        device = torch.device(self.device)
        ti, tm = self.tokenize(captions)
        tf = self.encode_text_tokens(ti.to(device), tm.to(device))
        return tf

    # -----------------------------
    # Text views for TPT
    # -----------------------------
    def _apply_template(self, captions: List[str], template: str) -> List[str]:
        if "{}" in template:
            return [template.format(c) for c in captions]
        return [template + " " + c for c in captions]

    @torch.no_grad()
    def make_text_views(self, captions: List[str], max_len: int = 250) -> Tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(self.device)
        captions = list(captions)

        base_ids, base_mask = self.tokenize(captions, max_len=max_len)
        base_ids = base_ids.to(device)
        base_mask = base_mask.to(device)

        M = self.num_views
        view_ids = base_ids.unsqueeze(0).repeat(M, 1, 1).contiguous()
        view_mask = base_mask.unsqueeze(0).repeat(M, 1, 1).contiguous()

        cls_id = getattr(self.TOKENIZER, "cls_token_id", 101)
        sep_id = getattr(self.TOKENIZER, "sep_token_id", 102)
        pad_id = getattr(self.TOKENIZER, "pad_token_id", 0)
        mask_id = getattr(self.TOKENIZER, "mask_token_id", None)
        if mask_id is None:
            mask_id = getattr(self.TOKENIZER, "unk_token_id", 100)

        # template-based views (optionally)
        if self.view_mode in ["templates", "mix"]:
            n_tpl = M if self.view_mode == "templates" else max(1, M // 2)
            for m in range(n_tpl):
                template = self.text_templates[m % len(self.text_templates)]
                cap_m = self._apply_template(captions, template)
                ids_m, mask_m = self.tokenize(cap_m, max_len=max_len)
                view_ids[m] = ids_m.to(device)
                view_mask[m] = mask_m.to(device)

        # masking-based views (optionally)
        if self.view_mode in ["mask", "mix"]:
            start_m = 0 if self.view_mode == "mask" else max(1, M // 2)
            for m in range(start_m, M):
                ids = view_ids[m]
                am = view_mask[m]
                valid = (am == 1) & (ids != cls_id) & (ids != sep_id) & (ids != pad_id)
                r = torch.rand_like(ids.float())
                do_mask = valid & (r < self.mask_prob)
                ids = ids.clone()
                ids[do_mask] = mask_id
                view_ids[m] = ids

        return view_ids, view_mask

    # -----------------------------
    # TPT objective
    # -----------------------------
    def tpt_objective(self, captions: List[str]) -> torch.Tensor:
        assert self.Z0 is not None and self.adapter is not None
        view_ids, view_mask = self.make_text_views(captions)  # (M,B,L)
        M, B = view_ids.shape[:2]

        Zt = self.tuned_prompt_bank()  # (L,d)

        feats = []
        with torch.no_grad():
            for m in range(M):
                tfm = self.encode_text_tokens(view_ids[m], view_mask[m])  # (B,d)
                feats.append(tfm)
        Zv = torch.stack(feats, dim=0)  # (M,B,d)
        Zv = l2_normalize(Zv)

        logits = torch.einsum("mbd,ld->mbl", Zv, Zt)  # (M,B,L)
        q = torch.softmax(logits / self.text_temp, dim=-1)

        if self.view_select_frac < 1.0:
            ent = -(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum(dim=-1)  # (M,B)
            keep = max(1, int(round(self.view_select_frac * M)))
            idx = torch.topk(ent, k=keep, dim=0, largest=False).indices  # (keep,B)
            q = torch.gather(q, dim=0, index=idx.unsqueeze(-1).expand(-1, -1, q.size(-1)))

        q_bar = q.mean(dim=0)  # (B,L)
        loss_marg = entropy(q_bar)

        if self.tpt_loss == "marginal_entropy":
            return loss_marg
        if self.tpt_loss == "marginal_minus_conditional":
            loss_cond = entropy(q.reshape(-1, q.size(-1)))
            return loss_marg - loss_cond

        raise ValueError(f"Unknown tpt_loss={self.tpt_loss}")

    # -----------------------------
    # Retrieval scoring
    # -----------------------------
    @torch.no_grad()
    def reid_scores(self, q_feat: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        device = q_feat.device
        G = G.to(device)

        Zt = self.tuned_prompt_bank()  # (L,d)

        q_norm = l2_normalize(q_feat)
        G_norm = l2_normalize(G)

        S_id = q_norm @ G_norm.t()

        if self.att_weight == 0.0:
            return S_id

        q_post = torch.softmax((q_norm @ Zt.t()) / self.text_temp, dim=1)  # (B,L)

        g_posts = []
        for s in range(0, G_norm.size(0), self.gallery_chunk):
            e = min(s + self.gallery_chunk, G_norm.size(0))
            logits_g = (G_norm[s:e] @ Zt.t()) / self.text_temp
            g_posts.append(torch.softmax(logits_g, dim=1))
        g_post = torch.cat(g_posts, dim=0)  # (Ng,L)

        S_att = q_post @ g_post.t()
        return S_id + self.att_weight * S_att

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, captions: List[str], G: torch.Tensor, **kwargs):
        assert self.model is not None and self.Z0 is not None and self.adapter is not None, \
            "Call init_model() + precompute_prompt_bank() + initialize() first."
        assert self.optimizer is not None, "Call initialize() first."

        captions = list(captions)

        if self.episodic:
            self.reset()

        self.adapter.train()
        loss_val = 0.0
        for _ in range(self.steps):
            loss = self.tpt_objective(captions)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            loss_val = float(loss.item())

        with torch.no_grad():
            q_feat = self.encode_text(captions).to(torch.device(self.device))
            S = self.reid_scores(q_feat, G.to(torch.device(self.device)))
            out_pred = torch.softmax(S / self.id_temp, dim=1) if self.return_probs else S

        if kwargs.get("hybrid_eval", False):
            return out_pred.detach().cpu(), q_feat.detach().cpu(), loss_val
        return q_feat.detach().cpu(), loss_val
