from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm
from transformers import BertTokenizer

# Project imports (keep as in your repo)
from backbones.PLIP.PLIPmodel import Create_PLIP_Model
from utils.utils import Test_parse_args


# ------------------------------------------------------------
# Atomic prompt bank (recommended for ACS/TPT baselines)
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
    "accessories": "A photo of a person with {}."
}

_prompt_set = {
    k: [_prompt_templates[k].format(p) for p in _prompts[k]]
    for k in _prompts.keys()
}
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
# Prompt adapter (low-rank additive, small parameter count)
# Z_t = normalize(Z0 + A @ B)
# A: (L, r), B: (r, d)
# ------------------------------------------------------------
class LowRankPromptAdapter(nn.Module):
    def __init__(self, num_prompts: int, dim: int, rank: int = 4, init_scale: float = 1e-3):
        super().__init__()
        self.A = nn.Parameter(torch.randn(num_prompts, rank) * init_scale)
        self.B = nn.Parameter(torch.randn(rank, dim) * init_scale)

    def forward(self, Z0: torch.Tensor) -> torch.Tensor:
        # Z0: (L, d)
        delta = self.A @ self.B  # (L, d)
        return Z0 + delta


# ============================================================
# ReID-oriented TPT baseline
# ============================================================
class TPT_REID(nn.Module):
    """
    A ReID-oriented version of Test-Time Prompt Tuning (TPT):
      - freeze VLM (image+text encoders)
      - tune a small prompt adapter on a prompt bank (attributes)
      - use TPT marginal entropy across augmentations

    For evaluation, compute retrieval scores:
      S = S_id + att_weight * S_att
      where S_att is similarity between attribute posteriors q(x) and q(g).
    """
    def __init__(
        self,
        steps: int = 10,
        device: str = "cuda:0",
        lr: float = 1e-3,
        episodic: bool = True,

        # TPT knobs
        num_views: int = 16,
        view_select_frac: float = 0.5,   # keep lowest-entropy views (per image)
        text_temp: float = 0.01,         # temperature for image->prompt posterior
        tpt_loss: str = "marginal_entropy",  # or "marginal_minus_conditional"

        # ReID scoring knobs
        id_temp: float = 2.0,
        att_weight: float = 0.3,         # fusion weight for attribute similarity
        gallery_chunk: int = 8192,

        # Prompt bank knobs
        prompt_groups: Optional[List[str]] = None,
        neutral_text: str = "A photo of a person.",

        # Prompt adapter
        prompt_adapter_rank: int = 4,

        # Augmentations (you can swap this to match your setup)
        img_size: Tuple[int, int] = (256, 128),
    ):
        super().__init__()
        self.steps = steps
        self.device = device
        self.lr = lr
        self.episodic = episodic

        self.num_views = num_views
        self.view_select_frac = view_select_frac
        self.text_temp = text_temp
        self.tpt_loss = tpt_loss

        self.id_temp = id_temp
        self.att_weight = att_weight
        self.gallery_chunk = gallery_chunk

        self.prompt_groups = prompt_groups if prompt_groups is not None else ["gender", "hair", "ubcc", "lbcc", "accessories"]
        self.neutral_text = neutral_text
        self.prompt_adapter_rank = prompt_adapter_rank

        # Model + tokenizer
        self.args = Test_parse_args(notebook=True)
        self.TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = None

        # Prompt bank buffers
        self.prompts: List[str] = []
        self.Z0: Optional[torch.Tensor] = None  # (L, d) base prompt embeddings (centered)
        self.neutral_feat: Optional[torch.Tensor] = None

        # Prompt adapter (learned at test time)
        self.adapter: Optional[nn.Module] = None
        self.optimizer = None
        self._adapter_state = None
        self._opt_state = None

        # Tensor-friendly test-time augmentation
        H, W = img_size
        self.aug = T.Compose([
            T.RandomResizedCrop((H, W), scale=(0.7, 1.0), ratio=(0.75, 1.33), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        ])

    # -----------------------------
    # Model init
    # -----------------------------
    def init_model(self):
        model = Create_PLIP_Model(self.args).to(self.device)
        checkpoint = torch.load(self.args.model_path, map_location="cpu")
        model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
        model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"], strict=False)
        self.model = model.to(self.device)

        # Freeze EVERYTHING (TPT tunes prompts only)
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
        # flatten prompts
        prompts = []
        for g in self.prompt_groups:
            prompts.extend(prompt_set[g])
        self.prompts = prompts

        device = torch.device(self.device)

        # neutral embedding (for centering)
        ti, tm = self.tokenize([self.neutral_text])
        tf0, _ = self.model.text_encoder(ti.to(device), tm.to(device))
        tf0 = l2_normalize(tf0.squeeze(0))
        self.neutral_feat = tf0

        feats = []
        for i in tqdm(range(0, len(prompts), chunksize), desc="TPT prompt bank"):
            chunk = prompts[i:i + chunksize]
            ti, tm = self.tokenize(chunk)
            tf, _ = self.model.text_encoder(ti.to(device), tm.to(device))
            feats.append(tf.detach())
        Z = torch.cat(feats, dim=0)  # (L, d)
        Z = l2_normalize(Z)

        # Center prompts (optional but usually helps)
        Z = l2_normalize(Z - tf0.unsqueeze(0))

        self.Z0 = Z.detach().clone()  # keep as base bank

        # Create prompt adapter now that we know dims
        L, d = self.Z0.shape
        self.adapter = LowRankPromptAdapter(L, d, rank=self.prompt_adapter_rank).to(device)

        print(f"Prompt bank built: L={L}, d={d}, groups={self.prompt_groups}")
        if was_training:
            print("Debug Notice: Model was indeed in train mode.")
            self.model.train()
        

    # -----------------------------
    # Optimizer setup + episodic reset
    # -----------------------------
    def initialize(self):
        assert self.adapter is not None, "Call precompute_prompt_bank() first."
        params = list(self.adapter.parameters())
        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, weight_decay=0.0)
        self._adapter_state = deepcopy(self.adapter.state_dict())
        self._opt_state = deepcopy(self.optimizer.state_dict())

    def reset(self):
        if self._adapter_state is None or self._opt_state is None:
            raise RuntimeError("reset called before initialize()")
        self.adapter.load_state_dict(self._adapter_state, strict=True)
        self.optimizer.load_state_dict(self._opt_state)

    # -----------------------------
    # Prompt bank (tuned)
    # -----------------------------
    def tuned_prompt_bank(self) -> torch.Tensor:
        """
        Returns Z_t: (L, d), L2-normalized.
        """
        Z = self.adapter(self.Z0)  # (L, d)
        Z = l2_normalize(Z)
        return Z

    # -----------------------------
    # Image encoder wrapper
    # -----------------------------
    @torch.no_grad()
    def encode_images(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) tensor on device
        returns: (B, d)
        """
        feats = self.model.get_image_embeddings(x)
        return feats

    def make_views(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create M augmented views:
          returns: (M, B, C, H, W)
        Note: torchvision transforms operate per-image; we loop.
        """
        assert x.dim() == 4, "Expected (B,C,H,W) tensor"
        B = x.size(0)
        device = x.device
        views = []
        # Move to CPU for torchvision transforms if needed; keep simple and robust:
        x_cpu = x.detach().cpu()
        for _ in range(self.num_views):
            v = torch.stack([self.aug(x_cpu[i]) for i in range(B)], dim=0)  # (B,C,H,W)
            views.append(v)
        views = torch.stack(views, dim=0).to(device)  # (M,B,C,H,W)
        return views

    # -----------------------------
    # TPT loss on prompt posterior
    # -----------------------------
    def tpt_objective(self, x_views: torch.Tensor) -> torch.Tensor:
        """
        x_views: (M,B,C,H,W)
        Implements TPT marginal entropy (with optional view selection).
        """
        M, B = x_views.shape[:2]
        device = x_views.device
        
        Zt = self.tuned_prompt_bank()  # (L,d)
        # Encode each view (no grad through encoders; grad flows to adapter via Zt)
        feats = []
        with torch.no_grad():
            for m in range(M):
                fm = self.encode_images(x_views[m])  # (B,d)
                feats.append(fm)
        Zv = torch.stack(feats, dim=0)  # (M,B,d)
        Zv = l2_normalize(Zv)
        # breakpoint()
        # Prompt posterior q^{(m)} = softmax(Zv @ Zt^T / tau)
        logits = torch.einsum("mbd,ld->mbl", Zv, Zt)  # (M,B,L)
        q = torch.softmax(logits / self.text_temp, dim=-1)  # (M,B,L)

        # View selection: keep lowest-entropy views per sample
        if self.view_select_frac < 1.0:
            # ent: (M,B)
            ent = -(q.clamp_min(1e-12) * q.clamp_min(1e-12).log()).sum(dim=-1)
            keep = max(1, int(round(self.view_select_frac * M)))
            # indices: (keep,B)
            idx = torch.topk(ent, k=keep, dim=0, largest=False).indices
            # gather -> (keep,B,L)
            q = torch.gather(q, dim=0, index=idx.unsqueeze(-1).expand(-1, -1, q.size(-1)))

        # marginal entropy: H(mean_m q^{(m)})
        q_bar = q.mean(dim=0)  # (B,L)
        loss_marg = entropy(q_bar)

        if self.tpt_loss == "marginal_entropy":
            return loss_marg

        if self.tpt_loss == "marginal_minus_conditional":
            # conditional entropy term encourages each view to be confident too
            loss_cond = entropy(q.reshape(-1, q.size(-1)))
            return loss_marg - loss_cond

        raise ValueError(f"Unknown tpt_loss={self.tpt_loss}")

    # -----------------------------
    # ReID scoring using tuned prompts
    # -----------------------------
    @torch.no_grad()
    def reid_scores(self, x_feat: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        x_feat: (B,d) query features
        G:     (Ng,d) gallery features
        Returns:
          S: (B,Ng) similarity scores (higher is better)
        """
        device = x_feat.device
        G = G.to(device)

        Zt = self.tuned_prompt_bank()  # (L,d)

        x_norm = l2_normalize(x_feat)
        G_norm = l2_normalize(G)

        # Identity similarity
        S_id = (x_norm @ G_norm.t())  # (B,Ng)

        # Attribute posterior for queries
        q_b = torch.softmax((x_norm @ Zt.t()) / self.text_temp, dim=1)  # (B,L)

        # Attribute posterior for gallery (chunked)
        qg_chunks = []
        for s in range(0, G_norm.size(0), self.gallery_chunk):
            e = min(s + self.gallery_chunk, G_norm.size(0))
            logits_g = (G_norm[s:e] @ Zt.t()) / self.text_temp  # (chunk,L)
            qg = torch.softmax(logits_g, dim=1)                 # (chunk,L)
            qg_chunks.append(qg)
        q_g = torch.cat(qg_chunks, dim=0)  # (Ng,L)

        # Attribute similarity: dot product of posteriors
        S_att = q_b @ q_g.t()  # (B,Ng)

        # Final fused similarity
        S = S_id + self.att_weight * S_att
        return S

    # -----------------------------
    # Forward (TTA)
    # -----------------------------
    def forward(self, x: torch.Tensor, G: torch.Tensor, **kwargs):
        """
        Returns:
          if not hybrid_eval: (x_feat, loss)
          else:              (S_probs, x_feat, loss)
        where S_probs are softmax-normalized retrieval probabilities.
        """
        assert self.model is not None and self.Z0 is not None and self.adapter is not None, \
            "Call init_model() + precompute_prompt_bank() + initialize() first."

        if self.episodic:
            self.reset()

        self.adapter.train()
        # TPT steps
        loss_val = None
        for _ in range(self.steps):
            x_views = self.make_views(x.to(self.device))  # (M,B,C,H,W)
            loss = self.tpt_objective(x_views)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_val = float(loss.item())

        # After tuning prompts, compute query features (no aug) + scores
        with torch.no_grad():
            x_feat = self.encode_images(x.to(self.device))
            S = self.reid_scores(x_feat, G)

            # For convenience (similar to your "hybrid_eval"), return distribution over gallery too
            P = torch.softmax(S / self.id_temp, dim=1)

        if kwargs.get("hybrid_eval", False):
            return P.detach().cpu(), x_feat.detach().cpu(), loss_val
        else:
            return x_feat.detach().cpu(), loss_val
