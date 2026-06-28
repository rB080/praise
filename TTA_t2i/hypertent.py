# hyper_plip_t2i.py
# T2I HyperTENT for PLIP
# Similar structure to your T2I ATTR_TTA template, but uses a hypernetwork
# to predict residual LN/BN affine updates for the text encoder at test time.

from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

from backbones.PLIP.PLIPmodel import Create_PLIP_Model
from utils.utils import Test_parse_args


# ============================================================
# 1) Utilities
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
# 2) Adaptive normalization wrappers
# ============================================================
class AdaptiveLayerNorm(nn.Module):
    def __init__(self, ln: nn.LayerNorm, name: str):
        super().__init__()
        self.name = name
        self.normalized_shape = ln.normalized_shape
        self.eps = ln.eps

        if not ln.elementwise_affine:
            raise ValueError(f"LayerNorm {name} must have affine=True")

        self.weight = nn.Parameter(ln.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(ln.bias.detach().clone(), requires_grad=False)

        self.delta_weight: Optional[torch.Tensor] = None
        self.delta_bias: Optional[torch.Tensor] = None

    def set_delta(self, delta_weight: Optional[torch.Tensor], delta_bias: Optional[torch.Tensor]):
        self.delta_weight = delta_weight
        self.delta_bias = delta_bias

    def clear_delta(self):
        self.delta_weight = None
        self.delta_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.delta_weight is None else self.weight + self.delta_weight
        bias = self.bias if self.delta_bias is None else self.bias + self.delta_bias
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class AdaptiveBatchNorm1d(nn.Module):
    def __init__(self, bn: nn.BatchNorm1d, name: str):
        super().__init__()
        self.name = name
        self.num_features = bn.num_features
        self.eps = bn.eps

        if not bn.affine:
            raise ValueError(f"BatchNorm1d {name} must have affine=True")

        self.weight = nn.Parameter(bn.weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bn.bias.detach().clone(), requires_grad=False)

        self.register_buffer("running_mean", bn.running_mean.detach().clone())
        self.register_buffer("running_var", bn.running_var.detach().clone())

        self.delta_weight: Optional[torch.Tensor] = None
        self.delta_bias: Optional[torch.Tensor] = None

    def set_delta(self, delta_weight: Optional[torch.Tensor], delta_bias: Optional[torch.Tensor]):
        self.delta_weight = delta_weight
        self.delta_bias = delta_bias

    def clear_delta(self):
        self.delta_weight = None
        self.delta_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.delta_weight is None else self.weight + self.delta_weight
        bias = self.bias if self.delta_bias is None else self.bias + self.delta_bias
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            weight,
            bias,
            training=False,
            momentum=0.0,
            eps=self.eps,
        )


# ============================================================
# 3) Hypernetwork
# ============================================================
class HyperNet(nn.Module):
    def __init__(self, descriptor_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 4) T2I HyperTENT
# ============================================================
class HYPER_TTA(nn.Module):
    def __init__(
        self,
        steps: int = 1,
        topk: int = 5,
        id_temp: float = 0.6,
        device: str = "cuda:0",
        lr: float = 5e-4,
        episodic: bool = True,

        # hypernetwork knobs
        hidden_dim: int = 256,
        descriptor_momentum: float = 0.9,
        delta_scale: float = 0.02,
        delta_reg_weight: float = 1e-4,

        # optional consistency
        use_consistency: bool = False,
        consistency_weight: float = 0.5,
        aug_dropout_text: bool = False,

        # text settings
        max_len: int = 250,

        # layer selection
        adapt_last_k_norms: Optional[int] = None,
    ):
        super().__init__()
        self.episodic = episodic
        self.steps = steps
        self.topk = topk
        self.id_temp = id_temp
        self.device = device
        self.lr = lr
        self.max_len = max_len

        self.hidden_dim = hidden_dim
        self.descriptor_momentum = descriptor_momentum
        self.delta_scale = delta_scale
        self.delta_reg_weight = delta_reg_weight

        self.use_consistency = use_consistency
        self.consistency_weight = consistency_weight
        self.aug_dropout_text = aug_dropout_text

        self.adapt_last_k_norms = adapt_last_k_norms

        self.model: Optional[nn.Module] = None
        self.hypernet: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.model_state = None
        self.hyper_state = None
        self.optimizer_state = None

        self.TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
        self.args = Test_parse_args(notebook=True)

        self._norm_layers: List[Tuple[str, nn.Module, int]] = []
        self.hyper_output_dim: Optional[int] = None
        self.descriptor_dim: Optional[int] = None
        self._descriptor_state: Optional[torch.Tensor] = None

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
    # Replace modules
    # -----------------------------
    def _replace_module(self, module_path: str, new_module: nn.Module):
        parts = module_path.split(".")
        parent = self.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_module)

    def wrap_text_norm_layers(self):
        """
        Replace text-encoder LN/BN layers with adaptive wrappers.
        """
        assert self.model is not None, "Call init_model() first."

        self._norm_layers = []
        named_modules = list(self.model.named_modules())

        for name, mod in named_modules:
            if name.split(".")[0] != "text_encoder":
                continue

            if isinstance(mod, nn.LayerNorm):
                wrapped = AdaptiveLayerNorm(mod, name)
                self._replace_module(name, wrapped)
                self._norm_layers.append((name, wrapped, wrapped.weight.numel()))

            elif isinstance(mod, nn.BatchNorm1d):
                wrapped = AdaptiveBatchNorm1d(mod, name)
                self._replace_module(name, wrapped)
                self._norm_layers.append((name, wrapped, wrapped.weight.numel()))

        if len(self._norm_layers) == 0:
            raise RuntimeError("No text-side LN/BN layers found.")

        if self.adapt_last_k_norms is not None:
            self._norm_layers = self._norm_layers[-self.adapt_last_k_norms:]

        self.hyper_output_dim = sum(2 * n for _, _, n in self._norm_layers)
        print(f"Wrapped {len(self._norm_layers)} text norm layers.")
        print(f"Hyper output dim = {self.hyper_output_dim}")

    # -----------------------------
    # Param collection / reset
    # -----------------------------
    def collect_params(self, disable_dropouts: bool = True) -> Tuple[List[nn.Parameter], List[str]]:
        """
        Only hypernetwork params are optimized.
        """
        params: List[nn.Parameter] = []
        names: List[str] = []

        for nm, p in self.hypernet.named_parameters():
            if p.requires_grad:
                params.append(p)
                names.append(f"hypernet.{nm}")

        if disable_dropouts and not self.aug_dropout_text:
            disable_dropout(self.model)

        return params, names

    def copy_model_and_optimizer(self):
        self.model_state = deepcopy(self.model.state_dict())
        self.hyper_state = deepcopy(self.hypernet.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def load_model_and_optimizer(self):
        if self.model_state is None or self.hyper_state is None or self.optimizer_state is None:
            raise RuntimeError("reset called before initialize()")
        self.model.load_state_dict(self.model_state, strict=True)
        self.hypernet.load_state_dict(self.hyper_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self._descriptor_state = None
        self.clear_deltas()

    def initialize(self, gallery_feats: torch.Tensor):
        assert self.model is not None, "Call init_model() first."

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.wrap_text_norm_layers()

        feat_dim = gallery_feats.shape[1]
        self.descriptor_dim = 2 * feat_dim + 2

        self.hypernet = HyperNet(
            descriptor_dim=self.descriptor_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hyper_output_dim,
        ).to(self.device)

        params, names = self.collect_params()
        print(f"Optimizing {len(params)} parameter tensors (hypernet only).")

        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, weight_decay=1e-4)
        self.copy_model_and_optimizer()

    def reset(self):
        self.load_model_and_optimizer()

    # -----------------------------
    # Descriptor
    # -----------------------------
    @torch.no_grad()
    def compute_descriptor(self, q_feat: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        Descriptor = [mu, sigma, entropy, mean_confidence]
        using the current batch text features and retrieval scores.
        """
        q_norm = l2_normalize(q_feat)

        mu = q_norm.mean(dim=0)
        sigma = q_norm.std(dim=0, unbiased=False)

        p = torch.softmax(S / self.id_temp, dim=1)
        ent = entropy_from_probs(p).view(1)
        conf = p.max(dim=1).values.mean().view(1)

        descriptor = torch.cat([mu, sigma, ent, conf], dim=0)

        if self._descriptor_state is None:
            self._descriptor_state = descriptor.detach()
        else:
            m = self.descriptor_momentum
            self._descriptor_state = m * self._descriptor_state + (1.0 - m) * descriptor.detach()

        return self._descriptor_state

    # -----------------------------
    # Delta assignment
    # -----------------------------
    def assign_deltas(self, flat_delta: torch.Tensor):
        idx = 0
        for _, mod, n in self._norm_layers:
            d_w = flat_delta[idx: idx + n]
            idx += n
            d_b = flat_delta[idx: idx + n]
            idx += n

            d_w = self.delta_scale * d_w.view_as(mod.weight)
            d_b = self.delta_scale * d_b.view_as(mod.bias)
            mod.set_delta(d_w, d_b)

    def clear_deltas(self):
        for _, mod, _ in self._norm_layers:
            mod.clear_delta()

    # -----------------------------
    # Optional text perturbation
    # -----------------------------
    def tokenize_with_light_dropout(self, captions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Very mild text corruption. Kept optional because strong text corruption
        usually hurts T2I more than it helps.
        """
        dropped = []
        for c in captions:
            words = c.split()
            if len(words) > 4:
                keep = [w for i, w in enumerate(words) if (i % 7) != 0]
                if len(keep) < 3:
                    keep = words
                dropped.append(" ".join(keep))
            else:
                dropped.append(c)
        return self.tokenize(dropped)

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, captions: List[str], G: torch.Tensor, **kwargs):
        if self.episodic:
            self.reset()

        out = None
        for _ in range(self.steps):
            out = forward_and_adapt(self, captions, G, **kwargs)
        return out


# ============================================================
# 5) Adaptation step (T2I HyperTENT)
# ============================================================
@torch.enable_grad()
def forward_and_adapt(adapter: HYPER_TTA, captions: List[str], G: torch.Tensor, **kwargs):
    adapter.model.eval()
    adapter.hypernet.train()
    adapter.optimizer.zero_grad()

    # ---------------------------------------
    # Pass 1: initial text features
    # ---------------------------------------
    ti0, tm0 = adapter.tokenize(captions)
    q_feat0, _ = adapter.model.text_encoder(ti0.to(adapter.device), tm0.to(adapter.device))

    G = G.to(q_feat0.device)
    q_norm0 = l2_normalize(q_feat0)
    G_norm = l2_normalize(G)

    S0 = q_norm0 @ G_norm.t()

    # ---------------------------------------
    # Descriptor -> hypernet -> norm deltas
    # ---------------------------------------
    descriptor = adapter.compute_descriptor(q_feat0, S0)
    flat_delta = adapter.hypernet(descriptor.unsqueeze(0)).squeeze(0)
    adapter.assign_deltas(flat_delta)

    # ---------------------------------------
    # Pass 2: adapted text features
    # ---------------------------------------
    ti, tm = adapter.tokenize(captions)
    q_feat, _ = adapter.model.text_encoder(ti.to(adapter.device), tm.to(adapter.device))

    q_norm = l2_normalize(q_feat)
    S = q_norm @ G_norm.t()

    # TENT-style top-k entropy
    p = torch.softmax(S / adapter.id_temp, dim=1)
    vals, _ = topk_rowwise(p, adapter.topk)
    vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-12)
    loss = entropy_from_probs(vals)

    # Optional consistency
    if adapter.use_consistency:
        if adapter.aug_dropout_text:
            ti_aug, tm_aug = adapter.tokenize_with_light_dropout(captions)
        else:
            ti_aug, tm_aug = adapter.tokenize(captions)

        q_feat_aug, _ = adapter.model.text_encoder(ti_aug.to(adapter.device), tm_aug.to(adapter.device))
        cons_loss = F.mse_loss(l2_normalize(q_feat), l2_normalize(q_feat_aug))
        loss = loss + adapter.consistency_weight * cons_loss

    # Delta regularization
    reg = q_feat.new_tensor(0.0)
    for _, mod, _ in adapter._norm_layers:
        if mod.delta_weight is not None:
            reg = reg + mod.delta_weight.pow(2).mean()
        if mod.delta_bias is not None:
            reg = reg + mod.delta_bias.pow(2).mean()

    loss = loss + adapter.delta_reg_weight * reg

    loss.backward()
    adapter.optimizer.step()
    adapter.optimizer.zero_grad()

    if kwargs.get("hybrid_eval", False):
        return p.detach(), q_feat.detach(), float(loss.item())
    else:
        return q_feat.detach(), float(loss.item())