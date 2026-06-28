from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import Test_parse_args
from backbones.PLIP.PLIPmodel import Create_PLIP_Model


# ============================================================
# 1) Utilities
# ============================================================
def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


def topk_rowwise(p: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(p, k=k, dim=1, largest=True)


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12, nll: bool = False) -> torch.Tensor:
    p = torch.clamp(p, min=eps)
    if not nll:
        return -(p * torch.log(p)).sum(dim=1).mean()
    else:
        return -(torch.log(p)).sum(dim=1).mean()


def disable_dropout(module):
    dropout_types = (
        nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
        nn.AlphaDropout, nn.FeatureAlphaDropout
    )
    count_dropouts_closed = 0
    for m in module.modules():
        if isinstance(m, dropout_types):
            m.eval()
            m.training = False
            count_dropouts_closed += 1
    print(f"Disabled dropout in {count_dropouts_closed} modules.")


# ============================================================
# 2) Adaptive normalization layers
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


class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d, name: str):
        super().__init__()
        self.name = name
        self.num_features = bn.num_features
        self.eps = bn.eps

        if not bn.affine:
            raise ValueError(f"BatchNorm2d {name} must have affine=True")

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
# 4) Main class
# ============================================================
class HYPER_TTA(nn.Module):
    def __init__(
        self,
        steps: int = 1,
        topk: int = 5,
        id_temp: float = 1.0,
        device: str = "cuda:0",
        lr: float = 1e-4,
        episodic: bool = True,

        # Hypernetwork settings
        hidden_dim: int = 256,
        descriptor_momentum: float = 0.9,
        delta_scale: float = 0.05,

        # Optional regularization
        delta_reg_weight: float = 1e-4,
        use_consistency: bool = False,
        consistency_weight: float = 0.5,
        aug_noise_std: float = 0.01,

        # Optional camera conditioning
        use_cam_labels: bool = False,
        cam_embed_dim: int = 16,
        max_cams: int = 64,
    ):
        super().__init__()
        self.episodic = episodic
        self.steps = steps
        self.topk = topk
        self.id_temp = id_temp
        self.device = device
        self.lr = lr

        self.hidden_dim = hidden_dim
        self.descriptor_momentum = descriptor_momentum
        self.delta_scale = delta_scale

        self.delta_reg_weight = delta_reg_weight
        self.use_consistency = use_consistency
        self.consistency_weight = consistency_weight
        self.aug_noise_std = aug_noise_std

        self.use_cam_labels = use_cam_labels
        self.cam_embed_dim = cam_embed_dim
        self.max_cams = max_cams

        self.model = None
        self.hypernet = None
        self.optimizer = None
        self.args = Test_parse_args(notebook=True)

        self.model_state = None
        self.hyper_state = None
        self.optimizer_state = None

        self._norm_layers: List[Tuple[str, nn.Module, int]] = []
        self.hyper_output_dim: Optional[int] = None
        self.descriptor_dim: Optional[int] = None
        self._descriptor_state: Optional[torch.Tensor] = None

        self.cam_embed = None

    # -----------------------------
    # 4.1 Model init
    # -----------------------------
    def init_model(self):
        model = Create_PLIP_Model(self.args).to(self.device)
        checkpoint = torch.load(self.args.model_path, map_location="cpu")
        model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
        model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"], strict=False)
        self.model = model.to(self.device)

    # -----------------------------
    # 4.2 Optional compatibility hook
    # -----------------------------
    @torch.no_grad()
    def precompute_attr_bank(self, *args, **kwargs):
        """
        Kept only for runner compatibility.
        This method does nothing for the hypernetwork version.
        """
        print("HYPER_TTA does not use an attribute bank. Skipping precompute_attr_bank().")

    # -----------------------------
    # 4.3 Replace modules
    # -----------------------------
    def _replace_module(self, module_path: str, new_module: nn.Module):
        parts = module_path.split(".")
        parent = self.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_module)

    def wrap_norm_layers(self):
        """
        Replace image-side LN/BN layers with adaptive wrappers.
        """
        assert self.model is not None, "Call init_model() first."

        self._norm_layers = []
        named_modules = list(self.model.named_modules())

        for name, mod in named_modules:
            if name.split(".")[0] == "text_encoder":
                continue

            if isinstance(mod, nn.LayerNorm):
                wrapped = AdaptiveLayerNorm(mod, name)
                self._replace_module(name, wrapped)
                self._norm_layers.append((name, wrapped, wrapped.weight.numel()))

            elif isinstance(mod, nn.BatchNorm1d):
                wrapped = AdaptiveBatchNorm1d(mod, name)
                self._replace_module(name, wrapped)
                self._norm_layers.append((name, wrapped, wrapped.weight.numel()))

            elif isinstance(mod, nn.BatchNorm2d):
                wrapped = AdaptiveBatchNorm2d(mod, name)
                self._replace_module(name, wrapped)
                self._norm_layers.append((name, wrapped, wrapped.weight.numel()))

        if len(self._norm_layers) == 0:
            raise RuntimeError("No image-side LN/BN layers found.")

        self.hyper_output_dim = sum(2 * n for _, _, n in self._norm_layers)
        print(f"Wrapped {len(self._norm_layers)} norm layers.")
        print(f"Hyper output dim = {self.hyper_output_dim}")

    # -----------------------------
    # 4.4 Param collection / reset
    # -----------------------------
    def collect_params(self, disable_dropouts=True) -> Tuple[List[nn.Parameter], List[str]]:
        """
        Collect only hypernetwork parameters.
        """
        params: List[nn.Parameter] = []
        names: List[str] = []

        for nm, p in self.hypernet.named_parameters():
            if p.requires_grad:
                params.append(p)
                names.append(f"hypernet.{nm}")

        if self.cam_embed is not None:
            for nm, p in self.cam_embed.named_parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(f"cam_embed.{nm}")

        if disable_dropouts:
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
        if self.cam_embed is not None and hasattr(self, "cam_embed_state"):
            self.cam_embed.load_state_dict(self.cam_embed_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self._descriptor_state = None
        self.clear_deltas()

    def initialize(self, gallery_feats: torch.Tensor, set_learnable_text: bool = False):
        """
        set_learnable_text is ignored. Kept only for compatibility with your runner.
        """
        assert self.model is not None, "Call init_model() first."

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.wrap_norm_layers()

        feat_dim = gallery_feats.shape[1]
        self.descriptor_dim = 2 * feat_dim + 2
        if self.use_cam_labels:
            self.descriptor_dim += self.cam_embed_dim
            self.cam_embed = nn.Embedding(self.max_cams, self.cam_embed_dim).to(self.device)

        self.hypernet = HyperNet(
            descriptor_dim=self.descriptor_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hyper_output_dim,
        ).to(self.device)

        params, names = self.collect_params()
        print(f"Optimizing {len(params)} parameter tensors (hypernet only).")

        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, weight_decay=1e-4)

        self.copy_model_and_optimizer()
        if self.cam_embed is not None:
            self.cam_embed_state = deepcopy(self.cam_embed.state_dict())

    def reset(self):
        self.load_model_and_optimizer()

    # -----------------------------
    # 4.5 Descriptor
    # -----------------------------
    @torch.no_grad()
    def compute_descriptor(
        self,
        x_feat: torch.Tensor,
        S: torch.Tensor,
        camids_batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Descriptor = [mu, sigma, entropy, mean_confidence, optional_camera_embedding]
        """
        x_norm = l2_normalize(x_feat)

        mu = x_norm.mean(dim=0)
        sigma = x_norm.std(dim=0, unbiased=False)

        p = torch.softmax(S / self.id_temp, dim=1)
        ent = entropy_from_probs(p).view(1)
        conf = p.max(dim=1).values.mean().view(1)

        parts = [mu, sigma, ent, conf]

        if self.use_cam_labels and camids_batch is not None and self.cam_embed is not None:
            camids_batch = camids_batch.to(x_feat.device).long().clamp(min=0, max=self.max_cams - 1)
            camvec = self.cam_embed(camids_batch).mean(dim=0)
            parts.append(camvec)

        descriptor = torch.cat(parts, dim=0)

        if self._descriptor_state is None:
            self._descriptor_state = descriptor.detach()
        else:
            m = self.descriptor_momentum
            self._descriptor_state = m * self._descriptor_state + (1.0 - m) * descriptor.detach()

        return self._descriptor_state

    # -----------------------------
    # 4.6 Delta application
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
    # 4.7 Feature augmentation
    # -----------------------------
    def augment_input(self, x):
        if torch.is_tensor(x):
            return x + torch.randn_like(x) * self.aug_noise_std
        return x

    # -----------------------------
    # 4.8 Forward loop
    # -----------------------------
    def forward(self, x, G: torch.Tensor, **kwargs):
        if self.episodic:
            self.reset()

        out = None
        for _ in range(self.steps):
            out = forward_and_adapt(self, x, G, **kwargs)
        return out


# ============================================================
# 5) Adaptation step
# ============================================================
@torch.enable_grad()
def forward_and_adapt(adapter: HYPER_TTA, x, G: torch.Tensor, **kwargs):
    """
    One TTA step.

    Contract:
        - if hybrid_eval=False: returns (x_feat, loss)
        - if hybrid_eval=True:  returns (preds, x_feat, loss)

    kwargs may optionally contain:
        - camids_batch
    """
    adapter.model.eval()
    adapter.hypernet.train()
    adapter.optimizer.zero_grad()

    camids_batch = kwargs.get("camids_batch", None)

    # ------------------------------------------------
    # 1) Initial pass without deltas
    # ------------------------------------------------
    x_feat0 = adapter.model.get_image_embeddings(x)
    G = G.to(x_feat0.device)

    x_norm0 = l2_normalize(x_feat0)
    G_norm = l2_normalize(G)
    S0 = x_norm0 @ G_norm.t()

    # ------------------------------------------------
    # 2) Build descriptor and predict deltas
    # ------------------------------------------------
    descriptor = adapter.compute_descriptor(
        x_feat=x_feat0,
        S=S0,
        camids_batch=camids_batch
    )
    flat_delta = adapter.hypernet(descriptor.unsqueeze(0)).squeeze(0)
    adapter.assign_deltas(flat_delta)

    # ------------------------------------------------
    # 3) Adapted forward pass
    # ------------------------------------------------
    x_feat = adapter.model.get_image_embeddings(x)
    x_norm = l2_normalize(x_feat)
    S = x_norm @ G_norm.t()

    p = torch.softmax(S / adapter.id_temp, dim=1)

    vals, idx = topk_rowwise(p, adapter.topk)
    vals = vals / (vals.sum(dim=1, keepdim=True) + 1e-12)
    loss = entropy_from_probs(vals)

    # Optional consistency
    if adapter.use_consistency:
        x_aug = adapter.augment_input(x)
        x_feat_aug = adapter.model.get_image_embeddings(x_aug)
        cons_loss = F.mse_loss(l2_normalize(x_feat), l2_normalize(x_feat_aug))
        loss = loss + adapter.consistency_weight * cons_loss

    # Delta regularization
    reg = x_feat.new_tensor(0.0)
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
        return p.detach(), x_feat.detach(), float(loss.item())
    else:
        return x_feat.detach(), float(loss.item())