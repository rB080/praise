from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

from utils.utils import *
import pickle
import argparse
from backbones.PLIP.PLIPmodel import Create_PLIP_Model
import sys
import os
import glob
import re
from transformers import BertTokenizer
from utils.metrics import *
from tqdm import tqdm
import torch.nn.functional as F


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
    "age": [_prompt_templates["age"].format(prompt) for prompt in _prompts["age"]],
    "gender": [_prompt_templates["gender"].format(prompt) for prompt in _prompts["gender"]],
    "hair": [_prompt_templates["hair"].format(prompt) for prompt in _prompts["hair"]],
    "ubcl": [_prompt_templates["ubcl"].format(prompt) for prompt in _prompts["ubcl"]],
    "ubcc": [_prompt_templates["ubcc"].format(prompt) for prompt in _prompts["ubcc"]],
    "lbcl": [_prompt_templates["lbcl"].format(prompt) for prompt in _prompts["lbcl"]],
    "lbcc": [_prompt_templates["lbcc"].format(prompt) for prompt in _prompts["lbcc"]],
    "lbct": [_prompt_templates["lbct"].format(prompt) for prompt in _prompts["lbct"]],
    "accessories": [_prompt_templates["accessories"].format(prompt) for prompt in _prompts["accessories"]] + ["A photo of a person without any accessories."]
}

_prompt_list = []
# _att_set = ["gender", "ubcc", "lbcc"]
# for attr, prompts in _prompt_set.items():
#     if attr not in _att_set: continue
#     # if attr not in ["ubcc"]: continue
#     for prompt in prompts:
#         _prompt_list.append(prompt)
_colours = ["black", "blue", "brown", "green", "gray", "orange", "pink", "purple", "red", "white", "yellow"]
for g, pronoun in {"man": "He", "woman": "She"}.items():
    # for h in ["", "with long black hair", "with black hair", "with short black hair", "with long brown hair", "with brown hair", "with short brown hair", "with long blonde hair", "with blonde hair", "with short blonde hair"]:
    for h in ["", "with long hair", "with hair", "with short hair"]:
        for l in ["sp", "d", "sk"]:
            for c in _colours:
                if not l == "d":
                    for cc in _colours:
                        # for ccc in ["black shoes", "brown shoes", "white shoes", "red shoes", "blue shoes", "green shoes"]:
                        for ccc in ["dark coloured shoes", "light coloured shoes"]:
                            # sp for shirt, d for dress, sk for skirt
                            _prompt_list.append(f"A {g} {h} wearing a {c} top. {pronoun} is also wearing {cc} {'pants' if l == 'sp' else 'skirt'} and {ccc}.")
                else:
                    # for ccc in ["black shoes", "brown shoes", "white shoes", "red shoes", "blue shoes", "green shoes"]:
                    for ccc in ["dark coloured shoes", "light coloured shoes"]:
                        _prompt_list.append(f"A {g} {h} wearing a {c} dress. {pronoun} is also wearing {ccc}.")
# consider adding type of shoes


class ATTR_TTA(nn.Module):
    def __init__(self, steps=1, 
                        alpha=0.2, 
                        k=5, 
                        k_pl=1, 
                        id_temp=0.6, 
                        att_temp=0.7,
                        device='cuda:1', 
                        lr=0.0005, 
                        episodic=True, 
                        mode=None, 
                        vectorize_alpha=False, 
                        zscore_text=False):
        super(ATTR_TTA, self).__init__()
        self.episodic = episodic
        
        self.steps = steps
        self.alpha = alpha
        
        self.vectorize_alpha = vectorize_alpha
        self.zscore_text = zscore_text
        self.topk = k
        self.k_pl = k_pl
        self.id_temp = id_temp
        self.att_temp = att_temp
        self.mode = mode
        self.lr = lr
        self.device = device
        self.args = Test_parse_args(notebook=True)

    def init_model(self):
        model = Create_PLIP_Model(self.args).to(self.device)
        model_file = self.args.model_path
        breakpoint()
        print(model_file)
        #if os.path.isdir(model_file): continue
        checkpoint = torch.load(model_file,map_location='cpu')

        model.image_encoder.load_state_dict(checkpoint["ImgEncoder_state_dict"])
        model.text_encoder.load_state_dict(checkpoint["TxtEncoder_state_dict"], strict=False)
        self.model = model.to(self.device)
        self.TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize(self, caption):
        result = self.TOKENIZER(caption, padding="max_length", max_length=250, truncation=True, return_tensors='pt')
        token, mask = result["input_ids"], result["attention_mask"]
        token, mask = token.squeeze(), mask.squeeze()
        return token, mask

    def precompute_embeddings(self, prompts=None, chunksize=32, path=None, neutral_text=None):
        if path is not None:
            self.preset_text_features = torch.load(path)
        else:
            input_prompts = prompts if prompts is not None else _prompt_list
            with torch.no_grad():
                for i in tqdm(range(0, len(input_prompts), chunksize), desc="Encoding texts"):
                    chunk = input_prompts[i:i + chunksize]
                    text_inputs, text_masks = self.tokenize(chunk)
                    # text_features = self.clip.encode_text(text_inputs)
                    text_features, text_part_features = self.model.text_encoder(text_inputs.to(self.device), text_masks.to(self.device))
                    # breakpoint()
                    if i == 0:
                        self.preset_text_features = text_features.detach().cpu()
                        self.preset_text_part_features = text_part_features.detach().cpu()
                    else:
                        self.preset_text_features = torch.cat((self.preset_text_features, text_features.detach().cpu()), dim=0)
                        self.preset_text_part_features = torch.cat((self.preset_text_part_features, text_part_features.detach().cpu()), dim=0)
        print("preset_text_features shape:", self.preset_text_features.shape)
        self.prompt_list = _prompt_list
        if neutral_text is not None:
            # breakpoint()
            self.neutral_text = neutral_text
            ti, tm = self.tokenize([neutral_text])
            ntf, ntpf = self.model.text_encoder(ti.unsqueeze(0).to(self.device), tm.unsqueeze(0).to(self.device))
            self.neutral_text_feature = ntf.detach().cpu()
            self.neutral_text_part_feature = ntpf.detach().cpu()
            self.preset_text_features -= self.neutral_text_feature.repeat(self.preset_text_features.shape[0], 1)
            # self.preset_text_part_features -= self.neutral_text_part_feature.repeat(self.preset_text_part_features.shape[0], 1, 1)
            print("Using neutral text:", neutral_text)

    # ---------- NEW: utilities ----------
    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Avoids div-by-zero; keeps graph intact."""
        return F.normalize(x, p=2, dim=-1, eps=eps)

    @staticmethod
    def _encode_ids(text_ids,
                    device):
        """
        Map arbitrary hashable IDs to contiguous indices [0..N_ids-1] in first-seen order.
        Returns:
            id_idx: LongTensor[N] with group index for each text.
            unique_ids: Python list of unique IDs aligned with columns of the output.
        """
        if isinstance(text_ids, torch.Tensor):
            if text_ids.dtype in (torch.long, torch.int32, torch.int64):
                seen, unique, mapped = {}, [], []
                for v in text_ids.tolist():
                    if v not in seen:
                        seen[v] = len(unique)
                        unique.append(v)
                    mapped.append(seen[v])
                return torch.tensor(mapped, device=device, dtype=torch.long), unique
            # non-integer tensors → fall through to list
            text_ids = text_ids.tolist()

        seen: dict = {}
        unique: List[Any] = []
        mapped: List[int] = []
        for v in text_ids:
            if v not in seen:
                seen[v] = len(unique)
                unique.append(v)
            mapped.append(seen[v])
        return torch.tensor(mapped, device=device, dtype=torch.long), unique

    def _ensure_text_bank_index(self, device: torch.device) -> None:
        """
        Lazily build and cache:
            self._text_id_idx: LongTensor[N] mapping each text row → group id.
            self._unique_text_ids: Python list of group IDs.
        """
        # if not hasattr(self, "text_feats") or not hasattr(self, "text_ids"):
         #    raise AttributeError("Expect self.text_feats (N,d) and self.text_ids (len N) to be set on PLIPATTR.")
        if (not hasattr(self, "_text_id_idx")) or (self._text_id_idx.device != device):
            id_idx, unique_ids = self._encode_ids(self.text_ids, device=device)
            self._text_id_idx = id_idx          # (N,)
            self._unique_text_ids = unique_ids  # list[Any]

    # ---------- NEW: main API ----------
    def image_text_max_cosine_by_id(
        self,
        image_feats: torch.Tensor,          # (B, d) requires_grad=True for images
        text_feats,
        *,
        normalize: bool = True,
        eps: float = 1e-8,
        text_chunk_size,
    ):
        """
        Compute B×N_ids matrix where each entry is the max cosine similarity between an image
        and *any* text embedding that shares the same ID. Gradients flow to `image_feats`.

        Args:
            image_feats: (B,d) tensor. Needs grad for images.
            normalize: L2-normalize before similarity for cosine.
            eps: numerical stability for normalization.
            text_chunk_size: if set, process texts in chunks to limit memory. Safe for autograd.

        Returns:
            scores: (B, N_ids) tensor with per-ID max cosine similarities.
            unique_ids: list[Any] of IDs aligned with scores columns.
        """
        if image_feats.dim() != 2:
            raise ValueError("image_feats must be 2D (B,d).")

        device = image_feats.device
        dtype = image_feats.dtype

        # # Pull text bank to the same device/dtype
        # text_feats: torch.Tensor = self.text_feats
        if text_feats.dim() != 2:
            raise ValueError("self.text_feats must be 2D (N,d).")
        if text_feats.size(1) != image_feats.size(1):
            raise ValueError("Feature dims must match between image_feats and self.text_feats.")

        text_feats = text_feats.to(device=device, dtype=dtype)

        # Build ID index once (cached)
        self._ensure_text_bank_index(device=device)
        id_idx: torch.Tensor = self._text_id_idx  # (N,)
        unique_ids: List[Any] = self._unique_text_ids
        num_ids = len(unique_ids)
        B = image_feats.size(0)

        # Normalize if requested
        x = self._l2_normalize(image_feats, eps=eps) if normalize else image_feats
        t = self._l2_normalize(text_feats, eps=eps) if normalize else text_feats

        # Fast path (no chunking): single matmul then grouped amax
        if not text_chunk_size:
            sim = x @ t.t()  # (B, N), keeps graph for grads to images
            # breakpoint()
            # Vectorized groupwise amax when available; fallback loop otherwise.
            try:
                out = torch.full((B, num_ids), float("-inf"), device=device, dtype=sim.dtype)
                idx = id_idx.unsqueeze(0).expand(B, -1)  # (B, N)
                # Why scatter_reduce_: avoids Python loops; autograd flows to argmax entries.
                out.scatter_reduce_(dim=1, index=idx, src=sim, reduce="amax", include_self=True)
            except AttributeError:
                cols = []
                for gid in range(num_ids):  # simple & robust; slower for very large N_ids
                    mask = (id_idx == gid)
                    cols.append(sim[:, mask].amax(dim=1).values)
                out = torch.stack(cols, dim=1)
            return out, unique_ids

        # Chunked path: reduce chunkwise then take elementwise max across chunks (grad-safe).
        out = torch.full((B, num_ids), float("-inf"), device=device, dtype=dtype)
        N = t.size(0)
        for s in range(0, N, text_chunk_size):
            e = min(s + text_chunk_size, N)
            sim_chunk = x @ t[s:e].t()              # (B, e-s)
            id_chunk = id_idx[s:e]                  # (e-s,)

            try:
                part = torch.full((B, num_ids), float("-inf"), device=device, dtype=sim_chunk.dtype)
                idx = id_chunk.unsqueeze(0).expand(B, -1)
                part.scatter_reduce_(dim=1, index=idx, src=sim_chunk, reduce="amax", include_self=True)
            except AttributeError:
                cols = []
                for gid in range(num_ids):
                    mask = (id_chunk == gid)
                    if mask.any():
                        cols.append(sim_chunk[:, mask].amax(dim=1).values)
                    else:
                        cols.append(torch.full((B,), float("-inf"), device=device, dtype=sim_chunk.dtype))
                part = torch.stack(cols, dim=1)

            # Why maximum: preserves gradient path to the chunk that wins the max.
            out = torch.maximum(out, part)

        return out, unique_ids

    def topk_distance_loss(self, q, g, temp=None):
        S = euclidean_distance(q, g, return_tensor=True)
        if temp is not None:
            #breakpoint()
            S = torch.exp(-S / temp)
            S = -torch.log(S / S.sum(1, keepdim=True))
        loss = 0.0
        for i in range(S.shape[0]):
            s = S[i]
            # Find the indices of the k smallest values
            _, topk_indices = torch.topk(s, self.topk, largest=False)
            # Create a mask with all False
            mask = torch.zeros_like(s, dtype=torch.bool)
            # Set True for top-k smallest indices
            mask[topk_indices] = True
            # Set all non-top-k elements to zero (or another value)
            s[~mask] = 0  # In-place operation
            loss += s.sum() / self.topk
        #breakpoint()
        return loss / S.shape[0]
    
    def topk_cosine_distance_loss(self, q, g, temp=None):
        #breakpoint()
        S = cosine_similarity(q, g, return_tensor=True, invert=False)
        if temp is not None:
            #breakpoint()
            S = torch.exp(-S / temp)
            S = -torch.log(S / S.sum(1, keepdim=True))
        loss = 0.0
        for i in range(S.shape[0]):
            s = S[i]
            # Find the indices of the k smallest values
            _, topk_indices = torch.topk(s, self.topk, largest=False)
            # Create a mask with all False
            mask = torch.zeros_like(s, dtype=torch.bool)
            # Set True for top-k smallest indices
            mask[topk_indices] = True
            # Set all non-top-k elements to zero (or another value)
            s[~mask] = 0  # In-place operation
            loss += s.sum()
        return loss / S.shape[0]

    def softmax_entropy(self, x: torch.Tensor, nll = True) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        if not nll: return -(x * torch.log(x)).sum(1).mean()
        else: return -torch.log(x).sum(1).mean()
    
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        
    # def attribute_entropy(self, x, atsim, idsim):
    #     pred = 
    
    def initialize(self, gallery_feats):
        params, names = collect_params(self.model)

        if self.vectorize_alpha:
            # breakpoint()
            self.alpha = nn.Parameter(torch.randn(32, gallery_feats.shape[0]).to(self.device) * self.alpha)
            params.append(self.alpha)
            names.append("alpha")
        # if self.tensor_alpha is not None:
        #     self.alpha = - alpha * torch.ones(tensor_alpha).to(device)
        #     self.alpha = nn.Parameter(self.alpha)
        print("PLIP-TTA is optimizing {} parameters:".format(len(params)))
        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, weight_decay=1e-4)
        # self.optimizer = torch.optim.Adam(
        #     [{"params": params[:-1], "lr": self.lr, "weight_decay": 1e-4},
        #     {"params": params[-1:], "lr": self.lr * 100, "weight_decay": 1e-4}]
        #     )
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, G, **kwargs):
        if self.episodic: self.reset()
        
        for _ in range(self.steps):
            # check if kwargs contains batch_pids and gpids, pass them to forward_and_adapt
            outputs = forward_and_adapt(self, x, G, **kwargs) # forward_and_adapt(self, x, cam_label)
            
        return outputs


def _masked_logsumexp(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    breakpoint()
    x = x.masked_fill(~mask, float('-inf'))
    return torch.logsumexp(x, dim=dim)

def _topk_mask_per_row(scores: torch.Tensor, k: int, largest: bool) -> torch.Tensor:
    idx = torch.topk(scores, k=k, dim=1, largest=largest).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(1, idx, True)



@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(adapter, x, G, **kwargs):
    
    loss = 0

    # criterions
    oracle_mode = False
    mask_using_id = False
    hybrid_eval = False
    use_CELoss = False
    if 'batch_pids' in kwargs and 'gpids' in kwargs and 'no_oracle' in kwargs:
        batch_pids = kwargs['batch_pids']
        gpids = kwargs['gpids']
        if not kwargs['no_oracle']: oracle_mode = True
        else: mask_using_id = True
    if 'hybrid_eval' in kwargs:
        hybrid_eval = kwargs['hybrid_eval']
    

    
    z_t = adapter.preset_text_features.to(adapter.device)
    # z_t = adapter.redo_text_embeddings(adapter.prompt_list, chunksize=8)
    # z_t = adapter.model.text_encoder.encoder.pooler(adapter.preset_text_part_features.to(x.device))
    # breakpoint()

    x, xmask = adapter.tokenize(x)
    x, _ = adapter.model.text_encoder(x.to(adapter.device), xmask.to(adapter.device))

    
    G = G.to(x.device)  # ensure G is on the same device as x_norm

    # norms
    x_norm = x / x.norm(dim=-1, keepdim=True)
    G_norm = G / G.norm(dim=-1, keepdim=True)
     
    
    A_id = x_norm @ G_norm.t()
    
    
    
    Ab_logits = x_norm @ z_t.t()            # [B,P]
    Ag_logits = G_norm @ z_t.t()            # [Ng,P]
    
    if adapter.zscore_text:
        mu = Ag_logits.mean(dim=0, keepdim=True)
        sd = Ag_logits.std(dim=0, keepdim=True) + 1e-6
        Ab_logits = (Ab_logits - mu) / sd
        Ag_logits = (Ag_logits - mu) / sd
    
    Ab_conf = torch.softmax(Ab_logits, dim=1)   # b_i
    Ag_conf = torch.softmax(Ag_logits / 0.3, dim=1)   # g_j

    
    A_att = Ab_logits @ Ag_logits.t()  # batch_at_sim is (B, N) and G_at_sim is (N, M), so result is (B, M)
    
    id_mask = None
    if oracle_mode:
        b = torch.as_tensor(batch_pids, device=x.device)
        g = torch.as_tensor(gpids, device=x.device)
        oracle_mask = b[:, None].eq(g[None, :]) # [B, N_g]
        oracle_mask = oracle_mask.float()
        # A_att = A_att * oracle_mask
        A_att = oracle_mask
    

   
    if adapter.vectorize_alpha: 
        preds = torch.softmax(A_id / adapter.id_temp, dim=1) + adapter.alpha[:A_att.shape[0]] * torch.softmax(A_att / adapter.att_temp, dim=1)
    else:
        preds = torch.softmax(A_id / adapter.id_temp, dim=1) + adapter.alpha * torch.softmax(A_att / adapter.att_temp, dim=1)

    # choose only top k values in preds
    preds_k = torch.topk(preds, adapter.topk).values
    # preds_k = preds * id_mask

    loss = adapter.softmax_entropy(preds_k)
    
    
    loss.backward()
    adapter.optimizer.step()
    adapter.optimizer.zero_grad()

    if not hybrid_eval: return x, loss.item()
    else: return preds, x, loss.item()




def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    # breakpoint()
    named_mods = model.named_modules() if hasattr(model, 'named_modules') else model.model.named_modules()
    for nm, m in named_mods:
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm): # or True:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    # if nm.split('.')[0] == 'text_encoder': continue
                    if nm.split('.')[0] == 'image_encoder': continue
                    params.append(p)
                    names.append(f"{nm}.{np}")

    # breakpoint()
    # for nm, m in model.text_encoder.encoder.pooler.named_modules():
    #     for np, p in m.named_parameters():
    #         params.append(p)
    #         names.append(f"{nm}.{np}")
    #         p.requires_grad = True
    # breakpoint()
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    if hasattr(model, 'state_dict'):
        model_state = deepcopy(model.state_dict())
    else:
        model_state = deepcopy(model.model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    if hasattr(model, 'load_state_dict'):
        model.load_state_dict(model_state, strict=True)
    else:
        model.model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    #breakpoint()
    for module in model.modules():
        for m in module.modules():
            # breakpoint()
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
