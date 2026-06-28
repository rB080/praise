# qwen_vl_clipish.py
from typing import List, Sequence, Optional, Union, Dict
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    # Optional but recommended for robust vision preproc on Qwen2-VL
    from qwen_vl_utils import process_vision_info  # pip install qwen-vl-utils
    _HAS_QWEN_VL_UTILS = True

except Exception:
    _HAS_QWEN_VL_UTILS = False
print(f"QwenVLClipish: qwen-vl-utils available: {_HAS_QWEN_VL_UTILS}")


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    x: (B, T, D)
    mask: (B, T) with 1 for valid tokens; if None, do plain mean
    """
    if mask is None:
        return x.mean(dim=1)
    w = mask.to(x.dtype).unsqueeze(-1)  # (B,T,1)
    s = (x * w).sum(dim=1)
    z = w.sum(dim=1).clamp_min(1e-6)
    return s / z


class QwenVLClipish:
    """
    CLIP-style convenience wrapper around Qwen2-VL.

    - .encode_image(images) -> (B, D) L2-normalized embeddings
    - .encode_text(texts)   -> (B, D) L2-normalized embeddings
    - .similarity(img_emb, txt_emb) -> cosine-sim matrix
    - .topk(img_emb, txt_emb, k)    -> indices/scores
    - .generate(images, prompt, ...) -> text generation with chat template
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        device_map: Union[str, Dict] = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ):
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).eval()

        # Backbone with the modules you printed
        self.base = getattr(self.model, "model", self.model)  # Qwen2VLModel
        self.visual = getattr(self.base, "visual")            # Qwen2VisionTransformerPretrainedModel
        self.text_model = getattr(self.base, "language_model")# Qwen2VLTextModel

        # Handy references
        self.tokenizer = self.processor.tokenizer
        self.device = next(self.model.parameters()).device

    # ------------- Vision/text preprocessing -------------

    def _prep_vision(self, images: Sequence[Image.Image]):
        """
        Returns a dict with keys expected by the vision stack:
        - pixel_values: FloatTensor of shape (B, 3, T, H, W)
        - image_grid_thw: LongTensor of shape (B, 3) (if available)
        - image_token_mask: BoolTensor (optional; used for pooling)
        """
        if _HAS_QWEN_VL_UTILS:
            pixel_values, grid_thw = process_vision_info(images)
            out = {
                "pixel_values": torch.tensor(pixel_values, device=self.device),
                "image_grid_thw": torch.tensor(grid_thw, device=self.device),
            }
        else:
            enc = self.processor(images=images, text=[""] * len(images), return_tensors="pt")
            # Move everything we know the vision block might use
            out = {k: v.to(self.device) for k, v in enc.items() if k in ("pixel_values", "image_grid_thw", "image_token_mask")}
            if "pixel_values" not in out:
                raise RuntimeError("processor did not return 'pixel_values'; install qwen-vl-utils or use the processor with images=...")

        return out

    def _prep_text(self, texts: Sequence[str]):
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    # ------------- Encoders -------------

    # @torch.inference_mode()
    def encode_image(self, images, normalize: bool = False):
        # one {image} per sample; build B chats
        chats = [
            self.processor.apply_chat_template(
                # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Focus on the person in the image."}]}],
                [{"role": "user", "content": [{"type": "image"}]}],
                add_generation_prompt=False,
                tokenize=False,
            )
            for _ in images
        ]
        batch = self.processor(
            text=chats,
            images=list(images),
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**batch, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[-1]                     # (B, L, D) from language model

        # Prefer processor-provided mask; else use image_token_id
        mask = getattr(batch, "image_token_mask", None)
        if mask is None:
            image_id = getattr(self.model.config, "image_token_id", None)
            if image_id is None:
                raise RuntimeError("Qwen config has no image_token_id; cannot locate image tokens.")
            mask = (batch.input_ids == image_id)       # (B, L)

        # safe masked mean
        denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)    # (B, 1)
        pooled = (hs * mask.unsqueeze(-1)).sum(dim=1) / denom # (B, D)

        return torch.nn.functional.normalize(pooled, dim=-1) if normalize else pooled



    @torch.inference_mode()
    def encode_text(
        self,
        texts: Sequence[str],
        pool: str = "mean",              # "mean" or "last"
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Returns (B, D) text embeddings from the decoder hidden states.
        """
        toks = self._prep_text(texts)
        tout = self.text_model(**toks, return_dict=True)
        h = tout.last_hidden_state  # (B, L, D)

        if pool == "last":
            # last non-pad position per sequence
            lengths = toks["attention_mask"].sum(dim=1) - 1
            pooled = h[torch.arange(h.size(0), device=h.device), lengths]
        else:
            pooled = _masked_mean(h, toks.get("attention_mask"))

        if normalize:
            pooled = F.normalize(pooled, dim=-1)
        return pooled

    # ------------- Similarity / ranking -------------

    @staticmethod
    @torch.inference_mode()
    def similarity(img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity matrix (B_img, B_txt) assuming inputs are L2-normalized.
        """
        return img_emb @ txt_emb.T

    @torch.inference_mode()
    def topk(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        k: int = 5,
    ):
        """
        Returns (indices, scores) where:
        - indices: (B_img, k) top-k text indices per image
        - scores:  (B_img, k) corresponding cosine scores
        """
        sims = self.similarity(img_emb, txt_emb)
        scores, idx = sims.topk(k, dim=1)
        return idx, scores

    # ------------- Generation (chat) -------------

    @torch.inference_mode()
    def generate(
        self,
        images: Optional[Sequence[Image.Image]],
        prompt: str,
        max_new_tokens: int = 128,
        do_sample: bool = False,
    ) -> List[str]:
        """
        Multimodal chat generation using the processor's chat template.
        If images is None, runs text-only.
        """
        if images is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            content = [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]
            messages = [{"role": "user", "content": content}]

        chat = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if images is None:
            inputs = self.processor(text=[chat], return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=[chat], images=list(images), return_tensors="pt").to(self.device)

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", eos_id)

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        # Strip the prompt tokens
        gen_ids = [out_ids[i, inputs.input_ids.shape[1]:] for i in range(out_ids.size(0))]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)



