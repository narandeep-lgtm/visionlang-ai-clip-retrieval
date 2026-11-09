"""Frozen zero-shot CLIP embedding extraction, without training or test leakage."""
from __future__ import annotations

import numpy as np
import torch
from PIL import ImageFilter, ImageOps
from torch.nn import functional as F

MODELS = {
    "clip-b32": "openai/clip-vit-base-patch32",
    "clip-b16": "openai/clip-vit-base-patch16",
}


def _projected_feature(output):
    """Support tensor-returning Transformers 4.x and structured 5.x outputs."""
    if isinstance(output, torch.Tensor):
        return output
    for field in ("pooler_output", "image_embeds", "text_embeds"):
        item = getattr(output, field, None)
        if isinstance(item, torch.Tensor):
            return item
    if isinstance(output, (tuple, list)) and len(output) and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported CLIP feature output: {type(output)}")


def corrupt_image(im, mode):
    if mode == "none":
        return im
    if mode == "grayscale":
        return ImageOps.grayscale(im).convert("RGB")
    if mode == "blur":
        return im.filter(ImageFilter.GaussianBlur(radius=3))
    raise ValueError(f"unknown corruption: {mode}")


class CLIPEncoder:
    def __init__(self, model_key: str, device: str = "auto"):
        if model_key not in MODELS:
            raise ValueError(f"unknown model: {model_key}")
        from transformers import CLIPModel, CLIPProcessor
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.model_id = MODELS[model_key]
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).eval().to(self.device)

    @torch.inference_mode()
    def images(self, images, batch_size=16, corruption="none") -> np.ndarray:
        chunks = []
        for s in range(0, len(images), batch_size):
            ims = [corrupt_image(im, corruption) for im in images[s:s+batch_size]]
            encoded = self.processor(images=ims, return_tensors="pt")
            emb = _projected_feature(self.model.get_image_features(pixel_values=encoded["pixel_values"].to(self.device)))
            chunks.append(F.normalize(emb.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    @torch.inference_mode()
    def texts(self, captions, batch_size=64) -> np.ndarray:
        chunks = []
        for s in range(0, len(captions), batch_size):
            encoded = self.processor(text=captions[s:s+batch_size], padding=True,
                                     truncation=True, max_length=77, return_tensors="pt")
            token_inputs = {k: v.to(self.device) for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
            emb = _projected_feature(self.model.get_text_features(**token_inputs))
            chunks.append(F.normalize(emb.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks, axis=0)
