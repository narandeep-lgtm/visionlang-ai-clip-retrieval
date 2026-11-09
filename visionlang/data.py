"""Flickr8k image-level split loader. Never leak captions between splits."""
from __future__ import annotations


def load_flickr8k(split: str = "test", max_images: int = 0, seed: int = 42):
    """Return images, captions, caption_to_image, filenames.

    `max_images=0` uses the complete official split. Positive values pick a
    reproducible random subset; subsampled recall is *not* a full benchmark.
    Requires internet for the first Hugging Face dataset download (~1.1 GB).
    """
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev, or test")
    from datasets import load_dataset
    ds = load_dataset("intro/flickr8k", split=split)
    if max_images < 0:
        raise ValueError("max_images must be 0 or positive")
    if max_images and max_images < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_images))
    images, captions, pairings, filenames = [], [], [], []
    for idx, ex in enumerate(ds):
        images.append(ex["image"].convert("RGB"))
        filenames.append(ex.get("file_name", f"{split}_{idx}"))
        for c in range(5):
            caption = ex[f"caption_{c}"]
            if caption and caption.strip():
                captions.append(caption.strip())
                pairings.append(idx)
    return images, captions, pairings, filenames
