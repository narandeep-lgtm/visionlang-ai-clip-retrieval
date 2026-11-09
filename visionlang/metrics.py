"""Correct bidirectional retrieval metrics for many-caption-per-image datasets.

All ranks use a *fixed* candidate gallery. For an image query, any of that
image's reference captions counts as correct. For a caption query, its paired
image counts as correct. Ties use optimistic ranks (all equal scores share rank).
"""
from __future__ import annotations

import numpy as np

KS = (1, 5, 10)


def retrieval_ranks(scores: np.ndarray, caption_to_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (text->image ranks, image->text ranks), both 1-indexed.

    scores has shape (n_captions, n_images). `caption_to_image[j]` is the
    0-based index of the relevant image for caption j.
    """
    scores = np.asarray(scores, dtype=np.float32)
    paired = np.asarray(caption_to_image, dtype=np.int64)
    if scores.ndim != 2 or len(paired) != scores.shape[0]:
        raise ValueError("scores must be [n_captions,n_images], and pairing must have n_captions entries")
    n_caps, n_images = scores.shape
    if n_caps < 1 or n_images < 1 or np.any(paired < 0) or np.any(paired >= n_images):
        raise ValueError("empty gallery or invalid pairing indexes")
    if not np.isfinite(scores).all():
        raise ValueError("all similarity scores must be finite")
    image_to_caps = [np.flatnonzero(paired == i) for i in range(n_images)]
    if any(len(x) == 0 for x in image_to_caps):
        raise ValueError("each image must have at least one matching caption")

    # Caption queries: position of its single matching image in the image gallery.
    correct_scores = scores[np.arange(n_caps), paired]
    t2i = 1 + np.sum(scores > correct_scores[:, None], axis=1)

    # Image queries: rank of the *best* among any of its matching captions.
    image_by_caption = scores.T
    best_relevant = np.array([image_by_caption[i, cap_ids].max() for i, cap_ids in enumerate(image_to_caps)])
    i2t = 1 + np.sum(image_by_caption > best_relevant[:, None], axis=1)
    return t2i.astype(np.int64), i2t.astype(np.int64)


def metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or len(ranks) == 0 or np.any(ranks < 1):
        raise ValueError("ranks must be a nonempty array of positive integers")
    results = {f"R@{k}": float(np.mean(ranks <= k) * 100) for k in KS}
    results["MRR"] = float(np.mean(1.0 / ranks))
    results["median_rank"] = float(np.median(ranks))
    results["n_queries"] = int(len(ranks))
    return results


def grouped_bootstrap_ci(
    ranks: np.ndarray,
    caption_to_image: np.ndarray | None = None,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, list[float]]:
    """95% CIs conditional on the fixed retrieval gallery.

    For text->image, resample at the *image* level so sibling captions are
    not incorrectly considered independent. For image->text, resample images.
    These are conditional CIs, not population/generalization guarantees.
    """
    ranks = np.asarray(ranks)
    if caption_to_image is None:
        group_outcomes = np.vstack([(ranks <= k).astype(float) for k in KS]).T
    else:
        paired = np.asarray(caption_to_image)
        unique = np.unique(paired)
        group_outcomes = np.array([
            [np.mean(ranks[paired == image_id] <= k) for k in KS]
            for image_id in unique
        ])
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    n = len(group_outcomes)
    means = np.empty((n_bootstrap, len(KS)))
    for i in range(n_bootstrap):
        means[i] = group_outcomes[rng.integers(0, n, size=n)].mean(axis=0) * 100
    return {
        f"R@{k}": np.percentile(means[:, j], [2.5, 97.5]).round(3).tolist()
        for j, k in enumerate(KS)
    }


def caption_length_analysis(ranks: np.ndarray, captions: list[str]) -> list[dict]:
    if len(ranks) != len(captions):
        raise ValueError("captions and ranks must be equal length")
    lengths = np.array([len(c.split()) for c in captions])
    # Fixed, predefined bins to avoid outcome-informed threshold selection.
    bins = [("short (<=8 words)", lengths <= 8),
            ("medium (9-13 words)", (lengths >= 9) & (lengths <= 13)),
            ("long (>=14 words)", lengths >= 14)]
    return [{"length_bin": name, "n_captions": int(mask.sum()),
             **metrics_from_ranks(np.asarray(ranks)[mask])}
            for name, mask in bins if mask.any()]
