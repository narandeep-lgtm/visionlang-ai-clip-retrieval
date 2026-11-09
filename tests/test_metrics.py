import numpy as np
import pytest
from visionlang.metrics import (caption_length_analysis, grouped_bootstrap_ci,
                                metrics_from_ranks, retrieval_ranks)


def test_perfect_bidirectional_retrieval_with_multiple_captions():
    # Each image has two captions. Caption 0+1 -> image 0; 2+3 -> image 1.
    scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    t2i, i2t = retrieval_ranks(scores, np.array([0, 0, 1, 1]))
    assert list(t2i) == [1, 1, 1, 1]
    assert list(i2t) == [1, 1]
    assert metrics_from_ranks(t2i)["R@1"] == 100.0


def test_image_to_text_accepts_any_valid_caption():
    scores = np.array([[0.2, 0.8], [0.95, 0.7], [0.9, 0.99], [0.1, 0.85]])
    # Image 0 gets caption 1 at top even though caption 0 is a poor match.
    t2i, i2t = retrieval_ranks(scores, np.array([0, 0, 1, 1]))
    assert i2t[0] == 1
    assert i2t[1] == 1
    assert t2i[0] == 2


def test_t2i_ranks_imperfect():
    s = np.array([[0.3, 0.6], [0.1, 0.9], [0.9, 0.3], [0.8, 0.1]])
    t2i, _ = retrieval_ranks(s, np.array([0, 0, 1, 1]))
    assert list(t2i) == [2, 2, 2, 2]
    assert metrics_from_ranks(t2i)["R@1"] == 0.0


def test_error_on_unpaired_image():
    with pytest.raises(ValueError):
        retrieval_ranks(np.array([[0.5, 0.4]]), np.array([0]))


def test_bootstrap_deterministic_and_bounds():
    ranks = np.array([1, 5, 2, 1, 3, 20])
    paired = np.array([0, 0, 1, 1, 2, 2])
    first = grouped_bootstrap_ci(ranks, paired, n_bootstrap=100, seed=12)
    second = grouped_bootstrap_ci(ranks, paired, n_bootstrap=100, seed=12)
    assert first == second
    assert all(0 <= lo <= hi <= 100 for lo, hi in first.values())


def test_caption_length_bins():
    caps = ["a " * 7, "a " * 12, "a " * 15]
    rows = caption_length_analysis(np.array([1, 2, 3]), caps)
    assert [r["n_captions"] for r in rows] == [1, 1, 1]
