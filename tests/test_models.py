import numpy as np
import torch
import pytest
from PIL import Image
from visionlang.models import _projected_feature, corrupt_image


def test_feature_extraction_supports_tensor_and_object():
    from types import SimpleNamespace
    t = torch.randn(2, 8)
    assert torch.equal(_projected_feature(t), t)
    assert torch.equal(_projected_feature(SimpleNamespace(pooler_output=t)), t)


def test_corruption_preserves_dimensions_and_rgb():
    src = Image.new("RGB", (20, 14), color=(10, 80, 110))
    for mode in ["none", "blur", "grayscale"]:
        res = corrupt_image(src, mode)
        assert res.size == (20, 14)
        assert res.mode == "RGB"
    with pytest.raises(ValueError):
        corrupt_image(src, "unknown")
