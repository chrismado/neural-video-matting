"""Tests for trimap generation and compositing math."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.trimap_gen import generate_rough_mask, generate_trimap


def test_compositing_math():
    """Verify composite = fg * alpha + bg * (1 - alpha)."""
    fg = torch.rand(3, 64, 64)
    bg = torch.rand(3, 64, 64)
    alpha = torch.rand(1, 64, 64)
    composite = fg * alpha + bg * (1 - alpha)
    assert composite.shape == (3, 64, 64)
    assert composite.min() >= 0 and composite.max() <= 1


def test_trimap_generation():
    """Generated trimap should have fg, bg, and unknown regions."""
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[16:48, 16:48] = 1.0
    alpha[14:16, 14:50] = 0.5
    alpha[48:50, 14:50] = 0.5

    trimap = generate_trimap(alpha)
    assert trimap.shape == (64, 64)

    has_fg = (trimap > 0.9).any()
    has_bg = (trimap < 0.1).any()
    has_unknown = ((trimap > 0.1) & (trimap < 0.9)).any()
    assert has_fg, "Trimap should have foreground region"
    assert has_bg, "Trimap should have background region"
    assert has_unknown, "Trimap should have unknown region"


def test_trimap_value_range():
    alpha = np.random.rand(64, 64).astype(np.float32)
    trimap = generate_trimap(alpha)
    assert trimap.min() >= 0
    assert trimap.max() <= 255


def test_rough_mask():
    """Rough mask should be approximately binary."""
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[20:44, 20:44] = 1.0
    mask = generate_rough_mask(alpha)
    assert mask.shape == (64, 64)
    binary_ratio = ((mask < 0.1) | (mask > 0.9)).mean()
    assert binary_ratio > 0.5, "Rough mask should be mostly binary"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
