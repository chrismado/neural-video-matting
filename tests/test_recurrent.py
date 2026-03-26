"""Tests for ConvGRU temporal recurrence."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.recurrent import ConvGRU


def test_convgru_output_shape():
    gru = ConvGRU(input_channels=64, hidden_channels=64)
    x = torch.randn(2, 64, 32, 32)
    h = gru(x, h_prev=None)
    assert h.shape == (2, 64, 32, 32)


def test_convgru_first_frame_none():
    """First frame with h_prev=None should produce valid output."""
    gru = ConvGRU(input_channels=64, hidden_channels=64)
    x = torch.randn(1, 64, 16, 16)
    h = gru(x, h_prev=None)
    assert torch.isfinite(h).all()


def test_convgru_sequential_changes():
    """Hidden state should change between frames."""
    gru = ConvGRU(input_channels=64, hidden_channels=64)
    x1 = torch.randn(1, 64, 16, 16)
    x2 = torch.randn(1, 64, 16, 16)

    h1 = gru(x1, h_prev=None)
    h2 = gru(x2, h_prev=h1)
    assert not torch.equal(h1, h2), "Hidden state should evolve over time"


def test_convgru_gradient_flow():
    """Gradients should flow through multiple timesteps."""
    gru = ConvGRU(input_channels=32, hidden_channels=32)
    frames = [torch.randn(1, 32, 8, 8, requires_grad=True) for _ in range(4)]

    h = None
    for x in frames:
        h = gru(x, h)

    loss = h.sum()
    loss.backward()

    # Gradients should reach first frame (BPTT)
    assert frames[0].grad is not None, "Gradient should reach first frame"
    assert frames[0].grad.abs().sum() > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
