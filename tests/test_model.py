"""Tests for encoder, decoder, refiner, and full matting network."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.matting_network import MattingNetwork
from src.model.refiner import Refiner


def test_encoder_output_shapes():
    enc = Encoder(in_channels=4)
    x = torch.randn(1, 4, 128, 128)
    features = enc(x)
    assert len(features) == 4
    assert features[0].shape == (1, 64, 32, 32)    # H/4
    assert features[1].shape == (1, 128, 16, 16)   # H/8
    assert features[2].shape == (1, 256, 8, 8)      # H/16
    assert features[3].shape == (1, 512, 4, 4)      # H/32


def test_decoder_alpha_shape():
    dec = Decoder(predict_foreground=True)
    recurrent_feat = torch.randn(1, 64, 32, 32)
    features = [
        torch.randn(1, 64, 32, 32),
        torch.randn(1, 128, 16, 16),
        torch.randn(1, 256, 8, 8),
        torch.randn(1, 512, 4, 4),
    ]
    alpha, fg, hidden = dec(recurrent_feat, features)
    assert alpha.shape == (1, 1, 128, 128)
    assert alpha.min() >= 0 and alpha.max() <= 1
    assert fg.shape == (1, 3, 128, 128)
    assert hidden.shape == (1, 64, 32, 32)


def test_refiner_output():
    ref = Refiner(in_channels=4, hidden_channels=16)
    x = torch.randn(1, 4, 64, 64)
    out = ref(x)
    assert out.shape == (1, 1, 64, 64)
    assert out.min() >= 0 and out.max() <= 1


def test_full_network_video():
    """Full network: (B, T, 3, H, W) + masks → (B, T, 1, H, W) alphas."""
    model = MattingNetwork(in_channels=4, recurrent_channels=32, predict_foreground=True)
    frames = torch.rand(1, 4, 3, 64, 64)
    masks = torch.rand(1, 4, 1, 64, 64)
    result = model(frames, masks=masks)
    assert result["alphas"].shape == (1, 4, 1, 64, 64)
    assert result["alphas"].min() >= 0
    assert result["alphas"].max() <= 1
    assert "foregrounds" in result
    assert result["foregrounds"].shape == (1, 4, 3, 64, 64)


def test_full_network_single_frame():
    """forward_single processes one frame with explicit state management."""
    model = MattingNetwork(in_channels=4, recurrent_channels=32)
    frame = torch.rand(1, 3, 64, 64)
    mask = torch.rand(1, 1, 64, 64)

    alpha, fg, h = model.forward_single(frame, guidance=mask, h_prev=None)
    assert alpha.shape == (1, 1, 64, 64)
    assert h.shape[1] == 32  # recurrent_channels

    # Second frame: hidden state propagates
    alpha2, _, h2 = model.forward_single(frame, guidance=mask, h_prev=h)
    assert alpha2.shape == (1, 1, 64, 64)
    # Hidden state should differ between frames
    assert not torch.equal(h, h2)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
