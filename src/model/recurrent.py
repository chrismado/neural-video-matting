"""
Convolutional GRU for temporal propagation between video frames.

Replaces fully-connected GRU gates with convolutional gates to preserve
spatial structure. The hidden state carries temporal context — edge memory,
motion boundaries, and matte consistency across frames. This is what makes
it VIDEO matting, not just per-frame matting.

Operates at H/4 resolution (64 channels) for memory efficiency.
"""

import torch
import torch.nn as nn


class ConvGRU(nn.Module):
    """Convolutional Gated Recurrent Unit operating on spatial feature maps.

    Args:
        input_channels: Channels of the input features.
        hidden_channels: Channels of the hidden state.
        kernel_size: Kernel size for convolutional gates.
    """

    def __init__(
        self,
        input_channels: int = 64,
        hidden_channels: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        combined = input_channels + hidden_channels

        # Update gate (z)
        self.conv_z = nn.Conv2d(combined, hidden_channels, kernel_size, padding=padding)
        # Reset gate (r)
        self.conv_r = nn.Conv2d(combined, hidden_channels, kernel_size, padding=padding)
        # Candidate hidden state
        self.conv_h = nn.Conv2d(combined, hidden_channels, kernel_size, padding=padding)

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, input_channels, H, W) current frame features.
            h_prev: (B, hidden_channels, H, W) previous hidden state.
                    None for first frame → initialised to zeros.

        Returns:
            h: (B, hidden_channels, H, W) updated hidden state.
        """
        if h_prev is None:
            B, _, H, W = x.shape
            h_prev = torch.zeros(B, self.hidden_channels, H, W, device=x.device)

        combined = torch.cat([x, h_prev], dim=1)

        z = torch.sigmoid(self.conv_z(combined))   # update gate
        r = torch.sigmoid(self.conv_r(combined))   # reset gate

        combined_r = torch.cat([x, r * h_prev], dim=1)
        h_candidate = torch.tanh(self.conv_h(combined_r))

        h = (1 - z) * h_prev + z * h_candidate
        return h
