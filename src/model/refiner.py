"""
Lightweight detail refinement network for high-resolution alpha edges.

The main decoder operates on downsampled features and misses fine details
like individual hair strands, fur, and semi-transparent edges. The refiner
processes full-resolution patches (RGB + coarse alpha) and outputs a
refined alpha prediction.

At inference, the refiner runs on overlapping patches stitched together
with weighted blending. At training, random patches from the "unknown"
region are sampled.
"""

import torch
import torch.nn as nn


class Refiner(nn.Module):
    """Patch-based alpha refinement for fine detail recovery.

    Input: (B, 4, patch_H, patch_W) — RGB patch + coarse alpha from decoder.
    Output: (B, 1, patch_H, patch_W) — refined alpha in [0, 1].

    Args:
        in_channels: Input channels (4 = RGB + coarse alpha).
        hidden_channels: Internal feature channels.
    """

    def __init__(self, in_channels: int = 4, hidden_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4, H, W) RGB patch concatenated with coarse alpha.

        Returns:
            (B, 1, H, W) refined alpha in [0, 1].
        """
        return torch.sigmoid(self.net(x))
