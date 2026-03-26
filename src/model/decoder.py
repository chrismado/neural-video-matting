"""
Multi-scale alpha matte decoder with skip connections.

Top-down decoder that fuses multi-scale encoder features to produce:
- Alpha matte (B, 1, H, W) via sigmoid activation
- Optional foreground prediction (B, 3, H, W)
- Hidden features at H/4 resolution for the recurrent module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    """Bilinear upsample 2x → Conv3x3 → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.conv(x)


class ConvBlock(nn.Module):
    """Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    """Multi-scale decoder producing alpha matte from encoder/recurrent features.

    Accepts recurrent hidden state (at H/4 resolution) + skip connections from
    the encoder at H/8, H/16, H/32.

    Args:
        predict_foreground: If True, also output a 3-channel foreground estimate.
    """

    def __init__(self, predict_foreground: bool = True, recurrent_channels: int = 64):
        super().__init__()
        self.predict_foreground = predict_foreground

        # f4 (512, H/32) → upsample + cat f3 (256) → 256
        self.up3 = UpBlock(512, 256)
        self.conv3 = ConvBlock(256 + 256, 256)

        # → upsample + cat f2 (128) → 128
        self.up2 = UpBlock(256, 128)
        self.conv2 = ConvBlock(128 + 128, 128)

        # → upsample + cat recurrent (recurrent_channels) → 64
        self.up1 = UpBlock(128, 64)
        self.conv1 = ConvBlock(64 + recurrent_channels, 64)

        # Final upsample 4x → alpha
        self.alpha_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

        # Optional foreground prediction
        if predict_foreground:
            self.fg_head = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, 1),
            )

    def forward(
        self,
        recurrent_features: torch.Tensor,
        encoder_features: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Args:
            recurrent_features: (B, 64, H/4, W/4) from ConvGRU (or f1 if no recurrence).
            encoder_features: [f1, f2, f3, f4] from encoder.

        Returns:
            alpha: (B, 1, H, W) predicted alpha matte in [0, 1].
            foreground: (B, 3, H, W) predicted foreground RGB, or None.
            hidden_features: (B, 64, H/4, W/4) features before final projection.
        """
        f1, f2, f3, f4 = encoder_features

        # Top-down with skip connections
        x = self.up3(f4)
        x = self.conv3(torch.cat([x, f3], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, f2], dim=1))

        x = self.up1(x)
        x = self.conv1(torch.cat([x, recurrent_features], dim=1))

        hidden_features = x  # (B, 64, H/4, W/4)

        # Upsample 4x to full resolution
        x_full = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)

        alpha = torch.sigmoid(self.alpha_head(x_full))

        foreground = None
        if self.predict_foreground:
            foreground = torch.sigmoid(self.fg_head(x_full))

        return alpha, foreground, hidden_features
