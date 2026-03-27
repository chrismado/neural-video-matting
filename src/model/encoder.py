"""
ResNet-34 style multi-scale feature encoder for video matting.

Extracts features at 4 resolution scales from input frames concatenated
with optional guidance (trimap or rough mask). Built from scratch — no
pretrained backbone weights.

Input: (B, C_in, H, W) where C_in = 3 (RGB) + 1 (mask/trimap) = 4
Output: list of multi-scale features [f1 (H/4), f2 (H/8), f3 (H/16), f4 (H/32)]
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet basic residual block: two 3x3 convolutions with skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_stage(in_ch: int, out_ch: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
    """Build a stage of BasicBlocks. First block may downsample."""
    blocks = [BasicBlock(in_ch, out_ch, stride=stride)]
    for _ in range(1, num_blocks):
        blocks.append(BasicBlock(out_ch, out_ch))
    return nn.Sequential(*blocks)


class Encoder(nn.Module):
    """ResNet-34 style encoder producing multi-scale features.

    Args:
        in_channels: Input channels (4 for RGB + mask, 3 for RGB-only).
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()
        # Stage 0: Conv7x7 stride 2 → BN → ReLU → MaxPool stride 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # Stage 1-4: ResNet-34 block counts [3, 4, 6, 3]
        self.stage1 = _make_stage(64, 64, 3)  # H/4, 64ch
        self.stage2 = _make_stage(64, 128, 4, stride=2)  # H/8, 128ch
        self.stage3 = _make_stage(128, 256, 6, stride=2)  # H/16, 256ch
        self.stage4 = _make_stage(256, 512, 3, stride=2)  # H/32, 512ch

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (B, C_in, H, W) input (RGB + guidance concatenated).

        Returns:
            [f1, f2, f3, f4] feature maps at H/4, H/8, H/16, H/32.
        """
        x = self.stem(x)  # (B, 64, H/4, W/4)
        f1 = self.stage1(x)  # (B, 64, H/4, W/4)
        f2 = self.stage2(f1)  # (B, 128, H/8, W/8)
        f3 = self.stage3(f2)  # (B, 256, H/16, W/16)
        f4 = self.stage4(f3)  # (B, 512, H/32, W/32)
        return [f1, f2, f3, f4]
