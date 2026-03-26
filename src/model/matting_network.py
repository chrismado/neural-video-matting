"""
Full neural video matting pipeline.

Combines encoder, ConvGRU temporal recurrence, decoder, and detail refiner
into a single forward pass that processes video sequences frame-by-frame
with recurrent state propagation.

Supports both trimap-guided and trimap-free (mask-guided) modes.
"""

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .recurrent import ConvGRU
from .refiner import Refiner


class MattingNetwork(nn.Module):
    """Recurrent video matting network.

    Architecture:
        frame + guidance → Encoder → multi-scale features
                                        ↓ (f1 at H/4)
                                     ConvGRU ← hidden state
                                        ↓
                                     Decoder ← skip connections (f2, f3, f4)
                                        ↓
                                   coarse alpha + foreground
                                        ↓ (optional)
                                     Refiner → refined alpha

    Args:
        in_channels: Input channels (4 for RGB + guidance, 3 for RGB-only).
        recurrent_channels: Hidden state channels for ConvGRU.
        predict_foreground: Whether the decoder also predicts foreground RGB.
        use_refiner: Whether to apply the detail refiner.
    """

    def __init__(
        self,
        in_channels: int = 4,
        recurrent_channels: int = 64,
        predict_foreground: bool = True,
        use_refiner: bool = False,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.recurrent = ConvGRU(
            input_channels=64,
            hidden_channels=recurrent_channels,
        )
        self.decoder = Decoder(
            predict_foreground=predict_foreground,
            recurrent_channels=recurrent_channels,
        )
        self.refiner = Refiner() if use_refiner else None
        self.use_refiner = use_refiner

    def forward(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor | None = None,
        trimaps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process a sequence of video frames.

        Args:
            frames: (B, T, 3, H, W) video frames in [0, 1].
            masks: (B, T, 1, H, W) optional rough masks (trimap-free mode).
            trimaps: (B, T, 1, H, W) optional trimaps (trimap mode).

        Returns:
            Dict with:
                'alphas': (B, T, 1, H, W) predicted alpha mattes in [0, 1].
                'foregrounds': (B, T, 3, H, W) predicted foregrounds (if enabled).
        """
        B, T = frames.shape[:2]
        h = None  # recurrent hidden state
        alphas = []
        foregrounds = []

        for t in range(T):
            frame = frames[:, t]  # (B, 3, H, W)

            # Prepare input: concatenate guidance channel
            if trimaps is not None:
                x = torch.cat([frame, trimaps[:, t]], dim=1)  # (B, 4, H, W)
            elif masks is not None:
                x = torch.cat([frame, masks[:, t]], dim=1)  # (B, 4, H, W)
            else:
                # No guidance — pad with zeros (for testing)
                x = torch.cat([
                    frame,
                    torch.zeros(B, 1, *frame.shape[2:], device=frame.device),
                ], dim=1)

            # Encode
            features = self.encoder(x)  # [f1 (H/4), f2 (H/8), f3 (H/16), f4 (H/32)]

            # Temporal recurrence at H/4 resolution
            h = self.recurrent(features[0], h)

            # Decode with temporal context
            alpha, fg, _ = self.decoder(h, features)

            alphas.append(alpha)
            if fg is not None:
                foregrounds.append(fg)

        result = {"alphas": torch.stack(alphas, dim=1)}
        if foregrounds:
            result["foregrounds"] = torch.stack(foregrounds, dim=1)

        return result

    def forward_single(
        self,
        frame: torch.Tensor,
        guidance: torch.Tensor | None = None,
        h_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Process a single frame (for inference with manual state management).

        Args:
            frame: (B, 3, H, W) single frame.
            guidance: (B, 1, H, W) optional mask or trimap.
            h_prev: Previous hidden state from ConvGRU.

        Returns:
            alpha: (B, 1, H, W) predicted alpha matte.
            foreground: (B, 3, H, W) or None.
            h: Updated hidden state.
        """
        if guidance is not None:
            x = torch.cat([frame, guidance], dim=1)
        else:
            x = torch.cat([
                frame,
                torch.zeros(frame.shape[0], 1, *frame.shape[2:], device=frame.device),
            ], dim=1)

        features = self.encoder(x)
        h = self.recurrent(features[0], h_prev)
        alpha, fg, _ = self.decoder(h, features)

        return alpha, fg, h
