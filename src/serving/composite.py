"""Foreground compositing with spill suppression."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def composite_foreground(
    fg: np.ndarray,
    alpha: np.ndarray,
    bg: np.ndarray,
    spill_suppression: bool = True,
    spill_strength: float = 0.5,
) -> np.ndarray:
    """Composite foreground onto a new background using the alpha matte.

    output = fg * alpha + bg * (1 - alpha)

    Args:
        fg: (H, W, 3) foreground image in [0, 255] uint8 or [0, 1] float32.
        alpha: (H, W) alpha matte in [0, 1] float32.
        bg: (H, W, 3) new background, same dtype convention as fg.
        spill_suppression: Whether to suppress color spill from the original background.
        spill_strength: Strength of spill suppression in [0, 1].

    Returns:
        (H, W, 3) composited image, same dtype as fg.
    """
    input_is_uint8 = fg.dtype == np.uint8

    # Normalize to float32
    if input_is_uint8:
        fg = fg.astype(np.float32) / 255.0
        bg = bg.astype(np.float32) / 255.0

    # Resize bg to match fg if needed
    if bg.shape[:2] != fg.shape[:2]:
        bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))

    # Ensure alpha is (H, W, 1) for broadcasting
    if alpha.ndim == 2:
        alpha_3 = alpha[..., np.newaxis]
    else:
        alpha_3 = alpha

    # Spill suppression: reduce color bleeding at semi-transparent edges
    if spill_suppression:
        fg = suppress_spill(fg, alpha, strength=spill_strength)

    # Composite
    output = fg * alpha_3 + bg * (1.0 - alpha_3)
    output = np.clip(output, 0.0, 1.0)

    if input_is_uint8:
        output = (output * 255).astype(np.uint8)

    return output


def suppress_spill(
    fg: np.ndarray,
    alpha: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """Suppress color spill (green/blue screen bleed) in semi-transparent regions.

    Detects dominant spill color in transparent edges and reduces it.

    Args:
        fg: (H, W, 3) float32 foreground in [0, 1].
        alpha: (H, W) float32 alpha in [0, 1].
        strength: Suppression strength in [0, 1].

    Returns:
        (H, W, 3) float32 spill-suppressed foreground.
    """
    # Find semi-transparent edge pixels
    edge_mask = (alpha > 0.05) & (alpha < 0.95)

    if not edge_mask.any():
        return fg

    # Estimate spill color from edge regions
    edge_pixels = fg[edge_mask]  # (N, 3)
    mean_color = edge_pixels.mean(axis=0)

    # Determine dominant spill channel (typically green for green screen)
    spill_channel = np.argmax(mean_color)
    other_channels = [c for c in range(3) if c != spill_channel]

    # Compute spill amount: how much the dominant channel exceeds the average of others
    fg_corrected = fg.copy()
    avg_other = np.mean(fg[..., other_channels], axis=-1)
    spill_amount = np.maximum(fg[..., spill_channel] - avg_other, 0)

    # Suppress in semi-transparent regions (weighted by 1 - alpha)
    suppression_weight = (1.0 - alpha) * strength
    fg_corrected[..., spill_channel] -= spill_amount * suppression_weight

    return np.clip(fg_corrected, 0.0, 1.0)


def composite_video(
    fg_frames: list,
    alphas: list,
    bg_frames: list,
    spill_suppression: bool = True,
    spill_strength: float = 0.5,
) -> list:
    """Composite a sequence of foreground frames onto background frames.

    Args:
        fg_frames: List of (H, W, 3) foreground frames.
        alphas: List of (H, W) alpha mattes.
        bg_frames: List of (H, W, 3) background frames. If shorter than fg_frames,
                   the last frame is repeated.
        spill_suppression: Whether to apply spill suppression.
        spill_strength: Spill suppression strength.

    Returns:
        List of (H, W, 3) composited frames.
    """
    results = []
    for i in range(len(fg_frames)):
        bg_idx = min(i, len(bg_frames) - 1)
        composited = composite_foreground(
            fg=fg_frames[i],
            alpha=alphas[i],
            bg=bg_frames[bg_idx],
            spill_suppression=spill_suppression,
            spill_strength=spill_strength,
        )
        results.append(composited)
    return results
