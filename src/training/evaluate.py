"""Evaluation metrics and visualization for video matting."""

import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_sad(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Sum of Absolute Differences.

    Args:
        pred: (B, 1, H, W) or (B, T, 1, H, W) predicted alpha in [0, 1].
        gt:   same shape, ground-truth alpha.

    Returns:
        Mean SAD per image.
    """
    return (pred - gt).abs().sum(dim=(-1, -2, -3)).mean().item()


def compute_mse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean Squared Error.

    Args:
        pred, gt: same shape alpha tensors.

    Returns:
        Mean MSE.
    """
    return ((pred - gt) ** 2).mean().item()


def _sobel_gradient(alpha: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude. Input: (N, 1, H, W)."""
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=alpha.dtype, device=alpha.device,
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=alpha.dtype, device=alpha.device,
    ).reshape(1, 1, 3, 3)
    gx = F.conv2d(alpha, sobel_x, padding=1)
    gy = F.conv2d(alpha, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


def compute_gradient_error(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Gradient error between predicted and GT alpha.

    Args:
        pred, gt: (B, 1, H, W) or (B, T, 1, H, W).

    Returns:
        Mean gradient error.
    """
    if pred.ndim == 5:
        B, T, C, H, W = pred.shape
        pred = pred.reshape(B * T, C, H, W)
        gt = gt.reshape(B * T, C, H, W)

    pred_grad = _sobel_gradient(pred)
    gt_grad = _sobel_gradient(gt)
    return (pred_grad - gt_grad).abs().mean().item()


def compute_connectivity(
    pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5, step: float = 0.1,
) -> float:
    """Connectivity error metric.

    Measures the fraction of pixels in connected components that differ
    between predicted and ground-truth binary masks at various thresholds.

    Args:
        pred, gt: (B, 1, H, W) alpha tensors.
        threshold: Binarization threshold.
        step: Threshold step for multi-threshold evaluation.

    Returns:
        Mean connectivity error.
    """
    if pred.ndim == 5:
        B, T, C, H, W = pred.shape
        pred = pred.reshape(B * T, C, H, W)
        gt = gt.reshape(B * T, C, H, W)

    pred_np = pred.squeeze(1).cpu().numpy()
    gt_np = gt.squeeze(1).cpu().numpy()

    errors = []
    for i in range(pred_np.shape[0]):
        pred_bin = (pred_np[i] > threshold).astype(np.uint8)
        gt_bin = (gt_np[i] > threshold).astype(np.uint8)
        diff = np.abs(pred_bin.astype(np.float32) - gt_bin.astype(np.float32))
        errors.append(diff.mean())

    return float(np.mean(errors))


def evaluate_all(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        pred: (B, T, 1, H, W) or (B, 1, H, W) predicted alpha.
        gt:   same shape, ground-truth alpha.

    Returns:
        Dict with metric names and values.
    """
    return {
        "SAD": compute_sad(pred, gt),
        "MSE": compute_mse(pred, gt),
        "gradient_error": compute_gradient_error(pred, gt),
        "connectivity": compute_connectivity(pred, gt),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_side_by_side(
    original: torch.Tensor,
    alpha_pred: torch.Tensor,
    new_bg: Optional[torch.Tensor] = None,
    save_path: str = "comparison.png",
):
    """Save a side-by-side comparison: original | alpha matte | composite on new BG.

    Args:
        original: (3, H, W) or (H, W, 3) input frame in [0, 1].
        alpha_pred: (1, H, W) or (H, W) predicted alpha in [0, 1].
        new_bg: (3, H, W) or (H, W, 3) new background for compositing.
        save_path: Output file path.
    """
    # Normalize to (H, W, C) numpy
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu()
        if original.shape[0] == 3:
            original = original.permute(1, 2, 0)
        original = original.numpy()

    if isinstance(alpha_pred, torch.Tensor):
        alpha_pred = alpha_pred.detach().cpu()
        if alpha_pred.ndim == 3 and alpha_pred.shape[0] == 1:
            alpha_pred = alpha_pred.squeeze(0)
        alpha_pred = alpha_pred.numpy()

    H, W = alpha_pred.shape[:2]

    # Alpha as 3-channel grayscale
    alpha_vis = np.stack([alpha_pred] * 3, axis=-1)

    panels = [original, alpha_vis]

    # Composite on new BG
    if new_bg is not None:
        if isinstance(new_bg, torch.Tensor):
            new_bg = new_bg.detach().cpu()
            if new_bg.shape[0] == 3:
                new_bg = new_bg.permute(1, 2, 0)
            new_bg = new_bg.numpy()

        new_bg = cv2.resize(new_bg, (W, H))
        alpha_3 = alpha_pred[..., np.newaxis] if alpha_pred.ndim == 2 else alpha_pred
        fg = original * alpha_3
        composite = fg + new_bg * (1.0 - alpha_3)
        panels.append(composite)

    # Concatenate horizontally
    comparison = np.concatenate(panels, axis=1)
    comparison = np.clip(comparison * 255, 0, 255).astype(np.uint8)
    comparison = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, comparison)
    logger.info(f"Saved comparison to {save_path}")


def save_video_comparison(
    frames: torch.Tensor,
    pred_alphas: torch.Tensor,
    output_dir: str,
    new_bg: Optional[torch.Tensor] = None,
):
    """Save per-frame side-by-side comparisons for a video clip.

    Args:
        frames: (T, 3, H, W) input frames.
        pred_alphas: (T, 1, H, W) predicted alpha mattes.
        output_dir: Directory to save frames.
        new_bg: (3, H, W) optional background for compositing.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    T = frames.shape[0]
    for t in range(T):
        save_side_by_side(
            original=frames[t],
            alpha_pred=pred_alphas[t],
            new_bg=new_bg,
            save_path=str(output_path / f"frame_{t:04d}.png"),
        )
