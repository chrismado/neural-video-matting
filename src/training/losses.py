"""Loss functions for neural video matting training."""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def alpha_l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 loss between predicted and ground-truth alpha mattes.

    Args:
        pred: (B, T, 1, H, W) or (B, 1, H, W) predicted alpha.
        gt:   same shape as pred, ground-truth alpha.

    Returns:
        Scalar loss.
    """
    return F.l1_loss(pred, gt)


def _build_laplacian_pyramid(img: torch.Tensor, levels: int = 4):
    """Build a Laplacian pyramid from an image tensor.

    Args:
        img: (N, C, H, W) tensor.
        levels: Number of pyramid levels.

    Returns:
        List of tensors at each pyramid level.
    """
    pyramid = []
    current = img
    for i in range(levels):
        down = F.avg_pool2d(current, 2, stride=2)
        up = F.interpolate(down, size=current.shape[2:], mode="bilinear", align_corners=False)
        lap = current - up
        pyramid.append(lap)
        current = down
    pyramid.append(current)
    return pyramid


def laplacian_loss(pred: torch.Tensor, gt: torch.Tensor, levels: int = 4) -> torch.Tensor:
    """Multi-level Laplacian pyramid L1 loss.

    Args:
        pred: (B, T, 1, H, W) or (B, 1, H, W) predicted alpha.
        gt:   same shape, ground-truth alpha.
        levels: Number of Laplacian pyramid levels.

    Returns:
        Scalar loss.
    """
    if pred.ndim == 5:
        B, T, C, H, W = pred.shape
        pred = pred.reshape(B * T, C, H, W)
        gt = gt.reshape(B * T, C, H, W)

    pred_pyr = _build_laplacian_pyramid(pred, levels)
    gt_pyr = _build_laplacian_pyramid(gt, levels)

    loss = torch.tensor(0.0, device=pred.device)
    for p_level, g_level in zip(pred_pyr, gt_pyr):
        loss = loss + F.l1_loss(p_level, g_level)

    return loss / (levels + 1)


def _sobel_filter(img: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude of an image.

    Args:
        img: (N, 1, H, W) tensor.

    Returns:
        (N, 1, H, W) gradient magnitude.
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device
    ).reshape(1, 1, 3, 3)

    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-6)


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Sobel gradient loss between predicted and GT alpha mattes.

    Args:
        pred: (B, T, 1, H, W) or (B, 1, H, W) predicted alpha.
        gt:   same shape, ground-truth alpha.

    Returns:
        Scalar loss.
    """
    if pred.ndim == 5:
        B, T, C, H, W = pred.shape
        pred = pred.reshape(B * T, C, H, W)
        gt = gt.reshape(B * T, C, H, W)

    pred_grad = _sobel_filter(pred)
    gt_grad = _sobel_filter(gt)

    return F.l1_loss(pred_grad, gt_grad)


def temporal_consistency_loss(pred_alpha: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Temporal consistency loss using simple frame-difference based warping.

    Warps previous frame's alpha prediction to current frame using the
    difference in frame intensities as a proxy for motion.

    Args:
        pred_alpha: (B, T, 1, H, W) predicted alpha sequence.
        frames: (B, T, 3, H, W) input composite frames.

    Returns:
        Scalar loss.
    """
    B, T, C, H, W = pred_alpha.shape
    if T < 2:
        return torch.tensor(0.0, device=pred_alpha.device)

    loss = torch.tensor(0.0, device=pred_alpha.device)

    for t in range(1, T):
        # Simple temporal coherence: consecutive alpha predictions should be similar
        # weighted by the similarity of corresponding frames
        frame_diff = (frames[:, t] - frames[:, t - 1]).abs().mean(dim=1, keepdim=True)
        # Weight: regions with small frame difference should have similar alpha
        weight = 1.0 - frame_diff.clamp(0, 1)

        alpha_diff = (pred_alpha[:, t] - pred_alpha[:, t - 1]).abs()
        loss = loss + (alpha_diff * weight).mean()

    return loss / (T - 1)


def foreground_l1_loss(
    pred_fg: torch.Tensor,
    gt_fg: torch.Tensor,
    alpha_gt: torch.Tensor,
) -> torch.Tensor:
    """Alpha-weighted foreground L1 loss.

    Args:
        pred_fg: (B, T, 3, H, W) predicted foreground.
        gt_fg:   (B, T, 3, H, W) ground-truth foreground.
        alpha_gt: (B, T, 1, H, W) ground-truth alpha (used as weight).

    Returns:
        Scalar loss.
    """
    diff = (pred_fg - gt_fg).abs()
    weighted = diff * alpha_gt
    return weighted.mean()


def combined_loss(
    pred_alpha: torch.Tensor,
    gt_alpha: torch.Tensor,
    frames: torch.Tensor,
    pred_fg: Optional[torch.Tensor] = None,
    gt_fg: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Combined training loss with configurable weights.

    Args:
        pred_alpha: (B, T, 1, H, W) predicted alpha.
        gt_alpha: (B, T, 1, H, W) ground-truth alpha.
        frames: (B, T, 3, H, W) composite input frames.
        pred_fg: (B, T, 3, H, W) predicted foreground (optional).
        gt_fg: (B, T, 3, H, W) GT foreground (optional).
        weights: Dict of loss name -> weight. Defaults provided if None.

    Returns:
        Dict with 'total' and individual loss terms.
    """
    if weights is None:
        weights = {
            "alpha_l1": 1.0,
            "laplacian": 1.0,
            "gradient": 0.5,
            "temporal": 0.5,
            "foreground": 0.5,
        }

    losses = {}

    # Alpha L1
    l_alpha = alpha_l1_loss(pred_alpha, gt_alpha)
    losses["alpha_l1"] = l_alpha

    # Laplacian
    l_lap = laplacian_loss(pred_alpha, gt_alpha)
    losses["laplacian"] = l_lap

    # Gradient
    l_grad = gradient_loss(pred_alpha, gt_alpha)
    losses["gradient"] = l_grad

    # Temporal consistency
    l_temp = temporal_consistency_loss(pred_alpha, frames)
    losses["temporal"] = l_temp

    # Foreground
    if pred_fg is not None and gt_fg is not None:
        l_fg = foreground_l1_loss(pred_fg, gt_fg, gt_alpha)
        losses["foreground"] = l_fg
    else:
        losses["foreground"] = torch.tensor(0.0, device=pred_alpha.device)

    # Combined
    total = torch.tensor(0.0, device=pred_alpha.device)
    for name, value in losses.items():
        w = weights.get(name, 0.0)
        total = total + w * value

    losses["total"] = total
    return losses
