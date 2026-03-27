"""Generate trimaps and rough masks from ground-truth alpha mattes."""

import random

import cv2
import numpy as np
import torch


def generate_trimap(
    alpha: np.ndarray,
    fg_threshold: float = 0.95,
    bg_threshold: float = 0.05,
    dilation_range: tuple = (5, 25),
) -> np.ndarray:
    """Generate a trimap from a ground-truth alpha matte.

    Regions:
        - Definite foreground (alpha > fg_threshold) = 1.0
        - Definite background (alpha < bg_threshold) = 0.0
        - Unknown (dilated boundary region)           = 0.5

    Args:
        alpha: (H, W) or (H, W, 1) float array in [0, 1].
        fg_threshold: Threshold above which pixels are definite foreground.
        bg_threshold: Threshold below which pixels are definite background.
        dilation_range: (min, max) kernel size for random dilation of unknown region.

    Returns:
        trimap: (H, W) float32 array with values in {0.0, 0.5, 1.0}.
    """
    if alpha.ndim == 3:
        alpha = alpha[..., 0]

    fg = (alpha > fg_threshold).astype(np.uint8)
    bg = (alpha < bg_threshold).astype(np.uint8)

    # Create unknown region by dilating the boundary
    kernel_size = random.randint(dilation_range[0], dilation_range[1])
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    fg_dilated = cv2.dilate(fg, kernel)
    bg_dilated = cv2.dilate(bg, kernel)

    trimap = np.full_like(alpha, 0.5, dtype=np.float32)
    # Only set definite regions where dilation of the opposite class hasn't reached
    trimap[fg == 1] = 1.0
    trimap[bg == 1] = 0.0
    # Force unknown in the transition zone
    unknown = (fg_dilated - fg) | (bg_dilated - bg)
    trimap[unknown == 1] = 0.5

    return trimap


def generate_rough_mask(
    alpha: np.ndarray,
    threshold: float = 0.5,
    dilation_range: tuple = (0, 15),
    erosion_range: tuple = (0, 10),
) -> np.ndarray:
    """Generate a rough binary mask from a GT alpha matte.

    Applies threshold then random morphological operations to simulate
    a noisy/imprecise input mask.

    Args:
        alpha: (H, W) or (H, W, 1) float array in [0, 1].
        threshold: Binarization threshold.
        dilation_range: (min, max) kernel size for random dilation.
        erosion_range: (min, max) kernel size for random erosion.

    Returns:
        mask: (H, W) float32 binary mask in {0.0, 1.0}.
    """
    if alpha.ndim == 3:
        alpha = alpha[..., 0]

    mask = (alpha > threshold).astype(np.uint8)

    # Random erosion
    erosion_size = random.randint(erosion_range[0], erosion_range[1])
    if erosion_size > 0:
        if erosion_size % 2 == 0:
            erosion_size += 1
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)
        )
        mask = cv2.erode(mask, k)

    # Random dilation
    dilation_size = random.randint(dilation_range[0], dilation_range[1])
    if dilation_size > 0:
        if dilation_size % 2 == 0:
            dilation_size += 1
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
        )
        mask = cv2.dilate(mask, k)

    return mask.astype(np.float32)


def generate_trimap_batch(alpha: torch.Tensor, **kwargs) -> torch.Tensor:
    """Generate trimaps for a batch of alpha mattes.

    Args:
        alpha: (B, 1, H, W) or (B, T, 1, H, W) tensor in [0, 1].

    Returns:
        trimaps: same shape as input.
    """
    orig_shape = alpha.shape
    if alpha.ndim == 5:
        B, T, C, H, W = alpha.shape
        alpha_flat = alpha.reshape(B * T, H, W).cpu().numpy()
    else:
        B, C, H, W = alpha.shape
        alpha_flat = alpha.reshape(B, H, W).cpu().numpy()

    trimaps = []
    for i in range(alpha_flat.shape[0]):
        t = generate_trimap(alpha_flat[i], **kwargs)
        trimaps.append(t)

    trimaps = np.stack(trimaps, axis=0)
    trimaps = torch.from_numpy(trimaps).to(alpha.device)

    if len(orig_shape) == 5:
        trimaps = trimaps.reshape(B, T, 1, H, W)
    else:
        trimaps = trimaps.unsqueeze(1)  # (B, 1, H, W)

    return trimaps


def generate_rough_mask_batch(alpha: torch.Tensor, **kwargs) -> torch.Tensor:
    """Generate rough masks for a batch of alpha mattes.

    Args:
        alpha: (B, 1, H, W) or (B, T, 1, H, W) tensor in [0, 1].

    Returns:
        masks: same shape as input.
    """
    orig_shape = alpha.shape
    if alpha.ndim == 5:
        B, T, C, H, W = alpha.shape
        alpha_flat = alpha.reshape(B * T, H, W).cpu().numpy()
    else:
        B, C, H, W = alpha.shape
        alpha_flat = alpha.reshape(B, H, W).cpu().numpy()

    masks = []
    for i in range(alpha_flat.shape[0]):
        m = generate_rough_mask(alpha_flat[i], **kwargs)
        masks.append(m)

    masks = np.stack(masks, axis=0)
    masks = torch.from_numpy(masks).to(alpha.device)

    if len(orig_shape) == 5:
        masks = masks.reshape(B, T, 1, H, W)
    else:
        masks = masks.unsqueeze(1)

    return masks
