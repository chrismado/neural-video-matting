"""Paired spatiotemporal augmentations for video matting training.

All transforms are applied consistently across ALL T frames and
between composite, alpha, foreground, and background arrays.
"""

import random
from typing import Tuple

import cv2
import numpy as np


class VideoMattingAugmentation:
    """Spatiotemporal augmentations that maintain consistency across frames and modalities.

    Args:
        crop_size: (H, W) output crop size.
        random_flip: Whether to apply random horizontal flip.
        random_affine: Whether to apply random affine transformations.
        affine_degree: Max rotation degrees for affine.
        affine_scale: (min_scale, max_scale) range.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int] = (512, 512),
        random_flip: bool = True,
        random_affine: bool = True,
        affine_degree: float = 10.0,
        affine_scale: Tuple[float, float] = (0.9, 1.1),
    ):
        self.crop_size = crop_size  # (H, W)
        self.random_flip = random_flip
        self.random_affine = random_affine
        self.affine_degree = affine_degree
        self.affine_scale = affine_scale

    def __call__(
        self,
        composite: np.ndarray,
        alpha: np.ndarray,
        foreground: np.ndarray,
        background: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply augmentations to all arrays consistently.

        Args:
            composite: (T, H, W, 3)
            alpha: (T, H, W, 1)
            foreground: (T, H, W, 3)
            background: (T, H, W, 3)

        Returns:
            Augmented (composite, alpha, foreground, background) with same shapes.
        """
        T, H, W, _ = composite.shape
        crop_h, crop_w = self.crop_size

        # --- Random crop (same location for all frames) ---
        if H > crop_h and W > crop_w:
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
        else:
            top, left = 0, 0
            crop_h, crop_w = H, W

        composite = composite[:, top : top + crop_h, left : left + crop_w, :]
        alpha = alpha[:, top : top + crop_h, left : left + crop_w, :]
        foreground = foreground[:, top : top + crop_h, left : left + crop_w, :]
        background = background[:, top : top + crop_h, left : left + crop_w, :]

        # --- Random horizontal flip (same decision for all frames) ---
        if self.random_flip and random.random() < 0.5:
            composite = composite[:, :, ::-1, :].copy()
            alpha = alpha[:, :, ::-1, :].copy()
            foreground = foreground[:, :, ::-1, :].copy()
            background = background[:, :, ::-1, :].copy()

        # --- Random affine (same transform matrix for all frames) ---
        if self.random_affine and random.random() < 0.5:
            angle = random.uniform(-self.affine_degree, self.affine_degree)
            scale = random.uniform(*self.affine_scale)
            center = (crop_w / 2.0, crop_h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, scale)

            def warp_sequence(seq: np.ndarray, interp: int) -> np.ndarray:
                out = np.empty_like(seq)
                for t in range(seq.shape[0]):
                    out[t] = cv2.warpAffine(
                        seq[t], M, (crop_w, crop_h), flags=interp,
                        borderMode=cv2.BORDER_REFLECT,
                    )
                    # Restore channel dim if squeezed by warpAffine
                    if out[t].ndim == 2:
                        out[t] = out[t][..., np.newaxis]
                return out

            composite = warp_sequence(composite, cv2.INTER_LINEAR)
            foreground = warp_sequence(foreground, cv2.INTER_LINEAR)
            background = warp_sequence(background, cv2.INTER_LINEAR)
            alpha = warp_sequence(alpha, cv2.INTER_LINEAR)

        return composite, alpha, foreground, background
