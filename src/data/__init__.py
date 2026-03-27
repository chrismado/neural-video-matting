from .augmentation import VideoMattingAugmentation
from .dataset import VideoMatteDataset
from .trimap_gen import (
    generate_rough_mask,
    generate_rough_mask_batch,
    generate_trimap,
    generate_trimap_batch,
)

__all__ = [
    "VideoMatteDataset",
    "VideoMattingAugmentation",
    "generate_trimap",
    "generate_trimap_batch",
    "generate_rough_mask",
    "generate_rough_mask_batch",
]
