"""VideoMatte240K-style dataset for neural video matting training."""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class VideoMatteDataset(Dataset):
    """Dataset that loads foreground videos with GT alpha mattes and composites
    them onto random background videos/images.

    Directory structure expected:
        root/
            fgr/         # foreground RGB frames (clip_name/frame_0000.png ...)
            pha/         # alpha matte frames   (clip_name/frame_0000.png ...)
            bgr_video/   # background video frames (clip_name/frame_0000.jpg ...)
            bgr_image/   # (optional) background images (img_0000.jpg ...)
    """

    def __init__(
        self,
        root: str,
        clip_length: int = 8,
        output_size: Tuple[int, int] = (512, 512),
        split: str = "train",
        augmentation=None,
    ):
        self.root = Path(root)
        self.clip_length = clip_length
        self.output_size = output_size  # (H, W)
        self.split = split
        self.augmentation = augmentation

        self.fgr_dir = self.root / "fgr"
        self.pha_dir = self.root / "pha"
        self.bgr_video_dir = self.root / "bgr_video"
        self.bgr_image_dir = self.root / "bgr_image"

        # Discover foreground clips
        self.fg_clips = sorted(
            [d.name for d in self.fgr_dir.iterdir() if d.is_dir()]
        )

        # Split: 90% train, 10% val
        split_idx = int(len(self.fg_clips) * 0.9)
        if split == "train":
            self.fg_clips = self.fg_clips[:split_idx]
        else:
            self.fg_clips = self.fg_clips[split_idx:]

        # Discover background clips / images
        self.bg_clips = []
        if self.bgr_video_dir.exists():
            self.bg_clips = sorted(
                [d.name for d in self.bgr_video_dir.iterdir() if d.is_dir()]
            )
        self.bg_images = []
        if self.bgr_image_dir.exists():
            self.bg_images = sorted(
                [f.name for f in self.bgr_image_dir.iterdir() if f.is_file()]
            )

    def __len__(self) -> int:
        return len(self.fg_clips)

    def _load_frames(self, clip_dir: Path, count: int, start: int) -> List[np.ndarray]:
        """Load `count` consecutive frames starting at `start` from a clip directory."""
        frame_files = sorted(clip_dir.iterdir())
        frames = []
        for i in range(start, start + count):
            idx = i % len(frame_files)
            img = cv2.imread(str(frame_files[idx]), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to read {frame_files[idx]}")
            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGRA2RGBA if img.shape[-1] == 4 else cv2.COLOR_BGR2RGB,
            )
            frames.append(img)
        return frames

    def _load_bg_sequence(self, length: int, h: int, w: int) -> np.ndarray:
        """Load a background sequence of shape (T, H, W, 3) as float32 [0,1]."""
        use_video = len(self.bg_clips) > 0 and (
            len(self.bg_images) == 0 or random.random() < 0.5
        )

        if use_video:
            clip_name = random.choice(self.bg_clips)
            clip_dir = self.bgr_video_dir / clip_name
            bg_frame_files = sorted(clip_dir.iterdir())
            start = random.randint(0, max(0, len(bg_frame_files) - length))
            bgs = []
            for i in range(length):
                idx = (start + i) % len(bg_frame_files)
                img = cv2.imread(str(bg_frame_files[idx]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (w, h))
                bgs.append(img.astype(np.float32) / 255.0)
            return np.stack(bgs, axis=0)
        else:
            img_name = random.choice(self.bg_images)
            img = cv2.imread(str(self.bgr_image_dir / img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (w, h))
            bg = img.astype(np.float32) / 255.0
            return np.stack([bg] * length, axis=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clip_name = self.fg_clips[idx]

        # Load foreground frames
        fgr_clip_dir = self.fgr_dir / clip_name
        pha_clip_dir = self.pha_dir / clip_name

        fgr_frame_files = sorted(fgr_clip_dir.iterdir())
        total_frames = len(fgr_frame_files)

        start = random.randint(0, max(0, total_frames - self.clip_length))

        fg_frames = []
        alpha_frames = []

        for i in range(self.clip_length):
            frame_idx = (start + i) % total_frames

            # Foreground RGB
            fgr_files = sorted(fgr_clip_dir.iterdir())
            fgr_img = cv2.imread(str(fgr_files[frame_idx]))
            fgr_img = cv2.cvtColor(fgr_img, cv2.COLOR_BGR2RGB)
            fgr_img = cv2.resize(fgr_img, (self.output_size[1], self.output_size[0]))
            fg_frames.append(fgr_img.astype(np.float32) / 255.0)

            # Alpha matte
            pha_files = sorted(pha_clip_dir.iterdir())
            pha_img = cv2.imread(str(pha_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
            pha_img = cv2.resize(pha_img, (self.output_size[1], self.output_size[0]))
            alpha_frames.append(pha_img.astype(np.float32) / 255.0)

        # Stack: (T, H, W, C) and (T, H, W)
        fg = np.stack(fg_frames, axis=0)       # (T, H, W, 3)
        alpha = np.stack(alpha_frames, axis=0)  # (T, H, W)
        alpha = alpha[..., np.newaxis]           # (T, H, W, 1)

        # Load background
        h, w = self.output_size
        bg = self._load_bg_sequence(self.clip_length, h, w)  # (T, H, W, 3)

        # Composite: fg * alpha + bg * (1 - alpha)
        composite = fg * alpha + bg * (1.0 - alpha)

        # Apply augmentation if provided
        if self.augmentation is not None:
            composite, alpha, fg, bg = self.augmentation(composite, alpha, fg, bg)

        # Convert to tensors: (T, H, W, C) -> (T, C, H, W)
        composite = torch.from_numpy(composite).permute(0, 3, 1, 2).float()
        alpha_gt = torch.from_numpy(alpha).permute(0, 3, 1, 2).float()
        foreground = torch.from_numpy(fg).permute(0, 3, 1, 2).float()
        background = torch.from_numpy(bg).permute(0, 3, 1, 2).float()

        return {
            "composite": composite,      # (T, 3, H, W)
            "alpha_gt": alpha_gt,         # (T, 1, H, W)
            "foreground": foreground,     # (T, 3, H, W)
            "background": background,     # (T, 3, H, W)
        }
