"""Inference engine for neural video matting."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.model.matting_network import MattingNetwork

logger = logging.getLogger(__name__)

# Supported video container formats (by file extension)
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

# Safety limit to prevent OOM on very long videos
MAX_FRAME_COUNT = 10_000
DEFAULT_MAX_FRAMES = 5_000


class VideoFormatError(ValueError):
    """Raised when an unsupported or invalid video format is encountered."""

    pass


class FrameCountExceededError(ValueError):
    """Raised when a video exceeds the maximum allowed frame count."""

    pass


def validate_video_path(video_path: str) -> Path:
    """Validate that a video path exists and has a supported format.

    Args:
        video_path: Path to the video file.

    Returns:
        Validated Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
        VideoFormatError: If the file extension is not supported.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not path.is_file():
        raise VideoFormatError(f"Path is not a file: {video_path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise VideoFormatError(
            f"Unsupported video format '{ext}'. "
            f"Supported formats: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
        )
    return path


def get_video_frame_count(video_path: str) -> int:
    """Get the total number of frames in a video without loading them all.

    Args:
        video_path: Path to the video file.

    Returns:
        Number of frames in the video.

    Raises:
        VideoFormatError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoFormatError(f"Cannot open video file: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


class MattingInference:
    """Load a checkpoint and run sequential video matting inference.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Torch device string.
        output_size: Optional (H, W) to resize output. None keeps original size.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        output_size: Optional[Tuple[int, int]] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_size = output_size

        self.model = MattingNetwork()
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {checkpoint_path} on {self.device}")

    @torch.no_grad()
    def process_video(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        max_frames: int = DEFAULT_MAX_FRAMES,
    ) -> List[np.ndarray]:
        """Process a sequence of video frames with guidance masks.

        Args:
            frames: List of (H, W, 3) uint8 RGB frames.
            masks: List of (H, W) float32 or uint8 guidance masks.
            max_frames: Maximum number of frames to process. Raises
                FrameCountExceededError if exceeded.

        Returns:
            List of (H, W) float32 alpha mattes in [0, 1].

        Raises:
            ValueError: If frames or masks are empty or have mismatched lengths.
            FrameCountExceededError: If the number of frames exceeds max_frames.
        """
        if len(frames) == 0:
            raise ValueError("No frames provided for processing.")
        if len(masks) == 0:
            raise ValueError("No masks provided for processing.")
        if len(frames) != len(masks):
            raise ValueError(
                f"Frame count ({len(frames)}) and mask count ({len(masks)}) must match."
            )
        if len(frames) > min(max_frames, MAX_FRAME_COUNT):
            raise FrameCountExceededError(
                f"Video has {len(frames)} frames, exceeding limit of "
                f"{min(max_frames, MAX_FRAME_COUNT)}. Consider splitting the video."
            )

        alphas = []
        h_prev = None

        for frame, mask in zip(frames, masks):
            alpha, h_prev = self._process_single(frame, mask, h_prev)
            alphas.append(alpha)

        return alphas

    def _process_single(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        h_prev: Optional[torch.Tensor],
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Process a single frame.

        Args:
            frame: (H, W, 3) uint8 RGB.
            mask: (H, W) guidance mask.
            h_prev: Previous recurrent hidden state or None.

        Returns:
            (alpha, h_new) where alpha is (H, W) float32 numpy.
        """
        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        if self.output_size is not None:
            proc_h, proc_w = self.output_size
            frame_resized = cv2.resize(frame, (proc_w, proc_h))
            mask_resized = cv2.resize(mask.astype(np.float32), (proc_w, proc_h))
        else:
            frame_resized = frame
            mask_resized = mask.astype(np.float32)

        # To tensor: (1, 3, H, W) and (1, 1, H, W)
        frame_t = (
            torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            / 255.0
        )
        mask_t = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).float().to(self.device)
        # Clamp mask to [0, 1]
        if mask_t.max() > 1.0:
            mask_t = mask_t / 255.0

        alpha_t, _, h_new = self.model.forward_single(frame_t, mask_t, h_prev)

        # To numpy
        alpha_np = alpha_t.squeeze().cpu().numpy()

        # Resize back to original if needed
        if self.output_size is not None:
            alpha_np = cv2.resize(alpha_np, (orig_w, orig_h))

        return alpha_np, h_new

    @torch.no_grad()
    def process_single_image(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Process a single image (no recurrent state).

        Args:
            frame: (H, W, 3) uint8 RGB.
            mask: (H, W) guidance mask.

        Returns:
            (H, W) float32 alpha matte in [0, 1].
        """
        alpha, _ = self._process_single(frame, mask, h_prev=None)
        return alpha


def load_video_frames(
    video_path: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
    validate: bool = True,
) -> List[np.ndarray]:
    """Load all frames from a video file as RGB uint8 numpy arrays.

    Args:
        video_path: Path to the video file.
        max_frames: Maximum frames to load. Raises FrameCountExceededError if exceeded.
        validate: Whether to validate the file extension before loading.

    Returns:
        List of (H, W, 3) uint8 RGB numpy arrays.

    Raises:
        FileNotFoundError: If the video file does not exist.
        VideoFormatError: If the file format is unsupported or cannot be opened.
        FrameCountExceededError: If the video exceeds max_frames.
    """
    if validate:
        validate_video_path(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoFormatError(f"Cannot open video file: {video_path}")

    # Check frame count before loading
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_limit = min(max_frames, MAX_FRAME_COUNT)
    if total > effective_limit:
        cap.release()
        raise FrameCountExceededError(
            f"Video has {total} frames, exceeding limit of {effective_limit}. "
            "Consider splitting the video into shorter segments."
        )

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Safety check during loading (CAP_PROP_FRAME_COUNT can be inaccurate)
        if len(frames) > effective_limit:
            cap.release()
            raise FrameCountExceededError(
                f"Video exceeded {effective_limit} frames during loading. "
                "Consider splitting the video into shorter segments."
            )
    cap.release()

    if len(frames) == 0:
        logger.warning(f"No frames could be read from {video_path}")

    return frames


def load_masks_from_dir(mask_dir: str) -> List[np.ndarray]:
    """Load mask images from a directory in sorted order."""
    mask_path = Path(mask_dir)
    mask_files = sorted(mask_path.glob("*.png")) + sorted(mask_path.glob("*.jpg"))
    masks = []
    for f in mask_files:
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        masks.append(m.astype(np.float32) / 255.0)
    return masks


def save_alpha_sequence(alphas: List[np.ndarray], output_dir: str):
    """Save alpha matte sequence as PNG files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, alpha in enumerate(alphas):
        alpha_uint8 = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(out / f"alpha_{i:06d}.png"), alpha_uint8)
    logger.info(f"Saved {len(alphas)} alpha mattes to {output_dir}")
