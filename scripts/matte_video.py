#!/usr/bin/env python3
"""CLI script to run video matting inference on a single video."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.serving.inference import (
    MattingInference,
    load_masks_from_dir,
    load_video_frames,
    save_alpha_sequence,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run video matting on a single video")
    parser.add_argument("--video", type=str, required=True, help="Input video path.")
    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Guidance mask: path to a single image or directory of mask frames.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint.")
    parser.add_argument("--output", type=str, default="output/alphas", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu).")
    args = parser.parse_args()

    engine = MattingInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Load video
    logger.info(f"Loading video: {args.video}")
    frames = load_video_frames(args.video)
    logger.info(f"Loaded {len(frames)} frames")

    # Load masks
    mask_path = Path(args.mask)
    if mask_path.is_dir():
        masks = load_masks_from_dir(str(mask_path))
        while len(masks) < len(frames):
            masks.append(masks[-1])
    else:
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        masks = [mask_img.astype(np.float32) / 255.0] * len(frames)

    # Run
    logger.info("Running matting inference...")
    alphas = engine.process_video(frames, masks)

    # Save
    save_alpha_sequence(alphas, args.output)
    logger.info(f"Done. Output saved to {args.output}")


if __name__ == "__main__":
    main()
