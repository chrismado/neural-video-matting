#!/usr/bin/env python3
"""CLI script to composite foreground video onto a new background."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.serving.composite import composite_video
from src.serving.inference import load_video_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Composite foreground onto new background")
    parser.add_argument("--fg-video", type=str, required=True, help="Foreground video path.")
    parser.add_argument("--alpha-dir", type=str, required=True, help="Directory of alpha PNGs.")
    parser.add_argument("--bg", type=str, required=True, help="Background video or image path.")
    parser.add_argument("--output", type=str, default="output/composited", help="Output directory.")
    parser.add_argument("--no-spill-suppression", action="store_true",
                        help="Disable spill suppression.")
    parser.add_argument("--spill-strength", type=float, default=0.5,
                        help="Spill suppression strength.")
    args = parser.parse_args()

    # Load foreground
    logger.info(f"Loading foreground: {args.fg_video}")
    fg_frames = load_video_frames(args.fg_video)
    logger.info(f"Loaded {len(fg_frames)} foreground frames")

    # Load alphas
    alpha_dir = Path(args.alpha_dir)
    alpha_files = sorted(alpha_dir.glob("*.png"))
    alphas = []
    for af in alpha_files:
        a = cv2.imread(str(af), cv2.IMREAD_GRAYSCALE)
        alphas.append(a.astype(np.float32) / 255.0)
    logger.info(f"Loaded {len(alphas)} alpha mattes")

    # Load background
    bg_img = cv2.imread(args.bg)
    if bg_img is not None:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
        bg_frames = [bg_img]
    else:
        bg_frames = load_video_frames(args.bg)
    logger.info(f"Loaded {len(bg_frames)} background frames")

    # Composite
    logger.info("Compositing...")
    results = composite_video(
        fg_frames=fg_frames,
        alphas=alphas,
        bg_frames=bg_frames,
        spill_suppression=not args.no_spill_suppression,
        spill_strength=args.spill_strength,
    )

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(results):
        if frame.dtype == np.float32:
            frame = (frame * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"composite_{i:06d}.png"), frame_bgr)
    logger.info(f"Saved {len(results)} frames to {args.output}")


if __name__ == "__main__":
    main()
