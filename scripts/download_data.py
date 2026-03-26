#!/usr/bin/env python3
"""Download VideoMatte240K dataset.

PLACEHOLDER SCRIPT - Manual steps required:

1. VideoMatte240K:
   - Visit: https://grail.cs.washington.edu/projects/background-matting-v2/
   - Download VideoMatte240K (foreground + alpha clips)
   - Extract into: data/VideoMatte240K/
       data/VideoMatte240K/fgr/    (foreground RGB frames per clip)
       data/VideoMatte240K/pha/    (alpha matte frames per clip)

2. Background videos (e.g., DVM dataset or custom):
   - Download background video clips
   - Extract into: data/VideoMatte240K/bgr_video/

3. Background images (optional, e.g., MS-COCO, BG-20K):
   - Download background images
   - Place into: data/VideoMatte240K/bgr_image/

4. Verify structure:
   data/VideoMatte240K/
       fgr/
           clip_0001/
               frame_0000.png
               frame_0001.png
               ...
           clip_0002/
               ...
       pha/
           clip_0001/
               frame_0000.png
               ...
       bgr_video/
           bg_clip_0001/
               frame_0000.jpg
               ...
       bgr_image/
           img_0001.jpg
           img_0002.jpg
           ...
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download VideoMatte240K dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/VideoMatte240K",
        help="Output directory for the dataset.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("VideoMatte240K Dataset Setup")
    print("=" * 60)
    print()
    print("This dataset must be downloaded manually due to license terms.")
    print()
    print("Steps:")
    print("1. Visit https://grail.cs.washington.edu/projects/background-matting-v2/")
    print("2. Download VideoMatte240K foreground + alpha clips.")
    print(f"3. Extract to: {output_dir.resolve()}")
    print()

    # Create directory structure
    for subdir in ["fgr", "pha", "bgr_video", "bgr_image"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {output_dir / subdir}")

    print()
    print("Directory structure created. Please add data files as described above.")
    print()

    # Verify if data already exists
    fgr_clips = list((output_dir / "fgr").iterdir()) if (output_dir / "fgr").exists() else []
    pha_clips = list((output_dir / "pha").iterdir()) if (output_dir / "pha").exists() else []
    print(f"Found {len(fgr_clips)} foreground clips, {len(pha_clips)} alpha clips.")

    if len(fgr_clips) == 0:
        print("WARNING: No foreground data found. Please download the dataset.")
        sys.exit(1)
    else:
        print("Dataset appears to be ready.")


if __name__ == "__main__":
    main()
