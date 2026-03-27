#!/usr/bin/env python3
"""Training entry point for neural video matting."""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentation import VideoMattingAugmentation  # noqa: I001, E402
from src.data.dataset import VideoMatteDataset  # noqa: E402
from src.model.matting_network import MattingNetwork  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train neural video matting model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config YAML.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data
    data_cfg = config.get("data", {})
    augmentation = VideoMattingAugmentation(
        crop_size=tuple(data_cfg.get("crop_size", [512, 512])),
        random_flip=data_cfg.get("random_flip", True),
        random_affine=data_cfg.get("random_affine", True),
    )

    train_dataset = VideoMatteDataset(
        root=data_cfg.get("root", "data/VideoMatte240K"),
        clip_length=data_cfg.get("clip_length", 8),
        output_size=tuple(data_cfg.get("output_size", [512, 512])),
        split="train",
        augmentation=augmentation,
    )
    val_dataset = VideoMatteDataset(
        root=data_cfg.get("root", "data/VideoMatte240K"),
        clip_length=data_cfg.get("clip_length", 8),
        output_size=tuple(data_cfg.get("output_size", [512, 512])),
        split="val",
        augmentation=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 2),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 2),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = MattingNetwork()

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
