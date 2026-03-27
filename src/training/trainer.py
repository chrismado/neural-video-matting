"""Training loop for neural video matting with TBPTT and recurrent state."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.data.trimap_gen import generate_rough_mask_batch
from src.training.losses import combined_loss

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for neural video matting with sequential clip processing and TBPTT.

    Args:
        model: MattingNetwork model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration dict.
        device: Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-2),
        )

        # LR schedule: warmup + cosine decay
        total_epochs = config.get("epochs", 100)
        warmup_epochs = config.get("warmup_epochs", 5)

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=config.get("min_lr", 1e-6),
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        # Mixed precision
        self.use_amp = config.get("mixed_precision", True)
        self.scaler = GradScaler(enabled=self.use_amp)

        # Loss weights
        self.loss_weights = config.get(
            "loss_weights",
            {
                "alpha_l1": 1.0,
                "laplacian": 1.0,
                "gradient": 0.5,
                "temporal": 0.5,
                "foreground": 0.5,
            },
        )

        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # W&B (optional)
        self.use_wandb = config.get("use_wandb", False)
        self.wandb_run = None
        if self.use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=config.get("wandb_project", "neural-video-matting"),
                    config=config,
                )
            except ImportError:
                logger.warning("wandb not installed, disabling W&B logging.")
                self.use_wandb = False

        self.global_step = 0
        self.epoch = 0

    def train(self):
        """Run the full training loop."""
        total_epochs = self.config.get("epochs", 100)

        for epoch in range(self.epoch, total_epochs):
            self.epoch = epoch
            train_metrics = self._train_epoch()
            self.scheduler.step()

            logger.info(
                f"Epoch {epoch + 1}/{total_epochs} | "
                f"Train Loss: {train_metrics['total']:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # Validation
            val_int = self.config.get("val_interval", 5)
            if self.val_loader is not None and (epoch + 1) % val_int == 0:
                val_metrics = self._validate()
                logger.info(f"  Val Loss: {val_metrics['total']:.4f}")

                if self.use_wandb:
                    import wandb

                    metrics = {f"val/{k}": v for k, v in val_metrics.items()}
                    wandb.log(metrics, step=self.global_step)

            # Save checkpoint
            if (epoch + 1) % self.config.get("save_interval", 10) == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pth")

            if self.use_wandb:
                import wandb

                wandb.log(
                    {
                        "train/total_loss": train_metrics["total"],
                        "train/alpha_l1": train_metrics.get("alpha_l1", 0),
                        "train/laplacian": train_metrics.get("laplacian", 0),
                        "train/gradient": train_metrics.get("gradient", 0),
                        "train/temporal": train_metrics.get("temporal", 0),
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch + 1,
                    },
                    step=self.global_step,
                )

        # Final checkpoint
        self.save_checkpoint("final.pth")
        logger.info("Training complete.")

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch. Returns average loss metrics."""
        self.model.train()
        running = {}
        count = 0

        for batch in self.train_loader:
            composite = batch["composite"].to(self.device)  # (B, T, 3, H, W)
            alpha_gt = batch["alpha_gt"].to(self.device)  # (B, T, 1, H, W)
            foreground = batch["foreground"].to(self.device)  # (B, T, 3, H, W)

            B, T, C, H, W = composite.shape

            # Generate rough guidance masks from GT alpha
            with torch.no_grad():
                guidance_masks = generate_rough_mask_batch(alpha_gt)  # (B, T, 1, H, W)

            # Forward pass: full clip through model
            with autocast(enabled=self.use_amp):
                outputs = self.model(composite, guidance_masks)
                pred_alpha = outputs["alphas"]  # (B, T, 1, H, W)

                pred_fg = outputs.get("foregrounds", None)

                losses = combined_loss(
                    pred_alpha=pred_alpha,
                    gt_alpha=alpha_gt,
                    frames=composite,
                    pred_fg=pred_fg,
                    gt_fg=foreground if pred_fg is not None else None,
                    weights=self.loss_weights,
                )

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate metrics
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()
            count += 1
            self.global_step += 1

        return {k: v / max(count, 1) for k, v in running.items()}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation. Returns average loss metrics."""
        self.model.eval()
        running = {}
        count = 0

        for batch in self.val_loader:
            composite = batch["composite"].to(self.device)
            alpha_gt = batch["alpha_gt"].to(self.device)
            foreground = batch["foreground"].to(self.device)

            guidance_masks = generate_rough_mask_batch(alpha_gt)

            outputs = self.model(composite, guidance_masks)
            pred_alpha = outputs["alphas"]
            pred_fg = outputs.get("foregrounds", None)

            losses = combined_loss(
                pred_alpha=pred_alpha,
                gt_alpha=alpha_gt,
                frames=composite,
                pred_fg=pred_fg,
                gt_fg=foreground if pred_fg is not None else None,
                weights=self.loss_weights,
            )

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()
            count += 1

        return {k: v / max(count, 1) for k, v in running.items()}

    def save_checkpoint(self, filename: str):
        """Save model and training state to a checkpoint file."""
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model and training state from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        logger.info(f"Checkpoint loaded from {path} (epoch {self.epoch})")
