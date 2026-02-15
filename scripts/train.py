#!/usr/bin/env python3
"""
Train ResNet-18 on CIFAR-100.

Usage
-----
    python scripts/train.py --epochs 200 --data-dir ./data --output-dir ./outputs

Resume from checkpoint:
    python scripts/train.py --resume outputs/checkpoints/checkpoint_epoch100.pth

Author: Aziz BEN AMIRA
Course: Theory of Deep Learning (MVA + ENSTA)
"""

import argparse
import json
import logging
import os
import sys

import torch
import numpy as np

# Allow running from the project root: `python scripts/train.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SEED, NUM_CLASSES, BATCH_SIZE, NUM_WORKERS,
    CIFAR100_MEAN, CIFAR100_STD,
    NUM_EPOCHS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, LR_MIN,
    CHECKPOINT_EVERY, PRINT_EVERY,
    DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, PLOT_DIR,
    ensure_dirs,
)
from src.models import build_resnet18_cifar
from src.data import get_dataloaders
from src.training import Trainer
from src.utils.visualization import plot_training_curves


def parse_args():
    p = argparse.ArgumentParser(
        description="Train ResNet-18 on CIFAR-100",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    p.add_argument("--data-dir", type=str, default=DATA_DIR,
                   help="Root directory for datasets")
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                   help="Root directory for all outputs")
    # training
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--momentum", type=float, default=MOMENTUM)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--lr-min", type=float, default=LR_MIN)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    # checkpointing
    p.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    p.add_argument("--print-every", type=int, default=PRINT_EVERY)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume training from")
    # reproducibility
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def setup_logging(log_dir: str):
    """Configure logging to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)


def main():
    args = parse_args()

    # Override output paths based on --output-dir
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs")
    plot_dir = os.path.join(args.output_dir, "plots")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    setup_logging(log_dir)
    logger = logging.getLogger(__name__)

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)
    if device.type == 'cuda':
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("Memory: %.1f GB",
                     torch.cuda.get_device_properties(0).total_mem / 1e9)

    # ---- Data ----
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cifar_mean=CIFAR100_MEAN,
        cifar_std=CIFAR100_STD,
    )

    # ---- Model ----
    model = build_resnet18_cifar(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("ResNet-18 for CIFAR-100: %s parameters", f"{total_params:,}")
    logger.info("Feature dimension (before FC): %d", model.fc.in_features)

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_min=args.lr_min,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        print_every=args.print_every,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ---- Train ----
    history = trainer.train(loaders['train'], loaders['test'])

    # ---- Plot training curves ----
    plot_training_curves(history, plot_dir)

    logger.info("All done! Best test accuracy: %.2f%%", trainer.best_acc * 100)
    logger.info("Outputs saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
