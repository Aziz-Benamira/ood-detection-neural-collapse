"""
Training loop with logging, checkpointing, and resume support.

Training Details
----------------
- Optimizer: SGD with momentum 0.9 and weight decay 5×10⁻⁴
  SGD + momentum is the standard for image classification. Adam can work but often
  leads to slightly worse generalization on CIFAR.
- Learning rate schedule: Cosine annealing from 0.1 to ~0
  Cosine schedule is smooth and avoids the need to manually tune step decay milestones.
- Epochs: 200 (standard for CIFAR-100; Neural Collapse typically emerges in the later
  epochs of training)

Author: Aziz BEN AMIRA
Course: Theory of Deep Learning (MVA + ENSTA)
"""

import copy
import json
import logging
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """
    Encapsulates the training loop, evaluation, checkpointing, and history
    logging for a PyTorch model on CIFAR-100.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    device : torch.device
        Device to train on (CPU or CUDA).
    num_epochs : int
        Total number of training epochs.
    lr : float
        Initial learning rate.
    momentum : float
        SGD momentum.
    weight_decay : float
        L2 regularization strength.
    lr_min : float
        Minimum learning rate for cosine annealing.
    checkpoint_dir : str
        Directory to save checkpoints.
    checkpoint_every : int
        Save a checkpoint every N epochs.
    print_every : int
        Print progress every N epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_epochs: int = 200,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        lr_min: float = 1e-6,
        checkpoint_dir: str = "./outputs/checkpoints",
        checkpoint_every: int = 10,
        print_every: int = 10,
    ):
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # Cosine annealing: LR starts at `lr`, smoothly decays to `lr_min`.
        # Unlike step decay, there's no sudden "jump" — the loss curve is smoother.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=lr_min,
        )

        self.history: Dict[str, list] = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [],  'test_acc': [],
            'lr': [],
        }
        self.best_acc = 0.0
        self.best_model_state = None
        self.start_epoch = 1  # 1-indexed

    # ------------------------------------------------------------------
    # Single-epoch routines
    # ------------------------------------------------------------------

    def _train_one_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Returns
        -------
        avg_loss : float
            Average cross-entropy loss over the epoch.
        accuracy : float
            Training accuracy (fraction of correct predictions).
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return running_loss / total, correct / total

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on a given dataset.

        We disable gradient computation to save memory and speed up inference.

        Returns
        -------
        avg_loss : float
            Average loss.
        accuracy : float
            Classification accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return running_loss / total, correct / total

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, path: Optional[str] = None):
        """Save a full training checkpoint (model + optimizer + scheduler + history)."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch:03d}.pth")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'best_model_state': self.best_model_state,
            'history': self.history,
        }
        torch.save(state, path)
        logger.info("Checkpoint saved → %s", path)

    def load_checkpoint(self, path: str):
        """Resume training from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_acc = checkpoint['best_acc']
        self.best_model_state = checkpoint['best_model_state']
        self.history = checkpoint['history']
        self.start_epoch = checkpoint['epoch'] + 1
        logger.info("Resumed from checkpoint (epoch %d, best_acc=%.2f%%)",
                     checkpoint['epoch'], self.best_acc * 100)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        """
        Full training loop with logging, early best-model tracking, and
        periodic checkpointing.

        Parameters
        ----------
        train_loader : DataLoader
            Training data.
        test_loader : DataLoader
            Validation / test data.
        """
        logger.info("Starting training from epoch %d to %d ...",
                     self.start_epoch, self.num_epochs)
        logger.info("=" * 70)

        epoch_iter = tqdm(
            range(self.start_epoch, self.num_epochs + 1),
            desc="Training",
            unit="epoch",
            initial=self.start_epoch - 1,
            total=self.num_epochs,
        )

        for epoch in epoch_iter:
            t0 = time.time()

            train_loss, train_acc = self._train_one_epoch(train_loader)
            test_loss, test_acc = self._evaluate(test_loader)

            # Step the scheduler AFTER the optimizer step (PyTorch convention)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            # Log metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(current_lr)

            # Save best model
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())

            # tqdm postfix
            epoch_iter.set_postfix(
                train_loss=f"{train_loss:.4f}",
                test_acc=f"{test_acc * 100:.1f}%",
                best=f"{self.best_acc * 100:.1f}%",
                lr=f"{current_lr:.1e}",
            )

            # Detailed log every N epochs
            if epoch % self.print_every == 0 or epoch == 1:
                elapsed = time.time() - t0
                logger.info(
                    "Epoch [%3d/%d]  "
                    "Train Loss: %.4f  Train Acc: %.1f%%  |  "
                    "Test Loss: %.4f  Test Acc: %.1f%%  |  "
                    "LR: %.6f  (%.1fs)",
                    epoch, self.num_epochs,
                    train_loss, train_acc * 100,
                    test_loss, test_acc * 100,
                    current_lr, elapsed,
                )

            # Periodic checkpoint
            if epoch % self.checkpoint_every == 0:
                self.save_checkpoint(epoch)

        logger.info("=" * 70)
        logger.info("Training finished! Best test accuracy: %.2f%%", self.best_acc * 100)

        # Save final best model as a standalone weights file
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        torch.save(self.best_model_state, best_path)
        logger.info("Best model weights saved → %s", best_path)

        # Save history as JSON for later plotting
        history_path = os.path.join(
            os.path.dirname(self.checkpoint_dir), "results", "training_history.json"
        )
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info("Training history saved → %s", history_path)

        return self.history
