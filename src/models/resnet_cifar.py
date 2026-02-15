"""
ResNet-18 adapted for CIFAR-100 (32×32 images).

The standard ResNet-18 from torchvision is designed for ImageNet (224×224 images).
For CIFAR (32×32), we need to modify the first convolutional layer:
- Original: 7×7 conv, stride 2, followed by maxpool → too aggressive downsampling for 32×32
- Modified: 3×3 conv, stride 1, no maxpool → preserves spatial information

This is a well-known trick from the CIFAR-ResNet literature (He et al., 2016).

Author: Aziz BEN AMIRA
Course: Theory of Deep Learning (MVA + ENSTA)
"""

import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18_cifar(num_classes: int = 100) -> nn.Module:
    """
    Build a ResNet-18 adapted for CIFAR-100 (32×32 images).

    The key modification is replacing the first conv layer + maxpool:
    - ImageNet version: Conv(7×7, stride=2) + MaxPool(3×3, stride=2)
      This downsamples 224×224 -> 56×56 very quickly.
    - CIFAR version: Conv(3×3, stride=1), no maxpool
      We keep the 32×32 resolution as long as possible because
      the images are already small.

    Parameters
    ----------
    num_classes : int
        Number of output classes (100 for CIFAR-100)

    Returns
    -------
    model : nn.Module
        Modified ResNet-18 ready for CIFAR-100
    """
    model = resnet18(weights=None)  # no pretrained weights — we train from scratch

    # Replace first conv: 7×7 stride 2 -> 3×3 stride 1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove maxpool (it would shrink our 32×32 images too much)
    model.maxpool = nn.Identity()

    # Replace the final FC layer to output num_classes instead of 1000
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
