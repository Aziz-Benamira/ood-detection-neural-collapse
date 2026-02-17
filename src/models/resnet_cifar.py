"""
ResNet-18 adapted for CIFAR-100.

Standard ResNet-18 is for ImageNet (224x224). For CIFAR (32x32), vanilla
torchvision ResNet-18 gave only ~45% accuracy. The 7x7 stride-2 conv + maxpool
downsamples 32x32 too aggressively.

Solution: 3x3 stride-1 conv, no maxpool (He et al., 2016).

"""

import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18_cifar(num_classes: int = 100) -> nn.Module:
    """ResNet-18 for CIFAR: 3x3 conv stride 1, no maxpool, train from scratch."""
    model = resnet18(weights=None)

    # 3x3 conv stride 1 instead of 7x7 stride 2
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # remove maxpool
    model.maxpool = nn.Identity()
    
    # adjust output for CIFAR-100
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
