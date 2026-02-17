"""
Feature extraction utilities.

Before we dive into OOD scoring, we need a way to extract **penultimate layer features.
Many OOD methods (Mahalanobis, ViM, NECO) work in this feature space rather thanon raw logits.
We use a **forward hook** — a PyTorch mechanism that lets us capture intermediate
activations without modifying the model's forward() method. 
"""

import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_features_and_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract penultimate layer features AND logits for every sample in the loader.

    How it works
    1. We register a hook on the avgpool layer (right before the FC layer).
    2. During forward pass, the hook captures the output of avgpool.
    3. We also get the final logits from the model's output.
    """
    all_features = []
    all_logits = []
    all_labels = []

    # Hook to capture features from the average pooling layer
    features_buffer: dict = {}

    def hook_fn(module, input, output):
        # output shape: (batch_size, 512, 1, 1) after avgpool
        # We flatten it to (batch_size, 512)
        features_buffer['feat'] = output.flatten(1)

    # Register hook on the avgpool layer
    handle = model.avgpool.register_forward_hook(hook_fn)

    model.eval()
    for images, labels in tqdm(loader, desc='Extracting features', leave=False):
        images = images.to(device)
        logits = model(images)  # this triggers the hook

        all_features.append(features_buffer['feat'].cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

    # Remove the hook (clean up)
    handle.remove()

    features = np.concatenate(all_features, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, logits, labels


@torch.no_grad()
def extract_multi_layer_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Extract features from multiple layers of ResNet-18.

    We hook into the output of each of the 4 residual blocks (layer1-layer4)
    plus the final avgpool layer.  This is used for the bonus analysis of Neural Collapse across layers.
    """
    # Layers to extract features from
    layers = {
        'layer1': model.layer1,  # 64-dim, 16×16 spatial
        'layer2': model.layer2,  # 128-dim,  8×8  spatial
        'layer3': model.layer3,  # 256-dim,  4×4  spatial
        'layer4': model.layer4,  # 512-dim,  2×2  spatial
    }

    buffers: Dict[str, list] = {name: [] for name in layers}
    all_labels = []
    handles = []

    # Register hooks for each layer
    def make_hook(name):
        def hook_fn(module, input, output):
            # Global average pool the feature maps to get a vector
            pooled = F.adaptive_avg_pool2d(output, 1).flatten(1)
            buffers[name].append(pooled.detach().cpu().numpy())
        return hook_fn

    for name, layer in layers.items():
        handles.append(layer.register_forward_hook(make_hook(name)))

    model.eval()
    for images, labels in tqdm(loader, desc='Multi-layer extraction', leave=False):
        images = images.to(device)
        model(images)
        all_labels.append(labels.numpy())

    # Cleanup
    for h in handles:
        h.remove()

    # Concatenate
    layer_features = {name: np.concatenate(buffers[name], axis=0) for name in layers}
    labels_arr = np.concatenate(all_labels, axis=0)

    return layer_features, labels_arr
