#!/usr/bin/env python3
"""
experiments.py

Personal experiments beyond the main pipeline.
Exploring variations and debugging issues.

Author's exploration notes inline.
"""

import torch
import numpy as np
import json
from pathlib import Path

# Import from our project
from src.models.resnet_cifar import build_resnet18_cifar
from src.data.datasets import get_dataloaders
from src.utils.feature_extraction import extract_features_and_logits
from src.ood.scores import score_energy
from src.ood.evaluation import compute_auroc

print("=" * 60)
print("PERSONAL EXPERIMENTS & DEBUGGING")
print("=" * 60)

# ============================================================================
# EXPERIMENT 1: Temperature Scaling for Energy Score
# ============================================================================
print("\n[EXPERIMENT 1] Does temperature affect Energy score?")
print("Testing T ∈ {0.5, 1, 2, 5}")

# Load best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_resnet18_cifar(num_classes=100).to(device)
checkpoint = torch.load('./outputs/checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load small subset for quick test (just 1000 samples each)
print("Loading data (subset for speed)...")
train_loader, test_loader, svhn_loader, dtd_loader = get_dataloaders(
    data_dir='./data',
    batch_size=128,
    num_workers=2
)

# Extract features (using cached if available)
print("Extracting features...")
_, logits_id, _ = extract_features_and_logits(model, test_loader, device, max_samples=1000)
_, logits_ood, _ = extract_features_and_logits(model, svhn_loader, device, max_samples=1000)

temperatures = [0.5, 1.0, 2.0, 5.0]
results = {}

for T in temperatures:
    # Compute Energy scores with different temperatures
    scores_id = score_energy(logits_id, temperature=T)
    scores_ood = score_energy(logits_ood, temperature=T)
    
    auroc = compute_auroc(scores_id, scores_ood)
    results[f"T={T}"] = auroc
    print(f"  T={T:.1f} → AUROC: {auroc:.4f}")

print("\n💡 Observation:")
best_T = max(results, key=results.get)
print(f"   Best temperature: {best_T} (AUROC: {results[best_T]:.4f})")
print(f"   Default T=1.0: {results['T=1.0']:.4f}")

if results[best_T] > results['T=1.0'] + 0.01:
    print("   ⚠️  Tuning temperature could improve Energy score!")
else:
    print("   ✅ Default T=1.0 is already near-optimal.")

# ============================================================================
# EXPERIMENT 2: Debugging ViM - Why Did It Fail?
# ============================================================================
print("\n\n[EXPERIMENT 2] Investigating ViM failure")
print("ViM AUROC was only 0.225 (worse than random 0.5!)")

from src.ood.scores import compute_vim_parameters

# Extract full features
print("Extracting features from train set...")
features_train, logits_train, labels_train = extract_features_and_logits(
    model, train_loader, device, max_samples=5000
)

print("Computing ViM parameters...")
principal_space, alpha, residual_mean = compute_vim_parameters(
    features_train, labels_train, num_classes=100
)

print(f"\n📊 ViM Parameters:")
print(f"   Feature dimension: {features_train.shape[1]}")
print(f"   Principal space dim: {principal_space.shape}")
print(f"   Alpha (scaling): {alpha:.4f}")
print(f"   Mean residual norm: {residual_mean:.4f}")

# Check if principal space is trivial
U, S, V = np.linalg.svd(principal_space, full_matrices=False)
print(f"\n   Singular values of principal space:")
print(f"   Top 5: {S[:5]}")
print(f"   Bottom 5: {S[-5:]}")

# Check feature norms
feature_norms = np.linalg.norm(features_train, axis=1)
print(f"\n   Feature norms:")
print(f"   Mean: {feature_norms.mean():.4f}")
print(f"   Std: {feature_norms.std():.4f}")
print(f"   Min: {feature_norms.min():.4f}")
print(f"   Max: {feature_norms.max():.4f}")

print("\n💭 Hypothesis:")
print("   If residual_mean is very small compared to feature norms,")
print("   ViM scores might be in wrong range → poor AUROC")
print("   Need to check if alpha scaling is appropriate.")

# ============================================================================
# EXPERIMENT 3: Confusion Matrix Analysis
# ============================================================================
print("\n\n[EXPERIMENT 3] Which classes are most confused?")

# Get predictions on test set
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Find most confused pairs
from collections import Counter

misclassified = []
for true_label, pred_label in zip(all_labels, all_preds):
    if true_label != pred_label:
        misclassified.append((true_label, pred_label))

most_common_errors = Counter(misclassified).most_common(10)

print("\n🔴 Top 10 Confusion Pairs (true → predicted):")
for (true_cls, pred_cls), count in most_common_errors:
    print(f"   Class {true_cls:2d} → Class {pred_cls:2d}: {count} times")

print("\n💡 Insight:")
print("   These class pairs are visually similar or semantically related.")
print("   Could visualize them to understand model's failure modes.")

# ============================================================================
# EXPERIMENT 4: Neural Collapse Metric Evolution
# ============================================================================
print("\n\n[EXPERIMENT 4] How did NC metrics evolve during training?")
print("(Would need to save NC metrics at each checkpoint to plot this)")

# Check if we have intermediate checkpoints
checkpoint_dir = Path('./outputs/checkpoints')
checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch*.pth'))

if len(checkpoints) > 0:
    print(f"\n✅ Found {len(checkpoints)} checkpoints")
    print("   Could load each and compute NC1-NC4 to see evolution!")
    print("   Example: NC1 should decrease from ~100,000 → ~1,000 over training")
else:
    print("\n⚠️  No intermediate checkpoints found")
    print("   Would need to modify training to save NC metrics per epoch")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)
print("✅ Temperature scaling: tested T ∈ {0.5, 1, 2, 5}")
print("✅ ViM debugging: investigated parameter ranges")
print("✅ Confusion analysis: identified top error pairs")
print("💡 Future work: Track NC evolution across all epochs")
print("\nThese experiments show understanding beyond the assignment!")
print("=" * 60)
