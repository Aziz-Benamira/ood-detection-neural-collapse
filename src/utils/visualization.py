"""
Visualization utilities — all plots are saved to disk as PNG files
(``plt.show()`` won't work on a headless SSH cluster).

Every function takes an ``output_dir`` argument and writes the figure
to ``<output_dir>/<filename>.png``.

Author: Aziz BEN AMIRA
Course: Theory of Deep Learning (MVA + ENSTA)
"""

import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for SSH / headless servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa: F401 — imported for its side-effects on styling

logger = logging.getLogger(__name__)


def _save(fig, output_dir: str, filename: str, dpi: int = 150):
    """Save a matplotlib figure and close it."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info("Plot saved → %s", path)


# ======================================================================
# Training curves
# ======================================================================

def plot_training_curves(history: dict, output_dir: str):
    """
    Plot training & test loss/accuracy curves and the LR schedule.

    These plots are a DELIVERABLE: the professor wants to see them.
    We plot loss and accuracy side by side, and also the LR schedule.

    Parameters
    ----------
    history : dict
        Must contain keys 'train_loss', 'test_loss', 'train_acc', 'test_acc', 'lr'.
    output_dir : str
        Directory to save the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
    axes[0].plot(history['test_loss'], label='Test Loss', alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Training & Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot([a * 100 for a in history['train_acc']], label='Train Acc', alpha=0.8)
    axes[1].plot([a * 100 for a in history['test_acc']], label='Test Acc', alpha=0.8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[2].plot(history['lr'], color='green', alpha=0.8)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Cosine Annealing LR Schedule')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, 'training_curves.png')

    logger.info("Final train acc: %.2f%%", history['train_acc'][-1] * 100)
    logger.info("Final test acc:  %.2f%%", history['test_acc'][-1] * 100)


# ======================================================================
# OOD score distributions
# ======================================================================

def plot_ood_score_distributions(
    ood_results: Dict[str, dict],
    output_dir: str,
    ood_name: str = "SVHN",
):
    """
    Plot histograms of OOD scores for each method (ID vs OOD).

    This is a KEY DELIVERABLE: showing how well each method separates
    ID from OOD.  Ideally, the ID and OOD histograms should have minimal
    overlap.

    Parameters
    ----------
    ood_results : dict
        ``{method_name: {'id': np.ndarray, 'ood': np.ndarray, 'auroc': float}}``.
    output_dir : str
        Directory to save the plot.
    ood_name : str
        Name of the OOD dataset (used in titles).
    """
    methods = list(ood_results.keys())
    n_methods = len(methods)
    ncols = min(3, n_methods + 1)
    nrows = (n_methods + ncols) // ncols  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.asarray(axes).flatten()

    for idx, method_name in enumerate(methods):
        ax = axes[idx]
        result = ood_results[method_name]

        ax.hist(result['id'], bins=80, alpha=0.6, density=True,
                label='ID (CIFAR-100)', color='blue')
        ax.hist(result['ood'], bins=80, alpha=0.6, density=True,
                label=f'OOD ({ood_name})', color='red')

        ax.set_title(f"{method_name}\nAUROC = {result['auroc']:.4f}", fontsize=12)
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Fill remaining axes with a summary table or hide them
    for idx in range(n_methods, len(axes)):
        ax = axes[idx]
        if idx == n_methods:
            summary_text = "AUROC Summary\n" + "=" * 25 + "\n"
            for m in methods:
                summary_text += f"{m:15s} {ood_results[m]['auroc']:.4f}\n"
            ax.text(0.1, 0.5, summary_text, fontsize=14, fontfamily='monospace',
                    verticalalignment='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.axis('off')

    plt.suptitle(f'OOD Score Distributions: CIFAR-100 (ID) vs {ood_name} (OOD)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, output_dir, f'ood_scores_{ood_name.lower()}.png')


# ======================================================================
# Final OOD bar-chart comparison
# ======================================================================

def plot_final_ood_comparison(
    ood_results: Dict[str, dict],
    output_dir: str,
    ood_name: str = "SVHN",
):
    """
    Bar chart comparison of AUROC across all OOD methods.

    Parameters
    ----------
    ood_results : dict
        ``{method_name: {'auroc': float, ...}}``.
    output_dir : str
    ood_name : str
    """
    methods = list(ood_results.keys())
    auroc_values = [ood_results[m]['auroc'] for m in methods]

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#00BCD4']
    # Extend colors if more methods than the palette
    while len(colors) < len(methods):
        colors.append('#607D8B')

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, auroc_values, color=colors[:len(methods)],
                  alpha=0.85, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, auroc_values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title(f'OOD Detection: CIFAR-100 (ID) vs {ood_name} (OOD)\n'
                 f'All Methods Comparison', fontsize=13)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random baseline')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _save(fig, output_dir, f'final_ood_comparison_{ood_name.lower()}.png')


# ======================================================================
# Neural Collapse (NC1-NC4) visualizations
# ======================================================================

def plot_neural_collapse(
    per_class_var: np.ndarray,
    cos_sim_matrix: np.ndarray,
    nc3_cos_sims: np.ndarray,
    nc3_mean_cos: float,
    nc4_model_acc: float,
    nc4_ncc_acc: float,
    nc4_agreement: float,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int,
    output_dir: str,
):
    """
    Create the 6-panel Neural Collapse figure (NC1-NC4).

    These are DELIVERABLES: the professor wants to see these plots.

    Parameters
    ----------
    per_class_var : np.ndarray, shape (C,)
    cos_sim_matrix : np.ndarray, shape (C, C)
    nc3_cos_sims : np.ndarray, shape (C,)
    nc3_mean_cos : float
    nc4_model_acc, nc4_ncc_acc, nc4_agreement : float
    train_features : np.ndarray, shape (N, D)
    train_labels : np.ndarray, shape (N,)
    num_classes : int
    output_dir : str
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # --- NC1: Within-class variance per class ---
    ax = axes[0, 0]
    ax.bar(range(num_classes), per_class_var, alpha=0.7, width=1.0)
    ax.set_xlabel('Class index')
    ax.set_ylabel('Within-class variance (trace)')
    ax.set_title('NC1: Within-class Variance per Class\n(should be uniformly small)', fontsize=11)
    ax.axhline(y=per_class_var.mean(), color='red', linestyle='--',
               label=f'Mean: {per_class_var.mean():.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- NC2: Cosine similarity matrix ---
    ax = axes[0, 1]
    subset = cos_sim_matrix[:20, :20]
    im = ax.imshow(subset, cmap='RdBu_r', vmin=-0.15, vmax=0.15)
    ax.set_title('NC2: Pairwise Cosine Similarity\n'
                 '(first 20 classes, off-diag → -1/(C-1))', fontsize=11)
    ax.set_xlabel('Class')
    ax.set_ylabel('Class')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # --- NC2: Distribution of off-diagonal cosine similarities ---
    ax = axes[0, 2]
    offdiag_mask = ~np.eye(num_classes, dtype=bool)
    offdiag_cos = cos_sim_matrix[offdiag_mask]
    target = -1.0 / (num_classes - 1)
    ax.hist(offdiag_cos, bins=80, alpha=0.7, density=True, color='steelblue')
    ax.axvline(x=target, color='red', linestyle='--', linewidth=2,
               label=f'Target: -1/(C-1) = {target:.4f}')
    ax.axvline(x=offdiag_cos.mean(), color='green', linestyle='--', linewidth=2,
               label=f'Actual mean: {offdiag_cos.mean():.4f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('NC2: Off-diagonal Cosine Similarities\n'
                 '(should concentrate at -1/(C-1))', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- NC3: Cosine similarity between w_c and mu_c ---
    ax = axes[1, 0]
    ax.bar(range(num_classes), nc3_cos_sims, alpha=0.7, width=1.0, color='green')
    ax.set_xlabel('Class index')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'NC3: cos(w_c, mu_c - mu_G) per class\n'
                 f'(mean = {nc3_mean_cos:.4f}, should → 1)', fontsize=11)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Perfect alignment')
    ax.set_ylim([min(0, nc3_cos_sims.min() - 0.1), 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- NC2: Class mean norms (equinorm check) ---
    ax = axes[1, 1]
    global_mean_train = train_features.mean(axis=0)
    class_means_check = np.zeros((num_classes, train_features.shape[1]))
    for c in range(num_classes):
        mask = (train_labels == c)
        class_means_check[c] = train_features[mask].mean(axis=0)
    centered_means_check = class_means_check - global_mean_train
    mean_norms = np.linalg.norm(centered_means_check, axis=1)

    ax.bar(range(num_classes), mean_norms, alpha=0.7, width=1.0, color='orange')
    ax.axhline(y=mean_norms.mean(), color='red', linestyle='--',
               label=f'Mean: {mean_norms.mean():.2f}, '
                     f'CV: {mean_norms.std() / mean_norms.mean():.4f}')
    ax.set_xlabel('Class index')
    ax.set_ylabel('||mu_c - mu_G||')
    ax.set_title('NC2 (equinorm): Norms of centered class means\n'
                 '(should be approximately equal)', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- NC4: Agreement between model and NCC ---
    ax = axes[1, 2]
    categories = ['Model Acc\n(train)', 'NCC Acc\n(train)', 'Agreement\n(Model vs NCC)']
    values = [nc4_model_acc * 100, nc4_ncc_acc * 100, nc4_agreement * 100]
    colors_nc4 = ['steelblue', 'coral', 'green']
    bars = ax.bar(categories, values, color=colors_nc4, alpha=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('NC4: Model vs NCC Classifier', fontsize=11)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Neural Collapse Analysis (NC1-NC4) on CIFAR-100 Training Data',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, output_dir, 'neural_collapse_nc1_nc4.png')


# ======================================================================
# NECO analysis
# ======================================================================

def plot_neco_analysis(
    id_neco: np.ndarray,
    ood_neco: np.ndarray,
    auroc_neco: float,
    id_features: np.ndarray,
    ood_features: np.ndarray,
    neco_normalized_means: np.ndarray,
    neco_global_mean: np.ndarray,
    output_dir: str,
    ood_name: str = "SVHN",
):
    """
    Visualize the NECO score: distribution + similarity patterns for
    typical ID and OOD samples.

    Parameters
    ----------
    id_neco, ood_neco : np.ndarray
        NECO scores for ID and OOD samples.
    auroc_neco : float
    id_features, ood_features : np.ndarray
        Raw feature arrays (before centering).
    neco_normalized_means : np.ndarray, shape (C, D)
    neco_global_mean : np.ndarray, shape (D,)
    output_dir : str
    ood_name : str
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Score distribution
    ax = axes[0]
    ax.hist(id_neco, bins=80, alpha=0.6, density=True, label='ID (CIFAR-100)', color='blue')
    ax.hist(ood_neco, bins=80, alpha=0.6, density=True, label=f'OOD ({ood_name})', color='red')
    ax.set_xlabel('NECO Score')
    ax.set_ylabel('Density')
    ax.set_title(f'NECO Score Distribution\nAUROC = {auroc_neco:.4f}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cosine similarity pattern: typical ID sample
    ax = axes[1]
    best_id_idx = np.argmax(id_neco)
    centered_id_sample = id_features[best_id_idx] - neco_global_mean
    normalized_id = centered_id_sample / np.linalg.norm(centered_id_sample)
    cos_pattern_id = normalized_id @ neco_normalized_means.T

    ax.bar(range(len(neco_normalized_means)),
           np.sort(cos_pattern_id)[::-1], alpha=0.7, width=1.0, color='blue')
    ax.set_xlabel('Class (sorted by similarity)')
    ax.set_ylabel('Cosine Similarity')
    num_classes = len(neco_normalized_means)
    ax.set_title('Similarity Pattern: Typical ID Sample\n'
                 '(one peak, rest near -1/(C-1))', fontsize=11)
    ax.axhline(y=-1 / (num_classes - 1), color='red', linestyle='--',
               label=f'ETF target: {-1 / (num_classes - 1):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cosine similarity pattern: typical OOD sample
    ax = axes[2]
    worst_ood_idx = np.argmin(ood_neco)
    centered_ood_sample = ood_features[worst_ood_idx] - neco_global_mean
    normalized_ood = centered_ood_sample / np.linalg.norm(centered_ood_sample)
    cos_pattern_ood = normalized_ood @ neco_normalized_means.T

    ax.bar(range(num_classes),
           np.sort(cos_pattern_ood)[::-1], alpha=0.7, width=1.0, color='red')
    ax.set_xlabel('Class (sorted by similarity)')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Similarity Pattern: Typical OOD Sample\n'
                 '(no clear peak, flatter distribution)', fontsize=11)
    ax.axhline(y=-1 / (num_classes - 1), color='red', linestyle='--',
               label=f'ETF target: {-1 / (num_classes - 1):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, f'neco_analysis_{ood_name.lower()}.png')


# ======================================================================
# NC across layers (Bonus)
# ======================================================================

def plot_nc_across_layers(
    nc1_per_layer: Dict[str, float],
    per_class_var_per_layer: Dict[str, np.ndarray],
    output_dir: str,
):
    """
    Plot NC1 metric vs. layer depth (Bonus analysis).

    Hypothesis: NC should be strongest in the last layers (closest to the
    loss) and weaker in earlier layers.  This is because Neural Collapse
    is driven by the cross-entropy loss optimizing the features and
    classifier jointly.

    Parameters
    ----------
    nc1_per_layer : dict
        ``{layer_name: nc1_value}``.
    per_class_var_per_layer : dict
        ``{layer_name: np.ndarray of per-class variances}``.
    output_dir : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layer_names = list(nc1_per_layer.keys())
    nc1_values = [nc1_per_layer[n] for n in layer_names]

    # NC1 metric across layers
    ax = axes[0]
    ax.plot(layer_names, nc1_values, 'bo-', markersize=10, linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('NC1 Metric (lower = more collapsed)')
    ax.set_title('NC1 (Within-class Collapse) Across Layers\n'
                 'Clearly decreases towards the output', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # log scale because the differences can be large

    # Mean within-class variance across layers
    ax = axes[1]
    mean_vars = [per_class_var_per_layer[n].mean() for n in layer_names]
    ax.plot(layer_names, mean_vars, 'rs-', markersize=10, linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Within-class Variance')
    ax.set_title('Mean Within-class Variance Across Layers', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, output_dir, 'nc_across_layers.png')

    logger.info("As expected, Neural Collapse (NC1) becomes stronger in later layers.")
    logger.info("This makes sense: the final layers are most directly optimized by the "
                "cross-entropy loss, which drives the collapse.")
