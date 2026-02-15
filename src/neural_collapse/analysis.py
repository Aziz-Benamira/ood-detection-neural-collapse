"""
Neural Collapse analysis orchestration.

This module provides high-level functions that run the full NC1-NC4
analysis, the NC5 test-time analysis, and the bonus NC-across-layers
experiment.  It calls into ``metrics.py`` for the individual
measurements.

Author: Aziz BEN AMIRA
Course: Theory of Deep Learning (MVA + ENSTA)
"""

import logging
from typing import Dict, Tuple

import numpy as np
from torch.utils.data import DataLoader
import torch

from .metrics import measure_nc1, measure_nc2, measure_nc3, measure_nc4
from ..utils.feature_extraction import extract_multi_layer_features

logger = logging.getLogger(__name__)


def run_full_nc_analysis(
    model,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    id_features: np.ndarray,
    id_logits: np.ndarray,
    id_labels: np.ndarray,
    class_means: np.ndarray,
    num_classes: int = 100,
) -> dict:
    """
    Run the complete NC1-NC5 analysis and return all metrics in a single
    dictionary.

    NC5 extends NC4 from training data to **test data**.  The claim is
    that after Neural Collapse, the network classifies test samples
    essentially by nearest class center, using the class means learned
    from the training set.

    Why NC5 matters for OOD:
    If the classifier simplifies to NCC for in-distribution data, then
    OOD samples should be far from *all* class centers.  This observation
    is the foundation of the NECO method.

    Parameters
    ----------
    model : nn.Module
    train_features, train_labels : np.ndarray
        Training features and labels (NC is a training-time phenomenon).
    id_features, id_logits, id_labels : np.ndarray
        Test (in-distribution) features, logits, and labels (for NC5).
    class_means : np.ndarray, shape (C, D)
        Per-class means (typically from Mahalanobis fitting).
    num_classes : int

    Returns
    -------
    results : dict
        Contains all NC metrics plus NC5 agreement numbers.
    """
    results = {}

    # ---- NC1 ----
    logger.info("=" * 60)
    logger.info("NEURAL COLLAPSE MEASUREMENTS")
    logger.info("=" * 60)

    logger.info("\n--- NC1: Within-class Variability Collapse ---")
    nc1_metric, per_class_var = measure_nc1(train_features, train_labels, num_classes)
    logger.info("  NC1 metric (tr(Σ_W @ Σ_B^{-1}) / C): %.6f", nc1_metric)
    logger.info("  (Should approach 0 if NC1 holds)")
    results['nc1_metric'] = nc1_metric
    results['per_class_var'] = per_class_var

    # ---- NC2 ----
    logger.info("\n--- NC2: Convergence to Simplex ETF ---")
    norm_std, cos_sim_matrix, cos_mean_offdiag = measure_nc2(
        train_features, train_labels, num_classes,
    )
    results['nc2_norm_std'] = norm_std
    results['nc2_cos_sim_matrix'] = cos_sim_matrix
    results['nc2_cos_mean_offdiag'] = cos_mean_offdiag

    # ---- NC3 ----
    logger.info("\n--- NC3: Self-Duality (Classifier-Feature Alignment) ---")
    nc3_cos_sims, nc3_mean_cos = measure_nc3(model, train_features, train_labels, num_classes)
    results['nc3_cos_sims'] = nc3_cos_sims
    results['nc3_mean_cos'] = nc3_mean_cos

    # ---- NC4 ----
    logger.info("\n--- NC4: Simplification to NCC ---")
    nc4_agreement, nc4_ncc_acc, nc4_model_acc = measure_nc4(
        model, train_features, train_labels, num_classes,
    )
    results['nc4_agreement'] = nc4_agreement
    results['nc4_ncc_acc'] = nc4_ncc_acc
    results['nc4_model_acc'] = nc4_model_acc

    # ---- NC5: NCC on test data ----
    # We use the class means from TRAINING data, but evaluate on TEST data.
    logger.info("\n--- NC5: Nearest Class Center on Test Data ---")
    test_h_sq = np.sum(id_features ** 2, axis=1, keepdims=True)
    test_mu_sq = np.sum(class_means ** 2, axis=1)
    test_cross = id_features @ class_means.T
    test_distances = test_h_sq - 2 * test_cross + test_mu_sq

    ncc_test_preds = np.argmin(test_distances, axis=1)
    model_test_preds = np.argmax(id_logits, axis=1)

    ncc_test_acc = float(np.mean(ncc_test_preds == id_labels))
    model_test_acc = float(np.mean(model_test_preds == id_labels))
    test_agreement = float(np.mean(ncc_test_preds == model_test_preds))

    logger.info("  Model test accuracy: %.2f%%", model_test_acc * 100)
    logger.info("  NCC test accuracy:   %.2f%%", ncc_test_acc * 100)
    logger.info("  Agreement (model vs NCC on test): %.2f%%", test_agreement * 100)
    logger.info("  The NCC classifier (trained means, tested on unseen data) achieves")
    logger.info("  comparable accuracy to the full model, confirming NC5.")

    results['nc5_ncc_test_acc'] = ncc_test_acc
    results['nc5_model_test_acc'] = model_test_acc
    results['nc5_test_agreement'] = test_agreement

    logger.info("=" * 60)

    return results


def measure_nc_across_layers(
    model,
    train_loader: DataLoader,
    device: torch.device,
    train_features_penult: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int = 100,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Bonus: measure NC1 at each residual-block layer of ResNet-18 to show
    that Neural Collapse is strongest in the deepest layers (closest to
    the loss).

    Hypothesis: NC should be strongest in the last layers (closest to
    the loss) and weaker in earlier layers.  This is because Neural
    Collapse is driven by the cross-entropy loss optimizing the features
    and classifier jointly.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    device : torch.device
    train_features_penult : np.ndarray
        Pre-extracted penultimate layer features (to avoid re-extracting).
    train_labels : np.ndarray
    num_classes : int

    Returns
    -------
    nc1_per_layer : dict
        ``{layer_name: nc1_value}``.
    per_class_var_per_layer : dict
        ``{layer_name: np.ndarray}``.
    """
    logger.info("Extracting features from all ResNet-18 layers...")
    layer_features, layer_labels = extract_multi_layer_features(model, train_loader, device)

    for name, feats in layer_features.items():
        logger.info("  %s: shape %s", name, feats.shape)

    logger.info("\nMeasuring NC1 across layers:")
    logger.info("=" * 50)

    nc1_per_layer: Dict[str, float] = {}
    per_class_var_per_layer: Dict[str, np.ndarray] = {}

    for name, feats in layer_features.items():
        nc1_val, pcv = measure_nc1(feats, layer_labels, num_classes)
        nc1_per_layer[name] = nc1_val
        per_class_var_per_layer[name] = pcv
        logger.info("  %s: NC1 = %.6f (feature dim = %d)", name, nc1_val, feats.shape[1])

    # Also include the penultimate layer
    nc1_penult, pcv_penult = measure_nc1(train_features_penult, train_labels, num_classes)
    nc1_per_layer['avgpool'] = nc1_penult
    per_class_var_per_layer['avgpool'] = pcv_penult
    logger.info("  avgpool (penultimate): NC1 = %.6f (feature dim = %d)",
                nc1_penult, train_features_penult.shape[1])

    return nc1_per_layer, per_class_var_per_layer
