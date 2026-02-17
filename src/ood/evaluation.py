"""
OOD detection evaluation utilities.

Provides:
- AUROC computation for OOD detection performance measurement
- Evaluation of 6 OOD scoring methods (MSP, Max Logit, Energy, Mahalanobis, ViM, NECO)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from .scores import (
    score_msp,
    score_max_logit,
    score_energy,
    score_mahalanobis,
    compute_class_statistics,
    compute_vim_parameters,
    score_vim,
    compute_neco_parameters,
    score_neco,
)

logger = logging.getLogger(__name__)


def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    Compute AUROC for OOD detection.

    Convention: ID samples are labeled 1 (positive), OOD samples labeled 0.
    A higher score should indicate ID, so AUROC measures how well the score
    separates the two.

    Parameters
    ----------
    id_scores : np.ndarray
        Scores for in-distribution samples.
    ood_scores : np.ndarray
        Scores for out-of-distribution samples.

    Returns
    -------
    auroc : float
        Area Under the ROC Curve.
    """
    # Label ID as 1 (positive), OOD as 0 (negative class)
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(labels, scores)


def evaluate_all_ood_methods(
    model,
    train_features: np.ndarray,
    train_logits: np.ndarray,
    train_labels: np.ndarray,
    id_features: np.ndarray,
    id_logits: np.ndarray,
    ood_features: np.ndarray,
    ood_logits: np.ndarray,
    num_classes: int = 100,
    temperature: float = 1.0,
) -> Dict[str, dict]:
    """
    Evaluate all 6 OOD scoring methods and compute AUROC for each.

    Methods evaluated: MSP, Max Logit, Energy, Mahalanobis, ViM, NECO.

    Parameters
    ----------
    model : nn.Module
        Trained model (needed for ViM weight extraction).
    train_features, train_logits, train_labels : np.ndarray
        Training set features / logits / labels (for fitting Mahalanobis, ViM, NECO).
    id_features, id_logits : np.ndarray
        In-distribution test features / logits.
    ood_features, ood_logits : np.ndarray
        Out-of-distribution features / logits.
    num_classes : int
    temperature : float
        Temperature for the energy score.

    Returns
    -------
    results : dict
        ``{method_name: {'id': scores, 'ood': scores, 'auroc': float}}``.
    """
    results: Dict[str, dict] = {}

    # Method 1: Maximum Softmax Probability (MSP)
    id_msp = score_msp(id_logits)
    ood_msp = score_msp(ood_logits)
    results['MSP'] = {'id': id_msp, 'ood': ood_msp, 'auroc': compute_auroc(id_msp, ood_msp)}
    logger.info("MSP          AUROC: %.4f", results['MSP']['auroc'])

    # Method 2: Maximum Logit
    id_ml = score_max_logit(id_logits)
    ood_ml = score_max_logit(ood_logits)
    results['Max Logit'] = {'id': id_ml, 'ood': ood_ml, 'auroc': compute_auroc(id_ml, ood_ml)}
    logger.info("Max Logit    AUROC: %.4f", results['Max Logit']['auroc'])

    # Method 3: Energy Score
    id_energy = score_energy(id_logits, temperature)
    ood_energy = score_energy(ood_logits, temperature)
    results['Energy'] = {'id': id_energy, 'ood': ood_energy,
                         'auroc': compute_auroc(id_energy, ood_energy)}
    logger.info("Energy       AUROC: %.4f", results['Energy']['auroc'])

    # Method 4: Mahalanobis Distance
    logger.info("Computing class statistics for Mahalanobis...")
    class_means, shared_cov, precision = compute_class_statistics(
        train_features, train_labels, num_classes,
    )
    id_maha = score_mahalanobis(id_features, class_means, precision)
    ood_maha = score_mahalanobis(ood_features, class_means, precision)
    results['Mahalanobis'] = {'id': id_maha, 'ood': ood_maha,
                              'auroc': compute_auroc(id_maha, ood_maha)}
    logger.info("Mahalanobis  AUROC: %.4f", results['Mahalanobis']['auroc'])

    # Method 5: Variation in Magnitude (ViM)
    logger.info("Computing ViM parameters...")
    vim_NS, vim_alpha, vim_u = compute_vim_parameters(
        model, train_features, train_logits, num_classes,
    )
    id_vim = score_vim(id_features, id_logits, vim_NS, vim_alpha, vim_u)
    ood_vim = score_vim(ood_features, ood_logits, vim_NS, vim_alpha, vim_u)
    results['ViM'] = {'id': id_vim, 'ood': ood_vim,
                      'auroc': compute_auroc(id_vim, ood_vim)}
    logger.info("ViM          AUROC: %.4f", results['ViM']['auroc'])

    # Method 6: Neural Collapse Eye Classifier (NECO)
    logger.info("Computing NECO parameters...")
    neco_scaler, neco_pca, neco_dim = compute_neco_parameters(
        train_features, neco_dim=num_classes,
    )
    id_neco = score_neco(id_features, neco_scaler, neco_pca, neco_dim)
    ood_neco = score_neco(ood_features, neco_scaler, neco_pca, neco_dim)
    results['NECO'] = {'id': id_neco, 'ood': ood_neco,
                       'auroc': compute_auroc(id_neco, ood_neco)}
    logger.info("NECO         AUROC: %.4f", results['NECO']['auroc'])

    # Attach fitted parameters so downstream code (e.g. visualization) can reuse them
    results['_fitted'] = {
        'class_means': class_means,
        'precision': precision,
        'vim_NS': vim_NS,
        'vim_alpha': vim_alpha,
        'vim_u': vim_u,
        'neco_scaler': neco_scaler,
        'neco_pca': neco_pca,
        'neco_dim': neco_dim,
    }

    return results
