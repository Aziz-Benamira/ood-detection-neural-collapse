"""
Neural Collapse measurement functions (NC1 – NC4).

Neural Collapse is a remarkable phenomenon observed at the **end of training**
(the "terminal phase") when networks are trained past zero training error.
Four properties simultaneously emerge:

NC1 — Within-class Variability Collapse
    The features of samples belonging to the same class converge to their
    class mean.  Formally, the within-class covariance Σ_W → 0.

NC2 — Convergence to Simplex ETF
    The class means (after centering to the global mean) converge to a Simplex
    Equiangular Tight Frame (ETF).  This means:
    - All class means are equidistant from each other
    - All pairwise angles between class means are equal: cos(θ) = -1/(C-1)
    - All class means have the same norm

NC3 — Convergence to Self-Duality (Classifier-Feature Alignment)
    The rows of the classifier weight matrix w_c align with the corresponding
    class means μ_c:
        w_c / ||w_c|| ≈ (μ_c - μ_G) / ||μ_c - μ_G||

NC4 — Simplification to Nearest Class Center (NCC)
    The network's prediction simplifies to a Nearest Class Center classifier.

Reference
---------
Papyan, Han & Donoho,
    "Prevalence of Neural Collapse during the terminal phase of deep
     learning training", PNAS 2020.

Author: Aziz BEN AMIRA
Course: Theory of Deep Learning (MVA + ENSTA)
"""

import logging
from typing import Tuple

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)


# NC1: Within-class Variability Collapse
# Σ_W = (1/N) Σ_c Σ_{i:y_i=c} (h_i - μ_c)(h_i - μ_c)^T → 0


def measure_nc1(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 100,
) -> Tuple[float, np.ndarray]:
    """
    Measure NC1: Within-class variability collapse.

    We compute the ratio of within-class covariance to between-class covariance:
        NC1_metric = trace(Σ_W @ Σ_B^{-1}) / C

    If NC1 holds, this ratio should approach 0 (within-class variance vanishes
    relative to between-class variance).

    We also return per-class variances so we can visualize them.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
    labels : np.ndarray, shape (N,)
    num_classes : int

    Returns
    -------
    nc1_metric : float
        The NC1 collapse metric (lower = more collapsed).
    per_class_var : np.ndarray, shape (C,)
        Trace of within-class covariance for each class.
    """
    D = features.shape[1]
    global_mean = features.mean(axis=0)  # shape (D,)

    # Within-class scatter matrix
    Sigma_W = np.zeros((D, D))
    per_class_var = np.zeros(num_classes)
    class_means_local = np.zeros((num_classes, D))

    for c in range(num_classes):
        mask = (labels == c)
        feats_c = features[mask]  # features of class c
        mu_c = feats_c.mean(axis=0)
        class_means_local[c] = mu_c

        # Within-class scatter for class c
        centered = feats_c - mu_c
        Sigma_c = centered.T @ centered / len(feats_c)
        Sigma_W += Sigma_c

        # Per-class variance (trace of class covariance)
        per_class_var[c] = np.trace(Sigma_c)

    Sigma_W /= num_classes  # average across classes

    # Between-class scatter matrix
    centered_means = class_means_local - global_mean
    Sigma_B = centered_means.T @ centered_means / num_classes

    # NC1 metric: tr(Σ_W @ Σ_B^{-1}) / C
    # We use pseudoinverse because Σ_B might be singular (if C > D)
    Sigma_B_inv = linalg.pinv(Sigma_B)
    nc1_metric = np.trace(Sigma_W @ Sigma_B_inv) / num_classes

    return nc1_metric, per_class_var


# NC2: Convergence to Simplex ETF


def measure_nc2(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 100,
) -> Tuple[float, np.ndarray, float]:
    """
    Measure NC2: Convergence to Simplex ETF.

    We check two things:
    1. Whether all centered class means have the same norm (equinorm).
    2. Whether all pairwise cosine similarities equal -1/(C-1) (equiangular).

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
    labels : np.ndarray, shape (N,)
    num_classes : int

    Returns
    -------
    norm_std : float
        Coefficient of variation of class mean norms (should be ~0 for
        perfect ETF).
    cos_sim_matrix : np.ndarray, shape (C, C)
        Pairwise cosine similarities of centered class means.
    cos_mean_off_diag : float
        Mean off-diagonal cosine similarity (should be ~-1/(C-1)).
    """
    global_mean = features.mean(axis=0)
    class_means_local = np.zeros((num_classes, features.shape[1]))

    for c in range(num_classes):
        mask = (labels == c)
        class_means_local[c] = features[mask].mean(axis=0)

    # Center the class means
    centered_means = class_means_local - global_mean  # shape (C, D)

    # Check equinorm: all norms should be equal
    norms = np.linalg.norm(centered_means, axis=1)  # shape (C,)
    norm_std = np.std(norms) / np.mean(norms)  # coefficient of variation

    # Check equiangularity: pairwise cosine similarities
    # Normalize each mean to unit length
    normalized = centered_means / norms[:, np.newaxis]
    cos_sim_matrix = normalized @ normalized.T  # shape (C, C)

    # Off-diagonal elements should all be -1/(C-1)
    mask_offdiag = ~np.eye(num_classes, dtype=bool)
    cos_mean_off_diag = cos_sim_matrix[mask_offdiag].mean()
    cos_std_off_diag = cos_sim_matrix[mask_offdiag].std()

    target_cos = -1.0 / (num_classes - 1)  # theoretical value for Simplex ETF

    logger.info("  Equinorm: coefficient of variation = %.6f (should be ~0)", norm_std)
    logger.info("  Equiangular: mean off-diag cos sim = %.6f", cos_mean_off_diag)
    logger.info("               std off-diag cos sim  = %.6f", cos_std_off_diag)
    logger.info("               target (Simplex ETF)  = %.6f", target_cos)

    return norm_std, cos_sim_matrix, cos_mean_off_diag


# NC3: Self-Duality (Classifier-Feature Alignment)
# w_c / ||w_c||  ≈  (μ_c - μ_G) / ||μ_c - μ_G||


def measure_nc3(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 100,
) -> Tuple[np.ndarray, float]:
    """
    Measure NC3: Self-duality (alignment between classifier weights and
    class means).

    We compute the cosine similarity between each classifier weight vector
    w_c and its corresponding centered class mean (μ_c - μ_G).

    Parameters
    ----------
    model : nn.Module
    features : np.ndarray, shape (N, D)
    labels : np.ndarray, shape (N,)
    num_classes : int

    Returns
    -------
    cos_sims : np.ndarray, shape (C,)
        Cosine similarity between w_c and (μ_c - μ_G) for each class c.
    mean_cos : float
        Mean cosine similarity across all classes (should be ~1).
    """
    W = model.fc.weight.detach().cpu().numpy()  # shape (C, D)

    global_mean = features.mean(axis=0)
    class_means_local = np.zeros((num_classes, features.shape[1]))
    for c in range(num_classes):
        mask = (labels == c)
        class_means_local[c] = features[mask].mean(axis=0)

    centered_means = class_means_local - global_mean

    # Compute cosine similarity between w_c and (μ_c - μ_G)
    cos_sims = np.zeros(num_classes)
    for c in range(num_classes):
        w_norm = W[c] / (np.linalg.norm(W[c]) + 1e-8)
        m_norm = centered_means[c] / (np.linalg.norm(centered_means[c]) + 1e-8)
        cos_sims[c] = np.dot(w_norm, m_norm)

    mean_cos = cos_sims.mean()
    logger.info("  Mean cosine similarity (w_c, μ_c - μ_G): %.6f (should be ~1)", mean_cos)
    logger.info("  Min:  %.6f", cos_sims.min())
    logger.info("  Max:  %.6f", cos_sims.max())
    logger.info("  Std:  %.6f", cos_sims.std())

    return cos_sims, mean_cos


# NC4: Simplification to Nearest Class Center (NCC)
# ŷ = argmax_c w_c^T h + b_c  ≈  argmin_c ||h - μ_c||²


def measure_nc4(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 100,
) -> Tuple[float, float, float]:
    """
    Measure NC4: Simplification to Nearest Class Center (NCC).

    We compare the model's actual predictions (argmax of logits) with the
    NCC predictions (argmin of Euclidean distance to class means).  If NC4
    holds, they should agree almost perfectly.

    Parameters
    ----------
    model : nn.Module
    features : np.ndarray, shape (N, D)
    labels : np.ndarray, shape (N,)
    num_classes : int

    Returns
    -------
    agreement : float
        Fraction of samples where model prediction == NCC prediction.
    ncc_accuracy : float
        Accuracy of the NCC classifier.
    model_accuracy : float
        Accuracy of the model (for reference).
    """
    # Model predictions
    W = model.fc.weight.detach().cpu().numpy()
    b = model.fc.bias.detach().cpu().numpy()
    logits = features @ W.T + b
    model_preds = np.argmax(logits, axis=1)

    # NCC predictions: classify to nearest class mean
    class_means_local = np.zeros((num_classes, features.shape[1]))
    for c in range(num_classes):
        mask = (labels == c)
        class_means_local[c] = features[mask].mean(axis=0)

    # Compute distances to all class means
    # ||h - μ_c||² = ||h||² - 2 h^T μ_c + ||μ_c||²
    # We can use this expanded form for efficient batch computation
    h_sq = np.sum(features ** 2, axis=1, keepdims=True)  # (N, 1)
    mu_sq = np.sum(class_means_local ** 2, axis=1)       # (C,)
    cross = features @ class_means_local.T                # (N, C)
    distances = h_sq - 2 * cross + mu_sq                  # (N, C)

    ncc_preds = np.argmin(distances, axis=1)

    # Compare
    agreement = np.mean(model_preds == ncc_preds)
    ncc_accuracy = np.mean(ncc_preds == labels)
    model_accuracy = np.mean(model_preds == labels)

    logger.info("  Model accuracy:    %.2f%%", model_accuracy * 100)
    logger.info("  NCC accuracy:      %.2f%%", ncc_accuracy * 100)
    logger.info("  Agreement (model vs NCC): %.2f%%", agreement * 100)

    return agreement, ncc_accuracy, model_accuracy


def measure_nc5_ood(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    ood_features: np.ndarray,
    num_classes: int = 100,
) -> float:
    """
    NC5 OOD: mean |cos(class_mean, ood_centroid)| across classes.
    Should be close to 0 if OOD features are orthogonal to class structure.
    """
    D = train_features.shape[1]
    class_means = np.zeros((num_classes, D))
    for c in range(num_classes):
        class_means[c] = train_features[train_labels == c].mean(axis=0)

    ood_mean = ood_features.mean(axis=0)
    ood_norm = np.linalg.norm(ood_mean)

    cos_abs = np.zeros(num_classes)
    for c in range(num_classes):
        c_norm = np.linalg.norm(class_means[c])
        cos_abs[c] = np.abs(np.dot(class_means[c], ood_mean) / (c_norm * ood_norm + 1e-10))

    nc5 = cos_abs.mean()
    logger.info("  NC5 OOD: mean |cos| = %.6f (should be ~0 for orthogonality)", nc5)

    return nc5
