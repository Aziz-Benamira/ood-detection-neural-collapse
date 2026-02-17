"""
Neural Collapse measurement functions (NC1 – NC4).
NC1 — Within-class Variability Collapse
NC2 — Convergence to Simplex ETF
NC3 — Convergence to Self-Duality (Classifier-Feature Alignment)
NC4 — Simplification to Nearest Class Center (NCC)
"""

import logging
from typing import Tuple

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)


# NC1: Within-class Variability Collapse
def measure_nc1(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 100,
) -> Tuple[float, np.ndarray]:
    #Measure NC1: Within-class variability collapse.
    
    D = features.shape[1]
    global_mean = features.mean(axis=0)  # shape (D,)

    # Within-class scatter matrix
    Sigma_W = np.zeros((D, D))
    per_class_var = np.zeros(num_classes)
    class_means_local = np.zeros((num_classes, D))

    for c in range(num_classes):
        mask = (labels == c)
        feats_c = features[mask] 
        mu_c = feats_c.mean(axis=0)
        class_means_local[c] = mu_c

        # Within-class scatter for class c
        centered = feats_c - mu_c
        Sigma_c = centered.T @ centered / len(feats_c)
        Sigma_W += Sigma_c

        per_class_var[c] = np.trace(Sigma_c)

    Sigma_W /= num_classes  # average across classes

    # Between-class scatter matrix
    centered_means = class_means_local - global_mean
    Sigma_B = centered_means.T @ centered_means / num_classes
    Sigma_B_inv = linalg.pinv(Sigma_B)
    nc1_metric = np.trace(Sigma_W @ Sigma_B_inv) / num_classes

    return nc1_metric, per_class_var


# NC2: Convergence to Simplex ETF
def measure_nc2(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 100,
) -> Tuple[float, np.ndarray, float]:
    global_mean = features.mean(axis=0)
    class_means_local = np.zeros((num_classes, features.shape[1]))

    for c in range(num_classes):
        mask = (labels == c)
        class_means_local[c] = features[mask].mean(axis=0)

    # Center the class means
    centered_means = class_means_local - global_mean 

    # Check equinorm: all norms should be equal
    norms = np.linalg.norm(centered_means, axis=1)  
    norm_std = np.std(norms) / np.mean(norms)

    normalized = centered_means / norms[:, np.newaxis]
    cos_sim_matrix = normalized @ normalized.T

    # Off-diagonal elements should all be -1/(C-1)
    mask_offdiag = ~np.eye(num_classes, dtype=bool)
    cos_mean_off_diag = cos_sim_matrix[mask_offdiag].mean()
    cos_std_off_diag = cos_sim_matrix[mask_offdiag].std()

    target_cos = -1.0 / (num_classes - 1)

    logger.info("  Equinorm: coefficient of variation = %.6f (should be ~0)", norm_std)
    logger.info("  Equiangular: mean off-diag cos sim = %.6f", cos_mean_off_diag)
    logger.info("               std off-diag cos sim  = %.6f", cos_std_off_diag)
    logger.info("               target (Simplex ETF)  = %.6f", target_cos)

    return norm_std, cos_sim_matrix, cos_mean_off_diag


# NC3: Self-Duality (Classifier-Feature Alignment)
def measure_nc3(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 100,
) -> Tuple[np.ndarray, float]:
    W = model.fc.weight.detach().cpu().numpy() 

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

def measure_nc4(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 100,
) -> Tuple[float, float, float]:
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
    h_sq = np.sum(features ** 2, axis=1, keepdims=True)  
    mu_sq = np.sum(class_means_local ** 2, axis=1)       
    cross = features @ class_means_local.T                
    distances = h_sq - 2 * cross + mu_sq               

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
