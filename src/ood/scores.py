"""
OOD (Out-of-Distribution) scoring functions.

Each function takes features and/or logits and returns a per-sample score
where **higher values indicate in-distribution** data.

Implemented methods
-------------------
1. MSP  — Maximum Softmax Probability  (Hendrycks & Gimpel, ICLR 2017)
2. Max Logit  (Hendrycks et al., ICML 2022)
3. Energy  (Liu et al., NeurIPS 2020)
4. Mahalanobis  (Lee et al., NeurIPS 2018)
5. ViM  — Virtual-logit Matching  (Wang et al., CVPR 2022)
6. NECO  — Neural Collapse inspired OOD detection  (Ammar et al., ICLR 2024)

Author: Aziz BEN AMIRA
Course: Theory of Deep Learning (MVA + ENSTA)
"""

import logging
from typing import Tuple

import numpy as np
from numpy.linalg import norm
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# MSP: Maximum Softmax Probability (Hendrycks & Gimpel, ICLR 2017)

def score_msp(logits: np.ndarray) -> np.ndarray:
    """
    Maximum Softmax Probability score. Higher = more likely ID.
    """
    # log-sum-exp trick for numerical stability
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return np.max(softmax_probs, axis=1)


# Max Logit Score (Hendrycks et al., ICML 2022)

def score_max_logit(logits: np.ndarray) -> np.ndarray:
    """
    Maximum raw logit. Higher = more likely ID.
    """
    return np.max(logits, axis=1)

# Energy Score (Liu et al., NeurIPS 2020)
# S = LogSumExp(z / T) where z are logits

def score_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Energy score = T * LogSumExp(logits / T). Higher = more likely ID."""
    return temperature * logsumexp(logits / temperature, axis=1)


# Mahalanobis Distance (Lee et al., NeurIPS 2018)

def compute_class_statistics(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class means and shared covariance. Returns (means, cov, precision)."""
    D = features.shape[1]
    class_means = np.zeros((num_classes, D))
    for c in range(num_classes):
        class_means[c] = features[labels == c].mean(axis=0)

    centered = []
    for c in range(num_classes):
        centered.extend(features[labels == c] - class_means[c])

    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(centered).astype(np.float64))

    return class_means, ec.covariance_, ec.precision_


def score_mahalanobis(
    features: np.ndarray,
    class_means: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    """
    Negative minimum Mahalanobis distance. Higher = more likely ID.
    """
    num_classes = class_means.shape[0]
    N = features.shape[0]

    all_distances = np.zeros((N, num_classes))

    for c in range(num_classes):
        diff = features - class_means[c]
        left = diff @ precision
        all_distances[:, c] = np.sum(left * diff, axis=1)

    return -np.min(all_distances, axis=1)

# ViM: Virtual-logit Matching (Wang et al., CVPR 2022)

def compute_vim_parameters(
    model,
    train_features: np.ndarray,
    train_logits: np.ndarray,
    num_classes: int = 100,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute ViM parameters: null space basis, alpha scaling, and origin u."""
    W = model.fc.weight.detach().cpu().numpy()
    b = model.fc.bias.detach().cpu().numpy()
    u = -np.linalg.pinv(W) @ b

    # for resnet18 (D=512): keep top 300 eigenvalues as principal space
    DIM = 300

    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(train_features - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)

    # null space = eigenvectors beyond the top DIM
    NS = np.ascontiguousarray(
        (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T
    )

    vlogit_train = norm(np.matmul(train_features - u, NS), axis=-1)
    alpha = train_logits.max(axis=-1).mean() / vlogit_train.mean()
    logger.info("ViM: DIM=%d, alpha=%.4f", DIM, alpha)

    return NS, alpha, u


def score_vim(
    features: np.ndarray,
    logits: np.ndarray,
    NS: np.ndarray,
    alpha: float,
    u: np.ndarray,
) -> np.ndarray:
    """ViM score = energy - alpha * vlogit. Higher = more likely ID."""
    vlogit = norm(np.matmul(features - u, NS), axis=-1) * alpha
    energy = logsumexp(logits, axis=-1)
    return -vlogit + energy


# NECO: Neural Collapse Based OOD Detection (Ammar et al., ICLR 2024)

def compute_neco_parameters(
    train_features: np.ndarray,
    neco_dim: int = 100,
):
    """Fit StandardScaler + PCA on training features for NECO scoring."""
    ss = StandardScaler()
    scaled = ss.fit_transform(train_features)

    pca = PCA(n_components=train_features.shape[1])
    pca.fit(scaled)

    return ss, pca, neco_dim


def score_neco(
    features: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    neco_dim: int,
) -> np.ndarray:
    """NECO score = ||h_reduced|| / ||h_full||. Higher = more likely ID."""
    scaled = scaler.transform(features)
    reduced_all = pca.transform(scaled)
    reduced = reduced_all[:, :neco_dim]

    full_norms = norm(scaled, axis=1)
    reduced_norms = norm(reduced, axis=1)

    # for resnet: just the norm ratio, no maxlogit multiplication
    return reduced_norms / (full_norms + 1e-10)
