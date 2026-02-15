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
from scipy import linalg

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Maximum Softmax Probability (MSP)
# ======================================================================
#
# Reference: Hendrycks & Gimpel,
#   "A Baseline for Detecting Misclassified and Out-of-Distribution
#    Examples in Neural Networks", ICLR 2017.
#
# Mathematical formulation:
#   Given logits z = (z_1, ..., z_C) for C classes, the softmax probability
#   of class c is:
#       p_c = exp(z_c) / sum_j exp(z_j)
#   The MSP score is simply:
#       S_MSP(x) = max_c p_c
#
# Intuition: a confident model assigns high probability to one class for
# ID data, and spreads probability more uniformly for OOD data.  So a
# *higher* max softmax probability suggests the sample is in-distribution.
#
# Limitation: softmax can be overconfident even for OOD inputs (Nguyen
# et al., 2015).  The softmax function normalizes logits, so even if all
# logits are small, the max softmax can still be high if one logit is
# slightly larger than the others.
# ======================================================================

def score_msp(logits: np.ndarray) -> np.ndarray:
    """
    Maximum Softmax Probability (MSP) score.

    The simplest OOD baseline: just take the max softmax probability.
    Higher score = more likely to be in-distribution.

    Parameters
    ----------
    logits : np.ndarray, shape (N, C)
        Raw logits for N samples across C classes.

    Returns
    -------
    scores : np.ndarray, shape (N,)
        MSP score for each sample.
    """
    # We use the log-sum-exp trick for numerical stability:
    #   softmax(z)_c = exp(z_c - logsumexp(z))
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return np.max(softmax_probs, axis=1)


# ======================================================================
# 2. Maximum Logit Score
# ======================================================================
#
# Reference: Hendrycks et al.,
#   "Scaling Out-of-Distribution Detection for Real-World Settings",
#   ICML 2022.
#
# Mathematical formulation:
#   S_MaxLogit(x) = max_c z_c
#
# Intuition: instead of using softmax (which normalizes and can hide
# information), we directly look at the largest raw logit.  The idea is
# that for ID data, the network produces at least one very large logit
# (strong evidence for some class), while for OOD data, all logits tend
# to be moderate.
#
# Why it can be better than MSP:  Softmax is invariant to adding a
# constant to all logits (softmax(z) == softmax(z + c)).  This means MSP
# loses information about the *magnitude* of the logits.  MaxLogit
# preserves this magnitude information.
# ======================================================================

def score_max_logit(logits: np.ndarray) -> np.ndarray:
    """
    Maximum Logit Score.

    Even simpler than MSP: just take the maximum raw logit value.
    No softmax normalization needed.

    Parameters
    ----------
    logits : np.ndarray, shape (N, C)

    Returns
    -------
    scores : np.ndarray, shape (N,)
    """
    return np.max(logits, axis=1)


# ======================================================================
# 3. Energy Score
# ======================================================================
#
# Reference: Liu et al.,
#   "Energy-based Out-of-distribution Detection", NeurIPS 2020.
#
# Mathematical formulation:
#   The energy of an input x is defined as the negative log of the
#   partition function:
#       E(x) = -log sum_c exp(z_c)
#   We use the *negative* energy as our score (higher = more likely ID):
#       S_Energy(x) = log sum_c exp(z_c)
#   This is also known as the LogSumExp (LSE) of the logits.
#
# Intuition: the energy score captures the overall "activation level"
# of the network.  For ID data, the logits tend to have large values
# (the network is excited about some class), giving a high LSE.  For
# OOD data, logits are more muted, giving a lower LSE.
#
# Why it's better than MSP: MSP normalizes away the total magnitude.
# Energy preserves it.  For example, logits [10, 10, 10] and [1, 1, 1]
# give the same MSP (1/3 each) but very different energies.
#
# Connection to Boltzmann distributions: in statistical physics, the
# partition function Z = sum_c exp(-E_c / kT) measures the total
# "density of states".  Here, the logits play the role of negative
# energies.
# ======================================================================

def score_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Energy Score (negative free energy).

    Computed as LogSumExp of the logits, optionally with a temperature.
    Temperature T > 1 smooths the distribution, T < 1 sharpens it.
    The original paper uses T=1.

    Parameters
    ----------
    logits : np.ndarray, shape (N, C)
    temperature : float
        Temperature scaling parameter (default: 1.0).

    Returns
    -------
    scores : np.ndarray, shape (N,)
        Energy score (higher = more likely ID).
    """
    scaled_logits = logits / temperature

    # LogSumExp with numerical stability:
    #   log(sum(exp(z))) = max(z) + log(sum(exp(z - max(z))))
    max_logits = np.max(scaled_logits, axis=1, keepdims=True)
    scores = max_logits.squeeze() + np.log(
        np.sum(np.exp(scaled_logits - max_logits), axis=1)
    )

    # Multiply by temperature to get the actual energy
    return temperature * scores


# ======================================================================
# 4. Mahalanobis Distance Score
# ======================================================================
#
# Reference: Lee et al.,
#   "A Simple Unified Framework for Detecting Out-of-Distribution
#    Samples and Adversarial Attacks", NeurIPS 2018.
#
# For each class c, we compute the class-conditional mean of the features:
#   mu_c = (1/N_c) sum_{i: y_i=c} h_i
#
# We also compute the **shared** covariance matrix (tied covariance):
#   Sigma = (1/N) sum_c sum_{i: y_i=c} (h_i - mu_c)(h_i - mu_c)^T
#
# The Mahalanobis distance of a test feature h to class c is:
#   d_c(h) = (h - mu_c)^T Sigma^{-1} (h - mu_c)
#
# The score is the *negative* minimum Mahalanobis distance:
#   S_Maha(x) = -min_c d_c(h)
#
# Intuition (ELI5): imagine each class forms a cloud of points in
# feature space.  The Mahalanobis distance measures how far a point is
# from the center of the *nearest* cloud, taking into account the
# cloud's shape (via the covariance matrix).  ID points should be close
# to some cloud; OOD points should be far from all clouds.
#
# Why covariance matters:  Euclidean distance treats all feature
# dimensions equally.  But some directions might have much more
# variance than others.  The Mahalanobis distance "stretches" the space
# so that each direction is normalized by its variance.  This is like
# converting to a standardized z-score, but in multiple dimensions.
#
# Why tied covariance?  Using a separate covariance per class would
# require more data (we'd need enough samples per class to estimate a
# 512×512 matrix reliably).  A shared covariance is a good compromise —
# it assumes the shape of the class clouds is similar, which is
# reasonable under neural collapse.
# ======================================================================

def compute_class_statistics(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-class means and shared covariance matrix from training
    features.  This is the "fitting" step for the Mahalanobis score —
    we need to know what the ID feature distribution looks like before
    we can measure distance from it.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Feature vectors from the penultimate layer.
    labels : np.ndarray, shape (N,)
        Class labels.
    num_classes : int
        Number of classes.

    Returns
    -------
    class_means : np.ndarray, shape (C, D)
        Mean feature vector for each class.
    shared_cov : np.ndarray, shape (D, D)
        Shared (tied) covariance matrix.
    precision : np.ndarray, shape (D, D)
        Inverse of the covariance matrix (precomputed for efficiency).
    """
    D = features.shape[1]  # feature dimension (512 for ResNet-18)
    class_means = np.zeros((num_classes, D))

    # Step 1: compute per-class means
    for c in range(num_classes):
        mask = (labels == c)
        class_means[c] = features[mask].mean(axis=0)

    # Step 2: compute shared covariance
    # For each sample, subtract its class mean, then compute outer product
    centered_features = features.copy()
    for c in range(num_classes):
        mask = (labels == c)
        centered_features[mask] -= class_means[c]

    # Covariance = (1/N) * X_centered^T @ X_centered
    shared_cov = np.cov(centered_features, rowvar=False)

    # Add small regularization to ensure the covariance matrix is invertible.
    # Without this, singular or near-singular matrices would cause numerical
    # issues.
    shared_cov += 1e-6 * np.eye(D)

    # Step 3: precompute precision matrix (inverse covariance)
    # This is the expensive step, but we only do it once.
    precision = linalg.inv(shared_cov)

    return class_means, shared_cov, precision


def score_mahalanobis(
    features: np.ndarray,
    class_means: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    """
    Mahalanobis distance-based OOD score.

    For each test sample, compute the Mahalanobis distance to every class
    mean, then return the negative of the minimum distance.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
    class_means : np.ndarray, shape (C, D)
    precision : np.ndarray, shape (D, D)

    Returns
    -------
    scores : np.ndarray, shape (N,)
        Negative minimum Mahalanobis distance (higher = more likely ID).
    """
    num_classes = class_means.shape[0]
    N = features.shape[0]

    # We compute distances to all class means at once for efficiency
    # d_c(h) = (h - mu_c)^T @ Sigma^{-1} @ (h - mu_c)
    all_distances = np.zeros((N, num_classes))

    for c in range(num_classes):
        diff = features - class_means[c]  # (N, D)
        # Batch Mahalanobis: each row's d = diff @ precision @ diff^T (diagonal)
        left = diff @ precision  # (N, D)
        all_distances[:, c] = np.sum(left * diff, axis=1)  # (N,)

    # Take negative of minimum distance across classes
    # Negative because higher score = more ID
    return -np.min(all_distances, axis=1)


# ======================================================================
# 5. ViM (Virtual-logit Matching)
# ======================================================================
#
# Reference: Wang et al.,
#   "ViM: Out-Of-Distribution with Virtual-logit Matching", CVPR 2022.
#
# Key insight: Most OOD scoring methods only look at the part of the
# feature space that the classifier *uses* (the row space of the weight
# matrix W).  But there's also a **null space** — directions in feature
# space that the classifier completely ignores.  OOD samples may have
# significant energy in this null space.
#
# Let W ∈ R^{C×D} be the weight matrix of the last linear layer.
#
# 1. Compute the principal subspace P of W (using SVD): this captures
#    the C directions in feature space that the classifier uses.
#
# 2. The **residual** of a feature h in the null space is:
#       r = h - P P^T h
#    This is the component of h that the classifier *cannot see*.
#
# 3. The **virtual logit** is the norm of this residual, scaled by α:
#       v = α ||r||
#
# 4. The ViM score combines the energy of the original logits with the
#    virtual logit:
#       S_ViM(x) = log( sum_c exp(z_c) + exp(v) )
#
# Intuition (ELI5): think of the classifier as looking at a shadow of
# the feature vector.  Two very different objects can cast the same
# shadow if they differ in a direction perpendicular to the screen.
# ViM also checks the "depth" (null space component) to distinguish
# them.
# ======================================================================

def compute_vim_parameters(
    model,
    train_features: np.ndarray,
    train_logits: np.ndarray,
    num_classes: int = 100,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Precompute the ViM parameters from the trained model and training data.

    Steps:
    1. Get the last layer's weight matrix W.
    2. Compute the principal subspace via SVD (directions the classifier uses).
    3. Compute the null space projection matrix.
    4. Compute the scaling constant alpha from training data.

    Parameters
    ----------
    model : nn.Module
    train_features : np.ndarray, shape (N, D)
    train_logits : np.ndarray, shape (N, C)
    num_classes : int

    Returns
    -------
    null_space_proj : np.ndarray, shape (D, D)
        Projection matrix onto the null space of W.
    alpha : float
        Scaling constant for the virtual logit.
    feature_mean : np.ndarray, shape (D,)
        Mean feature vector (we center features before projection).
    """
    # Get the weight matrix W of the final linear layer
    W = model.fc.weight.detach().cpu().numpy()  # shape (C, D)
    # b = model.fc.bias.detach().cpu().numpy()  # shape (C,)  (not used directly)

    # SVD of W: W = U @ diag(s) @ V^T
    # V contains the right singular vectors — the directions in feature space.
    # The first C columns of V span the row space of W (what the classifier "sees").
    U, s, Vt = np.linalg.svd(W, full_matrices=False)

    # Principal subspace: first C right singular vectors
    # V = Vt.T, so V[:, :C] = Vt[:C, :].T
    principal_space = Vt.T[:, :num_classes]  # shape (D, C)

    # Null space projection: I - V @ V^T
    # This removes the component in the classifier's row space
    D = W.shape[1]
    null_space_proj = np.eye(D) - principal_space @ principal_space.T  # shape (D, D)

    # Center features (subtract global mean)
    feature_mean = train_features.mean(axis=0)
    centered = train_features - feature_mean

    # Compute residuals for training data
    residuals = centered @ null_space_proj.T  # shape (N, D)
    residual_norms = np.linalg.norm(residuals, axis=1)  # shape (N,)

    # Alpha: scaling constant so that the virtual logit has similar
    # magnitude to the actual logits on training data.
    # We match the mean energy of the logits with the mean residual norm.
    logit_energy = np.log(np.sum(np.exp(train_logits), axis=1))  # log-sum-exp
    alpha = logit_energy.mean() / residual_norms.mean()

    logger.info("ViM parameters:")
    logger.info("  Feature dim D = %d, num classes C = %d", D, num_classes)
    logger.info("  Singular values range: [%.3f, %.3f]", s.min(), s.max())
    logger.info("  Alpha (scaling): %.4f", alpha)
    logger.info("  Avg residual norm (train): %.4f", residual_norms.mean())

    return null_space_proj, alpha, feature_mean


def score_vim(
    features: np.ndarray,
    logits: np.ndarray,
    null_space_proj: np.ndarray,
    alpha: float,
    feature_mean: np.ndarray,
) -> np.ndarray:
    """
    ViM (Virtual-logit Matching) score.

    Combines the energy of the logits with a "virtual logit" from the
    null space of the classifier's weight matrix.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
    logits : np.ndarray, shape (N, C)
    null_space_proj : np.ndarray, shape (D, D)
    alpha : float
    feature_mean : np.ndarray, shape (D,)

    Returns
    -------
    scores : np.ndarray, shape (N,)
    """
    # Center features
    centered = features - feature_mean

    # Project onto null space and compute norm
    residuals = centered @ null_space_proj.T
    virtual_logit = alpha * np.linalg.norm(residuals, axis=1)  # shape (N,)

    # Combine with actual logits: LogSumExp([logits, virtual_logit])
    # We append the virtual logit as an extra "class"
    augmented_logits = np.concatenate(
        [logits, virtual_logit[:, np.newaxis]], axis=1
    )  # shape (N, C+1)

    # Energy of augmented logits
    max_aug = np.max(augmented_logits, axis=1, keepdims=True)
    scores = max_aug.squeeze() + np.log(
        np.sum(np.exp(augmented_logits - max_aug), axis=1)
    )

    # ViM score: negative energy (because OOD should have HIGHER virtual
    # logit, leading to higher augmented energy — we want to flip the
    # direction)
    return -scores


# ======================================================================
# 6. NECO (Neural Collapse inspired OOD detection)
# ======================================================================
#
# Reference: Ammar et al.,
#   "NECO: NEural Collapse Based Out-of-distribution detection",
#   ICLR 2024.
#
# Key Idea
# --------
# Since Neural Collapse tells us that:
#   1. ID features cluster tightly around their class means (NC1)
#   2. These class means form an ETF (NC2)
#   3. The classifier aligns with these means (NC3)
#
# We can design an OOD score that leverages this geometric structure
# explicitly.
#
# NECO Score
# ----------
# The NECO method works in the feature space and uses the structure
# predicted by Neural Collapse:
#
# 1. Normalize features and class means to the unit sphere (since NC2
#    says the ETF lives on a sphere).
#
# 2. Project features onto the class mean directions and analyze the
#    resulting similarity pattern.
#
# 3. The score uses the angular structure: for an ID sample, the cosine
#    similarities to the class means should have a specific pattern
#    (high similarity to one class, equal low similarity to all others
#    — matching the ETF geometry).
#
# NECO score formulation:
#   Given a test feature h, compute the cosine similarities to all
#   (normalized) class means:
#       s_c = h^T mu_tilde_c / (||h|| ||mu_tilde_c||)
#   where mu_tilde_c = mu_c - mu_G are the centered class means.
#
#   The NECO score is based on how well the similarity pattern matches
#   the ETF structure:
#       S_NECO(x) = max_c s_c  -  (1/(C-1)) sum_{c' ≠ c*} s_{c'}
#   where c* = argmax_c s_c is the predicted class.
#
# Intuition: for an ID sample, the max similarity should be high and
# the rest should be low (close to -1/(C-1)), giving a large score.
# For OOD, the pattern is less structured, giving a smaller score.
# ======================================================================

def compute_neco_parameters(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the parameters needed for the NECO OOD score.

    This includes:
    1. Centered class means.
    2. Normalized class means (on unit sphere).

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Training features.
    labels : np.ndarray, shape (N,)
        Training labels.
    num_classes : int

    Returns
    -------
    normalized_means : np.ndarray, shape (C, D)
        Centered and normalized class means.
    global_mean : np.ndarray, shape (D,)
        Global feature mean.
    """
    D = features.shape[1]
    global_mean = features.mean(axis=0)

    # Compute per-class means
    class_means_local = np.zeros((num_classes, D))
    for c in range(num_classes):
        mask = (labels == c)
        class_means_local[c] = features[mask].mean(axis=0)

    # Center the means (subtract global mean)
    centered_means = class_means_local - global_mean

    # Normalize to unit sphere
    norms = np.linalg.norm(centered_means, axis=1, keepdims=True)
    normalized_means = centered_means / (norms + 1e-8)

    return normalized_means, global_mean


def score_neco(
    features: np.ndarray,
    normalized_means: np.ndarray,
    global_mean: np.ndarray,
) -> np.ndarray:
    """
    NECO (Neural Collapse based OOD detection) score.

    The score leverages the ETF structure: for ID samples, the cosine
    similarity pattern to class means should match the Simplex ETF.

    Score = max similarity - mean of remaining similarities

    This captures how "peaky" the similarity distribution is. ID samples
    should have one high similarity and many low ones; OOD samples should
    have a flatter distribution.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
    normalized_means : np.ndarray, shape (C, D)
    global_mean : np.ndarray, shape (D,)

    Returns
    -------
    scores : np.ndarray, shape (N,)
        NECO score (higher = more likely ID).
    """
    # Center features
    centered = features - global_mean

    # Normalize features to unit sphere
    feat_norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized_feats = centered / (feat_norms + 1e-8)

    # Compute cosine similarities to all class means
    # Since both are normalized, dot product = cosine similarity
    cos_sims = normalized_feats @ normalized_means.T  # shape (N, C)

    num_classes = normalized_means.shape[0]

    # For each sample:
    # score = max_c(cos_sim_c) - mean of other similarities
    max_sims = np.max(cos_sims, axis=1)  # shape (N,)
    sum_sims = np.sum(cos_sims, axis=1)  # shape (N,)

    # Mean of the non-max similarities: sum_others = sum_all - max
    mean_others = (sum_sims - max_sims) / (num_classes - 1)

    # NECO score: gap between max similarity and average of rest
    scores = max_sims - mean_others

    return scores
