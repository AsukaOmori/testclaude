"""Subspace-regularized PCA algorithm (Section 3.1-3.2 of the paper)."""

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from config import (
    US_TICKERS, JP_TICKERS, N_U, N_J, N, K, K0, LAMBDA, L,
    US_CYCLICAL, US_DEFENSIVE, JP_CYCLICAL, JP_DEFENSIVE,
)


def build_prior_vectors() -> np.ndarray:
    """Build V_0 ∈ R^{N×K_0}: three orthogonal prior exposure vectors.

    1. Global factor: v_1 ∝ 1 (equal weight on all N assets)
    2. Country spread: v_2 ∝ (1_{N_U}, -1_{N_J}), orthogonal to v_1
    3. Cyclical-Defensive: cyclical=+1, defensive=-1, orthogonal to v_1, v_2
    """
    all_tickers = US_TICKERS + JP_TICKERS

    # v1: global factor (equal weight)
    v1 = np.ones(N)
    v1 = v1 / np.linalg.norm(v1)

    # v2: country spread (US=+1, JP=-1)
    v2 = np.array([1.0 if t in US_TICKERS else -1.0 for t in all_tickers])
    # Gram-Schmidt: orthogonalize to v1
    v2 = v2 - np.dot(v2, v1) * v1
    v2 = v2 / np.linalg.norm(v2)

    # v3: cyclical-defensive factor
    cyclical_set = set(US_CYCLICAL + JP_CYCLICAL)
    defensive_set = set(US_DEFENSIVE + JP_DEFENSIVE)
    v3 = np.zeros(N)
    for i, t in enumerate(all_tickers):
        if t in cyclical_set:
            v3[i] = 1.0
        elif t in defensive_set:
            v3[i] = -1.0
    # Gram-Schmidt: orthogonalize to v1 and v2
    v3 = v3 - np.dot(v3, v1) * v1 - np.dot(v3, v2) * v2
    v3 = v3 / np.linalg.norm(v3)

    V0 = np.column_stack([v1, v2, v3])  # (N, K0)
    return V0


def compute_prior_correlation(
    cc_returns: pd.DataFrame,
    V0: np.ndarray,
) -> np.ndarray:
    """Compute prior correlation matrix C_0 (eq 10-12).

    Args:
        cc_returns: Joint (US+JP) close-to-close returns for the prior period
        V0: Prior exposure vectors (N, K0)

    Returns:
        C_0: Prior correlation matrix (N, N) with diag(C_0) = 1
    """
    # Compute full-sample correlation matrix C_full
    C_full = cc_returns.corr().values  # (N, N)

    # D_0 = diag(V_0^T C_full V_0) (eq 10)
    D0 = np.diag(np.diag(V0.T @ C_full @ V0))  # (K0, K0)

    # C_0^raw = V_0 D_0 V_0^T (eq 11)
    C0_raw = V0 @ D0 @ V0.T  # (N, N)

    # C_0 = Δ^{-1/2} C_0^raw Δ^{-1/2}, Δ = diag(C_0^raw) (eq 12)
    delta_diag = np.diag(C0_raw)
    # Handle near-zero diagonal elements
    delta_diag = np.maximum(delta_diag, 1e-10)
    delta_inv_sqrt = 1.0 / np.sqrt(delta_diag)
    C0 = C0_raw * np.outer(delta_inv_sqrt, delta_inv_sqrt)

    return C0


def compute_sample_correlation(Z: np.ndarray) -> np.ndarray:
    """Compute sample correlation matrix from standardized returns.

    C_t = Z_t^T Z_t / L, where Z_t ∈ R^{L×N} is the standardized return matrix.
    """
    L_obs = Z.shape[0]
    C = Z.T @ Z / L_obs  # (N, N)
    return C


def regularized_pca(
    C_t: np.ndarray,
    C_0: np.ndarray,
    lam: float = LAMBDA,
    k: int = K,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform subspace-regularized PCA (eq 13-16).

    C_t^reg = (1-λ)C_t + λC_0  (eq 13)
    Eigendecompose and return top K eigenvectors.

    Returns:
        eigenvalues: Top K eigenvalues (K,)
        eigenvectors: Top K eigenvectors (N, K)
    """
    C_reg = (1 - lam) * C_t + lam * C_0  # (eq 13)

    # Symmetric eigendecomposition (returns sorted ascending)
    eigenvalues, eigenvectors = eigh(C_reg)

    # Reverse to get descending order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Take top K
    top_eigenvalues = eigenvalues[:k]
    top_eigenvectors = eigenvectors[:, :k]  # (N, K)

    return top_eigenvalues, top_eigenvectors


def split_eigenvectors(
    V: np.ndarray,
    n_us: int = N_U,
) -> tuple[np.ndarray, np.ndarray]:
    """Split eigenvectors into US and Japan blocks (eq 16).

    V_{U,t}^(K) ∈ R^{N_U×K}, V_{J,t}^(K) ∈ R^{N_J×K}
    """
    V_US = V[:n_us, :]
    V_JP = V[n_us:, :]
    return V_US, V_JP
