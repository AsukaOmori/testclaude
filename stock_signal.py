"""Cross-market signal generation for stock-level strategy."""

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from stock_config import (
    US_STOCKS, SECTOR_MAPPING, WINDOW, K_COMPONENTS, LAMBDA_STOCK,
)
from stock_data import rolling_standardize


def _build_prior_vectors(
    us_tickers: list[str], target_tickers: list[str],
    us_sectors: dict[str, list[str]], target_sectors: dict[str, list[str]],
) -> np.ndarray:
    """Build prior subspace V_0 with 3 orthogonal vectors."""
    all_tickers = us_tickers + target_tickers
    n = len(all_tickers)
    n_us = len(us_tickers)

    # v1: global factor
    v1 = np.ones(n)
    v1 /= np.linalg.norm(v1)

    # v2: country spread (US=+1, Target=-1)
    v2 = np.array([1.0] * n_us + [-1.0] * (n - n_us))
    v2 -= v2.dot(v1) * v1
    v2 /= np.linalg.norm(v2)

    # v3: sector similarity — same sector across countries gets same sign
    v3 = np.zeros(n)
    # Assign sector indices
    sector_list = list(set(list(us_sectors.keys()) + list(target_sectors.keys())))
    for i, ticker in enumerate(all_tickers):
        for j, sector in enumerate(sector_list):
            tickers_in_sector = (
                us_sectors.get(sector, []) + target_sectors.get(sector, [])
            )
            if ticker in tickers_in_sector:
                v3[i] = 1.0 if j % 2 == 0 else -1.0
                break
    v3 -= v3.dot(v1) * v1
    v3 -= v3.dot(v2) * v2
    norm = np.linalg.norm(v3)
    if norm > 1e-8:
        v3 /= norm
    else:
        v3 = np.random.randn(n)
        v3 -= v3.dot(v1) * v1
        v3 -= v3.dot(v2) * v2
        v3 /= np.linalg.norm(v3)

    return np.column_stack([v1, v2, v3])


def _compute_prior_correlation(joint_z: np.ndarray, V0: np.ndarray) -> np.ndarray:
    """Compute prior correlation C_0 from V_0 and full-sample correlation."""
    C_full = np.corrcoef(joint_z.T)
    C_full = np.nan_to_num(C_full, nan=0.0)

    D0 = np.diag(np.diag(V0.T @ C_full @ V0))
    C0_raw = V0 @ D0 @ V0.T

    diag_vals = np.diag(C0_raw)
    diag_vals = np.where(diag_vals > 1e-10, diag_vals, 1e-10)
    inv_sqrt = np.diag(1.0 / np.sqrt(diag_vals))
    C0 = inv_sqrt @ C0_raw @ inv_sqrt
    return C0


def compute_sector_pca_signal(
    us_cc: pd.DataFrame,
    target_cc: pd.DataFrame,
    us_sectors: dict[str, list[str]],
    target_sectors: dict[str, list[str]],
    window: int = WINDOW,
    k: int = K_COMPONENTS,
    lam: float = LAMBDA_STOCK,
) -> pd.DataFrame:
    """Compute PCA-based lead-lag signal from US to target market.

    Uses sector-aware prior subspace regularization.
    """
    us_tickers = list(us_cc.columns)
    target_tickers = list(target_cc.columns)
    n_us = len(us_tickers)
    n_all = n_us + len(target_tickers)

    # Build prior vectors
    V0 = _build_prior_vectors(us_tickers, target_tickers, us_sectors, target_sectors)

    # Joint returns for standardization
    joint_cc = pd.concat([us_cc, target_cc], axis=1)
    joint_z = rolling_standardize(joint_cc, window)

    # Compute prior C_0 from first 2 years of data
    prior_end = min(window * 5, len(joint_z) // 3)
    prior_data = joint_z.iloc[:prior_end].dropna()
    if len(prior_data) < window:
        prior_data = joint_z.dropna().iloc[:window * 3]
    C0 = _compute_prior_correlation(prior_data.values, V0)

    # Rolling signal computation
    signals = {}
    dates = joint_z.index[window:]

    for i in range(window, len(joint_z)):
        date = joint_z.index[i]
        z_window = joint_z.iloc[i - window:i].dropna()
        if len(z_window) < window // 2:
            continue

        # Sample correlation
        C_sample = np.corrcoef(z_window.values.T)
        C_sample = np.nan_to_num(C_sample, nan=0.0)

        # Regularized correlation
        C_reg = (1 - lam) * C_sample + lam * C0

        # Eigendecompose
        eigenvalues, eigenvectors = eigh(C_reg)
        idx = np.argsort(eigenvalues)[::-1][:k]
        V_k = eigenvectors[:, idx]

        # Split into US and target blocks
        V_us = V_k[:n_us, :]
        V_tgt = V_k[n_us:, :]

        # Current US standardized return
        z_us = joint_z.iloc[i, :n_us].values
        if np.any(np.isnan(z_us)):
            continue

        # Factor score and signal
        f = V_us.T @ z_us
        signal = V_tgt @ f

        signals[date] = dict(zip(target_tickers, signal))

    return pd.DataFrame(signals).T


def compute_sector_xcorr_signal(
    us_cc: pd.DataFrame,
    target_cc: pd.DataFrame,
    us_sectors: dict[str, list[str]],
    target_sectors: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute cross-correlation signal: US sector avg → target stocks.

    For each target stock, the signal is the average return of
    the corresponding US sector stocks on the same day.
    """
    signals = pd.DataFrame(index=us_cc.index, columns=target_cc.columns, dtype=float)

    for target_sector, target_tickers in target_sectors.items():
        us_sector = SECTOR_MAPPING.get(target_sector, target_sector)
        us_tickers = us_sectors.get(us_sector, [])

        # Filter to available tickers
        us_available = [t for t in us_tickers if t in us_cc.columns]
        if not us_available:
            continue

        # Average US sector return as signal
        us_avg = us_cc[us_available].mean(axis=1)

        for ticker in target_tickers:
            if ticker in signals.columns:
                signals[ticker] = us_avg

    return signals.dropna(how="all")


def compute_momentum_signal(
    target_cc: pd.DataFrame, window: int = WINDOW,
) -> pd.DataFrame:
    """Momentum baseline: rolling mean of target close-to-close returns."""
    return target_cc.rolling(window=window, min_periods=window).mean()
