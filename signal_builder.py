"""Signal construction from subspace-regularized PCA (Section 3.3)."""

import numpy as np
import pandas as pd

from config import US_TICKERS, JP_TICKERS, N_U, N_J, K, L, LAMBDA, PRIOR_END_DATE
from data import rolling_standardize
from subspace_reg_pca import (
    build_prior_vectors,
    compute_prior_correlation,
    compute_sample_correlation,
    regularized_pca,
    split_eigenvectors,
)


def compute_signals(
    us_cc: pd.DataFrame,
    jp_cc: pd.DataFrame,
    lam: float = LAMBDA,
    k: int = K,
    window: int = L,
    prior_end: str = PRIOR_END_DATE,
) -> pd.DataFrame:
    """Compute lead-lag signals for all strategies.

    Args:
        us_cc: US close-to-close returns (T, N_U)
        jp_cc: Japan close-to-close returns (T, N_J)
        lam: Regularization parameter
        k: Number of principal components
        window: Rolling window length
        prior_end: End date for prior correlation estimation

    Returns:
        DataFrame with columns for each strategy's signals per JP ticker,
        indexed by date. MultiIndex columns: (strategy, ticker).
    """
    # Build prior subspace
    V0 = build_prior_vectors()

    # Joint close-to-close returns for correlation
    joint_cc = pd.concat([us_cc, jp_cc], axis=1)
    all_tickers = list(us_cc.columns) + list(jp_cc.columns)

    # Compute C_full from prior period (2010/1/1 to 2014/12/31)
    prior_data = joint_cc.loc[:prior_end]
    if len(prior_data) < window:
        raise ValueError(f"Prior period has only {len(prior_data)} observations, need at least {window}")
    C_0 = compute_prior_correlation(prior_data, V0)

    # Rolling standardization of joint returns
    joint_z = rolling_standardize(joint_cc, window)

    # Prepare output containers
    dates = joint_cc.index[window:]  # Dates for which we have valid windows
    pca_sub_signals = {}
    pca_plain_signals = {}
    mom_signals = {}

    print(f"Computing signals for {len(dates)} trading days...")

    for i, date in enumerate(dates):
        # Current window indices
        window_end_idx = joint_cc.index.get_loc(date)
        window_start_idx = window_end_idx - window

        if window_start_idx < 0:
            continue

        # Standardized returns in window (eq 8-9)
        Z_window = joint_z.iloc[window_start_idx:window_end_idx].values  # (L, N)

        # Skip if any NaN in the window
        if np.any(np.isnan(Z_window)):
            continue

        # Sample correlation matrix C_t (from standardized returns)
        C_t = compute_sample_correlation(Z_window)

        # --- PCA_SUB: regularized PCA (λ=0.9) ---
        eigenvalues_sub, V_sub = regularized_pca(C_t, C_0, lam=lam, k=k)
        V_US_sub, V_JP_sub = split_eigenvectors(V_sub, n_us=N_U)

        # --- PCA_PLAIN: no regularization (λ=0) ---
        eigenvalues_plain, V_plain = regularized_pca(C_t, C_0, lam=0.0, k=k)
        V_US_plain, V_JP_plain = split_eigenvectors(V_plain, n_us=N_U)

        # Current US standardized return z_{U,t} (eq 17)
        z_us = joint_z.iloc[window_end_idx, :N_U].values  # (N_U,)

        if np.any(np.isnan(z_us)):
            continue

        # PCA_SUB signal: ẑ_{J,t+1} = V_JP @ V_US^T @ z_{U,t} (eq 18-20)
        f_sub = V_US_sub.T @ z_us       # (K,) factor score (eq 18)
        s_sub = V_JP_sub @ f_sub         # (N_J,) predicted signal (eq 19)
        pca_sub_signals[date] = s_sub

        # PCA_PLAIN signal
        f_plain = V_US_plain.T @ z_us
        s_plain = V_JP_plain @ f_plain
        pca_plain_signals[date] = s_plain

        # MOM signal: m_{j,t} = (1/L) Σ r^cc_{j,τ} (eq 31)
        # Simple mean of Japan close-to-close returns in window
        jp_window = jp_cc.iloc[window_start_idx:window_end_idx]
        m_j = jp_window.mean().values  # (N_J,)
        mom_signals[date] = m_j

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(dates)} days")

    # Convert to DataFrames
    jp_tickers = list(jp_cc.columns)
    pca_sub_df = pd.DataFrame.from_dict(pca_sub_signals, orient="index", columns=jp_tickers)
    pca_plain_df = pd.DataFrame.from_dict(pca_plain_signals, orient="index", columns=jp_tickers)
    mom_df = pd.DataFrame.from_dict(mom_signals, orient="index", columns=jp_tickers)

    print(f"Signals computed: {len(pca_sub_df)} valid trading days")

    return pca_sub_df, pca_plain_df, mom_df
