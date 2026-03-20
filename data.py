"""Data download and return computation."""

import numpy as np
import pandas as pd

from config import US_TICKERS, JP_TICKERS, START_DATE, END_DATE
from data_provider import fetch_ohlc_multi


def download_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download OHLC data for given tickers via the multi-provider layer.

    Returns dict mapping ticker -> DataFrame with columns [Open, Close].
    """
    return fetch_ohlc_multi(tickers, start, end)


def compute_close_to_close_returns(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute close-to-close returns: r^cc_{i,t} = Close_t / Close_{t-1} - 1 (eq 1)."""
    closes = pd.DataFrame({ticker: df["Close"] for ticker, df in data.items()})
    returns = closes.pct_change()
    return returns.iloc[1:]  # Drop first NaN row


def compute_open_to_close_returns(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute open-to-close returns: r^oc_{j,t} = Close_t / Open_t - 1 (eq 2)."""
    oc_returns = {}
    for ticker, df in data.items():
        oc_returns[ticker] = df["Close"] / df["Open"] - 1
    return pd.DataFrame(oc_returns)


def align_us_jp_dates(
    us_cc_returns: pd.DataFrame,
    jp_cc_returns: pd.DataFrame,
    jp_oc_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align US and Japan data to common business days.

    For each Japan trading day t+1, pair with the most recent US trading day t.
    The US close-to-close return on day t is used to predict Japan open-to-close
    return on day t+1.
    """
    # Find common dates where both markets traded
    us_dates = us_cc_returns.index
    jp_dates = jp_cc_returns.index

    # For alignment: we need pairs (us_date_t, jp_date_{t+1})
    # Build aligned DataFrames where row i has:
    #   us_cc_returns[i] = US close-to-close return on us_date
    #   jp_cc_returns[i] = Japan close-to-close return on jp_date (same date alignment for correlation)
    #   jp_oc_returns[i] = Japan open-to-close return on jp_date (target for prediction)

    # Use intersection for correlation computation
    common_dates = us_dates.intersection(jp_dates)
    us_cc_aligned = us_cc_returns.loc[common_dates].copy()
    jp_cc_aligned = jp_cc_returns.loc[common_dates].copy()
    jp_oc_aligned = jp_oc_returns.loc[common_dates].copy()

    # Drop rows with any NaN
    valid_mask = us_cc_aligned.notna().all(axis=1) & jp_cc_aligned.notna().all(axis=1) & jp_oc_aligned.notna().all(axis=1)
    us_cc_aligned = us_cc_aligned[valid_mask]
    jp_cc_aligned = jp_cc_aligned[valid_mask]
    jp_oc_aligned = jp_oc_aligned[valid_mask]

    return us_cc_aligned, jp_cc_aligned, jp_oc_aligned


def load_and_prepare_data(
    start: str = START_DATE,
    end: str = END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full data pipeline: download, compute returns, align.

    Returns:
        us_cc: US close-to-close returns (aligned), shape (T, N_U)
        jp_cc: Japan close-to-close returns (aligned), shape (T, N_J)
        jp_oc: Japan open-to-close returns (aligned), shape (T, N_J)
    """
    print("Downloading US ETF data...")
    us_data = download_data(US_TICKERS, start, end)
    print("Downloading Japan ETF data...")
    jp_data = download_data(JP_TICKERS, start, end)

    print("Computing returns...")
    us_cc = compute_close_to_close_returns(us_data)
    jp_cc = compute_close_to_close_returns(jp_data)
    jp_oc = compute_open_to_close_returns(jp_data)

    print("Aligning dates...")
    us_cc, jp_cc, jp_oc = align_us_jp_dates(us_cc, jp_cc, jp_oc)

    print(f"Data ready: {len(us_cc)} common trading days, "
          f"{us_cc.shape[1]} US sectors, {jp_cc.shape[1]} JP sectors")

    return us_cc, jp_cc, jp_oc


def generate_synthetic_data(
    n_days: int = 3000,
    start: str = START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate synthetic data for testing when yfinance is unavailable.

    Simulates a factor model where US and Japan share common factors
    with a 1-day lag, mimicking the lead-lag structure.
    """
    np.random.seed(42)
    dates = pd.bdate_range(start=start, periods=n_days)

    # Common factors (K=3)
    K = 3
    factors = np.random.randn(n_days, K) * 0.01

    # US factor loadings (N_U=11)
    V_US = np.random.randn(len(US_TICKERS), K) * 0.5
    # Japan factor loadings (N_J=17)
    V_JP = np.random.randn(len(JP_TICKERS), K) * 0.5

    # US returns: driven by same-day factors + noise
    us_noise = np.random.randn(n_days, len(US_TICKERS)) * 0.005
    us_cc_vals = factors @ V_US.T + us_noise

    # Japan returns: driven by previous-day factors (lag) + noise
    jp_noise = np.random.randn(n_days, len(JP_TICKERS)) * 0.005
    jp_cc_vals = np.zeros((n_days, len(JP_TICKERS)))
    jp_cc_vals[1:] = factors[:-1] @ V_JP.T + jp_noise[1:]  # 1-day lag

    # Open-to-close returns (slightly different from close-to-close)
    jp_oc_vals = jp_cc_vals * 0.8 + np.random.randn(n_days, len(JP_TICKERS)) * 0.002

    us_cc = pd.DataFrame(us_cc_vals, index=dates, columns=US_TICKERS)
    jp_cc = pd.DataFrame(jp_cc_vals, index=dates, columns=JP_TICKERS)
    jp_oc = pd.DataFrame(jp_oc_vals, index=dates, columns=JP_TICKERS)

    print(f"Synthetic data generated: {n_days} trading days, "
          f"{len(US_TICKERS)} US sectors, {len(JP_TICKERS)} JP sectors")

    return us_cc, jp_cc, jp_oc


def rolling_standardize(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Standardize returns using rolling window mean and std (eq 8-9).

    z_{i,τ} = (r^cc_{i,τ} - μ_{i,t}) / σ_{i,t}
    where μ and σ are computed over window W_t = {t-L, ..., t-1}.
    """
    rolling_mean = returns.rolling(window=window, min_periods=window).mean()
    rolling_std = returns.rolling(window=window, min_periods=window).std(ddof=0)
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    standardized = (returns - rolling_mean) / rolling_std
    return standardized
