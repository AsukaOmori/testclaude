"""Data download and return computation for stock-level strategy."""

import numpy as np
import pandas as pd

from stock_config import (
    US_STOCKS, JP_STOCKS, KR_STOCKS,
    US_TICKERS_FLAT, JP_TICKERS_FLAT, KR_TICKERS_FLAT,
    N_US, N_JP, N_KR, WINDOW, START_DATE,
)
from data_provider import fetch_ohlc_multi


def download_stock_data(
    stock_dict: dict[str, list[str]], start: str, end: str,
) -> dict[str, pd.DataFrame]:
    """Download OHLC data for stocks via the multi-provider layer."""
    tickers = [t for group in stock_dict.values() for t in group]
    return fetch_ohlc_multi(tickers, start, end)


def compute_cc_returns(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Close-to-close returns: r^cc = Close_t / Close_{t-1} - 1."""
    closes = pd.DataFrame({t: df["Close"] for t, df in data.items()})
    return closes.pct_change().iloc[1:]


def compute_oc_returns(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Open-to-close returns: r^oc = Close_t / Open_t - 1."""
    return pd.DataFrame({t: df["Close"] / df["Open"] - 1 for t, df in data.items()})


def align_leader_target(
    us_cc: pd.DataFrame,
    target_cc: pd.DataFrame,
    target_oc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align US and target market dates to common business days."""
    common = us_cc.index.intersection(target_cc.index)
    us_out = us_cc.loc[common].copy()
    tgt_cc_out = target_cc.loc[common].copy()
    tgt_oc_out = target_oc.loc[common].copy()

    valid = (
        us_out.notna().all(axis=1)
        & tgt_cc_out.notna().all(axis=1)
        & tgt_oc_out.notna().all(axis=1)
    )
    return us_out[valid], tgt_cc_out[valid], tgt_oc_out[valid]


def rolling_standardize(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Standardize returns using rolling window mean and std."""
    mu = returns.rolling(window=window, min_periods=window).mean()
    sigma = returns.rolling(window=window, min_periods=window).std(ddof=0)
    sigma = sigma.replace(0, np.nan)
    return (returns - mu) / sigma


def load_stock_data(
    start: str = START_DATE, end: str = "2025-12-31",
) -> dict:
    """Full pipeline: download all markets, compute returns, align.

    Returns dict with keys 'jp' and 'kr', each containing:
        (us_cc, target_cc, target_oc)
    """
    print("Downloading US stock data...")
    us_data = download_stock_data(US_STOCKS, start, end)
    us_cc = compute_cc_returns(us_data)

    result = {}
    for market_name, stock_dict in [("jp", JP_STOCKS), ("kr", KR_STOCKS)]:
        print(f"Downloading {market_name.upper()} stock data...")
        tgt_data = download_stock_data(stock_dict, start, end)
        tgt_cc = compute_cc_returns(tgt_data)
        tgt_oc = compute_oc_returns(tgt_data)

        print(f"Aligning US-{market_name.upper()} dates...")
        us_aligned, tgt_cc_aligned, tgt_oc_aligned = align_leader_target(
            us_cc, tgt_cc, tgt_oc,
        )
        result[market_name] = (us_aligned, tgt_cc_aligned, tgt_oc_aligned)
        print(f"  {market_name.upper()}: {len(us_aligned)} common days, "
              f"{us_aligned.shape[1]} US stocks, {tgt_cc_aligned.shape[1]} target stocks")

    return result


def generate_synthetic_stock_data(
    n_days: int = 2500, start: str = START_DATE,
) -> dict:
    """Generate synthetic data with sector-based lead-lag structure."""
    np.random.seed(123)
    dates = pd.bdate_range(start=start, periods=n_days)

    # 5 sector factors
    n_factors = 5
    factors = np.random.randn(n_days, n_factors) * 0.015

    # US loadings: each stock loads on its sector factor
    us_loadings = np.zeros((N_US, n_factors))
    idx = 0
    for i, tickers in enumerate(US_STOCKS.values()):
        for _ in tickers:
            us_loadings[idx, i] = 0.6 + np.random.rand() * 0.4
            idx += 1

    us_noise = np.random.randn(n_days, N_US) * 0.008
    us_cc_vals = factors @ us_loadings.T + us_noise

    result = {}
    for market_name, stock_dict, tickers_flat, n_tgt in [
        ("jp", JP_STOCKS, JP_TICKERS_FLAT, N_JP),
        ("kr", KR_STOCKS, KR_TICKERS_FLAT, N_KR),
    ]:
        # Target loadings with sector alignment
        tgt_loadings = np.zeros((n_tgt, n_factors))
        idx = 0
        sector_to_factor = {"tech": 0, "financials": 1, "auto_ind": 2, "trading": 2, "industrial": 2, "energy": 3, "consumer": 4}
        for sector, tickers in stock_dict.items():
            factor_idx = sector_to_factor.get(sector, 0)
            for _ in tickers:
                tgt_loadings[idx, factor_idx] = 0.4 + np.random.rand() * 0.3
                idx += 1

        tgt_noise = np.random.randn(n_days, n_tgt) * 0.008
        tgt_cc_vals = np.zeros((n_days, n_tgt))
        # 1-day lag: target responds to yesterday's US factors
        tgt_cc_vals[1:] = factors[:-1] @ tgt_loadings.T + tgt_noise[1:]

        tgt_oc_vals = tgt_cc_vals * 0.8 + np.random.randn(n_days, n_tgt) * 0.003

        us_cc = pd.DataFrame(us_cc_vals, index=dates, columns=US_TICKERS_FLAT)
        tgt_cc = pd.DataFrame(tgt_cc_vals, index=dates, columns=tickers_flat)
        tgt_oc = pd.DataFrame(tgt_oc_vals, index=dates, columns=tickers_flat)

        result[market_name] = (us_cc, tgt_cc, tgt_oc)

    n_jp = result["jp"][1].shape[1]
    n_kr = result["kr"][1].shape[1]
    print(f"Synthetic data: {n_days} days, {N_US} US, {n_jp} JP, {n_kr} KR stocks")
    return result
