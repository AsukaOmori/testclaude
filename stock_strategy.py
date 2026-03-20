"""Portfolio construction with regime filter for stock-level strategy."""

import numpy as np
import pandas as pd

from stock_config import Q, US_STOCKS, JP_STOCKS, KR_STOCKS, SECTOR_MAPPING
from stock_signal import (
    compute_sector_pca_signal,
    compute_sector_xcorr_signal,
    compute_momentum_signal,
)
from regime_detector import compute_daily_ic, compute_rolling_ic, detect_regime
from evaluation import evaluate_strategy


def construct_portfolio(
    signals: pd.DataFrame,
    target_oc: pd.DataFrame,
    regime_flags: pd.Series | None = None,
    q: float = Q,
) -> pd.Series:
    """Construct long-short portfolio with optional regime filter.

    Args:
        signals: Signal values (date × tickers). Signal on day t predicts day t+1.
        target_oc: Open-to-close returns (date × tickers).
        regime_flags: Boolean series. If provided, only trade on True days.
        q: Quantile for long/short selection.
    """
    # Shift signals: signal on day t → trade on day t+1
    shifted_signals = signals.shift(1)

    common_dates = shifted_signals.index.intersection(target_oc.index)
    shifted_signals = shifted_signals.loc[common_dates]
    returns = target_oc.loc[common_dates]

    daily_pnl = {}
    for date in common_dates:
        # Regime filter
        if regime_flags is not None and date in regime_flags.index:
            if not regime_flags.loc[date]:
                daily_pnl[date] = 0.0
                continue

        sig = shifted_signals.loc[date].dropna()
        ret = returns.loc[date].reindex(sig.index).dropna()

        common_tickers = sig.index.intersection(ret.index)
        if len(common_tickers) < 3:
            continue

        sig = sig[common_tickers]
        ret = ret[common_tickers]

        # Long top-q, short bottom-q
        upper = sig.quantile(1 - q)
        lower = sig.quantile(q)

        longs = sig[sig >= upper].index
        shorts = sig[sig <= lower].index

        if len(longs) == 0 or len(shorts) == 0:
            continue

        w_long = 1.0 / len(longs)
        w_short = -1.0 / len(shorts)

        pnl = ret[longs].sum() * w_long + ret[shorts].sum() * w_short
        daily_pnl[date] = pnl

    return pd.Series(daily_pnl, name="daily_return")


def run_stock_strategies(
    us_cc: pd.DataFrame,
    target_cc: pd.DataFrame,
    target_oc: pd.DataFrame,
    market_name: str,
    target_sectors: dict[str, list[str]],
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Run all strategy variants for one target market.

    Returns:
        (strategy_returns, rolling_ic_series) — both as dicts keyed by signal type
    """
    us_sectors = US_STOCKS

    print(f"\n--- {market_name.upper()} Market ---")

    # Compute signals
    print(f"  Computing SECTOR_PCA signal...")
    pca_signals = compute_sector_pca_signal(
        us_cc, target_cc, us_sectors, target_sectors,
    )

    print(f"  Computing SECTOR_XCORR signal...")
    xcorr_signals = compute_sector_xcorr_signal(
        us_cc, target_cc, us_sectors, target_sectors,
    )

    print(f"  Computing MOM signal...")
    mom_signals = compute_momentum_signal(target_cc)

    # Compute regime flags for each signal type
    rolling_ics = {}
    regime_flags_dict = {}
    for name, sig in [("PCA", pca_signals), ("XCORR", xcorr_signals)]:
        daily_ic = compute_daily_ic(sig, target_oc)
        r_ic = compute_rolling_ic(daily_ic)
        regime = detect_regime(r_ic)
        rolling_ics[name] = r_ic
        regime_flags_dict[name] = regime

    # Build portfolios
    results = {}

    print(f"  Building portfolios...")
    results[f"{market_name}_PCA"] = construct_portfolio(pca_signals, target_oc)
    results[f"{market_name}_PCA_REGIME"] = construct_portfolio(
        pca_signals, target_oc, regime_flags_dict["PCA"],
    )
    results[f"{market_name}_XCORR"] = construct_portfolio(xcorr_signals, target_oc)
    results[f"{market_name}_XCORR_REGIME"] = construct_portfolio(
        xcorr_signals, target_oc, regime_flags_dict["XCORR"],
    )
    results[f"{market_name}_MOM"] = construct_portfolio(mom_signals, target_oc)

    return results, rolling_ics
