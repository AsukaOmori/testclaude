"""Rolling IC regime detection for lead-lag strategies."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stock_config import IC_WINDOW, IC_THRESHOLD


def compute_daily_ic(
    signals: pd.DataFrame, realized_returns: pd.DataFrame,
) -> pd.Series:
    """Compute daily Information Coefficient (Spearman rank correlation).

    IC(t) = SpearmanCorr(signal(t-1), realized_return(t))

    Signal on day t-1 predicts return on day t.
    """
    # Shift signals by 1 day to align prediction with realization
    shifted_signals = signals.shift(1)

    common_dates = shifted_signals.index.intersection(realized_returns.index)
    shifted_signals = shifted_signals.loc[common_dates]
    realized = realized_returns.loc[common_dates]

    ic_values = {}
    for date in common_dates:
        sig = shifted_signals.loc[date].dropna()
        ret = realized.loc[date].reindex(sig.index).dropna()

        # Need at least 4 stocks for meaningful rank correlation
        common_tickers = sig.index.intersection(ret.index)
        if len(common_tickers) < 4:
            continue

        s = sig[common_tickers].values
        r = ret[common_tickers].values

        if np.std(s) < 1e-10 or np.std(r) < 1e-10:
            ic_values[date] = 0.0
            continue

        corr, _ = spearmanr(s, r)
        ic_values[date] = corr if not np.isnan(corr) else 0.0

    return pd.Series(ic_values, name="daily_ic")


def compute_rolling_ic(
    daily_ic: pd.Series, window: int = IC_WINDOW,
) -> pd.Series:
    """Compute rolling mean of daily IC values."""
    return daily_ic.rolling(window=window, min_periods=window // 2).mean()


def detect_regime(
    rolling_ic: pd.Series, threshold: float = IC_THRESHOLD,
) -> pd.Series:
    """Detect active regime: trade only when rolling IC > threshold."""
    return rolling_ic > threshold
