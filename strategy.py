"""Trading strategy: long-short portfolio construction (Section 2.2)."""

import numpy as np
import pandas as pd

from config import Q


def construct_long_short_portfolio(
    signals: pd.DataFrame,
    jp_oc_returns: pd.DataFrame,
    q: float = Q,
) -> pd.Series:
    """Construct long-short portfolio returns from signals (eq 3-7).

    For each day:
    - s_{j,t} = ẑ_{j,t+1} (signal)
    - L_{t+1} = Top-q set (eq 3)
    - S_{t+1} = Bottom-q set (eq 4)
    - w = +1/|L| for longs, -1/|S| for shorts (eq 5)
    - R_{t+1} = Σ w_{j,t+1} r^oc_{j,t+1} (eq 7)

    Note: Signal on day t predicts day t+1, so we shift signals by 1 day.
    """
    # Align signals (day t) with next-day returns (day t+1)
    common_dates = signals.index.intersection(jp_oc_returns.index)
    signals_aligned = signals.loc[common_dates]
    # Shift: signal on date[i] → return on date[i+1]
    return_dates = common_dates[1:]
    signal_dates = common_dates[:-1]

    strategy_returns = []
    valid_dates = []

    for sig_date, ret_date in zip(signal_dates, return_dates):
        s = signals_aligned.loc[sig_date]
        r_oc = jp_oc_returns.loc[ret_date]

        # Drop NaN tickers
        valid = s.notna() & r_oc.notna()
        if valid.sum() < 3:
            continue

        s_valid = s[valid]
        r_valid = r_oc[valid]
        n_valid = len(s_valid)

        # Top-q and bottom-q (eq 3-4)
        n_long = max(1, int(np.floor(n_valid * q)))
        n_short = max(1, int(np.floor(n_valid * q)))

        sorted_idx = s_valid.argsort()
        long_idx = sorted_idx[-n_long:]   # Top-q
        short_idx = sorted_idx[:n_short]  # Bottom-q

        # Equal-weight portfolio (eq 5-6)
        weights = pd.Series(0.0, index=s_valid.index)
        weights.iloc[long_idx] = 1.0 / n_long
        weights.iloc[short_idx] = -1.0 / n_short

        # Strategy return (eq 7)
        ret = (weights * r_valid).sum()
        strategy_returns.append(ret)
        valid_dates.append(ret_date)

    return pd.Series(strategy_returns, index=valid_dates, name="strategy_return")


def construct_double_sort_portfolio(
    pca_signals: pd.DataFrame,
    mom_signals: pd.DataFrame,
    jp_oc_returns: pd.DataFrame,
) -> pd.Series:
    """DOUBLE strategy: 2×2 sort on MOM × PCA_SUB signals (Section 4.3).

    Median split on each signal, then:
    - High×High = long
    - Low×Low = short
    """
    common_dates = pca_signals.index.intersection(mom_signals.index).intersection(jp_oc_returns.index)
    pca_aligned = pca_signals.loc[common_dates]
    mom_aligned = mom_signals.loc[common_dates]

    return_dates = common_dates[1:]
    signal_dates = common_dates[:-1]

    strategy_returns = []
    valid_dates = []

    for sig_date, ret_date in zip(signal_dates, return_dates):
        s_pca = pca_aligned.loc[sig_date]
        s_mom = mom_aligned.loc[sig_date]
        r_oc = jp_oc_returns.loc[ret_date]

        valid = s_pca.notna() & s_mom.notna() & r_oc.notna()
        if valid.sum() < 4:
            continue

        s_pca_v = s_pca[valid]
        s_mom_v = s_mom[valid]
        r_valid = r_oc[valid]

        # Median split
        pca_high = s_pca_v >= s_pca_v.median()
        pca_low = s_pca_v < s_pca_v.median()
        mom_high = s_mom_v >= s_mom_v.median()
        mom_low = s_mom_v < s_mom_v.median()

        # High×High = long, Low×Low = short
        long_mask = pca_high & mom_high
        short_mask = pca_low & mom_low

        n_long = long_mask.sum()
        n_short = short_mask.sum()

        if n_long == 0 or n_short == 0:
            continue

        weights = pd.Series(0.0, index=r_valid.index)
        weights[long_mask] = 1.0 / n_long
        weights[short_mask] = -1.0 / n_short

        ret = (weights * r_valid).sum()
        strategy_returns.append(ret)
        valid_dates.append(ret_date)

    return pd.Series(strategy_returns, index=valid_dates, name="strategy_return")


def run_all_strategies(
    pca_sub_signals: pd.DataFrame,
    pca_plain_signals: pd.DataFrame,
    mom_signals: pd.DataFrame,
    jp_oc_returns: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Run all four strategies and return their daily returns."""
    print("Running PCA_SUB strategy...")
    pca_sub_ret = construct_long_short_portfolio(pca_sub_signals, jp_oc_returns)

    print("Running PCA_PLAIN strategy...")
    pca_plain_ret = construct_long_short_portfolio(pca_plain_signals, jp_oc_returns)

    print("Running MOM strategy...")
    mom_ret = construct_long_short_portfolio(mom_signals, jp_oc_returns)

    print("Running DOUBLE strategy...")
    double_ret = construct_double_sort_portfolio(pca_sub_signals, mom_signals, jp_oc_returns)

    return {
        "MOM": mom_ret,
        "PCA_PLAIN": pca_plain_ret,
        "PCA_SUB": pca_sub_ret,
        "DOUBLE": double_ret,
    }
