"""OHLC data-quality checks.

Each check returns a list of ValidationWarning.  The orchestrator
(`data_provider.py`) calls `validate_ohlc` after every provider fetch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from providers.base import ValidationWarning


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_ohlc_consistency(df: pd.DataFrame, ticker: str) -> list[ValidationWarning]:
    """Flag rows where Close < Low or Close > High."""
    warnings: list[ValidationWarning] = []
    if not {"Close", "Low", "High"}.issubset(df.columns):
        return warnings

    below_low = df["Close"] < df["Low"]
    above_high = df["Close"] > df["High"]
    bad = below_low | above_high

    if bad.any():
        n_bad = int(bad.sum())
        first_date = str(df.index[bad][0].date()) if hasattr(df.index[bad][0], "date") else str(df.index[bad][0])
        warnings.append(ValidationWarning(
            ticker=ticker,
            date=first_date,
            check="ohlc_consistency",
            message=f"{n_bad} rows with Close outside [Low, High] (first: {first_date})",
            severity="warn",
        ))
    return warnings


def check_nan_spikes(
    df: pd.DataFrame, ticker: str, threshold: float = 0.1,
) -> list[ValidationWarning]:
    """Flag if >threshold fraction of any column is NaN in a 20-day window."""
    warnings: list[ValidationWarning] = []
    window = 20
    if len(df) < window:
        return warnings

    nan_frac = df.isna().rolling(window=window).mean()
    max_frac = nan_frac.max().max()
    if max_frac > threshold:
        col = nan_frac.max().idxmax()
        warnings.append(ValidationWarning(
            ticker=ticker,
            date=None,
            check="nan_spike",
            message=f"NaN spike detected: {max_frac:.1%} in column '{col}' (threshold {threshold:.0%})",
            severity="warn",
        ))
    return warnings


def check_extreme_returns(
    df: pd.DataFrame, ticker: str, max_daily_return: float = 0.5,
) -> list[ValidationWarning]:
    """Flag single-day moves exceeding ±max_daily_return."""
    warnings: list[ValidationWarning] = []
    if "Close" not in df.columns or len(df) < 2:
        return warnings

    daily_ret = df["Close"].pct_change().abs()
    extreme = daily_ret > max_daily_return
    if extreme.any():
        n = int(extreme.sum())
        first_date = str(daily_ret[extreme].index[0].date()) if hasattr(daily_ret[extreme].index[0], "date") else str(daily_ret[extreme].index[0])
        warnings.append(ValidationWarning(
            ticker=ticker,
            date=first_date,
            check="extreme_return",
            message=f"{n} days with |return| > {max_daily_return:.0%} (first: {first_date})",
            severity="warn",
        ))
    return warnings


def check_stale_data(
    df: pd.DataFrame, ticker: str, max_gap_days: int = 5,
) -> list[ValidationWarning]:
    """Flag if Close repeats for >max_gap_days consecutive business days."""
    warnings: list[ValidationWarning] = []
    if "Close" not in df.columns or len(df) < max_gap_days + 1:
        return warnings

    close = df["Close"]
    same_as_prev = (close == close.shift(1))
    # Count consecutive True runs
    groups = (~same_as_prev).cumsum()
    run_lengths = same_as_prev.groupby(groups).sum()
    max_run = run_lengths.max()

    if max_run >= max_gap_days:
        warnings.append(ValidationWarning(
            ticker=ticker,
            date=None,
            check="stale_data",
            message=f"Close price unchanged for {int(max_run)} consecutive days (threshold {max_gap_days})",
            severity="warn",
        ))
    return warnings


def check_early_end(
    df: pd.DataFrame, ticker: str, end: str,
) -> list[ValidationWarning]:
    """Warn if data ends significantly before the requested end date."""
    warnings: list[ValidationWarning] = []
    if df.empty:
        return warnings

    data_end = pd.Timestamp(df.index[-1])
    requested_end = pd.Timestamp(end)
    gap = (requested_end - data_end).days

    if gap > 90:
        warnings.append(ValidationWarning(
            ticker=ticker,
            date=str(data_end.date()) if hasattr(data_end, "date") else str(data_end),
            check="early_end",
            message=f"Data ends {gap} days before requested end ({data_end.date()} vs {requested_end.date()}). Possible delisting.",
            severity="warn",
        ))
    return warnings


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def validate_ohlc(
    df: pd.DataFrame, ticker: str, end: str = "",
) -> tuple[pd.DataFrame, list[ValidationWarning]]:
    """Run all validation checks and return (df, warnings)."""
    warnings: list[ValidationWarning] = []
    warnings.extend(check_ohlc_consistency(df, ticker))
    warnings.extend(check_nan_spikes(df, ticker))
    warnings.extend(check_extreme_returns(df, ticker))
    warnings.extend(check_stale_data(df, ticker))
    if end:
        warnings.extend(check_early_end(df, ticker, end))
    return df, warnings
