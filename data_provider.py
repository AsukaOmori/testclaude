"""Unified data-fetch orchestrator.

Wires together: provider chain → cache → timezone normalisation → validation.
Exposes a single entry point `fetch_ohlc_multi` that is a drop-in replacement
for the old `yf.download` loops in data.py and stock_data.py.
"""

from __future__ import annotations

import pandas as pd

from providers.base import DataUnavailableError, RateLimitExceededError, ValidationWarning
from providers.registry import get_provider_chain
from data_cache import load_cached, save_to_cache, CacheMissError
from data_validation import validate_ohlc
from provider_config import DATA_SOURCE


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------

_EXCHANGE_TZ: dict[str, str] = {
    ".T": "Asia/Tokyo",
    ".KS": "Asia/Seoul",
}
_DEFAULT_TZ = "America/New_York"


def _get_exchange_tz(ticker: str) -> str:
    upper = ticker.upper()
    for suffix, tz in _EXCHANGE_TZ.items():
        if upper.endswith(suffix):
            return tz
    return _DEFAULT_TZ


def normalize_timezone(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise DatetimeIndex to tz-naive midnight dates."""
    if df.index.tz is not None:
        exchange_tz = _get_exchange_tz(ticker)
        df.index = df.index.tz_convert(exchange_tz).normalize().tz_localize(None)
    else:
        df.index = df.index.normalize()
    # Remove duplicate dates (possible after DST transitions)
    df = df[~df.index.duplicated(keep="first")]
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fetch_ohlc_multi(
    tickers: list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLC for multiple tickers using the provider chain.

    Returns dict[ticker → DataFrame with columns [Open, Close]].
    """
    result: dict[str, pd.DataFrame] = {}
    all_warnings: list[ValidationWarning] = []

    for ticker in tickers:
        df = _fetch_single(ticker, start, end, all_warnings)
        if df is not None and not df.empty:
            result[ticker] = df[["Open", "Close"]].copy()
        else:
            print(f"  Warning: No data for {ticker} from any provider")

    # Print validation summary
    if all_warnings:
        print(f"\n  Data validation: {len(all_warnings)} warning(s)")
        for w in all_warnings:
            print(f"    [{w.severity}] {w.ticker}: {w.message}")

    return result


def _fetch_single(
    ticker: str,
    start: str,
    end: str,
    all_warnings: list[ValidationWarning],
) -> pd.DataFrame | None:
    """Fetch OHLC for one ticker, trying cache then provider chain."""
    providers = get_provider_chain(ticker)

    # ---- cache-only mode ----
    if DATA_SOURCE == "cache_only":
        for provider in providers:
            cached = load_cached(provider.name, ticker, start, end)
            if cached is not None:
                return cached
        raise CacheMissError(
            f"No cached data for {ticker} (DATA_SOURCE=cache_only)"
        )

    # ---- normal / provider_only mode ----
    # Try cache first (unless provider_only)
    if DATA_SOURCE != "provider_only":
        for provider in providers:
            cached = load_cached(provider.name, ticker, start, end)
            if cached is not None:
                return cached

    # Fetch from providers
    for provider in providers:
        if not provider.supports_ticker(ticker):
            continue

        remaining = provider.rate_limit_remaining()
        if remaining is not None and remaining <= 0:
            continue

        try:
            df = provider.fetch_ohlc(ticker, start, end)
            df = normalize_timezone(df, ticker)
            df, warnings = validate_ohlc(df, ticker, end)
            all_warnings.extend(warnings)
            save_to_cache(provider.name, ticker, start, end, df)
            return df
        except (DataUnavailableError, RateLimitExceededError) as exc:
            print(f"    {provider.name} failed for {ticker}: {exc}")
            continue
        except Exception as exc:
            print(f"    {provider.name} unexpected error for {ticker}: {exc}")
            continue

    return None
