"""Stooq data provider — free, no API key required.

Best suited for Japanese (.T) equities via pandas_datareader.
"""

from __future__ import annotations

import pandas as pd

from providers.base import DataUnavailableError


# Stooq uses a different suffix for Tokyo-listed stocks.
_SUFFIX_MAP = {".T": ".JP"}


def _to_stooq_ticker(ticker: str) -> str:
    """Convert a yfinance-style ticker to Stooq format."""
    for yf_suffix, stooq_suffix in _SUFFIX_MAP.items():
        if ticker.upper().endswith(yf_suffix.upper()):
            return ticker[: -len(yf_suffix)] + stooq_suffix
    return ticker


class StooqProvider:
    """Fetches OHLC data from Stooq via pandas_datareader."""

    name: str = "stooq"

    def supports_ticker(self, ticker: str) -> bool:
        """Stooq is used for Tokyo-listed (.T) stocks."""
        return ticker.upper().endswith(".T")

    def fetch_ohlc(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            from pandas_datareader import data as pdr
        except ImportError as exc:
            raise DataUnavailableError(
                "pandas_datareader is not installed — run: pip install pandas_datareader"
            ) from exc

        stooq_ticker = _to_stooq_ticker(ticker)

        try:
            df = pdr.DataReader(stooq_ticker, "stooq", start=start, end=end)
        except Exception as exc:
            raise DataUnavailableError(
                f"Stooq fetch failed for {stooq_ticker}: {exc}"
            ) from exc

        if df.empty:
            raise DataUnavailableError(
                f"Stooq returned empty data for {stooq_ticker}"
            )

        # Stooq returns data in reverse-chronological order
        df = df.sort_index()

        required = {"Open", "High", "Low", "Close"}
        missing = required - set(df.columns)
        if missing:
            raise DataUnavailableError(
                f"Stooq missing columns {missing} for {stooq_ticker}"
            )

        return df[["Open", "High", "Low", "Close"]].astype("float64")

    def rate_limit_remaining(self) -> int | None:
        return None
