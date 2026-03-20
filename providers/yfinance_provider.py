"""yfinance data provider — universal fallback."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from providers.base import DataUnavailableError


class YFinanceProvider:
    """Fetches OHLC data via the yfinance library."""

    name: str = "yfinance"

    def supports_ticker(self, ticker: str) -> bool:
        return True  # universal fallback

    def fetch_ohlc(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Download adjusted OHLC from Yahoo Finance."""
        try:
            df = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False,
            )
        except Exception as exc:
            raise DataUnavailableError(f"yfinance error for {ticker}: {exc}") from exc

        if df.empty:
            raise DataUnavailableError(f"yfinance returned empty data for {ticker}")

        # yfinance sometimes returns MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = {"Open", "High", "Low", "Close"}
        missing = required - set(df.columns)
        if missing:
            raise DataUnavailableError(
                f"yfinance missing columns {missing} for {ticker}"
            )

        return df[["Open", "High", "Low", "Close"]].astype("float64")

    def rate_limit_remaining(self) -> int | None:
        return None
