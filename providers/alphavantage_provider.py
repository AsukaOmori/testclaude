"""Alpha Vantage data provider — requires API key.

Free tier: 25 requests/day.  The provider tracks daily usage in a JSON file
so that remaining calls survive across process restarts.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from providers.base import DataUnavailableError, RateLimitExceededError

_DAILY_LIMIT = 25
_BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageProvider:
    """Fetches adjusted daily OHLC from Alpha Vantage REST API."""

    name: str = "alphavantage"

    def __init__(self, api_key: str | None = None, cache_dir: str = "./data_cache"):
        self._api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        self._counter_path = Path(cache_dir) / "av_calls.json"

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def supports_ticker(self, ticker: str) -> bool:
        """Only enabled when an API key is configured; skips non-US tickers."""
        if not self._api_key:
            return False
        # Skip Japanese and Korean tickers
        upper = ticker.upper()
        if upper.endswith(".T") or upper.endswith(".KS"):
            return False
        return True

    def fetch_ohlc(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        if not self._api_key:
            raise DataUnavailableError("Alpha Vantage API key not configured")

        remaining = self.rate_limit_remaining()
        if remaining is not None and remaining <= 0:
            raise RateLimitExceededError("Alpha Vantage daily limit reached")

        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": self._api_key,
        }

        try:
            resp = requests.get(_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise DataUnavailableError(
                f"Alpha Vantage request failed for {ticker}: {exc}"
            ) from exc

        ts_key = "Time Series (Daily)"
        if ts_key not in payload:
            error_msg = payload.get("Note") or payload.get("Error Message") or str(payload)
            raise DataUnavailableError(
                f"Alpha Vantage bad response for {ticker}: {error_msg}"
            )

        self._increment_counter()

        raw = payload[ts_key]
        records = []
        for date_str, values in raw.items():
            records.append({
                "Date": pd.Timestamp(date_str),
                "Open": float(values["1. open"]),
                "High": float(values["2. high"]),
                "Low": float(values["3. low"]),
                "Close": float(values["5. adjusted close"]),
            })

        df = pd.DataFrame(records).set_index("Date").sort_index()
        df = df.loc[start:end]

        if df.empty:
            raise DataUnavailableError(
                f"Alpha Vantage returned no data for {ticker} in [{start}, {end}]"
            )

        return df[["Open", "High", "Low", "Close"]].astype("float64")

    def rate_limit_remaining(self) -> int | None:
        if not self._api_key:
            return 0
        today = date.today().isoformat()
        counter = self._load_counter()
        used = counter.get(today, 0)
        return max(0, _DAILY_LIMIT - used)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_counter(self) -> dict:
        if self._counter_path.exists():
            try:
                return json.loads(self._counter_path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _increment_counter(self) -> None:
        today = date.today().isoformat()
        counter = self._load_counter()
        counter[today] = counter.get(today, 0) + 1
        self._counter_path.parent.mkdir(parents=True, exist_ok=True)
        self._counter_path.write_text(json.dumps(counter))
