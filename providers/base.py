"""Base protocol and exceptions for data providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class DataProvider(Protocol):
    """Protocol that all market-data providers must satisfy."""

    name: str

    def supports_ticker(self, ticker: str) -> bool:
        """Return True if this provider can handle *ticker*."""
        ...

    def fetch_ohlc(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch OHLC data for a single ticker.

        Must return a DataFrame with:
          - DatetimeIndex (tz-naive, normalised to midnight)
          - Columns: at minimum [Open, High, Low, Close]
          - float64 values, split/dividend adjusted

        Raises DataUnavailableError when the ticker cannot be fetched.
        """
        ...

    def rate_limit_remaining(self) -> int | None:
        """Remaining API calls, or None if unlimited."""
        ...


class DataUnavailableError(Exception):
    """Raised when a provider cannot supply data for a ticker."""


class RateLimitExceededError(Exception):
    """Raised when a provider has exhausted its rate limit."""


@dataclass
class ValidationWarning:
    """Single data-quality warning emitted during validation."""

    ticker: str
    date: str | None
    check: str
    message: str
    severity: str  # "warn" or "error"
