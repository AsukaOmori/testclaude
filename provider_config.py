"""Configuration for the multi-provider data layer.

All settings are driven by environment variables with sensible defaults.
Supports .env files via python-dotenv (optional dependency).
"""

from __future__ import annotations

import os


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


_load_dotenv()


def _env_or(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
ALPHAVANTAGE_API_KEY: str | None = os.environ.get("ALPHAVANTAGE_API_KEY")
TIINGO_API_KEY: str | None = os.environ.get("TIINGO_API_KEY")
JQUANTS_API_KEY: str | None = os.environ.get("JQUANTS_API_KEY")

# ---------------------------------------------------------------------------
# Provider priority (comma-separated names)
# ---------------------------------------------------------------------------
PROVIDER_PRIORITY_JP: list[str] = _env_or("PROVIDER_PRIORITY_JP", "stooq,yfinance").split(",")
PROVIDER_PRIORITY_KR: list[str] = _env_or("PROVIDER_PRIORITY_KR", "yfinance").split(",")
PROVIDER_PRIORITY_US: list[str] = _env_or("PROVIDER_PRIORITY_US", "alphavantage,yfinance").split(",")

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
CACHE_DIR: str = _env_or("DATA_CACHE_DIR", "./data_cache")
CACHE_MAX_AGE_HOURS: int = int(_env_or("CACHE_MAX_AGE_HOURS", "24"))
CACHE_FORCE_REFRESH: bool = _env_or("CACHE_FORCE_REFRESH", "0") == "1"
DATA_SOURCE: str = _env_or("DATA_SOURCE", "auto")  # auto | cache_only | provider_only

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
VALIDATION_STRICT: bool = _env_or("VALIDATION_STRICT", "0") == "1"
