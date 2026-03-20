"""Provider registry — resolves ticker → ordered provider chain."""

from __future__ import annotations

from providers.base import DataProvider
from providers.yfinance_provider import YFinanceProvider
from providers.stooq_provider import StooqProvider
from providers.alphavantage_provider import AlphaVantageProvider

# Singleton instances (created lazily)
_instances: dict[str, DataProvider] = {}


def _get_instance(name: str) -> DataProvider | None:
    """Return (or create) a singleton provider instance by name."""
    if name in _instances:
        return _instances[name]

    cls_map: dict[str, type] = {
        "yfinance": YFinanceProvider,
        "stooq": StooqProvider,
        "alphavantage": AlphaVantageProvider,
    }

    cls = cls_map.get(name)
    if cls is None:
        return None

    _instances[name] = cls()
    return _instances[name]


def _detect_market(ticker: str) -> str:
    """Return market code based on ticker suffix."""
    upper = ticker.upper()
    if upper.endswith(".T"):
        return "jp"
    if upper.endswith(".KS"):
        return "kr"
    return "us"


def get_provider_chain(ticker: str) -> list[DataProvider]:
    """Return an ordered list of providers to try for *ticker*.

    The chain is determined by:
      1. Market detection from the ticker suffix
      2. PROVIDER_PRIORITY_* environment variables (via provider_config)
      3. yfinance is always the last fallback
    """
    from provider_config import (
        PROVIDER_PRIORITY_JP,
        PROVIDER_PRIORITY_KR,
        PROVIDER_PRIORITY_US,
    )

    market = _detect_market(ticker)
    if market == "jp":
        priority = PROVIDER_PRIORITY_JP
    elif market == "kr":
        priority = PROVIDER_PRIORITY_KR
    else:
        priority = PROVIDER_PRIORITY_US

    chain: list[DataProvider] = []
    seen: set[str] = set()

    for name in priority:
        name = name.strip()
        if name in seen:
            continue
        provider = _get_instance(name)
        if provider is not None and provider.supports_ticker(ticker):
            chain.append(provider)
            seen.add(name)

    # Always ensure yfinance is at the end as fallback
    if "yfinance" not in seen:
        yf_provider = _get_instance("yfinance")
        if yf_provider is not None:
            chain.append(yf_provider)

    return chain
