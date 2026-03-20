"""CSV-based local cache for OHLC data.

Cache layout::

    {CACHE_DIR}/
      {provider}/
        {ticker}_{start}_{end}.csv
      meta.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from provider_config import CACHE_DIR, CACHE_MAX_AGE_HOURS, CACHE_FORCE_REFRESH, DATA_SOURCE

_META_FILE = "meta.json"


def _cache_root() -> Path:
    return Path(CACHE_DIR)


def _meta_path() -> Path:
    return _cache_root() / _META_FILE


def _cache_key(provider: str, ticker: str, start: str, end: str) -> str:
    safe_ticker = ticker.replace("/", "_").replace("\\", "_")
    return f"{provider}/{safe_ticker}_{start}_{end}"


def _csv_path(key: str) -> Path:
    return _cache_root() / (key + ".csv")


# ---------------------------------------------------------------------------
# Meta helpers
# ---------------------------------------------------------------------------

def _load_meta() -> dict:
    path = _meta_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_meta(meta: dict) -> None:
    path = _meta_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cached(
    provider: str, ticker: str, start: str, end: str,
) -> pd.DataFrame | None:
    """Return cached DataFrame or None on miss."""
    if CACHE_FORCE_REFRESH:
        return None
    if DATA_SOURCE == "provider_only":
        return None

    key = _cache_key(provider, ticker, start, end)
    csv = _csv_path(key)

    if not csv.exists():
        return None

    meta = _load_meta()
    entry = meta.get(key)
    if entry is None:
        return None

    # Staleness check
    fetched_at = entry.get("fetched_at", 0)
    age_hours = (time.time() - fetched_at) / 3600

    # If end date is in the past, cached data is considered permanent
    end_in_past = pd.Timestamp(end) < pd.Timestamp.now()
    if not end_in_past and age_hours > CACHE_MAX_AGE_HOURS:
        return None

    try:
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        return df
    except Exception:
        return None


def save_to_cache(
    provider: str, ticker: str, start: str, end: str, df: pd.DataFrame,
) -> None:
    """Persist a DataFrame to the CSV cache."""
    key = _cache_key(provider, ticker, start, end)
    csv = _csv_path(key)
    csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv)

    meta = _load_meta()
    meta[key] = {
        "provider": provider,
        "ticker": ticker,
        "start": start,
        "end": end,
        "fetched_at": time.time(),
        "rows": len(df),
    }
    _save_meta(meta)


class CacheMissError(Exception):
    """Raised in cache_only mode when no cached data is available."""
