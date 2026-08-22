"""FRED macro series -- free, official, needs a free API key.

Used for regime context (curve slope, credit spreads, implied vol), not for
signals on its own.  Absent a key the rest of the project still works; anything
depending on FRED just degrades to unavailable rather than failing the run.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import requests

from stokker.data.store import STORE

log = logging.getLogger(__name__)

_BASE = "https://api.stlouisfed.org/fred/series/observations"

#: Handy defaults.  Keys are our names, values are FRED series ids.
SERIES = {
    "vix": "VIXCLS",              # equity implied vol
    "t10y2y": "T10Y2Y",           # 10y-2y curve slope
    "dgs10": "DGS10",             # 10y treasury yield
    "hy_oas": "BAMLH0A0HYM2",     # high-yield credit spread
    "dtwex": "DTWEXBGS",          # broad dollar index
}


class FredUnavailable(RuntimeError):
    """No API key configured."""


def _api_key() -> str:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise FredUnavailable(
            "FRED_API_KEY not set. Get a free key at "
            "https://fredaccount.stlouisfed.org/apikeys and put it in .env"
        )
    return key


def series(series_id: str, start: str = "2006-01-01", refresh: bool = False) -> pd.Series:
    key_name = f"fred_{series_id}"
    if not refresh and STORE.exists("macro", key_name):
        df = STORE.get("macro", key_name)
        return df["value"]

    resp = requests.get(
        _BASE,
        params={
            "series_id": series_id,
            "api_key": _api_key(),
            "file_type": "json",
            "observation_start": start,
        },
        timeout=30,
    )
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    if not obs:
        raise RuntimeError(f"FRED returned no observations for {series_id}")

    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    # FRED encodes missing observations as "."
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    df = df.set_index("date").dropna()
    STORE.put("macro", key_name, df)
    return df["value"]


def bundle(start: str = "2006-01-01") -> pd.DataFrame:
    """All default series as one daily frame, forward-filled."""
    cols = {}
    for name, sid in SERIES.items():
        try:
            cols[name] = series(sid, start=start)
        except Exception as exc:  # noqa: BLE001 - macro is optional context
            log.warning("FRED %s (%s) unavailable: %s", name, sid, exc)
    if not cols:
        raise FredUnavailable("no FRED series could be loaded")
    return pd.DataFrame(cols).sort_index().ffill()
