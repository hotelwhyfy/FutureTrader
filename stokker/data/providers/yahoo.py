"""yfinance price provider.

Prototype-grade on purpose.  It is free and it works, but it is an unofficial
scrape of Yahoo Finance: rate-limited, occasionally revised without notice, and
not licensed for redistribution.  Fine for research; replace before anything
ships to other people.  That is exactly why `PriceProvider` exists.

The `=F` tickers are *continuous front-month* series that Yahoo splices at
expiry WITHOUT back-adjusting, so roll dates show up as price gaps that are not
returns.  `stokker.contracts.roll` deals with that -- do not compute returns off
this data directly.
"""

from __future__ import annotations

import logging

import pandas as pd

from stokker.config import Instrument, get
from stokker.data.providers.base import validate_bars
from stokker.data.store import STORE

log = logging.getLogger(__name__)


class YahooProvider:
    name = "yahoo"

    def __init__(self, start: str = "2006-01-01") -> None:
        self.start = start

    def daily_bars(
        self, symbol: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        import yfinance as yf

        inst: Instrument = get(symbol)
        raw = yf.download(
            inst.yahoo,
            start=start or self.start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise RuntimeError(f"{symbol}: yfinance returned no data for {inst.yahoo}")

        # Recent yfinance returns MultiIndex columns even for a single ticker.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]

        return validate_bars(raw, symbol)

    def fetch_cached(self, symbol: str, refresh: bool = False) -> pd.DataFrame:
        key = f"{self.name}_{symbol}"
        if not refresh and STORE.exists("bars", key):
            return STORE.get("bars", key)
        bars = self.daily_bars(symbol)
        STORE.put("bars", key, bars)
        log.info("%s: %d bars %s..%s", symbol, len(bars),
                 bars.index.min().date(), bars.index.max().date())
        return bars
