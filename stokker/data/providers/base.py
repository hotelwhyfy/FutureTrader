"""Price provider interface.

Everything downstream depends on this shape, not on yfinance.  Swapping in a
paid feed (Databento, IBKR) later means writing one new class here, not
rewriting the research code.
"""

from __future__ import annotations

import logging
from typing import Protocol

import pandas as pd

log = logging.getLogger(__name__)

#: Canonical bar schema every provider must return.
#: DatetimeIndex named "date", tz-naive, sorted ascending, no duplicate days.
BAR_COLUMNS = ("open", "high", "low", "close", "volume")


class PriceProvider(Protocol):
    name: str

    def daily_bars(
        self, symbol: str, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Daily OHLCV for one instrument, indexed by date."""
        ...


def validate_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Enforce the canonical schema.  Raises rather than silently coercing."""
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: bars missing columns {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{symbol}: bars must have a DatetimeIndex")

    out = df.loc[:, list(BAR_COLUMNS)].copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["close"])

    # Non-positive futures prices are rare but REAL -- WTI settled at -$37.63 on
    # 2020-04-20.  Rejecting them would delete history.  Flag loudly instead, and
    # let `contracts.roll` handle the fact that log returns are undefined there.
    bad = out.index[out["close"] <= 0]
    if len(bad):
        log.warning(
            "%s: %d non-positive close(s) e.g. %s -- real for physical commodities; "
            "returns across these bars are neutralised, not computed",
            symbol, len(bad), bad[0].date(),
        )
    return out
