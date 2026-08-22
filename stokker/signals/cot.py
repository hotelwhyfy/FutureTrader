"""Positioning signals from CFTC Commitments of Traders.

The premise, stated so it can be tested rather than assumed: commercial hedgers
hold offsetting physical exposure and tend to lean against price moves, while
managed money tends to lean into them.  When speculative positioning reaches a
crowded extreme, the marginal buyer is exhausted and moves are prone to revert.

That is a hypothesis, not a fact, and it is weaker in financial futures than in
physical commodities -- TFF "dealers" are intermediaries, not hedgers with a
crop in the ground, so the commercial/spec contrast means less for ES than for
corn.  Check per-sector results before believing a pooled number.

Every signal here reads COT via `cot.as_of_frame`, which aligns on the Friday
RELEASE date, not the Tuesday report date.  Positioning you could not have
known is positioning you cannot trade.
"""

from __future__ import annotations

import pandas as pd

from stokker.data.providers import cot as cot_data
from stokker.signals.base import Signal

TRADING_DAYS = 252


def _aligned(cot: pd.DataFrame | None, index: pd.Index, columns: list[str]) -> pd.DataFrame | None:
    if cot is None or cot.empty:
        return None
    return cot_data.as_of_frame(cot, pd.DatetimeIndex(index), columns)


class CotExtreme(Signal):
    """Fade crowded speculative positioning.

    Speculative net position is normalised as a z-score against its own
    trailing history, then inverted: heavily long spec -> negative exposure.
    Below `deadband` z the signal is flat, so this only speaks at extremes.
    """

    name = "cot_extreme"

    def __init__(self, lookback_weeks: int = 156, deadband: float = 1.0, cap: float = 2.5) -> None:
        self.lookback_weeks = lookback_weeks
        self.deadband = deadband
        self.cap = cap

    def generate(self, bars, returns, inst, cot=None) -> pd.Series:
        if cot is None or cot.empty:
            return pd.Series(0.0, index=returns.index, name=self.name)

        spec = cot["spec_net"].astype(float)
        oi = cot["open_interest"].astype(float).replace(0.0, pd.NA)
        # Normalise by open interest so the series is comparable as contract
        # sizes and market participation change over two decades.
        frac = (spec / oi).astype(float)

        mean = frac.rolling(self.lookback_weeks, min_periods=52).mean()
        std = frac.rolling(self.lookback_weeks, min_periods=52).std()
        z = ((frac - mean) / std.replace(0.0, pd.NA)).clip(-self.cap, self.cap)

        weekly = pd.DataFrame({"z": z}, index=cot.index)
        weekly["release_date"] = cot["release_date"].to_numpy()
        aligned = _aligned(weekly.set_index(weekly.index), pd.DatetimeIndex(returns.index), ["z"])
        if aligned is None:
            return pd.Series(0.0, index=returns.index, name=self.name)

        zz = aligned["z"].astype(float)
        # Apply the deadband, then rescale the surviving range to [-1, 1].
        excess = zz.abs() - self.deadband
        magnitude = (excess.clip(lower=0.0) / (self.cap - self.deadband)).clip(0.0, 1.0)
        # Inverted: crowded longs -> short.
        return (-zz.apply(lambda v: 1.0 if v > 0 else -1.0) * magnitude).fillna(0.0).rename(self.name)


class CommercialFlow(Signal):
    """Follow the change in commercial net positioning.

    Rather than a level extreme, this reads the *direction* commercials are
    moving over recent weeks -- treating hedgers as informed about their own
    market. Distinct enough from `CotExtreme` to be worth testing separately.
    """

    name = "commercial_flow"

    def __init__(self, weeks: int = 8, lookback_weeks: int = 156) -> None:
        self.weeks = weeks
        self.lookback_weeks = lookback_weeks

    def generate(self, bars, returns, inst, cot=None) -> pd.Series:
        if cot is None or cot.empty:
            return pd.Series(0.0, index=returns.index, name=self.name)

        comm = cot["commercial_net"].astype(float)
        oi = cot["open_interest"].astype(float).replace(0.0, pd.NA)
        flow = (comm / oi).astype(float).diff(self.weeks)

        std = flow.rolling(self.lookback_weeks, min_periods=52).std()
        z = (flow / std.replace(0.0, pd.NA)).clip(-2.0, 2.0) / 2.0

        weekly = pd.DataFrame({"z": z}, index=cot.index)
        weekly["release_date"] = cot["release_date"].to_numpy()
        aligned = _aligned(weekly, pd.DatetimeIndex(returns.index), ["z"])
        if aligned is None:
            return pd.Series(0.0, index=returns.index, name=self.name)
        return aligned["z"].astype(float).fillna(0.0).clip(-1.0, 1.0).rename(self.name)
