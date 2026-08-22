"""Time-series momentum.

The reason this is the baseline and not an invention: cross-asset time-series
momentum in futures is one of the few effects with decades of out-of-sample
evidence behind it (Moskowitz, Ooi & Pedersen 2012, and the existence of the
managed-futures industry generally).  Starting from a documented effect and
trying to break it is a better use of time than inventing an indicator and
trying to confirm it.

Two formulations, because they fail differently:
  * `TimeSeriesMomentum` -- sign of the trailing N-day return.  Blunt, few
    parameters, hard to overfit.
  * `EWMACrossover`      -- normalised fast/slow EWMA spread.  Smoother, takes
    partial positions, but has more knobs to torture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stokker.signals.base import Signal

TRADING_DAYS = 252


class TimeSeriesMomentum(Signal):
    """Long if the trailing `lookback` return is positive, short if negative."""

    name = "ts_momentum"

    def __init__(self, lookback: int = 252, smooth: int = 5, binary: bool = True) -> None:
        self.lookback = lookback
        self.smooth = smooth
        self.binary = binary

    def generate(self, bars, returns, inst, cot=None) -> pd.Series:
        # Cumulative log return over the window; roll gaps already neutralised.
        trailing = returns.rolling(self.lookback, min_periods=self.lookback // 2).sum()

        if self.binary:
            raw = np.sign(trailing)
        else:
            # Scale by trailing vol so the magnitude means something comparable
            # across instruments, then squash.
            vol = returns.rolling(self.lookback, min_periods=20).std() * np.sqrt(self.lookback)
            raw = np.tanh(trailing / vol.replace(0.0, np.nan))

        sig = pd.Series(raw, index=returns.index)
        if self.smooth > 1:
            sig = sig.rolling(self.smooth, min_periods=1).mean()
        return sig.fillna(0.0).clip(-1.0, 1.0).rename(self.name)


class EWMACrossover(Signal):
    """Fast-minus-slow EWMA of price, normalised by trailing vol."""

    name = "ewma_crossover"

    def __init__(self, fast: int = 32, slow: int = 128, scale: float = 2.0) -> None:
        if fast >= slow:
            raise ValueError("fast span must be shorter than slow span")
        self.fast = fast
        self.slow = slow
        self.scale = scale

    def generate(self, bars, returns, inst, cot=None) -> pd.Series:
        # Use the return-implied price path rather than raw close, so roll gaps
        # neutralised upstream stay neutralised here.
        synthetic = returns.fillna(0.0).cumsum()
        spread = synthetic.ewm(span=self.fast).mean() - synthetic.ewm(span=self.slow).mean()
        vol = returns.rolling(self.slow, min_periods=20).std() * np.sqrt(self.slow)
        z = spread / vol.replace(0.0, np.nan)
        return np.tanh(z * self.scale).fillna(0.0).clip(-1.0, 1.0).rename(self.name)


class VolRegimeFilter(Signal):
    """Not a signal on its own -- a multiplier that de-risks in high-vol regimes.

    Blend it with a directional signal rather than trading it alone.
    """

    name = "vol_filter"

    def __init__(self, lookback: int = 60, percentile: float = 0.9) -> None:
        self.lookback = lookback
        self.percentile = percentile

    def generate(self, bars, returns, inst, cot=None) -> pd.Series:
        vol = returns.rolling(self.lookback, min_periods=20).std()
        rank = vol.rolling(TRADING_DAYS * 2, min_periods=TRADING_DAYS // 2).rank(pct=True)
        # Full size below the threshold, tapering to zero at the extreme.
        damp = ((1.0 - rank) / (1.0 - self.percentile)).clip(0.0, 1.0)
        return damp.fillna(1.0).rename(self.name)
