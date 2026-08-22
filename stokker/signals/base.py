"""Signal interface.

A signal maps market data to a desired exposure in [-1, 1] for each date, using
only information available on or before that date.  The backtester applies the
execution delay -- signals must NOT shift for it themselves, or the delay gets
applied twice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from stokker.config import Instrument


class Signal(ABC):
    name: str = "signal"

    @abstractmethod
    def generate(
        self,
        bars: pd.DataFrame,
        returns: pd.Series,
        inst: Instrument,
        cot: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Desired exposure in [-1, 1], indexed like `returns`."""

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({params})"


class Constant(Signal):
    """Always-long benchmark.  Every strategy should be compared against this."""

    name = "buy_and_hold"

    def __init__(self, exposure: float = 1.0) -> None:
        self.exposure = exposure

    def generate(self, bars, returns, inst, cot=None) -> pd.Series:
        return pd.Series(self.exposure, index=returns.index, name=self.name)


class Blend(Signal):
    """Weighted average of several signals, re-clipped to [-1, 1]."""

    name = "blend"

    def __init__(self, *components: tuple[Signal, float]) -> None:
        self.components = components

    def generate(self, bars, returns, inst, cot=None) -> pd.Series:
        total = pd.Series(0.0, index=returns.index)
        wsum = 0.0
        for sig, weight in self.components:
            total = total.add(sig.generate(bars, returns, inst, cot).fillna(0.0) * weight, fill_value=0.0)
            wsum += weight
        return (total / wsum if wsum else total).clip(-1.0, 1.0).rename(self.name)
