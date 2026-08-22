"""Orchestration: load data, build returns, run a signal, backtest it.

This is the layer the CLI and the dashboard both call, so that neither one
contains research logic of its own.
"""

from __future__ import annotations

import logging

import pandas as pd

from stokker.backtest import engine
from stokker.backtest.costs import DEFAULT, CostModel
from stokker.config import Instrument, get, symbols
from stokker.contracts import roll
from stokker.data.providers import cot as cot_data
from stokker.data.providers.yahoo import YahooProvider
from stokker.signals.base import Signal

log = logging.getLogger(__name__)


class Dataset:
    """Everything needed to evaluate one instrument."""

    def __init__(self, inst: Instrument, bars: pd.DataFrame, returns: pd.Series,
                 cot: pd.DataFrame | None) -> None:
        self.inst = inst
        self.bars = bars
        self.returns = returns
        self.cot = cot

    @property
    def price(self) -> pd.Series:
        return self.bars["close"].astype(float)

    def __repr__(self) -> str:
        span = f"{self.bars.index.min().date()}..{self.bars.index.max().date()}"
        cot_n = 0 if self.cot is None else len(self.cot)
        return f"<Dataset {self.inst.symbol} bars={len(self.bars)} {span} cot_weeks={cot_n}>"


def load(symbol: str, start: str = "2006-01-01", with_cot: bool = True,
         refresh: bool = False) -> Dataset:
    inst = get(symbol)
    bars = YahooProvider(start=start).fetch_cached(inst.symbol, refresh=refresh)
    bars = bars.loc[bars.index >= pd.Timestamp(start)]
    returns = roll.safe_returns(bars, inst)

    cot = None
    if with_cot:
        years = range(pd.Timestamp(start).year, pd.Timestamp.today().year + 1)
        try:
            cot = cot_data.for_instrument(inst.cot_code, inst.cot_report, years)
        except Exception as exc:  # noqa: BLE001 - COT is an overlay, not a requirement
            log.warning("%s: COT unavailable (%s)", symbol, exc)

    return Dataset(inst, bars, returns, cot)


def backtest(dataset: Dataset, signal: Signal, costs: CostModel = DEFAULT,
             target_vol: float = 0.15) -> engine.BacktestResult:
    sig = signal.generate(dataset.bars, dataset.returns, dataset.inst, dataset.cot)
    return engine.run(
        symbol=dataset.inst.symbol,
        signal=sig,
        returns=dataset.returns,
        price=dataset.price,
        inst=dataset.inst,
        costs=costs,
        target_vol=target_vol,
    )


def sweep(signal: Signal, universe: list[str] | None = None, start: str = "2006-01-01",
          costs: CostModel = DEFAULT) -> tuple[pd.DataFrame, dict]:
    """Run one signal across many instruments.  Returns (stats table, results)."""
    universe = universe or symbols()
    results: dict[str, engine.BacktestResult] = {}
    for sym in universe:
        try:
            ds = load(sym, start=start)
            results[sym] = backtest(ds, signal, costs=costs)
        except Exception as exc:  # noqa: BLE001 - one bad symbol shouldn't kill a sweep
            log.warning("%s: skipped (%s)", sym, exc)

    if not results:
        raise RuntimeError("no instruments produced results")

    portfolio = engine.combine(results)
    table = pd.DataFrame({s: r.summary() for s, r in results.items()}).T
    table.loc["PORTFOLIO"] = portfolio.summary()
    return table, {**results, "PORTFOLIO": portfolio}


def current_state(dataset: Dataset, signal: Signal) -> dict:
    """What the signal says right now -- the 'is it time to move' view.

    Reports the latest exposure plus how stale the inputs are, because a signal
    built on a COT release from nine days ago should say so.
    """
    sig = signal.generate(dataset.bars, dataset.returns, dataset.inst, dataset.cot)
    last_bar = dataset.bars.index[-1]
    latest = float(sig.iloc[-1])
    prev = float(sig.iloc[-2]) if len(sig) > 1 else 0.0

    state = {
        "symbol": dataset.inst.symbol,
        "name": dataset.inst.name,
        "signal": signal.name,
        "exposure": latest,
        "direction": "LONG" if latest > 0.05 else "SHORT" if latest < -0.05 else "FLAT",
        "changed": abs(latest - prev) > 0.05,
        "last_close": float(dataset.price.iloc[-1]),
        "last_bar": last_bar.date().isoformat(),
        "bar_age_days": (pd.Timestamp.today().normalize() - last_bar).days,
    }
    if dataset.cot is not None and not dataset.cot.empty:
        released = pd.Timestamp(dataset.cot["release_date"].iloc[-1])
        state["cot_report_date"] = dataset.cot.index[-1].date().isoformat()
        state["cot_age_days"] = (pd.Timestamp.today().normalize() - released).days
    return state
