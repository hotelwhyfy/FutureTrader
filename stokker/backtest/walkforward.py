"""Walk-forward evaluation: measuring how much of a backtest was fitting.

Every number elsewhere in this project is full-sample. That is acceptable while
nothing is being tuned, and dangerous the moment anything is: with a Sharpe
standard error near 0.3, trying ten parameter values buys roughly +0.3 Sharpe
from noise alone. This module exists so that any future "I found a better
lookback" arrives with a number attached showing whether it survived blind.

The method: cut time into rolling folds, choose parameters using ONLY the
training window, apply the winner to the untouched test window, roll forward,
and stitch the test slices into one out-of-sample curve.

    fold 1  [=== train ===]|[test]
    fold 2       [=== train ===]|[test]
    fold 3            [=== train ===]|[test]
                                      concatenate test slices -> honest curve

Three numbers come out, and the third is the one that matters:

  in-sample       best full-sample parameter -- what a naive backtest claims
  out-of-sample   walk-forward selection     -- what actually survived
  fixed baseline  never-tuned default        -- the control

If out-of-sample lands below fixed baseline, parameter selection is actively
destroying value and the correct action is to stop tuning. That is a common and
genuinely useful result, not a failure of the harness.

WHY NO EMBARGO BY DEFAULT. Purge/embargo gaps matter when labels overlap their
features -- predicting a 5-day forward return means the label at t consumes data
through t+5. Here the label is the next day's return, so there is no overlap,
every signal is causal (rolling and ewm windows only look backwards), and the
split is strictly chronological. An embargo is therefore not required for
correctness. The parameter is offered for experimentation and defaults to zero
rather than being cargo-culted in.

WHAT THIS DOES NOT DO. It does not narrow confidence intervals. Test windows are
shorter than the full sample, so the out-of-sample Sharpe carries a WIDER
interval than the full-sample estimate. Nothing but more years fixes that.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from stokker.backtest import engine
from stokker.backtest.costs import DEFAULT, CostModel
from stokker.config import Instrument
from stokker.signals.base import Signal

log = logging.getLogger(__name__)
TRADING_DAYS = 252


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    sd = r.std()
    return float((r.mean() * TRADING_DAYS) / (sd * np.sqrt(TRADING_DAYS))) if sd else np.nan


@dataclass(frozen=True)
class FoldSpec:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self) -> str:
        return (f"fold{self.index}: train {self.train_start.date()}..{self.train_end.date()} "
                f"test {self.test_start.date()}..{self.test_end.date()}")


@dataclass
class FoldResult:
    spec: FoldSpec
    params: dict[str, Any]
    train_sharpe: float
    test_sharpe: float
    candidates: dict[tuple, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Out-of-sample result plus the two comparisons that give it meaning."""

    label: str
    folds: list[FoldResult]
    oos_returns: pd.Series
    in_sample_params: dict[str, Any]
    in_sample_sharpe: float
    fixed_params: dict[str, Any]
    fixed_oos_returns: pd.Series

    # ---------------------------------------------------------------- views
    @property
    def oos_sharpe(self) -> float:
        return _sharpe(self.oos_returns)

    @property
    def fixed_sharpe(self) -> float:
        return _sharpe(self.fixed_oos_returns)

    @property
    def overfit_gap(self) -> float:
        """In-sample minus out-of-sample: the overfitting tax, in Sharpe units."""
        return self.in_sample_sharpe - self.oos_sharpe

    @property
    def selection_edge(self) -> float:
        """Walk-forward selection minus never tuning. Negative means stop tuning."""
        return self.oos_sharpe - self.fixed_sharpe

    def equity(self) -> pd.Series:
        return (1.0 + self.oos_returns.fillna(0.0)).cumprod()

    def stability(self) -> pd.DataFrame:
        """How often the selected parameters changed across folds.

        A stable optimum is evidence of signal. Selections that jump around
        every fold mean the objective surface is noise and the 'best' parameter
        is whatever the last few years happened to reward.
        """
        rows = [{"fold": f.spec.index, **f.params,
                 "train_sharpe": f.train_sharpe, "test_sharpe": f.test_sharpe}
                for f in self.folds]
        return pd.DataFrame(rows).set_index("fold")

    def summary(self) -> pd.Series:
        sel = self.stability()
        pcols = [c for c in sel.columns if c not in ("train_sharpe", "test_sharpe")]
        changes = sum(int(sel[c].nunique() > 1) for c in pcols)
        oos = self.oos_returns.dropna()
        eq = (1.0 + oos).cumprod()
        dd = (eq / eq.cummax() - 1.0).min() if len(eq) else np.nan
        return pd.Series({
            "folds": len(self.folds),
            "oos_years": len(oos) / TRADING_DAYS,
            "in_sample_sharpe": self.in_sample_sharpe,
            "oos_sharpe": self.oos_sharpe,
            "overfit_gap": self.overfit_gap,
            "fixed_sharpe": self.fixed_sharpe,
            "selection_edge": self.selection_edge,
            "oos_max_drawdown": dd,
            "params_that_varied": changes,
            "distinct_selections": len(sel[pcols].drop_duplicates()) if pcols else 1,
        }, name=self.label)


class WalkForward:
    """Rolling-origin evaluation with train-only parameter selection."""

    def __init__(
        self,
        train_years: float = 5.0,
        test_years: float = 1.0,
        step_years: float | None = None,
        embargo_days: int = 0,
        anchored: bool = False,
        objective: Callable[[pd.Series], float] = _sharpe,
    ) -> None:
        self.train_years = train_years
        self.test_years = test_years
        self.step_years = step_years if step_years is not None else test_years
        self.embargo_days = embargo_days
        self.anchored = anchored
        self.objective = objective

    # ------------------------------------------------------------- folds
    def folds(self, index: pd.DatetimeIndex) -> list[FoldSpec]:
        """Chronological folds over a date index.

        `anchored=True` grows the training window from a fixed origin (more data
        each fold); the default rolls a fixed-length window (adapts to regime
        change, discards old data).
        """
        index = pd.DatetimeIndex(index).sort_values()
        if len(index) == 0:
            return []
        origin, last = index[0], index[-1]
        train_td = pd.Timedelta(days=int(self.train_years * 365.25))
        test_td = pd.Timedelta(days=int(self.test_years * 365.25))
        step_td = pd.Timedelta(days=int(self.step_years * 365.25))
        embargo = pd.Timedelta(days=self.embargo_days)

        out: list[FoldSpec] = []
        train_end = origin + train_td
        while True:
            test_start = train_end + embargo
            test_end = min(test_start + test_td, last)
            if test_start >= last:
                break
            train_start = origin if self.anchored else max(origin, train_end - train_td)
            # Require the test window to hold enough bars to score at all.
            if (index >= test_start).sum() < 20:
                break
            out.append(FoldSpec(len(out) + 1, train_start, train_end, test_start, test_end))
            if test_end >= last:
                break
            train_end = train_end + step_td
        return out

    # ------------------------------------------------------------- runner
    def run(
        self,
        dataset,
        signal_cls: type[Signal],
        grid: dict[str, Sequence[Any]],
        costs: CostModel = DEFAULT,
        target_vol: float = 0.15,
        fixed: dict[str, Any] | None = None,
    ) -> WalkForwardResult:
        """Walk one instrument forward.

        `grid` maps constructor kwargs to candidate values. `fixed` is the
        never-tuned control (defaults to the signal class's own defaults).
        """
        inst: Instrument = dataset.inst
        rets, price = dataset.returns, dataset.price
        specs = self.folds(rets.index)
        if not specs:
            raise ValueError(
                f"{inst.symbol}: history too short for "
                f"{self.train_years}y train + {self.test_years}y test"
            )

        combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
        if not combos:
            raise ValueError("empty parameter grid")

        # Signals are causal, so generating over the full series and slicing
        # afterwards is safe: the value at date t consumes only data <= t. The
        # guard that matters is that SELECTION below reads the train slice only.
        cached: dict[tuple, pd.Series] = {}
        for c in combos:
            key = tuple(sorted(c.items()))
            cached[key] = signal_cls(**c).generate(dataset.bars, rets, inst, dataset.cot)

        # Backtest each candidate ONCE over the full history, then slice per fold.
        # This is both ~16x faster and more correct: slicing first would restart
        # vol-targeting cold at every fold boundary, whereas in live trading the
        # estimator carries its warm-up across the boundary. Slicing after is
        # also consistent with how the signals themselves are generated.
        full: dict[tuple, pd.Series] = {
            key: engine.run(inst.symbol, sig, rets, price, inst,
                            costs=costs, target_vol=target_vol).returns
            for key, sig in cached.items()
        }

        def bt(key: tuple, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
            r = full[key]
            out = r[(r.index >= lo) & (r.index <= hi)]
            return out if len(out) >= 20 else pd.Series(dtype=float)

        results: list[FoldResult] = []
        oos_parts: list[pd.Series] = []
        for spec in specs:
            scores: dict[tuple, float] = {}
            for c in combos:
                key = tuple(sorted(c.items()))
                scores[key] = self.objective(bt(key, spec.train_start, spec.train_end))

            usable = {k: v for k, v in scores.items() if np.isfinite(v)}
            if not usable:
                log.warning("%s %s: no scorable candidate, skipping", inst.symbol, spec)
                continue
            # Ties break toward the first grid entry, which keeps selection
            # deterministic instead of depending on dict ordering luck.
            best_key = max(usable, key=lambda k: usable[k])
            test_ret = bt(best_key, spec.test_start, spec.test_end)
            oos_parts.append(test_ret)
            results.append(FoldResult(spec, dict(best_key), usable[best_key],
                                      _sharpe(test_ret), scores))

        if not oos_parts:
            raise RuntimeError(f"{inst.symbol}: no fold produced out-of-sample returns")

        oos = pd.concat(oos_parts).sort_index()
        oos = oos[~oos.index.duplicated(keep="first")]

        # In-sample reference: best parameter chosen with full hindsight, scored
        # over the same span the OOS curve covers, so the gap is like-for-like.
        span = (oos.index.min(), oos.index.max())
        is_scores = {k: self.objective(bt(k, *span)) for k in full}
        is_scores = {k: v for k, v in is_scores.items() if np.isfinite(v)}
        is_key = max(is_scores, key=lambda k: is_scores[k]) if is_scores else None

        fixed = fixed or {}
        fixed_sig = signal_cls(**fixed).generate(dataset.bars, rets, inst, dataset.cot)
        fixed_full = engine.run(inst.symbol, fixed_sig, rets, price, inst,
                                costs=costs, target_vol=target_vol).returns
        fixed_ret = fixed_full[(fixed_full.index >= span[0]) & (fixed_full.index <= span[1])]

        return WalkForwardResult(
            label=inst.symbol,
            folds=results,
            oos_returns=oos,
            in_sample_params=dict(is_key) if is_key else {},
            in_sample_sharpe=is_scores[is_key] if is_key else np.nan,
            fixed_params=fixed,
            fixed_oos_returns=fixed_ret,
        )

    # -------------------------------------------------- pooled / portfolio
    def run_pooled(
        self,
        datasets: dict[str, Any],
        signal_cls: type[Signal],
        grid: dict[str, Sequence[Any]],
        costs: CostModel = DEFAULT,
        target_vol: float = 0.15,
        fixed: dict[str, Any] | None = None,
    ) -> WalkForwardResult:
        """Select ONE parameter set per fold across the whole universe.

        Preferred over per-instrument selection for a trend system: it spends a
        single degree of freedom instead of 43, which is both more robust and
        closer to how such a book is actually run.
        """
        if not datasets:
            raise ValueError("no datasets supplied")

        combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
        idx = sorted(set().union(*(d.returns.index for d in datasets.values())))
        specs = self.folds(pd.DatetimeIndex(idx))
        if not specs:
            raise ValueError("history too short for the requested fold geometry")

        cached: dict[str, dict[tuple, pd.Series]] = {}
        for sym, ds in datasets.items():
            cached[sym] = {
                tuple(sorted(c.items())): signal_cls(**c).generate(
                    ds.bars, ds.returns, ds.inst, ds.cot)
                for c in combos
            }

        # Equal weights are constant, so combining then slicing is identical to
        # slicing then combining -- which lets the whole portfolio curve be built
        # once per candidate instead of once per candidate per fold.
        keys = [tuple(sorted(c.items())) for c in combos]
        pooled: dict[tuple, pd.Series] = {}
        for key in keys:
            per = {sym: engine.run(sym, cached[sym][key], ds.returns, ds.price,
                                   ds.inst, costs=costs, target_vol=target_vol)
                   for sym, ds in datasets.items()}
            pooled[key] = engine.combine(per).returns

        def port(key: tuple, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
            r = pooled[key]
            out = r[(r.index >= lo) & (r.index <= hi)]
            return out if len(out) >= 20 else pd.Series(dtype=float)

        results, oos_parts = [], []
        for spec in specs:
            scores = {k: self.objective(port(k, spec.train_start, spec.train_end))
                      for k in keys}
            usable = {k: v for k, v in scores.items() if np.isfinite(v)}
            if not usable:
                continue
            best = max(usable, key=lambda k: usable[k])
            test_ret = port(best, spec.test_start, spec.test_end)
            oos_parts.append(test_ret)
            results.append(FoldResult(spec, dict(best), usable[best],
                                      _sharpe(test_ret), scores))

        if not oos_parts:
            raise RuntimeError("no fold produced out-of-sample returns")
        oos = pd.concat(oos_parts).sort_index()
        oos = oos[~oos.index.duplicated(keep="first")]

        span = (oos.index.min(), oos.index.max())
        is_scores = {k: self.objective(port(k, *span)) for k in keys}
        is_scores = {k: v for k, v in is_scores.items() if np.isfinite(v)}
        is_key = max(is_scores, key=lambda k: is_scores[k]) if is_scores else None

        fixed = fixed or {}
        fixed_key = tuple(sorted(fixed.items()))
        if fixed_key not in pooled:
            per = {}
            for sym, ds in datasets.items():
                sig = signal_cls(**fixed).generate(ds.bars, ds.returns, ds.inst, ds.cot)
                per[sym] = engine.run(sym, sig, ds.returns, ds.price, ds.inst,
                                      costs=costs, target_vol=target_vol)
            pooled[fixed_key] = engine.combine(per).returns

        return WalkForwardResult(
            label="PORTFOLIO",
            folds=results,
            oos_returns=oos,
            in_sample_params=dict(is_key) if is_key else {},
            in_sample_sharpe=is_scores[is_key] if is_key else np.nan,
            fixed_params=fixed,
            fixed_oos_returns=port(fixed_key, *span),
        )


# ---------------------------------------------------------------------------
# Universe bootstrap
# ---------------------------------------------------------------------------
# Walk-forward guards against overfitting in TIME. It cannot see overfitting in
# UNIVERSE: every fold trades the same instrument list, so a signal that only
# works on the particular markets you happened to pick sails straight through.
#
# This project ran into exactly that. `CotExtreme` scored -0.153 on the original
# 15 markets and +0.429 on the expanded 43 -- a swing of ~0.58 Sharpe caused by
# nothing but which markets were in the list. Resampling the universe is the
# complement that catches it.


@dataclass
class UniverseBootstrapResult:
    label: str
    sharpes: np.ndarray
    subset_size: int
    per_instrument: pd.Series

    def summary(self) -> pd.Series:
        a = self.sharpes[np.isfinite(self.sharpes)]
        return pd.Series({
            "draws": len(a),
            "subset_size": self.subset_size,
            "mean": a.mean(),
            "sd": a.std(),
            "q2.5": np.quantile(a, 0.025),
            "q97.5": np.quantile(a, 0.975),
            "p_negative": (a < 0).mean(),
            "range": a.max() - a.min(),
            "per_instrument_mean": self.per_instrument.mean(),
            "per_instrument_sd": self.per_instrument.std(),
            "pct_instruments_positive": (self.per_instrument > 0).mean(),
        }, name=self.label)

    def percentile_of(self, subset: Sequence[str], results: dict) -> float:
        """Where a specific instrument list falls in the resampled distribution.

        Use this to check whether a universe you chose is representative or a
        tail draw. The original 15-market list sits below the 2nd percentile for
        COT, which is why it read as 'refuted'.
        """
        s = _sharpe(engine.combine({k: results[k] for k in subset}).returns)
        a = self.sharpes[np.isfinite(self.sharpes)]
        return float((a < s).mean())


def universe_bootstrap(
    results: dict,
    subset_size: int = 15,
    draws: int = 1000,
    seed: int = 0,
    label: str = "universe",
) -> UniverseBootstrapResult:
    """Resample which instruments are traded, holding the signal fixed.

    `results` maps symbol -> BacktestResult (as produced by research.backtest).
    Returns the distribution of portfolio Sharpe across random subsets, which is
    the honest error bar on 'does this signal work' when the universe itself was
    a judgement call.
    """
    syms = sorted(results)
    if len(syms) <= subset_size:
        raise ValueError(
            f"need more than {subset_size} instruments to resample; got {len(syms)}"
        )
    rng = np.random.default_rng(seed)
    out = np.empty(draws, dtype=float)
    for i in range(draws):
        pick = rng.choice(syms, subset_size, replace=False)
        out[i] = _sharpe(engine.combine({s: results[s] for s in pick}).returns)
    per = pd.Series({s: _sharpe(r.returns) for s, r in results.items()})
    return UniverseBootstrapResult(label, out, subset_size, per)
