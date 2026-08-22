"""Walk-forward correctness.

The harness exists to detect overfitting, so its own leakage guarantees have to
be tested harder than the thing it measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stokker.backtest.costs import CostModel
from stokker.backtest.walkforward import WalkForward, _sharpe
from stokker.config import get
from stokker.signals.momentum import TimeSeriesMomentum

ES = get("ES")
FREE = CostModel(slippage_ticks=0.0, commission_mult=0.0)


class FakeDataset:
    def __init__(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2008-01-01", periods=n)
        self.returns = pd.Series(rng.normal(0.0003, 0.01, n), index=idx, name="r")
        close = 2000 * np.exp(self.returns.cumsum())
        self.bars = pd.DataFrame({"open": close, "high": close, "low": close,
                                  "close": close, "volume": 1.0}, index=idx)
        self.cot = None
        self.inst = ES

    @property
    def price(self):
        return self.bars["close"].astype(float)


# ------------------------------------------------------------------- folds
def test_folds_never_overlap_and_move_forward():
    wf = WalkForward(train_years=3, test_years=1)
    idx = pd.bdate_range("2010-01-01", periods=3000)
    folds = wf.folds(idx)
    assert len(folds) >= 3
    for f in folds:
        assert f.train_start < f.train_end <= f.test_start < f.test_end, f"malformed {f}"
    for a, b in zip(folds, folds[1:]):
        assert b.test_start > a.test_start, "folds must advance"
        assert a.test_end <= b.test_end


def test_test_windows_are_disjoint():
    wf = WalkForward(train_years=3, test_years=1, step_years=1)
    folds = wf.folds(pd.bdate_range("2010-01-01", periods=3000))
    for a, b in zip(folds, folds[1:]):
        assert a.test_end <= b.test_start + pd.Timedelta(days=1), (
            "overlapping test windows would double-count out-of-sample returns"
        )


def test_embargo_creates_the_requested_gap():
    idx = pd.bdate_range("2010-01-01", periods=3000)
    plain = WalkForward(train_years=3, test_years=1, embargo_days=0).folds(idx)
    gapped = WalkForward(train_years=3, test_years=1, embargo_days=30).folds(idx)
    assert (gapped[0].test_start - gapped[0].train_end).days == 30
    assert (plain[0].test_start - plain[0].train_end).days == 0


def test_anchored_grows_training_window():
    idx = pd.bdate_range("2010-01-01", periods=3000)
    folds = WalkForward(train_years=3, test_years=1, anchored=True).folds(idx)
    assert all(f.train_start == folds[0].train_start for f in folds)
    assert folds[-1].train_end > folds[0].train_end


# ---------------------------------------------------------------- leakage
def test_selection_uses_only_training_data():
    """The parameter chosen for a fold must not depend on that fold's test data.

    Mutating returns strictly after train_end must leave the selection intact.
    """
    ds = FakeDataset()
    wf = WalkForward(train_years=4, test_years=1)
    grid = {"lookback": [63, 126, 252]}

    base = wf.run(ds, TimeSeriesMomentum, grid, costs=FREE)

    poisoned = FakeDataset()
    first_test_start = base.folds[0].spec.test_start
    mask = poisoned.returns.index >= first_test_start
    rng = np.random.default_rng(99)
    poisoned.returns.loc[mask] = rng.normal(0.02, 0.04, int(mask.sum()))
    close = 2000 * np.exp(poisoned.returns.cumsum())
    poisoned.bars = pd.DataFrame({"open": close, "high": close, "low": close,
                                  "close": close, "volume": 1.0}, index=close.index)

    after = wf.run(poisoned, TimeSeriesMomentum, grid, costs=FREE)
    assert base.folds[0].params == after.folds[0].params, (
        "fold 1 selection changed when only post-train data was altered — leakage"
    )


def test_oos_returns_lie_inside_test_windows_only():
    ds = FakeDataset()
    wf = WalkForward(train_years=4, test_years=1)
    res = wf.run(ds, TimeSeriesMomentum, {"lookback": [126, 252]}, costs=FREE)

    windows = [(f.spec.test_start, f.spec.test_end) for f in res.folds]
    for ts in res.oos_returns.index:
        assert any(lo <= ts <= hi for lo, hi in windows), (
            f"{ts.date()} is outside every test window"
        )
    assert not res.oos_returns.index.has_duplicates, "a bar was counted twice"


def test_oos_starts_after_first_training_window():
    ds = FakeDataset()
    wf = WalkForward(train_years=4, test_years=1)
    res = wf.run(ds, TimeSeriesMomentum, {"lookback": [126, 252]}, costs=FREE)
    assert res.oos_returns.index.min() >= res.folds[0].spec.train_end


# --------------------------------------------------------------- reporting
def test_in_sample_beats_out_of_sample_on_noise():
    """On pure noise, hindsight selection must look better than walk-forward.

    This is the harness detecting overfitting on data with no signal at all —
    if the gap were <= 0 here, it would not be measuring anything.
    """
    ds = FakeDataset(seed=5)
    wf = WalkForward(train_years=4, test_years=1)
    res = wf.run(ds, TimeSeriesMomentum, {"lookback": [21, 63, 126, 252, 504]}, costs=FREE)
    assert res.in_sample_sharpe >= res.oos_sharpe, (
        "full-hindsight selection should not underperform walk-forward on noise"
    )
    assert np.isfinite(res.overfit_gap)


def test_summary_and_stability_shapes():
    ds = FakeDataset()
    wf = WalkForward(train_years=4, test_years=1)
    res = wf.run(ds, TimeSeriesMomentum, {"lookback": [126, 252]}, costs=FREE)

    s = res.summary()
    for k in ("folds", "oos_sharpe", "overfit_gap", "fixed_sharpe", "selection_edge"):
        assert k in s.index
    assert s["folds"] == len(res.folds)

    st = res.stability()
    assert len(st) == len(res.folds)
    assert "lookback" in st.columns


def test_selection_edge_is_oos_minus_fixed():
    ds = FakeDataset()
    wf = WalkForward(train_years=4, test_years=1)
    res = wf.run(ds, TimeSeriesMomentum, {"lookback": [126, 252]},
                 costs=FREE, fixed={"lookback": 252})
    assert np.isclose(res.selection_edge, res.oos_sharpe - res.fixed_sharpe)
    assert res.fixed_params == {"lookback": 252}


def test_short_history_raises_clearly():
    ds = FakeDataset(n=300)
    with pytest.raises(ValueError, match="too short"):
        WalkForward(train_years=5, test_years=1).run(
            ds, TimeSeriesMomentum, {"lookback": [126]}, costs=FREE)


# ------------------------------------------------------- universe bootstrap
def test_universe_bootstrap_spreads_and_reports():
    """Resampling the instrument list must expose composition sensitivity."""
    from stokker.backtest import engine
    from stokker.backtest.walkforward import universe_bootstrap

    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2012-01-01", periods=1500)

    class R:  # stand-in for BacktestResult: engine.combine reads all four
        def __init__(self, r):
            self.returns = r
            self.gross_returns = r
            self.costs = pd.Series(0.0, index=r.index)
            self.position = pd.Series(1.0, index=r.index)

    # Half the markets carry a positive drift, half negative: a subset draw
    # should therefore land anywhere across a wide range.
    results = {}
    for k in range(20):
        drift = 0.0006 if k < 10 else -0.0006
        results[f"M{k:02d}"] = R(pd.Series(rng.normal(drift, 0.01, len(idx)), index=idx))

    res = universe_bootstrap(results, subset_size=8, draws=300, seed=1)
    s = res.summary()
    assert s["draws"] == 300 and s["subset_size"] == 8
    assert s["range"] > 0.3, "resampling produced implausibly little spread"
    assert s["q2.5"] < s["q97.5"]
    assert len(res.per_instrument) == 20


def test_universe_bootstrap_rejects_undersized_universe():
    from stokker.backtest.walkforward import universe_bootstrap

    class R:
        def __init__(self, r):
            self.returns = r
            self.gross_returns = r
            self.costs = pd.Series(0.0, index=r.index)
            self.position = pd.Series(1.0, index=r.index)
    idx = pd.bdate_range("2015-01-01", periods=300)
    small = {f"M{k}": R(pd.Series(0.001, index=idx)) for k in range(5)}
    with pytest.raises(ValueError, match="more than"):
        universe_bootstrap(small, subset_size=15)
