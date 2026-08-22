"""The tests that matter most: proving the engine cannot see the future.

If these ever fail, every backtest number the project has produced is void.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stokker.backtest import engine
from stokker.backtest.costs import CostModel
from stokker.config import get
from stokker.data.providers import cot as cot_data

ES = get("ES")
FREE = CostModel(slippage_ticks=0.0, commission_mult=0.0)


def _series(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    rets = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    price = pd.Series(2000 * np.exp(rets.cumsum()), index=idx)
    return rets, price


def test_engine_delay_is_exactly_one_day():
    """Pin the execution delay to one day, from both sides.

    An oracle that sees today's return must NOT capture it (the engine lags it
    away).  Pre-shifting that same oracle one day forward must capture it
    exactly -- which proves the delay is one day, not zero and not two.
    """
    rets, price = _series()
    oracle = np.sign(rets)                    # knows ret[T] on day T
    prescient = oracle.shift(-1).fillna(0.0)  # hand the engine tomorrow's answer

    lagged = engine.run("ES", oracle, rets, price, ES, costs=FREE, size_by_vol=False)
    cheating = engine.run("ES", prescient, rets, price, ES, costs=FREE, size_by_vol=False)

    # Bar 0 has no prior bar to have acted on, so it is excluded from both sides.
    theoretical_max = rets.abs().iloc[1:].sum()

    # The honest run must fall far short of perfect foresight.
    assert lagged.gross_returns.iloc[1:].sum() < 0.25 * theoretical_max, (
        "oracle signal captured same-day returns: the execution delay is missing"
    )
    # The deliberately-cheating run reconstructs it almost exactly, confirming
    # the engine lags by one day and no more.
    assert np.isclose(cheating.gross_returns.iloc[1:].sum(), theoretical_max, rtol=1e-9), (
        "engine delay is not exactly one bar"
    )


def test_position_is_strictly_lagged():
    rets, price = _series(200)
    signal = pd.Series(0.0, index=rets.index)
    signal.iloc[50] = 1.0  # a single day of exposure

    res = engine.run("ES", signal, rets, price, ES, costs=FREE, size_by_vol=False)
    assert res.position.iloc[50] == 0.0, "position moved on the same bar as the signal"
    assert res.position.iloc[51] == 1.0, "position did not appear on the next bar"


def test_future_returns_cannot_leak_through_vol_sizing():
    """Vol-target sizing uses trailing vol shifted by one day."""
    rets, _ = _series(300)
    size = engine.vol_target_size(rets, target_vol=0.15, lookback=20)
    # Inject a huge move late; sizing on that same day must be unchanged.
    spiked = rets.copy()
    spiked.iloc[250] = 0.25
    size2 = engine.vol_target_size(spiked, target_vol=0.15, lookback=20)
    assert np.isclose(size.iloc[250], size2.iloc[250]), "sizing reacted to a same-day move"


def test_cot_alignment_uses_release_not_report_date():
    """Tuesday's positioning must not be visible before Friday's release."""
    report_dates = pd.to_datetime(["2024-01-02", "2024-01-09"])  # Tuesdays
    cot = pd.DataFrame(
        {"spec_net": [100.0, 999.0], "release_date": report_dates + pd.Timedelta(days=3)},
        index=report_dates,
    )
    daily = pd.bdate_range("2024-01-02", "2024-01-16")
    aligned = cot_data.as_of_frame(cot, daily, ["spec_net"])

    wed_after_first_report = pd.Timestamp("2024-01-03")
    assert pd.isna(aligned.loc[wed_after_first_report, "spec_net"]), (
        "Tuesday's report leaked into Wednesday, before it was published"
    )
    friday = pd.Timestamp("2024-01-05")
    assert aligned.loc[friday, "spec_net"] == 100.0, "release-day value missing"

    # The 999 report (Tue Jan 9) is released Fri Jan 12 -- Jan 11 must still see 100.
    assert aligned.loc[pd.Timestamp("2024-01-11"), "spec_net"] == 100.0
    assert aligned.loc[pd.Timestamp("2024-01-12"), "spec_net"] == 999.0


def test_costs_reduce_returns():
    rets, price = _series(400)
    signal = pd.Series(np.sign(np.sin(np.arange(400) / 3)), index=rets.index)  # churns

    free = engine.run("ES", signal, rets, price, ES, costs=FREE, size_by_vol=False)
    paid = engine.run("ES", signal, rets, price, ES,
                      costs=CostModel(slippage_ticks=2.0), size_by_vol=False)

    assert paid.returns.sum() < free.returns.sum()
    assert paid.stats["cost_drag_ann"] > 0
