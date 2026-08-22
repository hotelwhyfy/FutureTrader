"""Roll handling.

These tests encode the lesson from the rewrite: rolls come from the CONTRACT
CALENDAR, never from price outliers.  `test_price_jumps_are_not_used_for_rolls`
exists specifically to stop anyone re-wiring outlier detection into the return
path -- that bug inflated ES cumulative log return by ~0.6 before it was caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stokker.config import get
from stokker.contracts import roll


def _bars(n=800, seed=1, start="2018-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = 2000 * np.exp(rng.normal(0, 0.008, n).cumsum())
    return pd.DataFrame(
        {"open": close, "high": close * 1.002, "low": close * 0.998,
         "close": close, "volume": 1000.0},
        index=idx,
    )


# --- calendar correctness -------------------------------------------------

def test_es_rolls_on_third_fridays():
    dates = roll.roll_dates(get("ES"), "2024-01-01", "2024-12-31")
    assert len(dates) == 4, "quarterly contract should roll 4x per year"
    for d in dates:
        assert d.dayofweek == 4, f"{d.date()} is not a Friday"
        assert 15 <= d.day <= 21, f"{d.date()} is not the third Friday"
        assert d.month in {3, 6, 9, 12}


def test_cl_rolls_monthly():
    dates = roll.roll_dates(get("CL"), "2024-01-01", "2024-12-31")
    assert len(dates) == 12, "WTI should roll every month"
    # WTI expiry convention puts the roll in the third week of the prior month.
    assert all(17 <= d.day <= 23 for d in dates), [str(d.date()) for d in dates]


@pytest.mark.parametrize("sym,per_year", [
    ("ES", 4), ("NQ", 4), ("ZN", 4), ("6E", 4),
    ("CL", 12), ("NG", 12), ("GC", 6), ("ZS", 7), ("ZC", 5),
])
def test_roll_cadence_matches_delivery_cycle(sym, per_year):
    dates = roll.roll_dates(get(sym), "2015-01-01", "2025-01-01")
    assert abs(len(dates) / 10 - per_year) < 0.15, (
        f"{sym}: {len(dates)/10:.1f} rolls/yr, expected {per_year}"
    )


def test_roll_flags_land_on_trading_bars():
    inst = get("ES")
    bars = _bars(n=1500)
    flags = roll.roll_flags(bars, inst)
    expected = len(roll.roll_dates(inst, bars.index.min(), bars.index.max()))
    # Every roll date snaps forward to a bar; duplicates collapse.
    assert 0 < flags.sum() <= expected
    assert flags.index.equals(bars.index)


def test_roll_flags_are_price_independent():
    """The whole point of the rewrite: flags must not move when prices do."""
    inst = get("ES")
    a = _bars(seed=1)
    b = a.copy()
    b["close"] = b["close"] * np.exp(np.linspace(0, -0.5, len(b)))  # crash the series
    assert roll.roll_flags(a, inst).equals(roll.roll_flags(b, inst))


# --- the regression that motivated the rewrite ----------------------------

def test_price_jumps_are_not_used_for_rolls():
    """A crash must not be treated as a roll.

    Historically the detector flagged the 2010 flash crash, Brexit, Volmageddon
    and the COVID selloff as 'rolls' and zeroed them, deleting the left tail.
    """
    inst = get("ES")
    bars = _bars(n=600)
    crash_i = 300
    # A -10% day, well outside any threshold, on a non-roll date.
    bars.iloc[crash_i:, bars.columns.get_loc("close")] *= 0.90

    flags = roll.roll_flags(bars, inst)
    rets = roll.safe_returns(bars, inst)

    if not flags.iloc[crash_i]:  # not coincidentally a roll bar
        assert rets.iloc[crash_i] < -0.05, "a real crash day was neutralised"
    assert roll.price_jumps(bars).iloc[crash_i], "diagnostic should still see the jump"


# --- undefined prices -----------------------------------------------------

def test_negative_prices_do_not_produce_infinite_returns():
    """WTI settled at -$37.63 on 2020-04-20; log returns are undefined there."""
    inst = get("CL")
    bars = _bars(n=300)
    bars.loc[bars.index[150], "close"] = -37.63
    bars.loc[bars.index[150], "low"] = -40.32

    rets = roll.safe_returns(bars, inst)
    assert np.isfinite(rets).all(), "non-finite return leaked from a negative price"
    assert rets.iloc[150] == 0.0
    assert rets.iloc[151] == 0.0


def test_invalid_price_flags_both_sides():
    bars = _bars(n=50)
    bars.loc[bars.index[20], "close"] = -1.0
    flags = roll.invalid_price_bars(bars)
    assert flags.iloc[20] and flags.iloc[21]
    assert not flags.iloc[19] and not flags.iloc[22]


# --- adjustment -----------------------------------------------------------

def test_back_adjust_preserves_latest_price():
    inst = get("ES")
    bars = _bars()
    adj = roll.back_adjust(bars, inst)
    assert np.isclose(adj["close"].iloc[-1], bars["close"].iloc[-1])


def test_safe_returns_zeroes_roll_bars():
    inst = get("ES")
    bars = _bars(n=1000)
    flags = roll.roll_flags(bars, inst)
    rets = roll.safe_returns(bars, inst)
    assert (rets[flags] == 0.0).all(), "roll bars were not neutralised"
    assert (rets[~flags] != 0.0).any(), "everything was neutralised"
