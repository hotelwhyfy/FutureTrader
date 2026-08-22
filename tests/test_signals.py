"""Signals must respect the contract: bounded, aligned, no NaNs leaking out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stokker.config import get
from stokker.signals.base import Blend, Constant
from stokker.signals.cot import CommercialFlow, CotExtreme
from stokker.signals.momentum import EWMACrossover, TimeSeriesMomentum, VolRegimeFilter

ES = get("ES")


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-01", periods=900)
    rets = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    close = 2000 * np.exp(rets.cumsum())
    bars = pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1.0}, index=idx)
    return bars, rets


@pytest.fixture
def cot_frame():
    weeks = pd.bdate_range("2015-01-01", periods=250, freq="W-TUE")
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "spec_net": rng.normal(50_000, 20_000, len(weeks)),
            "commercial_net": rng.normal(-50_000, 20_000, len(weeks)),
            "open_interest": np.full(len(weeks), 2_000_000.0),
            "release_date": weeks + pd.Timedelta(days=3),
        },
        index=weeks,
    )


ALL = [TimeSeriesMomentum(), TimeSeriesMomentum(binary=False), EWMACrossover(),
       VolRegimeFilter(), Constant(), CotExtreme(), CommercialFlow()]


@pytest.mark.parametrize("sig", ALL, ids=lambda s: type(s).__name__ + repr(vars(s)))
def test_signal_contract(sig, data, cot_frame):
    bars, rets = data
    out = sig.generate(bars, rets, ES, cot_frame)

    assert isinstance(out, pd.Series)
    assert out.index.equals(rets.index), f"{sig.name}: index misaligned with returns"
    assert not out.isna().any(), f"{sig.name}: emitted NaN"
    assert out.abs().max() <= 1.0 + 1e-9, f"{sig.name}: exposure outside [-1, 1]"


@pytest.mark.parametrize("sig", [CotExtreme(), CommercialFlow()])
def test_cot_signals_degrade_to_flat_without_data(sig, data):
    bars, rets = data
    out = sig.generate(bars, rets, ES, None)
    assert (out == 0.0).all(), "COT signal should be flat when COT is unavailable"


def test_blend_stays_bounded(data, cot_frame):
    bars, rets = data
    blend = Blend((TimeSeriesMomentum(), 0.6), (CotExtreme(), 0.4))
    out = blend.generate(bars, rets, ES, cot_frame)
    assert out.abs().max() <= 1.0 + 1e-9


def test_momentum_follows_a_trend(data):
    """A relentlessly rising series should end up long."""
    idx = pd.bdate_range("2015-01-01", periods=600)
    rets = pd.Series(0.001, index=idx)
    close = 2000 * np.exp(rets.cumsum())
    bars = pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1.0}, index=idx)
    out = TimeSeriesMomentum().generate(bars, rets, ES, None)
    assert out.iloc[-1] > 0.9, "momentum failed to go long a pure uptrend"
