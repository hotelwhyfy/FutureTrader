"""Portfolio risk math.

Risk decomposition has exact algebraic identities. If these break, every number
the risk layer reports is wrong in a way that reads as plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stokker.risk import covariance as cov_mod
from stokker.risk import stress
from stokker.config import get
from stokker.risk.portfolio import Position, analyse, parse_positions, size_to_vol_target

EQUITY = 100_000.0


def _returns(symbols, n=800, seed=0, rho=0.0, vol=0.01):
    """Correlated gaussian returns with a controllable pairwise correlation."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    common = rng.normal(0, 1, n)
    out = {}
    for i, s in enumerate(symbols):
        idio = rng.normal(0, 1, n)
        z = np.sqrt(rho) * common + np.sqrt(max(1 - rho, 0.0)) * idio
        out[s] = pd.Series(z * vol, index=idx)
    return out


def _prices(symbols, value=100.0):
    return {s: value for s in symbols}


def _equal_notional(symbols, prices, target=10_000.0):
    """Equal dollar exposure per leg.

    Equal CONTRACT counts are not equal risk: ES is $50/point and CL is
    $1000/point, so one lot each is a 20x difference in exposure.
    """
    return [Position(s, target / (prices[s] * get(s).point_value)) for s in symbols]


# ------------------------------------------------------------------ parsing
def test_parse_positions_handles_signs_and_spacing():
    p = parse_positions(" ES:2 , CL:-1.5,GC:0.25 ")
    assert [x.symbol for x in p] == ["ES", "CL", "GC"]
    assert [x.contracts for x in p] == [2.0, -1.5, 0.25]


@pytest.mark.parametrize("bad", ["ES", "ES:", "ES:two", ""])
def test_parse_positions_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_positions(bad)


# -------------------------------------------------------------- identities
def test_risk_contributions_sum_to_portfolio_vol():
    """The defining property of marginal risk decomposition."""
    syms = ["ES", "NQ", "CL", "GC"]
    r = analyse([Position(s, 1.0) for s in syms], _prices(syms),
                _returns(syms, rho=0.3), EQUITY)
    assert np.isclose(r.positions["risk_contribution"].sum(), r.portfolio_vol, rtol=1e-9)
    assert np.isclose(r.positions["risk_contribution_pct"].sum(), 1.0, rtol=1e-9)


def test_effective_bets_equals_n_for_uncorrelated_equal_risk():
    syms = ["ES", "CL", "GC", "ZW"]
    px = _prices(syms)
    r = analyse(_equal_notional(syms, px), px, _returns(syms, rho=0.0, seed=4), EQUITY)
    assert 3.5 < r.effective_bets <= len(syms) + 1e-9, (
        f"uncorrelated equal-risk book should hold ~{len(syms)} bets, got {r.effective_bets:.2f}"
    )


def test_effective_bets_collapses_when_positions_are_redundant():
    """Four near-identical positions are not four bets."""
    syms = ["ES", "NQ", "YM", "RTY"]
    px = _prices(syms)
    r = analyse(_equal_notional(syms, px), px, _returns(syms, rho=0.97, seed=7), EQUITY)
    assert r.effective_bets < 2.0, (
        f"highly correlated book reported {r.effective_bets:.2f} effective bets"
    )
    assert any("effective bets" in w for w in r.concentration_warnings())


def test_effective_bets_ignores_correlation_but_concentration_does_not():
    """The two metrics answer different questions and must not be conflated."""
    syms = ["ES", "NQ", "YM", "RTY"]
    px = _prices(syms)
    r = analyse(_equal_notional(syms, px), px, _returns(syms, rho=0.97, seed=7), EQUITY)
    assert r.effective_bets < 2.0, "correlated book holds ~1 independent bet"
    assert r.risk_concentration > 3.0, (
        "risk is still spread evenly across four line items — that is what "
        "concentration measures, and why it cannot stand in for independence"
    )


def test_correlated_book_is_riskier_than_diversified_one():
    syms = ["ES", "NQ", "CL", "GC"]
    pos = [Position(s, 1.0) for s in syms]
    lo = analyse(pos, _prices(syms), _returns(syms, rho=0.0, seed=1), EQUITY)
    hi = analyse(pos, _prices(syms), _returns(syms, rho=0.9, seed=1), EQUITY)
    assert hi.portfolio_vol > lo.portfolio_vol
    assert hi.diversification_ratio < lo.diversification_ratio


def test_shorts_reduce_risk_against_a_correlated_long():
    syms = ["ES", "NQ"]
    rets = _returns(syms, rho=0.95, seed=2)
    both_long = analyse([Position("ES", 1), Position("NQ", 1)], _prices(syms), rets, EQUITY)
    hedged = analyse([Position("ES", 1), Position("NQ", -1)], _prices(syms), rets, EQUITY)
    assert hedged.portfolio_vol < both_long.portfolio_vol, (
        "shorting a highly correlated instrument must lower portfolio risk"
    )


def test_leverage_accounting():
    syms = ["ES", "CL"]
    r = analyse([Position("ES", 2), Position("CL", -1)], _prices(syms),
                _returns(syms), EQUITY)
    # ES: 2 * 100 * 50 = 10_000 ; CL: -1 * 100 * 1000 = -100_000
    assert np.isclose(r.gross_leverage, 110_000 / EQUITY)
    assert np.isclose(r.net_leverage, -90_000 / EQUITY)


def test_var_ordering_and_sign():
    syms = ["ES", "CL", "GC"]
    r = analyse([Position(s, 1.0) for s in syms], _prices(syms), _returns(syms), EQUITY)
    assert r.var_99 >= r.var_95 > 0, "VaR is reported as a positive loss, 99% >= 95%"
    assert r.cvar_95 >= r.var_95, "expected shortfall cannot be smaller than VaR"
    assert r.worst_day < 0 and r.worst_week <= 0


# --------------------------------------------------------------- covariance
def test_shrinkage_pulls_correlations_together_without_moving_vols():
    syms = ["ES", "NQ", "CL", "GC"]
    raw = cov_mod.ewma_cov(pd.DataFrame(_returns(syms, rho=0.5)))
    shrunk = cov_mod.shrink_correlation(raw, 0.5)
    assert np.allclose(np.diag(raw), np.diag(shrunk)), "shrinkage must not alter variances"

    def offdiag_spread(c):
        sd = np.sqrt(np.diag(c.to_numpy()))
        corr = c.to_numpy() / np.outer(sd, sd)
        return corr[~np.eye(len(sd), dtype=bool)].std()

    assert offdiag_spread(shrunk) < offdiag_spread(raw)


def test_full_shrinkage_makes_all_correlations_equal():
    syms = ["ES", "NQ", "CL"]
    c = cov_mod.shrink_correlation(cov_mod.ewma_cov(pd.DataFrame(_returns(syms, rho=0.4))), 1.0)
    sd = np.sqrt(np.diag(c.to_numpy()))
    corr = c.to_numpy() / np.outer(sd, sd)
    off = corr[~np.eye(len(sd), dtype=bool)]
    assert np.allclose(off, off[0])


def test_align_drops_short_history_rather_than_zero_filling():
    syms = ["ES", "CL"]
    rets = _returns(syms)
    rets["GC"] = rets["ES"].tail(10)  # too short
    frame, dropped = cov_mod.align(rets, ["ES", "CL", "GC"])
    assert dropped == ["GC"] and "GC" not in frame.columns


# ------------------------------------------------------------------ sizing
def test_size_to_vol_target_hits_the_target():
    syms = ["ES", "CL", "GC"]
    r = analyse([Position(s, 1.0) for s in syms], _prices(syms), _returns(syms), EQUITY)
    scaled = size_to_vol_target(r, target_vol=0.10)
    k = scaled["scaled"] / scaled["contracts"]
    assert np.allclose(k, k.iloc[0]), "every leg must scale by the same factor"
    assert np.isclose(r.portfolio_vol * k.iloc[0], 0.10)


def test_size_to_vol_target_flags_untradable_fractions():
    syms = ["ES", "CL"]
    r = analyse([Position(s, 1.0) for s in syms], _prices(syms), _returns(syms), EQUITY)
    scaled = size_to_vol_target(r, target_vol=0.0001)  # forces sizes below one lot
    assert scaled["untradable"].any()


# ------------------------------------------------------------------ stress
def test_stress_replay_reports_coverage_not_silent_gaps():
    syms = ["ES", "CL"]
    rets = _returns(syms, n=800)  # starts 2018, so GFC is uncovered
    r = analyse([Position(s, 1.0) for s in syms], _prices(syms), rets, EQUITY)
    out = stress.replay(r, rets)
    assert out.loc["gfc", "coverage"] == 0.0
    assert np.isnan(out.loc["gfc", "pnl_pct"])
    assert out.loc["covid", "coverage"] > 0 or np.isnan(out.loc["covid", "pnl_pct"])


def test_worst_windows_are_ordered_losses():
    syms = ["ES", "CL"]
    rets = _returns(syms)
    r = analyse([Position(s, 1.0) for s in syms], _prices(syms), rets, EQUITY)
    w = stress.worst_windows(r, rets, window=21, top=5)
    assert len(w) == 5
    assert list(w["loss_pct"]) == sorted(w["loss_pct"]), "worst first"
    assert w["loss_pct"].iloc[0] < 0


# ------------------------------------------------------------------ guards
def test_rejects_duplicate_and_zero_equity():
    syms = ["ES"]
    with pytest.raises(ValueError, match="duplicate"):
        analyse([Position("ES", 1), Position("ES", 2)], _prices(syms),
                _returns(syms), EQUITY)
    with pytest.raises(ValueError, match="equity"):
        analyse([Position("ES", 1)], _prices(syms), _returns(syms), 0.0)
