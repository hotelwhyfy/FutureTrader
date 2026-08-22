"""Vectorised backtester for continuous futures positions.

Design decisions that matter, stated plainly:

  * Positions are *continuous* (e.g. 0.37 contracts), because volatility
    targeting is how futures are actually sized.  Integer-contract rounding is
    a real constraint for small accounts and is reported, not simulated.

  * A signal computed from data through day T is traded at day T+1's close and
    earns day T+2's return.  The `signal.shift(1)` in `run()` is the entire
    defence against lookahead bias.  Do not remove it.

  * Costs are charged on every change in position size, not just on flips.

  * Returns are log returns from `contracts.roll.safe_returns`, so roll gaps
    are already neutralised before they reach here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stokker.backtest.costs import DEFAULT, CostModel
from stokker.config import Instrument

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    symbol: str
    equity: pd.Series          # cumulative growth of 1.0, net of costs
    returns: pd.Series         # daily net returns
    gross_returns: pd.Series
    position: pd.Series        # contracts held (continuous)
    costs: pd.Series           # daily cost drag in return terms
    stats: dict[str, float] = field(default_factory=dict)

    def summary(self) -> pd.Series:
        return pd.Series(self.stats, name=self.symbol)


def _stats(net: pd.Series, gross: pd.Series, pos: pd.Series, costs: pd.Series) -> dict[str, float]:
    net = net.dropna()
    if net.empty:
        return {}
    years = len(net) / TRADING_DAYS
    equity = (1.0 + net).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0

    vol = net.std() * np.sqrt(TRADING_DAYS)
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else np.nan
    downside = net[net < 0].std() * np.sqrt(TRADING_DAYS)

    return {
        "years": round(years, 2),
        "cagr": cagr,
        "vol": vol,
        "sharpe": (net.mean() * TRADING_DAYS) / vol if vol else np.nan,
        "sortino": (net.mean() * TRADING_DAYS) / downside if downside else np.nan,
        "max_drawdown": dd.min(),
        "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "hit_rate": (net > 0).mean(),
        "turnover_ann": pos.diff().abs().sum() / years if years > 0 else np.nan,
        "cost_drag_ann": costs.sum() / years if years > 0 else np.nan,
        "gross_sharpe": (
            (gross.mean() * TRADING_DAYS) / (gross.std() * np.sqrt(TRADING_DAYS))
            if gross.std() else np.nan
        ),
        "time_in_market": (pos.abs() > 1e-9).mean(),
    }


def vol_target_size(
    returns: pd.Series, target_vol: float = 0.15, lookback: int = 60, cap: float = 3.0
) -> pd.Series:
    """Scale exposure so each instrument contributes similar risk.

    Uses trailing realised vol, shifted by one day so today's sizing cannot see
    today's move.
    """
    realised = returns.rolling(lookback, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    scalar = (target_vol / realised.replace(0.0, np.nan)).shift(1)
    return scalar.clip(upper=cap).fillna(0.0)


def run(
    symbol: str,
    signal: pd.Series,
    returns: pd.Series,
    price: pd.Series,
    inst: Instrument,
    costs: CostModel = DEFAULT,
    target_vol: float = 0.15,
    size_by_vol: bool = True,
) -> BacktestResult:
    """Backtest one instrument.

    `signal` is a desired exposure in [-1, 1] indexed like `returns`, computed
    from information available *on* each date.  The engine handles the one-day
    execution delay itself.
    """
    idx = returns.index
    signal = signal.reindex(idx).fillna(0.0).clip(-1.0, 1.0)

    sizing = vol_target_size(returns, target_vol) if size_by_vol else pd.Series(1.0, index=idx)
    desired = signal * sizing

    # THE lookahead guard: act on yesterday's decision.
    position = desired.shift(1).fillna(0.0)

    gross = position * returns
    turn = position.diff().abs().fillna(position.abs())
    cost_rate = costs.cost_in_return_terms(inst, price.reindex(idx))
    drag = (turn * cost_rate).fillna(0.0)
    net = gross - drag

    equity = (1.0 + net.fillna(0.0)).cumprod()
    return BacktestResult(
        symbol=symbol,
        equity=equity,
        returns=net,
        gross_returns=gross,
        position=position,
        costs=drag,
        stats=_stats(net, gross, position, drag),
    )


def combine(results: dict[str, BacktestResult], weights: dict[str, float] | None = None) -> BacktestResult:
    """Equal-risk portfolio across instruments (equal-weight unless told otherwise)."""
    if not results:
        raise ValueError("nothing to combine")
    frame = pd.DataFrame({s: r.returns for s, r in results.items()}).fillna(0.0)
    gross = pd.DataFrame({s: r.gross_returns for s, r in results.items()}).fillna(0.0)
    cost = pd.DataFrame({s: r.costs for s, r in results.items()}).fillna(0.0)
    pos = pd.DataFrame({s: r.position for s, r in results.items()}).fillna(0.0)

    w = pd.Series(weights or {s: 1.0 / len(results) for s in results})
    w = w / w.sum()

    net = frame.mul(w, axis=1).sum(axis=1)
    return BacktestResult(
        symbol="PORTFOLIO",
        equity=(1.0 + net).cumprod(),
        returns=net,
        gross_returns=gross.mul(w, axis=1).sum(axis=1),
        position=pos.mul(w, axis=1).sum(axis=1),
        costs=cost.mul(w, axis=1).sum(axis=1),
        stats=_stats(
            net,
            gross.mul(w, axis=1).sum(axis=1),
            pos.mul(w, axis=1).sum(axis=1),
            cost.mul(w, axis=1).sum(axis=1),
        ),
    )
