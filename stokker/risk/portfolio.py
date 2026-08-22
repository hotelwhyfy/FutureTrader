"""Portfolio risk decomposition.

Answers the questions that are actually answerable from this data: what is this
book's volatility, where is the risk concentrated, and how many independent bets
does it really contain.

The headline number is `effective_bets`. Holding six positions feels diversified;
if four of them are energy contracts correlated at 0.9, the book holds closer to
two bets. That gap is invisible on a position blotter and obvious here.

Units. Futures have no purchase cost, so risk is expressed against a stated
account equity:

    notional_i = contracts_i * price_i * point_value_i
    weight_i   = notional_i / equity
    portfolio return = sum_i weight_i * return_i

A short is a negative contract count and therefore a negative weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stokker.config import Instrument, get
from stokker.risk import covariance as cov_mod

TRADING_DAYS = 252


def effective_bets(cov: np.ndarray, weights: np.ndarray, n_positions: int) -> float:
    """Number of independent bets, as the squared diversification ratio.

        DR   = sum_i |w_i| * sigma_i / sigma_portfolio
        bets = DR^2

    N uncorrelated equal-risk positions give exactly N; perfectly correlated
    positions give 1, whatever their number.

    WHY NOT PCA. The textbook alternative is Meucci's diversification
    distribution: rotate onto the principal axes and take the exponential
    entropy of their variance shares. It is correct in exact arithmetic and
    unusable here, because the eigenbasis is arbitrary precisely when
    eigenvalues are near-equal -- which is the well-diversified case this metric
    exists to identify. Four genuinely uncorrelated positions produced 2.13
    instead of 4.0 under that formula, because `eigh` happened to return the
    equally-weighted "market" direction as one axis and loaded the whole book
    onto it. The eigenvalues were fine; the basis was not.

    The diversification ratio needs no eigendecomposition, so it has no basis to
    be arbitrary about.
    """
    var = float(weights @ cov @ weights)
    if var <= 0:
        return float(n_positions)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    weighted_sa = float(np.sum(np.abs(weights) * sd))
    if weighted_sa <= 0:
        return float(n_positions)
    return float((weighted_sa ** 2) / var)


@dataclass(frozen=True)
class Position:
    symbol: str
    contracts: float          # signed: negative is short

    @property
    def instrument(self) -> Instrument:
        return get(self.symbol)

    def notional(self, price: float) -> float:
        return self.contracts * price * self.instrument.point_value


def parse_positions(spec: str) -> list[Position]:
    """Parse "ES:2,CL:-1,GC:0.5" into positions."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"malformed position {part!r}, expected SYMBOL:CONTRACTS")
        sym, _, qty = part.partition(":")
        try:
            n = float(qty)
        except ValueError:
            raise ValueError(f"{part!r}: contract count must be numeric") from None
        out.append(Position(sym.strip().upper(), n))
    if not out:
        raise ValueError("no positions parsed")
    return out


@dataclass
class RiskReport:
    equity: float
    positions: pd.DataFrame      # per-position detail
    portfolio_vol: float         # annualised, fraction of equity
    portfolio_vol_usd: float
    gross_leverage: float
    net_leverage: float
    effective_bets: float
    risk_concentration: float
    diversification_ratio: float
    var_95: float                # 1-day, fraction of equity (historical)
    var_99: float
    cvar_95: float
    worst_day: float
    worst_week: float
    worst_month: float
    dropped: list[str]
    history_days: int

    def summary(self) -> pd.Series:
        return pd.Series({
            "equity": self.equity,
            "portfolio_vol": self.portfolio_vol,
            "portfolio_vol_usd": self.portfolio_vol_usd,
            "gross_leverage": self.gross_leverage,
            "net_leverage": self.net_leverage,
            "positions": len(self.positions),
            "effective_bets": self.effective_bets,
            "risk_concentration": self.risk_concentration,
            "diversification_ratio": self.diversification_ratio,
            "var_95_1d": self.var_95,
            "var_99_1d": self.var_99,
            "cvar_95_1d": self.cvar_95,
            "worst_day": self.worst_day,
            "worst_week": self.worst_week,
            "worst_month": self.worst_month,
        }, name="risk")

    def concentration_warnings(self) -> list[str]:
        """Plain-language flags. Thresholds are conventions, not laws."""
        out = []
        p = self.positions
        if len(p) >= 2 and self.effective_bets < len(p) / 2:
            out.append(
                f"{len(p)} positions but only {self.effective_bets:.1f} effective bets — "
                f"the book is more concentrated than the blotter suggests"
            )
        top = p["risk_contribution_pct"].abs().max()
        if top > 0.5:
            sym = p["risk_contribution_pct"].abs().idxmax()
            out.append(f"{sym} carries {top:.0%} of total risk")
        sector = p.groupby("sector")["risk_contribution_pct"].sum()
        for sec, share in sector.items():
            if abs(share) > 0.6:
                out.append(f"{sec} is {share:.0%} of total risk")
        if self.gross_leverage > 10:
            out.append(f"gross leverage {self.gross_leverage:.1f}x")
        if self.diversification_ratio < 1.2 and len(p) > 2:
            out.append(
                f"diversification ratio {self.diversification_ratio:.2f} — positions are "
                f"moving together, so size is not being spread"
            )
        if self.dropped:
            out.append(f"excluded for short history: {', '.join(self.dropped)}")
        return out


def analyse(
    positions: list[Position],
    prices: dict[str, float],
    returns: dict[str, pd.Series],
    equity: float,
    halflife: int = 126,
    shrinkage: float = 0.2,
    lookback_days: int | None = 756,
) -> RiskReport:
    """Decompose a book's risk."""
    if equity <= 0:
        raise ValueError("account equity must be positive")

    syms = [p.symbol for p in positions]
    if len(set(syms)) != len(syms):
        raise ValueError("duplicate symbols in position list")

    frame, dropped = cov_mod.align(returns, syms)
    kept = [p for p in positions if p.symbol in frame.columns]
    if not kept:
        raise ValueError("no position had enough history to estimate risk")

    cov = cov_mod.estimate(frame, halflife=halflife, shrinkage=shrinkage,
                           lookback_days=lookback_days)
    order = [p.symbol for p in kept]
    cov = cov.loc[order, order]

    notionals = np.array([p.notional(prices[p.symbol]) for p in kept], dtype=float)
    w = notionals / equity

    S = cov.to_numpy()
    var = float(w @ S @ w)
    vol = float(np.sqrt(max(var, 0.0)))

    # Marginal contribution to risk: d(vol)/d(w_i). Component contributions
    # w_i * MCTR_i sum exactly to portfolio vol, which is what makes this the
    # standard decomposition rather than an ad-hoc attribution.
    if vol > 0:
        mctr = S @ w / vol
        contrib = w * mctr
        contrib_pct = contrib / vol
    else:
        mctr = np.zeros_like(w)
        contrib = np.zeros_like(w)
        contrib_pct = np.zeros_like(w)

    stand_alone = np.sqrt(np.diag(S))
    effective = effective_bets(S, w, len(kept))

    # Risk concentration is a DIFFERENT question from independence: it asks
    # whether risk is spread evenly across the line items, ignoring correlation.
    # An inverse Herfindahl of risk shares answers that, and only that -- using
    # it for "independent bets" would report four highly correlated equal-sized
    # positions as four bets, since each still contributes a quarter of the risk.
    denom = float(np.sum(contrib_pct ** 2))
    concentration = 1.0 / denom if denom > 0 else float(len(kept))

    weighted_sa = float(np.sum(np.abs(w) * stand_alone))
    div_ratio = weighted_sa / vol if vol > 0 else np.nan

    # Historical simulation on the actual book, which needs no distributional
    # assumption and keeps fat tails intact.
    port_ret = frame[order].fillna(0.0).to_numpy() @ w
    port_ret = pd.Series(port_ret, index=frame.index)
    q = lambda p: float(np.quantile(port_ret, p)) if len(port_ret) else np.nan

    detail = pd.DataFrame({
        "contracts": [p.contracts for p in kept],
        "price": [prices[p.symbol] for p in kept],
        "sector": [p.instrument.sector for p in kept],
        "notional": notionals,
        "weight": w,
        "vol_standalone": stand_alone,
        "mctr": mctr,
        "risk_contribution": contrib,
        "risk_contribution_pct": contrib_pct,
    }, index=order)
    detail.index.name = "symbol"

    return RiskReport(
        equity=equity,
        positions=detail.sort_values("risk_contribution_pct", key=abs, ascending=False),
        portfolio_vol=vol,
        portfolio_vol_usd=vol * equity,
        gross_leverage=float(np.abs(notionals).sum() / equity),
        net_leverage=float(notionals.sum() / equity),
        effective_bets=effective,
        risk_concentration=concentration,
        diversification_ratio=div_ratio,
        var_95=-q(0.05),
        var_99=-q(0.01),
        cvar_95=-float(port_ret[port_ret <= q(0.05)].mean()) if len(port_ret) else np.nan,
        worst_day=float(port_ret.min()) if len(port_ret) else np.nan,
        worst_week=float(port_ret.rolling(5).sum().min()) if len(port_ret) > 5 else np.nan,
        worst_month=float(port_ret.rolling(21).sum().min()) if len(port_ret) > 21 else np.nan,
        dropped=dropped,
        history_days=len(frame),
    )


def size_to_vol_target(
    report: RiskReport, target_vol: float = 0.15
) -> pd.DataFrame:
    """Scale the whole book to a target volatility.

    Also reports integer-contract rounding, because a suggestion of 0.4
    contracts is not tradable and the rounding error is real risk.
    """
    if report.portfolio_vol <= 0:
        raise ValueError("portfolio volatility is zero; cannot scale")
    k = target_vol / report.portfolio_vol
    out = report.positions[["contracts"]].copy()
    out["scaled"] = out["contracts"] * k
    out["tradable"] = out["scaled"].round()
    out["rounding_error"] = out["tradable"] - out["scaled"]
    out["untradable"] = out["tradable"].abs() < 1
    return out
