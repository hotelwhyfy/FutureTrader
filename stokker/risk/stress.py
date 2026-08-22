"""Historical stress replay.

Rather than simulating shocks from a distribution, this replays the current book
through windows that actually happened. Real crises carry correlation behaviour
no parametric model reproduces: diversification tends to fail exactly when it is
needed, and only the historical record shows by how much.

Coverage is reported per scenario. Several instruments here postdate 2008 (RTY
from 2017, UB from 2010, BZ from 2007), so an early scenario may cover only part
of the book. A partial replay is reported as partial rather than quietly scaled
up, because understating a stress loss is the one error worth ruling out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stokker.risk.portfolio import RiskReport


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    start: str
    end: str
    note: str


#: Windows chosen for distinct failure modes, not just size: an equity crash, a
#: rates shock, a vol-complex blowup, a commodity collapse, a correlated
#: everything-down year.
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("gfc", "Global Financial Crisis", "2008-09-01", "2008-12-31",
             "equity crash with commodity collapse and flight to Treasuries"),
    Scenario("euro", "Euro sovereign crisis", "2011-07-01", "2011-10-31",
             "equity selloff, gold spike, dollar bid"),
    Scenario("taper", "Taper tantrum", "2013-05-22", "2013-06-30",
             "bonds and equities fall together — diversification fails"),
    Scenario("cny", "China devaluation", "2015-08-17", "2015-09-01",
             "sharp equity air-pocket, industrial metals hit"),
    Scenario("volmageddon", "Volmageddon", "2018-02-01", "2018-02-14",
             "vol-complex unwind, equity-only shock"),
    Scenario("covid", "COVID crash", "2020-02-19", "2020-03-23",
             "everything down, correlations to 1, oil breaks"),
    Scenario("oil_neg", "Negative oil", "2020-04-01", "2020-05-01",
             "WTI settles below zero; energy-specific"),
    Scenario("inflation22", "2022 rates and inflation", "2022-01-01", "2022-10-31",
             "bonds and equities down together for a year"),
    Scenario("gilt", "UK gilt crisis", "2022-09-19", "2022-10-14",
             "rates disorder, sterling stress"),
)

BY_KEY = {s.key: s for s in SCENARIOS}


def replay(
    report: RiskReport,
    returns: dict[str, pd.Series],
    scenarios: tuple[Scenario, ...] = SCENARIOS,
) -> pd.DataFrame:
    """Apply today's weights to historical returns from each window.

    Positions are held fixed: this measures the shock to the book as it stands,
    not what a trader might have done during the event.
    """
    w = report.positions["weight"]
    rows = []
    for sc in scenarios:
        lo, hi = pd.Timestamp(sc.start), pd.Timestamp(sc.end)
        cols, missing = {}, []
        for sym, weight in w.items():
            ser = returns.get(sym)
            if ser is None:
                missing.append(sym)
                continue
            seg = ser[(ser.index >= lo) & (ser.index <= hi)]
            if seg.dropna().empty:
                missing.append(sym)
                continue
            cols[sym] = seg
        if not cols:
            rows.append({"scenario": sc.name, "key": sc.key, "days": 0,
                         "pnl_pct": np.nan, "pnl_usd": np.nan, "worst_day": np.nan,
                         "max_drawdown": np.nan, "coverage": 0.0, "note": sc.note,
                         "missing": ",".join(missing)})
            continue

        frame = pd.DataFrame(cols).fillna(0.0)
        ww = w.reindex(frame.columns).fillna(0.0)
        port = frame.to_numpy() @ ww.to_numpy()
        port = pd.Series(port, index=frame.index)
        eq = (1.0 + port).cumprod()
        dd = float((eq / eq.cummax() - 1.0).min())
        total = float(eq.iloc[-1] - 1.0)
        # Coverage weighted by risk share, so missing a large position matters
        # more than missing a token one.
        share = report.positions["risk_contribution_pct"].abs()
        covered = float(share.reindex(frame.columns).fillna(0.0).sum() / max(share.sum(), 1e-12))

        rows.append({
            "scenario": sc.name, "key": sc.key, "days": len(port),
            "pnl_pct": total, "pnl_usd": total * report.equity,
            "worst_day": float(port.min()), "max_drawdown": dd,
            "coverage": covered, "note": sc.note,
            "missing": ",".join(missing),
        })
    return pd.DataFrame(rows).set_index("key")


def worst_windows(
    report: RiskReport, returns: dict[str, pd.Series], window: int = 21, top: int = 5
) -> pd.DataFrame:
    """The worst rolling windows this book would have suffered, ever.

    A useful complement to named scenarios: it finds the book's own worst
    periods rather than the market's famous ones, which are not always the same.
    """
    w = report.positions["weight"]
    cols = {s: returns[s] for s in w.index if s in returns}
    frame = pd.DataFrame(cols).dropna(how="all").fillna(0.0)
    ww = w.reindex(frame.columns).fillna(0.0)
    port = pd.Series(frame.to_numpy() @ ww.to_numpy(), index=frame.index)

    rolled = port.rolling(window).sum().dropna()
    if rolled.empty:
        return pd.DataFrame()
    worst = rolled.nsmallest(top)
    return pd.DataFrame({
        "window_end": worst.index,
        "loss_pct": worst.to_numpy(),
        "loss_usd": worst.to_numpy() * report.equity,
        "window_start": [rolled.index[max(0, rolled.index.get_loc(d) - window + 1)]
                         for d in worst.index],
    }).set_index("window_end")
