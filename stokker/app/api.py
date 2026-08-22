"""JSON API over the research layer.

Every endpoint delegates to `stokker.research`, so the UI cannot disagree with
the CLI or the tests.  No research logic lives here -- this file only shapes
results for transport.

Run with `stokker ui`, or directly:
    uvicorn stokker.app.api:app --reload
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from stokker import research
from stokker.backtest.costs import DEFAULT, STRESSED
from stokker.config import UNIVERSE, get, symbols
from stokker.contracts import roll as roll_mod
from stokker.signals.base import Blend, Constant
from stokker.signals.cot import CommercialFlow, CotExtreme
from stokker.signals.momentum import EWMACrossover, TimeSeriesMomentum

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Stokker", docs_url="/api/docs")

SIGNALS = {
    "tsmom": ("Time-series momentum", lambda: TimeSeriesMomentum()),
    "tsmom-scaled": ("TS momentum (scaled)", lambda: TimeSeriesMomentum(binary=False)),
    "ewma": ("EWMA crossover", lambda: EWMACrossover()),
    "cot-extreme": ("COT extreme", lambda: CotExtreme()),
    "cot-flow": ("Commercial flow", lambda: CommercialFlow()),
    "blend": ("Blend (momentum + COT)", lambda: Blend((TimeSeriesMomentum(), 0.6),
                                                      (CotExtreme(), 0.4))),
    "hold": ("Buy and hold", lambda: Constant()),
}


def _signal(key: str):
    if key not in SIGNALS:
        raise HTTPException(400, f"unknown signal {key!r}")
    return SIGNALS[key][1]()


def _clean(value):
    """JSON cannot carry NaN/Inf; send null instead of emitting invalid JSON."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _downsample(s: pd.Series, max_points: int = 900) -> tuple[list[str], list]:
    """Thin a series for transport, keeping the last point exact."""
    s = s.dropna()
    if len(s) > max_points:
        step = int(np.ceil(len(s) / max_points))
        s = pd.concat([s.iloc[::step], s.iloc[[-1]]])
        s = s[~s.index.duplicated(keep="last")]
    return [d.strftime("%Y-%m-%d") for d in s.index], [_clean(v) for v in s.to_numpy()]


# ------------------------------------------------------------------ metadata
@app.get("/api/universe")
def api_universe():
    return {
        "instruments": [
            {"symbol": i.symbol, "name": i.name, "sector": i.sector,
             "cot_report": i.cot_report, "cot_code": i.cot_code,
             "point_value": i.point_value, "tick_size": i.tick_size}
            for i in UNIVERSE
        ],
        "sectors": sorted({i.sector for i in UNIVERSE}),
    }


@app.get("/api/signals")
def api_signals():
    return {"signals": [{"key": k, "label": v[0]} for k, v in SIGNALS.items()]}


# --------------------------------------------------------------- live state
@app.get("/api/state")
def api_state(signal: str = "tsmom", start: str = "2010-01-01",
              syms: str | None = Query(None)):
    sig = _signal(signal)
    wanted = [s.strip().upper() for s in syms.split(",")] if syms else symbols()
    rows, errors = [], []
    for sym in wanted:
        try:
            ds = research.load(sym, start=start)
            state = research.current_state(ds, sig)
            inst = get(sym)
            state["sector"] = inst.sector
            # A short trailing window gives the UI a sparkline for context.
            state["spark"] = _downsample(ds.price.tail(120), 120)[1]
            rows.append({k: _clean(v) for k, v in state.items()})
        except Exception as exc:  # noqa: BLE001
            log.warning("state %s: %s", sym, exc)
            errors.append({"symbol": sym, "error": str(exc)})
    return {"rows": rows, "errors": errors, "signal": signal}


# ---------------------------------------------------------------- backtests
@app.get("/api/backtest")
def api_backtest(symbol: str, signal: str = "tsmom", start: str = "2010-01-01",
                 stressed: bool = False, target_vol: float = 0.15):
    costs = STRESSED if stressed else DEFAULT
    try:
        ds = research.load(symbol.upper(), start=start)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, str(exc)) from exc

    res = research.backtest(ds, _signal(signal), costs=costs, target_vol=target_vol)
    bench = research.backtest(ds, Constant(), costs=costs, target_vol=target_vol)

    eq_dates, eq_vals = _downsample(res.equity)
    _, bench_vals = _downsample(bench.equity.reindex(res.equity.index).ffill())
    _, pos_vals = _downsample(res.position)

    drawdown = res.equity / res.equity.cummax() - 1.0
    _, dd_vals = _downsample(drawdown)

    return {
        "symbol": ds.inst.symbol,
        "name": ds.inst.name,
        "signal": signal,
        "stats": {k: _clean(v) for k, v in res.stats.items()},
        "benchmark": {k: _clean(v) for k, v in bench.stats.items()},
        "dates": eq_dates,
        "equity": eq_vals,
        "benchmark_equity": bench_vals,
        "position": pos_vals,
        "drawdown": dd_vals,
    }


@app.get("/api/sweep")
def api_sweep(signal: str = "tsmom", start: str = "2010-01-01", stressed: bool = False):
    costs = STRESSED if stressed else DEFAULT
    try:
        table, results = research.sweep(_signal(signal), start=start, costs=costs)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc

    rows = []
    for sym, row in table.iterrows():
        entry = {"symbol": sym, **{k: _clean(v) for k, v in row.items()}}
        if sym != "PORTFOLIO":
            entry["sector"] = get(sym).sector
            entry["name"] = get(sym).name
        rows.append(entry)

    port = results["PORTFOLIO"]
    dates, equity = _downsample(port.equity)
    dd = port.equity / port.equity.cummax() - 1.0
    return {
        "rows": rows,
        "portfolio": {"dates": dates, "equity": equity,
                      "drawdown": _downsample(dd)[1]},
        "signal": signal,
    }


# -------------------------------------------------------------- positioning
@app.get("/api/cot")
def api_cot(symbol: str, start: str = "2010-01-01"):
    try:
        ds = research.load(symbol.upper(), start=start)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, str(exc)) from exc
    if ds.cot is None or ds.cot.empty:
        return {"symbol": ds.inst.symbol, "available": False}

    cot = ds.cot
    oi = cot["open_interest"].replace(0, np.nan)
    spec = (cot["spec_net"] / oi).astype(float)
    comm = (cot["commercial_net"] / oi).astype(float)
    # Z-score of speculative crowding, the input CotExtreme actually reads.
    z = (spec - spec.rolling(156, min_periods=52).mean()) / spec.rolling(
        156, min_periods=52).std()

    dates, spec_v = _downsample(spec, 700)
    return {
        "symbol": ds.inst.symbol, "name": ds.inst.name, "available": True,
        "report": ds.inst.cot_report, "cot_code": ds.inst.cot_code,
        "weeks": len(cot),
        "dates": dates,
        "spec": spec_v,
        "commercial": _downsample(comm, 700)[1],
        "spec_z": _downsample(z, 700)[1],
        "price_dates": _downsample(ds.price, 700)[0],
        "price": _downsample(ds.price, 700)[1],
        "latest": {
            "report_date": cot.index[-1].strftime("%Y-%m-%d"),
            "release_date": pd.Timestamp(cot["release_date"].iloc[-1]).strftime("%Y-%m-%d"),
            "spec_net": _clean(cot["spec_net"].iloc[-1]),
            "commercial_net": _clean(cot["commercial_net"].iloc[-1]),
            "open_interest": _clean(cot["open_interest"].iloc[-1]),
            "spec_z": _clean(z.iloc[-1]),
        },
    }


# ------------------------------------------------------------------- rolls
@app.get("/api/rolls")
def api_rolls(symbol: str, start: str = "2010-01-01"):
    try:
        ds = research.load(symbol.upper(), with_cot=False, start=start)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, str(exc)) from exc

    impact = roll_mod.impact_report(ds.bars, ds.inst)
    validation = roll_mod.validate_calendar(ds.bars, ds.inst)
    recent = roll_mod.roll_report(ds.bars, ds.inst).tail(10)

    return {
        "symbol": ds.inst.symbol, "name": ds.inst.name,
        "impact": {k: _clean(v) for k, v in impact.items()},
        "validation": [
            {"offset": int(o), **{k: _clean(v) for k, v in row.items()}}
            for o, row in validation.iterrows()
        ],
        "recent": [
            {"date": d.strftime("%Y-%m-%d"), **{k: _clean(v) for k, v in row.items()}}
            for d, row in recent.iterrows()
        ],
    }


# ---------------------------------------------------------- walk-forward
WF_GRIDS = {
    "tsmom": {"lookback": [63, 126, 252, 504]},
    "tsmom-scaled": {"lookback": [63, 126, 252, 504]},
    "ewma": {"fast": [16, 32, 64], "slow": [128, 256]},
    "cot-extreme": {"lookback_weeks": [104, 156, 260], "deadband": [0.5, 1.0, 1.5]},
}
WF_CLASSES = {
    "tsmom": TimeSeriesMomentum, "tsmom-scaled": TimeSeriesMomentum,
    "ewma": EWMACrossover, "cot-extreme": CotExtreme,
}


@app.get("/api/walkforward")
def api_walkforward(signal: str = "tsmom", start: str = "2006-01-01",
                    train: float = 5.0, test: float = 1.0,
                    anchored: bool = False, stressed: bool = False,
                    syms: str | None = Query(None)):
    from stokker.backtest.walkforward import WalkForward

    if signal not in WF_GRIDS:
        raise HTTPException(400, f"no grid for {signal!r}")
    grid, cls = WF_GRIDS[signal], WF_CLASSES[signal]
    fixed = {"binary": False} if signal == "tsmom-scaled" else {}

    wanted = [s.strip().upper() for s in syms.split(",")] if syms else symbols()
    datasets = {}
    for sym in wanted:
        try:
            datasets[sym] = research.load(sym, start=start)
        except Exception as exc:  # noqa: BLE001
            log.warning("walkforward %s: %s", sym, exc)
    if not datasets:
        raise HTTPException(404, "no instruments could be loaded")

    wf = WalkForward(train_years=train, test_years=test, anchored=anchored)
    try:
        res = wf.run_pooled(datasets, cls, grid,
                            costs=STRESSED if stressed else DEFAULT, fixed=fixed)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc

    stab = res.stability()
    pcols = [c for c in stab.columns if "sharpe" not in c]
    dates, equity = _downsample(res.equity())
    fixed_eq = (1.0 + res.fixed_oos_returns.reindex(res.oos_returns.index).fillna(0.0)).cumprod()

    return {
        "signal": signal,
        "instruments": len(datasets),
        "grid": {k: list(v) for k, v in grid.items()},
        "summary": {k: _clean(v) for k, v in res.summary().items()},
        "in_sample_params": res.in_sample_params,
        "fixed_params": res.fixed_params or {},
        "param_names": pcols,
        "folds": [
            {"fold": int(i),
             "params": {c: _clean(row[c]) for c in pcols},
             "train_sharpe": _clean(row["train_sharpe"]),
             "test_sharpe": _clean(row["test_sharpe"]),
             "train_start": f.spec.train_start.strftime("%Y-%m-%d"),
             "train_end": f.spec.train_end.strftime("%Y-%m-%d"),
             "test_start": f.spec.test_start.strftime("%Y-%m-%d"),
             "test_end": f.spec.test_end.strftime("%Y-%m-%d")}
            for (i, row), f in zip(stab.iterrows(), res.folds)
        ],
        "dates": dates,
        "equity": equity,
        "fixed_equity": _downsample(fixed_eq)[1],
    }


# ------------------------------------------------------------------ static
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/", StaticFiles(directory=STATIC), name="static")
