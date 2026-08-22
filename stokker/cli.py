"""Command line entry point.

    stokker ui                       serve the web UI at localhost:8000
    stokker verify-universe          check every CFTC code resolves
    stokker search-cot corn          find a contract code by name
    stokker fetch [--symbols ES,CL]  populate the local cache
    stokker rolls ES                 inspect detected roll gaps
    stokker backtest ES --signal tsmom
    stokker sweep --signal tsmom
    stokker state                    current signal across the universe
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from stokker.backtest.costs import DEFAULT, STRESSED
from stokker.config import UNIVERSE, get, symbols
from stokker.contracts import roll as roll_mod
from stokker.data.providers import cot as cot_data
from stokker.signals.base import Blend, Constant
from stokker.signals.cot import CommercialFlow, CotExtreme
from stokker.signals.momentum import EWMACrossover, TimeSeriesMomentum, VolRegimeFilter

SIGNALS = {
    "tsmom": lambda: TimeSeriesMomentum(),
    "tsmom-scaled": lambda: TimeSeriesMomentum(binary=False),
    "ewma": lambda: EWMACrossover(),
    "cot-extreme": lambda: CotExtreme(),
    "cot-flow": lambda: CommercialFlow(),
    "hold": lambda: Constant(),
    "blend": lambda: Blend((TimeSeriesMomentum(), 0.6), (CotExtreme(), 0.4)),
}


def _show(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 50,
                           "display.float_format", lambda v: f"{v:,.3f}"):
        print(df)


def cmd_verify_universe(args) -> int:
    """Confirm every configured CFTC code exists in the real reports."""
    year = pd.Timestamp.today().year - 1
    cache: dict[str, pd.DataFrame] = {}
    bad = 0
    print(f"{'SYM':<5} {'CODE':<8} {'REPORT':<14} MARKET")
    for inst in UNIVERSE:
        if inst.cot_report not in cache:
            raw = cot_data._download(inst.cot_report, year)
            raw["_code"] = raw["CFTC_Contract_Market_Code"].astype(str).str.strip()
            cache[inst.cot_report] = raw
        raw = cache[inst.cot_report]
        hit = raw.loc[raw["_code"] == inst.cot_code, "Market_and_Exchange_Names"]
        if hit.empty:
            print(f"{inst.symbol:<5} {inst.cot_code:<8} {inst.cot_report:<14} *** NOT FOUND ***")
            bad += 1
        else:
            print(f"{inst.symbol:<5} {inst.cot_code:<8} {inst.cot_report:<14} {hit.iloc[0].strip()[:55]}")
    print(f"\n{len(UNIVERSE) - bad}/{len(UNIVERSE)} codes resolved.")
    return 1 if bad else 0


def cmd_search_cot(args) -> int:
    for report in ("tff", "disaggregated"):
        hits = cot_data.search(args.term, report)
        if not hits.empty:
            print(f"\n--- {report} ---")
            _show(hits)
    return 0


def cmd_fetch(args) -> int:
    from stokker import research

    syms = args.symbols.split(",") if args.symbols else symbols()
    for sym in syms:
        try:
            ds = research.load(sym.strip(), start=args.start, refresh=args.refresh)
            print(f"  {ds}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym}: FAILED - {exc}", file=sys.stderr)
    return 0


def cmd_rolls(args) -> int:
    from stokker import research

    ds = research.load(args.symbol, with_cot=False, start=args.start)
    impact = roll_mod.impact_report(ds.bars, ds.inst)

    print(f"\n{args.symbol} ({ds.inst.name}) -- {len(ds.bars)} bars")
    print(f"  rolls/year          {impact['rolls_per_year']:.2f}")
    print(f"  mean |roll gap|     {impact['mean_abs_roll_gap_pct']:.2f}%")
    print(f"  cum logret naive    {impact['naive_cum_logret']:+.4f}")
    print(f"  cum logret adjusted {impact['safe_cum_logret']:+.4f}")
    print(f"  roll effect         {impact['cum_delta']:+.4f} log units")

    if args.validate:
        print("\nCalendar validation -- |mean gap| should peak at offset 0.")
        print("Low-basis markets (equity/rates/FX) lack the power to show it; "
              "check the offset-0 value against cost-of-carry instead.")
        _show(roll_mod.validate_calendar(ds.bars, ds.inst).round(4))
    else:
        print("\nMost recent roll bars:")
        _show(roll_mod.roll_report(ds.bars, ds.inst).tail(8))
    return 0


def cmd_backtest(args) -> int:
    from stokker import research

    ds = research.load(args.symbol, start=args.start)
    sig = SIGNALS[args.signal]()
    costs = STRESSED if args.stressed else DEFAULT
    res = research.backtest(ds, sig, costs=costs)
    print(f"\n{ds}\nsignal={sig!r}  costs={'STRESSED' if args.stressed else 'default'}\n")
    _show(res.summary().to_frame().T)
    return 0


def cmd_sweep(args) -> int:
    from stokker import research

    sig = SIGNALS[args.signal]()
    costs = STRESSED if args.stressed else DEFAULT
    universe = args.symbols.split(",") if args.symbols else None
    table, _ = research.sweep(sig, universe=universe, start=args.start, costs=costs)
    print(f"\nsignal={sig!r}  costs={'STRESSED' if args.stressed else 'default'}\n")
    _show(table[["years", "cagr", "vol", "sharpe", "max_drawdown",
                 "gross_sharpe", "cost_drag_ann", "turnover_ann"]])
    return 0


def cmd_state(args) -> int:
    from stokker import research

    sig = SIGNALS[args.signal]()
    rows = []
    for sym in (args.symbols.split(",") if args.symbols else symbols()):
        try:
            ds = research.load(sym.strip(), start=args.start)
            rows.append(research.current_state(ds, sig))
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym}: skipped - {exc}", file=sys.stderr)
    if not rows:
        return 1
    _show(pd.DataFrame(rows).set_index("symbol"))
    print("\nExposure is a research output, not a recommendation.")
    return 0


def cmd_ui(args) -> int:
    """Serve the web UI."""
    import uvicorn

    print(f"\n  Stokker UI  ->  http://{args.host}:{args.port}\n")
    uvicorn.run("stokker.app.api:app", host=args.host, port=args.port,
                reload=args.reload, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stokker", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--start", default="2006-01-01")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify-universe").set_defaults(fn=cmd_verify_universe)

    s = sub.add_parser("ui", help="serve the web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_ui)

    s = sub.add_parser("search-cot"); s.add_argument("term"); s.set_defaults(fn=cmd_search_cot)

    s = sub.add_parser("fetch")
    s.add_argument("--symbols"); s.add_argument("--refresh", action="store_true")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("rolls")
    s.add_argument("symbol")
    s.add_argument("--validate", action="store_true",
                   help="offset-sensitivity test on the roll calendar")
    s.set_defaults(fn=cmd_rolls)

    for name, fn in (("backtest", cmd_backtest), ("sweep", cmd_sweep), ("state", cmd_state)):
        s = sub.add_parser(name)
        if name == "backtest":
            s.add_argument("symbol")
        else:
            s.add_argument("--symbols")
        s.add_argument("--signal", choices=sorted(SIGNALS), default="tsmom")
        if name != "state":
            s.add_argument("--stressed", action="store_true",
                           help="use pessimistic cost assumptions")
        s.set_defaults(fn=fn)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
