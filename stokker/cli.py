"""Command line entry point.

    stokker ui                       serve the web UI at localhost:8000
    stokker verify-universe          check every CFTC code resolves
    stokker search-cot corn          find a contract code by name
    stokker fetch [--symbols ES,CL]  populate the local cache
    stokker rolls ES                 inspect detected roll gaps
    stokker backtest ES --signal tsmom
    stokker sweep --signal tsmom
    stokker state                    current signal across the universe
    stokker walkforward              out-of-sample parameter evaluation
    stokker risk --positions ES:2    portfolio risk decomposition
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


# Grids are deliberately small and log-spaced. A dense grid is a licence to
# overfit: every extra candidate raises the best in-sample score on noise alone.
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


def cmd_walkforward(args) -> int:
    """Out-of-sample evaluation with train-only parameter selection."""
    from stokker import research
    from stokker.backtest.walkforward import WalkForward

    if args.signal not in WF_GRIDS:
        print(f"no grid defined for {args.signal!r}; "
              f"available: {', '.join(sorted(WF_GRIDS))}", file=sys.stderr)
        return 1

    cls, grid = WF_CLASSES[args.signal], WF_GRIDS[args.signal]
    fixed = {"tsmom-scaled": {"binary": False}}.get(args.signal, {})
    wf = WalkForward(train_years=args.train, test_years=args.test,
                     anchored=args.anchored, embargo_days=args.embargo)
    costs = STRESSED if args.stressed else DEFAULT

    syms = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else symbols()
    datasets = {}
    for sym in syms:
        try:
            datasets[sym] = research.load(sym, start=args.start)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym}: skipped ({exc})", file=sys.stderr)
    if not datasets:
        return 1

    if args.per_instrument:
        rows = []
        for sym, ds in datasets.items():
            try:
                r = wf.run(ds, cls, grid, costs=costs, fixed=fixed)
                rows.append(r.summary())
            except Exception as exc:  # noqa: BLE001
                print(f"  {sym}: {exc}", file=sys.stderr)
        if not rows:
            return 1
        _show(pd.DataFrame(rows))
        return 0

    res = wf.run_pooled(datasets, cls, grid, costs=costs, fixed=fixed)
    print(f"\nsignal={args.signal}  grid={grid}  {len(datasets)} instruments")
    print(f"{wf.train_years}y train / {wf.test_years}y test"
          f"{' (anchored)' if args.anchored else ' (rolling)'}\n")
    _show(res.summary().to_frame().T)
    print("\nPer-fold selection:")
    _show(res.stability())
    print(f"\n  in-sample (hindsight)   {res.in_sample_sharpe:+.3f}  {res.in_sample_params}")
    print(f"  out-of-sample           {res.oos_sharpe:+.3f}")
    print(f"  never-tuned baseline    {res.fixed_sharpe:+.3f}  {res.fixed_params or 'defaults'}")
    print(f"\n  overfitting tax         {res.overfit_gap:+.3f} Sharpe")
    print(f"  value of tuning         {res.selection_edge:+.3f} Sharpe", end="")
    print("   <- negative means stop tuning" if res.selection_edge < 0 else "")
    return 0


def cmd_risk(args) -> int:
    """Decompose the risk of a book."""
    import numpy as np

    from stokker import research
    from stokker.backtest.engine import vol_target_size
    from stokker.risk import stress
    from stokker.risk.portfolio import Position, analyse, parse_positions, size_to_vol_target

    if not args.positions and not args.from_signal:
        print("supply --positions 'ES:2,CL:-1' or --from-signal tsmom", file=sys.stderr)
        return 1

    # Load once; both the position source and the risk model need returns.
    wanted = symbols()
    datasets = {}
    for sym in wanted:
        try:
            datasets[sym] = research.load(sym, with_cot=bool(args.from_signal), start=args.start)
        except Exception:  # noqa: BLE001
            pass

    prices = {s: float(d.price.iloc[-1]) for s, d in datasets.items()}
    returns = {s: d.returns for s, d in datasets.items()}

    if args.positions:
        positions = parse_positions(args.positions)
        missing = [p.symbol for p in positions if p.symbol not in prices]
        if missing:
            print(f"no price data for: {', '.join(missing)}", file=sys.stderr)
            return 1
    else:
        # Reconstruct what the systematic book would hold today.
        #
        # `vol_target_size` scales each instrument to `target_vol` ON ITS OWN;
        # the backtest engine then equal-weights across the book. Applying the
        # per-instrument size to all 43 at once would be ~43x the intended
        # exposure -- a mistake the stress replay caught by reporting a -100%
        # GFC loss on a book nominally targeting 15% vol.
        sig_obj = SIGNALS[args.from_signal]()
        raw = []
        for sym, d in datasets.items():
            sig = sig_obj.generate(d.bars, d.returns, d.inst, d.cot)
            size = vol_target_size(d.returns, target_vol=args.target_vol)
            w = float(sig.iloc[-1]) * float(size.iloc[-1])
            if abs(w) < 1e-6:
                continue
            raw.append((sym, w, d.inst.point_value))
        if not raw:
            print("signal is flat everywhere; no book to analyse", file=sys.stderr)
            return 1

        n_active = len(raw)
        positions = [
            Position(sym, (w / n_active) * args.equity / (prices[sym] * pv))
            for sym, w, pv in raw
        ]

        # Equal-weighting leaves realised portfolio vol well below target,
        # because diversification cuts it by roughly sqrt(effective bets).
        # Rescale so `--target-vol` means what it says: portfolio volatility.
        probe = analyse(positions, prices, returns, equity=args.equity,
                        halflife=args.halflife, shrinkage=args.shrinkage)
        if probe.portfolio_vol > 0:
            k = args.target_vol / probe.portfolio_vol
            positions = [Position(p.symbol, p.contracts * k) for p in positions]

    rep = analyse(positions, prices, returns, equity=args.equity,
                  halflife=args.halflife, shrinkage=args.shrinkage)

    print(f"\nequity ${rep.equity:,.0f}   {len(rep.positions)} positions   "
          f"{rep.history_days} days of history\n")
    print(f"  portfolio vol      {rep.portfolio_vol:>8.1%}  (${rep.portfolio_vol_usd:,.0f}/yr)")
    print(f"  effective bets     {rep.effective_bets:>8.2f}  of {len(rep.positions)} positions")
    print(f"  risk concentration {rep.risk_concentration:>8.2f}")
    print(f"  diversification    {rep.diversification_ratio:>8.2f}x")
    print(f"  gross / net lev    {rep.gross_leverage:>8.2f}x / {rep.net_leverage:.2f}x")
    print(f"\n  1-day VaR 95%      {rep.var_95:>8.2%}  (${rep.var_95*rep.equity:,.0f})")
    print(f"  1-day VaR 99%      {rep.var_99:>8.2%}  (${rep.var_99*rep.equity:,.0f})")
    print(f"  1-day CVaR 95%     {rep.cvar_95:>8.2%}  (${rep.cvar_95*rep.equity:,.0f})")
    print(f"  worst day / month  {rep.worst_day:>8.2%} / {rep.worst_month:.2%}")

    warn = rep.concentration_warnings()
    if warn:
        print("\n  warnings:")
        for w in warn:
            print(f"    - {w}")

    print("\nRisk decomposition:")
    cols = ["contracts", "sector", "notional", "weight", "vol_standalone",
            "risk_contribution_pct"]
    _show(rep.positions[cols].head(args.top))

    if args.target_vol_scale:
        print(f"\nScaled to {args.target_vol_scale:.0%} portfolio vol:")
        _show(size_to_vol_target(rep, args.target_vol_scale).head(args.top))

    if args.stress:
        print("\nHistorical stress replay (positions held fixed):")
        out = stress.replay(rep, returns)
        _show(out[["scenario", "days", "pnl_pct", "pnl_usd", "max_drawdown", "coverage"]])
        print("\nWorst 21-day windows for this book:")
        _show(stress.worst_windows(rep, returns, window=21, top=5))
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

    s = sub.add_parser("risk", help="portfolio risk decomposition")
    s.add_argument("--positions", help='e.g. "ES:2,CL:-1,GC:3"')
    s.add_argument("--from-signal", choices=sorted(SIGNALS),
                   help="analyse what the systematic book would hold today")
    s.add_argument("--equity", type=float, default=100_000.0)
    s.add_argument("--target-vol", type=float, default=0.15,
                   help="vol target used when sizing --from-signal")
    s.add_argument("--target-vol-scale", type=float,
                   help="also show the book rescaled to this portfolio vol")
    s.add_argument("--halflife", type=int, default=126)
    s.add_argument("--shrinkage", type=float, default=0.2)
    s.add_argument("--stress", action="store_true", help="replay historical crises")
    s.add_argument("--top", type=int, default=20)
    s.set_defaults(fn=cmd_risk)

    s = sub.add_parser("walkforward", help="out-of-sample parameter evaluation")
    s.add_argument("--symbols")
    s.add_argument("--signal", choices=sorted(WF_GRIDS), default="tsmom")
    s.add_argument("--train", type=float, default=5.0, help="training years")
    s.add_argument("--test", type=float, default=1.0, help="test years per fold")
    s.add_argument("--embargo", type=int, default=0)
    s.add_argument("--anchored", action="store_true", help="grow train window from a fixed origin")
    s.add_argument("--per-instrument", action="store_true",
                   help="select parameters per market instead of one set for all")
    s.add_argument("--stressed", action="store_true")
    s.set_defaults(fn=cmd_walkforward)

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
