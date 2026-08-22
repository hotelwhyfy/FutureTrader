"""Streamlit research dashboard.

Deliberately thin: every number here comes from `stokker.research`, so the app
cannot drift from what the CLI and the tests see.  Run with:

    streamlit run stokker/app/dashboard.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stokker import research
from stokker.backtest.costs import DEFAULT, STRESSED, CostModel
from stokker.config import UNIVERSE, get, symbols
from stokker.contracts import roll as roll_mod
from stokker.signals.base import Blend, Constant
from stokker.signals.cot import CommercialFlow, CotExtreme
from stokker.signals.momentum import EWMACrossover, TimeSeriesMomentum

st.set_page_config(page_title="Stokker", page_icon="📈", layout="wide")

SIGNALS = {
    "Time-series momentum": TimeSeriesMomentum,
    "TS momentum (scaled)": lambda: TimeSeriesMomentum(binary=False),
    "EWMA crossover": EWMACrossover,
    "COT extreme": CotExtreme,
    "Commercial flow": CommercialFlow,
    "Buy and hold": Constant,
    "Blend (momentum + COT)": lambda: Blend((TimeSeriesMomentum(), 0.6), (CotExtreme(), 0.4)),
}


@st.cache_data(show_spinner="Loading data…")
def _load(symbol: str, start: str):
    ds = research.load(symbol, start=start)
    return ds.bars, ds.returns, ds.cot


def _dataset(symbol: str, start: str) -> research.Dataset:
    bars, rets, cot = _load(symbol, start)
    return research.Dataset(get(symbol), bars, rets, cot)


st.title("Stokker")
st.caption(
    "Futures trend research on free data. Everything here is a research output, "
    "not investment advice."
)

with st.sidebar:
    st.header("Setup")
    sig_name = st.selectbox("Signal", list(SIGNALS))
    start = st.text_input("Start date", "2010-01-01")
    stressed = st.checkbox("Pessimistic costs", value=False,
                           help="3 ticks slippage and double commission")
    target_vol = st.slider("Target volatility", 0.05, 0.40, 0.15, 0.01)
    st.divider()
    st.caption(
        "Data: Yahoo (prices, unofficial) + CFTC COT (official, weekly). "
        "COT is aligned on its Friday release date, never its Tuesday report date."
    )

signal = SIGNALS[sig_name]()
costs: CostModel = STRESSED if stressed else DEFAULT

tab_state, tab_bt, tab_cot, tab_rolls = st.tabs(
    ["Current state", "Backtest", "Positioning", "Roll diagnostics"]
)

# --------------------------------------------------------------- current state
with tab_state:
    st.subheader("Where the signal stands right now")
    picks = st.multiselect("Instruments", symbols(), default=symbols())
    if picks and st.button("Compute", type="primary"):
        rows = []
        bar = st.progress(0.0)
        for i, sym in enumerate(picks):
            try:
                rows.append(research.current_state(_dataset(sym, start), signal))
            except Exception as exc:  # noqa: BLE001
                st.warning(f"{sym}: {exc}")
            bar.progress((i + 1) / len(picks))
        bar.empty()
        if rows:
            df = pd.DataFrame(rows).set_index("symbol")
            longs = (df["direction"] == "LONG").sum()
            shorts = (df["direction"] == "SHORT").sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Long", longs)
            c2.metric("Short", shorts)
            c3.metric("Flat", len(df) - longs - shorts)
            st.dataframe(
                df[["name", "direction", "exposure", "changed", "last_close",
                    "last_bar", "bar_age_days"]
                   + ([c for c in ("cot_age_days",) if c in df.columns])],
                width='stretch',
            )
            st.caption(
                "`changed` marks a signal that flipped on the latest bar. "
                "`bar_age_days` matters: a stale bar means a stale signal."
            )

# -------------------------------------------------------------------- backtest
with tab_bt:
    st.subheader("Backtest")
    mode = st.radio("Scope", ["Single instrument", "Whole universe"], horizontal=True)

    if mode == "Single instrument":
        sym = st.selectbox("Instrument", symbols())
        if st.button("Run backtest", type="primary"):
            ds = _dataset(sym, start)
            res = research.backtest(ds, signal, costs=costs, target_vol=target_vol)
            bench = research.backtest(ds, Constant(), costs=costs, target_vol=target_vol)

            cols = st.columns(4)
            cols[0].metric("Sharpe", f"{res.stats['sharpe']:.2f}",
                           f"{res.stats['sharpe'] - bench.stats['sharpe']:+.2f} vs hold")
            cols[1].metric("CAGR", f"{res.stats['cagr']:.2%}")
            cols[2].metric("Max drawdown", f"{res.stats['max_drawdown']:.1%}")
            cols[3].metric("Cost drag / yr", f"{res.stats['cost_drag_ann']:.2%}")

            fig = go.Figure()
            fig.add_scatter(x=res.equity.index, y=res.equity, name=sig_name)
            fig.add_scatter(x=bench.equity.index, y=bench.equity, name="Buy and hold",
                            line=dict(dash="dot"))
            fig.update_layout(height=420, yaxis_type="log", yaxis_title="Growth of 1",
                              margin=dict(t=30))
            st.plotly_chart(fig, width='stretch')

            st.plotly_chart(
                go.Figure(go.Scatter(x=res.position.index, y=res.position, name="Position"))
                .update_layout(height=200, yaxis_title="Contracts", margin=dict(t=10)),
                width='stretch',
            )
            st.dataframe(res.summary().to_frame("value").T, width='stretch')
    else:
        if st.button("Run sweep", type="primary"):
            with st.spinner("Backtesting every instrument…"):
                table, results = research.sweep(signal, start=start, costs=costs)
            st.dataframe(
                table[["years", "cagr", "vol", "sharpe", "max_drawdown",
                       "gross_sharpe", "cost_drag_ann", "turnover_ann"]]
                .style.format("{:.3f}")
                .background_gradient(subset=["sharpe"], cmap="RdYlGn", vmin=-1, vmax=1),
                width='stretch',
            )
            port = results["PORTFOLIO"]
            st.plotly_chart(
                go.Figure(go.Scatter(x=port.equity.index, y=port.equity))
                .update_layout(height=380, yaxis_title="Growth of 1",
                               title="Equal-weight portfolio", margin=dict(t=40)),
                width='stretch',
            )

# ----------------------------------------------------------------- positioning
with tab_cot:
    st.subheader("CFTC positioning")
    sym = st.selectbox("Instrument", symbols(), key="cot_sym")
    ds = _dataset(sym, start)
    if ds.cot is None or ds.cot.empty:
        st.info("No COT data loaded for this instrument.")
    else:
        cot = ds.cot
        oi = cot["open_interest"].replace(0, np.nan)
        fig = go.Figure()
        fig.add_scatter(x=cot.index, y=cot["spec_net"] / oi, name="Speculators (net / OI)")
        fig.add_scatter(x=cot.index, y=cot["commercial_net"] / oi, name="Commercials (net / OI)")
        fig.update_layout(height=380, yaxis_title="Net position / open interest",
                          margin=dict(t=30))
        st.plotly_chart(fig, width='stretch')

        st.plotly_chart(
            go.Figure(go.Scatter(x=ds.bars.index, y=ds.bars["close"], name="Price"))
            .update_layout(height=280, margin=dict(t=10)),
            width='stretch',
        )
        st.caption(
            f"Report type: **{ds.inst.cot_report}** · contract code `{ds.inst.cot_code}` · "
            f"{len(cot)} weekly observations. Positions are as of Tuesday and published "
            "the following Friday; signals use the release date."
        )

# ------------------------------------------------------------ roll diagnostics
with tab_rolls:
    st.subheader("Roll diagnostics")
    st.markdown(
        "Rolls are derived from each contract's **delivery calendar**, not from price "
        "outliers. An earlier statistical detector flagged the 2010 flash crash, Brexit, "
        "Volmageddon and the COVID selloff as rolls and zeroed them — inflating ES "
        "cumulative return by roughly 0.6 log units. Price-based roll detection does not work."
    )
    sym = st.selectbox("Instrument", symbols(), key="roll_sym")
    ds = _dataset(sym, start)
    impact = roll_mod.impact_report(ds.bars, ds.inst)

    c = st.columns(4)
    c[0].metric("Rolls / year", f"{impact['rolls_per_year']:.1f}")
    c[1].metric("Mean |roll gap|", f"{impact['mean_abs_roll_gap_pct']:.2f}%")
    c[2].metric("Cum. return, naive", f"{impact['naive_cum_logret']:+.3f}")
    c[3].metric("Roll effect", f"{impact['cum_delta']:+.3f}",
                help="How much roll neutralisation moves cumulative log return")

    st.markdown("**Calendar validation** — |mean gap| should peak at offset 0.")
    st.dataframe(roll_mod.validate_calendar(ds.bars, ds.inst).round(4),
                 width='stretch')
    st.caption(
        "Low-basis markets (equity index, rates, FX) have too little basis relative to "
        "daily noise for the peak to be visible; for those, compare the offset-0 value "
        "against theoretical cost-of-carry instead."
    )
