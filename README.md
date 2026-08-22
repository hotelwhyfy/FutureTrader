# Stokker

Futures trend research on free data. Built to answer one question honestly:
**is there a signal here worth trading?**

The current answer, measured over 2006–2026, is **not yet** — see
[Findings](#findings). That is a real result, not a placeholder. The framework
exists so that claim can be checked, contradicted, and improved.

> Research output only. Nothing here is investment advice.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[app,dev]"

.venv/bin/stokker verify-universe          # confirm CFTC codes resolve
.venv/bin/stokker fetch                    # populate the cache (~2 min)
.venv/bin/stokker ui                       # web UI at localhost:8000
```

Or stay in the terminal:

```bash
.venv/bin/stokker sweep --signal tsmom     # backtest the universe
.venv/bin/stokker state                    # what the signal says today
.venv/bin/stokker rolls CL --validate      # roll calendar diagnostics
```

Optional: put a free [FRED key](https://fredaccount.stlouisfed.org/apikeys) in
`.env` for macro context. Everything else works without any key or account.

---

## Web UI

`stokker ui` serves a FastAPI backend plus a dependency-free frontend at
`localhost:8000`. Four tabs:

- **Overview** — current signal per instrument, with bar/COT staleness, plus the
  universe backtest
- **Backtest** — equity vs buy-and-hold, drawdown, position over time
- **Positioning** — COT speculators vs commercials, normalised by open interest
- **Roll diagnostics** — impact metrics and the calendar validation test

Every endpoint delegates to `stokker.research`, so the UI cannot disagree with
the CLI or the tests — `/api/backtest?symbol=ES&signal=tsmom` returns the same
0.346 Sharpe the CLI prints. Charts are hand-drawn SVG (no CDN, works offline),
and the page is dark-first with a light theme.

An older Streamlit dashboard remains at `stokker/app/dashboard.py`
(`pip install -e ".[streamlit]"`), superseded by the above.

---

## Data

| Source | What | Cost | Caveat |
|---|---|---|---|
| **CFTC COT** | Weekly positioning by trader category, 2010– | Free, official | Tuesday positions, published Friday |
| **Yahoo (`=F`)** | Daily OHLCV continuous front-month, 2006– | Free | Unofficial scrape; not licensed for redistribution |
| **FRED** | Macro regime series | Free w/ key | Optional |

15 instruments across equity index, rates, FX, energy, metals and ags. All 15
CFTC contract codes are verified against live CFTC files by
`stokker verify-universe` — they are not trusted from memory.

Yahoo is prototype-grade and will not survive contact with a product. That is
why `data/providers/base.py` defines a `PriceProvider` protocol: swapping in
Databento or IBKR is one new class, not a rewrite.

---

## Findings

Portfolio = equal-weight across 15 instruments, each vol-targeted to 15%.
`hold` is the always-long benchmark. Costs: 1 tick slippage + commission
(`stressed` = 3 ticks + double commission).

| Signal | Costs | CAGR | Vol | **Sharpe** | Max DD | Turnover |
|---|---|---|---|---|---|---|
| Buy and hold | default | 1.8% | 6.6% | **0.298** | −25.7% | 2.6 |
| TS momentum | default | 1.7% | 5.6% | **0.332** | −20.3% | 6.2 |
| TS momentum | stressed | 1.4% | 5.6% | **0.275** | −21.5% | 6.2 |
| TS momentum (scaled) | default | 1.1% | 3.9% | **0.310** | −10.5% | 2.8 |
| EWMA crossover | default | 0.9% | 3.3% | **0.284** | −7.2% | 2.0 |
| COT extreme | default | −0.2% | 1.4% | **−0.153** | −8.0% | 1.6 |
| Commercial flow | default | −0.8% | 2.1% | **−0.363** | −18.1% | 3.3 |
| Blend (mom + COT) | default | 1.0% | 3.2% | **0.321** | −11.8% | 3.9 |

**1. No signal here beats buy-and-hold convincingly.** Momentum's 0.332 versus
hold's 0.298 is well inside noise over 20 years. Treat it as "not disproven",
not as an edge.

**2. The COT hypothesis, as implemented, loses money.** Both positioning signals
are negative, and commercial-flow is the worst thing tested. The premise that
crowded speculative positioning predicts reversion did not survive contact with
data. Worth trying per-sector before abandoning — the commercial/spec contrast
is far weaker for financial futures, where TFF "dealers" are intermediaries, not
hedgers with a crop in the ground.

**3. Costs are not the binding constraint — signal quality is.** Cost drag runs
0.1–0.5%/yr; stressing costs moves Sharpe by ~0.06. There is no version of this
where better execution rescues it.

**4. Diversification is the only thing clearly working.** Individual Sharpes
average ~0.11; the equal-weight portfolio gets 0.33 by cutting vol from 15% to
5.6%. The portfolio construction is doing more than any signal in it.

**5. Low CAGRs are a vol-targeting artefact, not failure.** Portfolio vol is
3–6%, so 1–2% CAGR is consistent. Compare Sharpe, then lever deliberately.

---

## Two things that will silently break a futures backtest

### Roll gaps

Yahoo's `=F` series splice front-month contracts without back-adjusting. The
overnight jump at a roll is a *contract change*, not a return. Differencing the
close series books it as P&L.

**Roll dates come from the delivery calendar, never from price.** The first
implementation detected rolls statistically — any overnight move that was a
large outlier versus trailing vol. Measured against real data it had **zero
precision**. For ES it flagged 21 dates over 20 years and every one was a real
market event: the 2010 flash crash, the 2011 US downgrade, 2015-08-24, Brexit,
Volmageddon, six days of the COVID crash. Not one was a roll. Neutralising them
inflated ES cumulative log return by ~0.6 — roughly 80% compounded — by deleting
the fat left tail.

It cannot work: adjacent equity-index contracts differ by a few basis points,
one to two orders of magnitude below daily vol. Any threshold that catches rolls
catches every crash first.

Roll treatment is not a rounding error. Measured effect on cumulative log return:

| | ES | NQ | ZN | CL | NG | ZC | ZS |
|---|---|---|---|---|---|---|---|
| Naive | +1.80 | +2.85 | −0.01 | **+0.92** | −1.33 | +0.84 | +0.70 |
| Roll-adjusted | +1.61 | +2.63 | −0.09 | **−0.18** | −1.97 | +0.44 | +0.00 |

For crude, ignoring rolls is the difference between "+150%" and "−16%" over 20
years. That gap is the roll yield, and it is why front-month commodity ETFs bleed.

`stokker rolls CL --validate` shifts the calendar ±5 bars and measures the gap
signature at each. It peaks cleanly at offset 0 for CL, ZC and ZS — the markets
whose storage costs make basis large. For equity/rates/FX the per-roll basis is
~0.2%, at the level of daily noise, so the argmax wanders; there the check is
that offset-0 matches theoretical cost-of-carry (ES ~0.22%/quarter ≈ 0.9%/yr,
which it does).

### Lookahead through COT release timing

A COT report published Friday 15:30 ET describes positions as of the **prior
Tuesday**. Aligning signals on the report date leaks three days of hindsight
into every week. `cot.as_of_frame` merges on `release_date`, and
`tests/test_lookahead.py` asserts Tuesday's data is invisible until Friday.

The engine's other guard is `signal.shift(1)` in `backtest/engine.py`: a signal
computed from data through day T is traded at T+1. `test_engine_delay_is_exactly_one_day`
pins this from both sides — an oracle seeing today's return must *not* capture
it, and pre-shifting that oracle one day must capture it exactly.

**If those tests fail, every number this project has produced is void.**

---

## Also handled

**Negative prices are real.** WTI settled at −$37.63 on 2020-04-20. The initial
validator rejected the series as corrupt. It isn't — log returns are simply
undefined across a sign change, so those bars are neutralised and flagged rather
than dropped.

**CFTC schema drift.** Files through ~2016 use `Report_Date_as_MM_DD_YYYY`,
later ones `Report_Date_as_YYYY-MM-DD`. Both are handled by prefix match. The
archive starts in **2010**, not 2006. The disaggregated header contains a
genuine CFTC typo (`Swap__Positions_Short_All`, doubled underscore).

---

## Layout

```
stokker/
  config.py            universe + verified CFTC contract codes
  research.py          orchestration the CLI and app both call
  data/
    store.py           parquet cache + DuckDB query layer
    providers/         base protocol, yahoo, cot, fred
  contracts/roll.py    delivery calendar, roll neutralisation, validation
  backtest/
    costs.py           slippage + commission (no free default)
    engine.py          vectorised, vol-targeted, one-day execution delay
  signals/             base, momentum, cot
  app/
    api.py             FastAPI JSON endpoints over research.py
    static/            SPA: index.html, app.js (SVG charts), style.css
    dashboard.py       legacy Streamlit dashboard
tests/                 34 tests; lookahead + roll regressions
```

`stokker --help` for commands. `pytest -q` to verify.

---

## If you take this further

1. **Test COT per sector.** The pooled negative result may be masking something
   real in ags and energy, where hedgers genuinely hedge.
2. **Walk-forward, not full-sample.** Every number above is in-sample over one
   20-year path. Parameters were not tuned, which helps, but nothing here is
   out-of-sample.
3. **Better data before better signals.** Yahoo daily bars are the ceiling on
   what can be learned. Databento's free tier gives real CME history to validate
   against.
4. **Paper-trade before screens.** Route the signal to an IBKR or Tradovate
   paper account and compare fills against assumed costs for a few months.

Publishing signals to others for money can trigger investment-adviser rules
(SEC/state RIA). Personal research use is unaffected.
