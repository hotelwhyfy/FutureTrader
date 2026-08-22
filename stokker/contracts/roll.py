"""Contract rolls, handled by calendar rather than by guesswork.

WHY THIS MODULE WAS REWRITTEN -- read before "improving" it.

The first version detected rolls statistically: flag any overnight gap that is a
large outlier versus trailing volatility.  Measured against 20 years of real
data it had *zero* precision.  Every date it flagged for ES was a genuine market
event -- the 2010 flash crash, the 2011 US downgrade, 2015-08-24, Brexit,
Volmageddon, and six days of the COVID crash.  Not one was a roll.  Neutralising
them inflated ES cumulative log return by ~0.6 (about 80% compounded), because
the "fix" was quietly deleting the fat left tail.

The reason it cannot work: Yahoo's `=F` series are raw front-month splices (ES=F
in May 2010 reads ~1164, matching spot SPX, so no back-adjustment is applied),
but the basis between adjacent equity-index contracts is a few basis points --
one or two orders of magnitude below normal daily volatility.  A roll gap is
invisible against that noise, while a crash is not.  Any threshold that catches
rolls catches every crash first.

So rolls are now derived from each contract's *delivery calendar*: deterministic,
independent of price, and incapable of confusing a selloff for a contract change.

    roll_dates      estimated roll date per delivery cycle
    roll_flags      the bar in a price series where each roll lands
    safe_returns    <- the function to use; neutralises rolls + undefined bars
    price_jumps     DIAGNOSTIC ONLY. Large moves. Mostly real events, NOT rolls.
    back_adjust     Panama-adjusted levels for charting

CAVEATS, stated rather than buried.  These dates are exchange conventions
computed with plain business days and no holiday calendar, so they can be off by
a day around holidays.  Yahoo may also roll on volume rather than on the
official date.  Both are acceptable because the measured impact of roll
neutralisation on these markets is small (see `impact_report`) -- if you move to
per-contract data with true expiries, replace `roll_dates` and everything
downstream is exact without further change.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from stokker.config import Instrument

log = logging.getLogger(__name__)

#: Delivery months per symbol, as month numbers.  Sourced from exchange contract
#: specs; falls back to a sector default for anything unlisted.
_DELIVERY_MONTHS: dict[str, set[int]] = {
    # equity index + rates + fx: quarterly
    "ES": {3, 6, 9, 12}, "NQ": {3, 6, 9, 12}, "RTY": {3, 6, 9, 12}, "YM": {3, 6, 9, 12},
    "ZT": {3, 6, 9, 12}, "ZF": {3, 6, 9, 12}, "ZN": {3, 6, 9, 12},
    "ZB": {3, 6, 9, 12}, "UB": {3, 6, 9, 12},
    "6A": {3, 6, 9, 12}, "6B": {3, 6, 9, 12}, "6C": {3, 6, 9, 12}, "6E": {3, 6, 9, 12},
    "6J": {3, 6, 9, 12}, "6M": {3, 6, 9, 12}, "6N": {3, 6, 9, 12}, "6S": {3, 6, 9, 12},
    # energy: monthly
    "CL": set(range(1, 13)), "BZ": set(range(1, 13)), "NG": set(range(1, 13)),
    "RB": set(range(1, 13)), "HO": set(range(1, 13)),
    # metals
    "GC": {2, 4, 6, 8, 10, 12}, "SI": {3, 5, 7, 9, 12}, "HG": {3, 5, 7, 9, 12},
    "PL": {1, 4, 7, 10}, "PA": {3, 6, 9, 12},
    # grains + oilseeds
    "ZC": {3, 5, 7, 9, 12}, "ZS": {1, 3, 5, 7, 8, 9, 11}, "ZW": {3, 5, 7, 9, 12},
    "KE": {3, 5, 7, 9, 12}, "ZL": {1, 3, 5, 7, 8, 9, 10, 12},
    "ZM": {1, 3, 5, 7, 8, 9, 10, 12}, "ZO": {3, 5, 7, 9, 12},
    "ZR": {1, 3, 5, 7, 9, 11},
    # softs
    "KC": {3, 5, 7, 9, 12}, "CT": {3, 5, 7, 10, 12}, "CC": {3, 5, 7, 9, 12},
    "SB": {3, 5, 7, 10}, "OJ": {1, 3, 5, 7, 9, 11},
    # livestock -- these expire INSIDE the delivery month, unlike everything above
    "LE": {2, 4, 6, 8, 10, 12}, "HE": {2, 4, 5, 6, 7, 8, 10, 12},
    "GF": {1, 3, 4, 5, 8, 9, 10, 11},
}

_SECTOR_DEFAULT_MONTHS: dict[str, set[int]] = {
    "equity": {3, 6, 9, 12}, "rates": {3, 6, 9, 12}, "fx": {3, 6, 9, 12},
    "energy": set(range(1, 13)), "metals": {2, 4, 6, 8, 10, 12},
    "grains": {3, 5, 7, 9, 12}, "softs": {3, 5, 7, 9, 12},
    "livestock": {2, 4, 6, 8, 10, 12},
}

#: How each symbol's roll date is computed relative to its delivery month.
#: See `_ROLL_RULES` for what each name means.
_RULE_BY_SYMBOL: dict[str, str] = {
    "ES": "third_friday", "NQ": "third_friday", "RTY": "third_friday", "YM": "third_friday",
    "ZT": "prior_month_last_bd", "ZF": "prior_month_last_bd", "ZN": "prior_month_last_bd",
    "ZB": "prior_month_last_bd", "UB": "prior_month_last_bd",
    "6A": "third_wednesday_less_2bd", "6B": "third_wednesday_less_2bd",
    "6C": "third_wednesday_less_2bd", "6E": "third_wednesday_less_2bd",
    "6J": "third_wednesday_less_2bd", "6M": "third_wednesday_less_2bd",
    "6N": "third_wednesday_less_2bd", "6S": "third_wednesday_less_2bd",
    "CL": "prior_month_25th_less_3bd",
    # Brent settles against the month TWO months ahead, unlike WTI.
    "BZ": "two_months_prior_last_bd",
    "NG": "delivery_start_less_3bd",
    "RB": "prior_month_last_bd", "HO": "prior_month_last_bd",
    "GC": "prior_month_3rd_last_bd", "SI": "prior_month_3rd_last_bd",
    "HG": "prior_month_3rd_last_bd", "PL": "prior_month_3rd_last_bd",
    "PA": "prior_month_3rd_last_bd",
    "ZC": "prior_month_last_bd", "ZS": "prior_month_last_bd", "ZW": "prior_month_last_bd",
    "KE": "prior_month_last_bd", "ZL": "prior_month_last_bd", "ZM": "prior_month_last_bd",
    "ZO": "prior_month_last_bd", "ZR": "prior_month_last_bd",
    "KC": "prior_month_last_bd", "CT": "prior_month_last_bd", "CC": "prior_month_last_bd",
    "SB": "prior_month_last_bd", "OJ": "prior_month_last_bd",
    # Livestock contracts trade INTO their delivery month rather than expiring
    # before it, so the roll lands inside the month, not ahead of it.
    "LE": "delivery_month_last_bd",
    "HE": "delivery_month_10th_bd",
    "GF": "delivery_month_last_thursday",
}


def _third_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    """Third given weekday of a month (weekday: Mon=0 .. Fri=4)."""
    first = pd.Timestamp(year=year, month=month, day=1)
    return first + pd.offsets.WeekOfMonth(week=2, weekday=weekday)


def _delivery_start(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1)


def _roll_for_cycle(rule: str, year: int, month: int) -> pd.Timestamp:
    """Roll date for the contract delivering in (year, month)."""
    start = _delivery_start(year, month)
    if rule == "third_friday":
        return _third_weekday(year, month, 4)
    if rule == "third_wednesday_less_2bd":
        return _third_weekday(year, month, 2) - pd.offsets.BDay(2)
    if rule == "prior_month_last_bd":
        # One business day before delivery month begins.
        return start - pd.offsets.BDay(1)
    if rule == "prior_month_3rd_last_bd":
        return start - pd.offsets.BDay(3)
    if rule == "delivery_start_less_3bd":
        return start - pd.offsets.BDay(3)
    if rule == "prior_month_25th_less_3bd":
        prior = start - pd.offsets.MonthBegin(1)
        return prior.replace(day=25) - pd.offsets.BDay(3)
    if rule == "two_months_prior_last_bd":
        # Last business day of the month two months before delivery (Brent).
        return (start - pd.offsets.MonthBegin(1)) - pd.offsets.BDay(1)
    if rule == "delivery_month_last_bd":
        return (start + pd.offsets.MonthBegin(1)) - pd.offsets.BDay(1)
    if rule == "delivery_month_10th_bd":
        # BDay(0) rolls forward to the first business day if the 1st is a weekend.
        return start + pd.offsets.BDay(0) + pd.offsets.BDay(9)
    if rule == "delivery_month_last_thursday":
        last = start + pd.offsets.MonthEnd(0)
        return last - pd.Timedelta(days=(last.dayofweek - 3) % 7)
    raise ValueError(f"unknown roll rule {rule!r}")


def roll_dates(inst: Instrument, start: str | pd.Timestamp,
               end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Estimated roll dates for one instrument over a date range."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    months = _DELIVERY_MONTHS.get(inst.symbol) or _SECTOR_DEFAULT_MONTHS[inst.sector]
    rule = _RULE_BY_SYMBOL.get(inst.symbol, "prior_month_last_bd")

    out: list[pd.Timestamp] = []
    # Widen a year each side so cycles straddling the boundary are included.
    for year in range(start.year - 1, end.year + 2):
        for month in sorted(months):
            d = _roll_for_cycle(rule, year, month)
            if start <= d <= end:
                out.append(d)
    return pd.DatetimeIndex(sorted(set(out))).sort_values()


def roll_flags(bars: pd.DataFrame, inst: Instrument) -> pd.Series:
    """Boolean series marking the bar on which each roll takes effect.

    A roll date may fall on a holiday or weekend, so it is snapped forward to
    the first available trading bar.  At most one bar per roll cycle is flagged.
    """
    idx = pd.DatetimeIndex(bars.index)
    if len(idx) == 0:
        return pd.Series(dtype=bool, name="is_roll")

    rolls = roll_dates(inst, idx.min(), idx.max())
    flags = pd.Series(False, index=idx, name="is_roll")
    if len(rolls) == 0:
        return flags

    # searchsorted maps each roll date to the first bar at or after it.
    positions = idx.searchsorted(rolls, side="left")
    positions = positions[positions < len(idx)]
    flags.iloc[np.unique(positions)] = True

    expected = len(rolls)
    log.debug("%s: %d roll dates -> %d flagged bars", inst.symbol, expected, int(flags.sum()))
    return flags


def invalid_price_bars(bars: pd.DataFrame) -> pd.Series:
    """Bars where a log return cannot be defined.

    True when this close or the prior close is <= 0.  Both sides matter: the
    move *into* a negative print and the move back *out* of it are equally
    undefined in log space.  WTI in April 2020 is the canonical case.
    """
    close = bars["close"].astype(float)
    bad = close <= 0
    return (bad | bad.shift(1).fillna(False)).rename("invalid_price")


def safe_returns(bars: pd.DataFrame, inst: Instrument) -> pd.Series:
    """Daily log returns with roll bars and undefined bars neutralised to zero.

    This is the return series the backtester consumes.  Two classes are zeroed:

      * calendar roll bars -- the printed change spans two different contracts,
        so it is not a return anyone could have earned.
      * bars where price is non-positive, where log returns are undefined.

    Zeroing a roll bar does discard whatever real move happened that day, which
    is a genuine cost of using spliced continuous data.  It affects ~4-12 bars a
    year; `impact_report` quantifies it per instrument so the trade-off is
    visible rather than assumed.
    """
    close = bars["close"].astype(float)
    safe_close = close.where(close > 0)
    rets = np.log(safe_close / safe_close.shift(1))

    rolls = roll_flags(bars, inst)
    invalid = invalid_price_bars(bars)
    if invalid.any():
        log.warning(
            "%s: %d bar(s) with undefined returns from non-positive prices, neutralised",
            inst.symbol, int(invalid.sum()),
        )
    return rets.where(~(rolls | invalid), 0.0).fillna(0.0).rename(f"{inst.symbol}_ret")


def price_jumps(bars: pd.DataFrame, z_threshold: float = 6.0,
                lookback: int = 60) -> pd.Series:
    """DIAGNOSTIC ONLY -- large moves relative to trailing volatility.

    These are NOT rolls.  Empirically they are crashes, gap opens, and
    macro shocks, and neutralising them is data laundering.  Kept because it is
    genuinely useful to list an instrument's worst days, and because naming it
    honestly is the best defence against someone wiring it back into
    `safe_returns`.
    """
    close = bars["close"].astype(float).where(lambda c: c > 0)
    gap = np.log(close / close.shift(1))

    med = gap.rolling(lookback, min_periods=20).median()
    mad = (gap - med).abs().rolling(lookback, min_periods=20).median()
    # 1.4826 scales MAD to a consistent sigma estimate for normal data.
    scale = (mad * 1.4826).replace(0.0, np.nan)
    z = (gap - med).abs() / scale
    return (z > z_threshold).fillna(False).rename("price_jump")


def back_adjust(bars: pd.DataFrame, inst: Instrument) -> pd.DataFrame:
    """Panama (difference) adjusted OHLC, for charting and level indicators.

    Each roll gap is subtracted from all prior history, so the latest price is
    the real current price and older levels are shifted.  Deep history is
    therefore fictional in absolute terms -- the known cost of difference
    adjustment.  Do NOT compute returns from this; use `safe_returns`.
    """
    rolls = roll_flags(bars, inst)
    close = bars["close"].astype(float)
    gaps = (close - close.shift(1)).where(rolls, 0.0).fillna(0.0)

    # Adjustment for each bar is the total of all gaps occurring after it.
    offset = gaps.sum() - gaps.cumsum()

    out = bars.copy()
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col].astype(float) - offset
    out["roll_offset"] = offset
    out["is_roll"] = rolls
    return out


def roll_report(bars: pd.DataFrame, inst: Instrument) -> pd.DataFrame:
    """Table of each roll bar and the gap booked there.

    Sanity check: the count should match the contract's delivery cycle (4/yr for
    quarterly, ~12/yr for monthly).  Wildly more means the calendar is wrong.
    """
    rolls = roll_flags(bars, inst)
    close = bars["close"].astype(float)
    hits = bars.index[rolls]
    return pd.DataFrame(
        {
            "date": hits,
            "prev_close": close.shift(1).loc[hits].to_numpy(),
            "close": close.loc[hits].to_numpy(),
            "gap": (close - close.shift(1)).loc[hits].to_numpy(),
            "gap_pct": (close / close.shift(1) - 1).loc[hits].to_numpy() * 100,
        }
    ).set_index("date")


def impact_report(bars: pd.DataFrame, inst: Instrument) -> dict[str, float]:
    """How much roll neutralisation actually changes the return series.

    Report this rather than trusting it.  If `cum_delta` is large, the roll
    treatment is driving results and deserves better data.
    """
    close = bars["close"].astype(float).where(lambda c: c > 0)
    naive = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    safe = safe_returns(bars, inst)
    years = max(len(bars) / 252.0, 1e-9)
    rolls = roll_flags(bars, inst)
    return {
        "rolls_per_year": float(rolls.sum() / years),
        "naive_cum_logret": float(naive.sum()),
        "safe_cum_logret": float(safe.sum()),
        "cum_delta": float(safe.sum() - naive.sum()),
        "mean_abs_roll_gap_pct": float(
            (np.exp(naive[rolls]) - 1).abs().mean() * 100 if rolls.any() else 0.0
        ),
    }


def validate_calendar(bars: pd.DataFrame, inst: Instrument,
                      max_offset: int = 5) -> pd.DataFrame:
    """Shift the roll calendar +/- N bars and measure the gap signature at each.

    The logic: a real roll bar contains the basis between adjacent contracts, so
    the mean overnight gap on true roll bars should stand out against its
    neighbours.  If the calendar is right, |mean gap| peaks at offset 0.

    Measured over 2006-2026 this peaks cleanly at 0 for CL, ZC and ZS -- the
    markets whose storage costs make basis large.  For ES/NQ/ZN/6E the per-roll
    basis is only ~0.2%, at the level of daily noise, so the argmax wanders; for
    those, the offset-0 value matching theoretical cost-of-carry (ES ~0.22% per
    quarterly roll, i.e. ~0.9%/yr) is the meaningful cross-check, not the argmax.

    Low power on low-basis markets is a limitation of this test, not evidence
    against the calendar.
    """
    close = bars["close"].astype(float).where(lambda c: c > 0)
    gap = np.log(close / close.shift(1)).to_numpy()
    base = roll_flags(bars, inst).to_numpy()

    rows = []
    for off in range(-max_offset, max_offset + 1):
        shifted = np.roll(base, off)
        vals = gap[shifted]
        vals = vals[np.isfinite(vals)]
        n = len(vals)
        mean = float(vals.mean()) if n else np.nan
        se = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        rows.append({
            "offset": off,
            "n": n,
            "mean_gap_pct": mean * 100,
            "stderr_pct": se * 100,
            "t_stat": mean / se if se else np.nan,
        })
    return pd.DataFrame(rows).set_index("offset")
