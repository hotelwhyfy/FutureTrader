"""CFTC Commitments of Traders -- free, official, weekly positioning data.

Two reports cover our universe with different trader taxonomies:

  Traders in Financial Futures (TFF), for equity index / rates / FX:
      dealer | asset manager | leveraged money | other reportable | non-reportable
  Disaggregated, for physical commodities:
      producer-merchant | swap dealer | managed money | other reportable | non-reportable

We normalise both to a shared `commercial_net` / `spec_net` pair so a signal can
be written once and applied across sectors.  The mapping is a judgement call and
is documented at `_NORMALISE` below -- it is the one place in this module where
we are interpreting rather than transcribing.

IMPORTANT -- release timing.  A report published Friday 15:30 ET describes
positions as of the *prior Tuesday* close.  `report_date` is that Tuesday.
`release_date` is when you could actually have known it.  Signals must use
`release_date`, or the backtest is lookahead-biased.  See `as_of_frame()`.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import timedelta

import pandas as pd
import requests

from stokker.config import CACHE_DIR, CotReport
from stokker.data.store import STORE

log = logging.getLogger(__name__)

_URLS: dict[CotReport, str] = {
    "tff": "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip",
    "disaggregated": "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip",
}

# Earliest year each report is published in its current form.  Verified by
# probing the CFTC archive -- 2009 and earlier return 404 for both reports.
_FIRST_YEAR: dict[CotReport, int] = {"tff": 2010, "disaggregated": 2010}

_TIMEOUT = 60
_HEADERS = {"User-Agent": "stokker/0.1 (research; contact via github)"}

# Long/short column stems per report.  Note the double underscore in
# "Swap__Positions_Short_All" -- that typo is in the CFTC header itself.
_CATEGORIES: dict[CotReport, dict[str, tuple[str, str]]] = {
    "tff": {
        "dealer": ("Dealer_Positions_Long_All", "Dealer_Positions_Short_All"),
        "asset_mgr": ("Asset_Mgr_Positions_Long_All", "Asset_Mgr_Positions_Short_All"),
        "lev_money": ("Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All"),
        "other_rept": ("Other_Rept_Positions_Long_All", "Other_Rept_Positions_Short_All"),
        "nonrept": ("NonRept_Positions_Long_All", "NonRept_Positions_Short_All"),
    },
    "disaggregated": {
        "prod_merc": ("Prod_Merc_Positions_Long_All", "Prod_Merc_Positions_Short_All"),
        "swap": ("Swap_Positions_Long_All", "Swap__Positions_Short_All"),
        "m_money": ("M_Money_Positions_Long_All", "M_Money_Positions_Short_All"),
        "other_rept": ("Other_Rept_Positions_Long_All", "Other_Rept_Positions_Short_All"),
        "nonrept": ("NonRept_Positions_Long_All", "NonRept_Positions_Short_All"),
    },
}

# How raw categories roll up into the cross-sector view.
#   commercial = entities with underlying physical/balance-sheet exposure
#   spec       = the fast money whose crowding we care about
_NORMALISE: dict[CotReport, dict[str, tuple[str, ...]]] = {
    "tff": {"commercial": ("dealer",), "spec": ("lev_money",)},
    "disaggregated": {"commercial": ("prod_merc", "swap"), "spec": ("m_money",)},
}


def _download(report: CotReport, year: int) -> pd.DataFrame:
    """Fetch and parse one year of one report.  Raw zip is cached on disk."""
    raw_dir = CACHE_DIR / "raw" / "cot"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{report}_{year}.zip"

    if not zip_path.exists():
        url = _URLS[report].format(year=year)
        log.info("downloading %s", url)
        resp = requests.get(url, timeout=_TIMEOUT, headers=_HEADERS)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not names:
            raise ValueError(f"no .txt payload in {zip_path}")
        with zf.open(names[0]) as fh:
            df = pd.read_csv(io.BytesIO(fh.read()), low_memory=False)

    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


def _report_date(df: pd.DataFrame) -> pd.Series:
    """Locate and parse the report-date column.

    CFTC renamed it partway through the archive: files up to ~2016 use
    `Report_Date_as_MM_DD_YYYY`, later ones `Report_Date_as_YYYY-MM-DD`.
    Match on the stable prefix and let pandas infer the layout.
    """
    candidates = [c for c in df.columns if c.startswith("Report_Date_as")]
    if not candidates:
        raise KeyError(
            f"no Report_Date_as* column; CFTC schema changed. Saw: {list(df.columns)[:8]}"
        )
    col = candidates[0]
    parsed = pd.to_datetime(df[col], format="mixed", errors="coerce")
    if parsed.isna().all():
        raise ValueError(f"could not parse any dates from column {col!r}")
    return parsed


def _tidy(df: pd.DataFrame, report: CotReport) -> pd.DataFrame:
    """Raw CFTC columns -> tidy net-position frame."""
    cats = _CATEGORIES[report]
    missing = [c for pair in cats.values() for c in pair if c not in df.columns]
    if missing:
        raise KeyError(f"{report}: CFTC schema changed, missing columns: {missing}")

    out = pd.DataFrame(
        {
            "report_date": _report_date(df),
            "cot_code": df["CFTC_Contract_Market_Code"].astype(str).str.strip(),
            "market": df["Market_and_Exchange_Names"].astype(str).str.strip(),
            "open_interest": pd.to_numeric(df["Open_Interest_All"], errors="coerce"),
        }
    )

    for cat, (lcol, scol) in cats.items():
        lng = pd.to_numeric(df[lcol], errors="coerce")
        shr = pd.to_numeric(df[scol], errors="coerce")
        out[f"{cat}_long"] = lng
        out[f"{cat}_short"] = shr
        out[f"{cat}_net"] = lng - shr

    for bucket, members in _NORMALISE[report].items():
        out[f"{bucket}_net"] = sum(out[f"{m}_net"] for m in members)

    out["report"] = report
    # Positions as of Tuesday close; published the following Friday 15:30 ET.
    # Treat Friday as the first date the information was actionable.
    out["release_date"] = out["report_date"] + timedelta(days=3)
    return out


def fetch(report: CotReport, years: range | list[int], refresh: bool = False) -> pd.DataFrame:
    """Tidy COT for the given years.  Cached per (report, year).

    The current year is always refetched -- it grows a row every week.
    """
    current = pd.Timestamp.today().year
    frames = []
    for year in years:
        if year < _FIRST_YEAR[report]:
            log.warning("%s not published for %d, skipping", report, year)
            continue
        key = f"{report}_{year}"
        stale = refresh or year >= current
        if not stale and STORE.exists("cot", key):
            frames.append(STORE.get("cot", key))
            continue
        try:
            tidy = _tidy(_download(report, year), report)
        except requests.HTTPError as exc:
            log.warning("%s %d unavailable (%s), skipping", report, year, exc)
            continue
        STORE.put("cot", key, tidy)
        frames.append(tidy)

    if not frames:
        raise RuntimeError(f"no COT data retrieved for {report} over {list(years)}")
    return pd.concat(frames, ignore_index=True).sort_values("report_date")


def for_instrument(cot_code: str, report: CotReport, years: range | list[int]) -> pd.DataFrame:
    """COT history for one contract, indexed by report_date."""
    df = fetch(report, years)
    sub = df[df["cot_code"] == cot_code].copy()
    if sub.empty:
        raise ValueError(
            f"no rows for cot_code={cot_code!r} in {report}; "
            f"run `stokker verify-universe` or `stokker search-cot <name>`"
        )
    return sub.set_index("report_date").sort_index()


def as_of_frame(cot: pd.DataFrame, index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    """Reindex weekly COT onto a daily price index without lookahead.

    Each daily bar gets the most recent report whose *release_date* is at or
    before that bar.  This is the guard that keeps Tuesday's positioning out of
    Wednesday's signal.
    """
    src = cot.reset_index()[["release_date", *columns]].copy()
    # Parquet round-trips these as datetime64[ms] while a price index is [us];
    # pandas 3 refuses merge_asof across resolutions, so pin both to [ns].
    src["release_date"] = pd.to_datetime(src["release_date"]).astype("datetime64[ns]")
    src = src.sort_values("release_date")

    target_index = pd.DatetimeIndex(index).as_unit("ns")
    target = pd.DataFrame({"release_date": target_index}).sort_values("release_date")

    merged = pd.merge_asof(target, src, on="release_date", direction="backward")
    return merged.set_index("release_date").reindex(target_index).set_axis(index)


def search(term: str, report: CotReport, year: int | None = None) -> pd.DataFrame:
    """Find contract codes by market name -- for fixing a bad code in config."""
    year = year or pd.Timestamp.today().year - 1
    df = _download(report, year)
    df.columns = [c.strip() for c in df.columns]
    names = df["Market_and_Exchange_Names"].astype(str).str.strip()
    hit = df[names.str.contains(term, case=False, na=False)]
    return (
        hit.groupby(hit["CFTC_Contract_Market_Code"].astype(str).str.strip())
        ["Market_and_Exchange_Names"]
        .first()
        .reset_index()
        .rename(columns={"CFTC_Contract_Market_Code": "cot_code", "Market_and_Exchange_Names": "market"})
    )
