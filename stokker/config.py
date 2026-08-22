"""Project paths and the tradable universe.

The CFTC contract codes below are the join key between price data (yfinance)
and positioning data (COT).  They are transcribed from CFTC report headers and
should be treated as *unverified until checked* -- run `stokker verify-universe`
to confirm each code actually resolves to a contract in the downloaded reports.
A wrong code fails loudly there rather than silently producing an empty series.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "stokker.duckdb"

for _d in (DATA_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# COT comes in two flavours split by asset class.  Financials (equity index,
# rates, FX) are in the "Traders in Financial Futures" report; physical
# commodities are in the "Disaggregated" report.  They have different column
# names and different trader categories, so the report type travels with the
# instrument.
CotReport = Literal["tff", "disaggregated"]
Sector = Literal["equity", "rates", "fx", "energy", "metals", "ags"]


@dataclass(frozen=True)
class Instrument:
    """One futures market we track."""

    symbol: str          # our canonical id
    yahoo: str           # yfinance continuous-contract ticker
    name: str            # human label
    sector: Sector
    cot_code: str        # CFTC contract market code
    cot_report: CotReport
    point_value: float   # USD per 1.0 of price move, per contract
    tick_size: float     # minimum price increment
    commission: float    # USD per contract, round turn (retail ballpark)

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.point_value


UNIVERSE: tuple[Instrument, ...] = (
    # --- equity index (TFF) ---
    Instrument("ES", "ES=F", "E-mini S&P 500", "equity", "13874A", "tff", 50.0, 0.25, 4.0),
    Instrument("NQ", "NQ=F", "E-mini Nasdaq 100", "equity", "209742", "tff", 20.0, 0.25, 4.0),
    Instrument("RTY", "RTY=F", "E-mini Russell 2000", "equity", "239742", "tff", 50.0, 0.10, 4.0),
    # --- rates (TFF) ---
    Instrument("ZN", "ZN=F", "10-Year T-Note", "rates", "043602", "tff", 1000.0, 0.015625, 4.0),
    Instrument("ZB", "ZB=F", "30-Year T-Bond", "rates", "020601", "tff", 1000.0, 0.03125, 4.0),
    # --- fx (TFF) ---
    Instrument("6E", "6E=F", "Euro FX", "fx", "099741", "tff", 125000.0, 0.00005, 4.0),
    Instrument("6J", "6J=F", "Japanese Yen", "fx", "097741", "tff", 12500000.0, 0.0000005, 4.0),
    # --- energy (disaggregated) ---
    Instrument("CL", "CL=F", "WTI Crude Oil", "energy", "067651", "disaggregated", 1000.0, 0.01, 4.0),
    Instrument("NG", "NG=F", "Natural Gas", "energy", "023651", "disaggregated", 10000.0, 0.001, 4.0),
    # --- metals (disaggregated) ---
    Instrument("GC", "GC=F", "Gold", "metals", "088691", "disaggregated", 100.0, 0.10, 4.0),
    Instrument("SI", "SI=F", "Silver", "metals", "084691", "disaggregated", 5000.0, 0.005, 4.0),
    Instrument("HG", "HG=F", "Copper", "metals", "085692", "disaggregated", 25000.0, 0.0005, 4.0),
    # --- ags (disaggregated) ---
    Instrument("ZC", "ZC=F", "Corn", "ags", "002602", "disaggregated", 50.0, 0.25, 4.0),
    Instrument("ZS", "ZS=F", "Soybeans", "ags", "005602", "disaggregated", 50.0, 0.25, 4.0),
    Instrument("ZW", "ZW=F", "Wheat (SRW)", "ags", "001602", "disaggregated", 50.0, 0.25, 4.0),
)

BY_SYMBOL: dict[str, Instrument] = {i.symbol: i for i in UNIVERSE}
BY_YAHOO: dict[str, Instrument] = {i.yahoo: i for i in UNIVERSE}


def get(symbol: str) -> Instrument:
    try:
        return BY_SYMBOL[symbol.upper()]
    except KeyError:
        raise KeyError(
            f"unknown symbol {symbol!r}; known: {', '.join(sorted(BY_SYMBOL))}"
        ) from None


def symbols(sector: Sector | None = None) -> list[str]:
    return [i.symbol for i in UNIVERSE if sector is None or i.sector == sector]
