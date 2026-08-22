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
Sector = Literal["equity", "rates", "fx", "energy", "metals",
                 "grains", "softs", "livestock"]


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
    # ---------------------------------------------------------- equity index
    Instrument("ES", "ES=F", "E-mini S&P 500", "equity", "13874A", "tff", 50.0, 0.25, 4.0),
    Instrument("NQ", "NQ=F", "E-mini Nasdaq 100", "equity", "209742", "tff", 20.0, 0.25, 4.0),
    Instrument("RTY", "RTY=F", "E-mini Russell 2000", "equity", "239742", "tff", 50.0, 0.10, 4.0),
    Instrument("YM", "YM=F", "E-mini Dow", "equity", "124603", "tff", 5.0, 1.0, 4.0),
    # ----------------------------------------------------------------- rates
    Instrument("ZT", "ZT=F", "2-Year T-Note", "rates", "042601", "tff", 2000.0, 0.0078125, 4.0),
    Instrument("ZF", "ZF=F", "5-Year T-Note", "rates", "044601", "tff", 1000.0, 0.0078125, 4.0),
    Instrument("ZN", "ZN=F", "10-Year T-Note", "rates", "043602", "tff", 1000.0, 0.015625, 4.0),
    Instrument("ZB", "ZB=F", "30-Year T-Bond", "rates", "020601", "tff", 1000.0, 0.03125, 4.0),
    Instrument("UB", "UB=F", "Ultra T-Bond", "rates", "020604", "tff", 1000.0, 0.03125, 4.0),
    # -------------------------------------------------------------------- fx
    Instrument("6A", "6A=F", "Australian Dollar", "fx", "232741", "tff", 100000.0, 0.0001, 4.0),
    Instrument("6B", "6B=F", "British Pound", "fx", "096742", "tff", 62500.0, 0.0001, 4.0),
    Instrument("6C", "6C=F", "Canadian Dollar", "fx", "090741", "tff", 100000.0, 0.00005, 4.0),
    Instrument("6E", "6E=F", "Euro FX", "fx", "099741", "tff", 125000.0, 0.00005, 4.0),
    Instrument("6J", "6J=F", "Japanese Yen", "fx", "097741", "tff", 12500000.0, 0.0000005, 4.0),
    Instrument("6M", "6M=F", "Mexican Peso", "fx", "095741", "tff", 500000.0, 0.00001, 4.0),
    Instrument("6N", "6N=F", "New Zealand Dollar", "fx", "112741", "tff", 100000.0, 0.0001, 4.0),
    Instrument("6S", "6S=F", "Swiss Franc", "fx", "092741", "tff", 125000.0, 0.0001, 4.0),
    # ---------------------------------------------------------------- energy
    Instrument("CL", "CL=F", "WTI Crude Oil", "energy", "067651", "disaggregated", 1000.0, 0.01, 4.0),
    Instrument("BZ", "BZ=F", "Brent Crude Oil", "energy", "06765T", "disaggregated", 1000.0, 0.01, 4.0),
    Instrument("NG", "NG=F", "Natural Gas", "energy", "023651", "disaggregated", 10000.0, 0.001, 4.0),
    Instrument("RB", "RB=F", "RBOB Gasoline", "energy", "111659", "disaggregated", 42000.0, 0.0001, 4.0),
    Instrument("HO", "HO=F", "NY Harbor ULSD", "energy", "022651", "disaggregated", 42000.0, 0.0001, 4.0),
    # ---------------------------------------------------------------- metals
    Instrument("GC", "GC=F", "Gold", "metals", "088691", "disaggregated", 100.0, 0.10, 4.0),
    Instrument("SI", "SI=F", "Silver", "metals", "084691", "disaggregated", 5000.0, 0.005, 4.0),
    Instrument("HG", "HG=F", "Copper", "metals", "085692", "disaggregated", 25000.0, 0.0005, 4.0),
    Instrument("PL", "PL=F", "Platinum", "metals", "076651", "disaggregated", 50.0, 0.10, 4.0),
    Instrument("PA", "PA=F", "Palladium", "metals", "075651", "disaggregated", 100.0, 0.10, 4.0),
    # ---------------------------------------------------------------- grains
    Instrument("ZC", "ZC=F", "Corn", "grains", "002602", "disaggregated", 50.0, 0.25, 4.0),
    Instrument("ZS", "ZS=F", "Soybeans", "grains", "005602", "disaggregated", 50.0, 0.25, 4.0),
    Instrument("ZW", "ZW=F", "Wheat (SRW)", "grains", "001602", "disaggregated", 50.0, 0.25, 4.0),
    Instrument("KE", "KE=F", "Wheat (HRW)", "grains", "001612", "disaggregated", 50.0, 0.25, 4.0),
    Instrument("ZL", "ZL=F", "Soybean Oil", "grains", "007601", "disaggregated", 600.0, 0.01, 4.0),
    Instrument("ZM", "ZM=F", "Soybean Meal", "grains", "026603", "disaggregated", 100.0, 0.10, 4.0),
    Instrument("ZO", "ZO=F", "Oats", "grains", "004603", "disaggregated", 50.0, 0.25, 4.0),
    Instrument("ZR", "ZR=F", "Rough Rice", "grains", "039601", "disaggregated", 2000.0, 0.005, 4.0),
    # ----------------------------------------------------------------- softs
    Instrument("KC", "KC=F", "Coffee C", "softs", "083731", "disaggregated", 375.0, 0.05, 4.0),
    Instrument("CT", "CT=F", "Cotton No. 2", "softs", "033661", "disaggregated", 500.0, 0.01, 4.0),
    Instrument("CC", "CC=F", "Cocoa", "softs", "073732", "disaggregated", 10.0, 1.0, 4.0),
    Instrument("SB", "SB=F", "Sugar No. 11", "softs", "080732", "disaggregated", 1120.0, 0.01, 4.0),
    Instrument("OJ", "OJ=F", "Orange Juice", "softs", "040701", "disaggregated", 150.0, 0.05, 4.0),
    # ------------------------------------------------------------- livestock
    Instrument("LE", "LE=F", "Live Cattle", "livestock", "057642", "disaggregated", 400.0, 0.025, 4.0),
    Instrument("HE", "HE=F", "Lean Hogs", "livestock", "054642", "disaggregated", 400.0, 0.025, 4.0),
    Instrument("GF", "GF=F", "Feeder Cattle", "livestock", "061641", "disaggregated", 500.0, 0.025, 4.0),
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
