"""Transaction cost model.

Costs are a constructor argument with no free default, because the single most
common way a futures backtest lies is by omitting them.  Every turn of the
position pays: commission (fixed per contract) plus slippage (assumed half the
bid-ask spread, expressed in ticks).

Defaults below are retail ballpark for liquid CME contracts in normal
conditions.  They are optimistic during news, at the open, and for anything
outside the front month -- widen `slippage_ticks` before believing a strategy
that trades often.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stokker.config import Instrument


@dataclass(frozen=True)
class CostModel:
    slippage_ticks: float = 1.0   # ticks paid per side; 1.0 ~= crossing the spread
    commission_mult: float = 1.0  # scale the instrument's per-contract commission

    def cost_per_contract(self, inst: Instrument) -> float:
        """USD cost of trading one contract, one way."""
        slip = self.slippage_ticks * inst.tick_value
        comm = inst.commission * self.commission_mult / 2.0  # round-turn -> one side
        return slip + comm

    def cost_in_return_terms(self, inst: Instrument, price: pd.Series) -> pd.Series:
        """Cost of a full one-contract turn, as a fraction of notional.

        Lets the engine charge costs against a return series rather than
        tracking contract counts, which keeps position sizing continuous.
        """
        notional = price.astype(float) * inst.point_value
        return self.cost_per_contract(inst) / notional


#: Pessimistic preset -- use this to sanity-check anything that looks too good.
STRESSED = CostModel(slippage_ticks=3.0, commission_mult=2.0)
DEFAULT = CostModel()
