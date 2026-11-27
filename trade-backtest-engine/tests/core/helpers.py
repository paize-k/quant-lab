"""Small test doubles for the shared engine."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Sequence

from quant_lab.core import BaseStrategy, Bar, Signal
from quant_lab.core.types import OrderType, Position

T0 = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)


def make_bars(ohlc: Sequence[tuple], minutes: int = 1, symbol: str = "X") -> List[Bar]:
    return [Bar(ts=T0 + timedelta(minutes=minutes * (i + 1)), open=o, high=h, low=l, close=c,
                symbol=symbol) for i, (o, h, l, c) in enumerate(ohlc)]


@dataclass
class UnitModel:
    """1 unit = $1 per point, no slippage unless set, flat fee per side."""

    fee: float = 0.0
    slip: float = 0.0
    allow: bool = True

    def entry_allowed(self, bar): return self.allow
    def must_flatten(self, bar): return False
    def fill_price(self, side, price, bar, order_type):
        return price if order_type is OrderType.LIMIT else price + side * self.slip
    def size(self, equity, side, entry, stop, risk, bar): return (equity * risk) / abs(entry - stop)
    def round_quantity(self, q): return float(int(q))
    def fees(self, qty, price, bar): return self.fee
    def pnl(self, entry, exit, qty, side, bar): return (exit - entry) * side * qty
    def carry(self, position: Position, prev, bar): return 0.0


class Script(BaseStrategy):
    """Emits a pre-set signal at given bar indexes."""

    name = "script"
    default_params = {"plan": {}}

    def on_bar(self, bar, state):
        return self.params["plan"].get(state.bar_index, Signal.hold())
