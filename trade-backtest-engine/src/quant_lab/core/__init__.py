"""Shared, asset-class-agnostic primitives.

Everything in ``core`` must stay generic: bar/signal data models, the
``BaseStrategy`` contract, time-window math, and position-sizing math.
Nothing here may import from ``quant_lab.futures`` or ``quant_lab.forex``,
and no broker adapters, fee models, session calendars or live-state stores
live here (those belong to each engine).
"""

from quant_lab.core.types import Bar, Side, OrderType, SignalAction, Signal, Position
from quant_lab.core.strategy import BaseStrategy, StrategyState
from quant_lab.core.instrument import InstrumentSpec
from quant_lab.core.time import TimeWindow
from quant_lab.core.sizing import floor_to_step, size_by_risk
from quant_lab.core.data import bars_from_dataframe, normalize_ohlcv

__all__ = [
    "Bar", "Side", "OrderType", "SignalAction", "Signal", "Position",
    "BaseStrategy", "StrategyState", "InstrumentSpec", "TimeWindow",
    "floor_to_step", "size_by_risk", "bars_from_dataframe", "normalize_ohlcv",
]
