"""Core data models shared by the backtester, WFO and the live bots.

A ``Bar`` is always a *closed* bar. Strategies receive bar ``t`` only after it
has closed and answer with a ``Signal``; the engine executes that signal no
earlier than the open of bar ``t+1`` (or via a verified intra-bar stop/limit
trigger on ``t+1``). That is the zero look-ahead rule, and it is enforced by
the engine loop rather than trusted to each strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Optional


class Side(int, Enum):
    """Direction of a position or order. The int value doubles as a sign."""

    SHORT = -1
    FLAT = 0
    LONG = 1


class OrderType(str, Enum):
    MARKET = "market"   # filled at next bar's open (plus engine slippage)
    STOP = "stop"       # filled when next bar trades through ``entry_price``
    LIMIT = "limit"     # filled when next bar trades at/through ``entry_price``


class SignalAction(str, Enum):
    HOLD = "hold"               # do nothing
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT = "exit"               # flatten the current position
    MODIFY = "modify"           # move stop/target on the open position


@dataclass(frozen=True, slots=True)
class Bar:
    """One closed OHLCV bar.

    ``ts`` is the bar's *close* time and must be timezone-aware so futures
    (America/Chicago) and forex (UTC) engines never mix naive timestamps.

    Bid/ask fields are optional. The futures engine generally trades on the
    mid/last OHLC with modelled slippage; the forex engine fills shorts on
    the bid and covers on the ask, so it populates them when the data has
    them (or models them from a spread).
    """

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    symbol: str = ""
    bid_open: Optional[float] = None
    bid_high: Optional[float] = None
    bid_low: Optional[float] = None
    bid_close: Optional[float] = None
    ask_open: Optional[float] = None
    ask_high: Optional[float] = None
    ask_low: Optional[float] = None
    ask_close: Optional[float] = None

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None or self.ts.utcoffset() is None:
            raise ValueError("Bar.ts must be timezone-aware")
        if self.high < self.low:
            raise ValueError(f"Bar high {self.high} is below low {self.low}")
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError("Bar open/close must lie inside [low, high]")

    @property
    def has_quotes(self) -> bool:
        """True when full bid and ask OHLC are present."""
        return None not in (
            self.bid_open, self.bid_high, self.bid_low, self.bid_close,
            self.ask_open, self.ask_high, self.ask_low, self.ask_close,
        )


@dataclass(frozen=True, slots=True)
class Signal:
    """What a strategy wants done, decided on a closed bar.

    Prices are in instrument price units. Sizing is *not* decided here: the
    strategy states its protective stop and the engine's sizer turns
    ``risk_fraction`` of equity into contracts or units, so the same class
    sizes correctly in backtest, WFO and live trading.
    """

    action: SignalAction = SignalAction.HOLD
    order_type: OrderType = OrderType.MARKET
    entry_price: Optional[float] = None   # required for STOP/LIMIT orders
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_fraction: Optional[float] = None  # override of the engine default, e.g. 0.005
    tag: str = ""                          # which sub-strategy produced it (hybrids)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.order_type is not OrderType.MARKET and self.entry_price is None:
            raise ValueError(f"{self.order_type.value} orders need an entry_price")
        if self.risk_fraction is not None and not (0.0 < self.risk_fraction < 1.0):
            raise ValueError("risk_fraction must be in (0, 1)")
        if self.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT):
            ref = self.entry_price
            if ref is not None and self.stop_loss is not None:
                long = self.action is SignalAction.ENTER_LONG
                if (long and self.stop_loss >= ref) or (not long and self.stop_loss <= ref):
                    raise ValueError("stop_loss is on the wrong side of entry_price")

    @property
    def is_entry(self) -> bool:
        return self.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT)

    @property
    def side(self) -> Side:
        if self.action is SignalAction.ENTER_LONG:
            return Side.LONG
        if self.action is SignalAction.ENTER_SHORT:
            return Side.SHORT
        return Side.FLAT

    @classmethod
    def hold(cls) -> "Signal":
        return cls()


@dataclass(frozen=True, slots=True)
class Position:
    """Read-only view of the open position handed to strategies."""

    side: Side = Side.FLAT
    quantity: float = 0.0          # contracts (futures) or units (forex), always >= 0
    entry_price: Optional[float] = None
    entry_ts: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def is_flat(self) -> bool:
        return self.side is Side.FLAT or self.quantity == 0
