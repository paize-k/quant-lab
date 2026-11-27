"""Futures friction model and backtest entry point.

``FuturesExecutionModel`` plugs CME futures mechanics into the shared
``core.engine.Backtester``:

* prices rounded to the tick; market and stop orders pay ``slippage_ticks``
  (at least 1 tick) per side, limit orders none;
* commission + exchange + NFA + clearing fees per contract per side;
* sizing = floor(equity * risk / (stop ticks * tick value)), capped by the
  margin available (intraday margin while RTH-flat trading, else initial);
* entries only inside the chosen session, and an optional auto-flatten N
  minutes before the RTH close / Globex maintenance break;
* holding a position across a continuous-contract roll costs a round turn of
  fees plus slippage (you close the old month and reopen the new one).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from quant_lab.core.engine import BacktestConfig, BacktestResult, Backtester, ScaleOutRule
from quant_lab.core.strategy import BaseStrategy
from quant_lab.core.types import Bar, OrderType, Position
from quant_lab.futures.config.contracts import FuturesContractSpec
from quant_lab.futures.config.risk import FuturesRiskLimits
from quant_lab.futures.config.sessions import CHICAGO, FuturesSessions, get_sessions


@dataclass
class FuturesExecutionModel:
    spec: FuturesContractSpec
    sessions: Optional[FuturesSessions] = None   # None = no session rules (e.g. daily bars)
    rth_only: bool = True
    auto_flatten: bool = True
    intraday: bool = True                        # size against day-trade margin
    max_contracts: Optional[int] = None
    margin_buffer: float = 1.0                   # use only this share of equity for margin
    roll_count: int = field(default=0, init=False)

    # ---- sessions --------------------------------------------------------
    def entry_allowed(self, bar: Bar) -> bool:
        if self.sessions is None:
            return True
        if not self.sessions.is_tradable(bar.ts, rth_only=self.rth_only):
            return False
        return not (self.auto_flatten and self.sessions.should_flatten(bar.ts, self.rth_only))

    def must_flatten(self, bar: Bar) -> bool:
        if self.sessions is None or not self.auto_flatten:
            return False
        window = self.sessions.rth if self.rth_only else self.sessions.globex
        # Flatten in the last N minutes, and also if we're already outside the
        # session (data gap past the close).
        return self.sessions.should_flatten(bar.ts, self.rth_only) or not window.contains(bar.ts)

    # ---- prices and costs -----------------------------------------------
    def fill_price(self, side: int, price: float, bar: Bar, order_type: OrderType) -> float:
        tick = self.spec.tick_size
        px = self.spec.round_price(price)
        if order_type is OrderType.LIMIT:
            return px
        slip = max(self.spec.slippage_ticks, 1.0) * tick
        return self.spec.round_price(px + side * slip)

    def fees(self, qty: float, price: float, bar: Bar) -> float:
        return self.spec.fees_per_side * abs(qty)

    def pnl(self, entry_price: float, exit_price: float, qty: float, side: int, bar: Bar) -> float:
        return self.spec.pnl(entry_price, exit_price, qty, side)

    def round_quantity(self, qty: float) -> float:
        return float(math.floor(qty + 1e-9))

    def size(self, equity: float, side: int, entry_price: float, stop_price: float,
             risk_fraction: float, bar: Bar) -> float:
        stop_ticks = math.ceil(abs(entry_price - stop_price) / self.spec.tick_size - 1e-9)
        if stop_ticks <= 0:
            return 0.0
        n = self.spec.contracts_for_risk(equity, risk_fraction, stop_ticks,
                                         available_margin=equity * self.margin_buffer,
                                         intraday=self.intraday)
        if self.max_contracts is not None:
            n = min(n, self.max_contracts)
        return float(n)

    # ---- rolls -----------------------------------------------------------
    def carry(self, position: Position, prev_bar: Bar, bar: Bar) -> float:
        if position.is_flat or not prev_bar.symbol or prev_bar.symbol == bar.symbol:
            return 0.0
        self.roll_count += 1
        q = position.quantity
        return -(self.spec.round_trip_fees(q) + self.spec.slippage_cost(q, sides=2))


def run_backtest(
    strategy: BaseStrategy,
    bars: Sequence[Bar],
    spec: FuturesContractSpec,
    initial_equity: float = 100_000.0,
    limits: Optional[FuturesRiskLimits] = None,
    use_sessions: bool = True,
    rth_only: bool = True,
    scale_out: bool = True,
) -> BacktestResult:
    """Run ``strategy`` on futures ``bars`` with ``spec``'s friction.

    Risk per trade, the daily-loss and loss-streak breakers, auto-flatten and
    the 1.5R scale-out with breakeven + 1 tick all come from ``limits``
    (``FuturesRiskLimits()`` by default). Pass ``use_sessions=False`` for
    daily bars, where session times don't apply.
    """
    limits = limits or FuturesRiskLimits()
    sessions = get_sessions(spec.sessions_key) if use_sessions else None
    if sessions is not None:
        sessions = FuturesSessions(sessions.globex, sessions.rth, limits.flatten_minutes_before_close)
    model = FuturesExecutionModel(spec, sessions=sessions, rth_only=rth_only,
                                  auto_flatten=limits.auto_flatten_before_close and use_sessions,
                                  intraday=use_sessions and limits.auto_flatten_before_close,
                                  max_contracts=limits.max_contracts_per_symbol)
    cfg = BacktestConfig(
        initial_equity=initial_equity,
        risk_per_trade=limits.risk_per_trade,
        max_daily_loss=limits.max_daily_loss,
        max_consecutive_losses=limits.max_consecutive_losses,
        scale_out=ScaleOutRule(limits.scale_out_r, limits.scale_out_fraction,
                               limits.breakeven_offset_ticks * spec.tick_size) if scale_out else None,
        day_key_tz=CHICAGO,
    )
    return Backtester(strategy, model, cfg).run(bars)
