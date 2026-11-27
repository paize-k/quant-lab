"""Shared event-driven backtest loop.

The loop is asset-class agnostic. Everything about *how* an instrument
trades (fill prices, slippage or spread, fees, sizing, sessions, carry,
contract rolls) comes from an ``ExecutionModel`` supplied by the futures or
forex engine.

Timing, which is what removes look-ahead bias:

1. Bar ``t`` closes. The engine marks to market, applies session flattening,
   then calls ``strategy.on_bar(bar_t, state)``.
2. The resulting order is *pending*. It can only fill on bar ``t+1``:
   market orders at ``t+1`` open, stop/limit orders only if ``t+1`` trades
   through the price (gaps fill at the open, which is worse for you).
3. Protective stops and targets are checked against each bar's range. If a
   bar touches both, the stop is assumed to fill first (pessimistic). On the
   bar a stop/limit entry fills, only the stop is checked, because we cannot
   know whether the high or the low came after the fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Protocol

import pandas as pd

from quant_lab.core.strategy import BaseStrategy, StrategyState
from quant_lab.core.types import Bar, OrderType, Position, Side, Signal, SignalAction


class ExecutionModel(Protocol):
    """What an engine (futures/forex) must provide to the shared loop."""

    def entry_allowed(self, bar: Bar) -> bool:
        """May an order decided on this closed bar be placed for the next bar?"""
        ...

    def must_flatten(self, bar: Bar) -> bool:
        """Should any open position be closed at this bar's close?"""
        ...

    def fill_price(self, side: int, price: float, bar: Bar, order_type: OrderType) -> float:
        """Execution price for a trade in direction ``side`` (+1 buy, -1 sell)
        at reference ``price`` on ``bar``, including slippage or spread."""
        ...

    def size(self, equity: float, side: int, entry_price: float, stop_price: float,
             risk_fraction: float, bar: Bar) -> float:
        """Quantity to trade so that a stop-out loses about ``risk_fraction``."""
        ...

    def round_quantity(self, qty: float) -> float:
        ...

    def fees(self, qty: float, price: float, bar: Bar) -> float:
        """Fees for one side (entry *or* exit) of ``qty``, in account currency."""
        ...

    def pnl(self, entry_price: float, exit_price: float, qty: float, side: int, bar: Bar) -> float:
        ...

    def carry(self, position: Position, prev_bar: Bar, bar: Bar) -> float:
        """Account-currency cash flow for holding ``position`` from
        ``prev_bar`` to ``bar``: FX swaps, futures roll costs. Negative = cost."""
        ...


@dataclass
class ScaleOutRule:
    """Take ``fraction`` off at ``r_multiple`` R and move the stop to
    breakeven plus ``breakeven_offset`` (in price units, in your favour)."""

    r_multiple: float = 1.5
    fraction: float = 0.5
    breakeven_offset: float = 0.0


@dataclass
class BacktestConfig:
    initial_equity: float = 100_000.0
    risk_per_trade: float = 0.005
    max_daily_loss: Optional[float] = None          # e.g. 0.02 -> no entries rest of day at -2%
    max_consecutive_losses: Optional[int] = None    # lock out after N losses in a row ...
    lockout_resets_daily: bool = True               # ... until the next trading day
    scale_out: Optional[ScaleOutRule] = None
    day_key_tz: str = "UTC"                         # time zone that defines a "trading day"


@dataclass
class Trade:
    symbol: str
    side: int
    qty: float
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    gross_pnl: float
    fees: float
    slippage: float          # cost of slippage/spread vs. reference prices, already inside gross_pnl
    carry: float
    initial_risk: float      # money at risk at entry (qty * stop distance)
    exit_reason: str
    tag: str = ""

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees + self.carry

    @property
    def r_multiple(self) -> float:
        return self.net_pnl / self.initial_risk if self.initial_risk > 0 else float("nan")


@dataclass
class BacktestResult:
    equity: pd.Series                 # mark-to-market equity at each bar close
    trades: List[Trade]
    initial_equity: float
    strategy_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        rows = [{**t.__dict__, "net_pnl": t.net_pnl, "r_multiple": t.r_multiple} for t in self.trades]
        return pd.DataFrame(rows)


@dataclass
class _OpenPosition:
    side: int
    qty: float
    entry_ts: datetime
    entry_price: float
    stop: Optional[float]
    target: Optional[float]
    initial_stop: Optional[float]
    initial_risk: float
    tag: str
    entry_qty: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    carry: float = 0.0
    realized: float = 0.0     # gross P&L already booked by partial exits
    scaled: bool = False

    def view(self) -> Position:
        return Position(side=Side(self.side), quantity=self.qty, entry_price=self.entry_price,
                        entry_ts=self.entry_ts, stop_loss=self.stop, take_profit=self.target)


class Backtester:
    def __init__(self, strategy: BaseStrategy, model: ExecutionModel,
                 config: Optional[BacktestConfig] = None) -> None:
        self.strategy = strategy
        self.model = model
        self.cfg = config or BacktestConfig()

    # ------------------------------------------------------------------ run
    def run(self, bars: Iterable[Bar]) -> BacktestResult:
        cfg, model = self.cfg, self.model
        self.strategy.reset()
        state = StrategyState(equity=cfg.initial_equity)
        cash = cfg.initial_equity
        pos: Optional[_OpenPosition] = None
        pending: Optional[Signal] = None
        trades: List[Trade] = []
        eq_ts: List[datetime] = []
        eq_val: List[float] = []
        prev: Optional[Bar] = None
        day: Optional[date] = None
        day_start_equity = cash
        consecutive_losses = 0
        locked = False
        seen = 0  # trades already counted for the loss-streak breaker

        def close(p: _OpenPosition, qty: float, ref: float, bar: Bar, reason: str,
                  otype: OrderType = OrderType.MARKET) -> float:
            """Close ``qty`` of ``p``; returns realised net cash change."""
            nonlocal pos
            px = model.fill_price(-p.side, ref, bar, otype)
            gross = model.pnl(p.entry_price, px, qty, p.side, bar)
            slip = abs(model.pnl(ref, px, qty, 1, bar))
            fee = model.fees(qty, px, bar)
            p.fees += fee
            p.slippage += slip
            p.realized += gross
            p.qty -= qty
            if p.qty <= 1e-12:
                trades.append(Trade(
                    symbol=bar.symbol, side=p.side, qty=p.entry_qty, entry_ts=p.entry_ts,
                    entry_price=p.entry_price, exit_ts=bar.ts, exit_price=px,
                    gross_pnl=p.realized, fees=p.fees, slippage=p.slippage, carry=p.carry,
                    initial_risk=p.initial_risk, exit_reason=reason, tag=p.tag))
                pos = None
            return gross - fee

        for bar in bars:
            # -- new trading day: reset daily loss / streak lockouts
            bday = bar.ts.astimezone(_zone(cfg.day_key_tz)).date()
            if bday != day:
                day = bday
                day_start_equity = eq_val[-1] if eq_val else cash
                if cfg.lockout_resets_daily:
                    locked = False
                    consecutive_losses = 0

            # -- carry (swaps / rolls) for holding through prev -> bar
            if pos is not None and prev is not None:
                c = model.carry(pos.view(), prev, bar)
                pos.carry += c
                cash += c

            entered_by_stop = False
            # -- execute pending order at this bar
            if pending is not None:
                sig, pending = pending, None
                if sig.action is SignalAction.EXIT and pos is not None:
                    cash += close(pos, pos.qty, bar.open, bar, "signal")
                elif sig.is_entry:
                    side = int(sig.side)
                    if pos is not None and pos.side != side:
                        cash += close(pos, pos.qty, bar.open, bar, "reverse")
                    if pos is None:
                        ref = _trigger_price(sig, side, bar)
                        if ref is not None:
                            px = model.fill_price(side, ref, bar, sig.order_type)
                            stop = sig.stop_loss
                            if stop is None or (side > 0 and stop < px) or (side < 0 and stop > px):
                                # No stop, no trade: sizing is risk-based.
                                risk = sig.risk_fraction or cfg.risk_per_trade
                                qty = model.size(cash, side, px, stop, risk, bar) if stop is not None else 0.0
                                qty = model.round_quantity(qty)
                                if qty > 0:
                                    fee = model.fees(qty, px, bar)
                                    cash -= fee
                                    slip = abs(model.pnl(ref, px, qty, 1, bar))
                                    risk_money = abs(model.pnl(px, stop, qty, 1, bar))
                                    pos = _OpenPosition(side, qty, bar.ts, px, stop, sig.take_profit, stop,
                                                        risk_money, sig.tag, entry_qty=qty,
                                                        fees=fee, slippage=slip)
                                    entered_by_stop = sig.order_type is not OrderType.MARKET

            # -- protective stop / target / scale-out inside this bar
            if pos is not None:
                cash += self._manage_intrabar(pos, bar, close, entered_by_stop)
                if pos is not None and pos.qty <= 0:
                    pos = None

            # -- session flatten at the close
            if pos is not None and model.must_flatten(bar):
                cash += close(pos, pos.qty, bar.close, bar, "session_flatten")

            # -- streak / daily-loss bookkeeping for trades that just closed
            while seen < len(trades):
                t = trades[seen]
                seen += 1
                consecutive_losses = consecutive_losses + 1 if t.net_pnl < 0 else 0
                if cfg.max_consecutive_losses and consecutive_losses >= cfg.max_consecutive_losses:
                    locked = True

            # -- mark to market
            unreal = model.pnl(pos.entry_price, bar.close, pos.qty, pos.side, bar) if pos else 0.0
            equity = cash + unreal
            eq_ts.append(bar.ts)
            eq_val.append(equity)
            if cfg.max_daily_loss and equity <= day_start_equity * (1 - cfg.max_daily_loss):
                locked = True

            # -- strategy decides on the closed bar
            state.equity = equity
            state.position = pos.view() if pos else Position()
            state.push(bar, self.strategy.max_history)
            sig = self.strategy.on_bar(bar, state)

            if sig.action is SignalAction.MODIFY and pos is not None:
                if sig.stop_loss is not None:
                    pos.stop = sig.stop_loss
                if sig.take_profit is not None:
                    pos.target = sig.take_profit
            elif sig.action is SignalAction.EXIT and pos is not None:
                pending = sig
            elif sig.is_entry:
                allowed = model.entry_allowed(bar) and not locked
                same_side = pos is not None and pos.side == int(sig.side)
                if allowed and not same_side:
                    pending = sig
                elif pos is not None and not same_side and not allowed:
                    pending = Signal(SignalAction.EXIT, tag=sig.tag)  # reversal blocked -> just exit
            prev = bar

        # close anything left at the final close so every trade is counted
        if pos is not None and prev is not None:
            cash += close(pos, pos.qty, prev.close, prev, "end_of_data")
            eq_val[-1] = cash
        return BacktestResult(
            equity=pd.Series(eq_val, index=pd.DatetimeIndex(eq_ts), name="equity", dtype=float),
            trades=trades, initial_equity=cfg.initial_equity,
            strategy_name=self.strategy.name, params=dict(self.strategy.params))

    # ------------------------------------------------------------ intrabar
    def _manage_intrabar(self, pos: _OpenPosition, bar: Bar, close, entered_by_stop: bool) -> float:
        cash = 0.0
        long = pos.side > 0
        # stop first (pessimistic)
        if pos.stop is not None:
            hit = bar.low <= pos.stop if long else bar.high >= pos.stop
            if hit:
                gap = bar.open < pos.stop if long else bar.open > pos.stop
                ref = bar.open if gap and bar.ts != pos.entry_ts else pos.stop
                return close(pos, pos.qty, ref, bar, "stop", OrderType.STOP)
        if entered_by_stop:
            return cash
        # partial scale-out at N R, then stop to breakeven
        rule = self.cfg.scale_out
        if rule and not pos.scaled and pos.initial_stop is not None:
            r_dist = abs(pos.entry_price - pos.initial_stop)
            level = pos.entry_price + pos.side * rule.r_multiple * r_dist
            reached = bar.high >= level if long else bar.low <= level
            if reached and r_dist > 0:
                part = self.model.round_quantity(pos.qty * rule.fraction)
                pos.scaled = True
                if 0 < part < pos.qty:
                    ref = max(level, bar.open) if long else min(level, bar.open)
                    cash += close(pos, part, ref, bar, "scale_out", OrderType.LIMIT)
                    pos.stop = pos.entry_price + pos.side * rule.breakeven_offset
        if pos.target is not None and pos.qty > 0:
            hit = bar.high >= pos.target if long else bar.low <= pos.target
            if hit:
                ref = max(pos.target, bar.open) if long else min(pos.target, bar.open)
                cash += close(pos, pos.qty, ref, bar, "target", OrderType.LIMIT)
        return cash


def _trigger_price(sig: Signal, side: int, bar: Bar) -> Optional[float]:
    """Reference fill price for a pending entry on ``bar``, or ``None`` if
    a stop/limit order is not triggered."""
    if sig.order_type is OrderType.MARKET:
        return bar.open
    px = sig.entry_price
    if sig.order_type is OrderType.STOP:   # buy stop above / sell stop below
        if side > 0 and bar.high >= px:
            return max(px, bar.open)
        if side < 0 and bar.low <= px:
            return min(px, bar.open)
        return None
    # LIMIT: buy limit below / sell limit above
    if side > 0 and bar.low <= px:
        return min(px, bar.open)
    if side < 0 and bar.high >= px:
        return max(px, bar.open)
    return None


def _zone(name: str):
    from zoneinfo import ZoneInfo
    return ZoneInfo(name)
