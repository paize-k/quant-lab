"""Opening range breakout after a quiet overnight session."""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from quant_lab.core.strategy import BaseStrategy, StrategyState
from quant_lab.core.types import Bar, Signal, SignalAction
from quant_lab.futures.strategies._session import SessionClock


class OpeningRangeBreakout(BaseStrategy):
    """Build the high/low of the first ``or_minutes`` of RTH. Trade a close
    beyond it, but only if the overnight (pre-market) range was tight:
    ``overnight_range <= compression_ratio * previous RTH range``.

    Stop at the opposite side of the range (or its midpoint), target at
    ``target_r`` times the risk. At most ``max_trades_per_day`` entries; the
    engine's auto-flatten closes anything left before the RTH close.
    """

    name = "futures_orb"
    default_params = {
        "sessions_key": "cme_equity", "or_minutes": 30, "compression_ratio": 0.6,
        "target_r": 2.0, "stop_at": "opposite", "max_trades_per_day": 1, "last_entry_minutes": 120,
    }

    def validate_params(self) -> None:
        p = self.params
        if not p["sessions_key"]:
            raise ValueError("ORB needs intraday bars with a sessions_key")
        if not (5 <= p["or_minutes"] <= 120) or p["target_r"] <= 0:
            raise ValueError("or_minutes must be 5..120 and target_r > 0")
        if p["stop_at"] not in ("opposite", "mid"):
            raise ValueError("stop_at must be 'opposite' or 'mid'")

    @property
    def max_history(self):
        return 2

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.reset()

    def reset(self) -> None:
        self.clock = SessionClock(self.params["sessions_key"])
        self.session: Optional[object] = None
        self.or_high = self.or_low = None
        self.on_high = self.on_low = None      # overnight range since last RTH close
        self.rth_high = self.rth_low = None    # current RTH range
        self.prev_rth_range: Optional[float] = None
        self.trades_today = 0
        self.compressed = False

    def on_bar(self, bar: Bar, state: StrategyState) -> Signal:
        p = self.params
        anchor = self.clock.rth_open(bar.ts)
        if anchor is None:
            # Overnight / pre-market: track its range.
            if self.rth_high is not None:  # first bar after an RTH session ended
                self.prev_rth_range = self.rth_high - self.rth_low
                self.rth_high = self.rth_low = None
                self.on_high = self.on_low = None
            self.on_high = bar.high if self.on_high is None else max(self.on_high, bar.high)
            self.on_low = bar.low if self.on_low is None else min(self.on_low, bar.low)
            return Signal.hold()

        if anchor != self.session:  # new RTH session
            self.session = anchor
            self.or_high = self.or_low = None
            self.trades_today = 0
            on_range = None if self.on_high is None else self.on_high - self.on_low
            self.compressed = (on_range is not None and self.prev_rth_range is not None
                               and on_range <= p["compression_ratio"] * self.prev_rth_range)
        self.rth_high = bar.high if self.rth_high is None else max(self.rth_high, bar.high)
        self.rth_low = bar.low if self.rth_low is None else min(self.rth_low, bar.low)

        or_end = anchor + timedelta(minutes=p["or_minutes"])
        if bar.ts <= or_end:
            self.or_high = bar.high if self.or_high is None else max(self.or_high, bar.high)
            self.or_low = bar.low if self.or_low is None else min(self.or_low, bar.low)
            return Signal.hold()

        if (not state.position.is_flat or not self.compressed or self.or_high is None
                or self.trades_today >= p["max_trades_per_day"]
                or bar.ts > anchor + timedelta(minutes=p["last_entry_minutes"])):
            return Signal.hold()

        mid = (self.or_high + self.or_low) / 2
        if bar.close > self.or_high:
            stop = self.or_low if p["stop_at"] == "opposite" else mid
            risk = bar.close - stop
            self.trades_today += 1
            return Signal(SignalAction.ENTER_LONG, stop_loss=stop, take_profit=bar.close + p["target_r"] * risk,
                          tag=self.name, meta={"or_high": self.or_high, "or_low": self.or_low})
        if bar.close < self.or_low:
            stop = self.or_high if p["stop_at"] == "opposite" else mid
            risk = stop - bar.close
            self.trades_today += 1
            return Signal(SignalAction.ENTER_SHORT, stop_loss=stop, take_profit=bar.close - p["target_r"] * risk,
                          tag=self.name, meta={"or_high": self.or_high, "or_low": self.or_low})
        return Signal.hold()
