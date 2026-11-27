"""Futures hybrid meta-strategy: a regime router over the three strategies."""

from __future__ import annotations

from dataclasses import replace
from datetime import time
from typing import Dict, Optional

from quant_lab.core.indicators import ADX, ATR, PercentileRank
from quant_lab.core.strategy import BaseStrategy, StrategyState
from quant_lab.core.types import Bar, Position, Signal, SignalAction
from quant_lab.futures.config.sessions import CHICAGO
from quant_lab.futures.strategies.opening_range_breakout import OpeningRangeBreakout
from quant_lab.futures.strategies.trend_momentum import TrendMomentum
from quant_lab.futures.strategies.vwap_reversion import VWAPReversion

TREND, RANGE, NEUTRAL, CLOSED = "trend", "range", "neutral", "closed"


class FuturesHybrid(BaseStrategy):
    """Classify each bar's regime with ADX and the ATR percentile:

    * **trend**: ADX >= ``adx_trend`` and ATR percentile >= ``atr_pct_high``.
      New entries come from ORB first, then Trend Momentum.
    * **range**: ADX <= ``adx_range``. New entries come from VWAP reversion.
    * **neutral**: in between. No new entries.
    * **closed**: the lunch window (``lunch_start``-``lunch_end`` CT) or
      outside RTH. No new entries.

    All three sub-strategies see every bar so their indicators stay warm. The
    one that opened the position keeps managing it (trailing stop, exit) even
    if the regime changes; the others are shown a flat position.
    """

    name = "futures_hybrid"
    default_params = {
        "sessions_key": "cme_equity", "adx_period": 14, "adx_trend": 25.0, "adx_range": 20.0,
        "atr_period": 14, "atr_pct_window": 100, "atr_pct_high": 0.5,
        "lunch_start": time(11, 30), "lunch_end": time(13, 0),
        "trend_params": {}, "vwap_params": {}, "orb_params": {},
    }

    def validate_params(self) -> None:
        p = self.params
        if p["adx_range"] > p["adx_trend"]:
            raise ValueError("adx_range must be <= adx_trend")
        if not (0 <= p["atr_pct_high"] <= 1):
            raise ValueError("atr_pct_high must be in [0, 1]")

    @property
    def max_history(self):
        return 2

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.reset()

    def reset(self) -> None:
        p = self.params
        key = p["sessions_key"]
        self.subs: Dict[str, BaseStrategy] = {
            "orb": OpeningRangeBreakout(**{"sessions_key": key, **p["orb_params"]}),
            "trend": TrendMomentum(**p["trend_params"]),
            "vwap": VWAPReversion(**{"sessions_key": key, **p["vwap_params"]}),
        }
        self.adx, self.atr = ADX(p["adx_period"]), ATR(p["atr_period"])
        self.atr_rank = PercentileRank(p["atr_pct_window"])
        self.owner: Optional[str] = None
        self.pending_owner: Optional[str] = None
        self.regime = NEUTRAL

    def classify(self, bar: Bar, adx: Optional[float], atr_pct: Optional[float]) -> str:
        p = self.params
        clock = self.subs["vwap"].clock  # type: ignore[attr-defined]
        if clock.sessions is not None and clock.rth_open(bar.ts) is None:
            return CLOSED
        local = bar.ts.astimezone(_tz()).time()
        if p["lunch_start"] <= local < p["lunch_end"]:
            return CLOSED
        if adx is None:
            return NEUTRAL
        if adx >= p["adx_trend"] and atr_pct is not None and atr_pct >= p["atr_pct_high"]:
            return TREND
        if adx <= p["adx_range"]:
            return RANGE
        return NEUTRAL

    def on_bar(self, bar: Bar, state: StrategyState) -> Signal:
        adx = self.adx.update(bar.high, bar.low, bar.close)
        atr = self.atr.update(bar.high, bar.low, bar.close)
        atr_pct = self.atr_rank.update(atr) if atr is not None else None
        self.regime = self.classify(bar, adx, atr_pct)

        flat = state.position.is_flat
        if flat:
            self.owner = None
        elif self.owner is None:
            self.owner = self.pending_owner
        self.pending_owner = None

        flat_state = StrategyState(equity=state.equity, position=Position(), history=state.history,
                                   scratch=state.scratch, bar_index=state.bar_index)
        signals = {name: sub.on_bar(bar, state if name == self.owner else flat_state)
                   for name, sub in self.subs.items()}

        if not flat:
            if self.owner is not None:
                sig = signals[self.owner]
                if sig.action in (SignalAction.EXIT, SignalAction.MODIFY):
                    return replace(sig, tag=f"{self.name}:{self.owner}")
            return Signal.hold()

        order = {TREND: ("orb", "trend"), RANGE: ("vwap",)}.get(self.regime, ())
        for name in order:
            sig = signals[name]
            if sig.is_entry:
                self.pending_owner = name
                return replace(sig, tag=f"{self.name}:{name}",
                               meta={**sig.meta, "regime": self.regime})
        return Signal.hold()


def _tz():
    from zoneinfo import ZoneInfo
    return ZoneInfo(CHICAGO)
