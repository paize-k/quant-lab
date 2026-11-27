"""Trend momentum: dual EMA + Donchian (or Keltner) breakout, ADX filter,
ATR trailing stop."""

from __future__ import annotations

from quant_lab.core.indicators import ADX, ATR, EMA, Donchian, Keltner
from quant_lab.core.strategy import BaseStrategy, StrategyState
from quant_lab.core.types import Bar, Side, Signal, SignalAction


class TrendMomentum(BaseStrategy):
    """Long when fast EMA > slow EMA, close breaks the channel high and ADX is
    strong; short on the mirror image. The stop starts ``atr_stop`` ATRs away
    and trails behind the close; the trade also exits on an EMA cross back."""

    name = "futures_trend_momentum"
    default_params = {
        "fast": 20, "slow": 50, "channel": 20, "use_keltner": False, "keltner_mult": 2.0,
        "adx_period": 14, "adx_min": 20.0, "atr_period": 14, "atr_stop": 2.0,
    }

    def validate_params(self) -> None:
        p = self.params
        if not (0 < p["fast"] < p["slow"]):
            raise ValueError("need 0 < fast < slow")
        if p["channel"] < 2 or p["atr_stop"] <= 0:
            raise ValueError("channel must be >= 2 and atr_stop > 0")

    @property
    def warmup_bars(self) -> int:
        p = self.params
        return max(p["slow"], p["channel"] + 1, 2 * p["adx_period"] + 1, p["atr_period"])

    @property
    def max_history(self):
        return 2

    def reset(self) -> None:
        p = self.params
        self.fast, self.slow = EMA(p["fast"]), EMA(p["slow"])
        self.channel = Donchian(p["channel"])
        self.keltner = Keltner(p["channel"], p["atr_period"], p["keltner_mult"])
        self.adx, self.atr = ADX(p["adx_period"]), ATR(p["atr_period"])
        self.n = 0

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.reset()

    def on_bar(self, bar: Bar, state: StrategyState) -> Signal:
        p = self.params
        f, s = self.fast.update(bar.close), self.slow.update(bar.close)
        up, dn = self.channel.update(bar.high, bar.low)
        k_mid, k_up, k_dn = self.keltner.update(bar.high, bar.low, bar.close)
        adx, atr = self.adx.update(bar.high, bar.low, bar.close), self.atr.update(bar.high, bar.low, bar.close)
        self.n += 1
        if self.n < self.warmup_bars or None in (adx, atr, up, dn) or (p["use_keltner"] and k_up is None):
            return Signal.hold()
        if p["use_keltner"]:
            up, dn = k_up, k_dn

        pos = state.position
        dist = p["atr_stop"] * atr
        if pos.side is Side.LONG:
            if f < s:
                return Signal(SignalAction.EXIT, tag=self.name)
            trail = bar.close - dist
            if pos.stop_loss is None or trail > pos.stop_loss:
                return Signal(SignalAction.MODIFY, stop_loss=trail, tag=self.name)
            return Signal.hold()
        if pos.side is Side.SHORT:
            if f > s:
                return Signal(SignalAction.EXIT, tag=self.name)
            trail = bar.close + dist
            if pos.stop_loss is None or trail < pos.stop_loss:
                return Signal(SignalAction.MODIFY, stop_loss=trail, tag=self.name)
            return Signal.hold()

        if adx < p["adx_min"]:
            return Signal.hold()
        if f > s and bar.close > up:
            return Signal(SignalAction.ENTER_LONG, stop_loss=bar.close - dist, tag=self.name,
                          meta={"adx": adx, "atr": atr})
        if f < s and bar.close < dn:
            return Signal(SignalAction.ENTER_SHORT, stop_loss=bar.close + dist, tag=self.name,
                          meta={"adx": adx, "atr": atr})
        return Signal.hold()
