"""Intraday RTH VWAP mean reversion in low-ADX regimes."""

from __future__ import annotations

from quant_lab.core.indicators import ADX, ATR, RSI
from quant_lab.core.strategy import BaseStrategy, StrategyState
from quant_lab.core.types import Bar, Signal, SignalAction
from quant_lab.futures.strategies._session import SessionClock, SessionVWAP


class VWAPReversion(BaseStrategy):
    """Fade stretches beyond the anchored RTH VWAP +/- ``band_k`` standard
    deviations when RSI(2) is exhausted and ADX says there is no trend.
    Target: back to VWAP. Stop: ``atr_stop`` ATRs beyond the entry."""

    name = "futures_vwap_reversion"
    default_params = {
        "sessions_key": "cme_equity", "band_k": 2.0, "rsi_period": 2, "rsi_low": 10.0,
        "rsi_high": 90.0, "adx_period": 14, "adx_max": 20.0, "atr_period": 14, "atr_stop": 1.5,
        "min_session_bars": 6,
    }

    def validate_params(self) -> None:
        p = self.params
        if not p["sessions_key"]:
            raise ValueError("VWAP reversion needs intraday bars with a sessions_key")
        if p["band_k"] <= 0 or not (0 < p["rsi_low"] < p["rsi_high"] < 100):
            raise ValueError("bad band_k / rsi thresholds")

    @property
    def warmup_bars(self) -> int:
        return 2 * self.params["adx_period"] + 1

    @property
    def max_history(self):
        return 2

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.reset()

    def reset(self) -> None:
        p = self.params
        self.clock = SessionClock(p["sessions_key"])
        self.vwap = SessionVWAP()
        self.rsi, self.adx, self.atr = RSI(p["rsi_period"]), ADX(p["adx_period"]), ATR(p["atr_period"])

    def on_bar(self, bar: Bar, state: StrategyState) -> Signal:
        p = self.params
        rsi = self.rsi.update(bar.close)
        adx = self.adx.update(bar.high, bar.low, bar.close)
        atr = self.atr.update(bar.high, bar.low, bar.close)
        anchor = self.clock.rth_open(bar.ts)
        if anchor is None:
            return Signal.hold()  # outside RTH: no VWAP
        if anchor != self.vwap.anchor:
            self.vwap.reset(anchor)
        vwap = self.vwap.update(bar.high, bar.low, bar.close, bar.volume)
        sd = self.vwap.std
        if None in (rsi, adx, atr) or self.vwap.bars < p["min_session_bars"] or not sd:
            return Signal.hold()

        if not state.position.is_flat:
            return Signal.hold()  # target at VWAP and the stop handle exits
        if adx > p["adx_max"]:
            return Signal.hold()
        lower, upper = vwap - p["band_k"] * sd, vwap + p["band_k"] * sd
        if bar.close < lower and rsi < p["rsi_low"]:
            return Signal(SignalAction.ENTER_LONG, stop_loss=bar.close - p["atr_stop"] * atr,
                          take_profit=vwap, tag=self.name, meta={"vwap": vwap, "sd": sd})
        if bar.close > upper and rsi > p["rsi_high"]:
            return Signal(SignalAction.ENTER_SHORT, stop_loss=bar.close + p["atr_stop"] * atr,
                          take_profit=vwap, tag=self.name, meta={"vwap": vwap, "sd": sd})
        return Signal.hold()
