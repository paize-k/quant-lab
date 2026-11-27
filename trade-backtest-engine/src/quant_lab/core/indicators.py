"""Streaming (one bar at a time) indicators.

Strategies run through ``on_bar`` in backtests *and* live, so indicators must
update incrementally from closed bars only. Each class has ``update(...)``
which returns the current value (``None`` until warmed up) and a ``value``
property. ``tests/core/test_indicators.py`` checks SMA/EMA against the
vectorised versions in ``quant_lab.features.indicators`` so both stay in step.

Wilder smoothing (ATR, ADX, RSI) seeds with a simple average of the first
``period`` values, the classic definition.
"""

from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections import deque
from typing import Deque, Optional, Tuple


class SMA:
    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self.period = period
        self._win: Deque[float] = deque(maxlen=period)
        self._sum = 0.0
        self.value: Optional[float] = None

    def update(self, x: float) -> Optional[float]:
        if len(self._win) == self.period:
            self._sum -= self._win[0]
        self._win.append(x)
        self._sum += x
        self.value = self._sum / self.period if len(self._win) == self.period else None
        return self.value


class EMA:
    """Matches ``pandas.Series.ewm(span=period, adjust=False).mean()``."""

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self.count = 0
        self.value: Optional[float] = None

    def update(self, x: float) -> Optional[float]:
        self.value = x if self.value is None else self.value + self.alpha * (x - self.value)
        self.count += 1
        return self.value

    @property
    def ready(self) -> bool:
        return self.count >= self.period


class Wilder:
    """Wilder's smoothing: SMA seed, then ``v += (x - v) / period``."""

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self.period = period
        self._seed: list = []
        self.value: Optional[float] = None

    def update(self, x: float) -> Optional[float]:
        if self.value is None:
            self._seed.append(x)
            if len(self._seed) == self.period:
                self.value = sum(self._seed) / self.period
                self._seed = []
        else:
            self.value += (x - self.value) / self.period
        return self.value


class RollingStd:
    """Sample standard deviation over a fixed window."""

    def __init__(self, period: int) -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        self.period = period
        self._win: Deque[float] = deque(maxlen=period)
        self.value: Optional[float] = None

    def update(self, x: float) -> Optional[float]:
        self._win.append(x)
        if len(self._win) < self.period:
            self.value = None
        else:
            m = sum(self._win) / self.period
            self.value = math.sqrt(sum((v - m) ** 2 for v in self._win) / (self.period - 1))
        return self.value


class Donchian:
    """Highest high / lowest low of the *previous* ``period`` bars.

    Excluding the current bar means ``close > upper`` is a genuine breakout
    (same convention as ``donchian_breakout_position`` with ``shift(1)``).
    """

    def __init__(self, period: int) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        self.period = period
        self._highs: Deque[float] = deque(maxlen=period)
        self._lows: Deque[float] = deque(maxlen=period)
        self.upper: Optional[float] = None
        self.lower: Optional[float] = None

    def update(self, high: float, low: float) -> Tuple[Optional[float], Optional[float]]:
        if len(self._highs) == self.period:
            self.upper, self.lower = max(self._highs), min(self._lows)
        else:
            self.upper = self.lower = None
        self._highs.append(high)
        self._lows.append(low)
        return self.upper, self.lower


class ATR:
    def __init__(self, period: int = 14) -> None:
        self._w = Wilder(period)
        self._prev_close: Optional[float] = None
        self.value: Optional[float] = None

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        tr = high - low if self._prev_close is None else max(
            high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        self._prev_close = close
        self.value = self._w.update(tr)
        return self.value


class ADX:
    """Wilder's ADX with +DI / -DI."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._tr, self._pdm, self._mdm = Wilder(period), Wilder(period), Wilder(period)
        self._dx = Wilder(period)
        self._prev: Optional[Tuple[float, float, float]] = None
        self.plus_di: Optional[float] = None
        self.minus_di: Optional[float] = None
        self.value: Optional[float] = None

    def update(self, high: float, low: float, close: float) -> Optional[float]:
        if self._prev is None:
            self._prev = (high, low, close)
            return None
        ph, pl, pc = self._prev
        self._prev = (high, low, close)
        up, down = high - ph, pl - low
        pdm = up if up > down and up > 0 else 0.0
        mdm = down if down > up and down > 0 else 0.0
        tr = max(high - low, abs(high - pc), abs(low - pc))
        atr, sp, sm = self._tr.update(tr), self._pdm.update(pdm), self._mdm.update(mdm)
        if atr is None or sp is None or sm is None or atr == 0:
            return self.value
        self.plus_di, self.minus_di = 100 * sp / atr, 100 * sm / atr
        denom = self.plus_di + self.minus_di
        dx = 0.0 if denom == 0 else 100 * abs(self.plus_di - self.minus_di) / denom
        self.value = self._dx.update(dx)
        return self.value


class RSI:
    def __init__(self, period: int = 14) -> None:
        self._gain, self._loss = Wilder(period), Wilder(period)
        self._prev: Optional[float] = None
        self.value: Optional[float] = None

    def update(self, close: float) -> Optional[float]:
        if self._prev is None:
            self._prev = close
            return None
        change, self._prev = close - self._prev, close
        g, l = self._gain.update(max(change, 0.0)), self._loss.update(max(-change, 0.0))
        if g is None or l is None:
            return None
        self.value = 100.0 if l == 0 else 100.0 - 100.0 / (1.0 + g / l)
        return self.value


class Keltner:
    """EMA(close) +/- ``mult`` * ATR."""

    def __init__(self, period: int = 20, atr_period: int = 10, mult: float = 2.0) -> None:
        self._ema, self._atr, self.mult = EMA(period), ATR(atr_period), mult
        self.mid = self.upper = self.lower = None

    def update(self, high: float, low: float, close: float):
        mid, atr = self._ema.update(close), self._atr.update(high, low, close)
        if atr is None or not self._ema.ready:
            self.mid = self.upper = self.lower = None
        else:
            self.mid, self.upper, self.lower = mid, mid + self.mult * atr, mid - self.mult * atr
        return self.mid, self.upper, self.lower


class PercentileRank:
    """Fraction of the last ``period`` values strictly below the new value."""

    def __init__(self, period: int) -> None:
        self.period = period
        self._win: Deque[float] = deque()
        self._sorted: list = []
        self.value: Optional[float] = None

    def update(self, x: float) -> Optional[float]:
        if len(self._win) == self.period:
            old = self._win.popleft()
            del self._sorted[bisect_left(self._sorted, old)]
        rank = bisect_left(self._sorted, x) / len(self._sorted) if self._sorted else None
        self._win.append(x)
        insort(self._sorted, x)
        self.value = rank if len(self._win) == self.period else None
        return self.value
