"""Session helpers shared by the futures strategies (futures-only)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

from quant_lab.futures.config.sessions import FuturesSessions, get_sessions

_EPS = timedelta(seconds=1)


class SessionClock:
    """Answers "which RTH session is this bar in?" using the bar's close time.

    A bar closing exactly at 15:00 belongs to the session that closes at
    15:00, so we probe one second before the close.
    """

    def __init__(self, sessions_key: Optional[str]) -> None:
        self.sessions: Optional[FuturesSessions] = get_sessions(sessions_key) if sessions_key else None

    def rth_open(self, ts: datetime) -> Optional[datetime]:
        return None if self.sessions is None else self.sessions.rth.open_for(ts - _EPS)

    def in_globex(self, ts: datetime) -> bool:
        return self.sessions is None or self.sessions.globex.contains(ts - _EPS)


class SessionVWAP:
    """Volume-weighted average price anchored at each RTH open, with a
    volume-weighted standard deviation for the bands."""

    def __init__(self) -> None:
        self.anchor: Optional[datetime] = None
        self._pv = self._pv2 = self._v = 0.0
        self.bars = 0
        self.value: Optional[float] = None
        self.std: Optional[float] = None

    def reset(self, anchor: Optional[datetime]) -> None:
        self.anchor = anchor
        self._pv = self._pv2 = self._v = 0.0
        self.bars = 0
        self.value = self.std = None

    def update(self, high: float, low: float, close: float, volume: float) -> Optional[float]:
        tp = (high + low + close) / 3.0
        v = volume if volume > 0 else 1.0  # no volume data -> equal weights
        self._pv += tp * v
        self._pv2 += tp * tp * v
        self._v += v
        self.bars += 1
        self.value = self._pv / self._v
        self.std = math.sqrt(max(self._pv2 / self._v - self.value ** 2, 0.0))
        return self.value
