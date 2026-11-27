"""CME Globex / RTH session calendars (America/Chicago).

Globex for these products runs Sunday 17:00 to Friday 16:00 CT with a daily
16:00-17:00 CT maintenance halt, modelled as a 17:00-16:00 window that opens
Sunday through Thursday. RTH windows follow the underlying cash/pit hours.
Exchange holidays and early closes are not modelled yet (Milestone 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, Optional

from quant_lab.core.time import TimeWindow

CHICAGO = "America/Chicago"
_SUN_TO_THU = frozenset({6, 0, 1, 2, 3})
_WEEKDAYS = frozenset({0, 1, 2, 3, 4})


@dataclass(frozen=True)
class FuturesSessions:
    globex: TimeWindow
    rth: TimeWindow
    flatten_minutes_before_close: int = 10

    def is_tradable(self, ts: datetime, rth_only: bool = False) -> bool:
        return (self.rth if rth_only else self.globex).contains(ts)

    def should_flatten(self, ts: datetime, rth_only: bool = True) -> bool:
        """True in the last N minutes before RTH close (or Globex maintenance)."""
        window = self.rth if rth_only else self.globex
        left: Optional[float] = window.minutes_to_close(ts)
        return left is not None and left <= self.flatten_minutes_before_close


def _globex() -> TimeWindow:
    return TimeWindow("globex", time(17, 0), time(16, 0), CHICAGO, _SUN_TO_THU)


SESSIONS: Dict[str, FuturesSessions] = {
    "cme_equity": FuturesSessions(
        globex=_globex(), rth=TimeWindow("rth", time(8, 30), time(15, 0), CHICAGO, _WEEKDAYS)),
    "cme_metals": FuturesSessions(
        globex=_globex(), rth=TimeWindow("rth", time(7, 20), time(12, 30), CHICAGO, _WEEKDAYS)),
    "cme_fx": FuturesSessions(
        globex=_globex(), rth=TimeWindow("rth", time(7, 20), time(14, 0), CHICAGO, _WEEKDAYS)),
}


def get_sessions(key: str) -> FuturesSessions:
    return SESSIONS[key]
