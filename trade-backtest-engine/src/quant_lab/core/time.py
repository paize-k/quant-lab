"""Timezone-aware time-of-day windows.

This is generic math only. The actual session calendars (CME Globex/RTH in
America/Chicago, FX Asian/London/NY in UTC) live in each engine's config.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import FrozenSet, Optional
from zoneinfo import ZoneInfo

ALL_WEEKDAYS: FrozenSet[int] = frozenset(range(7))


@dataclass(frozen=True)
class TimeWindow:
    """A daily window ``[start, end)`` in ``tz``; may wrap past midnight.

    ``weekdays`` (Mon=0 .. Sun=6) are the days on which the window *opens*.
    A window such as 17:00-16:00 opening Sun-Thu therefore covers Sunday
    evening through Friday afternoon.
    """

    name: str
    start: time
    end: time
    tz: str = "UTC"
    weekdays: FrozenSet[int] = ALL_WEEKDAYS

    def __post_init__(self) -> None:
        ZoneInfo(self.tz)  # fail fast on a bad zone name
        if self.start == self.end:
            raise ValueError("start and end must differ")
        if not self.weekdays <= ALL_WEEKDAYS:
            raise ValueError("weekdays must be in 0..6")

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)

    @property
    def wraps_midnight(self) -> bool:
        return self.end < self.start

    def _local(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return ts.astimezone(self.zone)

    def open_for(self, ts: datetime) -> Optional[datetime]:
        """Open time of the window instance containing ``ts``, else ``None``."""
        local = self._local(ts)
        t = local.time()
        if not self.wraps_midnight:
            opened_day = local.date() if self.start <= t < self.end else None
        elif t >= self.start:
            opened_day = local.date()
        elif t < self.end:
            opened_day = local.date() - timedelta(days=1)
        else:
            opened_day = None
        if opened_day is None or opened_day.weekday() not in self.weekdays:
            return None
        return datetime.combine(opened_day, self.start, tzinfo=self.zone)

    def contains(self, ts: datetime) -> bool:
        return self.open_for(ts) is not None

    def close_for(self, ts: datetime) -> Optional[datetime]:
        """Close time of the window instance containing ``ts``, else ``None``."""
        opened = self.open_for(ts)
        if opened is None:
            return None
        close_day = opened.date() + timedelta(days=1 if self.wraps_midnight else 0)
        return datetime.combine(close_day, self.end, tzinfo=self.zone)

    def minutes_to_close(self, ts: datetime) -> Optional[float]:
        close = self.close_for(ts)
        if close is None:
            return None
        return (close - ts.astimezone(self.zone)).total_seconds() / 60.0
