"""FX session windows, all in UTC.

Fixed UTC windows ignore London/New York daylight-saving shifts on purpose,
matching the spec. Revisit in Milestone 3 if you want DST-tracking windows.
"""

from __future__ import annotations

from datetime import datetime, time

from quant_lab.core.time import TimeWindow

_WEEKDAYS = frozenset({0, 1, 2, 3, 4})

ASIAN = TimeWindow("asian", time(0, 0), time(7, 0), "UTC", _WEEKDAYS)
LONDON = TimeWindow("london", time(7, 0), time(16, 0), "UTC", _WEEKDAYS)
NEW_YORK = TimeWindow("new_york", time(13, 0), time(21, 0), "UTC", _WEEKDAYS)
LONDON_NY_OVERLAP = TimeWindow("london_ny_overlap", time(13, 0), time(17, 0), "UTC", _WEEKDAYS)
# Daily rollover: spreads widen and new entries are blocked. Sun-Fri evenings.
ROLLOVER = TimeWindow("rollover", time(21, 0), time(23, 0), "UTC", frozenset({6, 0, 1, 2, 3, 4}))


def is_rollover_window(ts: datetime) -> bool:
    return ROLLOVER.contains(ts)
