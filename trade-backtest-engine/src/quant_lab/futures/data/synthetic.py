"""Synthetic CME-style intraday bars for tests and demos.

Random-walk prices with regime switching (trending / mean-reverting days),
a U-shaped volume profile and full Globex hours in America/Chicago. This is
*not* market data; use it to check that the plumbing works, never to judge
a strategy.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quant_lab.futures.config.contracts import FuturesContractSpec

CT = ZoneInfo("America/Chicago")


def synthetic_intraday(
    spec: FuturesContractSpec,
    start: date = date(2026, 1, 5),
    days: int = 60,
    bar_minutes: int = 5,
    start_price: float = 5_000.0,
    daily_vol: float = 0.012,
    seed: Optional[int] = 7,
    rth_only: bool = False,
) -> pd.DataFrame:
    """Close-labelled OHLCV bars for ``days`` weekdays starting ``start``."""
    rng = np.random.default_rng(seed)
    rows, idx = [], []
    price = start_price
    d = start
    made = 0
    while made < days:
        if d.weekday() < 5:
            made += 1
            # Session opens 17:00 the previous evening, closes 16:00.
            open_dt = datetime.combine(d - timedelta(days=1), time(17, 0), tzinfo=CT)
            close_dt = datetime.combine(d, time(16, 0), tzinfo=CT)
            n = int((close_dt - open_dt).total_seconds() // (bar_minutes * 60))
            trend_day = rng.random() < 0.4
            drift = rng.choice([-1, 1]) * daily_vol * 0.6 / n if trend_day else 0.0
            sigma = daily_vol / np.sqrt(n)
            anchor = price
            for k in range(n):
                ts = open_dt + timedelta(minutes=bar_minutes * (k + 1))
                local_t = ts.time()
                rth = time(8, 30) < local_t <= time(15, 0)
                s = sigma * (1.6 if rth else 0.6)
                ret = drift + rng.normal(0, s)
                if not trend_day:  # pull back toward the day's anchor
                    ret -= 0.02 * (price - anchor) / anchor
                o = price
                c = o * (1 + ret)
                spread = abs(rng.normal(0, s)) * o
                h, l = max(o, c) + spread, min(o, c) - spread
                mins = (ts - open_dt).total_seconds() / 60
                vol = (300 if rth else 40) * (1 + 2 * abs(np.cos(np.pi * mins / ((close_dt - open_dt).total_seconds() / 60))))
                price = c
                if rth_only and not rth:
                    continue
                idx.append(ts)
                rows.append((spec.round_price(o), spec.round_price(h), spec.round_price(l),
                             spec.round_price(c), float(int(vol * rng.uniform(0.5, 1.5)))))
        d += timedelta(days=1)
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=["open", "high", "low", "close", "volume"])
    # Rounding can push open/close a tick outside high/low; fix that.
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df
