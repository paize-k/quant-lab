"""Bar normalisation for the futures engine (America/Chicago).

Most vendors label intraday bars by their *open* time; the engine wants the
*close* time so a bar is only visible once it is finished. ``to_engine_bars``
converts to Chicago time, shifts open-labelled bars to their close, drops
duplicates and (optionally) keeps only Globex or RTH bars.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from quant_lab.core.data import normalize_ohlcv
from quant_lab.futures.config.sessions import CHICAGO, FuturesSessions


def to_engine_bars(
    df: pd.DataFrame,
    bar_minutes: Optional[int] = None,
    label: str = "open",
    source_tz: str = "UTC",
    sessions: Optional[FuturesSessions] = None,
    rth_only: bool = False,
) -> pd.DataFrame:
    """Return OHLCV in Chicago time, indexed by bar close time.

    ``source_tz`` localises a naive index. ``bar_minutes`` is required when
    ``label="open"`` for intraday data; daily bars can pass ``None``.
    """
    out = normalize_ohlcv(df, tz=source_tz).tz_convert(CHICAGO)
    for extra in ("contract", "roll"):
        if extra in df.columns:
            out[extra] = df[extra].reindex(df.index).values
    if label == "open" and bar_minutes:
        out.index = out.index + pd.Timedelta(minutes=bar_minutes)
    elif label not in ("open", "close"):
        raise ValueError("label must be 'open' or 'close'")
    if sessions is not None:
        window = sessions.rth if rth_only else sessions.globex
        # A bar belongs to the session if its last instant is inside it.
        probe = out.index - pd.Timedelta(seconds=1)
        out = out[[window.contains(ts.to_pydatetime()) for ts in probe]]
    return out


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate close-labelled bars (e.g. 1min -> 5min), keeping close labels."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "contract" in df.columns:
        agg["contract"] = "last"
    out = df.resample(rule, label="right", closed="right").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])
