"""Bridge from the existing ``quant_lab.data.loader`` DataFrames to ``Bar``s.

``load_prices`` returns a yfinance DataFrame (often with ``(field, ticker)``
MultiIndex columns and a naive index). ``normalize_ohlcv`` flattens that to
lower-case ``open/high/low/close/volume`` columns with a tz-aware index, and
``bars_from_dataframe`` turns it into the ``Bar`` stream the event-driven
engines consume. The existing vectorised backtester keeps working unchanged.
"""

from __future__ import annotations

from typing import Iterator, Optional

import pandas as pd

from quant_lab.core.types import Bar

_QUOTE_COLS = (
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
)


def normalize_ohlcv(df: pd.DataFrame, tz: str = "UTC", ticker: Optional[str] = None) -> pd.DataFrame:
    """Return a copy with flat lower-case OHLCV columns and a tz-aware index.

    Naive indexes are *localised* to ``tz``; aware ones are converted to it.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        tickers = out.columns.get_level_values(-1).unique()
        if ticker is None:
            if len(tickers) != 1:
                raise ValueError(f"DataFrame holds several tickers {list(tickers)}; pass ticker=")
            ticker = tickers[0]
        out = out.xs(ticker, axis=1, level=-1)
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "close" not in out.columns and "adj_close" in out.columns:
        out = out.rename(columns={"adj_close": "close"})
    missing = {"open", "high", "low", "close"} - set(out.columns)
    if missing:
        raise ValueError(f"missing OHLC columns: {sorted(missing)}")
    if "volume" not in out.columns:
        out["volume"] = 0.0

    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    idx = idx.tz_localize(tz) if idx.tz is None else idx.tz_convert(tz)
    out.index = idx
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    keep = ["open", "high", "low", "close", "volume"] + [c for c in _QUOTE_COLS if c in out.columns]
    return out[keep].dropna(subset=["open", "high", "low", "close"])


def bars_from_dataframe(df: pd.DataFrame, symbol: str = "", tz: str = "UTC") -> Iterator[Bar]:
    """Yield closed ``Bar`` objects in time order from an OHLCV DataFrame."""
    norm = normalize_ohlcv(df, tz=tz)
    quote_cols = [c for c in _QUOTE_COLS if c in norm.columns]
    for ts, row in norm.iterrows():
        quotes = {c: float(row[c]) for c in quote_cols if pd.notna(row[c])}
        yield Bar(
            ts=ts.to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]) if pd.notna(row["volume"]) else 0.0,
            symbol=symbol,
            **quotes,
        )
