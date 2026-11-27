"""Continuous futures series with back-adjustment.

Give ``build_continuous`` one OHLCV DataFrame per contract month. It picks
the front contract, rolls ``roll_days_before_expiry`` business days before
each expiry, and back-adjusts older prices so the series has no artificial
gap at the roll:

* ``method="difference"`` (default) adds the price gap at each roll to all
  earlier bars. Point P&L is preserved exactly, which is what matters for
  tick-value accounting.
* ``method="ratio"`` multiplies instead, preserving percentage moves.
* ``method="none"`` splices raw prices (gaps left in).

The output keeps a ``contract`` column. ``bars_from_dataframe`` turns it into
``Bar.symbol``, and the futures execution model charges a round turn of fees
and slippage whenever a held position crosses a roll.

Expiry rules are approximations without exchange holidays (CME publishes the
exact calendar). They decide the roll date, not P&L, so a day's error only
moves the roll slightly.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Mapping, Tuple

import pandas as pd

from quant_lab.futures.config.contracts import FuturesContractSpec

ContractKey = Tuple[int, int]  # (year, month)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _business_days_back(d: date, n: int) -> date:
    while n > 0:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def _last_business_day(year: int, month: int) -> date:
    nxt = date(year + month // 12, month % 12 + 1, 1)
    d = nxt - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def expiry_date(spec: FuturesContractSpec, year: int, month: int) -> date:
    """Approximate last trading day of ``spec``'s (year, month) contract."""
    if spec.exchange == "COMEX":
        # Metals: third-last business day of the contract month.
        return _business_days_back(_last_business_day(year, month), 2)
    if spec.symbol.startswith("6"):
        # CME FX: two business days before the third Wednesday.
        return _business_days_back(_nth_weekday(year, month, 2, 3), 2)
    # Equity index: third Friday.
    return _nth_weekday(year, month, 4, 3)


def roll_date(spec: FuturesContractSpec, year: int, month: int, days_before: int | None = None) -> date:
    n = spec.roll_days_before_expiry if days_before is None else days_before
    return _business_days_back(expiry_date(spec, year, month), n)


def build_continuous(
    spec: FuturesContractSpec,
    contracts: Mapping[ContractKey, pd.DataFrame],
    method: str = "difference",
    days_before: int | None = None,
    tz: str = "America/Chicago",
) -> pd.DataFrame:
    """Splice per-contract bars into one back-adjusted continuous series.

    ``contracts`` maps ``(year, month)`` to a DataFrame with tz-aware index
    and ``open/high/low/close[/volume]`` columns.
    """
    if method not in ("difference", "ratio", "none"):
        raise ValueError("method must be 'difference', 'ratio' or 'none'")
    keys = sorted(contracts)
    if not keys:
        raise ValueError("no contracts given")

    pieces: List[pd.DataFrame] = []
    start = None
    for i, (y, m) in enumerate(keys):
        df = contracts[(y, m)].copy()
        if df.index.tz is None:
            raise ValueError(f"contract {(y, m)} has a naive index")
        df.index = df.index.tz_convert(tz)
        local_dates = df.index.date
        end = roll_date(spec, y, m, days_before) if i < len(keys) - 1 else None
        mask = pd.Series(True, index=df.index)
        if start is not None:
            mask &= local_dates >= start
        if end is not None:
            mask &= local_dates < end
        piece = df[mask.values].copy()
        piece["contract"] = spec.contract_code(y, m)
        pieces.append(piece)
        start = end

    if method != "none":
        # Gap at each roll, measured on raw prices: the new contract's close
        # at the old contract's last bar (or its first close if missing).
        gaps = []
        for i in range(len(pieces) - 1):
            old, new = pieces[i], pieces[i + 1]
            if old.empty or new.empty:
                gaps.append(0.0 if method == "difference" else 1.0)
                continue
            raw_new = contracts[keys[i + 1]]["close"].tz_convert(tz)
            new_px = float(raw_new.get(old.index[-1], new["close"].iloc[0]))
            old_px = float(old["close"].iloc[-1])
            gaps.append(new_px - old_px if method == "difference" else new_px / old_px)
        cols = ["open", "high", "low", "close"]
        # Piece i is shifted by every roll after it.
        for i in range(len(pieces) - 1):
            later = gaps[i:]
            if method == "difference":
                pieces[i][cols] = pieces[i][cols] + sum(later)
            else:
                factor = 1.0
                for g in later:
                    factor *= g
                pieces[i][cols] = pieces[i][cols] * factor

    out = pd.concat([p for p in pieces if not p.empty]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out["roll"] = out["contract"] != out["contract"].shift(1)
    out.iloc[0, out.columns.get_loc("roll")] = False
    return out
