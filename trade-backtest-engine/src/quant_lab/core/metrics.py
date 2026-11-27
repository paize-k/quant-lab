"""Tear-sheet metrics for ``BacktestResult`` objects.

Return-based ratios (CAGR, Sharpe, Sortino, Calmar) are computed on *daily*
equity (last mark of each calendar day), so they mean the same thing for
1-minute futures bars and 4-hour FX bars. This avoids the hard-coded
"252 bars per year" problem in ``quant_lab.metrics.performance``.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd

from quant_lab.core.engine import BacktestResult

TEAR_SHEET_FIELDS = (
    "Net PnL", "Return %", "CAGR %", "Sharpe", "Sortino", "Calmar", "Max Drawdown %",
    "Trades", "Win Rate %", "Profit Factor", "Expectancy", "Avg R",
    "Total Fees", "Total Slippage/Spread", "Total Carry", "Exposure %",
)


def max_drawdown_pct(equity: pd.Series | np.ndarray) -> float:
    """Largest peak-to-trough fall, as a positive percentage."""
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0, (peak - eq) / peak, 0.0)
    return float(dd.max() * 100)


def daily_equity(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    idx = equity.index
    local = equity.copy()
    if idx.tz is not None:
        local.index = idx.tz_convert("UTC")
    return local.groupby(local.index.normalize()).last()


def tear_sheet(result: BacktestResult, periods_per_year: int = 252) -> pd.Series:
    eq = result.equity
    trades = result.trades
    start = result.initial_equity
    end = float(eq.iloc[-1]) if len(eq) else start

    daily = pd.concat([pd.Series([start]), daily_equity(eq).reset_index(drop=True)], ignore_index=True)
    rets = daily.pct_change().dropna()
    years = max(len(rets) / periods_per_year, 1e-9)
    growth = end / start if start > 0 else np.nan
    cagr = (growth ** (1 / years) - 1) * 100 if growth and growth > 0 else -100.0
    sd = rets.std(ddof=1)
    downside = rets[rets < 0]
    dsd = np.sqrt((downside ** 2).sum() / len(rets)) if len(rets) else 0.0
    sharpe = float(rets.mean() / sd * np.sqrt(periods_per_year)) if sd and sd > 0 else np.nan
    sortino = float(rets.mean() / dsd * np.sqrt(periods_per_year)) if dsd and dsd > 0 else np.nan
    mdd = max_drawdown_pct(np.concatenate([[start], eq.values]))
    calmar = cagr / mdd if mdd > 0 else np.nan

    pnl = np.array([t.net_pnl for t in trades], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    pf = wins.sum() / -losses.sum() if losses.size else (np.inf if wins.size else np.nan)
    r = np.array([t.r_multiple for t in trades], dtype=float)
    exposure = 0.0
    if trades and len(eq) > 1:
        held = sum(((eq.index >= t.entry_ts) & (eq.index < t.exit_ts)).sum() for t in trades)
        exposure = held / len(eq) * 100

    return pd.Series({
        "Net PnL": end - start,
        "Return %": (growth - 1) * 100 if start > 0 else np.nan,
        "CAGR %": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown %": mdd,
        "Trades": len(trades),
        "Win Rate %": float((pnl > 0).mean() * 100) if pnl.size else np.nan,
        "Profit Factor": float(pf),
        "Expectancy": float(pnl.mean()) if pnl.size else np.nan,
        "Avg R": float(np.nanmean(r)) if r.size and not np.all(np.isnan(r)) else np.nan,
        "Total Fees": float(sum(t.fees for t in trades)),
        "Total Slippage/Spread": float(sum(t.slippage for t in trades)),
        "Total Carry": float(sum(t.carry for t in trades)),
        "Exposure %": exposure,
    }, name=result.strategy_name)[list(TEAR_SHEET_FIELDS)]


def compare(results: Mapping[str, BacktestResult], periods_per_year: int = 252) -> pd.DataFrame:
    """Side-by-side tear sheet, one column per strategy."""
    return pd.DataFrame({name: tear_sheet(r, periods_per_year) for name, r in results.items()})
