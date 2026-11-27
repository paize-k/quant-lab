"""Walk-forward optimisation.

Split the bar history into consecutive folds. In each fold, grid-search the
strategy's parameters on the in-sample window, then run the best set on the
following out-of-sample window that the optimiser never saw. Stitching the
out-of-sample results together gives an honest estimate of live performance.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Type

import numpy as np
import pandas as pd

from quant_lab.core.engine import BacktestResult, Trade
from quant_lab.core.metrics import tear_sheet
from quant_lab.core.strategy import BaseStrategy
from quant_lab.core.types import Bar

RunFn = Callable[[BaseStrategy, Sequence[Bar], float], BacktestResult]


def param_grid(space: Mapping[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(list(space[k]) for k in keys))]


@dataclass
class Fold:
    index: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    best_params: Dict[str, Any]
    in_sample_score: float
    out_of_sample: BacktestResult


@dataclass
class WalkForwardReport:
    folds: List[Fold]
    oos_equity: pd.Series
    oos_trades: List[Trade]
    initial_equity: float
    objective: str
    summary: pd.Series = field(default_factory=pd.Series)

    def folds_frame(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            ts = tear_sheet(f.out_of_sample)
            rows.append({"fold": f.index, "train_start": f.train_start, "train_end": f.train_end,
                         "test_start": f.test_start, "test_end": f.test_end,
                         **{f"param_{k}": v for k, v in f.best_params.items()},
                         f"IS {self.objective}": f.in_sample_score,
                         "OOS Net PnL": ts["Net PnL"], "OOS Sharpe": ts["Sharpe"],
                         "OOS Max DD %": ts["Max Drawdown %"], "OOS Trades": ts["Trades"]})
        return pd.DataFrame(rows)


def _score(result: BacktestResult, objective: str, min_trades: int) -> float:
    if len(result.trades) < min_trades:
        return -np.inf
    v = tear_sheet(result)[objective]
    return float(v) if np.isfinite(v) else -np.inf


def walk_forward(
    strategy_cls: Type[BaseStrategy],
    space: Mapping[str, Iterable[Any]],
    bars: Sequence[Bar],
    run: RunFn,
    n_folds: int = 4,
    train_fraction: float = 0.7,
    anchored: bool = False,
    objective: str = "Sharpe",
    min_trades: int = 5,
    initial_equity: float = 100_000.0,
    fixed_params: Optional[Mapping[str, Any]] = None,
) -> WalkForwardReport:
    """Rolling (or ``anchored``) walk-forward over ``n_folds``.

    The data is cut into ``n_folds`` equal segments; each segment is split
    ``train_fraction`` in-sample / the rest out-of-sample. With ``anchored``
    the in-sample window always starts at the first bar. Each fold's OOS run
    starts from the previous fold's ending equity so the stitched curve is
    continuous. ``run(strategy, bars, initial_equity)`` builds and runs the
    engine-specific backtester.
    """
    bars = list(bars)
    if n_folds < 1 or not (0 < train_fraction < 1):
        raise ValueError("need n_folds >= 1 and 0 < train_fraction < 1")
    seg = len(bars) // n_folds
    if seg < 10:
        raise ValueError("not enough bars for this many folds")
    grid = param_grid(space)
    fixed = dict(fixed_params or {})
    folds: List[Fold] = []
    eq_parts: List[pd.Series] = []
    trades: List[Trade] = []
    equity = initial_equity

    for k in range(n_folds):
        seg_start = k * seg
        seg_end = (k + 1) * seg if k < n_folds - 1 else len(bars)
        split = seg_start + int((seg_end - seg_start) * train_fraction)
        train = bars[0 if anchored else seg_start: split]
        test = bars[split:seg_end]
        if not train or not test:
            continue
        best_params, best_score = None, -np.inf
        for params in grid:
            try:
                strat = strategy_cls(**fixed, **params)
            except (ValueError, TypeError):
                continue  # invalid combination (e.g. fast >= slow)
            score = _score(run(strat, train, initial_equity), objective, min_trades)
            if best_params is None or score > best_score:
                best_params, best_score = params, score
        if best_params is None:
            raise ValueError("no valid parameter combination in the grid")
        oos = run(strategy_cls(**fixed, **best_params), test, equity)
        folds.append(Fold(k, train[0].ts, train[-1].ts, test[0].ts, test[-1].ts,
                          best_params, best_score, oos))
        eq_parts.append(oos.equity)
        trades.extend(oos.trades)
        equity = float(oos.equity.iloc[-1]) if len(oos.equity) else equity

    oos_equity = pd.concat(eq_parts) if eq_parts else pd.Series(dtype=float)
    report = WalkForwardReport(folds, oos_equity, trades, initial_equity, objective)
    stitched = BacktestResult(oos_equity, trades, initial_equity, strategy_cls.name + " (WFO OOS)")
    report.summary = tear_sheet(stitched)
    return report
