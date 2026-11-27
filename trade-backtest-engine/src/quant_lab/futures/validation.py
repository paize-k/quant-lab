"""Futures validation suite: comparative tear sheet, walk-forward
optimisation and 1,000-run Monte Carlo for every strategy.

    from quant_lab.futures.validation import validate, DEFAULT_SPACES
    report = validate(bars, get_contract("MES"))
    print(report.tear_sheets)        # one column per strategy
    print(report.robustness)         # WFO OOS + Monte Carlo p95 drawdown, pass/fail

Run ``python -m quant_lab.futures.validation`` for a demo on synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Type

import pandas as pd

from quant_lab.core.engine import BacktestResult
from quant_lab.core.metrics import compare
from quant_lab.core.monte_carlo import MonteCarloReport, reshuffle
from quant_lab.core.strategy import BaseStrategy
from quant_lab.core.types import Bar
from quant_lab.core.wfo import WalkForwardReport, walk_forward
from quant_lab.futures.config.contracts import FuturesContractSpec
from quant_lab.futures.config.risk import FuturesRiskLimits
from quant_lab.futures.engine import run_backtest
from quant_lab.futures.strategies import (
    FuturesHybrid, OpeningRangeBreakout, TrendMomentum, VWAPReversion,
)

DEFAULT_STRATEGIES: Dict[str, Type[BaseStrategy]] = {
    "Trend Momentum": TrendMomentum,
    "VWAP Reversion": VWAPReversion,
    "Opening Range Breakout": OpeningRangeBreakout,
    "Hybrid": FuturesHybrid,
}

# Small grids on purpose: every extra combination raises the risk of fitting noise.
DEFAULT_SPACES: Dict[str, Dict[str, Sequence]] = {
    "Trend Momentum": {"fast": [10, 20], "slow": [40, 80], "atr_stop": [1.5, 2.5]},
    "VWAP Reversion": {"band_k": [1.5, 2.0, 2.5], "adx_max": [20.0, 25.0]},
    "Opening Range Breakout": {"or_minutes": [15, 30, 60], "compression_ratio": [0.5, 0.8]},
    "Hybrid": {"adx_trend": [22.0, 28.0], "adx_range": [15.0, 20.0]},
}


@dataclass
class ValidationReport:
    results: Dict[str, BacktestResult]
    tear_sheets: pd.DataFrame
    wfo: Dict[str, WalkForwardReport] = field(default_factory=dict)
    monte_carlo: Dict[str, MonteCarloReport] = field(default_factory=dict)

    @property
    def robustness(self) -> pd.DataFrame:
        rows = {}
        for name in self.results:
            w, mc = self.wfo.get(name), self.monte_carlo.get(name)
            rows[name] = {
                "WFO OOS Net PnL": None if w is None else w.summary["Net PnL"],
                "WFO OOS Sharpe": None if w is None else w.summary["Sharpe"],
                "WFO OOS Trades": None if w is None else w.summary["Trades"],
                "MC p50 DD %": None if mc is None else mc.drawdown_p50,
                "MC p95 DD %": None if mc is None else mc.drawdown_p95,
                "MC P(loss)": None if mc is None else mc.prob_loss,
                "MC pass": None if mc is None else mc.passed,
            }
        return pd.DataFrame(rows)


def validate(
    bars: Sequence[Bar],
    spec: FuturesContractSpec,
    strategies: Optional[Mapping[str, BaseStrategy]] = None,
    spaces: Optional[Mapping[str, Mapping[str, Sequence]]] = None,
    initial_equity: float = 100_000.0,
    limits: Optional[FuturesRiskLimits] = None,
    use_sessions: bool = True,
    wfo_folds: int = 4,
    mc_runs: int = 1_000,
    mc_dd_limit: float = 20.0,
    run_wfo: bool = True,
) -> ValidationReport:
    bars = list(bars)
    strategies = dict(strategies) if strategies else {n: c() for n, c in DEFAULT_STRATEGIES.items()}
    spaces = DEFAULT_SPACES if spaces is None else spaces

    def run(strat: BaseStrategy, b: Sequence[Bar], eq: float) -> BacktestResult:
        return run_backtest(strat, b, spec, eq, limits=limits, use_sessions=use_sessions)

    results = {name: run(s, bars, initial_equity) for name, s in strategies.items()}
    report = ValidationReport(results, compare(results))
    for name, res in results.items():
        report.monte_carlo[name] = reshuffle([t.net_pnl for t in res.trades], initial_equity,
                                             runs=mc_runs, max_dd_limit=mc_dd_limit)
        if run_wfo and name in spaces:
            strat = strategies[name]
            fixed = {k: v for k, v in strat.params.items() if k not in spaces[name]}
            report.wfo[name] = walk_forward(type(strat), spaces[name], bars, run, n_folds=wfo_folds,
                                            initial_equity=initial_equity, fixed_params=fixed,
                                            min_trades=3)
    return report


def _demo() -> None:  # pragma: no cover - manual entry point
    from quant_lab.core.data import bars_from_dataframe
    from quant_lab.futures.config import get_contract
    from quant_lab.futures.data.synthetic import synthetic_intraday

    spec = get_contract("MES")
    df = synthetic_intraday(spec, days=80, bar_minutes=5)
    bars = list(bars_from_dataframe(df, symbol="MES", tz="America/Chicago"))
    rep = validate(bars, spec, mc_runs=1_000)
    pd.set_option("display.width", 160)
    print("Synthetic data - checks the plumbing only, says nothing about real edge.\n")
    print(rep.tear_sheets.round(2).to_string())
    print()
    print(rep.robustness.round(2).to_string())


if __name__ == "__main__":  # pragma: no cover
    _demo()
