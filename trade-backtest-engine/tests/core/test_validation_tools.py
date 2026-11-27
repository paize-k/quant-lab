import numpy as np
import pandas as pd
import pytest

from quant_lab.core import (
    Backtester, BacktestConfig, BacktestResult, Signal, SignalAction, max_drawdown_pct, reshuffle,
    tear_sheet, walk_forward,
)
from quant_lab.core.strategy import BaseStrategy
from tests.core.helpers import UnitModel, make_bars


def test_max_drawdown():
    assert max_drawdown_pct([100, 120, 90, 130]) == pytest.approx(25.0)
    assert max_drawdown_pct([100, 101, 102]) == 0


def test_monte_carlo_seeded_and_ordered():
    pnl = [100, -50, 200, -300, 80, -20] * 10
    a = reshuffle(pnl, 10_000, runs=500, seed=1)
    b = reshuffle(pnl, 10_000, runs=500, seed=1)
    assert a == b
    assert a.drawdown_p95 >= a.drawdown_p50 > 0
    assert a.final_equity_p50 == pytest.approx(10_000 + sum(pnl))  # shuffling keeps the total
    assert a.passed is True
    assert reshuffle(pnl, 10_000, runs=200, max_dd_limit=0.1).passed is False
    boot = reshuffle(pnl, 10_000, runs=200, bootstrap=True)
    assert boot.final_equity_p5 <= boot.final_equity_p50


class Alternate(BaseStrategy):
    """Goes long every ``every`` bars and exits the next bar."""

    name = "alternate"
    default_params = {"every": 4}

    def on_bar(self, bar, state):
        i = state.bar_index
        if i % self.params["every"] == 0:
            return Signal(SignalAction.ENTER_LONG, stop_loss=bar.close - 5)
        if i % self.params["every"] == 1:
            return Signal(SignalAction.EXIT)
        return Signal.hold()


def _bars(n=400):
    rng = np.random.default_rng(3)
    px = 100 + rng.normal(0, 1, n).cumsum()
    return make_bars([(p, p + 1, p - 1, p) for p in px], minutes=60 * 24)


def _run(strat, bars, eq):
    return Backtester(strat, UnitModel(fee=0.5), BacktestConfig(initial_equity=eq)).run(bars)


def test_tear_sheet_fields():
    res = _run(Alternate(), _bars(), 10_000)
    ts = tear_sheet(res)
    for k in ("Net PnL", "CAGR %", "Sharpe", "Sortino", "Calmar", "Max Drawdown %", "Win Rate %",
              "Profit Factor", "Expectancy", "Total Fees", "Total Slippage/Spread"):
        assert k in ts.index
    assert ts["Trades"] == len(res.trades) > 0
    assert ts["Total Fees"] == pytest.approx(len(res.trades) * 1.0)
    assert ts["Net PnL"] == pytest.approx(sum(t.net_pnl for t in res.trades))


def test_walk_forward_folds_are_out_of_sample():
    bars = _bars()
    rep = walk_forward(Alternate, {"every": [3, 4, 6]}, bars, _run, n_folds=4, min_trades=1,
                       initial_equity=10_000)
    assert len(rep.folds) == 4
    prev_test_end = None
    for f in rep.folds:
        assert f.train_end < f.test_start          # optimiser never saw the test window
        if prev_test_end is not None:
            assert f.test_start > prev_test_end    # folds don't overlap
        prev_test_end = f.test_end
        assert f.best_params["every"] in (3, 4, 6)
    assert rep.oos_equity.index.is_monotonic_increasing
    assert "Sharpe" in rep.summary.index
    assert len(rep.folds_frame()) == 4


def test_walk_forward_anchored_starts_at_first_bar():
    bars = _bars()
    rep = walk_forward(Alternate, {"every": [3, 4]}, bars, _run, n_folds=3, anchored=True,
                       min_trades=1, initial_equity=10_000)
    assert all(f.train_start == bars[0].ts for f in rep.folds)
