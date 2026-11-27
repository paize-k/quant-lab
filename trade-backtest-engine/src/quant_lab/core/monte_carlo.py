"""Monte Carlo trade reshuffling.

Reorders the realised trade P&Ls many times to see how bad the drawdown
could have been with the same trades in a different order. A strategy
"passes" when the 95th-percentile drawdown stays within your limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class MonteCarloReport:
    runs: int
    drawdown_p50: float
    drawdown_p95: float
    drawdown_max: float
    final_equity_p5: float
    final_equity_p50: float
    prob_loss: float           # share of runs that end below starting equity
    max_dd_limit: Optional[float]

    @property
    def passed(self) -> Optional[bool]:
        return None if self.max_dd_limit is None else self.drawdown_p95 <= self.max_dd_limit


def reshuffle(
    trade_pnls: Sequence[float],
    initial_equity: float,
    runs: int = 1_000,
    max_dd_limit: Optional[float] = 20.0,
    bootstrap: bool = False,
    seed: Optional[int] = 42,
) -> MonteCarloReport:
    """Shuffle (or, with ``bootstrap``, resample with replacement) trade P&Ls.

    Drawdowns are percentages of the running peak. ``max_dd_limit`` is the
    95th-percentile drawdown % you are willing to accept.
    """
    pnl = np.asarray(trade_pnls, dtype=float)
    if pnl.size == 0:
        return MonteCarloReport(runs, 0.0, 0.0, 0.0, initial_equity, initial_equity, 0.0, max_dd_limit)
    rng = np.random.default_rng(seed)
    if bootstrap:
        sims = rng.choice(pnl, size=(runs, pnl.size), replace=True)
    else:
        sims = np.array([rng.permutation(pnl) for _ in range(runs)])
    curves = initial_equity + np.cumsum(sims, axis=1)
    curves = np.hstack([np.full((runs, 1), initial_equity), curves])
    peaks = np.maximum.accumulate(curves, axis=1)
    dd = np.where(peaks > 0, (peaks - curves) / peaks, 1.0).max(axis=1) * 100
    final = curves[:, -1]
    return MonteCarloReport(
        runs=runs,
        drawdown_p50=float(np.percentile(dd, 50)),
        drawdown_p95=float(np.percentile(dd, 95)),
        drawdown_max=float(dd.max()),
        final_equity_p5=float(np.percentile(final, 5)),
        final_equity_p50=float(np.percentile(final, 50)),
        prob_loss=float((final < initial_equity).mean()),
        max_dd_limit=max_dd_limit,
    )
