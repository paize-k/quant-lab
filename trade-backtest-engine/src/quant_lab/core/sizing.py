"""Fixed-fractional position-sizing math shared by both engines.

Futures:  contracts = floor(equity * risk / (stop_ticks * tick_value))
Forex:    units     = floor(equity * risk / (stop_pips * pip_value_per_unit))

Both are ``size_by_risk`` with a different "money per unit of stop distance"
and lot step. Margin caps and lot rounding rules stay in each engine.
"""

from __future__ import annotations

import math


def floor_to_step(quantity: float, step: float) -> float:
    """Round ``quantity`` down to a multiple of ``step`` (never up)."""
    if step <= 0:
        raise ValueError("step must be positive")
    if quantity <= 0:
        return 0.0
    # The epsilon stops 2.9999999 from flooring to 2 because of float error.
    return math.floor(quantity / step + 1e-9) * step


def size_by_risk(
    equity: float,
    risk_fraction: float,
    stop_distance: float,
    money_per_unit_distance: float,
    step: float = 1.0,
) -> float:
    """Largest size whose loss at the stop is at most ``equity * risk_fraction``.

    ``stop_distance`` is in ticks or pips; ``money_per_unit_distance`` is the
    account-currency value of one tick/pip for one contract/unit.
    """
    if equity <= 0:
        return 0.0
    if not (0.0 < risk_fraction < 1.0):
        raise ValueError("risk_fraction must be in (0, 1)")
    if stop_distance <= 0:
        raise ValueError("stop_distance must be positive")
    if money_per_unit_distance <= 0:
        raise ValueError("money_per_unit_distance must be positive")
    raw = (equity * risk_fraction) / (stop_distance * money_per_unit_distance)
    return floor_to_step(raw, step)
