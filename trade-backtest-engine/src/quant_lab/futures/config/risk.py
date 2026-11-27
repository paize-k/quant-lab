"""Futures bot risk limits. Kept separate from forex on purpose."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FuturesRiskLimits:
    risk_per_trade: float = 0.005          # 0.5% of equity at the stop
    max_daily_loss: float = 0.02           # lock out for the day at -2.0%
    max_consecutive_losses: int = 4
    stale_feed_seconds: float = 2.0
    max_contracts_per_symbol: int = 10
    auto_flatten_before_close: bool = True
    flatten_minutes_before_close: int = 10
    scale_out_r: float = 1.5               # take partial profit at 1.5R
    scale_out_fraction: float = 0.5
    breakeven_offset_ticks: int = 1        # move stop to entry + 1 tick after scale-out

    def __post_init__(self) -> None:
        for name in ("risk_per_trade", "max_daily_loss", "scale_out_fraction"):
            v = getattr(self, name)
            if not (0.0 < v < 1.0):
                raise ValueError(f"{name} must be in (0, 1), got {v}")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be >= 1")
