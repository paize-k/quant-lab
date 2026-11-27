"""Forex bot risk limits. Kept separate from futures on purpose."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ForexRiskLimits:
    risk_per_trade: float = 0.005
    max_daily_loss: float = 0.02
    max_consecutive_losses: int = 4
    stale_feed_seconds: float = 2.0
    block_rollover_entries: bool = True
    # Cap on summed risk per currency across open trades, as a fraction of
    # equity. Long EUR/USD + long GBP/USD + short USD/CHF are all short USD,
    # so together they count against the USD cap.
    max_currency_exposure: float = 0.01
    per_currency_overrides: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("risk_per_trade", "max_daily_loss", "max_currency_exposure"):
            v = getattr(self, name)
            if not (0.0 < v < 1.0):
                raise ValueError(f"{name} must be in (0, 1), got {v}")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be >= 1")

    def currency_cap(self, currency: str) -> float:
        return self.per_currency_overrides.get(currency.upper(), self.max_currency_exposure)
