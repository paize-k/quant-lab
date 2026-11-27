"""Broker fee and margin profiles for the futures engine.

A profile holds everything about a contract that depends on *who you trade
through*: commission, exchange/NFA/clearing fees, slippage assumption and
margins. Contract specs stay exchange-only; ``get_contract(symbol, profile)``
combines the two. To use a real broker later, build a new
``FuturesBrokerProfile`` with its published numbers and pass it in.

PAPER_DEFAULT (researched 2026-11-27)
-------------------------------------
A deliberately *conservative* profile for paper trading and backtests while
no broker is chosen: on each line it takes the more expensive of the retail
figures found, so live costs at a mainstream discount broker should come in
at or below it. All sources are third-party broker reviews and broker margin
pages; check the broker's own pages before trading real money.

Per side, micro contracts (MES, MNQ):
* Commission $0.39: Tradovate/NinjaTrader free plan. IBKR Pro is $0.25.
  microfutures.app/learn/micro-futures-broker-commissions (2026-06-04),
  quantvps.com/blog/tradovate-review (2026-11-14),
  tradewize.io/blog/best-futures-brokers (2026-11-19).
* CME exchange fee $0.35 (IBKR lists $0.353), NFA fee $0.02: tradewize.io.
* Clearing $0.19: Tradovate, per quantvps.com.
  => $0.95/side, $1.90 round turn (IBKR all-in is about $1.24, Tradovate free $1.88).

MGC and 6E: the exchange fee was **not found** in any source read; the values
below are estimates and are flagged ``verified=False``. 6E commission uses
the IBKR standard-futures rate of $0.85 (interactivebrokers.com/en/pricing/
commissions-futures.php).

Margins (USD per contract):
* Day-trade: MES $50, MNQ $100 (Tradovate, Ironbeam, Optimus); MGC $125
  (Optimus, higher than Ironbeam's $100); 6E $250 (Ironbeam, Optimus).
  ironbeam.com/margins (2026-04-15), optimusfutures.com/Margin-Rates.php.
* Overnight initial: the higher of the posted figures. MES $2,924.50 and
  MNQ $4,589.02 (Tradovate, via quantvps.com 2026-11-14); MGC $2,640 and
  6E $2,970 (Ironbeam). Exchange margins change daily.
* Maintenance: initial / 1.10, the CME ratio Optimus describes; no source
  listed maintenance figures directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:  # avoid a runtime import cycle with contracts.py
    from quant_lab.futures.config.contracts import FuturesContractSpec


@dataclass(frozen=True)
class FuturesFeeSchedule:
    commission_per_side: float
    exchange_fee_per_side: float
    regulatory_fee_per_side: float
    clearing_fee_per_side: float
    intraday_margin: float
    initial_margin: float
    maintenance_margin: float
    slippage_ticks: float = 1.0
    verified: bool = True   # False when any figure is an estimate, not a published number

    @property
    def fees_per_side(self) -> float:
        return (self.commission_per_side + self.exchange_fee_per_side
                + self.regulatory_fee_per_side + self.clearing_fee_per_side)


@dataclass(frozen=True)
class FuturesBrokerProfile:
    name: str
    as_of: str                       # ISO date the numbers were collected
    schedules: Dict[str, FuturesFeeSchedule]
    sources: Tuple[str, ...] = field(default_factory=tuple)

    def schedule(self, symbol: str) -> FuturesFeeSchedule:
        try:
            return self.schedules[symbol.upper()]
        except KeyError:
            raise KeyError(f"profile {self.name!r} has no fees for {symbol!r}") from None

    def apply(self, spec: "FuturesContractSpec") -> "FuturesContractSpec":
        s = self.schedule(spec.symbol)
        return replace(
            spec,
            commission_per_side=s.commission_per_side,
            exchange_fee_per_side=s.exchange_fee_per_side,
            regulatory_fee_per_side=s.regulatory_fee_per_side,
            clearing_fee_per_side=s.clearing_fee_per_side,
            intraday_margin=s.intraday_margin,
            initial_margin=s.initial_margin,
            maintenance_margin=s.maintenance_margin,
            slippage_ticks=s.slippage_ticks,
        )


def _maint(initial: float) -> float:
    return round(initial / 1.10, 2)


PAPER_DEFAULT = FuturesBrokerProfile(
    name="paper_default",
    as_of="2026-11-27",
    sources=(
        "https://microfutures.app/learn/micro-futures-broker-commissions",
        "https://tradewize.io/blog/best-futures-brokers",
        "https://www.quantvps.com/blog/tradovate-review",
        "https://www.interactivebrokers.com/en/pricing/commissions-futures.php",
        "https://www.ironbeam.com/margins/",
        "https://optimusfutures.com/Margin-Rates.php",
    ),
    schedules={
        "MES": FuturesFeeSchedule(0.39, 0.35, 0.02, 0.19, 50.0, 2_924.50, _maint(2_924.50)),
        "MNQ": FuturesFeeSchedule(0.39, 0.35, 0.02, 0.19, 100.0, 4_589.02, _maint(4_589.02)),
        # Exchange fee estimated (not found in sources).
        "MGC": FuturesFeeSchedule(0.39, 0.60, 0.02, 0.19, 125.0, 2_640.0, _maint(2_640.0), verified=False),
        # Exchange fee estimated (not found in sources).
        "6E": FuturesFeeSchedule(0.85, 1.60, 0.02, 0.19, 250.0, 2_970.0, _maint(2_970.0), verified=False),
    },
)

PROFILES: Dict[str, FuturesBrokerProfile] = {PAPER_DEFAULT.name: PAPER_DEFAULT}
