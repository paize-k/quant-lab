"""CME futures contract specifications.

Tick sizes, tick values and contract months come from the CME contract
specs. **Margins and fees are placeholders**: CME changes margins often and
commissions depend on your broker, so check them against your broker before
trusting any backtest or sizing number. Override per run with
``dataclasses.replace(get_contract("MES"), initial_margin=...)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, Tuple

from quant_lab.core.sizing import size_by_risk

MONTH_CODES: Dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}


@dataclass(frozen=True)
class FuturesContractSpec:
    symbol: str                    # root, e.g. "MES"
    name: str
    exchange: str                  # CME, COMEX, ...
    tick_size: float               # minimum price increment, in index points / price units
    tick_value: float              # USD per tick per contract
    contract_months: Tuple[str, ...]  # month codes that are traded/rolled, e.g. ("H","M","U","Z")
    sessions_key: str              # key into futures.config.sessions.SESSIONS
    currency: str = "USD"
    initial_margin: float = 0.0    # USD per contract (overnight), placeholder
    maintenance_margin: float = 0.0
    intraday_margin: float = 0.0   # broker day-trade margin, placeholder; 0 = use initial
    commission_per_side: float = 0.0   # broker commission per contract per side
    exchange_fee_per_side: float = 0.0
    regulatory_fee_per_side: float = 0.0  # NFA etc.
    slippage_ticks: float = 1.0    # min slippage applied to market orders, per side
    roll_days_before_expiry: int = 8   # default roll offset for the continuous series

    def __post_init__(self) -> None:
        if self.tick_size <= 0 or self.tick_value <= 0:
            raise ValueError("tick_size and tick_value must be positive")
        bad = [m for m in self.contract_months if m not in MONTH_CODES]
        if bad:
            raise ValueError(f"unknown month codes {bad}")
        if self.maintenance_margin > self.initial_margin > 0:
            raise ValueError("maintenance_margin cannot exceed initial_margin")

    # ---- price/tick math -------------------------------------------------
    @property
    def point_value(self) -> float:
        """USD per 1.0 price move per contract (the multiplier)."""
        return self.tick_value / self.tick_size

    def round_price(self, price: float) -> float:
        ticks = round(price / self.tick_size)
        # Round the product too so 0.00005 * n doesn't leave float dust.
        return round(ticks * self.tick_size, 10)

    def price_to_ticks(self, price_delta: float) -> float:
        return round(price_delta / self.tick_size, 6)

    def ticks_to_price(self, ticks: float) -> float:
        return ticks * self.tick_size

    def pnl(self, entry_price: float, exit_price: float, quantity: float, side: int) -> float:
        """Gross P&L in USD before fees and slippage. ``side`` is +1/-1."""
        return (exit_price - entry_price) * int(side) * quantity * self.point_value

    # ---- friction --------------------------------------------------------
    @property
    def fees_per_side(self) -> float:
        return self.commission_per_side + self.exchange_fee_per_side + self.regulatory_fee_per_side

    def round_trip_fees(self, contracts: float) -> float:
        return 2.0 * self.fees_per_side * abs(contracts)

    def slippage_cost(self, contracts: float, sides: int = 2) -> float:
        return self.slippage_ticks * self.tick_value * abs(contracts) * sides

    # ---- sizing ----------------------------------------------------------
    def margin_per_contract(self, intraday: bool = True) -> float:
        if intraday and self.intraday_margin > 0:
            return self.intraday_margin
        return self.initial_margin

    def contracts_for_risk(
        self,
        equity: float,
        risk_fraction: float,
        stop_ticks: float,
        available_margin: float | None = None,
        intraday: bool = True,
    ) -> int:
        """floor(equity * risk / (stop_ticks * tick_value)), capped by margin."""
        n = int(size_by_risk(equity, risk_fraction, stop_ticks, self.tick_value, step=1.0))
        margin = self.margin_per_contract(intraday)
        if available_margin is not None and margin > 0:
            n = min(n, int(math.floor(available_margin / margin)))
        return max(n, 0)

    # ---- contract months -------------------------------------------------
    def contract_code(self, year: int, month: int) -> str:
        """e.g. ``MESZ6`` for Dec 2026 (single-digit year, CME style)."""
        code = next((c for c, m in MONTH_CODES.items() if m == month), None)
        if code is None or code not in self.contract_months:
            raise ValueError(f"{self.symbol} has no {month:02d} contract")
        return f"{self.symbol}{code}{year % 10}"

    def listed_months(self) -> Tuple[int, ...]:
        return tuple(sorted(MONTH_CODES[c] for c in self.contract_months))

    def front_month(self, on: date) -> Tuple[int, int]:
        """(year, month) of the nearest listed contract month on/after ``on``.

        This uses calendar months only; exact expiry and roll dates are the
        job of the Milestone 2 rollover handler.
        """
        for offset in range(0, 13):
            y, m = divmod(on.month - 1 + offset, 12)
            year, month = on.year + y, m + 1
            if month in self.listed_months():
                return year, month
        raise RuntimeError("no listed month found")  # unreachable for valid specs


_QUARTERLY = ("H", "M", "U", "Z")

CONTRACTS: Dict[str, FuturesContractSpec] = {
    "MES": FuturesContractSpec(
        symbol="MES", name="Micro E-mini S&P 500", exchange="CME",
        tick_size=0.25, tick_value=1.25, contract_months=_QUARTERLY,
        sessions_key="cme_equity",
        initial_margin=2_400.0, maintenance_margin=2_200.0, intraday_margin=50.0,
        commission_per_side=0.25, exchange_fee_per_side=0.35, regulatory_fee_per_side=0.02,
    ),
    "MNQ": FuturesContractSpec(
        symbol="MNQ", name="Micro E-mini Nasdaq-100", exchange="CME",
        tick_size=0.25, tick_value=0.50, contract_months=_QUARTERLY,
        sessions_key="cme_equity",
        initial_margin=3_500.0, maintenance_margin=3_200.0, intraday_margin=100.0,
        commission_per_side=0.25, exchange_fee_per_side=0.35, regulatory_fee_per_side=0.02,
    ),
    "MGC": FuturesContractSpec(
        symbol="MGC", name="Micro Gold", exchange="COMEX",
        tick_size=0.10, tick_value=1.00, contract_months=("G", "J", "M", "Q", "V", "Z"),
        sessions_key="cme_metals",
        initial_margin=1_800.0, maintenance_margin=1_650.0, intraday_margin=100.0,
        commission_per_side=0.25, exchange_fee_per_side=0.60, regulatory_fee_per_side=0.02,
    ),
    "6E": FuturesContractSpec(
        symbol="6E", name="Euro FX", exchange="CME",
        tick_size=0.00005, tick_value=6.25, contract_months=_QUARTERLY,
        sessions_key="cme_fx",
        initial_margin=2_700.0, maintenance_margin=2_450.0, intraday_margin=500.0,
        commission_per_side=0.85, exchange_fee_per_side=1.60, regulatory_fee_per_side=0.02,
    ),
}


def get_contract(symbol: str) -> FuturesContractSpec:
    try:
        return CONTRACTS[symbol.upper()]
    except KeyError:
        raise KeyError(f"unknown futures contract {symbol!r}; known: {sorted(CONTRACTS)}") from None
