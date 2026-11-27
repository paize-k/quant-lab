"""The minimal interface the shared backtest loop needs from an instrument.

``FuturesContractSpec`` and ``ForexPairSpec`` both satisfy this protocol, but
each engine owns its spec, fee model and sessions. ``core`` only relies on
the methods below to round prices and turn price moves into account money.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class InstrumentSpec(Protocol):
    symbol: str

    def round_price(self, price: float) -> float:
        """Round a price to the instrument's minimum increment."""
        ...

    def pnl(self, entry_price: float, exit_price: float, quantity: float, side: int) -> float:
        """Gross P&L in account currency, before fees, spread and financing."""
        ...
