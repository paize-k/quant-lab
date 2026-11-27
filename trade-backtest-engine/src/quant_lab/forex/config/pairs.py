"""Spot FX pair specifications.

Pip sizes are market convention (0.0001, or 0.01 for JPY quotes). Typical
spreads, swap rates and margin rates are **placeholders**: they vary by
broker and change daily, so load real values from your broker before relying
on a backtest. Swaps are expressed in pips per unit per night, so they scale
with position size the same way price P&L does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Mapping, Optional

from quant_lab.core.sizing import floor_to_step, size_by_risk


@dataclass(frozen=True)
class ForexPairSpec:
    symbol: str                   # OANDA style, e.g. "EUR_USD"
    base: str
    quote: str
    pip_size: float               # 0.0001, or 0.01 for JPY quotes
    price_precision: int          # quoted decimals (fractional pips), e.g. 5 or 3
    lot_step: float = 1.0         # 1 for OANDA units, 1_000 for micro lots
    min_units: float = 1.0
    margin_rate: float = 0.0333   # placeholder (30:1)
    typical_spread_pips: float = 1.0      # placeholder normal-hours spread
    rollover_spread_multiplier: float = 3.0  # spread widening 21:00-23:00 UTC
    swap_long_pips: float = 0.0   # financing per unit per night, in pips (+ = credit)
    swap_short_pips: float = 0.0
    triple_swap_weekday: int = 2  # Wednesday rollover books the weekend (T+2 spot)

    def __post_init__(self) -> None:
        if self.symbol != f"{self.base}_{self.quote}":
            raise ValueError(f"symbol {self.symbol!r} must be '{self.base}_{self.quote}'")
        if self.pip_size <= 0 or self.lot_step <= 0:
            raise ValueError("pip_size and lot_step must be positive")
        if not (0 <= self.triple_swap_weekday <= 6):
            raise ValueError("triple_swap_weekday must be 0..6")

    # ---- price/pip math --------------------------------------------------
    @property
    def tick_size(self) -> float:
        return 10.0 ** -self.price_precision

    def round_price(self, price: float) -> float:
        return round(price, self.price_precision)

    def price_to_pips(self, price_delta: float) -> float:
        return round(price_delta / self.pip_size, 6)

    def pips_to_price(self, pips: float) -> float:
        return pips * self.pip_size

    # ---- currency conversion --------------------------------------------
    def quote_to_account_rate(
        self, account_currency: str, price: float, rates: Optional[Mapping[str, float]] = None
    ) -> float:
        """Account-currency value of 1 unit of the quote currency.

        Handles quote == account (EUR/USD with a USD account), base == account
        (USD/JPY with a USD account, divide by price), and otherwise looks up
        a cross in ``rates`` keyed like ``"JPY_USD"`` or ``"USD_JPY"``.
        """
        acct = account_currency.upper()
        if self.quote == acct:
            return 1.0
        if self.base == acct:
            if price <= 0:
                raise ValueError("price must be positive")
            return 1.0 / price
        rates = rates or {}
        direct, inverse = f"{self.quote}_{acct}", f"{acct}_{self.quote}"
        if direct in rates:
            return float(rates[direct])
        if inverse in rates and rates[inverse] > 0:
            return 1.0 / float(rates[inverse])
        raise KeyError(f"need a {direct} or {inverse} rate to value {self.symbol} in {acct}")

    def pip_value_per_unit(
        self, account_currency: str, price: float, rates: Optional[Mapping[str, float]] = None
    ) -> float:
        return self.pip_size * self.quote_to_account_rate(account_currency, price, rates)

    def pnl(
        self, entry_price: float, exit_price: float, quantity: float, side: int,
        account_currency: str = "USD", rates: Optional[Mapping[str, float]] = None,
    ) -> float:
        """Gross P&L in account currency, valued at the exit price."""
        quote_pnl = (exit_price - entry_price) * int(side) * quantity
        return quote_pnl * self.quote_to_account_rate(account_currency, exit_price, rates)

    # ---- sizing ----------------------------------------------------------
    def round_units(self, units: float) -> float:
        u = floor_to_step(units, self.lot_step)
        return u if u >= self.min_units else 0.0

    def units_for_risk(
        self, equity: float, risk_fraction: float, stop_pips: float,
        account_currency: str, price: float, rates: Optional[Mapping[str, float]] = None,
    ) -> float:
        """floor(equity * risk / (stop_pips * pip_value_per_unit)), to lot step."""
        pv = self.pip_value_per_unit(account_currency, price, rates)
        return self.round_units(size_by_risk(equity, risk_fraction, stop_pips, pv, step=self.lot_step))

    # ---- financing -------------------------------------------------------
    def swap_nights(self, rollover_day: date) -> int:
        """Nights of financing charged at the rollover on ``rollover_day``."""
        wd = rollover_day.weekday()
        if wd >= 5:
            return 0  # no rollover on Sat/Sun
        return 3 if wd == self.triple_swap_weekday else 1

    def swap_pips(self, side: int, rollover_day: date) -> float:
        per_night = self.swap_long_pips if side > 0 else self.swap_short_pips
        return per_night * self.swap_nights(rollover_day)


def _pair(base: str, quote: str, **kw) -> ForexPairSpec:
    jpy = quote == "JPY"
    return ForexPairSpec(
        symbol=f"{base}_{quote}", base=base, quote=quote,
        pip_size=0.01 if jpy else 0.0001, price_precision=3 if jpy else 5, **kw,
    )


PAIRS: Dict[str, ForexPairSpec] = {
    p.symbol: p
    for p in (
        _pair("EUR", "USD", typical_spread_pips=0.8),
        _pair("GBP", "USD", typical_spread_pips=1.2),
        _pair("AUD", "USD", typical_spread_pips=1.0),
        _pair("USD", "JPY", typical_spread_pips=1.0),
        _pair("USD", "CHF", typical_spread_pips=1.5),
        _pair("USD", "CAD", typical_spread_pips=1.6),
        _pair("EUR", "JPY", typical_spread_pips=1.6),
    )
}


def get_pair(symbol: str) -> ForexPairSpec:
    key = symbol.upper().replace("/", "_")
    try:
        return PAIRS[key]
    except KeyError:
        raise KeyError(f"unknown FX pair {symbol!r}; known: {sorted(PAIRS)}") from None
