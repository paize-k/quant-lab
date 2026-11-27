"""Broker cost profiles for the forex engine.

A profile holds everything about a pair that depends on *who you trade
through*: spread, commission, overnight swap and margin. Pair specs stay
market-convention only; ``get_pair(symbol, profile)`` combines the two. To use
a real broker later, build a ``ForexCostProfile`` from its pricing pages (or
from its API: OANDA v20 serves live financing rates) and pass it in.

PAPER_DEFAULT (researched 2026-11-27, USD account, spread-only pricing)
-----------------------------------------------------------------------
Deliberately *conservative* for paper trading and backtests while no broker
is chosen: each spread is the widest of the retail OANDA figures found.
All sources are third-party reviews; check the broker's own pages before
trading real money.

Spreads in pips (widest of): bestbrokers.com/reviews/oanda/spreads-fees-
and-commissions (2026-02-19), iamforextrader.com/en/broker/oanda/united-states
(2026-06-12, US Standard), newyorkcityservers.com/blog/oanda-review
(2026-10-01), compareforexbrokers.com/us/oanda-review (2026-11-13, EUR/USD
1.4 advertised), forexbrokers.com/reviews/oanda (measured EUR/USD average
1.68 incl. rollover, Aug 2025, which is why EUR/USD is 1.6 here).

Commission: 0, because spread-only accounts build the cost into the spread.
(OANDA "core" pricing is tighter spreads plus roughly $5 per 100k per side,
i.e. $0.50 per 10k; sources disagree and it may not be offered to US clients.)

Swaps: OANDA table dated 2026-02-19 via bestbrokers.com, USD per 100k units
per night. Swaps track central-bank rates and move often; these are stale and
are only there so carry is not ignored. Replace with live rates.

Rollover spread widening: 3x normal from 21:00 to 23:00 UTC is an estimate,
not a published figure.

Margin: US retail limits are 2% of notional on major currencies (50:1) and 5%
on others (20:1), per 17 CFR 5.9 as summarised by compareforexbrokers.com.
All seven pairs here are majors.

Triple swap on Wednesday is the standard T+2 spot convention; none of the
sources read confirmed OANDA's exact rule, so check it when you pick a broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:  # avoid a runtime import cycle with pairs.py
    from quant_lab.forex.config.pairs import ForexPairSpec


@dataclass(frozen=True)
class PairCosts:
    spread_pips: float
    swap_long_per_100k: float
    swap_short_per_100k: float
    commission_per_10k_per_side: float = 0.0
    margin_rate: float = 0.02
    rollover_spread_multiplier: float = 3.0
    verified: bool = True


@dataclass(frozen=True)
class ForexCostProfile:
    name: str
    as_of: str
    account_currency: str
    costs: Dict[str, PairCosts]
    sources: Tuple[str, ...] = field(default_factory=tuple)

    def pair_costs(self, symbol: str) -> PairCosts:
        try:
            return self.costs[symbol.upper()]
        except KeyError:
            raise KeyError(f"profile {self.name!r} has no costs for {symbol!r}") from None

    def apply(self, spec: "ForexPairSpec") -> "ForexPairSpec":
        c = self.pair_costs(spec.symbol)
        return replace(
            spec,
            typical_spread_pips=c.spread_pips,
            rollover_spread_multiplier=c.rollover_spread_multiplier,
            commission_per_10k_per_side=c.commission_per_10k_per_side,
            swap_long_per_100k=c.swap_long_per_100k,
            swap_short_per_100k=c.swap_short_per_100k,
            margin_rate=c.margin_rate,
            account_currency=self.account_currency,
        )


PAPER_DEFAULT = ForexCostProfile(
    name="paper_default",
    as_of="2026-11-27",
    account_currency="USD",
    sources=(
        "https://www.bestbrokers.com/reviews/oanda/spreads-fees-and-commissions/",
        "https://iamforextrader.com/en/broker/oanda/united-states/",
        "https://newyorkcityservers.com/blog/oanda-review",
        "https://www.compareforexbrokers.com/us/oanda-review/",
        "https://www.forexbrokers.com/reviews/oanda",
    ),
    costs={
        #                    spread  swap long  swap short
        "EUR_USD": PairCosts(1.6,    -8.83,     2.33),
        "GBP_USD": PairCosts(2.0,    -3.66,    -3.81),
        "AUD_USD": PairCosts(1.6,    -1.80,    -2.13),
        "USD_JPY": PairCosts(1.6,     5.48,   -11.01),
        "USD_CHF": PairCosts(1.8,     7.73,   -13.48),
        "USD_CAD": PairCosts(2.2,     3.30,   -21.63),
        "EUR_JPY": PairCosts(1.8,     0.87,    -7.44),
    },
)

PROFILES: Dict[str, ForexCostProfile] = {PAPER_DEFAULT.name: PAPER_DEFAULT}
