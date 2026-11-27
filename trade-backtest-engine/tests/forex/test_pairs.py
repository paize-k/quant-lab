from dataclasses import replace
from datetime import date, datetime, timezone

import pytest

from quant_lab.core import InstrumentSpec
from quant_lab.forex.config import (
    ASIAN, LONDON_NY_OVERLAP, PAIRS, PAPER_DEFAULT, ForexCostProfile, ForexRiskLimits, PairCosts,
    get_pair, is_rollover_window,
)


def test_pip_sizes():
    assert get_pair("EUR_USD").pip_size == 0.0001
    assert get_pair("USD/JPY").pip_size == 0.01
    assert get_pair("EUR_JPY").price_precision == 3


def test_satisfies_instrument_protocol():
    assert isinstance(get_pair("EUR_USD"), InstrumentSpec)


def test_pip_value_conversion_usd_account():
    assert get_pair("EUR_USD").pip_value_per_unit("USD", 1.08) == pytest.approx(0.0001)
    # USD/JPY @ 150: 0.01 JPY / 150 = 0.0000667 USD per unit -> $6.67 per 100k
    assert get_pair("USD_JPY").pip_value_per_unit("USD", 150.0) * 100_000 == pytest.approx(6.6667, rel=1e-4)
    # EUR/JPY needs a JPY->USD cross
    with pytest.raises(KeyError):
        get_pair("EUR_JPY").pip_value_per_unit("USD", 162.0)
    pv = get_pair("EUR_JPY").pip_value_per_unit("USD", 162.0, rates={"USD_JPY": 150.0})
    assert pv == pytest.approx(0.01 / 150.0)


def test_units_for_risk_rounded_to_lot():
    eur = get_pair("EUR_USD")
    # $10k * 1% / (20 pips * $0.0001) = 50,000 units
    assert eur.units_for_risk(10_000, 0.01, 20, "USD", 1.08) == 50_000
    micro = replace(eur, lot_step=1_000, min_units=1_000)
    assert micro.units_for_risk(10_000, 0.01, 23, "USD", 1.08) == 43_000
    assert micro.units_for_risk(10, 0.01, 23, "USD", 1.08) == 0


def test_short_pnl_sign():
    assert get_pair("EUR_USD").pnl(1.1000, 1.0950, 10_000, -1) == pytest.approx(50.0)


def test_triple_swap_wednesday():
    eur = get_pair("EUR_USD")  # paper default: long -$8.83 per 100k per night
    assert eur.swap_nights(date(2026, 11, 25)) == 3   # Wednesday
    assert eur.swap_nights(date(2026, 11, 24)) == 1
    assert eur.swap_nights(date(2026, 11, 28)) == 0   # Saturday
    assert eur.swap_amount(+1, 10_000, date(2026, 11, 25)) == pytest.approx(-0.883 * 3)
    assert eur.swap_amount(-1, 100_000, date(2026, 11, 24)) == pytest.approx(2.33)


def test_paper_default_costs():
    eur = get_pair("EUR_USD")
    assert eur.typical_spread_pips == 1.6 and eur.margin_rate == 0.02
    assert eur.spread_pips(in_rollover=True) == pytest.approx(4.8)
    assert eur.commission(100_000) == 0
    assert set(PAPER_DEFAULT.costs) == set(PAIRS)
    bare = get_pair("EUR_USD", profile=None)
    assert bare.typical_spread_pips == 0 and bare.swap_long_per_100k == 0


def test_custom_profile_swaps_in():
    core = ForexCostProfile("core", "2026-11-27", "USD", {"EUR_USD": PairCosts(0.4, -9.0, 2.0, 0.5)})
    eur = get_pair("EUR_USD", profile=core)
    assert eur.commission(100_000) == pytest.approx(10.0)  # $0.50/10k/side, round trip
    with pytest.raises(KeyError):
        get_pair("GBP_USD", profile=core)


def test_sessions():
    utc = timezone.utc
    assert LONDON_NY_OVERLAP.contains(datetime(2026, 11, 26, 14, 0, tzinfo=utc))
    assert ASIAN.contains(datetime(2026, 11, 26, 3, 0, tzinfo=utc))
    assert is_rollover_window(datetime(2026, 11, 26, 21, 30, tzinfo=utc))
    assert not is_rollover_window(datetime(2026, 11, 26, 23, 0, tzinfo=utc))


def test_risk_limits():
    r = ForexRiskLimits(per_currency_overrides={"JPY": 0.005})
    assert r.currency_cap("usd") == 0.01 and r.currency_cap("jpy") == 0.005
