from datetime import date, datetime, timezone

import pytest

from quant_lab.core import InstrumentSpec
from quant_lab.forex.config import (
    ForexRiskLimits, LONDON_NY_OVERLAP, ASIAN, get_pair, is_rollover_window,
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
    micro = type(eur)(**{**eur.__dict__, "lot_step": 1_000, "min_units": 1_000})
    assert micro.units_for_risk(10_000, 0.01, 23, "USD", 1.08) == 43_000
    assert micro.units_for_risk(10, 0.01, 23, "USD", 1.08) == 0


def test_short_pnl_sign():
    assert get_pair("EUR_USD").pnl(1.1000, 1.0950, 10_000, -1) == pytest.approx(50.0)


def test_triple_swap_wednesday():
    p = type(get_pair("EUR_USD"))(**{**get_pair("EUR_USD").__dict__, "swap_long_pips": -0.5})
    assert p.swap_nights(date(2026, 11, 25)) == 3   # Wednesday
    assert p.swap_nights(date(2026, 11, 24)) == 1
    assert p.swap_nights(date(2026, 11, 28)) == 0   # Saturday
    assert p.swap_pips(+1, date(2026, 11, 25)) == pytest.approx(-1.5)


def test_sessions():
    utc = timezone.utc
    assert LONDON_NY_OVERLAP.contains(datetime(2026, 11, 26, 14, 0, tzinfo=utc))
    assert ASIAN.contains(datetime(2026, 11, 26, 3, 0, tzinfo=utc))
    assert is_rollover_window(datetime(2026, 11, 26, 21, 30, tzinfo=utc))
    assert not is_rollover_window(datetime(2026, 11, 26, 23, 0, tzinfo=utc))


def test_risk_limits():
    r = ForexRiskLimits(per_currency_overrides={"JPY": 0.005})
    assert r.currency_cap("usd") == 0.01 and r.currency_cap("jpy") == 0.005
