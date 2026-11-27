from dataclasses import replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from quant_lab.core import InstrumentSpec
from quant_lab.futures.config import (
    PAPER_DEFAULT, FuturesBrokerProfile, FuturesFeeSchedule, FuturesRiskLimits, get_contract, get_sessions,
)

CT = ZoneInfo("America/Chicago")


@pytest.mark.parametrize("sym, point_value", [("MES", 5.0), ("MNQ", 2.0), ("MGC", 10.0), ("6E", 125_000.0)])
def test_point_values(sym, point_value):
    assert get_contract(sym).point_value == pytest.approx(point_value)


def test_satisfies_instrument_protocol():
    assert isinstance(get_contract("MES"), InstrumentSpec)


def test_tick_math_and_pnl():
    mes = get_contract("MES")
    assert mes.round_price(5012.37) == 5012.25
    assert mes.price_to_ticks(2.0) == 8
    assert mes.pnl(5000, 5010, 2, +1) == pytest.approx(100.0)  # 10 pts * $5 * 2
    assert mes.pnl(5000, 5010, 2, -1) == pytest.approx(-100.0)
    e6 = get_contract("6e")
    assert e6.round_price(1.083337) == pytest.approx(1.08335)
    assert e6.pnl(1.0800, 1.0810, 1, 1) == pytest.approx(125.0)


def test_fees_and_slippage():
    mes = replace(get_contract("MES"), commission_per_side=0.5, exchange_fee_per_side=0.3,
                  regulatory_fee_per_side=0.02, clearing_fee_per_side=0.0)
    assert mes.round_trip_fees(3) == pytest.approx(2 * 0.82 * 3)
    assert mes.slippage_cost(3) == pytest.approx(1 * 1.25 * 3 * 2)


def test_contracts_for_risk_capped_by_margin():
    mes = replace(get_contract("MES"), intraday_margin=500.0, initial_margin=2_400.0, maintenance_margin=2_200.0)
    # $50k * 1% / (8 ticks * $1.25) = 50 contracts
    assert mes.contracts_for_risk(50_000, 0.01, 8) == 50
    assert mes.contracts_for_risk(50_000, 0.01, 8, available_margin=10_000) == 20
    assert mes.contracts_for_risk(50_000, 0.01, 8, available_margin=10_000, intraday=False) == 4


def test_contract_months():
    mes = get_contract("MES")
    assert mes.contract_code(2026, 12) == "MESZ6"
    assert mes.front_month(date(2026, 11, 27)) == (2026, 12)
    assert mes.front_month(date(2026, 12, 30)) == (2026, 12)
    assert get_contract("MGC").front_month(date(2026, 3, 1)) == (2026, 4)
    with pytest.raises(ValueError):
        mes.contract_code(2026, 1)
    with pytest.raises(KeyError):
        get_contract("ES_NOT_REAL")


def test_sessions_rth_and_flatten():
    s = get_sessions(get_contract("MES").sessions_key)
    assert s.is_tradable(datetime(2026, 11, 30, 9, 0, tzinfo=CT), rth_only=True)
    assert not s.is_tradable(datetime(2026, 11, 30, 16, 30, tzinfo=CT))  # maintenance
    assert s.is_tradable(datetime(2026, 11, 29, 18, 0, tzinfo=CT))      # Sunday Globex open
    assert s.should_flatten(datetime(2026, 11, 30, 14, 51, tzinfo=CT))
    assert not s.should_flatten(datetime(2026, 11, 30, 14, 40, tzinfo=CT))


def test_risk_limits_validate():
    assert FuturesRiskLimits().max_daily_loss == 0.02
    with pytest.raises(ValueError):
        FuturesRiskLimits(risk_per_trade=2.0)


def test_paper_default_profile_applied():
    mes = get_contract("MES")
    # $0.39 commission + $0.35 CME + $0.02 NFA + $0.19 clearing, both sides
    assert mes.round_trip_fees(1) == pytest.approx(1.90)
    assert mes.intraday_margin == 50 and mes.initial_margin > mes.maintenance_margin > 0
    assert all(sym in PAPER_DEFAULT.schedules for sym in ("MES", "MNQ", "MGC", "6E"))
    assert not PAPER_DEFAULT.schedule("6E").verified
    bare = get_contract("MES", profile=None)
    assert bare.round_trip_fees(1) == 0 and bare.initial_margin == 0


def test_custom_profile_swaps_in():
    cheap = FuturesBrokerProfile("cheap", "2026-11-27", {
        "MES": FuturesFeeSchedule(0.25, 0.35, 0.02, 0.0, 40.0, 2_000.0, 1_800.0)})
    mes = get_contract("MES", profile=cheap)
    assert mes.round_trip_fees(1) == pytest.approx(1.24)
    assert mes.margin_per_contract() == 40
    with pytest.raises(KeyError):
        get_contract("MNQ", profile=cheap)
