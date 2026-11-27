from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from quant_lab.core import Bar, Signal, SignalAction
from quant_lab.core.data import bars_from_dataframe
from quant_lab.core.types import OrderType, Position, Side
from quant_lab.futures.config import FuturesRiskLimits, get_contract, get_sessions
from quant_lab.futures.data.synthetic import synthetic_intraday
from quant_lab.futures.engine import FuturesExecutionModel, run_backtest
from quant_lab.futures.strategies import ALL_STRATEGIES, FuturesHybrid

CT = ZoneInfo("America/Chicago")
MES = get_contract("MES")


def bar(ts, px=5000.0, symbol="MESH6"):
    return Bar(ts=ts, open=px, high=px + 1, low=px - 1, close=px, symbol=symbol)


def test_fill_price_rounds_and_slips_one_tick():
    m = FuturesExecutionModel(MES)
    b = bar(datetime(2026, 3, 2, 9, 0, tzinfo=CT))
    assert m.fill_price(+1, 5000.1, b, OrderType.MARKET) == 5000.25   # 5000.00 + 1 tick
    assert m.fill_price(-1, 5000.1, b, OrderType.MARKET) == 4999.75
    assert m.fill_price(+1, 5000.1, b, OrderType.LIMIT) == 5000.0


def test_sizing_uses_ticks_and_margin():
    m = FuturesExecutionModel(MES)
    b = bar(datetime(2026, 3, 2, 9, 0, tzinfo=CT))
    # $100k * 0.5% = $500; 4 points = 16 ticks * $1.25 = $20 -> 25 contracts
    assert m.size(100_000, 1, 5000, 4996, 0.005, b) == 25
    m.max_contracts = 10
    assert m.size(100_000, 1, 5000, 4996, 0.005, b) == 10


def test_roll_charges_round_turn():
    m = FuturesExecutionModel(MES)
    t = datetime(2026, 3, 9, 9, 0, tzinfo=CT)
    held = Position(side=Side.LONG, quantity=2, entry_price=5000)
    cost = m.carry(held, bar(t, symbol="MESH6"), bar(t, symbol="MESM6"))
    assert cost == pytest.approx(-(MES.round_trip_fees(2) + MES.slippage_cost(2)))
    assert m.carry(held, bar(t), bar(t)) == 0
    assert m.carry(Position(), bar(t, symbol="MESH6"), bar(t, symbol="MESM6")) == 0


def test_session_rules():
    m = FuturesExecutionModel(MES, sessions=get_sessions("cme_equity"))
    assert m.entry_allowed(bar(datetime(2026, 3, 2, 10, 0, tzinfo=CT)))
    assert not m.entry_allowed(bar(datetime(2026, 3, 2, 7, 0, tzinfo=CT)))    # pre-market
    assert not m.entry_allowed(bar(datetime(2026, 3, 2, 14, 55, tzinfo=CT)))  # flatten window
    assert m.must_flatten(bar(datetime(2026, 3, 2, 14, 50, tzinfo=CT)))
    assert not m.must_flatten(bar(datetime(2026, 3, 2, 14, 0, tzinfo=CT)))


@pytest.fixture(scope="module")
def synth_bars():
    df = synthetic_intraday(MES, days=30, bar_minutes=5, seed=11)
    return list(bars_from_dataframe(df, symbol="MES", tz="America/Chicago"))


@pytest.mark.parametrize("cls", ALL_STRATEGIES)
def test_strategies_trade_and_respect_rth(cls, synth_bars):
    res = run_backtest(cls(), synth_bars, MES)
    assert res.trades, f"{cls.__name__} made no trades"
    rth = get_sessions("cme_equity").rth
    for t in res.trades:
        local = t.entry_ts.astimezone(CT)
        assert rth.contains(local), f"entry outside RTH at {local}"
        assert t.exit_ts.astimezone(CT).date() == local.date()  # flat by the close
        assert t.fees == pytest.approx(MES.round_trip_fees(t.qty))  # entry + all exits
        assert t.qty <= FuturesRiskLimits().max_contracts_per_symbol


def test_auto_flatten_before_close(synth_bars):
    res = run_backtest(FuturesHybrid(), synth_bars, MES)
    for t in res.trades:
        if t.exit_reason == "session_flatten":
            assert t.exit_ts.astimezone(CT).time() >= datetime(2026, 1, 1, 14, 50).time()


def test_hybrid_tags_name_the_sub_strategy(synth_bars):
    res = run_backtest(FuturesHybrid(), synth_bars, MES)
    tags = {t.tag for t in res.trades}
    assert tags and all(t.startswith("futures_hybrid:") for t in tags)


def test_daily_bars_without_sessions():
    import numpy as np
    import pandas as pd
    from quant_lab.futures.strategies import TrendMomentum

    rng = np.random.default_rng(5)
    px = 5000 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 400)))
    idx = pd.date_range("2024-01-01", periods=400, freq="B", tz="America/Chicago")
    df = pd.DataFrame({"open": px, "high": px * 1.005, "low": px * 0.995, "close": px}, index=idx)
    bars = list(bars_from_dataframe(df, symbol="MES", tz="America/Chicago"))
    res = run_backtest(TrendMomentum(fast=10, slow=30, adx_min=10), bars, MES, use_sessions=False)
    assert res.trades
