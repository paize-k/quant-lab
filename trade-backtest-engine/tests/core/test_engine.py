import math

import pytest

from quant_lab.core import Backtester, BacktestConfig, ScaleOutRule, Signal, SignalAction
from quant_lab.core.types import OrderType
from tests.core.helpers import Script, UnitModel, make_bars

L, S, X = SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT, SignalAction.EXIT
CFG = BacktestConfig(initial_equity=10_000, risk_per_trade=0.01)


def run(plan, ohlc, model=None, cfg=CFG):
    return Backtester(Script(plan=plan), model or UnitModel(), cfg).run(make_bars(ohlc))


def test_signal_fills_at_next_bar_open_never_same_close():
    ohlc = [(100, 101, 99, 100), (103, 104, 102, 103), (103, 104, 102, 104)]
    r = run({0: Signal(L, stop_loss=90), 1: Signal(X)}, ohlc)
    t = r.trades[0]
    assert t.entry_price == 103          # bar 1 open, not bar 0 close (100)
    assert t.exit_price == 103           # exit decided on bar 1, filled bar 2 open
    assert t.entry_ts == r.equity.index[1]
    # risk 1% of 10k = 100; stop distance 13 -> 7 units
    assert t.qty == 7


def test_slippage_and_fees_reduce_net():
    ohlc = [(100, 101, 99, 100), (100, 101, 99, 100), (110, 111, 109, 110)]
    r = run({0: Signal(L, stop_loss=90), 1: Signal(X)}, ohlc, UnitModel(fee=2, slip=0.5))
    t = r.trades[0]
    assert t.entry_price == 100.5 and t.exit_price == 109.5
    assert t.fees == 4
    assert t.net_pnl == pytest.approx(t.gross_pnl - 4)
    assert t.slippage == pytest.approx(1.0 * t.qty)
    assert r.equity.iloc[-1] == pytest.approx(10_000 + t.net_pnl)


def test_stop_and_target_same_bar_assumes_stop():
    ohlc = [(100, 100, 100, 100), (100, 101, 99, 100), (100, 120, 80, 100)]
    r = run({0: Signal(L, stop_loss=95, take_profit=110)}, ohlc)
    assert r.trades[0].exit_reason == "stop" and r.trades[0].exit_price == 95


def test_gap_through_stop_fills_at_open():
    ohlc = [(100, 100, 100, 100), (100, 101, 99, 100), (90, 91, 89, 90)]
    r = run({0: Signal(L, stop_loss=95)}, ohlc)
    assert r.trades[0].exit_price == 90


def test_target_hit():
    r = run({0: Signal(S, stop_loss=120, take_profit=90), 1: Signal.hold()},
            [(100, 100, 100, 100), (100, 101, 99, 100), (95, 96, 85, 88)])
    t = r.trades[0]
    assert t.side == -1 and t.exit_reason == "target" and t.exit_price == 90
    assert t.gross_pnl > 0


def test_stop_entry_only_when_triggered():
    plan = {0: Signal(L, OrderType.STOP, entry_price=105, stop_loss=100)}
    r = run(plan, [(100, 100, 100, 100), (101, 104, 100, 103), (103, 104, 102, 103)])
    assert not r.trades  # never traded through 105 on the next bar; order expired
    r = run(plan, [(100, 100, 100, 100), (101, 106, 100, 105), (105, 106, 104, 105)])
    assert r.trades[0].entry_price == 105


def test_entry_blocked_when_not_allowed():
    r = run({0: Signal(L, stop_loss=90)}, [(100, 100, 100, 100)] * 3, UnitModel(allow=False))
    assert not r.trades


def test_no_stop_no_trade():
    r = run({0: Signal(L)}, [(100, 100, 100, 100)] * 3)
    assert not r.trades


def test_reversal_closes_then_opens():
    ohlc = [(100, 100, 100, 100), (100, 101, 99, 100), (102, 103, 101, 102), (101, 102, 100, 101)]
    r = run({0: Signal(L, stop_loss=90), 1: Signal(S, stop_loss=112)}, ohlc)
    assert [t.side for t in r.trades] == [1, -1]
    assert r.trades[0].exit_reason == "reverse"


def test_modify_moves_stop():
    ohlc = [(100, 100, 100, 100), (100, 101, 99, 100), (100, 101, 97, 100)]
    r = run({0: Signal(L, stop_loss=90), 1: Signal(SignalAction.MODIFY, stop_loss=98)}, ohlc)
    assert r.trades[0].exit_reason == "stop" and r.trades[0].exit_price == 98


def test_scale_out_then_breakeven():
    cfg = BacktestConfig(initial_equity=10_000, risk_per_trade=0.01,
                         scale_out=ScaleOutRule(1.5, 0.5, breakeven_offset=0.25))
    # entry 100, stop 90 (10 pts), 1.5R = 115; next bar back to 100.1 hits BE stop at 100.25
    ohlc = [(100, 100, 100, 100), (100, 101, 99, 100), (101, 116, 100, 112), (112, 112, 100.1, 101)]
    r = run({0: Signal(L, stop_loss=90)}, ohlc, cfg=cfg)
    t = r.trades[0]
    assert t.qty == 10
    # 5 units +15, 5 units +0.25
    assert t.gross_pnl == pytest.approx(5 * 15 + 5 * 0.25)
    assert t.exit_reason == "stop"


def test_consecutive_loss_lockout():
    cfg = BacktestConfig(initial_equity=10_000, risk_per_trade=0.01, max_consecutive_losses=2,
                         lockout_resets_daily=False)
    flat, drop = (100, 100, 100, 100), (100, 100, 80, 90)
    ohlc = [flat, drop, flat, drop, flat, drop, flat]
    plan = {0: Signal(L, stop_loss=95), 2: Signal(L, stop_loss=95), 4: Signal(L, stop_loss=95)}
    r = run(plan, ohlc, cfg=cfg)
    assert len(r.trades) == 2  # third entry refused after two straight losses


def test_daily_loss_lockout():
    cfg = BacktestConfig(initial_equity=10_000, risk_per_trade=0.05, max_daily_loss=0.02)
    flat, drop = (100, 100, 100, 100), (100, 100, 80, 90)
    plan = {0: Signal(L, stop_loss=95), 2: Signal(L, stop_loss=95)}
    r = run(plan, [flat, drop, flat, drop, flat], cfg=cfg)
    assert len(r.trades) == 1  # -5% on the first trade blocks the rest of the day


def test_open_position_closed_at_end():
    r = run({0: Signal(L, stop_loss=90)}, [(100, 100, 100, 100), (100, 101, 99, 100), (100, 106, 99, 105)])
    assert r.trades[0].exit_reason == "end_of_data" and r.trades[0].exit_price == 105
    assert math.isclose(r.equity.iloc[-1], 10_000 + r.trades[0].net_pnl)
