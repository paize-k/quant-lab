from datetime import datetime, timezone

import pytest

from quant_lab.core import Bar, OrderType, Signal, SignalAction, Side

TS = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)


def test_bar_requires_tz():
    with pytest.raises(ValueError):
        Bar(ts=datetime(2026, 1, 5), open=1, high=2, low=0.5, close=1.5)


def test_bar_rejects_inconsistent_ohlc():
    with pytest.raises(ValueError):
        Bar(ts=TS, open=3, high=2, low=1, close=1.5)


def test_bar_quotes_flag():
    b = Bar(ts=TS, open=1, high=1, low=1, close=1)
    assert not b.has_quotes
    q = Bar(ts=TS, open=1, high=1, low=1, close=1, bid_open=1, bid_high=1, bid_low=1,
            bid_close=1, ask_open=1, ask_high=1, ask_low=1, ask_close=1)
    assert q.has_quotes


def test_signal_validation():
    with pytest.raises(ValueError):
        Signal(SignalAction.ENTER_LONG, OrderType.STOP)  # no entry price
    with pytest.raises(ValueError):
        Signal(SignalAction.ENTER_LONG, OrderType.LIMIT, entry_price=100, stop_loss=101)
    with pytest.raises(ValueError):
        Signal(SignalAction.ENTER_SHORT, OrderType.LIMIT, entry_price=100, stop_loss=99)
    s = Signal(SignalAction.ENTER_SHORT, stop_loss=105, risk_fraction=0.01)
    assert s.is_entry and s.side is Side.SHORT
    assert Signal.hold().side is Side.FLAT
