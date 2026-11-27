from datetime import datetime, time, timezone

import numpy as np
import pandas as pd
import pytest

from quant_lab.core import TimeWindow, bars_from_dataframe, floor_to_step, normalize_ohlcv, size_by_risk


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def test_simple_window():
    w = TimeWindow("x", time(13), time(17), "UTC")
    assert w.contains(utc(2026, 1, 5, 13, 0))
    assert not w.contains(utc(2026, 1, 5, 17, 0))  # end is exclusive
    assert w.minutes_to_close(utc(2026, 1, 5, 16, 50)) == pytest.approx(10)


def test_wrapping_window_weekdays():
    # Opens Sun-Thu 17:00, closes next day 16:00 (Globex-style)
    w = TimeWindow("g", time(17), time(16), "UTC", frozenset({6, 0, 1, 2, 3}))
    assert w.contains(utc(2026, 1, 4, 18))     # Sunday evening
    assert w.contains(utc(2026, 1, 9, 10))     # Friday morning (opened Thursday)
    assert not w.contains(utc(2026, 1, 9, 18))  # Friday evening
    assert not w.contains(utc(2026, 1, 5, 16, 30))  # maintenance gap
    assert w.close_for(utc(2026, 1, 4, 18)) == utc(2026, 1, 5, 16)


def test_window_rejects_naive():
    with pytest.raises(ValueError):
        TimeWindow("x", time(1), time(2)).contains(datetime(2026, 1, 1, 1, 30))


def test_sizing_floors():
    assert floor_to_step(2.9999999999, 1) == 3
    assert floor_to_step(2.7, 1) == 2
    assert floor_to_step(12_345, 1_000) == 12_000
    # $100k * 1% / (10 ticks * $1.25) = 80 contracts
    assert size_by_risk(100_000, 0.01, 10, 1.25) == 80
    assert size_by_risk(-5, 0.01, 10, 1.25) == 0
    with pytest.raises(ValueError):
        size_by_risk(1_000, 0.01, 0, 1.25)


def test_normalize_yfinance_multiindex():
    idx = pd.date_range("2026-01-01", periods=3, freq="D")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["SPY"]],
                                      names=["Price", "Ticker"])
    data = np.array([[1, 2, 0.5, 1.5, 100]] * 3, dtype=float)
    df = pd.DataFrame(data, index=idx, columns=cols)
    out = normalize_ohlcv(df, tz="America/Chicago")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert str(out.index.tz) == "America/Chicago"
    bars = list(bars_from_dataframe(df, symbol="SPY"))
    assert len(bars) == 3 and bars[0].symbol == "SPY" and bars[0].ts.tzinfo is not None
