import numpy as np
import pandas as pd
import pytest

from quant_lab.core.indicators import ADX, ATR, EMA, RSI, SMA, Donchian, Keltner, PercentileRank, RollingStd
from quant_lab.features.indicators import ema as v_ema, sma as v_sma

RNG = np.random.default_rng(0)
CLOSE = pd.Series(100 + RNG.normal(0, 1, 300).cumsum())


def test_sma_ema_match_vectorised_versions():
    s, e = SMA(20), EMA(20)
    ss = [s.update(x) for x in CLOSE]
    ee = [e.update(x) for x in CLOSE]
    np.testing.assert_allclose(pd.Series(ss, dtype=float).values, v_sma(CLOSE, 20).values, equal_nan=True)
    np.testing.assert_allclose(ee, v_ema(CLOSE, 20).values)


def test_rolling_std_matches_pandas():
    r = RollingStd(10)
    out = pd.Series([r.update(x) for x in CLOSE], dtype=float)
    np.testing.assert_allclose(out.values, CLOSE.rolling(10).std().values, equal_nan=True)


def test_donchian_excludes_current_bar():
    d = Donchian(3)
    for h in (1, 2, 3):
        d.update(h, h - 1)
    up, lo = d.update(10, 0)
    assert (up, lo) == (3, 0)  # previous three bars only


def test_atr_constant_range():
    a = ATR(5)
    for _ in range(10):
        v = a.update(11, 9, 10)
    assert v == pytest.approx(2.0)


def test_rsi_bounds_and_extremes():
    r = RSI(2)
    vals = [r.update(x) for x in range(1, 20)]
    assert vals[-1] == 100.0
    r = RSI(14)
    vals = [r.update(x) for x in CLOSE]
    assert all(0 <= v <= 100 for v in vals if v is not None)


def test_adx_strong_trend_is_high():
    a = ADX(14)
    for i in range(100):
        v = a.update(100 + i + 1, 100 + i - 1, 100 + i)
    assert v > 90 and a.plus_di > a.minus_di


def test_keltner_and_percentile():
    k = Keltner(5, 5, 2.0)
    for _ in range(10):
        mid, up, lo = k.update(11, 9, 10)
    assert mid == pytest.approx(10) and up == pytest.approx(14) and lo == pytest.approx(6)
    p = PercentileRank(4)
    out = [p.update(x) for x in (1, 2, 3, 4, 0, 10)]
    assert out[:3] == [None, None, None]
    assert out[-1] == 1.0 and out[-2] == 0.0
