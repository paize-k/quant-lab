from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant_lab.core.data import bars_from_dataframe
from quant_lab.futures.config import get_contract
from quant_lab.futures.data import build_continuous, expiry_date, roll_date, to_engine_bars

MES = get_contract("MES")


def test_expiry_rules():
    assert expiry_date(MES, 2026, 12) == date(2026, 12, 18)               # third Friday
    assert expiry_date(get_contract("6E"), 2026, 12) == date(2026, 12, 14)  # 2 bd before 3rd Wed
    assert expiry_date(get_contract("MGC"), 2026, 12) == date(2026, 12, 29)
    assert roll_date(MES, 2026, 12) == date(2026, 12, 8)                  # 8 business days earlier


def _contract(start, end, level):
    idx = pd.date_range(start, end, freq="B", tz="America/Chicago") + pd.Timedelta(hours=15)
    px = level + np.arange(len(idx), dtype=float)
    return pd.DataFrame({"open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 1.0}, index=idx)


def test_back_adjusted_difference_removes_gap():
    h6 = _contract("2026-01-01", "2026-03-20", 5000.0)
    m6 = _contract("2026-02-15", "2026-06-19", 5100.0)   # trades 100+ points above
    cont = build_continuous(MES, {(2026, 3): h6, (2026, 6): m6})
    assert list(cont["contract"].unique()) == ["MESH6", "MESM6"]
    assert cont["roll"].sum() == 1
    roll_at = cont.index[cont["roll"]][0]
    assert roll_at.date() >= roll_date(MES, 2026, 3)
    # Newest contract is unadjusted.
    assert (cont.loc[cont["contract"] == "MESM6", "close"] == m6.loc[cont.index[cont["contract"] == "MESM6"], "close"]).all()
    # Bar-to-bar change across the roll equals the new contract's own change.
    i = cont.index.get_loc(roll_at)
    jump = cont["close"].iloc[i] - cont["close"].iloc[i - 1]
    new_prev = m6.loc[cont.index[i - 1], "close"]
    assert jump == pytest.approx(cont["close"].iloc[i] - new_prev)


def test_ratio_and_none_methods():
    h6 = _contract("2026-01-01", "2026-03-20", 5000.0)
    m6 = _contract("2026-02-15", "2026-06-19", 5100.0)
    raw = build_continuous(MES, {(2026, 3): h6, (2026, 6): m6}, method="none")
    assert raw["close"].iloc[0] == 5000.0
    ratio = build_continuous(MES, {(2026, 3): h6, (2026, 6): m6}, method="ratio")
    assert ratio["close"].iloc[0] > 5000.0
    with pytest.raises(ValueError):
        build_continuous(MES, {(2026, 3): h6}, method="bad")


def test_contract_column_becomes_bar_symbol():
    h6 = _contract("2026-01-01", "2026-03-20", 5000.0)
    m6 = _contract("2026-02-15", "2026-06-19", 5100.0)
    cont = build_continuous(MES, {(2026, 3): h6, (2026, 6): m6})
    bars = list(bars_from_dataframe(cont, tz="America/Chicago"))
    assert bars[0].symbol == "MESH6" and bars[-1].symbol == "MESM6"


def test_to_engine_bars_shifts_open_labels_and_filters_rth():
    idx = pd.date_range("2026-03-02 13:00", periods=12 * 24, freq="5min", tz="UTC")  # 07:00 CT on
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    from quant_lab.futures.config import get_sessions
    out = to_engine_bars(df, bar_minutes=5, label="open", sessions=get_sessions("cme_equity"), rth_only=True)
    assert str(out.index.tz) == "America/Chicago"
    assert out.index[0].strftime("%H:%M") == "08:35"   # first RTH bar closes 08:35
    assert out.index[-1].strftime("%H:%M") == "15:00"
