from datetime import datetime, timedelta, timezone

import pytest

from quant_lab.core import Bar, BaseStrategy, Signal, SignalAction, StrategyState


class CloseAboveOpen(BaseStrategy):
    name = "close_above_open"
    default_params = {"lookback": 3}

    @property
    def warmup_bars(self):
        return self.params["lookback"]

    def validate_params(self):
        if self.params["lookback"] < 1:
            raise ValueError("lookback")

    def on_bar(self, bar, state):
        if len(state.history) < self.warmup_bars:
            return Signal.hold()
        if all(b.close > b.open for b in list(state.history)[-self.warmup_bars:]):
            return Signal(SignalAction.ENTER_LONG, stop_loss=bar.low - 1)
        return Signal.hold()


def _bars(n):
    t0 = datetime(2026, 1, 5, tzinfo=timezone.utc)
    return [Bar(ts=t0 + timedelta(minutes=i), open=10 + i, high=12 + i, low=9 + i, close=11 + i)
            for i in range(n)]


def test_params_and_with_params():
    s = CloseAboveOpen()
    assert s.params == {"lookback": 3}
    assert s.with_params(lookback=5).params["lookback"] == 5
    with pytest.raises(TypeError):
        CloseAboveOpen(nope=1)
    with pytest.raises(ValueError):
        CloseAboveOpen(lookback=0)


def test_base_strategy_is_abstract():
    with pytest.raises(TypeError):
        BaseStrategy()


def test_on_bar_sees_only_closed_bars():
    s = CloseAboveOpen()
    state = StrategyState(equity=10_000)
    actions = []
    for bar in _bars(5):
        state.push(bar, s.max_history)
        assert state.history[-1] is bar  # current bar is the newest visible
        actions.append(s.on_bar(bar, state).action)
    assert actions[:2] == [SignalAction.HOLD] * 2
    assert actions[2] is SignalAction.ENTER_LONG
    assert len(state.history) == 3  # bounded by max_history
    assert state.bar_index == 4


def test_state_rejects_out_of_order_bars():
    b = _bars(2)
    state = StrategyState(equity=1)
    state.push(b[1])
    with pytest.raises(ValueError):
        state.push(b[0])
