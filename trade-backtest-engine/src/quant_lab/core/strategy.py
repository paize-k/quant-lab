"""The unified strategy contract.

Every strategy in ``futures/`` and ``forex/`` subclasses ``BaseStrategy`` and
implements ``on_bar(bar, state) -> Signal``. The backtester, the walk-forward
optimiser and the live bots all drive the *same* class through this one
method, so research results carry over to live trading unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, ClassVar, Deque, Dict, Mapping, Optional

from quant_lab.core.types import Bar, Position, Signal


@dataclass
class StrategyState:
    """Everything a strategy may look at when deciding on bar ``t``.

    ``history`` holds closed bars up to and including the current one, so a
    strategy physically cannot see bar ``t+1``. ``scratch`` is free-form
    per-strategy memory (e.g. an anchored VWAP accumulator) that the live bot
    can persist between restarts.
    """

    equity: float
    position: Position = field(default_factory=Position)
    history: Deque[Bar] = field(default_factory=deque)
    scratch: Dict[str, Any] = field(default_factory=dict)
    bar_index: int = -1

    def push(self, bar: Bar, max_history: Optional[int] = None) -> None:
        """Append a closed bar. Called by the engine, never by a strategy."""
        if self.history and bar.ts <= self.history[-1].ts:
            raise ValueError(
                f"bars must arrive in strictly increasing time order: "
                f"{bar.ts} after {self.history[-1].ts}"
            )
        if max_history is not None and self.history.maxlen != max_history:
            self.history = deque(self.history, maxlen=max_history)
        self.history.append(bar)
        self.bar_index += 1


class BaseStrategy(ABC):
    """Base class for every backtestable *and* live-tradable strategy.

    Subclasses set ``name``, declare tunable parameters in
    ``default_params`` (WFO searches over these), and implement ``on_bar``.

    Rules a subclass must follow:

    * Decide only from ``bar`` and ``state``; never fetch data or peek ahead.
    * Return a ``Signal`` on every call (``Signal.hold()`` to do nothing).
    * Keep any rolling state in ``state.scratch`` or on ``self`` and clear it
      in ``reset()`` so WFO folds start clean.
    """

    name: ClassVar[str] = "base"
    default_params: ClassVar[Mapping[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise TypeError(f"{type(self).__name__} got unknown params: {sorted(unknown)}")
        self.params: Dict[str, Any] = {**self.default_params, **params}
        self.validate_params()

    @property
    def warmup_bars(self) -> int:
        """Closed bars needed before the first non-HOLD signal. Override."""
        return 0

    @property
    def max_history(self) -> Optional[int]:
        """How many closed bars the engine should keep in ``state.history``.

        ``None`` keeps everything; strategies should override with a bound
        so live bots don't grow memory without limit.
        """
        return max(self.warmup_bars, 1) if self.warmup_bars else None

    def validate_params(self) -> None:
        """Raise ``ValueError`` for an invalid parameter set. Override."""

    def reset(self) -> None:
        """Clear internal state between backtest runs / WFO folds. Override."""

    @abstractmethod
    def on_bar(self, bar: Bar, state: StrategyState) -> Signal:
        """Decide what to do after ``bar`` has closed."""

    def with_params(self, **params: Any) -> "BaseStrategy":
        """A fresh instance with some params changed (used by WFO)."""
        return type(self)(**{**self.params, **params})

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"
