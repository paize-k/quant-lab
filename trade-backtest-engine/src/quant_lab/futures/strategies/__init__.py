"""Futures strategies: trend momentum, RTH VWAP reversion, ORB, hybrid meta-strategy."""

from quant_lab.futures.strategies.hybrid import FuturesHybrid
from quant_lab.futures.strategies.opening_range_breakout import OpeningRangeBreakout
from quant_lab.futures.strategies.trend_momentum import TrendMomentum
from quant_lab.futures.strategies.vwap_reversion import VWAPReversion

ALL_STRATEGIES = (TrendMomentum, VWAPReversion, OpeningRangeBreakout, FuturesHybrid)

__all__ = ["TrendMomentum", "VWAPReversion", "OpeningRangeBreakout", "FuturesHybrid", "ALL_STRATEGIES"]
