# Milestone 2: futures engine

## Quick start

```python
from quant_lab.core.data import bars_from_dataframe
from quant_lab.futures.config import get_contract
from quant_lab.futures.data import to_engine_bars
from quant_lab.futures.engine import run_backtest
from quant_lab.futures.strategies import FuturesHybrid
from quant_lab.futures.validation import validate

spec = get_contract("MES")                       # paper-default fees and margins
df = to_engine_bars(raw_df, bar_minutes=5, label="open", source_tz="UTC")
bars = list(bars_from_dataframe(df, symbol="MES", tz="America/Chicago"))

result = run_backtest(FuturesHybrid(), bars, spec)   # one strategy
report = validate(bars, spec)                        # all four + WFO + Monte Carlo
print(report.tear_sheets)
print(report.robustness)
```

`python -m quant_lab.futures.validation` runs the whole suite on synthetic
bars. Synthetic data only proves the plumbing works. It says nothing about
whether a strategy has an edge.

For real intraday data you need a vendor (Databento, your broker's API, or
CSV exports). yfinance's `MES=F` only gives about 60 days of 5-minute bars of
a continuous series without contract codes, enough for a smoke test.

## How a bar is processed

1. Bar `t` closes. Carry (roll costs) is booked, then any order from bar
   `t-1` executes at this bar's open (market) or when the range trades
   through it (stop/limit). Gaps fill at the open.
2. Stops, targets and the 1.5R scale-out (then stop to breakeven + 1 tick)
   are checked against the bar's range. If a bar touches both stop and
   target, the stop wins.
3. Auto-flatten closes positions in the last 10 minutes before the RTH
   close.
4. Equity is marked to market and the daily-loss (-2%) and loss-streak (4)
   breakers are checked. Then the strategy sees the closed bar and returns
   a signal for the *next* bar.

## Friction

| Cost | Model |
|---|---|
| Fees | commission + exchange + NFA + clearing per contract per side (from the broker profile) |
| Slippage | `slippage_ticks` (min 1 tick) on market and stop fills; none on limits |
| Rolls | a held position crossing a contract roll pays a round turn of fees + slippage |
| Sizing | floor(equity x risk / (stop ticks x tick value)), capped by margin and `max_contracts_per_symbol` |

## Strategies

| Strategy | Entry | Exit |
|---|---|---|
| `TrendMomentum` | fast EMA vs slow EMA + Donchian (or Keltner) breakout, ADX >= 20 | ATR trailing stop, EMA cross back |
| `VWAPReversion` | close beyond RTH VWAP +/- 2 sd, RSI(2) < 10 / > 90, ADX <= 20 | target VWAP, ATR stop |
| `OpeningRangeBreakout` | close beyond the first 30 min range, only after a quiet overnight | stop at the opposite side, 2R target |
| `FuturesHybrid` | trend regime (ADX >= 25, ATR pct >= 50%) routes to ORB then Trend; range (ADX <= 20) routes to VWAP; lunch and outside RTH stay flat | the sub-strategy that opened the trade manages it |

## Validation

* `core.metrics.tear_sheet`: Net PnL, CAGR, Sharpe, Sortino, Calmar, Max DD %,
  win rate, profit factor, expectancy, avg R, total fees, slippage and carry.
  Ratios use daily equity, so intraday bar size doesn't distort them.
* `core.wfo.walk_forward`: rolling or anchored folds; each fold grid-searches
  in-sample and runs the winner on the following unseen window.
* `core.monte_carlo.reshuffle`: 1,000 reorderings of the trade P&Ls; passes
  if the 95th-percentile drawdown is within the limit (20% by default).
