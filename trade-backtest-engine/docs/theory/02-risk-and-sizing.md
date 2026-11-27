# 2. Risk, position sizing, R-multiples and circuit breakers

**The one idea of this chapter:** you don't choose how many contracts to trade.
You choose how much money you're willing to lose if the trade is wrong, and
the stop distance turns that into a number of contracts.

---

## 2.1 Fixed-fractional position sizing

**Intuition.** Risk the same small *fraction* of your account on every trade.
A wide stop means fewer contracts; a tight stop means more. Each trade then
costs about the same if it fails, whatever the market's volatility, and as the
account grows or shrinks the bet size follows it.

**The formula** (`src/quant_lab/core/sizing.py`, `size_by_risk`):

```math
q = \left\lfloor \frac{E \cdot \rho}{d \cdot v} \right\rfloor
```

* $E$: equity (the engine uses cash at the moment of entry)
* $\rho$: risk fraction, default 0.5% (`FuturesRiskLimits.risk_per_trade`)
* $d$: stop distance in ticks, rounded **up** (`math.ceil` in
  `FuturesExecutionModel.size`), so the risk estimate is never too small
* $v$: tick value

The floor $\lfloor\cdot\rfloor$ rounds *down*: you can never trade a
fraction of a contract, and rounding up would break the risk budget.

Then two caps are applied (`FuturesContractSpec.contracts_for_risk` and
`FuturesExecutionModel.size`):

```math
q_{\text{final}} = \min\!\Big(q,\ \Big\lfloor \tfrac{E}{\text{margin per contract}} \Big\rfloor,\ q_{\max}\Big)
```

with $q_{\max} = 10$ (`max_contracts_per_symbol`).

### Worked example (checked against the engine)

$E = \$100{,}000$, $\rho = 0.5\%$, so the risk budget is **$500**.

The engine fills a long at 6001.25 with a stop at 5993.25:

* Stop distance: $6001.25 - 5993.25 = 8$ points $= 32$ ticks.
* Risk per contract: $32 \times \$1.25 = \$40$.
* Raw size: $\lfloor 500 / 40 \rfloor = \lfloor 12.5 \rfloor = 12$.
* Margin cap: $\lfloor 100{,}000 / 50 \rfloor = 2{,}000$ (not binding).
* Max-contracts cap: **10** (binding).

So $q = 10$, and the money at risk is $10 \times \$40 = \$400$. This $400 is
the trade's **1R** (next section).

**Pitfall: friction isn't in the risk budget.** If that trade is stopped out,
the real loss is

```math
\underbrace{\$400}_{\text{stop distance}}
+ \underbrace{10 \times 1 \times \$1.25}_{\text{exit slippage}}
+ \underbrace{10 \times 2 \times \$0.95}_{\text{fees}}
= \$431.50
```

That's 0.43% here only because the 10-contract cap bit. Without the cap,
12 contracts would lose $517.80, slightly *more* than the 0.5% budget.
With tight stops (few ticks) friction becomes a large share of the loss, so
the realised risk can be well above the target. Keep this in mind when you
look at a strategy's average losing trade.

**Why fixed-fractional?** Two reasons.

1. *Survival.* Losing $n$ trades in a row at fraction $\rho$ leaves
   $(1-\rho)^n$ of the account. At 0.5% risk, 20 losses in a row cost
   $1 - 0.995^{20} = 9.5\%$. At 5% risk the same streak costs 64%.
2. *Recovery is asymmetric.* After a drawdown $D$, you need a gain of
   $D / (1 - D)$ to get back. A 10% drawdown needs +11%; a 50% drawdown needs
   +100%. Small risk keeps you in the region where recovery is easy.

---

## 2.2 R-multiples

**Intuition.** Measure each trade's result in units of what you risked. A
trade that risked $400 and made $800 is **+2R**. A full stop-out is about
**−1R** (a bit worse with costs). This makes trades comparable across
contracts, stop widths and account sizes.

```math
R\text{-multiple} = \frac{\text{net P\&L}}{\text{initial risk}},
\qquad
\text{initial risk} = q \cdot |P_{\text{entry}} - P_{\text{stop}}| \cdot M
```

Code: `Trade.r_multiple` and `Trade.initial_risk` in `core/engine.py`.

### Expectancy in R

If your win rate is $p$, average win is $\bar W$ R and average loss is
$\bar L$ R:

```math
\mathbb{E}[R] = p\,\bar W - (1-p)\,\bar L
```

The **break-even win rate** for a fixed target of $T$ R and a 1R loss is

```math
p^{*} = \frac{1}{1 + T}
```

| Target $T$ | Break-even win rate |
|---|---|
| 1R | 50% |
| 1.5R | 40% |
| 2R (the ORB default) | 33.3% |
| 3R | 25% |

Costs push all of these up a little.

**Pitfall.** High win rate ≠ good strategy. A 90% win rate with +0.1R wins
and −2R losses has expectancy $0.9(0.1) - 0.1(2) = -0.11$R per trade.

---

## 2.3 The 1.5R scale-out and breakeven stop

**The rule** (`ScaleOutRule` in `core/engine.py`; settings in
`futures/config/risk.py`): when price reaches $+1.5R$, close **half** the
position with a limit order and move the stop on the rest to **entry + 1
tick**.

```math
P_{\text{scale}} = P_{\text{entry}} + s \cdot 1.5 \cdot |P_{\text{entry}} - P_{\text{stop}}|
\qquad
P_{\text{new stop}} = P_{\text{entry}} + s\cdot\Delta
```

**Intuition.** You lock in some profit, and the remainder becomes a "free"
trade that can no longer lose (much). The cost is that you give up half the
profit on the trades that would have run to a big target.

### Worked example (same trade as above, verified in the engine)

Entry 6001.25, stop 5993.25, so 1R = 8 points and $1.5R = 12$ points.

1. A bar's high reaches 6014, through the scale level $6001.25 + 12 =
   6013.25$. Sell 5 at 6013.25 (limit, no slippage):
   $12 \times 5 \times \$5 = +\$300$.
2. The stop moves to $6001.25 + 0.25 = 6001.50$.
3. Price falls back to 6001. The stop fills at $6001.50 - 0.25$ (slippage)
   $= 6001.25$, exactly the entry, so $0 on the remaining 5.

Fees: $0.95 × (10 + 5 + 5) = $19. Net = $300 − $19 = **$281**, or
$281/400 = **+0.70R**.

The "+1 tick" was meant to guarantee a tiny profit on the remainder, but one
tick of stop slippage exactly cancels it. That's realistic: breakeven stops
often fill at or slightly below breakeven.

**The trade-off in numbers.** Ignoring costs, suppose a trade either goes to
−1R, or reaches 1.5R and then either reverses to breakeven or runs to 3R.

* Without scale-out: a trade reaching 1.5R that later runs to 3R gives 3R;
  one that reverses to the stop gives −1R.
* With scale-out: those become $0.5 \times 1.5 + 0.5 \times 3 = 2.25$R and
  $0.5 \times 1.5 + 0 = 0.75$R.

Scale-out cuts the best outcome and lifts the worst outcome of trades that
went your way. It lowers variance (smoother equity) and may lower or raise
expectancy depending on how often winners run. The backtest tells you which,
for your strategy.

---

## 2.4 Circuit breakers

Settings in `futures/config/risk.py`, enforced in `Backtester.run`.

### Daily loss limit (−2%)

If mark-to-market equity falls to

```math
E_t \le E_{\text{day start}}\,(1 - 0.02)
```

no new entries for the rest of the trading day (Chicago time). At 0.5% risk
per trade that's roughly four full losses, or fewer if slippage was bad.

### Loss-streak limit (4)

Four losing trades in a row locks out new entries until the next day.

**How often should a streak happen by chance?** With win rate $p$, the chance
that a given run of 4 trades are all losses is $(1-p)^4$:

| Win rate | $P(\text{4 losses in a given 4 trades})$ |
|---|---|
| 60% | 2.6% |
| 45% | 9.2% |
| 35% (typical for 2R targets) | 17.9% |

Over hundreds of trades, streaks of 4 are certain to happen even with a
genuine edge. The breaker isn't saying "the strategy is broken"; it caps the
damage on days when conditions don't suit it (or something is wrong with the
data feed or the bot).

**Pitfall.** Breakers change the trade sequence. A strategy optimised *with*
breakers may behave differently without them, and vice versa. Always backtest
with the same breakers you'll run live.

---

## Self-check

1. $E = \$50{,}000$, $\rho = 0.5\%$, MNQ (tick $0.50), stop 30 points away.
   How many contracts?
2. In Q1, what is 1R in dollars?
3. A trade risked $300 and made $450 net. What R-multiple?
4. Your ORB strategy wins 38% of trades with a 2R target. Roughly what
   expectancy per trade, before costs?
5. Why does the sizing round the stop distance *up* to whole ticks but the
   contract count *down*?
6. After a 25% drawdown, what return do you need to recover?

<details><summary>Answers</summary>

1. 30 points = 120 ticks, $120 × 0.50 = \$60$ per contract.
   $\lfloor 250/60 \rfloor = 4$ contracts.
2. $4 \times \$60 = \$240$.
3. $450/300 = +1.5$R.
4. $0.38 \times 2 - 0.62 \times 1 = +0.14$R per trade.
5. Both are the conservative direction: a slightly bigger stop estimate and a
   slightly smaller position both make the real loss ≤ the budget.
6. $0.25/0.75 = 33.3\%$.

</details>
