# 1. Foundations: returns, the backtest loop, and how futures make money

This chapter covers the plumbing every later idea sits on: how returns are
measured, how the backtester walks through time without cheating, what a
futures contract actually is, and what it costs to trade one.

File paths are relative to `trade-backtest-engine/`. Files under
`src/quant_lab/core/` and `src/quant_lab/futures/` beyond the config folder
arrive with Milestone 2 (PR #2).

---

## 1.1 Returns: simple vs log

**Intuition.** A return is "how much did my money grow this period". There
are two ways to write it, and each is convenient for different jobs.

```math
r_t^{\text{simple}} = \frac{P_t}{P_{t-1}} - 1
\qquad\qquad
\ell_t = \ln\!\left(\frac{P_t}{P_{t-1}}\right)
```

They are linked by $r = e^{\ell} - 1$, and for small moves $\ell \approx r$.

**Why log returns are handy.** They *add* across time:

```math
\ell_{1\to T} = \sum_{t=1}^{T} \ell_t
\quad\Longrightarrow\quad
\frac{P_T}{P_0} = \exp\!\Big(\sum_t \ell_t\Big)
```

Simple returns *compound* instead: $P_T/P_0 = \prod_t (1 + r_t)$.

**Worked example.** A price goes 100 → 110 → 99.

| Step | Simple | Log |
|---|---|---|
| 100 → 110 | +10.00% | +9.53% |
| 110 → 99 | −10.00% | −10.54% |
| Sum | 0.00% (wrong!) | −1.01% |
| True total | −1.00% | $e^{-0.0101}-1 = -1.00\%$ |

Adding simple returns says you broke even; you actually lost 1%. Adding log
returns and exponentiating gives the right answer.

**Where it lives.** `src/quant_lab/features/indicators.py` (`simple_returns`,
`log_returns`). The Milestone 1 vectorised backtester
`src/quant_lab/engine/backtest.py` builds equity as
`np.exp(strat_log.cumsum())`, exactly the formula above. Its trading cost
is written in log space as $\ln(1 - c)$ per unit of position change, so a 0.1%
cost becomes $\ln(0.999) \approx -0.001$.

**Pitfall.** Portfolio returns across *assets* add in simple space, not log
space: $r_p = \sum_i w_i r_i$. Log returns add across *time*, simple returns
add across *assets*.

---

## 1.2 The backtest loop and look-ahead bias

**Intuition.** A backtest is a replay. At each moment it must only use what
you *could have known* at that moment. Using anything from the future, even by
one bar, is called **look-ahead bias**, and it is the most common reason a
backtest looks brilliant and the live bot loses money.

The classic mistake: "if today's close is above the moving average, buy at
today's close." You only know the close once the bar has finished, by which
point the close price is gone. You can buy at the *next* bar's open at best.

### How the engine enforces this

`src/quant_lab/core/engine.py` (`Backtester.run`) processes each bar $t$ in
this order:

1. **Carry.** If a position is held across a contract roll, charge the roll
   cost (section 1.6).
2. **Execute the pending order** decided on bar $t-1$:
   * market orders fill at $O_t$ (bar $t$'s open);
   * stop and limit orders fill only if bar $t$'s range trades through the
     price (`_trigger_price`).
3. **Protective stops, targets, scale-out** are checked against $[L_t, H_t]$.
4. **Session flatten** near the close.
5. **Mark to market**: $E_t = \text{cash}_t + \text{unrealised P\&L at } C_t$.
6. **Strategy decides** using bar $t$, now closed. The signal becomes the
   pending order for bar $t+1$.

So a decision made on information up to $C_t$ can only ever fill at $O_{t+1}$
or later. The test `test_signal_fills_at_next_bar_open_never_same_close` in
`tests/core/test_engine.py` proves it.

The Milestone 1 vectorised engine does the same thing with one line:
`position.shift(1)`. The position you hold during bar $t$ is the one you
decided at the end of bar $t-1$:

```math
\ell^{\text{strategy}}_t = \text{pos}_{t-1}\cdot \ell_t
```

### Bar labelling, a hidden look-ahead

Most data vendors stamp a 5-minute bar with its **open** time: the bar
labelled 09:30 contains 09:30 to 09:35. If the engine treated "09:30" as the
moment the bar became known, every decision would be made five minutes early.
`src/quant_lab/futures/data/normalize.py` (`to_engine_bars`) shifts
open-labelled bars to their **close** time for this reason.

### What happens *inside* a bar

A bar only tells you four numbers (O, H, L, C), not the order they happened
in. If a long's stop *and* its target are both inside the bar, did price hit
the stop first or the target? We can't know, so the engine assumes the worse
case: **the stop wins** (`_manage_intrabar`). Similarly:

* **Gaps.** If the market opens *through* your stop, you're filled at the
  open, not at the stop. A long with a stop at 5990 that opens at 5980 loses
  the extra 10 points.
* **Entry bar.** On the bar a stop-entry fills, only the protective stop is
  checked, not the target, because we don't know if the high came before or
  after the fill.

Pessimistic assumptions make the backtest slightly worse than reality on
average. That is the right direction to be wrong in.

---

## 1.3 What a futures contract is

**Intuition.** A futures contract is an agreement to buy or sell something
later at a price fixed today. You don't pay the full value up front; you post
a deposit (margin) and the exchange moves cash in or out of your account every
day as the price moves. For a short-term trader that means: **you gain or lose
a fixed number of dollars for every tick the price moves.**

### Tick size, tick value, point value

* **Tick size** $\Delta$: the smallest price step.
* **Tick value** $v$: dollars gained or lost per tick per contract.
* **Point value** (multiplier) $M = v / \Delta$: dollars per 1.0 price move.

For the contracts in `src/quant_lab/futures/config/contracts.py`:

| Contract | Tick size $\Delta$ | Tick value $v$ | Point value $M$ |
|---|---|---|---|
| MES (Micro E-mini S&P 500) | 0.25 | $1.25 | $5 |
| MNQ (Micro E-mini Nasdaq-100) | 0.25 | $0.50 | $2 |
| MGC (Micro Gold) | 0.10 | $1.00 | $10 |
| 6E (Euro FX) | 0.00005 | $6.25 | $125,000 |

### P&L

```math
\text{P\&L} = (P_{\text{exit}} - P_{\text{entry}})\times s \times q \times M
```

where $s = +1$ for long, $-1$ for short, and $q$ is the number of contracts.
This is `FuturesContractSpec.pnl`.

**Worked example (MES).** Long 2 MES at 6001.25, exit at 6011.25:

```math
(6011.25 - 6001.25)\times(+1)\times 2 \times 5 = \$100
```

Equivalently: 10 points = 40 ticks, and $40 \times \$1.25 \times 2 = \$100$.

### Notional and leverage

**Notional** is the value of what you control: $N = P \times M$. One MES at
6000 controls $6000 \times 5 = \$30{,}000$ of S&P 500 exposure.

**Margin** is the deposit. The paper profile (`futures/config/fees.py`) uses
$50 intraday and $2,924.50 overnight for MES. So intraday, $50 controls
$30,000: a leverage of 600:1. A 1% move in the S&P (60 points) is $300 per
contract, six times the day margin.

**Pitfall.** Margin is *not* your risk. Margin is what the broker demands to
let you open the trade; your risk is set by where your stop is and how many
contracts you hold (chapter 2). Low day margins make it *possible* to
over-leverage, not sensible.

### Contract months and codes

Futures expire. Equity index futures trade quarterly: March (H), June (M),
September (U), December (Z). `MESZ6` is the December 2026 MES. The month
letters live in `MONTH_CODES`, and `contract_code(2026, 12)` builds `MESZ6`.

---

## 1.4 Friction: fees and slippage

**Intuition.** Every trade pays the broker and the exchange, and usually gets
a slightly worse price than the one on the screen. Strategies that trade a lot
for small gains are killed by this, so it must be in the backtest.

### Fees

Per contract **per side** (entry is one side, exit is another), the paper
profile charges:

| Item | $ / side |
|---|---|
| Broker commission | 0.39 |
| CME exchange fee | 0.35 |
| NFA regulatory fee | 0.02 |
| Clearing | 0.19 |
| **Total** $f$ | **0.95** |

A round turn (in and out) of $q$ contracts costs $2fq$. For 10 MES that's
$19.00. In ticks: $1.90 / $1.25 ≈ 1.5 ticks per contract round turn.

### Slippage

`FuturesExecutionModel.fill_price` in `src/quant_lab/futures/engine.py`:

```math
P_{\text{fill}} =
\begin{cases}
\text{round}(P_{\text{ref}}) & \text{limit orders} \\
\text{round}(P_{\text{ref}} + s \cdot \max(k, 1)\cdot\Delta) & \text{market and stop orders}
\end{cases}
```

where $k$ is `slippage_ticks` (default 1). A market buy pays one tick more
than the open; a market sell receives one tick less. Limit orders get their
price or nothing, so no slippage (but in reality they don't always fill even
when touched, which this model ignores; see pitfalls).

### The full cost of a round trip

For a market-in, market-out MES trade, the hurdle per contract is:

```math
\underbrace{2 \times 1 \text{ tick}}_{\text{slippage}} \times \$1.25
+ \underbrace{2 \times \$0.95}_{\text{fees}} = \$4.40 \approx 3.5 \text{ ticks}
```

A strategy needs to make more than 3.5 ticks (0.875 points) per trade on
average just to break even. That's why an indicator crossover that "works"
with zero costs on 1-minute bars usually dies once costs are added.

**Pitfalls.**

* **Limit fills.** Price touching your limit doesn't guarantee a fill, because
  others were in the queue ahead of you. The engine fills limits on a touch,
  which is optimistic. A stricter model requires price to trade *through* the
  limit by one tick.
* **Slippage isn't constant.** It's larger at the open, around news, and for
  bigger orders. One tick is a reasonable floor for liquid micros, not a
  ceiling.

---

## 1.5 Sessions

CME Globex runs Sunday 17:00 to Friday 16:00 Chicago time, with a daily
16:00 to 17:00 halt. **RTH** (regular trading hours) for equity index futures
is 08:30 to 15:00 CT, the hours of the underlying stock market
(`futures/config/sessions.py`). That's 14:30 to 21:00 UK time (it shifts for
a week or two in spring and autumn when the US and UK change clocks on
different dates, which is why everything is stored in `America/Chicago` and
converted, never hard-coded in UK time).

The engine only allows entries in RTH by default and **auto-flattens** 10
minutes before the close. Holding overnight needs the much higher overnight
margin ($2,924.50 vs $50) and exposes you to overnight gaps.

---

## 1.6 Continuous contracts, back-adjustment and rolls

**The problem.** You want a long price history, but each contract lives only
about three months. Gluing the contracts end to end leaves a **jump** at each
roll, because the next contract trades at a different price (for equity
indices it's usually higher, reflecting interest minus dividends until
expiry). A strategy would see that jump as a real move, and a backtest would
book fake P&L.

**The fix: back-adjustment.** `src/quant_lab/futures/data/rollover.py`
(`build_continuous`) rolls a fixed number of business days before expiry
(`roll_days_before_expiry`, default 8) and adjusts all *older* prices.

Let $g_j$ be the gap at roll $j$, measured at the same moment on both
contracts:

```math
g_j^{\text{diff}} = P^{\text{new}}_j - P^{\text{old}}_j
\qquad\qquad
g_j^{\text{ratio}} = \frac{P^{\text{new}}_j}{P^{\text{old}}_j}
```

Each piece $i$ is shifted by every roll that happens *after* it:

```math
\tilde P^{(i)} = P^{(i)} + \sum_{j \ge i} g_j^{\text{diff}}
\qquad\text{or}\qquad
\tilde P^{(i)} = P^{(i)} \times \prod_{j \ge i} g_j^{\text{ratio}}
```

**Worked example.** On the roll date MESZ5 closes at 6000.00 and MESH6 is at
6040.00, so $g = +40$. Every MESZ5 bar gets +40 added. A Z5 move from 5950 to
5960 becomes 5990 to 6000: still +10 points, so **point P&L is preserved
exactly**, which is what matters when P&L is ticks × tick value. That's why
`difference` is the default.

With `ratio`, $g = 6040/6000 = 1.00667$; the Z5 move becomes 5989.67 to
5999.73, which is +10.07 points but the same **percentage**.

**Rule of thumb.** Difference adjustment for P&L in ticks (futures). Ratio
adjustment when a signal uses percentage returns over long histories.

**Pitfalls.**

* Difference-adjusted prices far back in history can be very different from
  what actually traded (even negative for some commodities). Never use
  adjusted prices to compute a *percentage* or a notional value.
* Back-adjustment changes every old price each time you add a new contract.
  Results from last month's file and this month's won't match exactly. That
  is expected.

### The cost of rolling

If you hold through a roll you must sell the old month and buy the new one,
so `FuturesExecutionModel.carry` charges a round turn of fees plus two ticks
of slippage per contract whenever the bar's `symbol` (contract code) changes
while a position is open:

```math
\text{roll cost} = q\,(2f + 2k v) = q\,(2\times0.95 + 2\times1.25) = \$4.40\,q
```

For a day trader who's flat every night this never triggers.

---

## Self-check

1. A strategy's daily log returns over a week are +1%, −0.5%, +0.2%, −0.3%,
   +0.4%. What is the total simple return?
2. Your signal fires on the 10:05 bar's close. What's the earliest price you
   can realistically get, and what does the engine use?
3. A bar has O=6000, H=6012, L=5988, C=6005. You're long from 6000 with stop
   5990 and target 6010. What does the engine record, and why?
4. How many dollars is a 3.5-point move on 4 MNQ contracts?
5. Why does difference back-adjustment preserve P&L in ticks but distort
   percentage returns?
6. A strategy averages 0.6 points per MES trade before costs. Is it viable
   with the paper cost model?

<details><summary>Answers</summary>

1. Sum of logs = 0.8%, so simple = $e^{0.008} - 1 \approx 0.803\%$.
2. The 10:10 bar's open (the next bar). The engine fills the market order at
   that open plus one tick of slippage.
3. A stop-out at 5990 (minus one tick slippage, so 5989.75). Both levels were
   inside the bar, so the engine assumes the stop came first.
4. $3.5 \times \$2 \times 4 = \$28$.
5. Adding a constant to every price leaves *differences* unchanged but
   changes the base, so ratios (percentages) shift.
6. No. 0.6 points = 2.4 ticks, below the ~3.5-tick round-trip cost.

</details>
