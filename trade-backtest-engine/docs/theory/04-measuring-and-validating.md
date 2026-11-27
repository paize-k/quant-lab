# 4. Measuring a strategy, and checking it isn't luck

A backtest produces an equity curve and a list of trades. This chapter
explains the numbers on the tear sheet, then the two tools that ask "would
this survive the future?": walk-forward optimisation and Monte Carlo
reshuffling. It ends with the enemy of all of it: overfitting.

---

## 4.1 The tear sheet

Code: `src/quant_lab/core/metrics.py` (`tear_sheet`, `compare`).

**Daily equity first.** Return-based ratios use the *last equity mark of each
day*, not every bar. A Sharpe ratio computed from 5-minute returns and one
from daily returns aren't comparable, and annualising 5-minute returns with
"252 per year" (as Milestone 1's `metrics/performance.py` does for any bar
size) gives nonsense. Daily returns are

```math
r_d = \frac{E_d}{E_{d-1}} - 1
```

### Return metrics

**Total return** $= E_T / E_0 - 1$.

**CAGR** (compound annual growth rate): the constant yearly rate that would
turn $E_0$ into $E_T$ over $Y$ years.

```math
\text{CAGR} = \left(\frac{E_T}{E_0}\right)^{1/Y} - 1,
\qquad Y = \frac{\text{number of daily returns}}{252}
```

*Example:* $100k → $121k over 2 years: $1.21^{1/2} - 1 = 10\%$ a year, not
10.5%.

### Risk-adjusted metrics

**Sharpe ratio**: average return per unit of volatility, annualised.

```math
\text{Sharpe} = \frac{\bar r_d}{s_d}\sqrt{252}
```

(The risk-free rate is taken as 0. For futures that's reasonable: the margin
cash can earn interest separately, so futures P&L is already an excess
return.) The $\sqrt{252}$ comes from returns adding over time while
volatility grows with the *square root* of time: mean scales by 252, standard
deviation by $\sqrt{252}$, so their ratio scales by $\sqrt{252}$.

*Example:* mean daily return 0.05%, standard deviation 0.8%:
$0.0005 / 0.008 \times 15.87 = 0.99$.

**Sortino ratio**: like Sharpe but only penalises *downside* volatility,
because nobody minds upside surprises.

```math
\text{Sortino} = \frac{\bar r_d}{\sigma_-}\sqrt{252},
\qquad
\sigma_- = \sqrt{\frac{1}{n}\sum_{d} \min(r_d, 0)^2}
```

Note the divisor is *all* $n$ days, not only the losing ones. That's the
standard definition, and it's what the code does.

**Maximum drawdown**: the worst peak-to-trough fall.

```math
\text{DD}_t = \frac{\max_{s \le t} E_s - E_t}{\max_{s \le t} E_s},
\qquad
\text{MaxDD} = \max_t \text{DD}_t
```

*Example:* equity 100k → 110k → 99k → 115k. Peak 110k, trough 99k:
MaxDD = 11/110 = 10%. (MaxDD in the tear sheet uses every bar's mark, so it
includes intraday dips.)

**Calmar ratio** $= \text{CAGR} / \text{MaxDD}$. "How many years of growth
does the worst drawdown cost?" 12% CAGR with 8% MaxDD gives Calmar 1.5.

### Trade metrics

From the list of net trade P&Ls $x_1, \dots, x_n$:

```math
\text{Win rate} = \frac{\#\{x_i > 0\}}{n}
\qquad
\text{Profit factor} = \frac{\sum_{x_i > 0} x_i}{\left|\sum_{x_i < 0} x_i\right|}
\qquad
\text{Expectancy} = \frac{1}{n}\sum_i x_i
```

*Example:* ten trades: +500, −200, +300, −200, −200, +800, −200, +100, −200,
+400.

* Win rate: 5/10 = 50%
* Gross wins 2,100, gross losses 1,000, so **profit factor 2.1**
* Expectancy: 1,100 / 10 = **$110 per trade**

**Avg R** is the mean R-multiple (chapter 2). **Exposure %** is the share of
bars with a position open. **Total fees / slippage / carry** show how much
friction cost; if fees + slippage are a large share of gross profit, the
strategy is fragile to cost assumptions.

### How much should you trust a Sharpe ratio?

A Sharpe ratio estimated from data has error. A standard approximation
(Lo, 2002) for the standard error of an annualised Sharpe from $T$ daily
returns, when daily Sharpe is small, is

```math
\text{SE}(\widehat{\text{SR}}) \approx \sqrt{\frac{252}{T}}
```

| History | SE of annual Sharpe |
|---|---|
| 1 year (252 days) | ≈ 1.0 |
| 3 years | ≈ 0.58 |
| 5 years | ≈ 0.45 |

So **a Sharpe of 1.0 from one year of data is statistically
indistinguishable from zero.** This single fact explains most disappointing
live results.

> **Review note on PR #2 (spotted while writing this).** `daily_equity`
> groups bars by *UTC* calendar date. Globex reopens Sunday 17:00 CT, which is
> 23:00 UTC on Sunday, so Sunday evening becomes a sixth "day" each week. On
> the synthetic demo, 20 trading days produce 24 daily points. With
> annualisation at 252 per year, $Y$ is overstated by about 6/5, which
> understates CAGR and slightly distorts Sharpe and Sortino. Grouping by the
> Chicago trading day (the 17:00 CT boundary) would fix it. RTH-only data
> isn't affected.

---

## 4.2 Walk-forward optimisation (WFO)

Code: `src/quant_lab/core/wfo.py` (`walk_forward`); default grids in
`src/quant_lab/futures/validation.py` (`DEFAULT_SPACES`).

**Intuition.** If you choose parameters by looking at all the data, then
test on the same data, you've marked your own homework. WFO simulates what
you'd actually do in real time: optimise on the past, trade the next chunk
you haven't seen, then move forward and repeat.

### How the folds are cut

With $n$ bars, $K$ folds and train fraction $f$ (default $K = 4$, $f = 0.7$):

```math
\text{seg} = \lfloor n / K \rfloor,
\quad
\text{train}_k = [k\cdot\text{seg},\ k\cdot\text{seg} + \lfloor f\cdot\text{seg}\rfloor),
\quad
\text{test}_k = [k\cdot\text{seg} + \lfloor f\cdot\text{seg}\rfloor,\ (k+1)\cdot\text{seg})
```

*Example:* 20,000 bars, 4 folds: segments of 5,000. Fold 0 trains on bars
0 to 3,499 and tests on 3,500 to 4,999; fold 1 trains on 5,000 to 8,499 and
tests on 8,500 to 9,999; and so on.

```
fold 0: [TTTTTTT|ooo]
fold 1:             [TTTTTTT|ooo]
fold 2:                         [TTTTTTT|ooo]
fold 3:                                     [TTTTTTT|ooo]
```

With `anchored=True`, every training window starts at bar 0 and grows.

In each fold, every parameter combination in the grid is backtested on the
training window and scored by in-sample Sharpe (combinations with fewer than
`min_trades` trades score $-\infty$). The winner is run once on the test
window, starting from the previous fold's ending equity. Stitching the test
windows gives the **out-of-sample (OOS) equity curve**, the most honest
number the engine produces.

### Reading the result

* **Walk-forward efficiency** $\approx \text{OOS performance} / \text{IS
  performance}$. Values well below 0.5 mean most of the in-sample result was
  fitting noise.
* **Parameter stability.** If the best `fast` EMA flips between 10 and 20
  every fold, the strategy has no stable optimum; that's a warning.
* **Trade counts.** OOS windows are 30% of each segment. If an OOS window has
  five trades, its Sharpe means almost nothing (see 4.1).

### Pitfalls

* **Cold start.** Each run calls `strategy.reset()`, so every test window
  starts with empty indicators and can't trade until it's warmed up (50 bars
  for Trend Momentum, about 115 for the Hybrid's ATR percentile). On short
  test windows that's a real chunk of lost opportunity.
* **The grid itself is a choice.** If you try a grid, dislike the WFO
  result, and try another, the WFO is no longer out-of-sample. Your brain was
  the optimiser.

---

## 4.3 Monte Carlo trade reshuffling

Code: `src/quant_lab/core/monte_carlo.py` (`reshuffle`).

**Intuition.** Your backtest produced one particular *order* of trades. With
the same trades in a different order, the worst drawdown could have been much
worse. Shuffling the order 1,000 times shows the range of drawdowns you
should be prepared for.

For each run $j$, take a random permutation $\pi_j$ of the trade P&Ls and
build an equity path:

```math
E^{(j)}_k = E_0 + \sum_{i=1}^{k} x_{\pi_j(i)},
\qquad
\text{DD}^{(j)} = \max_k \frac{\max_{m \le k} E^{(j)}_m - E^{(j)}_k}{\max_{m \le k} E^{(j)}_m}
```

The report gives the 50th and **95th percentile** of $\text{DD}^{(j)}$. The
strategy "passes" if the p95 drawdown is at most 20% (`max_dd_limit`).

**Why the 95th percentile?** It means "in 19 of 20 alternative histories,
the drawdown was no worse than this". Size your expectations, and your risk
per trade, so you could live through that number.

### Permutation vs bootstrap

* **Permutation** (default): every run uses each trade exactly once. Only
  the order changes.
* **Bootstrap** (`bootstrap=True`): sample $n$ trades *with replacement*.
  Some trades appear twice, some not at all, so the total P&L varies too.

> **A mathematical fact worth understanding (and a review note on PR #2).**
> Addition is commutative, so under permutation every run ends at exactly the
> same equity: $E_0 + \sum_i x_i$. That means `final_equity_p5`,
> `final_equity_p50` and `prob_loss` carry no information in the default
> mode: they're always the real final equity and either 0 or 1. I confirmed
> this by running it. Only the drawdown statistics are meaningful under
> permutation. For a distribution of outcomes, use `bootstrap=True`, or the
> report could compute those fields from a bootstrap automatically.

### Pitfalls

* **Independence assumption.** Shuffling assumes trades are independent. Real
  losses cluster (a bad regime produces several in a row). Shuffling breaks
  those clusters, which can make drawdowns look *smaller* than reality.
  Block bootstrapping (resample runs of consecutive trades) is the fix, and a
  candidate for Milestone 5.
* **Closed-trade equity only.** The Monte Carlo paths step from trade to
  trade, so they miss drawdown *during* trades. The tear sheet's MaxDD is
  bar by bar and can be larger.
* **Dollar P&L, no compounding.** Trades are added in dollars at their
  original size. Reasonable for short histories at 0.5% risk; less so over
  years of growth.

---

## 4.4 Overfitting

**Definition.** Overfitting is when a strategy has learned the *noise* in
your historical data rather than a repeatable *signal*. It looks great
in-sample and fails out-of-sample.

### Why it's so easy: multiple testing

Suppose you test $M$ parameter combinations, and **none** of them has any
edge. Their measured Sharpe ratios are just noise, roughly
$\mathcal{N}(0, \text{SE}^2)$. The best of them is *expected* to look like

| $M$ tried | Expected best Sharpe, in units of SE |
|---|---|
| 1 | 0 |
| 8 | 1.42 |
| 100 | 2.51 |
| 1,000 | 3.24 |

These are the expected maxima of $M$ independent standard normals (roughly
$\sqrt{2 \ln M}$ for large $M$). With one year of data (SE ≈ 1), trying 8
combinations of a strategy with *zero* edge will typically hand you a winner
with Sharpe ≈ 1.4. The grids in `DEFAULT_SPACES` are small on purpose (8, 6,
6 and 4 combinations) for exactly this reason.

### Symptoms

* In-sample far better than out-of-sample (low walk-forward efficiency).
* Sharp parameter "cliffs": EMA 20 is great, EMA 18 and 22 are terrible. A
  real effect should degrade gracefully.
* Few trades, many rules. Each extra condition (lunch filter, ADX
  threshold, compression ratio) is another degree of freedom.
* Results depend on one or two huge trades.

### Defences already in this project

1. Small grids, chosen before looking.
2. Walk-forward OOS as the headline number, not the full-sample backtest.
3. Monte Carlo p95 drawdown, not the lucky actual drawdown.
4. Pessimistic fills and real costs, so the edge has to be large enough to
   survive friction.

### Defences planned for Milestone 5

* **Deflated Sharpe ratio** (Bailey and López de Prado): adjusts a Sharpe for
  the number of trials, the length of the sample and the skew and fat tails
  of returns.
* **PBO** (probability of backtest overfitting): how often the in-sample
  winner ranks below median out-of-sample.
* A final **untouched holdout** period you look at once.

---

## Self-check

1. $50k grows to $72k over 3 years. CAGR?
2. Why is Sharpe annualised with $\sqrt{252}$ and not 252?
3. A strategy's Sharpe is 1.2 on 18 months of data. Roughly how confident
   can you be it's positive?
4. In a permutation Monte Carlo, why is the p95 *final equity* equal to the
   actual final equity?
5. You test 100 parameter sets on 1 year of data and the best has Sharpe 2.3.
   Impressed?
6. WFO folds show best `fast` = 10, 20, 10, 20 and OOS Sharpe 1.5, −0.8,
   0.9, −1.1. What would you conclude?

<details><summary>Answers</summary>

1. $(72/50)^{1/3} - 1 = 1.44^{1/3} - 1 \approx 12.9\%$.
2. Mean return grows linearly with time; standard deviation grows with the
   square root of time. The ratio grows with $\sqrt{t}$.
3. SE $\approx \sqrt{252/378} \approx 0.82$, so 1.2 is about 1.5 SE above
   zero: suggestive, not convincing (roughly 93% one-sided, ignoring the
   multiple-testing problem).
4. Every permutation contains the same trades, and their sum doesn't depend
   on order.
5. No. The expected best of 100 zero-edge trials is about 2.5 SE ≈ 2.5. A
   Sharpe of 2.3 is what pure luck looks like.
6. Unstable parameters and OOS results that alternate sign: there's no
   reliable edge. Don't trade it.

</details>
