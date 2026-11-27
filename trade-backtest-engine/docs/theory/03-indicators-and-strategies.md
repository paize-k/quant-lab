# 3. Indicators and the four futures strategies

Every indicator here lives in `src/quant_lab/core/indicators.py` (streaming,
one closed bar at a time, so the same code runs in backtests and live bots).
The Milestone 1 vectorised versions are in `src/quant_lab/features/indicators.py`,
and `tests/core/test_indicators.py` checks they agree.

**Notation:** $O_t, H_t, L_t, C_t, V_t$ are bar $t$'s open, high, low, close
and volume.

---

## 3.1 Moving averages: SMA and EMA

**SMA** (simple moving average): the plain average of the last $N$ closes.

```math
\text{SMA}_t = \frac{1}{N}\sum_{i=0}^{N-1} C_{t-i}
```

**EMA** (exponential moving average): a weighted average where recent closes
count more, and weights decay geometrically.

```math
\text{EMA}_t = \text{EMA}_{t-1} + \alpha\,(C_t - \text{EMA}_{t-1}),
\qquad \alpha = \frac{2}{N+1}
```

Unrolled, the weight on the close $i$ bars ago is $\alpha(1-\alpha)^i$. For
$N=20$, $\alpha = 0.0952$: today's close gets 9.5% of the weight, the close
10 bars ago gets $0.0952 \times 0.9048^{10} \approx 3.5\%$.

**Why EMA reacts faster.** Its average "age" of information is
$(1-\alpha)/\alpha = (N-1)/2$ bars, the same as an SMA of the same $N$, but
more weight sits on the newest bars, so it turns sooner after a reversal.

**Worked example.** EMA(3), so $\alpha = 0.5$. Closes 100, 102, 101, 105,
seeded with the first close:
100 → $100 + 0.5(102-100) = 101$ → $101 + 0.5(101 - 101) = 101$ →
$101 + 0.5(105 - 101) = 103$.

**Pitfall: warm-up.** An EMA seeded at the first close takes a few multiples
of $N$ bars to "forget" that seed. `EMA.ready` is only true after $N$ bars,
and strategies wait longer still (`warmup_bars`).

---

## 3.2 Wilder smoothing

J. Welles Wilder's ATR, RSI and ADX use their own smoothing (class `Wilder`):

```math
W_t = W_{t-1} + \frac{1}{N}(x_t - W_{t-1})
```

seeded with the simple average of the first $N$ values. It's an EMA with
$\alpha = 1/N$ instead of $2/(N+1)$, so it's slower: Wilder(14) behaves like
an EMA(27), because $1/14 = 2/(27+1)$.

---

## 3.3 ATR: Average True Range

**Intuition.** "How far does this market typically move in one bar?" It's
the main volatility ruler for stops and regimes.

**True range** includes gaps from the previous close:

```math
\text{TR}_t = \max\big(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|\big)
\qquad
\text{ATR}_t = \text{Wilder}_N(\text{TR})_t
```

**Example.** Previous close 6000; bar $H=6004$, $L=5998$. TR $= \max(6,
4, 2) = 6$. If instead the bar gapped to $H=6015$, $L=6010$:
TR $= \max(5, 15, 10) = 15$. The gap counts.

**Use.** Stops at $k \times$ ATR scale with volatility, so the strategy
doesn't get stopped out by normal noise in a wild market or leave stops
miles away in a quiet one. Combined with chapter 2's sizing, a wider ATR stop
automatically means fewer contracts.

---

## 3.4 Donchian channels

The highest high and lowest low of the **previous** $N$ bars:

```math
U_t = \max_{1 \le i \le N} H_{t-i},\qquad D_t = \min_{1 \le i \le N} L_{t-i}
```

The current bar is excluded on purpose. If it were included, $C_t > U_t$ could
never happen (the close can't exceed a max that includes its own high), and a
version using $H_t \ge U_t$ would be trivially true whenever today makes the
high. Excluding it makes "close above the channel" a real breakout.
(Milestone 1's `donchain_breakout.py` does the same with `shift(1)`.)

---

## 3.5 Keltner channels

```math
\text{mid}_t = \text{EMA}_N(C)_t,\qquad
\text{upper/lower}_t = \text{mid}_t \pm m\cdot\text{ATR}_t
```

Like Bollinger bands but with ATR instead of standard deviation, so the band
width follows range rather than close-to-close dispersion. `TrendMomentum`
can use it instead of Donchian (`use_keltner=True`).

---

## 3.6 ADX: is there a trend?

**Intuition.** ADX measures trend *strength*, not direction. High ADX means
price is moving persistently one way; low ADX means chop.

Steps (class `ADX`):

1. Directional movement: $u = H_t - H_{t-1}$, $d = L_{t-1} - L_t$.

```math
+\text{DM} = \begin{cases} u & u > d,\ u > 0 \\ 0 & \text{else}\end{cases}
\qquad
-\text{DM} = \begin{cases} d & d > u,\ d > 0 \\ 0 & \text{else}\end{cases}
```

2. Smooth TR, +DM and −DM with Wilder($N$), then

```math
+\text{DI} = 100\,\frac{\widetilde{+\text{DM}}}{\widetilde{\text{TR}}},\qquad
-\text{DI} = 100\,\frac{\widetilde{-\text{DM}}}{\widetilde{\text{TR}}}
```

3. The directional index and its smoothed version:

```math
\text{DX} = 100\,\frac{|{+\text{DI}} - {-\text{DI}}|}{{+\text{DI}} + {-\text{DI}}},
\qquad
\text{ADX} = \text{Wilder}_N(\text{DX})
```

DX is near 0 when up and down movement balance, near 100 when one side
dominates. Because ADX is smoothed twice, it needs about $2N$ bars to warm
up (hence `warmup_bars = 2 * adx_period + 1`).

**Conventional reading.** Below 20: no trend. Above 25: trending. The
strategies use exactly these thresholds.

---

## 3.7 RSI and RSI(2)

```math
\text{RS} = \frac{\text{Wilder}_N(\text{gains})}{\text{Wilder}_N(\text{losses})},
\qquad
\text{RSI} = 100 - \frac{100}{1 + \text{RS}}
```

RSI runs from 0 (all losses) to 100 (all gains). With $N = 2$ it's
hypersensitive: two down closes in a row push it near 0. That's the point.
**RSI(2)** is a short-term *exhaustion* gauge, used to fade stretched moves,
not the classic 14-period overbought/oversold reading.

**Example.** RSI(2) seeded on changes −3, −2: average gain 0, average loss
2.5, so RS $= 0$ and RSI $= 0$. Next change +1: gain $= 0 + (1 - 0)/2 = 0.5$,
loss $= 2.5 + (0 - 2.5)/2 = 1.25$; RS $= 0.4$; RSI $= 100 - 100/1.4 = 28.6$.

---

## 3.8 Anchored VWAP and its bands

**Intuition.** VWAP is the average price paid today, weighted by volume.
Institutions benchmark execution against it, so price tends to be pulled
back toward it in quiet, range-bound sessions.

`SessionVWAP` in `futures/strategies/_session.py`, anchored at each RTH open
(08:30 CT), uses the typical price $\text{tp} = (H + L + C)/3$:

```math
\text{VWAP}_t = \frac{\sum_{i \le t} \text{tp}_i V_i}{\sum_{i \le t} V_i}
```

The bands use a volume-weighted standard deviation, via the identity
$\text{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$:

```math
\sigma_t = \sqrt{\frac{\sum \text{tp}_i^2 V_i}{\sum V_i} - \text{VWAP}_t^2},
\qquad
\text{bands} = \text{VWAP}_t \pm k\,\sigma_t
```

Keeping three running sums ($\sum \text{tp}V$, $\sum \text{tp}^2V$, $\sum V$)
means each update is $O(1)$, important for live bots.

**Pitfalls.**

* Early in the session $\sigma$ is tiny and the bands hug VWAP, so every
  wiggle looks "stretched". That's why `min_session_bars = 6` waits 30
  minutes on 5-minute bars.
* The $E[X^2] - E[X]^2$ formula can go very slightly negative from floating
  point error; the code clamps with `max(..., 0)`.
* With no volume data, every bar gets weight 1 and VWAP becomes a plain
  cumulative average of typical prices.

---

## 3.9 ATR percentile rank

`PercentileRank` reports what fraction of the last $W$ ATR values were below
today's:

```math
\text{pct}_t = \frac{\#\{\,j \in \text{window} : \text{ATR}_j < \text{ATR}_t\,\}}{W}
```

0.5 means today's volatility is median for the last $W$ bars. It's
*relative* volatility, which travels between quiet and wild years better than
a fixed ATR threshold.

---

## 3.10 The strategies

All live in `src/quant_lab/futures/strategies/`.

### Trend Momentum (`trend_momentum.py`)

* **Long entry:** $\text{EMA}_{20} > \text{EMA}_{50}$ (trend up), $C_t > U_t$
  (Donchian breakout) and $\text{ADX} \ge 20$ (trend is strong).
* **Stop:** $C_t - 2\,\text{ATR}$, then *trails*: each bar it moves to
  $C_t - 2\,\text{ATR}$ if that's higher, never lower.
* **Exit:** EMA cross back ($\text{EMA}_{20} < \text{EMA}_{50}$).
* Shorts are the mirror image.

The logic: momentum persists for a while once it starts (trend following
works because big moves are rare but large). Expect a low win rate and a few
big winners. The trailing stop is what lets winners run.

### VWAP Reversion (`vwap_reversion.py`)

* **Long entry:** $C_t < \text{VWAP} - 2\sigma$, RSI(2) $< 10$, ADX $\le 20$.
* **Target:** VWAP at the time of entry. **Stop:** $1.5\,\text{ATR}$ below.
* Shorts mirror.

The logic: in a non-trending session, stretched moves tend to snap back. The
ADX filter is essential, because in a trend "stretched" stays stretched and
mean reversion gets run over. Expect a high win rate and small wins.

### Opening Range Breakout (`opening_range_breakout.py`)

* Record the high and low of the first 30 minutes of RTH (the opening range).
* Trade only if the **overnight range was compressed**:
  $\text{overnight range} \le 0.6 \times \text{previous RTH range}$.
* **Long:** close above the OR high; stop at the OR low (or midpoint); target
  $2R$. At most one trade per day, only in the first 120 minutes.

The logic: a quiet overnight session builds pressure, and the first decisive
move out of the opening range often carries.

### The Hybrid: regime classification (`hybrid.py`)

A router that decides which sub-strategy may open trades:

| Regime | Condition | Allowed entries |
|---|---|---|
| trend | ADX $\ge 25$ and ATR percentile $\ge 0.5$ | ORB first, then Trend Momentum |
| range | ADX $\le 20$ | VWAP Reversion |
| neutral | between 20 and 25 | none |
| closed | lunch (11:30 to 13:00 CT) or outside RTH | none |

All three sub-strategies see every bar so their indicators stay warm, and the
one that opened a trade keeps managing it even if the regime changes.

**Why a neutral band?** It's hysteresis: without a gap between thresholds,
ADX wobbling around 22 would flip the regime back and forth every bar.

**Pitfall.** Regime filters add parameters (25, 20, 0.5, lunch times). Each
one is another dial that can be tuned to the past. Chapter 4 is about this.

---

## Self-check

1. What is $\alpha$ for EMA(50)? What weight does the close 50 bars ago get?
2. Why does `Donchian` exclude the current bar?
3. Previous close 100, bar H=103, L=97. What's TR? What if the bar was H=108,
   L=105?
4. ADX is 35 and −DI > +DI. Is the market trending? Which way?
5. Why does VWAP Reversion need ADX ≤ 20 and Trend Momentum need ADX ≥ 20?
6. Wilder(10) is roughly equivalent to which EMA period?

<details><summary>Answers</summary>

1. $2/51 = 0.0392$. Weight $0.0392 \times 0.9608^{50} \approx 0.0053$ (0.5%).
2. So that a close beyond it is a genuine breakout of *prior* bars. Including
   the current bar makes the test impossible or trivially true.
3. $\max(6, 3, 3) = 6$. Gapped: $\max(3, 8, 5) = 8$.
4. Yes, strongly (ADX > 25), and down, because −DI dominates.
5. Mean reversion fails in trends, momentum fails in chop. The filters send
   each to the environment it's built for.
6. $1/10 = 2/(N+1)$ gives $N = 19$.

</details>
