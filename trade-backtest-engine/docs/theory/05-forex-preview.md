# 5. Preview: forex theory for Milestone 3

The forex engine reuses everything in chapters 1 to 4 (the same backtester,
sizing formula, metrics, WFO and Monte Carlo). What changes is how prices,
costs and P&L are measured. The pair specs are already in
`src/quant_lab/forex/config/pairs.py` and the cost profile in
`src/quant_lab/forex/config/costs.py`.

---

## 5.1 Quotes, base and quote currency

`EUR_USD = 1.08000` means **1 euro (base) costs 1.08 US dollars (quote)**.
Buying EUR/USD means buying euros and selling dollars. Your P&L comes out in
the **quote** currency:

```math
\text{P\&L}_{\text{quote}} = (P_{\text{exit}} - P_{\text{entry}})\times s \times \text{units}
```

where *units* are units of the base currency. Lots are conventions:
standard = 100,000 units, mini = 10,000, micro = 1,000. OANDA lets you trade
any whole number of units (`lot_step = 1`).

---

## 5.2 Pips and pipettes

A **pip** is the conventional unit of price change: 0.0001 for most pairs,
0.01 for JPY-quoted pairs. Brokers quote one more decimal, the **pipette**
(fractional pip): 1.08000 is 5 decimals, so `price_precision = 5`, and the
last digit is a tenth of a pip.

| Pair | Pip size | Quoted decimals |
|---|---|---|
| EUR/USD, GBP/USD, AUD/USD, USD/CHF, USD/CAD | 0.0001 | 5 |
| USD/JPY, EUR/JPY | 0.01 | 3 |

---

## 5.3 Pip value and currency conversion

**Pip value** is how much one pip is worth to *you*, in your account
currency. One pip on 1 unit is `pip_size` in the quote currency; convert it:

```math
\text{pip value per unit} = \text{pip size} \times \underbrace{X_{\text{quote}\to\text{acct}}}_{\texttt{quote\_to\_account\_rate}}
```

`ForexPairSpec.quote_to_account_rate` handles three cases:

1. **Quote = account currency:** $X = 1$.
2. **Base = account currency:** $X = 1 / P$ (divide by the pair's own price).
3. **Neither:** look up a cross rate, e.g. `USD_GBP` or `GBP_USD`.

### Worked examples for a GBP account (you're in the UK)

Illustrative rates: EUR/USD 1.0800, GBP/USD 1.2700, USD/JPY 150.00,
GBP/JPY 190.50.

| Position | Pip value in quote ccy | Conversion | Pip value in GBP |
|---|---|---|---|
| 100k EUR/USD | 0.0001 × 100,000 = $10 | case 3: ÷ 1.27 | **£7.87** |
| 100k GBP/USD | $10 | case 2: ÷ 1.27 | **£7.87** |
| 100k USD/JPY | 0.01 × 100,000 = ¥1,000 | case 3: ÷ 190.50 | **£5.25** |

For a USD account the same USD/JPY position is ¥1,000 ÷ 150 = $6.67 per pip
(case 2).

**Pitfall.** Pip value moves with exchange rates. If GBP/USD rises, every
USD-quoted pip is worth fewer pounds. P&L is converted at the exit price,
which is what `ForexPairSpec.pnl` does.

The current `PAPER_DEFAULT` assumes a **USD** account. A GBP account will need
its own profile (swap and commission figures in GBP) and a GBP conversion
rate for every pair. That's on the Milestone 3 list.

---

## 5.4 Position sizing in forex

Same formula as futures (`core/sizing.py`), with pips instead of ticks:

```math
\text{units} = \left\lfloor \frac{E\cdot\rho}{d_{\text{pips}}\times\text{pip value per unit}} \right\rfloor
```

**Example.** £10,000 account, 0.5% risk = £50. EUR/USD, 20-pip stop. Pip
value per unit = 0.0001 / 1.27 = £0.00007874.

```math
\text{units} = \left\lfloor \frac{50}{20 \times 0.00007874} \right\rfloor = \lfloor 31{,}750.1 \rfloor = 31{,}750
```

That's `ForexPairSpec.units_for_risk`.

---

## 5.5 Bid, ask and spread

There are two prices at any moment: the **bid** (what you can sell at) and
the **ask** (what you can buy at). The **spread** is ask − bid. The **mid**
is their average, and most charts show mid or bid.

* You buy at the ask and later sell at the bid, so a round trip at an
  unchanged mid loses exactly one spread.
* Each side therefore costs half a spread relative to mid.

```math
\text{spread cost} = \text{spread (pips)} \times \text{pip value} \times \text{units}
```

**Example.** The paper profile's EUR/USD spread is 1.6 pips. On 100k units
that's $16 (≈ £12.60) per round trip. Most FX accounts charge no separate
commission; the spread *is* the commission.

**Rollover widening.** Around the daily rollover (about 22:00 UTC, 5pm New
York), liquidity thins and spreads widen. The profile assumes 3× the normal
spread from 21:00 to 23:00 UTC (`forex/config/sessions.py`,
`is_rollover_window`).

**Pitfall.** If your backtest data is mid prices and you fill at mid, you've
removed the entire cost of trading. The forex execution model must fill buys
at mid + half spread and sells at mid − half spread.

---

## 5.6 Swaps (overnight financing)

**Intuition.** Holding a currency pair overnight means borrowing one currency
to hold the other. You *earn* interest on the currency you hold and *pay* on
the one you borrowed. The net is the **swap** (or rollover, or financing),
booked at about 22:00 UTC each day.

Roughly, per night, in the quote currency:

```math
\text{swap} \approx \text{units}\times P \times \frac{i_{\text{base}} - i_{\text{quote}} - \text{broker markup}}{365}
```

for a long (flip the sign of the rate difference for a short).

**Example (illustrative rates).** Long 100k EUR/USD at 1.08, EUR rate 2%,
USD rate 4%, ignoring markup:

```math
100{,}000 \times 1.08 \times \frac{0.02 - 0.04}{365} = -\$5.92 \text{ per night}
```

The short side would *earn* roughly that, minus the broker's markup (the
markup means both sides often pay).

**Triple Wednesday.** Spot FX settles two business days after the trade
(T+2). A position held through Wednesday's rollover settles on Friday → Monday,
so it's charged for Saturday and Sunday too: three nights. That's
`ForexPairSpec.swap_nights`.

In the code, swaps are given per 100k units per night in the account currency
(`swap_long_per_100k`, `swap_short_per_100k`), and `swap_amount` multiplies by
size and nights. The figures in `PAPER_DEFAULT` are from February 2026 and
stale; OANDA's API serves live financing rates, which Milestone 3 or 4 should
use.

**Why it matters.** A swing strategy holding for weeks can have its edge
eaten (or boosted) by carry. Day strategies flat by 22:00 UTC never pay it.

---

## 5.7 Margin and leverage in the UK

Margin is a percentage of notional. The current paper profile uses **2%**
(50:1), the **US** retail limit for majors. **UK retail** clients are capped
by the FCA at **30:1 on major pairs (3.33% margin)** and 20:1 on minors. In
the UK, retail FX at brokers like OANDA is usually offered as a CFD or spread
bet. Pips, spreads and financing work the same way, but check the exact terms
for your account.

**Example.** The 31,750-unit EUR/USD position above has notional
€31,750 × 1.08 = $34,290 ≈ £27,000. At 3.33% the margin is about £900, 9% of
a £10,000 account, while the *risk* is still only £50, because risk is set by
the stop (chapter 2).

---

## Self-check

1. You buy 50,000 GBP/USD at 1.27000 and sell at 1.27350. How many pips, and
   what's the P&L in USD and in GBP?
2. What is the pip value of 20,000 USD/JPY in GBP, with GBP/JPY at 190.50?
3. Why do day traders usually not care about swaps?
4. Why is the Wednesday swap three times the normal one?
5. A backtest on mid prices shows +0.8 pips per trade on EUR/USD. Viable with
   a 1.6-pip spread?

<details><summary>Answers</summary>

1. 35 pips. $(1.27350 - 1.27000) \times 50{,}000 = \$175$. In GBP at the
   exit: $175 / 1.2735 ≈ £137.42$.
2. $0.01 \times 20{,}000 = ¥200$; $200 / 190.50 ≈ £1.05$ per pip.
3. Swaps are charged only on positions held through the daily rollover.
4. T+2 settlement: a trade rolled on Wednesday settles across the weekend,
   so it's financed for Saturday and Sunday as well.
5. No. The spread alone is 1.6 pips per round trip, double the edge.

</details>
