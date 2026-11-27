# 0. How to review a pull request

A pull request (PR) is a proposal: "here are some changes on a branch;
please check them and merge them into `main`." Reviewing is how you make sure
you understand and trust what goes into your project. You are the owner here,
so **nothing should be merged that you couldn't explain to someone else.**

You don't need to read every line with equal care. A good review has four
passes, from cheap to expensive.

---

## Pass 1: Read the description (5 minutes)

Open the PR on GitHub (the **Pull requests** tab, then the PR). Read the
description and ask:

* What does it claim to do? Write the claim down in one sentence.
* Does that match the milestone we agreed?
* What does it say it *doesn't* do, or what's assumed? (Paper fees,
  synthetic data only, and so on.)

If you can't summarise it, ask in a comment before going further.

---

## Pass 2: Run it (10 to 15 minutes)

Code that doesn't run isn't worth reading yet. On Windows, in PowerShell or
the VS Code terminal, from your clone of the repo:

```powershell
git fetch origin
git checkout <branch-name>           # shown at the top of the PR, e.g. claude/project-thread-fke3pb
cd trade-backtest-engine
python -m venv .venv                 # first time only
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q                            # every test should pass
```

Then run whatever demo the PR mentions. For Milestone 2:

```powershell
python -m quant_lab.futures.validation
```

Check:

* Do all tests pass? Note the count (PR #2 currently shows `81 passed`).
* Does the demo run, and do the numbers look *plausible*? (On synthetic data
  you're checking for sanity, not profit: no NaN columns, trade counts above
  zero, fees not absurd.)

When done, go back with `git checkout main`.

---

## Pass 3: Read the diff, in the right order (30 to 60 minutes)

On GitHub, open **Files changed**. Don't read alphabetically. Read in the
order the ideas depend on each other:

1. **Docs and config** first (`docs/`, `config/`): what are the assumptions
   and numbers?
2. **Types and interfaces** (`core/types.py`, the `ExecutionModel` protocol
   in `core/engine.py`): what are the building blocks?
3. **The core logic** (`core/engine.py`, `futures/engine.py`): how does it
   actually work?
4. **Strategies and features**.
5. **Tests last**, asking "does this test prove the thing it's named after?"

For each file, try to answer: *what is this for, and what would break if it
were wrong?* Use the theory chapters in this folder alongside.

### A checklist for quant code

These are the bugs that make backtests lie. Check each one explicitly.

| Question | Where to look |
|---|---|
| Does any decision use data from the bar it trades on, or later? | Signals vs fills, `shift(1)`, indicator windows |
| Are bars labelled by close time before they're used? | Data loading / normalisation |
| Are fees, slippage and spread charged on **every** side? | Execution model, trade records |
| Are units consistent: points vs ticks vs pips, account currency vs quote currency? | P&L and sizing functions |
| What happens at edges: first bar, last bar, gaps, missing volume, zero-range bars? | Engine loop, indicator seeding |
| Are intrabar assumptions pessimistic (stop before target)? | Stop/target logic |
| Is anything annualised with the wrong period count? | Metrics |
| Can the optimiser see the test data? | WFO train/test split |
| Does a test check the *behaviour*, or just that code runs? | `tests/` |

### Check one thing by hand

The best way to trust an engine is to predict its output yourself. Pick one
small case, do the maths on paper, then run it. For example, chapter 2's
worked MES trade (long at 6001.25, stop 5993.25, scale-out at 1.5R) predicts
10 contracts, $281 net, +0.70R. This script reproduces it:

```python
# run from trade-backtest-engine/ with the venv active
from tests.core.helpers import Script, make_bars
from quant_lab.core import Signal, SignalAction as A
from quant_lab.futures.config import get_contract
from quant_lab.futures.engine import run_backtest

bars = make_bars([(6000, 6000.5, 5999.5, 6000),   # signal decided here
                  (6001, 6003, 6000.5, 6002),     # fills at 6001 + 1 tick
                  (6005, 6014, 6004, 6012),       # reaches 1.5R, half off
                  (6010, 6011, 6001, 6003)],      # breakeven stop hit
                 symbol="MES")
plan = {0: Signal(A.ENTER_LONG, stop_loss=5993.25)}
r = run_backtest(Script(plan=plan), bars, get_contract("MES"), 100_000, use_sessions=False)
t = r.trades[0]
print(t.qty, t.entry_price, t.net_pnl, t.r_multiple)   # 10.0 6001.25 281.0 0.7025
```

If your prediction and the engine disagree, one of you is wrong, and
finding out which is the most valuable learning you can do.

---

## Pass 4: Decide and respond

On **Files changed**, hover over a line and click the blue **+** to leave a
comment on it. Good comments are questions or specific concerns:

* "Why is the stop checked before the target here?" (a question is fine; it's
  your codebase and you should understand it)
* "This annualises with 252 but the data has Sunday sessions, so is Y
  overstated?"
* "Nit: typo in the docstring." (mark small things as nits)

Then click **Review changes** (top right) and choose:

* **Comment**: questions, no verdict yet.
* **Request changes**: something must be fixed before merging.
* **Approve**: you understand it, it runs, and you're happy.

After approving, **Merge pull request** at the bottom of the Conversation tab
puts it into `main`. Merging is easy to undo with a revert, but it's cleaner
to merge only when you're confident.

You can also just ask me in the project chat: "explain `core/engine.py`
lines 200 to 240" or "why is X done this way?". Every question you ask at
review time is cheaper than a bug found with real money.

---

## What to expect from each milestone PR

Each milestone PR will come with a matching theory chapter (or an update to
this folder) covering the ideas it introduces, the formulas, where they live
in the code and self-check questions, so you can review the maths and the
code side by side.
