# MI Agent commercial quality recovery — strict end-to-end semantic fidelity

Baseline `0af2d9f` (frozen 7/7). Work: `277b4ad`, `2fed76f`.

Graded by a strict oracle (`cfo_oracle.py`) against truth computed independently
from the fixture tapes (`truth.py`, which imports no MI code). Both runs
re-graded with the **same** oracle, so the comparison is one ruler.

---

## 1. Product quality: before → after

| Category | Before | After | Change |
|---|---|---|---|
| EXACT | 56 | **65** | **+9** |
| DISCLOSED ASSUMPTION | 2 | 2 | 0 |
| TRUE SAFE REFUSAL | 12 | 13 | +1 |
| FALSE REFUSAL | 13 | 11 | −2 |
| **WRONG / SILENT SUBSTITUTION** | **8** | **0** | **−8** |

Ten questions moved. **Every one improved; none regressed.**

| family | EXACT | WRONG | FALSE REFUSAL |
|---|---|---|---|
| size | 6 → 8 | 2 → 0 | 0 → 0 |
| composition | 10 → 10 | 0 → 0 | 0 → 0 |
| trends | 9 → 10 | 1 → 0 | 0 → 0 |
| comparisons | 3 → 5 | 2 → 0 | 1 → 1 |
| filters | 11 → 11 | 0 → 0 | 1 → 1 |
| ranking | 8 → 10 | 0 → 0 | 4 → 2 |
| concentration | 3 → 3 | 0 → 0 | 4 → 4 |
| pipeline | 2 → 3 | 1 → 0 | 3 → 3 |
| specialist | 4 → 5 | 1 → 0 | 0 → 0 |
| insufficiency | 0 → 0 | 1 → 0 | 0 → 0 |

### Two corrections to my own oracle, recorded

The first oracle could not see a KPI payload or more than three sample rows, so
it scored every single-figure answer "value not present" and every five-point
series a length mismatch against its own truncation — 27 false WRONGs. It also
took `ok=True` at face value, so `geo_exposure`'s "I can't build a geographic
exposure view" counted as a delivered answer. **An oracle blind to the answer is
not an oracle**; both were fixed before the baseline was frozen.

## 2. The five P0 classes, each closed at the first divergence

| | root cause (first divergence) | fix | proof |
|---|---|---|---|
| **A** average balance | **executor** — `resolve_metric_key` read the aggregation for `count` and ignored it otherwise, so average balance fell through to `funded_balance` | `avg_balance` as a per-period metric, derived from the two figures either side of it | series now £216,944 → £268,837, matching truth exactly; the total series is byte-identical to before |
| **B** requested period | **plan** — read only a NAMED start period, so "for last month" arrived `window_periods=1` and opened at the earliest snapshot | the plan declares the window; the executor opens that many periods back | last month May→June **+£22.6m**; last 3 months March→June +£50.5m; unqualified unchanged |
| **C** forecast milestone | **renderer + projector** — `_ms` fell back to `milestones[-1]`, answering from the nearest threshold the ladder carried | the arithmetic decides `already_reached`; the requested target is passed to the projector | £100m/£172m reached; £200m→2026-08, £250m→2026-12, £500m→2028-05. Monotonic |
| **D** units | **renderer** — the summary rendered `wa_interest_rate`, declared a FRACTION in `_METRIC_DISPLAY`, with the points formatter | rendering reads the declaration | **6.26%** (true 6.2644%), was 0.06% |
| **E** forecast horizon | **contract** — no governed concept for a forward horizon, so five years, twelve months and five years all returned the same open-pipeline composition | a governed `forecast_horizon_months` reader; the composition declines a horizon it cannot reach | honest refusal naming that no forward projection was run and nothing was substituted |

Plus the pipeline mislabel: the receipt names its dataset, so £3.6m over 8 cases
reads "entire pipeline" rather than "entire funded portfolio".

## 3. Coverage recovered

`geo_exposure` returned `ok=True` with a can't-do message and, having claimed
the question, blocked everything behind it. It now **defers** — the estate's own
pre-claim pattern — so "which region has the largest/smallest balance?" answers
from the governed obligor-region breakdown that always existed.

## 4. Regression and architecture

MI-only manifest, 278 modules, before vs after:

```
modules 278 → 278     passed 5957 → 5957     failed 81 → 81
skipped 711 → 711     xfailed 15 → 15        errors 4 → 4      timeouts 1 → 1
failing names 85 → 85     INTRODUCED 0     FIXED/REMOVED 0
```

OCC, onboarding, Annex 2, regulatory XML and mail are outside this denominator.
Post-claim semantic census **0**; substitution detector **0 of 2**; frozen canary
intact; 68 guard tests pass, including the new leading-filter suite.

## 5. The blocker: a silent filter drop the bank did not sample

```
"How many lump_sum loans do we have?"   →  640   (the whole book)
truth                                   →  396
disclosure                              →  none
```

Its siblings fail differently: `"How many drawdown loans do we have?"` is
misread as a pipeline question and refuses; `"What is the balance for lump sum
loans?"` refuses citing **the wrong field** (`geographic_region_obligor`).
Geography categoricals work (`London` → 83 loans); product-type categoricals do
not.

This is the same defect class as the five just closed — a user constraint
silently lost — and it sits in **filters**, a core recurring family. It is
reported rather than fixed because categorical value resolution is not a bounded
local patch, and the operating instruction is explicit that a clean NOT READY
with one specific blocker beats a green verdict.

## 6. Remaining false refusals (11) — coverage, not safety

concentration 4 · pipeline 3 · ranking 2 (filtered ranked movement) ·
comparisons 1 · filters 1. All safe; none substitutes a figure.

**Filtered ranked movement** remains the known live wiring gap, and the
distinction stays explicit: the contract carries the predicate and the
plan-level proof executes it, but the shipped `period_change` executor does not
apply it, so the live product refuses.

## 7. UX defects, separate from semantic correctness

1. Single-figure answers put the number in a KPI while the prose says only
   *"Here is the result for your query, covering 1 group(s)."* A CFO reading the
   sentence sees no figure.
2. Refusals leak internal identifiers — *"the population
   current_loan_to_value gt 50.0"*.
3. *"How does the current month compare with the previous month?"* refuses
   citing **"a comparison between two books"**, which is not what was asked.
