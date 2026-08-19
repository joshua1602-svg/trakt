# ESCALATION — Silent aggregation substitution produces a plausible incorrect successful answer

**Raised during:** Commercial Beta Readiness Review (review/measure only)
**Status:** OPEN — escalated, not fixed. No production code has been changed.
**Baseline:** `983a755` (clean tree, P1L suite 8,675 passed / 0 failed)
**Severity:** BETA BLOCKER (candidate) — see §6 for the reasoning and the counter-argument.

The Commercial Beta Readiness brief requires that a safety defect capable of producing a
plausible incorrect successful answer be escalated separately rather than absorbed into a
breadth exercise. This is that escalation. The readiness review is paused pending direction.

---

## 1. The defect in one sentence

When a question asks for an aggregation that the target field's registry entry does not
permit, the MI Agent **silently substitutes the field's default aggregation, returns
`ok=True`, and asserts the substituted figure as the answer** — with no warning, no facet,
and no disclosed limitation.

## 2. The headline case

> **"What is the median LTV?"**

| | |
|---|---|
| Delivered | **43.1562** (`ok=True`, single KPI, no warnings) |
| Requested statistic (true median LTV) | **39.6757** |
| Error | **+3.48 LTV points / +8.77%** |
| Reproducibility | **5/5 genuine-LLM runs, 5/5 deterministic runs** — identical every time |
| Parser provenance | `llm_repaired`, 2 real model calls per run |

The only on-screen trace of the substitution is the calculation line, which reads
`Calculated: Weighted-average Current LTV`. Nothing states that the median the user asked
for was not computed. 43.2 against a true 39.7 is entirely plausible for this book; no
reader could detect the substitution from the number.

## 3. Evidence matrix — median across the four principal measures

Truth computed independently in pandas from the fixture; the production implementation was
not used to validate itself.

### Deterministic parser

| Case | ok | Delivered | Truth | Error | Calculation label | Verdict |
|---|---|---|---|---|---|---|
| median LTV | True | 43.1562 | 39.6757 | 8.77% | Weighted-average Current LTV | **SUBSTITUTED** |
| median interest rate | True | 6.5597 | 6.5750 | 0.23% | Weighted-average Interest Rate | **SUBSTITUTED** |
| median borrower age | True | 71.3976 | 71.0000 | 0.56% | Average Borrower Age | **SUBSTITUTED** |
| median loan balance | True | 1,964,886,258.21 | 156,864.66 | **1,252,499%** | Total Balance | **SUBSTITUTED** |

4 of 4 substituted, all `ok=True`.

### Genuine LLM parser (production configuration)

| Case | ok | Delivered | Truth | Error | Calculation label | Verdict |
|---|---|---|---|---|---|---|
| median LTV | True | 43.1562 | 39.6757 | **8.77%** | Weighted-average Current LTV | **SUBSTITUTED** |
| median interest rate | True | 6.5682 | 6.5750 | 0.10% | Average Interest Rate | **SUBSTITUTED** |
| median borrower age | True | 71.0000 | 71.0000 | 0.00% | Median Borrower Age | MATCH |
| median loan balance | True | 156,864.66 | 156,864.66 | 0.00% | Median Balance | MATCH |

2 of 4 substituted.

## 4. Root cause

The two measures that substitute on the LLM path are exactly the two whose registry
entries exclude `median` from `allowed_aggregations`. The correlation is perfect:

| Field | `allowed_aggregations` | `default_aggregation` | median request |
|---|---|---|---|
| `current_loan_to_value` | avg, weighted_avg, distribution | weighted_avg | **substituted** |
| `current_interest_rate` | avg, weighted_avg, distribution | weighted_avg | **substituted** |
| `youngest_borrower_age` | avg, **median**, distribution | avg | honoured (LLM path) |
| `current_outstanding_balance` | sum, avg, **median** | sum | honoured (LLM path) |

So: **a disallowed aggregation is coerced to the field default instead of being refused.**
The executor itself supports `median` correctly — `mi_query_executor.py:293` — and returns
the exact true value when it is actually asked for. The failure is entirely in the
permission-check path, which downgrades rather than declines.

The deterministic parser additionally never emits `median` at all, so it substitutes even
on fields that permit it (rows 3 and 4 of the deterministic table).

## 5. Why the P0 facet ledger did not catch it

The ledger governs **measures, dimensions, populations, scopes, thresholds, comparison
periods, rankings, shares and contributions**. It does not govern the **aggregation
function**. There is no facet kind for "the requested statistic".

`unresolved_measure_slots` cannot catch it by design: it is a *coordinated-list* guard that
returns `()` when no measure list is recognised, and in "what is the median LTV?" the
measure (LTV) resolves perfectly well. It is the *statistic* that is lost, and nothing is
watching that axis.

Note the architecture already proves the concept is expressible: **max/min are governed**
and refuse correctly —

> "the question asks for a single extreme value, but the calculation covered the whole book
> without ranking it. I have not returned the substituted breakdown."

That is the correct behaviour. It exists for extremes and is absent for median, percentile,
quartile, standard deviation and spread.

This is the same class of defect as P1K/P1L — a material intent disappearing without
evidence — on an axis the ledger does not yet cover.

## 6. Severity

**Argued as BETA BLOCKER:**
- It produces a wrong number asserted as the answer, with `ok=True` and no disclosure.
- It is fully reproducible (5/5 both paths), not a stochastic LLM wobble.
- It hits **LTV and interest rate** — the central credit and pricing metrics a lending
  CFO asks about, not an exotic corner.
- "Median LTV" is ordinary MI vocabulary, not an adversarial probe.
- The error is plausible in magnitude (8.77%), so it is undetectable by the reader.

**Counter-argument, stated fairly:** the calculation trace does name the statistic actually
computed ("Weighted-average Current LTV"), so a reader who reads that line can see the
substitution. This is a genuine mitigation and it is why the answer is not *silent* in the
strictest sense. It is not, in my assessment, sufficient — the figure is presented as the
result, and the whole point of the P1E/P1L work was that naming the substitution in a trace
is not the same as refusing to make it.

**Recommendation:** treat as a Beta blocker. The governing product rule established in
P1I-A applies directly: *prevent an invalid result from being created in the first place;
do not create it and disclose it later.*

## 7. Related findings from the same probe (lower severity, not escalated)

Recorded here so they are not lost, but they do **not** on their own meet the escalation bar.

| # | Finding | Path | Severity |
|---|---|---|---|
| R-1 | `90th percentile / standard deviation / interquartile range / upper quartile LTV` refuse, but with a raw internal string: `Execution failed: 'distribution' is not a scalar aggregation`. Safe, but not a governed refusal message. | LLM | Presentation |
| R-2 | `"What is the LTV spread?"` returns `ok=True` with £1,964,886,258.21 labelled `Count of`. Wrong measure entirely, but implausible enough to be self-evident. | LLM | Medium |
| R-3 | `"average loan size"` / `"mean loan size"` / `"average ticket size"` return a *distribution over Ticket Size* rather than a scalar mean. Deterministic only — the LLM path answers correctly (178,059.47). | Deterministic | Breadth |
| R-4 | `"What is the average ticket size in the back book?"` renders a malformed measure label: `Calculated: Count of ·` with nothing following. | Deterministic | Presentation |
| R-5 | Same five statistics on the deterministic path (percentile, stddev, IQR, quartile, spread) all return the weighted-average LTV with `ok=True` — the same substitution as §2. | Deterministic | **Same defect as §2** |

## 8. Pre-existing, not introduced by P0–P1L

`median` support and the `allowed_aggregations` permission model both long predate this
programme (`2823ce7`, `63c3075`, `db8329e`). No commit in the P0–P1L series touched the
aggregation permission path. This is a latent defect that the readiness review surfaced,
not a regression introduced by recent work.

## 9. What I have NOT done

- No production code changed.
- No fix attempted.
- No vocabulary broadened.
- No test modified.
- Nothing pushed.

## 10. Reproduction

```
scratchpad/cbr_agg_matrix.py deterministic     # 4/4 substituted
scratchpad/cbr_agg_matrix.py llm               # 2/4 substituted, LTV + rate
scratchpad/cbr_med_rep.py                      # 5x repeat, genuine LLM, 1 distinct outcome
scratchpad/cbr_stat.py {deterministic|llm}     # the wider statistic sweep
```

## 11. Direction requested

1. Confirm the **BETA BLOCKER** classification, or overrule it to a disclosed Beta
   limitation with the counter-argument in §6 accepted.
2. Confirm the intended semantics for a disallowed aggregation: **refuse** (my
   recommendation, consistent with the max/min precedent) versus **compute-and-disclose**.
3. Confirm whether `median LTV` should become *supported* (a weighted median is a genuine
   product question, not merely a permission change) or remain a governed refusal.

The readiness review is paused. On direction I will either resume it with this finding
carried into the scorecard as a named blocker, or take a separate implementation brief.
