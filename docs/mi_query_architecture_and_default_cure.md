# MI Query architecture, and the default & cure close-out

*Sprint 2.5E close-out. Baseline `ccbcf56` → `8ab5d14`.*

## 1. What actually calculates the answer

In plain terms: **a canonical field, and one of seven fixed arithmetic
operations.** Nothing more exotic, and nothing invented at question time.

Traced from the code rather than from the architecture diagram:

```
"What is WA current LTV?"
  → llm_query_parser / deterministic parser        picks metric + aggregation
  → MIQuerySpec(metric=current_loan_to_value, aggregation=weighted_avg)
  → mi_query_validator                             checks the field exists here
  → mi_query_executor.aggregate_series             does the arithmetic
  → resolve_weight_field                           weight from the MI registry
  → 62.33%
  → metadata.metric_definition + reconciliation    the explanation
```

The five layers, and which does what:

| Layer | Role | Example |
|---|---|---|
| **Canonical facts** | the source of truth; columns on a governed frame | `current_principal_balance`, `number_of_days_in_arrears` |
| **Generic deterministic aggregation** | `aggregate_series` — sum, avg, median, count, count_distinct, weighted_avg, balance_sum | total balance, average borrower age |
| **Shared specialist analytics** | `analytics_lib` and the concentration library | stratification, concentration, cohort, migration |
| **Registered KPIs** | owned named metrics with versioned methodology | CPR, WAL, YTM, loss severity, default rate, cure rate |
| **LLM** | understands the question, picks metric/dimension/filter, phrases the answer | — |

**The LLM contributes no arithmetic**, and that is structural rather than
policed: it emits a `MIQuerySpec` naming a metric, an aggregation from a closed
enum, dimensions and filters. There is no expression language, and the executor
raises on an unsupported aggregation. Confirmed by reading
`aggregate_series` — the complete set of arithmetic MI Query can perform is
those seven branches.

**Where the registry sits.** MI Query resolves *fields*; the capability
registry resolves *metrics that are not fields*. CPR, WAL, YTM, severity,
default and cure are not columns and never will be, so the parser cannot reach
them. Before this pass they fell through to "I couldn't map this question".
Now the registry answers first.

---

## 2. The 9/50 finding, resolved

The framework now holds **52** metrics (the two added below). Classified by
what each actually needs:

| Type | Count | Capability link appropriate? | Current state |
|---|---:|---|---|
| **A. Registered derived metric** | **11** | **Yes** | all linked: total balance, loan count, average loan size, WA LTV, geographic concentration, contractual WAL, contractual YTM, prepayment (CPR), losses, default rate, cure rate |
| **B. Generic deterministic aggregation** | 2 | No | field + aggregation; MI Query owns these correctly |
| **C. Shared analytics operation** | 12 | No | stratify, concentration, cohort, migration, period change, valuation age — operations, not named KPIs |
| **D. Covenant-library metric** | 22 | No | resolve through `readiness_metrics` + the concentration library, which is already one shared implementation with thresholds |
| **E. Data/readiness control** | 2 | No | regulatory field coverage |
| **F. Judgement-only** | 1 | No | deliberately has no deterministic calculation |
| **G. Data gap** | 1 | No | external evidence Trakt does not hold |
| *Unclassified* | 1 | — | `ELIG_BREACH_DRILLTHROUGH`, a navigation affordance rather than a metric |

> **Is 9/50 a problem? No — it was a mis-framed statistic, and my 2.5E report
> should not have presented it as a coverage gap.**

The 41 unlinked metrics are overwhelmingly **category D** (22) and **C** (12):
covenant metrics that already run through one shared library with thresholds
attached, and shared analytical *operations* that are not named KPIs at all.
Forcing them into the capability registry would add a second identity for
something already governed once — precisely the duplication the registry
exists to prevent.

The genuine answer is that **capability links belong on named derived KPIs**,
and after this pass all of those that exist are linked (11). The correct
denominator was never 50.

---

## 3. KPI precedence — and the defect that proved it was needed

**The principle is now enforced, and it was not before.**

Tracing the real path turned up this, which is the worst kind of defect
because it answers confidently:

```
Q: "What is realised loss severity?"
   ok=True   metric=loss_given_default   agg=avg   →   45.0
```

`loss_given_default` carried **"loss severity"** as a synonym in the MI
semantics registry. So the question resolved to an unweighted mean of a
**supplied bank LGD estimate** — a modelled input — and returned it as
realised severity, while Trakt's own `OBSERVED_LOSS_SEVERITY@v1` measures
allocated losses over the balance at default. On a programme whose standing
rule is that observed loss is not expected loss, MI Query was quietly breaking
it.

Two fixes, both structural:

1. **The alias is removed at its source** — the generator, not the generated
   file — with the reason recorded there.
2. **Owned KPIs are marked** (`owned_kpi: true`) on 11 capabilities, and MI
   Query consults the registry before answering. Field-plus-aggregation
   capabilities are deliberately *not* marked: total balance, WA LTV and WA
   coupon are computed correctly by the generic path and it owns them.

A second, quieter defect from 2.5E: the capability explanation was wired to the
`unresolved_metric` branch, and **CPR, WAL, YTM and default rate all arrive on
the `unmapped` branch** — so it never fired for any of them, and my 2.5E report
claimed an integration that did not work on the real path. Both branches
consult it now.

**Measured behaviour after the fix:**

| Question | Before | After |
|---|---|---|
| "realised loss severity?" | **45.0** (a supplied LGD) | routed to `OBSERVED_LOSS_SEVERITY@v1` |
| "total outstanding balance?" | £10,080,000 | unchanged — generic path owns it |
| "WA current LTV?" | 62.33% balance-weighted | unchanged |
| "contractual WAL?" (fixed book) | "I couldn't map this question" | names `CONTRACTUAL_WAL@v1` and its module |
| "contractual WAL?" (ERM) | same generic refusal | `NOT_APPLICABLE`, contingent repayment |

Tests: `test_an_owned_kpi_that_is_available_names_its_methodology_instead`,
`test_a_generic_metric_that_can_answer_is_left_to_the_ordinary_query_path`.

**One honest limitation.** MI Query is a **single-frame** engine: one dataset
in, one answer out. Multi-snapshot KPIs cannot be *executed* there whatever the
deployment holds. The refusal says so and names the tool, rather than reporting
"1 snapshot available", which would misreport a limitation of the path as a gap
in the client's data.

---

## 4. Default methodology

**Source: ESMA, quoted verbatim.** The Annex 12 investor-report schema in the
repository (`DRAFT1auth.098.001.04`, `AnlsdCstDfltRate`):

> "Annualised Constant Default Rate (CDR) ... **Periodic CDR is equal to the
> [(total current balance of underlying exposures classified as defaulted
> during the period) / (total current balance of non-defaulted underlying
> exposures at the beginning of the period)]**. This value is then annualised
> as follows: `100*(1-((1-Periodic CDR)^number of collection periods in a
> year))`"

| | |
|---|---|
| **Methodology** | `OBSERVED_DEFAULT_RATE_CDR@v1` |
| **Numerator** | balance classified as defaulted **during** the period |
| **Denominator** | **non-defaulted** balance at the **beginning** of the period |
| **Basis** | **balance**, because the definition says "total current balance" |
| **Time** | period flow, annualised geometrically `1-(1-CDR)^n` |
| **Stock** | reported separately, and never called a rate |

Edge cases, each following from the definition rather than from preference:

| Case | Treatment |
|---|---|
| Already defaulted at opening | in **neither** side — it cannot default again, and counting it each period would compound one event |
| Originated / acquired during the period | not in the opening denominator; a default is counted if evidenced at close |
| Redeemed, matured | leaves the tape; not a default |
| Disappears from the tape | **not** a default — absence is not evidence |
| Cure then re-default | a default in the period it defaults; the series shows churn rather than netting |
| Partial repayment | irrelevant — balances are read, not derived |

**Default event.** Read from `default_date` (RREL72) or a defaulted
`account_status` (RREL69) — the **same** definition `classify_exits` already
uses, not a second one. **Never** derived from days past due.

---

## 5. Cure methodology

| | |
|---|---|
| **Methodology** | `OBSERVED_CURE_RATE@v1` |
| **Numerator** | delinquent balance returning to **current** (0 DPD) |
| **Denominator** | delinquent balance **able to cure** at the opening |
| **Eligible** | DPD ≥ 1 at opening and **not** already in default |
| **Cure** | destination = current. **Improvement is not cure** |
| **Excluded** | already-defaulted (a write-back is a different event); exited tape (disclosed) |
| **Minimum performing period** | none — a single snapshot pair cannot evidence one, and requiring one would need data Trakt does not have |

The worked case, hand-checked:

```
opening   A £100 current   B £100 90 DPD   C £100 60 DPD
closing   A £100 current   B £100 CURRENT  C £100 30 DPD

eligible  = B + C = £200        (A is current and cannot cure)
cured     = B     = £100        → 50.0%
improved  = C     = £100        60 → 30 days: better, still delinquent
```

Counting C would give 100%; dividing by the whole £300 book would give 33.3%.
Both are asserted as *not* the answer.

---

## 6. Where conventions differ — stated, not smoothed

**Default rate: one authoritative formula, two definitions of the event.**
ESMA itself carries both — `DfltdXpsr` uses "the definition of default
specified in the securitisation documentation", while
`DfltdXpsrCptlRqrmntRgltn` uses Article 178 CRR (more than 90 days past due,
on a *material* obligation, over *consecutive* days). Trakt reads whichever
the tape asserts and imposes neither. A third definition — a DPD threshold —
is deliberately not offered, because it would silently disagree with both.

**Cure rate: no regulator definition exists at all.** ESMA's only "cure"
fields are `CurePrd`, `BrchCureDt` and `CurePmtPssblty` — covenant and trigger
cure periods on CRE loans, not delinquency cures. Market sources agree that a
cure is a return to current and that a roll is progression between buckets, but
this is a **convention Trakt has selected and versioned**, not a standard it can
quote. The result carries that distinction in its `authority` field, and a test
asserts the wording is present.

That asymmetry is the honest headline of Part B/C: **one of these two metrics
is the regulator's and one is ours**, and a consumer quoting them externally
should know which is which.

---

## 7. Registry

Both capabilities move from `METHODOLOGY_NOT_APPROVED` to resolving normally:

| | `default_rate` | `cure_rate` |
|---|---|---|
| Methodology | `OBSERVED_DEFAULT_RATE_CDR@v1` | `OBSERVED_CURE_RATE@v1` |
| Authority | ESMA Annex 12 `AnlsdCstDfltRate` | Trakt-selected market convention |
| Requires | ≥2 snapshots; RREL72 or RREL69 | ≥2 snapshots; RREL68 |
| With history | **AVAILABLE** | **AVAILABLE** |
| Without | `UNAVAILABLE` / `INSUFFICIENT_HISTORY` | same |
| Aliases | default rate, CDR, constant default rate, periodic/annualised default rate | cure rate, cures, cure performance |
| Owned KPI | yes | yes |

**`METHODOLOGY_NOT_APPROVED` stays in the state model** even though nothing
now uses it. It is the right answer the next time Trakt holds data for a metric
whose definition is unsettled, and deleting it would guarantee that metric gets
mislabelled `UNAVAILABLE` — sending someone to a client for a field already on
the tape. A test records that reasoning where the assertion changed.

---

## 8. MI Query, measured

| Question | Result |
|---|---|
| "total outstanding balance?" | **£10,080,000** — generic path, `sum` |
| "WA current LTV?" | **62.33%** — generic path, `weighted_avg`, balance-weighted |
| "default rate over the last 12 months?" | names `OBSERVED_DEFAULT_RATE_CDR@v1` and `analytics_lib.history.default_rate`; explains MI Query is single-frame |
| "cure rate?" | names `OBSERVED_CURE_RATE@v1` likewise |
| "loss severity?" | routed to the owned metric — **no longer 45.0** |
| "contractual WAL?" (ERM) | `NOT_APPLICABLE` — contingent repayment |
| "unicorn ratio?" | generic refusal, unchanged |

---

## 9. Readiness

Two metrics added, and deliberately **beside** rather than instead of what
existed:

- `PERF_DEFAULT_RATE` → capability `default_rate`, tool `default_analysis`
- `PERF_CURE_RATE` → capability `cure_rate`, tool `cure_analysis`

`PERF_DEFAULT_SHARE` — the default **stock** — is untouched. It answers a
different question, and its guidance now tells an agent to read the two
together. The FACT / RULE / JUDGEMENT separation holds: the framework names the
capability, the capability names the one implementation, and thresholds stay in
rule packs. Framework: 52 metrics, 50 READY, 11 capability-linked. Tests assert
a named capability exists and that framework and registry cannot declare
different implementations.

---

## 10. Regression

Baseline `ccbcf56`, candidate `8ab5d14`, each from a pinned, clean worktree,
verified by `rev-parse` before and after and neither edited while running.

| | Baseline `ccbcf56` | Candidate `8ab5d14` | Δ |
|---|---|---|---|
| passed | 5,277 | 5,299 | **+22** |
| failed | 64 | 64 | **0** |
| errors | 13 | 13 | **0** |
| skipped | 33 | 33 | 0 |
| collected | 5,374 | 5,396 | +22 |

**Full ID comparison, both directions — all four sets empty:**

```
FAILED only at baseline  : none
FAILED only at candidate : none
ERROR  only at baseline  : none
ERROR  only at candidate : none
```

**The +22 is fully accounted for**, counted by collection in both worktrees:

| Source | Tests |
|---|---|
| `tests/test_default_and_cure.py` (new file) | 19 |
| `tests/test_mi_capability_registry.py` (27 → 30) | 3 |
| **Total** | **22** |

**Regression neutrality is claimed.** No test passing at `ccbcf56` fails at
`8ab5d14`, no new error appears, and every additional passing test is one this
pass wrote.

**Tests changed rather than added, and why.** Four assertions in
`test_mi_capability_registry.py` were rewritten, each because the behaviour
they pinned was deliberately superseded:

| Test | Change |
|---|---|
| `..._capability_that_can_never_be_available...` | `default_rate` and `cure_rate` removed from the always-refused set — their methodologies are now owned. `expected_wal` remains |
| `..._every_state_in_the_model_is_reachable...` | `METHODOLOGY_NOT_APPROVED` is no longer reachable from the four portfolios. The state stays in the model, and the test records why removing it would be wrong |
| `..._book_that_can_answer_is_left_to_the_ordinary_query_path` | split in two: a *generic* capability stays silent when available; an *owned KPI* names its methodology instead |
| `..._asking_for_default_rate_admits_the_methodology_is_unowned` | replaced — the answer is now the owned methodology identifier, not a methodology refusal |

Two further sweeps were run before committing, which is the direct lesson from
the stale-OpenAPI miss in 2.5D: `mi_agent/tests` (928 passed) and a 758-test
targeted sweep across MI Query, capability, readiness, history, OpenAPI and the
tool registry (754 passed, 2 failed — both already failing at `3ddf7af` and in
the 64-strong known set).

---

## 11. Sprint 3 go/no-go

> **Can we stop building the analytical foundation and build the
> Securitisation Readiness Agent?**

**Yes. There is no remaining blocker to autonomous portfolio investigation.**

The two metrics named as blockers at the end of 2.5E are now owned, versioned
and registered, and the default rate is the regulator's own formula rather than
a convention Trakt picked. An agent can discover what exists, see whether it is
available *here*, retrieve it, read the methodology and its authority, and
receive a reasoned refusal — distinguishing "not applicable", "missing input",
"needs an assumption", "needs a model" and "definition unsettled".

What remains is genuinely not blocking: React does not consume the capability
registry; 22 covenant metrics resolve through the library rather than the
registry, correctly; WA seasoning and remaining term are unbuilt; YTW needs an
option set that does not exist. None of these stops an agent investigating a
portfolio.

**One caution to carry into Sprint 3, from this pass rather than theory.** The
loss-severity defect was live for the whole programme and was found only by
executing a real query and reading the number. Every layer above it looked
correct — the registry was right, the analytics were right, the tests passed.
An agent will hit paths no test covers, so Sprint 3 should assume that the
first agent runs will surface this class of defect again, and should be built
to make that visible rather than plausible.
