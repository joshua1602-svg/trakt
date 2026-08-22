# P1M — Governed Statistic Identity

**Scope:** remove one demonstrated Beta blocker — a user's requested statistic must never be
silently replaced by a different statistic. No new analytics, no widened vocabulary.
**Baseline:** `5623c09` (Commercial Beta Readiness Review, clean tree).

---

## 1. Executive verdict

The blocker is closed, generically, on both parser paths.

"What is the median LTV?" returned **43.1562** — the exposure-weighted average — against a
true median of **39.6757**. It now refuses, naming the statistic asked for and stating that
the governed one was not substituted. "Median loan balance" returned the whole-book total of
**£1,964,886,258.21** for a **£156,864.66** question; it now returns the median. Neither fix
is median-specific: the mechanism is the statistic, not the word.

Nothing was widened to achieve it. The commercial bank is unchanged on the production path
(30 correct / 4 safe refusal), the immutable 40-question bank moved 14/40 → 14/40 with **zero**
changed answers, and the accumulated P-gates are green.

## 2. Root cause

Two independent routes produced the same wrong number, which is why a median-specific patch
would have fixed neither properly.

**Deterministic path — the request never reached governance.** The parser had no vocabulary for
"median", so the spec was built with the field's default aggregation and nothing recorded that
a different statistic had been asked for. Validation had nothing to refuse.

```
"What is the median LTV?"  ->  metric=current_loan_to_value  aggregation='weighted_avg'
```

**LLM path — governance detected the violation and the repair loop negotiated around it.** The
model emitted `median`; validation correctly refused it; the repair re-prompted; the second
attempt returned a permitted statistic; and that spec validated cleanly and was returned as a
**success**.

```
attempt1: ok=False  Aggregation 'median' not allowed for metric 'current_loan_to_value'
                    (allowed: ['avg', 'distribution', 'weighted_avg'])
attempt2: ok=True
FINAL:    aggregation='weighted_avg'   <-- returned as a successful parse
```

The correlation was exact: on the LLM path the only two measures that substituted were LTV and
interest rate — precisely the two whose `allowed_aggregations` exclude `median`. Balance and
borrower age, which permit it, answered correctly. **`allowed_aggregations` was enforced by
downgrading rather than by declining.**

A third mechanism sat behind both: even with repair stopped, the **deterministic safety net**
(`if det_vr.ok: return deterministic_fallback`) would have handed back the very substitution the
LLM path had just refused to make.

## 3. Smallest correction

Three changes, none of them about medians.

| # | Change | Why it is the smallest form |
|---|---|---|
| 1 | `_statistic_not_permitted(errors, requested)` — an ungoverned statistic is **not a repairable parse error** | Mirrors the existing `_missing_column_only` precedent, including its stated reason: "the LLM cannot fix this without an unapproved substitution". |
| 2 | `resolve_statistic_role(spec, question, semantics)` — carry the named statistic into the spec | Closes the deterministic hole: the request now reaches the governance layer that refuses it. **One-directional**: it only overwrites an aggregation that does *not* already satisfy the request. |
| 3 | `KIND_STATISTIC` facet, reconciled against execution evidence | The backstop. Catches a statistic lost *inside a measure set*, which the parse boundary cannot see. |

The guard on the deterministic safety net is what makes change 1 hold; without it the refusal was
re-answered one branch later.

**Change 1 is conditioned on the statistic the user named**, which the first cut got wrong. Two
specs fail with the identical validation error and deserve opposite treatment:

| Spec | What happened | Correct outcome |
|---|---|---|
| "what is the **median** LTV?" | the model asked for a median; the registry governs none for LTV; the only repair available is a *different* statistic | **refuse** |
| "**weighted** ltv by region" | the user asked for a weighted average and the model returned a `sum` on a percent metric; repair can only move the spec back *towards* the request | **repair** |

So a permission error stops repair only when the rejected statistic is the one the question
named, and the deterministic fallback is withheld only when it would not honour that statistic.
A question naming no statistic has none that a repair could move away from. The first cut
treated every aggregation-permission error as non-repairable, which withheld correct answers to
protect a statistic the user never asked for — caught by the full suite (§13).

**Why change 2 is one-directional.** "Average LTV" must keep its exposure-weighted definition —
the house convention for a ratio measure is what a plain "average" means. Rewriting it to a
simple mean would have been P1M's own silent substitution. So a named statistic that the
current aggregation already satisfies is left untouched.

## 4. Governed statistic identity

`mi_agent/statistic.py`. Narrow by construction: the families the MI Agent already needs, and
no others.

```
GOVERNED_STATISTICS = sum, count, count_distinct, avg, weighted_avg, median, min, max
```

Percentile, quartile, standard deviation, variance and spread are **not** added and are not
recognised — a test asserts it.

The relation is identity, with exactly one deliberate relaxation:

| Requested | Satisfied by | Reasoning |
|---|---|---|
| a plain **average / mean** | `avg` **or** `weighted_avg` | The question does not choose between them; the field registry does. This is what keeps "average LTV" correct. |
| an explicit **weighted average** | `weighted_avg` only | Naming the weighting is a specific request. |
| **median**, **min**, **max**, **sum** | themselves only | A median is not a weighted average; a maximum is not an average; a total is not a count. |

Three things are deliberately **outside** the check, each because a different governed guard
already owns it:

- **Analytic modes** (`contribution`, `share`, `distribution`, `loan_level`) — a contribution
  *is* the decomposition of a weighted average, not a rival statistic. P1A/P1D own these.
- **Routed answers** — specialist routes publish no statistic evidence. Safe, because an
  ungoverned statistic is refused at the parse boundary and never reaches a route.
- **Questions naming more measures than the contract carries** — P1E owns that refusal, and
  attributing the question's statistic to whichever measure survived named the wrong fault.

**Vocabulary is deliberately minimal.** Only a statistic a registry can *deny* can produce the
substitution, so `sum` and `count` are not recognised — their English is ambiguous ("the total
number of loans" is a count) and recognising them would risk refusing sound questions for no
safety gain. Ranking words are excluded for the same reason: in "which region has the highest
average LTV", "highest" ranks groups and does not name the statistic on the measure.

## 5. Before / after

| Question | Before | After |
|---|---|---|
| What is the median LTV? | **43.1562** (weighted average), `ok=True` | **Refusal** naming the statistic; no KPI, chart or table |
| What is the median interest rate? | 6.5597 / 6.5682 (weighted or simple mean), `ok=True` | **Refusal** |
| What is the median loan balance? | **£1,964,886,258.21** (whole-book total), `ok=True` | **£156,864.66** — the median |
| What is the median borrower age? | 71.3976 (the mean), `ok=True` | **71.0000** — the median |
| What is the average LTV? | 43.1562 weighted average | **unchanged** |
| Weighted-average interest rate | 6.5597 | **unchanged** |
| Total balance / loan count | unchanged | **unchanged** |
| Maximum / minimum LTV | governed refusal | **unchanged** |
| Give me median LTV and total loan balance | balance alone, presented as the answer | **Refusal** naming the median |

The refusal text (brief §8):

> I understood that you asked for median Current LTV, but median is not currently a governed
> statistic for Current LTV. I have not substituted weighted average Current LTV.

Verified: no KPI, chart or table artifact accompanies it, and the weighted average appears
nowhere in the text. The refusal carries no `Calculated:` line, so it cannot read as though
something ran (brief §12).

Receipts on successful answers name the statistic that executed — `Median Balance`,
`Weighted-average Current LTV`, `Median Borrower Age`.

## 6. Independent truth reconciliation

Computed directly in pandas from the fixture. The MI executor was not used as its own oracle.

| Figure | Expected | Actual | Variance |
|---|---|---|---|
| Median loan balance | 156,864.66 | 156,864.66 | **0** |
| Median borrower age | 71.0000 | 71.0000 | **0** |
| Weighted-average current LTV | 43.1562462674 | 43.1562462674 | **0** |
| Total balance | 1,964,886,258.21 | 1,964,886,258.21 | **0** |
| Weighted-average interest rate | 6.5597233425 | 6.5597233425 | **0** |
| Average borrower age | 71.3975532397 | 71.3975532397 | **0** |
| Loan count | 11,035 | 11,035 | **0** |
| Median current LTV (39.6757) | — | **not returned** | refused, as required |
| Max / min LTV | — | **not returned** | refused, unchanged |

Zero unexplained variance.

## 7. Genuine-LLM repeated results

Live API, 5 runs per case, provenance captured at the parse seam.

| Case | Expected | Distinct outcomes | Provenance | Verdict |
|---|---|---|---|---|
| median LTV | refuse | **1 of 5** | `validation_failed` ×5 | PASS |
| median loan balance | 156,864.66 | **1 of 5** | `llm` ×5 | PASS |
| median borrower age | 71.0000 | **1 of 5** | `llm` ×5 | PASS |
| average LTV | 43.156246 | **1 of 5** | `llm` ×5 | PASS |
| total balance | 1,964,886,258.21 | **1 of 5** | `llm` ×5 | PASS |
| median LTV + total balance | refuse | **1 of 5** | `llm` ×5 | PASS |

**Gate: GREEN.** 25 genuine model calls. Every case is fully deterministic across repeats.

One honest qualification: **median LTV is now refused before any model call is made**
(`validation_failed`, 0 calls). The deterministic spec already carries the ungoverned statistic,
so there is nothing to ask a model about — correct behaviour, and cheaper, but that row is not a
genuine LLM run and is not claimed as one. The other five cases are.

The short-circuit was checked for over-reach. It fires only when the metric genuinely denies the
statistic; questions the deterministic parser reads differently still reach the model and answer
correctly — "median loan size", "median balance in the acquired book" and "median loan balance by
region" all return Median Balance via `parser=llm`.

## 8. Commercial Beta regression

Full 34-question Commercial Beta bank, both paths.

| Outcome | Deterministic before | Deterministic after | Production before | Production after |
|---|---|---|---|---|
| CORRECT | 29 | **29** | 30 | **30** |
| SAFE_REFUSAL | 5 | **5** | 4 | **4** |
| **INCORRECT_SUCCESSFUL** | 0 | **0** | 0 | **0** |
| **SILENT_SEMANTIC_ERROR** | 0 | **0** | 0 | **0** |
| **HARD_FAILURE** | 0 | **0** | 0 | **0** |

**Production path: zero changed answers.** Deterministic path: **one** change, an improvement —

| Q | Before | After |
|---|---|---|
| C03 "What is the average loan size?" | `Total Balance · grouped by Ticket Size` | `Average Balance · grouped by Ticket Size` |

The requested statistic is now honoured. The question is still answered as a distribution rather
than a scalar; that is a parser phrasing issue explicitly out of P1M scope, and the production
path already answers it correctly (£178,059.47).

No refusal was weakened to hold the score.

## 9. P-gate regression

Accumulated semantic suites, run together:

```
tests/test_p0_cohort_identity.py        tests/test_p1f_exposure_semantics.py
tests/test_p1c_ranked_movement_e2e.py   tests/test_p1g_measure_identity.py
tests/test_p1d_aggregate_contribution   tests/test_p1i_scope_resolution.py
tests/test_p1e_golden_bank.py           tests/test_p1j1_vintage_seasoning.py
tests/test_p1e_measure_safety.py        tests/test_p1l_population_propagation.py
tests/test_p1e_multi_measure.py         tests/test_p1m_statistic_identity.py

496 passed, 1 xfailed
```

No previously governed semantic identity was weakened. Three scoping errors in the first cut of
P1M were caught by these suites, not by my own tests — recorded in §13.

`tests/test_p1m_statistic_identity.py` adds **54** tests: the identity relation including every
adversarial substitution the brief names (§10), vocabulary narrowness, registry permission, the
parser seam, end-to-end positives and refusals, receipts, and P1E composition.

## 10. Immutable 40-question bank

| | Before | After |
|---|---|---|
| Answered | **14 / 40** | **14 / 40** |
| Changed answers | — | **0** |

No churn, explained or otherwise. The bank was not optimised and was not touched.

## 11. Full repository suite

Definitive run, against the corrected code:

```
8,784 passed, 30 skipped, 21 xfailed, 48 warnings, 6 subtests passed
0 failed                                        in 1746.97s (0:29:06)
```

Baseline for comparison was 8,675 passed at `983a755`; the increase is the 54 tests P1M adds
(52 in the new bank, plus the two added for the repair discriminator) alongside the growth from
the readiness review. No test was deleted, skipped or weakened.

The first run of this suite failed 3 — see §13. Those failures were correct and the code was
wrong; this run is against the fix.

## 12. Telemetry status: **DEFERRED**

**What exists.** A governed audit seam is already present: `AuditMetadata` (capability, request
and correlation id, tenant, organisation, actor, channel, portfolio, snapshot, outcome,
`started_at`, `duration_ms`, `error_code`), projected by `trakt_core.audit.audit_event_from_result`
and emitted as one structured JSON log line by `emit_audit_event`, behind a `_FORBIDDEN_KEYS`
filter that already excludes answers, tokens, URLs and paths.

**Why it is not a small sink for MI semantic telemetry.** Two findings:

1. `emit_audit_event` is called only from `trakt_tools/execution.py`. It is **not wired into the
   `/mi/query` path**, so nothing is emitted today for the React MI Agent or Copilot.
2. `AuditMetadata` is a **shared governed envelope** used by every capability, and its
   `_FORBIDDEN_KEYS` design deliberately keeps free text out of audit events. Carrying the user's
   question, parser provenance, requested statistic and facet ledger through it means either
   widening that shared contract or introducing a second event shape.

Either is more than "minimum telemetry over an existing seam", and widening a shared governed
envelope three days from deployment is not a trade I would make for a nice-to-have. Per brief §18
I have **deferred** it. The blocker took priority.

**Smallest next step** (one focused change, not part of P1M): emit one MI-specific event at the
existing `_audit` construction point in `mi_service`, carrying what is *already computed
in-process* — parser provenance, route, requested statistic, facet ledger outcome, refusal reason,
`populationApplied`, duration — keyed by `request_id` so it joins to the existing audit event
without widening it. Whether the question text itself is captured is a privacy decision for the
client, and should be a config flag defaulting to off.

Refusals remain the most valuable signal a Beta can collect, and none of it is being captured
today. This should land at Beta start.

## 13. Remaining product decisions

1. **Weighted median for LTV.** Deliberately not invented (brief §3). Until a methodology is
   agreed, median LTV and median interest rate are governed refusals. The choice between a simple
   median, an exposure-weighted average and a weighted median is a business decision.
2. **Median for other ratio/rate measures** follows the same rule and the same decision.
3. **C03's shape** — "average loan size" answers as a distribution on the deterministic path. The
   statistic is now right; the shape is a parser phrasing matter, out of P1M scope.
4. **Registry omissions.** No `allowed_aggregations` entry was widened. If any omission is later
   judged accidental rather than deliberate, correcting it is a registry decision with evidence,
   not a code change (brief §6, Case B). None was found to be clearly accidental.

**Three scoping errors in the first cut**, each caught by an existing suite and each recorded
because they show where the check must *not* reach:

| Error | Caught by | Correction |
|---|---|---|
| Rewrote `contribution` to `weighted_avg`, destroying the P1D analytic | `test_p1d_aggregate_contribution_e2e` (12 failures) | Analytic modes are exempt |
| Refused every routed answer containing "average" | `test_p0_cohort_identity`, `test_p1g`, `test_p1j1` (8 failures) | Routes publish no statistic evidence; guarded upstream instead |
| Named the wrong fault for an over-cap measure list | `test_p1e_measure_safety` (1 failure) | P1E owns that refusal |

A fourth, found by my own bank: the statistic facet is raised by the word "average", so it was
re-triggering the low-confidence caveat on exactly the plain KPI answers that test exists to
keep it off.

**A fifth, and the most substantive, found only by the full repository suite.**
`mi_agent/tests/test_streamlit_mi_agent.py` exercises the repair loop and the deterministic
fallback using an invalid LLM spec built as **`sum` on a percent metric** — which raises the
*same* validation error as an ungoverned median. Treating that error as categorically
non-repairable stopped repair and blocked the fallback for a question ("weighted ltv by region")
whose user request was perfectly governable, and which the deterministic parser answers
correctly.

Three tests failed and all three were right. The correction is in §3: the block is conditioned
on the statistic the *user* named rather than on the one the *model* produced. This is a sharper
and more defensible rule than the one it replaced, and it is the reason the brief's instruction
to run the full suite before claiming a verdict earned its place — the focused bank, the
commercial bank, the 40-bank and every P-gate were green while this was still wrong.

Two of my own tests were updated to the refined signature, and two were added for the new
discriminator: repair is permitted when the model invented the statistic, and blocked when the
invented one is what was asked for.

## 14. Beta-blocker verdict

**The Commercial Beta blocker B1 is CLOSED.**

- An unsupported requested statistic can no longer fall back to a field default — on either path,
  through the repair loop, or via the deterministic safety net.
- The requested statistic survives into execution and is reconciled against what ran.
- Median LTV no longer returns the weighted average. Median loan balance no longer returns the
  total.
- Supported statistics are unchanged and independently reconciled to zero variance.
- Multi-measure questions cannot hide an ungoverned statistic.
- Receipts identify the statistic that executed; refusals do not imply one did.

Condition 1 of the Commercial Beta Readiness Review's four launch conditions is met. Conditions
2–4 (telemetry capture, published envelope, stated Beta scope) remain open and are unaffected by
this work — with the note that telemetry is now explicitly deferred rather than assumed.

---

P1M STATISTIC IDENTITY: PASS
