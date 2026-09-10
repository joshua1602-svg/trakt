# Phase 2A · Phase 4 probe gate

45 questions, 15 canonicals, `claude-opus-5`, 45/45 successful, 0 failures.
Cohort and metrics: `policy_probe.py`. Raw: `phase2a_probe.json`.

## Outcomes by group, against Run 6

| group | Run 6 | probe |
|---|---|---|
| concise (must keep planning) | PLAN 12 | **PLAN 12** |
| preservation | PLAN 6 | **PLAN 6** |
| policy | PLAN 21 | PLAN 18, CLARIFY 3 |
| ambiguity | PLAN 4, CLARIFY 2 | PLAN 5, CLARIFY 1 |
| **total** | PLAN 43, CLARIFY 2 | **PLAN 41, CLARIFY 4** |

Verdicts: fully correct 40 → **40**, partially correct 3 → **1**, clarification
2 → **4**, incorrect 0 → **0**.

## Defect movement

| metric | Run 6 | probe |
|---|---|---|
| DROPPED_EXPLICIT_FILTER | 1 (Q16B) | **0** |
| DROPPED_EXPLICIT_TIME | 1 (Q25C) | **0** |
| every other fixture defect | 0 | 0 |
| elaboration: measure spread | 13 | **10** |
| elaboration: dimension spread | 4 | **2** |
| canonicals showing any spread | 9 | **4** |

## Every question that moved — six of 45

| question | Run 6 | probe | adjudication |
|---|---|---|---|
| Q16B | PLAN / partial | **PLAN / fully correct** | the explicit product filter is preserved. The preservation policy working. |
| Q25C | PLAN / partial | **PLAN / fully correct** | the explicit forward-looking temporal is preserved. Same. |
| BB01C | CLARIFY | **PLAN / fully correct** | see below — this corrects an earlier claim of mine. |
| NL7A | PLAN / fully correct | CLARIFY | GENUINE_AMBIGUITY — see below. |
| NL7B | PLAN / fully correct | CLARIFY | GENUINE_AMBIGUITY. |
| NL7C | PLAN / fully correct | CLARIFY | GENUINE_AMBIGUITY. |

### BB01C — "What is the headroom?" now plans, and that is right

In the Phase 1.5 report I wrote that BB01C *should* clarify. That was my
editorial view, not the evidence. The pre-authored fixture expects
`capability: borrowing_base, operation: headroom` — taken from the source bank,
which gives BB01 the measure `borrowing_base_headroom` on the borrowing-base
route — and `get_portfolio_semantic_context` reports exactly one configured
funding facility. There is one materially clear governed reading, so the
anti-overconservatism rule correctly resolved it. The probe scores it fully
correct against truth authored before any run.

### NL7 — the three clarifications are NOT converted enrichment

This is the gate's central question, and the distinction is sharp.

The forbidden pattern is declining an **optional companion** measure by asking a
question instead. That is not what happened. The compiler's own reasons say
`MISSING_REQUIRED_SLOT: operation 'compare' needs at least one measure` — the
model declined to invent the **primary** measure, which the question does not
name. "From a risk perspective", "riskier", "risk profile": Trakt carries
current LTV, borrower age, arrears, PD, LGD and an internal risk grade, and each
produces different work.

Four independent pieces of evidence that this is real ambiguity, not timidity:

1. The NL7 fixture leaves `measures` unstated and is marked `human_review`,
   authored before any run, precisely because "riskier" names no governed
   measure.
2. `due_diligence/evidence/analytical_intent_v1/expected_semantics.yaml` records
   an unresolved TENSION on this same question (Q7.2) — it pre-dates this sprint
   entirely.
3. In Run 6 the three readers produced **three different risk baskets** (two,
   one and four measures). Three readers disagreeing about the primary measure
   is the signature of an ambiguous question, not of enrichment.
4. All three variants now clarify **identically, for the same stated reason**, so
   NL7 becomes paraphrase-invariant. The clarification names the competing
   governed readings rather than merely refusing.

Run 6 scored NL7 "fully correct" three times because the fixture states only
operation, comparison and temporal — it could not see that the measure baskets
disagreed. The apparent regression is the scoring catching up with a
disagreement that was always there.

## Gate

| condition | result |
|---|---|
| unrequested elaboration decreases materially | **PASS** — canonicals with spread 9 → 4; measure spread 13 → 10; dimension spread 4 → 2 |
| explicit semantic preservation does not worsen | **PASS** — it improved, 2 defects → 0 |
| clear questions still predominantly PLAN | **PASS** — concise control 12/12 PLAN; 41/45 overall |
| no new systematic clarification behaviour | **PASS** — confined to one canonical whose ambiguity is documented in two places that pre-date the run; no concise or preservation question clarified |
| policy merely converts enrichment to CLARIFY | **NO** — the declined slot is required, not optional |

**GATE = PASS.** Proceed to the full 135.

## The qualification, stated rather than buried

The elaboration reduction is real but **uneven**, and roughly half of it comes
from canonicals resolving rather than from restraint:

| canonical | Run 6 spread (measure + dim) | probe | |
|---|---|---|---|
| Q21 | 0 + 1 | 0 + 0 | resolved |
| Q24 | 1 + 0 | 0 + 0 | resolved |
| Q25 | 2 + 0 | 0 + 0 | resolved |
| NL7 | 3 + 1 | 0 + 0 | resolved via clarification |
| BB01 | 1 + 0 | 0 + 0 | resolved |
| Q08 | 2 + 0 | 2 + 0 | unchanged |
| NL6 | 1 + 0 | **2 + 0** | worse |
| Q10 | 0 + 2 | **2 + 2** | worse |
| SM09 | 3 + 0 | **4 + 0** | worse |

The three that worsened are the specialist-measure enumeration cases — a
reconciliation, a limit rank, a pipeline summary — which is exactly where the
`owned_measures` inventory pressure lives. The policy note addressed it in words
and the measurement says the words are not sufficient there.

Per the brief's own instruction, that is recorded as a candidate
**REPRESENTATIONAL** issue for the contract-normalisation phase rather than
chased further here: if a capability owns which measures a reconciliation
reports, the intent arguably should not carry them at all, and no prompt wording
fixes a slot that exists.
