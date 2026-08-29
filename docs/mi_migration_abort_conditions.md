# MI compositional migration — abort conditions

**Pre-registered. Written and committed before any route migration begins.**

Baseline: `migration_phase0/BASELINE.json`. Regenerate with
`python -m migration_phase0.freeze_baseline`.

Every stage of this programme has run past its estimate. A five-phase migration
without a stated stopping point has no end condition other than completion, so
these are the conditions under which it stops — decided now, against evidence
that will exist, rather than argued later against evidence that will be
contested.

Each condition names **what is measured**, **the instrument**, and **the
threshold**. A threshold expressed as "materially worse" with no measurement
attached is not a threshold and none is used below.

---

## A1 — Cost explosion

**Stops when:** after the first three route conversions, a subsequent
conversion's measured cost exceeds the bound those three establish.

**Measured, per converted route:**

| metric | instrument |
|---|---|
| production lines changed | `git diff --stat` for that route's conversion commits, excluding tests and docs |
| test files touched | same diff |
| new primitive implementations introduced | must be **zero** — see A2 |
| route-identity decision sites removed | `python -m migration_phase0.route_identity_inventory` before/after |
| conversion commits required to reach equivalence | count |

**Threshold.** Let the first three conversions (`portfolio_summary`,
`temporal_compare`, `funded_bridge`) give a per-route median *m* for production
lines changed and *c* for commits-to-equivalence. **The migration stops for
reassessment when any later route needs more than `2 × m` lines or more than
`2 × c` commits.**

Why a multiple of an observed median rather than a number chosen now: the study
did not measure a conversion, so any absolute figure written today would be an
invention. The first three conversions ARE the estimate; this condition says the
estimate must hold, not that it must hit a number nobody has evidence for.

**Recorded before the first conversion:** *m* and *c* are **not yet known**.
`portfolio_summary` did not reach a conversion in Phase 0 (see A4).

---

## A2 — Failure to reconcile

**Stops when:** a migrated route cannot reproduce the existing economic result
without route-specific bespoke exceptions or altered economic semantics.

**Measured:** the shadow-vs-shipped equivalence harness for that route, over
every case the route's own recogniser claims — verified per case, not assumed.

**Thresholds, all of which stop:**

1. any economic figure differs by ≥ **£0.005** or one unit of the measure;
2. equivalence is reached only by adding a branch that names the route, a
   period, a book, or a dimension — i.e. a bespoke exception;
3. equivalence requires changing what a figure MEANS (a different denominator,
   a different weighting, a different missing-value policy) even where the
   number happens to match;
4. a new implementation of a primitive is introduced rather than an existing one
   reused. Phase 4 is the consolidation of 4 `group` implementations into 1; a
   migration that adds a 5th has made its own successor phase larger.

**Not a stop:** a difference in prose wording where prose was never asserted
stable. Text is compared only where the baseline records it as stable.

---

## A3 — Governance cannot generalise

**Stops when:** a governance property required for composition cannot be
expressed independently of route identity.

**The four properties, each with its instrument:**

| property | today | instrument |
|---|---|---|
| facet proof | `grouping_proven` already reads `metadata.groupedBy` — a per-step declaration, not a route name | `route_identity_inventory` |
| thin-sample / denominator disclosure | arity-1 only — **known-open defect**, pre-registered | `tests/test_migration_preregistered.py::TestArityIndependentDisclosure` |
| population / denominator disclosure | population ledger accepts execution evidence only | P1L tests |
| partial-execution refusal | `AnalyticalPlan.required_kinds` + `AnalyticalResult.satisfied` | `tests/test_analytical_capability_layer.py` |

**Threshold.** The migration stops if, for any one of these four, the only way to
keep current answer semantics is to consult which route answered. Concretely:
**if the route-identity decision-site count measured by
`route_identity_inventory` does not fall to zero for a property after that
property's generalisation commit, that property has not generalised.**

**Recorded now, so the target is not moved later:** **14 production decision
sites** across **11 route-keyed constants** in three modules
(`execution_receipt.py` 12, `chat_routing.py` 1, `mi_service.py` 1).

**A correction to the scoping study.** The study reported "54 route-name
literals". That count included declaration sites — a route naming *itself* on
its own envelope — and the allowlist definitions themselves. Those are the
channel through which execution declares what it was, and a compositional layer
replaces *what* declares rather than the fact of declaring. Counting only
consumers that BRANCH on identity gives **14**. The coupling is smaller and more
tractable than the study implied. It is also broader in kind: the study named 7
allowlists; the inventory finds **11 route-keyed constants**, including four the
study missed — `ROUTE_FIXED_MEASURE`, `_ROUTE_GRANULARITY`, `_ROUTE_TIME_GRAIN`
(`execution_receipt`), `_ROUTE_NOUN` (`chat_routing`) and `_RUN_SCOPED_ROUTES`
(`mi_service`). The last of those decides whether an answer may proceed against
a run-scoped dataset, which is a permission, not a label.

---

## A4 — Interpretation ownership cannot be made singular

**Stops when:** the plan path requires rereading raw question text to recover a
decision the interpretation contract already owns.

**Distinguish two cases, because only one of them is an abort:**

* **The contract owns it and the plan rereads anyway** → **ABORT.** This is a
  second semantic owner and it is the defect the programme spent a month
  removing.
* **The contract cannot represent it at all** → **STOP AND REPORT**, extend the
  contract deliberately, test the extension. Not an abort; the prescribed path.

**Instrument:** `migration_phase0/shadow_portfolio_summary.py::assert_no_question_read`
asserts structurally that the plan builder's signature carries no question
parameter. A plan step that cannot be built from the contract is emitted as a
`Step` carrying `blocked=`, and a plan with any blocked step is a **refusal, not
an answer with the step quietly omitted**.

**Already fired, in Phase 0, as the second case:** the contract has no claim for
a source-portfolio lens. **9 of 9** `portfolio_summary` surface cases are
blocked. See `docs/mi_phase0_report.md` §4.

---

## A5 — Unattributable regression

**Stops when:** any existing delivered or refused behaviour moves and cannot be
explained as an intended migration equivalence change.

**Measured, by case/test name, never by count:**

* `mi_agent/tests/test_mi_calibration_bank.py` — 267 passed
* `question_interpretation.run_robustness_deterministic` — 32/6/4/2, **seasoning families Q1, Q7, Q8 by name**
* `question_interpretation.shipped_shapes` — 15 correct, 0 wrong
* `question_interpretation.routed_surface` — 31 passed, `rt_004` known-failing
* `question_interpretation.mi_recognition_diagnosis` — 15/7/10/29
* `question_interpretation.time_series_surface` — **silent drops must remain 0**

**Threshold.** One unexplained movement stops the step. Not a budget, not a
percentage: the first one.

**The P0 property is absolute.** `time_series_surface` silent drops = 0. A
migration step that reintroduces a silent drop has failed regardless of its
economics, and no equivalence result overrides it.

**Baselines move only when** the movement was pre-registered, the reason is
documented, and the change is part of an authorised migration step. **A baseline
is never updated to make a test green.**

---

## What is already known-open, and therefore not the migration's fault

Recorded so it cannot be discovered mid-migration and attributed to it:

1. **`test_q7_compares_the_two_governed_sides_and_reconciles`** — fails identically on a clean tree at `42cef00`.
2. **`routed_surface::rt_004`** — expectation pinned at `1b90fe4`; behaviour moved by `42cef00`'s time-axis widening; expectation not updated with it.
3. **Arity-2 disclosure defect** — 16 of 88 leaf groups thin and undisclosed at arity 2; live on shipped shapes C1–C5, which pass.
4. **Interpretation lens gap** — the contract cannot carry a source-portfolio lens.
5. **Filter clause join** — 71 questions carry unjoined halves, `clause_id` set on 0.

---

## Phase 4 discipline, pre-registered now

Duplicate-decision removal is later work and is not performed in this task. Its
rule is recorded now because it is the highest-risk operation in the programme:

* **exactly one duplicated decision removed per commit**;
* before deletion, prove the surviving mechanism is reached on **every path** the deleted one previously served;
* prove no consumer relies on the deleted mechanism's **disagreement** with the survivor;
* run every registered surface in A5 and compare **by case/test name**;
* **stop at the first unattributable movement**;
* **never batch deletions.**

There are 42 duplicated shape decisions across the three cascades (13 parser
branches, 15 recognisers, 14 executor branches). At one per commit that is a
floor of 42 commits for Phase 4 alone, and that floor should be part of any
schedule quoted for this programme.

---

## What ships during this phase

**Nothing client-visible.** T3, T4, T5, T6 and T7 remain closed. No new answer
shapes, no newly permissive routing, and no conversion of an existing refusal
into an answer merely because a shadow plan can compute it.

The only permitted user-visible change before Phase 5 is an explicitly
authorised governance correction, separately measured and separately
attributable.
