# Contract normalisation — result

`CONTRACT_NORMALISATION = PASS`

Two defects removed, one cleanly bounded, which is what the brief's third
criterion allows. Validated entirely by deterministic replay: **0 live Opus
calls**.

## The three normalisations

### 1. Period labels in plan identity — REMOVED

**Before.** `PeriodBinding.labels` was part of `plan_id`. Q19A/B carried
`labels=["last month"]` and Q19C carried `["month-on-month"]`; every other field
was identical and the three plans hashed differently. Run 6 had already measured
the same fault on NL9, where one variant's `["today's pipeline"]` was the sole
difference between three plans.

**After.** `plan.identity_labels()` excludes labels from the identity for the
forms whose window is settled by `form` + `grain` + `periods_back` alone:
`current`, `previous_reporting_period`, `relative_pair`. The plan still carries
every label it was given, so a disclosure can quote the period the user named.

The other forms keep their labels in identity because a label can be
load-bearing there. `explicit_period` is nothing BUT its label — the compiler
refuses one with no labels. `range` and `series` accept a label as their only
statement of span, which the compiler proves by raising `AMBIGUOUS_PERIOD` for a
range with no labels, no grain and no period count. `forward_looking` may state a
horizon only in words.

**Evidence.** 42 of 327 replayed plan identities moved for this reason alone.
Tests: April and May remain two different plans; every wording-only form returns
an empty identity label set.

### 2. Duplicate relative-period representation — REMOVED

**Before.** Three spellings of "this period against the previous one":

```
previous_reporting_period                  (Q18A, Q18C, Q20A)
relative_pair + periods_back = 1           (Q18B, Q19A, Q19B, Q20B)
relative_pair + periods_back absent        (Q19C, Q20C)
```

**After.** One canonical form, `relative_pair` + `periods_back=1`, established
before plan identity. For an operation that spans two periods,
`previous_reporting_period` cannot mean one period — a movement over a single
snapshot is not a movement — so it means the pair. And a `relative_pair` with no
distance is accepted as COMPLETE by the compiler (unlike a range or a series it
raises no `AMBIGUOUS_PERIOD`), which leaves adjacent as the only available
reading.

The third spelling is the reason to record a correction: my first implementation
normalised only `previous_reporting_period`, and it converged **nothing** —
invariance stayed flat at 7/19, 18/45, 20/45 because Q19C and Q20C still differed
from their siblings over `periods_back` absent against 1. The replay caught it
before any of it was committed.

Scope limits, both tested: `periods_back=0` is not folded in (a pair zero periods
apart is not the adjacent pair), and no other form gains an implied distance —
NL5's forward-looking absent-against-zero belongs to the representational backlog
this sprint is told not to broaden into.

**Evidence.** 25 normalisations applied across 327 intents. Both spellings were
already members of the compiler's `_MULTI_PERIOD_FORMS`, so the rewrite provably
cannot turn a plan into an unsupported composition — and the replay confirms it:
0 outcome changes.

### 3. Movement / bridge owner duplication — CLEANLY BOUNDED

**Before.** A specialist measure is owned by exactly one capability, yet the model
also states `capability` and could name a different one. Choosing between
internal implementation owners is not the interpreter's job.

**After.** The owner is DERIVED, not claimed: where every measure in an intent is
owned by exactly one capability, the compiler binds that capability regardless of
what the model said. The rewrite is withheld when the derived owner does not
support the stated operation, so it can never convert a plan into
`UNSUPPORTED_OPERATION`.

**The bound, and it matters.** This does **not** collapse the Q18/Q19 divergence
the brief describes, and the replay shows why. Across all 327 recorded intents:

| | count |
|---|---|
| owner matched the model's claim | 180 |
| no single specialist owner to derive from | 147 |
| **owner differed from the claim** | **0** |

Normalisation 3 never fired. The model has never, in any measured run, named a
capability inconsistent with the measure it asked for. So the
`period_movement`-against-`funded_bridge` divergence is not an owner-claim defect
at all — it is a **measure** choice:

```
period_movement / current_outstanding_balance     "net change in balance"
funded_bridge   / funded_balance_movement         the bridge's own movement figure
```

`period_movement` owns no measures; it reports the period change of whatever
generic measure it is given. `funded_bridge` owns `funded_balance_movement` and
decides its own arithmetic. Deciding which of those answers "how did the book
change?" is deterministic analytical behaviour, and this sprint may not change
it — so per the brief's own instruction, it stops there. The stop is recorded in
`normalise.BOUNDED` rather than only in this report, and a test asserts it.

What normalisation 3 delivers is therefore a closed freedom rather than a
corrected defect: the model can no longer author the implementation owner, which
narrows its authority whether or not it was abusing it.

## Convergence

| run | paraphrase-invariant before | after | gained | lost |
|---|---|---|---|---|
| 57-question probe | 7/19 | **9/19** | Q19, Q20 | none |
| Run 6 | 18/45 | **19/45** | Q20 | none |
| Run 7 | 20/45 | **21/45** | Q20 | none |

The target canonicals, on the probe:

* **Q19 — 3 plans → 1.** All three variants now compile to `plan_5681bb5`.
  Fully invariant.
* **Q20 — 3 plans → 1.** All three now compile to `plan_e7b18ee`. Fully
  invariant.
* **Q18 — 3 plans → 2.** A and C converged (both `funded_bridge` / `movement` /
  `funded_balance_movement`). B remains separate, and correctly: "What changed in
  the portfolio since last month?" was read as `portfolio_summary` / `compare` /
  `portfolio_overview` — a different measure and a different operation. That is
  an interpretation difference, not a representational one.

Run 6 gains only Q20 because its Q19 variants disagreed about the MEASURE, which
is the bounded case above; the policy recalibration, not normalisation, is what
fixed Q19's measure disagreement.

These are deltas from one instrument applied identically to both sides. The
absolute figures differ slightly from the headline numbers in earlier reports
because this replay recompiles from recorded payloads and counts a row with no
payload as non-invariant; only the deltas are claimed here.

## Safety

| claim | evidence |
|---|---|
| no outcome changed | 0 of 327, PLAN/CLARIFY/REFUSE identical |
| no reason code changed | 0 of 327 |
| no semantic information lost | 0 of 327 — labels, claimed capability, claimed time and measures all preserved |
| no unrelated plan changes | the ONLY authorised field that differs anywhere is `period`, on 47 rows, every one attributable to a named normalisation; 0 unexplained |
| deterministic | 327/327 stable over repeated compiles |
| idempotent | 327/327 — a second normalisation pass is a no-op and the canonical form is a fixed point |
| Opus policy unchanged | `opus_interpreter.py` byte-identical to `5ae1f73`; `system_blocks` hash `f1c8cd9402aec324` on both sides |
| contract shown to Opus unchanged | `intent_json_schema`, `tool_schema`, `system_blocks`, `metadata_tools` all hash identically to the baseline |
| metadata unchanged | `metadata.py`, `vocabulary.py`, `outcomes.py`, `banks/` byte-identical |
| deterministic execution unchanged | nothing outside `interpretation_v2` imports the package; the period contract names appear nowhere else in the repository |
| model-authored executable bindings | 0 — the intent schema is byte-identical, still `additionalProperties: false`, still 14 semantic slots and no binding slot |

The model's claim survives every rewrite. `intent_claims` records what the model
said, `compiler_bindings["normalisation"]` records what the contract then did to
it, and `CompileResult.intent` returns the original — a caller cannot receive a
rewritten intent as though it were the model's. "Did the model pick this?" stays
answerable.

## Why no live Opus calls were needed

The brief permits a live probe only if the contract presented to Opus changed in
a way replay cannot validate. It did not change at all: all four model-facing
surfaces hash identically to the baseline. Opus would receive the same prompt,
the same tool schema and the same metadata tools, and would therefore emit the
same payloads — which are already on disk and were replayed. Spending 135 live
calls here would have bought reassurance and no information.

## Boundary

Changed: `plan.py` (+42 lines), `compiler.py` (+38), `normalise.py` (new),
`test_contract_normalisation.py` (new), `test_interpretation_policy.py`
(two phase-scoped boundary guards repointed).

The two repointed guards are worth naming. They asserted the compiler had not
moved since `28422ec`, which was correct while the interpreter was the subject
and is wrong now that the compiler is. One is repointed to assert the
INTERPRETER has not moved since `5ae1f73` — the commit whose probe produced the
evidence — and the other is pinned between the policy phase's own endpoints so it
keeps stating a true fact about history. The live boundary for this sprint is
asserted in `test_contract_normalisation.py`.

Unchanged: `intent.py`, `equivalence.py`, `metadata.py`, `vocabulary.py`,
`outcomes.py`, `banks/`, `opus_interpreter.py`, every registry, and every
production surface (`mi_agent_api`, `frontend`, `trakt_notifications`,
`mi_workflows`, `engine`, `analytics_lib`, `trakt_tools` — 0 files).

229 tests pass in `tests/interpretation_v2/`, 33 of them new.

## Left alone, deliberately

The remaining representational backlog — overlapping operation members
(breakdown/distribution, departures/transition), the undefined `population.base`
for forward-looking questions, `origin_stage` against `pipeline_stage`, rank
against compare on a two-valued dimension, and NL5's forward horizon — is
untouched. So is the deterministic-engine recovery work.

The one residual inside this sprint's own targets is Q18B, and it is not a
contract defect: two readings of "the portfolio" picked two different governed
measures. If that should converge, it converges in interpretation or in the
registry, not here.
