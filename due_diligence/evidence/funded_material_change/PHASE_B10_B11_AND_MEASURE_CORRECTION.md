# Sprint B — the material-summary correction, and the B10/B11 gates

```
PRODUCT_BASELINE   dcb71a20
SPRINT             B_FUNDED_MATERIAL_CHANGE
LIVE_MODEL_CALLS   0    LIVE_MI_QUERY_CALLS 0    DEPLOYMENTS 0
```

## Part 1 — the target-state correction

### Which field is authoritative where `operation` and `change_form` overlap

Stated before anything was written, and now written into
`vocabulary.CHANGE_FORM_CANONICAL_OPERATION`:

> **`change_form` is authoritative over the analytical FORM** — which owner
> implements the question, which governed mode that owner runs in, and therefore
> **who decides the candidate set of measures**. **`operation` is authoritative
> over the result SHAPE** requested within that form, and is checked for
> compatibility against `CAPABILITY_OPERATIONS` for the form's capability.
>
> A compatible linguistic variant of the form's own action is canonicalised to
> it. An operation stating a shape the form's owner does not produce is refused.

This is not a new principle. The compiler's `_MEASURE_OPTIONAL` set already says
*"Operations that need no measure — the capability decides what it reports"*.
Before Sprint A there was no slot that could say "the owner decides the
candidate set" other than the operation, so the rule was expressed through the
operation. `change_form` now says it directly.

### How it was implemented — one rewrite, not a second rule

`normalise.py` gained rule 5, in the seam that already holds rules 1–4:

```
change_form = material_summary
operation   ∈ {summary, movement, compare}      →  operation := summary
```

That is the whole change. Because `summary` is **already** in `_MEASURE_OPTIONAL`,
the measure requirement then resolves itself through the existing mechanism. No
second measure rule was added, and `_MEASURE_OPTIONAL` was not touched — a
parallel rule keyed on the form would have been exactly the duplication this
sprint is meant to avoid.

Rule 5 does not fire where rule 4 recorded a form/measure conflict. Rule 4
deliberately leaves such an intent as it stands so the compiler can see the
disagreement; rewriting the operation on top of it would half-resolve it.

Measured, from structured intent only:

| `change_form` | `operation` | measure | outcome | plan operation |
|---|---|---|---|---|
| material_summary | summary | absent | **PLAN** | `summary` |
| material_summary | movement | absent | **PLAN** | `summary` |
| material_summary | compare | absent | **PLAN** | `summary` |
| material_summary | rank / breakdown | absent | REFUSE | — |
| material_summary | series | absent | CLARIFY | — |
| metric_delta | movement | absent | **CLARIFY** | — |
| metric_delta | compare | absent | **CLARIFY** | — |
| attribution | movement | absent | CLARIFY | — |
| attribution | bridge | absent | PLAN | `bridge` (unchanged) |
| level_comparison | compare | absent | CLARIFY | — |
| *(no form)* | movement | absent | **CLARIFY** | — |

The last row is the inference that was **not** added. Whether a bare "what
changed" IS a material summary is an interpretation question; with the slot
empty the compiler does not guess it.

`rank` and `breakdown` are refused rather than flattened: they ask for an
ordering and a grouping this composition does not produce, and turning them into
a summary would answer a different question while reporting the one asked.

### The runtime perimeter narrowed as a result

`plan_material_summary.OPERATIONS` went from `{summary, movement, compare}` to
`{summary}`. Canonicalisation happens once, in the compiler's normal form, so
the perimeter sees exactly one shape. A non-canonical operation reaching it means
the plan skipped that seam and is **refused rather than re-canonicalised** — a
second rewriter is a second place the two could disagree.

## Part 2 — the maintainability review

```
DUPLICATED_CALCULATION_LOGIC  = NO
DUPLICATED_OWNER_PROJECTION   = YES  (extracted)
REPEATED_DECLINATION_PATTERN  = YES  (left alone, deliberately)
```

**No duplicated calculation.** Ten `BinOp` nodes in the whole file: five are
string / set / list concatenation, two are the ×100 inside the two formatters,
three are `threshold / 100.0`. Not one re-derives a figure an owner computed.
The threshold comparisons divide the THRESHOLD rather than multiplying the
governed value, so no owner's number is rescaled even transiently.

**Duplicated projection — found and extracted.** `change.to_dict()` was called
three times on the same object, `bridge.to_dict()` three times, `dist.to_dict()`
twice: nine calls for four projections, each rebuilding a whole dict and
discarding most of it. `_split(x, keys)[0]` / `[1]` became `_blocks(owner,
context=…, quality=…)` returning all three blocks from ONE read. This is the
extraction the brief permits: no new framework, no governance obscured, and it
turns "one projection of one owner contract" from a claim in a comment into a
property of the code. It was made **after** the B10 bank existed to cover it,
not before.

It did not reduce line count materially (567 vs 546 after the two product fixes
below), and is not claimed to. What it reduces is nine projection sites to four
and five redundant dict constructions to zero.

**Repeated declination — left alone, as instructed.** Nineteen `Omission` sites,
each with a different reason sentence. Factoring them needs either a reason-code
registry (a new framework) or a sentence table indexed by code, which would hide
which generator declines for which reason. That is precisely the case the brief
says to leave.

## Part 3 — two product defects the gates found

Neither would have been visible without the fixture bank.

**1. A suppressed improvement was dropped silently.** With
`funded_limits.report_improvements: false` and at least one deterioration
present, `limit_status_transitions` returned `out, []` — so a configuration that
hides recoveries hid the fact that it was hiding them. That is the single
failure mode the whole `Omission` contract exists to prevent. Suppression is now
reported whether or not anything else qualified.

**2. `relative_change` was published unlabelled.** The projection refactor in
B2–B9 dropped the `relative_change_is` key. `relative_change` is a FRACTION of
the starting value; a consumer reading `0.05` as five percentage points would be
wrong by a factor of twenty. The label is restored — a label, not a rescaled
number, because publishing a converted figure is the thing this module must not
do.

**3. And one in the plan adapter.** `RELATIVE_MODE_BY_GRAIN` required a grain,
but the compiler's own normal form produces `relative_pair` + `periods_back=1`
with **no grain** for the commonest request in this family — and states what it
means: *"a pair with no distance IS the adjacent pair"*. The resolver has a
method for exactly that (`current_vs_previous`, the two adjacent GOVERNED
snapshots, whatever cadence the book reports on). Mapping an absent grain to
`month_on_month` would have assumed a monthly book; refusing it rejected the
question the contract was built for. Both are now wrong and the third answer is
right. A distance of more than one period is still refused rather than
approximated, because synthesising the dates here would be the period arithmetic
the resolver owns.

## Part 4 — B10, the deterministic fixture bank

`mi_agent_api/tests/test_insight_funded.py` — **51 controls, all passing.** No
I/O, no frames, no model, no deployment. Every fixture is built from the
workflow's own frozen dataclasses (`MetricChange`, `DistributionChange`,
`BalanceBridge`) rather than from hand-written dicts, so a fixture cannot drift
from the contract it stands for and keep passing.

| required proof | controls |
|---|---|
| quiet period | stated with what was examined; **not** emitted for an absent analysis; **not** emitted when nothing is comparable; switchable off |
| material KPI movement | reported with the owner's figures verbatim; balance separated as its own type |
| below-threshold suppression | suppressed AND named; the gate is chosen by unit, not by name |
| PASS → BREACH | concern, transition and breach amount carried |
| BREACH → PASS | info, "recovered", never a breach |
| headroom deterioration | attention short of a breach |
| eligibility transition | never called a recovery — the `deteriorated`-flag trap |
| concentration / composition movement | the workflow's own leaders, not a re-sort; below-gate suppression named |
| attribution enrichment | decomposition carried; withheld when its movement was immaterial; a non-reconciling bridge is a limitation, not an explanation |
| deterministic ranking | severity → type priority; attribution follows its movement; quiet never displaces a finding; ties broken by discriminator regardless of input order |
| omission evidence | every finding names its threshold AND its governed owner; every omission has a category and a reason; a cap reports as capped |
| incompatible-date refusal | unresolvable pair → `unavailable` brief; non-pair period forms; untranslatable grain; untranslatable distance |
| unsupported-scope refusal | pipeline / forecast / whole_book / absent base |
| configured threshold override | a moved threshold changes exactly one finding set; partial override leaves others alone; improvements switchable; quiet switchable |
| deterministic repeatability | byte-identical brief; stable `insight_id` across regenerations; different across dates; **unchanged when only the values change** |

Plus the composition receipt (mode, resolved pair, owners, counts read from the
RESULT not the intent), both reporting dates with provenance, no loan-level
data, and partial failure costing one section rather than the brief.

Structured-plan controls **A–D** are in
`tests/interpretation_v2/test_change_form_semantic_bank.py`, built from
structured intent with no recognition involved, exactly as required.

## Part 5 — B11, target-state controls

`tests/interpretation_v2/test_change_intelligence_target_controls.py` —
**20 controls, all passing.**

**Why they are not scored against the 135 bank.** That bank records, for all
nine questions of this family, `capability=period_movement, operation=movement`.
All nine expect the same thing, so the fixture cannot express the distinction
this sprint exists to make. It is not wrong about the capability; it predates the
semantic. These are target-state controls instead.

| control | target | proved |
|---|---|---|
| Q18 broad (3 operations) | material_summary execution | `period_movement` / `summary` / `portfolio_overview`, claimed and admitted by the runtime, **no measure named or asked for** |
| Q18 attribution variant | attribution owner | `funded_bridge` / `bridge`; the material-summary runtime does **not** claim it |
| Q19 broad + `direct` lens (3 operations) | material_summary execution | same contract, and the SCOPE SURVIVES as `source_portfolio_type = direct` |
| Q19 attribution variant | attribution owner | `funded_bridge`, scope intact |
| Q20 | metric_delta → period_movement **unchanged** | `period_movement` / `movement` — **not** canonicalised — mode `requested_metric`, measure preserved; not claimed by the material-summary runtime |
| the family side by side | three distinct contracts | three distinct `(capability, operation, mode)` triples; exactly one claimed by the material-summary runtime |

No question is parsed anywhere. The wording appears only in comments saying
which reader's request each control stands for. **No production module
references Q18, Q19 or Q20.**

**What these do NOT prove.** That the interpreter assigns the right form to any
particular sentence. That is the live interpretation gate, still pending, and a
separate measurement.

## Part 6 — LOC

```
FINAL_PRODUCT_LOC_ADDED   = 769
FINAL_PRODUCT_LOC_REMOVED = 6

  +567   mi_agent_api/insight_funded.py            (new)
  +126   mi_agent/plan_material_summary.py         (new)
  +25    mi_agent_api/insight_config.py
  +17 -1 mi_agent_api/insight_contract.py
  +15 -3 mi_agent/interpretation_v2/normalise.py
  +12 -2 mi_agent/interpretation_v2/vocabulary.py
  +7     mi_agent/interpretation_v2/compiler.py
```

The correction moved the figure from **717 to 769** (+52), against the 717
ceiling set for B2–B9. Where it went:

| | lines | why |
|---|---|---|
| the correction itself | +12 vocabulary, +12 normalise | rule 5 and its two tables |
| defect: silent suppression | ~+10 | an omission that was being dropped |
| defect: unlabelled fraction | ~+6 | the unit label restored |
| defect: absent grain | ~+10 | the adjacent-pair method and the distance guard |
| the projection extraction | ~+2 | nine sites to four; not a line saving |

Three of those five are bug fixes the gates found, not the correction's cost.
The correction proper is ~24 lines.

## Part 7 — regression

| suite | result |
|---|---|
| Weekly Portfolio Brief (3 files) | **90 / 90** — the gate, unmoved |
| `test_insight_funded.py` (B10) | 51 / 51 |
| `test_change_intelligence_target_controls.py` (B11) | 20 / 20 |
| change-form contract + semantic bank | 64 / 64 (44 at Sprint A) |
| `period_change` (mi_agent + mi_agent_api) | 308 / 308 |
| concentration / risk-limit | 22 / 22 |

### The whole suite, run twice

`mi_agent_api/tests/ tests/interpretation_v2/` run in full against a clean
worktree at the pre-correction commit `5e3a45eb`, and against the working tree:

```
5e3a45eb       21 failed, 2229 passed, 7 skipped, 118 subtests passed
working tree   21 failed, 2314 passed, 7 skipped, 118 subtests passed
```

**The twelve failing test ids are identical, in the same order, in both runs.**
Zero new failures. The 85 extra passes are the B10 bank, the B11 controls and
the six added change-form controls.

The twelve, none of which this work touches:

```
test_channel_parity.py                 3   geography / run-id parity
test_chat_routing_e2e.py               1   cumulative cohort conversion
test_concentration_tests_api.py        2   route absence and never-500
test_copilot_actions.py                2   deterministic path, row truncation
test_single_parse_and_substitution.py  1   unavailable dimension
tests/interpretation_v2/ guards        3   the boundary guards below
```

I did not widen the two boundary guards (`test_only_contract_modules_moved`,
`test_production_surfaces_are_untouched`). They are already red for unrelated
prior reasons, and widening a broken guard teaches nothing. Sprint B's
`mi_agent_api` insight files do appear in their already-red lists, which is
recorded rather than suppressed.
