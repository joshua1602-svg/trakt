# Conversion 5 — `temporal_compare` — **STOP — REGIME MODEL INCOMPLETE**

**Stopped at §4, dependency verification. No production code was changed.**

Base: `e9e05b7`. Stop conditions committed first, at `7d9f4c6`, and they stand
unchanged.

---

## The one-line result

The re-baseline predicted `temporal_compare` needed **three new generic
accessors and nothing else**. Measured against the code, it needs **two** — and
it also needs **two pieces of generic semantic work the re-baseline did not
enumerate**, one of which is not an accessor at all but a change to a governed
owner whose blast radius reaches every question in the product.

Stop condition **S9** fired: *a generic semantic dependency appears that the
re-baseline did not enumerate.*

## What was measured

Two instruments, both committed:

| instrument | denominator | result |
|---|---|---|
| `migration_phase0/route_ownership_temporal_compare.py` | 20 cases × 2 workspace tabs = **40 renders** | 13 owned, 7 correctly not claimed, **0 disagreements** |
| `migration_phase0/dependency_verification_temporal_compare.py` | 13 owned cases × 2 tabs = **26 readings** | see below |

The ownership instrument was **wrong twice on first run and was corrected by
the measurement, not the other way round**: `"Compare October and November
forecast balance."` is claimed by `analytical_composition`, and `"Compare Narnia
and November funded balance."` reaches no route at all because an unresolvable
token leaves `compare_periods` short of two. Both are now declared `X` cases
naming their measured owner. Neither is on the owned surface, which is why the
third view never appears in the dependency table below.

The dependency instrument was also wrong on first run, in a way worth recording
because it would have manufactured a much larger and entirely false finding: it
fed `_dataset_for` the **raw workspace tab**, where `mi_service` in fact feeds it
`workspace.resolve_active_view(question, tab)`. That mistake produced four
phantom disagreements and an invented claim that production and the contract use
opposite precedence rules. They do not. Corrected, the two agree on precedence
exactly.

## §4 — the enumerated dependencies, classified

| dependency | re-baseline said | measured | classification |
|---|---|---|---|
| **`dataset`** | new generic accessor, ~20 lines, "a guarded field read" | accessor is thin, but the **contract's dataset axis and production's `_dataset_for` are two different semantic rules** | **new generic accessor + UNPLANNED semantic-owner change** |
| **`subject` / measure** | new generic accessor, ~28 lines | `subject.candidate_concept` reconstructs `resolve_metric_key`'s two inputs on **26 of 26** readings | **already readable — accessor only, no new semantics** |
| **`time` / comparison periods** | *nothing* — "`comparison_period` already bridged in C4", "no unread-field surprises identified" | the pair is carried **nowhere structurally**: **0 of 26** readings have a structural field; the claim carries `raw_text="October, November"`, a display join | **UNPLANNED existing-axis field extension** |
| **`operation`** | new generic accessor | **not required at all** — the recogniser reads `spec.temporal_mode`, and C1–C4 all left recognisers on the spec | **over-enumerated; not needed** |
| **`source_scope`** | not mentioned | not honoured today: `_route_compare` never passes `scope` to `run_temporal_compare` | **route-specific only; must stay unhonoured** |
| **`dimensions` / `filters` / `population`** | not mentioned | none involved | **not involved** |

## The blocking dependency, in detail

Production's dataset rule is:

```python
# chat_routing._dataset_for(question, view), where view is already
# workspace.resolve_active_view(question, tab)
pipeline if any undisclaimed mention of (pipeline|case|kfi|application|offer)
        else resolve_active_view(question, tab)
```

The contract's rule, `projection._dataset`, is `resolve_active_view(question,
tab)` exactly — the same precedence, the same default, **a narrower vocabulary**.
It recognises the three *view names* and not the four *pipeline artefacts*.

Measured over the owned surface:

```
readings                                    : 26
dataset disagreements, contract AS BUILT    : 10
dataset disagreements, WITH the view wired  : 3
measure disagreements (at the same dataset) : 0
readings whose periods are STRUCTURAL       : 0
```

Seven of the ten disagreements are **wiring**, and that part is genuinely cheap:
the routed `_build_interpretation` never passes `caller_dataset`, so `qi.dataset`
always falls back to the default. Handing it the resolved view closes seven.

The remaining **three** are the vocabulary, and they are load-bearing:

| case | question | tab | production | contract |
|---|---|---|---|---|
| P3 | Compare October and November **case** count. | funded | `pipeline` → *Pipeline case count* | `funded` → *Loan count* |
| P4 | Compare October and November **KFI** count. | funded | `pipeline` → *Pipeline case count* | `funded` → *Loan count* |
| P5 | Compare October and November **application** count. | funded | `pipeline` → *Pipeline case count* | `funded` → *Loan count* |

A different tape and a different metric — not a label difference. Switching the
route onto the contract as it stands would change all three answers.

### Why this is not a bounded accessor

Four ways to close it, and none is both bounded and permitted:

1. **Widen `workspace.view_named_by_question`.** It is the sole input to
   `resolve_active_view`, which chooses the **frame** for every question in the
   product — every route, converted and unconverted. Measured over the 882
   distinct questions in the Stage 1 + Stage 2 corpora, this changes the frame
   decision for **8 (0.9%)** — and three of the eight are *forecast* questions
   that would be demoted to pipeline:

   ```
   named=forecast ['pipeline'] :: How much of the forecast comes from pipeline?
   named=forecast ['pipeline'] :: Compare current weighted pipeline forecast with run-rate extrapolation.
   named=forecast ['pipeline'] :: Where is the funded balance forecast to get to from today's pipeline?
   ```

   So the naive widening is not merely broad, it is **wrong**. The two readings
   are not the same reading at different widths: `_dataset_for` picks a **tape**
   *after* the view is chosen, while `view_named_by_question` picks the **view**
   under a `forecast > pipeline > funded` precedence. How a tape vocabulary
   composes with that precedence is an unanswered design question at the owner.

2. **Add the vocabulary to `projection._dataset`.** A second dataset owner
   inside the contract. Forbidden by §5.
3. **Put it in the plan accessor.** A phrase list in the plan layer. Forbidden
   by §5 three times over: no phrase lists, no second semantic owner, no
   raw-question rereading.
4. **Keep `_dataset_for` in the converted handler.** The route would still
   reread the raw question, which is not a conversion — and it is exactly the
   "shallower conversion than C1–C4 received" the re-baseline's 90-line
   route-specific *floor* exists to catch.

## The second unplanned dependency

`time.comparison_period` is a `Slot`, and `Slot` carries `state`, `raw_text`,
`span`, `reason`, `source` — no list. `projection._time` writes
`", ".join(spec.compare_periods)`. Recovering the pair means splitting a
display join, which is re-parsing a serialised form rather than reading a field.

This is the **same shape** as the closure C2 already made once, and the schema
says so in its own words about `window_periods`: *"`trend_window` carried the
WORDING and not the MAGNITUDE, so a consumer that needed the number had to ask
the owner again."* `comparison_period` carries the wording and not the periods.

Regime B, anchor **20 lines**, on an already-bridged axis. Absorbable in
principle — but §4 says unplanned generic dependencies are not to be absorbed,
and on its own it takes predicted shared to **62 + 20 = 82**, past the
pre-registered **75**. Absorbing it would have meant reporting REGIME MODEL
INCOMPLETE on cost while never naming why.

## Why this is not a candidate-selection error

The blocker is **not specific to `temporal_compare`**. `_dataset_for` has two
callers and they are the two cheapest remaining routes:

```
chat_routing.py:804   _route_compare      (63 lines)   — C5 candidate
chat_routing.py:930   _route_evolution   (167 lines)   — next best candidate
```

`period_change_analysis` reads the dataset too, and additionally needs
`operation`/ranking, which the contract does not represent at all.

**All three remaining compositional-core routes are blocked on the same
prerequisite.** Re-selecting the candidate does not avoid it; it only changes
which route discovers it. That is why this is a stop and not a re-rank.

## What the model got right, and what it missed

Right:

* The **regime vocabulary** held. Both misses are known regimes (a
  semantic-owner change; a Regime B field extension), not a new kind of cost.
* **Measure**: 0 disagreements on 26 readings. `subject` is genuinely ready.
* **Route-specific** work looks as small as predicted — 63 lines, delegating to
  an existing module.
* The re-baseline's central finding — *no reuse-only candidate remains* — is
  reinforced, not weakened.

Missed:

* It inspected **which axes** a route needs and **which fields exist**. It did
  not check **whether the contract's answer equals production's answer**. An
  axis can be bridged, filled, and still say something different. That check is
  now an instrument and should run before any future candidate is chosen.
* It counted `operation` as required. It is not — recognisers stayed on the spec
  through all four prior conversions, and nothing in C5 changes that.

## Verdict

**STOP — REGIME MODEL INCOMPLETE**, declared at §4 before any production change,
under stop condition **S9** pre-registered at `7d9f4c6`.

Thresholds are **not revised**: shared ≤ 75, route-specific 90–150, total ≤ 225
stand for the retry.

## The prerequisite, named

> **The governed dataset owner must carry the tape vocabulary production
> actually uses, or production must stop using it.**
>
> `chat_routing._dataset_for` treats an undisclaimed mention of
> `pipeline | case | kfi | application | offer` as selecting the pipeline tape.
> `workspace.view_named_by_question` does not. Until one governed owner answers
> that question for both, no route that reads the dataset can be converted
> without changing answers — and that is all three remaining core routes.
>
> This is a **product-semantics task with its own blast-radius proof**, in the
> shape of the contract-role and Defect A prerequisites that preceded C4. It is
> not migration work and must not be done inside a conversion.

A second, smaller prerequisite can be closed in the same task or separately:

> **`time.comparison_period` must carry the periods, not a joined string** —
> the closure `window_periods` already made once for `trend_window`.

## Recommended next step

**Close the dataset-owner prerequisite as its own task**, with a pre-registered
blast-radius bound over the 882-question corpora, then re-run
`migration_phase0.dependency_verification_temporal_compare` and expect
**0 dataset disagreements with the view wired**. C5 restarts from there against
the thresholds already committed at `7d9f4c6`.

Do not re-select the candidate. Every remaining core route is behind this
prerequisite.
