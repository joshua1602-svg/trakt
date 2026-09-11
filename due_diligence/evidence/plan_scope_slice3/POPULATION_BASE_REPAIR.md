# Slice 3 — population base enforcement + legacy owner repair

Repaired at `c71eed49`. Offline throughout: no live model call, no deployment.

---

## 1. The two sides, and who owns each

```
REQUESTED   GovernedQueryPlan.population.base
            plan_runtime_adapter.requested_population_base(plan)

EXECUTED    the governed dataset identity the runtime resolved its frame with
            mi_service passes `view` to serve(); the same string it called
            datasets._resolve_query_frame(view, portfolio_id) with

RECONCILED  plan_runtime_adapter.check_population_base(plan, executed)
            called in plan_serving_canary._attempt, ABOVE the runtime dispatch
            and BEFORE any execution — so a refusal touches zero rows

PROVEN      mi_service._governed_plan_coverage adds one ledger entry,
            "governed_plan:population", on the SERVED envelope
```

Neither side is inferred. The question is not re-read, the rows are not counted,
no field is sniffed. `check_population_base` is asserted by test to contain no
reference to the question, a row, a frame or a regex.

**No second parameter was added.** `view` was already passed to `serve()` for
the evidence record and is already the identity the frame was resolved with;
introducing an `execution_population` beside it would have created two fields
that must agree, which is the defect class this repair exists to close.

---

## 2. Why it was a defect, measured

```
spec_for_plan   base=funded  →  filters {}  metric current_outstanding_balance
                base=pipeline→  filters {}  metric current_outstanding_balance   identical
                base=whole_book→filters {}  metric current_outstanding_balance   identical

check_eligibility  base=pipeline + generic_analysis  →  (True, '', '')
_governed_plan_coverage  reconciled filters and dimensions only
mi_service               supplies the FUNDED frame
```

Of the 135 signed-off plans: **83 funded, 34 pipeline, 8 forecast, 1
whole_book** — and every one the perimeter already admitted asks for funded. The
43 non-funded plans were stopped only by unrelated capability refusals. That is
luck, not a control, and the lifecycle ontology landed earlier in Slice 3 is
what makes `base = pipeline` reachable from ordinary language.

`workspace.resolve_dataset("What is in the front book?")` returns `funded`, so
the pipeline plan would have met the funded frame.

---

## 3. Controls 1–5 — the gate

| | | |
|---|---|---|
| 1 | funded base, funded runtime | **executes** |
| 2 | pipeline base, funded runtime | `POPULATION_NOT_EXECUTABLE`, zero rows touched |
| 2b | pipeline base, *pipeline* frame | still refused — this runtime does not execute pipeline |
| 3 | base absent (governed default) | **executes**, Slice 1 behaviour unchanged |
| 4 | runtime declares nothing (`None`, `""`, `"   "`) | `EXECUTED_POPULATION_UNPROVEN` |
| 5 | funded base, pipeline runtime | `POPULATION_BASE_MISMATCH` |
| — | `whole_book` | refused: it spans two frames and cannot be proven against one |

`EXECUTABLE_POPULATIONS = {"funded"}` is the registration point for a future
owner: `check_population_base(plan, "pipeline", executable={"pipeline"})`
passes, so a deterministic pipeline runtime declares itself and nothing in
`CandidateIntent`, the compiler or the plan is redesigned. **No pipeline
execution is added here.**

---

## 4. Controls 6–10 — the legacy lexical owner

### Two defects in my own first cut, caught by the controls

The first attempt deleted the four phrases from `_SEGMENT_PHRASES` outright.
Both failures below were found by running the controls end to end rather than
against the seasoning module alone:

1. **A signed-off canonical regressed.** `lending_windows_named` reads the same
   table, so *"the risk profile of recent originations versus the back book"*
   went from naming **two** windows (and correctly selecting neither) to naming
   **one** and narrowing to it. Deleting a phrase turned a comparison into a
   narrowing.
2. **The phrase came back as a grouping.** `resolve_seasoning_role` used to
   *consume* the axis the parser's dimension binder had raised from the same
   words. With no predicate it stopped early, so *"What is the back book
   balance?"* became a two-bar seasoning **breakdown** — the same
   reinterpretation wearing a grouping instead of a filter.

### What the repair actually is

**Recognition and masking stay whole; only SELECTION changed.** All ten phrases
remain in `_SEGMENT_PHRASES`; four are marked `_LIFECYCLE_OWNED`, and only
`resolve_population_predicate` consults the mark. `resolve_seasoning_role` then
completes the role decision — a lifecycle phrase is neither a filter nor an
axis — while leaving an axis the reader asked for in seasoning words alone.

Through the **production** parse seam (`ParsedQuestion.parse` →
`parse_with_repair`), which is what `mi_service` calls:

| question | before | after |
|---|---|---|
| "What is the back book balance?" | `seasoning_segment = Back Book` | `{}` / no axis |
| "What is in the front book?" | `seasoning_segment = Front Book` | `{}` / no axis |
| "…for ALP Acquired Back Book?" | `seasoning_segment = Back Book` | `{}` / no axis |
| "What is the acquired back book balance?" | `seasoning_segment = Back Book` | `{}` / no axis |
| "What is the balance of seasoned loans?" | `Back Book` | **`Back Book`** unchanged |
| "…recent originations versus the back book?" | `{}` | **`{}`** unchanged |

Forced through a safe fallback (canary on, principal allow-listed, interpreter
unreachable), `serve()` returns `None` and the legacy envelope now carries
`filters = {}` for both phrases. **The meaning no longer changes on fallback.**

Control 9 — six vintage phrases still select: `seasoned loans`, `legacy book`,
`seasoned book`, `recently originated`, `newly originated`, `recent
originations`. Control 10 — every phrase, including the four, is still masked
before the place-resolver, so P1J-1 ("Front" read as a region) stays closed.

---

## 5. Controls 11–14 — nothing else moved

* **Role**: `direct`/`acquired` still bind exactly one `source_portfolio_type`.
* **Named source**: `ALP Acquired Back Book` still binds
  `source_portfolio_id = alp_acquired`.
* **Slice 1**: the 135-case replay — every plan the perimeter admitted before is
  admitted now; the gate refuses **zero** of them.
* **Slice 2**: temporal plans share the one gate; a series is refused if it
  cannot name its population, and refused if it names another.

### Suite results, against a worktree at `c71eed49`

```
tests/interpretation_v2 + mi_agent/tests
    baseline   23 failed / 2018 passed
    repaired   21 failed / 2049 passed
    NEW failures: 0      FIXED: 1 (a stub predating `source_registry`)

mi_agent_api/tests
    identical failure sets: 18 before, 18 after, 0 new, 0 gone
```

One baseline-only failure, `test_registry_governance::test_checked_in_registry_matches_generator`,
is an artefact of running in a worktree and **not** something this change fixed:
its three inputs are byte-identical between the trees.

---

## 6. Verdict

```
POPULATION_BASE_ENFORCED                  = YES
PIPELINE_OVER_FUNDED_FRAME_POSSIBLE       = NO
POPULATION_BASE_COVERAGE_PROVEN           = YES
BACK_BOOK_LEGACY_SEASONING_OWNER          = REMOVED (selection); recognition and masking retained
FRONT_BOOK_LEGACY_SEASONING_OWNER         = REMOVED (selection); recognition and masking retained
SEASONING_MASKING_BEHAVIOUR_PRESERVED     = YES
LEGITIMATE_SEASONING_VOCABULARY_PRESERVED = YES
SEMANTIC_MEANING_CAN_CHANGE_ON_FALLBACK   = NO
RAW_TEXT_POPULATION_REREADS_AFTER_PLAN    = 0

SLICE_3_POPULATION_BASE_REPAIR = PASS
```
