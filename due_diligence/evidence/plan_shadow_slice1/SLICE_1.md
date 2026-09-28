# Implementation Slice 1 — the signed interpretation control plane, in shadow

`START_SHA`  `6b994336`
`FINAL_SHA`  `acc41be2` (implementation and evidence; this document is committed on top)

| | |
|---|---|
| `SLICE_1_IMPLEMENTATION` | **PASS** |
| `SERVED_TO_USERS` | **NO** — the old production path answered every request |
| `LIVE_OPUS_CALLS` | **0** |
| `MI_BEARER_USED` | **NO** |
| `DEPLOYED` | **NO** |
| quarantined deterministic branch merged | **NO** |

## What was built

Five commits, none squashed:

| | |
|---|---|
| `59cacf9d` | `mi_agent/plan_runtime_adapter.py` — a `GovernedQueryPlan` handed to the existing deterministic executor, plus the focused suite |
| `e1c4f9c2` | the bounded shadow call site in `mi_agent_api/mi_service.py`, the comparison ledger, and the does-not-serve suite |
| `27e2a58f` | the corpus replay harness |
| `91d50603` | three facets the replay caught the shadow dropping |
| `acc41be2` | Gate 5 evidence: `corpus_replay.json`, `shadow_ledger.jsonl`, `REPORT.md` |

The slice is the generic / funded / current-single-period / single-output /
at-most-two-dimension / no-specialist / no-explicit-lens shape and nothing else.
It is a **dispatcher, not a reader**: it never sees the question text (a test
asserts the module's source contains no `question` identifier), never imports
`re`, never defaults a value the plan left empty, and performs no arithmetic —
`mi_agent.mi_query_executor.execute_mi_query` owns every figure, as it always
has.

Exactly one legacy file was modified. In `mi_agent_api/mi_service.py`, four
lines before the point-in-time `return _governed_context(...)`:

```python
from mi_agent import plan_runtime_adapter as _plan_shadow
_plan_shadow.observe(result=result, frame=df, semantics=semantics, view=view,
                     portfolio_id=authorised.portfolio_id)
```

The import is inside the function deliberately, so an off flag loads no
interpreter; `observe` cannot touch `result` and swallows every exception.

Blast radius over the whole slice, `6b994336..acc41be2`: **8 files, 4435 lines
added, nothing removed and nothing rewritten.** One of the eight is legacy
(`mi_agent_api/mi_service.py`, `+9`), three are new tests and evidence, and
`mi_service.py` has not been touched since commit 2.

## Gates

### Gate 1 — focused tests: PASS

`mi_agent/tests/test_plan_runtime_adapter.py` **55 passed**;
`mi_agent/tests/test_plan_shadow_does_not_serve.py` **12 passed**.

The eligible-execution tests check figures against
`mi_agent/tests/portfolio_truth_oracle.py`, which imports nothing from the
product. The discipline tests assert the module reads no question text, imports
no `re`, no parser, no interpreter and no routing, that the flag defaults to off,
that an executor failure is captured rather than raised, and that no facet of the
plan contract is silently ignored.

### Gate 2 — the independent truth banks have not moved: PASS

Re-run at the final SHA, compared against the committed Phase A baselines:

| bank | verdicts | result file |
|---|---|---|
| `deterministic_baseline_v2/plan_execution_bank_v2.py` | 65 cases — 47 PASS, 11 FAIL, 1 OUTSIDE_CURRENT_SCOPE, 1 UNJUDGEABLE, 5 NOT_PLAN_LEVEL_REACHABLE | **byte-identical** |
| `deterministic_baseline_v2/scope_bank.py` | 28 cases — 4 PASS, 24 FAIL | **byte-identical** |
| CORE_GEOGRAPHY (C01–C05) | **5/5 PASS**; C07 OUTSIDE_CURRENT_SCOPE (ITL3 is a secondary Funded→Geography lens, not an MI-query route) | — |

The 11 FAILs and the 24 scope FAILs are the pre-existing, already-reported
scope-propagation findings, unchanged by this slice. The architecture trace
established that `dataset`, `portfolio_lens` and `period` not reaching the spec
is the *contract* — `query_plan_adapter`'s own docstring says those three select
the FRAME and are resolved upstream — so they are recorded, not treated as
regressions, and slice 1 does not touch them.

### Gate 3 — off by default: PASS

With `MI_AGENT_PLAN_SHADOW` unset, `observe` returns before building a plan: no
interpreter is imported, no model is called, no execution happens and no ledger
row is written. Only the exact word `shadow` (case-insensitive, trimmed) enables
anything — `serve`, `primary`, `on`, `1`, `true`, `shadow_mode` and `""` all read
as off, each asserted. A test also reads `mi_service.py` and asserts neither
`interpretation_v2` nor `plan_runtime_adapter` appears above
`def _run_analysis`.

### Gate 4 — a shadow failure cannot reach a user: PASS

Four failure modes are **forced**, not hoped for: interpretation returns nothing;
the compiler refuses; the plan provider itself raises; the deterministic executor
raises. Each ends in silence or a ledger row, never an exception and never a
changed answer — the control envelope is compared before and after every call. A
ledger path that cannot be written stays silent. An eligible shadow reaches
parity and the control is still what is served. A numerical difference is
reported for adjudication rather than resolved, because the control is not the
truth oracle.

### Gate 5 — the corpus replay: PASS, with the substitution stated

Full detail in `REPORT.md`. The 843-question corpus is **not present at HEAD** —
it lives only on the quarantined deterministic branch — so the substitute is the
seven complete recorded benchmark runs at HEAD, replayed with zero live calls.

```
distinct questions                                 135
recorded interpretations replayed                  945   (135 × 7 runs)
plan_id identical to the signed-off recording      126 / 126
SLICE1_ELIGIBLE (signed-off run)                    17 of 135
SLICE1_ELIGIBLE (aggregate)                        115 of 945
executions                                          73
    EXACT_SEMANTIC_PARITY                           58
    PRESENTATION_ONLY (grouped, no scalar)          15  → 15 GROUPED_CELL_PARITY
    NUMERICAL_DIFFERENCE                             0
    SHADOW_EXECUTION_ERROR                           0
    DISPOSITION_DIFFERENCE                           0
OUTSIDE_REPLAY_FIXTURE                              42
LIVE_OPUS_CALLS                                      0
```

Parity is judged on figures and semantics. Answer text is never compared — two
paths can phrase one correct figure differently, and a prose match would hide a
population difference behind identical wording.

### Gate 6 — regression: PASS

Full capture, no truncation. The earlier voided broad run was voided precisely
because it was piped through `tail -3`; nothing here is read through a filter
that could hide a line, and the one filter that *did* hide two lines was caught
and corrected (below).

**`mi_agent_api/tests` — the suite carrying the modified call site.** Run three
times: with the change at commit 2, with `mi_service.py` stashed, and again at
the final SHA.

| | result |
|---|---|
| with the change (commit 2) | `20 failed, 1820 passed, 7 skipped` |
| without the change (`mi_service.py` stashed) | `20 failed, 1820 passed, 7 skipped` |
| at the final SHA | `20 failed, 1820 passed, 7 skipped` |

The failing sets are **identical in all three**, diffed line by line. All are
AssertionError/IndexError in 11 distinct pre-existing groups, not environment
faults. **Zero failures attributable to this slice, and zero accidentally
fixed** — the second matters as much as the first: a test that starts passing
because of an unrelated change is an unexplained movement.

**`mi_agent/tests` — the unit suite.** `13 failed, 1497 passed, 264 skipped,
7 xfailed` at the final SHA. Every one of the 13 was then re-run at `START_SHA`
in a separate git worktree, so nothing in the working tree was mutated to get the
comparison:

| | START_SHA | final SHA |
|---|---|---|
| the 5 files holding the 11 `FAILED` tests | `11 failed, 150 passed` | `11 failed, 150 passed` |
| `test_a_span_has_one_role.py` (2 `SUBFAILED`) | `2 failed, 7 passed` | `2 failed, 7 passed` |

Failing sets **identical at both SHAs**, so all 13 are pre-existing and none is
attributable to the slice. They sit in parser, predicate-extraction, receipt and
ranked-movement code that `plan_runtime_adapter` does not import.

A filter defect of my own, recorded: my first pass grepped `^FAILED` and so
counted 11 where the suite reported 13. The two it missed were `SUBFAILED` lines
in a twelfth file, and that file was then checked at both SHAs too. An 11-versus-13
discrepancy between a total and a list is exactly the kind of gap a regression
claim must not be built on.

## What the slice found, and what was fixed

Three silent-loss paths in the new adapter, all in code written during this
slice, all fixed in `91d50603`:

1. **A geography axis was dropped instead of refused.** Five eligible recorded
   plans carry a geography binding with `group_by: True` on
   `canonical_region_reporting`; the adapter grouped by their other dimension
   alone and would have presented that as the authorised breakdown. Geography
   bindings are now an ineligibility.
2. **Two predicates on one field collapsed.** `MIQuerySpec.filters` is keyed by
   field, so an LTV band stated as two bounds would have reached the executor as
   `< 80` alone. Refused now; a band stated once as a `between` still executes,
   as the list the executor's filters have always held.
3. **The ledger's requested half could not state what was asked** — filters were
   `{field: value}`, losing the direction the resolved half keeps in
   `applied_predicates.op`. Now `{field, comparator, value}` entries in order.

And the facet guard that let (1) through was rewritten structurally: it now reads
the facet list off `GovernedQueryPlan` itself and requires every field to be
declared carried, refused, or identity, so a facet added to the contract later
fails the test until somebody classifies it.

One harness defect of mine, recorded because the engine was right: the
control's case-folding guard tested `column.dtype == object`, which pandas
reports as `str` for this book, so it never fired and the control read 0 rows
where the engine correctly read 93.

## What this slice does NOT establish

Stated plainly, because a shadow that overclaims is worse than no shadow:

* **The old-versus-new disposition comparison is unmeasured.** The recorded runs
  hold interpretation outputs only — no legacy answer was ever stored beside
  them — so the control in the replay is an independent oracle, and every ledger
  row says `control_route: independent_control_oracle`. Comparing the two paths'
  dispositions on the same questions needs the shadow running against live
  traffic, or live calls on the legacy route. Neither was authorised here.
* **13% of the bank is in scope.** 17 of 135 for the signed-off run. The bank is
  dominated by specialist capabilities, which is what an MI bank looks like.
* **6 eligible plans per run were not executed.** `ticket_bucket` and
  `interest_rate_bucket` are governed registry fields the independent replay book
  does not carry; those cases are untested, not broken.
* **Nothing about serving.** No answer from this path has ever reached a user,
  and enabling that is not part of this slice.

### One observation handed to the interpretation workstream, not fixed here

`Q12A`/`Q12B` compile "balance by LTV bucket and age bucket" to
`operation: breakdown`; `Q12C` — "Plot portfolio balance across LTV buckets and
borrower-age buckets" — compiles the same two-axis request to
`operation: distribution`, and so falls out of the slice. The slice is not
paraphrase-invariant at the operation boundary. That is a property of the
interpretation, which this slice is forbidden to touch, so it is recorded and
left alone.

## Not done, by instruction

No serving, no deployment, no merge or cherry-pick of the quarantined
deterministic branch, no change to `interpretation_v2`, the Opus prompt, compiler
policy, the temporal architecture, `SnapshotStore`, geography, lens, formulas or
specialists, and no legacy code deleted. Slice 2 is not started.
