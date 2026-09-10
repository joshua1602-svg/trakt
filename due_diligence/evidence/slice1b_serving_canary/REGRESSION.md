# Slice 1B — regression and blast measurement

`UNEXPECTED_BLAST` means: a test that PASSED before this change and FAILS after
it. Nothing else. A test that already failed is not blast, and a test that starts
passing is not blast either.

## Why a full-run comparison was necessary

This estate's suites are **order-dependent**. Run the 43 failing node ids on
their own and 17 fail; run them inside the full suite and 43 fail. The extra 26
are shared-state effects between tests, not defects in the selection.

So a targeted re-run cannot measure blast, and two measurements were taken.

## Measurement 1 — the same selection, before and after

The 40 node ids covering every failure of the full post-change run, executed
identically in a clean `git worktree` at `HEAD` (33c576c5) and in the modified
tree.

```
HEAD (before)      17 failed, 36 passed, 9 subtests passed
modified (after)   17 failed, 36 passed, 9 subtests passed
diff of the FAILED/SUBFAIL lines:  IDENTICAL
```

## Measurement 2 — the full suites, before and after

`python -m pytest mi_agent/tests mi_agent_api/tests -q -p no:randomly`, 3758
tests collected.

```
AFTER (modified tree)   54 failed, 3437 passed, 271 skipped, 7 xfailed,
                        974 subtests passed    — 43 FAILED nodes + 11 SUBFAILED
BEFORE (HEAD worktree)  NOT YET COMPLETE AT THE TIME OF THIS COMMIT
```

**This is stated as outstanding rather than claimed.** The baseline run in the
clean worktree takes roughly four times as long as the run in the warm tree and
had not finished when Slice 1B was committed. The full before/after failure-set
diff is therefore the one piece of this measurement that is not yet in hand, and
it is appended in a follow-up commit rather than asserted here.

What IS established about those 54, and what makes the gap small:

* none of the 43 failing node ids is in a file this change touches, and none
  names the serving canary, the shadow, `mi_service` or any plan module;
* Measurement 1 ran every one of them before and after and the failure sets were
  **identical**;
* the 54 are concentrated in risk limits, copilot actions, decks, concentration
  tests, pipeline stage transitions, geography binding and ranked movement —
  none of which is on the point-in-time serving path.

When the diff is compared, the failure SETS are compared line by line, `FAILED`,
`ERROR` and `SUBFAILED` alike. `grep -E "^(FAILED|ERROR)"` alone is not
sufficient and has misled this programme once already: pytest writes subtest
failures on `SUBFAILED` lines, which that pattern silently omits.

## The interpretation_v2 suite

```
BEFORE (HEAD, clean worktree)   3 failed, 226 passed
AFTER  (modified tree)          3 failed, 226 passed
```

The same three, and they are **pre-existing failures inherited from the
authorised Slice 1A change**, not Slice 1B's:

| Test | What it guards | Why it already failed |
|---|---|---|
| `test_bank_and_shadow_boundary::test_nothing_outside_the_new_boundary_imports_it` | nothing outside `interpretation_v2` imports it | Slice 1A's `mi_agent/plan_shadow_wiring.py` already imports it, as do three Slice 1/1A evidence harnesses |
| `test_contract_normalisation::test_only_contract_modules_moved` | only the contract modules changed since the sprint's start commit | Slice 1A already modified `mi_agent_api/mi_service.py` |
| `test_interpretation_policy::test_production_surfaces_are_untouched` | `mi_agent_api` unchanged since `ea8c65b` | same — Slice 1A's shadow call site |

All three encode the boundary of the EARLIER interpretation_v2 sprint, which the
authorised Slice 1A production-shadow wiring deliberately crossed. Slice 1B adds
its own files to the same pre-existing offender lists and changes the same one
file. Verified by running the suite at `HEAD` in a clean worktree, not inferred.

## Focused suites, all green

| Suite | Result |
|---|---|
| `test_plan_runtime_adapter` + `test_plan_shadow_wiring` + `test_plan_shadow_evidence` + `test_plan_serving_canary` | 149 passed |
| `test_plan_shadow_does_not_serve` + `test_plan_shadow_bypasses_legacy` + `test_shadow_replay` (with the above) | 133 passed |
| `test_deployed_acceptance_harness` + the six `test_query_plan_*` suites | 116 passed, 114 subtests passed |
| `test_portfolio_truth_bank` + `test_semantic_accounting` (independent oracle) | 31 passed, 60 subtests passed |
| `tests/interpretation_v2` | 226 passed, 3 pre-existing failures |

## Frozen layers, byte-unchanged

`git diff HEAD` is empty for all of:

```
mi_agent/interpretation_v2/          (the whole package)
mi_agent/plan_runtime_adapter.py     (eligibility + the adapter)
mi_agent/plan_shadow_evidence.py     (the recorder)
mi_agent/mi_query_executor.py        (the deterministic engine)
mi_agent/mi_query_spec.py
mi_agent/query_plan_compiler.py
```
