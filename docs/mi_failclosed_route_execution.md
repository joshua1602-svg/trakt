# Fail-closed route execution

Base `656279c`. Pre-registration `85156a9`, committed **before** the production
edit and not adjusted afterwards. Control `28a1ae2`.

---

## 1. The defect, reproduced from execution

A controlled `InjectedExecutionFault` was placed at
`period_change_route.movement_receipt_for` — **after** the governed
period-change analysis had already run — and every recogniser was instrumented
to record recognition, handler entry, handler return and handler raise.

| | baseline | faulted |
|---|---|---|
| recognised candidates | `period_change_analysis`, `temporal_compare` | same |
| handlers **executed** | `period_change_analysis` | `period_change_analysis`, **`temporal_compare`** |
| raised | — | `period_change_analysis` → `InjectedExecutionFault` |
| final route | `period_change_analysis` | **`temporal_compare`** |
| ok | `True` | `False` |
| answer | the ranked movement | a plausible refusal about ranking |
| receipt | present | **absent** |

The site, in `chat_routing`'s dispatch loop:

```python
except Exception as exc:
    _logger.warning("route %s failed: %s", recogniser.name, exc)
    continue
```

A second, independent instance was found later on a **C1–C6** route:
`period_movement` failing and `period_change_analysis` answering.

## 2. The boundary, from the existing architecture

No new concept, and no route name in the fix. The registry already separates two
stages with different documented contracts:

* **`recognise()` — pre-claim.** A raise is already skipped and logged inside
  `RecogniserRegistry.candidates`. **Untouched.**
* **`handle()` — post-claim.** The documented decline is `return None`
  (*"falls through to the next candidate, exactly as the old chain did"*). A
  raise is not a decline; it is the failure of the route the registry selected.

**Entering `handle` is the claim.**

### Measured before changing it

All 882 corpus questions, every handler instrumented:

```
HANDLER RAISES TOTAL 0     BY ROUTE {}
handlers entered per question: 0 → 704 questions, 1 → 167, 2 → 11
```

Nothing on the normal path declines by raising. The 11 two-handler questions
fall through by `None`, which this fix does not touch.

## 3. The control

28 lines of code at the one dispatch site. On a post-claim raise: log with
traceback, return a governed execution-failure envelope stamped with the
**claimed** route, and stop routing.

Error discipline reuses what exists. `_envelope(ok=False, …)` that is neither
`controlledUnsupported` nor `unmappedQuestion` nor a no-rows case is already
classified `ErrorCode.CALCULATION_FAILED` by
`mi_service._classify_analytical_failure` — the governed code for "the
calculation broke", distinct from "I will not answer that". **No new public
taxonomy.** The exception class, message and traceback are logged and never
published; `metadata` retains `route`, `claimedRoute`, `executionFailure` and
`claimBoundaryCrossed`.

## 4. The correction that mattered most

**The first route-substitution detector was vacuous.** It derived
`claim_boundary_crossed` from `metadata.claimBoundaryCrossed` — a flag only the
fix publishes. Run against the defective tree the flag was absent, so the
invariant read `False`, and the detector printed `FAILS CLOSED · SUBSTITUTIONS
0` and **exited 0** over a run in which `temporal_compare` had visibly answered
after `period_change_analysis` failed.

A control whose signal only exists once the defect is fixed cannot detect the
defect. The boundary is now derived from the detector's own run — the fault site
executing *and* the claimed route's handler having been entered — both true in
either tree. `tests/test_failclosed_route_execution.py` fails if the detector
ever reads the published flag as its signal again.

```
migration_phase0.route_substitution_detector
  pre-fix tree  : SUBSTITUTIONS 2 of 2   exit 1
  post-fix tree : SUBSTITUTIONS 0 of 2   exit 0
```

## 5. Deliberate-fault tests

`tests/test_failclosed_route_execution.py` — 9 tests. Faults are injected
**inside** the claimed execution path, and run the real callable *before*
raising, so "the boundary was crossed" is established rather than assumed.

| | requirement | result |
|---|---|---|
| **F1** | fault after `period_change_analysis` claims | fault executed 1×, `temporal_compare` **not entered**, alternate executions **0**, final route `period_change_analysis`, `executionFailure` true |
| **F2** | the same in a different claimed route (`period_movement`) | identical fail-closed behaviour — the control is not C7-specific |
| **F3** | first candidate returns `None` before executing | next candidate **still runs**, boundary **not** crossed, no execution-failure marker |
| **F3b** | the fallthrough answer is identical to baseline | `temporal_compare`, `ok=false`, unchanged prose |
| **F4** | a governed refusal stays a refusal | the no-implicit-period refusal unchanged, no execution-failure marker, route `period_change` |
| **F5** | a normal delivered answer, no fault | route, ranked movement, receipt schema, element ranks and artifacts unchanged |

Plus: the failure answer leaks no exception class, message, traceback or repo
path; and an AST test asserts the control names no route.

### Discrimination, proven in both trees

Run against a worktree at `85156a9`: **F1 and F2 fail**; F3, F3b, F4, F5 and the
no-leak test **pass in both**, which is what negative controls should do. (The
no-leak test is not discriminating — the substituted answer leaked nothing
either — so it is a safety assertion, not evidence of the fix.)

## 6. Blast — no fault injected

882 corpus questions through the live `/mi/query` path in a worktree at
`85156a9` and in the working tree, one shared fixture, comparing route, `ok`,
answer, warnings and every artifact field by field:

```
QUESTIONS CHANGED 0     ROUTES CHANGED 0
ok=True   before/after  381 / 381
artifacts before/after  932 / 932
exceptions              0 / 0
keys removed / added    0 / 0
```

**ZERO NORMAL-PATH BLAST.** Every movement in this task is under an injected
fault, and is reported as the intended change in §5.

## 7. Preservation

21 modules in the working tree, 19 in the `85156a9` worktree, serial,
module by module:

| | before | after |
|---|---|---|
| modules | 19 | 21 |
| passed | 438 | 467 |
| failed | 9 | 9 |
| skipped / xfailed / xpassed / errors | 0 | 0 |
| timeouts | 1 | 1 |

```
INTRODUCED:    (none)
FIXED/REMOVED: (none)
summary lines: identical, module by module
```

The +29 passes are the two new modules (`test_failclosed_route_execution` 9,
`test_live_movement_receipt_evidence` 20), which exist only in the working tree
and are not in the baseline denominator. The 9 pre-existing failures
(`test_conversion2_period_movement` 5, `test_p0_execution_receipt` 3,
`test_mi_predicate_extraction` 1) and the one timeout
(`test_assurance_measurement_failure`) are unchanged and stay in the
denominator. The frozen canary bank passes 11/11 with its history untouched.

## 8. Cost

| | added | deleted |
|---|---|---|
| router control (`mi_agent_api/chat_routing.py`) | 55 | 3 |
| — of which executable code | 28 | 3 |
| — of which commentary | 27 | 0 |
| assurance instrument (`route_substitution_detector.py`) | 231 | 0 |
| tests (`test_failclosed_route_execution.py`) | 343 | 0 |

One dispatch site and one envelope builder. No new routing architecture, no
recogniser ownership change, no new semantic concept, no executor changes, no
route-by-route exception handling.
