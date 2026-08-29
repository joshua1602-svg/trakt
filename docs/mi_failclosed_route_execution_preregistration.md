# Fail-closed route execution — pre-registration

**Written and committed BEFORE the production edit.** Base `656279c`. Nothing
here is adjusted afterwards; a breach is reported as a breach.

---

## 1. The defect, reproduced from execution

Question: *"Which two geographic region obligors added the most balance since
last month?"* — contract `ranking of movement, increase, absolute, limit 2`,
dimension `collateral_geography` (alt `geographic_region_obligor`), comparison
`["last month", "latest"]`, measure `current_outstanding_balance`.

A controlled fault (`InjectedExecutionFault`) was injected at
`period_change_route.movement_receipt_for` — **after** the governed analysis had
already run. Every recogniser was instrumented to record recognition, handler
entry, handler return and handler raise.

| | baseline | faulted |
|---|---|---|
| recognised candidates | `period_change_analysis`, `temporal_compare` | same |
| handlers **executed** | `period_change_analysis` | `period_change_analysis`, **`temporal_compare`** |
| handlers that raised | — | `period_change_analysis` → `InjectedExecutionFault` |
| final route | `period_change_analysis` | **`temporal_compare`** |
| ok | `True` | `False` |
| answer | the ranked movement | a plausible refusal about ranking |
| receipt / rankedMovement | present | **absent** |

Fault site executed exactly once. **POST-CLAIM ROUTE FALLTHROUGH REPRODUCED.**

The site is `mi_agent_api/chat_routing.py`, in the dispatch loop:

```python
try:
    envelope = recogniser.handle(request)
except Exception as exc:            # a broken route defers, never 500s
    _logger.warning("route %s failed: %s", recogniser.name, exc)
    continue
```

## 2. The boundary, from the existing architecture

No new concept is introduced. The registry **already** separates the two stages
and documents different contracts for them:

* `recognise(request) -> bool | Recognition` — *pre-claim*. A recogniser that
  raises here is already skipped and logged inside
  `RecogniserRegistry.candidates`. **Untouched by this fix.**
* `handle(request) -> dict | None` — *post-claim*. The documented decline
  signal is **`return None`**: *"``None`` falls through to the next candidate,
  exactly as the old chain did."* A raise is not a decline; it is the failure of
  the route the registry selected to answer.

So the boundary is: **entering `handle` is the claim.** No route name appears in
the fix.

### Measured justification

Every one of the **882** Stage 1 + Stage 2 corpus questions was executed through
the live path with every handler instrumented:

```
HANDLER RAISES TOTAL 0        BY ROUTE {}
handlers executed per question: 0 → 704 questions, 1 → 167, 2 → 11
```

No route relies on raising out of `handle` as a way to decline. The 11 questions
that execute two handlers do so by the documented `None` route, which this fix
does not touch.

## 3. Pre-registered blast

### Intended change — only under an injected execution fault

`route A raises after claim → route B answers` becomes
`route A raises after claim → governed execution failure, stamped route A`.

### Must remain unchanged — expected blast 0

Normal successful routing and precedence; pre-claim recogniser fallback;
applicability deferral via `None`; honest refusals; every successful C1–C7
answer; ranked-movement economics; receipts; narration; contract interpretation;
planner behaviour; filters and populations; dataset selection; pipeline / stage
/ funnel; cohort progression ownership; the frozen canary baseline.

### Prohibited

Removing the candidate-route model; collapsing recognisers; changing precedence;
changing interpretation ownership; route-specific exception lists;
special-casing `period_change_analysis` or `temporal_compare`; converting
`recognise` failures into hard failures; turning governed refusals into internal
errors.

## 4. Error discipline, reusing what exists

The failure envelope is built with the existing `chat_routing._envelope(ok=False,
…)` and carries **no exception class, message or traceback**. `mi_service.
_classify_analytical_failure` already returns `ErrorCode.CALCULATION_FAILED`
(category CAPABILITY, HTTP 200) for an `ok:false` envelope that is neither
`controlledUnsupported` nor `unmappedQuestion` nor a no-rows case — which is the
existing governed code for "the calculation broke". **No new public taxonomy.**

Internally: the exception is logged with traceback, and `metadata` retains the
claimed route plus an explicit execution-failure marker so a substitution is
detectable after the fact.

## 5. Deliberate-fault tests, registered

* **F1** — fault in `movement_receipt_for` after `period_change_analysis`
  claims. Require: claimed route `period_change_analysis`; fault fires;
  `temporal_compare` **not executed**; no alternate route returns; governed
  execution failure; claimed route still attributable.
* **F2** — the same, in a **different** converted route, to prove the control is
  not C7-specific.
* **F3** — first candidate returns `None` (the documented decline) before
  executing. Require the next route still runs and the answer is identical to
  baseline. **The key negative control.**
* **F4** — a question the claimed route refuses for a governed reason. Require
  refusal semantics unchanged and no conversion into an internal error.
* **F5** — a normal delivered ranked movement, no fault. Require equivalence of
  route, result, receipt, metadata and narration.

For each fault: `claim boundary crossed = True`, `fault executed = True`,
`alternate route execution count = 0`. For F3: `claim boundary crossed = False`,
`next candidate execution count > 0`.

## 6. Stop conditions

* **STOP — FAIL-CLOSED DEFECT NOT REPRODUCED** — not triggered; §1 reproduces it.
* **STOP — CLAIM BOUNDARY REQUIRES ARCHITECTURAL CHANGE** — not triggered; the
  boundary already exists.
* **STOP — ZERO-BLAST CONDITION BREACHED** on any no-fault movement.
* **STOP — FAIL-CLOSED CONTROL ALTERED NORMAL ROUTING**.
* **STOP — FAIL-CLOSED FIX IS NOT LOCAL** if a new routing architecture,
  recogniser ownership, semantic concept, widespread executor change or
  route-by-route exception handling is required.
