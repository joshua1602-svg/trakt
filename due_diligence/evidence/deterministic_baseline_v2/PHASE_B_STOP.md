# Phase B — STOPPED before product code, on the trace

```
PHASE_A_SHA            = 5150bead
PHASE_B_SHA            = none — no product code was written
FILES_CHANGED          = 0 product files
PERIOD_BOUNDARY        = NOT ATTEMPTED
```

The brief required the trace first and said to stop if routing through the
existing owner needed a broader architectural change than expected, or if
repository evidence showed a scope dimension was already bound authoritatively
outside the compiler. Both conditions are met. Three findings, in ascending order
of importance.

## 1. The product's temporal architecture is snapshot-per-period, not row-per-period

* `snapshot/store.py` — `SnapshotHeader.reporting_date`: **one snapshot carries
  one reporting date.**
* `mi_agent/states/selectors.py` — `SnapshotSelector.as_of(client_id,
  reporting_date, route)` selects **which snapshot to load from a store**. It does
  not filter a frame.
* `mi_agent/period_change/periods.py` — "governed two-snapshot period resolution",
  resolving a requested date against a **sequence of `SnapshotFrame`s**, with a
  governed max-gap refusal (`_guard_gap`) that is exactly the
  materially-distant-observation guard Phase B wanted.
* `mi_runtime.run_mi_query` docstring: *"Flat mode preserves the existing
  single-CSV MI Agent exactly; state / temporal / risk modes run behind this
  boundary using a `SnapshotStore`."*
* `mi_query_executor.py` contains **no** `reporting_date`, `as_of_date`,
  `temporal_mode` or `execution_mode` handling. Nothing anywhere in `mi_agent/`,
  `engine/` or `analytics_lib/` selects a period from a flat multi-period frame,
  and **no flat-frame→`SnapshotFrame` adapter exists.**

So the flat path is period-blind **by design**: one flat dataset *is* one
reporting date.

**This makes my Phase A temporal fixture the wrong shape**, and that is a genuine
harness defect, reported rather than worked around as the brief requires. It
encodes three periods as rows in one DataFrame. Honouring a period on that shape
would mean filtering rows on `reporting_date` — which would create the second
temporal semantic owner the brief explicitly forbids, and would bake in a data
architecture the product does not use. The fixture would have driven the design.

The lens and dataset fixtures are unaffected by this finding; only the temporal
one is mis-shaped.

## 2. `AnalyticalScope.period` is a frame-selection DECLARATION, not an execution instruction

`mi_agent/query_plan_adapter.py`, `plan_from_spec`, verbatim:

> `dataset`, `portfolio_lens` and `period` **select the FRAME and are resolved
> upstream of the spec**, so they are **supplied by the caller** rather than
> guessed here.

And the lift direction is spec → plan:

```python
period = period or spec.reporting_date or spec.as_of_date
```

The module's stated invariant is a **round trip**, `compile_query_plan(
plan_from_spec(spec)) == spec`, and it is scoped in its own words to *"the
semantics that decide an answer: population, measure, aggregation, dimension."*
`mi_agent/tests/test_query_plan_is_the_live_contract.py` asserts exactly that set
— group dims, filters, metric, aggregation — and asserts **nothing** about
`reporting_date`, `as_of_date`, `portfolio_lens` or `dataset`. It passes today
(11 passed, 43 subtests).

So `compile_query_plan` not emitting those three fields is **the contract, not a
defect**. The frame has already been chosen by the time execution happens, which
is why `execute_query_plan(plan, data, semantics)` takes the frame from its
caller. Baseline v2's measurement stands — the asymmetry is real — but its
*interpretation* as "incomplete scope propagation the compiler must repair" is
contradicted by the architecture's own written contract.

Making the compiler emit a period would also put the round-trip identity at risk
for every spec that did not carry one, with a blast radius well outside
"period cases only".

## 3. The surface measured has no production caller

`grep` for importers of `query_plan_execution` / `query_plan` outside tests and
evidence returns only the cluster itself: `query_plan_result.py`,
`query_plan_compiler.py`, `query_plan_execution.py`, `query_plan_adapter.py`.
Production `/mi/query` runs `mi_agent.mi_agent_workflow.run_mi_agent_query`
(`mi_agent_api/app.py`, `adapters.py`, `chat_routing.py`). The adapter's own
docstring describes the migration posture: *"the live route can take the plan path
only where the lift is exact, and the shipped path everywhere else."*

So the plan→execution boundary is a **migration surface**, and the 11 v2 failures
plus 24 scope-bank failures are defects **of that surface**, not of what production
currently serves.

## What this does and does not change

Unchanged and still valid: core arithmetic sound, 0 numerical errors, filters
sound, core geography 5/5, bridge sound, expected refusals honoured. The
measurement was correct.

Changed: the *owner* of the scope-loss defect, and therefore the fix. It is not a
compiler patch.

## Options, for your decision

**Option 1 — re-shape the temporal fixture to the product's architecture, then
bind.** Build three real snapshots under a `store_root` via the snapshot store,
each one reporting date with independently known totals. Phase B then becomes:
`AnalyticalScope.period` → `spec.as_of_date` + `temporal_mode="as_of"` +
`snapshot_client_id`, and `execute_query_plan` routes through
`mi_runtime.run_mi_query` instead of calling `execute_mi_query` directly, so
`_run_state`/`_run_temporal` and `SnapshotSelector` keep all semantic authority.
This is faithful to the existing owner and needs no new temporal logic — but
routing the plan executor through the runtime **is** the broader architectural
change the brief told me to stop for, and it would need the round-trip contract
re-settled deliberately.

**Option 2 — accept that the three dimensions are caller-resolved, and fix the
receipt only.** The defect that survives finding 2 intact is governance, not
binding: the receipt does not evidence which period, dataset or lens the frame it
executed represented. A caller-resolved frame still has an identity, and an
answer that cannot say which period it is for is unauditable. This is small,
additive, inside the authorised movement set, and it closes E01/E06/P09/DS04/L05
without touching semantic authority.

**Option 3 — decide the plan path's intended future first.** If the migration
intends `QueryPlan` to become the live contract, period/dataset/lens must become
execution instructions and Option 1 is the right shape. If the plan path stays a
shadow surface, Option 2 is the whole of the honest work and baseline v2's eleven
failures are largely a statement about an unfinished migration.

My recommendation is **Option 2 now, Option 3 as the decision to take before any
Option 1 work**. Option 2 is attributable, small and useful whichever way the
migration goes; Option 1 spends architectural change on a surface nothing serves
yet.

Phase A remains intact at `5150bead`. No product file was changed. Phases C, D and
E are not started.
