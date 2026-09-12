# live_change_form_gate_v2 — RETIRED UNRUN, superseded by v3

```
BANK_ID      live_change_form_gate_v2
BANK_SHA256  b9a842c6115ccf16f3091da22615a35c4e978b48663363519d01154a0abad1e4
LIVE_MODEL_CALLS_EVER_MADE   0
MANIFEST_EDITED_AFTER_HASH   NO
```

This bank was pre-registered and hashed **before** the current-anchor temporal
contract was closed. It was never run, and it has not been edited.

## Why it is invalid, specifically

Its own stated reason for not gating the temporal reading was:

> "the deterministic owners do not currently honour a `current` anchor for
> material_summary or attribution, which is an independent temporal-contract
> defect reported separately and not repaired in this sprint"

That sentence is now **false for `material_summary`**. Its runtime admits a
`current` anchor and completes it through the resolver's own
`current_vs_previous`, which resolves to the two adjacent governed snapshots. So:

- v2's `temporal_forms` list excludes `current` for **every** case, which now
  misstates the contract for `material_summary` and `attribution`; and
- v2 declines to gate temporal at all, on a justification that no longer holds.

Either way its pre-registered expectation no longer describes the target
contract.

## Why it was not simply edited

An already-hashed bank is never altered after a semantic policy change. Editing
the temporal expectation in place would let the expectations follow the finding —
which is the exact failure mode pre-registration exists to prevent, and it would
also break the hash that makes the pre-registration meaningful.

So v2 is retired as it stands, with its hash intact and zero calls against it, and
`live_change_form_gate_v3` is a fresh pre-registration.

## What v3 changes, and what it keeps

**Keeps:** all eighteen questions, byte-identical — 4 material_summary,
6 metric_delta over five distinct governed quantities, 4 attribution,
4 level_comparison; the same four scope-bearing cases.

**Changes:** the temporal expectation, which is now **per form and gated**:

| form | `current` | why |
|---|---|---|
| `material_summary` | **valid** | a comparison state is part of what the form is; its runtime completes the anchor |
| `attribution` | **valid** | `bridge` owns its own window, so one anchor denotes two states |
| `metric_delta` | invalid | names one quantity; its owner still requires more than a single point |
| `level_comparison` | invalid | asks for two states the reader has in mind, so the reader names them |

An accepted anchor is reported as `ANCHOR_ACCEPTED`, distinct from
`PAIR_PRESERVED`, so the evidence says which of the two actually happened rather
than collapsing them.

```
V3_BANK_SHA256  2ff59ff12277c66d1814ec496945a338d69fc9ea97098195915c626e65060a98
V3_RUN          NOT RUN — awaiting approval
```
