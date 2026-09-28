# Governance reconciliation + the current-anchor temporal contract

```
PRODUCT_BASELINE  665176864d2f5dbceedc277e351c5d2cd978777f
REPAIR_HEAD       d3d22c6a6db73bdf4328bd9ac2a6e883ed390f17  (completeness repair)
LIVE_MODEL_CALLS  0     DEPLOYMENTS 0     LIVE_MI_QUERY_CALLS 0
```

The completeness repair is **retained unchanged in principle**: a change-oriented
intent that requires `change_form` and does not state it still never becomes an
executable plan, and the compiler still never manufactures the missing form.

## A — governance reconciliation

### A1. All 19, classified

| category | count |
|---|---|
| 1. SUPERSEDED_CONTRACT_TEST | **18** |
| 2. HISTORICAL_REPLAY_INVARIANCE | **1** |
| 3. GENUINE_UNEXPECTED_REGRESSION | **0** |

Every one of the 19 was proved to share a single cause before anything was
touched: neutralising only `requires_change_form` made all 19 pass.

**Category 1 splits by treatment, not by cause.** One test asserts the superseded
behaviour *as its subject*; seventeen merely *use* an authored fixture whose
premise it was, while testing something orthogonal.

| test | old assertion | new behaviour | treatment |
|---|---|---|---|
| `test_change_form_contract::test_a_plan_with_no_form_records_no_form_binding` | a movement + pair + measure + no form → PLAN | CLARIFY / `MISSING_REQUIRED_SLOT` `change_form` | assertion migrated; original purpose re-pinned on a point-in-time request that legitimately has no form |
| `test_contract_normalisation` × 8 | same, as a precondition for testing label/period/owner normalisation | no plan | authored fixture completed: `metric_delta` for the generic measure, `attribution` where the measure is `funded_balance_movement` (owned by `funded_bridge`) |
| `test_corrections_from_the_first_live_run::test_a_period_on_period_movement_needs_no_comparison_flag` | as above | no plan | fixture completed (`metric_delta`) |
| `test_kl_multi_output_and_specialists::…[funded_bridge-bridge-…]` | a bridge is representable → PLAN | no plan | fixture completed (`attribution`) for the one capability that is a change form |
| `test_slice3_portfolio_affordance::test_proof_7` | movement under an acquired lens → PLAN | no plan | fixture completed (`metric_delta`) |
| `test_plan_temporal_runtime` × 6 | `compare`/`movement` payloads → PLAN, then snapshot assertions | no plan | the change-oriented payloads state their form |
| `test_insight_funded::…[current]`¹ | `current` → `PERIOD_NOT_A_PAIR` | admitted as an anchor | migrated (see B) |

¹ surfaced by B, not by the completeness repair; same category.

**Why completing an authored fixture is not relaxation.** These are hand-written
structured inputs, not recorded model output. A period-on-period movement of one
named measure *is* a metric delta; stating it makes the fixture valid under the
current contract without touching what the test measures. The recorded corpus is
treated completely differently — see A2.

One genuine subtlety found while doing it: stating `metric_delta` on a fixture
whose measure is `funded_balance_movement` makes normalise rule 4 record a
form/measure conflict on every pass, which surfaced a **pre-existing
non-idempotency in that rule**. Rather than absorb it, those call sites state
`attribution`, whose capability *is* `funded_bridge` — `applied` stays at 2 and
nothing is perturbed. The rule-4 behaviour is noted and left alone as out of scope.

### A2. The historical evidence is untouched

`run8_135_signoff_2b00172.json` is unchanged. No recorded payload gained a
`change_form` it never had — and
`test_the_signed_off_evidence_itself_is_never_rewritten` now asserts that, so a
future attempt to forge it fails.

What changed is the **guard's expectation**, and only for ids pinned by name:

```
EXPECTED_MOVED_CASES = 10   Q18A Q18B Q18C Q19A Q19B Q19C Q20A Q20B Q20C NL1B
ACTUAL_MOVED_CASES   = 10   the same ten
UNEXPECTED_MOVES     = 0
```

The migration guard is deliberately narrow. It fails if a case moves that is not
on the list; if a case **on** the list does *not* move; if a move lands anywhere
other than `CLARIFY`; or if the reason is anything other than
`MISSING_REQUIRED_SLOT` on `change_form`. The other 125 are still compared on
outcome **and** plan identity, so a disturbance to parsing or binding anywhere
else still fails here. It is not a blanket "output may change".

Nine are the change-intelligence family the Sprint B evidence already recorded as
predating the slot. The tenth, **NL1B** — *"are we originating different types of
loans now compared with a few months ago?"* (`period_movement` / `compare` /
`relative_pair`, periods_back 3) — is a genuine period-over-period comparison whose
analytical form is genuinely unstated, so asking is the correct outcome.

### A3 / CI

```
TARGET_STATE_CI = reconciled to the original baseline, exactly

  tests/interpretation_v2/ + mi_agent_api/tests/   21 failed, 2385 passed
  original baseline                               21 failed, 2314 passed
  identical 12 FAILED ids; +71 passes from the new controls

  mi_agent/tests/                                 16 FAILED ids
  like-for-like baseline                          16 FAILED ids, identical set
```

Zero new failures and zero disappeared. All 16 remaining MI failures were checked
individually: none mentions `change_form`, `MISSING_REQUIRED_SLOT`,
`PERIOD_NOT_A_PAIR` or `current_vs_previous` — they are the pre-existing legacy
parser, receipt-disclosure and shadow-wiring failures.

## B — the current-anchor temporal contract

### B1. The period owner already existed

```
PERIOD_REQUEST_TYPE               mi_agent.period_change.periods.PeriodRequest
CURRENT_VS_PREVIOUS_REPRESENTATION METHOD_CURRENT_VS_PREVIOUS = "current_vs_previous"
                                   (mi_agent/period_change/models.py:301)
PERIOD_RESOLVER_OWNER             period_change.periods.resolve_periods / _resolve_pair
SNAPSHOT_SELECTOR_OWNER           period_change_route.build_snapshots (catalogue)
CURRENT_VS_PREVIOUS_ALREADY_EXISTED = YES
```

`_resolve_pair` returns `ordered[-2], ordered[-1]` for that method — the two
adjacent governed snapshots, **with no calendar arithmetic at all**. Proved by
execution: three snapshots → FROM `2026-05-31 (s2)`, TO `2026-06-30 (s3)`,
neither end adjusted. One snapshot → refuses (*"needs two governed portfolio
snapshots"*) rather than inventing a date.

No new period framework was created.

### B2. material_summary at a current anchor

```
MATERIAL_SUMMARY_CURRENT_ANCHOR_SUPPORTED = YES
MATERIAL_SUMMARY_FROM_TO_RECEIPTED        = YES
```

`current` is admitted as an **anchor**, in its own set:

```python
PAIR_PERIOD_FORMS      # forms that denote two snapshots on their own
ANCHOR_PERIOD_FORMS    # {"current"} — one endpoint, completed by THIS form
ADMITTED_PERIOD_FORMS  # the union
```

It is deliberately **not** added to `PAIR_PERIOD_FORMS`: `current` still does not
mean two snapshots, and keeping the two sets apart is what keeps the provenance
honest. The intent is never rewritten — `plan.period.form` stays `current` — and
the receipt records both halves separately:

```
interpreted_period_form   "current"          <- the reader's reading, unrewritten
period_completed_by_form  true               <- the owner supplied the other end
period_resolution         resolution_method  "current_vs_previous"
                          FROM  s2 / 2026-05-31
                          TO    s3 / 2026-06-30
```

The Direct/Acquired scope survives a current anchor (the v1 CF05 case exactly).

**Nothing else was broadened.** `metric_delta` at a bare `current` still REFUSES;
`level_comparison` at a bare `current` is still refused by its own owner
(`PERIOD_NOT_TEMPORAL`); `series` and `forward_looking` are still not anchors; an
untranslatable grain is still refused. `material_summary` gained the anchor because
a comparison state is part of what *that* form is — not as a general relaxation.

### B3. attribution

```
ATTRIBUTION_TEMPORAL_CALCULATION_EXISTS = YES
ATTRIBUTION_PLAN_CONNECTIVITY_EXISTS    = NO
ATTRIBUTION_CURRENT_VS_PREVIOUS_SUPPORTED = YES at the calculation owner,
                                            NOT reachable through a governed plan
```

`period_change.workflow` takes a `PeriodRequest` and `include_bridge`, calls the
same `resolve_periods` (workflow.py:172) and computes `balance_bridge(start, end)`
from the pair it returns (workflow.py:279). So the bridge already accepts
`current_vs_previous` — it shares the resolver with every other period-change
figure.

What does not exist is a governed-plan runtime for `funded_bridge`:
`plan_temporal_runtime` covers `{point_in_time, breakdown, series, compare}` and
`plan_material_summary` covers `{summary}`. So `attribution + current` compiles
with `owned_by_capability=True` and nothing routes it.

**This is plan connectivity, not missing temporal behaviour.** Per the brief the
adapter was *not* built here. Recorded as a bounded connectivity item for the
serving sprint.

### B5 / B6

24 structured controls in
`tests/interpretation_v2/test_current_anchor_temporal_contract.py`, plus the
migrated B10 control. They pin: the anchor resolving to two governed snapshots;
explicit pairs preserved (`relative_pair` → `current_vs_previous`, with a grain →
`month_on_month`; explicit labels passed through); an unavailable comparison state
refusing without inventing a date; the other three forms unbroadened; attribution
only to the level B3 proved; the receipt carrying FROM/TO; and — structurally —
that the period decision has no `question` parameter, performs no attribute or key
read of one, and imports no regex.

```
RAW_QUESTION_ROUTING_ADDED      = NO
CHANGE_FORM_INFERRED_BY_COMPILER = NO
NEW_TEMPORAL_SLOT_OR_ENUM       = NO
```

Temporal normalisation does not infer `change_form` (a `current`-anchored bridge
with no form still does not reach a plan), and the change-form rules invent no
dates. The two responsibilities stay separate.

## v2 → v3

```
V2_BANK_RUN                = NO
V2_EXPECTATIONS_STILL_VALID = NO
V3_BANK_SHA256             = 2ff59ff12277c66d1814ec496945a338d69fc9ea97098195915c626e65060a98
V3_RUN                     = NO
```

v2's justification for not gating temporal is now false for `material_summary`, and
its `temporal_forms` list excludes `current` for every case. An already-hashed bank
is never edited after a semantic policy change, so v2 is retired unrun with its
hash intact and v3 is a fresh pre-registration: the same eighteen questions,
byte-identical, with the temporal expectation corrected to be **per form and
gated** — `current` valid for `material_summary` and `attribution`, invalid for
`metric_delta` and `level_comparison`. Proved to discriminate before any call.
