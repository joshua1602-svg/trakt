# Temporal presence repair + v4 pre-registration

```
PRODUCT_BASELINE  665176864d2f5dbceedc277e351c5d2cd978777f
PRODUCT_HEAD      0ee52863
LIVE_MODEL_CALLS  0     DEPLOYMENTS 0     LIVE_MI_QUERY_CALLS 0
```

`change_form` is accepted: v3 measured **18/18** live. Nothing about the four-way
analytical form was reopened.

## Root cause

`SemanticTime.form` carries a **construction** default of `current`, applied in the
PARSER — `intent._parse_time`, once for an absent `time` block and again for a block
with no `form` key. So three v3 readings that correctly named their analytical form
and deliberately stated no period arrived at the governed boundary
indistinguishable from a reader asking about the current state. All three said so
outright: *"no period is stated on the intent because the funded_bridge capability
defines the movement period itself."*

Two of them then compiled to **correct plans by coincidence** — the default landed
on the one value `attribution` accepts. That coincidence was the defect.

    A CONSTRUCTION DEFAULT IS NOT A BUSINESS DEFAULT.

## Phase 1 — what existed, and why the stop condition did not trigger

```
EXISTING_TEMPORAL_PRESENCE_SIGNAL  NONE
    IntentProvenance carries question / model_id / versions / usage only;
    parse_candidate_intent recorded no key presence; SemanticTime had no flag.

EXISTING_PROVENANCE_OWNER          the compiler's binding-level `defaulted` pattern
    GeographyBinding.defaulted + default_reason, MeasureBinding.statistic_defaulted,
    stripped from plan identity by GovernedQueryPlan._DERIVATION_KEYS.
    THIS IS WHAT WAS REUSED.

CURRENT_DEFAULT_APPLICATION_POINT  intent._parse_time
    line 1: `if raw is None: return SemanticTime()`
    line 2: `form=_enum(raw.get("form", "current"), ...)`
    i.e. the construction layer, which had become the business owner.
```

**The stop condition did not trigger.** Presence is available as a structural fact —
`"form" in raw` — so nothing rereads natural language, a source span or a phrase.

## The repair

**Presence metadata, not a new semantic slot.** `SemanticTime.stated` is provenance:
absent from `SemanticTime.key()` and added to `_DERIVATION_KEYS`, so it cannot change
a plan's identity or make two readings of one question look divergent. Asserted, not
claimed — an absent-time plan and an explicit-`current` plan have the **same
`plan_id`**.

**One authorised-defaults table**, each entry derived from an owner contract that
already exists, never from wording and never from the measure:

| form | absent time → | derived from |
|---|---|---|
| `material_summary` | `current_vs_previous` | its runtime already completes a `current` anchor through that method |
| `attribution` | `current_vs_previous` | its calculation owner shares `resolve_periods` with every other period-change figure |
| `metric_delta` | **no default** | its owner requires a period richer than a single point |
| `level_comparison` | **no default** | its owner wants the pair named |

Every governed form has an explicit entry, so a future form cannot inherit a default
by omission — asserted.

**A bug the matrix caught.** `normalise` rebuilds `SemanticTime` in both
relative-period rewrites and was dropping `stated`, so a normalised **explicit**
pair looked unstated and the deterministic layer applied a default over the reader's
own words. A rewrite changes the spelling of a stated form; it cannot unstate it.
Fixed in both branches.

### The three states, measured

```
case                        outcome  stated  defaulted  method               owner
material_summary ABSENT     PLAN     False   True       current_vs_previous  material_summary
material_summary @current   PLAN     True    False      -                    -
material_summary @rel_pair  PLAN     True    False      -                    -
attribution      ABSENT     PLAN     False   True       current_vs_previous  attribution
attribution      @current   PLAN     True    False      -                    -
metric_delta     ABSENT     REFUSE   -       -          -                    -
metric_delta     @rel_pair  PLAN     True    False      -                    -
level_comparison ABSENT     PLAN     False   False      -                    -   (refused by its owner)
level_comparison @rel_pair  PLAN     True    False      -                    -
```

The receipt now carries `interpreted_time_present`, `interpreted_period_form` (null
when unstated), `temporal_default_applied`, `temporal_default_method` and
`temporal_default_owner`, beside the resolver's own method and its FROM/TO
snapshots. The absent and explicit-`current` routes execute the **same pair** and
remain **separable** in evidence — the requirement stated exactly.

```
PRODUCT: 6 files, +164 / -6
MODEL_CHANGED = NO   (no prompt, tool description, example, synonym, term or retry)
NEW_TEMPORAL_SEMANTIC_SLOT_ADDED = NO
RAW_QUESTION_ROUTING_ADDED = NO
```

## Regression

| gate | result |
|---|---|
| B10 | **51 / 51** |
| B11 | **20 / 20** |
| Weekly Brief | **90 / 90** |
| period_change | **336 / 336** |
| completeness + current-anchor | **69 / 69** |
| new presence controls | **30 / 30** |
| `tests/interpretation_v2/` + `mi_agent_api/tests/` | 12 FAILED ids — identical set, +101 passes |
| `mi_agent/tests/` | 16 FAILED ids — identical set |

No new failures.

## v3 evidence policy

v3 is **unchanged** — manifest, hash, result and scoring all untouched, confirmed by
diff. Its two authoring defects are recorded here and were **not** used to justify
any product change:

- **V06** — the question said *"live funded loans"*. "Live" is a restricting
  adjective, so a governed `account_status` predicate was the reader's filter, not an
  invention. v3's blanket `filters: []` was wrong.
- **V13** — the question asked which *components* explain the balance.
  `bridge_component` is a governed `funded_bridge` concept and arguably the better
  reading. v3's expectation was too narrow.

**V10 and V14 remain the clean residual evidence** for the defect this repair
addresses.

## v4 — pre-registered, NOT run

```
BANK_ID      live_change_form_gate_v4
BANK_SHA256  26345762e04537246183d3218b82f787669c844c157ad4f1df9f1d0ee4762dce
QUESTIONS    17   (material_summary 4, metric_delta 5, attribution 4, level_comparison 4)
RUN          NOT RUN — awaiting approval
```

**Seventeen fresh paraphrases; not one of v3's questions is reused** — verified
programmatically, not asserted.

It separates the three temporal routes by pinning, per case, which its wording
actually supports:

```
EXPLICIT_PAIR    11 cases   the wording names two reporting states
EXPLICIT_ANCHOR   2 cases   the wording names only the current endpoint
OWNER_DEFAULT     4 cases   no temporal relation stated, and the form owns the window
```

A form that owns a default accepts **all three** routes — a model that states a
period for "summarise the material changes" is not wrong — so the gate scores
route *validity* and reports which route each case took, rather than preferring one.
`metric_delta` and `level_comparison` accept only a stated pair.

**Both v3 authoring defects corrected and exercised**, not just declared:

- `filters_permitted` is per case. **W06** — *"the number of loans in arrears"* —
  permits governed arrears predicates, so the corrected rule is tested.
- `ATTRIBUTION_TARGET` accepts the decomposed quantity **or** `bridge_component`.
  **W12** — *"break the funded balance movement into its components"* — exercises it.

Harness proved at zero cost, in both directions:

```
material_summary ABSENT            PASS  OWNER_DEFAULT_ABSENT
metric_delta     ABSENT            FAIL  class C   (owns no default)
metric_delta     @current          FAIL  class C
level_comparison ABSENT            FAIL  class C
attribution      ABSENT + bridge_component  PASS
metric_delta     + permitted arrears filter PASS
metric_delta     + invented LTV filter      FAIL  class E
```

All four bank manifests — v1, v2, v3, v4 — still hash to their own pins. v1 remains
immutable historical evidence (17/18); v2 remains immutable and unrun.
