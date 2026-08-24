# Conversion 5 — `temporal_compare` — pre-registered stop conditions

**Committed before any production change.**

Base: `e9e05b7`. Working tree clean at the time of writing.

Every number below is transcribed verbatim from
`docs/mi_c5_cost_regime_rebaseline.md` §7, which was committed at `e9e05b7`
before this conversion was authorised. Nothing here was chosen with knowledge of
what C5 turned out to need.

---

## The experiment

Conversions 1–4 budgeted, in turn, from architecture (C1), generalisation (C2),
conversion number (C3) and axis count (C4). The first three predictors were
falsified by measurement. The re-baseline replaced them with **which generic
capabilities the candidate still needs**, enumerated at field level.

Conversion 5 is the first conversion budgeted from that model, so it is the
first that can falsify it. The hypothesis:

> A route requiring known, pre-enumerated generic capabilities can be migrated
> within a budget derived from those capabilities, without discovering another
> hidden semantic layer.

C5 is **not** testing whether routes get cheaper. It is testing whether the
required shared work can be **predicted before the conversion starts**.

## Thresholds

| # | condition | threshold |
|---|---|---|
| **S1** | **shared** production lines | **≤ 75** |
| **S2** | **route-specific** production lines | **90–150** |
| **S3** | **total** production lines | **≤ 225** |

### Where 75 comes from

Bottom-up from measured analogues, not from the previous conversion's total:

| component | analogue | measured | estimate |
|---|---|---|---|
| `dataset` accessor | `comparison_period` (guarded field read) | 20 | **20** |
| measure accessor (`subject` + `operation`) | `grouping_concepts` | 31 for one axis | **28** for both as one accessor |
| section documentation | every prior conversion's banner | 14 | **14** |
| | | | **62 predicted** |
| justified margin | one unforeseen guard or state case | | **+13** |
| | | | **≤ 75** |

**This is higher than C4's measured 65, deliberately.** The model says shared
cost tracks how many generic capabilities the candidate needs, and
`temporal_compare` needs three where `funded_bridge` needed two. Setting it
below 65 would be budgeting from conversion number again — the exact error the
re-baseline exists to correct.

### Where 90–150 comes from

Observed route-specific: C1 176, C2 144, C3 129, C4 154. `_route_compare` is
**63 lines**, the smallest handler yet converted (geo 94 → 129; bridge 111 →
154). The **floor of 90 matters as much as the ceiling**: far below it would
mean a shallower conversion than C1–C4 received, making the comparison
meaningless in the other direction.

### Where 225 comes from

Derived (75 + 150). Predicted landing zone: ~62 shared + ~115 route-specific
≈ **180**.

## S4–S9, carried forward

| # | condition |
|---|---|
| S4 | a **new primitive** is required |
| S5 | economics breach the **A2** tolerance (£0.005) |
| S6 | payload/receipt equivalence needs a **bespoke `temporal_compare` exception** |
| S7 | any **silent drop** or **silent population widening** |
| S8 | any **unexplained regression**, by exact case/test name |
| S9 | a generic semantic dependency appears that the re-baseline did not enumerate |

## Measurement method

Unchanged since the first re-baseline: *is this production code reachable from
more than one route at the end of Conversion 5?* → **shared** /
**route-specific** / **product hardening** / **cleanup**. Tests, instruments and
docs are not production-line cost.

`cleanup` remains a **floor**, not a measurement: a net base→head diff cannot see
logic added and then removed inside one conversion.

## Verdicts, defined now

**REGIME MODEL SUPPORTED** — shared ≤ 75; route-specific 90–150; total ≤ 225;
only the pre-enumerated generic semantic work was required; no new primitive;
economics equivalent; payload and receipt equivalent; regressions clean.
*Means:* field-level dependency inspection is now a useful predictor of
conversion cost.

**REGIME MODEL INCOMPLETE** — technically clean, but unexpected generic semantic
work appears, or shared exceeds 75, or total exceeds 225 without architectural
redesign. **Do not proceed automatically to C6.**

**MIGRATION ECONOMICS FALSIFIED** — cost materially exceeds the model;
substantial hidden generic architecture appears; or C5 requires redesign rather
than bounded accessors. Stop and re-evaluate before C6.

---

## Conditions carried in

* Preserve the known safe-refusal cases, existing portfolio-scope semantics, the
  known Back Book label collision, C4 bridge behaviour, and C1–C4 outputs.
* The route's **source scope is not honoured today** — `_route_compare` never
  passes `scope` to `run_temporal_compare`. C5 must not start honouring it;
  that would be broadening capability, not converting a route.
* If C5 exposes a live wrong-number defect, it is **STOP — LIVE PRODUCT
  DEFECT**, fixed at its owner in its own task, not buried in migration work.

**Do not optimise the implementation to pass the threshold. Let the measured
result answer the question.**
