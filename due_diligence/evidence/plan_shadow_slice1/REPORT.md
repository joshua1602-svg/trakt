# Slice 1, Gate 5 — the recorded corpus, replayed through the shadow

Harness: `due_diligence/evidence/plan_shadow_slice1/corpus_replay.py`
Output: `corpus_replay.json`, `shadow_ledger.jsonl` (73 rows)
Subject: `mi_agent/plan_runtime_adapter.py`
`LIVE_OPUS_CALLS = 0`  `MI_BEARER_USED = NO`  `DEPLOYED = NO`

---

## 1. The 843-question corpus is not present at HEAD

The brief asked for the 843-question corpus and, before anything else, for a
determination of whether it can be replayed from already-recorded outputs. It
cannot be replayed, because it is not here:

* the 843/882-question corpus exists only on the quarantined deterministic
  branch (`3399d2ef`), which this slice is not authorised to merge, cherry-pick
  or read into the product;
* what is at HEAD is seven **complete** recorded benchmark runs under
  `mi_agent/interpretation_v2/evidence/`, each holding 135 interpretations with
  the model's raw payload, the compiled `GovernedQueryPlan` and its `plan_id`.

So the substitute corpus is those recordings. Stated precisely, because the
numbers are not interchangeable:

| | |
|---|---|
| distinct questions | **135** (45 canonical × 3 paraphrases) |
| recorded interpretations replayed | **945** (135 × 7 complete runs) |
| distinct questions eligible in at least one run | **18** |
| distinct eligible `plan_id`s | **24** |
| live model calls to obtain any of it | **0** |

945 replayed interpretations is not 843 distinct questions, and this report does
not present it as such. The 135-question bank is drawn from the same four
sources as the larger corpus (including
`due_diligence/evidence/analytical_intent_v1/nl_bank.py`), but turning the
remaining questions into plans needs live interpretation, which the brief
requires be stopped for and authorised first. It was not done.

`run5_PARTIAL_60of135_api_balance_exhausted.json` is excluded: its unattempted
questions carry no plan and would have counted as `NOT_A_PLAN`, misreporting the
census rather than widening it.

## 2. Is the replayed plan the plan that was signed off?

A replay proves nothing if it quietly recompiles to a different plan. Stage 1
parses each recorded raw payload and compiles it through the real
`DeterministicCompiler`, then compares the resulting `plan_id` with the recorded
one. For the signed-off run (`run8_135_signoff_2b00172.json`):

```
PLAN_ID_IDENTICAL    126
NO_PLAN_BOTH_TIMES     9     (8 CLARIFY + 1 REFUSE, reproduced as non-plans)
PLAN_ID_DIVERGED       0
REPLAY_FAILED          0
```

Every plan replays to the same content hash the signed-off run recorded, so the
plans executed below are those plans and no interpretation was bought again.

Older runs are **not** held to that and read `NOT_CHECKED`: the normal form
changed between them (see `contract_normalisation_report.md`), so a divergence
there would be a recorded change, not a defect, and asserting it would
manufacture a false failure.

## 3. Eligibility census

Signed-off run, 135 records:

```
TOTAL                     135
SLICE1_ELIGIBLE            17
NOT_ELIGIBLE              118
    CAPABILITY_NOT_GENERIC  91    specialist capability, owns its own arithmetic
    EXPLICIT_LENS           12    Direct/Acquired, resolved by portfolio_lens
    NOT_A_PLAN               9    CLARIFY or REFUSE — there is no plan to run
    GEOGRAPHY_REQUESTED      5    a governed region axis this slice cannot bind
    OPERATION_NOT_GENERIC    1    `distribution`, owned by period_change
```

Aggregate over all seven complete runs:

```
TOTAL                     945
SLICE1_ELIGIBLE           115
NOT_ELIGIBLE              830
    CAPABILITY_NOT_GENERIC 585
    NOT_A_PLAN             111
    EXPLICIT_LENS           75
    GEOGRAPHY_REQUESTED     31
    OPERATION_NOT_GENERIC   28
```

**This supersedes the preliminary figure of 31 I recorded while deriving the
allow-list.** That count applied the capability, operation, period and
dimension tests only; it had not yet applied the lens, comparison or geography
tests. The measured figure is 17 of 135 for the signed-off run.

17 of 135 is 13% coverage, and that is the honest shape of slice 1: the bank is
dominated by specialist capabilities (`period_movement`,
`pipeline_stage_movement`, `forecast`, `borrowing_base`, `limit_assessment`),
which is what one would expect of an MI bank and what makes a generic-only slice
narrow. Widening the allow-list to raise the number would be the defect.

## 4. Execution and parity

The 17 eligible plans of the signed-off run; the same shape holds per run and in
aggregate.

```
EXACT_SEMANTIC_PARITY       9     figure matches the independent control
PRESENTATION_ONLY           2     grouped: no scalar to compare (see below)
OUTSIDE_REPLAY_FIXTURE      6     the replay book does not carry the field
NUMERICAL_DIFFERENCE        0
SHADOW_EXECUTION_ERROR      0
DISPOSITION_DIFFERENCE      0
OLD/NEW_PATH_SEMANTIC_DIFF  0
```

Aggregate over 945: **58 `EXACT_SEMANTIC_PARITY`, 15 `PRESENTATION_ONLY`, 42
`OUTSIDE_REPLAY_FIXTURE`, and zero differences, errors or disposition
divergences of any kind.** 73 ledger rows were written.

**The grouped half is compared cell for cell, not waved through.** A grouped
execution has no scalar, so `classify` correctly declines to compare one and
says `PRESENTATION_ONLY`. The harness then executes the adapter's own spec and
checks every cell against the control:

```
GROUPED_CELL_PARITY        15     every cell within 0.01, groups identical
GROUPED_CELL_DIFFERENCE     0
```

### The control is an independent oracle, not the old path

The recorded runs hold interpretation outputs only — no legacy answer was ever
stored beside them — so an old-path-versus-new-path figure comparison is not
available from recorded evidence, and manufacturing one would need live calls on
the legacy route. The control here is arithmetic written out longhand in the
harness over a deterministic 400-row book, importing nothing from the product.
Every ledger row records `control_route: independent_control_oracle` so no
reader can mistake it for the legacy path. That is a *stronger* control than the
old path for the figures and a *weaker* one for disposition; both are reported
as what they are, and the old-versus-new disposition comparison remains
unmeasured by this gate.

One normalisation is applied and disclosed: the plan carries the governed
canonical value (`erm_product_type == 'drawdown'`) where the book carries the
display form (`'Drawdown'`), so strings compare case-insensitively after
trimming, and no product alias table is consulted.

### Fixture limits, stated as limits

6 of the 17 (`Q11A/B/C`, `Q13A/B/C`) reference `ticket_bucket` and
`interest_rate_bucket`. Both are governed registry fields
(`mi_semantics_field_registry.yaml:3786` and `:1899`); they are simply absent
from the independent replay book, which was built before it was known which
plans would be eligible. Those six are **untested here, not broken**. They are
pre-filtered rather than executed, because an executor failure caused by a
missing fixture column would be a false defect in the ledger. Extending the book
to cover them would be tuning the fixture to the corpus, which this run is not
permitted to do.

## 5. What the replay found

Three silent-loss paths in the new adapter, all three fixed in `91d50603`:

1. **A geography axis was dropped, not refused.** Five eligible plans
   (`Q14A/B/C`, `Q16A`, `Q16C`) carry a geography binding with `group_by: True`
   on `canonical_region_reporting`. The adapter read neither the binding nor the
   axis, grouped by the plan's other dimension alone, and would have reported
   that as the breakdown the reader authorised. This is the exact failure class
   the control plane exists to prevent, and it was in the new code. A geography
   binding is now an ineligibility.
2. **Two predicates on one field silently collapsed.** `MIQuerySpec.filters` is
   keyed by field, so an LTV band stated as two bounds would have reached the
   executor as `< 80` with the `> 50` gone — a wider population than the plan
   authorised. No plan in the corpus does this, so the corpus alone could not
   have caught it; it was found by reading the bind against the wire format.
   Refused now, and a band stated once as a `between` still executes, as a list.
3. **The ledger's requested half could not state what was asked.** Filters were
   recorded as `{field: value}`, losing the direction that the resolved half
   keeps in `applied_predicates.op`. `age > 55` and `age == 55` were
   indistinguishable on the requested side, which would have made a direction
   divergence unadjudicable from the row. Now a list of
   `{field, comparator, value}`.

And one harness defect of my own, recorded because the engine was right and I
was wrong: the case-folding guard tested `column.dtype == object`, which pandas
reports as `str` for this book, so it never fired and the control read 0 rows
where the engine correctly read 93.

### One observation for the interpretation workstream, not fixed here

`Q12A` and `Q12B` compile to `operation: breakdown` over
`[ltv_bucket, age_bucket]` and are eligible; `Q12C` — "Plot portfolio balance
across LTV buckets and borrower-age buckets" — compiles the same two-axis
request to `operation: distribution` and is therefore `OPERATION_NOT_GENERIC`.
The slice is not paraphrase-invariant at the operation boundary. That is a
property of the interpretation, not of the adapter, and the brief forbids
touching interpretation in this slice, so it is recorded and left alone.

## 6. Privacy

The ledger holds aggregates, field names, group keys and row counts. Checked on
the committed file: **0 of the book's 400 loan identifiers appear anywhere in
it**, and no borrower field does either.
