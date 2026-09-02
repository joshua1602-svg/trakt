# Pipeline stage-transition engine

Sprint: governed pipeline stage-transition capability — **engine only**.
Branch: `claude/pipeline-stage-transition-engine-2ewble`
Starting `main`: `e7678c8` (*Merge pull request #382 — trakt pre-go-live audit*).

---

## 1. Executive verdict

**YES.** The governed stage-transition capability is implemented correctly.

Every governed case identifier in the two-snapshot window is classified exactly
once as a new arrival, a stayer, a stage transition or a departure. Per-stage
case-count reconciliation is exact (residual `0` at every stage), per-stage
amount reconciliation is exact (residual `0.00` at every stage, against a
declared `0.01` floating-point tolerance), and the global identifier
reconciliation is exact (`14 = 4 prior-only + 8 both + 2 latest-only`, `residual
0`).

The work is additive. Two production files changed, **645 insertions and zero
deletions**. Every governed pipeline and forecast output on the existing
five-week pack is byte-identical to the starting `main` SHA.

---

## 2. Starting architecture — the seam that was extended

The pre-flight confirmed the previously identified `movement_components` seam is
**still the correct production owner**, and it was extended in place. No second
snapshot-matching engine, no second pipeline register, no new case identity.

| Question | Answer (verified in this tree, not inferred from prior reports) |
|---|---|
| 1. Who loads/receives prior and latest snapshots | `mi_agent_api.movement_detail.resolve_movement_detail` → `pipeline_contract.weekly_extract_inventory` + `pipeline_contract.load_prepared_pipeline`; the neighbour pair is chosen by `movement_detail.select_pair` |
| 2. Who joins them on `pipeline_case_identifier` | `movement_detail._case_level` (one row per case) then an outer join inside `movement_detail.movement_components` (`movement_detail.py:168`) |
| 3. Prior/current stage columns after the join | `_stage` (latest) and `_stage_prior`, from the prepared `pipeline_stage` column (`movement_detail.STAGE`) |
| 4. Prior/current amount fields | `_measure` and `_measure_prior`, from `current_outstanding_balance` (`movement_detail.MEASURE`) — the measure whose sum IS `pipeline_amount` |
| 5. ID population handling | outer join: latest-only → `new`, prior-only → `removed`, both → increased/decreased/unchanged/progressed_out. Duplicates were **summed** and reported; unkeyed rows excluded and counted by `_unkeyed` |
| 6. Stage canonicalisation source | `mi_agent_api.pipeline_prep.canonical_stage` over the single `_STAGE_CANON` map → `KFI / APPLICATION / OFFER / COMPLETED / WITHDRAWN / UNKNOWN` |
| 7. Live/terminal semantics | `movement_detail.TERMINAL_STAGES = ("COMPLETED", "WITHDRAWN")`; `pipeline_prep.ACTIVE_STAGES = ("KFI","APPLICATION","OFFER")`; `_OPEN_PIPELINE_STAGES` drives `pipeline_status` |

**What was missing.** The seam publishes a NET decomposition only. An OFFER stock
falling 3 → 1 is a net −2, equally consistent with 2 out / 0 in, 4 out / 2 in, or
3 out / 1 in. The seam held prior and current stage on every joined case but
never published the source→destination pair.

**One deviation from the seam's existing convention, deliberate.** For duplicate
identifiers the net decomposition sums and reports. The transition capability
**refuses** instead: summing two cases into one would silently corrupt a gross
count. Both behaviours now coexist; neither changed the other.

---

## 3. Event model

Four classes, mutually exclusive and collectively exhaustive, in
`movement_detail.stage_transition_events`:

| Class | Rule | Published fields |
|---|---|---|
| `new_arrival` | ID absent from prior, present in latest | `destination_stage`, `latest_amount`. `source_stage` is `None` — the engine never pretends it moved from a real stage |
| `stayer` | ID in both, `prior_stage == latest_stage` | stage, `prior_amount`, `latest_amount`, `amount_change`, count |
| `stage_transition` | ID in both, `prior_stage != latest_stage` | `source_stage`, `destination_stage`, `prior_amount`, `latest_amount`, `amount_change`, count |
| `departure` | ID in prior, absent from latest | `source_stage`, `governed_outcome`, `outcome_evidence`, `prior_amount`, count |

---

## 4. Identifier — stable `pipeline_case_identifier` ownership

`movement_detail.CASE_KEY = "pipeline_case_identifier"`, the governed natural key
already declared `role: identifier` in `config/mi/pipeline_field_contract.yaml`.
Matching goes through the existing `_case_level`, so the transition capability
and the net decomposition cannot key differently.

No fallback identity exists. Loan amount, row number, a hash of mutable fields
and the funded loan identifier are all unused for identity — proved by
`test_matching_uses_the_governed_case_identifier` (row order reversed between
snapshots, classification unchanged) and
`test_no_fallback_identity_is_invented_for_an_unkeyed_row`.

**Amount changes never change identity.** A case going KFI £500k → APPLICATION
£520k stays ONE case with ONE transition and a +£20k amendment — never a
departure plus an arrival
(`test_an_amendment_on_a_transitioning_case_stays_one_case`).

**Missing identifier** — mirrors the severity the preparation layer already
applies. The `missing_case_identifier` BLOCKER fires only when the column is
absent or blank for every row, so:

* column absent, or blank for every row in a non-empty snapshot →
  `available: false`, `reason_code: missing_case_identifier`;
* partially blank → reported in `methodology.unmatched_current` /
  `unmatched_comparison` and excluded from the population, never guessed;
* an *empty* snapshot beside a populated one is **not** an identifier defect —
  everything departed is a real answer, and refusing it would withhold one;
* neither snapshot carrying a matchable case → `reason_code: no_governed_cases`.

**Duplicate identifiers** → `available: false`,
`reason_code: duplicate_case_identifiers`, with the per-snapshot counts in
`methodology.duplicate_case_identifiers`. No arbitrary deduplication.

---

## 5. Stage model — existing canonical vocabulary reused

No second stage list exists. Every stage the capability publishes is a token
produced by `pipeline_prep.canonical_stage`, asserted directly:
`test_the_stage_vocabulary_is_the_one_the_pipeline_views_use` round-trips every
stage in the payload through `canonical_stage` and requires it to be unchanged.

Ordering is **derived, not restated**. One additive helper,
`pipeline_prep.canonical_stage_order()`, sorts the existing `_STAGE_BUCKET` map
by funnel position — the same map `canonical_stage` normalises onto, and the same
derivation `question_interpretation.lexical.canonical_pipeline_stages()` already
uses. A product that adds a stage gets the ordering with no caller edited.
`KFI`, `APPLICATION`, `OFFER`, `COMPLETED` are not hard-coded by this sprint;
they are read from the governed map.

Terminal semantics reuse `movement_detail.TERMINAL_STAGES` — `COMPLETED` and
`WITHDRAWN`. Note that the governed model already folds *declined* / *rejected* /
*cancelled* / *lapsed* onto `WITHDRAWN` (`_STAGE_CANON`); no "decline" outcome
was invented.

---

## 6. Structured output — the new capability contract

`detail_type: "PIPELINE_STAGE_TRANSITION"`. Engine entry points:

| Function | Role |
|---|---|
| `stage_transition_events(current, prior)` | per-case classified frame — the single source every aggregation groups |
| `transition_matrix` / `new_arrival_summary` / `stayer_summary` / `departure_summary` / `event_totals` | deterministic aggregations |
| `stage_reconciliation` / `global_reconciliation` | the published proofs |
| `build_stage_transition_detail(...)` | the governed payload |
| `resolve_stage_transition_detail(root, client_id, ...)` | production resolution from the governed extract inventory |
| `stage_transition_unavailable(...)` | the controlled "no detail" envelope |

```
detail_type, portfolio_id, scope, run_id, as_of_date, comparison_date,
available, reason, reason_code,
identifier, measure, stage_field,
counts { current, comparison, change }

transitions[]   { source_stage, destination_stage, case_count,
                  prior_amount, latest_amount, amount_change }
new_arrivals[]  { destination_stage, case_count, latest_amount }
stayers[]       { stage, case_count, prior_amount, latest_amount, amount_change }
departures[]    { source_stage, governed_outcome, outcome_evidence,
                  case_count, prior_amount }
event_totals    { new_arrival|stayer|stage_transition|departure:
                  { case_count, prior_amount, latest_amount } }

reconciliation {
  by_stage[] { stage,
               opening_case_count, new_arrivals, transitions_in,
               transitions_out, departures, stayers, closing_case_count,
               count_reconciliation_residual,
               opening_amount, new_arrival_amount,
               transferred_in_latest_amount, transferred_out_prior_amount,
               departure_prior_amount, stayer_amount_change, closing_amount,
               amount_reconciliation_residual },
  count_reconciliation_residual, amount_reconciliation_residual,
  global { prior_population, latest_population, union_population,
           prior_only, in_both, latest_only,
           classified_events, duplicate_classifications, residual },
  amount_tolerance, count_identity, amount_identity }

methodology { capability_definition, movement_basis: "gross", identity_basis,
              identity_note, stage_vocabulary, terminal_stages,
              departure_outcome_basis, version,
              unmatched_current, unmatched_comparison,
              duplicate_case_identifiers }
source_dates { pipeline_as_of, pipeline_comparison, funded_as_of,
               forecast_generated_at }
sources { current, comparison }
```

Following the module's existing conventions: same flat envelope shape as the
movement payload, the same keys in the available and unavailable cases, amounts
rounded to 2dp, and — as with the movement payload — **no case identifiers and no
loan-level rows** (`test_the_payload_carries_no_case_identifiers`). Explicit
event rows were deliberately NOT added to the payload for that reason; the
per-case frame is available to callers through `stage_transition_events`, exactly
as `movement_components` already is.

---

## 7. Reconciliation

Fixture: `tests/fixtures/pipeline_transition_2w`, prior `2026-06-05` (12 cases)
vs latest `2026-06-12` (10 cases), resolved through the production path.

### Count identity — exact integer equality

`opening + new arrivals + transitions in − transitions out − departures = closing`

```
stage         open  +arr   +in  -out  -dep  =close  resid
KFI              4     1     0     2     0       3      0
APPLICATION      4     1     2     2     1       4      0
OFFER            2     0     2     1     1       2      0
COMPLETED        1     0     1     0     1       1      0
WITHDRAWN        1     0     0     0     1       0      0
```

`count_reconciliation_residual = 0` at every stage and in total.

### Amount identity

```
opening_amount + new_arrival_amount + transferred_in_latest_amount
              − transferred_out_prior_amount − departure_prior_amount
              + stayer_amount_change = closing_amount
```

Each side of a transition carries the amount it actually had **in that
snapshot**: the source stage releases the case's `prior_amount`, the destination
receives its `latest_amount`. An amendment on a moving case therefore travels
with the case to its destination, which is why no separate transition-amendment
term is needed. This is the mathematically correct identity for the actual
prior/latest amount definitions, and it is published in the payload as
`reconciliation.amount_identity`.

```
stage           opening  +arrivals        +in       -out  -departed   +amend   =closing  resid
KFI           1,200,000    900,000          0    900,000          0   20,000  1,220,000   0.00
APPLICATION   2,900,000    150,000    920,000  1,300,000  1,300,000  -20,000  1,350,000   0.00
OFFER         2,000,000          0  1,290,000    800,000  1,200,000        0  1,290,000   0.00
COMPLETED     1,000,000          0    800,000          0  1,000,000        0    800,000   0.00
WITHDRAWN     1,100,000          0          0          0  1,100,000        0          0   0.00
```

`amount_reconciliation_residual = 0.00` at every stage, against a declared
tolerance of `0.01`. No residual is hidden — both residuals are payload fields,
computed from the identity, so a real break surfaces as a number rather than
vanishing.

Opening and closing are additionally proved to be the snapshots' **real** stage
stocks, not totals the capability invented for itself
(`test_the_per_stage_opening_and_closing_are_the_real_stage_stocks` compares them
against `value_counts()` on the prepared frames).

### Global reconciliation

```
prior_population 12   latest_population 10   union_population 14
prior_only 4  +  in_both 8  +  latest_only 2  =  14
classified_events 14   duplicate_classifications 0   residual 0
```

Event classes partition the union exactly:
`new_arrival 2 + stayer 3 + stage_transition 5 + departure 4 = 14`.
No case disappears; no case appears in two classes.

---

## 8. Example transition matrix

Prior `2026-06-05` → latest `2026-06-12`:

| Source | Destination | Cases | Prior £ | Latest £ | Change £ |
|---|---|---:|---:|---:|---:|
| KFI | APPLICATION | 2 | 900,000 | 920,000 | +20,000 |
| APPLICATION | OFFER | 2 | 1,300,000 | 1,290,000 | −10,000 |
| OFFER | COMPLETED | 1 | 800,000 | 800,000 | 0 |

New arrivals: KFI 1 case £900,000; APPLICATION 1 case £150,000.
Stayers: KFI 2 cases £300,000 → £320,000 (+£20,000); APPLICATION 1 case
£300,000 → £280,000 (−£20,000).

Departures:

| Source | Governed outcome | Evidence | Cases | Prior £ |
|---|---|---|---:|---:|
| APPLICATION | `unclassified_departure` | `none` | 1 | 1,300,000 |
| OFFER | `unclassified_departure` | `none` | 1 | 1,200,000 |
| COMPLETED | `COMPLETED` | `prior_terminal_stage` | 1 | 1,000,000 |
| WITHDRAWN | `WITHDRAWN` | `prior_terminal_stage` | 1 | 1,100,000 |

**The departure rule.** A departure's outcome is the governed terminal stage the
case was **last recorded at**. Absence from the latest extract is never treated
as evidence of an outcome, so cases 3013 (last seen at OFFER) and 3014 (last seen
at APPLICATION) stay `unclassified_departure` with `outcome_evidence: none`
rather than being called withdrawals. The distinction is a first-class payload
field, not a footnote.

APPLICATION is the case a net figure cannot describe: 4 opening, 4 closing —
net zero — while gross shows 2 transitions in, 1 new arrival, 2 transitions out,
1 departure and 1 stayer.

---

## 9. Existing-output parity

Every governed pipeline and forecast output was dumped on the five-week pack at
the starting `main` SHA and at HEAD and compared byte-for-byte:

* `pipeline_evolution` (all five periods, all metrics)
* `pipeline_funnel_evolution` (stage series, flow series, conversions)
* per-extract prep reports: `total_pipeline_amount`,
  `weighted_expected_funded_amount`, `stage_counts`, `row_count`, `data_quality`
* per-extract stock: case count, pipeline amount, per-stage stock, weighted
  expected funded amount

**Result: identical — 19,240 bytes, `sha256 6d168ff7…`.**

This is also enforced as a permanent test. `TestExistingOutputsUnchanged`
computes each governed output, exercises the new capability against the **same**
prepared frames, recomputes, and requires equality — covering live pipeline
amount and case count, pipeline stage stock, weighted expected pipeline, pipeline
evolution, funnel/conversion, and the existing net movement decomposition. It
also asserts via `pd.testing.assert_frame_equal` that the capability does not
mutate the prepared frames, so no downstream number can depend on whether the
capability was called.

The new capability owns no existing metric and is a distinct `detail_type`; it
does not re-label or replace `PIPELINE_WEEKLY_MOVEMENT` or
`COMPLETIONS_WEEKLY_MOVEMENT`.

---

## 10. Test results

`mi_agent_api/tests/test_pipeline_stage_transition.py` — **57 passed, 0 failed**.

| Group | Tests | Covers |
|---|---:|---|
| `TestIdentity` | 10 | key ownership, amendment ≠ identity, duplicate refusal (both snapshots), missing/blank identifier, no fallback identity, empty-snapshot handling, no-comparison |
| `TestClassification` | 9 | latest-only → arrival, both/same → stayer, both/different → transition, prior-only → departure, evidence-based terminal outcomes, unclassified never guessed, exactly-one-class |
| `TestFixtureTruth` | 14 | KFI→APPLICATION count/amount, APPLICATION→OFFER count/amount, terminal transition, aggregation, funnel ordering, arrivals, departures, amendments up/down/in-flight |
| `TestReconciliation` | 8 | per-stage count residual `0`, per-stage amount within tolerance, opening/closing = real stage stock, global exactness, class partition, matrix totals, residual visibility |
| `TestProductionReachability` | 6 | governed inventory, immediate-neighbour rule, earliest-snapshot refusal, canonical stage vocabulary, a **second** governed pack, no identifier leakage |
| `TestExistingOutputsUnchanged` | 8 | the six non-regression requirements plus frame immutability and distinct detail type |
| `TestGrossIsNotNet` | 2 | the brief's worked 3→1 example; `movement_basis: "gross"` |

### Production reachability

The capability is exercised through `resolve_stage_transition_detail`, which uses
the same `weekly_extract_inventory`, the same `load_prepared_pipeline` and the
same `select_pair` neighbour rule as production MI. It is proved against **two**
independently built governed packs — the new two-snapshot pack and the existing
five-week pack — so it is not tuned to one fixture. On the five-week pack it
correctly finds `KFI→OFFER 1` and `APPLICATION→OFFER 1` for week 4 → week 5, with
all residuals zero. There is no test-only computation path.

---

## 11. Regression — baseline vs HEAD

Baseline is the starting `main` SHA `e7678c8`, run in the same environment.

| Scope | Baseline (`e7678c8`) | HEAD | New failures |
|---|---|---|---|
| Targeted (pipeline prep, stock, evolution, movement, forecast, funnel/conversion, stage contract, serving cache, source selection, runtime materialisation) | 199 passed, 0 failed | 256 passed, 0 failed | **0** |
| Broad `mi_agent_api/tests` (75 files) | 1319 passed, 3 failed, 1 skipped | 1376 passed, 3 failed, 1 skipped | **0** |
| Repo-level pipeline surface (22 files) | 271 passed, 26 failed, 51 skipped | 271 passed, 26 failed, 51 skipped | **0** |

Pass counts rise by exactly the 57 new tests. The failure sets were diffed, not
just counted, and are **identical** at baseline and HEAD:

* `mi_agent_api/tests` — 3 pre-existing failures in `test_chat_routing_e2e`,
  `test_currency_authority`, `test_single_parse_and_substitution`;
* repo-level — 26 pre-existing failures, `diff` reports the sets as identical.

None are touched by this sprint and none were repaired, per the brief.

**Environment note.** Test dependencies were absent from the container and were
installed to run anything at all (`pandas` pinned to the `<3.0.0` range
`requirements.txt` declares, plus `numpy`, `PyYAML`, `plotly`, `matplotlib`,
`openpyxl`, `python-pptx`, `scikit-learn`, `Jinja2`, `rapidfuzz`, `fastapi`,
`uvicorn`, `httpx`, `pytest`). A broken Debian `cryptography` binding initially
failed 12 API tests **on `main` as well**; repairing the environment cleared all
12 on both sides. No repository file was changed for any of this, and both
baseline and HEAD were measured in the identical repaired environment.

---

## 12. Sprint 2 recommendation — presentation (NOT implemented)

Nothing in React, PPTX or deck configuration was written, and none should be
written against anything other than this payload.

**Service exposure.** The current architecture does **not** require every
analytical capability to be reachable through the service layer — the engine is
importable and tested directly — so no API route was added. Recommended for
Sprint 2: a single read-only route mirroring the existing movement-detail route
shape, e.g. `GET /mi/pipeline/stage-transitions?portfolioId=&asOf=`, delegating
straight to `resolve_stage_transition_detail` with no recomputation, and gated by
the existing `TRAKT_MI_ENHANCED_HOVERS`-style flag mechanism rather than a new
one.

**Consumption rules for React and PPTX:**

1. Render `transitions[]` directly as a Sankey or matrix. Do not recompute,
   re-aggregate across stages, or net arrivals against departures — the whole
   point is that they are separate.
2. Draw `new_arrivals[]` from a synthetic `NEW` node **in the visual only**. The
   engine's `source_stage: null` must survive in any data the deck exports.
3. Draw `departures[]` to a synthetic `EXIT` node, and render
   `governed_outcome: "unclassified_departure"` with visibly different treatment
   from `COMPLETED` / `WITHDRAWN`. Never label it a withdrawal.
4. Show `stayers[]` as self-loops or omit them from the flow diagram, but keep
   `amount_change` visible — it is a real amendment, not noise.
5. Respect `available: false`: render the governed `reason` and nothing else.
   An empty matrix must never be drawn as "nothing moved".
6. Surface the residuals. If `count_reconciliation_residual` or
   `amount_reconciliation_residual` is non-zero, the visual must say so rather
   than plotting a diagram that does not add up.
7. Sequence stages by `pipeline_prep.canonical_stage_order()`; the payload is
   already in that order.

---

## 13. Sprint 3 readiness

**YES.** The capability is sufficiently defined and tested for the existing
stage-movement Query question bank to route against it.

* The result is deterministic, stable and reconciled, with a fixed detail type
  and a frozen field contract.
* Gross source→destination counts and amounts are directly answerable, as are
  arrivals, stayers, departures and per-stage opening/closing.
* Unavailability is governed and typed (`no_prior_snapshot`,
  `missing_case_identifier`, `duplicate_case_identifiers`, `no_governed_cases`),
  so a router can refuse in the existing pattern instead of inventing an answer.
* Stage tokens are the same canonical vocabulary the question layer already
  reads, so no new vocabulary is required.

No Query file was modified in this sprint: `llm_query_parser.py`,
`chat_routing.py`, the recogniser registry, `mi_query_spec.py`, the Query
adapters, the Query executor and the stage-movement question banks are all
untouched.

---

## Return schedule

| # | Item | Answer |
|---|---|---|
| 1 | Starting `main` SHA | `e7678c8` |
| 2 | Branch | `claude/pipeline-stage-transition-engine-2ewble` |
| 3 | Commits | one, on top of `e7678c8` |
| 4 | Production seam extended | `mi_agent_api/movement_detail.py` — the `_case_level` + `movement_components` join, `select_pair`, `load_prepared_pipeline` |
| 5 | New capability | `build_stage_transition_detail` / `resolve_stage_transition_detail` / `stage_transition_events`, `detail_type = PIPELINE_STAGE_TRANSITION` |
| 6 | Structured output fields | see §6 |
| 7 | Stable identifier | `pipeline_case_identifier` — no fallback |
| 8 | Stage vocabulary owner | `pipeline_prep.canonical_stage` / `_STAGE_CANON`; order derived by the additive `canonical_stage_order()` |
| 9 | New arrivals | latest-only; `source_stage` stays `null`, never a synthetic stage |
| 10 | Stayers | both snapshots, same stage; amendment recorded, still one case |
| 11 | Transitions | both snapshots, different stage; source, destination, both amounts, change |
| 12 | Departures | prior-only; outcome = prior terminal stage where evidenced, else `unclassified_departure` with `outcome_evidence: none` |
| 13 | Amount amendments | attribute of a case, never evidence of identity; never split into departure + arrival |
| 14 | KFI→APPLICATION fixture truth | 2 cases, £900,000 → £920,000, +£20,000 |
| 15 | APPLICATION→OFFER fixture truth | 2 cases, £1,300,000 → £1,290,000, −£10,000 |
| 16 | Terminal fixture truth | OFFER→COMPLETED 1 case £800,000, change 0; plus departures with governed `COMPLETED` and `WITHDRAWN` outcomes |
| 17 | Per-stage count reconciliation | residual `0` at all five stages (§7) |
| 18 | Per-stage amount reconciliation | residual `0.00` at all five stages, tolerance `0.01` (§7) |
| 19 | Global event reconciliation | `4 + 8 + 2 = 14`, classified 14, residual `0` |
| 20 | Duplicate-ID behaviour | refuse — `available: false`, `duplicate_case_identifiers`, counts reported, no deduplication |
| 21 | Missing-ID behaviour | absent/all-blank → `available: false`, `missing_case_identifier`; partial → reported and excluded; neither snapshot keyed → `no_governed_cases` |
| 22 | Pipeline stock unchanged | **confirmed** — byte-identical, plus a permanent test |
| 23 | Pipeline evolution unchanged | **confirmed** — byte-identical, plus a permanent test |
| 24 | Forecast unchanged | **confirmed** — weighted expected funded amount and prep reports byte-identical |
| 25 | Targeted tests | 57 new pass; targeted suite 199 → 256 passed, 0 failed |
| 26 | Broad regression | `mi_agent_api/tests` 1319→1376 passed, same 3 pre-existing failures; repo-level pipeline surface identical (271 passed / 26 pre-existing failures) |
| 27 | Production files changed | `mi_agent_api/movement_detail.py`, `mi_agent_api/pipeline_prep.py` — 645 insertions, 0 deletions |
| 28 | React modified | **NO** |
| 29 | PPTX modified | **NO** |
| 30 | MI Query modified | **NO** |
| 31 | Report path | `docs/reports/pipeline_stage_transition_engine.md` |
| 32 | Merge recommendation | **MERGE.** Additive, zero deletions, zero new failures, existing governed outputs byte-identical, all reconciliations exact |

### Also not modified

PPTX, MI Query Agent, OCC, onboarding, transformation, validation, projection,
Annex 2, canonical schema, field registry, pipeline-stock semantics, forecast
methodology, concentration, cohort, funded analytics, existing Query
parsing/routing, existing presentation code. No surrounding pipeline
architecture was refactored: the implementation stayed inside the seam plus one
derived-ordering helper in the stage vocabulary owner, and never spread.
