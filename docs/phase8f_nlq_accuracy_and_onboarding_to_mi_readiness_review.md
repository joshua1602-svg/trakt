# Phase 8F — NLQ accuracy & Onboarding→MI production-readiness review

*Read-only review. No features built, no code/tests/configs modified, no PR
opened, no live Anthropic call made. Evidence is cited to files/tests; the
onboarding inventory was produced by code search and key claims were
spot-verified (file existence and the critical integration-gap claim were
checked directly).*

---

## 1. Executive summary

* **Measured NLQ success rate today:** **100% on the curated synthetic
  benchmarks** — 30/30 golden questions under the deterministic baseline, 45/45
  fake-adapter tests, and 24/24 end-to-end fake NLQ→runtime tests (12 executed +
  12 guard/contract). **145/145** NLQ-related tests pass.
* **What that measures:** the **deterministic baseline** and **fake Anthropic
  outputs** only. It measures *plumbing, governance, and the curated question
  coverage* — **not** a real model's interpretation accuracy.
* **Has live Anthropic actually been tested?** **No.** The Phase 8E harness
  exists and fails safely without a key, but no `ANTHROPIC_API_KEY` is available,
  so **live NLQ accuracy is unknown**.
* **Does the existing Onboarding Agent constitute the "production mapper"?**
  **Largely yes as a mapper.** There is a real, reasonably mature Onboarding
  Agent that detects source files, infers/aligns schema, matches aliases, runs a
  human review workbench, emits a canonical lender/pipeline tape with full
  lineage, persists mapping decisions, and supports multi-artefact consolidation
  and recurring uploads. My earlier "production mapper is not built" framing was
  **too pessimistic.**
* **What remains to connect onboarding into MI snapshots:** the **narrower, real
  gap** is integration/certification. There is **no code path or test** that
  takes onboarding's canonical output, registers it into a `SnapshotStore` with a
  proper `SnapshotHeader`, and runs `run_mi_query` over it. Phase 6B/6C prove the
  snapshot→MI half using **synthetic** frames that explicitly state "no
  onboarding." Plus two policy gaps: forecast fields
  (`forecast_funding_probability`, `forecast_funded_balance`) are not derived by
  onboarding (only passed through if present), and field defaulting happens in the
  downstream Transformation Agent, not onboarding.

**Corrected one-line state:** *The onboarding mapper exists and is fairly mature,
but it is not yet wired or certified into the MI recurring-snapshot ingestion
path, and live LLM interpretation accuracy has not been measured.*

---

## 2. NLQ accuracy review

Commands run (from repo root, `ANTHROPIC_API_KEY` unset):

```
python -m pytest tests/test_phase8a_mi_interpreter_harness.py \
  tests/test_phase8b_anthropic_interpreter_adapter.py \
  tests/test_phase8c_nlq_to_runtime_smoke.py \
  tests/test_phase8e_live_anthropic_dev_smoke.py -q
→ 145 passed
```

Golden dataset shape (`tests/fixtures/mi_interpreter/golden_questions.yaml`,
counted programmatically):

```
total golden: 30
expected_valid: 20
expected invalid/ambiguous: 10
clarification_required: 5
interpreter_supported (deterministic-graded): 25
spec-graded (validation-only): 5
```

### A. Deterministic baseline accuracy
* **Questions:** 30 golden (20 valid; 10 invalid/ambiguous, of which 5 are
  clarification cases and 5 are spec-graded validation failures).
* **Pass/fail:** **30/30 pass** (`test_phase8a…::test_golden_example`); the full
  Phase 8A suite is 67/67.
* **What success means:** each supported question maps to the *expected
  normalised MIQuerySpec fields*, validates as expected, and clarifies exactly
  when the gold says it should (graded by `evaluator.evaluate_interpretation`).
  Source: `mi_agent/interpreter/deterministic.py`, `…/evaluator.py`.
* **Caveat:** this is a *curated, hand-written* keyword interpreter graded
  against its *own* designed question set. 100% here means "the baseline covers
  its catalogue," not "robust to arbitrary phrasings."

### B. Fake Anthropic adapter accuracy
* **Tests:** **45/45 pass** (`tests/test_phase8b_anthropic_interpreter_adapter.py`).
* **What fake outputs prove:** the adapter safely parses model output (markdown
  fences, prose-wrapped JSON, lists, scalars, malformed JSON, code), enforces
  governance (unknown field → warning; unsupported chart type / hallucinated
  field → error → clarification), routes through `normalized()` +
  `validate_query_spec()`, and is safe-by-construction (invalid never `ok`).
  Source: `mi_agent/interpreter/anthropic.py`.
* **What they do NOT prove:** anything about a **real** model's accuracy. The
  "model" is a canned string; these tests measure *our* handling of outputs, not
  Claude's ability to produce correct specs.

### C. End-to-end fake NLQ→runtime accuracy
* **Tests:** **24/24 pass** (`tests/test_phase8c_nlq_to_runtime_smoke.py`).
* **Executed:** **12** questions interpreted → validated → run through
  `run_mi_query` over synthetic Phase 6B snapshots (total funded/pipeline/
  forecast; funded trend; funded compare; by portfolio/region/stage;
  concentration by region; grade/IFRS/PD migration).
* **Blocked:** **8** guard cases (clarification object, ambiguous via
  deterministic, malformed output, invalid enum, hallucinated field, missing
  temporal dates, regulatory+MI state, M&A+pipeline state) + **4** contract
  tests.
* **What was asserted:** specific runtime outputs (e.g. funded total 620;
  trend `[300, 620, 620]` with a line chart instruction; PF_001 400 / PF_002 220;
  B→C `deteriorated`), and that blocked cases yield `executed=False`,
  `runtime_result is None`.
* **Caveat:** fake clients + synthetic data. Proves the **bridge and gates**, not
  model accuracy or real data behaviour.

### D. Live Anthropic accuracy
* **Has the smoke been run?** **No.** No `ANTHROPIC_API_KEY` is available;
  this environment confirms the key is unset.
* **Success rate:** **UNKNOWN.**
* **Command needed:**
  `ANTHROPIC_API_KEY=sk-... python scripts/phase8e_live_anthropic_smoke.py`
* **Where results would be saved:**
  `artifacts/phase8e_live_anthropic_smoke_results.json` (gitignored; may contain
  provider responses). The harness grades each of the 17 controlled questions as
  execute/clarify. Source: `scripts/phase8e_live_anthropic_smoke.py`,
  `docs/phase8e_live_anthropic_dev_smoke_review.md`. Verified: the script returns
  exit 2 without a key and imports no SDK.

### Summary table

| Metric | Dataset | Question/Test count | Passed | Failed | Success rate | Caveat |
|---|---|---|---|---|---|---|
| Deterministic baseline | Golden questions (synthetic, curated) | 30 questions (67 tests) | 30 (67) | 0 | **100%** | Graded against its own catalogue; not arbitrary phrasings |
| Fake adapter | Canned model outputs | 45 tests | 45 | 0 | **100%** | Proves handling/safety, **not** model accuracy |
| E2E fake NLQ→runtime | 12 exec + 12 guard/contract | 24 tests | 24 | 0 | **100%** | Fake clients + synthetic snapshots |
| Live Anthropic | 17 controlled questions | — | — | — | **UNKNOWN (not run)** | No API key; never executed |

---

## 3. NLQ failure-mode review

"Safe" = cannot silently execute a wrong/ambiguous query. Sources:
`anthropic.py`, `runtime_bridge.py`, `mi_spec_validation.py`, `deterministic.py`.

| Failure mode | Current behaviour | Test evidence | Safe? | UX good enough? |
|---|---|---|---|---|
| Malformed JSON | `llm_malformed_json` → clarification; not executed | 8b `test_parse_malformed_json`, 8c `test_malformed_output_not_executed` | ✅ | OK (generic "rephrase") |
| Prose around JSON | Fences stripped; first balanced `{…}` extracted | 8b `test_parse_strips_markdown_fences`, `…extracts_object_from_prose` | ✅ | ✅ |
| Code/pandas/SQL output | `llm_output_contains_code` → clarification | 8b `test_parse_detects_code_output` | ✅ | OK |
| Invalid enum | `invalid_enum_value` (ERROR) → adapter clarifies; not executed | 8b validation, 8c `test_invalid_enum_not_executed` | ✅ | OK |
| Unknown field | `llm_unknown_field` (**WARNING**); dropped by `from_dict`; still executes if otherwise valid | 8b `test_unknown_field_is_warned_but_not_fatal` (anthropic.py:224) | ✅ | ✅ (tolerant by design) |
| Hallucinated dimension | `llm_hallucinated_field` (ERROR) when semantics supplied → clarification | 8b `test_hallucinated_field_*`, 8c `test_hallucinated_field_not_executed` | ✅ (only when semantics passed) | ⚠ depends on caller passing semantics |
| Missing dates (compare/trend) | `temporal_selector_incomplete` → not executed | 8c `test_missing_temporal_dates_not_executed`, 8b compare-without-dates | ✅ | OK |
| Vague "risk" | Clarification (which risk view) | golden 8a, 8c `test_ambiguous_question_via_deterministic_not_executed` | ✅ | ✅ |
| Vague "changes" | Clarification (which period) | golden 8a | ✅ | ✅ |
| Ambiguous "stage" | Clarification; `ambiguous_dimension` if forced into a spec | golden 8a; `mi_spec_validation` | ✅ | ✅ |
| Ambiguous "portfolio" | Clarification unless portfolio config available | golden 8a (`portfolio_config_available: false`) | ✅ | ✅ |
| Ambiguous "rate" | Clarification (metric vs buckets) | golden 8a | ✅ | ✅ |
| Multi-intent question | **No explicit handling.** Deterministic picks first matching branch; a model would pick one intent. Not detected/blocked | none (gap) | ⚠ partial — result is valid but may answer only one intent silently | ❌ not ideal |
| Unsupported metric/dimension | If named field unknown: `ambiguous_dimension` (bare terms) or `llm_hallucinated_field` (with semantics) or flat-mode `virtual_field_not_available`; otherwise grouping on a non-existent column may pass validation | partial (validation + hallucination tests) | ⚠ mostly, but **no semantic existence check unless semantics passed** | ⚠ |

**Net:** the safety posture is strong — every classic injection/ambiguity mode is
gated. Two soft spots: **multi-intent** questions aren't detected (silently
answer one intent), and **field-existence enforcement depends on the caller
passing the semantics registry** to the adapter (the bridge supports it but
callers may omit it).

---

## 4. Onboarding Agent inventory

Verified that all cited files exist. Maturity judgements are from code/test
inventory.

| Component | File(s) | What it does | Evidence/test | Maturity | Caveat |
|---|---|---|---|---|---|
| Source-file detection | `engine/onboarding_agent/document_extractor.py`, `…/onboarding_context.py` | Loads CSV/XLSX; detects file/sheet/column signals, asset class/jurisdiction | `tests/test_onboarding_deterministic_first.py` | Mature | — |
| Schema/field inference | `engine/gate_1_alignment/semantic_alignment.py` | Tokenise + fuzzy match to canonical registry with confidence tiers | onboarding alignment tests | Mature | Conservative thresholds; low-confidence routed to review |
| Alias matching | `engine/gate_1_alignment/aliases/alias_builder.py` + `aliases/*.yaml` | Curated alias sets; exact>alias>fuzzy resolution | `tests/test_onboarding_alias_integration.py` | Mature | — |
| Defaulting | `engine/transformation_agent/transformation_agent.py`, `…/onboarding_agent/target_first_decisions.py` | Default materialisation (TS_DEFAULT etc.) | transformation tests | **Partial** | **Defaults applied downstream (Transformation), not in onboarding output** |
| User questions / workbench | `ui/onboarding_review.py`, `engine/onboarding_agent/review_pack_builder.py`, `agents/onboarding_schemas.py` | HTML review pack; field/enum review items; LLM advisor | `tests/test_onboarding_workbench_smoke.py`, `…review_interpreter.py` | Mature | — |
| Canonical output generation | `engine/onboarding_agent/central_tape_builder.py` | Builds `18_central_lender_tape.csv` + `18a_central_pipeline_tape.csv` (canonical fields, row per loan) | `tests/test_onboarding_deterministic_first.py` | Mature | Output is a CSV tape, **not** a registered snapshot |
| Lineage / mapping artefacts | `central_tape_builder.py` (lineage cols), `05_mapping_candidates.json`, `05c_mapping_trace.json`, `18b_central_tape_lineage.csv` | Per-field source→canonical trace, method, confidence | `…deterministic_first.py::test_value_match_evidence_recorded` | Mature | — |
| Validation / data-quality | `snapshot/model.py` (header/key/segmentation checks), `central_tape_builder.py` gaps, `transformation_agent.py` status codes | Required-field, key, enum, date/numeric checks | snapshot + transformation tests | Mature | DQ split across onboarding/transformation/snapshot |
| Persistence of mapping decisions | `engine/onboarding_agent/mapping_persistence.py`, `12_approved_mapping_overrides.yaml` | Save/reload approved mappings | `tests/test_onboarding_approved_mapping_persistence.py` | Mature | — |
| Multiple files / joins | `central_tape_builder.py` (loan/borrower/collateral domains), `13_source_precedence_rules.yaml` | Consolidate artefacts on `loan_identifier`; conflict/precedence | `tests/helpers/phase6c_consolidation.py` (synthetic) | Mature | The 6C *MI* test synthesises consolidation; it does not call the onboarding engine |
| Recurring uploads | `engine/onboarding_agent/schema_drift.py`, `mapping_persistence.py`, `mode_policy.py` | Detect schema drift; reapply persisted mappings on new periods | `tests/test_onboarding_mapping_memory.py` | Mature | — |

---

## 5. Onboarding-to-MI gap analysis

| Check | Finding | Evidence |
|---|---|---|
| Canonical names compatible with `fields_registry.yaml` & MI semantics? | **Yes.** Onboarding canonicalises via the field registry; MI semantics (`mi_agent/mi_semantics_field_registry.yaml`) is a curated subset, so names align. Onboarding emits a superset (incl. regulatory non-core); no MI-scoping filter. | `central_tape_builder.py` uses `load_field_registry`; registries compared |
| Materialises MI virtual fields (`portfolio_id`, `funded_status`, `pipeline_stage`, `reporting_date`, `cut_off_date`, `forecast_funding_probability`, `forecast_funded_balance`)? | **Partial.** Segmentation (`portfolio_id`/`spv_id`/`acquired_portfolio_id`) and status (`funded_status`/`pipeline_stage`) are mapped if present (else gap issued). `reporting_date`/`cut_off_date` handled at snapshot stamping. **Forecast fields are NOT derived** — only passed through if the source has them. | `snapshot/model.py` reserved columns; `central_tape_builder.py`; agent inventory |
| Preserves `reporting_date` vs `upload_timestamp`? | **Yes, strictly enforced.** `reporting_date` must not default to `upload_timestamp` (ERROR otherwise); dates normalised to ISO and never conflated. | `snapshot/model.py` `HEADER_DATE_FIELDS`/`HEADER_DATETIME_FIELDS`, `MISSING_REPORTING_DATE`; `tests/test_run_context_date.py`, `test_phase2_snapshot_layer.py` |
| Stable `loan_id` / `opportunity_id` keys? | **Yes**, derived at snapshot registration: funded via `select_stable_loan_key`, pipeline via `make_pipeline_opportunity_id`. Onboarding emits `loan_identifier`; snapshot layer derives the rest. | `snapshot/keys.py` |
| Multiple artefacts and joins? | **Yes** in onboarding (domains merged on `loan_identifier`; precedence rules). | `central_tape_builder.py` |
| Lineage sufficient for MI audit? | **Yes** at onboarding output level (`18b_central_tape_lineage.csv`). Whether snapshot registration carries that lineage forward is **untested**. | lineage cols; (no snapshot-lineage test) |
| Can output be registered directly into `LocalFsSnapshotStore`? | **Not demonstrated.** No adapter constructs a `SnapshotHeader` (client_id, route, reporting_date, source_file_id, cadence, upload_timestamp) from onboarding output and calls `register_snapshot`. | search: no onboarding→`register_snapshot` glue |
| Existing test: onboarding output → SnapshotStore → `run_mi_query`? | **No.** Verified directly: no test imports the onboarding engine and runs `run_mi_query`/`register_snapshot`. Phase 6B/6C use **synthetic** frames and their headers/comments explicitly say "no onboarding." | `grep` of `tests/` (see commands below) |
| If not, what exact test is missing? | A certification test that: (1) runs onboarding on a fixture source tape → `18_central_lender_tape.csv`; (2) builds a `SnapshotHeader` from onboarding/run-context metadata; (3) `store.register_snapshot(header, tape_df)`; (4) `run_mi_query(state/temporal/risk, store=store)`; (5) asserts MI results and that `portfolio_id`/`funded_status`/`pipeline_stage` survive the round-trip. | — |

Verification commands run:

```
grep -rl "register_snapshot" tests/ | xargs grep -lE "onboarding|central_tape|central_lender"
→ only phase6b/6c (matches are in comments saying "no onboarding")
grep -rlE "from engine.onboarding_agent|engine\.onboarding" tests/ \
  | xargs grep -lE "run_mi_query|mi_runtime|register_snapshot"
→ NONE
```

---

## 6. Corrected current-state claim

**Classification B — "Onboarding mapper exists, but not yet wired/tested into MI
snapshots."**

The repository contains a genuine, fairly mature Onboarding Agent (detection,
inference, alias matching, human review workbench, canonical tape generation,
lineage, mapping persistence, multi-artefact consolidation, recurring-upload
support). The previous "production mapper is not built" claim was inaccurate.
The accurate, narrower gap is that onboarding's canonical output has **not been
connected, tested, or certified** as the recurring MI snapshot ingestion path:
there is no registration adapter and no onboarding→snapshot→`run_mi_query` test.

*Two B-internal caveats worth flagging (they do not downgrade to D, because
onboarding itself — including recurring uploads — is mature): forecast fields are
not derived by onboarding, and field defaulting occurs in the downstream
Transformation Agent rather than in the onboarding tape.*

---

## 7. Recommended next step

**Build an onboarding-to-MI certification test (plus the thin registration
adapter it exercises).**

This is the right move because the evidence shows two *separately* proven halves —
a mature onboarding mapper and a working snapshot→MI runtime — with an
**untested seam** between them. The cheapest, highest-value action is to certify
that seam: drive a fixture source tape through the real onboarding engine,
register the resulting canonical tape into `LocalFsSnapshotStore` via a small
adapter, and run `run_mi_query` over it. This converts "we believe these connect"
into "we have proven they connect," and will surface the concrete policy
decisions (forecast-field derivation, MI-scoping filter, lineage carry-forward,
where `SnapshotHeader` metadata comes from) before any cloud/UI investment.

Deferred until that passes: live-Anthropic accuracy measurement (independent
workstream, do when a key is available), prompt hardening (only if live accuracy
is poor), and the Azure snapshot adapter (only once the local onboarding→MI path
is certified).

*No implementation started.*

---

## 8. Output

* **Measured NLQ success rates by evidence type:**
  * Deterministic baseline: **100%** (30/30 golden; 67/67 suite).
  * Fake adapter: **100%** (45/45) — handling/safety only.
  * E2E fake NLQ→runtime: **100%** (24/24; 12 executed, 12 guard/contract).
  * Live Anthropic: **UNKNOWN** — not run.
* **Actual status of live Anthropic testing:** **Not run.** Harness exists and
  fails safely without a key; no key available in this environment.
* **Actual status of Onboarding Agent mapping capability:** **Exists and is
  fairly mature** — detection, inference, alias matching, review workbench,
  canonical tape, lineage, mapping persistence, multi-artefact consolidation, and
  recurring uploads are all present with tests. Gaps: forecast-field derivation
  and onboarding-stage defaulting.
* **Exact missing proof before claiming production MI ingestion:** a test (and
  small adapter) proving **onboarding canonical output → `SnapshotHeader` +
  `register_snapshot` → `run_mi_query`** with MI fields surviving the round-trip.
  It does not exist today.
* **Recommended next phase:** **Phase 9 — Onboarding→MI snapshot ingestion
  certification.** Drive a fixture tape through the real onboarding engine,
  register it into the snapshot store via a thin adapter, and run the MI runtime
  over it, asserting state/temporal/risk results and field round-trip. This
  certifies the one unproven seam between two already-working halves, is low-cost
  and read-mostly, and de-risks every later production decision (Azure, UI, live
  LLM) by forcing the outstanding ingestion-policy choices into the open first.
