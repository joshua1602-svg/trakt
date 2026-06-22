# Spine Audit — Single Source of Truth

**Status:** Architecture/spine audit. Read-only. No new MI features, no UI redesign,
no new analytics. This document maps the end-to-end MI/funded data path and
identifies every place the system **duplicates responsibility** or **bypasses the
intended source of truth**.

**Date:** 2026-06-21
**Branch:** `claude/spine-architecture-audit-rflfhz`
**Scope:** the production MI/funded path — `engine/onboarding_agent/` →
`mi_agent_api/` → `mi_agent/` → `frontend/mi-agent-ui/`. The legacy
`analytics/` Streamlit ERM app is in scope **only** as a parallel/legacy path to
name and ring-fence.

> Companion doc: `docs/mi_analytics_architecture_current_state_audit.md` covers
> *target architecture* (states/routes/risk monitor). This doc is narrower: it
> audits the **single source of truth** for the live data spine and the specific
> recent regressions.

---

## 0. Headline findings

1. **Yes — the system has parallel pipelines.** There are three distinct
   parallelisms, each a duplicated source of truth:
   - **Prep:** `mi_agent_api/funded_prep.py` (production, React/API) vs
     `analytics/mi_prep.py` (legacy Streamlit ERM). Different numeric parser,
     different bucketing, different LTV/age derivation.
   - **Numeric/format rendering:** Python `mi_chart_factory.format_percent` and
     `adapters._format_kpi_value` vs React `lib/utils.ts::formatValue` /
     `ChartArtifactView::valueFormatter`. Percentage scale is computed in one
     place and re-decided (differently) in three others.
   - **Transport:** `HttpAgentClient` (production) vs `MockAgentClient` +
     `data/mockArtifacts.ts` (demo) — selected at build time; **the app defaults
     to the mock** when `VITE_AGENT_API_URL` is unset.
2. **Production vs legacy/demo split:**
   - **Production:** onboarding gates → `18_central_lender_tape.csv` →
     `funded_prep.prepare_funded_mi_dataset` → `mi_query_executor` →
     `adapters` → `HttpAgentClient` → React artifact renderers.
   - **Legacy:** `analytics/streamlit_app_erm.py` + `analytics/mi_prep.py` +
     `analytics/charts_plotly.py` (ERM dashboard; still imported and runnable).
   - **Demo:** `MockAgentClient` + `mockArtifacts.ts` + `mockResponses.ts`.
   - **Offline diagnostic:** `mi_agent_api/scripts/generate_dashboard_review.py`
     (its own `_num` parser, see Stage 12).
3. **Every recent bug in the brief maps to a single root pattern:** *the same
   responsibility implemented twice and allowed to drift* — header detection,
   numeric parsing, percent scaling, chart-key derivation. Each is cited to the
   exact stage below and summarised in §15.

---

## 1. Stage map (one screen)

| # | Stage | Primary module (production) | Source-of-truth status | Risk |
|---|---|---|---|---|
| 1 | Raw client files | input dirs; fixtures in `synthetic_*`, `demo/` | n/a | low |
| 2 | Source discovery / classification | `engine/onboarding_agent/file_classifier.py` | single | low |
| 3 | Header / table detection | `engine/onboarding_agent/source_table_loader.py::redetect_header` | **duplicated readers; partial bypass** | **high** |
| 4 | Field mapping / alias resolution | `mapping_proposer.py` + `engine/gate_1_alignment/semantic_alignment.py` | single (multiple readers feed it) | medium |
| 5 | Period eligibility | `source_period_eligibility.py` + `central_tape_builder.py` | single (config-driven) | medium |
| 6 | Entity-key resolution / joins | `entity_key_resolver.py` | single | medium |
| 7 | Central lender tape promotion | `central_tape_builder.py` | single (consumes Gate-4 selections) | medium |
| 8 | Funded MI preparation | `mi_agent_api/funded_prep.py` | **duplicated by `analytics/mi_prep.py`** | **high** |
| 9 | MI field/dimension/metric contract | `mi_agent/mi_semantics_field_registry.yaml` | single (generated) | low |
| 10 | MI query parsing | `mi_agent/interpreter/deterministic.py` + `llm_query_parser.py` | single validator, two parsers | low |
| 11 | MI query execution | `mi_agent/mi_query_executor.py` | single | medium |
| 12 | API response / artifact schema | `mi_agent_api/adapters.py`, `app.py` | **format logic duplicated; review generator parallel** | **high** |
| 13 | React rendering / display formatting | `frontend/mi-agent-ui/src/components/artifacts/*`, `lib/utils.ts` | **percent/format duplicated; mock parallel** | **high** |

---

## Stage 1 — Raw client files

- **Responsible files:** raw client extracts enter as `.csv` / `.xlsx` / `.xls`
  (and `.pdf`/`.docx` for documents) under an input directory per client/run.
  Fixtures: `synthetic_onboarding_pack*/`, `synthetic_demo/`, `demo/`,
  `demo_static_6_loans.csv`, `spanish_raw_sample.csv`.
- **Inputs:** none (the boundary).
- **Outputs:** files on disk discovered by Stage 2.
- **Source of truth:** n/a.
- **Parallel/bypass paths:** real client packs vs the `synthetic_*`/`demo*`
  fixtures used by tests and the demo. These are clearly labelled fixtures, not a
  shadow production path.
- **Tests:** fixtures are consumed throughout `tests/` and
  `mi_agent_api/tests/`.
- **Known bugs:** the structural trap is *row-2 headers* in client extracts
  (e.g. "PropertyExtract" puts the real header on row 2). The data is fine; the
  failure is downstream readers — see Stage 3.
- **Recommended consolidation:** none for this stage.
- **Risk:** low.

---

## Stage 2 — Source discovery / file classification

- **Responsible files:** `engine/onboarding_agent/file_classifier.py`
  — `classify_directory()`, `classify_file()`, `_detect_file_type()`,
  `_read_structured_headers()`. Produces `01_file_inventory.json`
  (classification + confidence per file: `current_loan_report`,
  `collateral_report`, `pipeline_report`, `cashflow_report`, …).
- **Inputs:** raw files (Stage 1).
- **Outputs:** `01_file_inventory.json` (classification, row/column counts, sheet).
- **Source of truth:** single. Classification is computed once here and cached in
  the inventory; later stages read it for context hints (this is delegation, not
  duplication).
- **Parallel/bypass paths:** documents (PDF/DOCX) are classified **by filename
  only** — never read for content. Low impact (documents are metadata-only at
  this stage) but a misclassification risk.
- **Tests:** `tests/test_onboarding_mapping_quality.py` (header-related).
- **Known bugs:** `_read_structured_headers()` reads with a plain
  `pd.read_csv` / `pd.ExcelFile.parse` and **does not** call `redetect_header`.
  A row-2-header file therefore reports `Unnamed:*` columns and wrong
  row/column counts in the inventory. Same root cause as Stage 3.
- **Recommended consolidation:** route this reader through the same
  header-redetecting loader used in Stage 3 (see Stage 3 action).
- **Risk:** low–medium (inventory counts only; mapping happens later).

---

## Stage 3 — Header / table detection  ⚠️ HIGH

This is the first place a single responsibility is implemented many times.

- **Intended source of truth:**
  `engine/onboarding_agent/source_table_loader.py::redetect_header(df)` — the
  deterministic re-detector that, when >40% of columns are `Unnamed:*`, scans the
  first ~15 rows and promotes the best-scoring row to the header.
- **Inputs:** a raw frame (or file path) per source/sheet.
- **Outputs:** a frame with the real header; coverage report
  (`03_source_table_coverage.csv`).

### Every place a CSV/Excel is read (the duplication)

| Reader | Calls `redetect_header`? | Path |
|---|---|---|
| `source_table_loader.load_source_tables` | ✅ yes | canonical loader |
| `central_tape_builder._read_df` (`central_tape_builder.py:152–182`) | ✅ yes (`:178–179`) | **promotion enrichment** |
| `file_classifier._read_structured_headers` | ❌ no | Stage 2 inventory |
| `file_profiler` (`stream_dataframes`) | ❌ no | Stage 4 column profiling |
| `onboarding_orchestrator` structured-frame cache | ❌ no | internal cache |
| `llm_assisted_mapping` | ❌ no | optional LLM mapping |
| `compare_semantic_alignment` | ❌ no | debug tool |

- **Known bug (named in brief):** *"One loader handled row-2 headers, but another
  promotion/enrichment loader previously bypassed that logic."* This is exactly
  `central_tape_builder._read_df`: it previously did a plain `pd.read_csv`/
  `pd.read_excel`, so enrichment frames (PropertyExtract) came back as
  `Unnamed:*` and **the LoanExtract↔PropertyExtract join silently failed**.
  Fixed in commit `5af5074` (Finding 2) by reusing `redetect_header`
  (`central_tape_builder.py:173–181`). **The fix is local to `_read_df`;** the
  other four readers above are still bypassing the detector.
- **Tests:** `tests/test_onboarding_mapping_quality.py` (redetect row);
  `mi_agent_api/tests/test_funded_central_tape.py` (added in `5af5074`,
  promotion reads row-2 PropertyExtract headers).
- **Recommended consolidation:** make `redetect_header` the **only** entry point
  for reading a source table. Provide one `read_source_frame(path, sheet)` helper
  (it already exists in spirit as `_read_df`) and route `file_classifier`,
  `file_profiler`, `onboarding_orchestrator`, `llm_assisted_mapping`, and
  `compare_semantic_alignment` through it. Better still: read each source frame
  **once** after Stage 3 and pass the cached, header-corrected frame to
  profiling/mapping/promotion so the file is never re-read with different logic.
- **Risk:** **high** — divergent header handling is the documented cause of the
  mi_2025_11 LTV miss (Stage 8) and any "column looks absent" enrichment failure.

---

## Stage 4 — Field mapping / alias resolution

- **Responsible files:**
  - `engine/onboarding_agent/mapping_proposer.py` (`MappingProposer.propose_mappings`)
    → `05_mapping_candidates.json`.
  - `engine/gate_1_alignment/semantic_alignment.py` (`HeaderMapper`,
    `tokenize`, `normalise_name`) — the matching engine.
  - `engine/gate_1_alignment/aliases/alias_builder.py` — alias registry.
  - `engine/onboarding_agent/file_profiler.py` → `02_column_profiles.json`
    (types, null-rate, `likely_identifier`).
  - MI enrichment scope: `config/system/onboarding_agent.yaml`
    `central_lender_tape.mi_enrichment_fields` (`:288–309`).
- **Inputs:** header-detected frames (Stage 3), field registry
  (`config/system/fields_registry.yaml`), alias registry.
- **Outputs:** `05_mapping_candidates.json`, `02_column_profiles.json`,
  `03_candidate_keys.csv`.
- **Source of truth:** single matching engine (`HeaderMapper`). The authoritative
  *decisions* are `12_approved_mapping_overrides.yaml` +
  `13_source_precedence_rules.yaml` consumed by promotion (Stage 7).
- **Duplicated logic / smells:**
  - Header normalisation is reimplemented in several modules:
    `semantic_alignment.normalise_name`, `entity_key_resolver` norm helpers,
    `central_tape_builder._norm`, `mapping_proposer` context hints. They mostly
    agree (lowercase + collapse non-alphanumeric) but are independent copies that
    can drift.
  - Mapping operates on frames from readers that may **not** have run
    `redetect_header` (Stage 3), so a row-2 file can produce low-confidence or
    missing candidates.
- **Parallel/bypass paths:** `llm_assisted_mapping.py` can augment/override the
  deterministic candidates (optional, gated by config) and reads data without
  header redetection.
- **Tests:** `tests/test_onboarding_mapping_quality.py`, plus orchestrator
  integration tests.
- **Recommended consolidation:** (a) one shared `normalise_header()` used by
  alignment, entity-key, and promotion; (b) feed mapping only header-corrected
  frames from the Stage 3 cache.
- **Risk:** medium.

---

## Stage 5 — Period eligibility

- **Responsible files:** `engine/onboarding_agent/source_period_eligibility.py`
  (`compute_eligibility`) and the row-filter/universe-selection logic inside
  `central_tape_builder.py`. Config: `config/system/onboarding_agent.yaml`
  `source_period_eligibility` block + `expected_balance_checks` (`:258–271`).
- **Inputs:** `01_file_inventory.json`, file frames, run period (e.g.
  `mi_2025_10`), config.
- **Outputs:** `04c_source_period_eligibility.{csv,json}`; per-row period filter
  that defines the funded universe; `18f_central_universe_debug.json`.
- **Source of truth:** single, config-driven. Reporting period is resolved by a
  documented precedence (operator override → in-data period column → filename
  date → folder → profiler date).
- **Concrete behaviour (and the mi_2025_10 vs _11 setup):** a cumulative
  current-book file is **row-filtered** by the run period. Config encodes the
  expectation: `mi_2025_10` → 33 loans / ~£4.2m; `mi_2025_11` → 73 loans / ~£8.9m
  (`onboarding_agent.yaml:258–271`). This is the universe that makes
  `current_loan_to_value` differ between periods — see Stage 8.
- **Parallel/bypass paths:** domain-specific universes (`central_lender_tape`,
  `pipeline_mi`, `forward_exposure`, regulatory) are selected from the same
  eligibility computation — not parallel logic, but parallel *outputs*.
- **Tests:** `tests/test_onboarding_period_eligibility.py`.
- **Known bugs:** period-column detection is name-list driven; an unlisted /
  oddly-cased period column would silently include rows. No confirmed regression.
- **Recommended consolidation:** none structurally; keep the row-filter in one
  place (it already is). Add a regression asserting both period counts.
- **Risk:** medium.

---

## Stage 6 — Entity-key resolution / joins

- **Responsible files:** `engine/onboarding_agent/entity_key_resolver.py`
  (`resolve_entity_keys`, `normalise_key`), with overlap scoring in
  `source_consolidator.py`. Promotion consumes it via
  `central_tape_builder._norm_key` (`central_tape_builder.py:185–197`).
- **Inputs:** candidate key columns + sample frames.
- **Outputs:** `04b_entity_key_resolution.{csv,json}` (join key per source pair,
  normalisation rule, overlap %, `needs_operator_review`).
- **Source of truth:** single. Normalisation rules (`numeric_string`,
  `strip_decimal_suffix`, `strip_trailing_<suffix>`, `remove_separators`) are
  applied only as a *comparison* key; source values are never mutated.
- **Parallel/bypass paths:** `central_tape_builder._norm_key` is a *second*
  numeric-key normaliser (handles `76034101.0 == 76034101`) living next to
  `entity_key_resolver.normalise_key`. They serve different moments (resolution
  vs promotion join) but are duplicated key-normalisation code.
- **Tests:** covered in `tests/test_onboarding_central_tape_builder.py`
  (joined-field scenarios) and `mi_agent_api/tests/test_funded_enrichment.py`.
- **Known bugs:** when the entity-key intersection is 0, enrichment silently
  nulls — surfaced as a reason code in `18e/18f` debug artefacts. This is the
  *other* half of the mi_2025_11 LTV miss (join_failed vs header bypass).
- **Recommended consolidation:** share one key-normalisation function between
  `entity_key_resolver` and `central_tape_builder`.
- **Risk:** medium.

---

## Stage 7 — Central lender tape promotion

- **Responsible files:** `engine/onboarding_agent/central_tape_builder.py`
  (build + lineage). Config: `config/system/onboarding_agent.yaml`
  `central_lender_tape` (`:277–339`).
- **Inputs:** `05_mapping_candidates.json`, `12_approved_mapping_overrides.yaml`,
  `13_source_precedence_rules.yaml`, `14_enum_review_decisions.yaml`,
  `28a_target_coverage_matrix.json` (MI modes), eligibility (Stage 5),
  entity keys (Stage 6), source frames via `_read_df`.
- **Outputs:** `18_central_lender_tape.csv` (one row per funded loan, canonical
  columns) + `18b` lineage + `18c` gaps + `18d` summary +
  `18e/18f` materialisation/universe debug.
- **Source of truth:** single. Promotion **does not re-discover mappings** — it
  consumes Gate-4 selections (`28a`) and approved overrides. Field selection is
  precedence-ordered with validation/conflict status recorded in lineage.
- **MI enrichment:** `mi_enrichment_fields` (valuations, LTVs, original balance,
  borrower DOBs, region/postcode, channels) are joined onto funded loans from
  period-eligible enrichment sources via the entity-key bridge. The DOB and
  postcode fields were added in `5af5074` so Stage 8 can derive age/region.
- **Parallel/bypass paths:** `_read_df` now redetects headers (good), but it is a
  *parallel reader* to `source_table_loader` (Stage 3 duplication).
- **Tests:** `tests/test_onboarding_central_tape_builder.py`,
  `mi_agent_api/tests/test_funded_central_tape.py`.
- **Known bugs:** see Stage 3 (`_read_df` bypass, fixed) and Stage 6 (join
  failures). Enrichment diagnostics now reason-code these.
- **Recommended consolidation:** fold `_read_df` into the single shared
  header-redetecting reader from Stage 3.
- **Risk:** medium.

---

## Stage 8 — Funded MI preparation  ⚠️ HIGH (duplicated by legacy)

- **Production module:** `mi_agent_api/funded_prep.py`
  (`prepare_funded_mi_dataset`, `_derive_source_fields`).
  - Numeric parsing: `analytics_lib/numeric.py::coerce_numeric` (the **single**
    robust parser: commas/£/accounting-negatives; defensively squeezes a
    one-column DataFrame and raises a clear `ValueError` on a duplicate-named
    multi-column frame — `numeric.py:36–44`).
  - LTV derivation: `_derive_ltv` (`funded_prep.py:113–149`) — prefer explicit,
    else `balance/valuation`, normalised to a 0..1 ratio (`_to_ratio:105–110`).
  - Bucketing: `analytics_lib/buckets.py::materialise_buckets` over
    `config/mi/buckets.yaml` (the **same** engine Streamlit *should* use).
  - Age: `_derive_youngest_age` (`funded_prep.py:170–190`) from borrower DOBs.
  - Duplicate-column collapse: `_dedupe_columns` (`funded_prep.py:193–214`).
- **Legacy module (parallel source of truth):** `analytics/mi_prep.py::add_buckets`,
  still imported by `analytics/streamlit_app_erm.py:127` and
  `analytics/tab_pipeline.py`. It:
  - parses numbers with **plain `pd.to_numeric`** (no comma/£ handling) — the
    exact failure `coerce_numeric` was written to fix;
  - **hardcodes** bucket edges in Python (does not read `config/mi/buckets.yaml`);
  - has **no** LTV-from-balance derivation and **no** age-from-DOB derivation.
- **Inputs:** `18_central_lender_tape.csv` (Stage 7), `config/mi/buckets.yaml`.
- **Outputs:** analytics-ready DataFrame (LTV, original-LTV, vintage_year,
  months_on_book, age, bucket dimensions) + a report with `derived_fields`,
  `ltv_derivation_basis`, `buckets_applied`, `group_aliases`,
  `duplicate_columns_collapsed`, `dimensions_available`, `missing_dimensions`
  (reason-coded).
- **Source of truth:** **split.** `funded_prep` is authoritative for React/API;
  `mi_prep` is authoritative for the Streamlit ERM dashboard. The same portfolio
  can yield **different dimensions and different balances** depending on which one
  ran (e.g. comma-formatted balances → 0 in Streamlit, correct in API; derived
  LTV present in API, absent in Streamlit).
- **Known bugs (named in brief):**
  - *"current_loan_to_value worked for mi_2025_10 but not mi_2025_11."* Root
    cause is upstream: the 40 November loans need valuation enrichment from
    PropertyExtract; with the Stage 3 `_read_df` bypass (pre-`5af5074`) or a
    failed entity-key join (Stage 6), `current_valuation_amount` is null for those
    rows → `_derive_ltv` returns `derivation_inputs_missing` → `ltv_bucket` empty
    for November. `funded_prep` now reports this honestly via
    `missing_dimensions` reason codes rather than a bare "absent".
  - *"Duplicate column names previously caused `out[c]` to return a DataFrame and
    crash `/mi/query`."* `_dedupe_columns` collapses duplicate-named columns
    (coalescing values) and records `duplicate_columns_collapsed`; `coerce_numeric`
    has a defensive guard (`numeric.py:36–44`). See Stage 11 for the executor-side
    guard. Fixed in `ba3f3c5`.
- **Tests:** `mi_agent_api/tests/test_funded_prep.py` (comma/£ parsing, LTV
  derivation, age-from-DOB, postcode region reason, duplicate collapse),
  `tests/test_phase1_analytics_lib.py` (bucket engine). **`analytics/mi_prep.py`
  has no equivalent robustness tests.**
- **Recommended consolidation:** retire `analytics/mi_prep.py::add_buckets` as a
  separate engine. Point Streamlit at `analytics_lib.buckets` +
  `analytics_lib.numeric` (and ideally at `funded_prep` for LTV/age derivation),
  or freeze Streamlit as legacy and stop maintaining it. Until then, `funded_prep`
  is the declared source of truth for the product.
- **Risk:** **high.**

---

## Stage 9 — MI field / dimension / metric contract

- **Responsible files:** `mi_agent/mi_semantics_field_registry.yaml` (generated)
  via `mi_agent/build_mi_semantics_registry.py` (a curated allowlist projected
  from `config/system/fields_registry.yaml`). `mi_agent/mi_query_spec.py` defines
  the `MIQuerySpec` vocabulary.
- **Inputs:** canonical `fields_registry.yaml` + `CURATION` allowlist.
- **Outputs:** per-field MI metadata: `role`, `format` (currency/percent/…),
  `allowed_aggregations`, `allowed_chart_roles`, `weight_field`, `bucket_field`.
- **Source of truth:** single. The MI registry *references* canonical fields via
  `canonical_field`; it does not redefine them. Buckets/virtual fields are
  MI-only additions.
- **Duplicated logic:** `config/mi/mi_equity_release_uk_applicability.yaml` is a
  separate overlay noting the registry is "GENERIC, not yet asset/regime-aware" —
  a second place that asserts field applicability.
- **Known bug (named in brief) — the percentage contract gap:** the registry
  marks fields `format: percent` but **does not record whether values are stored
  as fractions (0.51) or whole-number percent (51).** That scale is left to be
  *detected at runtime* (Stage 11) and is **not propagated** into the artifact
  (Stage 12) — the origin of the "0.51 displayed as 0.5%" defect.
- **Tests:** `mi_agent/tests/test_mi_semantics_*.py`.
- **Recommended consolidation:** make percent scale an explicit field-level
  contract (e.g. `percent_storage: fraction|whole`) so detection/formatting stop
  guessing. Single owner = registry.
- **Risk:** low (but it seeds the Stage 12/13 high-risk bug).

---

## Stage 10 — MI query parsing

- **Responsible files:** `mi_agent/interpreter/deterministic.py`
  (`interpret_deterministic`, production default), `mi_agent/llm_query_parser.py`
  (optional Claude path, data-free), both validated by
  `mi_agent/mi_spec_validation.py::validate_query_spec` and
  `mi_agent/mi_query_validator.py`. Orchestrated by
  `mi_agent/mi_agent_workflow.py::run_mi_agent_query`.
- **Inputs:** NL question + the **semantic catalogue only** (never raw data).
- **Outputs:** a validated `MIQuerySpec`.
- **Source of truth:** **single validator**, two parsers. Both paths converge on
  `validate_query_spec`, so the spec contract is one source of truth.
- **Duplicated logic:** field/dimension resolution heuristics exist in **both**
  `deterministic.py` and `llm_query_parser.py` (e.g. region→`geographic_region_obligor`,
  broker→`broker_channel`). Minor drift risk.
- **Parallel/bypass paths:** parser mode is env-selected
  (`MI_AGENT_PARSER_MODE` = deterministic | llm | auto); LLM failures fall back to
  deterministic.
- **Tests:** `mi_agent/tests/test_parser_cost_hardening.py`,
  `test_mi_query_validator.py`.
- **Recommended consolidation:** extract the shared field-resolution into one
  helper consumed by both parsers.
- **Risk:** low.

---

## Stage 11 — MI query execution

- **Responsible files:** `mi_agent/mi_query_executor.py`
  (`execute_mi_query`, `_execute_summary/_grouped/_loan_level`,
  `_guard_duplicate_columns`, `_detect_percent_scale`).
- **Inputs:** validated spec + the prepared DataFrame (Stage 8) + semantics.
- **Outputs:** `MIQueryResult` (data rows, resolved fields, warnings, metadata
  including `percent_scale`).
- **Source of truth:** single. Deterministic, no LLM, no rendering, no mutation
  (operates on a copy).
- **Known bugs (named in brief):**
  - *Duplicate columns crashing `/mi/query`.* `_guard_duplicate_columns` raises a
    controlled `MIDuplicateColumnError` (carrying the duplicate names + affected
    fields) before any `df[col]` selection can return a DataFrame; `_execute_loan_level`
    has a second defensive check. Surfaced as `validation.ok = False` (200, never a
    raw 500). Fixed in `ba3f3c5`; tested in
    `mi_agent/tests/test_mi_query_executor.py`.
  - *Percentage scale.* `_detect_percent_scale` computes a median heuristic
    (≤1.5 ⇒ `fraction`, else `whole_number_percent`) and records it in metadata
    **but deliberately does not rescale** the data, emitting a warning to that
    effect. This is the *correct* single detection point — the defect is that the
    detection is **not carried forward** (Stages 12–13 re-guess).
- **Tests:** `mi_agent/tests/test_mi_query_executor.py`,
  `test_mi_chart_factory.py` (percent scale), `test_mi_semantics_buckets.py`.
- **Recommended consolidation:** emit `percent_scale` (and per-column scale) into
  the result so downstream consumers stop re-detecting. Make this the only place
  scale is decided.
- **Risk:** medium.

---

## Stage 12 — API response / artifact schema  ⚠️ HIGH

- **Responsible files:** `mi_agent_api/app.py` (`POST /mi/query`),
  `mi_agent_api/adapters.py` (`adapt_workflow_result`, `_chart_artifact`,
  `_infer_col_format`, `_format_kpi_value`), `data_source.py`, `catalogue.py`.
- **Inputs:** the workflow result (spec + execution result + chart result).
- **Outputs:** the React artifact JSON: `{type, chartType, xKey, yKey, valueKey,
  series[], rows[], valueFormat, source{nativeChartType, figure?}}`.
- **Source of truth:** single adapter builds the artifact — but it **re-implements
  formatting and re-derives chart keys** rather than carrying forward Stage 11
  metadata.
- **Known bugs (named in brief):**
  - *"Bubble chart backend returned data but artifact had `yKey: null`."*
    Confirmed by design: for `scatter`/`bubble`, `_chart_artifact`
    (`adapters.py:218–230`) builds `series = [x, y, (size)]` but **never assigns
    `y_key`**, so the returned `yKey` (`adapters.py:288`) is `null`. Likewise
    `yKey` is null for bar/line/treemap; it is populated **only for heatmap**
    (`adapters.py:242`). The y dimension for scatter/bubble lives in `series[1]`.
    A real fixture shows this: `frontend/.../test/fixtures/funded_strat_ltv_mi_2025_10.json`
    has `"yKey": null` with the data carried in `series`/`rows`. Any consumer that
    reads top-level `yKey` for a non-heatmap chart gets `null` — the latent trap
    behind the reported bug. React reads `series` instead (Stage 13), so it works
    *now*, but the schema overloads `yKey` and is a recurring footgun.
  - *Percentages as fractions.* `_format_kpi_value` (`adapters.py:128–129`) formats
    `pct` as `f"{value:.1f}%"` with **no rescale**, and `percent_scale` from
    Stage 11 is **not** included in the artifact. So `0.51` ships as data and is
    rendered `0.5%`.
  - *"Numeric parsing fixed in API path but the static review generator used a
    separate parser and showed £0."* `mi_agent_api/scripts/generate_dashboard_review.py`
    has its **own** `_num` (`:50–54`, naive `str.replace(",","").replace("£","")`).
    `5af5074` wired its printed aggregate to `coerce_numeric` (`:149–151`) so
    balances are non-zero, but the standalone `_num` parser still exists in the
    file — a parallel parser kept alive.
- **Tests:** `mi_agent_api/tests/test_adapters.py` (x/y/size series, heatmap,
  treemap, fidelity), `test_api.py`.
- **Recommended consolidation:** (a) carry `percent_scale` and a per-column
  `valueFormat`+scale from Stage 11 into the artifact; (b) one formatting module
  shared by `_format_kpi_value` and the React formatter (or format only on one
  side); (c) normalise the chart-key contract so `yKey` is never a null trap
  (either always populate the semantic y, or formally document `series`-only for
  scatter/bubble and remove `yKey` from those); (d) delete the standalone `_num`
  in the review generator in favour of `coerce_numeric`.
- **Risk:** **high.**

---

## Stage 13 — React rendering / display formatting  ⚠️ HIGH

- **Responsible files:** `frontend/mi-agent-ui/src/domain/artifacts.ts` (schema),
  `domain/guards.ts`, `components/artifacts/ArtifactRenderer.tsx` (dispatch),
  `ChartArtifactView.tsx`, `HeatmapArtifactView.tsx`, `TreemapArtifactView.tsx`,
  `PlotlyArtifactView.tsx`, `lib/utils.ts` (`formatValue`, `formatPct`).
- **Inputs:** the artifact JSON (Stage 12).
- **Outputs:** rendered Recharts/Plotly figures.
- **Source of truth:** the TypeScript `ChartArtifact` interface
  (`artifacts.ts:91–103`) is the contract. It documents `yKey?` as the
  "second categorical axis (heatmap)" — i.e. **heatmap-only**, matching the
  backend. Scatter/bubble read `const [sx, sy, sz] = artifact.series`
  (`ChartArtifactView.tsx:191`). So API and React **agree** on the current
  `series` convention; the disagreement is historical (`yKey:null` consumers).
- **Duplicated logic (the core problem):** percent/number formatting is
  implemented **independently** from the backend:
  - `lib/utils.ts::formatValue` `case "pct": return ${value.toFixed(1)}%`
    (`utils.ts:64`) and `formatPct` (`utils.ts:24–25`) — **no ×100**.
  - `ChartArtifactView::valueFormatter` `if (fmt === "pct") return ${v}%`
    (`ChartArtifactView.tsx:30`) — **no ×100**.
  - Meanwhile Python `mi_chart_factory.format_percent` **does ×100** when
    `percent_scale == "fraction"`. The two sides disagree, and React never
    receives `percent_scale`. ⇒ `0.51` → `0.5%`.
- **Parallel/bypass paths (demo vs production):**
  - `api/index.ts::createAgentClient` returns `HttpAgentClient` only when
    `VITE_AGENT_API_URL` is set and mode ≠ "mock"; **otherwise it returns
    `MockAgentClient`** (`index.ts:23–26`). Every artifact carries a `mock:
    boolean` field (`artifacts.ts:59`).
  - `data/mockArtifacts.ts` + `data/mockResponses.ts` are a hand-crafted artifact
    pipeline (pre-formatted KPI strings, rounded values). It can drift from the
    adapter output it imitates.
- **Tests:** `components/artifacts/ArtifactRenderer.test.tsx`,
  `ChartArtifactView`/`PlotlyArtifactView` tests, and
  `test/fundedCentralTape.test.tsx` against frozen real-backend fixtures
  (`funded_summary_mi_2025_10/_11.json`, `funded_strat_ltv_*`).
- **Recommended consolidation:** one percent/number formatter, fed an explicit
  scale from the artifact (from Stage 11/12). Either format entirely server-side
  (ship display strings) or entirely client-side with scale metadata — not both.
- **Risk:** **high.**

---

## 14. Are there parallel pipelines? (explicit answer)

**Yes.** Three, plus one offline diagnostic:

| Parallel path | Production sibling | Status | Where |
|---|---|---|---|
| `analytics/mi_prep.py` + `streamlit_app_erm.py` | `mi_agent_api/funded_prep.py` + React | **legacy**, still imported/runnable | Stage 8 |
| `MockAgentClient` + `data/mockArtifacts.ts` | `HttpAgentClient` + `adapters.py` | **demo**, default when no API URL | Stage 13 |
| React `lib/utils.ts`/`valueFormatter` percent/number | Python `mi_chart_factory.format_percent` / `_format_kpi_value` | **both live**, disagree on scale | Stage 12/13 |
| `generate_dashboard_review.py::_num` | `analytics_lib.numeric.coerce_numeric` | **offline diagnostic**, partly rewired | Stage 12 |

Multiple **CSV/Excel readers** (Stage 3) are a fourth, lower-level parallelism:
one redetects headers, five do not.

---

## 15. Recent issues → stage → status (traceability)

| Brief item | Stage | Root cause | Status |
|---|---|---|---|
| Row-2 header loader bypassed by promotion/enrichment loader | 3 / 7 | `central_tape_builder._read_df` did plain read | **Fixed** `5af5074`; 5 other readers still bypass |
| API numeric fixed but static review generator showed £0 (separate parser) | 12 | `generate_dashboard_review._num` naive parser | **Partly fixed** `5af5074` (aggregate uses `coerce_numeric`; `_num` remains) |
| `funded_prep` and older `mi_prep.py` overlap | 8 | two prep engines (parser/buckets/derivation) | **Open** (mi_prep is legacy, still active) |
| API execution vs React/artifact disagree on chart schema | 12 / 13 | `yKey` null for non-heatmap; y in `series` | **Reconciled** via `series` convention; `yKey` still a null trap |
| `current_loan_to_value` worked mi_2025_10 not mi_2025_11 | 5 / 6 / 8 | Nov loans' valuation enrichment failed (header bypass / join) → LTV derivation inputs missing | **Fixed** `5af5074` (+ reason codes); guard with period regression |
| Bubble backend had data but artifact `yKey: null` | 12 / 13 | adapter never sets `y_key` for scatter/bubble | **Works** (React reads `series`); schema overload remains |
| Percentage fractions (0.51) shown as 0.5% | 9 / 11 / 12 / 13 | scale detected once (executor) but not propagated; 3 formatters re-guess, none ×100 | **Open** |
| Duplicate column names crashed `/mi/query` (`out[c]` → DataFrame) | 8 / 11 | duplicate-named column selection returns DataFrame | **Fixed** `ba3f3c5` (prep collapse + numeric guard + executor `MIDuplicateColumnError`) |

---

## 16. Ordered consolidation plan

Sequenced by leverage (each unblocks/derisks the next). All are
consolidation/diagnostic work; none requires new MI features.

1. **One source reader (Stage 3).** Make `redetect_header` the only way a source
   table is read. Route `file_classifier`, `file_profiler`,
   `onboarding_orchestrator`, `llm_assisted_mapping`, `compare_semantic_alignment`,
   and `central_tape_builder._read_df` through a single `read_source_frame`, ideally
   reading each frame **once** and caching it. *Kills the row-2 class of bugs and
   the mi_2025_11 LTV miss at the source.* Risk: medium; high payoff.

2. **One percent/scale contract end-to-end (Stages 9→11→12→13).**
   - Stage 9: add explicit `percent_storage: fraction|whole` to the registry.
   - Stage 11: emit per-result `percent_scale` (already detected) into the result.
   - Stage 12: include scale in the artifact; pick **one** formatting side.
   - Stage 13: delete the duplicate `formatValue`/`valueFormatter` percent logic
     or have it consume the scale. *Closes the 0.51→0.5% defect permanently.*
   Risk: medium.

3. **Retire the second prep engine (Stage 8).** Point `analytics/mi_prep.py` at
   `analytics_lib.numeric.coerce_numeric` + `analytics_lib.buckets` (and LTV/age
   derivation), or formally freeze Streamlit ERM as legacy. *Removes the
   parser/bucket/derivation divergence.* Risk: medium (touches legacy UI) — can be
   staged as "Streamlit reads `analytics_lib`" first.

4. **Normalise the chart-key contract (Stages 12/13).** Either always populate the
   semantic `yKey`, or formally document `series`-only for scatter/bubble and drop
   `yKey` from those types in the schema + guards. *Removes the `yKey:null` trap.*
   Risk: low.

5. **Single key-normaliser (Stages 4/6).** One `normalise_header` and one
   `normalise_key` shared across `semantic_alignment`, `entity_key_resolver`, and
   `central_tape_builder`. Risk: low.

6. **Delete the standalone `_num` (Stage 12).** Make
   `generate_dashboard_review.py` use `coerce_numeric` everywhere. Risk: trivial.

7. **Read-only diagnostics to lock the wins (no behaviour change).**
   - A period regression asserting `mi_2025_10`=33 / `mi_2025_11`=73 loans and the
     LTV bucket populates for both.
   - A reader-parity check asserting every source reader applies `redetect_header`.
   - A contract test asserting any `format: percent` field round-trips
     fraction-stored values to the correct displayed percent.
   These are the only code additions recommended, and they are read-only/test-only.

---

## 17. Notes on scope

- No production code was changed by this audit.
- Legacy (`analytics/`) is documented only to ring-fence it; it is not the
  product spine.
- Annex 2/12 regulatory promotion paths in `central_tape_builder` are out of
  scope (MI/funded focus) and untouched here.

*Descriptive audit. Citations are to files/functions verified in the repo on
2026-06-21.*

---

## 18. Consolidation log — stages 1–7 (raw files → central lender tape)

**Date:** 2026-06-22. Scope: stages 1–7 only. No React, MI parser, or chart
changes. The first half of the spine now has one production path for reading
source tables, detecting headers, mapping fields, selecting period-eligible
sources, joining entities, and promoting values.

### Root cause found (the November-LTV regression)

The exact stage was **period eligibility**, confirmed by reproducing it: a
collateral / PropertyExtract delivered for October was dropped from the November
run with `reason="period_mismatch"` (visible in `18f_central_universe_debug.json`
`excluded_sources`). Enrichment roles (`collateral_report`, `warehouse_agreement`,
`cashflow_report`) were matched with the **strict funded-book cadence**
(`source_period_eligibility._funded_period_match`, `period == run_p`), the same
rule that defines the funded universe. An October valuation is, however, the
*latest-available* valuation for a loan still funded in November, so it must
enrich — not be excluded.

### Changes (production code)

1. **Enrichment cadence** — `source_period_eligibility.py`: new
   `_enrichment_period_match` (on-or-before the run period is eligible; only
   *future* enrichment is excluded). The `central_lender_tape` domain now splits
   universe roles (strict `_funded_period_match`, preserves 33/73) from enrichment
   roles (as-of cadence). Enrichment never seeds the universe, so counts are
   unchanged.
2. **Row-level filter is role-aware** — `central_tape_builder._build_lender_tape`:
   universe rows are still filtered to `period == run_period`; a *cumulative
   enrichment* file keeps rows `period <= run_period`, ordered latest-wins.
3. **Shared header detection in profiling** — `file_profiler._read_structured`
   now applies the same `source_table_loader.redetect_header` that promotion
   (`central_tape_builder._read_df`) and the source-table loader use, so a row-2
   PropertyExtract header profiles its real columns. Header detection is now one
   shared function across profiling and promotion/enrichment (item B).
4. **Run-level diagnostic** — `central_tape_builder` now writes
   `…/<run_id>/output/diagnostics/spine_stage_1_7_report.json` with, per target
   field: raw source candidates, selected source, per-source period eligibility
   (`inferred_reporting_period`, `period_eligible`, `rows_raw`,
   `rows_after_period_filter`, key overlap), promoted non-null count, derivation
   basis, and failure reason when unavailable; plus the funded-universe summary
   (counts, balance, excluded sources, pipeline exclusion). Built from the existing
   18e/18f debug data — no recomputation.

### Result (verified)

- `mi_2025_10`: 33 rows, ≈£4.208MM; valuation 33/33 (unchanged).
- `mi_2025_11`: 73 rows, ≈£8.903MM; **valuation 73/73**, and downstream
  `funded_prep` derives `current_loan_to_value` + `ltv_bucket` **73/73**.
- No pipeline rows promoted as funded; enrichment matches funded rows only.
- Tests: `mi_agent_api/tests/test_funded_period_enrichment.py` (12 cases — cadence
  units, both PropertyExtract filename/header variants, entity-key drift,
  end-to-end Oct/Nov, spine diagnostic). Full onboarding + API suite: 190 passed.

The Stage-3 reader-consolidation item in §16(1) is partially delivered:
profiling + promotion/enrichment now share `redetect_header`;
`onboarding_orchestrator`, `llm_assisted_mapping`, and `compare_semantic_alignment`
remain follow-ups.

---

## 19. Consolidation log — stages 8–13 (central tape → prepared dataset → query → API → React)

**Date:** 2026-06-22. Scope: stages 8–13. No change to raw discovery/promotion.
Principle delivered: **one prepared MI dataset and one metadata contract**; no
frontend, review script, or executor independently infers field meaning, numeric
parsing, bucket availability, or display scale.

### The single dataset contract

- **New `mi_agent/mi_dataset_profile.py`** — the one place that computes, per field:
  semantic type, **storage scale** (`percent_fraction` 0.51 vs `percent_points`
  51, by column-median), display format, non-null + numeric-parse counts, and
  dimension/metric availability. Also `validate_query_data` — the data-aware
  query validator.
- **New `mi_agent_api/mi_dataset_contract.py`** — builds the API/health-facing
  contract from the profile + the funded_prep report (source field / derivation
  basis / bucket source), and `summary_numbers` (shared review helper).
- `/health` now exposes `datasetContract` (per-field metadata) and dimensions
  reflect **actual non-null prepared values** (`mi_agent_api/app.py`,
  `data_source.py`).

### Query parsing / validation (C, D)

- `llm_query_parser._deterministic_parse` now routes **filtered count/balance**:
  "how many loans with youngest age more than 70" → `intent=summary`,
  `aggregation=count`, `filters={youngest_borrower_age: {op: gt, value: 70}}` —
  not a bar chart. Numeric operators (`gt/ge/lt/le/eq/between`) are parsed from
  natural language and applied in `mi_query_executor._apply_filters`
  (percent-scale aware, using the same single scale source).
- `mi_agent_workflow` runs `validate_query_data` **before** "Validation: Passed":
  a metric with no numeric values, a dimension with no non-blank values, a filter
  field with no values, or a loan-level x/y/size with no usable rows now fails
  with an **exact reason** (`metric_no_numeric_values`, `dimension_no_values`,
  `loan_level_no_usable_rows`, …). A post-execution empty result fails as
  `no_values_after_preparation`. Duplicate columns remain a controlled validation
  failure — `/mi/query` never returns a raw 500.

### API artifact schema (E, F)

- Bubble/scatter artifacts carry **explicit role keys** `xKey` / `yKey` /
  `sizeKey` + `xLabel` / `yLabel` / `sizeLabel` (`adapters._chart_artifact`).
  `yKey` is never null for a bubble; React no longer infers axes from series
  order. Every chart/table/KPI artifact carries per-column `displayHints`
  (`{format, scale}`) from the contract. The executor keeps internal values
  unchanged and never rescales.

### React rendering / display (G, H)

- `lib/utils.ts::formatValue` + `toPercentPoints` apply the contract scale, so a
  fraction (0.51) renders **51.0%**, not 0.5% — consistently across tables
  (`TableArtifactView`), chart tooltips/axes (`ChartArtifactView`), and KPI values
  (formatted server-side from the same contract). `ChartArtifactView` consumes
  `xKey`/`yKey`/`sizeKey` directly. Failed validation surfaces the controlled
  validation artifact (no silent empty chart).

### Static review generator (I)

- `generate_dashboard_review.py::_num` now delegates to the shared
  `analytics_lib.numeric.coerce_numeric` (no separate parser); summary numbers use
  the shared contract helper.

### Result (verified, real pipeline + units)

- `balance by ltv by age` on mi_2025_10 → API 200, `ok:true`, rowCount 33, chart
  artifact with `xKey`/`yKey`/`sizeKey`; **LTV 0.29–0.56 displays 29%–56%**.
- `current outstanding balance by ltv bucket` renders for October.
- Missing/empty LTV → graceful failure with exact reason; never "Validation:
  Passed"; no chart artifact emitted.
- `/health` lists available + missing dimensions from non-null prepared values
  and the per-field contract.
- Tests: `mi_agent_api/tests/test_mi_dataset_contract.py` (12), frontend
  `lib/utils.test.ts` + `ChartArtifactView.test.tsx`. Full MI + onboarding suite:
  **239 passed**; frontend **55 passed**.

---

## 20. Closure status — finding → action → remaining risk

Final status of every duplicated-authority / parallel-path finding from §0/§14,
after the Stage 1–7 and Stage 8–13 consolidation. Read each row as *initial
finding → consolidation action → remaining risk*.

| # | Parallel path / finding (initial) | Consolidation action | Status |
|---|---|---|---|
| 1 | Production funded MI prep vs legacy Streamlit `analytics/mi_prep.py` | Production React/API path is `funded_prep.prepare_funded_mi_dataset` only; `mi_prep.py` is **legacy Streamlit-only** and not on the React path | **Ring-fenced.** Production single; legacy retirement deferred |
| 2 | Shared `coerce_numeric` vs local `pd.to_numeric` | Production prep, executor, profiler-derivation and the review generator all use `analytics_lib.numeric.coerce_numeric`; `mi_prep.py`'s `pd.to_numeric` is Streamlit-only | **Fixed on production path.** Legacy Streamlit usage deferred |
| 3 | Backend percent formatting vs React percent formatting | Single percent storage scale computed once in `mi_dataset_profile`; carried in the artifact `displayHints`/KPI; React `formatValue`/`toPercentPoints` and server `_format_kpi_value` both consume it | **Fixed** |
| 4 | API real (`HttpAgentClient`) vs mock (`MockAgentClient`) | `createAgentClient` uses HTTP when `VITE_AGENT_API_URL` is set; `.env.development` sets it (`/` via Vite proxy) so `npm run dev` is live by default; mock requires explicit `VITE_AGENT_MODE=mock` | **Fixed (mock explicit-only)** |
| 5 | Static review generator's own parser vs API/contract | `generate_dashboard_review.py::_num` delegates to `coerce_numeric`; summary numbers via the shared contract helper | **Fixed** |
| 6 | Chart artifact schema vs React renderer assumptions (series-order, `yKey:null`) | Backend emits explicit `xKey`/`yKey`/`sizeKey` + labels + `displayHints`; React consumes them directly | **Fixed** |
| 7 | Stage-3 header detection bypassed by promotion/enrichment loader | `central_tape_builder._read_df` and `file_profiler` now share `redetect_header` | **Fixed (production path)**; 3 non-production readers deferred (§16/§18) |
| 8 | Period eligibility excluding October valuation from November | Enrichment cadence split from universe cadence (`_enrichment_period_match`, as-of ≤ run period) | **Fixed** |

### Production sources of truth (post-consolidation)

| Concern | Production module (single source) |
|---|---|
| Source-table read + header detection | `engine/onboarding_agent/source_table_loader.redetect_header` (via `central_tape_builder._read_df`, `file_profiler`) |
| Period eligibility (funded vs enrichment cadence) | `engine/onboarding_agent/source_period_eligibility.py` |
| Entity-key resolution / join | `engine/onboarding_agent/entity_key_resolver.py` |
| Central tape promotion + snapshot | `engine/onboarding_agent/central_tape_builder.py` (one tape per run id) |
| Funded MI preparation | `mi_agent_api/funded_prep.prepare_funded_mi_dataset` |
| Numeric parsing | `analytics_lib/numeric.coerce_numeric` |
| Bucketing | `analytics_lib/buckets` + `config/mi/buckets.yaml` |
| Per-field metadata + storage scale + data validation | `mi_agent/mi_dataset_profile.py` |
| API dataset contract (/health, review) | `mi_agent_api/mi_dataset_contract.py` |
| Query parse → validate → execute | `mi_agent/llm_query_parser` (deterministic) → `mi_query_validator` + `mi_dataset_profile.validate_query_data` → `mi_query_executor` |
| API artifact schema | `mi_agent_api/adapters.py` |
| React rendering / display | `frontend/mi-agent-ui/src/components/artifacts/*` + `lib/utils.ts` |

### Legacy / demo modules retained (NOT production paths)

- `analytics/mi_prep.py`, `analytics/streamlit_app_erm.py` and the rest of
  `analytics/` — the legacy Streamlit ERM dashboard. Not imported by the React MI
  Agent path. Retained; retirement is out of scope here.
- `frontend/.../api/MockAgentClient.ts` + `data/mockArtifacts.ts` /
  `mockResponses.ts` — demo path, used only when `VITE_AGENT_MODE=mock` is set
  explicitly.

### Deferred follow-ups (intentionally NOT done in this work)

1. Route the remaining non-production readers
   (`onboarding_orchestrator`, `llm_assisted_mapping`, `compare_semantic_alignment`)
   through the shared `redetect_header` (§16/§18).
2. Fold the duplicate field-resolution heuristics in the two parser layers
   (`mi_agent/interpreter/deterministic.py` vs the flat parser in
   `mi_agent/llm_query_parser.py`) into one helper (§16 item).
3. Retire / wrap legacy Streamlit `analytics/` prep so it consumes
   `analytics_lib` + `funded_prep` (or freeze it formally as legacy).
4. Declare percent storage scale in the registry contract itself (currently the
   single scale is computed per-column by `mi_dataset_profile`).

### Snapshot semantics (confirmed)

One run id = one central lender tape snapshot = one reporting-period funded
universe. `mi_2025_10` (33 loans / ≈£4.208MM) and `mi_2025_11` (73 loans /
≈£8.903MM) are written to separate `…/<run_id>/output/central/` snapshots;
November does not overwrite October. Funded universe is strict-period
(`period == run period`); enrichment is as-of-period (latest valid source on or
before the run period, never future); enrichment never seeds the universe and
pipeline rows never create funded rows. The November LTV (73/73) is populated by
the as-of October valuation enrichment (diagnostic: selected source period
`2025-10`, eligible, key overlap 73/73) — proven by
`spine_stage_1_7_report.json`, not by merging the two runs.
