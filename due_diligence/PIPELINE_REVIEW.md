# Trakt Pipeline — Final Production Readiness Review

**Date:** 2026-02-11
**Scope:** All 11 Python gate scripts, orchestrator, config layer, Azure deployment
**Verdict:** Not production-grade. An institution would not place enterprise value on this in its current state.

---

## Executive Summary

The pipeline has genuine architectural merit — the 5-gate structure, config hierarchy, lineage tracking, and materiality-based validation aggregation are all sound design choices. But the implementation has **5 critical bugs**, **~13 high-severity gaps**, and several deployment-layer issues that collectively mean: a regulatory submission produced by this pipeline today would be **non-compliant and unauditable**.

---

## Pipeline Inventory

| Gate | Script | Lines | Purpose |
|------|--------|-------|---------|
| Orchestrator | `trakt_run.py` | 710 | Multi-mode dispatcher (MI / Annex 12 / Regulatory), subprocess orchestration, manifest writer |
| Gate 1 | `semantic_alignment.py` | 635 | Maps raw loan tape headers to canonical fields via 6-tier strategy (exact, normalized, alias, Jaccard, fuzzy token-set, fuzzy ratio) |
| Transform | `canonical_transform.py` | 581 | Standardizes formats (dates, decimals, booleans), derives fields (LTV, geography, reporting date), emits typed canonical CSV |
| Gate 2 | `validate_canonical.py` | 611 | Validates canonical data for core field presence, format compliance, enum correctness |
| Gate 2.5 | `lineage_tracker.py` | 417 | Generates field_lineage.json and value_lineage.json for audit traceability |
| Gate 3 | `validate_business_rules.py` | 704 | Applies ~40 cross-field business logic rules (dates, balances, LTV, arrears, identifiers) |
| Gate 3b | `aggregate_validation_results.py` | 326 | Aggregates row-level violations into field-level summary with materiality classification |
| Gate 4 (Annex 12) | `annex12_projector.py` | 425 | Projects single deal-level ESMA Annex 12 record from canonical data with IVSR/IVSF support |
| Gate 4 (Regulatory) | `regime_projector.py` | 709 | Projects canonical data to ESMA Annex 2-9 schemas with enum mapping and ND defaults |
| Gate 5 (Annex 12) | `xml_builder_investor.py` | 477 | Builds ESMA Annex 12 XML from projected CSV, XSD-aware with namespace enforcement |
| Gate 5 (Regulatory) | `xml_builder.py` | 40 | Stub Jinja2-based XML builder for Annex 2-9 — requires template upload to function |

---

## CRITICAL BUGS (must fix before any production use)

### C1. ND Regex Accepts Invalid Codes Across Pipeline

**Context:** ESMA only accepts ND1 through ND5. These are "No Data" reason codes used when a field value is unavailable.

| File | Line | Pattern | Accepts ND0? | Accepts ND6+? | Correct? |
|------|------|---------|:---:|:---:|:---:|
| `canonical_transform.py` | 33 | `^ND\d+$` | YES | YES | **NO** |
| `validate_canonical.py` | 32 | `^ND[1-9]\d*$` | No | YES | **NO** |
| `xml_builder_investor.py` | 150 | `startswith("ND")` | YES | YES | **NO** |
| `annex12_projector.py` | 80-85 | Explicit `{ND1..ND5}` | No | No | YES |

**Impact:** Invalid ND codes (ND0, ND6, ND7, ND99) silently pass through the entire pipeline. The canonical transform treats them as legitimate and strips them during type conversion. The validator accepts them. The XML builder emits them. An ESMA submission containing ND7 would be rejected by the regulator.

**Fix:** `r"^ND[1-5]$"` (case-insensitive) in all four files.

---

### C2. Pipe Delimiter Mismatch Between Projector and XML Builder

**Location:** `annex12_projector.py:31` vs `xml_builder_investor.py:305`

The projector joins repeatable-group values (IVSR, IVSF investor tranches) with:
```python
PIPE_DELIM = "\x1f"   # Unit Separator (0x1F)
```

The XML builder splits them with:
```python
parts = [v.strip() for v in raw_val.split("|")]   # Pipe character (0x7C)
```

These are **different characters**. The XML builder will never split IVSR/IVSF values because the delimiter doesn't match. All repeatable values are treated as a single monolithic string, producing malformed XML where each investor tranche section contains the entire concatenated dataset instead of individual per-tranche records.

**Impact:** Any Annex 12 submission with multiple investor tranches produces structurally invalid XML.

**Fix:** Align both files to use the same delimiter (pipe `"|"` is the conventional choice).

---

### C3. Validation Mutates Data Before Checking It

**Location:** `validate_business_rules.py:665-666`

```python
df = _backfill_ltv(df, "current_principal_balance",
               "current_valuation_amount", "current_loan_to_value")
```

`_backfill_ltv` modifies the DataFrame **in place**, inserting computed LTV values where they are missing. The subsequent validation rules then check the **modified** data, not the actual data in the canonical CSV.

**Impact:**
- LTV-related rules (LTV001, LTV002) can never detect missing LTV because it gets backfilled first
- The validation report gives a false sense of data quality
- A validation step should be **read-only** — it must assess the data as-is, not improve it first

**Fix:** Remove `_backfill_ltv` from the validator. LTV derivation belongs in the transform step (Gate 2), which already does it.

---

### C4. Arrears Bucket Boundary Error

**Location:** `config/regime/annex12_template.yaml:138`

The IVSS40 bucket (60-89 days in arrears) is configured with `min: 50` instead of `min: 60`. This creates an overlap with IVSS39 (30-59 days).

**Impact:** Loans in the 50-59 day range are double-counted across two buckets. Bucket sums exceed total balance. An auditor or ESMA validation tool catches this immediately.

**Fix:** Change `min: 50` to `min: 60`.

---

### C5. Regulatory Mode Non-Functional

**Location:** `xml_builder.py` (40 lines)

The file is a stub that requires a Jinja2 template (`esma_template.xml`) which does not exist. Running `--mode regulatory` crashes at Gate 5 with `TemplateNotFound`.

**Impact:** The entire regulatory pipeline (ESMA Annex 2-9) cannot produce XML output.

**Fix:** Upload the Jinja template file (user has this planned).

---

## HIGH-SEVERITY ISSUES

### Pipeline Engine

#### H1. Validators Never Exit Non-Zero — Manifest Always Says "pass"

**Location:** `validate_canonical.py` main(), `validate_business_rules.py` main()

Neither validator calls `sys.exit(1)` when errors are found. The orchestrator checks exit codes to set `canonical_status` in the manifest, but they are always 0. Combined with `allow_fail=True`, the pipeline **always runs to completion** regardless of data quality.

**Impact:** The `run_manifest.json` always reports `"status": "pass"` for validation gates. Automated systems consuming the manifest are misled. There is no circuit breaker.

---

#### H2. String "NaT" Leakage Through Pipeline

**Location:** `semantic_alignment.py:315`, `canonical_transform.py:124`

Both files use `pd.Series.dt.strftime("%Y-%m-%d")` on series containing `NaT` (Not a Time). Pandas `strftime` converts `NaT` to the literal string `"NaT"`, which then:
- Passes downstream date format checks (it's a non-empty string)
- Appears in regime projections as a real value
- Gets written to ESMA XML as the literal text `NaT`

**Impact:** Silently corrupted date fields in regulatory submissions.

**Fix:** Replace NaT with `pd.NA` after strftime, or use a wrapper that returns NA for NaT inputs.

---

#### H3. Date Comparisons Are String-Based, Not Datetime

**Location:** `validate_business_rules.py:101-103`

```python
"test": lambda df: df["origination_date"].isna()
                   | (df["origination_date"] <= df["reporting_date"]),
```

This is lexicographic string comparison. It works **only** if dates are in `YYYY-MM-DD` format. If any date slips through as `DD/MM/YYYY` or `MM-DD-YYYY` (e.g., from a parse failure upstream), the comparison produces silently wrong results. Rules DAT001, DAT002, DAT003, DAT004 all have this issue.

---

#### H4. Subprocess Output Not Captured in Manifest

**Location:** `trakt_run.py:77`

```python
result = subprocess.run(args, env=env)
```

No `capture_output=True`. If a child process crashes, the stack trace goes to the terminal but is not recorded in the manifest or any log artifact. For a production audit trail, subprocess output should be captured and persisted.

---

### Configuration Issues

#### H5. Enum Code Mismatch Between Template and Constraints

**Location:** `annex12_template.yaml` vs `annex12_field_constraints.yaml`

The template uses ESMA codes (`VSLC`, `ORIG`, `INVS`, `ARRR`). The constraints file uses informal labels (`VR`, `Originator`, `InvestorReport`, `Arrears`). No mapping layer reconciles them. Field validation will either pass everything or flag false positives depending on which config the validator consults.

---

#### H6. Empty Computation Methods for Mandatory Fields

**Location:** `annex12_template.yaml:103-112`

Fields IVSS21 (current principal balance), IVSS22 (current balance), and IVSS24 (outstanding nominal amount) have empty `computation_method` entries. These are **mandatory monetary fields** in Annex 12 — the projector needs to know how to compute them from canonical data.

---

#### H7. Arrears Format Conflict

**Location:** `annex12_template.yaml` vs `annex12_field_constraints.yaml`

The template uses `BUCKET_SUM` computation (produces monetary amounts). The constraints specify `{PERCENTAGE}` format. These are incompatible — one produces GBP values, the other expects percentages.

---

#### H8. Static Reporting Date Mismatch

**Location:** `config_client_ERM_UK.yaml:13` uses `2025-11-30`; `config_client_annex12.yaml:52` uses `2025-10-31`

A one-month mismatch between the main client config and the Annex 12 overlay. The pipeline could produce an Annex 12 report dated October with data filtered for November (or vice versa).

---

#### H9. XSD Files Are DRAFT Versions

**Location:** `config/system/DRAFT1auth.098.*.xsd`

The XSD schemas used for validation are labelled `DRAFT1`. Production ESMA submissions must validate against the final published schemas. Validation against draft schemas may pass XML that the regulator rejects, or reject XML that the regulator accepts.

---

#### H10. No rule_registry.yaml — Violations Aggregate Generically

**Location:** Missing file

Without `rule_registry.yaml`, all business rule violations in the aggregate summary are assigned to the generic field name `"PORTFOLIO"`. The field-level dashboard becomes meaningless — all violations appear under one bucket instead of being attributed to specific canonical fields.

---

### Deployment Issues

#### H11. Temp Directory Never Cleaned Up

**Location:** `function_app.py:85-90`

Each blob trigger invocation creates a temporary directory but never removes it. Over time, the Azure Function's storage fills up.

---

#### H12. No Subprocess Timeout

**Location:** `function_app.py:122`

The pipeline subprocess has no timeout. A hung gate blocks the Azure Function for the full 30-minute function timeout, consuming compute resources.

---

#### H13. No Idempotency Guard

**Location:** `function_app.py`

Blob triggers can fire multiple times for the same file (at-least-once delivery). Without a deduplication mechanism (e.g., checking if output already exists for a given input hash), the pipeline may process the same file multiple times, producing duplicate outputs.

---

## MEDIUM-SEVERITY ISSUES

### Code Quality

| Issue | Location | Detail |
|-------|----------|--------|
| Abbreviation expansion false positives | `semantic_alignment.py:99-105` | Token `"material"` matches prefix `"mat"` → expanded to `"maturity"` + `"erial"` |
| LEI rule checks URL, not LEI format | `validate_business_rules.py:366-372` | Checks for `http://standards.iso.org/iso/17442` instead of 20-char alphanumeric |
| Module-level side effects | `validate_business_rules.py:20-21`, `aggregate_validation_results.py:36-37` | Create directories at import time |
| `is_nd` matches any ND-prefixed string | `xml_builder_investor.py:150-151` | Would match `"NDAKOTA"`, NUTS codes, etc. |
| Missing closing brace in format check | `annex12_projector.py:156` | `"{INTEGER"` missing `}` — works by accident |
| No validation of ND range in defaults | `regime_projector.py:346-355` | Config could insert ND7 without error |
| Hardcoded default year 2025 | `canonical_transform.py:58,385,505` | Stale for 2026 production use |
| `safe_read_json` not actually safe | `lineage_tracker.py:26-29` | No try/except — malformed JSON crashes lineage step |
| Division by zero possible | `lineage_tracker.py:324-329` | If `len(core_fields)` is 0 |
| Current LTV unconditionally overwritten | `canonical_transform.py:311-314` | Provider-reported LTV silently replaced with recalculated value |

### Dashboard / Streamlit

| Issue | Detail |
|-------|--------|
| Local filesystem reads only | Cannot read from Azure blob storage (outbound container) |
| `MAX_ROWS` guard defined but never applied | Large files crash the dashboard |
| `generate_pptx_erm.py` referenced but missing | PowerPoint export broken |
| Deprecated `is_categorical_dtype` | Will break on pandas 3.0 |
| Relative config paths | Dashboard only works if run from project root |

### Security

| Issue | Detail |
|-------|--------|
| No path traversal protection on blob names | Malicious filename could write outside temp directory |
| `unsafe_allow_html=True` with data injection | 29 occurrences in Streamlit app — CSV values rendered as HTML |
| Connection-string auth | Should use managed identity for production Azure deployment |

---

## What Has Enterprise Value Today

Despite the bugs, the **design** is solid and represents significant domain knowledge:

1. **5-gate architecture** with clean separation of concerns — alignment, transform, validation, projection, delivery
2. **Config hierarchy** (system → asset → client → regime) — correct pattern for multi-client securitisation platforms
3. **Fields registry** (5,508 lines) — comprehensive ESMA data dictionary covering all Annex field definitions
4. **Materiality-based validation aggregation** — the issue_policy.yaml + field-level summary approach is exactly what due diligence teams need
5. **Lineage tracking** (field + value level) — audit requirement for any regulated reporting
6. **Six-tier semantic alignment** — genuinely useful for ingesting heterogeneous loan tapes from different originators
7. **Azure blob-trigger deployment** — correct serverless pattern for file-driven pipelines
8. **Multi-mode support** — MI, Annex 12, and Regulatory modes share Gates 1-3 with mode-specific Gates 4-5

The *architecture* has enterprise value. The *implementation* needs a hardening pass before an institution would deploy it. The gap is roughly **2-3 weeks of focused engineering** on the critical/high items, not a fundamental redesign.

---

## Recommended Fix Priority

| Priority | Item | Effort | Impact |
|:---:|------|--------|--------|
| 1 | Fix ND regex to `^ND[1-5]$` across all 4 files | 30 min | Prevents non-compliant ESMA submissions |
| 2 | Fix pipe delimiter mismatch (projector ↔ XML builder) | 15 min | Enables valid Annex 12 XML for multi-tranche deals |
| 3 | Make validation read-only (remove `_backfill_ltv` from validator) | 30 min | Ensures validation report reflects actual data quality |
| 4 | Fix arrears bucket boundary min=50→60 | 5 min | Prevents double-counting in arrears buckets |
| 5 | Upload Jinja template for xml_builder.py | User action | Enables regulatory mode (Annex 2-9) |
| 6 | Add non-zero exit codes to validators when errors found | 30 min | Enables manifest to report true pass/fail status |
| 7 | Handle NaT → pd.NA instead of string "NaT" | 1 hr | Prevents corrupted dates in submissions |
| 8 | Reconcile enum codes between template and constraints | 2 hrs | Ensures consistent field validation |
| 9 | Add temp cleanup + subprocess timeout to function_app.py | 30 min | Production stability |
| 10 | Replace hardcoded dates/test values with parameterized config | 1 hr | Removes dev leftovers |

**Items 1-4** are the minimum to produce a **technically valid** ESMA Annex 12 submission.
**Items 5-10** make it **production-grade**.

---

## Deployment Note: Azure Flex Consumption Blob Trigger

The blob trigger requires `source="EventGrid"` in the decorator and an Event Grid subscription on the storage account. Standard polling-based blob triggers are not supported on the Flex Consumption plan. This has been identified and the fix is:

1. Add `source="EventGrid"` to the `@app.blob_trigger` decorator in `function_app.py`
2. Create an Event Grid subscription on the `trakt8ba0` storage account pointing to the `trakt_blob_trigger` function
3. Filter for `Blob Created` events only on the `inbound` container

---

*Review conducted against commit `3f4d402` on branch `claude/formalize-due-diligence-structure-qUtn2`*
