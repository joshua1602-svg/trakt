# ESMA Annex 2 Regulatory Watch — ESMA_Annex2

*Generated 2026-01-01T00:00:00Z · report contract v1.1.0 · Stage 1, observational only — no active configuration was modified.*

**Outcome: `REGULATORY_SPEC_CHANGED`** — is the implemented machine-readable Annex 2 specification current? **NO**

- gating sources: `SOURCE_CHANGED` · corroborating sources: `SOURCE_CHECK_FAILED`
- parsed specification: `SPEC_CHANGED`
- 1 regulatory delta(s), 3 implementation impact(s)

## 1. Baseline source / version

- **Spec version:** `auth.099.001.04 / workbook 1.3.1`
- **Fields parsed:** 104
- **Normalizer:** `annex2-normalizer/1.1.0`
- **Scope:** {"asset_branch": "ResdtlRealEsttLn/PrfrmgLn", "cancellation_rows": "excluded", "nd_schema_cross_check": "enabled", "performance": "PRF", "schema_namespace": "urn:esma:xsd:DRAFT1auth.099.001.04", "sheet": "DRAFT1auth.099.001.04", "templates": ["ALL", "RRE"]}

| artefact | criticality | external version | sha256 | status |
| --- | --- | --- | --- | --- |
| `esma_annex2_message_workbook` | **gating** | 1.3.1 | `983e33c851a6621a…` | OK |
| `esma_annex2_reporting_instructions` | corroborating | UNKNOWN | `—` | SOURCE_CHECK_FAILED — no local copy vendored and retrieval is disabled |
| `esma_annex2_sample_message` | corroborating | UNKNOWN | `64e30a51b1019acb…` | OK |
| `esma_annex2_xml_schema` | **gating** | 1.3.0 | `48e587ec5905ab32…` | OK |

## 2. Candidate source / version

- **Spec version:** `auth.099.001.04 / workbook 1.3.1 + candidate schema`
- **Fields parsed:** 104
- **Normalizer:** `annex2-normalizer/1.1.0`
- **Scope:** {"asset_branch": "ResdtlRealEsttLn/PrfrmgLn", "cancellation_rows": "excluded", "nd_schema_cross_check": "enabled", "performance": "PRF", "schema_namespace": "urn:esma:xsd:DRAFT1auth.099.001.04", "sheet": "DRAFT1auth.099.001.04", "templates": ["ALL", "RRE"]}

| artefact | criticality | external version | sha256 | status |
| --- | --- | --- | --- | --- |
| `esma_annex2_message_workbook` | **gating** | 1.3.1 | `983e33c851a6621a…` | OK |
| `esma_annex2_reporting_instructions` | corroborating | UNKNOWN | `—` | SOURCE_CHECK_FAILED — no local copy vendored and retrieval is disabled |
| `esma_annex2_sample_message` | corroborating | UNKNOWN | `64e30a51b1019acb…` | OK |
| `esma_annex2_xml_schema` | **gating** | 1.3.0 | `3f4ed4e0030d0fbf…` | OK |

## 3. Did the authoritative source content change?

- **Gating sources** (the specification is derived from these): `SOURCE_CHANGED`
- **Corroborating sources** (tracked, nothing derived from them): `SOURCE_CHECK_FAILED`

Unverified sources:

- `esma_annex2_reporting_instructions` (corroborating, baseline) — SOURCE_CHECK_FAILED: no local copy vendored and retrieval is disabled
- `esma_annex2_reporting_instructions` (corroborating, candidate) — SOURCE_CHECK_FAILED: no local copy vendored and retrieval is disabled

> A **corroborating** source is unverified. No compared attribute is derived from it, so it does not change the determination about the machine-readable specification — but the obligation it carries has not been reviewed by this run.

- baseline source digest: `esma_annex2_message_workbook:983e33c851a6621af6b91e43bf032f6ecde5be1fc62770fd1c233789b287676b|esma_annex2_reporting_instructions:|esma_annex2_sample_message:64e30a51b1019acbea41a77575106af982ba13b63fcd0fb826da694bd404fe7e|esma_annex2_xml_schema:48e587ec5905ab322eb98e52a9989ed65194d3a1fd20f3f55aac33a7d492ab1c`
- candidate source digest: `esma_annex2_message_workbook:983e33c851a6621af6b91e43bf032f6ecde5be1fc62770fd1c233789b287676b|esma_annex2_reporting_instructions:|esma_annex2_sample_message:64e30a51b1019acbea41a77575106af982ba13b63fcd0fb826da694bd404fe7e|esma_annex2_xml_schema:3f4ed4e0030d0fbf41f40095e474013cee64a706afebd285765788dd09458d3a`

## 4. Did the parsed Annex 2 specification change?

**`SPEC_CHANGED`**

## 5. Regulatory deltas

| code | change | severity | confidence | old | new |
| --- | --- | --- | --- | --- | --- |
| `RREL42` | ENUM_CHANGED | MEDIUM | deterministic | {"enum_values": ["CAPP", "DISC", "FINX", "FLCA", "FLCF", "FLFL", "FLIF", "FXPR", "FXRL", "MODE", "OBLS", "OTHR", "SWIC"]} | {"enum_values": ["CAPP", "DISC", "FINX", "FLCA", "FLCF", "FLFL", "FLIF", "FXPR", "MODE", "OBLS", "OTHR", "SWIC"]} |

## 6. Likely Trakt implementation impact

Findings by status: `CONFIG_CHANGE_REQUIRED` × 1, `TEST_CHANGE_REQUIRED` × 1, `VALIDATION_CHANGE_REQUIRED` × 1

| code | change | component | status | locations | current |
| --- | --- | --- | --- | --- | --- |
| `RREL42` | ENUM_CHANGED | annex2_fixtures_and_tests | **TEST_CHANGE_REQUIRED** | `tests/fixtures/annex2_delivery_ready_no_npe.csv`, `tests/fixtures/annex2_projected_ci.csv`, `tests/test_annex2_delivery_normalizer.py`, `tests/test_onboarding_annex2_workflow.py` | {"existing_references": ["tests/fixtures/annex2_delivery_ready_no_npe.csv", "tests/fixtur… |
| `RREL42` | ENUM_CHANGED | enum_mapping | **CONFIG_CHANGE_REQUIRED** | `config/system/enum_mapping.yaml`, `config/regime/annex2_delivery_rules.yaml` | {"delivery_rule_enum_map": {"CAPP": "CAPP", "DISC": "DISC", "FINX": "FINX", "FLCA": "FLCA… |
| `RREL42` | ENUM_CHANGED | validation_rules | **VALIDATION_CHANGE_REQUIRED** | `config/regime/annex2_delivery_rules.yaml` | {"delivery_rule_enum_map": {"CAPP": "CAPP", "DISC": "DISC", "FINX": "FINX", "FLCA": "FLCA… |

## 7. Unresolved parser / review items

- **baseline** `RREC8` (attributes: —) — ND_BRANCH_PATH_INCONSISTENT: ND branch '/Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/ScrtstnRpt/UndrlygXpsrRcrd/UndrlygXpsrData/ResdtlRealEsttLn/PrfrmgLn/Coll/CollCmonData/Dtls/CmonData/Lien/NoDataOptn' is not nested under the value element '/Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/ScrtstnRpt/UndrlygXpsrRcrd/UndrlygXpsrData/ResdtlRealEsttLn/PrfrmgLn/Coll/CollCmonData/Dtls/CmonData/LienVal'
- **baseline** `RREL80` (attributes: —) — ND_BRANCH_PATH_INCONSISTENT: ND branch '/Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/ScrtstnRpt/UndrlygXpsrRcrd/UndrlygXpsrData/ResdtlRealEsttLn/PrfrmgLn/UndrlygXpsrCmonData/InstnlDtls/OrgnlLndr/LEI/NoDataOptn' is not nested under the value element '/Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/ScrtstnRpt/UndrlygXpsrRcrd/UndrlygXpsrData/ResdtlRealEsttLn/PrfrmgLn/UndrlygXpsrCmonData/InstnlDtls/OrgnlLndr/LEICd'
- **candidate** `RREC8` (attributes: —) — ND_BRANCH_PATH_INCONSISTENT: ND branch '/Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/ScrtstnRpt/UndrlygXpsrRcrd/UndrlygXpsrData/ResdtlRealEsttLn/PrfrmgLn/Coll/CollCmonData/Dtls/CmonData/Lien/NoDataOptn' is not nested under the value element '/Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/ScrtstnRpt/UndrlygXpsrRcrd/UndrlygXpsrData/ResdtlRealEsttLn/PrfrmgLn/Coll/CollCmonData/Dtls/CmonData/LienVal'
- **candidate** `RREL80` (attributes: —) — ND_BRANCH_PATH_INCONSISTENT: ND branch '/Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/ScrtstnRpt/UndrlygXpsrRcrd/UndrlygXpsrData/ResdtlRealEsttLn/PrfrmgLn/UndrlygXpsrCmonData/InstnlDtls/OrgnlLndr/LEI/NoDataOptn' is not nested under the value element '/Document/ScrtstnNonAsstBckdComrclPprUndrlygXpsrRpt/NewCrrctn/ScrtstnRpt/UndrlygXpsrRcrd/UndrlygXpsrData/ResdtlRealEsttLn/PrfrmgLn/UndrlygXpsrCmonData/InstnlDtls/OrgnlLndr/LEICd'

## 8. Evidence and provenance

Every delta in the JSON report carries `evidence[]` entries naming the artefact and the exact locator (`sheet=<name>;row=<n>` for the workbook, `xsd:simpleType/<name>` for a schema code list).

- **demonstration:** candidate schema withdraws FXRL from InterestRateType2Code; the workbook is byte-identical on both sides
- **mutating_writes:** none — Stage 1 is observational
- **source_manifest:** /home/user/trakt/config/regulatory_watch/esma_annex2_sources.yaml

---

*Stage 1 is observational. No Annex 2 configuration, regime version, projection, enum behaviour, ND behaviour or XML output was created or modified by this run.*
