# ESMA Annex 2 Regulatory Watch — Stage 1

**Status: Stage 1 complete. Observational only.**

Stage 1 answers one narrow question, deterministically and offline:

> Has the authoritative ESMA Annex 2 regulatory specification changed relative
> to the version Trakt currently implements, and if so which existing Trakt
> Annex 2 components are likely affected?

It preserves evidence of the source versions and emits a structured delta
report. It does **not** update configuration, promote a regime version,
generate XML, notify anyone, or use a model anywhere in the comparison path.

---

## 1. Architecture

```
config/regulatory_watch/esma_annex2_sources.yaml   explicit ESMA allowlist
        │
        ▼
  manifest.py ── retrieval.py (opt-in, isolated, off by default)
        │
        ▼
  snapshots.py            immutable content-addressed snapshot + sha256
        │
        ▼
  annex2_spec.py          workbook + XSD ─► NormalizedSpec (per-field, with
        │                                    provenance and `unresolved`)
        ▼
  comparator.py           spec A vs spec B ─► SpecDelta[]  (11 change types)
        │
        ▼
  impact.py               delta ─► ImpactFinding[] against the LIVE Trakt
        │                          Annex 2 config (read-only)
        ▼
  report.py               JSON (UI-agnostic contract) + Markdown
```

`changelog.py` sits alongside the normalizer: it parses ESMA's own published
amendment history out of the workbook and is used as corroborating evidence
and for historical baseline reconstruction. It never overrides a parsed value.

### Files

| path | role |
| --- | --- |
| `config/regulatory_watch/esma_annex2_sources.yaml` | authoritative source allowlist |
| `regulatory_watch/contracts.py` | data contract + controlled vocabularies |
| `regulatory_watch/manifest.py` | allowlist loading/validation |
| `regulatory_watch/retrieval.py` | optional, isolated network fetch |
| `regulatory_watch/snapshots.py` | immutable sha256 snapshot store |
| `regulatory_watch/annex2_spec.py` | deterministic Annex 2 normalizer |
| `regulatory_watch/changelog.py` | ESMA published amendment history |
| `regulatory_watch/comparator.py` | deterministic spec-vs-spec diff |
| `regulatory_watch/impact.py` | delta → Trakt component impact |
| `regulatory_watch/report.py` | JSON + Markdown emitters |
| `regulatory_watch/cli.py` | `snapshot` / `compare` commands |
| `scripts/run_regulatory_watch_demo.py` | end-to-end local demonstration |

---

## 2. Sources of truth

Stage 1 treats **primary ESMA artefacts only** as authoritative. The allowlist
is explicit; there is no crawling and no link-following.

| artefact | why it is authoritative | what is derived from it |
| --- | --- | --- |
| `DRAFT1auth.099.001.04_non-ABCP Underlying Exposure Report_Version_1.3.1.xlsx` | ESMA's message/mapping workbook for the auth.099 non-ABCP underlying-exposure report — the Annex 2 message. The repo already treats it as the authoritative crosswalk (`scripts/build_annex2_field_xsd_path_map.py`, Gate 5 mapping specs). | field code, label, definition, XML tag, full XML path, multiplicity, data type / code-list name, pattern, validation-rule text, field ordering, ND branch structure |
| `DRAFT1auth.099.001.04_1.3.0.xsd` | the ESMA schema the submission is validated against | enumeration values for coded fields; the `NoDataAllowedJustification{1,2,3,4}Code` vocabularies that state exactly which ND options each field permits; target namespace |
| `DRAFT1auth.099.001.04_non-ABCP Underlying Exposure Report.xml` | ESMA sample message | hash-tracked only — Stage 1 never derives a requirement from a sample |
| ESMA technical reporting instructions | authoritative, **not vendored here** | nothing — declared in the allowlist so its absence is reported as `SOURCE_CHECK_FAILED` rather than silently omitted |

### Scope

The residential-real-estate **performing** branch
(`ResdtlRealEsttLn/PrfrmgLn`), templates `ALL`/`RRE`, cancellation (`/Cxl/`)
rows excluded — the same branch the live Trakt Annex 2 pathway targets. That
yields **104** Annex 2 codes.

---

## 3. Derivation rules (all deterministic)

| attribute | derivation |
| --- | --- |
| `xml_path` | the element node: the longest common ancestor of the code's **value** rows (the ND branch is a sibling for a few codes, so it is excluded from this calculation) |
| `value_paths` | every published path under the element node, excluding `NoDataOptn` |
| `data_type` / `format_pattern` | the first value leaf in sheet order carrying a `DATA TYPE / CODE` |
| `nd_allowed` | the **union** of the XSD enumerations of every published `NoData` leaf under the code's `NoDataOptn` branch(es); `[]` when the workbook publishes no ND branch |
| `enum_values` | the XSD enumeration of the value leaf's code-list type |
| `mandatory` | minimum occurrence of the element node's multiplicity |
| `validation_rules` | verbatim `Validation rule` cells, de-duplicated in sheet order |
| `order_index` | the code's position in workbook sheet order |

Nothing is inferred. An attribute that cannot be derived is set to `UNKNOWN`
and named in `SpecField.unresolved`; any delta touching it is downgraded to
`review-required`, and a comparison over a spec with unresolved attributes can
never report `SPEC_UNCHANGED`.

### Independent corroboration

The normalizer derives ND eligibility from the message workbook + XSD. The
committed `config/regime/annex2_field_universe.yaml` derives it from the ESMA
*template* workbook (which is **not** vendored in this repository). The two
independent derivations agree on **all 104** codes — pinned by
`tests/test_regulatory_watch_spec_parser.py`. The normalizer also independently
reproduces the documented fact that RREL18 / RREL28 / RREC22 carry no XML path
(they are currency *attributes*, per the delivery-rules representation block).

---

## 4. Statuses

**Source** — `SOURCE_UNCHANGED` · `SOURCE_CHANGED` · `SOURCE_CHECK_FAILED` ·
`SOURCE_MISSING`

**Specification** — `SPEC_UNCHANGED` · `SPEC_CHANGED` · `SPEC_UNDETERMINED`

**Outcome** — `NO_CHANGE_DETECTED` · `SOURCE_CHANGED_SPEC_UNCHANGED` ·
`REGULATORY_SPEC_CHANGED` · `SPEC_CHANGE_UNDETERMINED` · `WATCH_INCONCLUSIVE`

A hash change alone is never a regulatory change: source-byte comparison and
parsed-requirement comparison are separate axes and are both reported. A failed
source check can never resolve to `NO_CHANGE_DETECTED` or
`SOURCE_CHANGED_SPEC_UNCHANGED`. A confirmed spec change is still reported when
another source is unchecked — an unchecked source can only add findings.

**Change types** — `FIELD_ADDED`, `FIELD_REMOVED`, `FIELD_DESCRIPTION_CHANGED`,
`FORMAT_CHANGED`, `MANDATORY_STATUS_CHANGED`, `ND_PERMISSION_CHANGED`,
`ENUM_CHANGED`, `XML_PATH_CHANGED`, `MULTIPLICITY_CHANGED`, `ORDER_CHANGED`,
`VALIDATION_RULE_CHANGED`.

**Impact statuses** — `NO_IMPLEMENTATION_CHANGE`, `CONFIG_CHANGE_REQUIRED`,
`VALIDATION_CHANGE_REQUIRED`, `XML_CHANGE_REQUIRED`, `TEST_CHANGE_REQUIRED`,
`MANUAL_REVIEW_REQUIRED`. One delta may produce several findings.

---

## 5. Running it

```bash
# capture a baseline from the vendored artefacts
python -m regulatory_watch.cli snapshot --out-dir output/regulatory_watch/baseline

# compare a candidate against it
python -m regulatory_watch.cli compare \
    --out-dir output/regulatory_watch/run \
    --baseline-spec output/regulatory_watch/baseline/annex2_spec.json \
    --candidate-workbook <new workbook.xlsx> \
    --candidate-schema  <new schema.xsd>

# end-to-end demonstration with one controlled regulatory change
python scripts/run_regulatory_watch_demo.py
```

Exit codes: `2` = a source could not be parsed (never "no change"), `3` = the
comparison was refused (regime or parser-version mismatch).

---

## 6. Known limitations

1. **No second historical source version is vendored.** The repository holds
   one version of the ESMA artefacts (workbook 1.3.1 / schema 1.3.0), so a true
   "parse version N, parse version N+1, diff" replay cannot be run here and no
   such difference has been invented. What *is* replayed is ESMA's own
   published change log (see below). A real historical snapshot can be dropped
   in as a normalized-spec JSON with **no comparator change**.
2. **The technical reporting instructions are not vendored.** Every run
   therefore reports `SOURCE_CHECK_FAILED` for that artefact and resolves to
   `WATCH_INCONCLUSIVE` unless a spec change is confirmed. This is accurate,
   not a defect: Trakt genuinely cannot assert that source is current. Vendoring
   it (or enabling retrieval) removes the condition.
3. **The vendored artefacts are ESMA DRAFTs** (`DRAFT1auth`, namespace
   `urn:esma:xsd:DRAFT1auth.099.001.04`). The watch compares what the repo
   actually relies on; confirming the final published auth.099.001.04 package
   is a separate, pre-existing repository issue (see
   `docs/annex2_production_xml_structure_contract.md`).
4. **Two real ESMA workbook inconsistencies are reported, not silently fixed.**
   RREC8 (Lien) and RREL80 (Original Lender LEI) publish their ND branch as a
   *sibling* of the value branch (`.../Lien/NoDataOptn` vs `.../LienVal/Lien`).
   The normalizer resolves both correctly and emits a
   `ND_BRANCH_PATH_INCONSISTENT` warning.
5. **One real metadata conflict between ESMA workbooks.** RREC2's label is
   "New Underlying Exposure Identifier" in the message workbook and
   "Underlying Exposure Identifier" in the committed field universe. Asserted in
   the tests so a *new* divergence fails.
6. **Scope is the performing residential branch only.** Non-performing (`NPRF`)
   and other asset classes are out of Stage 1 scope, matching the live pathway.
7. **The XSD is normalized only for code lists and namespace.** Full schema
   structural diffing is not part of Stage 1; XML paths come from the workbook,
   which is what the repo's existing path-map builder also does.

### Historical replay actually performed

ESMA ships its amendment history inside the workbook
(`Change log - Summary`, `Change log - XML path`). Stage 1 uses it for:

* **vocabulary coverage** — every amendment type ESMA has actually used
  (element tag addition/removal/move/amendment, element path amendment, code
  change, element codes added, multiplicity change, XLS-only rule edits) maps
  onto a comparator change type. The test fails if ESMA history contains a
  change the comparator cannot express.
* **baseline reconstruction** — v1.3.0 amendment #78 ("Gross Annual Rental
  income" element added for residential real estate) publishes its new XML
  paths in full and is therefore reversible. Reverse-applying it produces a
  genuine pre-1.3.0 baseline. The result is a real-world proof of the
  *source changed / Annex 2 spec unchanged* case: ESMA added the element under
  `ResdtlRealEsttLn/**NonPrfrmgLn**` with no RTS field code, so it lies outside
  the performing branch Trakt projects, and the comparator correctly reports
  zero Annex 2 deltas.
* Amendments whose pre-change value ESMA does not publish (e.g. v1.3.1 #79,
  which deleted validation rules on RREL24 / RREL36 / RREL73) are reported as
  **not reconstructible**, never guessed.

---

## 7. What Stage 1 deliberately does not do

No OCC Regulatory Config UI. No human approval workflow. No versioned config
promotion. No autonomous config updates. No Teams / Copilot / email
notifications. No FCA. No Annex 12. No generic regulation framework. No LLM in
the comparison path.

## 8. Stage 2 (not implemented)

1. **OCC Regulatory Config view** — consumes the JSON report as-is; the
   contract is already UI-agnostic and versioned (`contract_version`).
2. **Human approval** — a reviewer accepts/rejects each delta and each impact
   finding; today every finding is advisory.
3. **Versioned config promotion** — turning an approved delta into a new,
   explicitly-activated Annex 2 regime version, with the current version left
   untouched until promotion.
4. **Scheduled retrieval** — enabling `retrieval_enabled` with a real fetcher
   plus a cadence, and alerting on `SOURCE_CHECK_FAILED`.
