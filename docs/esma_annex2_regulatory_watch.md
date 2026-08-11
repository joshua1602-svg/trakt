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
| `value_paths` | every published path under the element node, excluding `NoDataOptn` (evidence only; not compared) |
| `value_leaves` | every typed leaf as `{relative_path, data_type, pattern}`, keyed **relative** to the element node |
| `data_type` / `format_pattern` | the first value leaf in sheet order carrying a `DATA TYPE / CODE` |
| `nd_allowed` | the **union** of the XSD enumerations of every published `NoData` leaf under the code's `NoDataOptn` branch(es), **cross-checked against the schema's own definition**; `[]` when neither publishes an ND branch |
| `enum_values` | the XSD enumeration of the value leaf's code-list type |
| `mandatory` | minimum occurrence of the element node's multiplicity |
| `validation_rules` | verbatim `Validation rule` cells, de-duplicated in sheet order |
| `order_index` | the code's position in workbook sheet order |

Nothing is inferred. An attribute that cannot be derived is set to `UNKNOWN`
and named in `SpecField.unresolved`; any delta touching it is downgraded to
`review-required`, and a comparison over a spec with unresolved attributes can
never report `SPEC_UNCHANGED`.

### Why the ND union rule is correct (not merely agreed-with)

`NoDataOptn` is never a bare element. In the schema it is always typed as one
of three complex types, and every one of them is an **`xs:choice`** whose
branches are the alternative ways of populating a single ND slot:

| choice type | branches | permitted ND set |
| --- | --- | --- |
| `NoDataJustification1Choice` | `NoData: Justification1Code {ND1,ND2,ND3,ND5}` or `NoData4: NoDataFour1` | ND1–ND5 |
| `NoDataJustification4Choice` | `NoData: Justification2Code {ND1,ND2,ND3}` or `NoData4: NoDataFour1` | ND1–ND4 |
| `NoDataJustification3Choice` | `NoData: Justification3Code {ND5}` | ND5 |

`NoDataFour1` is an `xs:sequence` of `(Dt: ISODate, NoData:
Justification4Code {ND4})` — ND4 means *"data collected but will only be
available from `<date>`"*, which is why that branch carries a date. It is a
branch of the choice, not a separate field.

Over a choice, **the union of the branch value sets IS the permitted value
set**. The union rule is therefore the schema's own semantics, not a heuristic
that happens to fit.

Distribution across the 104 in-scope codes: 41 × `Justification1Choice`,
17 × `Justification4Choice`, 29 × `Justification3Choice`, 17 with no
`NoDataOptn` branch at all.

### The schema cross-check (why this stays true on a future artefact)

Agreement with today's configuration proves today's answer. It cannot prove the
*method* survives a future workbook — and the workbook does contain errors (see
RREC8/RREL80 below). So `SchemaNdModel` resolves each derived element path
against the real XSD element tree and independently computes the permitted ND
set. On disagreement the run **fails closed**: `nd_allowed` becomes unresolved
with an `ND_SCHEMA_DISAGREEMENT` warning, and neither source is trusted. An
element path the schema does not define raises `XSD_PATH_UNRESOLVED`.

On the vendored artefacts the cross-check is clean for all 104 codes, and it
simultaneously proves the *element-path* derivation: every path the normalizer
derived is a path the schema defines.

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

**Source criticality** — `gating` · `corroborating`. Declared **per artefact**
in the manifest and **required**, so classifying a source as non-gating is
always a visible decision. Gating = the normalized specification is derived from
it (the workbook and the XSD). Corroborating = authoritative and tracked, but no
compared attribute comes from it (the sample message, the reporting
instructions).

**Source** — `SOURCE_UNCHANGED` · `SOURCE_CHANGED` · `SOURCE_CHECK_FAILED` ·
`SOURCE_MISSING`, reported separately for gating and corroborating sources plus
a `combined` (strictest) value.

**Specification** — `SPEC_UNCHANGED` · `SPEC_CHANGED` · `SPEC_UNDETERMINED`

**Outcome** — `CURRENT` · `SOURCE_CHANGED_SPEC_UNCHANGED` ·
`REGULATORY_SPEC_CHANGED` · `SPEC_CHANGE_UNDETERMINED` · `WATCH_INCONCLUSIVE`

**`specification_current`** — `yes` / `no` / `unknown`. The unambiguous
headline, independent of which outcome fired. It is a claim about the
**machine-readable** specification only.

Semantics:

* `CURRENT` only when **every gating source is verified** and no regulatory
  delta exists;
* `REGULATORY_SPEC_CHANGED` whenever a deterministic delta exists — reported
  even if a gating source is unverified, since an unchecked source can only add
  findings, never retract a confirmed one;
* `WATCH_INCONCLUSIVE` when a **gating** source cannot be verified, or parsing
  is indeterminate;
* a **corroborating** failure is always listed in `unverified_sources`, shown in
  the Markdown, and raises the `combined` source status — but does not withhold
  the machine-readable determination.

A hash change alone is never a regulatory change: source-byte comparison and
parsed-requirement comparison are separate axes and both are reported.

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
2. **The technical reporting instructions are not vendored, and are classified
   corroborating.** Nothing the comparator compares is derived from them, so
   they cannot decide the machine-readable determination — but they *can* carry
   a real obligation change: narrative guidance on how a field must be
   *populated* can move without the workbook or XSD changing a byte. Stage 1
   does not model interpretation and does not claim to detect that. Every run
   lists the artefact in `unverified_sources` and says so in the Markdown.
   Vendoring it (or enabling retrieval) closes the gap. **`CURRENT` means the
   machine-readable specification is current; it does not mean every ESMA
   obligation has been reviewed.**
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
5. **One real metadata conflict between ESMA workbooks, fully explained.**
   ESMA publishes two multi-code cells — `RREL1|RREC1` and `RREL3|RREC2` — where
   the collateral section cross-references the same element as the exposure
   section. A shared cell carries one `RTS Field name`, so RREC2 inherits
   RREL3's label ("New Underlying Exposure Identifier") while the template
   workbook names it "Underlying Exposure Identifier". Consequence: a label
   change on RREL3 would also raise `FIELD_DESCRIPTION_CHANGED` on RREC2. Both
   codes genuinely map to that element, so this is accurate, not a defect.
   Asserted in the tests so a *new* divergence fails.
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

### Impact assessment is a controlled decision table

`regulatory_watch/impact.py` is a declarative table — one auditable row per
(change type, component) — rather than procedural branching, so the control can
be reviewed as data. `decision_table()` renders it. Two boundaries are
structural: the comparator's factual output cannot be created, suppressed or
reworded by the impact pass, and **no status authorises an automatic change** —
every status names a location for a human. Where the evidence does not decide,
the row resolves to `MANUAL_REVIEW_REQUIRED`; `NO_IMPLEMENTATION_CHANGE` is
reserved for cases where Trakt demonstrably holds nothing that depends on the
changed attribute. Configuration that *predates* the authority publishing a code
is treated as review-required, never as reassurance.

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
