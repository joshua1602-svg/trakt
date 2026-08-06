# Annex 2 delivery — migration status

Agreed direction: **evolve the existing Annex 2 delivery tail into the governed
production route.** Retain the shared projector, repair and govern the existing
normaliser and XML builder, eliminate silent fabrication, surface ND usage, and
integrate the delivery stages into the governed OCC pathway. **Do not build a
second Annex 2 implementation.**

> **XML delivery is NOT part of the governed production route today.** The
> production-invoked pathway ends at the projected Annex 2 CSV. The delivery
> tail is proven but not promoted, and this document does not claim otherwise.
> `config/delivery/annex2_field_xsd_path_map.yaml` still records
> `production_ready: 0/107` and `do_not_generate_production_xml: true`.

## Current Annex 2 status

| Property | State |
|---|---|
| Projector | **deterministic** |
| Delivery normaliser | **deterministic** |
| XML builder | **deterministic** |
| Delivery-rule defaults | **explicit and declared** (46 rules) |
| Builder ND insertion | **zero** |
| Builder fabrication | **zero** |
| Latent fabrication anywhere in the delivery path | **zero** |
| Instrumentation | **complete** — every ND and every value transform attributed |
| Benchmark | **reproducible** — current baseline `8018abb9…3da5`; byte-identical to `a21f8a4c…d685d` apart from the deliberate RREL20/RREL21 correction |
| Production collateral mapping | **corrected** — no demo overlay required |

**Remaining work is operational governance (Phase 3), not XML correctness.**
The delivery tail produces a schema-valid submission in which every value comes
from client data or a declared rule. What it still lacks is *governance*: it is
not invoked by the production route, its warnings are not surfaced to an
operator, and its artefacts are not staged behind approval. Those are the
Phase 3 questions. Two builder functions in the non-performing branch still hold
reporting policy — `_ensure_hstrcl_colltn_nd_defaults` and
`_ensure_nprf_nonprfrmgdata_defaults`. Neither fires on this benchmark (it is
PRF-mode with zero `NonPrfrmgLn` nodes), so there is no measured behaviour to
preserve and no evidence on which to choose the right rule shape; both are
instrumented and will report the moment they do.

---

## Phase status

| Phase | Scope | Status |
|---|---|---|
| **0** | Reproduce the committed 105/107 benchmark at HEAD | ✅ **Complete — byte-identical** |
| **1** | Make ND usage and value coercion visible, output-neutral | ✅ **Complete** |
| **2** | Remove the RREL12 fabrication; move RREL20/RREL21 ND into declared rules | ✅ **Complete — byte-identical** |
| 3 | Promote the delivery stages into the governed OCC pathway | ❌ Not implemented |
| 4 | Retire duplicate configuration; consolidate enum truth | ❌ Not implemented |
| 5 | Replace the wide-row builder (multi-collateral) | ❌ Not implemented |

## Phase 0 — benchmark reproduction (complete)

The route that produced the historical result — `regime_projector.py` →
`annex2_delivery_normalizer.py` → `xml_builder_annex2.py` + XSD — reproduces at
HEAD **byte-identically**:

| Measure | Committed (`0ed7b4c`, 2026-07-26) | At HEAD | |
|---|---|---|---|
| Fields projected & delivered | 105 of 107 | 105 of 107 | ✅ |
| Omitted codes | RREL20, RREL21 (Optional) | RREL20, RREL21 | ✅ |
| Exposure records | 11,035 | 11,035 | ✅ |
| XML size | 208,822,291 bytes | 208,822,291 bytes | ✅ |
| SHA-256 | `a21f8a4c…d685d` | `a21f8a4c…d685d` | ✅ |
| SHA-256 *(after the RREL20/21 correction)* | — | **`8018abb9…3da5`** | new baseline |
| XSD validation | PASSED | PASSED | ✅ |

The same SHA-256 held through Phase 2 and every subsequent change **except one**:
the deliberate RREL20/RREL21 ND5 → ND1 regulatory correction, which re-baselined
the benchmark to **`8018abb960e2472b80bf928c354d712a604c1b05f3aed1997942dd86b0a63da5`**
(size unchanged at 208,822,291 bytes). A byte-level diff of the two files shows
**22,070 differing bytes, every one a `5` → `1`**, distributed exactly 11,035
under `IncmVal` and 11,035 under `Vrfctn`. Nothing else in the 208 MB submission
changed.

Reproduce (writes only to the scratch root — the repository is untouched):

```bash
TRAKT_DEMO_ROOT=<scratch> python -m demo_platform.run_demo --generate --onboard --orchestrate
TRAKT_DEMO_ROOT=<scratch> python -m demo_platform.run_demo --artefacts --no-reset
sha256sum <scratch>/artefacts/regulatory/annex2_submission.xml
```

**What 105/107 measures:** field codes projected *and* delivered into the XML,
out of the 107-code field universe. It is **not** mapping certification, not
workbook sign-off, and not submission readiness. 54 of the 105 delivered values
in a sample record are permitted no-data codes.

Since Phase 2 the count is stated as a split rather than as one number, because
"a field is present" and "a field carries client data" are different claims:

> **105** fields populated from projected/source data ·
> **2** fields populated by governed delivery rules as `ND1` ·
> **107** fields represented in the final Annex 2 submission

The delivery-ready CSV now has 107 columns where it had 105, because RREL20 and
RREL21 are created by rule instead of being injected by the builder. **Coverage
did not increase** — the same two fields occupy the same XML positions, now
carrying `ND1` (see the correction recorded below) and visible one stage
earlier. Gate 4b reports the split
in `*_delivery_report.json` → `field_provenance`, and the demo manifest carries
`fieldsFromProjectedSource` / `fieldsFromDeliveryRule` beside `fields`.

## Production collateral mapping no longer depends on the demo overlay

`config/system/enum_mapping.yaml::ESMA_Annex2.collateral_type` previously
targeted `R1`/`R2`/`C1`/`C2`, none of which are members of `CollateralType7Code`
in the delivery XSD. This was **not** dead configuration: the projector's
synonym resolver discards a synonym whose *target* is absent from the regime
table, so real-data values reached Gate 5 unmapped and the XSD rejected them.
`demo_platform/artefacts.py::_demo_enum_mapping` worked around it with a
per-run overlay.

Production configuration now resolves this on its own:

| Input | Before | After |
|---|---|---|
| `Residential property` (demo canonical) | `Residential property` ✗ | `RBLD` ✅ |
| `Residential Building (RBLD)` (real ERE canonical) | passed through ✗ | `RBLD` ✅ |
| `HOUSE` / `FLAT` | `R1` / `R2` ✗ | `RBLD` ✅ |
| `OFFICE` / `INDUSTRIAL` | `C2` ✗ | `CBLD` / `IBLD` ✅ |
| `MIXED_USE` | `C1` ✗ | `MIXD` ✅ |
| `LAND` / `AGRICULTURAL` / `DEVELOPMENT` | `OTHR` ✅ | `OTHR` ✅ *(unchanged — see below)* |

`LAND`/`AGRICULTURAL`/`DEVELOPMENT` were briefly moved to `OTRE` and have been
**reverted**. `OTHR` was already XSD-valid, no failing case required the change,
and "other real estate" is a semantic reclassification needing its own
regulatory decision.

Labelled forms (`Label (CODE)`) are **explicit keys**, not a parenthetical-code
parser: a parser would have to validate the extracted code against this field's
enumeration before it could be trusted. An unrecognised value stays
unrecognised so the XSD rejects it visibly, rather than becoming a
plausible-looking code.

The demo overlay is now a no-op for `collateral_type`
(`tests/test_annex2_collateral_projection.py` pins this).

## Phase 1 — delivery behaviour is now visible (complete)

**Output-neutral by construction.** Removing every instrumentation call would
leave the delivery-ready CSV and the XML byte-identical. Verified end-to-end:
the benchmark SHA-256 is unchanged, and the delivery-ready CSV is byte-identical
when re-normalised.

Three categories are reported **separately**, because they carry different
weight: upstream truth, a governed decision, and an ungoverned one.

> The figures below are **as measured at the end of Phase 1**, kept as the
> record of what Phase 1 found. Phase 2 moved the third category into the
> second; the current figures are in the Phase 2 table further down.

| Category | Where reported | Phase 1 benchmark value |
|---|---|---|
| ND already present in the projected input | `*_delivery_report.json` → `delivery_instrumentation.nd_present_in_input` | **154,506** (ND1 22,082 / ND5 132,424) |
| ND applied by **declared delivery rules** | `…nd_applied_by_rules` | **441,367** (ND1 165,527 / ND5 275,840) |
| ND **injected by the builder** | `annex2_submission_delivery_instrumentation.json` → `nd_injected_by_builder` | **22,070** (all ND5) |
| Rule-declared value transforms | `…coercions` (normaliser) | 121,355 |
| **Builder value coercions (fabrication)** | `…coercions` (builder) | **0** |

The accounting closes exactly:

```
154,506 (input) + 441,367 (rules) = 595,873 ND cells in the delivery-ready CSV
595,873 + 22,070 (builder)        = 617,943 ND nodes in the XML   ✅
```

Builder injection is attributed by code, field and branch:

```json
"nd_injected_by_builder": {
  "total": 22070,
  "by_code":   {"ND5": 22070},
  "by_field":  {"RREL20": 11035, "RREL21": 11035},
  "by_branch": {"ScndryOblgrIncm/IncmVal": 11035,
                "ScndryOblgrIncm/Vrfctn":  11035}
}
```

**This explains a previously unexplained result:** RREL20 and RREL21 are the two
codes *not* projected or delivered — yet they appear in the XML carrying `ND5`.
The builder inserts them. Until Phase 1 that was invisible. **Phase 2 acted on
this finding**: the injection is gone and the two codes are now declared rules.

A run with no coercions **records zero explicitly**; an absent entry is never
treated as evidence of absence:

```json
"coercions": {"count": 0, "records": [], "truncated": false, "record_cap": 5000}
```

Counts are always exact. Individual records are capped at 5,000 with
`truncated: true` when exceeded, so a report stays usable without under-stating.

## Phase 2 — nothing is invented; the ND answer is configuration (complete)

Phase 1 made two behaviours visible. Phase 2 removes one and governs the other.
**Output-neutral, proven end-to-end:** the benchmark SHA-256 is unchanged.

### 1. The RREL12 fabrication is gone

`_coerce_record_value_for_branch` substituted the hardcoded string `"2026"` for
any RREL12 value that was not an ISO year. It now returns an empty string, so
the value routes to the NoData branch the mapping permits — the behaviour its
own docstring already specified — and records the event with the original
value, field code, exposure identifier and reason.

It did not fire on this benchmark (every RREL12 value is `2021`), so this was a
latent landmine rather than an active corruption. The builder report now asserts
its absence explicitly rather than leaving it unsaid:

```json
"routed_to_nodata":   {"count": 0, "records": [], "truncated": false, "record_cap": 5000},
"fabricated_values":  {"count": 0, "note": "Phase 2 removed the only fabrication path …"}
```

### 2. RREL20 / RREL21 moved from builder code into declared rules

The builder used to write `ND5` into `ScndryOblgrIncm/IncmVal` and
`ScndryOblgrIncm/Vrfctn` because the XSD mandates the element and the projection
carried no column. That is a **regulatory judgement** — "there is no secondary
obligor, so this is not applicable" — and it was expressed as two lines of
Python inside an XML writer, where no reviewer would look for it.

It is now two rules in `config/regime/annex2_delivery_rules.yaml`:

```yaml
RREL20:
  workbook_semantic: ScndryOblgrIncm/IncmVal
  mandatory: false
  enforce_presence: false
  nd_allowed: [ND1, ND2, ND3, ND4, ND5]
  default_allowed: true
  default_value: ND1      # not collected — outside ERM underwriting
```

Gate 4b creates a column for any absent field whose rule authorises a default.
The mechanism is **entirely rule-driven** — no field code appears in the
normaliser — and a field that sets `enforce_presence` is still reported missing
rather than quietly filled.

RREL21 also gained an `enum_map`. It is a workbook `{LIST}` field
(`SCRT`/`SCNF`/`VRFD`/`NVRF`/`SCRG`/`OTHR`) and the first draft of the rule left
it unconstrained — which the onboarding enum reconciliation flagged as
`unconstrained_no_enum_map`. That is the same latent defect Phase 2 exists to
remove, one source column away from firing. The map is the **identity** over the
workbook's own six codes, matching the sibling RREL19: it constrains the field
without inventing a translation from a source vocabulary that does not yet
exist. The regime rule count is therefore **70**, not 68.

The builder no longer injects. If RREL20/RREL21 reach it with neither a value
nor a NoData branch it now **raises**, naming the rules file, instead of
inventing an answer:

```
[Gate 5] RREL20 (secondary obligor income) reached the builder with neither a
value nor a NoData branch. … Declare RREL20 in
config/regime/annex2_delivery_rules.yaml with default_allowed: true and a
default_value your product rationale supports (ND1 = not collected because the
underwriting criteria did not require it; ND5 = not applicable to this product).
```

### Which no-data code — RESOLVED as ND1 (regulatory correction)

**Equity-release loans commonly have joint borrowers**, so a secondary obligor
usually *does* exist. An earlier version of this document justified `ND5` on the
basis that no secondary obligor exists. **That reasoning was wrong** and has
been removed everywhere it appeared, including a production error message.

The real product fact is different: borrower income — primary and secondary
alike — is generally not part of equity-release underwriting, which runs on age,
property value and LTV. The lender does not collect it because the underwriting
criteria do not require it. **That is ND1, not ND5.**

| Evidence | Finding |
|---|---|
| `standards_library.yaml` | **ND1 = "Data not collected as not required by the lending or underwriting criteria"** — a verbatim description of the equity-release position. ND5 = "Not applicable". |
| RREL16 (Primary Income) | Carries the **identical** workbook wording ("…income used to underwrite the underlying exposure…") and ESMA sets `nd5_allowed: **False**`. The standard-setter has already rejected the reading that "income was not used to underwrite" makes an income field not-applicable — so the secondary field cannot take a different answer on the same rationale. |
| `product_defaults_ERM.yaml` | Applies **ND1 uniformly** to the whole income family (RREL16/17/18/19 **and** RREL20/21), and reserves ND5 for features the product genuinely lacks — `maturity_date` ("No fixed term"), `scheduled_principal_payment_frequency` ("No payments"). |
| Provenance of the previous ND5 | **Inherited from the pre-Phase-2 builder**, which hard-coded it. Phase 2 preserved the value to keep the benchmark byte-identical — not because ND5 had been established as correct. The benchmark was therefore never evidence for it. |

**This is a deliberate regulatory correctness correction, not a regression.** It
is the one change in this work that intentionally alters the submission. Both
configuration layers now declare ND1 and are asserted to agree by
`tests/test_annex2_secondary_income_applicability.py`.

`ND5` remains **permitted** for these codes (the workbook allows it) but is no
longer the default: a book whose secondary-income field genuinely does not apply
can declare it without a code change.

The properties that hold regardless of the code, all pinned by test:

* a **real supplied value is never overwritten** by a default — a joint-borrower
  loan that does report secondary income delivers the reported figure;
* **borrower count alone never determines the code** — a joint-borrower loan
  with no income data reports exactly what a single-borrower loan reports,
  because the rationale is the underwriting methodology, not the obligor count;
* the decision is **configuration, never Python** — no module names RREL20 or
  RREL21 against an ND code;
* **canonical truth carries no regime ND code**; ND is a delivery decision.

### The ND accounting after the move

The 22,070 builder-injected nodes moved into the rules category exactly. Nothing
appeared, nothing vanished:

| Category | Phase 1 | Phase 2 |
|---|---|---|
| ND present in the projected input | 154,506 | 154,506 |
| ND applied by **declared delivery rules** | 441,367 | **463,437** |
| ND **injected by the builder** | 22,070 | **0** |
| **ND nodes in the XML** | **617,943** | **617,943** ✅ |

```
154,506 + 463,437 = 617,943 ND cells in the delivery-ready CSV
617,943 +       0 = 617,943 ND nodes in the XML   ✅
```

The CSV total and the XML total are now the same number, because the builder
contributes nothing of its own.

### What Phase 2 deliberately does NOT do

* **The non-performing historical-collection ND logic is unchanged.**
  `_ensure_hstrcl_colltn_nd_defaults` still writes 144 ND5 nodes per record from
  builder code. It did not fire on this benchmark — the submission is PRF-mode
  with zero `NonPrfrmgLn` nodes — so there is no measured behaviour to preserve
  and no evidence on which to choose the right rule shape. Moving it blind would
  be guesswork dressed as governance. It stays instrumented, and it is named
  here as outstanding rather than left implicit. It is the one remaining
  builder-side ND decision.
* **It does not promote the delivery tail into the production route.** Phase 2
  changed where a decision lives, not which pipeline runs. XML delivery is still
  outside the governed OCC pathway (Phase 3).
* **It does not surface builder ND counts as operator warnings.** They are
  reported in the instrumentation JSON; wiring them to an operator surface
  belongs with the governed stages in Phase 3.
* **It does not regenerate the committed demo fixture.**
  `demo-video/public/fixtures/demo_manifest.json` still records `"fields": 105`
  from the `0ed7b4c` run. That number is correct *for that run*. A live run now
  reports `fields: 107` with `fieldsFromProjectedSource: 105` and
  `fieldsFromDeliveryRule: 2`. Regenerating the fixture means the React surface
  reading `fields` would show 107 without the split beside it — an overstatement
  of coverage — so the fixture and its consumer should be updated together, and
  that is React work this phase was scoped out of.

## Phase 2b — the last fabrication, and a repository-wide audit (complete)

Phase 2 removed the builder-side fabrication and recorded a second one it had
been scoped out of. Phase 2b removes that one too and then goes looking for
others. **Output-neutral: the benchmark SHA-256 is unchanged.**

### The RREL12 `geography_map` is gone

`config/regime/annex2_delivery_rules.yaml` declared a `transform.geography_map`
for RREL12 mapping ten region *names* (`London`, `Scotland`, …) onto the literal
classification year `"2026"`. Three things settled it:

1. **It has no authoritative basis.** The workbook says only "Enter the year of
   the NUTS3 classification used … e.g. 2013 for NUTS3 2013". A region name
   cannot imply a vintage — `West Midlands` is equally consistent with NUTS3
   2013, 2016 or 2021. The authoritative source is client configuration
   (`nuts_classification_year`), which Gate 2 writes into
   `geographic_region_classification`.
2. **It contradicted the stage that populates the field.**
   `canonical_transform.py` states the governing policy outright — *"Never
   writes a readable label or a region code into the classification year"* — and
   `regime_projector.py` carries an explicit semantic guard excluding
   classification-year fields from the UK `GBZZZ` override. The delivery rule
   was doing the mirror image of what both upstream stages refuse to do.
3. **It defeated its own guard.** Transforms run *before* validators, so the map
   converted a region label into `"2026"` and handed it to a regex whose entire
   purpose was to reject region labels. The value arrived looking valid.

The repository already named this defect: `test_annex2_path_map_promotion.py`
and `test_annex2_path_acceptance_gate.py` both block `RREL12 -> "2026"` from
being imported into the XSD path map, by name.

**It had never executed.** All 11,035 projected RREL12 values are `2021`; zero
RREL12 coercions were recorded.

The rule now carries **no transform at all**. The regex guard is retained, so:

| Input | Before | After |
|---|---|---|
| `2021`, `2013` | accepted | accepted |
| `ND1` (permitted) | accepted | accepted |
| blank (optional) | passes through | passes through |
| `West Midlands` | **silently became `"2026"`** | **governed `pattern` error, run fails** |
| `GBZZZ`, `TLG31` | pattern error | pattern error |

An unrepresentable value now fails the run deterministically instead of being
substituted. `geography_map` remains declared by **RREL11 only**, where it is
what it claims to be: a region *label* → region *code* translation.

### Repository-wide fabrication audit

Every value-changing mechanism reachable by the Annex 2 delivery path, and the
equivalent patterns elsewhere, were enumerated and classified. Summary: **no fabrication remains anywhere in the Annex 2 delivery path.** Two
findings outside it, both pre-existing on `main` and neither addressed here:
`annex12_projector.fallback_from_constraints` substitutes `0`/`0.0` for a
missing monetary value where no ND code is permitted (zero is a meaningful
financial figure, not an absence); and the XML preview guardrails are correct in
configuration but their test reads a key the policy file does not have, so all
ten error at setup and the guard is unverified.

### Every value transformation is now declared

Across 70 rules the complete set of value-changing mechanisms is:

| Mechanism | Rules | Category |
|---|---|---|
| `transform.enum_map` | 23 | declared enum translation (strict — unknown value errors) |
| `transform.geography_map` | 1 | declared enum translation (RREL11 label → code) |
| `transform.boolean` | 5 | declared formatting |
| `precision` | 11 | declared formatting/precision |
| `default_value` + `default_allowed` | 46 | declared delivery-rule default |
| `derive` | 3 | declared derivation from other canonical fields |
| `validators` | 25 | validation only — changes no value |

`derive` is a fourth category beyond the three the delivery tail was expected to
have. It is legitimate but worth naming: it computes `months_between_dates` from
two real dates, or selects the first non-blank of named fields. It returns `""`
rather than a substitute when its inputs are absent, so an absent input stays
absent. The `generator` mechanism (securitisation-ID composition) exists in code
but **no rule declares it**; it too refuses rather than invents.

## Configuration ownership and representation (complete)

### The layer hierarchy is now real

```
Global regime -> Asset pack -> Client -> Deal/SPV -> Reporting cycle -> Loan tape
      -> Projection -> Delivery normalisation -> XML
```

**The asset pack is active in the production projection path.** It previously
was not, for two independent reasons: no caller passed `--product-defaults`
(neither `trakt_run.py` nor the demo route), and the pack publishes
`nd_defaults` at the top level while Gate 4 reads `defaults.nd_defaults`.
Fixing only the first would have changed nothing. `trakt_run.py` now carries an
`ASSET_PACKS` registry keyed by portfolio type, so a new asset class registers a
pack rather than editing a call site.

**Product defaults no longer live in client configuration.** Ten product
decisions moved to the pack — payment frequencies, grace period, rate index and
tenor, margin, reset interval, cap, floor, deposit. The client keeps deal facts
and source-system limits only: 17 ND entries down to 6, none owned twice.

**Precedence is explicit and fail-closed.** Where the asset pack and the client
declare different no-data codes for the same field, Gate 4 **refuses**, naming
the field and both values, unless the exception is declared under
`nd_override_rationale`. Agreement passes; a declared override passes and is
evidenced.

### Workbook concept vs XML representation

Three Annex 2 concepts are workbook fields but **XML attributes**:

| Code | Workbook concept | XML representation | Qualifies |
|---|---|---|---|
| RREL18 | Primary Income Currency | `Ccy` attribute | RREL16 income amount |
| RREL28 | Currency Denomination | `Ccy` attribute | exposure amounts |
| RREC22 | Collateral Currency | `Ccy` attribute | RREC13 valuation amount |

The XSD carries a currency as `Ccy` on the amount it qualifies
(`ActiveOrHistoricCurrencyAndAmount`, `use="required"`), and its own
documentation says so: *"Include the currency in which the amount is
denominated, using {CURRENCYCODE_3} format."* The authoritative workbook declares
**zero** paths for all three, while sibling elements RREL16/17/19 have 9/7/7.

A currency is due **only where an amount is reported**. On this benchmark
RREL16 is `ND1`, so there is no income amount and no currency is missing.
**No delivery rule, XML path or node may be created for these three** — emitting
an element the schema does not define would fail validation.

Across all 107 concepts: **87 value-vs-NoData choices, 17 elements, 3
attributes.**

This is declared as governed metadata in
`annex2_delivery_rules.yaml::reconciliation_scope.representation`
(`representation_type`, `attribute_name`, `attribute_of`, `emission_condition`).
**The structure is INTERIM** — deliberately scoped to the concepts that need it,
designed to move into a generic per-regime representation model once one exists
so Annex 12 and FCA reuse it without special-case code.

### ND reconciliation is scoped to emitting concepts

```
ND in delivery-ready data      628,978
  on non-emitting concepts      11,035   (RREL18, disclosed not hidden)
  on emitting concepts         617,943   <- the XML tie-out
ND nodes in the XML            617,943   ✅
```

The non-emitting cells are legitimate delivery-ready data that produce no XML
node. They are disclosed separately rather than counted as a shortfall.

### KNOWN LIMITATION — single run-level currency

The builder stamps **every** monetary `Ccy` from the run-level `--currency`
(default `GBP`), applied uniformly to all 121,381 amount attributes. It is not
read from `primary_income_currency`, `exposure_currency_denomination` or
`collateral_currency`.

* Correct for this **GBP-only** benchmark.
* **Not proven for multi-currency portfolios**, or where income, exposure and
  collateral are denominated differently — every amount would be stamped with
  the run currency regardless of what was reported.
* Per-amount currency lineage is **not implemented**. Income, exposure and
  collateral currency may each need a distinct source.
* **OCC production should block or clearly mark multi-currency submissions**
  until this is resolved.

Deliberately not fixed in this phase: it would change delivered values for any
non-GBP book and cannot be validated against the accepted benchmark.

## Remaining work

**Phase 3 — governed OCC promotion.** Add delivery normalisation and XML+XSD as
governed stages after the existing `project` step, with the instrumentation
above surfaced as operator-visible warnings and the XML staged until explicit
approval. Only at that point does XML delivery enter the production route. The
hardening framework must then be repointed at the governed stages rather than
`trakt_run.py --mode regulatory` — **not yet implemented**.

**Not production-ready.** The delivery tail is proven, observable, and invents
nothing. It is still not governed, not promoted, and not certified: the XSD is a
draft, `config/delivery/annex2_field_xsd_path_map.yaml` still records
`production_ready: 0/107` and `do_not_generate_production_xml: true`, and no
real client tape has been submitted.
