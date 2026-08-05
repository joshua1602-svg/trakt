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

---

## Phase status

| Phase | Scope | Status |
|---|---|---|
| **0** | Reproduce the committed 105/107 benchmark at HEAD | ✅ **Complete — byte-identical** |
| **1** | Make ND usage and value coercion visible, output-neutral | ✅ **Complete** |
| 2 | Remove the RREL12 fabrication; govern the NPRF historical-collection ND | ❌ Not implemented |
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
| XSD validation | PASSED | PASSED | ✅ |

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

| Category | Where reported | Benchmark value |
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
The builder inserts them. Until Phase 1 that was invisible.

A run with no coercions **records zero explicitly**; an absent entry is never
treated as evidence of absence:

```json
"coercions": {"count": 0, "records": [], "truncated": false, "record_cap": 5000}
```

Counts are always exact. Individual records are capped at 5,000 with
`truncated: true` when exceeded, so a report stays usable without under-stating.

### What Phase 1 deliberately does NOT do

* `_coerce_record_value_for_branch` still substitutes `"2026"` for a non-ISO-year
  RREL12 value. It is now **recorded** with original value, resulting value,
  field code, exposure identifier and reason — but not removed. Its own
  docstring already specifies the correct behaviour ("route to NoData branch");
  implementing that is **Phase 2**. It did not fire on the benchmark (every
  RREL12 value is `2021`), so it is a latent landmine, not an active corruption.
* The non-performing historical-collection ND logic
  (`_ensure_hstrcl_colltn_nd_defaults`, 144 ND5 nodes per record) is unchanged.
  It did not fire on this benchmark — the submission is PRF-mode with zero
  `NonPrfrmgLn` nodes — but it is instrumented and will report if it does.

## Remaining work

**Phase 2 — remove fabrication, govern ND.** Make RREL12 route to the NoData
branch as documented. Decide whether builder-injected ND for RREL20/RREL21 is
legitimate no-data reporting (probably yes) and, if so, move it into a
*declared* rule so it is governed rather than hard-coded. Surface builder ND
counts as operator warnings.

**Phase 3 — governed OCC promotion.** Add delivery normalisation and XML+XSD as
governed stages after the existing `project` step, with the instrumentation
above surfaced as operator-visible warnings and the XML staged until explicit
approval. Only at that point does XML delivery enter the production route. The
hardening framework must then be repointed at the governed stages rather than
`trakt_run.py --mode regulatory` — **not yet implemented**.

**Not production-ready.** The delivery tail is proven and now observable. It is
not governed, not promoted, and not certified: the XSD is a draft, no field is
`production_ready`, and no real client tape has been submitted.
