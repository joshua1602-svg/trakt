# Repository fabrication audit — Annex 2 hardening, final pass

**Question asked:** does any code or configuration reachable by the Annex 2
delivery path invent a regulatory value the client never supplied?

**Answer: no.** The last instance — the RREL12 `geography_map` — was removed in
this pass. Two findings *outside* the Annex 2 delivery path are recorded below;
both pre-date this branch and exist on `main`.

Scope of the sweep: every `.py` and `.yaml` under `engine/`,
`operations_control/`, `config/`, `demo_platform/`, `mi_agent/`, `trakt_core/`
and `scripts/`, searched for hardcoded years and dates, hardcoded regulatory and
ND codes, fallback substitutions, catch-all enum entries, value-inventing
`else`/`except` branches, and coercions applied outside declared configuration.

---

## 1. Classification summary

| Class | Count | Action |
|---|---|---|
| Latent fabrication (never executed, could) | **1** | **Removed** — RREL12 `geography_map` |
| Active fabrication (executes today) | **0** | — |
| Fabrication outside the Annex 2 path | 1 | Reported, not changed (Annex 12) |
| Legitimate constant | 9 | Unchanged |
| Declared configuration | 6 | Unchanged |
| Test / benchmark fixture | 5 | Unchanged |
| Synthetic preview generator (guarded) | 2 | Reported — guardrails untested |
| Documentation / example | many | Unchanged |
| False positive | many | Unchanged |

---

## 2. The finding that was removed

### RREL12 `geography_map` — latent fabrication

`config/regime/annex2_delivery_rules.yaml` mapped ten UK region *names* onto the
literal year `"2026"`.

| Question | Finding |
|---|---|
| Why did it exist? | Copied from the RREL11 rule directly above it — identical key set, target changed from the region code `GBZZZ` to a year. Introduced in `af7471e`. |
| Authoritative basis? | **None.** The workbook defines RREL12 as "the year of the NUTS3 classification used". A region name cannot imply a vintage. |
| Contradicts anything? | **Yes, two upstream stages.** `canonical_transform.py`: *"Never writes a readable label or a region code into the classification year"*. `regime_projector.py` carries an explicit guard excluding classification-year fields from the `GBZZZ` override. |
| Has it ever executed? | **No.** All 11,035 projected RREL12 values are `2021`; zero RREL12 coercions recorded. |
| Any test depend on it? | **No.** `tests/test_annex2_delivery_normalizer.py` exercises the *mechanism* with its own inline fixture. Two committed tests already blocked `RREL12 -> "2026"` by name. |
| Aggravating factor | Transforms run **before** validators, so it converted a region label into `"2026"` and handed it to a regex whose purpose was to reject region labels. |

**Resolution:** `transform` removed entirely; the regex guard retained. An
unrepresentable value now produces a governed `pattern` error and fails the run.
`ND1` remains available for a genuinely unavailable year.

> **Note on the fix itself:** the first attempt removed the `validators` block
> along with the transform, which would have been *worse* than the fabrication —
> a region label would have reached Gate 5 unchecked. Caught by running values
> through Gate 4b rather than trusting the diff. `test_the_semantic_guard_survived_the_removal`
> now pins it.

---

## 3. Findings outside the Annex 2 delivery path

### 3.1 Annex 12 projector — numeric-zero fallback (genuine, not fixed here)

`engine/gate_4_projection/annex12_projector.py::fallback_from_constraints`:

```python
if spec.get("nd5_allowed"): return "ND5"
if spec.get("nd1_nd4_allowed"): return "ND1"
if is_numeric:
    if "{INTEGER" in fmt.upper(): return 0
    return 0.0
raise ValueError(f"{code}: no legal fallback ...")
```

The ND branches are workbook-driven and legitimate. The **numeric branch
substitutes `0` for a missing monetary or percentage value** where no ND code is
permitted. Zero is a meaningful financial figure, not an absence: reporting a
balance of `0` asserts something false about the exposure.

Mitigating: it raises rather than inventing when no fallback is legal at all,
and the *eligibility* comes from the workbook constraints.

**Not changed here.** Annex 12 is a different regime with its own delivery path,
no benchmark in this branch, and changing it would need its own reproduction
evidence. Recommended for the Annex 12 equivalent of Phase 2.

### 3.2 XML preview synthetic generators — guarded, but the guard is untested

`engine/delivery_xml_agent/preview_readiness.py` (`_DUMMY_BY_TOKEN`) and
`xsd_structured_preview_builder.py` (`_builtin`) generate synthetic values —
`"2025-01-01"`, `"2025"`, `"2024-12-31"`, `"SYNTHDUMMY…"`, a `DUMY` LEI — to
build a structurally valid document for XSD conformance testing.

**These are correctly guarded.** `config/delivery/xml_preview_policy.yaml`:

* all four preview modes `enabled: false`
* every mode carries a mandatory watermark ("NOT FOR REGULATORY SUBMISSION")
* five production guardrails, all `true`, including
  `never_promote_preview_values_to_production` and
  `never_write_preview_to_production_output`

**Classification: legitimate synthetic test-data generator.**

**But the test that verifies this cannot run.**
`tests/test_delivery_xml_agent_review.py::TestXmlPreviewPolicy` reads
`yaml.safe_load(...)["preview_policy"]`, and the policy file has no
`preview_policy` key — its keys are top level. All 10 tests error at setup.

Verified present on `main`, so this pre-dates the branch. It is the only place
in the sweep where a real guardrail against value fabrication is **unverified**,
which is why it is called out rather than filed under "documentation".

**Recommended:** fix the key path so the guardrails are actually asserted. Not
done here — different subsystem, and this pass was scoped to the Annex 2 path.

---

## 4. Classified as legitimate — not changed

| Finding | Location | Why it is legitimate |
|---|---|---|
| `Ccy` attribute stamped on `Amt` leaves | `xml_builder_annex2.py:513` | XSD requires a currency on amount values; sourced from the explicit `--currency` run setting (default `GBP`), not invented per-row. Disclosed as informational evidence by the OCC intervention report. |
| `set_to_zero_if_missing` / `set_to_nd5_if_missing` / `set_to_nd1_if_missing` | `annex12_projector.py:456-461` | Read from a `defaults` configuration block — declared, reviewable, per-deployment. |
| `nd_defaults` application | `regime_projector.py:426` | Entirely client-config driven (`config.defaults.nd_defaults`). No hardcoded ND. |
| ND vocabulary construction | `rules_adapter.py:70`, `target_coverage.py:365`, `target_contract_completion.py:763` | Builds the *allowed* ND list from workbook constraints. Constructs a vocabulary, assigns no value. |
| `_parse_multiplicity` returning `(0, None)` | `xml_builder_annex2.py:283` | Parses XSD cardinality metadata. Not a data value. |
| `generate_securitisation_id` | `annex2_delivery_normalizer.py:152` | Composes the ESMA identifier from a validated LEI + 4-digit year + sequence. Returns `None` (→ governed error) if either input is invalid. Derivation, never invention. **No rule declares it.** |
| `derive: months_between_dates` / `first_non_blank_from_fields` | 3 rules | Arithmetic and selection over real canonical values; returns `""` when inputs are absent rather than substituting. |
| `OTHER` / `UNKNOWN` → `OTHR` enum entries | `enum_mapping.yaml` | **Explicit source values, not wildcards.** A source asserting "UNKNOWN" mapped to the regulatory catch-all is a declared translation. **No `*`/`default`/`fallback` key exists anywhere in the file** — an unrecognised value stays unrecognised so the XSD rejects it visibly. |
| `REPORTING_DATE = "2026-06-30"` and go-live dates | `operations_control/occ_agent/fixtures.py` | Demo fixture data. |

### Enum-layer inconsistency — fail-closed, already tracked

`enum_mapping.yaml` declares targets for `employment_status`, `repayment_type`,
`borrower_type` that are readable labels rather than ESMA codes
(`EMPLOYED`, `Annuity`, `O`). This is the same *shape* as the `R1`/`R2`/`C1`/`C2`
collateral defect corrected earlier in this branch.

It is **not** a fabrication, because the delivery layer catches it: RREL13's rule
carries a **strict** `enum_map` (`EMRS`, `EMBL`, `EMUK`, `UNEM`, `SFEM`, `NOEM`,
`STNT`, `PNNR`, `OTHR`), and a strict `enum_map` returns a governed error for an
unknown value rather than passing it through. A layer mismatch fails the run.

It is also already tracked: `TestAnnex2ConfigAlignment::test_remaining_asset_conflicts_require_manual_review`
and the `46_annex2_enum_coverage_reconciliation` report exist for exactly this,
and are among the known pre-existing failures. Belongs with the Phase 4
enum-truth consolidation.

---

## 5. Remaining builder policy (Annex 2 Gate 5)

Every place `xml_builder_annex2.py` writes a value, and what it is:

| Site | What it writes | Class |
|---|---|---|
| `leaf.text = v` (598) | the row's own value | structural ✅ |
| `root.set(schemaLocation)` (736) | XSD namespace metadata | structural ✅ |
| `node.set("Ccy", currency)` (513) | run-setting currency on `Amt` leaves | configuration ✅ |
| `_coerce_record_value_for_branch` (241) | nothing — routes to NoData | **no policy since Phase 2** ✅ |
| `_ensure_scndry_oblgr_incm_defaults` (610) | nothing — raises if unsupplied | **no policy since Phase 2** ✅ |
| `_ensure_hstrcl_colltn_nd_defaults` (689) | `ND5` × 144 per record | ⚠️ **policy** |
| `_ensure_nprf_nonprfrmgdata_defaults` (711) | `ND5` for NPE codes | ⚠️ **policy** |

**Two functions still contain reporting policy rather than structural XML
construction. Both are in the non-performing (NPRF) branch.**

**Why acceptable today:** the benchmark is PRF-mode with zero `NonPrfrmgLn`
nodes, so neither fires — there is no measured behaviour to preserve and no
evidence on which to choose the right rule shape. Moving them blind would be
guesswork dressed as governance. Both are instrumented and will report the
moment they do fire.

**Recommendation:** migrate to declared rules the same way RREL20/RREL21 were,
as part of Phase 3 — but only once a non-performing book exercises them, so the
change can be evidenced rather than assumed. In
`_ensure_nprf_nonprfrmgdata_defaults` the *eligibility* is already
workbook-derived (NPE codes carrying a `NoData` branch under `/NonPrfrmgData/`);
only the choice of `ND5` is in code.

---

## 6. Method note

Grep alone cannot answer "does this fabricate". Each candidate was resolved by
reading the code path and, where behaviour was in question, executing it:
RREL12 values were run through Gate 4b to observe accept/error outcomes, the
builder refusal was triggered directly, and the benchmark's own instrumentation
was read to confirm which paths executed. The claim "it never fires" is
measured — 11,035 of 11,035 projected values — not inferred.
