# Review F — Annex 2 XML delivery route investigation

Investigation only; nothing modified. Evidence: forensic module reads with
file:line citations, git history, instrumented re-execution, and per-field XML
analysis. Companion data: [`annex2_field_population.csv`](annex2_field_population.csv)
(all 107 codes) and [`annex2_38_code_reconciliation.csv`](annex2_38_code_reconciliation.csv).

## A. Executive answer

**The latest, most efficient, most complete and most successfully proven route
from canonical loan data to XSD-valid Annex 2 XML is the Gate chain (Route B):**

```
platform_canonical_typed.csv
→ engine/gate_4_projection/regime_projector.py            (projection, 105/107 codes)
→ engine/gate_4b_delivery/annex2_delivery_normalizer.py   (delivery normalisation)
→ engine/gate_5_delivery/xml_builder_annex2.py            (XML + whole-document XSD validation)
   against config/system/DRAFT1auth.099.001.04_1.3.0.xsd
```

Re-executed unchanged in this investigation: **11,035 exposure records, 105
fields, 208.8 MB XML, XSD Validation: PASSED, 171.5 s wall, 772.7 MB peak
child RSS.** It is the *only* route in the repository that produces validated
XML. It is also the **oldest** route (all three scripts landed 2026-07-01) —
"latest" and "proven" are not the same implementation here, and the newer
agentic route deliberately refuses to emit XML (§D/§F). The proven route has
two disqualifying behaviours for production as-is — silent ND5 injection and
value fabrication in the Gate 5 builder — documented in
`docs/legacy_gate5_annex2_xml_builder_review.md`, which is why the
recommendation (§G/§H) is: **keep Gate 4 projector and Gate 4b normaliser as
authoritative, wrap them through the orchestrator, and gate the Gate 5 builder
behind an explicit governed step with its injections surfaced as operator
warnings — not silently re-platform, and not wait for the unfinished v2
agent.**

## B. Evidence-backed route map (all routes)

**Route A — current orchestrator/OCC route (production-invoked).**
`orchestrator_agent` → `RealAgentAdapters.onboard(regulatory_mi)` →
transform/validate agents → stamp → assemble →
`adapters.project()` → `assembler_agent.build_regime_command()` →
**`regime_projector.py`** (same script as Route B stage 1, invoked with
`config/client/config_client_ERM_UK.yaml` + `config/system/enum_mapping.yaml`)
→ `*_ESMA_Annex2_projected.csv` + provenance companion → **ENDS**
(`orchestrator.py:293-309`; no normaliser, no XML, no XSD). Callers:
`apps/blob_trigger_app/router.py:471,524,749-759` (production),
`operations_control` (OCC). *A projected CSV is not a completed delivery.*

**Route B — Gate 4b + Gate 5 (demo-invoked; the proven route).** Exact
commands in `demo_platform/artefacts.py:335-435`; also
`trakt_run.py --mode regulatory` (`:97-100,664-716`) and
`synthetic_demo/run_pipeline.sh:68,95`. Mapping source for XML is the
**workbook** `DRAFT1auth.099.001.04_…_Version_1.3.1.xlsx` sheet
`DRAFT1auth.099.001.04` (`xml_builder_annex2.py:168-214`) — *not*
`annex2_field_xsd_path_map.yaml`. Validation: `etree.XMLSchema(...).validate`
on the whole in-memory DOM (`:682-684`).

**Route C — agentic route (projection_agent → delivery_xml_agent).**
`40_validation_manifest` → `projection_agent.build_projection_package`
(long loan×field audit frame `51_…`, built from
`annex2_delivery_rules.yaml::field_rules` — 68/107 codes; the field universe
YAML is never loaded) → `delivery_xml_agent.build_delivery_package` →
`62_delivery_normalised_frame` (**classifier, not a normaliser** — values are
copied unchanged, `delivery_xml_agent.py:368`) → `65_xml_preview.xml` is a
**2-line comment placeholder**; `66_…` says `production_xml: false`,
"deferred to v2" (`:792-804`); `ready_for_xml_delivery` is a literal `False`
(`:774,857`). **No callers outside its own CLI and tests** (grep across
`engine/orchestrator_agent`, `apps/`, `function_app.py`, `demo_platform`:
zero). Untouched since its single landing commit `f09a898` (2026-07-04).

**Route D — XSD structured preview (non-production).**
`preview_readiness.py` + `xsd_structured_preview_builder.py`: real nested-path
construction from `annex2_field_xsd_path_map.yaml` (100/107 codes
builder-accepted) and real `lxml` XSD validation (`:398-407`) — but 5-record,
watermarked, all modes `enabled: false`, and self-declared expected-to-fail
(`known_limitations`, `:367-381`). It is the seed of a future real builder,
not a route.

## C. Historical-result reconciliation

| Claim | Verdict | Evidence |
|---|---|---|
| "106 of 107" | **No repository evidence anywhere** (full tree + `git log --all -S` pickaxe). Closest real numbers: **100/107** codes `xsd_validated=True` in `output/config_review/annex2_path_acceptance_decisions.csv`, and preview artefacts *numbered* `106_…`/`107_…` (sequence numbers) | prior review + re-verified |
| "107" | The **field-universe size** (`annex2_field_universe.yaml:6 field_count: 107`), never a pass count | config |
| **105 fields** | **Strongest evidenced result**: commit `0ed7b4c` (2026-07-26, "Produce a genuinely XSD-valid ESMA Annex 2 submission from the demo platform"), committed record `demo-video/public/fixtures/demo_manifest.json:4823` (`fields:105, exposureRecords:11035, xsdValidated:true`). Route B | git + fixture |
| **104 fields** | `synthetic_demo` committed reports (`…projection_report.json: regime_fields:104`, delivery report 0 errors, `ANNEX2_XML_REVIEW.md` "XSD Validation: PASSED"). Route B, earlier fixture | committed artefacts |
| 11,035 | Exposure-record count of the demo platform's consolidated 2026-06 canonical; reproduced | this run |
| "XSD valid" | True for Route B outputs — with the material caveat that validity is partly achieved by builder-injected ND5/coerced values (§E, §F) | `legacy_gate5…review.md:87-93` |

## D. Reproduction report

| Field | Value |
|---|---|
| commit / branch | `c57ff43` on `claude/trakt-operations-control-centre-mqjyy7` (chain scripts untouched since `b777cc1`/`0ed7b4c`) |
| entry_point / command | `python -m demo_platform.run_demo --generate --onboard --orchestrate` then instrumented `--artefacts --no-reset` |
| input_fixture | demo-generated synthetic client (`alderbridge`), consolidated `platform_canonical_typed.csv`, 2026-06, 11,035 loans |
| projector | `engine/gate_4_projection/regime_projector.py` |
| normaliser | `engine/gate_4b_delivery/annex2_delivery_normalizer.py` |
| xml_builder | `engine/gate_5_delivery/xml_builder_annex2.py` |
| rules_file | demo copy of `config/regime/annex2_delivery_rules.yaml` — **identical 68 `field_rules`**; sole diff is one RREL83 demo-LEI enum entry (generated-file header states this) |
| field_universe / template / code_order | `annex2_field_universe.yaml` (informational; not loaded by this route) / workbook `…Version_1.3.1.xlsx` sheet `DRAFT1auth.099.001.04` / `config/system/esma_code_order.yaml` |
| xsd / validator | `config/system/DRAFT1auth.099.001.04_1.3.0.xsd` / `lxml etree.XMLSchema` whole-document |
| record_count / field_count / xml_size | **11,035 / 105 / 208.8 MB** |
| elapsed / peak_memory | **171.5 s** (whole artefact stage incl. deck; Annex 2 chain is the dominant cost) / **772.7 MB** peak child RSS |
| xsd_result / errors | **PASSED** / none; stage returncodes all 0 |
| exactness | Same commands, code and configs as `0ed7b4c`; input is freshly generated synthetic data from the same generator (seeded run — counts and field set match the committed record exactly). Not byte-identical XML (demo LEI/sequence values), and stated as such |

Independent check: `pytest tests/test_xml_builder_annex2_shape_fixes.py::test_delivery_ready_fixture_without_npe_columns_still_xsd_valid_for_prf_and_nprf` — PASSED (real schema validation on the committed 104-column fixture).

## E. Field-population report (why rule-count ≠ XML completeness)

Full per-code table: `annex2_field_population.csv`. Summary of the 107 codes
in the reproduced XSD-valid output:

- **105 projected and delivered** (all but `RREL20`/`RREL21`, Optional,
  ND-allowed, omitted end-to-end).
- Population mechanisms: **68 rule-governed** (46 validation/format rules, 22
  enum-transform rules) and **37 registry-mapping only** — the projector
  selects and renames columns from `fields_registry.yaml::regime_mapping`
  regardless of `field_rules`; the normaliser **passes unlisted columns
  through verbatim** (`out_df = df.copy()`, iterates `field_rules` only,
  `annex2_delivery_normalizer.py:334,339`).
- **54 of the 105** delivery values in the sample record are permitted
  no-data codes (ND1/ND5) — many from the 40+ `default_value: ND…` rules,
  some injected by the Gate 5 builder itself (`_ensure_hstrcl_colltn_nd_defaults`
  writes 144 ND5 month-nodes per record, `xml_builder_annex2.py:498-517`).
- 100/107 codes resolve to an XML value via the accepted path map; the
  unresolved handful are the polluted-path conflicts (RREC1/RREC2), the
  deferred RREC22, and nested/header cases.
- **Therefore `field_rules` count (68) measures *governed treatment*, not
  emission.** Projection breadth comes from the registry (107 mapped);
  delivery survival is pass-through; XSD validity is then guaranteed largely
  by ND defaulting — in rules (legitimate, declared) and in the builder
  (silent, the documented reason the builder must not ship as-is).

> **Superseded in part by Phase 2** (see
> [`annex2_delivery_migration.md`](../../annex2_delivery_migration.md)). The
> rule count is now **70**: `RREL20`/`RREL21` moved out of the builder into
> declared rules, so they are delivered as rule-populated `ND5` rather than
> omitted end-to-end. The builder injects **no** ND for them — the only
> remaining builder-side ND insertion is `_ensure_hstrcl_colltn_nd_defaults`,
> which did not fire on this benchmark. The XML is byte-identical
> (`a21f8a4c…d685d`): what changed is where the decision is made, not the
> output. The delivered field split is now stated as 105 from projected source
> + 2 by declared rule = 107 represented.

## F. Route comparison (same 11,035-record input)

Route A's projector stage was executed with the exact orchestrator arguments
(`config_client_ERM_UK.yaml` + production `enum_mapping.yaml`) against the
same canonical:

| Aspect | Route A (orchestrator projector) | Route B (gate chain) | Route C (agentic) | Route D (preview) |
|---|---|---|---|---|
| Projected shape | 11,035 × 105, identical column order to B | 11,035 × 105 | long audit frame, 68-rule subset | n/a |
| Value diff vs B | **22 columns differ — all configuration-driven**: demo client config supplies LEI (RREL1/RREC1 ScrtstnIdr generated vs blank), `nd_defaults` (ND5 in RREL8/44/45/46… vs blank), augmented enum map (RREC5 `RBLD` vs raw `Residential property`) | baseline (sha `0f449fbe…`) | not comparable (different shape) | n/a |
| Normalised set | — (stage absent) | 105 cols, preflight PASS | 62_ frame classifies, mutates nothing | n/a |
| XML / namespaces / ordering | none | `urn:esma:xsd:DRAFT1auth.099.001.04`, workbook-order sequence insertion | placeholder comment | real nesting, 5 records, watermarked |
| XSD outcome | none | **PASSED** | none | honest report, expected fail |
| Runtime / memory | projector alone ≈ seconds | 171.5 s / 772.7 MB (stage total) | n/a end-to-end (blocked by 68/107 gate upstream) | n/a |
| Classification of differences | configuration-driven (client config completeness), expected | — | missing functionality (by design, "deferred to v2") | non-production by policy |

Key operational implication of the A↔B diff: with the current production
client config (`config_client_ERM_UK.yaml` — no LEI, no `nd_defaults`), the
Gate 4b preflight would **fail on blank mandatory identifiers** — client-config
completeness (LEI, securitisation identity, ND policy) is a real onboarding
prerequisite for any client, independent of code.

## G. Scores, "latest vs best", and authoritative components

| Criterion | Route A (projector-only) | **Route B (4b+5)** | Route C (agentic) | Route D (preview) |
|---|---|---|---|---|
| Correctness | projection proven; no delivery | **only XSD-proven route**; but builder fabricates (RREL12 "2026" coercion `:109-118`) and silently injects ND5 — flagged unsafe | refuses XML by design | expected-fail |
| Completeness | stops at CSV | **full: projection→normalisation→XML→XSD** | readiness reports only | partial |
| Efficiency | light | 171.5 s / 773 MB for 11k records; in-memory DOM, no streaming; O(rows×68) scalar normaliser; production path additionally runs the projector twice (`router.py:749-759`) | n/a | n/a |
| Maintainability | shared with B | wide-row model, workbook ordering (not XSD-derived), duplicated header guards; but strongest test suite (14 normaliser + 10 builder tests incl. the real-XSD test) | cleanest contracts (40→50→60 manifests, lineage, readiness) but unfinished | policy-frozen |
| Production suitability | already wrapped/governed | deterministic and idempotent, but `sys.exit(2)` gates, silent fills, no governed result | best manifest/readiness chain for wrapping; no product | no |

**Latest ≠ best, resolved explicitly:** *newest* = Route C (landed 2026-07-04,
never touched since, one commit); *currently invoked* = Route A (production +
OCC); *most complete & most successfully tested* = **Route B**; *fastest /
most memory-efficient for full delivery* = Route B (only contender; 773 MB is
acceptable, streaming is a future concern); *easiest to govern* = Route C's
manifest pattern wrapped around Route B's engines. **Gate 4b/Gate 5 were
retired NOT because they were wrong wholesale or replaced** — the review says
the workbook crosswalk "resolved ~89 previously-unmapped fields" and the
verdict is "map reused, runtime retired": the *builder's silent ND5
injection/value coercion and wide-row singleton-collateral model* are the
disqualifiers; the replacement (v2) **does not exist yet**. The normaliser and
its rules remained actively maintained to 2026-07-28.

**Authoritative components (recommendation):**

| Role | Authoritative file |
|---|---|
| Canonical input | assembled `platform_canonical_typed.csv` (existing platform assembler output) |
| Annex 2 projector | `engine/gate_4_projection/regime_projector.py` |
| Delivery normaliser | `engine/gate_4b_delivery/annex2_delivery_normalizer.py` |
| XML builder | `engine/gate_5_delivery/xml_builder_annex2.py` — **interim**, behind a governed step that surfaces its ND5-injection/coercion counts as operator-visible warnings; long-term replacement = promoting `xsd_structured_preview_builder.build_tree` per the acceptance policy |
| XSD | `config/system/DRAFT1auth.099.001.04_1.3.0.xsd` (retire the duplicate repo-root copy) |
| XSD validator | `lxml etree.XMLSchema` as invoked in `xml_builder_annex2.py:682-684` |
| Delivery rules | `config/regime/annex2_delivery_rules.yaml` (complete the 38-code gap per E1) |
| Field universe | `config/regime/annex2_field_universe.yaml` |
| Template/mapping | workbook `DRAFT1auth.099.001.04_…_Version_1.3.1.xlsx` sheet `DRAFT1auth.099.001.04` (interim); `config/delivery/annex2_field_xsd_path_map.yaml` once its 7 open codes are accepted |
| Code ordering | `config/system/esma_code_order.yaml` |
| Field constraints / enums | `field_rules` precision/regex/nd_allowed + `config/system/enum_mapping.yaml` (note: enum truth duplicated with `field_rules.transform.enum_map` — consolidation candidate) |

## H. Governed integration proposal (smallest safe production route)

Wrap, don't rewrite: **two new adapter stages** in the OCC engine (additive
files only), executing the proven scripts exactly as `demo_platform/artefacts.py`
does, after the existing `project` step:

| Stage | Input | Output | Success | Blocking | Warning | Persisted | Rerun | Plain-English result |
|---|---|---|---|---|---|---|---|---|
| Delivery normalisation | `*_ESMA_Annex2_projected.csv` + `annex2_delivery_rules.yaml` | `*_delivery_ready.csv`, report, issues | preflight PASS (rc 0) | rc 2 (mandatory missing / ND-not-allowed / enum reject) → operator decisions from the issues CSV | optional-field issues | all three artefacts in run staging + ops container refs | idempotent re-exec | "The report data was checked and prepared for the regulator. N items need your review." |
| XML + XSD | delivery-ready CSV, workbook, code order, XSD | `annex2_submission.xml`, stdout capture | "XSD Validation: PASSED" + rc 0 | rc ≠ 0 / FAILED | **parsed injection counts** (ND5 defaults injected, values coerced — from builder output/diff) surfaced as warnings the operator must see before publication | XML + validation summary | idempotent re-exec | "The submission file was generated and passed the regulator's format check. M values were filled as 'no data' — review before publishing." |

Publication then follows the existing OCC gate: XML is a staged artefact until
explicit approval; `persist_regime_dir` (existing) publishes it. No agent,
gate script, or config is modified for this; the orchestrator itself also
stays untouched (the OCC engine already owns post-`project` staging).
Prerequisites that ARE existing-file changes (below): delivery-rules
completion (or the run parks at the current onboarding gate) and client-config
completeness (LEI etc., per §F).

## I. Approval checkpoint — changes required for the recommended route

| # | File(s) | Change | Type | Up/downstream impact | MI-only | Blob trigger | OCC | Regression tests | Rollback |
|---|---|---|---|---|---|---|---|---|---|
| I1 | `operations_control/engine.py`, `operations_control/adapters.py` (+~150 lines) | two governed stages invoking `annex2_delivery_normalizer.py` and `xml_builder_annex2.py` as subprocesses; injection-count parsing; stage GARs | **new additive adapter code** | none upstream; OCC-only downstream | none | none | new stages | new stage tests + golden run (below) | revert commit |
| I2 | `config/regime/annex2_delivery_rules.yaml` | complete/defer the 38 unruled codes (= prior review E1) | **config** | clears onboarding gate; normaliser behaviour for those codes changes from pass-through to governed | none | same gate exists there — also unblocked | unblocks regulatory onboarding | golden Route B rerun must stay XSD-PASSED with unchanged treated columns | git revert |
| I3 | `config/client/config_client_ERM_UK.yaml` (or per-client config) | real LEI, securitisation identity, ND policy | **config** | projector output completeness (§F diff) | none | affects its regime runs | required for preflight PASS | Route B rerun on client data | git revert |
| I4 | `xml_builder_annex2.py` (optional, later) | replace silent ND5 injection/coercion with declared, rule-driven behaviour; or promote preview builder per `annex2_path_map_promotion_policy.md` acceptance conditions | **existing pipeline file — defer** until golden test exists | changes XSD outcome risk | none | none | warning counts drop | golden test is the gate | git revert |
| I5 | repo-root duplicate XSD; unused `annex2_xml_structure_contract.yaml`; double projector run in `router.py:749-759` | housekeeping/de-duplication | **removal/cleanup — separate approvals** | low | none | router change needs care | none | existing suites | git revert |

**Required regression suite before calling the route production-ready**
(mostly new tests, additive): golden reproduction of the 105-field/11,035-record
XSD-PASS (fixture: the demo generator seed + committed expected counts and
projected-CSV hash); exact record/field counts; whole-document XSD validation;
element-order and namespace assertions; mandatory/conditional/optional and
ND-value matrices per `field_rules`; enum-map round-trips; deterministic rerun
(hash-stable); corrupt input / invalid XML / wrong XSD / missing config
failure paths (exit codes + operator-safe messages); large-file runtime+RSS
budget (≤2× the measured 171.5 s / 773 MB); OCC-level restart recovery,
publication gating, tenant isolation and audit-chain assertions over the new
stages (extending the existing 52-test suite).

**No change is implemented. Approval is requested for I1–I3 (in that order)
to make the proven route available through the OCC; I4 and I5 are flagged
separately and are not prerequisites.**
