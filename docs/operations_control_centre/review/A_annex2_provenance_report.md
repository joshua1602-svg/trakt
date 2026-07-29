# Review A — Annex 2 pipeline provenance report

Verification exercise, no code changed. Evidence sources: instrumented runtime
traces (Python audit hooks over real OCC runs), static call-graph inspection
with file:line citations, git history, and re-execution of the prior
best-evidenced Annex 2 XSD run.

## 1. Actual runtime call graph (instrumented, not inferred)

Two real workflows were executed through the Operations Control engine with a
`sys.addaudithook` recording every file open, subprocess exec and imported
module (instrumentation was temporary and is not committed).

**MI + ESMA Annex 2 workflow (`outcome=mi_annex2`), observed:**

```
React "Start"  (POST /ops/workflows {outcome: "mi_annex2"})
→ operations_control.api.app.create_workflow
→ operations_control.engine.OpsEngine._execute
→ engine.orchestrator_agent.orchestrator.run_orchestration(target="all",
      full_pipeline=True)                                  [current conductor]
→ GovernedAdapters → engine.orchestrator_agent.adapters.RealAgentAdapters
   .onboard(mode="regulatory_mi")
→ engine.onboarding_agent.workflow.run_operator_workflow   [current agent]
   configs OBSERVED opened: config/regime/annex2_delivery_rules.yaml,
   config/regime/annex2_field_universe.yaml, config/system/fields_registry.yaml,
   config/system/aliases_*.yaml, config/system/onboarding_modes.yaml,
   config/asset/product_profiles.yaml, product_defaults_ERM.yaml,
   config/system/run_context_policy.yaml
→ PARKS: onboarding status NEEDS_CONFIGURATION → workflow needs_review with
   5 operator decisions; transform/validate/project never reached.
   Subprocesses observed: none (parked before projection).
```

Had the run proceeded, the projection stage is proven (by constructing the
exact command the adapter executes) to be:

```
engine/orchestrator_agent/adapters.py:project()
→ engine/assembler_agent.py:build_regime_command()
→ subprocess: python engine/gate_4_projection/regime_projector.py <central.csv>
     --regime ESMA_Annex2
     --registry config/system/fields_registry.yaml
     --enum-mapping config/system/enum_mapping.yaml
     --config config/client/config_client_ERM_UK.yaml
     --template-order config/system/esma_code_order.yaml
→ outputs: *_ESMA_Annex2_projected.csv + *_ESMA_Annex2_provenance.csv
→ END. No delivery normalisation, no XML builder, no XSD validation exists in
   the orchestrator_agent path (orchestrator.py:293-309 returns after project).
```

**MI-only workflow, observed:** identical onboarding/agent modules
(`mi_only` mode), ran to completion, published via
`ProductionPersistence.persist_platform` only after explicit approval.

**Production parity.** The blob-trigger production path constructs
`RealAgentAdapters` identically (`apps/blob_trigger_app/orchestrator_invoke.py:239-251`:
`onboarding_mode_for_target(target)`, `full_pipeline`, `managed_service=True`)
and for `regime_required` sources runs `target="all"`
(`target_selection.py:29-31`, `router.py:471,524`) then the same
`build_regime_command` again via `regime_runner.py:37`, persisting **CSV only**
(`router.py:749-759`). Production does not produce Annex 2 XML/XSD today.

## 2. Current vs legacy path comparison

| Candidate path | Current or legacy | Used by OCC | Used by prior XSD-validated runs | Evidence |
|---|---|---|---|---|
| **Orchestrator-agent conductor** (`orchestrator_agent` + `RealAgentAdapters`: onboarding regulatory_mi → transformation_agent → validation_agent → stamp → assemble → `regime_projector` via `build_regime_command`; CSV only) | **Current production** | **Yes** (instrumented trace above) | No | `apps/blob_trigger_app/orchestrator_invoke.py:239-266`; `adapters.py:147`; audit-hook trace |
| **Gate chain** `trakt_run.py --mode regulatory` → `regime_projector.py` → `gate_4b_delivery/annex2_delivery_normalizer.py` → `gate_5_delivery/xml_builder_annex2.py` + XSD (`config/system/DRAFT1auth.099.001.04_1.3.0.xsd`) | **Legacy runtime, demo-only** — docs explicitly retire it: `docs/legacy_gate5_annex2_xml_builder_review.md:12-14` "must not be wired into the production path as-is"; `docs/annex2_path_map_promotion_policy.md:7` "Keep the map. Retire the runtime." | No (but its **first stage, `regime_projector.py`, is the same script** OCC's project step runs) | **Yes** — this is the only complete Annex2→XML→XSD chain in the repo | `trakt_run.py:652,690,1051-1053`; `demo_platform/artefacts.py:335-435`; `synthetic_demo/run_pipeline.sh:68,95` |
| **Agentic v1 chain** (`projection_agent` → `delivery_xml_agent`) | Forward path, partially wired — transform+validate live in the conductor; projection_agent/delivery_xml_agent have **no caller** outside their CLIs and tests; `66_xml_validation_report.json` hardcodes `production_xml: false` ("deferred to v2", `delivery_xml_agent.py:786-803`) | Transform+validate stages only | No | `orchestrator_agent/adapters.py:360,375`; `gate5_adapter.py:18-23` |
| **XSD structured preview** (`preview_readiness.py`, watermarked `105_xsd_structured_preview.xml`) | Frozen, non-production by design ("none ever touches production", `preview_readiness.py:7`) | No | No (engineering preview only) | `preview_readiness.py:72-76` |
| Generic Jinja `xml_builder.py` | Dead for Annex 2 (no XSD; `trakt_run.py:1163` stage label is stale) | No | No | `trakt_run.py:716,728` |
| Streamlit-era Annex 2 code | **Does not exist** — zero Annex 2 references in `ui/`, `analytics_lib/`, `cli/`, `tools/` | — | — | repo-wide grep |

**Config duplication findings (informational):** `annex2_xml_structure_contract.yaml`
is loaded by no code; `annex2_field_xsd_path_map.yaml` is preview/tooling-only;
the XSD exists in two copies (repo root and `config/system/`); Annex 2 enum
truth is duplicated between `enum_mapping.yaml` and
`annex2_delivery_rules.yaml::field_rules.transform.enum_map`.

## 3. The prior "≈106/107" result

**No "106 of 107" artefact, log, test, commit message or document exists
anywhere in the repository or its git history** (full-tree and `git log --all
-S` pickaxe searches). The figure is not reproducible because it was never
recorded. The real, evidenced prior results are:

1. **105 fields, XSD-validated, 11,035 exposure records** — commit `0ed7b4c`
   ("Produce a genuinely XSD-valid ESMA Annex 2 submission from the demo
   platform"), artefact record committed in
   `demo-video/public/fixtures/demo_manifest.json:4823`
   (`"fields": 105, "xsdValidated": true`). Produced by the **gate chain**
   (`demo_platform/artefacts.py:335-435`): `regime_projector` →
   `annex2_delivery_normalizer` → `xml_builder_annex2 --xsd
   config/system/DRAFT1auth.099.001.04_1.3.0.xsd`.
2. **104 fields, "XSD Validation: PASSED", 36 rows** — `synthetic_demo`
   committed artefacts: `..._ESMA_Annex2_projection_report.json`
   (`regime_fields: 104, mandatory_fields: 47, optional_fields: 57`),
   `..._delivery_report.json` (`errors_total: 0`), `ANNEX2_XML_REVIEW.md`.
3. The number closest to "106/107": `output/config_review/annex2_path_acceptance_decisions.csv`
   records **100 of 107** codes `xsd_validated=True` in the path-mapping
   review (and the preview artefacts are *numbered* `106_…`/`107_…` — sequence
   numbers, not counts). "107" is the **field-universe size**
   (`annex2_field_universe.yaml:6 field_count: 107`), not a pass count.

**Reproduction rerun (unchanged commands, outputs redirected off-repo):**

- `pytest tests/test_xml_builder_annex2_shape_fixes.py::test_delivery_ready_fixture_without_npe_columns_still_xsd_valid_for_prf_and_nprf`
  → **PASSED** (real `etree.XMLSchema.validate` against the in-repo XSD, on the
  104-column committed fixture).
- `python -m demo_platform.run_demo --all` with `TRAKT_LOCAL_BLOB_ROOT`
  pointed at a scratch directory → result recorded in §5 below.

## 4. Reconciliation of the "31 missing rules" statement

The Phase 2 report said "31 Annex 2 codes lack rules in
`config/regime/annex2_delivery_rules.yaml`". Verified numbers from the parked
run's own `43_annex2_field_universe_reconciliation.json` and the configs:

- Universe: **107** codes (`annex2_field_universe.yaml`).
- With `field_rules`: **68**. Deferred: **1** (`RREC22`).
- **Missing from field_rules: 38** — of which **31 are Mandatory-priority**
  (that is where "31" came from; the Phase 2 sentence conflated the blocking
  count with the total), 2 Optional, 5 Analytics.
- Of the 38: **30** gate as `pending_regime_rule`, **7** are already
  `source_mapped` in 28a coverage, **1** is `not_applicable`.

**The decisive cross-check:** of the 38 codes with no `field_rules` entry,
**36 are present and populated in the re-executed proven path's XSD-valid
delivery output** (105 columns, 11,035 records — §5), because the projector
fills them from `fields_registry.yaml` `regime_mapping` (all **107** codes are
registry-mapped: `registry_mapped_count: 107`). Only **2** codes are absent
from the proven output as well: `RREL20` / `RREL21` (Secondary Income /
Verification — both Optional, ND-allowed). The proven run consumed a
delivery-rules file whose `field_rules` are **identical to production's 68**
(the demo copy differs only by one demo-LEI enum entry, per its generated
header) — so the prior XSD success was achieved *with* the same 38-code rule
gap in place, proving the gap gates the new chain's onboarding contract, not
XML capability.

Machine-readable per-code reconciliation (all 38 rows, all required fields):
[`annex2_38_code_reconciliation.csv`](annex2_38_code_reconciliation.csv).

**Root cause of the park:** `field_rules` in `annex2_delivery_rules.yaml` is
the contract of the **new agentic chain** (onboarding target-coverage:
`target_coverage.py:248-261,453`; gate at `workflow.py:443-445` →
`NEEDS_CONFIGURATION`), which the current production conductor enforces at
onboarding for every regulatory run. It is **not** a capability gap of the
proven projector, and it is **not** something the OCC introduced,
misconfigured or mis-reported: the OCC surfaced, verbatim, a gate that the
production blob path would hit identically for the same delivery. The Phase 2
wording "31 codes lack delivery rules" was accurate about the gate but
imprecise about the count (38 total / 31 mandatory) and did not explain that
the proven XML path populates almost all of them anyway.

## 5. Prior-result rerun outcome

`python -m demo_platform.run_demo --all` was re-executed unchanged (outputs
land in the gitignored `demo_platform/workspace/`, removed after inspection).
Result — **reproduced exactly**:

```
[artefacts] ESMA Annex 2: 11,035 exposure records, 105 fields, 208.8 MB XML, XSD valid
[artefacts] OK  regulatoryOutput
```

matching commit `0ed7b4c`'s committed record (`"fields": 105,
"exposureRecords": 11035, "xsdValidated": true`). The demo later failed in an
unrelated stage-6 MI-fixture export (`ask_trakt_mi()` signature drift) — after
the regulatory output had completed and validated. The independent unit-level
check also passed:
`tests/test_xml_builder_annex2_shape_fixes.py::…_still_xsd_valid_for_prf_and_nprf`
(real `XMLSchema.validate` against the in-repo XSD).

**Field-by-field comparison with the OCC path:** the OCC's `mi_annex2` run
produces no Annex 2 output to compare — it parks at the onboarding
`NEEDS_CONFIGURATION` gate; and even past the gate, its final artefact is the
projector CSV (the *same projector, same registry/enum configs* as the proven
run's first stage). The reproduced delivery output's 105 columns therefore
represent the proven superset; the per-code disposition of every disputed
code is in the reconciliation CSV. No case was found where the OCC selected a
different or older implementation of any shared stage.

## 6. Conclusion

**The Operations Control Centre uses the correct, current production Annex 2
pipeline** — the same conductor, agents, adapter construction and projector
invocation as the blob-trigger production path, proven by instrumented
execution and call-graph evidence.

Two important qualifications:

1. **No current path — OCC or production — produces Annex 2 XML or XSD
   validation.** The only complete XML+XSD chain is the legacy gate chain,
   demo-only and explicitly retired as runtime. The prior XSD-validated
   results (105/104 fields) came from that chain. Anyone expecting the OCC's
   "MI + ESMA Annex 2" outcome to end in validated XML should know the
   current production path ends at the projected CSV; wiring a governed XML
   step is a Phase 3 decision, not a bug in the OCC's path selection.
2. The `NEEDS_CONFIGURATION` park is the current pipeline's own governance
   gate (68/107 codes ruled). Clearing it requires completing (or formally
   deferring codes in) `config/regime/annex2_delivery_rules.yaml` — an
   existing-file change that remains approval-gated (see report E).
