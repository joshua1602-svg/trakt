# Annex 2 Part B — pre-merge XML/XSD smoke test note

Branch under review: `claude/annex2-mapping-corrections-proposed`
Base: `claude/annex2-target-first-workflow-frnu7h`
Date: 2026-06-16

## 1. Annex 2 XML generation path

- **Is Annex 2 XML generation wired to the onboarding workflow directly: NO.**
  The onboarding workflow produces target-coverage / reconciliation artefacts
  (28a, 42–48), not XML.
- **Closest available XML/XSD path:** the gate-4b + gate-5 delivery tools that
  `engine/orchestrator/trakt_run.py` chains:
  1. `engine/gate_4b_delivery/annex2_delivery_normalizer.py` — applies the
     regime rules (`nd_allowed`, `default_value`, `boolean` transform,
     `enum_map`) to a projected Annex 2 CSV. **This is the component that the
     Part B changes reach.** (It does NOT read `validators.regex`, so the new
     date/LEI/country regexes are onboarding-side only.)
  2. `engine/gate_5_delivery/xml_builder_annex2.py` — builds the XML (element
     placement from `code_order` / `esma_model_structure` / ESMA code, NOT
     `workbook_semantic`) and validates against the XSD.
- **Commands run:**
  - Onboarding (corrected): `python -m engine.onboarding_agent.workflow ... --regime-config config/regime/annex2_delivery_rules.yaml ...`
  - Gate 4b: `python engine/gate_4b_delivery/annex2_delivery_normalizer.py --input <projected.csv> --rules <rules.yaml> --output-dir <dir>`
  - Gate 5: `python engine/gate_5_delivery/xml_builder_annex2.py --input <delivery_ready.csv> --output annex2.xml --mapping-workbook "DRAFT1auth.099.001.04_non-ABCP Underlying Exposure Report_Version_1.3.1.xlsx" --sheet DRAFT1auth.099.001.04 --code-order-yaml config/system/esma_code_order.yaml --xsd config/system/DRAFT1auth.099.001.04_1.3.0.xsd`
  - Fixtures: projected CSV `synthetic_demo/output/SYNTHETIC_ERE_Portfolio_012026_ESMA_Annex2_projected.csv`.

## 2. Onboarding workflow (corrected rules)

Run `run_annex2_premerge_xsd` — confirmed:

| check | result |
|---|---|
| 28a rows | **107** |
| 47 semantic mismatches | **0** |
| 43 missing_from_28a | **0** |
| 44 regime_broader | **0** |
| 48 proposals | **0** |

## 3. XML/XSD validation — a genuine defect was found and fixed

| run | rules | XSD result |
|---|---|---|
| before | base branch | **PASSED** |
| after  | Part B (as committed for review) | **FAILED** — `RREL13` value `manual` not in the Employment Status enum |
| after-fix | Part B + RREL13 fix | **PASSED** |

**Root cause (genuine Part B defect, not a fixture gap):** the original RREL13
rule mapped the source value `manual → OTHR`. The Part B correction replaced
RREL13's `enum_map` with a pure workbook-code identity map, dropping that
source-value synonym. The projected fixture carries `manual` for Employment
Status, so the gate-4b normalizer flagged an enum error and passed `manual`
through, which the XSD then rejected.

**Fix applied (permitted — test exposed a genuine defect):** restored the
`manual → OTHR` (and `MANUAL → OTHR`) synonym on RREL13 alongside the workbook
identity codes. Re-validation: **XSD PASSED**, normalizer reports **0** enum/
type errors. This was the only correction where a pre-existing source-value
synonym had been dropped (RREL26's synonyms were retained in Part B; the other
15 codes had no prior synonyms to lose).

## 4. Placement vs value impact

- **XML placement changed: no** — placement is driven by `code_order` /
  `esma_model_structure` / ESMA code; unaffected by the corrections.
- **XML schema path / order changed: no.**
- **Value-selection mapping changed for the 17 corrected codes: yes, by design**
  (`projected_source_field` re-pointed; `nd_allowed`/enum/boolean mechanics
  aligned to the workbook).
- **Schema validation: PASSES** after the RREL13 fix.

## 5. Tests

- `pytest tests/test_onboarding_annex2_workflow.py tests/test_xml_builder_annex2_shape_fixes.py tests/test_annex2_delivery_normalizer.py tests/test_regime_projector_annex2_guards.py tests/test_onboarding_target_coverage.py -q` → **112 passed**.
- `pytest tests -q` → see commit/PR status (run at merge time).

## 6. Verdict

- The corrected Annex 2 workflow runs; onboarding metrics are clean.
- Annex 2 XSD validation **PASSES** with the corrected rules after the RREL13
  source-synonym fix.
- No XML placement/order regression.
- No date/LEI/country/boolean validation failure introduced; the only enum issue
  (RREL13 `manual`) is fixed.
- Remaining caveats (unchanged from the 49 impact report): the 17
  `workbook_semantic` XML-path TODOs are non-blocking; several corrected codes
  fall back to a valid ND default until their canonical source columns are wired
  in real client data.
