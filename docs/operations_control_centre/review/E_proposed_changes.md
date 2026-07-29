# Review E — Proposed changes requiring approval

None of these are implemented. Each is submitted for individual approval.

## E1. Complete (or formally defer) the Annex 2 delivery rules — clears R2

- **Exact file:** `config/regime/annex2_delivery_rules.yaml` (existing).
- **Change:** add `field_rules` entries for the 30 `pending_regime_rule`
  codes (start from the reconciliation CSV; 36/38 already have proven
  registry-driven population to copy semantics from), move genuinely
  non-applicable codes into `reconciliation_scope.deferred_fields`, and decide
  RREL20/RREL21 (Optional, ND-allowed — likely `confirm_ND` defaults).
- **Purpose:** unblock every regulatory onboarding (OCC and production blob
  path park identically today). ~+150–300 YAML lines.
- **Upstream impact:** none (config only).
- **Downstream impact:** onboarding target-coverage gate clears
  (`NEEDS_CONFIGURATION` → proceed); Gate 4b delivery normaliser reads the
  same file — new rules alter normalisation for those codes, which is the
  intended effect but must be regression-checked.
- **Regression risk:** medium — the proven XML path consumes this file at the
  normaliser stage.
- **Expected effect on the prior 105-field result:** none-to-positive; the
  105/XSD-valid run succeeded with today's 68 rules, and added rules must be
  verified not to change existing column treatment. Re-run
  `demo_platform.run_demo --orchestrate --artefacts` and the
  `due_diligence/50_annex2_premerge_xsd_smoke_test.md` commands before/after.
- **Rollback:** `git revert` of the config commit.

## E2. Honest labelling of the Annex 2 outcome in the OCC — mitigates R1 now

- **Exact files:** `operations_control/language.py` (stage label
  "Regulatory report" → "Regulatory data prepared"),
  `operations_control/adapters.py` (projection-stage summary sentence to state
  the submission XML is a separate, not-yet-wired step),
  `frontend/operations-control-ui/src/lib/copy.ts` (matching copy).
- **Purpose:** stop implying a submittable XML results from the `mi_annex2`
  outcome while R1 is open. ~±15 lines.
- **Impact:** OCC-only; no pipeline effect. **Testing:** language contract
  tests + UI copy test. **Rollback:** revert.

## E3. Consolidation batch (scenario B) — clears R3, R6, R8, R9; ~−1,000 lines

- **Exact files:** `frontend/…/src/api/MockOpsClient.ts` (slim to fixture or
  dynamic import, −500…−850), `operations_control/api/presenters.py` merge
  into `language.py` (−~80), `operations_control/classification.py` fold into
  `engine.py` (−1 file), `operations_control/adapters.py` split the two
  F-grade translators into per-stage functions, `operations_control/api/app.py`
  remove unused `Field` import + either add the `retire` route or delete the
  engine method, drop unread `result_history` writes; add tests to raise
  `app.py` coverage ≥80%; either wrap API responses in
  `trakt_core.GovernedResult` or amend design doc 05 (pick one, R3);
  document the two derived-state invariants (R5, R7) in
  `operations_control/README.md`.
- **Impact:** OCC-only, zero behaviour change intended; full backend + UI test
  suites are the gate. **Rollback:** revert (single batch commit).

## E4. Phase 3 decision — governed Annex 2 XML step (resolves R1 properly)

- **Options:** (a) wire the existing `annex2_delivery_normalizer` +
  `xml_builder_annex2` + XSD as a governed OCC publication step (fastest;
  contradicts the "retire the runtime" doc unless that policy is revisited);
  (b) finish `delivery_xml_agent` v2 to production XML and wire it into the
  conductor (aligned with the documented direction; larger).
- **Exact files:** to be scoped after direction is chosen — new adapter stage
  in `operations_control/adapters.py`+`engine.py` for (a); engine/agent work
  for (b). Requesting a direction decision, not yet a diff.
- **Expected effect on prior result:** (a) reuses the code that produced it.

## E5. Housekeeping backlog (pre-existing pipeline, low priority — R10)

Root `.gitignore` +1 line (`.ops_state/`); delete the unused duplicate XSD at
repo root or document why two copies exist; delete or wire
`config/delivery/annex2_xml_structure_contract.yaml`; fix stale
`trakt_run.py:1163` stage label; de-duplicate the double projector run in
`router.py`. Each is an existing-file change; approve individually.
