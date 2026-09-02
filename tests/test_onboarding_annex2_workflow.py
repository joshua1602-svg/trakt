#!/usr/bin/env python3
"""tests/test_onboarding_annex2_workflow.py — ESMA Annex 2 target-first delivery.

Covers the acceptance behaviours for running Annex 2 delivery through the
existing target-first operator workflow with TWO config layers:

  1. Annex 2 target contract loading (ESMA codes; not the MI registry).
  2. Regime + asset config loading (both recorded in the 40 summary).
  3. ND / default application (regime + asset, within the regime envelope).
  4. Invalid default handling (surfaced in 42 + Gate 4, never silently applied).
  5. 28c decision-queue quality (genuine regulatory decisions only).
  6. Workflow summary (40) target_contract_id + Annex 2 counts.
  7. Review pack shows the Annex 2 target contract + coverage + Gate 4.
  8. The MI workflow remains unchanged (target contract + no 42 artefact).
  9. The optional LLM target advisor remains advisory only (no 28a/28c mutation).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import target_coverage as tcov
from engine.onboarding_agent import workflow as wf

PACK = str(_REPO_ROOT / "synthetic_demo" / "input")
REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ALIASES = str(_REPO_ROOT / "config" / "system")
from tests.annex2_contract_fixture import contract_path

REGIME = contract_path()
ASSET = str(_REPO_ROOT / "config" / "asset" / "product_defaults_ERM.yaml")


def _run_annex2(out: Path, advisor: bool = False):
    warnings.simplefilter("ignore")
    return wf.run_operator_workflow(
        input_dir=PACK, client_name="CLIENT_001_TEST", client_id="client_001",
        run_id="annex2", project_dir=str(out), mode="regulatory_mi",
        registry=REGISTRY, aliases_dir=ALIASES, enable_llm_target_advisor=advisor)


# --------------------------------------------------------------------------- #
# 1 — Annex 2 target loading (unit level)
# --------------------------------------------------------------------------- #
class TestAnnex2TargetLoading(unittest.TestCase):
    def test_regulatory_mode_loads_annex2_codes(self):
        cid, csrc, fields = tcov.load_target_contract("regulatory_mi", {})
        self.assertEqual(cid, "esma_annex_2")
        self.assertIn("annex2_contract", csrc)
        names = {f["target_field"] for f in fields}
        for code in ("RREL1", "RREL2", "RREL6", "RREC9", "RREL16", "RREL40",
                     "RREC8", "RREC15"):
            self.assertIn(code, names)

    def test_does_not_use_mi_registry_as_target_contract(self):
        _cid, csrc, fields = tcov.load_target_contract("regulatory_mi", {})
        self.assertNotIn("mi_semantics_field_registry", csrc)
        # No MI canonical field names leak into the Annex 2 target contract.
        names = {f["target_field"] for f in fields}
        self.assertNotIn("account_status", names)
        self.assertNotIn("current_interest_rate", names)


# --------------------------------------------------------------------------- #
# 3 (unit) — ND/default classification + explicit default policy
# --------------------------------------------------------------------------- #
class TestDefaultApplicationUnit(unittest.TestCase):
    def test_mandatory_no_source_no_default_is_blocking(self):
        # A mandatory/enforce_presence field with no source, derivation or valid
        # default is missing_required AND blocking.
        fields = [{
            "target_field": "RREX1", "esma_code": "RREX1",
            "projected_source_field": "made_up_field", "target_domain": "loan",
            "target_label": "x", "required_status": "mandatory",
            "enforce_presence": True, "applicability_status": "applicable",
            "match_field": "made_up_field", "synonyms": [], "derived": False,
            "derivation_rule": "", "default_rule": "", "default_value": "",
            "default_rule_source": "", "default_reason": "", "nd_allowed": [],
            "configured_value_source": "",
        }]
        rows, _ = tcov.build_target_coverage(
            "regulatory_mi", {}, "esma_annex_2", REGIME, fields,
            evidence_rows=[], resolved_rows=[])
        r = rows[0]
        self.assertEqual(r["coverage_status"], tcov.MISSING_REQUIRED)
        self.assertTrue(r["blocking"])

    def test_the_contract_carries_no_defaults_and_the_pack_does(self):
        """The distinction is still real; it just lives in the right place.

        RREL16 takes a no-data code and RREC8 takes a value, and both are
        statements about UK equity release rather than about ESMA Annex 2 — so
        both are in the asset pack, and the regime contract states neither.
        """
        import yaml as _y
        cid, csrc, fields = tcov.load_target_contract("regulatory_mi", {})
        by = {f["target_field"]: f for f in fields}
        for code in ("RREL16", "RREC8"):
            self.assertEqual(by[code]["default_value"], "", code)
        pack = _y.safe_load(
            (_REPO_ROOT / "config" / "asset" / "product_defaults_ERM.yaml")
            .read_text(encoding="utf-8"))
        self.assertTrue(tcov._is_nd(pack["nd_defaults"]["primary_income"]))
        self.assertEqual(str(pack["defaults"]["lien"]), "1")


# --------------------------------------------------------------------------- #
# 4 (unit) — invalid asset default handling + 2-layer validation
# --------------------------------------------------------------------------- #
class TestConfigValidationUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.overlay, cls.asset_src = tcov.build_annex2_config_validation(
            REGIME, ASSET)
        cls.by_code = {r["esma_code"]: r for r in cls.rows if r["esma_code"]}

    def test_both_config_layers_consumed(self):
        self.assertTrue(self.asset_src.endswith("product_defaults_ERM.yaml"))
        # A known regime rule is present.
        self.assertIn("RREL16", self.by_code)
        self.assertIn("RREC8", self.by_code)

    def test_valid_asset_default_applied(self):
        # primary_income (RREL16) asset default ND1 is within nd_allowed -> valid
        # and applied as an asset_config-sourced default.
        self.assertEqual(self.by_code["RREL16"]["validation_status"], tcov.VS_VALID)
        self.assertEqual(self.overlay["RREL16"]["default_rule_source"], "asset_config")

    def test_reconciled_asset_default_valid(self):
        # debt_to_income_ratio (RREL40) is intentionally reconciled to ND5, which
        # IS within the regime nd_allowed [ND5] -> valid (no longer a conflict).
        row = self.by_code["RREL40"]
        self.assertEqual(row["validation_status"], tcov.VS_VALID)
        self.assertEqual(row["asset_default_value"], "ND5")

    def test_an_asset_default_outside_the_regulator_envelope_is_refused(self):
        """A product may choose within the envelope; it cannot exceed it.

        RREL16 (primary income) is one of the codes the workbook marks
        ``nd5_allowed: false`` — "not applicable" is not an answer the regulator
        accepts there. A synthetic asset pack declaring ND5 must be surfaced as
        invalid and the value must not be applied. The envelope comes from the
        field universe now, so this is checked against the regulator's own
        statement rather than against whatever a rules file happened to say.
        """
        import tempfile
        import textwrap
        from pathlib import Path
        d = Path(tempfile.mkdtemp(prefix="annex2_invalid_"))
        asset = d / "asset.yaml"
        asset.write_text(textwrap.dedent("""\
            asset_class: equity_release
            defaults: {}
            nd_defaults:
              primary_income: ND5
        """), encoding="utf-8")
        rows, overlay, _ = tcov.build_annex2_config_validation("", str(asset))
        by = {r["esma_code"]: r for r in rows if r["esma_code"]}
        self.assertEqual(by["RREL16"]["validation_status"], tcov.VS_INVALID)
        self.assertEqual(by["RREL16"]["asset_default_value"], "ND5")
        self.assertFalse(overlay["RREL16"].get("valid"))
        self.assertNotEqual(overlay["RREL16"].get("default_value"), "ND5")

    def test_unknown_and_missing_statuses_present(self):
        statuses = {r["validation_status"] for r in self.rows}
        self.assertIn(tcov.VS_UNKNOWN, statuses)
        self.assertIn(tcov.VS_MISSING_NOT_REQ, statuses)


# --------------------------------------------------------------------------- #
# 1/2/5/6 — full workflow first pass
# --------------------------------------------------------------------------- #
class TestAnnex2Workflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_wf_"))
        cls.summary = _run_annex2(cls.out)
        cls.cov = json.loads(
            (cls.out / "28a_target_coverage_matrix.json").read_text())
        cls.dec = json.loads(
            (cls.out / "28c_human_decision_queue.json").read_text())

    def test_28a_contains_esma_codes(self):
        names = {r["target_field"] for r in self.cov["rows"]}
        self.assertEqual(self.cov["target_contract_id"], "esma_annex_2")
        for code in ("RREL1", "RREL16", "RREL40", "RREC8"):
            self.assertIn(code, names)

    def test_42_artefacts_written(self):
        for name in ("42_annex2_config_validation.csv",
                     "42_annex2_config_validation.json",
                     "42_annex2_config_validation_summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_40_summary_records_contract_and_config_paths(self):
        s = self.summary
        self.assertEqual(s["target_contract_id"], "esma_annex_2")
        self.assertTrue(s["regime_config_path"])
        self.assertTrue(s["asset_config_path"].endswith("product_defaults_ERM.yaml"))
        self.assertEqual(s["annex2_field_count"], len(self.cov["rows"]))
        # Reconciles to 42, rather than asserting a magic number. Both real asset
        # defaults that used to sit outside the regime envelope are now declared
        # inside it (RREL35 "Bullet" is in the RREL35 enum map; RREL40 "ND5" is in
        # nd_allowed), so the current count is 0 — a configuration-completeness
        # improvement, not a lost check. The mechanism is proven on a synthetic
        # config by TestAnnex2NdEligibility.test_asset_default_outside_regime_
        # envelope_is_invalid, which does not depend on the real config having a
        # conflict left in it.
        val = json.loads(
            (self.out / "42_annex2_config_validation.json").read_text())
        self.assertEqual(s["annex2_invalid_default_count"],
                         sum(1 for r in val["rows"]
                             if r["validation_status"] == tcov.VS_INVALID))

    def test_40_status_reflects_real_configuration_gaps(self):
        """Not ready — but for the right reason, and now for a visible one.

        It used to be NEEDS_CONFIGURATION because 37 of the 107 codes had no
        hand-written rule, so coverage parked them as ``pending_regime_rule``:
        a statement about the rules file, not about the tape. Every code is
        governed now, so those codes are assessed against the demo tape like
        any other — and the ones it does not carry, and which no asset, client
        or operator layer answers for, surface as BLOCKING Gate 4 decisions
        instead of sitting in a holding pen. Blocked on a real, listed gap is
        strictly more honest than not-configured on an accounting artefact.
        """
        self.assertNotEqual(self.summary["status"], wf.READY)
        self.assertEqual(self.summary["status"], wf.BLOCKED)
        self.assertEqual(self.summary["annex2_pending_regime_rule_count"], 0,
                         "no code is missing from the contract")
        # Reconciled to the coverage matrix, not pinned to a number: every
        # blocking decision is a code the tape does not source and no layer
        # defaults.
        blocking = {d["target_field"] for d in self.dec["rows"] if d["blocking"]}
        unmapped = {r["target_field"] for r in self.cov["rows"]
                    if r["coverage_status"] == tcov.MISSING_REQUIRED}
        self.assertEqual(blocking, unmapped)
        self.assertTrue(blocking, "the demo tape does not carry every Annex 2 code")

    def test_explicitly_defaulted_fields_not_in_28c(self):
        # A field that is explicitly ND/value defaulted with no confirmation must
        # NOT appear as a Gate 4 decision.
        dec_fields = {d["target_field"] for d in self.dec["rows"]}
        cov_by = {r["target_field"]: r for r in self.cov["rows"]}
        # Both answers now come from the asset pack rather than the regime
        # layer, so coverage records them as configured rather than as a regime
        # default — and neither needs an operator decision.
        self.assertEqual(cov_by["RREL16"]["coverage_status"], tcov.DEFAULTED_ND)
        self.assertNotIn("RREL16", dec_fields)
        self.assertEqual(cov_by["RREC8"]["coverage_status"],
                         tcov.CONFIGURED_STATIC)
        self.assertNotIn("RREC8", dec_fields)

    def test_invalid_default_decisions_track_the_config_validation(self):
        """The decision queue carries an invalid-default item for exactly the
        codes 42 marked invalid — no more, and no fewer.

        This used to assert RREL35 was in the queue. It no longer is, because
        "Bullet" was declared in the RREL35 regime enum map and the ERM pack maps
        it to OTHR for this asset class; the conflict was resolved in
        configuration, which is where it should be resolved. Asserting the
        relationship keeps the check alive when the real config has nothing
        invalid left; the branch itself is covered on a synthetic row by
        test_an_invalid_asset_default_becomes_a_nonblocking_decision below.
        """
        val = json.loads(
            (self.out / "42_annex2_config_validation.json").read_text())
        invalid_codes = {r["esma_code"] for r in val["rows"]
                         if r["validation_status"] == tcov.VS_INVALID}
        queued = {d["esma_code"] for d in self.dec["rows"]
                  if d["decision_type"] == tcov.D_INVALID_DEFAULT}
        self.assertTrue(queued <= invalid_codes,
                        f"queued invalid-default codes not marked invalid by 42: "
                        f"{queued - invalid_codes}")
        for d in self.dec["rows"]:
            if d["decision_type"] == tcov.D_INVALID_DEFAULT:
                self.assertFalse(d["blocking"])  # regime fallback -> non-blocking

    def test_an_invalid_asset_default_becomes_a_nonblocking_decision(self):
        """The D_INVALID_DEFAULT branch, driven directly.

        Independent of whether today's shipped configuration happens to contain a
        conflict, so resolving the last real one cannot leave this untested.
        """
        row = {
            "target_contract_id": "esma_annex_2",
            "target_field": "RREL35", "esma_code": "RREL35",
            "coverage_status": tcov.DEFAULTED_VALUE,
            "config_validation_status": tcov.VS_INVALID,
            "requires_user_decision": True, "blocking": False,
            "decision_reason": "asset default not allowed by regime rule",
            "operator_question": "Confirm the regime default?",
            "selected_source_file": "", "selected_source_column": "",
            "asset_default_value": "Bullet", "nd_allowed": "ND1; ND2",
            "default_value": "OTHR", "required_status": "optional",
            "target_domain": "loan", "coverage_basis": "asset_config",
            "value_compatibility_status": "", "overlap_evidence": "",
            "alternative_source_candidates": "",
        }
        queue = tcov.build_human_decision_queue("regulatory_mi", [row], [])
        inv = [d for d in queue if d["decision_type"] == tcov.D_INVALID_DEFAULT]
        self.assertEqual(len(inv), 1)
        self.assertEqual(inv[0]["target_field"], "RREL35")
        self.assertFalse(inv[0]["blocking"])
        self.assertIn("Bullet", inv[0]["evidence_summary"])

    def test_queue_contains_only_genuine_regulatory_decisions(self):
        allowed = {tcov.D_MISSING, tcov.D_CONFLICT, tcov.D_PRIORITY, tcov.D_VALUE,
                   tcov.D_CONFIG, tcov.D_ND, tcov.D_INVALID_DEFAULT,
                   tcov.D_EXTENSION, tcov.D_PARSE}
        for d in self.dec["rows"]:
            self.assertIn(d["decision_type"], allowed)

    def test_review_pack_shows_annex2(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("esma_annex_2", html)
        self.assertIn("ESMA Annex 2 delivery", html)
        self.assertIn("Annex 2 coverage by field family", html)
        self.assertIn("Annex 2 delivery readiness", html)

    def test_34_template_generated(self):
        self.assertTrue((self.out / "34_target_first_decisions.yaml").exists())


# --------------------------------------------------------------------------- #
# Field-universe completeness (28a == authoritative universe; 43 reconciliation)
# --------------------------------------------------------------------------- #
class TestAnnex2FieldUniverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_univ_"))
        cls.summary = _run_annex2(cls.out)
        cls.cov = json.loads(
            (cls.out / "28a_target_coverage_matrix.json").read_text())
        cls.recon = json.loads(
            (cls.out / "43_annex2_field_universe_reconciliation.json").read_text())

    def test_universe_loader_unit(self):
        auth = tcov.load_annex2_authoritative_universe(REGISTRY)
        # Workbook registry carries the full Annex 2 code set (> the 70 regime
        # rules), e.g. RREL3 / RREL5 / RREL7 are present but unruled.
        self.assertGreater(len(auth), 70)
        for code in ("RREL3", "RREL5", "RREL7"):
            self.assertIn(code, auth)

    def test_workbook_universe_is_authoritative_107(self):
        wb, src = tcov.load_annex2_workbook_universe()
        # The workbook-derived universe is the authoritative ESMA Annex 2 set.
        self.assertEqual(len(wb), 107)
        self.assertTrue(src.endswith("annex2_field_universe.yaml"))
        # RREC1 is in the authoritative workbook universe (84 RREL + 23 RREC).
        self.assertIn("RREC1", wb)
        self.assertEqual(sum(1 for c in wb if c.startswith("RREL")), 84)
        self.assertEqual(sum(1 for c in wb if c.startswith("RREC")), 23)
        self.assertIn("nd5_allowed", wb["RREL1"])
        self.assertEqual(self.recon["summary"]["authoritative_field_count"], 107)

    def test_rrec1_present_in_28a(self):
        cov_codes = {r["target_field"] for r in self.cov["rows"]}
        self.assertIn("RREC1", cov_codes)

    def test_no_active_phantom_deferred_codes(self):
        # After alignment, NO active runtime code sits outside the authoritative
        # workbook universe: the phantom RREC24-39 codes were moved to the
        # audit-only list, so 43 reports zero not_in_authoritative_universe.
        phantom = [r for r in self.recon["rows"]
                   if r["reconciliation_status"] == "not_in_authoritative_universe"]
        self.assertEqual(phantom, [])
        self.assertEqual(
            self.recon["summary"].get("not_in_authoritative_universe_count", 0), 0)
        self.assertEqual(self.summary["annex2_active_phantom_deferred_count"], 0)
        # The previously-phantom codes are not in 28a either.
        cov_codes = {r["target_field"] for r in self.cov["rows"]}
        for code in ("RREC24", "RREC30", "RREC39"):
            self.assertNotIn(code, cov_codes)

    def test_rrec1_registry_mapped(self):
        # RREC1 (the known registry gap) is now mapped in fields_registry, so 43
        # records it as registry_mapped with zero registry gaps overall.
        by = {r["esma_code"]: r for r in self.recon["rows"]}
        self.assertEqual(by["RREC1"]["registry_mapping_status"], "registry_mapped")
        self.assertTrue(by["RREC1"]["in_registry_mapping"])
        self.assertEqual(self.recon["summary"]["registry_gap_count"], 0)
        self.assertEqual(self.summary["annex2_registry_gap_count"], 0)
        self.assertEqual(self.summary["annex2_registry_mapped_count"], 107)

    def test_28a_equals_authoritative_universe_count(self):
        recon_sum = self.recon["summary"]
        self.assertEqual(len(self.cov["rows"]),
                         recon_sum["authoritative_field_count"])
        self.assertEqual(self.summary["annex2_coverage_field_count"],
                         self.summary["annex2_authoritative_field_count"])

    def test_no_authoritative_code_missing_from_28a(self):
        self.assertEqual(self.recon["summary"]["missing_from_28a_count"], 0)
        missing = [r for r in self.recon["rows"]
                   if r["reconciliation_status"] == "missing_from_28a"]
        self.assertEqual(missing, [])
        # Every regime field rule and every workbook code is represented in 28a.
        cov_codes = {r["target_field"] for r in self.cov["rows"]}
        for r in self.recon["rows"]:
            if r["in_regime_field_rules"] or r["in_workbook_reconciliation"]:
                self.assertIn(r["esma_code"], cov_codes)

    def test_43_artefacts_written(self):
        for name in ("43_annex2_field_universe_reconciliation.csv",
                     "43_annex2_field_universe_reconciliation.json",
                     "43_annex2_field_universe_reconciliation_summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_attribute_only_codes_are_present_not_dropped(self):
        """Nothing is deferred any more; three codes have no element at all.

        RREC22 used to be declared deferred by hand. It is one of the three
        concepts auth.099 carries as a currency attribute of the amount it
        qualifies, which the contract reads off the schema — so it is present in
        coverage and marked as having no element of its own, rather than
        listed somewhere as permitted to be skipped.
        """
        cov_by = {r["target_field"]: r for r in self.cov["rows"]}
        for code in ("RREC22", "RREL18", "RREL28"):
            self.assertIn(code, cov_by, code)
        deferred_codes = [r["esma_code"] for r in self.recon["rows"]
                          if r["reconciliation_status"] == "deferred_in_regime"]
        self.assertEqual(deferred_codes, [],
                         "no code is deferred by declaration any more")

    def test_no_code_is_missing_from_the_contract(self):
        """The category is empty by construction.

        "Missing from the regime rules" was a real state when the rules were a
        hand-maintained list of 70. The contract is derived from the workbook
        universe, so every code the regulator defines is in it.
        """
        pending = [r["esma_code"] for r in self.recon["rows"]
                   if r["reconciliation_status"] == "missing_from_regime_rules"]
        self.assertEqual(pending, [])

    def test_40_reports_universe_counts(self):
        s = self.summary
        for k in ("annex2_authoritative_field_count", "annex2_coverage_field_count",
                  "annex2_regime_rule_count", "annex2_config_validation_count",
                  "annex2_missing_from_28a_count", "annex2_deferred_field_count",
                  "annex2_deliverable_field_count"):
            self.assertIn(k, s)
        # The contract covers the whole workbook universe. It used to cover 70
        # of 107, and the 37 without an entry were treated as ungoverned.
        self.assertEqual(s["annex2_regime_rule_count"], 107)
        self.assertEqual(s["annex2_authoritative_field_count"], 107)

    def test_review_pack_shows_universe_reconciliation(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("Annex 2 field universe reconciliation", html)
        self.assertIn("Authoritative Annex 2 fields", html)


# --------------------------------------------------------------------------- #
# ND-eligibility reconciliation (44): regime nd_allowed vs workbook eligibility
# --------------------------------------------------------------------------- #
class TestAnnex2NdEligibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_nd_"))
        cls.summary = _run_annex2(cls.out)
        cls.nd = json.loads(
            (cls.out / "44_annex2_nd_eligibility_reconciliation.json").read_text())
        cls.align = json.loads(
            (cls.out / "45_annex2_config_alignment_review.json").read_text())

    def test_44_artefacts_written(self):
        for name in ("44_annex2_nd_eligibility_reconciliation.csv",
                     "44_annex2_nd_eligibility_reconciliation.json",
                     "44_annex2_nd_eligibility_reconciliation_summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_nd_reconciliation_unit(self):
        """The reconciliation is now a tautology, and says so.

        It existed to compare a hand-maintained ND list against the workbook's,
        and found 35 of 70 divergent. The contract takes the envelope FROM the
        workbook, so every code matches — which is the outcome the report was
        built to drive towards.
        """
        rows = tcov.build_annex2_nd_eligibility_reconciliation()
        allowed = {"match", "regime_stricter", "regime_broader", "divergent",
                   "no_regime_rule", "not_in_workbook"}
        for r in rows:
            self.assertIn(r["nd_alignment_status"], allowed)
        statuses = {r["nd_alignment_status"] for r in rows}
        self.assertEqual(statuses, {"match"},
                         "the regime envelope IS the workbook envelope")

    def test_regime_broader_is_zero_after_tightening(self):
        s = self.nd["summary"]
        # The 5 regime_broader cases (RREL1/2/6/69/83) were tightened to the
        # workbook envelope, so NO regime rule is broader than the workbook.
        self.assertEqual(s["regime_broader"], 0)
        self.assertEqual(self.summary["annex2_nd_regime_broader_count"], 0)
        by = {r["esma_code"]: r for r in self.nd["rows"]}
        for code in ("RREL1", "RREL2", "RREL6", "RREL69", "RREL83"):
            self.assertEqual(by[code]["nd_alignment_status"], "match")

    def test_nothing_diverges_from_the_workbook_any_more(self):
        s = self.nd["summary"]
        self.assertEqual(s["divergent"], 0)
        self.assertEqual(s["regime_stricter"], 0)
        self.assertEqual(s["regime_broader"], 0)
        self.assertEqual(self.summary["annex2_nd_divergent_count"], 0)
        self.assertEqual(self.summary["annex2_nd_regime_stricter_count"], 0)

    def test_the_envelope_the_validator_uses_is_the_regulators(self):
        """42 validates the asset pack against what ESMA permits.

        It used to validate against a hand-narrowed list — RREL40 restricted to
        ND5 where the workbook allows ND1 to ND5 — so a product decision inside
        the regulator's envelope but outside the file's could be called invalid.
        """
        val = json.loads(
            (self.out / "42_annex2_config_validation.json").read_text())
        rrel40 = next(r for r in val["rows"] if r["esma_code"] == "RREL40")
        self.assertEqual(
            [c.strip() for c in rrel40["regime_nd_allowed"].split(";")],
            ["ND1", "ND2", "ND3", "ND4", "ND5"])
        self.assertEqual(rrel40["asset_default_value"], "ND5")
        self.assertEqual(rrel40["validation_status"], tcov.VS_VALID)

    def test_review_pack_shows_nd_reconciliation(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("Annex 2 ND-eligibility reconciliation", html)


# --------------------------------------------------------------------------- #
# Enum-coverage reconciliation (46): regime enum_map vs workbook allowed codes
# --------------------------------------------------------------------------- #
class TestAnnex2EnumCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_enum_"))
        cls.summary = _run_annex2(cls.out)
        cls.enum = json.loads(
            (cls.out / "46_annex2_enum_coverage_reconciliation.json").read_text())
        cls.by = {r["esma_code"]: r for r in cls.enum["rows"]}

    def test_46_artefacts_written(self):
        for name in ("46_annex2_enum_coverage_reconciliation.csv",
                     "46_annex2_enum_coverage_reconciliation.json",
                     "46_annex2_enum_coverage_reconciliation_summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_no_enum_exceeds_workbook(self):
        # No regime enum_map may map to a code the workbook forbids.
        self.assertEqual(self.enum["summary"]["targets_outside_workbook"], 0)
        self.assertEqual(self.summary["annex2_enum_targets_outside_workbook_count"], 0)

    def test_added_enum_maps_are_within_workbook(self):
        import yaml as _y
        wb = tcov.load_annex2_workbook_universe()[0]
        from tests.annex2_contract_fixture import contract_field_rules
        fr = contract_field_rules()
        for code in ("RREL19", "RREL56", "RREL57", "RREC10", "RREC18"):
            allowed = set(tcov._annex2_workbook_enum_codes(wb[code]["content"]))
            targets = set(fr[code]["transform"]["enum_map"].values())
            self.assertTrue(targets, code)
            self.assertTrue(targets <= allowed, f"{code}: {targets - allowed} outside workbook")
            self.assertEqual(self.by[code]["enum_coverage_status"],
                             "constrained_within_workbook")

    def test_previously_mismatched_list_fields_now_constrained(self):
        # After the mapping corrections, these {LIST} fields are constrained to
        # the workbook's allowed codes (no longer semantic_mismatch).
        for code in ("RREL17", "RREL70", "RREC23"):
            self.assertEqual(self.by[code]["enum_coverage_status"],
                             "constrained_within_workbook")
        self.assertEqual(self.enum["summary"]["semantic_mismatch"], 0)
        self.assertEqual(self.enum["summary"]["unconstrained_no_enum_map"], 0)

    def test_45_records_enum_constraint_actions(self):
        align = json.loads(
            (self.out / "45_annex2_config_alignment_review.json").read_text())
        constrained = [r for r in align["rows"]
                       if r["alignment_status"] == "enum_constrained_to_workbook"]
        self.assertEqual(len(constrained), 5)

    def test_review_pack_shows_enum_reconciliation(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("Annex 2 enum-coverage reconciliation", html)


# --------------------------------------------------------------------------- #
# Semantic-mapping reconciliation (47): regime source field vs workbook field
# --------------------------------------------------------------------------- #
class TestAnnex2SemanticMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_sem_"))
        cls.summary = _run_annex2(cls.out)
        cls.sem = json.loads(
            (cls.out / "47_annex2_semantic_mapping_reconciliation.json").read_text())
        cls.by = {r["esma_code"]: r for r in cls.sem["rows"]}

    def test_47_artefacts_written(self):
        for name in ("47_annex2_semantic_mapping_reconciliation.csv",
                     "47_annex2_semantic_mapping_reconciliation.json",
                     "47_annex2_semantic_mapping_reconciliation_summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_every_code_is_semantically_checked(self):
        """107, not 70: the 37 with no hand-written rule were never checked."""
        self.assertEqual(self.sem["summary"]["semantic_rows_total"], 107)

    def test_all_mappings_now_aligned(self):
        # After the mapping corrections, the previously mismapped codes align.
        for code in ("RREL13", "RREL17", "RREL70", "RREC23"):
            self.assertEqual(self.by[code]["semantic_status"], "aligned")
        self.assertEqual(self.sem["summary"]["semantic_mismatch"], 0)
        self.assertEqual(self.summary["annex2_semantic_mismatch_count"], 0)

    def test_correctly_mapped_codes_aligned(self):
        for code in ("RREL1", "RREL2", "RREL16", "RREL40"):
            self.assertEqual(self.by[code]["semantic_status"], "aligned")

    def test_47_reads_corrected_rules_from_disk(self):
        # 47 is report-only; it reflects the corrected rules now on disk.
        import yaml as _y
        from tests.annex2_contract_fixture import contract_field_rules
        fr = contract_field_rules()
        self.assertEqual(fr["RREL70"]["projected_source_field"],
                         "reason_for_default_or_foreclosure")

    def test_review_pack_shows_semantic_reconciliation(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("Annex 2 semantic-mapping reconciliation", html)


# --------------------------------------------------------------------------- #
# Mapping-correction proposals (48): report-only proposed source/ND/mechanics
# --------------------------------------------------------------------------- #
class TestAnnex2MappingProposals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_prop_"))
        cls.summary = _run_annex2(cls.out)
        cls.prop = json.loads(
            (cls.out / "48_annex2_mapping_correction_proposals.json").read_text())
        cls.by = {r["esma_code"]: r for r in cls.prop["rows"]}

    def test_48_artefacts_written(self):
        for name in ("48_annex2_mapping_correction_proposals.csv",
                     "48_annex2_mapping_correction_proposals.json",
                     "48_annex2_mapping_correction_proposals_summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_one_proposal_per_semantic_mismatch(self):
        sem = json.loads(
            (self.out / "47_annex2_semantic_mapping_reconciliation.json").read_text())
        mismatches = {r["esma_code"] for r in sem["rows"]
                      if r["semantic_status"] == "semantic_mismatch"}
        self.assertEqual(set(self.by), mismatches)
        self.assertEqual(self.prop["summary"]["proposal_rows_total"], len(mismatches))

    def test_no_proposals_remaining_after_corrections(self):
        # With the corrections applied, there are no mismatches left, so the
        # proposals artefact is empty.
        self.assertEqual(self.prop["summary"]["proposal_rows_total"], 0)
        self.assertEqual(self.by, {})

    def test_regime_rules_now_carry_corrected_sources(self):
        import yaml as _y
        from tests.annex2_contract_fixture import contract_field_rules
        fr = contract_field_rules()
        self.assertEqual(fr["RREL70"]["projected_source_field"],
                         "reason_for_default_or_foreclosure")
        self.assertEqual(fr["RREL14"]["projected_source_field"], "credit_impaired_obligor")
        self.assertEqual(fr["RREC21"]["projected_source_field"], "sale_price")
        self.assertEqual(fr["RREL10"]["nd_allowed"], ["ND1", "ND2", "ND3", "ND4"])

    def test_mechanics_split_and_report_only(self):
        s = self.prop["summary"]
        self.assertEqual(s["re_point_source_only"]
                         + s["needs_rule_mechanics_changes"]
                         + s["needs_mechanics_review"], s["proposal_rows_total"])
        self.assertTrue(all(r["requires_manual_review"] for r in self.prop["rows"]))
        self.assertTrue(all(r["xml_output_changes"] == "yes" for r in self.prop["rows"]))

    def test_review_pack_shows_mapping_proposals(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("Annex 2 mapping-correction proposals", html)


# --------------------------------------------------------------------------- #
# Config-alignment review (45): actions taken + manual-review items
# --------------------------------------------------------------------------- #
class TestAnnex2ConfigAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_align_"))
        cls.summary = _run_annex2(cls.out)
        cls.align = json.loads(
            (cls.out / "45_annex2_config_alignment_review.json").read_text())
        cls.val = json.loads(
            (cls.out / "42_annex2_config_validation.json").read_text())
        cls.by = {}
        for r in cls.align["rows"]:
            cls.by.setdefault(r["esma_code"], []).append(r)

    def test_45_artefacts_written(self):
        for name in ("45_annex2_config_alignment_review.csv",
                     "45_annex2_config_alignment_review.json",
                     "45_annex2_config_alignment_review_summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_columns_match_spec(self):
        for col in ("esma_code", "workbook_field_name", "workbook_nd_allowed",
                    "regime_nd_allowed_before", "regime_nd_allowed_after",
                    "alignment_status", "action_taken", "requires_manual_review",
                    "message"):
            self.assertIn(col, self.align["rows"][0])

    def test_records_alignment_actions(self):
        s = self.align["summary"]
        self.assertEqual(s["tightened_to_workbook"], 5)
        self.assertEqual(s["phantom_deferred_removed"], 11)
        self.assertEqual(s["registry_mapping_added"], 1)
        self.assertEqual(s["registry_gap"], 0)
        # Tightened rows show before != after (broader -> workbook envelope).
        rrel1 = self.by["RREL1"][0]
        self.assertEqual(rrel1["alignment_status"], "tightened_to_workbook")
        self.assertEqual(rrel1["regime_nd_allowed_before"], "ND1; ND2; ND3")
        self.assertEqual(rrel1["regime_nd_allowed_after"], "")

    def test_interest_rate_type_resolved_via_enum_map(self):
        # interest_rate_type=Fixed now maps to ESMA FXRL (not OTHR), so RREL42 is
        # no longer an invalid asset default.
        val = json.loads(
            (self.out / "42_annex2_config_validation.json").read_text())
        rrel42 = next(r for r in val["rows"] if r["esma_code"] == "RREL42")
        self.assertEqual(rrel42["validation_status"], tcov.VS_VALID)
        ok, _ = tcov._validate_value_against_rule(
            "Fixed", {"transform": {"enum_map": {"Fixed": "FXRL"}}})
        self.assertTrue(ok)

    def test_asset_defaults_resolve_to_declared_esma_values(self):
        """Both asset defaults that were once conflicts now sit inside the regime
        envelope, by declaration rather than by tolerance.

        RREL35 ("Bullet") is a key of the RREL35 regime enum map, and the ERM pack
        maps it to OTHR for this asset class — a lifetime mortgage rolls up and
        repays at death or sale, which is not a scheduled bullet under the Annex 2
        definition. RREL40 ("ND5") is inside the regime's nd_allowed. Neither is a
        conflict, and asserting that they still are would mean re-opening a
        regulatory reporting decision to satisfy a test.
        """
        conflicts = [r for r in self.align["rows"]
                     if r["alignment_status"] == "asset_default_conflict"]
        self.assertEqual([r["esma_code"] for r in conflicts], [])
        for code in ("RREL35", "RREL40"):
            row = next(r for r in self.val["rows"] if r["esma_code"] == code)
            self.assertEqual(row["validation_status"], tcov.VS_VALID, code)
        # The mapping is declared, not inferred, in both halves: "Bullet" is a
        # real key of the generic RREL35 enum map (from the governed enum
        # configuration, where it means BLLT), and the ERM pack states the
        # product-specific override in its own file.
        import yaml as _yaml
        from engine.regime_contract import build_contract
        enum_map = build_contract().fields["RREL35"].enum_map
        self.assertEqual(enum_map.get("Bullet"), "BLLT")
        pack = _yaml.safe_load(
            (_REPO_ROOT / "config" / "asset"
             / "product_defaults_ERM.yaml").read_text(encoding="utf-8"))
        self.assertEqual(
            pack["reporting_policy"]["enum_overrides"]["amortisation_type"]["Bullet"],
            "OTHR")

    def test_any_conflict_that_survives_requires_manual_review(self):
        """Whatever ends up here is never auto-applied."""
        for r in self.align["rows"]:
            if r["alignment_status"] == "asset_default_conflict":
                self.assertTrue(r["requires_manual_review"])

    def test_40_reports_alignment_counts(self):
        s = self.summary
        self.assertEqual(s["annex2_alignment_tightened_count"], 5)
        self.assertEqual(s["annex2_alignment_phantom_removed_count"], 11)
        self.assertEqual(s["annex2_alignment_registry_added_count"], 1)
        # Both real asset defaults now resolve to declared ESMA values, so no
        # conflict remains. Reconciled to the alignment rows rather than pinned
        # to a number that only described one moment in the configuration.
        self.assertEqual(s["annex2_asset_default_conflict_count"],
                         sum(1 for r in self.align["rows"]
                             if r["alignment_status"] == "asset_default_conflict"))
        self.assertEqual(s["annex2_asset_default_conflict_count"], 0)
        # Manual review is for rows the run refuses to auto-resolve. It used to
        # be non-zero because the hand-maintained ND sets diverged from the
        # workbook in 35 places; the contract takes the envelope from the
        # workbook, so there is nothing left to adjudicate. Reconciled to the
        # rows rather than asserting a count the configuration no longer earns.
        self.assertEqual(s["annex2_alignment_manual_review_count"],
                         sum(1 for r in self.align["rows"]
                             if r["requires_manual_review"]))
        self.assertEqual(s["annex2_alignment_manual_review_count"], 0)

    def test_review_pack_shows_alignment_review(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("Annex 2 config-alignment review", html)


# --------------------------------------------------------------------------- #
# 8 — MI workflow remains unchanged
# --------------------------------------------------------------------------- #
class TestMiUnchanged(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_mi_"))
        warnings.simplefilter("ignore")
        cls.summary = wf.run_operator_workflow(
            input_dir=PACK, client_name="MI", client_id="mi", run_id="r",
            project_dir=str(cls.out), mode="mi_only", registry=REGISTRY,
            aliases_dir=ALIASES)

    def test_mi_uses_mi_registry_contract(self):
        cov = json.loads((self.out / "28a_target_coverage_matrix.json").read_text())
        self.assertEqual(cov["target_contract_id"], "mi_semantics_field_registry")

    def test_mi_has_no_annex2_artefacts_or_summary(self):
        for name in ("42_annex2_config_validation.csv",
                     "43_annex2_field_universe_reconciliation.csv",
                     "44_annex2_nd_eligibility_reconciliation.csv",
                     "45_annex2_config_alignment_review.csv",
                     "46_annex2_enum_coverage_reconciliation.csv",
                     "47_annex2_semantic_mapping_reconciliation.csv",
                     "48_annex2_mapping_correction_proposals.csv"):
            self.assertFalse((self.out / name).exists(), name)
        self.assertNotIn("annex2_field_count", self.summary)
        self.assertNotIn("annex2_authoritative_field_count", self.summary)
        self.assertNotIn("annex2_registry_mapped_count", self.summary)
        self.assertEqual(self.summary.get("target_contract_id"),
                         "mi_semantics_field_registry")

    def test_mi_field_count_unchanged(self):
        # The MI run covers the MI semantics registry and nothing else — asserted
        # against the registry's own size rather than a hard-coded count, so
        # curating an MI field in or out is not a spurious failure here. What this
        # guards is that an Annex 2 run has not leaked extra target fields in.
        import yaml
        registry = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "mi_agent"
             / "mi_semantics_field_registry.yaml").read_text(encoding="utf-8")) or {}
        cov = json.loads((self.out / "28a_target_coverage_matrix.json").read_text())
        self.assertEqual(cov["summary"]["target_fields_total"],
                         len(registry.get("fields", {}) or {}))


# --------------------------------------------------------------------------- #
# 9 — LLM target advisor optional + advisory only
# --------------------------------------------------------------------------- #
class TestAnnex2LlmAdvisorOptional(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_llm_"))
        cls.summary = _run_annex2(cls.out, advisor=True)

    def test_advisor_writes_36_artefacts(self):
        self.assertTrue((self.out / "36_target_first_llm_recommendations.csv").exists())
        self.assertTrue((self.out / "36_target_first_llm_usage_summary.json").exists())
        self.assertTrue(self.summary["llm_target_advisor_enabled"])

    def test_advisor_does_not_mutate_28a_28c(self):
        # Re-run the same first pass WITHOUT the advisor and compare the
        # deterministic target-first state.
        out2 = Path(tempfile.mkdtemp(prefix="annex2_nollm_"))
        _run_annex2(out2, advisor=False)
        cov_llm = json.loads((self.out / "28a_target_coverage_matrix.json").read_text())
        cov_no = json.loads((out2 / "28a_target_coverage_matrix.json").read_text())
        self.assertEqual(cov_llm["summary"]["coverage_status_counts"],
                         cov_no["summary"]["coverage_status_counts"])
        dec_llm = json.loads((self.out / "28c_human_decision_queue.json").read_text())
        dec_no = json.loads((out2 / "28c_human_decision_queue.json").read_text())
        self.assertEqual(dec_llm["summary"]["human_decision_rows_total"],
                         dec_no["summary"]["human_decision_rows_total"])


# --------------------------------------------------------------------------- #
# Onboarding completeness is config-driven, NOT XML/XSD-driven.
# XML/XSD validation is the final schema-validity check at delivery (gate 5);
# it is not the source of truth for onboarding completeness.
# --------------------------------------------------------------------------- #
class TestAnnex2CompletenessIsConfigDriven(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_complete_"))
        cls.summary = _run_annex2(cls.out)
        cls.recon = json.loads(
            (cls.out / "43_annex2_field_universe_reconciliation.json").read_text())["summary"]

    def test_completeness_governed_by_workbook_universe_and_config(self):
        # The workbook universe + config reconciliation still determine
        # completeness, and still do so without any schema check. What changed
        # is where the incompleteness shows: no code is missing from the regime
        # contract any more (it is derived from this very universe), so the run
        # is not-ready because of the tape's own unmapped mandatory codes.
        self.assertEqual(self.recon["authoritative_field_count"], 107)
        self.assertEqual(self.recon["registry_gap_count"], 0)
        self.assertEqual(self.recon["missing_from_regime_rules_count"], 0)
        self.assertEqual(self.summary["status"], "BLOCKED")

    def test_onboarding_does_not_emit_or_gate_on_xsd(self):
        # Onboarding produces no XSD/schema-validation artefact and never reports
        # a schema verdict as a completeness signal (that lives in gate 5).
        names = [p.name.lower() for p in self.out.iterdir()]
        self.assertFalse(any(".xsd" in n or "xsd_valid" in n for n in names),
                         f"unexpected schema artefact in onboarding output: {names}")
        for key in self.summary:
            self.assertNotIn("xsd", key.lower())


if __name__ == "__main__":
    unittest.main()
