#!/usr/bin/env python3
"""tests/test_delivery_xml_preview_readiness.py

Two SEPARATE non-production XML artefact modes for the Delivery/XML Agent:

  * client_safe_preview                  (Client Preview XML)
  * synthetic_full_coverage_schema_test  (Synthetic Full-Coverage Schema Test XML)

Covers (per the task acceptance criteria):
  * both modes disabled by default;
  * disabled modes emit no XML;
  * client preview emits ONLY when explicitly enabled AND its verdict allows;
  * synthetic schema test emits ONLY when explicitly enabled;
  * production XML gates remain false and unchanged;
  * preview output is separate from production output (under preview/);
  * client preview uses placeholders ONLY for approved fields;
  * client preview excludes preview_exclusion (operator-ambiguous) fields;
  * client preview does NOT fabricate valuation/rate/economic fields;
  * synthetic schema test populates every Annex 2 field with dummy values;
  * synthetic values are labelled source = synthetic_schema_test;
  * RREL35 is real (not placeholder/exclusion/synthetic) because it is resolved;
  * RREL82 is placeholder-only in the client preview;
  * XML files contain watermarks;
  * no preview artefact sets xml_generation_allowed / ready_for_xml_delivery true.
"""

from __future__ import annotations

import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.delivery_xml_agent import delivery_xml_agent as da
from engine.delivery_xml_agent import preview_readiness as pr

REGIME = str(_REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml")
REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
UNIVERSE = str(_REPO_ROOT / "config" / "regime" / "annex2_field_universe.yaml")
POLICY = str(_REPO_ROOT / "config" / "delivery" / "xml_preview_policy.yaml")

_LOANS = ["LN0001", "LN0002"]

_FRAME_COLUMNS = [
    "row_id", "loan_identifier", "record_group", "esma_code", "canonical_field",
    "projected_value", "projected_value_type", "value_source", "projection_status",
    "nd_applied", "default_applied", "source_field", "source_value_sample",
    "rule_id", "blocking_for_delivery",
]
_PROJ_ISSUE_COLUMNS = [
    "issue_id", "source_issue_id", "esma_code", "canonical_field", "record_group",
    "issue_type", "projection_status", "severity", "blocking_for_delivery",
    "blocking_for_xml_delivery", "recommended_action", "downstream_owner", "description",
]

# A "clean-enough" frame: only deliverable + placeholderable identifiers +
# operator-ambiguous (excluded) fields. No config/source/format blockers, so the
# client preview verdict is ALLOWED.
_CLEAN_FRAME_SPEC = [
    ("RREC9", "property_type", "RREC", "projected_from_transformed", ["RHOS", "RFLT"]),
    ("RREL35", "amortisation_type", "RREL", "projected_from_transformed", ["OTHR", "OTHR"]),
    ("RREL2", "original_underlying_exposure_identifier", "RREL",
     "blocked_client_onboarding_dependency", ["", ""]),
    ("RREL82", "originator_name", "RREL",
     "blocked_client_onboarding_dependency", ["", ""]),
    ("RREC17", "original_valuation_amount", "RREC",
     "blocked_operator_or_config_dependency", ["", ""]),
]
_CLEAN_ISSUES = [
    dict(issue_id="PRJ-0001", esma_code="RREL2", record_group="RREL",
         issue_type="client_onboarding_dependency_unresolved",
         projection_status="blocked_client_onboarding_dependency"),
    dict(issue_id="PRJ-0002", esma_code="RREL82", record_group="RREL",
         issue_type="client_onboarding_dependency_unresolved",
         projection_status="blocked_client_onboarding_dependency"),
    dict(issue_id="PRJ-0003", esma_code="RREC17", record_group="RREC",
         issue_type="operator_dependency_unresolved",
         projection_status="blocked_operator_or_config_dependency"),
]

# A frame that still has a real config/source blocker (RREC7 source mapping) so
# the client preview verdict is NOT allowed.
_BLOCKED_FRAME_SPEC = _CLEAN_FRAME_SPEC + [
    ("RREC7", "occupancy", "RREC", "unresolved_source_mapping", ["", ""]),
]
_BLOCKED_ISSUES = _CLEAN_ISSUES + [
    dict(issue_id="PRJ-0004", esma_code="RREC7", record_group="RREC",
         issue_type="source_mapping_unresolved",
         projection_status="unresolved_source_mapping"),
]


def _write_projection_package(root: Path, frame_spec, issues) -> Path:
    proj = root / "output" / "projection"
    proj.mkdir(parents=True, exist_ok=True)

    rows = []
    rid = 0
    for code, canonical, group, status, values in frame_spec:
        for i, loan in enumerate(_LOANS):
            rid += 1
            rows.append({
                "row_id": f"R{rid:04d}", "loan_identifier": loan,
                "record_group": group, "esma_code": code, "canonical_field": canonical,
                "projected_value": values[i] if i < len(values) else "",
                "projected_value_type": "raw", "value_source": "transformed_tape",
                "projection_status": status, "nd_applied": "False",
                "default_applied": "False", "source_field": canonical,
                "source_value_sample": "", "rule_id": code,
                "blocking_for_delivery": "True" if status not in (
                    "projected_from_transformed", "projected_nd_default") else "False",
            })
    with open(proj / "51_projected_annex2_target_frame.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_FRAME_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    with open(proj / "55_projection_issues.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_PROJ_ISSUE_COLUMNS)
        w.writeheader()
        for it in issues:
            row = {k: "" for k in _PROJ_ISSUE_COLUMNS}
            row.update(it)
            row["severity"] = "warn"
            row["blocking_for_delivery"] = "True"
            row["blocking_for_xml_delivery"] = "True"
            w.writerow(row)

    (proj / "54_projection_lineage.json").write_text(json.dumps({
        "onboarding_lineage": [], "transformation_lineage": [],
        "validation_lineage": [], "projection_lineage": [],
    }), encoding="utf-8")

    manifest = {
        "agent": "projection_agent", "agent_version": "1.0",
        "client_id": "client_001", "run_id": "run_pre_xml_final_check_3",
        "target_contract_id": "ESMA_Annex2",
        "consumes_validation_package": True, "performed_xml_delivery": False,
        "invoked_gate5_xml_builder": False,
        "regime_config_path": REGIME, "registry_path": REGISTRY,
        "projection_ran": True, "projection_complete": False,
        "ready_for_delivery_normalisation": False, "ready_for_xml_delivery": False,
    }
    mpath = proj / "50_projection_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return mpath


def _build_delivery(root: Path, frame_spec, issues) -> Path:
    mpath = _write_projection_package(root, frame_spec, issues)
    da.build_delivery_package(mpath, field_universe_path=UNIVERSE)
    return root / "output" / "delivery_xml"


def _enabled_policy(tmp: Path, *, client: bool = False, synthetic: bool = False) -> str:
    """Copy the real policy and flip the requested modes to enabled."""
    data = copy.deepcopy(yaml.safe_load(Path(POLICY).read_text(encoding="utf-8")))
    data["preview_modes"]["client_safe_preview"]["enabled"] = client
    data["preview_modes"]["synthetic_full_coverage_schema_test"]["enabled"] = synthetic
    p = tmp / "policy_enabled.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
class TestDefaultsDisabled(unittest.TestCase):
    """Both modes disabled by default; no XML emitted; production gates false."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="prev_default_"))
        cls.out = _build_delivery(cls.root, _BLOCKED_FRAME_SPEC, _BLOCKED_ISSUES)
        cls.result = pr.evaluate_and_emit(cls.out, field_universe_path=UNIVERSE)

    def test_policy_modes_disabled_by_default(self):
        policy = pr.load_policy()
        modes = policy["preview_modes"]
        self.assertFalse(modes["client_safe_preview"]["enabled"])
        self.assertFalse(modes["synthetic_full_coverage_schema_test"]["enabled"])

    def test_readiness_artefacts_written(self):
        preview = self.out / "preview"
        for name in ("70_xml_preview_readiness.json", "71_xml_preview_readiness.md",
                     "72_xml_preview_policy_application.csv",
                     "73_xml_preview_assumptions.csv", "74_xml_preview_blockers.csv",
                     "75_synthetic_schema_test_readiness.json",
                     "76_synthetic_schema_test_readiness.md",
                     "77_synthetic_schema_field_plan.csv"):
            self.assertTrue((preview / name).exists(), name)

    def test_no_xml_emitted_when_disabled(self):
        self.assertFalse(self.result["flags"]["xml_preview_generated"])
        self.assertFalse(self.result["flags"]["synthetic_schema_test_generated"])
        self.assertFalse((self.out / "preview" / "client_preview").exists())
        self.assertFalse((self.out / "preview" / "synthetic_schema_test").exists())
        # no XML anywhere under preview/.
        self.assertEqual(list((self.out / "preview").rglob("*.xml")), [])

    def test_production_gates_unchanged_and_false(self):
        manifest = json.loads((self.out / "60_delivery_manifest.json").read_text())
        self.assertFalse(manifest["xml_generation_allowed"])
        self.assertFalse(manifest["ready_for_xml_delivery"])
        self.assertFalse(manifest["xml_generated"])
        pf = self.result["flags"]["production_flags_unchanged"]
        self.assertFalse(pf["xml_generation_allowed"])
        self.assertFalse(pf["ready_for_xml_delivery"])

    def test_client_preview_not_allowed_with_config_source_blockers(self):
        v = self.result["client_preview_verdict"]
        self.assertFalse(v["allowed"])
        self.assertIn("RREC7", v["must_resolve_codes"])

    def test_no_preview_artefact_sets_production_true(self):
        # scan every preview artefact text for an accidental production-true set.
        for p in (self.out / "preview").rglob("*"):
            if p.is_file():
                text = p.read_text(encoding="utf-8", errors="ignore")
                self.assertNotIn('"xml_generation_allowed": true', text)
                self.assertNotIn('"ready_for_xml_delivery": true', text)


# --------------------------------------------------------------------------- #
class TestClientPreviewEnabledAndAllowed(unittest.TestCase):
    """Clean frame + client mode enabled -> client preview emits."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="prev_client_"))
        cls.out = _build_delivery(cls.root, _CLEAN_FRAME_SPEC, _CLEAN_ISSUES)
        cls.policy = _enabled_policy(cls.root, client=True)
        cls.result = pr.evaluate_and_emit(
            cls.out, policy_path=cls.policy, field_universe_path=UNIVERSE)
        cls.verdict = cls.result["client_preview_verdict"]
        cls.cdir = cls.out / "preview" / "client_preview"
        cls.apps = {a["esma_code"]: a for a in cls.verdict["applications"]}

    def test_verdict_allowed_and_generated(self):
        self.assertTrue(self.verdict["allowed"])
        self.assertTrue(self.result["flags"]["xml_preview_generated"])
        self.assertTrue(self.result["flags"]["ready_for_xml_preview"])

    def test_all_client_artefacts_written(self):
        for name in ("80_client_preview_frame.csv", "81_client_preview_lineage.json",
                     "82_client_preview_assumptions.csv",
                     "83_client_preview_exclusions.csv",
                     "84_client_preview_watermark.txt", "85_client_preview.xml",
                     "86_client_preview_summary.md"):
            self.assertTrue((self.cdir / name).exists(), name)

    def test_placeholders_only_for_approved_fields(self):
        # RREL2 and RREL82 are approved identifier placeholders.
        self.assertEqual(self.apps["RREL2"]["disposition"], pr.DISP_PLACEHOLDER)
        self.assertEqual(self.apps["RREL82"]["disposition"], pr.DISP_PLACEHOLDER)
        self.assertEqual(self.apps["RREL2"]["preview_value"], "PREVIEW_ONLY_RREL2")
        # a deliverable, non-identifier field is NOT placeholdered.
        self.assertEqual(self.apps["RREC9"]["disposition"], pr.DISP_REAL)
        self.assertNotIn("PREVIEW_ONLY", self.apps["RREC9"]["preview_value"])

    def test_excludes_operator_ambiguous_field(self):
        self.assertEqual(self.apps["RREC17"]["disposition"], pr.DISP_EXCLUDED)
        self.assertIn("RREC17", self.verdict["excluded_codes"])

    def test_does_not_fabricate_economic_field(self):
        # RREC17 (Original Valuation Amount, {MONETARY}) must never carry a value.
        self.assertEqual(self.apps["RREC17"]["preview_value"], "")
        self.assertNotEqual(self.apps["RREC17"]["value_source"],
                            pr.SRC_PLACEHOLDER)
        # and it must not appear in the emitted XML at all.
        xml_text = (self.cdir / "85_client_preview.xml").read_text()
        self.assertNotIn('code="RREC17"', xml_text)

    def test_rrel35_real_not_placeholder_or_excluded(self):
        a = self.apps["RREL35"]
        self.assertEqual(a["disposition"], pr.DISP_REAL)
        self.assertEqual(a["preview_value"], "OTHR")
        self.assertNotIn("RREL35", self.verdict["placeholder_codes"])
        self.assertNotIn("RREL35", self.verdict["excluded_codes"])

    def test_rrel82_placeholder_only_in_preview(self):
        self.assertIn("RREL82", self.verdict["placeholder_codes"])
        self.assertEqual(self.apps["RREL82"]["preview_value"], "PREVIEW_ONLY_RREL82")
        # RREL82 has no ND allowed in the universe -> still a production blocker.
        uni = yaml.safe_load(Path(UNIVERSE).read_text())["fields"]["RREL82"]
        self.assertFalse(uni["nd1_4_allowed"])
        self.assertFalse(uni["nd5_allowed"])

    def test_xml_has_watermark_and_namespace(self):
        xml_text = (self.cdir / "85_client_preview.xml").read_text()
        self.assertIn("NON-PRODUCTION CLIENT PREVIEW", xml_text)
        self.assertIn("urn:trakt:nonproduction:preview", xml_text)
        self.assertIn("PREVIEW_ONLY_RREL2", xml_text)
        # placeholder identifiers + real values present; not a submission.
        self.assertIn("NOT A REGULATORY SUBMISSION", xml_text)

    def test_summary_language(self):
        s = (self.cdir / "86_client_preview_summary.md").read_text()
        self.assertIn("not a reportable regulatory XML", s)
        self.assertIn("Production XML remains blocked", s)

    def test_separate_from_production_output(self):
        self.assertEqual(self.cdir.parent.name, "preview")
        self.assertEqual(self.cdir.parent.parent.name, "delivery_xml")
        # production manifest still false.
        manifest = json.loads((self.out / "60_delivery_manifest.json").read_text())
        self.assertFalse(manifest["xml_generation_allowed"])


# --------------------------------------------------------------------------- #
class TestClientPreviewEnabledButNotAllowed(unittest.TestCase):
    """Even when enabled, the client preview does NOT emit while real
    config/source blockers remain."""

    def test_enabled_but_blocked_emits_no_xml(self):
        root = Path(tempfile.mkdtemp(prefix="prev_client_blk_"))
        out = _build_delivery(root, _BLOCKED_FRAME_SPEC, _BLOCKED_ISSUES)
        policy = _enabled_policy(root, client=True)
        result = pr.evaluate_and_emit(out, policy_path=policy, field_universe_path=UNIVERSE)
        self.assertFalse(result["client_preview_verdict"]["allowed"])
        self.assertFalse(result["flags"]["xml_preview_generated"])
        self.assertFalse((out / "preview" / "client_preview").exists())


# --------------------------------------------------------------------------- #
class TestSyntheticSchemaTest(unittest.TestCase):
    """Synthetic full-coverage schema test populates every Annex 2 field."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="prev_synth_"))
        cls.out = _build_delivery(cls.root, _CLEAN_FRAME_SPEC, _CLEAN_ISSUES)
        cls.policy = _enabled_policy(cls.root, synthetic=True)
        cls.result = pr.evaluate_and_emit(
            cls.out, policy_path=cls.policy, field_universe_path=UNIVERSE)
        cls.verdict = cls.result["synthetic_full_coverage_verdict"]
        cls.sdir = cls.out / "preview" / "synthetic_schema_test"

    def test_allowed_and_generated(self):
        self.assertTrue(self.verdict["allowed"])
        self.assertTrue(self.result["flags"]["synthetic_schema_test_generated"])

    def test_all_synthetic_artefacts_written(self):
        for name in ("90_synthetic_schema_frame.csv",
                     "91_synthetic_schema_lineage.json",
                     "92_synthetic_values_catalog.csv",
                     "93_synthetic_schema_watermark.txt",
                     "94_synthetic_schema_test.xml",
                     "95_synthetic_schema_summary.md"):
            self.assertTrue((self.sdir / name).exists(), name)

    def test_full_field_coverage(self):
        universe = yaml.safe_load(Path(UNIVERSE).read_text())["fields"]
        self.assertEqual(self.verdict["planned_field_count"], len(universe))
        plan_codes = {p["esma_code"] for p in self.verdict["plan"]}
        self.assertEqual(plan_codes, set(universe.keys()))
        # every planned field has a non-empty value (dummy or real).
        self.assertTrue(all(p["value"] for p in self.verdict["plan"]))

    def test_synthetic_values_labelled(self):
        rows = list(csv.DictReader(open(self.sdir / "92_synthetic_values_catalog.csv")))
        synth = [r for r in rows if r["source"] == "synthetic_schema_test"]
        self.assertGreater(len(synth), 0)
        # the XML labels synthetic source on synthetic fields.
        xml_text = (self.sdir / "94_synthetic_schema_test.xml").read_text()
        self.assertIn('source="synthetic_schema_test"', xml_text)

    def test_rrel35_real_not_synthetic(self):
        plan = {p["esma_code"]: p for p in self.verdict["plan"]}
        # RREL35 is deliverable (OTHR) -> reused as a real value, not synthetic.
        self.assertEqual(plan["RREL35"]["value_source"], pr.SRC_REAL)
        self.assertEqual(plan["RREL35"]["value"], "OTHR")

    def test_xml_watermark_engineering_only(self):
        xml_text = (self.sdir / "94_synthetic_schema_test.xml").read_text()
        self.assertIn("SYNTHETIC FULL-COVERAGE SCHEMA TEST", xml_text)
        self.assertIn("ENGINEERING ONLY", xml_text)
        self.assertIn("<EngineeringOnly>true</EngineeringOnly>", xml_text)
        self.assertIn("<Reportable>false</Reportable>", xml_text)

    def test_summary_language(self):
        s = (self.sdir / "95_synthetic_schema_summary.md").read_text()
        self.assertIn("synthetic engineering artefact", s)
        self.assertIn("not reportable", s.lower())

    def test_production_gates_unchanged(self):
        manifest = json.loads((self.out / "60_delivery_manifest.json").read_text())
        self.assertFalse(manifest["xml_generation_allowed"])
        self.assertFalse(manifest["ready_for_xml_delivery"])


# --------------------------------------------------------------------------- #
class TestInspectPreviewHelper(unittest.TestCase):
    def test_inspect_preview_reports_state(self):
        root = Path(tempfile.mkdtemp(prefix="prev_inspect_"))
        out = _build_delivery(root, _CLEAN_FRAME_SPEC, _CLEAN_ISSUES)
        policy = _enabled_policy(root, client=True, synthetic=True)
        pr.evaluate_and_emit(out, policy_path=policy, field_universe_path=UNIVERSE)
        from scripts.inspect_delivery_xml_readiness import inspect_preview
        p = inspect_preview(out)
        self.assertTrue(p["preview_exists"])
        self.assertFalse(p["production_xml"]["xml_generation_allowed"])
        self.assertTrue(p["client_preview"]["xml_exists"])
        self.assertTrue(p["synthetic_schema_test"]["xml_exists"])
        self.assertGreater(p["synthetic_schema_test"]["synthetic_value_count"], 0)
        self.assertGreaterEqual(p["client_preview"]["placeholder_count"], 2)


if __name__ == "__main__":
    unittest.main()
