#!/usr/bin/env python3
"""tests/test_delivery_xml_agent_workflow.py

Trakt Delivery/XML Agent v1 — consume the Projection package and produce a
delivery package (artefacts 60..64). NO production XML.

Covers:
  * rejects a missing / non-projection manifest;
  * refuses XML when projection_complete=false;
  * refuses XML when blocked target-frame statuses exist;
  * blocked rows are carried into 63_delivery_issues (by category);
  * projected_from_transformed values flow into 62_delivery_normalised_frame;
  * projected_nd_default values flow into 62_delivery_normalised_frame;
  * RREL/RREC record_group is preserved;
  * no XML file is written by default;
  * --allow-xml-preview does not bypass the readiness gates;
  * the delivery manifest links all artefacts;
  * delivery does not silently fill blocked values.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.delivery_xml_agent import delivery_xml_agent as da
from engine.delivery_xml_agent.delivery_xml_agent import ProjectionHandoffError

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
REGIME = str(_REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml")
UNIVERSE = str(_REPO_ROOT / "config" / "regime" / "annex2_field_universe.yaml")

# --------------------------------------------------------------------------- #
# Synthetic projection package builder
# --------------------------------------------------------------------------- #

_FRAME_COLUMNS = [
    "row_id", "loan_identifier", "record_group", "esma_code", "canonical_field",
    "projected_value", "projected_value_type", "value_source", "projection_status",
    "nd_applied", "default_applied", "source_field", "source_value_sample",
    "rule_id", "blocking_for_delivery",
]

# (esma_code, canonical, record_group, projection_status, projected_value)
_FRAME_SPEC = [
    # deliverable — materialised enum values.
    ("RREC9", "property_type", "RREC", "projected_from_transformed", ["RHOS", "RFLT"]),
    # deliverable — allowed ND default.
    ("RREL16", "primary_income", "RREL", "projected_nd_default", ["ND1", "ND1"]),
    # blocked — client onboarding identifier policy.
    ("RREL2", "original_underlying_exposure_identifier", "RREL",
     "blocked_client_onboarding_dependency", ["", ""]),
    # blocked — operator dependency (ambiguous valuation source).
    ("RREC17", "original_valuation_amount", "RREC",
     "blocked_operator_or_config_dependency", ["", ""]),
    # blocked — config mapping dependency (purpose).
    ("RREL27", "purpose", "RREL", "blocked_operator_or_config_dependency", ["", ""]),
    # blocked — not materialised, no nd/default.
    ("RREL40", "debt_to_income_ratio", "RREL", "unresolved_not_materialised", ["", ""]),
    # blocked — unresolved source mapping.
    ("RREC7", "occupancy", "RREC", "unresolved_source_mapping", ["", ""]),
]

# 55 projection issues (subset of projection _ISSUE_COLUMNS).
_PROJ_ISSUE_COLUMNS = [
    "issue_id", "source_issue_id", "esma_code", "canonical_field", "record_group",
    "issue_type", "projection_status", "severity", "blocking_for_delivery",
    "blocking_for_xml_delivery", "recommended_action", "downstream_owner", "description",
]
_PROJ_ISSUES = [
    dict(issue_id="PRJ-0001", esma_code="RREL2", canonical_field="original_underlying_exposure_identifier",
         record_group="RREL", issue_type="client_onboarding_dependency_unresolved",
         projection_status="blocked_client_onboarding_dependency",
         downstream_owner="client_onboarding"),
    dict(issue_id="PRJ-0002", esma_code="RREC17", canonical_field="original_valuation_amount",
         record_group="RREC", issue_type="operator_dependency_unresolved",
         projection_status="blocked_operator_or_config_dependency",
         downstream_owner="operator"),
    dict(issue_id="PRJ-0003", esma_code="RREL27", canonical_field="purpose",
         record_group="RREL", issue_type="config_dependency_unresolved",
         projection_status="blocked_operator_or_config_dependency",
         downstream_owner="config_policy"),
    dict(issue_id="PRJ-0004", esma_code="RREL40", canonical_field="debt_to_income_ratio",
         record_group="RREL", issue_type="nd_default_rule_missing",
         projection_status="unresolved_not_materialised",
         downstream_owner="config_policy"),
    dict(issue_id="PRJ-0005", esma_code="RREC7", canonical_field="occupancy",
         record_group="RREC", issue_type="source_mapping_unresolved",
         projection_status="unresolved_source_mapping",
         downstream_owner="projection"),
]

_LOANS = ["LN0001", "LN0002"]


def _write_projection_package(root: Path, *, agent: str = "projection_agent",
                              projection_complete: bool = False,
                              ready_norm: bool = False,
                              performed_xml: bool = False,
                              write_frame: bool = True) -> Path:
    output = root / "output"
    proj = output / "projection"
    proj.mkdir(parents=True, exist_ok=True)

    # 51 target frame (long).
    if write_frame:
        rows = []
        rid = 0
        for code, canonical, group, status, values in _FRAME_SPEC:
            for i, loan in enumerate(_LOANS):
                rid += 1
                rows.append({
                    "row_id": f"R{rid:04d}", "loan_identifier": loan,
                    "record_group": group, "esma_code": code,
                    "canonical_field": canonical,
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

    # 52 contract (minimal).
    with open(proj / "52_projection_field_contract.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["esma_code", "canonical_field", "record_group",
                                           "mandatory", "field_projection_status"])
        w.writeheader()
        for code, canonical, group, status, _ in _FRAME_SPEC:
            w.writerow({"esma_code": code, "canonical_field": canonical,
                        "record_group": group, "mandatory": "True",
                        "field_projection_status": status})

    # 55 issues.
    with open(proj / "55_projection_issues.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_PROJ_ISSUE_COLUMNS)
        w.writeheader()
        for it in _PROJ_ISSUES:
            row = {k: "" for k in _PROJ_ISSUE_COLUMNS}
            row.update(it)
            row["source_issue_id"] = ""
            row["severity"] = "warn"
            row["blocking_for_delivery"] = "True"
            row["blocking_for_xml_delivery"] = "True"
            row["recommended_action"] = "resolve upstream"
            w.writerow(row)

    # 56 blocker resolution (minimal).
    with open(proj / "56_projection_blocker_resolution.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["validation_issue_id", "esma_code",
                                           "projection_status", "resolved"])
        w.writeheader()
        for it in _PROJ_ISSUES:
            w.writerow({"validation_issue_id": it["issue_id"], "esma_code": it["esma_code"],
                        "projection_status": it["projection_status"], "resolved": "False"})

    # 54 lineage.
    (proj / "54_projection_lineage.json").write_text(json.dumps({
        "onboarding_lineage": [{"target_field": "RREC9"}],
        "transformation_lineage": [{"target_field": "RREC9"}],
        "validation_lineage": [{"field": "property_type"}],
        "projection_lineage": [{"esma_code": "RREC9", "canonical_field": "property_type"}],
    }), encoding="utf-8")

    # 53 readiness.
    (proj / "53_projection_readiness.json").write_text(json.dumps({
        "projection_ran": True, "projection_complete": projection_complete,
        "ready_for_delivery_normalisation": ready_norm,
        "ready_for_xml_delivery": False}), encoding="utf-8")

    manifest = {
        "agent": agent, "agent_version": "1.0",
        "client_id": "client_001", "run_id": "run_pre_xml_final_check_3",
        "target_contract_id": "ESMA_Annex2",
        "consumes_validation_package": True,
        "performed_xml_delivery": performed_xml,
        "invoked_gate5_xml_builder": False,
        "regime_config_path": REGIME, "registry_path": REGISTRY,
        "projection_ran": True,
        "projection_complete": projection_complete,
        "ready_for_delivery_normalisation": ready_norm,
        "ready_for_xml_delivery": False,
    }
    mpath = proj / "50_projection_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return mpath


def _write_restricted_code_order(root: Path) -> Path:
    """A code-order that omits the mandatory RREL2 to exercise the
    template_order_incomplete gate deterministically."""
    p = root / "esma_code_order_restricted.yaml"
    p.write_text("Record:\n  - RREC9\n  - RREL16\n", encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Manifest validation
# --------------------------------------------------------------------------- #
class TestManifestValidation(unittest.TestCase):
    def test_missing_manifest_raises(self):
        with self.assertRaises(ProjectionHandoffError):
            da.build_delivery_package("/no/such/50_projection_manifest.json")

    def test_non_projection_agent_raises(self):
        root = Path(tempfile.mkdtemp(prefix="da_badagent_"))
        with self.assertRaises(ProjectionHandoffError):
            da.build_delivery_package(
                _write_projection_package(root, agent="validation_agent"))

    def test_performed_xml_raises(self):
        root = Path(tempfile.mkdtemp(prefix="da_xmldone_"))
        with self.assertRaises(ProjectionHandoffError):
            da.build_delivery_package(
                _write_projection_package(root, performed_xml=True))

    def test_missing_frame_raises(self):
        root = Path(tempfile.mkdtemp(prefix="da_noframe_"))
        with self.assertRaises(ProjectionHandoffError):
            da.build_delivery_package(
                _write_projection_package(root, write_frame=False))


# --------------------------------------------------------------------------- #
# Full delivery run on an INCOMPLETE projection (the current-run scenario).
# --------------------------------------------------------------------------- #
class TestDeliveryRunRefusesXml(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="da_run_"))
        cls.mpath = _write_projection_package(cls.root)
        cls.order = _write_restricted_code_order(cls.root)
        cls.result = da.build_delivery_package(
            cls.mpath, esma_code_order_path=str(cls.order), field_universe_path=UNIVERSE)
        cls.out = cls.root / "output" / "delivery_xml"
        cls.manifest = json.loads((cls.out / "60_delivery_manifest.json").read_text())
        cls.readiness = json.loads((cls.out / "61_delivery_readiness.json").read_text())
        cls.frame = json.loads(
            (cls.out / "62_delivery_normalised_frame.json").read_text())["rows"]
        cls.issues = json.loads((cls.out / "63_delivery_issues.json").read_text())["rows"]

    def _frame_cells(self, code):
        return [r for r in self.frame if r["esma_code"] == code]

    def test_all_artefacts_written(self):
        for name in ("60_delivery_manifest.json", "60_delivery_manifest.yaml",
                     "61_delivery_readiness.json", "61_delivery_readiness.md",
                     "62_delivery_normalised_frame.csv", "62_delivery_normalised_frame.json",
                     "63_delivery_issues.csv", "63_delivery_issues.json",
                     "64_delivery_lineage.json"):
            self.assertTrue((self.out / name).exists(), name)

    def test_output_under_delivery_xml_dir(self):
        self.assertEqual(self.out.name, "delivery_xml")
        self.assertEqual(self.out.parent.name, "output")

    def test_refuses_xml_when_projection_incomplete(self):
        self.assertFalse(self.manifest["xml_generation_allowed"])
        self.assertFalse(self.manifest["xml_generated"])
        self.assertFalse(self.manifest["ready_for_xml_delivery"])
        self.assertTrue(self.manifest["delivery_xml_ran"])
        self.assertFalse(self.manifest["delivery_normalisation_complete"])
        self.assertEqual(self.manifest["next_agent"],
                         "operator_config_projection_remediation")
        # the projection_complete gate explicitly failed.
        gate = next(g for g in self.readiness["gates"] if g["gate"] == "projection_complete")
        self.assertFalse(gate["passed"])

    def test_refuses_xml_when_blocked_statuses_exist(self):
        gate = next(g for g in self.readiness["gates"]
                    if g["gate"] == "no_blocked_target_frame_rows")
        self.assertFalse(gate["passed"])
        self.assertGreater(self.manifest["blocked_row_count"], 0)

    def test_projected_from_transformed_in_frame(self):
        cells = self._frame_cells("RREC9")
        self.assertTrue(cells)
        self.assertTrue(all(c["delivery_status"] == da.DS_DELIVERABLE for c in cells))
        self.assertEqual({c["delivery_value"] for c in cells}, {"RHOS", "RFLT"})

    def test_projected_nd_default_in_frame(self):
        cells = self._frame_cells("RREL16")
        self.assertTrue(cells)
        self.assertTrue(all(c["delivery_status"] == da.DS_DELIVERABLE for c in cells))
        self.assertTrue(all(c["delivery_value"] == "ND1" for c in cells))
        self.assertTrue(all(c["is_nd_value"] for c in cells))

    def test_blocked_rows_not_filled(self):
        for code in ("RREL2", "RREC17", "RREL27", "RREL40", "RREC7"):
            cells = self._frame_cells(code)
            self.assertTrue(cells)
            self.assertTrue(all(c["delivery_status"] == da.DS_BLOCKED for c in cells), code)
            # delivery never promotes / fills a blocked value.
            self.assertTrue(all(c["delivery_value"] == "" for c in cells), code)

    def test_blocked_rows_carried_into_issues(self):
        issue_codes = {i["esma_code"] for i in self.issues}
        for code in ("RREL2", "RREC17", "RREL27", "RREL40", "RREC7"):
            self.assertIn(code, issue_codes, code)
        # each blocked frame row links to a delivery issue.
        for code in ("RREL2", "RREC17"):
            for c in self._frame_cells(code):
                self.assertTrue(c["delivery_issue_id"], code)

    def test_issue_categories_present(self):
        cats = {i["delivery_blocker_type"] for i in self.issues}
        for expected in (da.BT_CLIENT, da.BT_OPERATOR_OR_CONFIG, da.BT_CONFIG,
                         da.BT_SOURCE_MAPPING, da.BT_ND_DEFAULT_MISSING,
                         da.BT_STRUCTURE_DEFERRED, da.BT_TEMPLATE_ORDER):
            self.assertIn(expected, cats, expected)

    def test_record_group_preserved(self):
        groups = {r["record_group"] for r in self.frame}
        self.assertIn("RREL", groups)
        self.assertIn("RREC", groups)
        # xml_record_group mapped consistently.
        for r in self.frame:
            if r["record_group"] == "RREL":
                self.assertEqual(r["xml_record_group"], "underlying_exposure")
            elif r["record_group"] == "RREC":
                self.assertEqual(r["xml_record_group"], "collateral")

    def test_no_xml_written_by_default(self):
        self.assertFalse((self.out / "65_xml_preview.xml").exists())
        self.assertFalse((self.out / "66_xml_validation_report.json").exists())
        self.assertEqual(list((self.root / "output").rglob("*.xml")), [])

    def test_did_not_invoke_frozen_builders(self):
        self.assertFalse(self.manifest["invoked_gate5_xml_builder"])
        self.assertFalse(self.manifest["invoked_gate4b_normalizer"])
        self.assertFalse(self.manifest["silently_filled_blocked_values"])

    def test_manifest_links_all_artefacts(self):
        for key in ("output_readiness_json", "output_readiness_md",
                    "output_delivery_frame_csv", "output_delivery_frame_json",
                    "output_delivery_issues_csv", "output_delivery_issues_json",
                    "output_lineage_json"):
            p = Path(self.manifest[key])
            self.assertTrue(p.exists(), key)
            self.assertEqual(p.parent, self.out)
        self.assertTrue(self.manifest["consumes_projection_package"])
        self.assertEqual(Path(self.manifest["input_projection_manifest_path"]), self.mpath)

    def test_upstream_not_mutated(self):
        proj = self.root / "output" / "projection"
        names = sorted(p.name for p in proj.iterdir())
        self.assertNotIn("60_delivery_manifest.json", names)
        self.assertIn("51_projected_annex2_target_frame.csv", names)

    def test_template_order_incomplete_gate_failed(self):
        gate = next(g for g in self.readiness["gates"]
                    if g["gate"] == "template_code_order_complete")
        self.assertFalse(gate["passed"])
        self.assertGreater(self.manifest["missing_required_order_code_count"], 0)

    def test_lineage_extended(self):
        lin = json.loads((self.out / "64_delivery_lineage.json").read_text())
        self.assertTrue(lin["projection_lineage"])
        self.assertTrue(lin["delivery_lineage"])


# --------------------------------------------------------------------------- #
# --allow-xml-preview must NOT bypass the readiness gates.
# --------------------------------------------------------------------------- #
class TestAllowXmlPreviewDoesNotBypass(unittest.TestCase):
    def test_preview_flag_blocked_run_writes_no_xml(self):
        root = Path(tempfile.mkdtemp(prefix="da_preview_"))
        mpath = _write_projection_package(root)  # incomplete projection
        result = da.build_delivery_package(
            mpath, allow_xml_preview=True, field_universe_path=UNIVERSE)
        out = root / "output" / "delivery_xml"
        manifest = json.loads((out / "60_delivery_manifest.json").read_text())
        self.assertFalse(manifest["xml_generation_allowed"])
        self.assertTrue(manifest["allow_xml_preview_flag"])
        self.assertFalse(manifest["xml_preview_written"])
        self.assertFalse(manifest["xml_generated"])
        self.assertFalse((out / "65_xml_preview.xml").exists())
        self.assertFalse((out / "66_xml_validation_report.json").exists())


class TestRealStylePathLayout(unittest.TestCase):
    """The agent must consume a manifest at the real nested run layout and write
    artefacts alongside it, under the SAME run output directory."""

    def test_nested_run_layout_consumed(self):
        root = Path(tempfile.mkdtemp(prefix="da_realpath_"))
        run_root = root / "onboarding_output" / "client_001" / "run_pre_xml_final_check_3"
        mpath = _write_projection_package(run_root)
        # sanity: manifest lives at the expected real-style path.
        self.assertTrue(str(mpath).endswith(
            "run_pre_xml_final_check_3/output/projection/50_projection_manifest.json"))
        result = da.build_delivery_package(mpath, field_universe_path=UNIVERSE)
        out = run_root / "output" / "delivery_xml"
        self.assertEqual(Path(result["delivery_dir"]), out)
        for name in ("60_delivery_manifest.json", "61_delivery_readiness.md",
                     "62_delivery_normalised_frame.csv", "63_delivery_issues.csv",
                     "64_delivery_lineage.json"):
            self.assertTrue((out / name).exists(), name)
        manifest = json.loads((out / "60_delivery_manifest.json").read_text())
        self.assertFalse(manifest["xml_generation_allowed"])
        self.assertEqual(manifest["run_id"], "run_pre_xml_final_check_3")


class TestInspectHelperAndGrouping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="da_inspect_"))
        mpath = _write_projection_package(cls.root)
        order = _write_restricted_code_order(cls.root)
        da.build_delivery_package(
            mpath, esma_code_order_path=str(order), field_universe_path=UNIVERSE)
        cls.out = cls.root / "output" / "delivery_xml"

    def test_inspect_summary(self):
        from scripts.inspect_delivery_xml_readiness import inspect
        s = inspect(self.out)
        self.assertTrue(s["exists"])
        self.assertFalse(s["flags"]["xml_generation_allowed"])
        self.assertEqual(s["flags"]["next_agent"], "operator_config_projection_remediation")
        self.assertEqual(s["xml_files"], [])
        self.assertGreater(s["status_mix"].get("blocked", 0), 0)
        self.assertGreater(s["status_mix"].get("deliverable", 0), 0)

    def test_inspect_missing_dir(self):
        from scripts.inspect_delivery_xml_readiness import inspect
        s = inspect(self.root / "nope")
        self.assertFalse(s["exists"])

    def test_grouping_on_produced_issues(self):
        from engine.delivery_xml_agent.remediation import group_delivery_issues
        issues = json.loads((self.out / "63_delivery_issues.json").read_text())["rows"]
        g = group_delivery_issues(issues)
        self.assertIn("RREL2", g["client_onboarding"]["codes"])
        self.assertIn("RREC17", g["operator_review"]["codes"])
        self.assertIn("RREL27", g["config_mapping"]["codes"])
        self.assertIn("RREC7", g["source_projection"]["codes"])
        self.assertGreater(g["template_order"]["issue_count"], 0)
        self.assertGreater(g["delivery_structure"]["issue_count"], 0)


if __name__ == "__main__":
    unittest.main()
