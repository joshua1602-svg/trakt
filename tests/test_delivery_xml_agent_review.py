#!/usr/bin/env python3
"""tests/test_delivery_xml_agent_review.py

Delivery/XML Agent v1 — review + skeleton checks.

Covers:
  * the review doc exists and answers all 16 review questions + the Gate 4b /
    Gate 5 reviews;
  * the agent package skeleton is importable and exposes the public API;
  * the gate5_adapter is non-raising and reuses only safe predicates;
  * the delivery-readiness gates refuse XML when any gate fails and only allow
    XML when every gate passes.
"""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DOC = _REPO_ROOT / "docs" / "delivery_xml_agent_v1_review.md"
ROADMAP = _REPO_ROOT / "docs" / "xml_readiness_remediation_roadmap.md"
PREVIEW_PLAN = _REPO_ROOT / "docs" / "minimum_xml_preview_remediation_plan.md"
PREVIEW_MATRIX = (_REPO_ROOT / "output" / "config_review"
                  / "minimum_xml_preview_remediation_matrix.csv")
PREVIEW_POLICY = _REPO_ROOT / "config" / "delivery" / "xml_preview_policy.yaml"
PREVIEW_SPEC = _REPO_ROOT / "docs" / "xml_preview_policy_spec.md"


class TestReviewDoc(unittest.TestCase):
    def setUp(self):
        self.assertTrue(DOC.exists(), "docs/delivery_xml_agent_v1_review.md must exist")
        self.text = DOC.read_text(encoding="utf-8")

    def test_all_review_questions_present(self):
        # 16 numbered review questions must each appear as a "### N." heading.
        for n in range(1, 17):
            self.assertRegex(self.text, rf"###\s*{n}\.")

    def test_gate_reviews_present(self):
        self.assertIn("Gate 4b review", self.text)
        self.assertIn("Gate 5 review", self.text)

    def test_records_no_production_xml(self):
        self.assertIn("No production XML", self.text)
        self.assertIn("ready_for_xml_delivery", self.text)

    def test_documents_reuse_and_adapt(self):
        # acceptance criterion 8: documents reuse from Gate 4b/Gate 5 + adapt.
        self.assertIn("safe to reuse", self.text.lower())
        self.assertIn("unsafe", self.text.lower())

    def test_record_group_questions_covered(self):
        for token in ("RREL", "RREC", "collateral", "nest"):
            self.assertIn(token, self.text)


class TestSkeletonImportable(unittest.TestCase):
    def test_public_api(self):
        from engine.delivery_xml_agent import (
            build_delivery_package, ProjectionHandoffError)
        self.assertTrue(callable(build_delivery_package))
        self.assertTrue(issubclass(ProjectionHandoffError, RuntimeError))

    def test_modules_present(self):
        import importlib
        for mod in ("workflow", "cli", "delivery_xml_agent", "gate5_adapter",
                    "delivery_readiness"):
            importlib.import_module(f"engine.delivery_xml_agent.{mod}")

    def test_workflow_parser_has_allow_xml_preview(self):
        from engine.delivery_xml_agent import workflow as wf
        p = wf.build_parser()
        # parsing without the flag yields allow_xml_preview False.
        ns = p.parse_args(["--projection-manifest", "x"])
        self.assertFalse(ns.allow_xml_preview)
        ns2 = p.parse_args(["--projection-manifest", "x", "--allow-xml-preview"])
        self.assertTrue(ns2.allow_xml_preview)


class TestGate5AdapterNonRaising(unittest.TestCase):
    def test_is_nd_value(self):
        from engine.delivery_xml_agent import gate5_adapter as g5
        for v in ("ND1", "ND5", "nd3"):
            self.assertTrue(g5.is_nd_value(v))
        for v in ("", "RHOS", "ND6", "NODATA", "123"):
            self.assertFalse(g5.is_nd_value(v))

    def test_record_group_to_xml_group(self):
        from engine.delivery_xml_agent import gate5_adapter as g5
        self.assertEqual(g5.record_group_to_xml_group("RREL"), "underlying_exposure")
        self.assertEqual(g5.record_group_to_xml_group("RREC"), "collateral")
        self.assertEqual(g5.record_group_to_xml_group("other"), "header_pool_report")
        self.assertEqual(g5.record_group_to_xml_group(""), "header_pool_report")

    def test_loaders_non_raising_on_missing(self):
        from engine.delivery_xml_agent import gate5_adapter as g5
        self.assertEqual(g5.load_record_order("/no/such.yaml"), [])
        self.assertEqual(g5.field_universe_index("/no/such.yaml"), {})
        self.assertEqual(g5.xsd_type_for_code("RREL1", {}), "")

    def test_format_and_enum_pass_for_nd_and_blank(self):
        from engine.delivery_xml_agent import gate5_adapter as g5
        rule = {"validators": {"regex": "^[A-Z]{4}$"}, "transform": {"enum_map": {"a": "RHOS"}}}
        self.assertTrue(g5.format_valid("", rule))
        self.assertTrue(g5.format_valid("ND5", rule))
        self.assertTrue(g5.format_valid("ABCD", rule))
        self.assertFalse(g5.format_valid("abc1", rule))
        self.assertTrue(g5.enum_valid("RHOS", rule))     # mapped target value
        self.assertTrue(g5.enum_valid("a", rule))         # known key
        self.assertFalse(g5.enum_valid("zzz", rule))

    def test_loads_real_record_order(self):
        from engine.delivery_xml_agent import gate5_adapter as g5
        order = g5.load_record_order(
            _REPO_ROOT / "config" / "system" / "esma_code_order.yaml")
        self.assertIn("RREL1", order)


class TestReadinessGates(unittest.TestCase):
    def _all_good_kwargs(self):
        return dict(
            projection_complete=True,
            ready_for_delivery_normalisation=True,
            ready_for_xml_delivery=True,
            delivery_blocking_projection_issue_count=0,
            blocked_frame_row_count=0,
            mandatory_blank_without_nd_count=0,
            format_violation_count=0,
            missing_header_metadata=[],
            rows_without_record_group=0,
            missing_required_order_codes=[],
        )

    def test_all_pass_allows_xml(self):
        from engine.delivery_xml_agent.delivery_readiness import compute_delivery_readiness
        r = compute_delivery_readiness(**self._all_good_kwargs())
        self.assertTrue(r["xml_generation_allowed"])
        self.assertTrue(all(g["passed"] for g in r["gates"]))

    def test_any_blocker_refuses_xml(self):
        from engine.delivery_xml_agent.delivery_readiness import compute_delivery_readiness
        kw = self._all_good_kwargs()
        kw["blocked_frame_row_count"] = 3
        r = compute_delivery_readiness(**kw)
        self.assertFalse(r["xml_generation_allowed"])

    def test_projection_incomplete_refuses_xml(self):
        from engine.delivery_xml_agent.delivery_readiness import compute_delivery_readiness
        kw = self._all_good_kwargs()
        kw["projection_complete"] = False
        r = compute_delivery_readiness(**kw)
        self.assertFalse(r["xml_generation_allowed"])

    def test_gate_names_stable(self):
        from engine.delivery_xml_agent.delivery_readiness import (
            compute_delivery_readiness, GATE_NAMES)
        r = compute_delivery_readiness(**self._all_good_kwargs())
        self.assertEqual([g["gate"] for g in r["gates"]], GATE_NAMES)


class TestRemediationRoadmap(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ROADMAP.exists(),
                        "docs/xml_readiness_remediation_roadmap.md must exist")
        self.text = ROADMAP.read_text(encoding="utf-8")

    def test_seven_groups_present(self):
        for title in ("Client onboarding decisions", "Operator decisions",
                      "Config mapping decisions", "Source / projection mapping gaps",
                      "ND / default policy gaps", "Delivery structure gaps",
                      "Template / order gaps"):
            self.assertIn(title, self.text)

    def test_action_plan_sections_present(self):
        # A..F plain-English action plan.
        for letter in "ABCDEF":
            self.assertRegex(self.text, rf"###\s*{letter}\.")

    def test_per_field_dimensions_present(self):
        for token in ("Field codes", "Current blocker type", "Business meaning",
                      "Recommended owner", "Recommended action",
                      "Needed before XML preview", "Needed before production XML"):
            self.assertIn(token, self.text)

    def test_known_codes_referenced(self):
        for code in ("RREL1", "RREL2", "RREC9", "RREC13", "RREC17", "RREL43", "RREL27"):
            self.assertIn(code, self.text)

    def test_no_production_xml_statement(self):
        self.assertIn("No production XML", self.text)


class TestIssueGrouping(unittest.TestCase):
    def setUp(self):
        from engine.delivery_xml_agent import remediation as rem
        from engine.delivery_xml_agent import delivery_xml_agent as da
        self.rem = rem
        self.da = da

    def test_groups_in_stable_order(self):
        groups = self.rem.group_delivery_issues([])
        self.assertEqual(
            list(groups.keys()),
            ["client_onboarding", "operator_review", "config_mapping",
             "source_projection", "nd_default", "delivery_structure", "template_order"])
        # empty groups still present with zero counts.
        self.assertTrue(all(g["issue_count"] == 0 for g in groups.values()))

    def test_blocker_types_map_to_expected_groups(self):
        da = self.da
        issues = [
            {"delivery_issue_id": "DEL-0001", "delivery_blocker_type": da.BT_CLIENT, "esma_code": "RREL1"},
            {"delivery_issue_id": "DEL-0002", "delivery_blocker_type": da.BT_OPERATOR_OR_CONFIG, "esma_code": "RREC17"},
            {"delivery_issue_id": "DEL-0003", "delivery_blocker_type": da.BT_CONFIG, "esma_code": "RREL27"},
            {"delivery_issue_id": "DEL-0004", "delivery_blocker_type": da.BT_SOURCE_MAPPING, "esma_code": "RREC7"},
            {"delivery_issue_id": "DEL-0005", "delivery_blocker_type": da.BT_FORMAT, "esma_code": "RREL16"},
            {"delivery_issue_id": "DEL-0006", "delivery_blocker_type": da.BT_ND_DEFAULT_MISSING, "esma_code": "RREL40"},
            {"delivery_issue_id": "DEL-0007", "delivery_blocker_type": da.BT_STRUCTURE_DEFERRED, "esma_code": ""},
            {"delivery_issue_id": "DEL-0008", "delivery_blocker_type": da.BT_TEMPLATE_ORDER, "esma_code": "RREL2,RREL3"},
        ]
        g = self.rem.group_delivery_issues(issues)
        self.assertEqual(g["client_onboarding"]["codes"], ["RREL1"])
        self.assertEqual(g["operator_review"]["codes"], ["RREC17"])
        self.assertEqual(g["config_mapping"]["codes"], ["RREL27"])
        # source/projection absorbs both source-mapping and format-invalid.
        self.assertEqual(g["source_projection"]["codes"], ["RREC7", "RREL16"])
        self.assertEqual(g["source_projection"]["issue_count"], 2)
        self.assertEqual(g["nd_default"]["codes"], ["RREL40"])
        self.assertEqual(g["delivery_structure"]["issue_count"], 1)
        # comma-joined template-order codes are split.
        self.assertEqual(g["template_order"]["codes"], ["RREL2", "RREL3"])

    def test_preview_vs_production_flags(self):
        g = self.rem.group_delivery_issues([])
        # delivery_structure is the only group deferred past preview.
        self.assertFalse(g["delivery_structure"]["needed_before_preview"])
        self.assertTrue(g["delivery_structure"]["needed_before_production"])
        for key in ("client_onboarding", "operator_review", "config_mapping",
                    "source_projection", "nd_default", "template_order"):
            self.assertTrue(g[key]["needed_before_preview"], key)
            self.assertTrue(g[key]["needed_before_production"], key)


_ALLOWED_PREVIEW_TREATMENTS = {
    "must_resolve", "explicit_preview_assumption", "preview_exclusion",
    "synthetic_placeholder_for_demo_only", "defer_until_production",
    "not_required_for_preview",
}
_MATRIX_COLUMNS = [
    "esma_code", "canonical_field", "current_blocker_type", "issue_group",
    "affected_rows", "preview_required", "production_required",
    "recommended_preview_treatment", "recommended_production_treatment",
    "owner", "risk_level", "reason",
]


class TestMinimumXmlPreviewPlan(unittest.TestCase):
    def test_plan_doc_present_and_answers_questions(self):
        self.assertTrue(PREVIEW_PLAN.exists(),
                        "docs/minimum_xml_preview_remediation_plan.md must exist")
        text = PREVIEW_PLAN.read_text(encoding="utf-8")
        for n in range(1, 11):  # the ten questions
            self.assertRegex(text, rf"###\s*{n}\.")
        self.assertIn("smallest safe path", text.lower())
        self.assertIn("No silent fills", text)
        self.assertIn("xml_generation", text)

    def test_plan_classifies_rrel82_as_onboarding_static_reference(self):
        text = PREVIEW_PLAN.read_text(encoding="utf-8")
        self.assertIn("onboarding_static_reference", text)
        self.assertIn("RREL82", text)
        # explicitly: captured during onboarding + ND not allowed.
        self.assertRegex(text, r"RREL82[\s\S]{0,400}?[Oo]nboarding")
        self.assertRegex(text, r"RREL82[\s\S]{0,400}?[Nn]o ND")
        # not framed as an ND/default policy item.
        self.assertNotRegex(text, r"RREL82[^\n]*ND/default rule")

    def test_matrix_generates_and_is_well_formed(self):
        from scripts.build_minimum_xml_preview_matrix import build_rows
        rows = build_rows()
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(set(r.keys()), set(_MATRIX_COLUMNS))
            self.assertIn(r["recommended_preview_treatment"], _ALLOWED_PREVIEW_TREATMENTS)
            self.assertIn(r["recommended_production_treatment"], _ALLOWED_PREVIEW_TREATMENTS)

    def test_no_silent_fill_or_fake_production(self):
        from scripts.build_minimum_xml_preview_matrix import build_rows
        for r in build_rows():
            # synthetic placeholders are PREVIEW-only, never a production treatment.
            self.assertNotEqual(r["recommended_production_treatment"],
                                "synthetic_placeholder_for_demo_only", r["esma_code"])

    def test_client_identifiers_not_guessed(self):
        from scripts.build_minimum_xml_preview_matrix import build_rows
        by_code = {r["esma_code"]: r for r in build_rows()}
        for code in ("RREL1", "RREL2"):
            self.assertEqual(by_code[code]["owner"], "client_onboarding")
            # production must be earned via onboarding, not a placeholder.
            self.assertEqual(by_code[code]["recommended_production_treatment"], "must_resolve")
            self.assertEqual(by_code[code]["recommended_preview_treatment"],
                             "synthetic_placeholder_for_demo_only")

    def test_operator_valuations_stay_operator(self):
        from scripts.build_minimum_xml_preview_matrix import build_rows
        by_code = {r["esma_code"]: r for r in build_rows()}
        for code in ("RREC17", "RREC13", "RREC9", "RREL43"):
            self.assertEqual(by_code[code]["owner"], "operator")
            self.assertEqual(by_code[code]["recommended_production_treatment"], "must_resolve")
            # never fabricated for preview.
            self.assertEqual(by_code[code]["recommended_preview_treatment"], "preview_exclusion")

    def test_rrel82_is_onboarding_static_reference_not_nd_default(self):
        from scripts.build_minimum_xml_preview_matrix import build_rows
        rrel82 = next(r for r in build_rows() if r["esma_code"] == "RREL82")
        # business group reclassified away from nd_default.
        self.assertEqual(rrel82["issue_group"], "onboarding_static_reference")
        self.assertNotEqual(rrel82["issue_group"], "nd_default")
        # owned by onboarding, production must_resolve, preview placeholder demo-only.
        self.assertIn("onboarding", rrel82["owner"])
        self.assertEqual(rrel82["recommended_production_treatment"], "must_resolve")
        self.assertEqual(rrel82["recommended_preview_treatment"],
                         "synthetic_placeholder_for_demo_only")
        # reason must state ND is not allowed and never fabricate.
        self.assertIn("ND is NOT allowed", rrel82["reason"])
        self.assertIn("onboarding", rrel82["reason"].lower())

    def test_no_code_grouped_as_nd_default(self):
        # the nd_default business group is now empty for this run.
        from scripts.build_minimum_xml_preview_matrix import build_rows
        self.assertFalse([r for r in build_rows() if r["issue_group"] == "nd_default"])

    def test_rrel35_documented_as_resolved(self):
        from scripts.build_minimum_xml_preview_matrix import build_rows
        rrel35 = next(r for r in build_rows() if r["esma_code"] == "RREL35")
        self.assertEqual(rrel35["recommended_preview_treatment"], "not_required_for_preview")
        self.assertEqual(rrel35["current_blocker_type"], "resolved")

    def test_matrix_csv_written_matches_builder(self):
        # the committed CSV exists and has the contract columns.
        self.assertTrue(PREVIEW_MATRIX.exists(),
                        "minimum_xml_preview_remediation_matrix.csv must exist")
        with open(PREVIEW_MATRIX, newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        self.assertEqual(header, _MATRIX_COLUMNS)


class TestXmlPreviewPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = yaml.safe_load(PREVIEW_POLICY.read_text(encoding="utf-8"))["preview_policy"]
        rows = list(csv.DictReader(open(PREVIEW_MATRIX, encoding="utf-8")))
        cls.m_ph = {r["esma_code"] for r in rows
                    if r["recommended_preview_treatment"] == "synthetic_placeholder_for_demo_only"}
        cls.m_ex = {r["esma_code"] for r in rows
                    if r["recommended_preview_treatment"] == "preview_exclusion"}
        cls.m_mr = {("record_structure" if r["esma_code"] == "(structural)" else r["esma_code"])
                    for r in rows if r["recommended_preview_treatment"] == "must_resolve"}

    def test_spec_and_config_exist(self):
        self.assertTrue(PREVIEW_SPEC.exists())
        self.assertTrue(PREVIEW_POLICY.exists())

    def test_disabled_by_default(self):
        self.assertFalse(self.policy["enabled"])
        self.assertEqual(self.policy["mode"], "non_production_preview")
        self.assertTrue(self.policy["preserve_production_gates"])
        self.assertFalse(self.policy["allow_silent_defaults"])
        self.assertFalse(self.policy["allow_nd_without_policy"])

    def test_preview_and_production_flags_are_separate(self):
        prev = set(self.policy["preview_gate_flags"])
        prod = set(self.policy["production_gate_flags_unchanged"])
        self.assertEqual(prev, {"xml_preview_allowed", "xml_preview_generated",
                                "ready_for_xml_preview"})
        self.assertEqual(prod, {"xml_generation_allowed", "ready_for_xml_delivery",
                                "xml_generated"})
        self.assertFalse(prev & prod)  # disjoint — gates do not collapse.

    def test_production_guardrails(self):
        g = self.policy["production_guardrails"]
        self.assertTrue(g["never_set_xml_generation_allowed"])
        self.assertTrue(g["never_set_ready_for_xml_delivery"])
        self.assertTrue(g["never_set_xml_generated"])
        self.assertTrue(g["preview_output_must_be_separate"])
        self.assertTrue(g["do_not_promote_placeholders_to_production"])
        self.assertTrue(g["do_not_fabricate_valuation_or_source"])

    def test_field_lists_match_matrix(self):
        ph = set(self.policy["placeholder_policy"]["fields"])
        ex = set(self.policy["exclusion_policy"]["fields"])
        mr = set()
        for v in self.policy["must_resolve_before_preview"].values():
            mr.update(v)
        self.assertEqual(ph, self.m_ph)
        self.assertEqual(ex, self.m_ex)
        self.assertEqual(mr, self.m_mr)

    def test_placeholder_prefix_and_non_reportable(self):
        pp = self.policy["placeholder_policy"]
        self.assertEqual(pp["prefix"], "PREVIEW_ONLY_")
        self.assertTrue(pp["non_reportable"])

    def test_rrel82_is_onboarding_static_reference_placeholder(self):
        f = self.policy["placeholder_policy"]["fields"]["RREL82"]
        self.assertEqual(f["business_group"], "onboarding_static_reference")
        self.assertIn("onboarding", f["owner"])
        self.assertIn("ND is NOT allowed", f["reason"])

    def test_rrel35_resolved_not_in_preview_logic(self):
        ph = set(self.policy["placeholder_policy"]["fields"])
        ex = set(self.policy["exclusion_policy"]["fields"])
        mr = set()
        for v in self.policy["must_resolve_before_preview"].values():
            mr.update(v)
        self.assertNotIn("RREL35", ph | ex | mr)
        self.assertIn("RREL35", self.policy["resolved_examples"])

    def test_operator_valuations_excluded_not_fabricated(self):
        ex = self.policy["exclusion_policy"]["fields"]
        for code in ("RREC13", "RREC17", "RREC9", "RREL43"):
            self.assertIn(code, ex, code)

    def test_watermark_present(self):
        self.assertIn("NON-PRODUCTION PREVIEW", self.policy["watermark"])
        self.assertIn("NON-PRODUCTION PREVIEW", PREVIEW_SPEC.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
