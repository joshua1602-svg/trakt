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

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DOC = _REPO_ROOT / "docs" / "delivery_xml_agent_v1_review.md"


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


if __name__ == "__main__":
    unittest.main()
