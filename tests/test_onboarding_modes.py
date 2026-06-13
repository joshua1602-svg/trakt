#!/usr/bin/env python3
"""
tests/test_onboarding_modes.py

PART 9 tests 1-6 — onboarding mode support and mode-aware gap severity /
readiness / review pack.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent.cli import build_parser
from engine.onboarding_agent.mode_policy import VALID_MODES, load_mode_policy
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding

PACK = _REPO_ROOT / "synthetic_onboarding_pack"
REGISTRY = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
ALIASES = _REPO_ROOT / "config" / "system"


def _run(mode):
    tmp = tempfile.mkdtemp(prefix=f"mode_{mode}_")
    project = run_onboarding(
        input_dir=str(PACK), client_name="TEST", output_dir=tmp,
        registry_path=str(REGISTRY), aliases_dir=str(ALIASES), mode=mode,
    )
    return project, Path(tmp)


def _sev_by_category(project):
    sev = {}
    for q in project.gap_questions:
        sev.setdefault(q.category, q.severity)
    return sev


class TestModeCLI(unittest.TestCase):
    def test_cli_accepts_all_modes(self):
        parser = build_parser()
        for mode in ("mi_mna", "regulatory_mi", "warehouse_securitisation"):
            args = parser.parse_args(
                ["--input-dir", "x", "--client-name", "c", "--output-dir", "o", "--mode", mode]
            )
            self.assertEqual(args.mode, mode)

    def test_valid_modes_constant(self):
        self.assertEqual(
            set(VALID_MODES), {"mi_mna", "regulatory_mi", "warehouse_securitisation"}
        )


class TestModeSeverity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mi, _ = _run("mi_mna")
        cls.reg, _ = _run("regulatory_mi")
        cls.wh, _ = _run("warehouse_securitisation")

    def test_same_pack_different_severities(self):
        mi = _sev_by_category(self.mi)
        reg = _sev_by_category(self.reg)
        wh = _sev_by_category(self.wh)
        # geography differs: regulatory escalates, MI/warehouse de-prioritise.
        self.assertEqual(reg["geography"], "high")
        self.assertEqual(mi["geography"], "info")
        self.assertEqual(wh["geography"], "info")

    def test_mi_mna_not_blocked_by_esma_only_gaps(self):
        sev = _sev_by_category(self.mi)
        # ESMA-only gaps (geography) must not be blocking/high in MI/M&A.
        self.assertEqual(sev["geography"], "info")
        # The only blocking gaps should be the (genuinely shared) reporting date,
        # never an ESMA-only category.
        blocking_cats = {q.category for q in self.mi.gap_questions if q.severity == "blocking"}
        self.assertNotIn("geography", blocking_cats)
        self.assertNotIn("config", blocking_cats)

    def test_regulatory_blocks_on_regulatory_requirements(self):
        sev = _sev_by_category(self.reg)
        self.assertEqual(sev["enum"], "high")
        self.assertEqual(sev["geography"], "high")
        # config mandatory gaps may escalate to blocking in regulatory mode.
        policy = load_mode_policy("regulatory_mi")
        self.assertIn("config", policy.blocking_gap_categories)
        self.assertIn("geography", policy.blocking_gap_categories)

    def test_warehouse_flags_warehouse_terms_high(self):
        wh_qs = [q for q in self.wh.gap_questions if q.category == "warehouse"]
        self.assertTrue(wh_qs)
        self.assertTrue(all(q.severity in ("high", "blocking") for q in wh_qs))
        # In regulatory mode the same warehouse terms are low priority.
        reg_wh = [q for q in self.reg.gap_questions if q.category == "warehouse"]
        self.assertTrue(all(q.severity in ("low", "info") for q in reg_wh))

    def test_mode_recorded_on_project(self):
        self.assertEqual(self.mi.onboarding_mode, "mi_mna")
        self.assertEqual(self.reg.onboarding_mode, "regulatory_mi")
        self.assertEqual(self.wh.onboarding_mode, "warehouse_securitisation")


class TestModeReviewPack(unittest.TestCase):
    def test_review_pack_shows_mode_and_readiness(self):
        project, out = _run("warehouse_securitisation")
        html = (out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("warehouse_securitisation", html)
        self.assertIn("mode-specific readiness", html.lower())
        self.assertIn("Outputs in scope", html)


if __name__ == "__main__":
    unittest.main()
