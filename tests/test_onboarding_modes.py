#!/usr/bin/env python3
"""
tests/test_onboarding_modes.py

Mode support + mode-aware gap severity / readiness / review pack, updated for the
mi_only | mna_dd | regulatory_mi | warehouse_securitisation mode set.
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
from engine.onboarding_agent.mode_policy import (
    VALID_MODES,
    load_mode_policy,
    resolve_mode_alias,
)
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
        for mode in ("mi_only", "mna_dd", "regulatory_mi", "warehouse_securitisation"):
            args = parser.parse_args(
                ["--input-dir", "x", "--client-name", "c", "--output-dir", "o", "--mode", mode]
            )
            self.assertEqual(args.mode, mode)

    def test_valid_modes_constant(self):
        self.assertEqual(
            set(VALID_MODES),
            {"mi_only", "mna_dd", "regulatory_mi", "warehouse_securitisation"},
        )

    def test_mi_mna_alias_maps_to_mna_dd(self):
        canonical, msg = resolve_mode_alias("mi_mna")
        self.assertEqual(canonical, "mna_dd")
        self.assertTrue(msg)  # deprecation message present
        # And the CLI still accepts it.
        args = build_parser().parse_args(
            ["--input-dir", "x", "--client-name", "c", "--output-dir", "o", "--mode", "mi_mna"]
        )
        self.assertEqual(args.mode, "mi_mna")


class TestModeSeverity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mi, _ = _run("mi_only")
        cls.mna, _ = _run("mna_dd")
        cls.reg, _ = _run("regulatory_mi")
        cls.wh, _ = _run("warehouse_securitisation")

    def test_same_pack_different_severities(self):
        reg = _sev_by_category(self.reg)
        wh = _sev_by_category(self.wh)
        # Regulatory escalates geography to high; warehouse de-prioritises it.
        self.assertEqual(reg["geography"], "high")
        self.assertNotIn("geography", _sev_by_category(self.mi))  # mi_only: no geography gap
        self.assertEqual(wh["warehouse"], "high")
        self.assertEqual(reg["warehouse"], "low")

    def test_mi_only_no_esma_blocking(self):
        # Geography/regime gaps are not generated for mi_only at all.
        cats = {q.category for q in self.mi.gap_questions}
        self.assertNotIn("geography", cats)
        blocking_cats = {q.category for q in self.mi.gap_questions if q.severity == "blocking"}
        self.assertNotIn("geography", blocking_cats)
        self.assertNotIn("config", blocking_cats)

    def test_warehouse_flags_warehouse_terms_high(self):
        wh_qs = [q for q in self.wh.gap_questions if q.category == "warehouse"]
        self.assertTrue(wh_qs)
        self.assertTrue(all(q.severity in ("high", "blocking") for q in wh_qs))
        reg_wh = [q for q in self.reg.gap_questions if q.category == "warehouse"]
        self.assertTrue(all(q.severity in ("low", "info") for q in reg_wh))

    def test_mode_recorded_on_project(self):
        self.assertEqual(self.mi.onboarding_mode, "mi_only")
        self.assertEqual(self.mna.onboarding_mode, "mna_dd")
        self.assertEqual(self.reg.onboarding_mode, "regulatory_mi")
        self.assertEqual(self.wh.onboarding_mode, "warehouse_securitisation")

    def test_regulatory_requires_regime_config(self):
        self.assertTrue(load_mode_policy("regulatory_mi").regime_config_required)
        self.assertFalse(load_mode_policy("mi_only").regime_config_required)
        self.assertFalse(load_mode_policy("mna_dd").regime_config_required)

    # --- PART 3: out-of-scope regulatory enum suppression ---
    def _emp_enum(self, project):
        return [q for q in project.gap_questions
                if q.category == "enum" and q.subject == "employment_status"]

    def test_mi_only_suppresses_manual_enum_gap(self):
        # employment_status is regulatory non-core -> out of scope in mi_only.
        self.assertEqual(self._emp_enum(self.mi), [])

    def test_mna_dd_shows_manual_enum_visible_nonblocking(self):
        qs = self._emp_enum(self.mna)
        self.assertTrue(qs)
        self.assertNotEqual(qs[0].severity, "blocking")  # visible but non-blocking

    def test_regulatory_shows_manual_enum_active(self):
        qs = self._emp_enum(self.reg)
        self.assertTrue(qs)
        self.assertIn(qs[0].severity, ("high", "blocking"))

    # --- PART 5: missing in-scope core-field gaps ---
    def test_mi_only_emits_missing_core_field_gaps(self):
        core_qs = [q for q in self.mi.gap_questions if q.category == "core_field"]
        self.assertTrue(core_qs)
        self.assertTrue(all(q.severity == "blocking" for q in core_qs))

    def test_mna_dd_missing_core_nonblocking(self):
        core_qs = [q for q in self.mna.gap_questions if q.category == "core_field"]
        self.assertTrue(core_qs)
        # mna_dd blocks only on structural viability; non-structural core is high.
        self.assertTrue(all(q.severity != "blocking" for q in core_qs))

    def test_mi_only_no_core_gap_for_regulatory_noncore(self):
        # A regulatory non-core field (employment_status) must never appear as a
        # missing-core-field gap.
        subjects = {q.subject for q in self.mi.gap_questions if q.category == "core_field"}
        self.assertNotIn("employment_status", subjects)


class TestModeReviewPack(unittest.TestCase):
    def test_review_pack_shows_mode_and_field_scope(self):
        project, out = _run("mi_only")
        html = (out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("mi_only", html)
        self.assertIn("mode-specific readiness", html.lower())
        self.assertIn("Field scope for this onboarding mode", html)
        self.assertIn("Regulatory non-core fields are excluded", html)


if __name__ == "__main__":
    unittest.main()
