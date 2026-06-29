#!/usr/bin/env python3
"""
tests/test_onboarding_field_scope.py

PART 10 — field-scope resolver + its application to mapping, gaps, config and
answer ingestion, driven by registry `category` and `core_canonical`.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.gate_1_alignment.semantic_alignment import load_field_registry
from engine.onboarding_agent.answer_ingestion import STATUS_READY, ingest_answers
from engine.onboarding_agent.field_scope import resolve_field_scope
from engine.onboarding_agent.mode_policy import load_mode_policy
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding

PACK = _REPO_ROOT / "synthetic_onboarding_pack"
REGISTRY_PATH = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
ALIASES = _REPO_ROOT / "config" / "system"
REGISTRY = load_field_registry(REGISTRY_PATH)
_FIELDS = REGISTRY["fields"]


def _scope(mode, **kw):
    return resolve_field_scope(REGISTRY, load_mode_policy(mode), **kw)


def _cat(f):
    return _FIELDS.get(f, {}).get("category")


def _core(f):
    return _FIELDS.get(f, {}).get("core_canonical") is True


class TestRegistryKey(unittest.TestCase):
    """PART 1 — the real registry uses core_canonical (not canonical_core)."""

    def test_core_canonical_key_present_and_counts(self):
        core = [f for f, m in _FIELDS.items() if m.get("core_canonical") is True]
        self.assertEqual(len(core), 14)
        # The wrong key must not exist anywhere in the registry.
        self.assertFalse(any("canonical_core" in m for m in _FIELDS.values()))

    def test_category_counts(self):
        reg = [f for f, m in _FIELDS.items() if m.get("category") == "regulatory"]
        ana = [f for f, m in _FIELDS.items() if m.get("category") == "analytics"]
        # 316 includes collateral_unique_identifier (Annex 2 RREC1), added to
        # close the fields_registry ESMA_Annex2 mapping gap.
        self.assertEqual(len(reg), 316)
        # 183 analytics = 177 prior + 6 source-portfolio provenance fields
        # (source_portfolio_id/type/label, acquisition_date, seller_name,
        # portfolio_cohort). The prior literal (156) was already stale vs the
        # registry before provenance was added.
        self.assertEqual(len(ana), 183)

    def test_regulatory_core_field_example(self):
        # current_principal_balance is regulatory + core -> stays in MI-only scope.
        self.assertEqual(_cat("current_principal_balance"), "regulatory")
        self.assertTrue(_core("current_principal_balance"))
        # employment_status is regulatory + non-core -> excluded in MI-only.
        self.assertEqual(_cat("employment_status"), "regulatory")
        self.assertFalse(_core("employment_status"))


class TestResolver(unittest.TestCase):
    def test_mi_only_excludes_regulatory_noncore(self):
        s = _scope("mi_only")
        # Every excluded field is regulatory and not core.
        self.assertTrue(s.excluded_fields)
        for f in s.excluded_fields:
            self.assertEqual(_cat(f), "regulatory")
            self.assertFalse(_core(f))
        # No analytics field is excluded.
        self.assertFalse(s.excluded_fields & s.analytics_fields)

    def test_mi_only_includes_core_canonical_even_if_regulatory(self):
        s = _scope("mi_only")
        self.assertIn("current_principal_balance", s.included_fields)  # core + regulatory
        self.assertTrue(s.core_canonical_fields.issubset(s.included_fields))
        # A regulatory core field is included despite being regulatory category.
        self.assertEqual(_cat("current_principal_balance"), "regulatory")
        self.assertTrue(_core("current_principal_balance"))

    def test_mi_only_includes_analytics(self):
        s = _scope("mi_only")
        analytics_examples = [f for f in s.analytics_fields if not _core(f)][:5]
        for f in analytics_examples:
            self.assertIn(f, s.included_fields)

    def test_mi_only_blocking_is_core_canonical_only(self):
        s = _scope("mi_only")
        self.assertTrue(s.blocking_fields)
        self.assertTrue(s.blocking_fields.issubset(s.core_canonical_fields))

    def test_mi_only_legal_entity_fields_not_blocking_but_included(self):
        # Regulatory legal-entity identifiers must not block MI-only promotion,
        # yet remain in scope (visible/mappable, raised as a non-blocking gap).
        s = _scope("mi_only")
        for f in ("originator_legal_entity_identifier", "originator_name"):
            self.assertIn(f, s.core_canonical_fields)
            self.assertIn(f, s.included_fields)        # still in scope
            self.assertNotIn(f, s.blocking_fields)     # but not blocking

    def test_mi_only_mi_critical_core_fields_remain_blocking(self):
        # MI-critical core fields are unaffected and still block where required.
        s = _scope("mi_only")
        for f in ("loan_identifier", "current_principal_balance"):
            if f in s.included_fields:
                self.assertIn(f, s.blocking_fields, f)

    def test_regulatory_mi_legal_entity_fields_still_blocking(self):
        # Regulatory mode requirements must NOT be weakened.
        s = _scope("regulatory_mi")
        for f in ("originator_legal_entity_identifier", "originator_name"):
            self.assertIn(f, s.blocking_fields, f)

    def test_warehouse_legal_entity_blocking_follows_regulatory_flag(self):
        s_off = _scope("warehouse_securitisation", regulatory_reporting_enabled=False)
        s_on = _scope("warehouse_securitisation", regulatory_reporting_enabled=True)
        for f in ("originator_legal_entity_identifier", "originator_name"):
            self.assertNotIn(f, s_off.blocking_fields, f)   # non-blocking when off
            self.assertIn(f, s_on.blocking_fields, f)       # re-blocks when on

    def test_mna_dd_full_universe(self):
        s = _scope("mna_dd")
        self.assertFalse(s.excluded_fields)  # nothing excluded
        self.assertTrue(s.regulatory_fields.issubset(s.included_fields))

    def test_mna_dd_blocking_structural_only(self):
        s = _scope("mna_dd")
        # Blocking limited to structural viability essentials, not all core.
        self.assertTrue(s.blocking_fields.issubset({"loan_identifier", "current_principal_balance"}))
        self.assertLess(len(s.blocking_fields), len(s.core_canonical_fields))

    def test_regulatory_mi_full_universe_and_regime(self):
        s = _scope("regulatory_mi")
        self.assertFalse(s.excluded_fields)
        self.assertTrue(load_mode_policy("regulatory_mi").regime_config_required)

    def test_warehouse_excludes_regulatory_unless_enabled(self):
        s_off = _scope("warehouse_securitisation", regulatory_reporting_enabled=False)
        self.assertTrue(s_off.excluded_fields & s_off.regulatory_fields)
        s_on = _scope("warehouse_securitisation", regulatory_reporting_enabled=True)
        self.assertTrue(s_on.regulatory_fields.issubset(s_on.included_fields))


class TestScopeApplied(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_mi = Path(tempfile.mkdtemp(prefix="fs_mi_"))
        cls.mi = run_onboarding(
            input_dir=str(PACK), client_name="MI", output_dir=str(cls.tmp_mi),
            registry_path=str(REGISTRY_PATH), aliases_dir=str(ALIASES), mode="mi_only",
        )
        cls.tmp_mna = Path(tempfile.mkdtemp(prefix="fs_mna_"))
        cls.mna = run_onboarding(
            input_dir=str(PACK), client_name="MNA", output_dir=str(cls.tmp_mna),
            registry_path=str(REGISTRY_PATH), aliases_dir=str(ALIASES), mode="mna_dd",
        )

    def test_mi_only_mapping_no_excluded_targets(self):
        scope = _scope("mi_only")
        targets = {m.candidate_canonical_field for m in self.mi.mapping_candidates if m.candidate_canonical_field}
        self.assertFalse(targets & scope.excluded_fields)

    def test_mi_only_out_of_scope_artifact(self):
        self.assertTrue((self.tmp_mi / "05a_out_of_scope_fields.csv").exists())
        self.assertTrue(self.mi.out_of_scope_fields)
        # Every diverted field is a regulatory category target.
        for o in self.mi.out_of_scope_fields:
            self.assertEqual(o["category"], "regulatory")

    def test_mi_only_no_regime_classification_config(self):
        fields = {c.field for c in self.mi.config_suggestions}
        self.assertNotIn("regime", fields)
        self.assertNotIn("classification_year", fields)
        self.assertNotIn("geography_policy", fields)

    def test_mi_only_out_of_scope_summary_question(self):
        scope_qs = [q for q in self.mi.gap_questions if q.category == "scope"]
        self.assertTrue(scope_qs)
        self.assertIn("regulatory fields", scope_qs[0].question)

    def test_mna_dd_includes_regulatory_in_mapping_universe(self):
        # No fields diverted out of scope in mna_dd (full coverage).
        self.assertEqual(self.mna.out_of_scope_fields, [])

    def test_mna_dd_regulatory_gaps_nonblocking(self):
        # mna_dd blocks only on structural viability; geography/config gaps absent
        # or non-blocking.
        blocking = [q for q in self.mna.gap_questions if q.severity == "blocking"]
        for q in blocking:
            self.assertEqual(q.category, "date")  # only structural date is blocking

    def test_mna_dd_no_required_regime(self):
        fields = {c.field for c in self.mna.config_suggestions}
        self.assertNotIn("regime", fields)  # only possible_regime, not required regime
        self.assertIn("possible_regime", fields)


class TestRegulatoryReportingFlag(unittest.TestCase):
    def test_warehouse_regulatory_flag_activates_regulatory(self):
        tmp = Path(tempfile.mkdtemp(prefix="fs_reg_"))
        proj = run_onboarding(
            input_dir=str(PACK), client_name="WH", output_dir=str(tmp),
            registry_path=str(REGISTRY_PATH), aliases_dir=str(ALIASES),
            mode="warehouse_securitisation", regulatory_reporting_enabled=True,
        )
        # With the flag on, no regulatory field is diverted out of scope.
        self.assertEqual(proj.out_of_scope_fields, [])

    def test_warehouse_without_flag_excludes_regulatory(self):
        tmp = Path(tempfile.mkdtemp(prefix="fs_noreg_"))
        proj = run_onboarding(
            input_dir=str(PACK), client_name="WH", output_dir=str(tmp),
            registry_path=str(REGISTRY_PATH), aliases_dir=str(ALIASES),
            mode="warehouse_securitisation",
        )
        self.assertTrue(proj.out_of_scope_fields)


class TestAnswerIngestionScope(unittest.TestCase):
    def test_mi_only_approval_without_regulatory_answers(self):
        tmp = Path(tempfile.mkdtemp(prefix="fs_ing_"))
        run_onboarding(
            input_dir=str(PACK), client_name="MI", output_dir=str(tmp),
            registry_path=str(REGISTRY_PATH), aliases_dir=str(ALIASES), mode="mi_only",
        )
        # Answer only the (non-regulatory) questions from the generated template.
        answers = yaml.safe_load((tmp / "example_answers.yaml").read_text())
        report = ingest_answers(tmp, tmp / "example_answers.yaml", confirm=True)
        # No regime/geography/classification answers were needed to reach ready.
        self.assertEqual(report["approval_status"], STATUS_READY)
        cfg = yaml.safe_load((tmp / "11_approved_config.yaml").read_text())
        self.assertNotIn("regime", cfg)
        self.assertNotIn("classification_year", cfg)


if __name__ == "__main__":
    unittest.main()
