#!/usr/bin/env python3
"""tests/test_onboarding_mapping_memory.py — PART 14 (6–14).

Client-specific mapping memory: save/load, application order, client scoping,
mode/field-scope safety, material-conflict warnings, and gap suppression on a
future run (enum / ignore / source-precedence memory).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from onboarding_domain_fixtures import ALIASES, PACK, REGISTRY, SCENARIO_A
from engine.onboarding_agent import mapping_memory as mm
from engine.onboarding_agent.field_scope import resolve_field_scope
from engine.onboarding_agent.mode_policy import load_mode_policy
from engine.onboarding_agent.onboarding_models import MappingCandidate
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding


def _tmp_memory_dir(client_id: str = "client_a") -> Path:
    return Path(tempfile.mkdtemp(prefix="mem_")) / client_id / "client_memory"


def _scope(mode: str):
    return resolve_field_scope(str(REGISTRY), load_mode_policy(mode))


def _run_with_memory(memory_dir, client_id, mode="regulatory_mi"):
    out = Path(tempfile.mkdtemp(prefix="memrun_")) / "run"
    return run_onboarding(
        input_dir=str(PACK / SCENARIO_A), client_name="CLIENT_A",
        output_dir=str(out), registry_path=str(REGISTRY), aliases_dir=str(ALIASES),
        mode=mode, client_id=client_id, run_id="run_001",
        client_memory_dir=str(memory_dir), apply_client_memory=True,
    )


class TestSaveLoad(unittest.TestCase):
    # 6. Save mapping override to client memory.
    def test_save_mapping_override(self):
        d = _tmp_memory_dir()
        store = mm.MappingMemoryStore(d, client_id="client_a")
        store.save_entry(mm.MemoryEntry(
            client_id="client_a", decision_type=mm.DECISION_MAPPING_OVERRIDE,
            source_file_pattern="master_loan_collateral_tape*", source_column="loan amount",
            canonical_field="original_principal_balance", mode="regulatory_mi",
            evidence={"value_match_rate": 1.0}))
        self.assertTrue((d / "mapping_memory.yaml").exists())
        self.assertEqual(store.counts()["total"], 1)

    # 7. Load mapping memory for same client.
    def test_load_same_client(self):
        d = _tmp_memory_dir()
        mm.MappingMemoryStore(d, client_id="client_a").save_entry(mm.MemoryEntry(
            client_id="client_a", decision_type=mm.DECISION_MAPPING_OVERRIDE,
            source_column="loan_amount", canonical_field="original_principal_balance"))
        reloaded = mm.MappingMemoryStore(d, client_id="client_a")
        self.assertEqual(reloaded.counts()["total"], 1)
        e = reloaded.by_type(mm.DECISION_MAPPING_OVERRIDE)[0]
        self.assertEqual(e.canonical_field, "original_principal_balance")
        # normalisation is stable across spellings.
        self.assertEqual(e.normalized_source_column, "loan amount")


class TestApplication(unittest.TestCase):
    # 8. Apply memory before alias/registry mapping (memory wins over a prior map).
    def test_memory_wins_over_prior_mapping(self):
        d = _tmp_memory_dir()
        store = mm.MappingMemoryStore(d, client_id="client_a")
        store.save_entry(mm.MemoryEntry(
            client_id="client_a", decision_type=mm.DECISION_MAPPING_OVERRIDE,
            source_file_pattern="master_loan_collateral_tape*", source_column="loan_amount",
            canonical_field="original_principal_balance", mode="regulatory_mi",
            evidence={"value_match_rate": 1.0}))
        cand = MappingCandidate(
            source_file="master_loan_collateral_tape.csv", source_column="loan_amount",
            candidate_canonical_field="loan_amount", confidence=0.4, method="alias")
        res = mm.apply_mapping_memory([cand], store, field_scope=_scope("regulatory_mi"),
                                      mode="regulatory_mi", conflict_signals={})
        self.assertEqual(res["applied"], 1)
        self.assertEqual(cand.candidate_canonical_field, "original_principal_balance")
        self.assertEqual(cand.method, "client_memory")

    # 9. Memory does not apply to a different client.
    def test_other_client_isolated(self):
        d_a = _tmp_memory_dir("client_a")
        mm.MappingMemoryStore(d_a, client_id="client_a").save_entry(mm.MemoryEntry(
            client_id="client_a", decision_type=mm.DECISION_MAPPING_OVERRIDE,
            source_column="loan_amount", canonical_field="original_principal_balance"))
        # client_b has its own (empty) memory dir.
        d_b = _tmp_memory_dir("client_b")
        store_b = mm.MappingMemoryStore(d_b, client_id="client_b")
        self.assertTrue(store_b.is_empty)

    # 10. Memory respects mode / field scope (out-of-scope target is rejected).
    def test_mode_field_scope_safe(self):
        d = _tmp_memory_dir()
        store = mm.MappingMemoryStore(d, client_id="client_a")
        # property_post_code is a regulatory non-core field excluded in mi_only.
        store.save_entry(mm.MemoryEntry(
            client_id="client_a", decision_type=mm.DECISION_MAPPING_OVERRIDE,
            source_file_pattern="*", source_column="some_col",
            canonical_field="property_post_code", mode="regulatory_mi"))
        cand = MappingCandidate(source_file="x.csv", source_column="some_col",
                                candidate_canonical_field="", confidence=0.0)
        res = mm.apply_mapping_memory([cand], store, field_scope=_scope("mi_only"),
                                      mode="mi_only", conflict_signals={})
        self.assertEqual(res["rejected"], 1)
        self.assertEqual(res["applied"], 0)
        self.assertEqual(cand.candidate_canonical_field, "")

    # 11. Memory warning is generated when value evidence conflicts.
    def test_material_conflict_warns(self):
        d = _tmp_memory_dir()
        store = mm.MappingMemoryStore(d, client_id="client_a")
        store.save_entry(mm.MemoryEntry(
            client_id="client_a", decision_type=mm.DECISION_MAPPING_OVERRIDE,
            source_file_pattern="*", source_column="loan_amount",
            canonical_field="original_principal_balance", mode="regulatory_mi",
            evidence={"value_match_rate": 1.0}))
        cand = MappingCandidate(source_file="x.csv", source_column="loan_amount",
                                candidate_canonical_field="loan_amount", confidence=0.4)
        res = mm.apply_mapping_memory(
            [cand], store, field_scope=_scope("regulatory_mi"), mode="regulatory_mi",
            conflict_signals={"original_principal_balance": 0.55})
        self.assertEqual(res["warned"], 1)
        self.assertTrue(res["gap_questions"])
        self.assertTrue(cand.requires_review)


class TestFutureRunSuppression(unittest.TestCase):
    # 12. Ignored column memory suppresses future gaps for that column.
    def test_ignored_column_suppresses_gaps(self):
        d = _tmp_memory_dir()
        mm.MappingMemoryStore(d, client_id="client_a").save_entry(mm.MemoryEntry(
            client_id="client_a", decision_type=mm.DECISION_IGNORE_COLUMN,
            source_file_pattern="master_loan_collateral_tape*",
            source_column="employment_status", mode="regulatory_mi"))
        project = _run_with_memory(d, "client_a")
        enum_gaps = [q for q in project.gap_questions
                     if q.category == "enum" and q.subject == "employment_status"]
        self.assertEqual(enum_gaps, [])

    # 13. Enum memory applies to a future run.
    def test_enum_memory_applies(self):
        d = _tmp_memory_dir()
        store = mm.MappingMemoryStore(d, client_id="client_a")
        for raw in ("manual", "PART_TIME"):
            store.save_entry(mm.MemoryEntry(
                client_id="client_a", decision_type=mm.DECISION_ENUM_MAPPING,
                canonical_field="employment_status", source_value=raw,
                mode="regulatory_mi", evidence={"decision": "treat_as_missing"}))
        project = _run_with_memory(d, "client_a")
        enum_gaps = [q for q in project.gap_questions
                     if q.category == "enum" and q.subject == "employment_status"]
        self.assertEqual(enum_gaps, [])

    # 14. Source precedence memory applies to a future run.
    def test_source_precedence_memory_applies(self):
        d = _tmp_memory_dir()
        mm.MappingMemoryStore(d, client_id="client_a").save_entry(mm.MemoryEntry(
            client_id="client_a", decision_type=mm.DECISION_SOURCE_PRECEDENCE,
            canonical_field="current_principal_balance", mode="regulatory_mi",
            evidence={"primary_source_file": "master_loan_collateral_tape.csv"}))
        project = _run_with_memory(d, "client_a")
        sot_gaps = [q for q in project.gap_questions
                    if q.category == "source_of_truth"
                    and q.subject == "current_principal_balance"]
        self.assertEqual(sot_gaps, [])


if __name__ == "__main__":
    unittest.main()
