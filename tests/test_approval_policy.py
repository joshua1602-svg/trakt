#!/usr/bin/env python3
"""tests/test_approval_policy.py

Phase 2 — the APPROVAL POLICY materiality/evidence classifier + its router wiring.

Proves the four acceptance criteria:
  (a) a brand-new source        → pending_review (one-click);
  (b) recurring, cosmetic change → AUTO-APPROVE, fingerprint re-pinned, governance
      evidence written, pipeline completes with no human action;
  (c) recurring, mandatory field removed → pending_review (material);
  (d) exact-schema recurring     → deterministic (no LLM, no approval, no classify).

Plus unit coverage of the pure classifier and the LLM-mode policy resolution.

Run: python -m unittest tests.test_approval_policy
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app import approval_policy as AP
from apps.blob_trigger_app import router as R
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.repin import repin_source
from apps.blob_trigger_app.schema_fingerprint import fingerprint_pack
from apps.blob_trigger_app.source_registry import SourceRegistry, STATUS_ACTIVE
from apps.blob_trigger_app.storage import Storage

_NOW = "2026-10-01T00:00:00+00:00"
_CONTAINER = "raw-v2"
_LOAN = ["loan_id", "balance", "rate", "origination_date", "maturity_date"]


class _Invoker:
    def __init__(self, status="done"):
        self.calls = []
        self.status = status

    def __call__(self, **kw):
        self.calls.append(kw)
        return {"run_id": "orun", "status": self.status,
                "central_canonical_path": None, "blockers": []}


def _stub_assembler(**kw):
    return {}


def _write_pack(dirpath, files):
    dirpath.mkdir(parents=True, exist_ok=True)
    out = []
    for name, cols in files.items():
        p = dirpath / name
        p.write_text(",".join(cols) + "\n" + ",".join(["x"] * len(cols)) + "\n")
        out.append(str(p))
    return sorted(out)


# --------------------------------------------------------------------------- #
# Pure classifier
# --------------------------------------------------------------------------- #

class TestClassifier(unittest.TestCase):

    def test_cosmetic_header_is_non_material_auto(self):
        r = AP.classify(
            old_role_schemas={"loan_extract": ["loan_id", "balance", "rate"]},
            new_role_schemas={"loan_extract": ["Loan ID", "Balance", "Rate"]})
        self.assertFalse(r.material)
        self.assertTrue(r.auto_approvable)
        self.assertTrue(r.evidence["cosmetic_only"])

    def test_reorder_is_non_material_auto(self):
        r = AP.classify(
            old_role_schemas={"loan_extract": ["a", "b", "c"]},
            new_role_schemas={"loan_extract": ["c", "b", "a"]})
        self.assertTrue(r.auto_approvable)

    def test_additive_optional_is_non_material_auto(self):
        r = AP.classify(
            old_role_schemas={"loan_extract": ["a", "b", "c"]},
            new_role_schemas={"loan_extract": ["a", "b", "c", "d"]})
        self.assertTrue(r.auto_approvable)
        self.assertEqual(r.evidence["added_columns"], ["loan_extract:d"])

    def test_removed_column_is_material(self):
        r = AP.classify(
            old_role_schemas={"loan_extract": ["a", "b", "c"]},
            new_role_schemas={"loan_extract": ["a", "b"]})
        self.assertTrue(r.material)
        self.assertFalse(r.auto_approvable)

    def test_new_role_is_material(self):
        r = AP.classify(
            old_role_schemas={"loan_extract": ["a", "b"]},
            new_role_schemas={"loan_extract": ["a", "b"], "cashflow_extract": ["x"]})
        self.assertTrue(r.material)

    def test_low_llm_conf_below_threshold_is_material(self):
        # Not structurally cosmetic (a column removed) AND weak evidence → material.
        r = AP.classify(
            old_role_schemas={"loan_extract": ["a", "b", "c"]},
            new_role_schemas={"loan_extract": ["a", "b"]},
            llm_conf=0.5)
        self.assertTrue(r.material)

    def test_thresholds_config_driven(self):
        t = AP.load_thresholds({"TRAKT_APPROVAL_AUTO_LLM_CONF": "0.80"})
        self.assertEqual(t.llm_conf, 0.80)


# --------------------------------------------------------------------------- #
# LLM mode policy
# --------------------------------------------------------------------------- #

class TestLLMModePolicy(unittest.TestCase):

    def test_resolving_mode_enables_mapping_resolver(self):
        from apps.blob_trigger_app.llm_recommendations import resolve_llm_policy
        pol = resolve_llm_policy({"TRAKT_LLM_ENABLED": "true", "TRAKT_LLM_MODE": "resolving",
                                  "ANTHROPIC_API_KEY": "k"})
        self.assertTrue(pol["enabled"])
        self.assertTrue(pol["resolve_mapping"])

    def test_advisory_mode_does_not_enable_resolver(self):
        from apps.blob_trigger_app.llm_recommendations import resolve_llm_policy
        pol = resolve_llm_policy({"TRAKT_LLM_ENABLED": "true", "TRAKT_LLM_MODE": "advisory",
                                  "ANTHROPIC_API_KEY": "k"})
        self.assertFalse(pol["resolve_mapping"])

    def test_orchestrator_invoke_forwards_resolver_flag(self):
        # The mapping resolver is wired for a new/changed source when resolving.
        import apps.blob_trigger_app.orchestrator_invoke as OI
        from unittest import mock
        captured = {}

        class _FakeState:
            run_id = "r"; status = "done"; central_canonical_path = None
            blockers = []
            def state_path(self):
                return "/tmp/x/run_state.json"

        def _fake_adapters(**kw):
            captured.update(kw)
            return object()

        with mock.patch.object(OI, "resolve_llm_policy" if hasattr(OI, "resolve_llm_policy") else "_noop", create=True):
            pass
        with mock.patch("apps.blob_trigger_app.llm_recommendations.resolve_llm_policy",
                        return_value={"enabled": True, "resolve_mapping": True,
                                      "available": True, "model": "claude-haiku-4-5-20251001"}), \
             mock.patch("engine.orchestrator_agent.adapters.RealAgentAdapters", _fake_adapters), \
             mock.patch("engine.orchestrator_agent.run_orchestration", return_value=_FakeState()), \
             mock.patch("engine.orchestrator_agent.orchestrator.onboarding_mode_for_target",
                        return_value="mi_only"):
            OI.default_orchestrator_invoker(
                processing_mode="source_onboarding", client_id="ERE",
                source_portfolio_id="direct_001", source_portfolio_type="direct",
                dataset="funded", frequency="monthly", reporting_period="2025-11-30",
                input_path="/tmp/in", target="mi", run_regime=False,
                mapping_config_path=None, out_dir="/tmp/out")
        self.assertTrue(captured.get("enable_llm_mapping_review"))


# --------------------------------------------------------------------------- #
# Router integration (a)–(d)
# --------------------------------------------------------------------------- #

class TestRouterApprovalPolicy(unittest.TestCase):

    def _ctx(self, td):
        storage = Storage(Path(td))
        layout = Layout()
        persistence = ProductionPersistence(storage, layout)
        registry = SourceRegistry("blob://trakt-state/registry/source_registry.yaml",
                                  storage=storage)
        return storage, layout, persistence, registry

    def _route(self, registry, persistence, out_dir, marker, data_files, invoker):
        role_schemas = R.role_schemas_for_pack(registry, marker, _CONTAINER)
        aliases = R.aliases_for_pack(registry, marker, _CONTAINER)
        schema = fingerprint_pack(data_files, role_schemas=role_schemas, aliases=aliases)
        return R.handle_blob_event(
            marker, registry=registry, out_dir=out_dir, container=_CONTAINER,
            pack_marker="_READY.json", schema_info=schema,
            input_dir_override=str(Path(data_files[0]).parent),
            orchestrator_invoker=invoker, assembler_refresher=_stub_assembler,
            persistence=persistence, now=_NOW)

    def _pin(self, registry, pack):
        repin_source(registry, client_id="ERE", source_portfolio_id="direct_001",
                     dataset="pipeline", frequency="weekly", data_files=pack,
                     source_book_type="direct", regime_required=False)

    def test_a_new_source_pending_review(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, persistence, registry = self._ctx(td)
            pack = _write_pack(Path(td) / "wk", {"PipelineExtract.csv": _LOAN})
            marker = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W02/_READY.json"
            m = self._route(registry, persistence, td, marker, pack, _Invoker())
            self.assertEqual(m["decision"], R.DECISION_SOURCE_ONBOARDING)
            self.assertEqual(m["status"], R.STATUS_PENDING_REVIEW)
            self.assertEqual(m["event_decision"], R.EVT_NEW_SOURCE_PENDING)
            self.assertIsNotNone(m.get("approval_id"))  # one-click artifact present

    def test_d_exact_schema_deterministic_no_classify(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, persistence, registry = self._ctx(td)
            pack = _write_pack(Path(td) / "wk", {"PipelineExtract.csv": _LOAN})
            self._pin(registry, pack)
            marker = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W03/_READY.json"
            m = self._route(registry, persistence, td, marker, pack, _Invoker())
            self.assertEqual(m["decision"], R.DECISION_DETERMINISTIC)
            self.assertEqual(m["status"], R.STATUS_PROCESSED)
            self.assertNotIn("materiality", m)         # exact match never classifies
            self.assertFalse(m.get("auto_approved"))

    def test_b_cosmetic_change_auto_approve_repin_governance(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, persistence, registry = self._ctx(td)
            old = _write_pack(Path(td) / "wk1", {"PipelineExtract.csv": _LOAN})
            self._pin(registry, old)
            old_fp = registry.lookup("ERE", "direct_001", "pipeline", "weekly").expected_schema_fingerprint

            # Cosmetic header text change (same normalised headers) → different
            # fingerprint but header-first role match.
            new = _write_pack(Path(td) / "wk2", {
                "PipelineExtract.csv": ["Loan ID", "Balance", "Rate",
                                        "Origination Date", "Maturity Date"]})
            marker = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W04/_READY.json"
            m = self._route(registry, persistence, td, marker, new, _Invoker())

            self.assertEqual(m["decision"], R.DECISION_DETERMINISTIC)
            self.assertTrue(m.get("auto_approved"))
            self.assertEqual(m["event_decision"], R.EVT_AUTO_APPROVED)
            self.assertEqual(m["status"], R.STATUS_PROCESSED)
            # Registry re-pinned to the new fingerprint.
            rec = registry.lookup("ERE", "direct_001", "pipeline", "weekly")
            self.assertEqual(rec.expected_schema_fingerprint, m["schema_fingerprint"])
            self.assertNotEqual(rec.expected_schema_fingerprint, old_fp)
            self.assertEqual(rec.status, STATUS_ACTIVE)
            # Governance artifact written durably with old→new fingerprint evidence.
            gov = persistence.load_governance_artifact(m["pack_key"])
            self.assertIsNotNone(gov)
            self.assertEqual(gov["old_fingerprint"], old_fp)
            self.assertEqual(gov["new_fingerprint"], m["schema_fingerprint"])
            self.assertIn("materiality_evidence", gov)

            # The NEXT identical upload is now a clean exact-match deterministic run.
            marker2 = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W05/_READY.json"
            m2 = self._route(registry, persistence, td, marker2, new, _Invoker())
            self.assertEqual(m2["decision"], R.DECISION_DETERMINISTIC)
            self.assertFalse(m2.get("auto_approved"))

    def test_c_material_change_pending_review(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, persistence, registry = self._ctx(td)
            old = _write_pack(Path(td) / "wk1", {"PipelineExtract.csv": _LOAN})
            self._pin(registry, old)
            # Drop a mandatory column (still header-matches the role: Jaccard 4/5).
            new = _write_pack(Path(td) / "wk2", {
                "PipelineExtract.csv": ["loan_id", "balance", "rate", "origination_date"]})
            marker = "raw-v2/ERE/direct/pipeline/weekly/direct_001/2026-W04/_READY.json"
            m = self._route(registry, persistence, td, marker, new, _Invoker())
            self.assertEqual(m["decision"], R.DECISION_SOURCE_ONBOARDING)
            self.assertEqual(m["status"], R.STATUS_PENDING_REVIEW)
            self.assertEqual(m["event_decision"], R.EVT_MATERIAL_CHANGE_PENDING)
            self.assertTrue(m.get("material_change"))
            self.assertTrue(m["materiality"]["material"])


if __name__ == "__main__":
    unittest.main()
