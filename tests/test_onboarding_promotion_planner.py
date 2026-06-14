#!/usr/bin/env python3
"""tests/test_onboarding_promotion_planner.py — PART 15 (18–24)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml

from onboarding_domain_fixtures import REGISTRY, SCENARIO_A, build_run
from engine.onboarding_agent import central_tape_builder, promotion_planner


def _promote(mode="regulatory_mi", reg=False, storage_backend="local",
             input_uri="", output_uri=""):
    project, pdir, rp = build_run(
        SCENARIO_A, mode=mode, ingest=True,
        storage_backend=storage_backend, input_uri=input_uri, output_uri=output_uri,
        regulatory_reporting_enabled=reg,
    )
    res = central_tape_builder.build_central_tapes(
        pdir, rp, str(REGISTRY), mode=mode, regulatory_reporting_enabled=reg
    )
    plan = promotion_planner.build_promotion_plan(
        pdir, rp, res, project.domain_coverage, mode, reg,
        client_name="CLIENT_X", project_id="client_x",
    )
    return project, pdir, rp, res, plan


class TestPromotionArtefacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project, cls.pdir, cls.rp, cls.res, cls.plan = _promote()
        cls.manifests = Path(cls.rp.manifests_dir)

    # 19. Promotion plan is written.
    def test_promotion_plan_written(self):
        self.assertTrue((self.manifests / "19_promotion_plan.yaml").exists())

    # 20. Handoff manifest is written.
    def test_handoff_manifest_written(self):
        self.assertTrue((self.manifests / "20_pipeline_handoff_manifest.yaml").exists())

    # 21. Readiness JSON is written.
    def test_readiness_written(self):
        p = self.manifests / "21_pipeline_handoff_readiness.json"
        self.assertTrue(p.exists())
        rj = json.loads(p.read_text())
        for key in ("ready_for_mi_agent", "ready_for_gate1_handoff",
                    "ready_for_regulatory_projection", "ready_for_warehouse_analysis",
                    "domain_coverage", "loan_count", "pipeline_count"):
            self.assertIn(key, rj)

    # 22. Pipeline trigger JSON is written.
    def test_trigger_written(self):
        p = self.manifests / "23_pipeline_trigger.json"
        self.assertTrue(p.exists())
        t = json.loads(p.read_text())
        self.assertIn(t["event_type"],
                      ("trakt.onboarding.handoff.ready", "trakt.onboarding.handoff.blocked"))

    # 24. Promote does not run Gates 1–5.
    def test_no_gates_run(self):
        manifest = yaml.safe_load(
            (self.manifests / "20_pipeline_handoff_manifest.yaml").read_text()
        )
        self.assertEqual(manifest["run_gates"], "none")
        self.assertTrue(manifest["dry_run_only"])
        # No Gate output directories were created under the run folder.
        for stray in ("gate_1_alignment", "gate_2_transform", "gate_outputs"):
            self.assertFalse((self.pdir / stray).exists())


class TestTriggerAzureURIs(unittest.TestCase):
    # 23. Pipeline trigger JSON contains Azure-compatible URIs when provided.
    def test_trigger_has_azure_uris(self):
        _, _, rp, _, plan = _promote(
            storage_backend="azure_blob_compatible",
            input_uri="azure://c/clients/client_x/onboarding/run_001/input/uploaded/",
            output_uri="azure://c/clients/client_x/onboarding/run_001/output/",
        )
        t = json.loads(Path(plan["pipeline_trigger_path"]).read_text())
        self.assertEqual(t["storage_backend"], "azure_blob_compatible")
        self.assertTrue(str(t["central_lender_tape_uri"]).startswith("azure://"))
        self.assertTrue(str(t["handoff_manifest_uri"]).startswith("azure://"))
        self.assertTrue(str(t["readiness_uri"]).startswith("azure://"))


class TestWarehouseMode(unittest.TestCase):
    # 18. Warehouse mode uses cashflow / warehouse domains for readiness.
    def test_warehouse_readiness(self):
        project, _, _, _, plan = _promote(mode="warehouse_securitisation")
        readiness = plan["readiness"]
        self.assertIn("ready_for_warehouse_analysis", readiness)
        cov = {d.domain: d.status for d in project.domain_coverage}
        # Warehouse + cashflow domains are assessed (not out-of-scope) in this mode.
        self.assertNotEqual(cov.get("warehouse_terms"), "out_of_scope")
        self.assertNotEqual(cov.get("cashflow"), "out_of_scope")


if __name__ == "__main__":
    unittest.main()
