#!/usr/bin/env python3
"""tests/test_onboarding_workbench_smoke.py — PART 14 (1–5).

Smoke tests for the Streamlit review workbench. They never import or run
Streamlit — all tested behaviour lives in plain, importable functions.
"""

from __future__ import annotations

import importlib
import json
import sys
import unittest
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from onboarding_domain_fixtures import SCENARIO_A, build_run


class TestWorkbenchSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A real run dir to load.
        cls.project, cls.pdir, cls.rp = build_run(SCENARIO_A, mode="regulatory_mi", ingest=False)

    # 1. Workbench module imports without running Streamlit side effects.
    def test_imports_without_streamlit(self):
        # Ensure streamlit is genuinely absent during import.
        self.assertNotIn("streamlit", sys.modules)
        mod = importlib.import_module(
            "engine.onboarding_agent.streamlit_onboarding_workbench"
        )
        self.assertTrue(hasattr(mod, "load_project"))
        self.assertTrue(hasattr(mod, "main"))
        # Importing must not have pulled in streamlit.
        self.assertNotIn("streamlit", sys.modules)

    # 2. Workbench can load a project directory artefact set.
    def test_load_project(self):
        from engine.onboarding_agent import streamlit_onboarding_workbench as wb
        ctx = wb.load_project(self.pdir, client_id="client_x", run_id="run_001",
                              mode="regulatory_mi")
        self.assertTrue(ctx.inventory)
        self.assertTrue(ctx.mapping_candidates)
        self.assertTrue(ctx.gap_questions)
        ov = wb.run_overview(ctx)
        self.assertEqual(ov["input_files"], len(ctx.inventory))
        self.assertIn(ov["readiness_status"], ("Ready", "Needs review", "Blocked"))
        self.assertTrue(wb.domain_rows(ctx))
        self.assertTrue(wb.mapping_rows(ctx))
        # In-scope canonical fields for the dropdown are mode-aware.
        fields = wb.in_scope_canonical_fields(ctx.registry_path, "mi_only", False)
        self.assertIn("current_principal_balance", fields)

    # 3. Workbench decision serializer writes 24_workbench_pending_decisions.yaml.
    def test_write_pending_decisions(self):
        from engine.onboarding_agent import streamlit_onboarding_workbench as wb
        path = wb.write_pending_decisions(
            self.pdir, {"client_id": "client_x", "gap_answers": {"Q1": {"answer": "x"}}}
        )
        self.assertEqual(path.name, "24_workbench_pending_decisions.yaml")
        self.assertTrue(path.exists())
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(data["gap_answers"]["Q1"]["answer"], "x")

    # 4. Workbench answers generator writes 25_workbench_answers.yaml (ingestible).
    def test_generate_answers_yaml(self):
        from engine.onboarding_agent import streamlit_onboarding_workbench as wb
        ctx = wb.load_project(self.pdir, client_id="client_x", run_id="run_001")
        answers = wb.answers_from_decisions(ctx, {"gap_answers": {}})
        path = wb.generate_answers_yaml(self.pdir, answers, project_id="client_x")
        self.assertEqual(path.name, "25_workbench_answers.yaml")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Compatible shape: {project_id, answers: {qid: {answer, ...}}}
        self.assertIn("answers", data)
        self.assertTrue(data["answers"])
        first = next(iter(data["answers"].values()))
        self.assertIn("answer", first)

    # 5. Action log appends to 26_workbench_action_log.json.
    def test_action_log_appends(self):
        from engine.onboarding_agent import streamlit_onboarding_workbench as wb
        log_path = self.pdir / "26_workbench_action_log.json"
        if log_path.exists():
            log_path.unlink()
        wb.append_action_log(self.pdir, "client_x", "run_001", "action_one",
                             outputs_written=["a"])
        wb.append_action_log(self.pdir, "client_x", "run_001", "action_two",
                             outputs_written=["b"], status="ok")
        log = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(len(log), 2)
        self.assertEqual(log[0]["action"], "action_one")
        self.assertEqual(log[1]["action"], "action_two")
        for entry in log:
            for key in ("timestamp", "client_id", "run_id", "action", "inputs",
                        "outputs_written", "status"):
                self.assertIn(key, entry)


if __name__ == "__main__":
    unittest.main()
