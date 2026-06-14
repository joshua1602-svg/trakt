#!/usr/bin/env python3
"""tests/test_onboarding_demo_v1.py — PART 14 (15–21).

End-to-end demo: runs the v1 onboarding story on synthetic scenario A and
asserts it reads files, detects domains, generates+answers gaps, saves client
memory, builds consolidated tapes, produces the Azure-ready trigger, and that a
memory-applied rerun has fewer-or-equal unresolved mapping gaps.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent import demo_onboarding_v1 as demo


class TestDemoEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="demo_v1_"))
        cls.out = cls.tmp / "demo_onboarding_v1"
        cls.result = demo.run_demo(str(cls.out), client_id="demo_client",
                                   run_id="demo_run_001")

    # 15. Demo script runs end-to-end on synthetic scenario A.
    def test_runs_end_to_end(self):
        r = self.result
        self.assertEqual(r["input_files"], 4)
        for dom in ("loan", "borrower", "collateral", "cashflow", "pipeline"):
            self.assertIn(dom, r["domains_detected"])
        self.assertEqual(r["blocking_after"], 0)
        self.assertLess(r["blocking_after"], r["blocking_before"])

    # 16. Demo produces the central lender tape.
    def test_central_lender_tape(self):
        tape = self.out / "output" / "central" / "18_central_lender_tape.csv"
        self.assertTrue(tape.exists())
        self.assertGreater(self.result["lender_tape_rows"], 0)

    # 17. Demo produces the central pipeline tape.
    def test_central_pipeline_tape(self):
        tape = self.out / "output" / "central" / "18a_central_pipeline_tape.csv"
        self.assertTrue(tape.exists())
        self.assertGreater(self.result["pipeline_rows"], 0)

    # 18. Demo produces the pipeline trigger JSON.
    def test_pipeline_trigger(self):
        trigger = Path(self.result["pipeline_trigger_path"])
        self.assertTrue(trigger.exists())
        data = json.loads(trigger.read_text(encoding="utf-8"))
        self.assertEqual(data["client_id"], "demo_client")
        self.assertIn("event_type", data)

    # 19. Demo saves client memory.
    def test_client_memory_saved(self):
        self.assertEqual(self.result["client_memory_entries_saved"], 4)
        mem = self.out.parent / "demo_client" / "client_memory" / "mapping_memory.yaml"
        self.assertTrue(mem.exists())

    # 20. Second/reapplied run has fewer or equal unresolved mapping gaps.
    def test_rerun_fewer_or_equal_mapping_gaps(self):
        self.assertLessEqual(self.result["mapping_gaps_after"],
                             self.result["mapping_gaps_before"])
        # The applied memory is reported on the rerun.
        summary = self.result["memory_applied_summary"]
        self.assertTrue(summary.get("client_mapping_memory_loaded"))

    def test_ready_for_handoff(self):
        self.assertEqual(self.result["readiness_status"], "ready_for_pipeline")
        self.assertTrue(self.result["ready_for_mi"])


class TestDemoDocs(unittest.TestCase):
    # 21. Demo README exists and includes commands.
    def test_readme_exists_with_commands(self):
        readme = _REPO_ROOT / "docs" / "onboarding_v1_demo.md"
        self.assertTrue(readme.exists())
        text = readme.read_text(encoding="utf-8")
        self.assertIn("demo_onboarding_v1", text)
        self.assertIn("streamlit run engine/onboarding_agent/streamlit_onboarding_workbench.py", text)


if __name__ == "__main__":
    unittest.main()
