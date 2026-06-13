#!/usr/bin/env python3
"""tests/test_onboarding_mapping_trace.py — PART 9 (16, 17 + trace structure)."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from onboarding_domain_fixtures import SCENARIO_A, build_run
from engine.onboarding_agent import mapping_trace

_EXPECTED_COLUMNS = set(mapping_trace._TRACE_COLUMNS)


class TestMappingTraceArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project, cls.pdir, cls.rp = build_run(SCENARIO_A, ingest=False)

    # 16. 05c_mapping_trace.csv / json written.
    def test_trace_csv_and_json_written(self):
        self.assertTrue((self.pdir / "05c_mapping_trace.csv").exists())
        self.assertTrue((self.pdir / "05c_mapping_trace.json").exists())

    # 17. 05d_mapping_explanation.md written.
    def test_explanation_md_written(self):
        md = self.pdir / "05d_mapping_explanation.md"
        self.assertTrue(md.exists())
        text = md.read_text(encoding="utf-8")
        self.assertIn("Deterministic-first", text)
        self.assertIn("alias", text.lower())

    def test_trace_records_loaded_alias_files(self):
        data = json.loads((self.pdir / "05c_mapping_trace.json").read_text())
        loaded = data["summary"]["alias_files_loaded"]
        self.assertIn("aliases_mandatory.yaml", loaded)
        self.assertIn("aliases_analytics.yaml", loaded)

    def test_trace_rows_have_required_fields(self):
        with (self.pdir / "05c_mapping_trace.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(set(r.keys()), _EXPECTED_COLUMNS)
            self.assertTrue(r["final_status"])
            self.assertTrue(r["source_column"])
            # Every row records the alias libraries that were loaded.
            self.assertIn("aliases_mandatory.yaml", r["alias_files_loaded"])

    def test_summary_counts_consistent(self):
        s = self.project.mapping_trace_summary
        total = (s["mapped_by_alias"] + s["mapped_by_registry_header"]
                 + s["mapped_by_value_or_context"] + s["out_of_scope"]
                 + s["ambiguous_needs_review"] + s["unmapped"])
        self.assertEqual(total, s["columns_total"])


if __name__ == "__main__":
    unittest.main()
