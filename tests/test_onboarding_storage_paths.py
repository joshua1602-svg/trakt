#!/usr/bin/env python3
"""tests/test_onboarding_storage_paths.py — PART 15 (1–3)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import storage_paths


class TestStoragePaths(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sp_"))

    # 1. Creates the expected local output folders.
    def test_creates_local_output_folders(self):
        rp = storage_paths.resolve_run_paths(
            project_dir=str(self.tmp / "proj"),
            input_dir=str(self.tmp / "proj" / "input" / "uploaded"),
            client_id="client_x", run_id="run_001",
        )
        for d in (rp.central_dir, rp.lineage_dir, rp.gaps_dir, rp.manifests_dir,
                  rp.logs_dir, rp.working_dir):
            self.assertTrue(Path(d).is_dir(), f"missing {d}")
        self.assertTrue(Path(rp.output_root).is_dir())

    # 2. Azure-compatible manifest URIs when input/output URIs are supplied.
    def test_azure_manifest_uris(self):
        rp = storage_paths.resolve_run_paths(
            project_dir=str(self.tmp / "proj"),
            output_root=str(self.tmp / "proj" / "output"),
            client_id="client_x", run_id="run_001",
            storage_backend="azure_blob_compatible",
            input_uri="azure://c/clients/client_x/onboarding/run_001/input/uploaded/",
            output_uri="azure://c/clients/client_x/onboarding/run_001/output/",
        )
        tape = Path(rp.central_dir) / "18_central_lender_tape.csv"
        uri = rp.to_manifest_uri(tape)
        self.assertEqual(
            uri, "azure://c/clients/client_x/onboarding/run_001/output/central/18_central_lender_tape.csv"
        )
        # Local manifest path stays a project-relative POSIX path.
        self.assertEqual(rp.to_manifest_path(tape), "output/central/18_central_lender_tape.csv")

    def test_manifest_uri_null_without_azure(self):
        rp = storage_paths.resolve_run_paths(project_dir=str(self.tmp / "proj"))
        tape = Path(rp.central_dir) / "18_central_lender_tape.csv"
        self.assertIsNone(rp.to_manifest_uri(tape))

    # 3. Storage guard prevents writing outside project/output root.
    def test_guard_blocks_outside_writes(self):
        rp = storage_paths.resolve_run_paths(project_dir=str(self.tmp / "proj"))
        with self.assertRaises(ValueError):
            rp.guard("/etc/passwd")
        with self.assertRaises(ValueError):
            storage_paths.assert_within_project(self.tmp / "elsewhere" / "x.csv", rp.project_dir)
        # Inside is fine.
        inside = Path(rp.central_dir) / "ok.csv"
        self.assertEqual(rp.guard(inside), inside.resolve())


if __name__ == "__main__":
    unittest.main()
