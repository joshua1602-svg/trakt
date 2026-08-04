#!/usr/bin/env python3
"""tests/test_simulation_regression_fixtures.py — the pinned economics.

A handful of simulation cases have their MANIFEST and their independently
computed EXPECTED TRUTH committed under ``tests/fixtures/simulation/``. These
tests regenerate each case from its committed manifest and assert the expected
truth is byte-identical.

That is what makes a generated case a permanent regression fixture: if a change
to the generator, the knobs or an asset profile silently alters a case's
economics, this fails immediately and names the case and its seed — without
committing a single generated CSV.

Refresh deliberately, never casually:

    python -m simulation.tools.build_regression_fixtures

Run: python -m pytest tests/test_simulation_regression_fixtures.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from simulation import manifests as M  # noqa: E402
from simulation.history import generate_history  # noqa: E402
from simulation.reference_truth import build_expected_truth  # noqa: E402
from simulation.tools.build_regression_fixtures import FIXTURE_ROOT, PINNED  # noqa: E402


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class TestRegressionFixtures(unittest.TestCase):
    def setUp(self):
        self.index = {entry["case_id"]: entry
                      for entry in _load(FIXTURE_ROOT / "index.json")}

    def test_the_index_covers_every_pinned_case(self):
        self.assertEqual(set(self.index), set(PINNED))

    def test_a_clean_case_is_pinned_for_every_asset_family(self):
        families = {self.index[c]["asset_class"] for c in self.index
                    if c.endswith("clean_origination_v1")
                    or "clean_" in c}
        self.assertEqual(families, {"equity_release", "bridge", "asset_finance"})

    def test_a_schema_drift_case_and_both_guards_are_pinned(self):
        self.assertIn("bridge_maturity_wall_v1", self.index)
        self.assertIn("bridge_unsupported_schema_guard_v1", self.index)
        self.assertIn("equity_release_missing_mandatory_guard_v1", self.index)

    def test_the_committed_manifest_matches_the_catalogue(self):
        for case_id in PINNED:
            committed = M.read_manifest(FIXTURE_ROOT / case_id / "manifest.json")
            self.assertEqual(committed, M.get_case(case_id), case_id)

    def test_the_manifest_hash_is_unchanged(self):
        for case_id, entry in self.index.items():
            self.assertEqual(M.get_case(case_id).content_hash(),
                             entry["manifest_hash"],
                             f"{case_id}: the manifest changed — regenerate the "
                             f"fixtures deliberately if that was intended")

    def test_the_economics_reproduce_exactly_from_the_committed_manifest(self):
        for case_id in PINNED:
            with self.subTest(case=case_id):
                case = M.read_manifest(FIXTURE_ROOT / case_id / "manifest.json")
                regenerated = build_expected_truth(case, generate_history(case))
                committed = _load(FIXTURE_ROOT / case_id / "expected_truth.json")
                self.assertEqual(
                    json.dumps(regenerated, sort_keys=True, default=str),
                    json.dumps(committed, sort_keys=True, default=str),
                    f"{case_id} (seed {case.seed}) no longer reproduces its "
                    f"committed economics. Reproduce with: python -m "
                    f"simulation.runner reproduce --case {case_id} "
                    f"--seed {case.seed}")

    def test_the_fixtures_stay_small(self):
        """The framework does not commit bulk generated data."""
        total = sum(p.stat().st_size for p in FIXTURE_ROOT.rglob("*") if p.is_file())
        self.assertLess(total, 2 * 1024 * 1024,
                        f"simulation fixtures grew to {total} bytes")
        self.assertEqual([p for p in FIXTURE_ROOT.rglob("*.csv")], [])
        self.assertEqual([p for p in FIXTURE_ROOT.rglob("*.xlsx")], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
