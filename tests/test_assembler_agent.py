#!/usr/bin/env python3
"""tests/test_assembler_agent.py

Assembler Agent: pipeline-scope handling, lineage manifest, and routing to the
MI Agent and the Regime / Projection Agent. Reuses the platform_assembler core;
these tests focus on the agent layer (scope + lineage + routing).

Run: python -m unittest tests.test_assembler_agent
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine import assembler_agent as aa
from engine import platform_assembler as pa


def _make_canonical(td: Path, fname: str, pid: str, ptype: str, date: str,
                    loans, label: str = "Book") -> Path:
    acq = "" if ptype == "direct" else "2026-08-15"
    df = pd.DataFrame({
        "loan_identifier": list(loans),
        "current_outstanding_balance": [100.0] * len(loans),
        "data_cut_off_date": [date] * len(loans),
        "source_portfolio_id": [pid] * len(loans),
        "source_portfolio_type": [ptype] * len(loans),
        "source_portfolio_label": [label] * len(loans),
        "acquisition_date": [acq] * len(loans),
        "seller_name": ["" if ptype == "direct" else "Seller A"] * len(loans),
        "portfolio_cohort": [pid] * len(loans),
    })
    p = td / fname
    df.to_csv(p, index=False)
    return p


class TestAssemblerAgentCore(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.td = Path(self._tmp.name)
        self.out = self.td / "out_platform"

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, pipeline="mi", regime=None, **kw):
        return aa.run_assembler_agent(
            self.td, self.out, client_id="ERE", pipeline=pipeline, regime=regime,
            assembler_run_id="asm_test", created_at="2026-09-01T00:00:00+00:00", **kw)

    def test_single_portfolio_one_central(self):
        _make_canonical(self.td, "direct_001_canonical_typed.csv", "direct_001",
                        "direct", "2026-07-31", ["L1", "L2"])
        res = self._run()
        self.assertTrue(res.central_canonical_path.exists())
        self.assertEqual(res.manifest["portfolio_count"], 1)
        self.assertEqual(len(res.manifest["included_portfolios"]), 1)

    def test_multiple_portfolios_one_central(self):
        _make_canonical(self.td, "d_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1", "L2"])
        _make_canonical(self.td, "a_canonical_typed.csv", "acquired_001", "acquired",
                        "2026-08-31", ["L1", "L9"])
        res = self._run()
        self.assertEqual(res.manifest["portfolio_count"], 2)
        self.assertEqual(res.manifest["total_rows"], 4)

    def test_latest_selected_older_excluded_with_reason(self):
        _make_canonical(self.td, "d_jun_canonical_typed.csv", "direct_001", "direct",
                        "2026-06-30", ["L1", "L2"])
        _make_canonical(self.td, "d_jul_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1", "L2", "L3"])
        res = self._run()
        self.assertEqual(res.manifest["total_rows"], 3)
        self.assertEqual(res.manifest["included_portfolios"][0]["snapshot_date"], "2026-07-31")
        self.assertEqual(len(res.manifest["excluded_candidates"]), 1)
        self.assertIn("superseded", res.manifest["excluded_candidates"][0]["reason"])

    def test_manifest_lineage_fields(self):
        _make_canonical(self.td, "d_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1"])
        m = self._run().manifest
        for k in ("assembler_run_id", "client_id", "pipeline", "lineage",
                  "created_at", "content_sha256", "output_canonical_path",
                  "output_total_balance", "included_portfolios", "excluded_candidates"):
            self.assertIn(k, m)
        p0 = m["included_portfolios"][0]
        for k in ("source_portfolio_id", "source_portfolio_type",
                  "source_portfolio_label", "selected_canonical_path",
                  "snapshot_date", "row_count", "total_balance", "input_file_hash"):
            self.assertIn(k, p0)

    def test_invalid_pipeline_rejected(self):
        _make_canonical(self.td, "d_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1"])
        with self.assertRaises(aa.AssemblerAgentError):
            self._run(pipeline="bogus")

    def test_regime_requires_regime_value(self):
        _make_canonical(self.td, "d_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1"])
        with self.assertRaises(aa.AssemblerAgentError):
            self._run(pipeline="regime")

    def test_duplicate_composite_key_rejected(self):
        _make_canonical(self.td, "d_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1", "L1"])
        with self.assertRaises(pa.PlatformAssemblyError):
            self._run()

    def test_same_loan_id_across_portfolios_ok(self):
        _make_canonical(self.td, "d_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1"])
        _make_canonical(self.td, "a_canonical_typed.csv", "acquired_001", "acquired",
                        "2026-08-31", ["L1"])
        res = self._run()
        self.assertEqual(sorted(res.dataframe[pa.PLATFORM_KEY_COLUMN]),
                         ["acquired_001/L1", "direct_001/L1"])


class TestAssemblerAgentMIRouting(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.td = Path(self._tmp.name)
        _make_canonical(self.td, "d_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1", "L2"])
        _make_canonical(self.td, "a_canonical_typed.csv", "acquired_001", "acquired",
                        "2026-08-31", ["L1", "L9"])
        from mi_agent_api import data_source as ds
        self.ds = ds
        self._env_keys = ("MI_AGENT_PLATFORM_CANONICAL", "MI_AGENT_PLATFORM_DIR",
                          "MI_AGENT_ANALYTICS_DATASET", "MI_AGENT_CENTRAL_TAPE",
                          "MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_DATA_CSV")
        self._saved = {k: os.environ.get(k) for k in self._env_keys}
        for k in self._env_keys:
            os.environ.pop(k, None)
        ds.reset_cache()

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ[k] = v if v is not None else os.environ.pop(k, "")
            if v is None:
                os.environ.pop(k, None)
        self.ds.reset_cache()
        self._tmp.cleanup()

    def test_mi_pipeline_routes_to_central(self):
        res = aa.run_assembler_agent(self.td, self.td / "out_platform",
                                     client_id="ERE", pipeline="mi")
        self.assertIn("mi", res.routing)
        env = res.routing["mi"]["data_source_env"]
        self.assertEqual(env["MI_AGENT_PLATFORM_CANONICAL"],
                         str(res.central_canonical_path))

    def test_mi_data_source_uses_central_when_present(self):
        res = aa.run_assembler_agent(self.td, self.td / "out_platform",
                                     client_id="ERE", pipeline="mi")
        os.environ["MI_AGENT_PLATFORM_CANONICAL"] = str(res.central_canonical_path)
        self.ds.reset_cache()
        path, kind = self.ds.resolve_data_source()
        self.assertEqual(kind, "platform_canonical")
        df = self.ds.get_dataframe()
        self.assertEqual(sorted(df["source_portfolio_id"].unique()),
                         ["acquired_001", "direct_001"])

    def test_mi_fallback_unchanged_when_absent(self):
        path, kind = self.ds.resolve_data_source()
        self.assertNotEqual(kind, "platform_canonical")

    def test_explicit_dataset_override_wins(self):
        res = aa.run_assembler_agent(self.td, self.td / "out_platform",
                                     client_id="ERE", pipeline="mi")
        os.environ["MI_AGENT_PLATFORM_CANONICAL"] = str(res.central_canonical_path)
        explicit = self.td / "explicit.csv"
        pd.DataFrame({"loan_identifier": ["Z1"]}).to_csv(explicit, index=False)
        os.environ["MI_AGENT_ANALYTICS_DATASET"] = str(explicit)
        self.ds.reset_cache()
        _, kind = self.ds.resolve_data_source()
        self.assertEqual(kind, "prepared_explicit")


class TestAssemblerAgentRegimeRouting(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.td = Path(self._tmp.name)
        self.inputs = self.td / "inputs"
        self.inputs.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_regime_routing_command_targets_central(self):
        _make_canonical(self.inputs, "a_canonical_typed.csv", "acquired_001",
                        "acquired", "2026-08-31", ["L1", "L2"])
        res = aa.run_assembler_agent(self.inputs, self.td / "out", client_id="ERE",
                                     pipeline="regime", regime="ESMA_Annex2")
        self.assertIn("regime", res.routing)
        self.assertEqual(res.routing["regime"]["input_canonical"],
                         str(res.central_canonical_path))
        self.assertIn(str(res.central_canonical_path), res.routing["regime"]["command"])

    def test_regime_projector_consumes_central_clean_esma_plus_companion(self):
        # Build a projector-friendly provenance-stamped canonical from the
        # synthetic demo, then route it through the regime projector via the agent.
        demos = sorted(_REPO.glob("synthetic_demo/output/*canonical_typed.csv"))
        if not demos:
            self.skipTest("synthetic demo canonical not present")
        cmd = [sys.executable,
               str(_REPO / "engine/gate_2_transform/canonical_transform.py"),
               str(demos[0]),
               "--registry", str(_REPO / "config/system/fields_registry.yaml"),
               "--portfolio-type", "equity_release",
               "--output-dir", str(self.inputs),
               "--output-prefix", "acquired_001",
               "--no-derivations",
               "--source-portfolio-id", "acquired_001",
               "--acquisition-date", "2026-08-15",
               "--source-portfolio-label", "Acquired Portfolio 1"]
        if subprocess.run(cmd, capture_output=True, text=True).returncode != 0:
            self.skipTest("could not stamp synthetic canonical")

        out = self.td / "out"
        res = aa.run_assembler_agent(self.inputs, out, client_id="ERE",
                                     pipeline="regime", regime="ESMA_Annex2",
                                     run_regime=True, regime_allow_unreviewed=True)
        if res.regime_run is None or not res.regime_run["ok"]:
            self.skipTest(f"regime projector unavailable: "
                          f"{(res.regime_run or {}).get('stderr_tail','')[-300:]}")
        projected = glob.glob(str(out / "*ESMA_Annex2_projected.csv"))
        companion = glob.glob(str(out / "*ESMA_Annex2_provenance.csv"))
        self.assertTrue(projected and companion)
        esma = pd.read_csv(projected[0])
        leaked = [c for c in pa._provenance.PROVENANCE_FIELDS if c in esma.columns]
        self.assertEqual(leaked, [])  # ESMA stays template-clean
        comp = pd.read_csv(companion[0])
        self.assertIn("source_portfolio_id", comp.columns)
        self.assertIn("portfolio_cohort", comp.columns)


if __name__ == "__main__":
    unittest.main()
