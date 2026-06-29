#!/usr/bin/env python3
"""tests/test_platform_assembler.py

Platform Portfolio Assembler: latest-per-portfolio combination, composite
uniqueness, rejection paths, and MI Agent data-source resolution (prefer the
platform canonical when present; otherwise unchanged behaviour).

Run: python -m unittest tests.test_platform_assembler
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine import platform_assembler as pa


def _make_canonical(td: Path, fname: str, pid: str, ptype: str,
                    date: str, loans, label: str = "Book") -> Path:
    acq = "" if ptype == "direct" else "2026-08-15"
    seller = "" if ptype == "direct" else "Seller A"
    df = pd.DataFrame({
        "loan_identifier": list(loans),
        "current_principal_balance": [100.0] * len(loans),
        "data_cut_off_date": [date] * len(loans),
        "source_portfolio_id": [pid] * len(loans),
        "source_portfolio_type": [ptype] * len(loans),
        "source_portfolio_label": [label] * len(loans),
        "acquisition_date": [acq] * len(loans),
        "seller_name": [seller] * len(loans),
        "portfolio_cohort": [pid] * len(loans),
    })
    p = td / fname
    df.to_csv(p, index=False)
    return p


class TestPlatformAssembler(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.td = Path(self._tmp.name)
        self.out = self.td / "out_platform"

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_portfolio(self):
        _make_canonical(self.td, "d_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1", "L2"])
        res = pa.assemble_platform_canonical(self.td, self.out, write=True)
        self.assertEqual(res.manifest["portfolio_count"], 1)
        self.assertEqual(len(res.dataframe), 2)
        self.assertTrue(res.output_csv.exists())

    def test_multiple_portfolios(self):
        _make_canonical(self.td, "d_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1", "L2"])
        _make_canonical(self.td, "a1_canonical_typed.csv", "acquired_001", "acquired",
                        "2026-08-31", ["L1", "L9"])
        _make_canonical(self.td, "a2_canonical_typed.csv", "acquired_002", "acquired",
                        "2026-09-30", ["L5"])
        res = pa.assemble_platform_canonical(self.td, self.out, write=True)
        self.assertEqual(res.manifest["portfolio_count"], 3)
        self.assertEqual(len(res.dataframe), 5)
        self.assertEqual(sorted(res.dataframe["source_portfolio_id"].unique()),
                         ["acquired_001", "acquired_002", "direct_001"])

    def test_latest_snapshot_replaces_older(self):
        _make_canonical(self.td, "d_jun_canonical_typed.csv", "direct_001", "direct",
                        "2026-06-30", ["L1", "L2"])
        _make_canonical(self.td, "d_jul_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1", "L2", "L3"])
        res = pa.assemble_platform_canonical(self.td, self.out, write=True)
        # Only July (3 rows), never June.
        self.assertEqual(len(res.dataframe), 3)
        self.assertEqual(list(res.dataframe["data_cut_off_date"].unique()), ["2026-07-31"])
        self.assertEqual(res.manifest["portfolios"][0]["snapshot_date"], "2026-07-31")

    def test_duplicate_portfolio_snapshot_rejected(self):
        # Same portfolio, same latest date in two files -> ambiguous -> reject.
        _make_canonical(self.td, "d_a_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1", "L2"])
        _make_canonical(self.td, "d_b_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L3", "L4"])
        with self.assertRaises(pa.PlatformAssemblyError):
            pa.assemble_platform_canonical(self.td, self.out, write=True)

    def test_duplicate_composite_key_rejected(self):
        # A single portfolio file with a duplicated loan_identifier -> duplicate
        # composite key (source_portfolio_id + loan_identifier) -> reject.
        _make_canonical(self.td, "d_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1", "L1"])
        with self.assertRaises(pa.PlatformAssemblyError) as ctx:
            pa.assemble_platform_canonical(self.td, self.out, write=True)
        self.assertIn("composite", str(ctx.exception).lower())

    def test_same_loan_id_across_portfolios_not_a_collision(self):
        _make_canonical(self.td, "d_canonical_typed.csv", "direct_001", "direct",
                        "2026-07-31", ["L1"])
        _make_canonical(self.td, "a_canonical_typed.csv", "acquired_001", "acquired",
                        "2026-08-31", ["L1"])
        res = pa.assemble_platform_canonical(self.td, self.out, write=True)
        self.assertEqual(sorted(res.dataframe[pa.PLATFORM_KEY_COLUMN]),
                         ["acquired_001/L1", "direct_001/L1"])

    def test_provenance_preserved_unchanged(self):
        src = _make_canonical(self.td, "a_canonical_typed.csv", "acquired_001",
                              "acquired", "2026-08-31", ["L1", "L2"],
                              label="Acquired Portfolio 1")
        before = pd.read_csv(src)
        res = pa.assemble_platform_canonical(self.td, self.out, write=True)
        for f in pa.read_portfolio_snapshot.__globals__["_provenance"].PROVENANCE_FIELDS:
            self.assertIn(f, res.dataframe.columns)
        # Values identical to the source (not modified/dropped).
        self.assertEqual(list(res.dataframe["source_portfolio_label"]),
                         list(before["source_portfolio_label"]))
        self.assertEqual(list(res.dataframe["seller_name"].astype(str)),
                         list(before["seller_name"].astype(str)))

    def test_unstamped_canonical_rejected(self):
        df = pd.DataFrame({"loan_identifier": ["L1"], "current_principal_balance": [1.0]})
        (self.td / "x_canonical_typed.csv").write_text(df.to_csv(index=False))
        with self.assertRaises(pa.PlatformAssemblyError):
            pa.assemble_platform_canonical(self.td, self.out, write=True)

    def test_individual_canonicals_not_overwritten(self):
        src = _make_canonical(self.td, "d_canonical_typed.csv", "direct_001",
                              "direct", "2026-07-31", ["L1", "L2"])
        before = src.read_text()
        pa.assemble_platform_canonical(self.td, self.out, write=True)
        self.assertEqual(src.read_text(), before)  # untouched


class TestMIDataSourceResolution(unittest.TestCase):
    """MI Agent prefers the platform canonical when present; else unchanged."""

    _PLATFORM_ENVS = ("MI_AGENT_PLATFORM_CANONICAL", "MI_AGENT_PLATFORM_DIR")
    _OTHER_ENVS = ("MI_AGENT_ANALYTICS_DATASET", "MI_AGENT_CENTRAL_TAPE",
                   "MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_DATA_CSV")

    def setUp(self):
        from mi_agent_api import data_source as ds
        self.ds = ds
        self._saved = {k: os.environ.get(k) for k in
                       self._PLATFORM_ENVS + self._OTHER_ENVS}
        for k in self._PLATFORM_ENVS + self._OTHER_ENVS:
            os.environ.pop(k, None)
        ds.reset_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.td = Path(self._tmp.name)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.ds.reset_cache()
        self._tmp.cleanup()

    def _write_platform(self) -> Path:
        df = pd.DataFrame({
            "loan_identifier": ["L1", "L2"],
            "current_outstanding_balance": [100.0, 200.0],
            "source_portfolio_id": ["direct_001", "acquired_001"],
            "source_portfolio_type": ["direct", "acquired"],
            "source_portfolio_label": ["Direct Book", "Acquired Portfolio 1"],
            "acquisition_date": ["", "2026-08-15"],
            "seller_name": ["", "Seller A"],
            "portfolio_cohort": ["direct_001", "acquired_001"],
            pa.PLATFORM_KEY_COLUMN: ["direct_001/L1", "acquired_001/L2"],
        })
        p = self.td / pa.PLATFORM_CANONICAL_NAME
        df.to_csv(p, index=False)
        return p

    def test_uses_platform_canonical_when_present(self):
        p = self._write_platform()
        os.environ["MI_AGENT_PLATFORM_CANONICAL"] = str(p)
        path, kind = self.ds.resolve_data_source()
        self.assertEqual(kind, "platform_canonical")
        self.assertEqual(Path(path), p)

    def test_platform_dir_resolution(self):
        self._write_platform()
        os.environ["MI_AGENT_PLATFORM_DIR"] = str(self.td)
        path, kind = self.ds.resolve_data_source()
        self.assertEqual(kind, "platform_canonical")

    def test_fallback_when_no_platform_canonical(self):
        # No platform canonical anywhere -> today's behaviour (synthetic demo).
        path, kind = self.ds.resolve_data_source()
        self.assertNotEqual(kind, "platform_canonical")
        self.assertIn(kind, ("synthetic_demo", "unavailable", "central_tape"))

    def test_explicit_dataset_still_wins_over_platform(self):
        # An explicit MI_AGENT_ANALYTICS_DATASET override must still take priority.
        p = self._write_platform()
        os.environ["MI_AGENT_PLATFORM_CANONICAL"] = str(p)
        explicit = self.td / "explicit.csv"
        pd.DataFrame({"loan_identifier": ["Z1"]}).to_csv(explicit, index=False)
        os.environ["MI_AGENT_ANALYTICS_DATASET"] = str(explicit)
        path, kind = self.ds.resolve_data_source()
        self.assertEqual(kind, "prepared_explicit")
        self.assertEqual(Path(path), explicit)


if __name__ == "__main__":
    unittest.main()
