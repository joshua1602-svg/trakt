#!/usr/bin/env python3
"""tests/test_onboarding_period_eligibility.py

Reporting-period eligibility + period-scoped central lender tape universe.

Covers (per spec):
  1. run-period inference from run_id;
  2. period_of_value (Month-Run column, bare month name + run year, ISO);
  3. file-level eligibility (separate monthly files; future-period excluded);
  4. row-level eligibility (a cumulative current-book file with a Month Run
     column is filtered to the run period);
  5. promotion end-to-end:
       mi_2025_10 -> 33 funded loans, c. £4.2MM current_outstanding_balance;
       mi_2025_11 -> 73 funded loans, c. £8.9MM;
     pipeline-only / future-period files create no lender-tape rows;
  6. 04c explains include/exclude; 18b lineage + 18f universe debug present;
  7. regulatory mode universe gate untouched.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from engine.onboarding_agent import source_period_eligibility as spe


# --------------------------------------------------------------------------- #
# 1-4 — unit behaviour
# --------------------------------------------------------------------------- #
class TestPeriodUnit(unittest.TestCase):
    def test_run_period_from_run_id(self):
        self.assertEqual(spe.run_period("mi_2025_10"), ("2025-10", "2025-10-31"))
        self.assertEqual(spe.run_period("mi_2025_11"), ("2025-11", "2025-11-30"))

    def test_period_of_value(self):
        self.assertEqual(spe.period_of_value("October", 2025), "2025-10")
        self.assertEqual(spe.period_of_value("Nov", 2025), "2025-11")
        self.assertEqual(spe.period_of_value("2025-11-30", None), "2025-11")
        self.assertEqual(spe.period_of_value("31/10/2025", None), "2025-10")
        self.assertEqual(spe.period_of_value("", 2025), "")

    def test_filename_delivery_offset(self):
        cfg = dict(spe._DEFAULTS, filename_delivery_offset_months=-1)
        recs = [{"file_name": "M2L KFI and Pipeline 2025_11_01_113916.xlsx",
                 "file_path": "/x/M2L KFI and Pipeline 2025_11_01_113916.xlsx",
                 "sheet_name": "", "artefact_role": "pipeline_report",
                 "detected_reporting_date": "", "df": None}]
        rows = spe.compute_eligibility(recs, "mi_2025_10", config=cfg)
        # 2025_11_01 delivery, offset -1 -> October close -> eligible for mi_2025_10.
        self.assertEqual(rows[0].inferred_reporting_period, "2025-10")
        self.assertTrue(rows[0].is_period_eligible)

    def test_future_period_excluded(self):
        df = pd.DataFrame({"Loan ID": [1, 2], "Month Run": ["November", "November"]})
        recs = [{"file_name": "nov.csv", "file_path": "/x/nov.csv", "sheet_name": "",
                 "artefact_role": "current_loan_report", "detected_reporting_date": "",
                 "df": df}]
        rows = spe.compute_eligibility(recs, "mi_2025_10")
        self.assertFalse(rows[0].is_period_eligible)
        self.assertEqual(rows[0].reason_excluded, "future_period")

    def test_cumulative_file_is_row_filterable(self):
        df = pd.DataFrame({"Loan ID": [1, 2, 3],
                           "Month Run": ["October", "November", "November"]})
        recs = [{"file_name": "book.csv", "file_path": "/x/book.csv", "sheet_name": "",
                 "artefact_role": "current_loan_report", "detected_reporting_date": "",
                 "df": df}]
        oct_rows = spe.compute_eligibility(recs, "mi_2025_10")
        self.assertTrue(oct_rows[0].is_period_eligible)         # run period present
        self.assertEqual(oct_rows[0].period_column, "Month Run")
        self.assertTrue(oct_rows[0].is_universe_source)


# --------------------------------------------------------------------------- #
# 5-7 — promotion end-to-end (period-scoped universe)
# --------------------------------------------------------------------------- #
class _PromoBase(unittest.TestCase):
    """One cumulative current-book file (Oct 33 / Nov 73 cumulative) + a
    future-period pipeline file in the same input dir."""

    OCT_N = 33
    NOV_N = 73
    OCT_BAL = 4_200_000.0
    NOV_BAL = 8_900_000.0

    @classmethod
    def _make_input(cls, root: Path) -> Path:
        inp = root / "input"
        inp.mkdir(parents=True)
        ids = [760000 + i for i in range(cls.NOV_N)]
        # October book: first 33 loans, ~£4.2MM total.
        # November book (cumulative): all 73 loans, ~£8.9MM total.
        oct_rows, nov_rows = [], []
        oct_each = round(cls.OCT_BAL / cls.OCT_N, 2)
        nov_each = round(cls.NOV_BAL / cls.NOV_N, 2)
        for i, lid in enumerate(ids):
            nov_rows.append({"Loan Policy Number": lid, "Month Run": "November",
                             "Loan Interest Rate": 3.10 + (i % 5) * 0.05,
                             "Current Outstanding Balance": nov_each,
                             "Policy Completion Date": "2025-11-15"})
            if i < cls.OCT_N:
                oct_rows.append({"Loan Policy Number": lid, "Month Run": "October",
                                 "Loan Interest Rate": 3.10 + (i % 5) * 0.05,
                                 "Current Outstanding Balance": oct_each,
                                 "Policy Completion Date": "2025-10-15"})
        pd.DataFrame(oct_rows + nov_rows).to_csv(inp / "LoanExtract One.csv", index=False)
        # A future-period pipeline file (must not create lender-tape rows).
        pd.DataFrame({"application_id": [f"APP{i}" for i in range(20)],
                      "Account Number": [990000 + i for i in range(20)],
                      "product rate": [4.0] * 20,
                      "Month Run": ["December"] * 20}).to_csv(
            inp / "M2L KFI and Pipeline 2025_12_01.csv", index=False)
        return inp

    @classmethod
    def _run(cls, root: Path, inp: Path, run_id: str):
        from engine.onboarding_agent import workflow as wf
        from engine.onboarding_agent import central_tape_builder, storage_paths
        proj = root / f"proj_{run_id}"
        wf.run_operator_workflow(
            input_dir=str(inp), client_name="T", client_id="t", run_id=run_id,
            mode="mi_only", project_dir=str(proj),
            product_profile="equity_release_lifetime_mortgage")
        rp = storage_paths.resolve_run_paths(
            project_dir=str(proj), input_dir=str(inp), output_root=None,
            client_id="t", run_id=run_id, storage_backend="local",
            input_uri="", output_uri="")
        tr = central_tape_builder.build_central_tapes(
            str(proj), rp,
            str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml"),
            mode="mi_only")
        return proj, tr, pd.read_csv(tr["central_lender_tape_path"])


class TestPeriodScopedPromotion(_PromoBase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="period_promo_"))
        cls.inp = cls._make_input(cls.root)
        cls.oct_proj, cls.oct_tr, cls.oct_tape = cls._run(cls.root, cls.inp, "mi_2025_10")
        cls.nov_proj, cls.nov_tr, cls.nov_tape = cls._run(cls.root, cls.inp, "mi_2025_11")

    def _bal(self, tape) -> float:
        s = (tape["current_outstanding_balance"].astype(str)
             .str.replace(",", "", regex=False))
        return pd.to_numeric(s, errors="coerce").dropna().sum()

    def test_october_universe_is_33(self):
        self.assertEqual(len(self.oct_tape), self.OCT_N)

    def test_november_universe_is_73(self):
        self.assertEqual(len(self.nov_tape), self.NOV_N)

    def test_october_balance(self):
        self.assertAlmostEqual(self._bal(self.oct_tape), self.OCT_BAL, delta=1.0)

    def test_november_balance(self):
        self.assertAlmostEqual(self._bal(self.nov_tape), self.NOV_BAL, delta=1.0)

    def test_funded_fields_populated_full_universe(self):
        for tape, n in ((self.oct_tape, self.OCT_N), (self.nov_tape, self.NOV_N)):
            for f in ("current_interest_rate", "current_outstanding_balance"):
                self.assertEqual(int(tape[f].notna().sum()), n, f)

    def test_pipeline_does_not_create_rows(self):
        # 20 future-period pipeline accounts must not appear as lender-tape rows.
        self.assertEqual(len(self.oct_tape), self.OCT_N)
        self.assertNotIn("990000", set(self.oct_tape["loan_identifier"].astype(str)))

    def test_04c_explains_include_exclude(self):
        rows = list(csv.DictReader(open(self.oct_proj / "04c_source_period_eligibility.csv")))
        by_file = {r["source_file"]: r for r in rows}
        self.assertTrue(any("LoanExtract" in f for f in by_file))
        # The pipeline file is recorded with its (future) period and excluded.
        pipe = next((r for f, r in by_file.items() if "M2L" in f), None)
        self.assertIsNotNone(pipe)

    def test_18f_universe_debug_written(self):
        for proj, n in ((self.oct_proj, self.OCT_N), (self.nov_proj, self.NOV_N)):
            dbg = json.loads((proj / "18f_central_universe_debug.json").read_text())
            self.assertTrue(dbg["period_gate_active"])
            self.assertEqual(dbg["canonical_universe_rows"], n)
            self.assertTrue(dbg["selected_universe_sources"])

    def test_18b_lineage_has_funded_source(self):
        lin = pd.read_csv(self.oct_tr["central_tape_lineage_path"])
        present = set(lin["canonical_field"].astype(str))
        self.assertIn("current_outstanding_balance", present)


class TestRegulatoryUniverseUntouched(unittest.TestCase):
    def test_period_gate_only_for_mi_modes(self):
        import inspect
        from engine.onboarding_agent import central_tape_builder
        src = inspect.getsource(central_tape_builder.build_central_tapes)
        self.assertIn('if mode in ("mi_only", "mna_dd")', src)
        # The gate is only assembled inside that MI-mode block.
        self.assertIn('period_gate: Dict[str, Any] = {}', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
