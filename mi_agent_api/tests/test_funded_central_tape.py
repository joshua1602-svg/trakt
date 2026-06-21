#!/usr/bin/env python3
"""mi_agent_api/tests/test_funded_central_tape.py

The MI Agent API (and therefore the React dashboard it backs) must be able to
serve the **promoted funded central lender tape** — proving the data layer
reflects real promoted output, not the synthetic demo:

  * mi_2025_10 -> 33 loans / c. £4.208MM / GBP / valuation populated;
  * mi_2025_11 -> 73 loans / c. £8.903MM / GBP / valuation populated;
  * the old 2,196-row universe never appears; pipeline/KFI rows are excluded;
  * /health reports the funded source (generic by client_id / run_id);
  * a "portfolio summary" query returns a KPI artifact with the funded count +
    outstanding balance (the same envelope the React renderer consumes).

These run the REAL onboarding + promotion pipeline, then point the API data
source at the resulting 18_central_lender_tape.csv.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from mi_agent_api import data_source

_REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
_OCT_N, _OCT_EACH = 33, 127515.15      # -> 4,207,999.95  (~£4.208MM)
_NOV_N, _NOV_EACH = 73, 121958.90      # -> 8,902,999.70  (~£8.903MM)

_FUNDED_ENV = ("MI_AGENT_CENTRAL_TAPE", "MI_AGENT_ONBOARDING_OUTPUT_ROOT",
               "MI_AGENT_CLIENT_ID", "MI_AGENT_RUN_ID", "MI_AGENT_DATA_CSV")


def _clear_funded_env():
    for k in _FUNDED_ENV:
        os.environ.pop(k, None)
    data_source.reset_cache()


def _to_num(v) -> float:
    return float(re.sub(r"[^0-9.\-]", "", str(v)) or 0)


def _make_pack(root: Path) -> Path:
    inp = root / "input"
    inp.mkdir(parents=True)
    ids = [760000 + i for i in range(_NOV_N)]
    long = [s * 100 + 1 for s in ids]
    oct_rows, nov_rows = [], []
    for i, lid in enumerate(ids):
        nov_rows.append({"Loan Policy Number": lid, "Month Run": "November",
                         "Loan Interest Rate": 3.10 + (i % 5) * 0.05,
                         "Current Outstanding Balance": _NOV_EACH,
                         "Policy Completion Date": "2025-06-01"})
        if i < _OCT_N:
            oct_rows.append({"Loan Policy Number": lid, "Month Run": "October",
                             "Loan Interest Rate": 3.10 + (i % 5) * 0.05,
                             "Current Outstanding Balance": _OCT_EACH,
                             "Policy Completion Date": "2025-06-01"})
    pd.DataFrame(oct_rows + nov_rows).to_csv(inp / "LoanExtract One.csv", index=False)
    pd.DataFrame({"Account Number": long,
                  "Latest Property Value": [250000.0 + i for i in range(_NOV_N)]}).to_csv(
        inp / "Collateral Extract.csv", index=False)
    pd.DataFrame({"application_id": [f"APP{i}" for i in range(20)],
                  "Account Number": [990000 + i for i in range(20)],
                  "product rate": [4.0] * 20}).to_csv(
        inp / "M2L KFI and Pipeline 2025_12_01.csv", index=False)
    return inp


def _promote(root: Path, inp: Path, run_id: str) -> Path:
    from engine.onboarding_agent import workflow as wf, storage_paths, central_tape_builder
    proj = root / f"proj_{run_id}"
    wf.run_operator_workflow(
        input_dir=str(inp), client_name="Client 001", client_id="client_001",
        run_id=run_id, mode="mi_only", project_dir=str(proj),
        product_profile="equity_release_lifetime_mortgage")
    rp = storage_paths.resolve_run_paths(
        project_dir=str(proj), input_dir=str(inp), output_root=None,
        client_id="client_001", run_id=run_id, storage_backend="local",
        input_uri="", output_uri="")
    res = central_tape_builder.build_central_tapes(str(proj), rp, _REGISTRY, mode="mi_only")
    return Path(res["central_lender_tape_path"])


class TestFundedCentralTapeServedByApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="api_funded_"))
        inp = _make_pack(cls.root)
        cls.oct_tape = _promote(cls.root, inp, "mi_2025_10")
        cls.nov_tape = _promote(cls.root, inp, "mi_2025_11")

    def tearDown(self):
        _clear_funded_env()

    @classmethod
    def tearDownClass(cls):
        _clear_funded_env()

    def _serve(self, tape: Path, client_id="client_001", run_id="mi_2025_10"):
        _clear_funded_env()
        os.environ["MI_AGENT_CENTRAL_TAPE"] = str(tape)
        os.environ["MI_AGENT_CLIENT_ID"] = client_id
        os.environ["MI_AGENT_RUN_ID"] = run_id
        data_source.reset_cache()

    # --- data layer ---
    def test_get_dataframe_october(self):
        self._serve(self.oct_tape, run_id="mi_2025_10")
        df = data_source.get_dataframe()
        self.assertEqual(len(df), 33)
        self.assertAlmostEqual(
            pd.to_numeric(df["current_outstanding_balance"]).sum(), 4_208_000, delta=2_000)
        self.assertEqual(set(df["exposure_currency_denomination"]), {"GBP"})
        self.assertEqual(int(df["current_valuation_amount"].notna().sum()), 33)

    def test_get_dataframe_november(self):
        self._serve(self.nov_tape, run_id="mi_2025_11")
        df = data_source.get_dataframe()
        self.assertEqual(len(df), 73)
        self.assertAlmostEqual(
            pd.to_numeric(df["current_outstanding_balance"]).sum(), 8_903_000, delta=2_000)

    def test_no_polluted_or_pipeline_rows(self):
        self._serve(self.oct_tape, run_id="mi_2025_10")
        df = data_source.get_dataframe()
        self.assertNotEqual(len(df), 2196)
        ids = set(df["loan_identifier"].astype(str))
        self.assertFalse(any(i.startswith("9900") for i in ids))  # no pipeline accounts

    # --- generic resolution by output-root + client/run ---
    def test_resolution_by_output_root_client_run(self):
        _clear_funded_env()
        os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(self.oct_tape.parent.parent)  # .../output
        os.environ["MI_AGENT_CLIENT_ID"] = "client_001"
        os.environ["MI_AGENT_RUN_ID"] = "mi_2025_10"
        data_source.reset_cache()
        self.assertEqual(data_source.data_source_kind(), data_source.KIND_PREPARED)
        self.assertEqual(len(data_source.get_dataframe()), 33)

    # --- /health ---
    def test_health_reports_funded_source(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        self._serve(self.oct_tape, run_id="mi_2025_10")
        body = TestClient(app).get("/health").json()
        self.assertTrue(body["dataAvailable"])
        self.assertEqual(body["dataSourceKind"], data_source.KIND_PREPARED)
        self.assertEqual(body["dataSourceInfo"]["run_id"], "mi_2025_10")
        self.assertEqual(body["dataSource"], "18_central_lender_tape.csv")

    # --- query envelope the React renderer consumes ---
    def _summary_kpis(self, tape: Path, run_id: str):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        self._serve(tape, run_id=run_id)
        body = TestClient(app).post(
            "/mi/query",
            json={"question": "portfolio summary", "portfolioId": f"client_001/{run_id}",
                  "asOfDate": "2025-10-31"}).json()
        self.assertTrue(body["ok"], body.get("validation"))
        kpi = next((a for a in body["artifacts"] if a["type"] == "kpi"), None)
        self.assertIsNotNone(kpi, body)
        items = kpi.get("kpis") or kpi.get("items") or []
        return {str(it.get("label", "")).lower(): it.get("value") for it in items}

    def test_query_summary_october(self):
        kpis = self._summary_kpis(self.oct_tape, "mi_2025_10")
        # loan count
        loan_kpi = next(v for k, v in kpis.items() if "loan" in k)
        self.assertEqual(int(_to_num(loan_kpi)), 33)
        bal_kpi = next(v for k, v in kpis.items() if "balance" in k)
        self.assertAlmostEqual(_to_num(bal_kpi), 4_208_000, delta=2_000)

    def test_query_summary_november(self):
        kpis = self._summary_kpis(self.nov_tape, "mi_2025_11")
        loan_kpi = next(v for k, v in kpis.items() if "loan" in k)
        self.assertEqual(int(_to_num(loan_kpi)), 73)


class TestMessyHeaderCsvLoading(unittest.TestCase):
    """PropertyExtract-style CSVs put the real header in row 2; the central tape
    loader must re-detect it (not read ``Unnamed:*`` columns)."""

    def test_read_df_redetects_row_two_header(self):
        from engine.onboarding_agent.central_tape_builder import _read_df
        d = Path(tempfile.mkdtemp(prefix="hdr_"))
        f = d / "PG_PropertyExtract Internal OMNI_test.csv"
        f.write_text(
            "PG PropertyExtract Internal OMNI,,,,\n"
            "Loan ID,Latest Valuation,Original Valuation,Youngest Age,Property Region\n"
            "76034101,500000,450000,67,London\n"
            "76034201,300000,280000,72,Wales\n", encoding="utf-8")
        df = _read_df(str(f))
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)  # rows_raw non-zero
        for col in ("Loan ID", "Latest Valuation", "Original Valuation",
                    "Youngest Age", "Property Region"):
            self.assertIn(col, df.columns)
        self.assertFalse(any(str(c).startswith("Unnamed") for c in df.columns))


class TestDefaultStillSyntheticDemo(unittest.TestCase):
    def test_demo_default_when_unconfigured(self):
        _clear_funded_env()
        kind = data_source.data_source_kind()
        # With no funded env, the API falls back to the synthetic demo (or
        # unavailable if the demo CSV isn't present) — never the funded tape.
        self.assertIn(kind, (data_source.KIND_SYNTHETIC_DEMO, data_source.KIND_UNAVAILABLE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
