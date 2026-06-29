#!/usr/bin/env python3
"""tests/test_source_portfolio_provenance.py

End-to-end coverage for source-portfolio provenance: onboarding stamping,
survival through canonical transformation + validation, regime companion
(ESMA stays template-clean), and the MI Agent portfolio lenses.

Run: python -m unittest tests.test_source_portfolio_provenance
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine import provenance as prov
from engine.gate_3_validation.validate_business_rules import run_rules


# --------------------------------------------------------------------------- #
# 1. Keystone contract: derivation, fail-closed, stamping
# --------------------------------------------------------------------------- #

class TestProvenanceContract(unittest.TestCase):

    def test_derive_type_from_prefix(self):
        self.assertEqual(prov.derive_portfolio_type("direct_001"), "direct")
        self.assertEqual(prov.derive_portfolio_type("acquired_002"), "acquired")
        self.assertIsNone(prov.derive_portfolio_type("legacy_001"))

    def test_direct_book_ok_and_cohort_defaults(self):
        p = prov.build_provenance("direct_001", source_portfolio_label="Direct Book")
        self.assertEqual(p.source_portfolio_type, "direct")
        self.assertEqual(p.portfolio_cohort, "direct_001")  # defaults to id
        self.assertIsNone(p.acquisition_date)

    def test_acquired_book_ok(self):
        p = prov.build_provenance("acquired_001", acquisition_date="2026-08-15",
                                  seller_name="Seller A",
                                  source_portfolio_label="Acquired Portfolio 1")
        self.assertEqual(p.source_portfolio_type, "acquired")
        self.assertEqual(p.acquisition_date, "2026-08-15")

    def test_missing_id_fails_closed(self):
        with self.assertRaises(prov.ProvenanceError):
            prov.build_provenance("")

    def test_acquired_without_date_fails_closed(self):
        with self.assertRaises(prov.ProvenanceError):
            prov.build_provenance("acquired_001")

    def test_acquired_without_date_allowed_with_override(self):
        p = prov.build_provenance("acquired_002", allow_unknown_acquisition_date=True)
        self.assertIsNone(p.acquisition_date)

    def test_unknown_prefix_without_type_fails_closed(self):
        with self.assertRaises(prov.ProvenanceError):
            prov.build_provenance("legacy_001")
        # explicit type rescues it
        p = prov.build_provenance("legacy_001", source_portfolio_type="acquired",
                                  acquisition_date="2025-01-01")
        self.assertEqual(p.source_portfolio_type, "acquired")

    def test_direct_with_acquisition_date_fails_closed(self):
        with self.assertRaises(prov.ProvenanceError):
            prov.build_provenance("direct_001", acquisition_date="2020-01-01")

    def test_stamp_dataframe_every_row(self):
        p = prov.build_provenance("acquired_001", acquisition_date="2026-08-15")
        df = pd.DataFrame({"loan_identifier": ["L1", "L2", "L3"]})
        prov.stamp_dataframe(df, p)
        for f in prov.PROVENANCE_FIELDS:
            self.assertIn(f, df.columns)
        self.assertEqual(list(df["source_portfolio_id"]), ["acquired_001"] * 3)
        self.assertEqual(list(df["portfolio_cohort"]), ["acquired_001"] * 3)

    def test_lineage_marks_run_metadata(self):
        p = prov.build_provenance("direct_001")
        entries = prov.lineage_entries(p)
        self.assertEqual(entries["source_portfolio_id"]["source"]["origin"],
                         "run_metadata")


# --------------------------------------------------------------------------- #
# 2. Validation: PROV* rules fire on missing, pass when present
# --------------------------------------------------------------------------- #

class TestProvenanceValidation(unittest.TestCase):

    def _base_df(self):
        return pd.DataFrame({
            "loan_identifier": ["L1", "L2"],
            "current_principal_balance": [100.0, 200.0],
        })

    def test_missing_provenance_fails(self):
        v = run_rules(self._base_df(), "ESMA_Annex2")
        ids = set(v["rule_id"])
        self.assertIn("PROV001", ids)  # missing source_portfolio_id
        self.assertIn("PROV002", ids)  # missing source_portfolio_type
        self.assertIn("PROV005", ids)  # missing portfolio_cohort
        errs = v[v["rule_id"] == "PROV001"]["severity"].iloc[0]
        self.assertEqual(errs, "error")

    def test_stamped_acquired_passes(self):
        df = self._base_df()
        p = prov.build_provenance("acquired_001", acquisition_date="2026-08-15")
        prov.stamp_dataframe(df, p)
        v = run_rules(df, "ESMA_Annex2")
        self.assertEqual(len(v[v["rule_id"].str.startswith("PROV")]), 0)

    def test_stamped_direct_passes(self):
        df = self._base_df()
        prov.stamp_dataframe(df, prov.build_provenance("direct_001"))
        v = run_rules(df, "ESMA_Annex2")
        self.assertEqual(len(v[v["rule_id"].str.startswith("PROV")]), 0)

    def test_direct_with_acq_date_warns(self):
        df = self._base_df()
        # Hand-craft an inconsistent row (direct + acquisition_date) to prove
        # the validation gate surfaces it even if it bypassed build_provenance.
        df["source_portfolio_id"] = "direct_001"
        df["source_portfolio_type"] = "direct"
        df["portfolio_cohort"] = "direct_001"
        df["acquisition_date"] = "2020-01-01"
        v = run_rules(df, "ESMA_Annex2")
        self.assertIn("PROV004", set(v["rule_id"]))


# --------------------------------------------------------------------------- #
# 3. Canonical transform stamps the typed CSV (subprocess = live code path)
# --------------------------------------------------------------------------- #

class TestCanonicalTransformStamping(unittest.TestCase):

    def test_typed_csv_carries_provenance(self):
        src = _REPO / "canonical_post_mapping_pre_ledger_sample.csv"
        if not src.exists():
            self.skipTest("sample canonical csv not present")
        with tempfile.TemporaryDirectory() as td:
            cmd = [
                sys.executable,
                str(_REPO / "engine/gate_2_transform/canonical_transform.py"),
                str(src),
                "--registry", str(_REPO / "config/system/fields_registry.yaml"),
                "--portfolio-type", "equity_release",
                "--output-dir", td,
                "--no-derivations",
                "--source-portfolio-id", "acquired_001",
                "--source-portfolio-label", "Acquired Portfolio 1",
                "--acquisition-date", "2026-08-15",
                "--seller-name", "Seller A",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            out = list(Path(td).glob("*_canonical_typed.csv"))
            self.assertTrue(out)
            d = pd.read_csv(out[0])
            for f in prov.PROVENANCE_FIELDS:
                self.assertIn(f, d.columns)
            self.assertTrue((d["source_portfolio_id"] == "acquired_001").all())
            self.assertTrue((d["portfolio_cohort"] == "acquired_001").all())


# --------------------------------------------------------------------------- #
# 4. Regime companion: ESMA template-clean + provenance preserved
# --------------------------------------------------------------------------- #

class TestRegimeCompanion(unittest.TestCase):

    def test_companion_written_and_esma_clean(self):
        from engine.gate_4_projection.regime_projector import write_provenance_companion
        with tempfile.TemporaryDirectory() as td:
            df_canonical = pd.DataFrame({
                "unique_identifier": ["U1", "U2"],
                "source_portfolio_id": ["acquired_001", "acquired_001"],
                "source_portfolio_type": ["acquired", "acquired"],
                "source_portfolio_label": ["Acquired Portfolio 1"] * 2,
                "acquisition_date": ["2026-08-15"] * 2,
                "seller_name": ["Seller A"] * 2,
                "portfolio_cohort": ["acquired_001", "acquired_001"],
            })
            manifest = write_provenance_companion(df_canonical, Path(td), "ERE", "ESMA_Annex2")
            self.assertIsNotNone(manifest)
            comp = pd.read_csv(Path(td) / "ERE_ESMA_Annex2_provenance.csv")
            self.assertIn("loan_identifier", comp.columns)
            self.assertIn("source_portfolio_id", comp.columns)
            self.assertEqual(manifest["rows_by_cohort"], {"acquired_001": 2})

    def test_no_companion_when_unstamped(self):
        from engine.gate_4_projection.regime_projector import write_provenance_companion
        with tempfile.TemporaryDirectory() as td:
            df = pd.DataFrame({"unique_identifier": ["U1"]})
            self.assertIsNone(write_provenance_companion(df, Path(td), "ERE", "ESMA_Annex2"))


# --------------------------------------------------------------------------- #
# 5. MI Agent portfolio lenses
# --------------------------------------------------------------------------- #

class TestPortfolioLensResolver(unittest.TestCase):

    def test_single_lenses(self):
        from mi_agent import portfolio_lens as pl
        self.assertEqual(pl.resolve_lens("Show total portfolio balance").name, "total")
        self.assertEqual(pl.resolve_lens("Show direct book balance").name, "direct")
        self.assertEqual(pl.resolve_lens("Show acquired book balance").name, "acquired")
        coh = pl.resolve_lens("stratifications for acquired_001 only")
        self.assertEqual(coh.name, "cohort")
        self.assertEqual(coh.filters, {"source_portfolio_id": "acquired_001"})

    def test_synonyms(self):
        from mi_agent import portfolio_lens as pl
        self.assertEqual(pl.resolve_lens("originated loans, organic book").name, "direct")
        self.assertEqual(pl.resolve_lens("the purchased back book").name, "acquired")
        self.assertEqual(pl.resolve_lens("whole book").name, "total")

    def test_comparisons(self):
        from mi_agent import portfolio_lens as pl
        ab = pl.resolve_comparison_lenses("Compare direct versus acquired by LTV")
        self.assertEqual([l.name for l in ab], ["direct", "acquired"])
        cc = pl.resolve_comparison_lenses("LTV for direct_001 vs acquired_001")
        self.assertEqual([l.label for l in cc], ["direct_001", "acquired_001"])
        aa = pl.resolve_comparison_lenses("acquired_001 vs acquired_002")
        self.assertEqual(len(aa), 2)


class TestMILensExecution(unittest.TestCase):

    def setUp(self):
        from mi_agent.mi_query_validator import load_mi_semantics
        self.sem = load_mi_semantics(str(_REPO / "mi_agent/mi_semantics_field_registry.yaml"))
        self.df = pd.DataFrame({
            "loan_identifier": ["L1", "L2", "L3", "L4"],
            "current_outstanding_balance": [100.0, 200.0, 300.0, 400.0],
            "source_portfolio_id": ["direct_001", "direct_001", "acquired_001", "acquired_002"],
            "source_portfolio_type": ["direct", "direct", "acquired", "acquired"],
            "portfolio_cohort": ["direct_001", "direct_001", "acquired_001", "acquired_002"],
        })

    def _balance(self, text):
        from mi_agent.mi_query_spec import MIQuerySpec
        from mi_agent.mi_query_executor import execute_mi_query
        from mi_agent import portfolio_lens as pl
        spec = MIQuerySpec(intent="summary", metric="current_outstanding_balance",
                           aggregation="sum")
        pl.resolve_and_apply(spec, text)
        res = execute_mi_query(spec, self.df, self.sem)
        val = float(res.data.iloc[0, -1]) if res.data is not None and len(res.data) else None
        return spec, val

    def test_total_lens_all_rows(self):
        spec, val = self._balance("total portfolio balance")
        self.assertEqual(spec.portfolio_lens["name"], "total")
        self.assertEqual(val, 1000.0)

    def test_direct_lens(self):
        spec, val = self._balance("direct book balance")
        self.assertEqual(spec.portfolio_lens["name"], "direct")
        self.assertEqual(val, 300.0)

    def test_acquired_lens(self):
        spec, val = self._balance("acquired book balance")
        self.assertEqual(val, 700.0)

    def test_cohort_lens(self):
        spec, val = self._balance("balance for acquired_001 only")
        self.assertEqual(spec.portfolio_lens["filters"], {"source_portfolio_id": "acquired_001"})
        self.assertEqual(val, 300.0)

    def test_total_lens_unchanged_behaviour(self):
        # A total-lens spec must equal a no-lens spec's result (no regression).
        from mi_agent.mi_query_spec import MIQuerySpec
        from mi_agent.mi_query_executor import execute_mi_query
        spec = MIQuerySpec(intent="summary", metric="current_outstanding_balance",
                           aggregation="sum")
        plain = execute_mi_query(spec, self.df, self.sem)
        _, val = self._balance("whole book")
        self.assertEqual(float(plain.data.iloc[0, -1]), val)


if __name__ == "__main__":
    unittest.main()
