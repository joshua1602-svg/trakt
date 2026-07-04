#!/usr/bin/env python3
"""
tests/test_synthetic_nonequity_portfolios.py

Characterisation tests that drive two *non-equity-release* synthetic funded
loan tapes through the Agentic onboarding model and pin down exactly how
"hardened" the existing registries + onboarding infrastructure are for asset
classes the platform was not built for:

  * auto_finance       — UK auto HP/PCP, secured on motor vehicles (ESMA Annex 5)
  * unsecured_consumer — UK unsecured personal loans (ESMA Annex 6)

These tests intentionally assert on *current* behaviour (including the
gaps/fail-open behaviour), so they double as an executable specification of the
hardening findings documented in
``synthetic_portfolios/HARDENING_FINDINGS.md``. If the platform is later
extended to first-class support for these asset classes, the assertions marked
``HARDENING GAP`` are the ones expected to change.

No product/engine code is modified by this work — only synthetic portfolios and
these tests are added.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent.onboarding_orchestrator import run_onboarding

PORTFOLIOS = _REPO_ROOT / "synthetic_portfolios"
REGISTRY = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
ALIASES_DIR = _REPO_ROOT / "config" / "system"

# Fields whose canonical home exists in the `common` registry block and which
# every asset class shares — the "core loan economics" that MUST generalise.
CORE_LOAN_FIELDS = {
    "loan_identifier",
    "original_principal_balance",
    "current_principal_balance",
    "current_interest_rate",
    "interest_rate_type",
    "origination_date",
    "maturity_date",
    "account_status",
}

# The common credit-risk block — the whole point of asking "is the risk
# infrastructure hardened for non-ERM assets?".
CREDIT_RISK_FIELDS = {
    "ifrs9_stage",
    "internal_risk_grade",
    "internal_risk_score",
    "probability_of_default",
    "loss_given_default",
    "exposure_at_default",
}


def _run(portfolio_dir: Path, client: str, mode: str):
    """Run the onboarding agent on a portfolio dir; return (project, out_dir)."""
    out = Path(tempfile.mkdtemp(prefix=f"synth_{client}_{mode}_")) / "run"
    project = run_onboarding(
        input_dir=str(portfolio_dir),
        client_name=client,
        output_dir=str(out),
        registry_path=str(REGISTRY),
        aliases_dir=str(ALIASES_DIR),
        mode=mode,
        client_id=client.lower(),
        run_id="run_001",
    )
    return project, out


def _tape_trace(out_dir: Path) -> pd.DataFrame:
    """Mapping trace rows for the funded loan tape only."""
    mt = pd.read_csv(out_dir / "05c_mapping_trace.csv")
    return mt[mt["source_column"].notna()
              & mt["source_file"].str.contains("funded_loan_tape")]


def _status(trace: pd.DataFrame) -> dict:
    return dict(zip(trace["source_column"], trace["final_status"]))


def _selected(trace: pd.DataFrame) -> dict:
    return dict(zip(trace["source_column"], trace["selected_candidate"]))


# ---------------------------------------------------------------------------
# 1. The pipeline INGESTS a never-seen asset class without crashing (base infra
#    is asset-agnostic enough to classify, profile and map the core economics).
# ---------------------------------------------------------------------------
class TestBaseInfraIngestsNonEquityAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.auto, cls.auto_out = _run(PORTFOLIOS / "auto_finance", "AUTO_FIN", "mi_only")
        cls.cons, cls.cons_out = _run(PORTFOLIOS / "unsecured_consumer", "UCL_FIN", "mi_only")

    def test_runs_complete_and_emit_core_artefacts(self):
        for out in (self.auto_out, self.cons_out):
            for name in ("09_onboarding_run_summary.json", "01_file_inventory.json",
                         "05c_mapping_trace.csv", "17_domain_coverage.json"):
                self.assertTrue((out / name).exists(), f"missing {name} in {out}")

    def test_funded_tape_classified_as_loan_report(self):
        # The raw *_funded_loan_tape.csv is recognised as a current loan report.
        for project in (self.auto, self.cons):
            roles = {Path(i.file_path).name: i.classification for i in project.file_inventory}
            tape = next(n for n in roles if "funded_loan_tape" in n)
            self.assertIn(roles[tape], ("current_loan_report", "historical_loan_report"))

    def test_core_loan_economics_map_for_both_asset_classes(self):
        for out in (self.auto_out, self.cons_out):
            mapped = {c for c, s in _status(_tape_trace(out)).items() if s == "mapped"}
            selected = set(_selected(_tape_trace(out)).values())
            # Every core field resolved to its canonical home.
            self.assertTrue(CORE_LOAN_FIELDS.issubset(selected),
                            f"missing core fields: {CORE_LOAN_FIELDS - selected}")
            self.assertTrue(len(mapped) >= 8)

    def test_credit_risk_block_generalises(self):
        # PD / LGD / EAD / IFRS9 stage / internal grade+score are `common`
        # analytics fields, so they map for auto AND consumer. This is the part
        # of the platform that IS hardened for non-ERM assets.
        for out in (self.auto_out, self.cons_out):
            selected = set(_selected(_tape_trace(out)).values())
            self.assertTrue(CREDIT_RISK_FIELDS.issubset(selected),
                            f"credit-risk gap: {CREDIT_RISK_FIELDS - selected}")


# ---------------------------------------------------------------------------
# 2. HARDENING GAP — asset-class identification is fail-OPEN.
#    Auto finance has NO asset signal at all, so it is silently classified as
#    equity release with full confidence and the ERM product profile is applied
#    to a motor-vehicle book. Consumer at least detects `consumer_loan`, but
#    there is no consumer product profile, so it falls back to generic handling.
# ---------------------------------------------------------------------------
class TestAssetClassIdentificationIsFailOpen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.auto, cls.auto_out = _run(PORTFOLIOS / "auto_finance", "AUTO_FIN", "regulatory_mi")
        cls.cons, cls.cons_out = _run(PORTFOLIOS / "unsecured_consumer", "UCL_FIN", "regulatory_mi")

    def test_auto_silently_mislabelled_as_equity_release(self):
        ppr = self.auto.product_profile_resolution
        # HARDENING GAP: a vehicle-finance tape is treated as a lifetime mortgage.
        self.assertTrue(ppr["applied"])
        self.assertEqual(ppr["profile_id"], "equity_release_lifetime_mortgage")
        self.assertEqual(ppr["decision"], "detected_high_confidence")
        self.assertGreaterEqual(ppr["confidence"], 0.8)
        self.assertIn("asset_class=equity_release_mortgage", ppr.get("evidence", []))

    def test_consumer_has_no_product_profile(self):
        ppr = self.cons.product_profile_resolution
        # HARDENING GAP: `consumer_loan` is a recognised asset signal but there
        # is no consumer product profile, so the engine reverts to generic
        # (stricter) behaviour rather than an asset-appropriate profile.
        self.assertFalse(ppr["applied"])
        self.assertEqual(ppr["decision"], "no_profile_generic_behaviour")

    def test_no_auto_asset_signal_exists_in_taxonomy(self):
        # Root cause: the asset-signal taxonomy has no auto/vehicle entry, and an
        # unmatched asset defaults to equity_release_mortgage.
        from engine.onboarding_agent import onboarding_context as ctx
        keys = set(ctx._ASSET_SIGNALS)
        self.assertNotIn("auto", keys)
        self.assertNotIn("auto_finance", keys)
        self.assertNotIn("vehicle", keys)
        self.assertIn("consumer_loan", keys)
        # The literal fail-open default.
        src = Path(ctx.__file__).read_text()
        self.assertIn('else "equity_release_mortgage"', src)


# ---------------------------------------------------------------------------
# 3. HARDENING GAP — collateral & asset-specific attributes have no canonical
#    home. Vehicle make/model/mileage/VIN and consumer affordability/dependents
#    fall out as unmapped; borrower age leaks onto an equity-release field.
# ---------------------------------------------------------------------------
class TestAssetSpecificAttributesAreUnmapped(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.auto, cls.auto_out = _run(PORTFOLIOS / "auto_finance", "AUTO_FIN", "regulatory_mi")
        cls.cons, cls.cons_out = _run(PORTFOLIOS / "unsecured_consumer", "UCL_FIN", "regulatory_mi")

    def test_vehicle_collateral_attributes_unmapped(self):
        status = _status(_tape_trace(self.auto_out))
        for col in ("Vehicle Make", "Vehicle Model", "Mileage",
                    "Vehicle Identification Number", "Fuel Type",
                    "Agreement Type", "Balloon Payment"):
            self.assertEqual(status.get(col), "unmapped",
                             f"expected {col!r} unmapped (no canonical field), got {status.get(col)!r}")

    def test_consumer_specific_attributes_unmapped(self):
        status = _status(_tape_trace(self.cons_out))
        for col in ("Secured / Unsecured", "Number of Dependents",
                    "Residential Status", "Affordability Assessment Result"):
            self.assertEqual(status.get(col), "unmapped",
                             f"expected {col!r} unmapped, got {status.get(col)!r}")

    def test_borrower_age_leaks_onto_equity_release_field(self):
        # HARDENING GAP: a non-ERM borrower's age is mapped to `borrower_1_age`,
        # which the registry tags portfolio_type: equity_release. There is no
        # asset-neutral borrower-age field.
        for out in (self.auto_out, self.cons_out):
            selected = _selected(_tape_trace(out))
            self.assertEqual(selected.get("Borrower Age"), "borrower_1_age")
        reg = yaml.safe_load(REGISTRY.read_text())
        self.assertEqual(reg["fields"]["borrower_1_age"]["portfolio_type"], "equity_release")


# ---------------------------------------------------------------------------
# 4. HARDENING GAP — the registry & regime layer are ESMA-Annex-2 (RRE) centric.
#    No auto (Annex 5) or consumer (Annex 6) portfolio_type or regime codes
#    exist; regulatory-mode projection reaches for RRE codes on a vehicle book.
# ---------------------------------------------------------------------------
class TestRegistryAndRegimeAreRRECentric(unittest.TestCase):
    def test_registry_has_no_auto_or_consumer_portfolio_type(self):
        reg = yaml.safe_load(REGISTRY.read_text())
        pts = {f.get("portfolio_type") for f in reg["fields"].values()}
        # Present: common / cre / rre / sme / equity_release / equipment / corporate
        self.assertIn("equity_release", pts)
        self.assertNotIn("auto", pts)
        self.assertNotIn("consumer", pts)
        self.assertNotIn("auto_finance", pts)

    def test_registry_has_no_annex5_or_annex6_regime_codes(self):
        reg = yaml.safe_load(REGISTRY.read_text())
        annexes = set()
        for f in reg["fields"].values():
            for regime in (f.get("regime_mapping") or {}):
                annexes.add(regime)
        # ESMA Annex 5 = automobile, Annex 6 = consumer — neither is present.
        self.assertNotIn("ESMA_Annex5", annexes)
        self.assertNotIn("ESMA_Annex6", annexes)
        self.assertIn("ESMA_Annex2", annexes)

    def test_enum_normalisation_only_covers_rre(self):
        enum = yaml.safe_load((ALIASES_DIR / "enum_mapping.yaml").read_text())
        self.assertEqual(list(enum.keys()), ["ESMA_Annex2"])
        # collateral_type enum is property-only — no motor-vehicle code.
        collat = {k.upper() for k in enum["ESMA_Annex2"].get("collateral_type", {})}
        self.assertNotIn("MOTOR VEHICLE", collat)
        self.assertNotIn("VEHICLE", collat)

    def test_regulatory_mode_blocks_and_reaches_for_rre_geography(self):
        # Regulatory-mode onboarding of the auto book blocks, and the geography
        # gate reaches for ESMA Annex 2 residential codes (RREL/RREC) — direct
        # evidence a motor-vehicle book is being projected onto the RRE regime.
        auto, out = _run(PORTFOLIOS / "auto_finance", "AUTO_FIN", "regulatory_mi")
        self.assertEqual(auto.review_status, "blocked")
        gq = pd.read_csv(out / "07_gap_questions.csv")
        text = " ".join(str(x) for x in gq.get("question", pd.Series([])).tolist())
        self.assertTrue(("RREL" in text) or ("RREC" in text) or ("Annex 2" in text),
                        "expected RRE (Annex 2) codes in the regulatory gap questions")


# ---------------------------------------------------------------------------
# 5. mi_only vs regulatory_mi scoping — regulatory (arrears/default/collateral)
#    fields are out-of-scope for base MI but in-scope for regulatory MI.
# ---------------------------------------------------------------------------
class TestModeScopingBehaviour(unittest.TestCase):
    def test_regulatory_fields_out_of_scope_in_mi_only(self):
        _, out = _run(PORTFOLIOS / "auto_finance", "AUTO_FIN", "mi_only")
        status = _status(_tape_trace(out))
        # These carry category: regulatory, so mi_only excludes them from scope.
        for col in ("Days Past Due", "Arrears Balance", "Default Amount",
                    "Collateral Type", "Current Loan-To-Value"):
            self.assertEqual(status.get(col), "out_of_scope",
                             f"{col!r} expected out_of_scope in mi_only, got {status.get(col)!r}")

    def test_same_fields_in_scope_under_regulatory_mi(self):
        _, out = _run(PORTFOLIOS / "auto_finance", "AUTO_FIN", "regulatory_mi")
        status = _status(_tape_trace(out))
        for col in ("Days Past Due", "Arrears Balance", "Default Amount",
                    "Collateral Type", "Current Loan-To-Value"):
            self.assertEqual(status.get(col), "mapped",
                             f"{col!r} expected mapped in regulatory_mi, got {status.get(col)!r}")


# ---------------------------------------------------------------------------
# 6. Synthetic data quality — the tapes are meaningful (credit-risk + collateral
#    coverage, internal consistency) and deterministic.
# ---------------------------------------------------------------------------
class TestSyntheticPortfolioQuality(unittest.TestCase):
    def test_generator_is_deterministic_and_consistent(self):
        from synthetic_portfolios.generate_portfolios import build_auto, build_consumer
        a1, a2 = build_auto(), build_auto()
        self.assertEqual(a1, a2)  # deterministic
        for rows, secured in ((build_auto(), True), (build_consumer(), False)):
            self.assertGreaterEqual(len(rows), 20)
            stages = {r["IFRS9 Stage"] for r in rows}
            self.assertTrue({"Stage 1", "Stage 2", "Stage 3"}.issubset(stages),
                            "portfolio should span all IFRS 9 stages")
            for r in rows:
                self.assertLessEqual(float(r["Current Principal Balance"]),
                                     float(r["Original Principal Balance"]) + 1e-6)
                # Every loan carries a full credit-risk record.
                self.assertTrue(0.0 <= float(r["Probability of Default"]) <= 1.0)
                self.assertTrue(0.0 <= float(r["Loss Given Default"]) <= 1.0)
                if secured:
                    self.assertEqual(r["Collateral Type"], "Motor Vehicle")
                else:
                    self.assertEqual(r["Secured / Unsecured"], "Unsecured")

    def test_committed_tapes_present_with_expected_shape(self):
        for sub, tape, min_cols in (
            ("auto_finance", "auto_finance_funded_loan_tape.csv", 40),
            ("unsecured_consumer", "unsecured_consumer_funded_loan_tape.csv", 35),
        ):
            df = pd.read_csv(PORTFOLIOS / sub / tape)
            self.assertGreaterEqual(len(df), 20)
            self.assertGreaterEqual(df.shape[1], min_cols)
            for col in ("Loan Identifier", "IFRS9 Stage", "Probability of Default",
                        "Loss Given Default", "Exposure at Default"):
                self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
