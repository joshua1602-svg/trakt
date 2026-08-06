#!/usr/bin/env python3
"""tests/test_simulation_platform_extensions.py — the production changes the
asset-class hardening work made, pinned.

Onboarding bridge lending and asset finance required a small number of governed
extensions and three genuine defect fixes. Each is pinned here so it cannot
silently regress, and so the reason for each change is readable next to it.

Also pins the two production LIMITATIONS the framework found and deliberately
did NOT paper over, so the current behaviour is documented rather than assumed.

Run: python -m pytest tests/test_simulation_platform_extensions.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine.gate_2_transform.canonical_transform import to_decimal  # noqa: E402
from engine.provenance import build_provenance, stamp_dataframe  # noqa: E402
from mi_agent.concentration_tests.library import load_library  # noqa: E402
from mi_agent.concentration_tests.metrics import evaluate_metric  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Governed limit library: new field roles, dimensions and metrics
# --------------------------------------------------------------------------- #
class TestGovernedLimitLibraryExtensions(unittest.TestCase):
    """Bridge and asset-finance limits run on the existing governed library."""

    @classmethod
    def setUpClass(cls):
        cls.lib = load_library()

    def test_the_new_field_roles_are_declared(self):
        for role in ("maturity_date", "original_term", "charge_type", "lien",
                     "contract_type", "manufacturer", "vendor", "industry",
                     "collateral_type", "balloon_amount",
                     "residual_value_current", "residual_value_original"):
            self.assertIsNotNone(self.lib.role(role), role)

    def test_the_new_composition_dimensions_are_declared(self):
        for metric_id in ("composition_dimension_share",
                          "composition_largest_group_share"):
            values = set((self.lib.get(metric_id).parameters or {})
                         ["dimension"]["values"])
            for dimension in ("charge_type", "contract_type", "collateral_type",
                              "manufacturer", "vendor", "industry"):
                self.assertIn(dimension, values, f"{metric_id}: {dimension}")

    def test_a_new_dimension_needs_no_code_change(self):
        """A dimension whose name IS a declared role resolves on its own."""
        from mi_agent.concentration_tests.metrics import _dimension_role
        self.assertEqual(_dimension_role(self.lib, "manufacturer"), "manufacturer")
        self.assertEqual(_dimension_role(self.lib, "region"), "region")
        self.assertIsNone(_dimension_role(self.lib, "not_a_governed_role"))

    def test_the_three_new_metrics_are_implemented(self):
        for metric_id in ("maturity_within_horizon_share", "extension_share",
                          "residual_value_exposure_share"):
            metric = self.lib.get(metric_id)
            self.assertIsNotNone(metric, metric_id)
            self.assertEqual(metric.implementation_status, "implemented")

    # -- evaluator behaviour -------------------------------------------- #
    def _frame(self):
        return pd.DataFrame({
            "loan_identifier": ["L1", "L2", "L3", "L4"],
            "current_outstanding_balance": [100.0, 200.0, 300.0, 400.0],
            "reporting_date": ["2025-06-30"] * 4,
            "origination_date": ["2024-06-30"] * 4,
            # L1..L3 mature exactly on their original term; only L4's maturity
            # is later than its 12-month term implies, so only L4 is extended.
            "original_term": [11, 14, 19, 12],
            # L1 already overdue, L2 inside 90 days, L3/L4 beyond it.
            "maturity_date": ["2025-05-31", "2025-08-31", "2026-01-31", "2026-06-30"],
            "balloon_amount": [0.0, 0.0, 50.0, 100.0],
            "manufacturer": ["A", "A", "B", "C"],
        })

    def test_maturity_horizon_counts_overdue_and_near_term(self):
        metric = self.lib.get("maturity_within_horizon_share")
        comp = evaluate_metric(self._frame(), self.lib, metric,
                               {"horizon_days": 90, "denominator": "current_balance",
                                "_reporting_date": "2025-06-30"})
        self.assertEqual(comp.numerator_value, 300.0)   # L1 + L2
        self.assertEqual(comp.value, 30.0)              # of 1000

    def test_maturity_horizon_can_exclude_the_already_overdue(self):
        metric = self.lib.get("maturity_within_horizon_share")
        comp = evaluate_metric(self._frame(), self.lib, metric,
                               {"horizon_days": 90, "include_past_due": False,
                                "_reporting_date": "2025-06-30"})
        self.assertEqual(comp.numerator_value, 200.0)   # L2 only

    def test_maturity_horizon_never_uses_todays_date(self):
        """A re-run of an old period must answer as it did the first time."""
        metric = self.lib.get("maturity_within_horizon_share")
        frame = self._frame()
        first = evaluate_metric(frame, self.lib, metric, {"horizon_days": 90})
        second = evaluate_metric(frame.drop(columns=["reporting_date"]), self.lib,
                                 metric, {"horizon_days": 90})
        self.assertEqual(first.numerator_value, 300.0)
        # With no reporting date anywhere, the horizon is NOT assumed.
        self.assertIsNone(second.value)
        self.assertEqual(second.data_status, "data_missing")

    def test_extension_share_is_derived_from_canonical_dates(self):
        metric = self.lib.get("extension_share")
        comp = evaluate_metric(self._frame(), self.lib, metric,
                               {"denominator": "current_balance"})
        self.assertEqual(comp.numerator_value, 400.0)   # L4 only
        self.assertEqual(comp.value, 40.0)

    def test_residual_value_share_measures_the_balloon_population(self):
        metric = self.lib.get("residual_value_exposure_share")
        comp = evaluate_metric(self._frame(), self.lib, metric,
                               {"residual_basis": "balloon", "minimum_amount": 0,
                                "denominator": "current_balance"})
        self.assertEqual(comp.numerator_value, 700.0)   # L3 + L4
        self.assertIn("balloon", comp.notes)

    def test_a_manufacturer_concentration_resolves_with_no_bespoke_evaluator(self):
        metric = self.lib.get("composition_largest_group_share")
        comp = evaluate_metric(self._frame(), self.lib, metric,
                               {"dimension": "manufacturer",
                                "denominator": "current_balance"})
        self.assertEqual(comp.numerator_value, 400.0)   # manufacturer C
        self.assertEqual(comp.resolved_columns, {"manufacturer": "manufacturer"})


# --------------------------------------------------------------------------- #
# 2. Defect fix: provenance must not blank a loan-level seller
# --------------------------------------------------------------------------- #
class TestProvenanceDoesNotEraseLoanLevelData(unittest.TestCase):
    """A run that supplies no seller must not blank the tape's own seller.

    ``seller_name`` is shared: for an acquired book it is run-level provenance,
    but an asset-finance tape uses it for the vendor who introduced each
    agreement. Overwriting it with ``None`` on every direct-origination run
    silently destroyed that value, and vendor-concentration analysis then
    resolved an empty column.
    """

    def _frame(self):
        return pd.DataFrame({"loan_identifier": ["L1", "L2"],
                             "seller_name": ["Vendor A", "Vendor B"],
                             "acquisition_date": ["2024-01-31", "2024-01-31"]})

    def test_an_absent_optional_field_leaves_the_tape_value_alone(self):
        provenance = build_provenance(source_portfolio_id="direct_001",
                                      source_portfolio_type="direct")
        out = stamp_dataframe(self._frame(), provenance)
        self.assertEqual(out["seller_name"].tolist(), ["Vendor A", "Vendor B"])

    def test_a_supplied_value_still_overrides_the_tape(self):
        provenance = build_provenance(source_portfolio_id="acquired_001",
                                      source_portfolio_type="acquired",
                                      acquisition_date="2024-01-31",
                                      seller_name="Run Level Seller")
        out = stamp_dataframe(self._frame(), provenance)
        self.assertEqual(set(out["seller_name"]), {"Run Level Seller"})

    def test_the_required_provenance_fields_are_always_stamped(self):
        provenance = build_provenance(source_portfolio_id="direct_001",
                                      source_portfolio_type="direct")
        out = stamp_dataframe(self._frame(), provenance)
        self.assertEqual(set(out["source_portfolio_id"]), {"direct_001"})
        self.assertEqual(set(out["portfolio_cohort"]), {"direct_001"})


# --------------------------------------------------------------------------- #
# 3. Defect fix: ESMA collateral-type codes must satisfy the XSD
# --------------------------------------------------------------------------- #
class TestCollateralTypeCodesMatchTheSchema(unittest.TestCase):
    """The mapped collateral codes must be members of the XSD's own enumeration.

    The previous values (R1 / R2 / C1 / C2) are not in the ``CollTp``
    enumeration, so every exposure-level submission carrying a mapped collateral
    type failed schema validation at Gate 5.
    """

    #: Verbatim from the auth.099.001.04 schema's CollTp enumeration.
    SCHEMA_CODES = {
        "INDV", "ITEQ", "MCHT", "CARX", "AERO", "CBLD", "CMTR", "ENEQ", "GUAR",
        "IBLD", "INDE", "MDEQ", "MIXD", "NACM", "NALV", "OFEQ", "OTHR", "OTFA",
        "OTGI", "OTRE", "OTHV", "OTHE", "RALV", "RBLD", "SECU",
    }

    def setUp(self):
        mapping = yaml.safe_load(
            (_REPO / "config" / "system" / "enum_mapping.yaml")
            .read_text(encoding="utf-8"))
        self.collateral = mapping["ESMA_Annex2"]["collateral_type"]

    def test_every_mapped_code_is_in_the_schema_enumeration(self):
        invalid = sorted({v for v in self.collateral.values()
                          if v not in self.SCHEMA_CODES})
        self.assertEqual(invalid, [])

    def test_residential_commercial_and_mixed_collateral_all_map(self):
        self.assertEqual(self.collateral["HOUSE"], "RBLD")
        self.assertEqual(self.collateral["FLAT"], "RBLD")
        self.assertEqual(self.collateral["OFFICE"], "CBLD")
        self.assertEqual(self.collateral["INDUSTRIAL"], "IBLD")
        self.assertEqual(self.collateral["MIXED_USE"], "MIXD")
        self.assertEqual(self.collateral["OTHER"], "OTHR")


# --------------------------------------------------------------------------- #
# 4. Defect fix: a non-Annex-2 regime must refuse, not crash
# --------------------------------------------------------------------------- #
class TestRegulatoryDeliveryRefusesCleanly(unittest.TestCase):
    """Only Annex 2 has a committed delivery template.

    Gate 5 previously invoked the generic builder with no ``--template``, so it
    fell through to a default file that does not exist and died inside a
    subprocess with a Jinja ``TemplateNotFound`` — a crash where a governed
    a governed refusal belongs.

    The refusal must say the delivery is NOT IMPLEMENTED, not merely "not
    configured": for every regime except Annex 2 there is no authored,
    XSD-validated delivery template in this repository, so telling an operator
    to supply a path would invite an unvalidated artefact to be mistaken for a
    submission.
    """

    def test_the_orchestrator_exposes_a_template_option(self):
        source = (_REPO / "engine" / "orchestrator" / "trakt_run.py").read_text(
            encoding="utf-8")
        self.assertIn("--regime-xml-template", source)

    def test_the_refusal_says_the_capability_is_unbuilt(self):
        source = (_REPO / "engine" / "orchestrator" / "trakt_run.py").read_text(
            encoding="utf-8")
        self.assertIn("is NOT IMPLEMENTED", source)
        self.assertIn("authored and", source)
        self.assertIn("UNVALIDATED artefact", source)

    def test_the_refusal_does_not_present_this_as_a_deployment_setting(self):
        """The previous wording invited exactly the wrong conclusion."""
        source = (_REPO / "engine" / "orchestrator" / "trakt_run.py").read_text(
            encoding="utf-8")
        refusal = source[source.index("[Gate 5] Regulatory delivery for"):]
        refusal = refusal[:refusal.index('")') + 2]
        self.assertNotIn("is not configured", refusal)
        self.assertIn("not a missing", refusal)

    def test_the_framework_asserts_that_exact_refusal(self):
        from simulation.assertions.regime import NOT_CONFIGURED_MARKER
        source = (_REPO / "engine" / "orchestrator" / "trakt_run.py").read_text(
            encoding="utf-8")
        self.assertIn(NOT_CONFIGURED_MARKER, source)


# --------------------------------------------------------------------------- #
# 5. Enhancement: machine-readable KPI evidence
# --------------------------------------------------------------------------- #
class TestKpiArtefactsCarryRawValues(unittest.TestCase):
    """A governed answer's headline figure must be verifiable by a machine.

    ``value`` is formatted for a human ("£18.9MM"), which is rounded and cannot
    be reconciled to the book. ``rawValue`` is additive alongside it.
    """

    def _artifact(self):
        from mi_agent_api.adapters import AdapterContext, _kpi_artifact
        qr = {"data": [{"current_outstanding_balance_sum": 18_900_916.49,
                        "loan_count": 142}],
              "resolved_fields": {"current_outstanding_balance_sum":
                                  {"canonical_field": "current_outstanding_balance"}}}
        return _kpi_artifact(qr, {"intent": "summary"},
                             AdapterContext("sim/latest", "2025-06-30"), {})

    def test_the_display_string_is_still_a_formatted_string(self):
        for kpi in self._artifact()["kpis"]:
            self.assertIsInstance(kpi["value"], str)
        display = {k["value"] for k in self._artifact()["kpis"]}
        self.assertIn("18,900,916", display)
        # Rounded for a human, so it cannot be reconciled to the book — which is
        # exactly why the raw figure has to travel beside it.
        self.assertNotIn("18900916.49", display)

    def test_the_raw_figure_is_carried_alongside(self):
        raw = {k["label"]: k.get("rawValue") for k in self._artifact()["kpis"]}
        self.assertIn(18_900_916.49, raw.values())
        self.assertIn(142.0, raw.values())

    def test_the_raw_figure_names_its_canonical_field(self):
        kpis = {k["label"]: k for k in self._artifact()["kpis"]}
        balance = next(k for k in kpis.values()
                       if k.get("rawValue") == 18_900_916.49)
        self.assertEqual(balance["canonicalField"], "current_outstanding_balance")
        self.assertEqual(balance["field"], "current_outstanding_balance_sum")
        self.assertTrue(balance["valueFormat"])


# --------------------------------------------------------------------------- #
# 6. Enhancement: the MI semantics registry knows the new asset classes
# --------------------------------------------------------------------------- #
class TestMiSemanticsCoversTheNewAssetClasses(unittest.TestCase):
    def setUp(self):
        self.fields = yaml.safe_load(
            (_REPO / "mi_agent" / "mi_semantics_field_registry.yaml")
            .read_text(encoding="utf-8"))["fields"]

    def test_bridge_dimensions_are_governed_mi_vocabulary(self):
        for field in ("charge_type", "collateral_type", "original_term"):
            self.assertIn(field, self.fields)

    def test_asset_finance_dimensions_are_governed_mi_vocabulary(self):
        for field in ("manufacturer", "seller_name", "nace_industry_code",
                      "balloon_amount", "current_residual_value_of_asset"):
            self.assertIn(field, self.fields)

    def test_industry_is_a_dimension_not_an_identifier(self):
        entry = self.fields["nace_industry_code"]
        self.assertEqual(entry["role"], "dimension")
        self.assertTrue(entry["chartable"])

    def test_the_vendor_synonyms_reach_the_seller_field(self):
        synonyms = self.fields["seller_name"]["synonyms"]
        for phrase in ("vendor", "dealer", "seller"):
            self.assertIn(phrase, synonyms)

    def test_the_new_vocabulary_does_not_claim_bare_head_nouns(self):
        """Regression: a curated synonym must not hijack an existing question.

        ``collateral_type`` originally carried the bare word "collateral",
        which pulled a dimension into "what is the total collateral
        valuation" and broke the routing (calibration case ``kpi_024``).
        ``charge_type`` carried the bare word "charge", which ``lien``
        already owned, making resolution order-dependent.
        """
        self.assertNotIn("collateral", self.fields["collateral_type"]["synonyms"])
        self.assertNotIn("charge", self.fields["charge_type"]["synonyms"])
        self.assertIn("charge", self.fields["lien"]["synonyms"])

    def test_no_curated_synonym_is_owned_by_two_fields(self):
        owners: dict = {}
        for key, entry in self.fields.items():
            for synonym in entry.get("synonyms") or []:
                owners.setdefault(str(synonym).strip().casefold(), []).append(key)
        self.assertEqual({s: k for s, k in owners.items() if len(k) > 1}, {})


# --------------------------------------------------------------------------- #
# 7. Recorded LIMITATIONS — pinned, not papered over
# --------------------------------------------------------------------------- #
class TestRecordedLimitations(unittest.TestCase):
    def test_a_continental_decimal_comma_is_silently_misread(self):
        """``to_decimal`` strips ``,`` as a thousands separator unconditionally.

        ``1234,56`` therefore becomes ``123456`` — a hundred-fold error reported
        as a clean parse, not as a failure. The simulation's locale dialect
        deliberately avoids this rendering; the behaviour is pinned here so a
        future fix has a test to turn around, and so nobody assumes the
        capability exists.
        """
        parsed = to_decimal(pd.Series(["1234,56"]))
        self.assertEqual(float(parsed.iloc[0]), 123456.0)
        self.assertNotEqual(float(parsed.iloc[0]), 1234.56)

    def test_thousands_separated_decimals_are_read_correctly(self):
        parsed = to_decimal(pd.Series(["1,234,567.89"]))
        self.assertAlmostEqual(float(parsed.iloc[0]), 1234567.89, places=2)

    def test_the_exposure_delivery_rules_enumerate_one_client_lei(self):
        """The shared delivery rules validate the originator LEI as an ENUM.

        An identifier is being checked for membership of a list rather than for
        its shape, so every other originator's LEI blocks delivery. The
        framework works around it with a run-scoped overlay through the existing
        ``--annex2-delivery-rules`` option rather than editing the regulatory
        contract; this pins the current behaviour.
        """
        rules = yaml.safe_load(
            (_REPO / "config" / "regime" / "annex2_delivery_rules.yaml")
            .read_text(encoding="utf-8"))
        enum_map = ((rules["field_rules"]["RREL83"].get("transform") or {})
                    .get("enum_map") or {})
        self.assertTrue(enum_map, "RREL83 has no enum map at all")
        self.assertLessEqual(len(enum_map), 4,
                             "RREL83 still enumerates specific LEIs")

    def test_the_canonical_model_has_no_funded_spv_field(self):
        """``spv_id`` is a reserved snapshot column and a governed risk role, but
        it is not in the canonical field registry, so a funded SPV segmentation
        cannot round-trip through canonical today. The framework carries the
        funding structure on ``source_portfolio_id`` instead."""
        from snapshot.model import RESERVED_OPTIONAL_LOAN_COLUMNS

        registry = yaml.safe_load(
            (_REPO / "config" / "system" / "fields_registry.yaml")
            .read_text(encoding="utf-8"))
        self.assertIn("spv_id", RESERVED_OPTIONAL_LOAN_COLUMNS)
        self.assertIsNotNone(load_library().role("spv"))
        self.assertNotIn("spv_id", registry["fields"])


# --------------------------------------------------------------------------- #
# 8. The verifier must survive the scale it exists to verify
# --------------------------------------------------------------------------- #
class TestArtefactCheckIsMemoryBounded(unittest.TestCase):
    """Regression: the well-formedness check must stream, not build a DOM.

    A 100 000-loan Annex 2 submission is a 2.6 GB XML. The check used to call
    ``ElementTree.parse``, whose DOM was several times that — the performance
    profile was OOM-killed by its own assertion rather than by the platform.
    """

    def _sample_xml(self, directory, elements: int = 60_000):
        path = Path(directory) / "sample.xml"
        with path.open("w", encoding="utf-8") as handle:
            handle.write("<Doc>")
            for index in range(elements):
                handle.write(f"<Ln><Id>L{index:08d}</Id><Bal>{index}.00</Bal></Ln>")
            handle.write("</Doc>")
        return path

    def test_streaming_uses_a_fraction_of_the_dom_footprint(self):
        import tempfile
        import tracemalloc
        from xml.etree import ElementTree

        from simulation.assertions.regime import _is_well_formed

        with tempfile.TemporaryDirectory() as tmp:
            path = self._sample_xml(tmp)

            tracemalloc.start()
            self.assertTrue(_is_well_formed(path))
            streamed = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

            tracemalloc.start()
            ElementTree.parse(str(path))
            dom = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

        self.assertLess(streamed, dom / 10,
                        f"streaming peak {streamed} is not bounded well below "
                        f"the DOM peak {dom}")

    def test_a_malformed_artefact_is_still_reported_as_malformed(self):
        import tempfile

        from simulation.assertions.regime import _is_well_formed

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.xml"
            bad.write_text("<Doc><Ln><Id>1</Id></Doc>", encoding="utf-8")
            self.assertFalse(_is_well_formed(bad))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
