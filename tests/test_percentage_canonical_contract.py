"""The governed percentage contract: canonical stores percentage points.

90% is 90.00 — in the canonical snapshot AND on the Annex wire, which is why
the regulatory projection is a pass-through and there is no conversion to get
wrong. Confirmed for RREC12, RREC16 and RREL40.

What went wrong, and what these tests hold shut:

* the contract had nowhere to live. Of 499 registry fields exactly one declared
  ``format: percentage``, and no field anywhere declared a unit, so every
  downstream consumer was left to work the scale out from the values.
* Gate 2 force-derived the CURRENT LTV into points but only gap-filled the
  ORIGINAL, so a source-supplied fraction survived untouched. Both reached the
  Annex output: RREC12 in points beside RREC16 as a 0-1 fraction, in the same
  record, and no rule objected because the original LTV had no rules.
* the derivation demanded ``current_principal_balance``. An acquired tape
  carries an outstanding balance instead, so its LTV was never derived and
  both LTV validators were skipped for the entire acquired book.

The resolver now prefers a DECLARED source unit, RECONCILES an undeclared one
against balance and valuation, and derives only what is genuinely absent. A
supplied value that reconciles is never overwritten, and one that reconciles at
neither scale is left alone and reported rather than silently rescaled.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

from engine.gate_2_transform.canonical_transform import (  # noqa: E402
    _resolve_ltv,
    percentage_source_unit,
)


def _rules():
    spec = importlib.util.spec_from_file_location(
        "vbr", REPO / "engine/gate_3_validation/validate_business_rules.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: True LTV for the standard frame: 100k/250k and 200k/500k are both 40.00.
BAL = [100_000, 200_000]
VAL = [250_000, 500_000]
TRUE_POINTS = 40.0

CURRENT_INPUTS = ("current_principal_balance", "current_outstanding_balance")
ORIGINAL_INPUTS = ("original_principal_balance",)


def frame(**cols):
    return pd.DataFrame(cols)


def resolve(df, col="current_loan_to_value", inputs=CURRENT_INPUTS,
            val="current_valuation_amount", config=None):
    out = df.copy()
    report = _resolve_ltv(out, col, inputs, val, config or {})
    return pd.to_numeric(out[col], errors="coerce"), report


# --------------------------------------------------------------------------- #
# U — Stage A: the contract is declared, and only where confirmed
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def registry():
    return yaml.safe_load(
        (REPO / "config/system/fields_registry.yaml").read_text())["fields"]


class TestDeclaredContract:

    def test_u01_confirmed_fields_declare_percentage_points(self, registry):
        confirmed = {"current_loan_to_value", "original_loan_to_value",
                     "debt_to_income_ratio", "protected_equity_percentage"}
        for field in confirmed:
            assert registry[field].get("unit") == "percentage_points", field

    def test_u02_unconfirmed_fields_are_left_undeclared(self, registry):
        """Only fields with regulatory percentage evidence are tagged. A name
        containing 'rate' or 'ratio' is not evidence."""
        for field in ("current_interest_rate", "indexed_loan_to_value",
                      "stressed_LTV", "ltv_cap", "current_interest_rate_margin",
                      "collateralisation_ratio"):
            assert "unit" not in registry[field], field

    def test_u02b_the_tagged_set_is_exactly_the_confirmed_set(self, registry):
        tagged = {k for k, v in registry.items()
                  if isinstance(v, dict) and v.get("unit")}
        assert tagged == {"current_loan_to_value", "original_loan_to_value",
                          "debt_to_income_ratio", "protected_equity_percentage"}

    def test_u02c_every_tagged_field_has_regulatory_evidence(self, registry):
        """Each tagged field maps to a PercentageRate3Choice code, or declares
        format: percentage itself."""
        m = yaml.safe_load((REPO / "config/delivery/annex2_field_xsd_path_map.yaml"
                            ).read_text())["field_xsd_path_map"]
        rows = [r for v in m.values() if isinstance(v, list)
                for r in v if isinstance(r, dict)]
        pct_codes = {r["esma_code"] for r in rows
                     if "percentage" in str(r.get("xsd_type", "")).lower()}
        for field, spec in registry.items():
            if not (isinstance(spec, dict) and spec.get("unit")):
                continue
            mapped = {rm.get("code") for rm in (spec.get("regime_mapping") or {}).values()
                      if isinstance(rm, dict)}
            assert (mapped & pct_codes) or spec.get("format") == "percentage", field

    def test_u02d_the_alias_mechanism_is_documented(self):
        text = (REPO / "config/system/aliases_analytics.yaml").read_text()
        assert "source_unit" in text
        assert "never decides from magnitude alone" in text

    def test_source_unit_is_read_when_declared(self):
        cfg = {"aliases": {"current_loan_to_value": {"source_unit": "fraction"}}}
        assert percentage_source_unit("current_loan_to_value", cfg) == "fraction"
        assert percentage_source_unit("current_loan_to_value", {}) is None


# --------------------------------------------------------------------------- #
# T — Stage B: normalise, reconcile, derive
# --------------------------------------------------------------------------- #
class TestTransformResolution:
    def test_t01_points_input_is_kept(self):
        v, r = resolve(frame(current_principal_balance=BAL,
                             current_valuation_amount=VAL,
                             current_loan_to_value=[40.0, 40.0]))
        assert v.tolist() == [TRUE_POINTS, TRUE_POINTS]
        assert r["source_unit"] == "percentage_points"

    def test_t02_fraction_input_is_normalised(self):
        v, r = resolve(frame(current_principal_balance=BAL,
                             current_valuation_amount=VAL,
                             current_loan_to_value=[0.40, 0.40]))
        assert v.tolist() == [TRUE_POINTS, TRUE_POINTS]
        assert (r["source_unit"], r["basis"]) == ("fraction", "reconciled")

    def test_t04_declared_unit_is_authoritative(self):
        """A declaration outranks reconciliation — it is evidence, not a guess."""
        cfg = {"aliases": {"current_loan_to_value": {"source_unit": "fraction"}}}
        v, r = resolve(frame(current_principal_balance=BAL,
                             current_valuation_amount=VAL,
                             current_loan_to_value=[0.40, 0.40]), config=cfg)
        assert v.tolist() == [TRUE_POINTS, TRUE_POINTS]
        assert r["basis"] == "declared"

    def test_t05_a_genuinely_small_percentage_survives(self):
        """0.40 means 0.4% when the book agrees. It must not be rescaled just
        for being small."""
        v, r = resolve(frame(current_principal_balance=[1_000],
                             current_valuation_amount=[250_000],
                             current_loan_to_value=[0.40]))
        assert v.tolist() == [0.40]
        assert r["source_unit"] == "percentage_points"

    def test_t06_missing_ltv_is_derived(self):
        v, r = resolve(frame(current_principal_balance=BAL,
                             current_valuation_amount=VAL))
        assert v.tolist() == [TRUE_POINTS, TRUE_POINTS]
        assert r["derived_rows"] == 2

    def test_t07_principal_balance_takes_precedence(self):
        v, r = resolve(frame(current_principal_balance=BAL,
                             current_outstanding_balance=[999_999, 999_999],
                             current_valuation_amount=VAL))
        assert r["numerator"] == "current_principal_balance"
        assert v.tolist() == [TRUE_POINTS, TRUE_POINTS]

    def test_t08_outstanding_balance_is_the_fallback(self, monkeypatch):
        """The acquired path: no principal balance on the tape. Gated, because
        it creates a disclosure — see TestAcquiredDisclosureIsGated."""
        monkeypatch.setenv("GATE2_LTV_ACQUIRED_DISCLOSURE", "true")
        v, r = resolve(frame(current_outstanding_balance=BAL,
                             current_valuation_amount=VAL))
        assert r["numerator"] == "current_outstanding_balance"
        assert v.tolist() == [TRUE_POINTS, TRUE_POINTS]

    def test_t09_no_balance_means_no_derivation(self):
        out = frame(current_valuation_amount=VAL).copy()
        assert _resolve_ltv(out, "current_loan_to_value", CURRENT_INPUTS,
                            "current_valuation_amount", {}) is None
        assert "current_loan_to_value" not in out.columns

    def test_a_supplied_value_that_reconciles_is_never_overwritten(self):
        """The derivation agrees the scale; it does not replace the client's
        number. 39.8 against a 40.0 book stays 39.8."""
        v, _ = resolve(frame(current_principal_balance=BAL,
                             current_valuation_amount=VAL,
                             current_loan_to_value=[39.8, 40.2]))
        assert v.tolist() == [39.8, 40.2]

    def test_an_unreconcilable_value_is_reported_not_rescaled(self):
        v, r = resolve(frame(current_principal_balance=BAL,
                             current_valuation_amount=VAL,
                             current_loan_to_value=[93.0, 93.0]))
        assert v.tolist() == [93.0, 93.0], "left exactly as supplied"
        assert r["basis"] == "unreconciled" and r["unresolved_rows"] == 2

    def test_t10_original_ltv_is_resolved_symmetrically(self):
        """The defect that produced RREC16 as a fraction."""
        v, r = resolve(frame(original_principal_balance=BAL,
                             original_valuation_amount=VAL,
                             original_loan_to_value=[0.40, 0.40]),
                       col="original_loan_to_value", inputs=ORIGINAL_INPUTS,
                       val="original_valuation_amount")
        assert v.tolist() == [TRUE_POINTS, TRUE_POINTS]
        assert r["source_unit"] == "fraction"

    def test_current_and_original_use_their_own_inputs(self):
        """Original LTV must not read a current balance, nor the reverse."""
        df = frame(current_principal_balance=[100_000, 200_000],
                   current_valuation_amount=VAL,
                   original_principal_balance=[125_000, 250_000],
                   original_valuation_amount=VAL)
        cur, _ = resolve(df)
        orig, _ = resolve(df, col="original_loan_to_value",
                          inputs=ORIGINAL_INPUTS, val="original_valuation_amount")
        assert cur.tolist() == [40.0, 40.0]
        assert orig.tolist() == [50.0, 50.0]


# --------------------------------------------------------------------------- #
# V — Stage C: validation
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_v01_original_ltv_now_has_rules(self):
        ids = {r["rule_id"] for r in _rules().RULES}
        assert {"LTV001", "LTV002", "LTV003", "LTV004"} <= ids

    def test_v02_ltv004_flags_a_fraction_scaled_original(self):
        rule = next(r for r in _rules().RULES if r["rule_id"] == "LTV004")
        df = pd.DataFrame({"original_loan_to_value": [0.346, 34.6],
                           "original_principal_balance": [86_500, 86_500],
                           "original_valuation_amount": [250_000, 250_000]})
        assert rule["test"](df).tolist() == [False, True]

    def test_v03_range_alone_cannot_catch_a_scale_error(self):
        """Why reconciliation is the load-bearing rule: 0.346 is inside 0–500."""
        rule = next(r for r in _rules().RULES if r["rule_id"] == "LTV003")
        df = pd.DataFrame({"original_loan_to_value": [0.346, 34.6]})
        assert rule["test"](df).tolist() == [True, True]

    def test_v05_the_validator_no_longer_repairs_its_own_input(self):
        source = (REPO / "engine/gate_3_validation/validate_business_rules.py"
                  ).read_text()
        assert "_backfill_ltv" not in source


# --------------------------------------------------------------------------- #
# R — Stage D: the regulatory path is a pass-through
# --------------------------------------------------------------------------- #
class TestRegulatoryPassThrough:
    def test_r01_no_percentage_conversion_in_the_delivery_stack(self):
        """Canonical and wire share one convention, so a conversion here would
        be a defect. This guard is what keeps it that way."""
        import re
        offenders = []
        for sub in ("projection_agent", "delivery_xml_agent", "gate_4_projection",
                    "gate_4b_delivery", "gate_5_delivery"):
            for path in (REPO / "engine" / sub).rglob("*.py"):
                for i, line in enumerate(path.read_text().splitlines(), 1):
                    if re.search(r"(\*|/)\s*100(\.0)?\b", line) and not line.lstrip().startswith("#"):
                        if re.search(r"ltv|loan_to_value|percent|pct|rate", line, re.I):
                            offenders.append(f"{path.relative_to(REPO)}:{i}")
        assert not offenders, f"conversion introduced into the regulatory path: {offenders}"

    @pytest.mark.parametrize("fixture", [
        "tests/fixtures/annex2_projected_ci.csv",
        "tests/fixtures/annex2_delivery_ready_no_npe.csv",
    ])
    def test_r03_both_annex_ltv_fields_are_points(self, fixture):
        df = pd.read_csv(REPO / fixture)
        for code in ("RREC12", "RREC16"):
            values = pd.to_numeric(df[code], errors="coerce").dropna()
            assert not values.empty, code
            assert (values > 1.5).all(), (
                f"{code} carries a fraction-scaled value: {values.tolist()}")

    def test_r03b_the_two_annex_ltvs_share_an_order_of_magnitude(self):
        """The defect signature was a 100x gap between two fields describing the
        same loan — RREC16 0.346 against RREC12 37.89, a ratio of 0.009. Their
        ORDERING is not asserted: a loan that amortises rather than rolling up
        legitimately has an original LTV above its current one, which DEMO-0003
        in this fixture does."""
        df = pd.read_csv(REPO / "tests/fixtures/annex2_projected_ci.csv")
        cur = pd.to_numeric(df["RREC12"], errors="coerce")
        orig = pd.to_numeric(df["RREC16"], errors="coerce")
        ratio = orig / cur
        assert ratio.between(0.1, 10.0).all(), (
            f"scale divergence between RREC12 and RREC16: {ratio.tolist()}")


# --------------------------------------------------------------------------- #
# A — the acquired disclosure ships dark
# --------------------------------------------------------------------------- #
class TestAcquiredDisclosureIsGated:
    """Deriving LTV from an outstanding balance populates RREC12/RREC16 for a
    book that reported No Data. That is a new regulatory disclosure, so the
    capability is built and tested but off until approved."""

    ACQUIRED = {"current_outstanding_balance": BAL, "current_valuation_amount": VAL}

    def test_a01_off_by_default_the_acquired_book_is_unchanged(self, monkeypatch):
        monkeypatch.delenv("GATE2_LTV_ACQUIRED_DISCLOSURE", raising=False)
        out = frame(**self.ACQUIRED)
        assert _resolve_ltv(out, "current_loan_to_value", CURRENT_INPUTS,
                            "current_valuation_amount", {}) is None
        assert "current_loan_to_value" not in out.columns

    def test_a02_on_the_fallback_derives_in_points(self, monkeypatch):
        monkeypatch.setenv("GATE2_LTV_ACQUIRED_DISCLOSURE", "true")
        v, r = resolve(frame(**self.ACQUIRED))
        assert v.tolist() == [TRUE_POINTS, TRUE_POINTS]
        assert r["numerator"] == "current_outstanding_balance"

    def test_a02b_config_can_enable_it_without_the_environment(self):
        v, r = resolve(frame(**self.ACQUIRED),
                       config={"gate2_ltv_acquired_disclosure": True})
        assert r["numerator"] == "current_outstanding_balance"
        assert v.tolist() == [TRUE_POINTS, TRUE_POINTS]

    def test_a03_the_flag_never_affects_a_book_with_a_principal_balance(
            self, monkeypatch):
        """Direct books resolve identically either way — the flag governs the
        fallback only, not the contract."""
        direct = frame(current_principal_balance=BAL, current_valuation_amount=VAL)
        monkeypatch.delenv("GATE2_LTV_ACQUIRED_DISCLOSURE", raising=False)
        off, _ = resolve(direct)
        monkeypatch.setenv("GATE2_LTV_ACQUIRED_DISCLOSURE", "true")
        on, _ = resolve(direct)
        assert off.tolist() == on.tolist() == [TRUE_POINTS, TRUE_POINTS]
