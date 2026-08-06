"""simulation.assets.equity_release — UK lifetime-mortgage / equity-release book.

Roll-up (interest-capitalising) lending with no contractual amortisation and no
fixed maturity: the loan redeems on death, move into long-term care, or sale.
This mirrors the ``equity_release_lifetime_mortgage`` product profile already in
``config/asset/product_profiles.yaml`` and runs under the production
``--portfolio-type equity_release``.

The existing equity-release scenario engine (``analytics/scenario_engine.py``)
is deliberately NOT replaced or re-implemented: this module produces a book for
it (and for the deterministic MI path) to be verified against.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..dates import months_between, parse_iso
from ..geo import REGION_VALUE_INDEX, postcode, weighted_region
from ..models import (
    ASSET_EQUITY_RELEASE,
    ASSET_SUBTYPES,
    MOVE_FURTHER_ADVANCE,
    MOVE_INTEREST,
    RISK_AMBER,
    RISK_RED,
    STATUS_PERFORMING,
    CaseManifest,
    LoanState,
)
from . import GenContext, register
from . import _common as C

#: Canonical fields specific to equity release, on top of the common set.
ERM_FIELDS = (
    "erm_product_type",
    "youngest_borrower_age",
    "negative_equity_guarantee",
    "further_advance_amount",
    "further_advance_flag",
    "ltv_cap",
    "loan_redemption_flag",
    "property_type",
    "occupancy_type",
    "current_loan_to_value",
    "original_loan_to_value",
)

#: ``erm_product_type`` values by subtype, and the drawdown behaviour each
#: implies. A drawdown product has a reserve facility it can call on; a lump-sum
#: product does not.
_SUBTYPE_PRODUCTS: Dict[str, List[str]] = {
    "lump_sum": ["Lump Sum"],
    "drawdown": ["Drawdown"],
    "mixed": ["Lump Sum", "Drawdown"],
}

#: Values taken verbatim from the governed enum vocabularies
#: (``config/system/enum_synonyms.yaml`` + ``enum_mapping.yaml``) so the ESMA
#: layer resolves them exactly rather than through a fuzzy guess.
_PROPERTY_TYPES = ["Detached house", "Semi-detached house", "House",
                   "Bungalow", "Flat"]


class EquityReleaseProfile:
    """Asset profile for equity release / lifetime mortgages."""

    asset_class = ASSET_EQUITY_RELEASE
    subtypes = ASSET_SUBTYPES[ASSET_EQUITY_RELEASE]
    canonical_fields = C.COMMON_CANONICAL_FIELDS + ERM_FIELDS
    product_profile_id = "equity_release_lifetime_mortgage"

    # ------------------------------------------------------------------ #
    def open_loan(self, rng, ctx: GenContext, *, loan_id: str,
                  origination_date: str, cohort: str) -> LoanState:
        case, knobs = ctx.case, ctx.knobs
        region = weighted_region(rng, knobs.region_weights)
        index = REGION_VALUE_INDEX[region]

        original_value = round(
            max(85_000.0, float(rng.normal(392_000.0, 116_000.0)) * index), 2)
        ltv = min(0.70, max(0.05, float(rng.normal(knobs.initial_ltv_mean,
                                                   knobs.initial_ltv_sd))))
        advance = round(original_value * ltv, 2)
        rate = round(max(2.0, float(rng.normal(knobs.rate_mean, knobs.rate_sd))), 3)

        products = _SUBTYPE_PRODUCTS.get(case.asset_subtype, ["Lump Sum"])
        product = products[int(rng.integers(0, len(products)))]
        age_at_origination = int(
            min(94, max(55, round(float(rng.normal(68.4, 5.2))))))

        loan = LoanState(
            loan_id=loan_id, status=STATUS_PERFORMING,
            opening_balance=advance, closing_balance=advance,
            origination_date=origination_date, funding_date=origination_date,
            original_principal=advance, interest_rate=rate,
            current_value=original_value, original_value=original_value,
            borrower_id="", region=region, postcode=postcode(rng, region),
            cohort=cohort,
        )
        loan.attributes.update({
            "erm_product_type": product,
            "age_at_origination": age_at_origination,
            "property_type": _PROPERTY_TYPES[int(rng.integers(0, len(_PROPERTY_TYPES)))],
            "occupancy_type": "Owner Occupied",
            "amortisation_type": "Other",
            "interest_rate_type": "Fixed",
            "valuation_method": "Full survey",
            "purpose": "Equity Release",
            "origination_channel": "Direct" if cohort == "direct" else "Broker",
            "collateral_type": "House",
            "monthly_payment_due": 0.0,          # roll-up: nothing contractually due
            "negative_equity_guarantee": "Y",     # NNEG is a product feature
            # ERM101: an NNEG product must state the LTV cap the guarantee bites at.
            "ltv_cap": 75.0,
            # Long-stop: the date the youngest borrower reaches 110. A lifetime
            # mortgage has no contractual repayment date, so this is what the
            # regulatory maturity field reports.
            "original_term_months": max(1, (110 - age_at_origination) * 12),
            # A drawdown product carries a reserve facility above the advance.
            "reserve_facility": (round(advance * 0.45, 2)
                                 if product == "Drawdown" else 0.0),
            "further_advance_amount": 0.0,
            "further_advance_flag": "N",
        })
        loan.attributes["credit_limit"] = round(
            advance + loan.attributes["reserve_facility"], 2)
        loan.attributes["maturity_date"] = C.maturity_from_term(
            origination_date, loan.attributes["original_term_months"])
        return loan

    # ------------------------------------------------------------------ #
    def roll(self, rng, loan: LoanState, ctx: GenContext) -> None:
        """One month: interest rolls up, the property revalues, exits happen."""
        knobs = ctx.knobs
        loan.attributes["pending_reporting_date"] = ctx.reporting_date

        # 1. Interest capitalises onto the balance — the defining ERM economic.
        interest = round(loan.closing_balance * C.annual_to_monthly(loan.interest_rate), 2)
        loan.closing_balance = round(loan.closing_balance + interest, 2)
        loan.record(MOVE_INTEREST, interest)

        # 2. A drawdown product may call on its reserve facility.
        reserve = float(loan.attributes.get("reserve_facility", 0.0))
        loan.attributes["further_advance_flag"] = "N"
        if reserve > 0 and float(rng.random()) < 0.020:
            tranche = round(min(reserve, max(5_000.0,
                                             float(rng.normal(24_000.0, 7_500.0)))), 2)
            loan.closing_balance = round(loan.closing_balance + tranche, 2)
            loan.attributes["reserve_facility"] = round(reserve - tranche, 2)
            loan.attributes["further_advance_amount"] = round(
                float(loan.attributes.get("further_advance_amount", 0.0)) + tranche, 2)
            loan.attributes["further_advance_flag"] = "Y"
            loan.record(MOVE_FURTHER_ADVANCE, tranche)

        # 3. Property revaluation (an adverse case supplies a negative growth).
        C.revalue(loan, knobs.valuation_monthly_growth)

        # 4. Performance ladder. Equity release rarely goes into arrears, but the
        #    ladder exists so a stress case can exercise it.
        C.advance_status(rng, loan, knobs)
        C.accrue_arrears(loan)

        # 5. Redemption: death, move into care, or voluntary sale. Older
        #    borrowers redeem more often, which is what makes the borrower-age
        #    stratification move directionally with the reporting date.
        age = self._age_at(loan, ctx.reporting_date)
        mortality_uplift = 1.0 + max(0, age - 75) * 0.06
        if float(rng.random()) < knobs.redemption_rate * mortality_uplift:
            C.redeem(loan, ctx.reporting_date, reason="redemption")

    @staticmethod
    def _age_at(loan: LoanState, reporting_date: str) -> int:
        elapsed_years = months_between(parse_iso(loan.origination_date),
                                       parse_iso(reporting_date)) // 12
        return int(loan.attributes["age_at_origination"]) + elapsed_years

    # ------------------------------------------------------------------ #
    def canonical_row(self, loan: LoanState, ctx: GenContext) -> Dict[str, Any]:
        row = C.common_canonical(loan, ctx.reporting_date, ctx.case)
        attrs = loan.attributes
        value = loan.current_value or 0.0
        row.update({
            "erm_product_type": attrs["erm_product_type"],
            "youngest_borrower_age": self._age_at(loan, ctx.reporting_date),
            "negative_equity_guarantee": attrs["negative_equity_guarantee"],
            "further_advance_amount": round(float(attrs.get("further_advance_amount", 0.0)), 2),
            "further_advance_flag": attrs.get("further_advance_flag", "N"),
            "ltv_cap": attrs["ltv_cap"],
            "loan_redemption_flag": "Y" if loan.exit_date else "N",
            "property_type": attrs["property_type"],
            "occupancy_type": attrs["occupancy_type"],
            # Percentage POINTS — the repository's canonical convention.
            "current_loan_to_value": round(loan.closing_balance / value * 100.0, 4) if value else 0.0,
            "original_loan_to_value": round(
                loan.original_principal / loan.original_value * 100.0, 4)
            if loan.original_value else 0.0,
        })
        return row

    # ------------------------------------------------------------------ #
    def risk_tests(self, case: CaseManifest) -> List[Dict[str, Any]]:
        """Governed concentration tests for an equity-release book.

        Every entry names a metric that already exists in
        ``config/risk/concentration_test_library.yaml``. Thresholds are chosen
        so a baseline case sits green and the declared stress case moves the
        named limit to amber or breach — nothing is hard-coded per client, and
        the thresholds travel with the case, not with the code.
        """
        stressed = case.risk_intent.intended_status in (RISK_AMBER, RISK_RED)
        targeted = set(case.risk_intent.targeted_limits)

        def threshold(limit: str, green: float, stress: float) -> float:
            return stress if (stressed and limit in targeted) else green

        return [
            {"test_id": "er_geo_largest_region", "metric_id": "geo_largest_region_share",
             "display_name": "Largest regional concentration",
             "category": "geography", "operator": "max",
             "threshold": threshold("geographic_concentration", 60.0, 45.0),
             "parameters": {"denominator": "current_balance"}},
            {"test_id": "er_property_value_top_exposure",
             "metric_id": "property_largest_exposure_share",
             "display_name": "Largest single property exposure",
             "category": "property_value", "operator": "max",
             "threshold": threshold("property_value_concentration", 12.0, 1.5),
             "parameters": {"denominator": "current_balance"}},
            {"test_id": "er_borrower_largest", "metric_id": "borrower_largest_share",
             "display_name": "Largest single-borrower exposure",
             "category": "borrower", "operator": "max",
             "threshold": threshold("borrower_concentration", 12.0, 2.5),
             "parameters": {"denominator": "current_balance"}},
            {"test_id": "er_borrower_age_over_85", "metric_id": "borrower_age_share",
             "display_name": "Exposure to borrowers aged over 85",
             "category": "borrower", "operator": "max",
             "threshold": threshold("age_concentration", 45.0, 12.0),
             "parameters": {"age": 85, "age_basis": "youngest",
                            "denominator": "current_balance"}},
            {"test_id": "er_rate_type_fixed", "metric_id": "rate_type_share",
             "display_name": "Fixed-rate exposure",
             "category": "rate_product",
             # A MINIMUM, not a maximum. The interest-rate risk in a roll-up
             # book is the VARIABLE-rate part: a fixed lifetime rate is what
             # makes the balance path knowable, so the covenant is a floor on
             # fixed-rate exposure. Written as a maximum, a wholly fixed-rate
             # book sat permanently at its own limit.
             "operator": "min",
             "threshold": threshold("interest_rate_structure", 50.0, 99.5),
             # The canonical transform normalises "Fixed" onto the governed
             # enum code, so the test names BOTH forms rather than assuming
             # which side of normalisation it is reading.
             "parameters": {"values": ["fixed", "fxrl"],
                            "denominator": "current_balance"}},
            {"test_id": "er_ltv_weighted_average", "metric_id": "ltv_weighted_average",
             "display_name": "Weighted-average current LTV",
             "category": "ltv", "operator": "max",
             # A roll-up book's LTV climbs for as long as it is on book:
             # interest capitalises faster than UK property appreciates. A
             # seasoned book at 75-85% is ordinary, so the covenant sits at 95%
             # and only the declared adverse-valuation case breaches it.
             "threshold": threshold("ltv_limit", 95.0, 40.0),
             "parameters": {"ltv_basis": "current"}},
        ]


PROFILE = register(EquityReleaseProfile())

__all__ = ["ERM_FIELDS", "EquityReleaseProfile", "PROFILE"]
