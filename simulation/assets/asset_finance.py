"""simulation.assets.asset_finance — funded asset-finance contracts.

Hire purchase, finance lease and secured asset loans over depreciating
collateral. **Equipment finance is a SUBTYPE here, not a separate platform**:
it is expressed through contract type, collateral category, manufacturer/vendor
configuration and residual-value behaviour, exactly as the task requires.

Runs under the production ``--portfolio-type equipment``, which selects the
existing equipment fields (``manufacturer``, ``model``,
``original_/current_residual_value_of_asset``, ``date_of_lease_expiration``,
``number_of_leased_objects``, ``year_of_manufacture_construction``) alongside the
``common`` superset (``balloon_amount``, ``collateral_type``, ``product_type``,
``nace_industry_code``, ``seller_name``, the payment-frequency fields). No new
canonical field was needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..dates import parse_iso
from ..geo import postcode, weighted_region
from ..models import (
    ASSET_FINANCE,
    ASSET_SUBTYPES,
    MOVE_AMORTISATION,
    MOVE_DISPOSAL,
    MOVE_INTEREST,
    RISK_AMBER,
    RISK_RED,
    STATUS_ENFORCEMENT,
    STATUS_PERFORMING,
    STATUS_REPOSSESSED,
    CaseManifest,
    LoanState,
)
from . import GenContext, register
from . import _common as C

#: Canonical fields specific to asset finance, on top of the common set.
ASSET_FINANCE_FIELDS = (
    "product_type",
    "loan_sub_type",
    "manufacturer",
    "model",
    "number_of_leased_objects",
    "year_of_manufacture_construction",
    "original_residual_value_of_asset",
    "current_residual_value_of_asset",
    "date_of_lease_expiration",
    "balloon_amount",
    "nace_industry_code",
    "scheduled_principal_payment_frequency",
    "scheduled_interest_payment_frequency",
    "current_loan_to_value",
    "original_loan_to_value",
    "prepayment_terms_description",
)

#: Contract types. ``product_type`` carries the canonical enum-ish label and
#: ``loan_sub_type`` the readable contract description.
CONTRACT_TYPES: Tuple[str, ...] = ("Hire Purchase", "Finance Lease", "Asset Loan")

#: Equipment categories, each with its manufacturers, a typical asset value and
#: a monthly depreciation multiplier relative to the case's baseline. This table
#: IS the equipment-finance configuration: adding a category is a data change.
EQUIPMENT_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "Construction Plant": {
        "manufacturers": ["Caterpillar", "JCB", "Komatsu", "Liebherr"],
        "value_mean": 148_000.0, "value_sd": 52_000.0, "depreciation": 1.00,
        "collateral_type": "Industrial", "leased_objects": (1, 3),
    },
    "Agricultural Machinery": {
        "manufacturers": ["John Deere", "New Holland", "Massey Ferguson"],
        "value_mean": 122_000.0, "value_sd": 41_000.0, "depreciation": 0.85,
        "collateral_type": "Industrial", "leased_objects": (1, 2),
    },
    "Commercial Vehicles": {
        "manufacturers": ["Mercedes-Benz", "Scania", "DAF", "Volvo"],
        "value_mean": 96_000.0, "value_sd": 28_000.0, "depreciation": 1.25,
        "collateral_type": "Other", "leased_objects": (1, 6),
    },
    "Materials Handling": {
        "manufacturers": ["Toyota Material Handling", "Linde", "Hyster"],
        "value_mean": 44_000.0, "value_sd": 14_000.0, "depreciation": 1.10,
        "collateral_type": "Industrial", "leased_objects": (1, 8),
    },
    "Medical Equipment": {
        "manufacturers": ["Siemens Healthineers", "GE HealthCare", "Philips"],
        "value_mean": 215_000.0, "value_sd": 78_000.0, "depreciation": 0.70,
        "collateral_type": "Other", "leased_objects": (1, 2),
    },
    "IT and Print": {
        "manufacturers": ["Ricoh", "Konica Minolta", "Dell Technologies"],
        "value_mean": 28_000.0, "value_sd": 9_000.0, "depreciation": 1.60,
        "collateral_type": "Other", "leased_objects": (2, 25),
    },
}

#: Which categories each subtype draws on. Equipment finance is the broad
#: equipment book; vehicle finance is the commercial-vehicle concentration.
SUBTYPE_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "equipment_finance": ("Construction Plant", "Agricultural Machinery",
                          "Materials Handling", "Medical Equipment",
                          "IT and Print"),
    "vehicle_finance": ("Commercial Vehicles",),
    "mixed_asset_finance": tuple(EQUIPMENT_CATEGORIES),
}

#: Vendors / dealers that introduce the business. Vendor concentration is a real
#: asset-finance limit, so the vendor is a first-class attribute.
VENDORS: Tuple[str, ...] = (
    "Northgate Plant Sales", "Ridgeway Equipment Ltd", "Calderwell Machinery",
    "Bramfield Asset Partners", "Stonecross Vendor Finance",
)

#: Lessee industries, carried on the canonical ``nace_industry_code`` field as
#: readable NACE section labels.
INDUSTRIES: Tuple[str, ...] = (
    "F - Construction", "A - Agriculture", "H - Transport and Storage",
    "C - Manufacturing", "Q - Human Health", "G - Wholesale and Retail",
)

PAYMENT_FREQUENCIES: Tuple[str, ...] = ("Monthly", "Quarterly")

#: The REPORTED status tokens that mean "delinquent", derived from the single
#: mapping so a change there cannot silently narrow a covenant.
_DELINQUENT_STATUSES = sorted({C.REPORTED_STATUS[s].lower()
                               for s in ("arrears", "default", "enforcement")})


class AssetFinanceProfile:
    """Asset profile for funded asset finance, including equipment finance."""

    asset_class = ASSET_FINANCE
    subtypes = ASSET_SUBTYPES[ASSET_FINANCE]
    canonical_fields = C.COMMON_CANONICAL_FIELDS + ASSET_FINANCE_FIELDS
    product_profile_id = "asset_finance_contract"

    # ------------------------------------------------------------------ #
    def open_loan(self, rng, ctx: GenContext, *, loan_id: str,
                  origination_date: str, cohort: str) -> LoanState:
        case, knobs = ctx.case, ctx.knobs
        region = weighted_region(rng, knobs.region_weights)

        categories = SUBTYPE_CATEGORIES.get(case.asset_subtype,
                                            SUBTYPE_CATEGORIES["mixed_asset_finance"])
        category = categories[int(rng.integers(0, len(categories)))]
        spec = EQUIPMENT_CATEGORIES[category]
        manufacturers = spec["manufacturers"]
        manufacturer = manufacturers[int(rng.integers(0, len(manufacturers)))]

        asset_value = round(max(6_000.0, float(rng.normal(spec["value_mean"],
                                                          spec["value_sd"]))), 2)
        advance_fraction = min(1.0, max(0.35, float(rng.normal(
            knobs.initial_ltv_mean, knobs.initial_ltv_sd))))
        financed = round(asset_value * advance_fraction, 2)
        rate = round(max(3.0, float(rng.normal(knobs.rate_mean, knobs.rate_sd))), 3)

        contract = CONTRACT_TYPES[int(rng.integers(0, len(CONTRACT_TYPES)))]
        term = int(rng.integers(24, 73))
        has_balloon = float(rng.random()) < knobs.balloon_share
        balloon = round(financed * knobs.balloon_fraction, 2) if has_balloon else 0.0
        # A finance lease always carries a residual; HP and loans only do when a
        # balloon was written.
        residual = (round(asset_value * 0.22, 2) if contract == "Finance Lease"
                    else balloon)

        objects_lo, objects_hi = spec["leased_objects"]
        loan = LoanState(
            loan_id=loan_id, status=STATUS_PERFORMING,
            opening_balance=financed, closing_balance=financed,
            origination_date=origination_date, funding_date=origination_date,
            original_principal=financed, interest_rate=rate,
            current_value=asset_value, original_value=asset_value,
            borrower_id="", region=region, postcode=postcode(rng, region),
            cohort=cohort,
        )
        # Straight-line capital repayment down to the balloon over the term. Kept
        # deliberately simple: a full annuity schedule would be a cashflow
        # waterfall, which is out of scope.
        repayable = max(0.0, financed - balloon)
        monthly_capital = round(repayable / term, 2) if term else 0.0
        monthly_interest = round(financed * C.annual_to_monthly(rate), 2)

        loan.attributes.update({
            "contract_type": contract,
            "product_type": "Other",
            "loan_sub_type": contract,
            "equipment_category": category,
            "manufacturer": manufacturer,
            "model": f"{manufacturer.split()[0][:4].upper()}-{int(rng.integers(100, 999))}",
            "vendor_name": VENDORS[int(rng.integers(0, len(VENDORS)))],
            "lessee_industry": INDUSTRIES[int(rng.integers(0, len(INDUSTRIES)))],
            "collateral_type": spec["collateral_type"],
            "number_of_leased_objects": int(rng.integers(objects_lo, objects_hi + 1)),
            "year_of_manufacture": int(parse_iso(origination_date).year
                                       - int(rng.integers(0, 3))),
            "depreciation_multiplier": float(spec["depreciation"]),
            "original_residual_value": residual,
            "current_residual_value": residual,
            "balloon_amount": balloon,
            "original_term_months": term,
            "monthly_capital": monthly_capital,
            "monthly_payment_due": round(monthly_capital + monthly_interest, 2),
            "payment_frequency": PAYMENT_FREQUENCIES[
                0 if float(rng.random()) < 0.86 else 1],
            "amortisation_type": "French",
            "interest_rate_type": "Fixed",
            "valuation_method": "automated valuation",
            "purpose": "Business finance",
            "origination_channel": "Other",
            "originator_name": "Simulated Asset Finance Ltd",
            "seller_name": VENDORS[int(rng.integers(0, len(VENDORS)))],
            "credit_limit": financed,
            "disposal_proceeds": 0.0,
        })
        loan.attributes["seller_name"] = loan.attributes["vendor_name"]
        loan.attributes["maturity_date"] = C.maturity_from_term(origination_date, term)
        loan.attributes["lease_expiry_date"] = loan.attributes["maturity_date"]
        return loan

    # ------------------------------------------------------------------ #
    def roll(self, rng, loan: LoanState, ctx: GenContext) -> None:
        knobs = ctx.knobs
        attrs = loan.attributes
        attrs["pending_reporting_date"] = ctx.reporting_date

        # 1. Interest accrues and is paid with the instalment, so it does not
        #    capitalise. Only capital repayment reduces the exposure.
        capital = min(loan.closing_balance - float(attrs.get("balloon_amount", 0.0)),
                      float(attrs.get("monthly_capital", 0.0)))
        capital = round(max(0.0, capital), 2)
        if loan.status == STATUS_PERFORMING and capital:
            loan.closing_balance = round(loan.closing_balance - capital, 2)
            loan.record(MOVE_AMORTISATION, capital)
        elif loan.status != STATUS_PERFORMING:
            # An impaired contract stops paying; interest is then capitalised,
            # which is what makes a delinquent book's balance drift upward.
            interest = round(loan.closing_balance
                             * C.annual_to_monthly(loan.interest_rate), 2)
            loan.closing_balance = round(loan.closing_balance + interest, 2)
            loan.record(MOVE_INTEREST, interest)

        # 2. The asset depreciates. ``asset_monthly_depreciation`` is expressed
        #    as a positive rate of decline, so it enters revaluation negated.
        C.revalue(loan, -abs(knobs.asset_monthly_depreciation)
                  * float(attrs.get("depreciation_multiplier", 1.0)))
        attrs["current_residual_value"] = round(
            min(float(attrs.get("original_residual_value", 0.0)),
                loan.current_value * 0.85), 2)

        # 3. Performance ladder.
        C.advance_status(rng, loan, knobs)
        C.accrue_arrears(loan)

        # 4. Repossession and disposal: the asset is taken and sold, the
        #    proceeds are applied, and the shortfall is written off.
        if loan.status == STATUS_ENFORCEMENT and float(rng.random()) < knobs.repossession_rate:
            proceeds = round(loan.current_value * knobs.disposal_fraction, 2)
            attrs["disposal_proceeds"] = proceeds
            # The proceeds are reported as an ASSET DISPOSAL rather than a
            # recovery: that is what an asset-finance loss table calls them.
            C.write_off(loan, ctx.reporting_date, knobs, proceeds=proceeds,
                        reason="repossession_and_disposal",
                        status=STATUS_REPOSSESSED,
                        proceeds_kind=MOVE_DISPOSAL)
            return

        # 5. Contract maturity: the balloon or residual is settled and the
        #    contract closes.
        remaining = C.months_to_maturity(ctx.reporting_date, attrs["maturity_date"])
        attrs["months_to_maturity"] = remaining
        if remaining <= 0 and loan.is_open:
            C.redeem(loan, ctx.reporting_date, reason="contract_maturity")
            return

        # 6. Early settlement.
        if loan.is_open and float(rng.random()) < knobs.redemption_rate:
            C.redeem(loan, ctx.reporting_date, reason="early_settlement")

    # ------------------------------------------------------------------ #
    def canonical_row(self, loan: LoanState, ctx: GenContext) -> Dict[str, Any]:
        row = C.common_canonical(loan, ctx.reporting_date, ctx.case)
        attrs = loan.attributes
        value = loan.current_value or 0.0
        row.update({
            "product_type": attrs["product_type"],
            "loan_sub_type": attrs["loan_sub_type"],
            "manufacturer": attrs["manufacturer"],
            "model": attrs["model"],
            "number_of_leased_objects": int(attrs["number_of_leased_objects"]),
            "year_of_manufacture_construction": int(attrs["year_of_manufacture"]),
            "original_residual_value_of_asset": round(
                float(attrs["original_residual_value"]), 2),
            "current_residual_value_of_asset": round(
                float(attrs["current_residual_value"]), 2),
            "date_of_lease_expiration": attrs["lease_expiry_date"],
            "balloon_amount": round(float(attrs["balloon_amount"]), 2),
            "nace_industry_code": attrs["lessee_industry"],
            "scheduled_principal_payment_frequency": attrs["payment_frequency"],
            "scheduled_interest_payment_frequency": attrs["payment_frequency"],
            "current_loan_to_value": round(loan.closing_balance / value * 100.0, 4) if value else 0.0,
            "original_loan_to_value": round(
                loan.original_principal / loan.original_value * 100.0, 4)
            if loan.original_value else 0.0,
            "prepayment_terms_description": (
                f"{attrs['contract_type']}; category {attrs['equipment_category']}; "
                f"vendor {attrs['vendor_name']}; disposal proceeds "
                f"{attrs.get('disposal_proceeds', 0.0):.2f}"),
        })
        return row

    # ------------------------------------------------------------------ #
    def risk_tests(self, case: CaseManifest) -> List[Dict[str, Any]]:
        """Governed concentration tests for an asset-finance book.

        Industry, vendor, manufacturer and collateral-type concentration all run
        through the existing ``composition_*`` metrics — the governed library
        gained the field roles and dimension values, not new bespoke evaluators.
        """
        stressed = case.risk_intent.intended_status in (RISK_AMBER, RISK_RED)
        targeted = set(case.risk_intent.targeted_limits)

        def threshold(limit: str, green: float, stress: float) -> float:
            return stress if (stressed and limit in targeted) else green

        return [
            {"test_id": "af_industry_largest",
             "metric_id": "composition_largest_group_share",
             "display_name": "Largest lessee-industry exposure",
             "category": "composition", "operator": "max",
             "threshold": threshold("industry_concentration", 60.0, 22.0),
             "parameters": {"dimension": "industry", "denominator": "current_balance"}},
            {"test_id": "af_vendor_largest",
             "metric_id": "composition_largest_group_share",
             "display_name": "Largest vendor exposure",
             "category": "composition", "operator": "max",
             "threshold": threshold("vendor_concentration", 60.0, 24.0),
             "parameters": {"dimension": "vendor", "denominator": "current_balance"}},
            {"test_id": "af_manufacturer_largest",
             "metric_id": "composition_largest_group_share",
             "display_name": "Largest manufacturer exposure",
             "category": "composition", "operator": "max",
             "threshold": threshold("manufacturer_concentration", 60.0, 22.0),
             "parameters": {"dimension": "manufacturer",
                            "denominator": "current_balance"}},
            {"test_id": "af_collateral_type_largest",
             "metric_id": "composition_largest_group_share",
             "display_name": "Largest collateral-type exposure",
             "category": "composition", "operator": "max",
             "threshold": threshold("collateral_type_concentration", 92.0, 45.0),
             "parameters": {"dimension": "collateral_type",
                            "denominator": "current_balance"}},
            {"test_id": "af_balloon_exposure",
             "metric_id": "residual_value_exposure_share",
             "display_name": "Balloon and residual-value exposure",
             "category": "residual_value", "operator": "max",
             "threshold": threshold("balloon_residual_concentration", 60.0, 14.0),
             "parameters": {"residual_basis": "balloon",
                            "denominator": "current_balance"}},
            {"test_id": "af_delinquency_share", "metric_id": "perf_status_share",
             "display_name": "Delinquent contracts",
             "category": "performance", "operator": "max",
             # A delinquency covenant on an SME asset-finance book bites early:
             # 5% of the book past due is already a servicing problem.
             "threshold": threshold("delinquency_concentration", 60.0, 5.0),
             "parameters": {"values": _DELINQUENT_STATUSES,
                            "denominator": "current_balance"}},
            {"test_id": "af_borrower_largest", "metric_id": "borrower_largest_share",
             "display_name": "Largest single-lessee exposure",
             "category": "borrower", "operator": "max",
             "threshold": threshold("borrower_concentration", 25.0, 5.0),
             "parameters": {"denominator": "current_balance"}},
        ]


PROFILE = register(AssetFinanceProfile())

__all__ = ["ASSET_FINANCE_FIELDS", "AssetFinanceProfile", "CONTRACT_TYPES",
           "EQUIPMENT_CATEGORIES", "INDUSTRIES", "PROFILE", "SUBTYPE_CATEGORIES",
           "VENDORS"]
