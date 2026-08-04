"""simulation.assets.bridge — short-term secured bridge lending.

A funded bridge book: short original terms, first or second charge over
residential, commercial or mixed collateral, interest either retained, serviced
or rolled up, and an exit route (refinance / sale / development exit) that is
the loan's actual repayment mechanism. Maturity is contractual and near, which
is what makes a maturity wall, an extension and an enforcement genuine events
rather than decorations.

Runs under the production ``--portfolio-type cre``. Every canonical field used
here is already in ``config/system/fields_registry.yaml`` — ``charge_type``,
``lien``, ``maturity_date``, ``original_term`` and both LTVs are all
``portfolio_type: common``, so no registry change was needed.

Development finance is deliberately out of scope for this phase: it needs a
drawdown-schedule and cost-to-complete model the bridge schema does not carry.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..dates import add_month_ends, parse_iso
from ..geo import REGION_VALUE_INDEX, postcode, weighted_index, weighted_region
from ..models import (
    ASSET_BRIDGE,
    ASSET_SUBTYPES,
    MOVE_INTEREST,
    MOVE_REPAYMENT,
    RISK_AMBER,
    RISK_RED,
    STATUS_ARREARS,
    STATUS_ENFORCEMENT,
    STATUS_PERFORMING,
    CaseManifest,
    LoanState,
)
from . import GenContext, register
from . import _common as C

#: Canonical fields specific to bridge lending, on top of the common set.
BRIDGE_FIELDS = (
    "charge_type",
    "lien",
    "property_type",
    "occupancy_type",
    "loan_sub_type",
    "product_type",
    "scheduled_interest_payment_frequency",
    "current_loan_to_value",
    "original_loan_to_value",
    "prepayment_terms_description",
)

#: Collateral mix by subtype: (collateral_type, property_type, weight).
#: Values are taken verbatim from the governed enum vocabularies, so canonical
#: validation and the ESMA layer resolve them exactly rather than by fuzzy match.
_SUBTYPE_COLLATERAL: Dict[str, List[tuple]] = {
    "residential_bridge": [("House", "Detached house", 0.30),
                           ("House", "Semi-detached house", 0.30),
                           ("Flat", "Flat", 0.40)],
    "commercial_bridge": [("Office", "Office", 0.30),
                          ("Retail", "Retail", 0.35),
                          ("Industrial", "Industrial", 0.35)],
    "mixed_bridge": [("House", "Detached house", 0.25),
                     ("Flat", "Flat", 0.25),
                     ("Retail", "Retail", 0.25),
                     ("Mixed_use", "Mixed_use", 0.25)],
}

#: How the interest is handled. This is a real bridge product distinction and it
#: changes the balance path: retained and rolled-up interest capitalise, serviced
#: interest is paid monthly and never touches the principal.
INTEREST_STRUCTURES = ("Retained", "Serviced", "Rolled Up")

#: The loan's actual repayment mechanism.
EXIT_ROUTES = ("Refinance", "Sale of Security", "Development Exit",
               "Sale of Other Asset")

#: The REPORTED status tokens that mean "impaired", derived from the single
#: mapping so a change there cannot silently narrow a covenant.
_IMPAIRED_STATUSES = sorted({C.REPORTED_STATUS[s].lower()
                             for s in ("arrears", "default", "enforcement")})


class BridgeProfile:
    """Asset profile for funded bridge lending."""

    asset_class = ASSET_BRIDGE
    subtypes = ASSET_SUBTYPES[ASSET_BRIDGE]
    canonical_fields = C.COMMON_CANONICAL_FIELDS + BRIDGE_FIELDS
    product_profile_id = "bridge_short_term_secured"

    # ------------------------------------------------------------------ #
    def open_loan(self, rng, ctx: GenContext, *, loan_id: str,
                  origination_date: str, cohort: str) -> LoanState:
        case, knobs = ctx.case, ctx.knobs
        region = weighted_region(rng, knobs.region_weights)
        index = REGION_VALUE_INDEX[region]

        mix = _SUBTYPE_COLLATERAL.get(case.asset_subtype,
                                      _SUBTYPE_COLLATERAL["mixed_bridge"])
        collateral_type, property_type, _weight = mix[
            weighted_index(rng, [w for _, _, w in mix])]

        original_value = round(
            max(150_000.0, float(rng.normal(1_050_000.0, 420_000.0)) * index), 2)
        second_charge = float(rng.random()) < knobs.second_charge_share
        # A second charge sits behind a senior lender, so the advance is smaller
        # relative to the security and the charge rank is 2.
        ltv = min(0.80, max(0.10, float(rng.normal(
            knobs.initial_ltv_mean * (0.55 if second_charge else 1.0),
            knobs.initial_ltv_sd))))
        advance = round(original_value * ltv, 2)
        rate = round(max(4.0, float(rng.normal(
            knobs.rate_mean + (1.9 if second_charge else 0.0), knobs.rate_sd))), 3)

        term = int(rng.integers(knobs.original_term_months[0],
                                knobs.original_term_months[1] + 1))
        structure = INTEREST_STRUCTURES[int(rng.integers(0, len(INTEREST_STRUCTURES)))]
        exit_route = EXIT_ROUTES[int(rng.integers(0, len(EXIT_ROUTES)))]

        loan = LoanState(
            loan_id=loan_id, status=STATUS_PERFORMING,
            opening_balance=advance, closing_balance=advance,
            origination_date=origination_date, funding_date=origination_date,
            original_principal=advance, interest_rate=rate,
            current_value=original_value, original_value=original_value,
            borrower_id="", region=region, postcode=postcode(rng, region),
            cohort=cohort,
        )
        monthly_interest = round(advance * C.annual_to_monthly(rate), 2)
        loan.attributes.update({
            "charge_type": "Second Charge" if second_charge else "First Charge",
            "lien": 2 if second_charge else 1,
            "collateral_type": collateral_type,
            "property_type": property_type,
            "occupancy_type": ("Owner Occupied"
                               if collateral_type in ("House", "Flat")
                               else "Tenanted"),
            "interest_structure": structure,
            "exit_route": exit_route,
            "loan_sub_type": f"Bridge - {exit_route}",
            "product_type": "Other",  # canonical product_type enum
            "amortisation_type": "Bullet",
            "interest_rate_type": "Fixed",
            "valuation_method": "Full survey",
            "purpose": "Refinance",
            "origination_channel": "Broker" if cohort == "direct" else "Other",
            "original_term_months": term,
            "extensions_granted": 0,
            # Serviced interest is contractually due each month; retained and
            # rolled-up interest are not.
            "monthly_payment_due": monthly_interest if structure == "Serviced" else 0.0,
            "scheduled_interest_payment_frequency": (
                "Monthly" if structure == "Serviced" else "Bullet"),
            "credit_limit": advance,
        })
        loan.attributes["maturity_date"] = C.maturity_from_term(origination_date, term)
        return loan

    # ------------------------------------------------------------------ #
    def roll(self, rng, loan: LoanState, ctx: GenContext) -> None:
        knobs = ctx.knobs
        loan.attributes["pending_reporting_date"] = ctx.reporting_date
        structure = loan.attributes["interest_structure"]

        # 1. Interest. Serviced interest is paid away; retained and rolled-up
        #    interest capitalises onto the balance.
        interest = round(loan.closing_balance * C.annual_to_monthly(loan.interest_rate), 2)
        if structure == "Serviced":
            # Cash interest never enters the exposure, so no movement is
            # recorded against the balance — recording one would break the
            # opening + movements == closing identity.
            pass
        else:
            loan.closing_balance = round(loan.closing_balance + interest, 2)
            loan.record(MOVE_INTEREST, interest)

        # 2. Security revaluation.
        C.revalue(loan, knobs.valuation_monthly_growth)

        # 3. Performance ladder.
        C.advance_status(rng, loan, knobs)
        C.accrue_arrears(loan)

        # 4. Maturity handling. Within the last month of term the loan either
        #    exits by its planned route, is extended, or rolls past maturity and
        #    starts down the enforcement ladder.
        remaining = C.months_to_maturity(ctx.reporting_date,
                                         loan.attributes["maturity_date"])
        loan.attributes["months_to_maturity"] = remaining
        if remaining <= 0:
            if float(rng.random()) < knobs.extension_rate:
                self._extend(loan, rng)
            elif float(rng.random()) < 0.55:
                C.redeem(loan, ctx.reporting_date, reason="exit_route")
                return
            else:
                # Past maturity and not extended: this is a default event.
                if loan.status == STATUS_PERFORMING:
                    loan.status = STATUS_ARREARS
                    loan.attributes["arrears_months"] = max(
                        1, int(loan.attributes.get("arrears_months", 0)))
        elif remaining <= 3 and float(rng.random()) < knobs.extension_rate:
            self._extend(loan, rng)

        # 5. Early exit before maturity (refinance away, sale completes early).
        if loan.is_open and float(rng.random()) < knobs.redemption_rate:
            C.redeem(loan, ctx.reporting_date, reason="early_exit")
            return

        # 6. Enforcement realises the security.
        if loan.status == STATUS_ENFORCEMENT and float(rng.random()) < knobs.write_off_rate:
            proceeds = round(loan.current_value * 0.78, 2)
            C.write_off(loan, ctx.reporting_date, knobs, proceeds=proceeds,
                        reason="security_realisation")
            return

        # 7. Partial repayment: a borrower selling one of several units.
        if (loan.is_open and structure != "Serviced"
                and float(rng.random()) < 0.010 and loan.closing_balance > 80_000):
            repayment = round(loan.closing_balance * 0.20, 2)
            loan.closing_balance = round(loan.closing_balance - repayment, 2)
            loan.record(MOVE_REPAYMENT, repayment)

    @staticmethod
    def _extend(loan: LoanState, rng) -> None:
        """Grant a contractual extension: maturity moves, the fact is recorded."""
        months = int(rng.integers(3, 7))
        loan.attributes["maturity_date"] = add_month_ends(
            parse_iso(loan.attributes["maturity_date"]), months).isoformat()
        loan.attributes["extensions_granted"] = int(
            loan.attributes.get("extensions_granted", 0)) + 1
        loan.attributes["original_term_months"] = int(
            loan.attributes["original_term_months"])  # ORIGINAL term is immutable
        loan.attributes["loan_sub_type"] = (
            f"Bridge - {loan.attributes['exit_route']} (Extended)")

    # ------------------------------------------------------------------ #
    def canonical_row(self, loan: LoanState, ctx: GenContext) -> Dict[str, Any]:
        row = C.common_canonical(loan, ctx.reporting_date, ctx.case)
        attrs = loan.attributes
        value = loan.current_value or 0.0
        row.update({
            "charge_type": attrs["charge_type"],
            "lien": int(attrs["lien"]),
            "property_type": attrs["property_type"],
            "occupancy_type": attrs["occupancy_type"],
            "loan_sub_type": attrs["loan_sub_type"],
            "product_type": attrs["product_type"],
            "scheduled_interest_payment_frequency":
                attrs["scheduled_interest_payment_frequency"],
            "current_loan_to_value": round(loan.closing_balance / value * 100.0, 4) if value else 0.0,
            "original_loan_to_value": round(
                loan.original_principal / loan.original_value * 100.0, 4)
            if loan.original_value else 0.0,
            "prepayment_terms_description": (
                f"Exit route: {attrs['exit_route']}; interest {attrs['interest_structure'].lower()}; "
                f"{attrs.get('extensions_granted', 0)} extension(s) granted"),
        })
        return row

    # ------------------------------------------------------------------ #
    def risk_tests(self, case: CaseManifest) -> List[Dict[str, Any]]:
        """Governed concentration tests for a bridge book.

        ``bridge_maturity_within_90_days`` and ``bridge_extension_share`` use the
        two metrics added to the governed library for short-dated secured
        lending; everything else reuses the cross-asset metrics that already
        existed.
        """
        stressed = case.risk_intent.intended_status in (RISK_AMBER, RISK_RED)
        targeted = set(case.risk_intent.targeted_limits)

        def threshold(limit: str, green: float, stress: float) -> float:
            return stress if (stressed and limit in targeted) else green

        return [
            {"test_id": "br_geo_largest_region", "metric_id": "geo_largest_region_share",
             "display_name": "Largest regional concentration",
             "category": "geography", "operator": "max",
             "threshold": threshold("geographic_concentration", 60.0, 45.0),
             "parameters": {"denominator": "current_balance"}},
            {"test_id": "br_borrower_largest", "metric_id": "borrower_largest_share",
             "display_name": "Largest single-borrower exposure",
             "category": "borrower", "operator": "max",
             "threshold": threshold("single_borrower_concentration", 25.0, 4.0),
             "parameters": {"denominator": "current_balance"}},
            {"test_id": "br_top_5_loans", "metric_id": "top_n_loans_share",
             "display_name": "Top 5 loans by exposure",
             "category": "loan_balance", "operator": "max",
             "threshold": threshold("loan_size_concentration", 70.0, 22.0),
             "parameters": {"n": 5, "denominator": "current_balance"}},
            {"test_id": "br_maturity_90_days",
             "metric_id": "maturity_within_horizon_share",
             "display_name": "Balance maturing within 90 days",
             "category": "maturity", "operator": "max",
             "threshold": threshold("maturity_concentration", 85.0, 30.0),
             "parameters": {"horizon_days": 90, "denominator": "current_balance"}},
            {"test_id": "br_extension_share", "metric_id": "extension_share",
             "display_name": "Extended loans",
             "category": "maturity", "operator": "max",
             "threshold": threshold("extension_concentration", 80.0, 20.0),
             "parameters": {"denominator": "current_balance"}},
            {"test_id": "br_impaired_share", "metric_id": "perf_status_share",
             "display_name": "Arrears, default and enforcement",
             "category": "performance", "operator": "max",
             "threshold": threshold("arrears_concentration", 60.0, 8.0),
             # The tape's REPORTED status tokens, not the simulator's internal
             # ladder names: enforcement is reported as DEFAULT with the
             # litigation flag set (see assets._common.REPORTED_STATUS).
             "parameters": {"values": _IMPAIRED_STATUSES,
                            "denominator": "current_balance"}},
            {"test_id": "br_second_charge_share",
             "metric_id": "composition_dimension_share",
             "display_name": "Second-charge exposure",
             "category": "composition", "operator": "max",
             "threshold": threshold("charge_rank_concentration", 60.0, 20.0),
             "parameters": {"dimension": "charge_type",
                            "values": ["second charge"],
                            "denominator": "current_balance"}},
        ]


PROFILE = register(BridgeProfile())

__all__ = ["BRIDGE_FIELDS", "BridgeProfile", "EXIT_ROUTES", "INTEREST_STRUCTURES",
           "PROFILE"]
