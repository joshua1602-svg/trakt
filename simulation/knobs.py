"""simulation.knobs — the economic knobs a case's declared intent resolves to.

A manifest states *intent* in words (``"redemption_heavy"``,
``"maturity_wall"``); this module turns that into explicit numbers. Keeping the
translation here means a manifest never carries magic constants and a reviewer
can see, in one place, exactly what "extension heavy" is engineered to do.

Every intent is a closed, named modifier. There is no free-form knob override
in a manifest: an unrecognised intent raises rather than silently doing nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

from .geo import REGIONS
from .models import (
    ASSET_BRIDGE,
    ASSET_EQUITY_RELEASE,
    ASSET_FINANCE,
)

#: Even regional spread — the baseline every concentration intent departs from.
_EVEN_REGIONS: Dict[str, float] = {r: 1.0 / len(REGIONS) for r in REGIONS}


@dataclass(frozen=True)
class Knobs:
    """Deterministic economic parameters for one case.

    Rates are MONTHLY probabilities per open loan unless stated otherwise.
    """

    # -- volume ---------------------------------------------------------- #
    #: New loans funded each month, as a fraction of the opening book.
    new_business_rate: float = 0.020
    #: Loans acquired (bought in) each month, as a fraction of the opening book.
    acquisition_rate: float = 0.0
    #: Share of the OPENING book that is an acquired cohort rather than direct.
    opening_acquired_share: float = 0.0

    # -- pricing / seasoning --------------------------------------------- #
    rate_mean: float = 6.40                  # annual percentage points
    rate_sd: float = 0.60
    initial_ltv_mean: float = 0.35           # fraction of collateral value
    initial_ltv_sd: float = 0.07
    #: Months of seasoning on the OPENING book, drawn uniformly.
    seasoning_months: Tuple[int, int] = (6, 84)
    #: Monthly collateral revaluation. Negative models an adverse market.
    valuation_monthly_growth: float = 0.0009

    # -- exits + performance --------------------------------------------- #
    redemption_rate: float = 0.006
    arrears_entry_rate: float = 0.004
    arrears_cure_rate: float = 0.25
    default_rate: float = 0.15               # of loans already in arrears
    enforcement_rate: float = 0.20           # of loans already in default
    write_off_rate: float = 0.10             # of loans in enforcement
    recovery_fraction: float = 0.55          # of the written-off balance

    # -- amortisation ----------------------------------------------------- #
    #: ``rollup`` (capitalising), ``bullet`` (interest serviced, principal at
    #: maturity) or ``amortising`` (scheduled principal repayment).
    amortisation_mode: str = "rollup"
    #: Monthly scheduled principal repayment as a fraction of opening balance,
    #: used only when ``amortisation_mode == "amortising"``.
    amortisation_rate: float = 0.0

    # -- geography / concentration ---------------------------------------- #
    region_weights: Dict[str, float] = field(
        default_factory=lambda: dict(_EVEN_REGIONS))
    #: Number of distinct borrowers per 100 loans. Lower ⇒ more borrower
    #: concentration (one obligor holding several facilities).
    borrowers_per_hundred: int = 100
    #: Multiplier applied to the largest few loans, to create size concentration.
    large_loan_multiplier: float = 1.0
    large_loan_count: int = 0

    # -- bridge ------------------------------------------------------------ #
    original_term_months: Tuple[int, int] = (9, 18)
    #: Monthly probability that a loan within 3 months of maturity is extended.
    extension_rate: float = 0.15
    #: Share of loans written on a second charge.
    second_charge_share: float = 0.0

    # -- asset finance ----------------------------------------------------- #
    #: Share of contracts carrying a balloon / residual-value payment.
    balloon_share: float = 0.0
    #: Balloon as a fraction of the financed amount, when present.
    balloon_fraction: float = 0.25
    #: Monthly probability that a defaulted contract is repossessed.
    repossession_rate: float = 0.30
    #: Disposal proceeds as a fraction of the asset's current value.
    disposal_fraction: float = 0.62
    #: Monthly asset depreciation (asset finance overrides the property growth).
    asset_monthly_depreciation: float = 0.012

    # -- restatement -------------------------------------------------------- #
    #: Period index whose PRIOR period is restated, or ``-1`` for none.
    restatement_period: int = -1
    #: Fraction of the restated period's balance the correction moves.
    restatement_fraction: float = 0.0

    def with_(self, **changes) -> "Knobs":
        return replace(self, **changes)


# --------------------------------------------------------------------------- #
# Per-asset baselines
# --------------------------------------------------------------------------- #
_BASELINES: Dict[str, Knobs] = {
    ASSET_EQUITY_RELEASE: Knobs(
        new_business_rate=0.018, rate_mean=6.45, rate_sd=0.65,
        initial_ltv_mean=0.335, initial_ltv_sd=0.075,
        seasoning_months=(6, 120), valuation_monthly_growth=0.0009,
        redemption_rate=0.0075,
        # A roll-up lifetime mortgage has NOTHING contractually due before
        # redemption, so it cannot fall into arrears. Modelling an arrears rate
        # here would invent a payment obligation the product does not have.
        arrears_entry_rate=0.0,
        amortisation_mode="rollup",
    ),
    ASSET_BRIDGE: Knobs(
        new_business_rate=0.085, rate_mean=10.75, rate_sd=1.20,
        initial_ltv_mean=0.62, initial_ltv_sd=0.09,
        seasoning_months=(1, 12), valuation_monthly_growth=0.0006,
        redemption_rate=0.070, arrears_entry_rate=0.010,
        amortisation_mode="rollup", original_term_months=(9, 18),
        extension_rate=0.12, second_charge_share=0.0,
    ),
    ASSET_FINANCE: Knobs(
        new_business_rate=0.045, rate_mean=8.20, rate_sd=1.05,
        initial_ltv_mean=0.85, initial_ltv_sd=0.06,
        seasoning_months=(1, 42), valuation_monthly_growth=-0.012,
        redemption_rate=0.008, arrears_entry_rate=0.012,
        amortisation_mode="amortising", amortisation_rate=0.021,
        balloon_share=0.0, repossession_rate=0.25,
        asset_monthly_depreciation=0.012,
    ),
}


def _skewed(primary: str, share: float) -> Dict[str, float]:
    """Region weights with *primary* carrying *share* and the rest even."""
    rest = [r for r in REGIONS if r != primary]
    remainder = (1.0 - share) / len(rest)
    weights = {primary: share}
    weights.update({r: remainder for r in rest})
    return weights


# --------------------------------------------------------------------------- #
# Intent modifiers (closed vocabulary)
# --------------------------------------------------------------------------- #
INTENT_MODIFIERS: Dict[str, Dict] = {
    # ---- shared ---------------------------------------------------------- #
    "clean_performing": dict(arrears_entry_rate=0.0, default_rate=0.0,
                             enforcement_rate=0.0, write_off_rate=0.0),
    "new_origination": dict(seasoning_months=(1, 18), new_business_rate=0.035),
    "seasoned_book": dict(seasoning_months=(60, 180), new_business_rate=0.004),
    "mixed_cohorts": dict(opening_acquired_share=0.42, acquisition_rate=0.006),
    "redemption_heavy": dict(redemption_rate=0.032, new_business_rate=0.006),
    "geographic_concentration_stress": dict(
        region_weights=_skewed("South East", 0.52)),
    "borrower_concentration_stress": dict(borrowers_per_hundred=34),
    "loan_size_concentration_stress": dict(large_loan_multiplier=9.0,
                                           large_loan_count=6),
    "arrears_transitions": dict(arrears_entry_rate=0.030, arrears_cure_rate=0.18,
                                default_rate=0.28, enforcement_rate=0.30,
                                write_off_rate=0.16),
    "controlled_restatement": dict(restatement_period=3,
                                   restatement_fraction=0.015),

    # ---- equity release --------------------------------------------------- #
    "rolled_up_interest": dict(amortisation_mode="rollup", rate_mean=6.95),
    "high_ltv_adverse_valuation": dict(initial_ltv_mean=0.52,
                                       initial_ltv_sd=0.085,
                                       valuation_monthly_growth=-0.0085,
                                       seasoning_months=(48, 168)),

    # ---- bridge ----------------------------------------------------------- #
    "mixed_charge_rank": dict(second_charge_share=0.34),
    "extension_heavy": dict(extension_rate=0.62, redemption_rate=0.020,
                            original_term_months=(6, 12)),
    "maturity_wall": dict(original_term_months=(3, 7), redemption_rate=0.030,
                          extension_rate=0.10, new_business_rate=0.020),

    # ---- asset finance ---------------------------------------------------- #
    "hire_purchase_and_lease_mix": dict(amortisation_mode="amortising",
                                        amortisation_rate=0.023),
    "mixed_collateral_categories": dict(amortisation_rate=0.020),
    "balloon_and_residual": dict(balloon_share=0.58, balloon_fraction=0.32,
                                 amortisation_rate=0.014),
    "delinquency_repossession": dict(arrears_entry_rate=0.038,
                                     arrears_cure_rate=0.15, default_rate=0.32,
                                     enforcement_rate=0.35,
                                     repossession_rate=0.45,
                                     write_off_rate=0.22),
    "vendor_concentration_stress": dict(borrowers_per_hundred=62),
}


class UnknownIntentError(ValueError):
    """Raised for an intent token that has no declared modifier."""


def resolve_knobs(asset_class: str, intents) -> Knobs:
    """Baseline knobs for *asset_class*, with each declared intent applied.

    Intents are applied in declaration order, so a later intent may deliberately
    override an earlier one. Every applied change is explicit — nothing is
    inferred from the case id or from the intent's spelling.
    """
    knobs = _BASELINES[asset_class]
    for intent in intents or ():
        modifier = INTENT_MODIFIERS.get(intent)
        if modifier is None:
            raise UnknownIntentError(
                f"economic intent {intent!r} has no declared modifier; add one "
                f"to simulation.knobs.INTENT_MODIFIERS")
        knobs = knobs.with_(**modifier)
    return knobs


__all__ = ["INTENT_MODIFIERS", "Knobs", "UnknownIntentError", "resolve_knobs"]
