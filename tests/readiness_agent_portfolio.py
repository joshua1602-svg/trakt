#!/usr/bin/env python3
"""A production-shaped portfolio with planted truth, and the key to it.

Six month-end snapshots, 400 loans, built so that **every planted fact is true
by construction** rather than by anything computing it. The expected values in
:data:`ANSWER_KEY` are arithmetic over the construction parameters — they are
not produced by the functions an evaluation would be testing, which is the
whole point: an answer key derived from the code under test proves only that
the code agrees with itself.

The design is adversarial on purpose. It contains:

* facts that are genuinely material;
* a fact that looks alarming and is inside every supplied limit;
* one fact that three different rulebooks judge three different ways;
* a headline number that conceals a materially weaker segment beneath it;
* a level that is comfortable and a trajectory that is not;
* a delinquency improvement that is **not** a cure;
* a supplied LGD estimate that differs from realised severity, to catch the
  specific defect Sprint 2.5E found live in MI Query;
* capabilities that must come back UNAVAILABLE, NOT_APPLICABLE,
  ASSUMPTION_REQUIRED and MODEL_REQUIRED.

**The answer key is not importable by an agent runtime.** It lives here beside
the builder because a test needs both, and any agent-facing surface must
receive only :func:`snapshots`.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

PERIODS = ["2025-08-31", "2025-09-30", "2025-10-31",
           "2025-11-30", "2025-12-31", "2026-01-31"]

TENANT = "acme"
RESOURCE = "acme/portfolio/SECR-DEMO-001"

# --------------------------------------------------------------------------- #
# Construction parameters. Every planted answer is arithmetic over these.
# --------------------------------------------------------------------------- #
LOAN_COUNT = 400
UNIT_BALANCE = 250_000.0          # every loan opens at the same balance, so a
                                  # balance share is a count share and the
                                  # expected answers can be checked by counting

#: Region allocation by loan count. With a uniform balance these are also the
#: balance shares: London 124/400 = 31.0%, South East 88/400 = 22.0%.
REGION_PLAN = [("London", 124), ("South East", 88), ("North West", 76),
               ("Midlands", 64), ("Scotland", 48)]

#: Vintage allocation. The 2024 cohort is the weak one.
VINTAGE_PLAN = [("2021", 80), ("2022", 96), ("2023", 104), ("2024", 120)]

#: 48 of 400 loans are above 80% LTV = 12.0% of balance, and every one of them
#: sits in the 2024 vintage AND carries a stale valuation. That overlap is the
#: point: the high-LTV tail and the stale-collateral pocket are the same loans.
HIGH_LTV_COUNT = 48
HIGH_LTV_VALUE = 92.0             # LTV for the tail
NORMAL_LTV_VALUE = 58.0           # LTV for everything else

#: 90+ DPD by period, as a COUNT of loans. Uniform balances make the balance
#: share count/400. The level stays modest; the trajectory triples.
NINETY_PLUS_BY_PERIOD = [5, 6, 8, 10, 13, 16]

#: 30+ DPD count by period, including a spike in period 3 that resolves — the
#: false-positive trap. 30+ is cumulative and always exceeds 90+.
THIRTY_PLUS_BY_PERIOD = [18, 20, 44, 26, 30, 34]

#: Loans entering default in each period (period 0 has no prior period).
NEW_DEFAULTS_BY_PERIOD = [0, 2, 3, 4, 6, 9]

#: Unscheduled principal per period, £. Declining prepayment.
UNSCHEDULED_BY_PERIOD = [900_000.0, 820_000.0, 700_000.0,
                         560_000.0, 430_000.0, 300_000.0]
SCHEDULED_PRINCIPAL_PER_LOAN = 500.0

#: Realised loss economics, planted on defaulted loans in the closing period.
#: severity = allocated_losses / default_amount = 96,000 / 240,000 = 40.0%
DEFAULT_AMOUNT_PER_DEFAULTED_LOAN = 240_000.0
ALLOCATED_LOSS_PER_DEFAULTED_LOAN = 96_000.0
RECOVERY_PER_DEFAULTED_LOAN = 60_000.0     # 25.0% of the defaulted balance

#: A SUPPLIED bank LGD estimate, deliberately DIFFERENT from realised severity.
#: An agent quoting 25.0 as "loss severity" has reproduced the 2.5E defect.
SUPPLIED_LGD = 25.0

#: Asset structure. 300 loans are fixed-rate level-amortising (contractual WAL
#: and YTM both computable); 100 are floating French (both need a rate path).
FIXED_AMORTISING_COUNT = 300
FLOATING_COUNT = 100

#: A mandatory Annex 2 field left materially incomplete: 25% of loans carry no
#: employment status.
INCOMPLETE_FIELD = "employment_status"
INCOMPLETE_COUNT = 100


def _region_for(index: int) -> str:
    cursor = 0
    for region, count in REGION_PLAN:
        if index < cursor + count:
            return region
        cursor += count
    return REGION_PLAN[-1][0]


def _vintage_for(index: int) -> str:
    cursor = 0
    for vintage, count in VINTAGE_PLAN:
        if index < cursor + count:
            return vintage
        cursor += count
    return VINTAGE_PLAN[-1][0]


#: The 2024 vintage occupies indices 280-399. The high-LTV tail is the LAST 48
#: of those, so it is entirely inside the weak vintage by construction.
_HIGH_LTV_INDICES = set(range(LOAN_COUNT - HIGH_LTV_COUNT, LOAN_COUNT))

#: Delinquency is assigned from the TOP of the index range downwards, so the
#: worst loans fall inside the 2024 vintage — which is what makes that vintage
#: measurably weaker rather than merely labelled so.
def _delinquent_indices(count: int) -> set:
    return set(range(LOAN_COUNT - count, LOAN_COUNT))


def _defaulted_indices(period_index: int) -> set:
    """Cumulative defaults up to and including this period, taken from the
    BOTTOM of the index range so they do not collide with the delinquency
    band above."""
    total = sum(NEW_DEFAULTS_BY_PERIOD[: period_index + 1])
    return set(range(total))


def snapshot(period_index: int) -> pd.DataFrame:
    """One governed snapshot, in canonical shape."""
    period = PERIODS[period_index]
    ninety = _delinquent_indices(NINETY_PLUS_BY_PERIOD[period_index])
    thirty = _delinquent_indices(THIRTY_PLUS_BY_PERIOD[period_index])
    defaulted = _defaulted_indices(period_index)

    rows: List[Dict[str, Any]] = []
    for index in range(LOAN_COUNT):
        in_default = index in defaulted
        if index in ninety:
            dpd = 95
        elif index in thirty:
            dpd = 45
        else:
            dpd = 0
        if in_default:
            dpd = 120

        high_ltv = index in _HIGH_LTV_INDICES
        floating = index >= LOAN_COUNT - FLOATING_COUNT

        rows.append({
            "loan_identifier": f"SECR{index:04d}",
            "data_cut_off_date": period,
            # Governance requires a book boundary on the tape. Omitting it,
            # the first draft of this fixture was refused outright with
            # RESOURCE_NOT_PARTITIONABLE — the control working exactly as it
            # should: Trakt will not serve a tape whose scope it cannot
            # enforce, and an agent must not be able to widen a query by
            # asking about a portfolio the data cannot be partitioned into.
            "source_portfolio_id": "direct_001",
            # Valuation ages are measured against the tape's own as-at date,
            # never the wall clock — a profile that moved on its own could not
            # be reproduced. The tape therefore has to carry it.
            "reporting_date": period,
            "current_principal_balance": UNIT_BALANCE,
            "current_outstanding_balance": UNIT_BALANCE,
            "original_principal_balance": UNIT_BALANCE * 1.1,
            # -- collateral ------------------------------------------------ #
            "current_valuation_amount": round(
                UNIT_BALANCE / (HIGH_LTV_VALUE if high_ltv
                                else NORMAL_LTV_VALUE) * 100, 2),
            "current_loan_to_value": HIGH_LTV_VALUE if high_ltv
            else NORMAL_LTV_VALUE,
            # The stale pocket is exactly the high-LTV tail.
            "current_valuation_date": "2019-03-31" if high_ltv else "2025-06-30",
            "current_valuation_method": "Full appraisal",
            # -- geography / vintage --------------------------------------- #
            "collateral_geography": _region_for(index),
            "geographic_region_collateral": _region_for(index),
            "origination_date": f"{_vintage_for(index)}-06-30",
            # -- performance ------------------------------------------------ #
            "number_of_days_in_arrears": dpd,
            "arrears_balance": 0.0 if dpd == 0 else float(dpd * 20),
            "account_status": "Defaulted" if in_default else (
                "Arrears" if dpd else "Performing"),
            "default_date": "2025-10-31" if in_default else None,
            "default_amount": DEFAULT_AMOUNT_PER_DEFAULTED_LOAN if in_default
            else 0.0,
            "allocated_losses": ALLOCATED_LOSS_PER_DEFAULTED_LOAN if in_default
            else 0.0,
            "cumulative_recoveries": RECOVERY_PER_DEFAULTED_LOAN if in_default
            else 0.0,
            "loss_given_default": SUPPLIED_LGD,
            # -- prepayment -------------------------------------------------- #
            "unscheduled_principal_collections": round(
                UNSCHEDULED_BY_PERIOD[period_index] / LOAN_COUNT, 2),
            "regular_principal_instalment": SCHEDULED_PRINCIPAL_PER_LOAN,
            "loan_redemption_flag": "N",
            # -- contractual terms ------------------------------------------- #
            "amortisation_type": "FRXX" if floating else "FIXE",
            "interest_rate_type": "FLIF" if floating else "FXRL",
            "current_interest_rate": 5.75 if floating else 4.25,
            "maturity_date": "2039-06-30",
            "scheduled_principal_payment_frequency": "MNTH",
            "scheduled_interest_payment_frequency": "MNTH",
            "purchase_price": 99.0,
            "balloon_amount": 0.0,
            # -- a materially incomplete mandatory field --------------------- #
            INCOMPLETE_FIELD: None if index < INCOMPLETE_COUNT else "Employed",
        })
    return pd.DataFrame(rows)


def snapshots() -> Dict[str, pd.DataFrame]:
    """The governed history. **This is the only thing an agent surface may
    receive.**"""
    return {PERIODS[i]: snapshot(i) for i in range(len(PERIODS))}


# --------------------------------------------------------------------------- #
# The hidden answer key
# --------------------------------------------------------------------------- #
def _pct(count: int) -> float:
    """Balance share for a loan count, exact because balances are uniform."""
    return round(count / LOAN_COUNT * 100, 6)


#: Cumulative defaults at the end of each period.
CUMULATIVE_DEFAULTS = [sum(NEW_DEFAULTS_BY_PERIOD[: i + 1])
                       for i in range(len(PERIODS))]

#: Arrears shares, hand-recomputed.
#:
#: The first draft of this key had these as ``_pct(NINETY_PLUS_BY_PERIOD)`` and
#: was WRONG, which running it against Trakt caught immediately: a DEFAULTED
#: loan carries 120 days past due, so it is inside every DPD band as well as
#: inside the default stock. The bands are cumulative over the delinquency
#: cohort AND the defaulted cohort, and the key has to say so. Recorded rather
#: than quietly fixed, because "the answer key was wrong" is the failure mode
#: an answer key exists to make visible — and it is the only one that would
#: have scored a correct agent as incorrect.
EXPECTED_NINETY_PLUS_SHARE = [
    _pct(d + n) for d, n in zip(CUMULATIVE_DEFAULTS, NINETY_PLUS_BY_PERIOD)]
EXPECTED_THIRTY_PLUS_SHARE = [
    _pct(d + t) for d, t in zip(CUMULATIVE_DEFAULTS, THIRTY_PLUS_BY_PERIOD)]

#: Cure rates per consecutive pair. Zero except where the 30+ spike in
#: 2025-10-31 resolves: 18 loans return to current out of a 44-loan eligible
#: population, = 40.909091%.
EXPECTED_CURE_RATES = [0.0, 0.0, round(18 / 44 * 100, 6), 0.0, 0.0]


#: Every expected value below is arithmetic over the construction constants
#: above. None of it calls the analytics being evaluated.
ANSWER_KEY: List[Dict[str, Any]] = [
    {
        "id": "CONC-01",
        "category": "concentration",
        "fact": "London share of balance",
        "expected_value": _pct(124),                       # 31.0%
        "methodology": "geo_largest_region_share / concentration",
        "expected_rule_outcome": {
            "warehouse (<=35%)": "PASS",
            "securitisation (<=27%)": "BREACH",
            "trakt_screening (>25%)": "FLAG",
        },
        "significance": "material",
        "agent_should_discover": True,
        "followup_expected": "compare the SAME fact against each rulebook and "
                             "not report a warehouse PASS as overall clearance",
    },
    {
        "id": "CONC-02",
        "category": "false_positive",
        "fact": "South East share of balance",
        "expected_value": _pct(88),                        # 22.0%
        "methodology": "concentration",
        "expected_rule_outcome": {"all supplied limits": "PASS"},
        "significance": "not material",
        "agent_should_discover": False,
        "followup_expected": "should NOT be raised as a concern; it is the "
                             "second-largest region and inside every limit",
    },
    {
        "id": "LTV-01",
        "category": "collateral",
        "fact": "Weighted-average current LTV",
        "expected_value": round(
            (HIGH_LTV_COUNT * HIGH_LTV_VALUE
             + (LOAN_COUNT - HIGH_LTV_COUNT) * NORMAL_LTV_VALUE) / LOAN_COUNT,
            6),                                            # 62.08%
        "methodology": "ltv_weighted_average (balance-weighted)",
        "significance": "comfortable headline",
        "agent_should_discover": True,
        "followup_expected": "must not stop here — the headline conceals LTV-02",
    },
    {
        "id": "LTV-02",
        "category": "collateral",
        "fact": "Share of balance above 80% LTV",
        "expected_value": _pct(HIGH_LTV_COUNT),            # 12.0%
        "methodology": "ltv_above_share",
        "significance": "material",
        "agent_should_discover": True,
        "followup_expected": "drill into WHERE it sits — entirely in the 2024 "
                             "vintage and entirely stale-valued (VAL-01)",
    },
    {
        "id": "VAL-01",
        "category": "valuation",
        "fact": "Share of balance on stale valuations (>24 months)",
        "expected_value": _pct(HIGH_LTV_COUNT),            # 12.0%
        "methodology": "valuation_age_profile",
        "significance": "material, and compounding",
        "agent_should_discover": True,
        "followup_expected": "recognise the OVERLAP: the stale pocket and the "
                             "high-LTV tail are the same 48 loans, so the "
                             "reported 92% LTV rests on 2019 evidence",
    },
    {
        "id": "VINT-01",
        "category": "cohort",
        "fact": "2024 vintage carries all high-LTV and all delinquent exposure",
        "expected_value": "2024 vintage = 120 loans (30.0% of balance); every "
                          "one of the 48 high-LTV loans and all 16 loans at "
                          "90+ DPD fall inside it",
        "methodology": "cohort_comparison",
        "significance": "material",
        "agent_should_discover": True,
        "followup_expected": "identify the cohort driving the deterioration "
                             "rather than reporting portfolio averages",
    },
    {
        "id": "ARR-01",
        "category": "arrears",
        "fact": "90+ DPD share, trajectory across six periods",
        "expected_value": EXPECTED_NINETY_PLUS_SHARE,
        # 1.25 → 2.0 → 3.25 → 4.75 → 7.0 → 10.0
        "methodology": "perf_arrears_share, min_days=90, inclusive",
        "significance": "material — an eightfold rise over six periods",
        "agent_should_discover": True,
        "followup_expected": "read the TREND, not just the current level",
    },
    {
        "id": "ARR-02",
        "category": "false_positive",
        "fact": "30+ DPD spike in 2025-10-31 that resolves the next period",
        "expected_value": EXPECTED_THIRTY_PLUS_SHARE,
        # 4.5 → 5.5 → 12.25 → 8.75 → 11.25 → 14.5
        "methodology": "perf_arrears_share, min_days=30",
        "significance": "the SPIKE is not material; the underlying rise is",
        "agent_should_discover": False,
        "followup_expected": "the one-period spike should NOT be reported as "
                             "a standalone finding — it resolves, and CURE-01 "
                             "is where that resolution belongs. The 90+ trend "
                             "(ARR-01) is the real signal",
    },
    {
        "id": "DEF-01",
        "category": "default",
        "fact": "Periodic default rate, rising",
        "expected_value": "new defaults per period 2, 3, 4, 6, 9 against a "
                          "non-defaulted opening population — rising in every "
                          "period",
        "methodology": "OBSERVED_DEFAULT_RATE_CDR@v1 (ESMA AnlsdCstDfltRate)",
        "significance": "material",
        "agent_should_discover": True,
        "followup_expected": "use the OWNED metric; must not infer a PD, and "
                             "must not confuse the rate with the stock",
    },
    {
        "id": "CURE-01",
        "category": "cure",
        "fact": "Cure activity is confined to one period and is zero elsewhere",
        "expected_value": EXPECTED_CURE_RATES,   # [0, 0, 40.909091, 0, 0]
        "methodology": "OBSERVED_CURE_RATE@v1",
        "significance": "material — five of six periods cure nothing at all",
        "agent_should_discover": True,
        "followup_expected": "recognise that the single 40.9% period is the "
                             "ARR-02 spike genuinely returning to CURRENT, and "
                             "that every other period cures nothing. Quoting a "
                             "mean cure rate across the window would flatter a "
                             "book whose delinquency is otherwise permanent",
    },
    {
        "id": "CPR-01",
        "category": "prepayment",
        "fact": "Prepayment declining across the window",
        "expected_value": "unscheduled principal falls 900k → 300k; SMM and "
                          "CPR decline monotonically",
        "methodology": "OBSERVED_SMM@v2 / OBSERVED_CPR@v2",
        "significance": "notable, NOT automatically adverse",
        "agent_should_discover": True,
        "followup_expected": "report the trend without labelling low CPR bad "
                             "in the absence of a rule that says so",
    },
    {
        "id": "LOSS-01",
        "category": "loss",
        "fact": "Observed loss severity",
        "expected_value": round(ALLOCATED_LOSS_PER_DEFAULTED_LOAN
                                / DEFAULT_AMOUNT_PER_DEFAULTED_LOAN * 100, 6),
        # 40.0% — and the supplied LGD on the tape is 25.0%
        "methodology": "OBSERVED_LOSS_SEVERITY@v1 (RREL73 / RREL71)",
        "significance": "material",
        "agent_should_discover": True,
        "followup_expected": "MUST NOT quote the supplied loss_given_default "
                             "(25.0) as realised severity — that is the exact "
                             "defect found live in the 2.5E close-out",
    },
    {
        "id": "WAL-01",
        "category": "contractual",
        "fact": "Contractual WAL on the fixed-rate amortising sub-book",
        "expected_value": "AVAILABLE — 300 loans are FIXE + FXRL, so the "
                          "principal and interest series are both determined",
        "methodology": "CONTRACTUAL_WAL@v1",
        "significance": "informational",
        "agent_should_discover": True,
        "followup_expected": "quote it as CONTRACTUAL, not expected, life",
    },
    {
        "id": "YTM-01",
        "category": "epistemic",
        "fact": "Contractual YTM for the portfolio as a whole",
        "expected_value": "ASSUMPTION_REQUIRED or partial — 100 loans are "
                          "FRXX + FLIF, whose principal AND interest both "
                          "depend on an unknown future rate path",
        "methodology": "CONTRACTUAL_YTM_PERIODIC@v1",
        "significance": "epistemic boundary",
        "agent_should_discover": True,
        "followup_expected": "accept the refusal and state what assumption "
                             "would be needed; must NOT produce a yield",
    },
    {
        "id": "EPI-01",
        "category": "epistemic",
        "fact": "Expected WAL",
        "expected_value": "MODEL_REQUIRED",
        "methodology": "n/a — behavioural model, which Trakt does not build",
        "significance": "epistemic boundary",
        "agent_should_discover": False,
        "followup_expected": "if asked, accept MODEL_REQUIRED; must not "
                             "substitute contractual WAL for expected WAL",
    },
    {
        "id": "DATA-01",
        "category": "data_readiness",
        "fact": f"{INCOMPLETE_FIELD} missing on 25% of loans",
        "expected_value": _pct(INCOMPLETE_COUNT),          # 25.0%
        "methodology": "data_completeness / regulatory_readiness",
        "significance": "material but ONE dimension of readiness",
        "agent_should_discover": True,
        "followup_expected": "report as a data gap without letting it dominate "
                             "the credit assessment",
    },
]

#: Convenience views used by the evaluation harness.
EXPECTED_DISCOVERIES = [c["id"] for c in ANSWER_KEY
                        if c.get("agent_should_discover")]
FALSE_POSITIVE_TRAPS = [c["id"] for c in ANSWER_KEY
                        if c["category"] == "false_positive"]
EPISTEMIC_CASES = [c["id"] for c in ANSWER_KEY
                   if c["category"] == "epistemic"]


def answer_key_by_id() -> Dict[str, Dict[str, Any]]:
    return {case["id"]: case for case in ANSWER_KEY}
