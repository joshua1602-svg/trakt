#!/usr/bin/env python3
"""scripts/prove_multivariate_pipeline_fixture.py — the oracle, before the bank.

READ-ONLY. Computes every expected answer in the multivariate pipeline bank
DIRECTLY from the governed prepared frame — the output of production
``pipeline_contract.load_prepared_pipeline`` — so each expectation is arithmetic
on governed data rather than an opinion about what the agent ought to say.

It also proves the fixture is DISCRIMINATING, which is the part that decides
whether the bank can detect anything. A bank whose Offer population is 100%
joint borrowers grades an agent that silently dropped the borrower-type filter
as correct. For every filtered question this prints the filtered figure beside
the unfiltered one it would be confused with, and fails if they coincide.

    python scripts/prove_multivariate_pipeline_fixture.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

FIXTURE = _REPO / "tests" / "fixtures" / "pipeline_multivariate"
CLIENT = "client_001"

STAGE = "pipeline_stage"
BORROWER = "borrower_type"
REGION = "geographic_region_obligor"
LTV = "current_loan_to_value"
BALANCE = "current_outstanding_balance"
RATE = "current_interest_rate"


class FixtureError(RuntimeError):
    """The fixture cannot support the bank. Never absorbed into a pass."""


def frames() -> Dict[str, Any]:
    from mi_agent_api import pipeline_contract as pc

    inv = pc.weekly_extract_inventory(str(FIXTURE), CLIENT)
    extracts = inv.get("extracts", [])
    if len(extracts) < 2:
        raise FixtureError("need at least two governed extracts, found %d"
                           % len(extracts))
    out = {}
    for entry in extracts:
        df, _ = pc.load_prepared_pipeline(entry)
        out[entry.get("pipeline_extract_date")] = df
    return out


def money(value: float) -> str:
    return "GBP %,.0f".replace(",", ",").format(value) if False else "GBP {:,.0f}".format(value)


def report(frames_by_date: Dict[str, Any]) -> Dict[str, Any]:
    dates = sorted(frames_by_date)
    latest, prior_month = dates[-1], dates[0]
    df = frames_by_date[latest]
    may = frames_by_date[prior_month]

    for col in (STAGE, BORROWER, REGION, LTV, BALANCE, RATE):
        if col not in df.columns:
            raise FixtureError("governed field %r is absent from the prepared "
                               "frame; the bank cannot ask about it" % col)

    offer = df[df[STAGE] == "OFFER"]
    application = df[df[STAGE] == "APPLICATION"]

    def wa_ltv(frame) -> float:
        w = frame[BALANCE]
        return float((frame[LTV] * w).sum() / w.sum()) if w.sum() else float("nan")

    facts: Dict[str, Any] = {
        "extracts": dates,
        "latest": latest,
        "prior_month_extract": prior_month,
        "rows_latest": int(len(df)),
        "stage_counts": {k: int(v) for k, v in df[STAGE].value_counts().items()},
        "borrower_split_overall": {k: int(v) for k, v
                                   in df[BORROWER].value_counts().items()},
        "ltv_range": [round(float(df[LTV].min()), 4), round(float(df[LTV].max()), 4)],
    }

    # Q1 stage + borrower type + balance
    facts["Q1_offer_joint_balance"] = round(
        float(offer[offer[BORROWER] == "joint"][BALANCE].sum()), 2)
    facts["Q1_offer_total_balance"] = round(float(offer[BALANCE].sum()), 2)
    facts["Q1_offer_joint_cases"] = int((offer[BORROWER] == "joint").sum())

    # Q2 share of Offer that is joint
    facts["Q2_offer_joint_share_pct"] = round(
        100.0 * facts["Q1_offer_joint_balance"] / facts["Q1_offer_total_balance"], 2)

    # Q3 stage + region + balance
    facts["Q3_application_london_balance"] = round(
        float(application[application[REGION] == "London"][BALANCE].sum()), 2)
    facts["Q3_application_total_balance"] = round(float(application[BALANCE].sum()), 2)

    # Q4 stage + region grouping
    facts["Q4_offer_by_region"] = {
        str(k): round(float(v), 2) for k, v
        in offer.groupby(REGION)[BALANCE].sum().sort_index().items()}

    # Q5 stage + LTV threshold (LTV is a governed RATIO, so 60% is 0.60)
    above = offer[offer[LTV] > 0.60]
    facts["Q5_offer_ltv_above_60_balance"] = round(float(above[BALANCE].sum()), 2)
    facts["Q5_offer_ltv_above_60_cases"] = int(len(above))

    # Q6 stage + LTV grouping
    bands = {"<=50%": (0.0, 0.50), "50-60%": (0.50, 0.60),
             "60-70%": (0.60, 0.70), ">70%": (0.70, 10.0)}
    facts["Q6_application_by_ltv_band"] = {
        name: round(float(application[(application[LTV] > lo)
                                      & (application[LTV] <= hi)][BALANCE].sum()), 2)
        for name, (lo, hi) in bands.items()}

    # Q7 stage + weighted metric
    facts["Q7_offer_wa_ltv_pct"] = round(100.0 * wa_ltv(offer), 3)
    facts["Q7_offer_simple_mean_ltv_pct"] = round(100.0 * float(offer[LTV].mean()), 3)

    # Q8 stage + dimension + weighted metric
    app_joint = application[application[BORROWER] == "joint"]
    facts["Q8_application_joint_wa_ltv_pct"] = round(100.0 * wa_ltv(app_joint), 3)
    facts["Q8_application_wa_ltv_pct"] = round(100.0 * wa_ltv(application), 3)
    facts["Q8_application_joint_simple_mean_ltv_pct"] = round(
        100.0 * float(app_joint[LTV].mean()), 3)

    # Q9 stage stock vs the prior-month governed extract
    facts["Q9_application_balance_latest"] = facts["Q3_application_total_balance"]
    may_app = may[may[STAGE] == "APPLICATION"]
    facts["Q9_application_balance_prior_month"] = round(float(may_app[BALANCE].sum()), 2)
    facts["Q9_application_change"] = round(
        facts["Q9_application_balance_latest"]
        - facts["Q9_application_balance_prior_month"], 2)

    # Q10 two-dimensional grouping
    facts["Q10_stage_x_borrower"] = {
        "%s|%s" % (a, b): round(float(v), 2) for (a, b), v
        in df.groupby([STAGE, BORROWER])[BALANCE].sum().items()}

    # Optional: stage + ticket band
    facts["Q11_offer_ticket_over_500k_balance"] = round(
        float(offer[offer[BALANCE] > 500_000][BALANCE].sum()), 2)

    return facts


def discrimination(facts: Dict[str, Any]) -> List[str]:
    """The checks that decide whether a dropped filter could pass unnoticed."""
    problems: List[str] = []

    def distinct(label, a, b, why):
        if abs(float(a) - float(b)) < 0.01:
            problems.append("%s: %s == %s — %s" % (label, a, b, why))

    distinct("Q1 borrower filter", facts["Q1_offer_joint_balance"],
             facts["Q1_offer_total_balance"],
             "a dropped borrower-type filter would score as correct")
    distinct("Q3 region filter", facts["Q3_application_london_balance"],
             facts["Q3_application_total_balance"],
             "a dropped region filter would score as correct")
    distinct("Q5 LTV threshold", facts["Q5_offer_ltv_above_60_balance"],
             facts["Q1_offer_total_balance"],
             "a dropped LTV threshold would score as correct")
    distinct("Q7 weighting basis", facts["Q7_offer_wa_ltv_pct"],
             facts["Q7_offer_simple_mean_ltv_pct"],
             "an UNWEIGHTED mean would be indistinguishable from WA LTV")
    distinct("Q8 weighting basis", facts["Q8_application_joint_wa_ltv_pct"],
             facts["Q8_application_joint_simple_mean_ltv_pct"],
             "an unweighted mean would be indistinguishable")
    distinct("Q8 borrower filter", facts["Q8_application_joint_wa_ltv_pct"],
             facts["Q8_application_wa_ltv_pct"],
             "a dropped borrower-type filter would score as correct")
    distinct("Q9 period", facts["Q9_application_balance_latest"],
             facts["Q9_application_balance_prior_month"],
             "an unchanged stock cannot show a substituted comparison period")

    if facts["Q5_offer_ltv_above_60_cases"] == 0:
        problems.append("Q5: no Offer case is above 60% LTV — the question is empty")
    if len(facts["Q4_offer_by_region"]) < 3:
        problems.append("Q4: fewer than three Offer regions — grouping is weak")
    share = facts["Q2_offer_joint_share_pct"]
    if share >= 99.9 or share <= 0.1:
        problems.append("Q2: Offer joint share is %s%% — not discriminating" % share)
    if len(facts["Q10_stage_x_borrower"]) < 6:
        problems.append("Q10: the stage x borrower cross-tab has too few cells")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    facts = report(frames())
    problems = discrimination(facts)

    print("GOVERNED PIPELINE FIXTURE — %s" % FIXTURE.relative_to(_REPO))
    print("  extracts        %s" % ", ".join(facts["extracts"]))
    print("  latest rows     %d" % facts["rows_latest"])
    print("  stages          %s" % facts["stage_counts"])
    print("  borrower split  %s" % facts["borrower_split_overall"])
    print("  LTV range       %s (a governed RATIO, so 60%% is 0.60)"
          % facts["ltv_range"])
    print()
    for key in sorted(k for k in facts if k[0] == "Q"):
        print("  %-42s %s" % (key, facts[key]))
    print()
    if problems:
        print("FIXTURE NOT DISCRIMINATING:")
        for p in problems:
            print("  - %s" % p)
    else:
        print("DISCRIMINATING: every filter changes the answer it would be "
              "confused with.")

    if args.json:
        args.json.write_text(json.dumps(
            {"facts": facts, "problems": problems}, indent=1), encoding="utf-8")
        print("\nwrote %s" % args.json)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
