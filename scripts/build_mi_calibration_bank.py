#!/usr/bin/env python3
"""scripts/build_mi_calibration_bank.py

Assemble the curated MI Agent calibration question bank
(``config/mi/golden_questions/ere_mi_calibration_250.yaml``) — 250+ realistic
business-user questions with per-case EXPECTED semantic behaviour, not just
"returns a response". Expectations are calibrated against the REAL deterministic
MI path (see ``mi_agent/tests/test_mi_calibration_bank.py`` which enforces them).

This is a curated bank (realistic lender / investor / credit-committee / portfolio-
manager / ops phrasings) and is ADDITIONAL to the generated registry harness
(``mi_agent/mi_query_harness.py``).

Where the real system does not (yet) do what a business user would expect, the
case is marked ``known_gap`` with a reason (never loosened to force a pass) so the
calibration report can surface it as a follow-up.

Regenerate:  python scripts/build_mi_calibration_bank.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

_OUT = Path(__file__).resolve().parents[1] / "config" / "mi" / "golden_questions" / "ere_mi_calibration_250.yaml"

# Canonical metric keys.
BAL = "current_outstanding_balance"
LTV = "current_loan_to_value"
RATE = "current_interest_rate"
AGE = "youngest_borrower_age"
VAL = "current_valuation_amount"
ORIG = "original_principal_balance"

# Dimension phrase -> semantic key (funded fixture dims).
DIMS = {
    "region": "geographic_region_obligor",
    "borrower type": "borrower_type",
    "product type": "erm_product_type",
    "LTV bucket": "ltv_bucket",
    "age bucket": "age_bucket",
    "broker": "broker_channel",
    "account status": "account_status",
    "origination channel": "origination_channel",
    "rate type": "interest_rate_type",
    "occupancy type": "occupancy_type",
    "term bucket": "term_bucket",
    "ticket size": "ticket_bucket",
}

_cid = 0


def _id(cat: str) -> str:
    global _cid
    _cid += 1
    return f"{cat}_{_cid:03d}"


def case(**kw) -> Dict[str, Any]:
    """A case with sensible defaults; only override what matters per question."""
    base = {
        "id": _id(kw.pop("_cat")),
        "category": kw.pop("category"),
        "question": kw.pop("question"),
        "expected_status": "answer",
        "execution": "full",
        "expected_scope": "funded",
        "expected_metric": None,
        "expected_metrics": None,
        "expected_dimensions": [],
        "expected_filters": [],
        "expected_artifact_type": "bar",
        "expected_reconciliation": True,
        "expected_dimension_invariant_ok": True,
        "expected_filter_invariant_ok": True,
        "expected_warnings": [],
        "expected_columns_include": [],
        "expected_min_columns": 1,
        "known_gap": None,
        "notes": "",
    }
    base.update(kw)
    return base


def build() -> List[Dict[str, Any]]:
    C: List[Dict[str, Any]] = []

    # --------------------------------------------------------------------- #
    # 1) Basic funded portfolio KPIs
    # --------------------------------------------------------------------- #
    kpi = dict(_cat="kpi", category="basic_kpi", expected_artifact_type="kpi",
               expected_dimensions=[], expected_min_columns=1)
    for q in ["What is our total funded balance?", "total funded balance",
              "show me the total balance of the book", "what is the total exposure"]:
        C.append(case(**kpi, question=q, expected_metric=BAL,
                      notes="Whole-book balance KPI."))
    for q in ["how many loans are in the book", "number of loans", "loan count",
              "how many mortgages do we have"]:
        C.append(case(**kpi, question=q, expected_metric=None,
                      expected_columns_include=["loan_count"],
                      notes="Whole-book loan count (count aggregation)."))
    C.append(case(**kpi, question="average loan balance", expected_metric=BAL,
                  notes="Mean loan balance (avg aggregation)."))
    C.append(case(**kpi, question="what is the mean loan balance", expected_metric=BAL))
    for q in ["weighted average LTV", "what's the weighted average LTV",
              "portfolio weighted average loan to value", "WA LTV"]:
        C.append(case(**kpi, question=q, expected_metric=LTV,
                      notes="Balance-weighted average LTV."))
    for q in ["weighted average interest rate", "average interest rate",
              "what is the weighted average coupon"]:
        C.append(case(**kpi, question=q, expected_metric=RATE))
    for q in ["average borrower age", "what is the average age of borrowers",
              "mean borrower age"]:
        C.append(case(**kpi, question=q, expected_metric=AGE,
                      notes="Mean borrower age (age is a simple mean, not balance-weighted)."))
    # Largest single loan -> a loan-level ranking table (not a KPI card).
    for q in ["what is the largest loan", "show me the biggest loan",
              "largest single loan by balance"]:
        C.append(case(_cat="kpi", category="basic_kpi", question=q,
                      expected_metric=BAL, expected_artifact_type="table",
                      expected_dimensions=[], expected_min_columns=2,
                      notes="Loan-level ranking: the single largest loan."))
    for q in ["what is the total collateral valuation", "total property valuation",
              "total original principal", "what is the total original balance"]:
        m = ORIG if "original" in q else VAL
        C.append(case(**kpi, question=q, expected_metric=m,
                      notes="Whole-book measure KPI."))
    for q in ["portfolio summary", "portfolio overview", "book overview", "key metrics"]:
        C.append(case(_cat="kpi", category="basic_kpi", question=q, expected_metric=None,
                      expected_artifact_type="kpi", expected_dimensions=[], expected_min_columns=1,
                      notes="Whole-book summary (count + balance)."))
    C.append(case(**kpi, question="average loan size", expected_metric=BAL,
                  known_gap="'size' is parsed as the ticket-size dimension, so this "
                            "returns balance by ticket bucket instead of a mean-size KPI. "
                            "Use 'average loan balance'. Follow-up: metric vocabulary for 'loan size'.",
                  notes="Known parser quirk."))

    # --------------------------------------------------------------------- #
    # 2) Single-dimension groupings
    # --------------------------------------------------------------------- #
    for phrase, key in DIMS.items():
        C.append(case(_cat="grp1", category="single_dim",
                      question=f"balance by {phrase}", expected_metric=BAL,
                      expected_dimensions=[key], expected_artifact_type="bar",
                      expected_min_columns=2, notes=f"Balance grouped by {phrase}."))
    for phrase, key in DIMS.items():
        C.append(case(_cat="grp1", category="single_dim",
                      question=f"loan count by {phrase}", expected_metric=None,
                      expected_dimensions=[key], expected_artifact_type="bar",
                      expected_columns_include=["loan_count"], expected_min_columns=2,
                      notes=f"Loan count grouped by {phrase}."))
    for phrase in ["region", "product type", "broker", "age bucket", "borrower type",
                   "account status", "rate type", "occupancy type"]:
        C.append(case(_cat="grp1", category="single_dim",
                      question=f"average LTV by {phrase}", expected_metric=LTV,
                      expected_dimensions=[DIMS[phrase]], expected_artifact_type="bar",
                      expected_min_columns=2, notes=f"WA LTV by {phrase}."))
    for phrase in ["region", "product type", "broker", "borrower type"]:
        C.append(case(_cat="grp1", category="single_dim",
                      question=f"average borrower age by {phrase}", expected_metric=AGE,
                      expected_dimensions=[DIMS[phrase]], expected_artifact_type="bar",
                      expected_min_columns=2))
    for phrase in ["region", "product type", "broker", "rate type", "borrower type", "age bucket"]:
        C.append(case(_cat="grp1", category="single_dim",
                      question=f"weighted average interest rate by {phrase}",
                      expected_metric=RATE, expected_dimensions=[DIMS[phrase]],
                      expected_artifact_type="bar", expected_min_columns=2))
    # Alternate business phrasings for balance breakdowns (still single-dim bars).
    for tmpl in ["show me balance by {p}", "break down balance by {p}",
                 "balance split by {p}", "exposure by {p}"]:
        for phrase in ["region", "borrower type", "product type", "broker"]:
            C.append(case(_cat="grp1", category="single_dim",
                          question=tmpl.format(p=phrase), expected_metric=BAL,
                          expected_dimensions=[DIMS[phrase]], expected_artifact_type="bar",
                          expected_min_columns=2, notes="Balance breakdown (phrasing variant)."))
    # Valuation / original balance measures grouped by a dimension.
    for phrase in ["region", "product type", "broker"]:
        C.append(case(_cat="grp1", category="single_dim",
                      question=f"total valuation by {phrase}", expected_metric=VAL,
                      expected_dimensions=[DIMS[phrase]], expected_artifact_type="bar",
                      expected_min_columns=2, notes="Collateral valuation by dimension."))
        C.append(case(_cat="grp1", category="single_dim",
                      question=f"original balance by {phrase}", expected_metric=ORIG,
                      expected_dimensions=[DIMS[phrase]], expected_artifact_type="bar",
                      expected_min_columns=2, notes="Original advance by dimension."))

    # --------------------------------------------------------------------- #
    # 3) Two-dimensional groupings -> heatmap/matrix
    # --------------------------------------------------------------------- #
    two = [
        ("balance by borrower type by region", BAL, ["borrower_type", "geographic_region_obligor"]),
        ("balance by LTV bucket by age bucket", BAL, ["ltv_bucket", "age_bucket"]),
        ("balance by region by product type", BAL, ["geographic_region_obligor", "erm_product_type"]),
        ("balance by broker by product type", BAL, ["broker_channel", "erm_product_type"]),
        ("balance by account status by region", BAL, ["account_status", "geographic_region_obligor"]),
        ("balance by product type by age bucket", BAL, ["erm_product_type", "age_bucket"]),
        ("balance by region by LTV bucket", BAL, ["geographic_region_obligor", "ltv_bucket"]),
        ("balance by occupancy type by region", BAL, ["occupancy_type", "geographic_region_obligor"]),
        ("balance by rate type by product type", BAL, ["interest_rate_type", "erm_product_type"]),
        ("WA LTV by product type and region", LTV, ["erm_product_type", "geographic_region_obligor"]),
        ("average LTV by region and age bucket", LTV, ["geographic_region_obligor", "age_bucket"]),
        ("average interest rate by product type and rate type", RATE, ["erm_product_type", "interest_rate_type"]),
    ]
    for q, m, dims in two:
        C.append(case(_cat="grp2", category="two_dim", question=q, expected_metric=m,
                      expected_dimensions=dims, expected_artifact_type="heatmap",
                      expected_min_columns=3, notes="Two categorical dims -> matrix."))
    for q, dims in [
        ("loan count by region and borrower type", ["geographic_region_obligor", "borrower_type"]),
        ("count by age bucket and LTV bucket", ["age_bucket", "ltv_bucket"]),
        ("number of loans by broker and product type", ["broker_channel", "erm_product_type"]),
    ]:
        C.append(case(_cat="grp2", category="two_dim", question=q, expected_metric=None,
                      expected_dimensions=dims, expected_artifact_type="heatmap",
                      expected_columns_include=[], expected_min_columns=3,
                      notes="Two-dim count matrix."))
    for q, m, dims in [
        ("balance by borrower type by product type", BAL, ["borrower_type", "erm_product_type"]),
        ("balance by region by age bucket", BAL, ["geographic_region_obligor", "age_bucket"]),
        ("balance by broker by region", BAL, ["broker_channel", "geographic_region_obligor"]),
        ("balance by account status by product type", BAL, ["account_status", "erm_product_type"]),
        ("balance by term bucket by LTV bucket", BAL, ["term_bucket", "ltv_bucket"]),
        ("balance by ticket size by region", BAL, ["ticket_bucket", "geographic_region_obligor"]),
        ("average LTV by borrower type and region", LTV, ["borrower_type", "geographic_region_obligor"]),
        ("balance by origination channel by region", BAL, ["origination_channel", "geographic_region_obligor"]),
        ("balance by rate type by region", BAL, ["interest_rate_type", "geographic_region_obligor"]),
        ("balance by borrower type by age bucket", BAL, ["borrower_type", "age_bucket"]),
        ("WA interest rate by broker and product type", RATE, ["broker_channel", "erm_product_type"]),
    ]:
        C.append(case(_cat="grp2", category="two_dim", question=q, expected_metric=m,
                      expected_dimensions=dims, expected_artifact_type="heatmap",
                      expected_min_columns=3, notes="Two categorical dims -> matrix."))

    # --------------------------------------------------------------------- #
    # 4) Filtered queries
    # --------------------------------------------------------------------- #
    # Ungrouped filtered KPIs (need a count/total cue to keep the balance + filter).
    for q, m, filt, cols in [
        ("how many loans have LTV above 50%", None, [LTV], ["loan_count"]),
        ("how many loans have LTV above 40%", None, [LTV], ["loan_count"]),
        ("how many borrowers are over 70", None, [AGE], ["loan_count"]),
        ("how many loans have a balance above £250k", None, [BAL], ["loan_count"]),
        ("total balance where LTV is above 50%", BAL, [LTV], []),
        ("how much balance is in loans with a balance above £250k", BAL, [BAL], []),
        ("how many joint borrowers are there", None, ["borrower_type"], ["loan_count"]),
    ]:
        C.append(case(_cat="filt", category="filtered", question=q,
                      expected_metric=m, expected_filters=filt,
                      expected_artifact_type="kpi", expected_dimensions=[],
                      expected_columns_include=cols, expected_min_columns=1,
                      notes="Filtered whole-book KPI."))
    # Grouped filtered.
    for q, m, dims, filt in [
        ("balance by region where LTV above 50%", BAL, ["geographic_region_obligor"], [LTV]),
        ("balance by broker where LTV above 50%", BAL, ["broker_channel"], [LTV]),
        ("balance by product type for loans above £250k", BAL, ["erm_product_type"], [BAL]),
        ("balance by region where LTV between 40 and 60", BAL, ["geographic_region_obligor"], [LTV]),
        ("count by borrower type where age is over 70", None, ["borrower_type"], [AGE]),
        ("WA LTV by region for joint borrowers", LTV, ["geographic_region_obligor"], ["borrower_type"]),
        ("average LTV by product type for joint borrowers", LTV, ["erm_product_type"], ["borrower_type"]),
        ("balance by broker for loans above £250k", BAL, ["broker_channel"], [BAL]),
    ]:
        C.append(case(_cat="filt", category="filtered", question=q, expected_metric=m,
                      expected_dimensions=dims, expected_filters=filt,
                      expected_artifact_type="bar", expected_min_columns=2,
                      notes="Grouped query + value filter (mask before grouping)."))
    # More grouped filters across thresholds / dimensions (all supported).
    for thr in [30, 40, 60, 70]:
        C.append(case(_cat="filt", category="filtered",
                      question=f"balance by region where LTV above {thr}%", expected_metric=BAL,
                      expected_dimensions=["geographic_region_obligor"], expected_filters=[LTV],
                      expected_artifact_type="bar", expected_min_columns=2,
                      notes="Grouped balance filtered by LTV threshold."))
    for phrase in ["broker", "product type", "borrower type", "age bucket"]:
        C.append(case(_cat="filt", category="filtered",
                      question=f"balance by {phrase} where LTV above 50%", expected_metric=BAL,
                      expected_dimensions=[DIMS[phrase]], expected_filters=[LTV],
                      expected_artifact_type="bar", expected_min_columns=2,
                      notes="Grouped balance + LTV filter."))
    for phrase in ["region", "broker", "product type"]:
        C.append(case(_cat="filt", category="filtered",
                      question=f"balance by {phrase} for loans above £250k", expected_metric=BAL,
                      expected_dimensions=[DIMS[phrase]], expected_filters=[BAL],
                      expected_artifact_type="bar", expected_min_columns=2,
                      notes="Grouped balance + balance-threshold filter."))
    for thr in [200000, 300000]:
        C.append(case(_cat="filt", category="filtered",
                      question=f"how many loans have a balance above {thr}", expected_metric=None,
                      expected_filters=[BAL], expected_artifact_type="kpi",
                      expected_dimensions=[], expected_columns_include=["loan_count"],
                      expected_min_columns=1, notes="Filtered loan count KPI."))
    for phrase in ["region", "product type", "broker"]:
        C.append(case(_cat="filt", category="filtered",
                      question=f"balance by {phrase} for joint borrowers", expected_metric=BAL,
                      expected_dimensions=[DIMS[phrase]], expected_filters=["borrower_type"],
                      expected_artifact_type="bar", expected_min_columns=2,
                      notes="Grouped balance + categorical (borrower-structure) filter."))
    for age in [70, 75, 80]:
        C.append(case(_cat="filt", category="filtered",
                      question=f"balance by region where borrower age is over {age}",
                      expected_metric=BAL, expected_dimensions=["geographic_region_obligor"],
                      expected_filters=[AGE], expected_artifact_type="bar", expected_min_columns=2,
                      notes="Grouped balance + age filter."))
    for phrase in ["broker", "product type"]:
        C.append(case(_cat="filt", category="filtered",
                      question=f"WA LTV by {phrase} where LTV above 40%", expected_metric=LTV,
                      expected_dimensions=[DIMS[phrase]], expected_filters=[LTV],
                      expected_artifact_type="bar", expected_min_columns=2,
                      notes="Grouped WA LTV + LTV filter."))

    # Bare "balance where LTV > 50%" — a KNOWN gap (drops filter, resolves LTV).
    C.append(case(_cat="filt", category="filtered", question="balance where LTV above 50%",
                  expected_metric=BAL, expected_filters=[LTV], expected_artifact_type="kpi",
                  expected_dimensions=[], expected_min_columns=1,
                  known_gap="Bare 'balance where <filter>' (no total/how-much/how-many cue) "
                            "resolves the metric to LTV and drops the filter. Use 'total balance "
                            "where LTV above 50%'. Follow-up: treat 'balance where ...' as filtered balance.",
                  notes="Known parser gap on bare filtered balance."))

    # --------------------------------------------------------------------- #
    # 5) Ranking / top-N
    # --------------------------------------------------------------------- #
    for q, dims in [
        ("top 10 regions by balance", ["geographic_region_obligor"]),
        ("top 5 brokers by balance", ["broker_channel"]),
        ("top 10 products by balance", ["erm_product_type"]),
        ("which region has the largest balance", ["geographic_region_obligor"]),
        ("which broker has the largest balance", ["broker_channel"]),
        ("which product type has the highest balance", ["erm_product_type"]),
        ("regions with the most loans", ["geographic_region_obligor"]),
        ("highest average LTV regions", ["geographic_region_obligor"]),
        ("which region has the highest average LTV", ["geographic_region_obligor"]),
        ("smallest region by balance", ["geographic_region_obligor"]),
    ]:
        m = LTV if "LTV" in q else BAL
        C.append(case(_cat="rank", category="ranking", question=q, expected_metric=m,
                      expected_dimensions=dims, expected_artifact_type="bar",
                      expected_min_columns=2, notes="Ranked grouped bar."))
    for q in ["largest loans", "show the largest loans", "highest LTV loans",
              "top 10 loans by balance", "biggest 20 loans"]:
        m = LTV if "LTV" in q else BAL
        C.append(case(_cat="rank", category="ranking", question=q, expected_metric=m,
                      expected_dimensions=[], expected_artifact_type="table",
                      expected_min_columns=2, notes="Loan-level top-N ranking table."))
    for n, phrase in [(5, "region"), (5, "broker"), (10, "product type"),
                      (3, "age bucket"), (5, "account status")]:
        C.append(case(_cat="rank", category="ranking",
                      question=f"top {n} {phrase} by balance",
                      expected_metric=BAL, expected_dimensions=[DIMS[phrase]],
                      expected_artifact_type="bar", expected_min_columns=2,
                      notes="Top-N grouped ranking."))
    for phrase in ["broker", "product type", "borrower type"]:
        C.append(case(_cat="rank", category="ranking",
                      question=f"which {phrase} has the highest average LTV",
                      expected_metric=LTV, expected_dimensions=[DIMS[phrase]],
                      expected_artifact_type="bar", expected_min_columns=2,
                      notes="Ranked WA LTV by dimension."))

    # --------------------------------------------------------------------- #
    # 6) Pipeline queries (parse-only — pipeline execution needs the runtime
    #    chat-routing harness with pipeline data, not the funded fixture).
    # --------------------------------------------------------------------- #
    pipe = dict(_cat="pipe", category="pipeline", execution="parse_only",
                expected_scope="pipeline", expected_reconciliation=False,
                expected_artifact_type="none")
    for q in ["pipeline amount by stage", "expected funded by stage", "pipeline by broker",
              "pipeline by stage", "completions by month", "pipeline conversion by stage",
              "top brokers by expected funded amount", "applications over the last four weeks",
              "pipeline by stage for broker Alpha", "weighted expected funded by week",
              "how many cases are in the pipeline", "pipeline amount by broker"]:
        C.append(case(**pipe, question=q,
                      notes="Pipeline scope: validated at parse level (no hallucinated "
                            "fields); full execution requires the pipeline runtime harness."))

    # --------------------------------------------------------------------- #
    # 7) Forecast queries (parse-only — forecast runtime required).
    # --------------------------------------------------------------------- #
    fc = dict(_cat="fcast", category="forecast", execution="parse_only",
              expected_scope="forecast", expected_reconciliation=False,
              expected_artifact_type="none")
    C.append(case(**fc, question="forecast funded balance by month",
                  expected_metric="forecast_funded_balance",
                  notes="Forecast metric recognised at parse level."))
    for q in ["expected funded amount by month", "show projected completions",
              "forecast by completion month", "compare current funded balance to expected funded",
              "forecast funded balance", "what is the projected funded balance next quarter",
              "expected completions over the next three months", "run-rate forecast of funded balance"]:
        C.append(case(**fc, question=q,
                      notes="Forecast scope: parse-level (no hallucinated fields); full "
                            "execution requires the forecast runtime harness."))

    # --------------------------------------------------------------------- #
    # 8) Risk / concentration
    # --------------------------------------------------------------------- #
    for q, dims in [
        ("balance by region", ["geographic_region_obligor"]),
        ("balance by LTV bucket", ["ltv_bucket"]),
        ("exposure by region", ["geographic_region_obligor"]),
    ]:
        C.append(case(_cat="risk", category="risk", question=q, expected_metric=BAL,
                      expected_dimensions=dims, expected_scope="risk",
                      expected_artifact_type="bar", expected_min_columns=2,
                      notes="Concentration view."))
    C.append(case(_cat="risk", category="risk", question="largest regional concentration",
                  expected_metric=BAL, expected_dimensions=["geographic_region_obligor"],
                  expected_scope="risk", expected_artifact_type="bar", expected_min_columns=2,
                  known_gap="Resolves to a loan-level ranking TABLE, not a region concentration "
                            "bar. Use 'which region has the largest balance'. Follow-up: route "
                            "'largest <dim> concentration' to a grouped concentration bar.",
                  notes="Known gap: concentration phrasing -> loan-level table."))
    for q, filt in [
        ("how many loans have a balance above £250k", [BAL]),
        ("how much balance is above 60% LTV", [LTV]),
        ("how many loans are over 80 years old", [AGE]),
    ]:
        C.append(case(_cat="risk", category="risk", question=q, expected_metric=None if "how many" in q else BAL,
                      expected_filters=filt, expected_scope="risk",
                      expected_artifact_type="kpi", expected_dimensions=[],
                      expected_min_columns=1, notes="Filtered risk exposure KPI."))
    for place in ["London", "the South East"]:
        C.append(case(_cat="risk", category="risk", question=f"exposure to {place}",
                      expected_metric=BAL, expected_filters=["geographic_region_obligor"],
                      expected_scope="risk", expected_artifact_type="kpi",
                      expected_dimensions=[], expected_min_columns=1,
                      known_gap="A place name ('exposure to London') is not turned into a "
                                "region filter — the whole-book balance is returned. Follow-up: "
                                "parse place names as region-value filters (or clarify).",
                      notes="Known gap: place-name exposure not filtered."))
    for phrase in ["region", "product type", "borrower type", "account status"]:
        C.append(case(_cat="risk", category="risk",
                      question=f"concentration by {phrase}", expected_metric=BAL,
                      expected_dimensions=[DIMS[phrase]], expected_scope="risk",
                      expected_artifact_type="bar", expected_min_columns=2,
                      notes="Balance concentration by a categorical dimension."))
    for phrase in ["LTV bucket", "age bucket"]:
        C.append(case(_cat="risk", category="risk",
                      question=f"concentration by {phrase}", expected_metric=BAL,
                      expected_dimensions=[DIMS[phrase]], expected_scope="risk",
                      expected_artifact_type="bar", expected_min_columns=2,
                      known_gap="The bucket field name pulls the MEASURE (WA LTV / mean age) "
                                "instead of balance, so 'concentration by LTV bucket' returns "
                                "WA LTV per bucket, not the balance concentration. Use 'balance "
                                "by LTV bucket'. Follow-up: prefer balance for 'concentration by "
                                "<bucket>'.",
                      notes="Known metric-resolution quirk on bucket concentration."))
    for q, filt in [("how many loans are above 70% LTV", [LTV]),
                    ("how much balance sits above 50% LTV", [LTV]),
                    ("how many loans are over 75 years old", [AGE])]:
        C.append(case(_cat="risk", category="risk", question=q,
                      expected_metric=None if "how many" in q else BAL,
                      expected_filters=filt, expected_scope="risk",
                      expected_artifact_type="kpi", expected_dimensions=[],
                      expected_min_columns=1, notes="Threshold risk exposure."))
    C.append(case(_cat="risk", category="risk", question="balance by property value band",
                  expected_metric=BAL, expected_dimensions=["age_bucket"],
                  expected_scope="risk", expected_artifact_type="bar", expected_min_columns=2,
                  known_gap="'property value band' is mis-mapped to the age bucket (no "
                            "valuation-band dimension in the tape). Follow-up: add a "
                            "valuation-band dimension or refuse.",
                  notes="Known dimension mis-mapping."))
    C.append(case(_cat="risk", category="risk", question="high age borrower exposure",
                  expected_metric=BAL, expected_filters=[AGE], expected_scope="risk",
                  expected_artifact_type="kpi", expected_dimensions=[], expected_min_columns=1,
                  known_gap="No age threshold is parsed from 'high age', so the whole-book "
                            "balance is returned unfiltered. Use 'balance for borrowers over 80'. "
                            "Follow-up: qualitative-threshold handling or clarify.",
                  notes="Known: qualitative 'high age' not turned into a filter."))

    # --------------------------------------------------------------------- #
    # 9) Data-quality / reconciliation
    # --------------------------------------------------------------------- #
    for q, dims in [("coverage for region", ["geographic_region_obligor"]),
                    ("coverage by borrower type", ["borrower_type"]),
                    ("data coverage by region", ["geographic_region_obligor"])]:
        C.append(case(_cat="dq", category="data_quality", question=q, expected_metric=BAL,
                      expected_dimensions=dims, expected_scope="data_quality",
                      expected_artifact_type="bar", expected_reconciliation=True,
                      expected_min_columns=2,
                      notes="Coverage is delivered via the reconciliation block on the grouped result."))
    C.append(case(_cat="dq", category="data_quality",
                  question="reconciliation for balance by region where LTV above 50%",
                  expected_metric=BAL, expected_dimensions=["geographic_region_obligor"],
                  expected_filters=[LTV], expected_scope="data_quality",
                  expected_artifact_type="bar", expected_reconciliation=True, expected_min_columns=2,
                  notes="Filtered grouped result carries a reconciliation block."))
    for q in ["missing region count", "missing borrower age count", "missing LTV count",
              "loans excluded from LTV analysis"]:
        C.append(case(_cat="dq", category="data_quality", question=q,
                      expected_status="clarify", expected_scope="data_quality",
                      expected_artifact_type="none", expected_reconciliation=False,
                      known_gap="No dedicated 'missing/excluded count' intent — the system "
                                "surfaces coverage via the reconciliation block on a normal "
                                "result, not as a standalone count. Follow-up: a data-quality "
                                "intent that answers missing/excluded counts directly.",
                      notes="Known gap: DQ counts are not a first-class query."))
    C.append(case(_cat="dq", category="data_quality", question="which fields are unavailable for this query",
                  expected_status="clarify", expected_scope="unknown",
                  expected_artifact_type="none", expected_reconciliation=False,
                  notes="Meta question — not a governed analytic; refused/clarified."))

    # --------------------------------------------------------------------- #
    # 10) Ambiguous / unsupported / fail-closed
    # --------------------------------------------------------------------- #
    for q, missing in [
        ("show defaulted balance by region", "default"),
        ("defaulted balance", "default"),
        ("show NNEG by region", "nneg"),
        ("NNEG exposure by LTV bucket", "nneg"),
        ("show credit score by broker", "credit"),
        ("average credit score by region", "credit"),
        ("show arrears by region", "arrears"),
        ("arrears balance by broker", "arrears"),
        ("show recoveries in period", "recover"),
        ("indexed value by region", "index"),
    ]:
        C.append(case(_cat="unsup", category="unsupported", question=q,
                      expected_status="refuse", expected_scope="unknown",
                      expected_artifact_type="none", expected_reconciliation=False,
                      notes=f"Field genuinely absent ({missing}); must refuse with a reason, "
                            "never fabricate or substitute."))
    for q in ["show risky loans", "show me the good loans", "which loans should I worry about",
              "loans above 500000"]:
        C.append(case(_cat="ambig", category="ambiguous", question=q,
                      expected_status="clarify", expected_scope="unknown",
                      expected_artifact_type="none", expected_reconciliation=False,
                      notes="Ambiguous / no governed metric — must clarify or refuse, not guess."))
    for q in [("show best brokers"), ("show bad regions"), ("show profitability by region"),
              ("show me interesting regions")]:
        C.append(case(_cat="ambig", category="ambiguous", question=q,
                      expected_status="clarify", expected_scope="unknown",
                      expected_artifact_type="none", expected_reconciliation=False,
                      known_gap="Currently answered as balance by a dimension (a subjective/"
                                "unavailable concept was silently substituted with balance). "
                                "Ideal: clarify or refuse. Follow-up: detect subjective/"
                                "unavailable concepts ('best', 'bad', 'interesting', 'profitability').",
                      notes="Known gap: subjective term silently mapped to balance."))
    # 3+ dimensions: ALL requested dimensions are preserved as a table/pivot
    # (never silently truncated at parse); a chart shows at most two.
    C.append(case(_cat="ambig", category="ambiguous",
                  question="balance by region by borrower type by LTV bucket",
                  expected_metric=BAL,
                  expected_dimensions=["geographic_region_obligor", "borrower_type", "ltv_bucket"],
                  expected_artifact_type="table", expected_min_columns=4,
                  expected_warnings=["Showing a table across 3 dimensions"],
                  notes="3+ dims: all requested dimensions preserved as a table (no silent "
                        "truncation); the dimension invariant sees all three applied."))
    # Filtered time-series: the value filter is applied to the mask BEFORE the
    # trend is built — never an unfiltered trend.
    C.append(case(_cat="ambig", category="ambiguous", question="balance trend where LTV above 50%",
                  expected_status="answer", expected_scope="funded",
                  expected_metric=BAL, expected_filters=[LTV],
                  expected_artifact_type="line", expected_reconciliation=True,
                  expected_min_columns=2,
                  notes="Filtered time-series: LTV filter applied before the trend is built; "
                        "reconciliation shows fewer records. Never a silently unfiltered trend."))

    return C


def main() -> int:
    import yaml
    cases = build()
    doc = {
        "meta": {
            "name": "ERE MI Agent curated calibration bank",
            "description": "250+ realistic business-user MI questions with expected "
                           "semantic behaviour + output checks. Additional to the generated "
                           "registry harness (mi_agent/mi_query_harness.py).",
            "count": len(cases),
            "fixture": "mi_agent.mi_query_harness.build_fixture (funded tape)",
            "enforced_by": "mi_agent/tests/test_mi_calibration_bank.py",
        },
        "questions": cases,
    }
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, width=100, allow_unicode=True)
    # Category counts for the console.
    from collections import Counter
    counts = Counter(c["category"] for c in cases)
    print(f"wrote {_OUT} ({len(cases)} questions)")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
