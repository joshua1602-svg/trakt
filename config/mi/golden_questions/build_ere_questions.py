#!/usr/bin/env python3
"""build_ere_questions.py

Generator for ``ere_mi_questions.yaml`` — the governed MI Agent golden-question
library (regression coverage + parameterised query-template coverage).

This is a DOCUMENTATION / TEST artifact generator. It does NOT define a semantic
registry and does not parse or execute anything — it only emits a YAML question
bank whose entries the test harness runs through the EXISTING parser / registry /
validator / executor.

Run:  python config/mi/golden_questions/build_ere_questions.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

# --------------------------------------------------------------------------- #
# Category defaults: (dataset, expected_intent, must_reconcile, supported)
# ``supported`` False => the ERE fixture lacks the field(s); the agent must give a
# controlled unsupported / missing-field response, never a hallucinated answer.
# --------------------------------------------------------------------------- #
CATEGORY_DEFAULTS = {
    "funded_kpi": ("funded", "kpi", True, True),
    "funded_breakdown_1d": ("funded", "grouped", True, True),
    "funded_crosstab_2d": ("funded", "crosstab", True, True),
    "funded_filtered_qa": ("funded", "filtered_kpi", True, True),
    "ranking_concentration": ("funded", "ranking", True, True),
    "drill_through": ("funded", "drill", True, True),
    "vintage_cohort": ("funded", "grouped", True, True),
    "funded_evolution": ("funded", "time_series", True, True),
    "pipeline": ("pipeline", "grouped", True, True),
    "pipeline_evolution": ("pipeline", "time_series", True, True),
    "forecast": ("forecast", "grouped", True, True),
    # Securitisation scale-up forecast (run-rate / KFI conversion extrapolation).
    # Dataset 'forecast' so these route to /mi/forecast/extrapolation, not a
    # point-in-time funded KPI.
    "forecast_scale": ("forecast", "forecast_scale", True, True),
    # Schedule 8 risk-limit / concentration monitor questions.
    "risk_limits": ("funded", "risk_limit", True, True),
    "data_quality": ("funded", "meta", False, True),
    "arrears_default": ("funded", "controlled_unsupported", False, False),
    "nneg_er": ("funded", "controlled_unsupported", False, False),
}

# Fields absent from the ERE fixture -> drive controlled unsupported responses.
UNSUPPORTED_FIELDS = {
    "arrears_default": ["arrears_balance", "days_in_arrears", "default_amount",
                        "allocated_losses", "recoveries_in_period"],
    "nneg_er": ["nneg_flag", "indexed_loan_to_value", "indexed_valuation_amount",
                "protected_equity_flag", "accrued_interest"],
}

# --------------------------------------------------------------------------- #
# Base questions, verbatim from the spec, grouped by category.
# --------------------------------------------------------------------------- #
QUESTIONS = {
    "funded_kpi": [
        "What is the funded balance?", "What is the current funded balance?",
        "What is total current outstanding balance?", "How many funded loans are there?",
        "How many loans are in the funded book?", "What is the weighted average current LTV?",
        "What is WA LTV?", "What is the weighted average interest rate?",
        "What is WA current interest rate?", "What is the simple average interest rate?",
        "What is the average borrower age?", "What is the weighted average borrower age?",
        "What is the average loan balance?", "What is the median loan balance?",
        "What is the largest loan?", "What is the smallest loan?",
        "What is the total original balance?", "What is the current valuation amount?",
        "What is the average current valuation?", "What is the weighted average current valuation?",
    ],
    "funded_breakdown_1d": [
        "Show balance by region.", "Show funded balance by collateral region.",
        "Show balance by obligor region.", "Show loan count by region.",
        "Show average loan balance by region.", "Show WA LTV by region.",
        "Show WA interest rate by region.", "Show average borrower age by region.",
        "Show balance by broker.", "Show loan count by broker.",
        "Show average loan balance by broker.", "Show WA LTV by broker.",
        "Show WA interest rate by broker.", "Show balance by LTV bucket.",
        "Show loan count by LTV bucket.", "Show WA interest rate by LTV bucket.",
        "Show balance by borrower age bucket.", "Show loan count by borrower age bucket.",
        "Show WA LTV by borrower age bucket.", "Show balance by origination channel.",
        "Show balance by product type.", "Show balance by account status.",
        "Show balance by borrower structure.", "Show balance by property type.",
        "Show balance by occupancy type.",
    ],
    "funded_crosstab_2d": [
        "Show balance by age bucket and LTV bucket.",
        "Show loan count by age bucket and LTV bucket.",
        "Show WA interest rate by age bucket and LTV bucket.",
        "Show WA LTV by region and age bucket.", "Show balance by broker and region.",
        "Show balance by broker and LTV bucket.", "Show balance by broker and age bucket.",
        "Show balance by region and LTV bucket.", "Show loan count by broker and LTV bucket.",
        "Show average loan balance by broker and region.", "Show WA LTV by broker and region.",
        "Show WA interest rate by broker and LTV bucket.",
        "Show balance by account status and LTV bucket.",
        "Show balance by product type and broker.",
    ],
    "funded_filtered_qa": [
        "How many borrowers are aged 70 or above?", "How many borrowers are aged 75 or above?",
        "How many borrowers are aged 80 or above?", "How many loans have LTV above 30%?",
        "How many loans have LTV above 40%?", "How many loans have LTV above 50%?",
        "How many loans have LTV above 60%?", "What is the balance of loans with LTV above 50%?",
        "What is the balance of loans with borrower age above 70?",
        "What is the balance of loans with borrower age above 80?",
        "How many loans have borrower age 70+ and LTV above 50%?",
        "What is the balance of loans with borrower age 70+ and LTV above 50%?",
        "What percentage of the book has LTV above 50%?",
        "What percentage of the book has borrower age above 80?",
        "How many loans are in the South East?", "What is the balance in the South East?",
        "How many South East loans have LTV above 50%?",
        "What is the balance of South East loans with LTV above 50%?",
        "How many joint borrowers are there?", "What is the balance for joint borrowers?",
        "What share of the book is joint borrowers?", "How many single borrowers are there?",
        "What is the balance for single borrowers?",
        "Show loans where current balance is above £200,000.",
        "Show loans where current balance is below £50,000.",
        "Show loans with interest rate above 10%.", "Show loans with interest rate below 7%.",
        "Show loans where valuation is above £500,000.",
        "Show loans where valuation is below £200,000.",
    ],
    "ranking_concentration": [
        "Which broker has the highest funded balance?",
        "Which broker has the highest average loan balance?",
        "Which broker has the most loans?", "Which broker has the highest WA LTV?",
        "Which broker has the highest WA interest rate?",
        "Which region has the highest funded balance?",
        "Which region has the highest average LTV?", "Which region has the highest interest rate?",
        "Which LTV bucket has the highest balance?", "Which age bucket has the highest balance?",
        "What are the top 10 brokers by funded balance?",
        "What are the top 10 regions by funded balance?", "What are the top 10 loans by balance?",
        "What are the top 10 loans by LTV?", "What are the oldest borrowers?",
        "What are the highest interest rate loans?", "What is the top 5 broker concentration?",
        "What is the top 10 broker concentration?", "What is the largest broker concentration?",
        "What is the largest regional concentration?",
        "What percentage of the book is in the top 5 brokers?",
        "What percentage of the book is in the top 3 regions?",
        "What percentage of the book is above 50% LTV?",
        "What percentage of the book is in borrowers aged 80+?",
    ],
    "drill_through": [
        "Show loans with LTV above 50%.", "Show loans with LTV above 60%.",
        "Show loans where borrower age is above 80.", "Show loans in the South East.",
        "Show loans for Equity Release Supermarket Limited.", "Show loans for the largest broker.",
        "Show loans in the highest LTV bucket.", "Show loans in the highest age bucket.",
        "Show loans with interest rate above 10%.", "Show loans with balance above £200,000.",
        "Show loans with balance below £50,000.", "Drill into the 50%+ LTV bucket.",
        "Drill into borrowers aged 80+.", "Drill into South East loans.",
        "Drill into broker Equity Release Supermarket Limited.",
        "Show the loan-level records behind the high LTV bucket.",
        "Show the loans included in Unknown / Missing age.", "Show loans missing LTV.",
        "Show loans missing interest rate.",
    ],
    "vintage_cohort": [
        "Show balance by origination year.", "Show loan count by origination year.",
        "Show WA LTV by origination year.", "Show WA interest rate by origination year.",
        "Show average loan balance by origination year.", "Show balance by vintage.",
        "Show vintage distribution.", "Which vintage has the highest balance?",
        "Which vintage has the highest WA LTV?", "Show 2025 vintage by broker.",
        "Show 2025 vintage by region.", "Show 2025 vintage by LTV bucket.",
        "Show balance by origination month.", "Show loan count by origination month.",
        "Show average loan balance by origination month.", "Show WA LTV by origination month.",
        "Show 2024 vintage performance.", "Show 2025 vintage concentration.",
        "Show origination cohort by broker.", "Show origination cohort by region.",
    ],
    "funded_evolution": [
        "Show funded balance evolution by month.", "Show funded balance trend by reporting date.",
        "Show loan count evolution by month.", "Show WA LTV evolution by month.",
        "Show WA interest rate evolution by month.", "Show average borrower age evolution by month.",
        "Show funded balance evolution from October to November.",
        "Compare October and November funded balance.",
        "Compare October and November loan count.", "Compare October and November WA LTV.",
        "Show monthly balance evolution by broker.", "Show monthly balance evolution by region.",
        "Show monthly balance evolution by LTV bucket.",
        "Show monthly loan count evolution by broker.",
        "Show October, November, December balance evolution.",
        "Show balance trend for Equity Release Supermarket Limited.",
        "Show LTV bucket evolution over time.", "Show age bucket evolution over time.",
        "Show regional concentration evolution over time.",
        "Show broker concentration evolution over time.",
    ],
    "pipeline": [
        "What is the pipeline amount?", "How many pipeline cases are there?",
        "What is the weighted expected funded amount?", "What is the pipeline conversion rate?",
        "What is the pipeline amount by stage?", "What is the pipeline case count by stage?",
        "What is the pipeline amount by broker?", "What is the pipeline amount by region?",
        "What is the pipeline amount by expected completion month?",
        "Which broker has the largest pipeline?", "Which stage has the largest pipeline?",
        "How much pipeline is expected to complete next month?", "How much pipeline is overdue?",
        "How much pipeline is current month?", "Show overdue pipeline cases.",
        "Show pipeline by expected completion date.", "Show pipeline stage distribution.",
        "Show pipeline weighted expected amount by stage.",
        "Show pipeline weighted expected amount by broker.", "Show pipeline amount by product.",
        "Show pipeline amount by LTV bucket.", "Show pipeline amount by age bucket.",
        "Show pipeline amount by source file.", "Show current month pipeline completions expected.",
    ],
    "pipeline_evolution": [
        "Show pipeline amount evolution by week.", "Show pipeline amount evolution by month.",
        "Show pipeline case count evolution by week.", "Show pipeline case count evolution by month.",
        "Show pipeline amount by stage over time.", "Show pipeline cases by stage over time.",
        "Show expected completions by month.", "Show weighted expected funded amount by month.",
        "Show pipeline conversion basis over time.", "Show pipeline by broker over time.",
        "Show pipeline by stage for October and November.",
        "Compare October and November pipeline amount.", "Compare latest pipeline with prior pipeline.",
        "Show pipeline growth from October to November.", "Show stage migration over time.",
    ],
    "forecast": [
        "What is the forecast funded balance?", "What is funded balance plus weighted pipeline?",
        "What is the expected funded balance?", "What is the weighted expected pipeline contribution?",
        "What is the forecast loan count?", "Show forecast balance by region.",
        "Show forecast balance by broker.", "Show forecast balance by LTV bucket.",
        "Show forecast balance by expected completion month.",
        "How much of the forecast comes from funded book?",
        "How much of the forecast comes from pipeline?", "What is the forecast bridge?",
        "Show funded vs pipeline contribution.", "Show forecast completion basis.",
        "Show historical conversion basis.", "Which stages use historical rates?",
        "Which stages use config fallback rates?",
        "How much active pipeline is excluded from weighting?",
        "How much pipeline is excluded because of missing probability?",
        "How much pipeline is excluded because of withdrawn status?",
        "Show forecast balance by stage.", "Show forecast balance by broker and stage.",
    ],
    "forecast_scale": [
        "When do we reach £25m funded balance?", "When do we reach £50m funded balance?",
        "When do we reach £75m funded balance?", "When do we reach £100m funded balance?",
        "When do we reach £150m funded balance?",
        "What is the current completion run rate?",
        "What is the annualised completion run rate?",
        "Show the funded balance extrapolation curve.",
        "Show the scale-up forecast.", "What is the downside forecast?",
        "What is the base forecast?", "What is the upside forecast?",
        "How much pipeline is needed to reach £100m?",
        "What happens if completion run rate falls by 25%?",
        "What completion rate is assumed from KFI to completion?",
        "What is the expected time to securitisation scale?",
        "Show milestone dates to funding thresholds.",
        "Compare current weighted pipeline forecast with run-rate extrapolation.",
        "What is the 8-week completion run rate?", "What is the 12-week completion run rate?",
        "Project funded balance over the next twelve months.",
        "When does the book reach scale?",
    ],
    "risk_limits": [
        "Are we within our concentration limits?", "Which limits are breached?",
        "Show risk limit headroom.", "Show the Schedule 8 concentration tests.",
        "What is the largest geographic concentration versus limit?",
        "Show geographic concentration against limits.",
        "Show broker concentration against limits.",
        "Which concentration test is closest to breach?",
        "Show the single largest loan against the large-loan limit.",
        "Are any regional limits breached?", "Show limit utilisation by category.",
        "What is the headroom on the London concentration limit?",
        "Show concentration limit status.", "Which risk limits need review?",
        "Show concentration breaches.", "Are we over any limit?",
        "Show the risk limit pass/warn/fail summary.",
        "What is the nearest limit breach?",
    ],
    "data_quality": [
        "Which fields are missing?", "Which required MI fields are missing?",
        "How complete is interest rate?", "How complete is LTV?", "How complete is borrower age?",
        "How complete is broker?", "How complete is region?", "How much balance is missing age?",
        "How much balance is missing LTV?", "How much balance is missing broker?",
        "How much balance is missing interest rate?", "Why does this chart not reconcile?",
        "What is excluded from this result?", "What records are in Unknown / Missing?",
        "Which fields are sourced from funded files?", "Which fields are sourced from pipeline files?",
        "Is interest rate sourced from the funded book?", "What is the source of current interest rate?",
        "What is the source of current valuation?", "What is the source of borrower age?",
        "What is the source of LTV?", "Show data completeness by field.",
        "Show data completeness by balance.", "Show records missing required MI fields.",
        "Show source notes for this answer.",
    ],
    "arrears_default": [
        "How many loans are in arrears?", "What is the arrears balance?",
        "What percentage of the book is in arrears?", "Show arrears balance by region.",
        "Show arrears balance by broker.", "Show arrears balance by LTV bucket.",
        "Show arrears balance by age bucket.", "Show loans more than 30 days in arrears.",
        "Show loans more than 60 days in arrears.", "Show loans more than 90 days in arrears.",
        "What is the balance of loans more than 90 days in arrears?",
        "Show number of days in arrears by broker.", "Show arrears trend by month.",
        "Show arrears evolution over time.", "Show arrears by vintage.",
        "Show arrears by origination year.", "Show defaulted loans.",
        "How many loans are defaulted?", "What is the defaulted balance?",
        "Show default amount by region.", "Show default amount by broker.",
        "Show default rate by vintage.", "Show default balance evolution over time.",
        "Show recoveries in period.", "Show cumulative recoveries.",
        "Show loss amount by vintage.", "Show defaulted loans with LTV above 50%.",
        "Show arrears by borrower age bucket.", "Show arrears by current LTV bucket.",
    ],
    "nneg_er": [
        "How many loans have negative equity guarantee?", "What percentage of the book has NNEG?",
        "Show NNEG exposure by region.", "Show NNEG exposure by LTV bucket.",
        "Show NNEG exposure by borrower age bucket.", "Show NNEG exposure by vintage.",
        "Show loans with NNEG and LTV above 50%.", "Show loans with NNEG and borrower age above 80.",
        "Show indexed LTV by age bucket.", "Show indexed LTV by region.",
        "Show indexed value by region.", "Show estimated equity by loan.",
        "Show equity by LTV bucket.", "Show loans with low remaining equity.",
        "Show loans with current balance above current valuation.",
        "Show no-negative-equity risk by age and LTV.", "Show NNEG risk concentration by region.",
        "Show NNEG risk concentration by broker.", "Show NNEG risk by origination vintage.",
        "Show protected equity loans.", "Show protected equity flag by product.",
        "Show roll-up interest exposure.", "Show accrued interest by vintage.",
        "Show interest roll-up by age bucket.",
    ],
}

# --------------------------------------------------------------------------- #
# Parameterised templates -> >1,000 supported phrasings without hardcoding them.
# --------------------------------------------------------------------------- #
PARAMETERISED = {
    # id of the representative base question -> template axes
    "How many loans have LTV above 50%?": {
        "field": "current_loan_to_value", "value_type": "percentage",
        "thresholds": [25, 30, 40, 45, 50, 60, 70],
        "operators": ["above", "over", "greater than", "below", "less than",
                      "at least", "between"],
        "field_synonyms": ["LTV", "current LTV", "loan to value"],
        "metric_synonyms": ["how many loans", "what balance", "what share"],
        "variations": [
            "How many loans have LTV above 30%?", "What balance is over 45% LTV?",
            "Show loans where LTV is between 20% and 40%.",
            "What percentage of the book has LTV below 25%?",
        ],
    },
    "How many borrowers are aged 70 or above?": {
        "field": "youngest_borrower_age", "value_type": "age",
        "thresholds": [65, 70, 75, 80, 85],
        "operators": ["aged ... or above", "above", "over", "below", "between"],
        "field_synonyms": ["borrower age", "age", "customer age", "youngest age"],
        "metric_synonyms": ["how many borrowers", "what balance", "what share"],
        "variations": [
            "How many borrowers are aged 75 or above?",
            "What balance is for borrowers over 80?",
            "Show borrowers aged between 70 and 85.",
        ],
    },
    "Show loans where current balance is above £200,000.": {
        "field": "current_outstanding_balance", "value_type": "currency",
        "thresholds": [50000, 100000, 200000, 300000, 500000],
        "operators": ["above", "over", "below", "less than", "between"],
        "field_synonyms": ["balance", "current balance", "exposure"],
        "metric_synonyms": ["show loans", "how many loans", "what balance"],
        "variations": [
            "Show loans where current balance is below £50,000.",
            "How many loans have exposure above £100,000?",
        ],
    },
    "Show loans with interest rate above 10%.": {
        "field": "current_interest_rate", "value_type": "percentage",
        "thresholds": [5, 6, 7, 8, 10, 12],
        "operators": ["above", "over", "below", "less than", "between"],
        "field_synonyms": ["interest rate", "rate", "current rate", "coupon"],
        "metric_synonyms": ["show loans", "how many loans", "what balance"],
        "variations": [
            "Show loans with interest rate below 7%.",
            "How many loans have rate over 8%?",
        ],
    },
    "Compare October and November funded balance.": {
        "field": "reporting_date", "value_type": "period",
        "thresholds": ["October", "November", "December"],
        "operators": ["compare", "evolution from ... to", "over time"],
        "field_synonyms": ["funded balance", "loan count", "WA LTV"],
        "metric_synonyms": ["compare", "show evolution", "trend"],
        "variations": [
            "Compare November and December funded balance.",
            "Show funded balance evolution from October to December.",
        ],
    },
}


def _entry(qid: str, category: str, question: str) -> dict:
    dataset, intent, reconcile, supported = CATEGORY_DEFAULTS[category]
    entry = {
        "id": qid,
        "category": category,
        "question": question,
        "dataset": dataset,
        "expected_intent": intent,
        "must_reconcile": reconcile,
        "supported": supported,
    }
    if not supported:
        entry["expected_response_type"] = "controlled_missing_field"
        entry["expected_missing_fields"] = UNSUPPORTED_FIELDS.get(category, [])
    tmpl = PARAMETERISED.get(question)
    if tmpl:
        entry["parameterised_template"] = {
            "field": tmpl["field"], "value_type": tmpl["value_type"],
            "operators": tmpl["operators"], "thresholds": tmpl["thresholds"],
        }
        entry["variations"] = list(tmpl["variations"])
        entry["variation_axes"] = {
            "thresholds": tmpl["thresholds"], "operators": tmpl["operators"],
            "field_synonyms": tmpl["field_synonyms"],
            "metric_synonyms": tmpl["metric_synonyms"],
        }
    return entry


def build() -> dict:
    questions = []
    counters: dict = {}
    for category, items in QUESTIONS.items():
        for q in items:
            counters[category] = counters.get(category, 0) + 1
            qid = f"{category}_{counters[category]:03d}"
            questions.append(_entry(qid, category, q))
    return {
        "version": 1,
        "client": "ERE",
        "description": ("Governed MI Agent golden-question library. Base questions "
                        "are regression coverage; parameterised templates + "
                        "variation_axes demonstrate >1,000 supported phrasings. The "
                        "library proves coverage; it does NOT constrain the free-form "
                        "agent to exact entries."),
        "categories": sorted(QUESTIONS.keys()),
        "controlled_unsupported_categories": ["arrears_default", "nneg_er"],
        "questions": questions,
    }


def main() -> None:
    out = Path(__file__).resolve().parent / "ere_mi_questions.yaml"
    bank = build()
    header = ("# AUTO-GENERATED by config/mi/golden_questions/build_ere_questions.py\n"
              "# Governed MI Agent golden-question library (regression + template coverage).\n"
              "# Edit the generator, then re-run it; do not hand-edit this file.\n")
    out.write_text(header + yaml.safe_dump(bank, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    n = len(bank["questions"])
    tmpl = sum(1 for q in bank["questions"] if "variation_axes" in q)
    print(f"wrote {out} | {n} base questions | {tmpl} parameterised templates")


if __name__ == "__main__":
    main()
