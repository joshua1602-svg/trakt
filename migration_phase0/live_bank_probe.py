#!/usr/bin/env python3
"""live_bank_probe — run the accepted question bank against the LIVE book.

RUNS WHERE THE DATA IS. This is meant to be executed on the App Service, in the
container that already holds the governed tape, so no client data has to travel
anywhere to be measured.

WHAT IT EMITS, AND WHAT IT DELIBERATELY DOES NOT
------------------------------------------------
Diagnostics, never figures. For each question it records the route taken,
whether it answered or refused, the governed outcome code, the reason it
refused, and the SHAPE of what it bound — metric key, dimension keys, filter
FIELDS. It never records a balance, a count, a category value or the prose of a
successful answer.

That is what makes the output file shareable: it carries the failure surface and
no client information. Whoever reads it can see WHICH questions fail and WHY,
and cannot see what the book contains.

Refusal text is passed through a redactor anyway, because a refusal may quote a
figure back ("the scope spans ..."). Currency amounts and long numbers are
replaced; ISO dates are kept, because a date mismatch is a diagnosis.

USAGE
-----
    python live_bank_probe.py --scope direct --out /home/probe_result.json

    --scope     source-portfolio lens: direct | acquired | total | a cohort id
                (default: direct). Sent as `sourcePortfolioLens`, which is the
                field `QueryRequest` actually declares — an unrecognised name is
                dropped in silence by the request model and the run quietly
                measures the whole book instead.
    --out       where to write the JSON                        (required)
    --limit     first N questions only, for a quick smoke test
    --as-of     reporting date override, e.g. 2026-06-30
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# Auth is a FRONT-DOOR concern and this is not the front door: we are calling the
# governed service in-process, inside the container, as an operator tool. Setting
# it here affects THIS process only — the running gunicorn workers are untouched.
os.environ["MI_AGENT_AUTH_ENABLED"] = "false"

#: Currency amounts and long/decimal numbers are redacted from any text that
#: reaches the output. ISO dates are matched FIRST and preserved, because
#: "the portfolios are at different reporting dates (2025-11-30 vs 2026-06-30)"
#: is exactly the kind of finding this probe exists to surface.
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_FIGURE = re.compile(r"[£$€]\s?[\d,.]+\s*(?:[kKmMbB]{1,2}|MM)?|\b\d[\d,]{3,}(?:\.\d+)?\b|\b\d+\.\d+\b")


def _redact(text):
    """Text with money and long numbers removed, ISO dates kept."""
    if not text:
        return ""
    keep = {}

    def _stash(m):
        token = "\x00%d\x00" % len(keep)
        keep[token] = m.group(0)
        return token

    out = _ISO_DATE.sub(_stash, str(text))
    out = _FIGURE.sub("[figure]", out)
    for token, original in keep.items():
        out = out.replace(token, original)
    return out.strip()


def _keys(value):
    """Field NAMES from a filter mapping. Never the values they select on."""
    if isinstance(value, dict):
        return sorted(str(k) for k in value)
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


def _scope_applied(env, meta):
    """What scope the ANSWER actually covers — never what we asked for.

    The requested lens travelling in the body is not evidence that it was
    applied. Two independent witnesses are recorded here, because different
    routes leave different traces:

      * routed answers carry `portfolioScope`, stamped by
        `mi_service._stamp_routed_scope` with the scope the answer HAS
        (an un-narrowed route is stamped Total, deliberately);
      * executed answers realise the lens as a provenance filter, so
        `portfolio_id` appears in `filter_fields`, recorded separately.

    `fell_back_to_total` is the one that matters most: it is the platform
    saying it could not honour the requested context and widened.
    """
    scope = env.get("portfolioScope")
    if not isinstance(scope, dict):
        return {"lens_applied": meta.get("lensApplied")}
    return {
        "context_id": scope.get("context_id"),
        "context_kind": scope.get("context_kind"),
        "requested_context_id": scope.get("requested_context_id"),
        "fell_back_to_total": scope.get("fell_back_to_total"),
        "lens_applied": meta.get("lensApplied"),
    }


QUESTIONS = [
 {
  "id": "Q01A",
  "bank": "BANK75",
  "q": "How many loans are to borrowers over 55 with LTV above 50%?"
 },
 {
  "id": "Q01B",
  "bank": "BANK75",
  "q": "How many loans have a borrower older than 55 and an LTV greater than 50%?"
 },
 {
  "id": "Q01C",
  "bank": "BANK75",
  "q": "Count the loans where borrower age is above 55 and current LTV is over 50%."
 },
 {
  "id": "Q02A",
  "bank": "BANK75",
  "q": "What is the balance of loans to borrowers over 75 with LTV above 40%?"
 },
 {
  "id": "Q02B",
  "bank": "BANK75",
  "q": "How much outstanding balance do we have where borrower age exceeds 75 and LTV is over 40%?"
 },
 {
  "id": "Q02C",
  "bank": "BANK75",
  "q": "Show the total balance for loans with borrowers older than 75 and current LTV greater than 40%."
 },
 {
  "id": "Q03A",
  "bank": "BANK75",
  "q": "How many drawdown loans have LTV above 50%?"
 },
 {
  "id": "Q03B",
  "bank": "BANK75",
  "q": "Of the drawdown loans, how many are over 50% LTV?"
 },
 {
  "id": "Q03C",
  "bank": "BANK75",
  "q": "Count drawdown cases where current LTV exceeds 50%."
 },
 {
  "id": "Q04A",
  "bank": "BANK75",
  "q": "What is the balance of Direct-book loans in London to borrowers over 75?"
 },
 {
  "id": "Q04B",
  "bank": "BANK75",
  "q": "How much balance is in the Direct portfolio for London loans where the borrower is older than 75?"
 },
 {
  "id": "Q04C",
  "bank": "BANK75",
  "q": "Show total outstanding balance for London loans in the Direct book with borrower age above 75."
 },
 {
  "id": "Q05A",
  "bank": "BANK75",
  "q": "What is the weighted-average LTV of lump sum loans in the Direct book?"
 },
 {
  "id": "Q05B",
  "bank": "BANK75",
  "q": "For Direct-book lump sum loans, what is the weighted average current LTV?"
 },
 {
  "id": "Q05C",
  "bank": "BANK75",
  "q": "Give me WA LTV for lump sum lending in the Direct portfolio."
 },
 {
  "id": "Q06A",
  "bank": "BANK75",
  "q": "Summarise the portfolio."
 },
 {
  "id": "Q06B",
  "bank": "BANK75",
  "q": "Give me a management summary of the current book."
 },
 {
  "id": "Q06C",
  "bank": "BANK75",
  "q": "Give me a concise overview of the funded portfolio."
 },
 {
  "id": "Q07A",
  "bank": "BANK75",
  "q": "Compare the Direct and Acquired books."
 },
 {
  "id": "Q07B",
  "bank": "BANK75",
  "q": "How do the Direct and Acquired portfolios differ?"
 },
 {
  "id": "Q07C",
  "bank": "BANK75",
  "q": "Give me a side-by-side comparison of Direct versus Acquired."
 },
 {
  "id": "Q08A",
  "bank": "BANK75",
  "q": "Where are our largest concentrations today?"
 },
 {
  "id": "Q08B",
  "bank": "BANK75",
  "q": "What are the biggest concentration exposures in the book?"
 },
 {
  "id": "Q08C",
  "bank": "BANK75",
  "q": "Summarise the main current portfolio concentrations."
 },
 {
  "id": "Q09A",
  "bank": "BANK75",
  "q": "Are any concentration limits currently breached or close to breach?"
 },
 {
  "id": "Q09B",
  "bank": "BANK75",
  "q": "Which of our concentration tests are most at risk today?"
 },
 {
  "id": "Q09C",
  "bank": "BANK75",
  "q": "Summarise our current position against the concentration limits."
 },
 {
  "id": "Q10A",
  "bank": "BANK75",
  "q": "Summarise the current pipeline."
 },
 {
  "id": "Q10B",
  "bank": "BANK75",
  "q": "Give me an overview of the pipeline by size and stage."
 },
 {
  "id": "Q10C",
  "bank": "BANK75",
  "q": "What does the current pipeline look like?"
 },
 {
  "id": "Q11A",
  "bank": "BANK75",
  "q": "Show a table of balance by LTV bucket and ticket-size bucket."
 },
 {
  "id": "Q11B",
  "bank": "BANK75",
  "q": "Cross-tab the outstanding balance by LTV band and ticket-size band."
 },
 {
  "id": "Q11C",
  "bank": "BANK75",
  "q": "Break the balance down by both LTV bucket and ticket-size bucket."
 },
 {
  "id": "Q12A",
  "bank": "BANK75",
  "q": "Chart the balance by LTV bucket and borrower-age bucket."
 },
 {
  "id": "Q12B",
  "bank": "BANK75",
  "q": "Show me balance split by both LTV band and age band."
 },
 {
  "id": "Q12C",
  "bank": "BANK75",
  "q": "Plot portfolio balance across LTV buckets and borrower-age buckets."
 },
 {
  "id": "Q13A",
  "bank": "BANK75",
  "q": "Show a table of balance by LTV bucket and interest-rate bucket."
 },
 {
  "id": "Q13B",
  "bank": "BANK75",
  "q": "Cross-tab balance by LTV band and interest-rate band."
 },
 {
  "id": "Q13C",
  "bank": "BANK75",
  "q": "Break down outstanding balance by both LTV bucket and rate bucket."
 },
 {
  "id": "Q14A",
  "bank": "BANK75",
  "q": "Show loan count by region and product type."
 },
 {
  "id": "Q14B",
  "bank": "BANK75",
  "q": "Give me a table of loan numbers split by region and loan type."
 },
 {
  "id": "Q14C",
  "bank": "BANK75",
  "q": "Break the number of loans down by both geographic region and product type."
 },
 {
  "id": "Q15A",
  "bank": "BANK75",
  "q": "For the Direct book, show balance by broker and product type."
 },
 {
  "id": "Q15B",
  "bank": "BANK75",
  "q": "Break Direct-book balance down by both broker channel and loan type."
 },
 {
  "id": "Q15C",
  "bank": "BANK75",
  "q": "Give me a broker-by-product balance table for the Direct portfolio."
 },
 {
  "id": "Q16A",
  "bank": "BANK75",
  "q": "For drawdown loans, show balance by region and LTV bucket."
 },
 {
  "id": "Q16B",
  "bank": "BANK75",
  "q": "Break drawdown balance down by both geography and LTV band."
 },
 {
  "id": "Q16C",
  "bank": "BANK75",
  "q": "Show me the regional balance by LTV bucket for drawdown loans."
 },
 {
  "id": "Q17A",
  "bank": "BANK75",
  "q": "For the Direct book, show balance by LTV bucket, ticket-size bucket and borrower-age bucket."
 },
 {
  "id": "Q17B",
  "bank": "BANK75",
  "q": "Give me a table of Direct-book balance split by LTV band, ticket-size band and age band."
 },
 {
  "id": "Q17C",
  "bank": "BANK75",
  "q": "Break Direct portfolio balance down across LTV, ticket size and borrower age."
 },
 {
  "id": "Q18A",
  "bank": "BANK75",
  "q": "How did the book change in the last month?"
 },
 {
  "id": "Q18B",
  "bank": "BANK75",
  "q": "What changed in the portfolio since last month?"
 },
 {
  "id": "Q18C",
  "bank": "BANK75",
  "q": "Give me a summary of how the funded book moved over the last month."
 },
 {
  "id": "Q19A",
  "bank": "BANK75",
  "q": "How did the Direct book change last month?"
 },
 {
  "id": "Q19B",
  "bank": "BANK75",
  "q": "What changed in the Direct portfolio since last month?"
 },
 {
  "id": "Q19C",
  "bank": "BANK75",
  "q": "Summarise the month-on-month movement in the Direct book."
 },
 {
  "id": "Q20A",
  "bank": "BANK75",
  "q": "How did drawdown loans change last month?"
 },
 {
  "id": "Q20B",
  "bank": "BANK75",
  "q": "What changed in the drawdown book since last month?"
 },
 {
  "id": "Q20C",
  "bank": "BANK75",
  "q": "Summarise the month-on-month movement for drawdown loans."
 },
 {
  "id": "Q21A",
  "bank": "BANK75",
  "q": "Which region added the most balance last month for loans with LTV above 50%?"
 },
 {
  "id": "Q21B",
  "bank": "BANK75",
  "q": "For loans over 50% LTV, which region contributed the most balance growth since last month?"
 },
 {
  "id": "Q21C",
  "bank": "BANK75",
  "q": "Among loans with current LTV above 50%, where did balance increase the most over the last month?"
 },
 {
  "id": "Q22A",
  "bank": "BANK75",
  "q": "Which source portfolio contributed most to balance growth last month?"
 },
 {
  "id": "Q22B",
  "bank": "BANK75",
  "q": "Did Direct or Acquired add more balance during the last month?"
 },
 {
  "id": "Q22C",
  "bank": "BANK75",
  "q": "Which of the Direct and Acquired books drove more of the month-on-month balance increase?"
 },
 {
  "id": "Q23A",
  "bank": "BANK75",
  "q": "When will we reach \u00a3100m of funded loans?"
 },
 {
  "id": "Q23B",
  "bank": "BANK75",
  "q": "At the current trajectory, when do we get to \u00a3100 million?"
 },
 {
  "id": "Q23C",
  "bank": "BANK75",
  "q": "When does the funded book reach the \u00a3100m milestone?"
 },
 {
  "id": "Q24A",
  "bank": "BANK75",
  "q": "At the current run rate, when will we reach \u00a3250m?"
 },
 {
  "id": "Q24B",
  "bank": "BANK75",
  "q": "When are we expected to get to \u00a3250 million of funded loans?"
 },
 {
  "id": "Q24C",
  "bank": "BANK75",
  "q": "Based on the current run rate, when does the book reach \u00a3250m?"
 },
 {
  "id": "Q25A",
  "bank": "BANK75",
  "q": "Do we expect to breach any concentration tests?"
 },
 {
  "id": "Q25B",
  "bank": "BANK75",
  "q": "Are any concentration limits likely to be breached as the book grows?"
 },
 {
  "id": "Q25C",
  "bank": "BANK75",
  "q": "Based on the current book and forward pipeline, which concentration tests are we at risk of breaching?"
 },
 {
  "id": "CFO01",
  "bank": "CFO91",
  "q": "What is our total funded balance?"
 },
 {
  "id": "CFO02",
  "bank": "CFO91",
  "q": "How many loans do we have?"
 },
 {
  "id": "CFO03",
  "bank": "CFO91",
  "q": "What is the average loan balance?"
 },
 {
  "id": "CFO04",
  "bank": "CFO91",
  "q": "What is our weighted average LTV?"
 },
 {
  "id": "CFO05",
  "bank": "CFO91",
  "q": "What is the average borrower age?"
 },
 {
  "id": "CFO06",
  "bank": "CFO91",
  "q": "What is the average interest rate on the book?"
 },
 {
  "id": "CFO07",
  "bank": "CFO91",
  "q": "Show balance by region."
 },
 {
  "id": "CFO08",
  "bank": "CFO91",
  "q": "Show loan count by region."
 },
 {
  "id": "CFO09",
  "bank": "CFO91",
  "q": "Show balance by product type."
 },
 {
  "id": "CFO10",
  "bank": "CFO91",
  "q": "Show balance by broker channel."
 },
 {
  "id": "CFO11",
  "bank": "CFO91",
  "q": "Show balance by origination channel."
 },
 {
  "id": "CFO12",
  "bank": "CFO91",
  "q": "Show balance by LTV bucket."
 },
 {
  "id": "CFO13",
  "bank": "CFO91",
  "q": "Show balance by age bucket."
 },
 {
  "id": "CFO14",
  "bank": "CFO91",
  "q": "Show loan count by product type."
 },
 {
  "id": "CFO15",
  "bank": "CFO91",
  "q": "What is the balance in the direct book?"
 },
 {
  "id": "CFO16",
  "bank": "CFO91",
  "q": "What is the balance in the acquired book?"
 },
 {
  "id": "CFO17",
  "bank": "CFO91",
  "q": "Summarise the portfolio."
 },
 {
  "id": "CFO18",
  "bank": "CFO91",
  "q": "Give me a portfolio overview."
 },
 {
  "id": "CFO19",
  "bank": "CFO91",
  "q": "Show funded balance over time."
 },
 {
  "id": "CFO20",
  "bank": "CFO91",
  "q": "Show loan count over time."
 },
 {
  "id": "CFO21",
  "bank": "CFO91",
  "q": "Show funded balance evolution."
 },
 {
  "id": "CFO22",
  "bank": "CFO91",
  "q": "How has the book grown over the last 3 months?"
 },
 {
  "id": "CFO23",
  "bank": "CFO91",
  "q": "Show average LTV over time."
 },
 {
  "id": "CFO24",
  "bank": "CFO91",
  "q": "Show average balance over time."
 },
 {
  "id": "CFO25",
  "bank": "CFO91",
  "q": "What has changed since last month?"
 },
 {
  "id": "CFO26",
  "bank": "CFO91",
  "q": "How did the book move last month?"
 },
 {
  "id": "CFO27",
  "bank": "CFO91",
  "q": "How did balance change since last month?"
 },
 {
  "id": "CFO28",
  "bank": "CFO91",
  "q": "How did average LTV change since last month?"
 },
 {
  "id": "CFO29",
  "bank": "CFO91",
  "q": "Compare this month with last month."
 },
 {
  "id": "CFO30",
  "bank": "CFO91",
  "q": "How does the current month compare with the previous month?"
 },
 {
  "id": "CFO31",
  "bank": "CFO91",
  "q": "Compare the direct and acquired books."
 },
 {
  "id": "CFO32",
  "bank": "CFO91",
  "q": "What is the balance difference between this month and last month?"
 },
 {
  "id": "CFO33",
  "bank": "CFO91",
  "q": "Show the balance bridge for last month."
 },
 {
  "id": "CFO34",
  "bank": "CFO91",
  "q": "What drove the movement in the book last month?"
 },
 {
  "id": "CFO35",
  "bank": "CFO91",
  "q": "What is the balance for loans with LTV above 50%?"
 },
 {
  "id": "CFO36",
  "bank": "CFO91",
  "q": "How many loans have LTV above 50%?"
 },
 {
  "id": "CFO37",
  "bank": "CFO91",
  "q": "Balance by region for loans with LTV above 50%."
 },
 {
  "id": "CFO38",
  "bank": "CFO91",
  "q": "For loans with LTV above 50%, balance by region"
 },
 {
  "id": "CFO39",
  "bank": "CFO91",
  "q": "What is the balance for loans with borrower age above 75?"
 },
 {
  "id": "CFO40",
  "bank": "CFO91",
  "q": "For loans with borrower age above 75, balance by region"
 },
 {
  "id": "CFO41",
  "bank": "CFO91",
  "q": "Balance by region for loans with borrower age above 75."
 },
 {
  "id": "CFO42",
  "bank": "CFO91",
  "q": "How many loans are above \u00a3300,000?"
 },
 {
  "id": "CFO43",
  "bank": "CFO91",
  "q": "What is the balance for loans in London?"
 },
 {
  "id": "CFO44",
  "bank": "CFO91",
  "q": "How many loans have an interest rate above 7%?"
 },
 {
  "id": "CFO45",
  "bank": "CFO91",
  "q": "Balance by product type for loans with LTV above 40%."
 },
 {
  "id": "CFO46",
  "bank": "CFO91",
  "q": "How many drawdown loans do we have?"
 },
 {
  "id": "CFO47",
  "bank": "CFO91",
  "q": "Which region has the largest balance?"
 },
 {
  "id": "CFO48",
  "bank": "CFO91",
  "q": "Which region has the smallest balance?"
 },
 {
  "id": "CFO49",
  "bank": "CFO91",
  "q": "Which broker channel has the largest balance?"
 },
 {
  "id": "CFO50",
  "bank": "CFO91",
  "q": "Which region added the most balance since last month?"
 },
 {
  "id": "CFO51",
  "bank": "CFO91",
  "q": "Which region lost the most balance since last month?"
 },
 {
  "id": "CFO52",
  "bank": "CFO91",
  "q": "Which two regions added the most balance since last month?"
 },
 {
  "id": "CFO53",
  "bank": "CFO91",
  "q": "Which three regions added the most balance since last month?"
 },
 {
  "id": "CFO54",
  "bank": "CFO91",
  "q": "Which region grew fastest in balance since last month?"
 },
 {
  "id": "CFO55",
  "bank": "CFO91",
  "q": "Which broker channel added the most balance since last month?"
 },
 {
  "id": "CFO56",
  "bank": "CFO91",
  "q": "Which region added the most balance since last month for loans with LTV above 50%?"
 },
 {
  "id": "CFO57",
  "bank": "CFO91",
  "q": "For loans with LTV above 50%, which region added the most balance since last month?"
 },
 {
  "id": "CFO58",
  "bank": "CFO91",
  "q": "Which product type grew the most since last month?"
 },
 {
  "id": "CFO59",
  "bank": "CFO91",
  "q": "What proportion of the book is in London?"
 },
 {
  "id": "CFO60",
  "bank": "CFO91",
  "q": "Show product concentration."
 },
 {
  "id": "CFO61",
  "bank": "CFO91",
  "q": "Show broker concentration."
 },
 {
  "id": "CFO62",
  "bank": "CFO91",
  "q": "Which product type has the largest share of the book?"
 },
 {
  "id": "CFO63",
  "bank": "CFO91",
  "q": "What share of the book is drawdown?"
 },
 {
  "id": "CFO64",
  "bank": "CFO91",
  "q": "Show origination channel concentration."
 },
 {
  "id": "CFO65",
  "bank": "CFO91",
  "q": "What proportion of the book is in the acquired portfolio?"
 },
 {
  "id": "CFO66",
  "bank": "CFO91",
  "q": "What is the pipeline balance?"
 },
 {
  "id": "CFO67",
  "bank": "CFO91",
  "q": "How many cases are in the pipeline?"
 },
 {
  "id": "CFO68",
  "bank": "CFO91",
  "q": "Show the pipeline by stage."
 },
 {
  "id": "CFO69",
  "bank": "CFO91",
  "q": "How has the pipeline evolved?"
 },
 {
  "id": "CFO70",
  "bank": "CFO91",
  "q": "Show pipeline evolution by stage."
 },
 {
  "id": "CFO71",
  "bank": "CFO91",
  "q": "What is the value of outstanding offers?"
 },
 {
  "id": "CFO72",
  "bank": "CFO91",
  "q": "Are any of our concentration limits at risk?"
 },
 {
  "id": "CFO73",
  "bank": "CFO91",
  "q": "Which of our limits are currently most at risk?"
 },
 {
  "id": "CFO74",
  "bank": "CFO91",
  "q": "At the current run rate, when do we reach \u00a3250m of loans?"
 },
 {
  "id": "CFO75",
  "bank": "CFO91",
  "q": "What is our largest single-name exposure?"
 },
 {
  "id": "CFO76",
  "bank": "CFO91",
  "q": "Show the largest 10 loan exposures."
 },
 {
  "id": "CFO77",
  "bank": "CFO91",
  "q": "Which region grew the most?"
 },
 {
  "id": "CFO78",
  "bank": "CFO91",
  "q": "What changed?"
 },
 {
  "id": "CFO79",
  "bank": "CFO91",
  "q": "Which region added the most?"
 },
 {
  "id": "CFO80",
  "bank": "CFO91",
  "q": "Show me the trend."
 },
 {
  "id": "CFO81",
  "bank": "CFO91",
  "q": "How much is in the Highgate Mortgages book?"
 },
 {
  "id": "CFO82",
  "bank": "CFO91",
  "q": "Show balance by risk grade."
 },
 {
  "id": "CFO83",
  "bank": "CFO91",
  "q": "What is our arrears rate?"
 },
 {
  "id": "CFO84",
  "bank": "CFO91",
  "q": "Show the cure rate by vintage."
 },
 {
  "id": "CFO85",
  "bank": "CFO91",
  "q": "What is the NNEG exposure?"
 },
 {
  "id": "CFO86",
  "bank": "CFO91",
  "q": "Show roll rates by bucket."
 },
 {
  "id": "CFO87",
  "bank": "CFO91",
  "q": "How many loans have a Risk Score above 700?"
 },
 {
  "id": "CFO88",
  "bank": "CFO91",
  "q": "Compare us with the market."
 },
 {
  "id": "CFO89",
  "bank": "CFO91",
  "q": "Show balance by servicer."
 },
 {
  "id": "CFO90",
  "bank": "CFO91",
  "q": "What will the book be worth in five years?"
 },
 {
  "id": "CFO91",
  "bank": "CFO91",
  "q": "Which cohort is performing best?"
 }
]


def _locate_app_root() -> None:
    """Put the deployed application on `sys.path`, wherever it was unpacked.

    On App Service the code does not live where you are standing: Oryx extracts
    it to /tmp/<build-id>/ and the site directory holds only the compressed
    artefact. Discovering it here means the probe can be launched from anywhere,
    including a fresh SSH session in /home, and does not depend on the operator
    knowing the build id.
    """
    import glob

    if os.path.isdir("mi_agent_api"):
        sys.path.insert(0, os.getcwd())
        return
    candidates = [os.path.dirname(p) for p in glob.glob("/tmp/*/mi_agent_api")]
    candidates += [os.path.dirname(p)
                   for p in glob.glob("/home/site/wwwroot/mi_agent_api")]
    for root in candidates:
        if os.path.isfile(os.path.join(root, "mi_agent_api", "app.py")):
            sys.path.insert(0, root)
            print("application found at", root, flush=True)
            return
    raise SystemExit(
        "could not find mi_agent_api. Run this from the extracted application "
        "directory, e.g.  cd $(dirname $(ls -d /tmp/*/mi_agent_api | head -1))")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="direct",
                    help="portfolio context: direct | acquired | total")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="",
                    help="comma-separated question ids, e.g. CFO50,Q21A. Lets a "
                         "hypothesis be tested on the questions it is about "
                         "instead of re-running the whole bank.")
    ap.add_argument("--as-of", default=None)
    args = ap.parse_args()

    _locate_app_root()
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    from mi_agent_api import data_source

    client = TestClient(app, raise_server_exceptions=False)

    # What the probe is measuring, recorded alongside the result so a file can
    # never be read against the wrong book.
    context = {
        "scope": args.scope,
        "as_of": args.as_of,
        "data_source_kind": data_source.data_source_kind(),
        "data_source_label": data_source.data_source_label(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rows = []
    asked = QUESTIONS
    if args.only:
        wanted = {t.strip().upper() for t in args.only.split(",") if t.strip()}
        asked = [q for q in asked if q["id"].upper() in wanted]
        missing = wanted - {q["id"].upper() for q in asked}
        if missing:
            raise SystemExit("unknown question id(s): %s" % ", ".join(sorted(missing)))
    if args.limit:
        asked = asked[: args.limit]
    print("probing %d questions | scope=%s | source=%s"
          % (len(asked), args.scope, context["data_source_kind"]), flush=True)

    for i, item in enumerate(asked):
        payload = {"question": item["q"], "sourcePortfolioLens": args.scope}
        if args.as_of:
            payload["asOfDate"] = args.as_of
        t0 = time.time()
        try:
            res = client.post("/mi/query", json=payload)
            env = res.json()
            status = res.status_code
        except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
            rows.append({"id": item["id"], "q": item["q"], "http": 0,
                         "error_class": type(exc).__name__})
            continue

        meta = env.get("metadata") or {}
        gov = env.get("governance") or {}
        spec = env.get("spec") if isinstance(env.get("spec"), dict) else {}
        answered = bool(env.get("ok"))

        # Coverage ledger entries: keep the QUESTION's own words (term) and the
        # field, drop the matched book value.
        coverage = []
        for entry in ((meta.get("semanticCoverage") or {}).get("entries") or []):
            if isinstance(entry, dict):
                coverage.append({"kind": entry.get("kind"),
                                 "field": entry.get("field"),
                                 "term": entry.get("term"),
                                 "disposition": entry.get("disposition")})

        rows.append({
            "id": item["id"],
            "bank": item.get("bank"),
            "q": item["q"],
            "http": status,
            "answered": answered,
            "route": meta.get("route") or "point_in_time_engine",
            "dataset": (env.get("reconciliation") or {}).get("dataset"),
            "outcome_code": ((gov.get("error") or {}).get("code")
                             if gov.get("error") else None),
            "controlled_refusal": bool(env.get("controlledRefusal")),
            "clarification": bool(env.get("clarificationRequested")),
            # Refusal reasons are the payload of this probe. Successful answer
            # prose is NOT recorded: it carries the client's figures.
            "reason": _redact(env.get("error")) if not answered else None,
            "warnings": [_redact(w) for w in (env.get("warnings") or [])][:4],
            # SHAPE only — which fields were bound, never which values.
            "metric": spec.get("metric"),
            "aggregation": spec.get("aggregation"),
            "dimensions": _keys(spec.get("dimensions") or spec.get("dimension")),
            "filter_fields": _keys(spec.get("filters")),
            # WHY the language-understanding step failed, in the arm's own
            # words. `_enforce_model_availability` deliberately publishes a
            # reader-facing sentence with no model name and no arm in it; the
            # cause it withholds is an API exception, not client data, and
            # without it a whole run of refusals names no fault to fix.
            # WHAT THE CALL ACTUALLY COST, when the envelope reports it.
            # Estimating cost from a prompt built against a DIFFERENT book is
            # how a $2 estimate is presented against a $20 bill: the vocabulary
            # is the whole prompt, and it is the part that varies by book.
            "llm": ({k: (meta.get("llm") or {}).get(k)
                     for k in ("calls", "input_tokens", "output_tokens",
                               "cache_read_tokens", "cache_write_tokens",
                               "estimated_total_cost", "cost_estimate_status")}
                    if isinstance(meta.get("llm"), dict) else None),
            "concept_merge": ({
                "status": (meta.get("conceptMerge") or {}).get("status"),
                "detail": _redact((meta.get("conceptMerge") or {}).get("detail")),
            } if isinstance(meta.get("conceptMerge"), dict) else None),
            "scope_applied": _scope_applied(env, meta),
            "coverage": coverage,
            "artefacts": [{"type": a.get("type"), "rows": len(a.get("rows") or [])}
                          for a in (env.get("artifacts") or [])],
            "ms": int((time.time() - t0) * 1000),
        })
        if (i + 1) % 20 == 0:
            print("  %d/%d" % (i + 1, len(asked)), flush=True)

    answered = sum(1 for r in rows if r.get("answered"))

    # DID THE LENS ACTUALLY BIND? The requested scope travels in the body; that
    # is a request, not a result. An unknown field would be dropped silently by
    # the request model and every question would run over the whole book — the
    # precise failure this run exists to avoid. So the file records the evidence
    # and the operator is told before they read anything into the numbers.
    widened = [r["id"] for r in rows
               if (r.get("scope_applied") or {}).get("fell_back_to_total")]
    narrowed = sum(1 for r in rows
                   if "portfolio_id" in (r.get("filter_fields") or [])
                   or (r.get("scope_applied") or {}).get("context_kind")
                   not in (None, "total"))
    context["scope_evidence"] = {
        "requested": args.scope,
        "answers_showing_a_narrowed_scope": narrowed,
        "answers_that_fell_back_to_total": widened,
    }
    if args.scope != "total" and answered and not narrowed:
        # Only meaningful when something answered: a run where every question
        # refused never got far enough to resolve a scope, and warning there
        # would cry wolf on exactly the runs whose refusals matter most.
        print("\n*** WARNING: %d answers, none showing the %s lens applied. "
              "Treat this run as whole-book. ***" % (answered, args.scope))

    out = {"context": context, "answered": answered, "total": len(rows),
           "rows": rows}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print("\nanswered %d / %d   refused %d" % (answered, len(rows), len(rows) - answered))
    # THE BILL, from the run's own usage rather than from a prompt built
    # against another book. Reported whenever the envelope carried it.
    billed = [r["llm"] for r in rows if isinstance(r.get("llm"), dict)
              and r["llm"].get("input_tokens")]
    if billed:
        tin = sum(b.get("input_tokens") or 0 for b in billed)
        tout = sum(b.get("output_tokens") or 0 for b in billed)
        tread = sum(b.get("cache_read_tokens") or 0 for b in billed)
        cost = sum(b.get("estimated_total_cost") or 0.0 for b in billed)
        out["llm_totals"] = {"questions_with_usage": len(billed),
                             "input_tokens": tin, "output_tokens": tout,
                             "cache_read_tokens": tread,
                             "estimated_total_cost": round(cost, 4)}
        print("\nLLM: %d questions | in %d (cache reads %d) | out %d | "
              "est $%.4f  = $%.4f/question"
              % (len(billed), tin, tread, tout, cost, cost / len(billed)))
        if not tread:
            print("     no cache reads — every call paid full price for the "
                  "vocabulary; check the TTL reached this build")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
