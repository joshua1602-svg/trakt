#!/usr/bin/env python3
"""live_pipeline_probe — the stage-movement bank against the LIVE pipeline.

The companion to `live_bank_probe.py`, and it grades itself, which that one
cannot. Every question in `STAGE_MOVEMENT_BANK.yaml` and its near-neighbour set
carries an `expect` written from the QUESTION and the GOVERNED DATA before any
answer existed: DELIVER where a governed capability can produce the analysis,
REFUSE where the capability set genuinely cannot and saying so is the only safe
outcome. 82 of the 116 expect a refusal.

That inversion is the point. On this bank a refusal is usually the CORRECT
outcome, so "answered 30 / 116" says nothing on its own — the measure is
agreement with `expect`, and the two failure directions are not equally bad:

  MISSED    expected DELIVER, refused        a capability gap or a defect
  OVERREACH expected REFUSE, answered        the serious one: an answer the
                                             governed layer cannot support

An OVERREACH on a stage-movement question means the agent produced a transition
figure from per-stage stock, which is exactly the fabrication the bank was
written to detect.

Emits the same diagnostics as the funded probe — route, outcome code, redacted
reason, bound shape — and no figures.

USAGE
-----
    python live_pipeline_probe.py --out /home/pipeline_result.json
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
  "id": "SM01",
  "bank": "STAGE88",
  "q": "How many cases moved from KFI to Application?",
  "family": "A_source_dest_count",
  "expect": "REFUSE"
 },
 {
  "id": "SM02",
  "bank": "STAGE88",
  "q": "How many KFI cases progressed to Application?",
  "family": "A_source_dest_count",
  "expect": "REFUSE"
 },
 {
  "id": "SM03",
  "bank": "STAGE88",
  "q": "How many cases went from Application to Offer?",
  "family": "A_source_dest_count",
  "expect": "REFUSE"
 },
 {
  "id": "SM04",
  "bank": "STAGE88",
  "q": "How many applications became offers?",
  "family": "A_source_dest_count",
  "expect": "REFUSE"
 },
 {
  "id": "SM05",
  "bank": "STAGE88",
  "q": "How many offers moved to Completion?",
  "family": "A_source_dest_count",
  "expect": "REFUSE"
 },
 {
  "id": "SM06",
  "bank": "STAGE88",
  "q": "How many cases progressed from Offer to Completion?",
  "family": "A_source_dest_count",
  "expect": "REFUSE"
 },
 {
  "id": "SM07",
  "bank": "STAGE88",
  "q": "What number of cases moved KFI to Application?",
  "family": "A_source_dest_count",
  "expect": "REFUSE"
 },
 {
  "id": "SM08",
  "bank": "STAGE88",
  "q": "Show case movement from Application into Offer.",
  "family": "A_source_dest_count",
  "expect": "REFUSE"
 },
 {
  "id": "SM09",
  "bank": "STAGE88",
  "q": "Cases progressing from KFI to Application this period?",
  "family": "A_source_dest_count",
  "expect": "REFUSE"
 },
 {
  "id": "SM10",
  "bank": "STAGE88",
  "q": "How many pipeline cases advanced from Offer to Completion?",
  "family": "A_source_dest_count",
  "expect": "REFUSE"
 },
 {
  "id": "SM11",
  "bank": "STAGE88",
  "q": "How many deals transitioned out of KFI into Application?",
  "family": "A_source_dest_count",
  "expect": "REFUSE"
 },
 {
  "id": "SM12",
  "bank": "STAGE88",
  "q": "How many cases flowed from Application to Offer between the last two extracts?",
  "family": "A_source_dest_count",
  "expect": "REFUSE"
 },
 {
  "id": "SM13",
  "bank": "STAGE88",
  "q": "How much balance moved from KFI to Application?",
  "family": "B_source_dest_amount",
  "expect": "REFUSE"
 },
 {
  "id": "SM14",
  "bank": "STAGE88",
  "q": "What value progressed from Application to Offer?",
  "family": "B_source_dest_amount",
  "expect": "REFUSE"
 },
 {
  "id": "SM15",
  "bank": "STAGE88",
  "q": "How much pipeline moved from Offer to Completion?",
  "family": "B_source_dest_amount",
  "expect": "REFUSE"
 },
 {
  "id": "SM16",
  "bank": "STAGE88",
  "q": "What was the balance of cases moving from Application into Offer?",
  "family": "B_source_dest_amount",
  "expect": "REFUSE"
 },
 {
  "id": "SM17",
  "bank": "STAGE88",
  "q": "How much exposure transitioned from KFI to Application?",
  "family": "B_source_dest_amount",
  "expect": "REFUSE"
 },
 {
  "id": "SM18",
  "bank": "STAGE88",
  "q": "What amount moved from Offer to Completion?",
  "family": "B_source_dest_amount",
  "expect": "REFUSE"
 },
 {
  "id": "SM19",
  "bank": "STAGE88",
  "q": "Show balance transferred from Application to Offer.",
  "family": "B_source_dest_amount",
  "expect": "REFUSE"
 },
 {
  "id": "SM20",
  "bank": "STAGE88",
  "q": "What exposure went from Offer to Completion this period?",
  "family": "B_source_dest_amount",
  "expect": "REFUSE"
 },
 {
  "id": "SM21",
  "bank": "STAGE88",
  "q": "Where did Offer-stage departures go?",
  "family": "C_departures",
  "expect": "REFUSE"
 },
 {
  "id": "SM22",
  "bank": "STAGE88",
  "q": "Where did cases leaving Application move to?",
  "family": "C_departures",
  "expect": "REFUSE"
 },
 {
  "id": "SM23",
  "bank": "STAGE88",
  "q": "What happened to cases that left KFI?",
  "family": "C_departures",
  "expect": "REFUSE"
 },
 {
  "id": "SM24",
  "bank": "STAGE88",
  "q": "Break down departures from Offer by destination.",
  "family": "C_departures",
  "expect": "REFUSE"
 },
 {
  "id": "SM25",
  "bank": "STAGE88",
  "q": "How many cases left Application and where did they go?",
  "family": "C_departures",
  "expect": "REFUSE"
 },
 {
  "id": "SM26",
  "bank": "STAGE88",
  "q": "What balance exited KFI by destination?",
  "family": "C_departures",
  "expect": "REFUSE"
 },
 {
  "id": "SM27",
  "bank": "STAGE88",
  "q": "Show destination mix for cases departing Offer.",
  "family": "C_departures",
  "expect": "REFUSE"
 },
 {
  "id": "SM28",
  "bank": "STAGE88",
  "q": "How many cases arrived in Application?",
  "family": "D_arrivals",
  "expect": "REFUSE"
 },
 {
  "id": "SM29",
  "bank": "STAGE88",
  "q": "Where did new Offer cases come from?",
  "family": "D_arrivals",
  "expect": "REFUSE"
 },
 {
  "id": "SM30",
  "bank": "STAGE88",
  "q": "What balance moved into Offer?",
  "family": "D_arrivals",
  "expect": "REFUSE"
 },
 {
  "id": "SM31",
  "bank": "STAGE88",
  "q": "Show arrivals into Completion by prior stage.",
  "family": "D_arrivals",
  "expect": "REFUSE"
 },
 {
  "id": "SM32",
  "bank": "STAGE88",
  "q": "How much entered Application during the period?",
  "family": "D_arrivals",
  "expect": "REFUSE"
 },
 {
  "id": "SM33",
  "bank": "STAGE88",
  "q": "Which stages contributed cases into Offer?",
  "family": "D_arrivals",
  "expect": "REFUSE"
 },
 {
  "id": "SM34",
  "bank": "STAGE88",
  "q": "How many cases stayed in Application?",
  "family": "E_stayers",
  "expect": "REFUSE"
 },
 {
  "id": "SM35",
  "bank": "STAGE88",
  "q": "How much balance remained in Offer?",
  "family": "E_stayers",
  "expect": "REFUSE"
 },
 {
  "id": "SM36",
  "bank": "STAGE88",
  "q": "What happened to cases that stayed at KFI?",
  "family": "E_stayers",
  "expect": "REFUSE"
 },
 {
  "id": "SM37",
  "bank": "STAGE88",
  "q": "Show persisting Application cases.",
  "family": "E_stayers",
  "expect": "REFUSE"
 },
 {
  "id": "SM38",
  "bank": "STAGE88",
  "q": "How many Offer cases remained at Offer?",
  "family": "E_stayers",
  "expect": "REFUSE"
 },
 {
  "id": "SM39",
  "bank": "STAGE88",
  "q": "What was the balance of cases staying in Application?",
  "family": "E_stayers",
  "expect": "REFUSE"
 },
 {
  "id": "SM40",
  "bank": "STAGE88",
  "q": "What was the amount amendment on cases that stayed in Application?",
  "family": "F_amendments",
  "expect": "REFUSE"
 },
 {
  "id": "SM41",
  "bank": "STAGE88",
  "q": "How much did persisting Offer cases change in value?",
  "family": "F_amendments",
  "expect": "REFUSE"
 },
 {
  "id": "SM42",
  "bank": "STAGE88",
  "q": "What balance change occurred on cases remaining at KFI?",
  "family": "F_amendments",
  "expect": "REFUSE"
 },
 {
  "id": "SM43",
  "bank": "STAGE88",
  "q": "Did Application-stage cases increase or decrease in amount?",
  "family": "F_amendments",
  "expect": "REFUSE"
 },
 {
  "id": "SM44",
  "bank": "STAGE88",
  "q": "Show amount movement on cases that stayed in Offer.",
  "family": "F_amendments",
  "expect": "REFUSE"
 },
 {
  "id": "SM45",
  "bank": "STAGE88",
  "q": "What was the net balance amendment for KFI stayers?",
  "family": "F_amendments",
  "expect": "REFUSE"
 },
 {
  "id": "SM46",
  "bank": "STAGE88",
  "q": "How many cases completed?",
  "family": "G_completions",
  "expect": "REFUSE"
 },
 {
  "id": "SM47",
  "bank": "STAGE88",
  "q": "How much balance completed?",
  "family": "G_completions",
  "expect": "REFUSE"
 },
 {
  "id": "SM48",
  "bank": "STAGE88",
  "q": "How many pipeline cases are at the Completion stage?",
  "family": "G_completions",
  "expect": "DELIVER"
 },
 {
  "id": "SM49",
  "bank": "STAGE88",
  "q": "What is the balance at the Completion stage of the pipeline?",
  "family": "G_completions",
  "expect": "DELIVER"
 },
 {
  "id": "SM50",
  "bank": "STAGE88",
  "q": "How many pipeline cases reached Completion?",
  "family": "G_completions",
  "expect": "REFUSE"
 },
 {
  "id": "SM51",
  "bank": "STAGE88",
  "q": "What value completed this period?",
  "family": "G_completions",
  "expect": "REFUSE"
 },
 {
  "id": "SM52",
  "bank": "STAGE88",
  "q": "What was completion flow by count?",
  "family": "G_completions",
  "expect": "REFUSE"
 },
 {
  "id": "SM53",
  "bank": "STAGE88",
  "q": "What was completion flow by balance?",
  "family": "G_completions",
  "expect": "REFUSE"
 },
 {
  "id": "SM54",
  "bank": "STAGE88",
  "q": "How many Offer cases completed?",
  "family": "G_completions",
  "expect": "REFUSE"
 },
 {
  "id": "SM55",
  "bank": "STAGE88",
  "q": "How many pipeline cases are withdrawn?",
  "family": "H_terminal",
  "expect": "DELIVER"
 },
 {
  "id": "SM56",
  "bank": "STAGE88",
  "q": "What is the withdrawn balance in the pipeline?",
  "family": "H_terminal",
  "expect": "DELIVER"
 },
 {
  "id": "SM57",
  "bank": "STAGE88",
  "q": "How many cases were withdrawn?",
  "family": "H_terminal",
  "expect": "REFUSE"
 },
 {
  "id": "SM58",
  "bank": "STAGE88",
  "q": "How much pipeline was withdrawn?",
  "family": "H_terminal",
  "expect": "REFUSE"
 },
 {
  "id": "SM59",
  "bank": "STAGE88",
  "q": "Where did pipeline drop out?",
  "family": "H_terminal",
  "expect": "REFUSE"
 },
 {
  "id": "SM60",
  "bank": "STAGE88",
  "q": "What stage had the most withdrawals?",
  "family": "H_terminal",
  "expect": "REFUSE"
 },
 {
  "id": "SM61",
  "bank": "STAGE88",
  "q": "How many Offer cases were withdrawn?",
  "family": "H_terminal",
  "expect": "REFUSE"
 },
 {
  "id": "SM62",
  "bank": "STAGE88",
  "q": "What balance left the pipeline without completing?",
  "family": "H_terminal",
  "expect": "REFUSE"
 },
 {
  "id": "SM63",
  "bank": "STAGE88",
  "q": "Where was the greatest pipeline attrition?",
  "family": "H_terminal",
  "expect": "REFUSE"
 },
 {
  "id": "SM64",
  "bank": "STAGE88",
  "q": "Reconcile Application stage this period.",
  "family": "I_reconciliation",
  "expect": "REFUSE"
 },
 {
  "id": "SM65",
  "bank": "STAGE88",
  "q": "Explain the movement in Offer-stage cases.",
  "family": "I_reconciliation",
  "expect": "REFUSE"
 },
 {
  "id": "SM66",
  "bank": "STAGE88",
  "q": "Why did KFI cases change from opening to closing?",
  "family": "I_reconciliation",
  "expect": "REFUSE"
 },
 {
  "id": "SM67",
  "bank": "STAGE88",
  "q": "Show opening, arrivals, departures and closing for Application.",
  "family": "I_reconciliation",
  "expect": "REFUSE"
 },
 {
  "id": "SM68",
  "bank": "STAGE88",
  "q": "Reconcile Offer balance between the two extracts.",
  "family": "I_reconciliation",
  "expect": "REFUSE"
 },
 {
  "id": "SM69",
  "bank": "STAGE88",
  "q": "What drove the change in Application-stage balance?",
  "family": "I_reconciliation",
  "expect": "REFUSE"
 },
 {
  "id": "SM70",
  "bank": "STAGE88",
  "q": "Reconcile Application cases from opening to closing.",
  "family": "I_reconciliation",
  "expect": "REFUSE"
 },
 {
  "id": "SM71",
  "bank": "STAGE88",
  "q": "Which stage had the most movement?",
  "family": "J_largest",
  "expect": "REFUSE"
 },
 {
  "id": "SM72",
  "bank": "STAGE88",
  "q": "What was the largest stage transition?",
  "family": "J_largest",
  "expect": "REFUSE"
 },
 {
  "id": "SM73",
  "bank": "STAGE88",
  "q": "Where did the most cases progress?",
  "family": "J_largest",
  "expect": "REFUSE"
 },
 {
  "id": "SM74",
  "bank": "STAGE88",
  "q": "Which transition moved the most balance?",
  "family": "J_largest",
  "expect": "REFUSE"
 },
 {
  "id": "SM75",
  "bank": "STAGE88",
  "q": "Where was pipeline attrition greatest?",
  "family": "J_largest",
  "expect": "REFUSE"
 },
 {
  "id": "SM76",
  "bank": "STAGE88",
  "q": "What was the biggest departure destination?",
  "family": "J_largest",
  "expect": "REFUSE"
 },
 {
  "id": "SM77",
  "bank": "STAGE88",
  "q": "Compare stage movement with the prior period.",
  "family": "K_period_comparison",
  "expect": "REFUSE"
 },
 {
  "id": "SM78",
  "bank": "STAGE88",
  "q": "Did more cases move from Application to Offer this period?",
  "family": "K_period_comparison",
  "expect": "REFUSE"
 },
 {
  "id": "SM79",
  "bank": "STAGE88",
  "q": "How has KFI-to-Application movement changed?",
  "family": "K_period_comparison",
  "expect": "REFUSE"
 },
 {
  "id": "SM80",
  "bank": "STAGE88",
  "q": "Was completion flow higher than last period?",
  "family": "K_period_comparison",
  "expect": "REFUSE"
 },
 {
  "id": "SM81",
  "bank": "STAGE88",
  "q": "Compare Offer departures with the previous reporting period.",
  "family": "K_period_comparison",
  "expect": "REFUSE"
 },
 {
  "id": "SM82",
  "bank": "STAGE88",
  "q": "Has pipeline progression improved month on month?",
  "family": "K_period_comparison",
  "expect": "REFUSE"
 },
 {
  "id": "SM83",
  "bank": "STAGE88",
  "q": "Summarise pipeline stage movement this period.",
  "family": "L_summary",
  "expect": "REFUSE"
 },
 {
  "id": "SM84",
  "bank": "STAGE88",
  "q": "Give me the stage movement summary.",
  "family": "L_summary",
  "expect": "REFUSE"
 },
 {
  "id": "SM85",
  "bank": "STAGE88",
  "q": "What changed in pipeline stages?",
  "family": "L_summary",
  "expect": "REFUSE"
 },
 {
  "id": "SM86",
  "bank": "STAGE88",
  "q": "What happened in the pipeline between the last two extracts?",
  "family": "L_summary",
  "expect": "REFUSE"
 },
 {
  "id": "SM87",
  "bank": "STAGE88",
  "q": "Show pipeline progression.",
  "family": "L_summary",
  "expect": "DELIVER"
 },
 {
  "id": "SM88",
  "bank": "STAGE88",
  "q": "How did cases move through the funnel?",
  "family": "L_summary",
  "expect": "DELIVER"
 },
 {
  "id": "NN01",
  "bank": "NEAR28",
  "q": "How has pipeline balance moved this month?",
  "family": "N_pipeline_evolution",
  "expect": "DELIVER"
 },
 {
  "id": "NN02",
  "bank": "NEAR28",
  "q": "Show pipeline evolution.",
  "family": "N_pipeline_evolution",
  "expect": "DELIVER"
 },
 {
  "id": "NN03",
  "bank": "NEAR28",
  "q": "What is pipeline by stage?",
  "family": "N_stage_stock",
  "expect": "DELIVER"
 },
 {
  "id": "NN04",
  "bank": "NEAR28",
  "q": "Show pipeline amount by stage.",
  "family": "N_stage_stock",
  "expect": "DELIVER"
 },
 {
  "id": "NN05",
  "bank": "NEAR28",
  "q": "How much pipeline is currently in Offer?",
  "family": "N_stage_stock",
  "expect": "DELIVER"
 },
 {
  "id": "NN06",
  "bank": "NEAR28",
  "q": "What is the current pipeline amount?",
  "family": "N_stage_stock",
  "expect": "DELIVER"
 },
 {
  "id": "NN07",
  "bank": "NEAR28",
  "q": "How much balance is in Application?",
  "family": "N_stage_stock",
  "expect": "DELIVER"
 },
 {
  "id": "NN08",
  "bank": "NEAR28",
  "q": "What is the Offer-stage balance?",
  "family": "N_stage_stock",
  "expect": "DELIVER"
 },
 {
  "id": "NN09",
  "bank": "NEAR28",
  "q": "What percentage of pipeline is in KFI?",
  "family": "N_stage_stock",
  "expect": "DELIVER"
 },
 {
  "id": "NN10",
  "bank": "NEAR28",
  "q": "Show weekly pipeline cases.",
  "family": "N_pipeline_evolution",
  "expect": "DELIVER"
 },
 {
  "id": "NN11",
  "bank": "NEAR28",
  "q": "How has the pipeline grown?",
  "family": "N_pipeline_evolution",
  "expect": "DELIVER"
 },
 {
  "id": "NN12",
  "bank": "NEAR28",
  "q": "Show pipeline case count over time.",
  "family": "N_pipeline_evolution",
  "expect": "DELIVER"
 },
 {
  "id": "NN13",
  "bank": "NEAR28",
  "q": "What is the conversion rate?",
  "family": "N_conversion",
  "expect": "DELIVER"
 },
 {
  "id": "NN14",
  "bank": "NEAR28",
  "q": "How has conversion changed?",
  "family": "N_conversion",
  "expect": "DELIVER"
 },
 {
  "id": "NN15",
  "bank": "NEAR28",
  "q": "What is the KFI to completion conversion rate?",
  "family": "N_conversion",
  "expect": "DELIVER"
 },
 {
  "id": "NN16",
  "bank": "NEAR28",
  "q": "What is funded balance movement?",
  "family": "N_funded_movement",
  "expect": "DELIVER"
 },
 {
  "id": "NN17",
  "bank": "NEAR28",
  "q": "Why did funded balance increase?",
  "family": "N_funded_movement",
  "expect": "DELIVER"
 },
 {
  "id": "NN18",
  "bank": "NEAR28",
  "q": "Show movement by region.",
  "family": "N_funded_movement",
  "expect": "DELIVER"
 },
 {
  "id": "NN19",
  "bank": "NEAR28",
  "q": "Show balance movement by portfolio.",
  "family": "N_funded_movement",
  "expect": "DELIVER"
 },
 {
  "id": "NN20",
  "bank": "NEAR28",
  "q": "How many funded loans were added?",
  "family": "N_funded_movement",
  "expect": "DELIVER"
 },
 {
  "id": "NN21",
  "bank": "NEAR28",
  "q": "How did the book move last month?",
  "family": "N_funded_movement",
  "expect": "DELIVER"
 },
 {
  "id": "NN22",
  "bank": "NEAR28",
  "q": "What is movement in LTV?",
  "family": "N_other_movement",
  "expect": "DELIVER"
 },
 {
  "id": "NN23",
  "bank": "NEAR28",
  "q": "How has regional concentration moved?",
  "family": "N_other_movement",
  "expect": "DELIVER"
 },
 {
  "id": "NN24",
  "bank": "NEAR28",
  "q": "How has average LTV changed since last month?",
  "family": "N_other_movement",
  "expect": "DELIVER"
 },
 {
  "id": "NN25",
  "bank": "NEAR28",
  "q": "Show balance by region.",
  "family": "N_other_movement",
  "expect": "DELIVER"
 },
 {
  "id": "NN26",
  "bank": "NEAR28",
  "q": "What is the forecast funded balance?",
  "family": "N_forecast",
  "expect": "DELIVER"
 },
 {
  "id": "NN27",
  "bank": "NEAR28",
  "q": "How much of the pipeline is expected to complete?",
  "family": "N_forecast",
  "expect": "DELIVER"
 },
 {
  "id": "NN28",
  "bank": "NEAR28",
  "q": "When will we reach 700 loans?",
  "family": "N_forecast",
  "expect": "DELIVER"
 }
]



QUESTIONS += [
 {
  "id": "SHIP-SM01-1",
  "bank": "SHIPPED36",
  "q": "How many cases moved from KFI to Application?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM01-2",
  "bank": "SHIPPED36",
  "q": "How many KFI cases progressed to Application?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM01-3",
  "bank": "SHIPPED36",
  "q": "How many cases went from KFI into Application?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM01-4",
  "bank": "SHIPPED36",
  "q": "What number of cases transitioned KFI to Application?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM02-1",
  "bank": "SHIPPED36",
  "q": "How much balance moved from Application to Offer?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM02-2",
  "bank": "SHIPPED36",
  "q": "What value progressed from Application to Offer?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM02-3",
  "bank": "SHIPPED36",
  "q": "How much pipeline moved from Application into Offer?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM02-4",
  "bank": "SHIPPED36",
  "q": "What amount transitioned from Application to Offer?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM03-1",
  "bank": "SHIPPED36",
  "q": "How many cases moved from Offer to Completion?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM03-2",
  "bank": "SHIPPED36",
  "q": "How many cases progressed from Offer into Completion?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM03-3",
  "bank": "SHIPPED36",
  "q": "What number of offers reached Completion?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM03-4",
  "bank": "SHIPPED36",
  "q": "How many cases advanced from Offer to Completed?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM04-1",
  "bank": "SHIPPED36",
  "q": "How much balance moved from Offer to Completion?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM04-2",
  "bank": "SHIPPED36",
  "q": "How much Offer-stage pipeline reached Completion?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM04-3",
  "bank": "SHIPPED36",
  "q": "What amount progressed from Offer to Completion?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM04-4",
  "bank": "SHIPPED36",
  "q": "What value went from Offer into Completed?",
  "family": "transition",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM05-1",
  "bank": "SHIPPED36",
  "q": "How many new cases entered KFI?",
  "family": "new_arrival",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM05-2",
  "bank": "SHIPPED36",
  "q": "How many new pipeline cases arrived in KFI?",
  "family": "new_arrival",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM05-3",
  "bank": "SHIPPED36",
  "q": "What number of cases were new arrivals into KFI?",
  "family": "new_arrival",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM05-4",
  "bank": "SHIPPED36",
  "q": "How many cases newly entered the KFI stage?",
  "family": "new_arrival",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM06-1",
  "bank": "SHIPPED36",
  "q": "How many cases stayed in Application?",
  "family": "stayer",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM06-2",
  "bank": "SHIPPED36",
  "q": "How many Application cases remained in Application?",
  "family": "stayer",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM06-3",
  "bank": "SHIPPED36",
  "q": "What number of cases stayed at Application stage?",
  "family": "stayer",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM06-4",
  "bank": "SHIPPED36",
  "q": "How many cases persisted in Application?",
  "family": "stayer",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM07-1",
  "bank": "SHIPPED36",
  "q": "What was the amount change on cases that stayed in Application?",
  "family": "stayer_amount_change",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM07-2",
  "bank": "SHIPPED36",
  "q": "How much did Application stayers change in value?",
  "family": "stayer_amount_change",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM07-3",
  "bank": "SHIPPED36",
  "q": "What was the balance amendment on cases remaining in Application?",
  "family": "stayer_amount_change",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM07-4",
  "bank": "SHIPPED36",
  "q": "How did the value of cases persisting in Application change?",
  "family": "stayer_amount_change",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM08-1",
  "bank": "SHIPPED36",
  "q": "Where did cases leaving Offer go?",
  "family": "departure",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM08-2",
  "bank": "SHIPPED36",
  "q": "Break down departures from Offer by destination.",
  "family": "departure",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM08-3",
  "bank": "SHIPPED36",
  "q": "What happened to cases that left Offer?",
  "family": "departure",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM08-4",
  "bank": "SHIPPED36",
  "q": "Show the destinations of Offer-stage departures.",
  "family": "departure",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM09-1",
  "bank": "SHIPPED36",
  "q": "Reconcile Application stage this period.",
  "family": "reconciliation",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM09-2",
  "bank": "SHIPPED36",
  "q": "Show opening, arrivals, departures and closing for Application.",
  "family": "reconciliation",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM09-3",
  "bank": "SHIPPED36",
  "q": "Reconcile the Application stage from opening to closing.",
  "family": "reconciliation",
  "expect": "DELIVER"
 },
 {
  "id": "SHIP-SM09-4",
  "bank": "SHIPPED36",
  "q": "Give me the Application stage reconciliation.",
  "family": "reconciliation",
  "expect": "DELIVER"
 }
]


#: THE AUDIT BANK'S `expect` VALUES ARE STALE, AND SILENTLY SO.
#:
#: `migration_phase0/STAGE_MOVEMENT_BANK.yaml` was frozen during an AUDIT whose
#: brief said "do not implement the capability in this task". Its 82 REFUSEs
#: were correct THEN: the governed layer held per-stage stock and no transition
#: could be derived from it. The capability shipped afterwards
#: (`mi_agent_api/stage_movement_query.py`, in the deployed build) and answers
#: exactly the families those REFUSEs were written against — so grading today's
#: answers against yesterday's expectations scores the feature WORKING as
#: OVERREACH, which is the most misleading verdict this probe could produce.
#:
#: The families the shipped capability declares, from the nine subtypes in
#: `tests/fixtures/mi_query_stage_movement/STAGE_MOVEMENT_BANK.yaml`:
#: transition (count and amount), new_arrival, stayer, stayer_amount_change,
#: departure, reconciliation. The audit families below map onto those.
#:
#: THIS MAPPING IS AN INFERENCE, not a measurement, and it is stated here so it
#: can be corrected rather than believed. The families left out — J_largest,
#: K_period_comparison, L_summary, G_completions, H_terminal — are NOT claimed
#: either way: their rows are graded UNSCORED, because a guess dressed as a
#: verdict is worse than an honest gap.
_SHIPPED_FAMILIES = {
    "A_source_dest_count", "B_source_dest_amount", "C_departures",
    "D_arrivals", "E_stayers", "F_amendments", "I_reconciliation",
}
_STALE_REFUSE_FAMILIES = _SHIPPED_FAMILIES


def _expected(item):
    """What this question SHOULD do against the build being probed."""
    if item.get("bank") == "SHIPPED36":
        return "DELIVER"
    if item.get("bank") == "NEAR28":
        return "DELIVER"
    fam = item.get("family")
    if fam in _STALE_REFUSE_FAMILIES:
        return "DELIVER"          # the capability now covers this family
    if item.get("expect") == "DELIVER":
        return "DELIVER"
    return "UNSCORED"             # frozen as REFUSE before the capability; unproven now


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
            "concept_merge": ({
                "status": (meta.get("conceptMerge") or {}).get("status"),
                "detail": _redact((meta.get("conceptMerge") or {}).get("detail")),
            } if isinstance(meta.get("conceptMerge"), dict) else None),
            "scope_applied": _scope_applied(env, meta),
            "coverage": coverage,
            "artefacts": [{"type": a.get("type"), "rows": len(a.get("rows") or [])}
                          for a in (env.get("artifacts") or [])],
            "ms": int((time.time() - t0) * 1000),
            "family": item.get("family"),
            "expect_frozen": item.get("expect"),
            "expect": _expected(item),
            # AGREEMENT, not success. A refusal where the bank expects REFUSE is
            # the right answer and is scored as one.
            "verdict": ("UNSCORED" if _expected(item) == "UNSCORED"
                        else "PASS" if (answered == (_expected(item) == "DELIVER"))
                        else ("OVERREACH" if answered else "MISSED")),
        })
        if (i + 1) % 20 == 0:
            print("  %d/%d" % (i + 1, len(asked)), flush=True)

    answered = sum(1 for r in rows if r.get("answered"))
    verdicts = {}
    for r in rows:
        verdicts[r.get("verdict")] = verdicts.get(r.get("verdict"), 0) + 1

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
           # AGREEMENT is the headline on this bank, not the answered count:
           # 82 of the 116 questions expect a refusal, so a high refusal rate
           # is the target rather than the problem.
           "verdicts": verdicts,
           "overreach": [r["id"] for r in rows if r.get("verdict") == "OVERREACH"],
           "missed": [r["id"] for r in rows if r.get("verdict") == "MISSED"],
           "rows": rows}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print("\nanswered %d / %d   refused %d" % (answered, len(rows), len(rows) - answered))
    print("agreement with the bank: PASS %d  MISSED %d  OVERREACH %d"
          % (verdicts.get("PASS", 0), verdicts.get("MISSED", 0),
             verdicts.get("OVERREACH", 0)))
    print("  (UNSCORED %d — frozen as REFUSE before the capability shipped; "
          "not graded either way)" % verdicts.get("UNSCORED", 0))
    if out["overreach"]:
        print("OVERREACH (answered where the bank expects a refusal): %s"
              % ", ".join(out["overreach"]))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
