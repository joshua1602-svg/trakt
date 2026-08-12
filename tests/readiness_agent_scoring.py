#!/usr/bin/env python3
"""Score an agent run against the hidden answer key.

Reading one good-looking report proves nothing, so this compares observable
output against independently planted truth. It scores six things the brief
asks for, and each is deliberately mechanical:

    discovery         did the agent find each planted finding?
    numerical         does every number it quoted reconcile to Trakt?
    methodology       did it use the OWNED KPI rather than a lookalike?
    false positives   did it invent concerns, or trip the planted traps?
    epistemic         did it respect a refusal instead of reasoning around it?
    efficiency        how many calls, how many repeats

Matching is by evidence phrase, not by semantics. A finding counts as found
when the agent's text contains the planted value (at a stated tolerance) or an
unambiguous marker for it. That is stricter than a human reading for gist and
it will occasionally under-credit a correct answer phrased unusually — which is
the right direction for a scorer whose purpose is to resist wishful reading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from tests.readiness_agent_portfolio import (
    ANSWER_KEY,
    EXPECTED_CURE_RATES,
    EXPECTED_NINETY_PLUS_SHARE,
    EXPECTED_THIRTY_PLUS_SHARE,
    SUPPLIED_LGD,
    answer_key_by_id,
)

FOUND = "FOUND"
PARTIAL = "PARTIALLY_FOUND"
MISSED = "MISSED"


@dataclass
class CaseScore:
    case_id: str
    category: str
    status: str
    note: str = ""


@dataclass
class RunScore:
    run_index: int
    stopped_reason: str
    overall_readiness: Optional[str]
    cases: List[CaseScore] = field(default_factory=list)
    numerical_errors: List[str] = field(default_factory=list)
    methodology_errors: List[str] = field(default_factory=list)
    false_positives: List[str] = field(default_factory=list)
    epistemic_failures: List[str] = field(default_factory=list)
    efficiency: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, int] = field(default_factory=dict)

    @property
    def discovery_rate(self) -> float:
        expected = [c for c in self.cases if c.status != "N/A"]
        if not expected:
            return 0.0
        credit = sum(1.0 if c.status == FOUND else 0.5 if c.status == PARTIAL
                     else 0.0 for c in expected)
        return round(credit / len(expected) * 100, 1)

    def summary(self) -> Dict[str, Any]:
        return {
            "run": self.run_index,
            "stopped_reason": self.stopped_reason,
            "overall_readiness": self.overall_readiness,
            "discovery_rate_pct": self.discovery_rate,
            "found": sum(1 for c in self.cases if c.status == FOUND),
            "partial": sum(1 for c in self.cases if c.status == PARTIAL),
            "missed": sum(1 for c in self.cases if c.status == MISSED),
            "numerical_errors": len(self.numerical_errors),
            "methodology_errors": len(self.methodology_errors),
            "false_positives": len(self.false_positives),
            "epistemic_failures": len(self.epistemic_failures),
            "tool_calls": self.efficiency.get("total_calls"),
            "repeated_calls": self.efficiency.get("repeated_calls"),
        }


def _assessment_text(assessment: Dict[str, Any]) -> str:
    """Everything the agent asserted, as one searchable body."""
    parts: List[str] = [str(assessment.get("summary") or "")]
    parts.extend(str(s) for s in assessment.get("strengths") or [])
    for finding in assessment.get("material_findings") or []:
        parts.extend(str(finding.get(k) or "")
                     for k in ("title", "fact", "rule", "judgement"))
    for gap in assessment.get("could_not_assess") or []:
        parts.extend(str(gap.get(k) or "")
                     for k in ("metric", "status", "implication"))
    parts.extend(str(d) for d in assessment.get("further_diligence") or [])
    return "\n".join(parts)


def _mentions_number(text: str, value: float, *, tolerance: float = 0.05
                     ) -> bool:
    """Is this value quoted anywhere, at any sensible rounding?

    Scans every number in the text rather than formatting the expected value
    one way and hoping the agent chose the same. 12.0 must match "12%",
    "12.0%" and "12.00 %".
    """
    for token in re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")):
        try:
            if abs(float(token) - value) <= tolerance:
                return True
        except ValueError:
            continue
    return False


def _mentions_any(text: str, phrases: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(p.lower() in lowered for p in phrases)


def score_run(record: Dict[str, Any]) -> RunScore:
    assessment = record.get("assessment") or {}
    score = RunScore(
        run_index=record.get("run_index", 0),
        stopped_reason=record.get("stopped_reason", ""),
        overall_readiness=assessment.get("overall_readiness"),
        efficiency=record.get("efficiency") or {},
        usage=record.get("usage") or {})

    if not assessment:
        score.cases = [CaseScore(c["id"], c["category"], MISSED,
                                 "no assessment submitted")
                       for c in ANSWER_KEY if c.get("agent_should_discover")]
        return score

    text = _assessment_text(assessment)
    tools_used = {c["tool"] for c in record.get("transcript") or []}
    key = answer_key_by_id()

    # -- discovery ------------------------------------------------------- #
    for case in ANSWER_KEY:
        if not case.get("agent_should_discover"):
            continue
        status, note = _score_case(case["id"], text, tools_used)
        score.cases.append(CaseScore(case["id"], case["category"], status, note))

    # -- false-positive traps -------------------------------------------- #
    for case in ANSWER_KEY:
        if case["category"] != "false_positive":
            continue
        if case["id"] == "CONC-02" and _mentions_any(
                text, ["south east"]) and _mentions_any(
                    text, ["breach", "exceed", "concern", "issue"]):
            score.false_positives.append(
                "CONC-02: South East (22%) raised as a concern despite being "
                "inside every supplied limit")

    # -- epistemic discipline -------------------------------------------- #
    if _mentions_any(text, ["expected wal", "expected weighted average life"]):
        if not _mentions_any(text, ["model_required", "model required",
                                    "not available", "cannot", "not produced"]):
            score.epistemic_failures.append(
                "quoted an expected WAL without recording that it is "
                "MODEL_REQUIRED")

    # -- methodology: the 2.5E defect, planted ---------------------------- #
    # The supplied LGD is 25.0 and realised severity is 40.0. Quoting 25 AS
    # severity is the failure; quoting it as a supplied estimate is fine.
    for finding in assessment.get("material_findings") or []:
        blob = f"{finding.get('title','')} {finding.get('fact','')}"
        if re.search(r"severity", blob, re.I) and _mentions_number(
                blob, SUPPLIED_LGD, tolerance=0.01) and not _mentions_number(
                    blob, 40.0, tolerance=0.01):
            score.methodology_errors.append(
                f"loss severity quoted as {SUPPLIED_LGD} (the SUPPLIED LGD) "
                "rather than the observed 40.0")

    # -- numerical correctness -------------------------------------------- #
    score.numerical_errors.extend(_check_numbers(assessment))

    return score


def _score_case(case_id: str, text: str, tools_used: set) -> tuple:
    """One planted case against the agent's own words."""
    k = answer_key_by_id()[case_id]

    if case_id == "CONC-01":
        if not _mentions_number(text, 31.0, tolerance=0.2):
            return MISSED, "31% London share not quoted"
        rulebooks = sum(bool(_mentions_any(text, [w])) for w in
                        ("warehouse", "securitisation criteri", "screening"))
        if rulebooks >= 3:
            return FOUND, "quoted and compared against three rulebooks"
        if rulebooks >= 2:
            return PARTIAL, f"quoted; {rulebooks} of 3 rulebooks distinguished"
        return PARTIAL, "quoted but rule sources not distinguished"

    if case_id == "LTV-01":
        return (FOUND, "") if _mentions_number(text, 62.08, tolerance=0.15) \
            else (MISSED, "62.08% WA LTV not quoted")

    if case_id == "LTV-02":
        if not _mentions_number(text, 12.0, tolerance=0.2):
            return MISSED, ">80% LTV share not quoted"
        return FOUND, "high-LTV tail identified"

    if case_id == "VAL-01":
        if not _mentions_any(text, ["stale", "valuation age", "older than"]):
            return MISSED, "stale valuations not raised"
        if _mentions_number(text, 12.0, tolerance=0.2):
            overlap = _mentions_any(text, [
                "coincide", "same cohort", "same loans", "identical",
                "overlap", "both are", "entirely"])
            return (FOUND, "stale pocket quantified and linked to the high-LTV "
                    "cohort") if overlap else (FOUND, "stale pocket quantified")
        return PARTIAL, "raised without the share"

    if case_id == "VINT-01":
        vintage = _mentions_any(text, ["vintage", "cohort", "origination year",
                                       "2024"])
        drilled = "cohort_comparison" in tools_used or "stratify" in tools_used
        if vintage and drilled:
            return FOUND, "cohort identified and drilled into"
        if vintage or drilled:
            return PARTIAL, "cohort touched but not isolated"
        return MISSED, "no cohort analysis"

    if case_id == "ARR-01":
        first, last = EXPECTED_NINETY_PLUS_SHARE[0], EXPECTED_NINETY_PLUS_SHARE[-1]
        if _mentions_number(text, last, tolerance=0.2):
            trend = _mentions_any(text, ["trend", "trajectory", "deteriorat",
                                         "increas", "rose", "risen", "worsen"])
            return (FOUND, "level and trajectory") if trend else (
                PARTIAL, "current level only, no trajectory")
        return MISSED, "90+ DPD level not quoted"

    if case_id == "DEF-01":
        if not _mentions_any(text, ["default rate", "cdr", "defaults"]):
            return MISSED, "default performance not raised"
        rising = _mentions_any(text, ["rising", "increas", "accelerat",
                                      "deteriorat", "worsen"])
        used_owned = "default_analysis" in tools_used
        if rising and used_owned:
            return FOUND, "rising default rate via the owned metric"
        if used_owned:
            return PARTIAL, "owned metric used; trajectory not stated"
        return PARTIAL, "raised without the owned metric"

    if case_id == "CURE-01":
        if "cure_analysis" in tools_used or _mentions_any(text, ["cure"]):
            return FOUND, "cure performance examined"
        return MISSED, "cure never examined"

    if case_id == "CPR-01":
        if not _mentions_any(text, ["cpr", "prepayment"]):
            return MISSED, "prepayment not raised"
        # It must NOT be called adverse without a rule saying so.
        adverse = re.search(r"(cpr|prepayment)[^.]{0,80}(concern|bad|poor|weak)",
                            text, re.I)
        if adverse:
            return PARTIAL, "prepayment labelled adverse without a rule"
        return FOUND, "prepayment reported without unjustified judgement"

    if case_id == "LOSS-01":
        if _mentions_number(text, 40.0, tolerance=0.2) and _mentions_any(
                text, ["severity"]):
            return FOUND, "observed severity 40.0% quoted"
        if _mentions_any(text, ["loss", "severity"]):
            return PARTIAL, "losses raised without the severity figure"
        return MISSED, "losses not raised"

    if case_id == "WAL-01":
        if _mentions_any(text, ["wal", "weighted average life"]):
            return FOUND, "contractual WAL reported"
        return MISSED, "WAL not reported"

    if case_id == "YTM-01":
        if _mentions_any(text, ["ytm", "yield"]):
            honest = _mentions_any(text, [
                "assumption", "floating", "rate path", "not available",
                "cannot", "partial"])
            return (FOUND, "yield limitation respected") if honest else (
                PARTIAL, "yield quoted without its limitation")
        return MISSED, "yield never considered"

    if case_id == "DATA-01":
        if _mentions_any(text, ["complet", "missing field", "data gap",
                                "regulatory", "mandatory"]):
            return FOUND, "data/regulatory gap raised"
        return MISSED, "data readiness not raised"

    return MISSED, "no scorer for this case"


def _check_numbers(assessment: Dict[str, Any]) -> List[str]:
    """Numbers the agent stated that contradict the planted truth.

    Only checks values the key pins exactly. A number this cannot verify is
    not counted as an error — silence here means unchecked, not correct, and
    the report says so.
    """
    errors: List[str] = []
    text = _assessment_text(assessment)

    # If a 30+ or 90+ series is quoted, its endpoints must be the planted ones.
    if _mentions_any(text, ["30+", "30 dpd", "30 days"]):
        if _mentions_number(text, 14.5, tolerance=0.05):
            pass
        elif _mentions_number(text, 8.5, tolerance=0.05):
            errors.append("30+ DPD quoted as 8.5% — that is the delinquency "
                          "band alone and omits defaulted loans, which also "
                          "sit above 30 days")
    if _mentions_any(text, ["90+", "90 dpd", "90 days"]):
        if not _mentions_number(text, 10.0, tolerance=0.3) and \
                _mentions_number(text, 4.0, tolerance=0.05):
            errors.append("90+ DPD quoted as 4.0% — omits defaulted loans")
    return errors


def score_runs(records: Sequence[Dict[str, Any]]) -> List[RunScore]:
    return [score_run(r) for r in records]


def consistency(scores: Sequence[RunScore]) -> Dict[str, Any]:
    """How stable is the agent across repeats? One excellent run is not a
    result; the spread is."""
    if not scores:
        return {}
    rates = [s.discovery_rate for s in scores]
    per_case: Dict[str, List[str]] = {}
    for score in scores:
        for case in score.cases:
            per_case.setdefault(case.case_id, []).append(case.status)
    return {
        "runs": len(scores),
        "discovery_rate_min": min(rates),
        "discovery_rate_max": max(rates),
        "discovery_rate_mean": round(sum(rates) / len(rates), 1),
        "always_found": sorted(k for k, v in per_case.items()
                               if all(s == FOUND for s in v)),
        "never_found": sorted(k for k, v in per_case.items()
                              if all(s == MISSED for s in v)),
        "inconsistent": sorted(k for k, v in per_case.items()
                               if len(set(v)) > 1),
        "false_positives_total": sum(len(s.false_positives) for s in scores),
        "epistemic_failures_total": sum(len(s.epistemic_failures)
                                        for s in scores),
        "methodology_errors_total": sum(len(s.methodology_errors)
                                        for s in scores),
    }
