"""mi_agent_api/concentration_query.py

Governed MI Query capabilities over the APPROVED concentration tests.

The chat route (and therefore Copilot, which delegates to the same governed
service) answers limit questions from ``compute_concentration_tests`` — the
single evaluation service — through a deterministic intent layer:

    list_concentration_tests        "show regional concentration tests"
    get_concentration_test          "what is the current London plus South East
                                     concentration?" / "headroom on average
                                     initial principal balance"
    get_concentration_summary       "are we breaching any concentration limits?"
    compare_concentration_periods   "what changed in risk limits this month?" /
                                    "which tests moved from pass to warning?"
    list_concentration_breaches     "which limits are we breaching?"
    list_concentration_warnings     "which tests are closest to breach?"
    explain_concentration_test      "what definition are we using for Net WAC?"
                                    / "show the source and approval history"
    get_concentration_test_drillthrough
                                    "which loans contribute to the East Anglia
                                     test?"
    list_concentration_unavailable  "which limits cannot currently be
                                     calculated?"

Everything answers from the evaluated envelope — this module never touches raw
rows, never invents a threshold, never infers an unapproved definition, and
reports unavailable values as unavailable, not zero.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_CATEGORY_WORDS = {
    "geography": ("region", "regional", "geographic", "geography", "postcode",
                  "county", "country"),
    "property_value": ("property value", "valuation"),
    "loan_balance": ("loan size", "balance", "principal", "large loan",
                     "ticket"),
    "borrower": ("borrower", "joint", "age"),
    "rate_product": ("rate", "coupon", "wac", "product"),
    "ltv": ("ltv", "loan to value", "loan-to-value"),
    "performance": ("arrears", "default", "performance", "delinquen"),
    "composition": ("originator", "seller", "servicer", "vintage", "tenure",
                    "purpose"),
    "external_index": ("hpi", "house price", "index"),
}

_STATUS_LABEL = {
    "pass": "PASS", "warning": "WARNING", "breach": "BREACH",
    "unavailable": "UNAVAILABLE", "insufficient_data": "INSUFFICIENT DATA",
    "pending_effective_date": "PENDING EFFECTIVE DATE", "expired": "EXPIRED",
    "indicative_only": "INDICATIVE ONLY",
}


def _fmt(value: Optional[float], unit: str) -> str:
    if value is None:
        return "unavailable"
    if unit == "percent":
        return f"{value:.2f}%"
    if unit == "count":
        return f"{value:.0f}"
    return f"{value:,.0f}"


def _headroom_phrase(t: Dict[str, Any]) -> str:
    h = t.get("headroom")
    if h is None:
        return "headroom unavailable"
    unit = "pp" if t.get("unit") == "percent" else t.get("unit") or ""
    return f"headroom {h:,.2f} {unit}".strip()


def _test_line(t: Dict[str, Any]) -> str:
    limit = _fmt(t.get("threshold"), t.get("unit") or "percent")
    op = "≤" if t.get("operator") == "max" else "≥"
    return (f"{t['displayName']}: {_fmt(t.get('currentValue'), t.get('unit') or 'percent')} "
            f"(limit {op} {limit}) — {_STATUS_LABEL.get(t['status'], t['status'])}")


def detect_intent(question: str) -> str:
    q = " ".join(str(question).lower().split())
    if re.search(r"\bwhich pipeline (loans|cases)\b|\bpipeline (loans|cases)? ?driv"
                 r"|\bdrives? the\b.*\b(increase|rise|movement)\b", q):
        return "get_concentration_pipeline_drivers"
    if re.search(r"\bhow (was|is) the (expected )?forecast\b|\bforecast "
                 r"(methodolog|calculat|assumption)|\bconversion assumptions\b"
                 r"|\bfive.week\b|\bhow reliable\b|\bforecast confidence\b"
                 r"|\bmethodology\b", q):
        return "get_forecast_methodology"
    if re.search(r"\bif all (the )?pipeline\b|\ball pipeline (loans? )?"
                 r"(complete|convert)|\bmaximum.exposure\b|\bfull.pipeline\b"
                 r"|\bstress (scenario|state|case)\b|\bonly breach in\b", q):
        return "list_full_pipeline_concentration_breaches"
    if re.search(r"\bwhy is\b.*\bexpected\b|\bexpected concentration change\b"
                 r"|\bwhy .*\bexpected (to )?(breach|rise|increase|worsen)\b", q):
        return "explain_expected_concentration_change"
    if re.search(r"\blikely to breach\b|\bexpected to breach\b|\bexpected "
                 r"breach|\bgoing to breach\b|\bwill (we )?breach\b"
                 r"|\bemerging risk|\bheading (for|towards)\b", q):
        return "identify_emerging_concentration_risks"
    if re.search(r"\bcompare\b.*\b(states?|funded|expected|pipeline)\b"
                 r"|\bfunded (vs|versus) expected\b", q):
        return "compare_concentration_states"
    if re.search(r"\bexpected headroom\b", q):
        return "get_concentration_test"
    if re.search(r"\bwhich loans\b|\bcontribut|\bdrill", q):
        return "get_concentration_test_drillthrough"
    if re.search(r"\bdefinition\b|\bdefined\b|\bhow (is|do we|are we) (it |this )?"
                 r"(calculat|measur)|what does .* mean", q):
        return "explain_concentration_test"
    if re.search(r"\bsource\b.*\bapproval\b|\bapproval history\b|\bwho approved\b"
                 r"|\bprovenance\b", q):
        return "explain_concentration_test"
    if re.search(r"\bmoved from\b|\bwhat( has|'s| has been)? changed\b|\bworsen"
                 r"|\bdeteriorat|\bmovement\b|\bthis month\b|\bsince last\b", q):
        return "compare_concentration_periods"
    if re.search(r"\bcannot\b.*\bcalculat|\bunavailable\b|\bmissing data\b", q):
        return "list_concentration_unavailable"
    if re.search(r"\bclosest to\b|\bnearest to\b|\bwarning", q):
        return "list_concentration_warnings"
    if re.search(r"\bbreach", q):
        return "list_concentration_breaches"
    if re.search(r"\bshow\b.*\btests\b|\blist\b|\bwhat tests\b|\ball (the )?"
                 r"(tests|limits)\b", q):
        return "list_concentration_tests"
    if re.search(r"\bheadroom on\b|\bwhat is the current\b|\bhow much\b", q):
        return "get_concentration_test"
    return "get_concentration_summary"


def _category_filter(question: str) -> Optional[str]:
    q = str(question).lower()
    for category, words in _CATEGORY_WORDS.items():
        if any(w in q for w in words):
            return category
    return None


def _find_test(question: str, tests: List[Dict[str, Any]]
               ) -> Optional[Dict[str, Any]]:
    """The test whose display name best overlaps the question. Deterministic:
    token-overlap score, ties broken by the longer name then declaration
    order."""
    q_tokens = set(re.findall(r"[a-z0-9]+", str(question).lower()))
    best = None
    best_score = 0.0
    for t in tests:
        name_tokens = set(re.findall(r"[a-z0-9]+",
                                     str(t.get("displayName", "")).lower()))
        name_tokens -= {"exposure", "share", "concentration", "the", "of"}
        if not name_tokens:
            continue
        overlap = len(q_tokens & name_tokens) / len(name_tokens)
        if overlap > best_score or (overlap == best_score and best is not None
                                    and len(name_tokens) > 0 and overlap > 0.5):
            best, best_score = t, overlap
    return best if best_score >= 0.5 else None


def _movement_phrase(t: Dict[str, Any]) -> str:
    change = t.get("absoluteChange")
    if change is None:
        return "no prior period available"
    unit = "pp" if t.get("unit") == "percent" else ""
    direction = "up" if change > 0 else ("down" if change < 0 else "flat")
    phrase = f"{direction} {abs(change):,.2f}{unit} vs {t.get('priorReportingDate') or 'the prior period'}"
    if t.get("statusTransition"):
        phrase += f" ({t['statusTransition'].replace('->', '→')})"
    return phrase


def answer(question: str, envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Answer a limit question from the evaluated envelope.

    Returns {intent, answer, rows, warnings, testId} — the chat route builds
    the artifacts. ``rows`` are the tests the answer is about, verbatim from
    the envelope.
    """
    intent = detect_intent(question)
    tests: List[Dict[str, Any]] = list(envelope.get("tests") or [])
    summary = envelope.get("summary") or {}
    forecast = envelope.get("forecast") or {}
    states_available = bool((envelope.get("states") or {}).get("available"))
    warnings: List[str] = []
    if forecast.get("stagesUsingConfigFallback"):
        warnings.append(
            "Forecast caveat: stage(s) "
            + ", ".join(forecast["stagesUsingConfigFallback"])
            + " fall back to configured assumptions — the observed sample is "
              "below the sufficiency floor.")
    if envelope.get("source") == "legacy_extracted":
        warnings.append(
            "These are extracted limits pending operator approval — not yet "
            "approved concentration tests.")
    if envelope.get("openProposals"):
        warnings.append(
            f"{envelope['openProposals']} concentration-test proposal(s) are "
            "still awaiting review or approval from onboarding.")

    category = _category_filter(question)
    scoped = [t for t in tests if t.get("category") == category] if category else tests

    def _no_states(reason_intent: str) -> Dict[str, Any]:
        reason = (envelope.get("states") or {}).get("reason") or \
            forecast.get("reason") or \
            "no governed pipeline forecast is available"
        return {"intent": reason_intent,
                "answer": ("Forward-looking concentration states are not "
                           f"available: {reason} Funded results are "
                           "unaffected."),
                "rows": scoped, "warnings": warnings, "testId": None}

    if intent == "identify_emerging_concentration_risks":
        risks = envelope.get("emergingRisks") or []
        if not states_available and not risks:
            return _no_states(intent)
        if not risks:
            text = ("No emerging concentration risks: nothing is in breach, "
                    "expected to breach, or materially deteriorating"
                    + (" (based on the current completion-trend model)"
                       if states_available else "") + ".")
            return {"intent": intent, "answer": text, "rows": scoped,
                    "warnings": warnings, "testId": None}
        parts = [f"{len(risks)} emerging concentration risk(s), ranked:"]
        for i, r in enumerate(risks[:5], start=1):
            parts.append(f"{i}. {r['statement']}")
        rows = [t for t in tests
                if t.get("testId") in {r.get("testId") for r in risks}]
        return {"intent": intent, "answer": " ".join(parts),
                "rows": rows or scoped, "warnings": warnings, "testId": None}

    if intent == "list_full_pipeline_concentration_breaches":
        if not states_available:
            return _no_states(intent)
        rows = [t for t in scoped if t.get("fullPipelineBreach")]
        stress_only = [t for t in rows
                       if not t.get("expectedBreach")
                       and t.get("status") != "breach"]
        if rows:
            text = (f"If ALL active pipeline converted — a maximum-exposure "
                    f"stress, not an expected outcome — {len(rows)} test(s) "
                    "would breach: "
                    + "; ".join(
                        f"{t['displayName']} "
                        f"({_fmt(t.get('currentValue'), t.get('unit'))} → "
                        f"{_fmt((t.get('fullPipeline') or {}).get('value'), t.get('unit'))} "
                        f"vs {_fmt(t.get('threshold'), t.get('unit'))})"
                        for t in rows[:5]) + ".")
            if stress_only:
                text += (f" {len(stress_only)} of these breach ONLY in the "
                         "stress state — they remain compliant on the funded "
                         "book and in the expected forecast.")
        else:
            text = ("No test would breach even if all active pipeline "
                    "converted (the maximum-exposure stress state).")
        return {"intent": intent, "answer": text, "rows": rows or scoped,
                "warnings": warnings, "testId": None}

    if intent == "get_forecast_methodology":
        if not forecast.get("available"):
            return _no_states(intent)
        rates = forecast.get("stageRates") or {}
        rate_bits = []
        for stage in ("KFI", "APPLICATION", "OFFER"):
            r = rates.get(stage) or {}
            if r.get("rate") is not None:
                rate_bits.append(
                    f"{stage.title()} {r['rate']:.2f} (observed {r.get('observed', 0)}"
                    + (", sufficient" if r.get("sufficient") else
                       ", below the sufficiency floor — configured fallback")
                    + ")")
        parts = [
            "The Expected Forecast uses the existing deterministic "
            "completion-trend model — no machine learning, no invented "
            "probabilities.",
            f"Observation window {forecast.get('observationWindowStart')} → "
            f"{forecast.get('observationWindowEnd')} across "
            f"{forecast.get('weeklyExtractsUsed')} weekly extract(s), tracking "
            f"{forecast.get('trackedCaseCount')} case(s) with "
            f"{forecast.get('observedCompletionCount')} observed completion(s).",
        ]
        if rate_bits:
            parts.append("Stage completion rates: " + "; ".join(rate_bits)
                         + f". A stage needs at least "
                           f"{forecast.get('minObservations')} observed cases "
                           "before its empirical rate is trusted.")
        excl = forecast.get("excludedStageCounts") or {}
        if excl:
            parts.append("Withdrawn/cancelled and unknown-stage cases are "
                         "never counted or weighted ("
                         + ", ".join(f"{k.lower()}: {v}" for k, v in excl.items())
                         + ").")
        parts.append("Each pipeline case contributes balance × its stage "
                     "completion probability to both numerator and "
                     "denominator. Full Pipeline ignores probabilities "
                     "entirely — it is the maximum-exposure stress, not a "
                     "prediction.")
        if forecast.get("pointInTimeNote"):
            parts.append(forecast["pointInTimeNote"])
        return {"intent": intent, "answer": " ".join(parts), "rows": [],
                "warnings": warnings, "testId": None}

    if intent == "compare_concentration_states":
        if not states_available:
            return _no_states(intent)
        t = _find_test(question, scoped) or _find_test(question, tests)
        rows = [t] if t else scoped
        if t:
            expected = t.get("expected") or {}
            full = t.get("fullPipeline") or {}
            text = (f"{t['displayName']} — Funded (contractual): "
                    f"{_fmt(t.get('currentValue'), t.get('unit'))} "
                    f"[{_STATUS_LABEL.get(t['status'], t['status'])}]; "
                    f"Expected Forecast: {_fmt(expected.get('value'), t.get('unit'))} "
                    f"[{_STATUS_LABEL.get(expected.get('status'), expected.get('status') or 'n/a')}]; "
                    f"Full Pipeline (stress): {_fmt(full.get('value'), t.get('unit'))} "
                    f"[{_STATUS_LABEL.get(full.get('status'), full.get('status') or 'n/a')}] "
                    f"against a limit of "
                    f"{'≤' if t.get('operator') == 'max' else '≥'} "
                    f"{_fmt(t.get('threshold'), t.get('unit'))}.")
        else:
            text = (f"Across {len(scoped)} test(s): "
                    f"{summary.get('breaches', 0)} funded breach(es), "
                    f"{summary.get('expectedBreaches', 0)} expected breach(es), "
                    f"{summary.get('fullPipelineBreaches', 0)} stress-state "
                    "breach(es).")
        return {"intent": intent, "answer": text, "rows": rows,
                "warnings": warnings, "testId": t.get("testId") if t else None}

    if intent in ("explain_expected_concentration_change",
                  "get_concentration_pipeline_drivers"):
        if not states_available:
            return _no_states(intent)
        t = _find_test(question, scoped) or _find_test(question, tests)
        if t is None:
            names = ", ".join(x["displayName"] for x in tests[:8])
            return {"intent": intent,
                    "answer": ("I could not match that to an active "
                               f"concentration test. Active tests: {names}."),
                    "rows": tests, "warnings": warnings, "testId": None}
        # The route attaches the drivers table; the text carries the movement.
        expected = t.get("expected") or {}
        change = t.get("changeFundedToExpected")
        text = (f"{t['displayName']}: funded "
                f"{_fmt(t.get('currentValue'), t.get('unit'))}, expected "
                f"{_fmt(expected.get('value'), t.get('unit'))}"
                + (f" ({change:+.2f}pp)" if change is not None
                   and t.get("unit") == "percent" else "")
                + f" against {_fmt(t.get('threshold'), t.get('unit'))}.")
        if t.get("expectedBreach"):
            over = expected.get("breachAmount")
            text += (" That is an expected breach"
                     + (f" of {over:.2f}pp" if over is not None
                        and t.get("unit") == "percent" else "") + ".")
        horizon = (t.get("expectedBreachHorizon") or {}).get("period")
        if horizon:
            text += f" The expected crossing period is {horizon}."
        return {"intent": "get_concentration_pipeline_drivers",
                "answer": text, "rows": [t], "warnings": warnings,
                "testId": t["testId"]}

    if intent == "list_concentration_breaches":
        rows = [t for t in scoped if t["status"] == "breach"]
        if rows:
            text = (f"{len(rows)} concentration limit(s) are in breach: "
                    + "; ".join(_test_line(t) for t in rows[:5]) + ".")
        else:
            measured = [t for t in scoped if t.get("currentValue") is not None]
            text = ("No concentration limits are in breach on the funded book"
                    + (f" ({len(measured)} measured)" if measured else "") + ".")
            un = [t for t in scoped if t["status"] in ("unavailable",
                                                       "insufficient_data")]
            if un:
                text += (f" {len(un)} test(s) could not be evaluated and are "
                         "reported unavailable, not passing.")
        expected_rows = [t for t in scoped if t.get("expectedBreach")]
        if expected_rows:
            text += (f" However, {len(expected_rows)} test(s) are EXPECTED to "
                     "breach based on the completion-trend model: "
                     + "; ".join(
                         f"{t['displayName']} "
                         f"({_fmt(t.get('currentValue'), t.get('unit'))} → "
                         f"{_fmt((t.get('expected') or {}).get('value'), t.get('unit'))})"
                         for t in expected_rows[:3]) + ".")
        return {"intent": intent, "answer": text,
                "rows": rows or expected_rows or scoped,
                "warnings": warnings, "testId": None}

    if intent == "list_concentration_warnings":
        rows = [t for t in scoped if t["status"] == "warning"]
        measured = sorted((t for t in scoped if t.get("headroom") is not None),
                          key=lambda t: t["headroom"])
        closest = measured[:3]
        parts = []
        if rows:
            parts.append(f"{len(rows)} test(s) at warning: "
                         + "; ".join(_test_line(t) for t in rows[:5]) + ".")
        if closest:
            parts.append("Closest to breach: "
                         + "; ".join(f"{t['displayName']} ({_headroom_phrase(t)})"
                                     for t in closest) + ".")
        if not parts:
            parts.append("No tests are currently at warning, and no measured "
                         "headroom is available to rank.")
        return {"intent": intent, "answer": " ".join(parts),
                "rows": rows or closest, "warnings": warnings, "testId": None}

    if intent == "list_concentration_unavailable":
        rows = [t for t in scoped if t["status"] in ("unavailable",
                                                     "insufficient_data")]
        if rows:
            reasons = "; ".join(
                f"{t['displayName']} ({t.get('notes') or t.get('dataStatus')})"
                for t in rows[:6])
            text = (f"{len(rows)} test(s) cannot currently be calculated: "
                    f"{reasons}. They are reported unavailable — never as "
                    "passing or zero.")
        else:
            text = "Every active concentration test evaluated successfully."
        return {"intent": intent, "answer": text, "rows": rows,
                "warnings": warnings, "testId": None}

    if intent == "compare_concentration_periods":
        if not envelope.get("priorAvailable"):
            return {"intent": intent,
                    "answer": ("No prior governed snapshot is available, so "
                               "period movement cannot be reported yet."),
                    "rows": scoped, "warnings": warnings, "testId": None}
        moved = [t for t in scoped if t.get("statusTransition")]
        worsened = [t for t in scoped if t.get("deteriorated")]
        changed = sorted((t for t in scoped if t.get("absoluteChange")),
                         key=lambda t: -abs(t.get("absoluteChange") or 0))
        parts = [f"Comparing {envelope.get('priorReportingDate')} → "
                 f"{envelope.get('reportingDate')}:"]
        if moved:
            parts.append("status changes — "
                         + "; ".join(f"{t['displayName']} "
                                     f"{t['statusTransition'].replace('->', '→')}"
                                     for t in moved) + ".")
        else:
            parts.append("no test changed status.")
        if changed:
            top = changed[:3]
            parts.append("Largest movements: "
                         + "; ".join(f"{t['displayName']} {_movement_phrase(t)}"
                                     for t in top) + ".")
        if worsened:
            parts.append(f"{len(worsened)} test(s) deteriorated.")
        rows = moved or changed[:5]
        return {"intent": intent, "answer": " ".join(parts), "rows": rows,
                "warnings": warnings, "testId": None}

    if intent in ("get_concentration_test", "explain_concentration_test",
                  "get_concentration_test_drillthrough"):
        t = _find_test(question, scoped) or _find_test(question, tests)
        if t is None:
            names = ", ".join(x["displayName"] for x in tests[:8])
            return {"intent": intent,
                    "answer": ("I could not match that to an active "
                               f"concentration test. Active tests: {names}."),
                    "rows": tests, "warnings": warnings, "testId": None}
        if intent == "get_concentration_test_drillthrough":
            return {"intent": intent, "answer": "", "rows": [t],
                    "warnings": warnings, "testId": t["testId"]}
        if intent == "explain_concentration_test":
            d = t.get("definition") or {}
            p = t.get("provenance") or {}
            parts = [f"{t['displayName']} — approved contractual limit: "
                     f"{'≤' if t['operator'] == 'max' else '≥'} "
                     f"{_fmt(t.get('threshold'), t.get('unit') or 'percent')}."]
            if d.get("description"):
                parts.append(f"Definition: {d['description']}")
            if d.get("numerator"):
                parts.append(f"Numerator: {d['numerator']}")
            if t.get("parameters"):
                parts.append(f"Approved parameters: {t['parameters']}.")
            if d.get("definitionNotes"):
                parts.append(f"Confirmed at onboarding: {d['definitionNotes']}")
            if p.get("sourceText"):
                parts.append(f"Source wording: \"{p['sourceText']}\"")
            if p.get("approvedBy"):
                parts.append(f"Approved by {p['approvedBy']} on "
                             f"{(p.get('approvedAt') or '')[:10]}"
                             + (f" ({p['approvalComments']})"
                                if p.get("approvalComments") else "") + ".")
            parts.append(f"Configuration version {t.get('configurationVersion')}.")
            return {"intent": intent, "answer": " ".join(parts), "rows": [t],
                    "warnings": warnings, "testId": t["testId"]}
        # get_concentration_test
        parts = [_test_line(t) + "."]
        parts.append(_headroom_phrase(t).capitalize() + " on the funded book.")
        if t.get("utilization") is not None:
            parts.append(f"Utilization {t['utilization']:.1f}% of the limit.")
        expected = t.get("expected") or {}
        full = t.get("fullPipeline") or {}
        if expected.get("value") is not None:
            exp_h = expected.get("headroom")
            parts.append(
                f"Expected Forecast: {_fmt(expected['value'], t.get('unit'))}"
                + (f" (expected headroom {exp_h:,.2f}"
                   + ("pp" if t.get("unit") == "percent" else "") + ")"
                   if exp_h is not None else "")
                + ("— an expected breach." if t.get("expectedBreach") else "."))
        elif expected.get("status") == "indicative_only":
            parts.append("The expected state is indicative only for this "
                         "metric family.")
        if full.get("value") is not None:
            parts.append(
                f"Full Pipeline (maximum-exposure stress, not a prediction): "
                f"{_fmt(full['value'], t.get('unit'))}"
                + (" — would breach." if t.get("fullPipelineBreach") else "."))
        parts.append("Movement: " + _movement_phrase(t) + ".")
        if t["status"] in ("unavailable", "insufficient_data"):
            parts.append(f"Data limitation: {t.get('notes') or t['dataStatus']}.")
        return {"intent": intent, "answer": " ".join(parts), "rows": [t],
                "warnings": warnings, "testId": t["testId"]}

    if intent == "list_concentration_tests":
        label = (category.replace("_", " ") + " " if category else "")
        if not scoped:
            return {"intent": intent,
                    "answer": f"No active {label}concentration tests.",
                    "rows": [], "warnings": warnings, "testId": None}
        text = (f"{len(scoped)} active {label}concentration test(s): "
                + "; ".join(_test_line(t) for t in scoped[:8]) + ".")
        return {"intent": intent, "answer": text, "rows": scoped,
                "warnings": warnings, "testId": None}

    # get_concentration_summary
    parts = [f"{summary.get('activeTests', 0)} active concentration test(s) at "
             f"{envelope.get('reportingDate') or 'the latest reporting date'}: "
             f"{summary.get('passes', 0)} pass, {summary.get('warnings', 0)} "
             f"warning(s), {summary.get('breaches', 0)} breach(es), "
             f"{summary.get('unavailable', 0)} unavailable."]
    if summary.get("breaches"):
        rows = [t for t in tests if t["status"] == "breach"]
        parts.append("Breached: " + "; ".join(t["displayName"] for t in rows) + ".")
    closest = summary.get("closestToLimit")
    if closest:
        unit = "pp" if closest.get("unit") == "percent" else closest.get("unit") or ""
        parts.append(f"Nearest to limit: {closest['displayName']} "
                     f"({closest['headroom']:,.2f} {unit} headroom).")
    if summary.get("deteriorations"):
        parts.append(f"{summary['deteriorations']} test(s) deteriorated since "
                     f"{envelope.get('priorReportingDate')}.")
    return {"intent": "get_concentration_summary", "answer": " ".join(parts),
            "rows": tests, "warnings": warnings, "testId": None}
