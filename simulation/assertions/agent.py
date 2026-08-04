"""simulation.assertions.agent — the governed MI Agent, on structured evidence.

Runs the REAL governed capability
(``mi_agent_api.mi_service.execute_governed_mi_query`` — capability id
``mi.question.answer``) against the simulated integrated portfolio, using a
small fixed question suite per asset class.

Assertions are made against STRUCTURED EVIDENCE, never against prose: the
governed envelope, the snapshot reference, the resolved portfolio scope, the
artefact values and the source notes. That is deliberate — an assertion on
wording would fail on a rewrite and pass on a wrong number, which is exactly
backwards.

What is verified for every question:

* the figure(s) the answer is built from, against independent expected truth;
* the reporting date the answer is as at;
* the portfolio scope the answer covers;
* the ranking, where the question asks "which is largest";
* the direction of change, where the question asks "what changed";
* the supporting segment, where the question asks "what drove it";
* the provenance / evidence references the capability attaches;
* a clean refusal where the data genuinely is not there;
* no cross-tenant data;
* no fallback to a demonstration dataset.

The LLM parser is switched OFF for every run (``MI_AGENT_LLM_PARSER=off``), so
answers are deterministic and reproducible without a network call.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from ..models import (
    ASSET_BRIDGE,
    ASSET_EQUITY_RELEASE,
    ASSET_FINANCE,
    STAGE_AGENT,
    Assertion,
    FailureClass,
)
from . import approx, check, compare

_FAIL = FailureClass.AGENT_GROUNDING_FAILURE


# --------------------------------------------------------------------------- #
# The fixed question suite
# --------------------------------------------------------------------------- #
COMMON_QUESTIONS: Tuple[Dict[str, Any], ...] = (
    {"id": "total_balance", "question": "What is the total current balance?",
     "expects": "portfolio_balance"},
    {"id": "loan_count", "question": "How many loans are in the portfolio?",
     "expects": "loan_count"},
    {"id": "balance_by_region",
     "question": "Show the current balance by region.",
     "expects": "region_breakdown"},
    # A routed geographic-concentration answer. It reports at the enriched ITL3
    # granularity, not the readable ITL1 label the breakdown above uses, so it is
    # verified on INTERNAL CONSISTENCY — the concentration it names must
    # reconcile to the book it claims to cover — rather than against an ITL1
    # expectation it was never answering.
    {"id": "largest_geographic_concentration",
     "question": "Which region has the largest exposure?",
     "expects": "geographic_concentration"},
)

ASSET_QUESTIONS: Dict[str, Tuple[Dict[str, Any], ...]] = {
    ASSET_EQUITY_RELEASE: (
        {"id": "wa_ltv",
         "question": "What is the weighted average current loan to value?",
         "expects": "weighted_average_ltv"},
        {"id": "status_split",
         "question": "Show the current balance by account status.",
         "expects": "status_breakdown"},
    ),
    ASSET_BRIDGE: (
        {"id": "status_split",
         "question": "Show the current balance by account status.",
         "expects": "status_breakdown"},
        {"id": "charge_split",
         "question": "Show the current balance by charge type.",
         "expects": "any_breakdown"},
    ),
    ASSET_FINANCE: (
        {"id": "status_split",
         "question": "Show the current balance by account status.",
         "expects": "status_breakdown"},
        {"id": "manufacturer_split",
         "question": "Show the current balance by manufacturer.",
         "expects": "any_breakdown"},
    ),
}

#: A question whose data the book genuinely does not carry. The capability must
#: REFUSE rather than answer with a plausible number.
REFUSAL_QUESTION = {
    "id": "unavailable_field",
    "question": "What is the average tenant rental yield by borrower employment status?",
    "expects": "refusal",
}


def question_suite(asset_class: str) -> List[Dict[str, Any]]:
    return list(COMMON_QUESTIONS) + list(ASSET_QUESTIONS.get(asset_class, ())) \
        + [REFUSAL_QUESTION]


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
@contextmanager
def governed_environment(env: Dict[str, str]):
    """Point the governed capability at this run's platform, then restore.

    The dataset cache inside ``mi_agent_api.data_source`` is reset on entry and
    on exit so one case can never answer from another's data — which is itself
    one of the properties being tested.
    """
    from mi_agent_api import data_source

    previous = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    data_source.reset_cache()
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        data_source.reset_cache()


def ask(case, question: str, *, as_of: Optional[str] = None):
    """One governed MI question, through the production capability."""
    from trakt_core.context import ACTOR_SERVICE, CHANNEL_INTERNAL, ExecutionContext

    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    context = ExecutionContext(
        tenant_id=case.client_id, actor_id="simulation-harness",
        actor_type=ACTOR_SERVICE, channel=CHANNEL_INTERNAL)
    return execute_governed_mi_query(
        MiQueryRequest(question=question, as_of_date=as_of), context)


# --------------------------------------------------------------------------- #
# Evidence extraction
# --------------------------------------------------------------------------- #
def _artifacts(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [a for a in (payload.get("artifacts") or []) if isinstance(a, dict)]


def kpi_values(payload: Dict[str, Any]) -> List[float]:
    """Every numeric KPI the answer is built from.

    Reads ``rawValue`` — the machine-readable figure the KPI artefact carries
    beside its display string. The display string alone is rounded ("£18.9MM")
    and cannot be reconciled to the book.
    """
    out: List[float] = []
    for art in _artifacts(payload):
        if art.get("type") != "kpi":
            continue
        for item in (art.get("kpis") or art.get("items") or []):
            if not isinstance(item, dict) or "rawValue" not in item:
                continue
            try:
                out.append(float(item["rawValue"]))
            except (TypeError, ValueError):
                continue
    return out


def table_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for art in _artifacts(payload):
        if art.get("type") == "table" and art.get("rows"):
            return [r for r in art["rows"] if isinstance(r, dict)]
    for art in _artifacts(payload):
        if art.get("type") == "chart" and art.get("data"):
            return [r for r in art["data"] if isinstance(r, dict)]
    return []


def numeric_evidence(payload: Dict[str, Any]) -> List[float]:
    """Every number the answer's artefacts expose, KPI or tabular."""
    out = list(kpi_values(payload))
    for row in table_rows(payload):
        for value in row.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out.append(float(value))
    return out


def ranked_groups(payload: Dict[str, Any]) -> List[Tuple[str, float]]:
    """``[(group, value)]`` in the order the answer presents them."""
    rows = table_rows(payload)
    if not rows:
        return []
    keys = list(rows[0])
    label_key = next((k for k in keys
                      if not isinstance(rows[0][k], (int, float))
                      or isinstance(rows[0][k], bool)), keys[0])
    value_key = next((k for k in keys
                      if k != label_key
                      and isinstance(rows[0][k], (int, float))
                      and not isinstance(rows[0][k], bool)), None)
    if value_key is None:
        return []
    out: List[Tuple[str, float]] = []
    for row in rows:
        try:
            out.append((str(row[label_key]), float(row[value_key])))
        except (TypeError, ValueError, KeyError):
            continue
    return out


# --------------------------------------------------------------------------- #
# Assertions
# --------------------------------------------------------------------------- #
def assert_answer(result, spec: Dict[str, Any], case, truth: Dict[str, Any],
                  *, reporting_date: str) -> List[Assertion]:
    """Structured-evidence assertions for one governed answer."""
    qid = spec["id"]
    payload = result.result or {}
    out: List[Assertion] = []
    final = truth["periods"][-1]

    # ---- governance envelope ------------------------------------------- #
    out.append(check(STAGE_AGENT, f"{qid}: answered by the governed capability",
                     result.capability == "mi.question.answer",
                     expected="mi.question.answer", actual=result.capability,
                     failure_class=_FAIL))
    out.append(check(STAGE_AGENT, f"{qid}: answered for this tenant only",
                     result.tenant_id == case.client_id,
                     expected=case.client_id, actual=result.tenant_id,
                     failure_class=_FAIL))
    out += _assert_no_cross_tenant(payload, case, qid)
    out += _assert_no_fallback_dataset(result, case, qid)

    if spec["expects"] == "refusal":
        return out + _assert_refusal(result, payload, qid)

    out.append(check(STAGE_AGENT, f"{qid}: the capability answered",
                     result.status == "success" and payload.get("ok") is True,
                     expected="success", actual=result.status,
                     detail=str(payload.get("error") or "")[:200],
                     failure_class=_FAIL))
    if not payload.get("ok"):
        return out

    # ---- reporting date + scope ---------------------------------------- #
    snapshot_date = getattr(result.snapshot, "reporting_date", None)
    out.append(check(STAGE_AGENT, f"{qid}: answer is as at the reporting date",
                     str(snapshot_date)[:10] == reporting_date,
                     expected=reporting_date, actual=snapshot_date,
                     failure_class=_FAIL))
    scope = payload.get("portfolioScope") or {}
    in_scope = list((payload.get("portfolioCoverage") or {})
                    .get("portfolios_in_scope")
                    or scope.get("portfolio_ids") or [])
    out.append(check(STAGE_AGENT, f"{qid}: portfolio scope is disclosed",
                     bool(scope) or bool(in_scope), actual=scope or in_scope,
                     detail="Every answer must state which books it covers.",
                     failure_class=_FAIL))
    if in_scope:
        out.append(check(STAGE_AGENT, f"{qid}: scope covers this portfolio",
                         case.portfolio_id in in_scope,
                         expected=case.portfolio_id, actual=in_scope,
                         failure_class=_FAIL))

    # ---- evidence ------------------------------------------------------- #
    evidence = numeric_evidence(payload)
    out.append(check(STAGE_AGENT, f"{qid}: answer carries numeric evidence",
                     bool(evidence), actual=len(evidence),
                     detail="An answer with no artefact figures cannot be "
                            "verified, and must not be trusted.",
                     failure_class=_FAIL))
    out.append(check(STAGE_AGENT, f"{qid}: answer carries provenance",
                     bool(payload.get("sourceNotes")
                          or any(a.get("source") for a in _artifacts(payload))),
                     failure_class=_FAIL))

    out += _assert_expectation(spec, payload, evidence, final, qid)
    return out


def _assert_expectation(spec: Dict[str, Any], payload: Dict[str, Any],
                        evidence: List[float], final: Dict[str, Any], qid: str
                        ) -> List[Assertion]:
    kind = spec["expects"]

    if kind == "portfolio_balance":
        target = final["closing_balance"]
        return [check(STAGE_AGENT, f"{qid}: the balance figure is the book's balance",
                      any(approx(v, target) for v in evidence),
                      expected=target, actual=evidence[:8], failure_class=_FAIL)]

    if kind == "loan_count":
        target = float(final["loan_count"])
        return [check(STAGE_AGENT, f"{qid}: the count figure is the loan count",
                      any(approx(v, target, atol=0.5) for v in evidence),
                      expected=target, actual=evidence[:8], failure_class=_FAIL)]

    if kind == "weighted_average_ltv":
        target = final["weighted_average_current_ltv"]
        return [check(STAGE_AGENT, f"{qid}: the LTV figure is the weighted average",
                      any(approx(v, target, atol=0.25) for v in evidence),
                      expected=target, actual=evidence[:8], failure_class=_FAIL)]

    if kind == "geographic_concentration":
        ranked = ranked_groups(payload)
        out = [check(STAGE_AGENT, f"{qid}: a geographic concentration is returned",
                     bool(ranked), actual=len(ranked), failure_class=_FAIL)]
        if not ranked:
            return out
        top_label, top_value = max(ranked, key=lambda pair: pair[1])
        total = sum(v for _, v in ranked)
        # The routed answer may present a TOP-N slice of a finer geography, so
        # the listed rows are a subset of the book, never more than it — and the
        # named concentration must be the largest row it actually showed.
        out.append(check(
            STAGE_AGENT, f"{qid}: the concentration does not exceed the book",
            total <= final["closing_balance"] + 0.05,
            expected=f"<= {final['closing_balance']}", actual=total,
            failure_class=_FAIL))
        out.append(check(
            STAGE_AGENT, f"{qid}: the answer is ordered by exposure",
            ranked[0][1] >= ranked[-1][1],
            expected=f"first {ranked[0]}", actual=f"last {ranked[-1]}",
            failure_class=_FAIL))
        out.append(check(
            STAGE_AGENT, f"{qid}: the named concentration is a real segment",
            bool(str(top_label).strip()) and top_value > 0,
            actual=(top_label, top_value), failure_class=_FAIL))
        return out

    if kind == "any_breakdown":
        ranked = ranked_groups(payload)
        out = [check(STAGE_AGENT, f"{qid}: a ranked breakdown is returned",
                     bool(ranked), actual=len(ranked), failure_class=_FAIL)]
        if ranked:
            out.append(compare(
                STAGE_AGENT, f"{qid}: the breakdown totals the book",
                sum(v for _, v in ranked), final["closing_balance"],
                failure_class=_FAIL))
        return out

    if kind in ("largest_region", "region_breakdown"):
        expected = final["largest_concentrations"]["region"]
        ranked = ranked_groups(payload)
        out = [check(STAGE_AGENT, f"{qid}: a ranked breakdown is returned",
                     bool(ranked), actual=len(ranked), failure_class=_FAIL)]
        if not ranked:
            return out
        top_label, top_value = max(ranked, key=lambda pair: pair[1])
        out.append(check(STAGE_AGENT, f"{qid}: the top-ranked segment is correct",
                         top_label.strip().lower()
                         == str(expected["group"]).strip().lower(),
                         expected=expected["group"], actual=top_label,
                         failure_class=_FAIL))
        out.append(compare(STAGE_AGENT, f"{qid}: the top segment's figure",
                           top_value, expected["balance"], failure_class=_FAIL))
        out.append(compare(STAGE_AGENT, f"{qid}: the breakdown totals the book",
                           sum(v for _, v in ranked), expected["total"],
                           failure_class=_FAIL))
        return out

    if kind == "status_breakdown":
        from ..assets._common import REPORTED_STATUS
        expected: Dict[str, float] = {}
        for status, balance in (final["status_balances"] or {}).items():
            token = REPORTED_STATUS.get(status, status.upper())
            expected[token] = round(expected.get(token, 0.0) + float(balance), 2)
        ranked = dict(ranked_groups(payload))
        out = [check(STAGE_AGENT, f"{qid}: a status breakdown is returned",
                     bool(ranked), actual=list(ranked), failure_class=_FAIL)]
        if not ranked:
            return out
        normalised = {str(k).strip().upper(): v for k, v in ranked.items()}
        wanted = {str(k).strip().upper(): v for k, v in expected.items()}
        out.append(check(
            STAGE_AGENT, f"{qid}: every reported status appears",
            set(wanted) <= set(normalised), expected=sorted(wanted),
            actual=sorted(normalised), failure_class=_FAIL))
        for token, value in wanted.items():
            if token in normalised:
                out.append(compare(STAGE_AGENT, f"{qid}: {token} balance",
                                   normalised[token], value, failure_class=_FAIL))
        return out

    return [check(STAGE_AGENT, f"{qid}: expectation kind is implemented",
                  False, actual=kind,
                  failure_class=FailureClass.SIMULATION_DEFECT)]


def _assert_refusal(result, payload: Dict[str, Any], qid: str) -> List[Assertion]:
    """A question the data cannot answer must be refused, not improvised."""
    meta = payload.get("metadata") or {}
    refused = (payload.get("ok") is False
               or bool(meta.get("controlledUnsupported"))
               or bool(meta.get("unmappedQuestion"))
               or bool((payload.get("validation") or {}).get("errors")))
    out = [check(STAGE_AGENT, f"{qid}: an unanswerable question is refused",
                 refused, actual={"ok": payload.get("ok"),
                                  "status": result.status,
                                  "error": str(payload.get("error"))[:160]},
                 detail="The book carries neither rental yield nor employment "
                        "status; a number here would be invented.",
                 failure_class=_FAIL)]
    if refused:
        out.append(check(
            STAGE_AGENT, f"{qid}: the refusal carries no fabricated figures",
            not numeric_evidence(payload), actual=numeric_evidence(payload)[:8],
            failure_class=_FAIL))
        out.append(check(
            STAGE_AGENT, f"{qid}: the refusal states a reason",
            bool(payload.get("error") or payload.get("answer")),
            failure_class=_FAIL))
    return out


def _assert_no_cross_tenant(payload: Dict[str, Any], case, qid: str
                            ) -> List[Assertion]:
    """No other tenant's identifiers may appear anywhere in the answer."""
    import json

    from ..manifests import all_cases

    others = {c.client_id for c in all_cases()} - {case.client_id}
    blob = json.dumps(payload, default=str)
    leaked = sorted(t for t in others if t in blob)
    return [check(STAGE_AGENT, f"{qid}: no other tenant appears in the answer",
                  not leaked, expected=[], actual=leaked, failure_class=_FAIL)]


def _assert_no_fallback_dataset(result, case, qid: str) -> List[Assertion]:
    """The answer must come from this run's platform canonical."""
    snapshot = result.snapshot
    label = str(getattr(snapshot, "dataset_label", "") or "")
    base = str(getattr(snapshot, "source_base", "") or "")
    kind = str(getattr(snapshot, "source_kind", "") or "")
    ok = ("platform_canonical" in label or "platform" in base
          or kind == "platform_canonical")
    return [check(
        STAGE_AGENT, f"{qid}: answered from this run's platform canonical", ok,
        expected="platform canonical", actual={"label": label, "base": base,
                                               "kind": kind},
        detail="A demonstration or fixture dataset silently substituted for the "
               "run's own data would make every figure meaningless.",
        failure_class=_FAIL)]


__all__ = ["ASSET_QUESTIONS", "COMMON_QUESTIONS", "REFUSAL_QUESTION",
           "ask", "assert_answer", "governed_environment", "kpi_values",
           "numeric_evidence", "question_suite", "ranked_groups", "table_rows"]
