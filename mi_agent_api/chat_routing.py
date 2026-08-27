"""mi_agent_api/chat_routing.py

End-to-end routing of the new governed analytical intents through POST /mi/query.

The deterministic parser already compiles questions into governed plans
(``temporal_mode='compare'``, ``forecast_mode='extrapolation'``,
``risk_limit_query``, evolution line specs). This module detects those plans and
executes them against the INTERNAL services already built for the dashboard —
``temporal_compare``, ``evolution``, ``forecast_extrapolation`` and
``risk_limits`` — then shapes the result into the SAME artifact union the React
chat/workspace already renders (chart | table | risk | kpi). No HTTP hop, no new
renderer, no parser rebuild.

``try_route`` returns a full ``/mi/query`` response envelope when it handles a
question, or ``None`` to defer to the existing point-in-time MI Agent path
(``run_mi_agent_query`` + ``adapt_workflow_result``) — so normal funded/pipeline/
forecast/data-quality questions are completely unaffected.
"""

from __future__ import annotations

import logging as _logging

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mi_agent import population as _population
from mi_agent.mi_query_executor import MIQueryExecutionError
from mi_agent.parsed_question import ParsedQuestion

from mi_agent import period_request as _period_request
from mi_agent import portfolio_lens as _portfolio_lens

_logger = _logging.getLogger("mi_agent_api.chat_routing")

from . import temporal_compare as compare_mod
from . import currency as currency_mod
from . import evolution as evolution_mod
from . import forecast_extrapolation as fx_mod
from . import movement_summary as movement_mod
from . import period_change_route as _period_change
from . import analytical_plan as _plan
from . import risk_limits as risk_mod
from . import scenario as scenario_mod
from . import workspace as _workspace
from .recogniser_registry import (
    REGISTRY,
    Recogniser,
    Recognition,
    RecogniserRegistry,
    RouteRequest,
    resolve_capability_state,
)

from mi_workflows import concentration_analysis as conc_mod
from mi_workflows import portfolio_risk_comparison as prc_mod
from mi_workflows.analytical import intent as analytical_intent
from mi_workflows.analytical import route as analytical_mod
from mi_workflows.semantics import load_business_semantics

from trakt_core.portfolio import (
    CAP_COHORTS,
    CAP_ORIGINATION_FORECAST,
    CAP_PIPELINE,
    CAP_RISK,
)

_PALETTE = ["#919dd1", "#36c2a8", "#e0a93b", "#c46b8f", "#3d4a82", "#6fcf97"]

# Per evolution-metric display: (answer_style, chart valueFormat, chart scale).
_METRIC_DISPLAY: Dict[str, Tuple[str, str, Optional[str]]] = {
    "funded_balance": ("gbp", "gbp", None),
    "avg_balance": ("gbp", "gbp", None),
    "pipeline_amount": ("gbp", "gbp", None),
    "weighted_expected_funded_amount": ("gbp", "gbp", None),
    "loan_count": ("count", "number", None),
    "pipeline_case_count": ("count", "number", None),
    # Both fractions: the evolution routes now emit one convention, so the
    # display scale is the same for every percent metric they carry.
    "wa_ltv": ("pct_fraction", "pct", "percent_fraction"),
    "wa_interest_rate": ("pct_fraction", "pct", "percent_fraction"),
    "avg_borrower_age": ("decimal", "decimal", None),
}


# --------------------------------------------------------------------------- #
# Small helpers (mirror adapters._uid / _now without coupling to it)
# --------------------------------------------------------------------------- #
def _uid(prefix: str = "art") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gbp(v: Optional[float]) -> str:
    # Name kept for call-site stability; the symbol is the request's resolved
    # currency (tape -> config -> GBP), not a hardcoded £.
    return currency_mod.format_money(v, suffixes=("bn", "m", "k"))


def _disp(value: Optional[float], metric_key: str,
          decimals: Optional[int] = None) -> str:
    """Render a metric in ITS OWN declared unit.

    `_METRIC_DISPLAY` is the canonical statement of whether a metric is carried
    as a FRACTION (0.0626) or as PERCENTAGE POINTS (36.26). A caller that picks
    a formatter by eye has to remember which, and the portfolio summary did not:
    it rendered `wa_interest_rate` — a fraction, so declared here — with the
    points formatter and published **0.06%** for a book yielding **6.2644%**, a
    hundredfold error in the first figure a reader sees.

    Reading the declaration is what makes that unrepresentable, which is why the
    fix is here and not a ×100 at the summary's call site.
    """
    if value is None:
        return "—"
    style = _METRIC_DISPLAY.get(metric_key, ("decimal", "decimal", None))[0]
    if style == "gbp":
        return _gbp(value)
    if style == "count":
        return f"{int(round(float(value))):,}"
    if style == "pct_fraction":
        return f"{float(value) * 100:.{1 if decimals is None else decimals}f}%"
    if style == "pct_points":
        return f"{float(value):.{2 if decimals is None else decimals}f}%"
    return f"{float(value):,.2f}"


def _source(label: str, spec: Dict[str, Any], portfolio_id: Optional[str],
            as_of: Optional[str], engine: str = "mi_agent.workflow") -> Dict[str, Any]:
    return {"engine": engine, "label": label, "spec": spec,
            "asOf": as_of, "portfolio": portfolio_id}


def _envelope(*, ok: bool, question: str, answer: str, spec: Dict[str, Any],
              artifacts: List[Dict[str, Any]], reconciliation: Optional[Dict[str, Any]] = None,
              source_notes: Optional[List[Dict[str, Any]]] = None,
              warnings: Optional[List[str]] = None, route: str = "",
              error: Optional[str] = None,
              lens_applied: Optional[bool] = None) -> Dict[str, Any]:
    notes = source_notes or []
    for art in artifacts:
        if art.get("type") in ("chart", "table", "kpi") and reconciliation:
            art.setdefault("reconciliation", reconciliation)
        if art.get("type") in ("chart", "table", "kpi") and notes:
            art.setdefault("sourceNotes", notes)
    return {
        "ok": ok,
        "error": error,
        "question": question,
        "answer": answer,
        "interpreted": "",
        "spec": spec,
        "validation": {"ok": ok, "errors": ([] if ok else [error or "unavailable"]),
                       "warnings": [], "resolved_fields": {}},
        "artifacts": artifacts,
        "reconciliation": reconciliation,
        "sourceNotes": notes,
        "warnings": warnings or [],
        "diagnostics": [],
        "assumptions": [],
        "metadata": {"engine": "mi_agent", "source": "python", "mock": False,
                     "route": route,
                     # Whether this route actually narrowed its figures to the
                     # requested portfolio lens. ``None`` means "not stated" and
                     # is resolved centrally in try_route; ``False`` means the
                     # numbers are whole-book and MUST NOT be labelled otherwise.
                     "lensApplied": lens_applied},
    }


def _undeliverable(**kwargs) -> Dict[str, Any]:
    """THE governed "the analysis you asked for was not produced" envelope.

    `ok: true` MEANS THE REQUESTED ANALYSIS WAS DELIVERED.

    Twenty-three routed sites used to say `ok=True` with no artifacts and an
    answer that explained an inability — "I can't build a geographic exposure
    view for this book", "No weekly pipeline extracts are available", "I
    couldn't resolve a dimension to attribute the bridge by". A caller reading
    the envelope saw a success; a reader reading the prose saw a refusal. On the
    API those are the same field, and it was telling them different things.

    NO NEW PUBLIC TAXONOMY. This is the existing governed contract for "I will
    not answer that": `ok:false` plus `metadata.controlledUnsupported`, which
    `mi_service` classifies as `UNSUPPORTED_QUESTION` (HTTP 200, `ok:false`) —
    the same shape `_capability_unavailable_envelope` already publishes.

    A ZERO-ROW ANSWER IS NOT THIS. "There are no funded loans in the acquired
    book" is an analysis that ran over an empty population, and it keeps
    `ok=True`: turning an empty result into a failure would be the opposite
    error. Three sites are deliberately left as they were — the two empty-scope
    statements, and the run-rate branch that reports the current balance and
    discloses that it could not extrapolate.
    """
    kwargs.pop("ok", None)
    kwargs.pop("artifacts", None)
    answer = kwargs.get("answer") or "This analysis could not be produced."
    kwargs.setdefault("error", answer)
    envelope = _envelope(ok=False, artifacts=[], **kwargs)
    # THE ESTATE'S EXISTING MARKERS, all three, exactly as the portfolio
    # comparison route already stamps them. A caller that recognised a
    # controlled refusal by `controlledRefusal` must keep recognising it.
    envelope["controlledRefusal"] = True
    envelope["metadata"]["controlledRefusal"] = True
    envelope["metadata"]["controlledUnsupported"] = True
    return envelope


# --------------------------------------------------------------------------- #
# Artifact builders (existing artifact union — chart | table | risk)
# --------------------------------------------------------------------------- #
def _chart_artifact(title: str, *, chart_type: str, x_key: str,
                    rows: List[Dict[str, Any]], series: List[Dict[str, str]],
                    value_format: str, spec: Dict[str, Any],
                    portfolio_id: Optional[str], as_of: Optional[str],
                    display_hints: Optional[Dict[str, Any]] = None,
                    description: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": _uid(), "type": "chart", "title": title,
        "description": description,
        "source": {**_source(f"MI Agent · {chart_type}", spec, portfolio_id, as_of),
                   "nativeChartType": chart_type},
        "createdAt": _now(), "mock": False,
        "chartType": chart_type, "xKey": x_key,
        "series": series, "rows": rows, "valueFormat": value_format,
        "displayHints": display_hints or {},
        "warnings": [],
    }


def _table_artifact(title: str, *, columns: List[Dict[str, Any]],
                    rows: List[Dict[str, Any]], spec: Dict[str, Any],
                    portfolio_id: Optional[str], as_of: Optional[str],
                    description: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": _uid(), "type": "table", "title": title,
        "description": description or f"{len(rows)} rows.",
        "source": _source("MI Agent · table", spec, portfolio_id, as_of),
        "createdAt": _now(), "mock": False,
        "columns": columns, "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Dataset / metric resolution
# --------------------------------------------------------------------------- #
# THE SECOND OWNER IS GONE.
#
# `_dataset_for` used to live here. It read its own tape vocabulary
# (`pipeline | case | kfi | application | offer`) and fell back to the resolved
# view, which made it a second answer to "which dataset is this question
# about?" — one that disagreed with the contract's on 3 of 26 readings of the
# `temporal_compare` surface, and with the point-in-time path on 29 of the 882
# corpus questions.
#
# Its vocabulary is now `workspace.PIPELINE_ARTEFACTS`, read by
# `workspace.resolve_dataset`, which is the single owner. Routes ask that owner;
# they do not re-decide. Converted routes will read `interpretation.dataset`,
# which the same owner populates.


def _split_portfolio(portfolio_id: Optional[str]) -> Tuple[str, Optional[str]]:
    if portfolio_id and "/" in portfolio_id:
        cid, rid = portfolio_id.split("/", 1)
        return cid or "client_001", rid
    return (portfolio_id or "client_001"), None


# --------------------------------------------------------------------------- #
# A. Temporal compare
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# A0. Governed composite answers: portfolio summary + period movement
#
# Both delegate to mi_agent_api.movement_summary, which composes the EXISTING
# evolution / bridge services — no new metric definitions. They exist so the
# React MI Agent and Microsoft 365 Copilot (which calls the same /mi/query
# handler) return one governed answer to the two questions a portfolio owner
# asks first, instead of each surface assembling its own narrative.
# --------------------------------------------------------------------------- #
_SUMMARY_MARKERS = (
    "summarise the portfolio", "summarize the portfolio", "portfolio summary",
    "summarise the book", "summarize the book", "summary of the portfolio",
    "overview of the portfolio", "overview of the book", "portfolio overview",
)

_MOVEMENT_MARKERS = (
    "what has changed", "what's changed", "whats changed", "what changed",
    "what has moved", "what moved", "how has the portfolio changed",
    "how has the book changed",
)

_PRIOR_PERIOD_MARKERS = (
    "prior month", "previous month", "last month", "prior period",
    "previous period", "last period", "prior reporting", "previous reporting",
    "month on month", "month-on-month",
)


#: Item 4 — the trigger half of the class. A VOCABULARY, and stated as one
#: rather than dressed up as a rule: it is the same kind of finite list as
#: `lexical.AXIS_MARKERS`, and the day someone writes "give me the top-line
#: picture" it will need a word added.
#:
#: What makes the class a RULE rather than these phrases is the second half
#: below — the question must name NOTHING ELSE. That is computed, and it is what
#: keeps "tell me about brokers", "tell me about arrears", "how is lending
#: doing" and "what is the CPR of this book" out. The CPR case matters most: an
#: honest "I cannot compute that" must not become a summary nobody asked for.
_SUMMARY_INTENT = (
    "summary", "summarise", "summarize", "overview", "snapshot",
    "the basics", "basics about", "headline", "key metrics", "key figures",
    "key numbers", "highlights", "top-line", "top line",
    "how is it doing", "how are we doing", "how is the book doing",
    "how is the portfolio doing", "how is the book looking",
    "where do we stand", "how do things stand", "how are things",
    "tell me about",
)

#: Spec markers that mean another governed capability owns this question. Read
#: rather than re-derived: each is set by the parser and consumed by its own
#: recogniser, so duplicating any of their vocabularies here would recreate the
#: multi-owner defect this programme has closed six times.
_SPECIALIST_SPEC_MARKS = ("risk_limit_query", "risk_monitor_mode", "forecast_mode",
                          "cohort_progression", "bridge_query", "temporal_mode")


def _names_something_else(question: str, spec=None) -> bool:
    """Whether the question asks for anything more specific than the book itself.

    A measure, a dimension, a filter, a comparison, or a specialist capability
    the spec already marks. This is the half of the class that is COMPUTED, and
    it does all the discriminating — the vocabulary above only decides whether a
    summary was asked for at all.
    """
    for mark in _SPECIALIST_SPEC_MARKS:
        if getattr(spec, mark, None):
            return True
    try:
        from mi_agent import llm_query_parser as _p
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.datasets import semantics_path
        from mi_workflows.analytical.intent import is_comparative as _comparative
        semantics = load_mi_semantics(semantics_path())
        if [k for _s, _e, k, _a in _p._measure_hits(question, semantics)
                if k != "loan_count"]:
            return True
        if _p._explicit_dimensions(question, semantics)[0]:
            return True
        if _p._parse_filters(question, semantics):
            return True
        if _comparative(question):
            return True
    except Exception:  # noqa: BLE001 - a summary is the fallback, not the default
        return True
    return False


def _is_portfolio_summary(question: str, spec=None) -> bool:
    """A whole-book summary request — the book's overall position, however worded.

    Item 4. This required one of nine LITERAL PHRASES, so "Tell me the basics
    about this book" was refused outright while "book overview", "key metrics"
    and "What are the headline numbers?" fell through to the generic executor
    and came back as a two-KPI card. Three different outcomes for one question,
    only one of them the governed summary.

    The generic path was never DECIDING those were summaries; it was failing to
    recognise them. Claiming them here is what makes the answer single.
    """
    q = f" {question.lower().strip()} "
    if not any(m in q for m in _SUMMARY_INTENT):
        return False
    # "summarise the portfolio by region" is a stratification, not a summary; and
    # "summarise what changed" is a movement question.
    if " by " in q or any(m in q for m in _MOVEMENT_MARKERS):
        return False
    if any(m in q for m in _PRIOR_PERIOD_MARKERS):
        return False
    return not _names_something_else(question, spec)


def _is_period_movement(question: str) -> bool:
    """A "what has changed versus the prior month" request.

    Deliberately narrow: an explicit change-marker AND an explicit prior-period
    reference. Anything else (a named two-period comparison, a single-metric
    trend) falls through to the existing compare / evolution routes unchanged.
    """
    q = f" {question.lower().strip()} "
    return (any(m in q for m in _MOVEMENT_MARKERS)
            and any(m in q for m in _PRIOR_PERIOD_MARKERS))


def _resolve_lens(question: str, source_lens) -> Any:
    """The portfolio lens for a routed intent, RESOLVED through the registry.

    The lens layer recognises the scope a caller named; the governed registry
    decides what that scope contains. Resolving here means a routed answer
    (evolution, bridge, cohort progression, forecast) narrows to the same
    explicit portfolio-id list the point-in-time path uses — so a group route
    picks up a newly onboarded member with no change here, and the scope a
    response DISCLOSES is always the scope it actually filtered on.
    """
    default_lens = (_portfolio_lens.lens_from_selection(source_lens)
                    if source_lens is not None else None)
    # PHASE 1E. The governed registry is what React renders its portfolio
    # selector from, so resolving against it is what makes a name the client can
    # SEE a name MI can READ. Best-effort: if the registry is unavailable this
    # falls back to exactly the pre-1E resolution rather than failing a route.
    #
    # Built ONCE and handed to `resolve_context` below, which would otherwise
    # build its own. The scope a route discloses and the scope its lens resolved
    # against are then the same object, not two registries that happen to agree.
    registry = None
    try:
        from . import portfolio_context as _ctx_registry

        registry = _ctx_registry.build_registry()
    except Exception as exc:  # noqa: BLE001 - identity must never break routing
        _logger.info("governed registry unavailable for lens resolution: %s", exc)
    lens = _portfolio_lens.resolve_lens_with_default(question, default_lens,
                                                     registry=registry)
    try:
        from . import portfolio_context as _ctx

        scope = _ctx.resolve_context(_portfolio_lens.context_id(lens),
                                     registry=registry,
                                     discover_pipeline=False).scope
        return _portfolio_lens.PortfolioLens(
            name=lens.name, label=lens.label, filters=dict(scope.filters),
            cohort_id=lens.cohort_id)
    except Exception as exc:  # noqa: BLE001 - routing must never break on scope
        _logger.info("routed lens scope resolution unavailable: %s", exc)
        return lens


class LensNotApplied(RuntimeError):
    """A lens named a scope, and this frame could not be narrowed to it.

    Raised instead of returning the frame unchanged. See `_apply_lens_filter`.
    """

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def _apply_lens_filter(df, lens, *, evidence_out=None) -> Any:
    """Narrow a dataframe to the portfolio ids a RESOLVED lens carries.

    This is the same narrowing the point-in-time executor performs — a group is
    exactly the sum of its current members. Total carries no filter and the
    frame is returned unchanged; a frame without provenance is likewise
    unchanged, because a book that records no source portfolio holds exactly
    one and IS the scope.

    IT FAILS CLOSED. The docstring here used to state a PRECONDITION —
    "``_resolve_lens`` returns the registry-resolved id list (never a type
    string)" — and nothing enforced it. When `period_change_route` stopped
    reading the sentence and began deriving its lens from the contract, that
    precondition quietly stopped holding: `contract_scope.lens_from_contract`
    returned ``{source_portfolio_type: "direct"}``, `ids` was empty, and this
    function returned the whole frame. Five snapshots went in at 520, 545, 570,
    600 and 640 rows and came out at 520, 545, 570, 600 and 640.

    "Summarise the month-on-month movement in the Direct book" then answered
    £22.6m — the whole book — for a book that moved £12.4m, and the receipt
    declared `portfolioScope: direct` beside it. The same question through
    `period_movement` answered £12.4m, and the two envelopes were identical in
    every field a consumer can read. A wrong figure vouched for by its own
    receipt is the outcome this estate ranks above every other, so the widening
    is now an exception and not a return value.

    ``evidence_out``, when given, receives the narrowing that was performed —
    the same "execution evidence only" rule `metadata.populationApplied`
    follows. A caller that narrows and stays silent is indistinguishable from
    one that did not narrow at all, which is precisely how the defect above
    survived being measured.
    """
    name = getattr(lens, "name", None) if lens is not None else None
    if df is None or lens is None or name in (None, _portfolio_lens.LENS_TOTAL):
        return df
    # THE IDS, WHEREVER THE LENS CARRIES THEM. A registry-resolved lens puts
    # them in `filters`; `_selection_lens` — "several books chosen explicitly,
    # exactly those, never their type" — puts them in `cohort_ids` and leaves
    # `filters` empty. Reading only `filters` treated an explicit multi-book
    # selection exactly as it treated the type lens that produced the wrong
    # movement figure: no ids, frame returned whole.
    ids = ((getattr(lens, "filters", None) or {}).get(_portfolio_lens.SOURCE_ID_FIELD)
           or getattr(lens, "cohort_ids", None)
           or getattr(lens, "cohort_id", None))
    rows_before = len(df)
    if _portfolio_lens.SOURCE_ID_FIELD not in getattr(df, "columns", []):
        # NO PROVENANCE COLUMN: the book records one source portfolio and is
        # already the scope. Recorded as applied, because it is.
        if evidence_out is not None:
            evidence_out.append({
                "context": name, "label": getattr(lens, "label", None),
                "rows_before": rows_before, "rows_after": rows_before,
                "detail": "this book records a single source portfolio"})
        return df
    if not ids:
        raise LensNotApplied(
            "the %s scope resolved to no portfolio id, and this book records "
            "provenance for more than one" % (getattr(lens, "label", None) or name))
    wanted = {str(i).strip().lower() for i in (ids if isinstance(ids, (list, tuple, set))
                                               else [ids])}
    col = df[_portfolio_lens.SOURCE_ID_FIELD].astype("string").str.strip().str.lower()
    out = df[col.isin(wanted)]
    if evidence_out is not None:
        evidence_out.append({
            "context": name, "label": getattr(lens, "label", None),
            "rows_before": rows_before, "rows_after": len(out),
            "detail": ", ".join(sorted(wanted))})
    return out


def _pct_points(value: Optional[float], decimals: int = 1) -> str:
    return "—" if value is None else f"{float(value):.{decimals}f}%"


def _years(value: Optional[float], decimals: int = 1) -> str:
    return "—" if value is None else f"{float(value):.{decimals}f} years"


def _count(value: Optional[float]) -> str:
    return "—" if value is None else f"{int(round(float(value))):,}"


def _date_label(iso: Optional[str]) -> str:
    """"30 June 2026" from an ISO date, falling back to the raw value."""
    if not iso:
        return "—"
    try:
        return datetime.strptime(str(iso)[:10], "%Y-%m-%d").strftime("%d %B %Y").lstrip("0")
    except Exception:  # noqa: BLE001 - never fail an answer on a date label
        return str(iso)


def _summary_kpi_artifact(title: str, kpis: List[Dict[str, str]], *, spec, portfolio_id,
                          as_of, description: str) -> Dict[str, Any]:
    return {
        "id": _uid(), "type": "kpi", "title": title, "description": description,
        "source": _source("MI Agent · summary", spec, portfolio_id, as_of),
        "createdAt": _now(), "mock": False, "kpis": kpis,
    }


def _summary_population(question, source_lens, interpretation, *, output_root,
                        client_id, run_id) -> Tuple[Dict[str, Any], str, bool]:
    """The governed headline position, and the scope it was computed over.

    CONVERSION 1 — the switch point, and the whole of it.

    The population is PLANNED from the interpretation contract: governed
    portfolio ids, the base population, and the provenance that decides
    precedence against a workspace selection. `mi_agent.portfolio_lens` is still
    the only thing that decides what "the acquired book" MEANS; the contract
    transports its answer, and this consumes the transported answer rather than
    asking the resolver a second time.

    `_resolve_lens` is deliberately NOT reachable from here. It remains the
    owner for the five other routes that call it, untouched.

    Returns ``(summary, scope label, narrowed)``. ``summary is None`` means the
    route defers. No lens object escapes, so no consumer downstream can read a
    second opinion off one.
    """
    if interpretation is None:
        # NO CONTRACT, NO ANSWER FROM THIS ROUTE. It defers, which is the
        # route's own pre-existing "I cannot answer this" behaviour and is what
        # every portfolio-summary question did for the whole of Phase 1G, when
        # the provider was raising and nobody noticed.
        #
        # The alternative — keeping the lens-resolved path here as a fallback —
        # would leave `_resolve_lens` in this route as a SECOND POPULATION
        # OWNER, reachable exactly when the first one failed, which is the worst
        # moment for two owners to disagree. One owner, or none.
        return None, "", False

    summary = _plan.portfolio_summary(
        output_root, client_id, interpretation=interpretation,
        to_run_id=run_id)
    scope = getattr(interpretation, "source_scope", None)
    label = summary.get("lens") or "Total"
    narrowed = bool(getattr(scope, "portfolio_ids", ()) or ())
    return summary, label, narrowed


def _route_portfolio_summary(question, spec, spec_dict, *, client_id, run_id,
                             output_root, portfolio_id, as_of, source_lens=None,
                             interpretation=None) -> Optional[Dict[str, Any]]:
    """The current reporting period's governed headline position."""
    summary, _scope_label, _narrowed = _summary_population(
        question, source_lens, interpretation, output_root=output_root,
        client_id=client_id, run_id=run_id)
    if summary is None:
        return None  # no contract: defer to the existing point-in-time path
    if not summary.get("available"):
        if _narrowed:
            # PHASE 1E. Deferring here hands a NARROWING question to a path that
            # cannot see the narrowing. Measured before this branch existed,
            # "Summarise the spv1_sponsored portfolio" — a governed portfolio
            # with no funded rows at this reporting date — came back with the
            # whole book's 11,035 loans and £1.96bn, with the scope it was asked
            # for mentioned nowhere. The scope was not unresolvable; it was
            # resolved, found empty, and then dropped.
            #
            # An empty governed scope is a FACT about the book, so it is stated
            # (the shape `geo_exposure` already uses for the same condition)
            # rather than replaced by a broader population.
            return _envelope(
                ok=True, question=question, spec=spec_dict, artifacts=[],
                answer=(f"There are no funded loans in {_scope_label} at the "
                        f"current governed reporting date, so there is no "
                        f"position to summarise for it. I have not answered "
                        f"for the whole book instead."),
                route="portfolio_summary", lens_applied=True,
                warnings=[f"no rows in scope for {_scope_label}: "
                          f"{summary.get('reason', 'scope is empty')}."])
        return None  # defer to the existing point-in-time summary path

    m = summary["metrics"]
    cut_off = _date_label(summary.get("reportingDate"))
    scope = "" if not _narrowed else f" ({_scope_label})"

    parts = [
        f"At {cut_off} the portfolio{scope} holds {_count(m['loan_count'])} loans "
        f"with a funded balance of {_gbp(m['funded_balance'])}."
    ]
    detail = []
    if m["wa_ltv_points"] is not None:
        detail.append(f"weighted-average current LTV is {_pct_points(m['wa_ltv_points'])}")
    if m["wa_interest_rate"] is not None:
        detail.append(f"the weighted-average interest rate is "
                      f"{_disp(m['wa_interest_rate'], 'wa_interest_rate', 2)}")
    if m["avg_borrower_age"] is not None:
        detail.append(f"the average youngest-borrower age is "
                      f"{_years(m['avg_borrower_age'])}")
    if detail:
        parts.append(_upper_first(_sentence_join(detail)) + ".")

    regions = summary.get("topRegions") or []
    if regions:
        # The answer names only the top few so it stays executive-length; the
        # supporting chart carries the fuller ranking.
        named = [f"{r['region']} ({_gbp(r['balance'])}, "
                 f"{_pct_points((r['share'] or 0) * 100)})"
                 for r in regions[:movement_mod.NAMED_REGIONS]]
        parts.append(f"The largest regional exposures are {_sentence_join(named)}.")

    cohorts = summary.get("cohortBalances") or {}
    labels = {c["id"]: c["label"] for c in (summary.get("cohorts") or [])}
    if len(cohorts) > 1 and not _narrowed:
        split = [f"{labels.get(cid, cid)} {_gbp(bal)}"
                 for cid, bal in sorted(cohorts.items(), key=lambda kv: -kv[1])]
        parts.append(f"By source portfolio: {_sentence_join(split)}.")

    answer = " ".join(parts)

    kpis = [
        {"label": "Loans funded", "value": _count(m["loan_count"])},
        {"label": "Funded balance", "value": _gbp(m["funded_balance"])},
        {"label": "WA current LTV", "value": _pct_points(m["wa_ltv_points"])},
        {"label": "WA interest rate",
         "value": _disp(m["wa_interest_rate"], "wa_interest_rate", 2)},
        {"label": "Avg borrower age", "value": _years(m["avg_borrower_age"])},
        {"label": "Data cut-off", "value": cut_off},
    ]
    artifacts: List[Dict[str, Any]] = [
        _summary_kpi_artifact(
            f"Portfolio summary — {cut_off}", kpis, spec=spec_dict,
            portfolio_id=portfolio_id, as_of=as_of,
            description=f"Governed position at the {summary.get('period')} "
                        f"reporting date ({_scope_label}).")
    ]
    if regions:
        artifacts.append(_chart_artifact(
            "Funded balance by region", chart_type="bar", x_key="region",
            rows=[{"region": r["region"], "value": r["balance"], "share": r["share"]}
                  for r in regions],
            series=[{"key": "value", "label": "Funded balance", "color": _PALETTE[0]}],
            value_format="gbp", spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
            display_hints={"value": {"format": "gbp", "scale": None}},
            description=f"Largest regional exposures at {cut_off}."))
    if len(cohorts) > 1:
        artifacts.append(_table_artifact(
            "Funded balance by source portfolio", columns=[
                {"key": "portfolio", "label": "Source portfolio", "align": "left",
                 "format": "text"},
                {"key": "balance", "label": "Funded balance", "align": "right",
                 "format": "gbp"},
            ],
            rows=[{"portfolio": labels.get(cid, cid), "balance": bal}
                  for cid, bal in sorted(cohorts.items(), key=lambda kv: -kv[1])],
            spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of))

    notes = [{"field": "reporting_period",
              "note": f"{summary.get('period')} ({summary.get('reportingDate')}); "
                      f"{summary.get('periodCount')} governed period(s) available."}]
    return _envelope(
        ok=True, question=question, answer=answer, spec=spec_dict,
        artifacts=artifacts,
        reconciliation=_workspace.reconciliation_for(
            _workspace.datasets_read(output_root=output_root),
            missing_dimension_policy="exclude"),
        source_notes=notes, route="portfolio_summary")


def _route_period_movement(question, spec, spec_dict, *, client_id, run_id,
                           output_root, portfolio_id, as_of, source_lens=None,
                           interpretation=None) -> Optional[Dict[str, Any]]:
    """Month-on-month movement across the governed metrics, with attribution.

    CONVERSION 2 — the switch point, and the whole of it.

    Both semantic inputs this route used to re-read from the question now come
    from the contract: the source scope (Phase 1G) and the STATED WINDOW's
    magnitude (target-state closure). Honouring the stated period still matters
    for the same reason it always did — a question that names "this year" and is
    answered over the latest month has had a declared element replaced, and
    disclosing the narrower window in the prose is not the same as honouring it
    — but the window is now read once, upstream, and carried.
    """
    if interpretation is None:
        # NO CONTRACT, NO ANSWER FROM THIS ROUTE. Same rule as Conversion 1:
        # one population owner, or none. Keeping the lens-resolved path as a
        # fallback would leave `_resolve_lens` reachable exactly when the
        # contract failed.
        return None

    mv = _plan.period_movement(output_root, client_id,
                               interpretation=interpretation, to_run_id=run_id)
    span = _period_request.span_from_claim(interpretation.time)
    if not mv.get("available"):
        if span is not None and mv.get("spanRequested"):
            message = _period_request.clarification(
                span, int(mv.get("periodsAvailable") or 0))
            return _envelope(
                ok=False, question=question, spec=spec_dict, artifacts=[],
                answer=message, error=message, route="period_movement",
                warnings=[message])
        return _undeliverable(
            question=question, spec=spec_dict, answer=f"I can't compare against the prior month yet: "
                   f"{mv.get('reason', 'insufficient reporting periods')}.",
            route="period_movement",
            warnings=["insufficient-data: a month-on-month comparison needs two "
                      "governed reporting periods."])

    cur, pri, dl = mv["current"], mv["prior"], mv["delta"]
    cur_label = _date_label(mv.get("currentReportingDate"))
    move = dl["funded_balance"]
    direction = movement_mod.balance_descriptor(move)

    # Sentence 1 — the movement, and its primary regional contributor. The
    # completion wording is used only when the completions evidence supports it.
    lead = f"Funded balances {direction} by {_gbp(abs(move))} during the month"
    primary = mv.get("primaryRegion")
    if primary and mv.get("primaryRegionIsDominant"):
        if mv.get("completionsEvidenced"):
            lead += f", primarily driven by completions in the {primary['region']}"
        else:
            lead += (f", with the largest contribution from the {primary['region']} "
                     f"({'+' if primary['delta'] >= 0 else '−'}"
                     f"{_gbp(abs(primary['delta']))})")
    elif primary:
        lead += (f". The largest single regional contribution came from the "
                 f"{primary['region']} ({'+' if primary['delta'] >= 0 else '−'}"
                 f"{_gbp(abs(primary['delta']))})")
    parts = [lead.rstrip(".") + "."]

    # Sentence 2 — LTV and borrower age, with materiality-aware wording.
    ltv_txt = movement_mod.ltv_descriptor(dl["wa_ltv_points"])
    age_txt = movement_mod.age_descriptor(dl["avg_borrower_age"])
    parts.append(
        f"Weighted-average LTV {ltv_txt} {_pct_points(cur['wa_ltv_points'])}, "
        f"while average borrower age {age_txt} {_years(cur['avg_borrower_age'])}.")

    # Sentence 3 — the source-portfolio contribution.
    cohorts = mv.get("cohortMovements") or []
    if len(cohorts) > 1:
        contrib = [f"{c['label']} contributed "
                   f"{'approximately ' if abs(c['delta']) >= 100_000 else ''}"
                   f"{'+' if c['delta'] >= 0 else '−'}{_gbp(abs(c['delta']))}"
                   for c in cohorts]
        parts.append(_upper_first(_sentence_join(contrib)) + " of the movement.")

    answer = " ".join(parts)

    kpis = [
        {"label": "Funded balance movement",
         "value": f"{'+' if (move or 0) >= 0 else '−'}{_gbp(abs(move))}"},
        {"label": f"Funded balance ({cur_label})", "value": _gbp(cur["funded_balance"])},
        {"label": "Loans funded",
         "value": f"{_count(cur['loan_count'])} "
                  f"({'+' if (dl['loan_count'] or 0) >= 0 else '−'}"
                  f"{_count(abs(dl['loan_count'] or 0))})"},
        {"label": "WA current LTV",
         "value": f"{_pct_points(cur['wa_ltv_points'])} "
                  f"({'+' if (dl['wa_ltv_points'] or 0) >= 0 else '−'}"
                  f"{_pct_points(abs(dl['wa_ltv_points'] or 0), 2)})"},
        {"label": "Avg borrower age",
         "value": f"{_years(cur['avg_borrower_age'])} "
                  f"({'+' if (dl['avg_borrower_age'] or 0) >= 0 else '−'}"
                  f"{abs(dl['avg_borrower_age'] or 0):.2f})"},
    ]
    artifacts: List[Dict[str, Any]] = [
        _summary_kpi_artifact(
            f"{mv['priorPeriod']} → {mv['currentPeriod']} movement", kpis,
            spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
            description=f"Governed month-on-month movement ({mv['lens']}).")
    ]

    contributions = mv.get("regionContributions") or []
    if contributions:
        top = contributions[:8]
        artifacts.append(_chart_artifact(
            f"Balance movement by region — {mv['priorPeriod']} → {mv['currentPeriod']}",
            chart_type="bar", x_key="region",
            rows=[{"region": c["region"], "value": c["delta"]} for c in top],
            series=[{"key": "value", "label": "Movement", "color": _PALETTE[0]}],
            value_format="gbp", spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
            display_hints={"value": {"format": "gbp", "scale": None}},
            description="Regional contributions sum exactly to the consolidated "
                        "movement."))
        artifacts.append(_table_artifact(
            "Regional contribution to the movement", columns=[
                {"key": "region", "label": "Region", "align": "left", "format": "text"},
                {"key": "start", "label": mv["priorPeriod"], "align": "right",
                 "format": "gbp"},
                {"key": "end", "label": mv["currentPeriod"], "align": "right",
                 "format": "gbp"},
                {"key": "delta", "label": "Δ", "align": "right", "format": "gbp"},
            ],
            rows=[{"region": c["region"], "start": c["start"], "end": c["end"],
                   "delta": c["delta"]} for c in top],
            spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of))
    if len(cohorts) > 1:
        artifacts.append(_table_artifact(
            "Source-portfolio contribution to the movement", columns=[
                {"key": "portfolio", "label": "Source portfolio", "align": "left",
                 "format": "text"},
                {"key": "prior", "label": mv["priorPeriod"], "align": "right",
                 "format": "gbp"},
                {"key": "current", "label": mv["currentPeriod"], "align": "right",
                 "format": "gbp"},
                {"key": "delta", "label": "Δ", "align": "right", "format": "gbp"},
            ],
            rows=[{"portfolio": c["label"], "prior": c["prior"],
                   "current": c["current"], "delta": c["delta"]} for c in cohorts],
            spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of))

    recon = mv.get("reconciliation") or {}
    notes = [
        {"field": "reporting_periods",
         "note": f"{mv['priorPeriod']} ({mv.get('priorReportingDate')}) → "
                 f"{mv['currentPeriod']} ({mv.get('currentReportingDate')})."},
        {"field": "attribution",
         "note": f"Regional contributions sum to "
                 f"{_gbp(recon.get('sum_of_region_movements'))} against a "
                 f"consolidated movement of "
                 f"{_gbp(recon.get('consolidated_movement'))} "
                 f"(residual {_gbp(recon.get('region_residual'))})."},
    ]
    if mv.get("completionsBalanceInMonth") is not None:
        notes.append({
            "field": "completions_in_month",
            "note": f"Loans completed in {mv['currentPeriod']} carry "
                    f"{_gbp(mv['completionsBalanceInMonth'])} of balance"
                    + (f", of which {_gbp(mv.get('completionsBalanceInPrimaryRegion'))} "
                       f"is in the {primary['region']}." if primary else "."),
        })
    out = _envelope(
        ok=True, question=question, answer=answer, spec=spec_dict,
        artifacts=artifacts,
        reconciliation=_workspace.reconciliation_for(
            _workspace.datasets_read(output_root=output_root),
            missing_dimension_policy="exclude"),
        source_notes=notes, route="period_movement")
    return _declare_scope(out, mv.get("scopeApplied"), label=mv.get("lens"))


def _upper_first(text: str) -> str:
    """Capitalise the first character only — never lowercase the remainder,
    which would mangle acronyms and proper nouns (LTV, ALP Origination Book)."""
    return text[:1].upper() + text[1:] if text else text


def _sentence_join(items: Sequence[str]) -> str:
    """"a, b and c" — UK English list punctuation."""
    vals = [str(i) for i in items if i]
    if not vals:
        return ""
    if len(vals) == 1:
        return vals[0]
    return f"{', '.join(vals[:-1])} and {vals[-1]}"


def _route_compare(question, spec_dict, *, client_id, run_id, output_root,
                   pipeline_root, portfolio_id, as_of, interpretation
                   ) -> Dict[str, Any]:
    """THE PARSE IS NO LONGER A PARAMETER.

    `spec` is gone from this signature, and that absence is the conversion's
    real result: there is nothing left for this route to read from it. The
    period pair, the measure and the dataset all arrive on the contract, and a
    route that cannot reach the parse cannot quietly re-decide any of them.

    `spec_dict` stays. It is echoed into the envelope for the receipt layer and
    is not consulted for any semantic fact.
    """
    # CONVERSION 5. Composed. Every semantic fact this route used to read from
    # the question now arrives on the contract: the dataset (the ownership
    # remediation made `workspace.resolve_dataset` the single owner and the
    # contract carries its answer), the measure (`subject`), and the period pair
    # (carried structurally by the time contract). The plan states all three and
    # the deterministic executor runs them.
    #
    # The two-period guard moves INTO the plan, which blocks rather than
    # defaults; the refusal below is unchanged in wording so the envelope, the
    # receipt and the prose are identical for a question naming one period.
    out = _plan.temporal_compare(output_root, pipeline_root, client_id, run_id,
                                 interpretation=interpretation)
    if out.get("planBlocked"):
        return _envelope(ok=False, question=question,
                         answer="I need two periods to compare.", spec=spec_dict,
                         artifacts=[], route="temporal_compare", error="missing periods")
    periods = list(_plan.comparison_periods(interpretation))
    dataset = out.get("dataset")
    metric_key = out.get("metric", "funded_balance")
    label = out.get("metricLabel", metric_key)

    if not out.get("available"):
        avail = out.get("availablePeriods") or []
        answer = (f"I can't compare {periods[0]} and {periods[1]} for {label.lower()}: "
                  f"{out.get('reason', 'a period is unavailable')}.")
        if len(avail) <= 1:
            answer += " Only one reporting period is available."
        return _undeliverable(question=question, answer=answer, spec=spec_dict,
                         route="temporal_compare",
                         warnings=["insufficient-data: cross-period comparison needs two periods."])

    va, vb = out["valueA"], out["valueB"]
    delta, pct = out["absoluteDelta"], out["percentageDelta"]
    direction = out["direction"]
    arrow = "up" if direction == "up" else ("down" if direction == "down" else "flat")
    answer = (f"{label} moved from {_disp(va, metric_key)} in {out['periodA']} to "
              f"{_disp(vb, metric_key)} in {out['periodB']} — a change of "
              f"{_disp(abs(delta), metric_key)} "
              f"({'+' if delta >= 0 else ''}{pct if pct is not None else '—'}%, {arrow}).")

    fmt = _METRIC_DISPLAY.get(metric_key, ("decimal", "decimal", None))
    chart = _chart_artifact(
        f"{label}: {out['periodA']} vs {out['periodB']}", chart_type="bar",
        x_key="period",
        rows=[{"period": out["periodA"], "value": va},
              {"period": out["periodB"], "value": vb}],
        series=[{"key": "value", "label": label, "color": _PALETTE[0]}],
        value_format=fmt[1], spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
        display_hints={"value": {"format": fmt[1], "scale": fmt[2]}})
    table = _table_artifact(
        f"{label} comparison", columns=[
            {"key": "metric", "label": "Metric", "align": "left", "format": "text"},
            {"key": "period_a", "label": out["periodA"], "align": "right", "format": fmt[1], "scale": fmt[2]},
            {"key": "period_b", "label": out["periodB"], "align": "right", "format": fmt[1], "scale": fmt[2]},
            {"key": "abs_delta", "label": "Δ absolute", "align": "right", "format": fmt[1], "scale": fmt[2]},
            {"key": "pct_delta", "label": "Δ %", "align": "right", "format": "pct"},
        ],
        rows=[{"metric": label, "period_a": va, "period_b": vb,
               "abs_delta": delta, "pct_delta": pct}],
        spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)

    recon = {"dataset": dataset, "coverage_by_balance_pct": 100.0,
             "missing_dimension_policy": "exclude"}
    notes = [{"field": "source_periods",
              "note": f"Period A: {out.get('sourcePeriods', [None, None])[0]}; "
                      f"Period B: {out.get('sourcePeriods', [None, None])[1]}"}]
    return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                     artifacts=[chart, table], reconciliation=recon, source_notes=notes,
                     route="temporal_compare")


# --------------------------------------------------------------------------- #
# B. Evolution / trend
# --------------------------------------------------------------------------- #
def _declare_scope(envelope: Dict[str, Any], applied,
                   *, context: Optional[str] = None,
                   label: Optional[str] = None) -> Dict[str, Any]:
    """Record the source-portfolio scope this answer was NARROWED to.

    The same execution-evidence rule `populationApplied` and `_declare_grain`
    follow: the route says what it did, and a route that says nothing is read
    as having covered everything.

    Saying nothing was the whole defect. `portfolioScope` publishes the scope a
    request RESOLVED, and both a correct Direct answer and a whole-book answer
    mislabelled Direct published `context_id: "direct"` beside figures of
    £12.4m and £22.6m. Nothing else in either envelope differed. This is the
    field that differs.
    """
    if not applied:
        return envelope
    envelope.setdefault("metadata", {})["scopeApplied"] = {
        "context": context or applied.get("context"),
        "label": label or applied.get("label"),
        "detail": applied.get("detail"),
        "rowsBefore": applied.get("rowsBefore"),
        "rowsAfter": applied.get("rowsAfter"),
        "snapshots": applied.get("snapshots"),
    }
    return envelope


def _declare_grain(envelope: Dict[str, Any], grain: str) -> Dict[str, Any]:
    """Record the reporting grain this answer was actually published at.

    The receipt owns a SECOND claim about grain — a static route -> grain map
    asserting that every series route reports months, on the premise that they
    all read the month-end funded snapshots. Three do not: the funnel and the
    by-stage series have always been keyed on the weekly extract date, and the
    single-metric series is weekly whenever the pipeline producer supplied it.

    While the two claims agreed by accident nobody noticed. Once the series was
    keyed correctly they disagreed, in both directions at once: a question asking
    for weeks was refused on the ground that the answer was monthly, and one
    asking for months was answered weekly with nothing disclosed.

    So the route says what it did, and the receipt reads that rather than an
    assertion made about the route elsewhere — the same execution-evidence rule
    `populationApplied` and `declared_series_periods` already follow. A route
    that declares nothing keeps the static fallback and is unaffected.
    """
    envelope.setdefault("metadata", {})["seriesGrain"] = grain
    return envelope


def _funded_metric_value(df, metric_key: str) -> Optional[float]:
    """A single funded metric for one period frame — the SAME computation
    assemble_funded_evolution uses per period, so a filtered series reconciles to
    the unfiltered one when the filter is a no-op."""
    if metric_key == "funded_balance":
        return evolution_mod._bal_sum(df)
    if metric_key == "loan_count":
        return int(len(df))
    # Fractions, matching assemble_funded_evolution exactly — this helper's
    # contract is that a filtered series reconciles to the unfiltered one.
    if metric_key == "wa_ltv":
        return evolution_mod._pct_fraction(df, "current_loan_to_value")
    if metric_key == "wa_interest_rate":
        return evolution_mod._pct_fraction(df, "current_interest_rate")
    if metric_key == "avg_borrower_age":
        return evolution_mod._simple_avg(df, "youngest_borrower_age")
    return None


def _filtered_funded_evo(output_root, client_id, run_id, predicates, semantics,
                         metric_key: str) -> Dict[str, Any]:
    """A funded single-metric series with the POPULATION applied within each period.

    ``predicates`` are the governed `Predicate` objects the compositional plan
    selected — `SELECT_POPULATION(kind=row_predicates)`, built from
    `RowPredicateClaim` and nothing else. This function does not receive the
    spec, so it cannot read a filter's meaning even by accident.

    Execution goes through `population.apply_population`, which since the
    predicate-parity work runs every predicate through
    `mi_query_executor.governed_predicate_mask` — the same single owner
    `_apply_filters` uses. That equivalence is what makes this substitution
    row-for-row identical rather than merely similar, and it is measured at
    119/119 by `migration_phase0.predicate_execution_parity`.

    Raises on a predicate that cannot be applied, so the caller defers to the
    controlled point-in-time validation path exactly as before.
    """
    frames = evolution_mod.funded_frames(output_root, client_id, run_id)
    periods: List[Dict[str, Any]] = []
    sources: List[str] = []
    for fr in frames:
        df = fr.get("df")
        if df is None:
            continue
        fdf, evidence = _population.apply_population(df, predicates, semantics)
        if fdf is None or not evidence.is_usable:
            # FAIL CLOSED, and in the shape this caller already handles: a
            # population that could not be applied is not a series with a
            # missing filter, it is no series at all.
            raise MIQueryExecutionError(
                "the requested population could not be applied to this book: "
                + "; ".join(evidence.unavailable or [evidence.blocked_reason or ""]))
        rd = fr.get("reporting_date") or fr.get("run_id")
        periods.append({
            "period": (str(rd)[:7] if rd else fr.get("run_id")),
            "reporting_date": rd,
            "metrics": {metric_key: _funded_metric_value(fdf, metric_key)},
            "filteredRows": int(len(fdf)),
        })
        if fr.get("source"):
            sources.append(str(fr["source"]))
    return {"periods": periods, "sourceFiles": sources}


def _filter_summary(predicates) -> str:
    """A short human description of the applied population, for answer and notes.

    Described from the governed `Predicate` rather than from the spec, through
    `Predicate.describe()` — the same wording the population ledger and the
    facet labels already use, so the prose, the evidence and the receipt cannot
    say three different things about one narrowing.
    """
    return "; ".join(p.describe() for p in (predicates or ()))


def _route_evolution(question, spec, spec_dict, *, client_id, run_id, output_root,
                     pipeline_root, portfolio_id, as_of, semantics=None,
                     interpretation=None) -> Optional[Dict[str, Any]]:
    # THE DATASET, THE STAGE AND THE STAGE AXIS, ALL FROM THE CONTRACT.
    #
    # This route used to re-read the raw question three times for facts the
    # interpretation layer had already settled: `resolve_dataset(question)` for
    # the dataset, a five-substring `_FUNNEL_KEYWORDS` map for the stage, and
    # `"by stage" in q` for the stage axis. Each was a second owner of a
    # governed decision, and the substring readers were narrower than the
    # governed vocabulary they shadowed — 21 spellings against 5.
    dataset = _plan.evolution_dataset(interpretation)
    if dataset is None:
        # No contract, no plan. Deferring is the fail-safe: the point-in-time
        # path validates and refuses, rather than this route guessing a dataset.
        return None
    funnel_stage, stage_axis = _plan.governed_stage(interpretation)
    is_count = spec.aggregation == "count"

    # NO IMPLICIT MEASURE. A series has to plot something, so the parser
    # supplies the governed balance when the question names no measure — and
    # for "show me the trend" that substitution WAS the answer: a funded
    # balance series, chosen entirely by us, presented as though it had been
    # asked for. The frozen acceptance bank has expected this to be refused
    # since it was written.
    #
    # The test is the BARE case and nothing wider, read from the contract
    # alone. A question that supplies any governed element the measure can be
    # determined FROM is not bare and is untouched: an explicit dataset
    # ("show pipeline evolution by stage" — the governed pipeline amount), a
    # named analytic ("show regional concentration evolution over time"), or a
    # dimension to break the series down by. Only a question that supplies
    # none of them leaves the measure with no owner but us.
    from question_interpretation.schema import PROV_DEFAULT as _PROV_DEFAULT

    _subject = getattr(interpretation, "subject", None)
    if (getattr(_subject, "provenance", None) == _PROV_DEFAULT
            and not is_count
            and not stage_axis
            # A NAMED STAGE DETERMINES THE MEASURE just as the dataset does:
            # "completions by month" is a pipeline question whose measure is
            # the governed pipeline amount, not a question with no measure.
            # Checking only the stage AXIS ("by stage") and not a named stage
            # made this guard refuse it.
            and not funnel_stage
            and getattr(getattr(interpretation, "operation", None),
                        "analytic", None) is None
            and not _plan.grouping_concepts(interpretation)
            and getattr(getattr(interpretation, "dataset", None),
                        "provenance", None) == _PROV_DEFAULT):
        message = (
            "I can show a trend, but you have not said which metric. "
            "For example: funded balance, loan count or weighted-average LTV. "
            "No metric has been chosen for you.")
        return _undeliverable(question=question, spec=spec_dict, artifacts=[],
                              answer=message, error=message, route="evolution",
                              warnings=[message])

    # THE POPULATION, PLANNED FROM THE CONTRACT. `spec.filters` still answers
    # "did the reader ask to narrow at all" — a gate, not a meaning — and every
    # question of WHICH rows is answered by the plan step.
    population_step = _plan.row_predicate_step(interpretation)
    predicates = _plan.row_predicates(population_step)
    requested = dict(getattr(spec, "filters", None) or {})
    filtered = bool(predicates)

    # A narrowing the plan does not carry must never be silently dropped. The
    # keys the contract excludes by design are SCOPE and reporting-basis keys,
    # which are not row predicates at all; if any remain, this route cannot
    # honour the whole request and defers to the point-in-time path, which is
    # the same fail-safe an invalid filter already takes. Measured on the
    # corpus: 121 spec.filters entries, 121 material predicates, 0 excluded —
    # so this defers on nothing that ships today.
    if requested and len(predicates) != len(requested):
        return None

    # Filtered trends are supported for the FUNDED single-metric series only
    # (applied per period below). A filtered pipeline / funnel / stage trend defers
    # to the point-in-time path, which handles filters within one snapshot.
    if filtered and dataset != "funded":
        return None

    # Funnel stage trend, for whatever stage the governed vocabulary resolved.
    if funnel_stage:
        funnel = evolution_mod.pipeline_funnel_evolution(pipeline_root, client_id, run_id)
        pts = funnel.get("series", {}).get(funnel_stage, [])
        summ = funnel.get("summary", {}).get(funnel_stage, {})
        if not pts:
            return _undeliverable(question=question,
                             answer=f"No weekly {funnel_stage.title()} extracts are available yet.",
                             spec=spec_dict, route="evolution_funnel",
                             warnings=["insufficient-data: no weekly pipeline extracts."])
        flow_pts = funnel.get("flowSeries", {}).get(funnel_stage, [])
        # Weekly-flow rows (bars); fall back to the stock level only when no flow
        # series is present. Each row carries the stock level too (cumulative line).
        rows = [{"week": p.get("week"),
                 "value": p.get("flowValue"),
                 "count": p.get("flowCount"),
                 "stock": s.get("value")}
                for p, s in zip(flow_pts, pts)] or \
            [{"week": p.get("week"), "value": p.get("value"), "count": p.get("count")} for p in pts]
        chart = _chart_artifact(
            f"{summ.get('label', funnel_stage.title())} weekly flow", chart_type="line",
            x_key="week", rows=rows,
            series=[{"key": "value", "label": "Weekly flow (£)", "color": _PALETTE[0]}],
            value_format="gbp", spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)
        table = _table_artifact(
            f"{summ.get('label', funnel_stage.title())} weekly flow trend", columns=[
                {"key": "week", "label": "Week", "align": "left", "format": "date"},
                {"key": "value", "label": "Weekly flow (£)", "align": "right", "format": "gbp"},
                {"key": "count", "label": "Weekly flow (count)", "align": "right", "format": "number"},
                {"key": "stock", "label": "Stock level (£)", "align": "right", "format": "gbp"},
            ], rows=rows, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)
        answer = (f"Latest week {summ.get('label', funnel_stage.title())} weekly flow: "
                  f"{_gbp(summ.get('latestFlowValue'))}; "
                  f"5-week average weekly flow {_gbp(summ.get('fiveWeekAvgFlowValue'))} "
                  f"({summ.get('trend', 'flat')} vs prior week). Current stock level "
                  f"{_gbp(summ.get('latestStockValue'))}.")
        notes = [{"field": "weekly_extracts",
                  "note": f"{funnel.get('uniqueWeeklyExtractsUsed') or len(rows)} governed weekly extract(s)."}]
        out = _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                        artifacts=[chart, table],
                        reconciliation={"dataset": "pipeline", "coverage_by_balance_pct": 100.0},
                        source_notes=notes, route="evolution_funnel")
        return _declare_grain(out, "week")

    # Pipeline amount by stage over time (multi-series).
    if dataset == "pipeline" and stage_axis:
        pipe = evolution_mod.pipeline_evolution(pipeline_root, client_id, run_id)
        by_stage = pipe.get("byStage", [])
        if not by_stage:
            return _undeliverable(question=question,
                             answer="No weekly pipeline extracts are available to build a stage trend.",
                             spec=spec_dict, route="evolution_pipeline_stage",
                             warnings=["insufficient-data: no weekly pipeline extracts."])
        periods = sorted({r["period"] for r in by_stage})
        stages = sorted({r["stage"] for r in by_stage})
        rows = []
        for per in periods:
            row: Dict[str, Any] = {"period": per}
            for st in stages:
                row[st] = sum(r["value"] for r in by_stage if r["period"] == per and r["stage"] == st)
            rows.append(row)
        chart = _chart_artifact(
            "Pipeline amount by stage over time", chart_type="line", x_key="period",
            rows=rows, series=[{"key": st, "label": st, "color": _PALETTE[i % len(_PALETTE)]}
                               for i, st in enumerate(stages)],
            value_format="gbp", spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)
        answer = (f"Pipeline amount by stage across {len(periods)} period(s): "
                  f"stages {', '.join(stages)}.")
        out = _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                        artifacts=[chart],
                        reconciliation={"dataset": "pipeline", "coverage_by_balance_pct": 100.0},
                        route="evolution_pipeline_stage")
        # DECLARE THE AXIS IT ACTUALLY SPLIT BY. This branch builds one series
        # per governed stage, so the answer IS broken down by `pipeline_stage` —
        # and until the word "stage" resolved to that governed dimension,
        # nothing raised a grouping facet for it and nothing had to say so.
        # Now that it does, an undeclared axis reads as a request the route
        # dropped, and the question was refused over a chart that had honoured
        # it. Declared from the EXECUTED series, exactly as the bridge route
        # declares `dimensionCol`.
        from question_interpretation.lexical import PIPELINE_STAGE_FIELD

        out.setdefault("metadata", {})["groupedBy"] = [PIPELINE_STAGE_FIELD]
        return _declare_grain(out, "week")

    # Funded / pipeline single-metric evolution.
    metric_key, label, fmt = compare_mod.resolve_metric_key(dataset, spec.metric, spec.aggregation)
    if dataset == "pipeline":
        evo = evolution_mod.pipeline_evolution(pipeline_root, client_id, run_id)
    elif filtered:
        # Filtered funded series: apply the filter within each period. On an invalid
        # filter, defer to the controlled point-in-time validation path.
        try:
            evo = _filtered_funded_evo(output_root, client_id, run_id, predicates,
                                       semantics or {}, metric_key)
        except Exception:  # noqa: BLE001 - invalid filter -> point-in-time path
            return None
    else:
        evo = evolution_mod.funded_evolution(output_root, client_id, run_id)
    periods = evo.get("periods", [])
    if not periods:
        return _undeliverable(question=question,
                         answer=f"No reporting periods are available to build a {label.lower()} trend.",
                         spec=spec_dict, route="evolution",
                         warnings=["insufficient-data: no governed reporting periods."])
    # The observation identity is whatever the series publishes as its own grain.
    # The pipeline producer publishes a day-level `week` per governed weekly
    # extract; the funded producers publish only a monthly `period`. Keying every
    # series on `period` collapsed five distinct weekly extracts onto one x-axis
    # point, under a chart this route already titled "by week" — the label and the
    # data disagreed, and the label was right. Read from what the producer returns
    # rather than from the dataset name, so a series is keyed by the grain it
    # actually carries.
    period_field = "week" if any("week" in p for p in periods) else "period"
    rows = [{"period": p.get(period_field), "value": (p.get("metrics") or {}).get(metric_key)}
            for p in periods]
    filter_txt = _filter_summary(predicates) if filtered else ""
    scope_suffix = f" — {filter_txt}" if filter_txt else ""
    disp = _METRIC_DISPLAY.get(metric_key, ("decimal", "decimal", None))
    chart = _chart_artifact(
        f"{label} by {'week' if dataset == 'pipeline' else 'month'}{scope_suffix}", chart_type="line",
        x_key="period", rows=rows,
        series=[{"key": "value", "label": label, "color": _PALETTE[0]}],
        value_format=disp[1], spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
        display_hints={"value": {"format": disp[1], "scale": disp[2]}})
    table = _table_artifact(
        f"{label} trend{scope_suffix}", columns=[
            {"key": "period", "label": "Period", "align": "left", "format": "text"},
            {"key": "value", "label": label, "align": "right", "format": disp[1], "scale": disp[2]},
        ], rows=rows, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)

    vals = [r["value"] for r in rows if r["value"] is not None]
    warnings: List[str] = []
    if len(vals) <= 1:
        warnings.append("Only one reporting period is available — showing the single point; "
                        "an evolution view reads best with two or more periods.")
    trend_txt = ""
    if len(vals) >= 2:
        d = vals[-1] - vals[0]
        trend_txt = f" ({'up' if d > 0 else 'down' if d < 0 else 'flat'} over the window)"
    scope_answer = f" (scoped to {filter_txt})" if filter_txt else ""
    answer = (f"{label} over {len(rows)} period(s){scope_answer}: latest "
              f"{_disp(vals[-1] if vals else None, metric_key)}{trend_txt}.")
    src_files = evo.get("sourceFiles") or []
    notes = [{"field": "source_periods",
              "note": f"{len(periods)} governed period(s); source: "
                      f"{src_files[-1] if src_files else 'governed runs'}"}]
    if filtered:
        kept = [p.get("filteredRows") for p in periods if p.get("filteredRows") is not None]
        notes.append({"field": "filter",
                      "note": (f"Filter applied within each period: {filter_txt}. "
                               f"Rows per period after filter: "
                               f"{', '.join(str(k) for k in kept) or 'n/a'}.")})
    last_recon = (periods[-1].get("reconciliation") if periods else None) or {
        "dataset": dataset, "coverage_by_balance_pct": 100.0}
    out = _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                    artifacts=[chart, table], reconciliation=last_recon,
                    source_notes=notes, warnings=warnings, route="evolution")
    _declare_grain(out, "week" if period_field == "week" else "month")
    if filtered:
        # P1L: this route genuinely applies the population — per period, which is
        # what makes a filtered trend meaningful — so it DECLARES that it did.
        # The population ledger accepts execution evidence only, and a route that
        # stays silent is treated as having widened. Silence from the one route
        # that was always correct would have refused it.
        last = next((p.get("filteredRows") for p in reversed(periods)
                     if p.get("filteredRows") is not None), None)
        before = next((p.get("rows") or p.get("totalRows") for p in reversed(periods)
                       if (p.get("rows") or p.get("totalRows")) is not None), None)
        out.setdefault("metadata", {})["populationApplied"] = {
            "applied": [f"{p.field} (applied within each period)"
                        for p in predicates],
            "unavailable": [], "rowsBefore": before, "rowsAfter": last,
        }
    return out


# --------------------------------------------------------------------------- #
# C. Forecast scale-up / extrapolation
# --------------------------------------------------------------------------- #
def _route_forecast(question, spec, spec_dict, *, client_id, run_id, output_root,
                    pipeline_root, history_model, portfolio_id, as_of) -> Dict[str, Any]:
    # THE TARGET THE READER NAMED goes to the projector, so the milestone it
    # answers from is the one asked about rather than the nearest round number
    # the fixed ladder happens to carry.
    _target = spec.forecast_target_value
    fx = fx_mod.build_extrapolation(output_root, pipeline_root, client_id, run_id,
                                    history_model=history_model,
                                    extra_thresholds=([_target] if _target else ()))
    rr = fx.get("completionRunRateForecast", {})
    kfi = fx.get("kfiConversionForecast", {})
    weighted = fx.get("currentWeightedPipelineForecast", {})
    cur = fx.get("currentFundedBalance", 0.0)
    kind = spec.forecast_question or "extrapolation_curve"
    target = _target
    caveat = "Downside/base/upside are indicative scenario bands, not statistically validated confidence intervals."
    warnings = [caveat]

    if not rr.get("available"):
        answer = (f"Current funded balance is {_gbp(cur)}. I can't extrapolate a completion "
                  f"run-rate yet: {rr.get('caveat', 'insufficient completion history')}.")
        wp = weighted.get("weightedExpectedPipeline")
        if wp is not None:
            answer += (f" The current weighted pipeline forecast adds {_gbp(wp)} "
                       f"(→ {_gbp(weighted.get('forecastFundedBalance'))}).")
        return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                         artifacts=[], route="forecast_extrapolation",
                         warnings=["insufficient-data: not enough completion history for a run-rate forecast."])

    base = rr.get("baseMonthlyRunRate")
    ann = rr.get("annualisedRunRate")
    scenarios = rr.get("scenarioMonthlyRunRate", {})
    milestones = rr.get("milestones", [])

    def _ms(thr: float) -> Optional[Dict[str, Any]]:
        """The milestone FOR THIS THRESHOLD, or the next one above it.

        NEVER `milestones[-1]`. That fallback returned the largest milestone the
        projection happened to carry whenever the requested threshold was beyond
        it, and the caller then reported that milestone's state as the answer for
        the threshold actually asked about. Measured: the milestone list tops out
        at £75m — all reached — so "when do we reach £250m?" answered "the book
        has already reached £250.0m" on a book holding £172.1m.
        """
        exact = next((m for m in milestones if m["threshold"] == thr), None)
        if exact:
            return exact
        above = [m for m in milestones if m["threshold"] >= thr]
        return above[0] if above else None

    if kind in ("reach_threshold",) and target:
        m = _ms(target)
        # THE ARITHMETIC DECIDES, not a milestone flag. "Already reached" is a
        # statement about the CURRENT balance and the REQUESTED target, and it is
        # true exactly when one is at least the other.
        if float(cur or 0) >= float(target):
            answer = f"The book has already reached {_gbp(target)} (current funded balance {_gbp(cur)})."
        elif m and m.get("reached"):
            answer = (f"Current funded balance is {_gbp(cur)}; {_gbp(target)} is "
                      f"beyond the projection horizon, so I cannot say when it is "
                      f"reached. {caveat}")
        elif m:
            answer = (f"At the current base completion run-rate (~{_gbp(base)}/month, "
                      f"{_gbp(ann)}/year), the book reaches {_gbp(target)} around "
                      f"{m.get('baseDate')} (downside {m.get('downsideDate')}, "
                      f"upside {m.get('upsideDate')}). {caveat}")
        else:
            answer = f"Current funded balance is {_gbp(cur)}; {_gbp(target)} is beyond the projection horizon."
    elif kind == "pipeline_needed" and target:
        gap = max(float(target) - float(cur), 0.0)
        answer = (f"To reach {_gbp(target)} from the current {_gbp(cur)} you need ~{_gbp(gap)} "
                  f"of additional completions — about {gap / base:.0f} month(s) at the base "
                  f"run-rate of {_gbp(base)}/month." if base else
                  f"To reach {_gbp(target)} you need ~{_gbp(gap)} of additional completions.")
    elif kind == "run_rate_annualised":
        answer = (f"The annualised completion run-rate is ~{_gbp(ann)} "
                  f"(base monthly {_gbp(base)} over {rr.get('observedMonths')} observed month(s)).")
    elif kind == "run_rate":
        answer = (f"The current completion run-rate is ~{_gbp(base)}/month "
                  f"({_gbp(ann)}/year) based on {rr.get('observedMonths')} month(s) of funded growth.")
    elif kind in ("scenario_downside", "scenario_upside", "scenario"):
        which = "downside" if "down" in kind else ("upside" if "up" in kind else "downside")
        answer = (f"{which.title()} scenario monthly run-rate is ~{_gbp(scenarios.get(which))} "
                  f"vs a base of {_gbp(scenarios.get('base'))}. {caveat}")
    elif kind == "conversion":
        if kfi.get("available"):
            # This is the COHORT completion rate from the historical weekly-snapshot
            # model — the share of KFI cases (tracked across snapshots) that have
            # since completed — NOT the Evolution funnel's observed same-week
            # KFI→Completion ratio. It reads lower because recent KFI cases have not
            # yet had time to complete (right-censored), so it is a floor.
            rate = kfi.get("conversionRate", 0) * 100
            lag = kfi.get("lagMonths")
            answer = (
                f"About {rate:.1f}% of KFI cases have since completed, tracked cohort-style "
                f"across the weekly snapshots, with a ~{lag}-month median KFI→completion lag. "
                f"This is a floor — recent KFIs haven't had time to complete yet — and is measured "
                f"differently from the Evolution funnel's observed same-week KFI→Completion ratio, "
                f"which reads higher.")
        else:
            answer = ("A KFI→completion conversion rate can't be derived from the current history; "
                      f"using the completion run-rate (~{_gbp(base)}/month) instead.")
    elif kind == "compare_models":
        wp = weighted.get("weightedExpectedPipeline")
        answer = (f"Current weighted pipeline forecast adds {_gbp(wp)} (point-in-time → "
                  f"{_gbp(weighted.get('forecastFundedBalance'))}); the completion run-rate "
                  f"extrapolation projects ~{_gbp(base)}/month ({_gbp(ann)}/year) forward. {caveat}")
    else:
        # HONOUR THE STATED GRANULARITY, OR CLARIFY. "Based on the last few
        # weeks" pins no COUNT, so there is no span to fail — but it does pin a
        # UNIT, and this run-rate is measured from month-end funded snapshots.
        # Disclosing "based on 2 month(s) of funded growth" told the reader what
        # was used; it did not answer the question they asked, which is the same
        # shape as answering "this year" over the latest month with a note.
        #
        # The weekly pipeline extracts cannot stand in: ten of the twelve carry
        # no completion at all, so a weekly completion rate would rest on two
        # observations in the final fortnight — the censoring artefact Tranche C
        # documents, not a rate.
        unit = _period_request.requested_unit(question)
        if _period_request.finer_than(unit, "month"):
            message = _period_request.granularity_clarification(
                unit, "month", "month-end funded snapshots")
            return _envelope(
                ok=False, question=question, spec=spec_dict, artifacts=[],
                answer=message, error=message, route="forecast_extrapolation",
                warnings=[message])
        # State the OBSERVATION WINDOW on the answers that do stand. The sibling
        # milestone answer above has always disclosed it; this one did not.
        observed = rr.get("observedMonths")
        basis = (f", based on {observed} month(s) of funded growth"
                 if observed else "")
        answer = (f"Current funded balance {_gbp(cur)}; base completion run-rate ~{_gbp(base)}/month "
                  f"({_gbp(ann)}/year){basis}, projected forward with "
                  f"downside/base/upside bands. {caveat}")

    artifacts: List[Dict[str, Any]] = []
    proj = rr.get("projectedBalances", [])
    if proj:
        rows = [{"month": p["month"], "downside": p["downside"], "base": p["base"], "upside": p["upside"]}
                for p in proj]
        artifacts.append(_chart_artifact(
            "Projected funded balance — downside / base / upside", chart_type="line",
            x_key="month", rows=rows, series=[
                {"key": "downside", "label": "Downside", "color": _PALETTE[3]},
                {"key": "base", "label": "Base", "color": _PALETTE[0]},
                {"key": "upside", "label": "Upside", "color": _PALETTE[1]},
            ], value_format="gbp", spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of))
    if milestones:
        mrows = [{"threshold": m["thresholdLabel"],
                  "downside": "reached" if m.get("reached") else (m.get("downsideDate") or "—"),
                  "base": "reached" if m.get("reached") else (m.get("baseDate") or "—"),
                  "upside": "reached" if m.get("reached") else (m.get("upsideDate") or "—")}
                 for m in milestones]
        artifacts.append(_table_artifact(
            "Milestone dates to funding thresholds", columns=[
                {"key": "threshold", "label": "Threshold", "align": "left", "format": "text"},
                {"key": "downside", "label": "Downside", "align": "right", "format": "text"},
                {"key": "base", "label": "Base", "align": "right", "format": "text"},
                {"key": "upside", "label": "Upside", "align": "right", "format": "text"},
            ], rows=mrows, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of))

    notes = [{"field": "assumptions",
              "note": f"Base run-rate {_gbp(base)}/mo over {rr.get('observedMonths')} month(s); "
                      f"signal = month-on-month funded growth. {caveat}"}]
    recon = {"dataset": "forecast", "coverage_by_balance_pct": 100.0}
    return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                     artifacts=artifacts, reconciliation=recon, source_notes=notes,
                     warnings=warnings, route="forecast_extrapolation")


# --------------------------------------------------------------------------- #
# C2. Scenario / what-if — perturb the completion run-rate, re-solve the milestone
# --------------------------------------------------------------------------- #
_SCENARIO_CONDITIONALS = ("if ", "were to", "suppose", "what if", "assuming",
                          "hypothetical", "scenario")
_SCENARIO_CHANGES = ("increase", "increas", "rise", "rose", "grow", "grew", "improv",
                     "higher", "up by", "boost", "double", "doubl", "fall", "fell",
                     "drop", "lower", "declin", "reduc", "cut", "halve", "halv", "down by")
_SCENARIO_LEVERS = ("conversion", "convert", "run rate", "run-rate", "completion")


def _is_scenario(question: str) -> bool:
    q = f" {question.lower()} "
    return (any(c in q for c in _SCENARIO_CONDITIONALS)
            and any(c in q for c in _SCENARIO_CHANGES)
            and any(l in q for l in _SCENARIO_LEVERS))


def _scenario_multiplier(question: str) -> Optional[Tuple[float, str]]:
    """(run-rate multiplier, human change label) from a what-if phrasing, or None
    when the magnitude can't be quantified (caller then defers)."""
    q = question.lower()
    if "double" in q or "doubl" in q:
        return 2.0, "doubled"
    if "halve" in q or "halv" in q:
        return 0.5, "halved"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|percentage points?|pts?|pp)\b", q)
    if not m:
        m = re.search(r"\bby\s+(\d+(?:\.\d+)?)\b", q)
    if not m:
        return None
    pct = float(m.group(1))
    down = any(w in q for w in ("decreas", "fall", "fell", "drop", "lower",
                                "declin", "reduc", "cut", "down"))
    signed = -pct if down else pct
    return scenario_mod.multiplier_from_conversion_delta(signed), f"{'-' if down else '+'}{pct:g}%"


def _scenario_target(spec, question: str) -> Optional[float]:
    tv = getattr(spec, "forecast_target_value", None)
    if tv:
        return float(tv)
    m = re.search(r"£?\s*(\d+(?:\.\d+)?)\s*(bn|billion|mm|m|million)\b", question.lower())
    if not m:
        return None
    val = float(m.group(1))
    return val * (1e9 if m.group(2) in ("bn", "billion") else 1e6)


def _route_scenario(question, spec, spec_dict, *, client_id, run_id, output_root,
                    pipeline_root, history_model, portfolio_id, as_of
                    ) -> Optional[Dict[str, Any]]:
    """A deterministic what-if: perturb the completion run-rate (a conversion
    change maps proportionally) and re-solve the milestone date to a target. The
    math lives in the pure ``scenario`` engine; here we resolve the base from the
    forecast and shape the answer. Returns None (defer) when the change magnitude
    can't be quantified."""
    parsed = _scenario_multiplier(question)
    if parsed is None:
        return None
    mult, change_txt = parsed
    target = _scenario_target(spec, question)

    fx = fx_mod.build_extrapolation(output_root, pipeline_root, client_id, run_id,
                                    history_model=history_model)
    rr = fx.get("completionRunRateForecast", {})
    cur = fx.get("currentFundedBalance", 0.0)
    if not rr.get("available"):
        return _undeliverable(question=question, spec=spec_dict, answer=("I can't run a what-if on the completion run-rate yet: "
                                 f"{rr.get('caveat', 'insufficient completion history')}."),
                         route="scenario",
                         warnings=["insufficient-data: no run-rate to perturb."])
    base = rr.get("baseMonthlyRunRate")
    proj = rr.get("projectedBalances") or []
    period = proj[0]["month"] if proj else (as_of or "2025-01")[:7]
    res = scenario_mod.apply_scenario(
        current_balance=cur, base_monthly_run_rate=base, reporting_period=period,
        run_rate_multiplier=mult, target_value=target)

    caveat = ("What-if assumption: a conversion change maps proportionally to the completion "
              "run-rate (KFI inflow held constant); dates share the base forecast's basis and "
              "carry the same downside/base/upside uncertainty.")
    rows = [{"month": p["month"], "base": p["base"], "scenario": p["scenario"]}
            for p in res["projectedSeries"]]
    chart = _chart_artifact(
        f"Projected funded balance — base vs scenario ({change_txt} run-rate)",
        chart_type="line", x_key="month", rows=rows,
        series=[{"key": "base", "label": "Base", "color": _PALETTE[0]},
                {"key": "scenario", "label": f"Scenario ({change_txt})", "color": _PALETTE[1]}],
        value_format="gbp", spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)
    artifacts: List[Dict[str, Any]] = [chart]

    if target is not None:
        trows = [
            {"metric": "Monthly run-rate", "base": _gbp(res["baseMonthlyRunRate"]),
             "scenario": _gbp(res["scenarioMonthlyRunRate"])},
            {"metric": f"Date to {_gbp(target)}",
             "base": res["baseTargetDate"] or "beyond horizon",
             "scenario": res["scenarioTargetDate"] or "beyond horizon"},
        ]
        artifacts.append(_table_artifact(
            "Base vs scenario", columns=[
                {"key": "metric", "label": "", "align": "left", "format": "text"},
                {"key": "base", "label": "Base", "align": "right", "format": "text"},
                {"key": "scenario", "label": f"Scenario ({change_txt})", "align": "right", "format": "text"},
            ], rows=trows, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of))

    if target is not None and float(cur or 0) >= float(target):
        answer = f"The book has already reached {_gbp(target)} (current funded balance {_gbp(cur)})."
    elif target is not None and res["baseTargetDate"] == "reached":
        # The projection says "reached" for a threshold the book has NOT reached:
        # the same substituted-milestone defect as above, seen from the scenario
        # side. Say what is true rather than repeat the flag.
        answer = (f"Current funded balance is {_gbp(cur)}; {_gbp(target)} is beyond "
                  f"the projection horizon, so a scenario cannot be dated to it. {caveat}")
    elif target is not None and res["monthsSaved"] is None:
        answer = (f"Even with a {change_txt} completion run-rate (~{_gbp(res['scenarioMonthlyRunRate'])}/mo) "
                  f"the book doesn't reach {_gbp(target)} within the projection horizon. {caveat}")
    elif target is not None:
        saved = res["monthsSaved"]
        faster = "sooner" if saved > 0 else ("later" if saved < 0 else "unchanged")
        answer = (f"A {change_txt} completion run-rate moves the {_gbp(target)} milestone from "
                  f"{res['baseTargetDate']} (base ~{_gbp(base)}/mo) to {res['scenarioTargetDate']} "
                  f"(~{_gbp(res['scenarioMonthlyRunRate'])}/mo) — about {abs(saved)} month(s) {faster}. {caveat}")
    else:
        answer = (f"A {change_txt} completion run-rate lifts the monthly run-rate from {_gbp(base)} "
                  f"to {_gbp(res['scenarioMonthlyRunRate'])} (annualised {_gbp(res['scenarioMonthlyRunRate'] * 12)}). "
                  f"{caveat}")
    notes = [{"field": "scenario", "note": caveat}]
    return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                     artifacts=artifacts,
                     reconciliation={"dataset": "forecast", "coverage_by_balance_pct": 100.0},
                     source_notes=notes, warnings=[caveat], route="scenario")


# --------------------------------------------------------------------------- #
# D. Risk limits / concentration
# --------------------------------------------------------------------------- #
_CONCENTRATION_TEST_COLUMNS = [
    {"key": "test", "label": "Test", "align": "left"},
    {"key": "category", "label": "Category", "align": "left"},
    {"key": "current", "label": "Current", "align": "right"},
    {"key": "limit", "label": "Limit", "align": "right"},
    {"key": "headroom", "label": "Headroom", "align": "right"},
    {"key": "change", "label": "Period change", "align": "right"},
    {"key": "status", "label": "Status", "align": "left"},
]


def _route_concentration_tests(question, spec, spec_dict, *, client_id,
                               run_id, output_root, portfolio_id, as_of
                               ) -> Optional[Dict[str, Any]]:
    """Answer a limit question from the approved concentration-test service.

    Returns None when no approved configuration exists (the legacy Schedule 8
    route then answers, unchanged). Every number comes from the ONE evaluation
    service — nothing here recalculates from raw rows, invents a threshold or
    reports an unavailable value as zero.
    """
    from . import concentration_query as conc_query
    from . import concentration_tests_api as conc_mod

    envelope = conc_mod.compute_concentration_tests(output_root, client_id,
                                                    run_id)
    if envelope.get("source") != conc_mod.SOURCE_APPROVED:
        return None
    result = conc_query.answer(question, envelope)

    artifacts: List[Dict[str, Any]] = []
    if result["intent"] == "get_concentration_pipeline_drivers" and \
            result.get("testId"):
        drivers = conc_mod.compute_pipeline_drivers(output_root, client_id,
                                                    run_id, result["testId"])
        answer_text = result["answer"]
        if not drivers.get("available"):
            answer_text += (" Pipeline-driver detail is not available: "
                            f"{drivers.get('reason')}")
        else:
            top = drivers["drivers"][:4]
            share = drivers.get("topShareOfMovement")
            lead = top[0] if top else None
            answer_text += (
                f" {len(top)} pipeline loan(s) drive "
                + (f"{share * 100:.0f}% of " if share is not None else "")
                + "the expected increase"
                + (f", led by {lead['caseId']} at £{lead['balance']:,.0f} "
                   f"currently at {str(lead.get('stage') or '').title()}"
                   if lead else "") + ".")
            artifacts.append(_table_artifact(
                f"Pipeline drivers — {result['rows'][0].get('displayName', '')}",
                columns=[
                    {"key": "caseId", "label": "Case", "align": "left"},
                    {"key": "balance", "label": "Balance", "align": "right"},
                    {"key": "stage", "label": "Stage", "align": "left"},
                    {"key": "completionProbability", "label": "Probability",
                     "align": "right"},
                    {"key": "expectedContribution", "label": "Expected contrib.",
                     "align": "right"},
                    {"key": "expectedCompletionMonth", "label": "Exp. month",
                     "align": "left"},
                    {"key": "impact", "label": "Impact", "align": "left"},
                ],
                rows=drivers["drivers"], spec=spec_dict,
                portfolio_id=portfolio_id, as_of=as_of,
                description="Forecast-engine probabilities and contributions; "
                            "reconciles to the expected numerator."))
        result = {**result, "answer": answer_text}
    elif result["intent"] == "get_concentration_test_drillthrough" and \
            result.get("testId"):
        drill = conc_mod.compute_drillthrough(output_root, client_id, run_id,
                                              result["testId"])
        test = result["rows"][0] if result.get("rows") else {}
        if not drill.get("available"):
            answer_text = (f"The contributing loans for "
                           f"{test.get('displayName', 'that test')} are not "
                           f"available: {drill.get('reason')}")
        else:
            answer_text = (
                f"{drill['loansInNumerator']} loan(s) contribute to "
                f"{test.get('displayName')} — numerator "
                f"{drill['numeratorValue']:,.0f} of denominator "
                f"{drill['denominatorValue']:,.0f} "
                f"({drill['denominatorBasis'].replace('_', ' ')}). "
                "The population below is the exact evaluated numerator.")
            artifacts.append(_table_artifact(
                f"Contributing loans — {test.get('displayName')}",
                columns=[{"key": c, "label": c.replace("_", " ").title(),
                          "align": "left"} for c in drill["columns"]],
                rows=drill["rows"], spec=spec_dict,
                portfolio_id=portfolio_id, as_of=as_of,
                description="Drill-through reuses the evaluator's own filter "
                            "and denominator."))
        result = {**result, "answer": answer_text}
    elif result.get("rows"):
        def _cell(v, unit):
            if v is None:
                return "—"
            if unit == "percent":
                return f"{v:.2f}%"
            if unit == "count":
                return f"{v:.0f}"
            return f"{v:,.0f}"
        trows = [{
            "test": t["displayName"], "category": t["category"],
            "current": _cell(t.get("currentValue"), t.get("unit")),
            "limit": ("≤ " if t.get("operator") == "max" else "≥ ")
                     + _cell(t.get("threshold"), t.get("unit")),
            "headroom": _cell(t.get("headroom"), t.get("unit")),
            "change": ("—" if t.get("absoluteChange") is None
                       else f"{t['absoluteChange']:+.2f}"),
            "status": t["status"],
        } for t in result["rows"]]
        artifacts.append(_table_artifact(
            "Concentration tests (approved configuration "
            f"v{envelope.get('configurationVersion')})",
            columns=_CONCENTRATION_TEST_COLUMNS, rows=trows, spec=spec_dict,
            portfolio_id=portfolio_id, as_of=as_of,
            description="Approved contractual limits vs the funded book, from "
                        "the governed evaluation service."))

    warnings = list(result.get("warnings") or [])
    envelope_out = _envelope(
        ok=True, question=question, answer=result["answer"], spec=spec_dict,
        artifacts=artifacts, route="risk_limits", warnings=warnings)
    envelope_out["metadata"]["concentrationIntent"] = result["intent"]
    envelope_out["metadata"]["configurationVersion"] = \
        envelope.get("configurationVersion")
    envelope_out["sourceNotes"].append({
        "label": "Concentration tests",
        "note": (f"Approved configuration v{envelope.get('configurationVersion')} "
                 f"(activated {str(envelope.get('activatedAt') or '')[:10]} by "
                 f"{envelope.get('activatedBy') or 'operator'}), evaluated at "
                 f"{envelope.get('reportingDate') or 'the latest snapshot'}."),
    })
    return envelope_out


def _route_risk(question, spec, spec_dict, *, client_id, run_id, output_root,
                portfolio_id, as_of) -> Dict[str, Any]:
    # The operator-APPROVED concentration-test configuration is the governed
    # truth for limit questions; the Schedule 8 extracted monitor remains the
    # calculator only while no approved configuration exists.
    approved = _route_concentration_tests(
        question, spec, spec_dict, client_id=client_id, run_id=run_id,
        output_root=output_root, portfolio_id=portfolio_id, as_of=as_of)
    if approved is not None:
        return approved
    rl = risk_mod.compute_risk_limits(output_root, client_id, run_id)
    summ = rl.get("summary", {})
    tests = rl.get("tests", [])
    category = getattr(spec, "risk_limit_category", None)

    if not rl.get("available"):
        answer = (f"Contractual risk limits are unavailable for this portfolio "
                  f"({rl.get('limitsReason', 'extraction required')}). "
                  "I can show observed concentrations once limits are provided.")
        return _undeliverable(question=question, answer=answer, spec=spec_dict,
                         route="risk_limits",
                         warnings=["limits unavailable / needs review."])

    # Scope to a single category when asked ("geographic concentration limits").
    cat_label = ""
    if category:
        scoped = [t for t in tests if t.get("category") == category]
        if scoped:
            tests = scoped
            summ = risk_mod._summary(tests)
            cat_label = category.replace("_", " ") + ": "
        else:
            return _undeliverable(
                question=question, spec=spec_dict, route="risk_limits",
                answer=(f"No {category.replace('_', ' ')} limits are configured for this "
                        "portfolio."),
                warnings=[f"no tests in category '{category}'."])

    closest = summ.get("closestHeadroom")
    largest = summ.get("largestConcentration")
    answer = (f"{cat_label}{summ.get('testsPassed', 0)} passed, {summ.get('warnings', 0)} warning(s), "
              f"{summ.get('breaches', 0)} breach(es), {summ.get('needsReview', 0)} need review, "
              f"{summ.get('unavailable', 0)} unavailable.")
    if closest:
        answer += f" Nearest to limit: {closest['label']} ({closest['headroom']:.1f} pp headroom)."
    if largest:
        answer += f" Largest concentration: {largest['label']} at {largest['actualValue']:.1f}%."

    # RISK artifact (RAG groups for percent-unit, computable tests).
    groups = []
    for t in tests:
        if (t.get("status") in ("green", "amber", "red") and t.get("actualValue") is not None
                and t.get("limitValue") and t.get("unit") == "percent"):
            groups.append({
                "name": t["label"], "balance": t["actualValue"],
                "share": float(t["actualValue"]) / 100.0,
                "status": t["status"], "limit": float(t["limitValue"]) / 100.0,
                "approaching": t["status"] == "amber",
            })
    artifacts: List[Dict[str, Any]] = []
    if groups:
        artifacts.append({
            "id": _uid(), "type": "risk",
            "title": "Concentration vs Schedule 8 limits",
            "description": "Funded exposure against extracted concentration limits.",
            "source": {**_source("Risk monitor · concentration", spec_dict, portfolio_id, as_of,
                                 engine="risk_monitor"), "state": "total_funded"},
            "createdAt": _now(), "mock": False,
            "mode": "limits", "dimension": "concentration", "groups": groups,
            "warnings": ([f"{summ.get('breaches')} limit(s) breached."] if summ.get("breaches") else []),
        })

    # TABLE artifact (ALL tests incl needs_review / unavailable).
    def _f(v, unit):
        if v is None:
            return "—"
        return f"{v:.1f}%" if unit == "percent" else (f"{int(v)}" if unit == "count" else _gbp(v))
    trows = [{
        "test": t["label"],
        "actual": _f(t.get("actualValue"), t.get("unit")),
        "limit": _f(t.get("limitValue"), t.get("unit")),
        "headroom": ("—" if t.get("headroom") is None else f"{t['headroom']:.1f}"),
        "status": t["status"], "movement": ("—" if t.get("movementVsPrior") is None
                                            else f"{t['movementVsPrior']:+.1f}"),
        "source": t.get("source", ""),
    } for t in tests]
    artifacts.append(_table_artifact(
        "Risk limit tests", columns=[
            {"key": "test", "label": "Test", "align": "left", "format": "text"},
            {"key": "actual", "label": "Actual", "align": "right", "format": "text"},
            {"key": "limit", "label": "Limit", "align": "right", "format": "text"},
            {"key": "headroom", "label": "Headroom", "align": "right", "format": "text"},
            {"key": "status", "label": "Status", "align": "left", "format": "text"},
            {"key": "movement", "label": "Movement", "align": "right", "format": "text"},
            {"key": "source", "label": "Source", "align": "left", "format": "text"},
        ], rows=trows, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of))

    notes = [{"field": "limit_source", "note": rl.get("limitsSource", "Schedule 8 extracted")}]
    recon = _workspace.reconciliation_for(
        _workspace.datasets_read(output_root=output_root),
        reporting_date=rl.get("reportingDate"))
    warnings = []
    if summ.get("unavailable"):
        warnings.append(f"{summ['unavailable']} test(s) unavailable (missing fields).")
    if summ.get("needsReview"):
        warnings.append(f"{summ['needsReview']} limit(s) need manual review.")
    envelope = _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                         artifacts=artifacts, reconciliation=recon, source_notes=notes,
                         warnings=warnings, route="risk_limits")
    # D7 (B12): the route states which book dimensions its EXECUTED tests are
    # written against. Without it the receipt certified "broken down by region"
    # — and "broken down by account status" — over a table whose columns are
    # test/actual/limit/headroom, because an axis existed that the reader could
    # not identify. Derived from the tests that actually computed, so a limit
    # reported unavailable certifies nothing.
    fields_tested = risk_mod.tested_fields(tests)
    if fields_tested:
        envelope.setdefault("metadata", {})["groupedBy"] = fields_tested
    return envelope


# --------------------------------------------------------------------------- #
# E. Funded balance bridge (attribution waterfall between two periods)
# --------------------------------------------------------------------------- #
# Preferred attribution dimension when the question names none.
_BRIDGE_DEFAULT_DIMS = ("geographic_region_obligor", "collateral_geography",
                        "broker_channel", "erm_product_type")


# The region family — any of these columns may carry the geography depending on
# the tape; the bridge resolves whichever is actually present.
_REGION_FAMILY = ("collateral_geography", "geographic_region_collateral",
                  "geographic_region_obligor")


def _bridge_dimension(concept: Optional[str],
                      semantics: Dict[str, Any]) -> Tuple[Optional[str], Any, str]:
    """(semantic_key, candidate_column(s), business_label) for the bridge
    attribution dimension — the governed CONCEPT the caller resolved, else a
    sensible default. Region resolves to the whole family so the bridge picks
    whichever geography column the funded tape actually carries.

    CONVERSION 4 changed this from reading `spec.bridge_dimension` itself to
    taking the concept as an argument, so the DECISION of which dimension is the
    axis moved to the interpretation contract while this kept its real job:
    turning a governed concept into the column(s) and label this tape spells it
    with. One owner of registry resolution, and it no longer owns the semantics.
    """
    fields = semantics.get("fields", {})
    key = concept
    if not key or key not in fields:
        key = next((k for k in _BRIDGE_DEFAULT_DIMS if k in fields), None)
    if not key:
        return None, None, ""
    entry = fields.get(key, {}) or {}
    label = entry.get("business_name") or entry.get("display_name") or key.replace("_", " ")
    if key in _REGION_FAMILY:
        cols = [fields.get(k, {}).get("canonical_field", k)
                for k in _REGION_FAMILY if k in fields]
        return key, (cols or [entry.get("canonical_field", key)]), label
    return key, entry.get("canonical_field", key), label


def _route_bridge(question, spec, spec_dict, *, client_id, run_id, output_root,
                  portfolio_id, as_of, semantics, source_lens=None,
                  interpretation=None) -> Optional[Dict[str, Any]]:
    """Governed funded-balance ATTRIBUTION bridge → a waterfall artifact.

    Opening balance (a named start period, else the earliest) → per-category
    change over the chosen dimension → the LATEST balance. A source-portfolio
    lens named in the question (or the active dropdown) scopes it — so a
    consolidated (Total) and cohort (direct / acquired / cohort id) bridge are
    both available. Deltas reconcile exactly to the net change."""
    if interpretation is None:
        # NO CONTRACT, NO ANSWER FROM THIS ROUTE. The rule Conversions 1-3
        # settled: one population owner, or none. Keeping the lens-resolved path
        # as a fallback would leave `resolve_lens_with_default` reachable from
        # here exactly when the contract failed.
        return None
    # CONVERSION 4 — the switch point, and the whole of it.
    #
    # Every semantic fact this route read from the question now arrives on the
    # contract: the source scope (Conversion 1), the attribution dimension (the
    # `dimensions` axis, bridged here) and the named start period
    # (`time.comparison_period`). `_bridge_dimension` keeps only the registry
    # resolution — concept to column and label.
    dim_key, dim_col, dim_label = _bridge_dimension(
        (_plan.grouping_concepts(interpretation) or (None,))[0], semantics)
    if not dim_col:
        return _undeliverable(question=question, spec=spec_dict, answer="I couldn't resolve a dimension to attribute the bridge by.",
                         route="funded_bridge", warnings=["no attribution dimension resolved."])

    br = _plan.funded_bridge(
        output_root, client_id, interpretation=interpretation,
        dimension_columns=dim_col, dimension_key=dim_key,
        dimension_label=dim_label, to_run_id=run_id)
    lens_label_text, lens_narrowed = br.get("lens") or "Total", None

    if not br.get("available"):
        return _undeliverable(question=question, spec=spec_dict, answer=(f"I can't build a funded balance bridge yet: "
                                 f"{br.get('reason', 'insufficient reporting periods')}."),
                         route="funded_bridge",
                         warnings=["insufficient-data: a bridge needs two funded reporting periods."])

    start, end = br["start"], br["end"]
    net = br["netChange"]
    arrow = "up" if net > 0 else ("down" if net < 0 else "flat")
    rows = [{"label": start["period"], "value": start["total"], "type": "total"}]
    for c in br["contributions"]:
        rows.append({"label": c["category"], "value": c["delta"], "type": "delta"})
    rows.append({"label": f"{end['period']} (latest)", "value": end["total"], "type": "total"})

    lens_suffix = "" if lens_label_text == "Total" else f" — {lens_label_text}"
    title = f"Funded balance bridge by {dim_label}{lens_suffix}"
    chart = _chart_artifact(
        title, chart_type="waterfall", x_key="label", rows=rows,
        series=[{"key": "value", "label": dim_label, "color": _PALETTE[0]}],
        value_format="gbp", spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
        display_hints={"value": {"format": "gbp", "scale": None}},
        description=(f"Opening {start['period']} → {dim_label.lower()} contributions "
                     f"→ latest {end['period']}."))

    top = max(br["contributions"], key=lambda c: abs(c["delta"]), default=None)
    top_txt = ""
    if top:
        td = top["delta"]
        top_txt = (f" Largest mover: {top['category']} "
                   f"({'+' if td >= 0 else '−'}{_gbp(abs(td))}).")
    answer = (f"{dim_label} bridge ({lens_label_text}): funded balance moved from "
              f"{_gbp(start['total'])} in {start['period']} to {_gbp(end['total'])} at "
              f"{end['period']} (latest) — a net change of "
              f"{'+' if net >= 0 else '−'}{_gbp(abs(net))} ({arrow}).{top_txt}")

    table = _table_artifact(
        f"{dim_label} contribution to balance change", columns=[
            {"key": "category", "label": dim_label, "align": "left", "format": "text"},
            {"key": "start", "label": start["period"], "align": "right", "format": "gbp"},
            {"key": "end", "label": end["period"], "align": "right", "format": "gbp"},
            {"key": "delta", "label": "Δ", "align": "right", "format": "gbp"},
        ],
        rows=[{"category": c["category"], "start": c["start"], "end": c["end"],
               "delta": c["delta"]} for c in br["contributions"]],
        spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)

    recon = _workspace.reconciliation_for(
        _workspace.datasets_read(output_root=output_root),
        reporting_date=end.get("reporting_date"))
    notes = [{"field": "bridge_periods",
              "note": f"Opening {start.get('reporting_date') or start['period']}; "
                      f"closing {end.get('reporting_date') or end['period']} (latest); "
                      f"attributed by {dim_label.lower()}; deltas reconcile to the net change."}]
    envelope = _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                         artifacts=[chart, table], reconciliation=recon,
                         source_notes=notes, route="funded_bridge")
    # D7: the route states the axis its bridge ACTUALLY attributed movement by,
    # so `grouping_proven` can certify a requested grouping from execution
    # evidence rather than from route identity. Without it this route declared
    # nothing, `declared_group_fields` returned no registry field, and every
    # question naming a dimension was refused with the grouping marked LOST —
    # over a waterfall that had attributed by exactly that dimension.
    #
    # `dimensionCol` is what `evolution.funded_bridge` REPORTS IT GROUPED BY —
    # the candidate it found present in the data, not the list it was offered
    # and not the dimension the question asked for. Declaring the executed axis
    # rather than the requested one is the safety property: a question naming a
    # dimension the bridge could not use leaves its request correctly unproven
    # and correctly refused. Reached only on a successful bridge, so an
    # unavailable result certifies nothing — the same rule `risk_limits` follows
    # for the tests that actually computed.
    executed_dim = br.get("dimensionCol")
    if executed_dim:
        envelope.setdefault("metadata", {})["groupedBy"] = [str(executed_dim)]
    return envelope


# --------------------------------------------------------------------------- #
# F. Cohort static-pool progression (a cohort's metrics across periods)
# --------------------------------------------------------------------------- #
# (question keyword) -> (metric key, label, chart valueFormat, display scale)
_PROG_METRICS: Dict[str, Tuple[str, str, str, Optional[str]]] = {
    "balance": ("funded_balance", "Funded balance", "gbp", None),
    "ltv": ("wa_ltv", "WA LTV", "pct", "percent_fraction"),
    "rate": ("wa_interest_rate", "WA interest rate", "pct", "percent_fraction"),
    "nneg": ("nneg_headroom_pct", "NNEG headroom", "pct", "percent_fraction"),
    "nneg_exposure": ("nneg_exposure", "NNEG exposure", "gbp", None),
    "count": ("loan_count", "Loan count", "number", None),
    "age": ("avg_borrower_age", "Avg borrower age", "decimal", None),
}


def _prog_metric_key(q: str) -> str:
    if "negative equity" in q or "nneg" in q or "no-negative" in q or "headroom" in q:
        return "nneg_exposure" if "exposure" in q else "nneg"
    if "ltv" in q or "loan to value" in q:
        return "ltv"
    if "rate" in q or "interest" in q or "coupon" in q:
        return "rate"
    if "how many" in q or "loan count" in q or "number of loans" in q:
        return "count"
    if "borrower age" in q or "age" in q:
        return "age"
    return "balance"


def _route_cohort_progression(question, spec, spec_dict, *, client_id, run_id,
                              output_root, portfolio_id, as_of, source_lens=None
                              ) -> Dict[str, Any]:
    """Governed static-pool cohort progression → a metric line across reporting
    periods for a cohort (source portfolio ± origination vintage) + a full
    metrics table."""
    default_lens = (_portfolio_lens.lens_from_selection(source_lens)
                    if source_lens is not None else None)
    lens = _portfolio_lens.resolve_lens_with_default(question, default_lens)
    vintage = getattr(spec, "cohort_vintage", None)
    grain = getattr(spec, "cohort_grain", None) or "Y"

    prog = evolution_mod.funded_cohort_progression(
        output_root, client_id, lens_filters=lens.filters or None,
        lens_label=lens.label, vintage=vintage, grain=grain, to_run_id=run_id)

    scope = lens.label + (f", {vintage} vintage" if vintage else "")
    if not prog.get("available"):
        return _undeliverable(question=question, spec=spec_dict, answer=(f"I can't build a progression for {scope}: "
                                 f"{prog.get('reason', 'no matching loans')}."),
                         route="cohort_progression",
                         warnings=[f"insufficient-data: {prog.get('reason', 'no matching cohort')}"])

    q = question.lower()
    mkey = _prog_metric_key(q)
    if mkey in ("nneg", "nneg_exposure") and not any(
            "nneg_exposure" in p["metrics"] for p in prog["periods"]):
        mkey = "balance"  # no valuation → NNEG not derivable; fall back to balance
    metric_key, label, vfmt, scale = _PROG_METRICS[mkey]

    periods = prog["periods"]
    rows = [{"period": p["period"], metric_key: (p["metrics"] or {}).get(metric_key)}
            for p in periods]
    chart = _chart_artifact(
        f"{label} — {scope}", chart_type="line", x_key="period", rows=rows,
        series=[{"key": metric_key, "label": label, "color": _PALETTE[0]}],
        value_format=vfmt, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
        display_hints={metric_key: {"format": vfmt, "scale": scale}},
        description=f"Static-pool {label.lower()} for {scope} across reporting periods.")

    # Full metrics table (all periods).
    tcols = [{"key": "period", "label": "Period", "align": "left", "format": "text"},
             {"key": "loan_count", "label": "Loans", "align": "right", "format": "number"},
             {"key": "funded_balance", "label": "Balance", "align": "right", "format": "gbp"},
             {"key": "wa_ltv", "label": "WA LTV", "align": "right", "format": "pct", "scale": "percent_fraction"},
             {"key": "wa_interest_rate", "label": "WA rate", "align": "right", "format": "pct", "scale": "percent_fraction"}]
    if "nneg_headroom_pct" in prog.get("metricsAvailable", []):
        tcols.append({"key": "nneg_headroom_pct", "label": "NNEG headroom", "align": "right",
                      "format": "pct", "scale": "percent_fraction"})
    trows = [{"period": p["period"], **{k: (p["metrics"] or {}).get(k)
              for k in ("loan_count", "funded_balance", "wa_ltv", "wa_interest_rate", "nneg_headroom_pct")}}
             for p in periods]
    table = _table_artifact(f"{scope} — metrics by period", columns=tcols, rows=trows,
                            spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)

    live = [p for p in periods if p["loanCount"]]
    first, last = (live[0] if live else None), (live[-1] if live else None)
    def _mv(p):
        return None if p is None else (p["metrics"] or {}).get(metric_key)
    fv, lv = _mv(first), _mv(last)
    trend = ""
    if fv is not None and lv is not None:
        trend = " up" if lv > fv else (" down" if lv < fv else " flat")
    answer = (f"{label} for {scope}: tracked across {len(live)} reporting period(s) "
              f"({first['period'] if first else '—'} → {last['period'] if last else '—'}"
              f"){trend}.")
    warnings = []
    if prog.get("singlePeriod"):
        warnings.append("Only one reporting period has loans for this cohort — a "
                        "progression reads best with two or more periods.")
    recon = _workspace.reconciliation_for(
        _workspace.datasets_read(output_root=output_root),
        reporting_date=(last or {}).get("reporting_date"))
    notes = [{"field": "cohort", "note": prog["lineage"]["note"]}]
    return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                     artifacts=[chart, table], reconciliation=recon, source_notes=notes,
                     warnings=warnings, route="cohort_progression")


# --------------------------------------------------------------------------- #
# G. Geographic exposure (ITL3) — "where is the book concentrated?"
# --------------------------------------------------------------------------- #
# Location words + an exposure/superlative intent route to the ITL3 exposure
# engine. A "limit / breach / appetite" framing is a risk-monitor question and is
# deliberately left to _route_risk (which owns concentration LIMITS).
_GEO_TERMS = ("geograph", "region", "area", "postcode", "post code", "itl3",
              "county", "city", "town")
# Deliberately NARROW: a superlative / concentration / map intent — NOT generic
# grouping words like "by region" (that is an ordinary point-in-time stratification
# and must reach the standard executor, not the ITL3 engine).
_GEO_MARKERS = ("concentrat", "largest", "biggest", "most exposed", "top ",
                "hotspot", "where is", "where are", "where's",
                "which region", "which area", "exposure")
_RISK_LIMIT_TERMS = ("limit", "breach", "appetite", "covenant", "threshold",
                     "rag", "amber", " red ")


#: An explicitly GROUPED ranking — "top 5 regions BY balance", "largest areas by
#: exposure". The "by <measure>" makes this an ordinary ranked stratification of
#: a named metric, which the point-in-time executor answers with the requested
#: top-N, the requested metric, the requested filters and the portfolio lens.
#: The ITL3 exposure engine honours none of those, so it must not capture it.
_GROUPED_RANKING_RE = re.compile(
    r"\b(?:top|bottom|largest|biggest|smallest|lowest|highest)\b[^?]*?\bby\b")


def _is_aggregate_contribution_question(question: str) -> bool:
    """True for "which region contributes most to the weighted average LTV?".

    Uses the SAME detector the P0 guard uses, so a route cannot stand down for
    one set of questions while the guard refuses a different set. No routed
    capability decomposes a weighted average across groups, so every one of them
    defers to the governed contribution aggregation.
    """
    try:
        from mi_agent import execution_receipt as _receipt

        return bool(_receipt._detect_contribution((question or "").lower()))
    except Exception:  # noqa: BLE001 - a detection fault must not lose a route
        logger.exception("contribution deference check failed for %r", question)
        return False


def _defers_to_period_change(question: str, *, spec: Any = None,
                             view: str = "funded") -> bool:
    """True when the governed period-change route positively claims ``question``.

    A point-in-time route that answers a two-period question does not produce a
    smaller answer — it produces a DIFFERENT one. Rather than duplicating
    period-change vocabulary here, this asks the owning capability's own
    recogniser, so the two can never drift apart.
    """
    try:
        return bool(_period_change.recognise(question, spec=spec,
                                             view=view or "funded").matched)
    except Exception:  # noqa: BLE001 - a recognition fault must not lose a route
        logger.exception("period-change deference check failed for %r", question)
        return False


def _is_a_generic_ranking(interpretation: Any) -> bool:
    """The CONTRACT says: a ranked stratification, with NO analytic named.

    ROUTE ENTITLEMENT, decided by the governed contract instead of by wording.

    The specialist route's wording tests cannot separate these two, and that was
    measured field by field before this existed — `OperationClaim.type`, all
    four ordering values, `modifiers`, the subject claim, the dimension claims
    and `residue` are identical on both:

        "Which region has the largest balance?"                 generic
        "What is the largest geographic area concentration?"    specialist

    `OperationClaim.analytic` is the fact that separates them. It is
    `mi_workflows.concentration_analysis`'s reading, carried on the contract —
    the owner of that vocabulary, asked once, before precedence. A RANKING with
    no analytic named is a measure ordered over an axis: the generic
    compositional path answers it with the requested direction, limit, filters
    and portfolio lens, none of which the ITL3 engine honours.

    A question that names BOTH — "the largest geographic area CONCENTRATION" —
    keeps the specialist route, because the analytic it named is the specialist
    one.

    The grouping dimension is required as well: a ranking with no axis is not a
    stratification of anything, and the specialist route keeps that too.

    ENTITLEMENT ONLY. This runs before any claim, so no handler has executed and
    nothing is handed on after a failure — `tests/test_failclosed_route_
    execution.py` is unchanged and still green.
    """
    if interpretation is None:
        return False
    try:
        from question_interpretation.schema import FILLED, GROUPING, RANKING
    except Exception:  # noqa: BLE001 - no contract vocabulary, no change
        return False
    operation = getattr(interpretation, "operation", None)
    if operation is None:
        return False
    if getattr(operation, "analytic", None) is not None:
        return False                      # a named analytic is the specialist's
    if getattr(operation, "type", None) != RANKING:
        return False
    if getattr(operation, "state", None) != FILLED:
        return False
    return any(getattr(d, "role", None) == GROUPING
               and getattr(d, "state", None) == FILLED
               for d in (getattr(interpretation, "dimensions", None) or ()))


def _is_geo_exposure(question: str, *, spec: Any = None,
                     view: str = "funded",
                     interpretation_provider: Any = None) -> bool:
    q = f" {question.lower()} "
    if any(t in q for t in _RISK_LIMIT_TERMS):
        return False  # a limit/breach question is a risk-monitor question
    if "bridge" in q:
        return False  # a balance bridge by region is the bridge route
    if _is_aggregate_contribution_question(question):
        return False  # a contribution to a weighted average is not an exposure map
    if _defers_to_period_change(question, spec=spec, view=view):
        # "Which region grew the most last month?" is a PERIOD-CHANGE question
        # that happens to name geography. This route answers at one date, so it
        # would have reported today's largest region and silently dropped the
        # comparison — exactly the substitution P0 refuses. Deference is narrow
        # by construction: it applies only when the governed period-change
        # recogniser positively claims the question, so anything that route
        # declines still lands here.
        return False
    if _GROUPED_RANKING_RE.search(q):
        # "show top 5 regions by balance" is a ranking question that happens to
        # mention geography — not a request for the ITL3 concentration view.
        # Routing it here discarded top_n, the metric and the lens.
        return False
    if not (any(t in q for t in _GEO_TERMS) and any(m in q for m in _GEO_MARKERS)):
        return False
    # THE CONTRACT HAS THE LAST WORD, and is asked LAST on purpose: building it
    # reads the frame and detects facets, so it is paid only for the handful of
    # questions the wording tests have already brought this far — and for those
    # it is memoised on the request, which this route's handler resolves anyway.
    if interpretation_provider is not None:
        try:
            if _is_a_generic_ranking(interpretation_provider()):
                return False
        except Exception:  # noqa: BLE001 - a contract fault must not lose a route
            logger.exception("geo entitlement contract check failed for %r", question)
    return True


def _route_geo(question, spec_dict, *, client_id, run_id, frame_resolver,
               portfolio_id, as_of, source_lens=None,
               interpretation=None) -> Optional[Dict[str, Any]]:
    """Funded exposure by UK ITL3 area → a ranked bar + table, from the ITL3
    exposure engine (tape ITL3 field, else postcode-derived). Answers "largest
    geographic concentration / where is the book". Degrades honestly when the
    tape carries no ITL3 or postcode.

    Geographic concentration is a POINT-IN-TIME question. With no run selected it
    answers from the ACTIVE governed dataset — the frame resolver returns exactly
    the frame the point-in-time executor would use — so it must never demand a
    run id. Genuinely temporal geography (evolution / comparison / cohort) is a
    different route.

    The frame is narrowed to the resolved portfolio scope BEFORE exposure is
    computed, so "where is the acquired book concentrated?" reports the acquired
    book — and the share-of-book percentages are shares of that scope, not of the
    platform. This route reads a dataframe, so it can honour a lens exactly as
    the point-in-time executor does; routes that read pre-aggregated run
    artefacts cannot, and disclose that instead (see ``try_route``)."""
    if interpretation is None:
        # NO CONTRACT, NO ANSWER FROM THIS ROUTE. The same rule Conversions 1
        # and 2 settled: one population owner, or none. Keeping the
        # lens-resolved path as a fallback would leave `_resolve_lens` reachable
        # from here exactly when the contract failed.
        return None
    if frame_resolver is None:
        return _undeliverable(question=question, spec=spec_dict, answer="I can't resolve the funded book for a geographic view here.",
                         route="geo_exposure", lens_applied=True,
                         warnings=["insufficient-data: no funded frame available."])
    try:
        df = frame_resolver(client_id, run_id)
    except Exception:  # noqa: BLE001 - a resolution hiccup degrades, never 500s
        df = None
    if df is None or not len(df):
        scope = "this run" if run_id else "the active reporting dataset"
        return _undeliverable(question=question, spec=spec_dict, answer=f"I couldn't load the funded book for {scope} to map exposure.",
                         route="geo_exposure", lens_applied=True,
                         warnings=[f"insufficient-data: no funded frame for {scope}."])

    # CONVERSION 3 — the switch point, and the whole of it.
    #
    # The one semantic fact this route ever read from the question is the source
    # scope, and it now comes from the contract. The plan narrows the frame with
    # the SAME governed id list, through the plan's own filters rather than
    # through a lens object — so `_apply_lens_filter` is no longer reachable
    # from here and the compositional layer has one narrowing owner.
    geo = _plan.geo_exposure(df, interpretation=interpretation)
    scope_label, narrowed = geo["lens"], geo["narrowed"]
    lens_warnings: List[str] = (
        [f"portfolio scope applied: {scope_label}"] if narrowed else [])
    if geo.get("empty_scope"):
        return _envelope(
            ok=True, question=question, spec=spec_dict, artifacts=[],
            answer=(f"There are no funded loans in {scope_label} to map, so I "
                    "can't report a geographic concentration for it."),
            route="geo_exposure", lens_applied=True,
            warnings=[f"no rows in scope for {scope_label}."])

    result = geo
    if not result.get("available"):
        # DEFER, DON'T DECLINE ON EVERYONE'S BEHALF.
        #
        # This returned ok=True with "I can't build a geographic exposure view
        # for this book" — a refusal wearing a success flag — and, because it
        # had already claimed the question, nothing else got to answer. So
        # "which region has the largest balance?" refused on a book whose very
        # next question, "show balance by region", returns seven regions: this
        # capability needs an ITL3 area or a property postcode, and the
        # governed obligor-region breakdown needs neither.
        #
        # Returning None is the estate's own pre-claim deferral: this route
        # cannot answer, so the next candidate may. A book that genuinely has
        # no region at all still refuses, one route further on, on its terms.
        # REVERTED, DELIBERATELY. An earlier pass had this defer so the generic
        # path could answer "which region has the largest balance?" — which it
        # does, completely and with disclosure. But that is exactly what
        # `test_geographic_exposure_degrades_honestly_without_itl3_or_postcode`
        # forbids: a specialist capability's failure handed to a different
        # answer. The guard is not wrongly formulated, no contract field
        # separates "where is the book concentrated" from "which region has the
        # largest balance" (both carry a measure), and this task's scope rules
        # out fixing a false refusal that its own change did not cause.
        #
        # So the route keeps the question and explains what it could not build.
        # The cost is two false refusals, documented rather than traded away.
        reason = result.get("reason", "no ITL3 area or property postcode on the tape")
        return _undeliverable(question=question, spec=spec_dict, answer=(f"I can't build a geographic exposure view for this book: "
                                 f"{reason}."),
                         route="geo_exposure", lens_applied=True,
                         warnings=lens_warnings + [f"insufficient-data: {reason}"])

    areas = result.get("areas", [])
    top = areas[0]
    rows = [{"area": a["itl3_name"] or a["itl3_code"], "code": a["itl3_code"],
             "balance": a["balance"], "count": a["count"],
             "share": (f"{a['sharePct']:.1f}%" if a.get("sharePct") is not None else "—")}
            for a in areas[:15]]
    chart = _chart_artifact(
        "Funded exposure by ITL3 area (top 15)", chart_type="bar", x_key="area",
        rows=rows, series=[{"key": "balance", "label": "Exposure", "color": _PALETTE[0]}],
        value_format="gbp", spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
        display_hints={"balance": {"format": "gbp", "scale": None}})
    table = _table_artifact(
        "Funded exposure by ITL3 area", columns=[
            {"key": "area", "label": "ITL3 area", "align": "left", "format": "text"},
            {"key": "code", "label": "Code", "align": "left", "format": "text"},
            {"key": "balance", "label": "Exposure", "align": "right", "format": "gbp"},
            {"key": "count", "label": "Loans", "align": "right", "format": "number"},
            {"key": "share", "label": "Book share", "align": "right", "format": "text"},
        ], rows=rows, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)
    top_name = top["itl3_name"] or top["itl3_code"]
    book = "the book" if not narrowed else scope_label
    answer = (f"Largest geographic concentration: {top_name} at {_gbp(top['balance'])} "
              f"({top['sharePct']:.1f}% of {book}) across {result.get('areaCount', len(areas))} "
              f"ITL3 area(s). Basis: {result.get('basis', 'tape')}; "
              f"resolved coverage {result.get('coveragePct', 0)}%.")
    notes = [{"field": "geo_basis",
              "note": (f"ITL3 basis {result.get('basis', 'tape')}: "
                       f"{result.get('resolvedFromItl3Field', 0)} from the ITL3 field, "
                       f"{result.get('resolvedFromPostcode', 0)} postcode-derived.")}]
    recon = {"dataset": "funded",
             "coverage_by_balance_pct": result.get("coveragePct", 100.0)}
    return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                     artifacts=[chart, table], reconciliation=recon,
                     source_notes=notes, route="geo_exposure",
                     warnings=lens_warnings, lens_applied=True)


# --------------------------------------------------------------------------- #
# H. Cumulative cohort conversion — the canonical "conversion" answer
# --------------------------------------------------------------------------- #
# One definition of conversion everywhere: the % of the ORIGINAL KFI cohort that
# has reached each milestone (KFI -> Application -> Offer -> Funded) to date. A
# forecast / threshold / scenario framing is a projection question and stays with
# the forecast route.
_COHORT_STAGE_LABELS = {"KFI": "KFI", "APPLICATION": "Application",
                        "OFFER": "Offer", "COMPLETED": "Funded"}
_CONVERSION_EXCLUDE = ("reach", "forecast", "project", "run rate", "run-rate",
                       "increase", "increased", "scenario", "if ", "target",
                       "milestone", "when will", "when do", "how long", "£", "$")


def _is_conversion(question: str) -> bool:
    q = f" {question.lower()} "
    if any(x in q for x in _CONVERSION_EXCLUDE):
        return False  # a projection / what-if question -> forecast route
    if "conversion" in q or "convert" in q:
        return True
    return "cohort" in q and ("fund" in q or "complet" in q)


def _route_conversion(question, spec_dict, *, history_model, portfolio_id, as_of
                      ) -> Dict[str, Any]:
    """Cumulative cohort conversion: the % of the original KFI cohort reaching each
    milestone (KFI → Application → Offer → Funded) by week, plus the headline
    latest Funded %. Matches the dashboard's canonical conversion metric."""
    model = history_model or {}
    prog = model.get("cohortProgression")
    conv = model.get("cumulativeCohortConversion")
    if not prog or not prog.get("weeks"):
        return _undeliverable(question=question, spec=spec_dict, answer=("I can't compute cumulative cohort conversion yet — it needs the "
                                 "weekly pipeline snapshots that track KFI cases through to funding."),
                         route="cohort_conversion",
                         warnings=["insufficient-data: no cohort-tracked pipeline history."])
    weeks = prog["weeks"]
    stages = prog.get("stages", [])
    series = prog.get("series", {})
    rows: List[Dict[str, Any]] = []
    for i, w in enumerate(weeks):
        row: Dict[str, Any] = {"week": w}
        for st in stages:
            vals = series.get(st) or []
            row[st] = vals[i] if i < len(vals) else None
        rows.append(row)
    chart = _chart_artifact(
        "Cumulative cohort conversion — % of the KFI cohort reaching each milestone",
        chart_type="line", x_key="week", rows=rows,
        series=[{"key": st, "label": _COHORT_STAGE_LABELS.get(st, st.title()),
                 "color": _PALETTE[i % len(_PALETTE)]} for i, st in enumerate(stages)],
        value_format="pct", spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
        display_hints={st: {"format": "pct", "scale": "percent_points"} for st in stages})
    table = _table_artifact(
        "Cohort progression by milestone",
        columns=[{"key": "week", "label": "Week", "align": "left", "format": "date"}]
        + [{"key": st, "label": _COHORT_STAGE_LABELS.get(st, st.title()),
            "align": "right", "format": "pct", "scale": "percent_points"} for st in stages],
        rows=rows, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)
    conv_txt = f"{conv:.1f}%" if isinstance(conv, (int, float)) else "n/a"
    answer = (f"Cumulative cohort conversion is {conv_txt} — the share of the original KFI "
              f"cohort ({prog.get('cohortSize')} case(s)) that has funded to date. The chart "
              f"tracks that cohort through KFI → Application → Offer → Funded across "
              f"{len(weeks)} week(s). This is the single definition of conversion used on the "
              f"KPI card, the funnel and the forecast.")
    notes = [{"field": "cohort_conversion",
              "note": ("Cumulative cohort progression: of the original KFI cohort, the % reaching "
                       "each milestone to date — case-tracked across weekly snapshots, monotonic.")}]
    recon = {"dataset": "pipeline", "coverage_by_balance_pct": 100.0}
    return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                     artifacts=[chart, table], reconciliation=recon, source_notes=notes,
                     route="cohort_conversion")


# --------------------------------------------------------------------------- #
# Detection + dispatch
# --------------------------------------------------------------------------- #
_EVOLUTION_MARKERS = ("evolution", "over time", "trend", "by month", "monthly",
                      "by week", "weekly", "per week", "by reporting", "stage over time",
                      "over the months")


def _is_evolution(question: str, spec) -> bool:
    if spec.chart_type != "line":
        return False
    if spec.x == "vintage_year":
        return False
    # A filter no longer forces the within-snapshot path: _route_evolution applies
    # the filter WITHIN each period for a funded series (and defers otherwise).
    q = question.lower()
    if any(m in q for m in _EVOLUTION_MARKERS):
        return True
    # THE AXIS QUESTION IS THE OWNER'S. `_EVOLUTION_MARKERS` above is a third
    # copy of "did this sentence ask for a time axis?" — after
    # `lexical.time_axis_request` (which owns it) and `llm_query_parser`'s
    # `is_line` (which sets the chart type). Two of the three deciding
    # differently is not academic: with only the chart type widened, "balance by
    # period" became a line, failed THIS list, missed the evolution route, and
    # was answered by the generic line executor as 13 VINTAGE YEARS — a cohort
    # distribution, not a reporting-period series. The parser's own note at that
    # path already warns a vintage "is a cohort label, not a point on a time
    # axis".
    #
    # Consulting the owner here keeps the chart type and the route on one
    # reading. See docs/mi_dual_mechanism_pattern.md.
    try:
        from question_interpretation.lexical import time_axis_request
    except Exception:  # noqa: BLE001 - routing must never break on an import
        return False
    return bool(time_axis_request(question))


# --------------------------------------------------------------------------- #
# Capability routing — the governed recogniser registry
# --------------------------------------------------------------------------- #
# This section used to be a hand-ordered ``if/elif`` chain inside ``try_route``.
# It is now a declarative registry (``recogniser_registry``): each capability
# states its own name, precedence, recogniser, handler, lens behaviour and
# governed capability gate in ONE place, and precedence is data rather than
# source-code line order. Behaviour is preserved — every recogniser below keeps
# its historical position via ``priority``, and all share DEFAULT_CONFIDENCE, so
# ordering collapses to exactly the old chain order.

# --------------------------------------------------------------------------- #
# Portfolio Risk Comparison — adapter for the governed workflow package.
#
# The workflow (mi_workflows.portfolio_risk_comparison) owns recognition
# predicates, scope resolution, Business Semantics Registry consumption and
# every calculation. This adapter only resolves the collaborators the workflow
# needs (frame, portfolio registry, BSR) and re-keys the result contract into
# the chat envelope — it performs no calculations and takes no decisions.
# --------------------------------------------------------------------------- #
#: Workflow recognisers declare higher confidence than the single-capability
#: default (0.5): a genuine workflow question outranks the single-capability
#: recogniser that would otherwise catch it (architecture doc §11.3).
_WORKFLOW_CONFIDENCE = 0.7


def _recognise_portfolio_comparison(request: RouteRequest) -> Recognition:
    matched, reason = prc_mod.is_portfolio_comparison_question(
        request.question, request.spec)
    return (Recognition.yes(_WORKFLOW_CONFIDENCE, reason) if matched
            else Recognition.no(reason))


def _share_pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _comparison_cell(value: Optional[float], unit: str,
                     aggregation: Optional[str] = None) -> str:
    if value is None:
        return "—"
    if aggregation == "count":
        # A cardinality is a whole number. Keyed off the AGGREGATION, not the
        # unit: the average of an integer-unit field (a term in months) is
        # legitimately fractional and must keep its decimals.
        return f"{float(value):,.0f}"
    if unit == "currency":
        return _gbp(value)
    if unit == "share":
        return _share_pct(value)
    if unit == "integer":
        return f"{float(value):,.1f}"
    return f"{float(value):,.4g}"


def _metric_comparison_table(result: Dict[str, Any], *, spec_dict, portfolio_id,
                             as_of) -> Optional[Dict[str, Any]]:
    comparisons = result.get("metric_comparisons") or []
    if not comparisons:
        return None
    sides = result.get("portfolio_results") or []
    label_a = sides[0]["label"] if sides else "Portfolio A"
    label_b = sides[1]["label"] if len(sides) > 1 else "Portfolio B"
    rows = []
    for c in comparisons:
        unit = c.get("unit") or "decimal"
        agg = c.get("aggregation")
        rows.append({
            "metric": c["display_name"],
            "aggregation": c["aggregation"]
            + (f" (wt: {c['weight_basis']})" if c.get("weight_basis") else ""),
            "a": _comparison_cell(c["portfolio_a"]["value"], unit, agg),
            "b": _comparison_cell(c["portfolio_b"]["value"], unit, agg),
            "difference": _comparison_cell(c["difference"]["absolute"], unit, agg),
            "direction": (c["directionality"]["higher"] or "—").replace(
                "portfolio_a", label_a).replace("portfolio_b", label_b),
        })
    return _table_artifact(
        f"Portfolio comparison — {label_a} vs {label_b}",
        columns=[
            {"key": "metric", "label": "Metric", "align": "left", "format": "text"},
            {"key": "aggregation", "label": "Aggregation", "align": "left",
             "format": "text"},
            {"key": "a", "label": label_a, "align": "right", "format": "text"},
            {"key": "b", "label": label_b, "align": "right", "format": "text"},
            {"key": "difference", "label": "Difference (A−B)", "align": "right",
             "format": "text"},
            {"key": "direction", "label": "Higher", "align": "left",
             "format": "text"},
        ], rows=rows, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
        description=f"{len(rows)} governed metric comparison(s).")


def _distribution_comparison_tables(result: Dict[str, Any], *, spec_dict,
                                    portfolio_id, as_of) -> List[Dict[str, Any]]:
    sides = result.get("portfolio_results") or []
    label_a = sides[0]["label"] if sides else "Portfolio A"
    label_b = sides[1]["label"] if len(sides) > 1 else "Portfolio B"
    tables = []
    for dist in result.get("distribution_comparisons") or []:
        rows = [{
            "category": row["category"],
            "count_a": _share_pct(row["count_share_a"]),
            "count_b": _share_pct(row["count_share_b"]),
            "exposure_a": _share_pct(row["exposure_share_a"]),
            "exposure_b": _share_pct(row["exposure_share_b"]),
        } for row in dist.get("categories") or []]
        tables.append(_table_artifact(
            f"{dist['display_name']} mix — {label_a} vs {label_b}",
            columns=[
                {"key": "category", "label": dist["display_name"],
                 "align": "left", "format": "text"},
                {"key": "count_a", "label": f"{label_a} (count)",
                 "align": "right", "format": "text"},
                {"key": "count_b", "label": f"{label_b} (count)",
                 "align": "right", "format": "text"},
                {"key": "exposure_a", "label": f"{label_a} (exposure)",
                 "align": "right", "format": "text"},
                {"key": "exposure_b", "label": f"{label_b} (exposure)",
                 "align": "right", "format": "text"},
            ], rows=rows, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
            description=f"Share of each scope; unknown values reported as "
                        f"'{prc_mod.engine.UNKNOWN_CATEGORY}'."))
    return tables


def _route_portfolio_comparison(request: RouteRequest) -> Optional[Dict[str, Any]]:
    """Adapter: collaborators in, workflow result out, envelope re-keying only."""
    route = prc_mod.WORKFLOW_ID
    if request.frame_resolver is None:
        return _undeliverable(
            question=request.question, spec=request.spec_dict,
            route=route, lens_applied=True,
            answer="I can't resolve the governed funded book to compare portfolios here.",
            warnings=["insufficient-data: no funded frame available."])
    try:
        df = request.frame_resolver(request.client_id, request.run_id)
    except Exception:  # noqa: BLE001 - a resolution hiccup degrades, never 500s
        df = None
    if df is None or not len(df):
        return _undeliverable(
            question=request.question, spec=request.spec_dict,
            route=route, lens_applied=True,
            answer="I couldn't load the governed funded book to compare portfolios.",
            warnings=["insufficient-data: no funded frame available."])
    try:
        bsr = load_business_semantics()
    except Exception as exc:  # noqa: BLE001 - a controlled outcome, never a 500
        _logger.warning("business semantics registry unavailable: %s", exc)
        return _undeliverable(
            question=request.question, spec=request.spec_dict,
            route=route, lens_applied=True,
            answer=("Portfolio comparison is unavailable: the Business "
                    "Semantics Registry could not be loaded, and comparison is "
                    "only performed over governed semantics."),
            warnings=["insufficient-data: business semantics registry unavailable."])
    try:
        from . import portfolio_context as _ctx
        registry = _ctx.build_registry(df, client_id=request.client_id)
    except Exception:  # noqa: BLE001 - the workflow builds its own from the frame
        registry = None

    result = prc_mod.run_portfolio_risk_comparison(
        df, question=request.question, bsr=bsr, mi_semantics=request.semantics,
        registry=registry, client_id=request.client_id, as_of=request.as_of,
        spec=request.spec)

    warnings = list(result.get("warnings") or [])
    warnings.extend(f"limitation: {note}" for note in result.get("limitations") or [])

    if not result.get("available"):
        envelope = _undeliverable(
            question=request.question, spec=request.spec_dict,
            route=route, lens_applied=True,
            answer=f"I can't compare portfolios here: {result.get('reason')}.",
            warnings=warnings)
        envelope["workflow"] = result
        return envelope

    sides = result["portfolio_results"]
    metric_comparisons = result.get("metric_comparisons") or []
    distribution_comparisons = result.get("distribution_comparisons") or []
    requested = result.get("requested_metric") or {}
    requested_set = result.get("requested_metrics") or (
        [requested] if requested else [])
    compared_ok = [r for r in requested_set if r.get("compared")]
    not_compared = [r for r in requested_set if not r.get("compared")]
    # Requested measures the workflow does not express at all (a loan count has
    # no BSR directionality, so it is not one of the measures it undertook to
    # compare). Named in the answer alongside `not_compared`, but kept out of
    # the requested-MEASURE ledger those two lists feed.
    uncomparable = result.get("uncomparable_measures") or []

    # ---- the requested-measure invariant -------------------------------- #
    # Every measure the caller named must be accounted for before this route may
    # answer. When NONE was compared it is a refusal naming them; when some were
    # and some were not, the answer stands and names what is missing (P1E's
    # explicit-partial rule). The failure this prevents: "how do the two books
    # compare on borrower age?" returning "no governed directional differences
    # were observed" when borrower age was never compared at all.
    if requested_set and not compared_ok:
        names = _sentence_join([r.get("display_name") or r.get("field")
                                for r in not_compared]) or "the requested measure"
        reasons = "; ".join(sorted({str(r.get("reason")) for r in not_compared}))
        message = (
            f"I could not compare {sides[0]['label']} with {sides[1]['label']} "
            f"on {names}: {reasons}. I have not compared a different "
            f"measure instead, and I have not reported this as 'no difference'.")
        envelope = _envelope(
            ok=False, question=request.question, spec=request.spec_dict,
            artifacts=[], route=route, lens_applied=True,
            answer=message, error=message, warnings=warnings + [message])
        envelope["workflow"] = result
        envelope["controlledRefusal"] = True
        meta = envelope["metadata"]
        meta["controlledRefusal"] = True
        meta["controlledUnsupported"] = True
        meta["portfolioComparison"] = {
            # This route compares PORTFOLIO SCOPES, so the cohort concept it
            # expresses is how the loans were SOURCED. Declared as evidence so a
            # question asking about a different cohort — seasoning, vintage —
            # cannot be marked answered merely because two books were compared.
            "cohortConcept": "sourcing",
            "requestedMetric": requested.get("field"),
            "requestedMetrics": [r.get("field") for r in requested_set],
            "requestedMetricCompared": False,
            "reason": requested.get("reason"),
            "measuresCompared": [],
        }
        return envelope

    # ---- empty-comparison safety ---------------------------------------- #
    # An empty comparison set means NOTHING WAS COMPARED. It does not mean the
    # portfolios are alike, and it may never be rendered as a negative finding.
    summary_lines = result.get("summary") or []
    if summary_lines:
        answer = " ".join(summary_lines)
        if not_compared or uncomparable:
            # Explicit partial: what ran, and what did not, by name. Never a
            # silent three-of-four.
            answer += (" Not compared: " + "; ".join(
                f"{r.get('display_name') or r.get('field')} ({r.get('reason')})"
                for r in list(not_compared) + list(uncomparable)) + ".")
    elif metric_comparisons or distribution_comparisons:
        # Something WAS compared and no direction emerged — a real observation.
        answer = (f"Compared {sides[0]['label']} with {sides[1]['label']} at "
                  f"{result.get('reporting_date') or 'the current reporting date'} "
                  f"across {len(metric_comparisons) + len(distribution_comparisons)} "
                  f"governed indicator(s): no directional difference was observed "
                  f"on any of them.")
    else:
        message = (
            f"I did not compare {sides[0]['label']} with {sides[1]['label']}: no "
            f"governed indicator was eligible for comparison on this book. "
            f"Nothing was measured, so I cannot say whether the two books "
            f"differ.")
        envelope = _envelope(
            ok=False, question=request.question, spec=request.spec_dict,
            artifacts=[], route=route, lens_applied=True,
            answer=message, error=message, warnings=warnings + [message])
        envelope["workflow"] = result
        envelope["controlledRefusal"] = True
        envelope["metadata"]["controlledRefusal"] = True
        envelope["metadata"]["controlledUnsupported"] = True
        envelope["metadata"]["portfolioComparison"] = {
            # This route compares PORTFOLIO SCOPES, so the cohort concept it
            # expresses is how the loans were SOURCED. Declared as evidence so a
            # question asking about a different cohort — seasoning, vintage —
            # cannot be marked answered merely because two books were compared.
            "cohortConcept": "sourcing",
            "requestedMetric": requested.get("field"),
            "requestedMetricCompared": False,
            "reason": "no governed indicator was eligible for comparison",
            "measuresCompared": [],
        }
        return envelope

    artifacts: List[Dict[str, Any]] = []
    table = _metric_comparison_table(result, spec_dict=request.spec_dict,
                                     portfolio_id=request.portfolio_id,
                                     as_of=request.as_of)
    if table is not None:
        artifacts.append(table)
    artifacts.extend(_distribution_comparison_tables(
        result, spec_dict=request.spec_dict, portfolio_id=request.portfolio_id,
        as_of=request.as_of))

    audit = result.get("audit") or {}
    notes = [{"field": "governance",
              "note": (f"Business Semantics Registry v{audit.get('bsr_version')} "
                       f"(schema {audit.get('bsr_schema_version')}); calculation "
                       f"version {audit.get('calculation_version')}; only fields "
                       "governed for portfolio comparison were compared.")}]
    recon = {"dataset": prc_mod.DATASET,
             "reporting_date": result.get("reporting_date"),
             "portfolio_a_rows": sides[0]["row_count"],
             "portfolio_b_rows": sides[1]["row_count"]}
    envelope = _envelope(
        ok=True, question=request.question, answer=answer,
        spec=request.spec_dict, artifacts=artifacts, reconciliation=recon,
        source_notes=notes, route=route, warnings=warnings, lens_applied=True)
    envelope["workflow"] = result
    # The EVIDENCE the P0 guard verifies and the receipt renders: which measures
    # were actually compared, and between which two books. Derived from executed
    # comparison metadata, never from the question's wording.
    envelope["metadata"]["portfolioComparison"] = {
        # This route compares PORTFOLIO SCOPES, so the cohort concept it
        # expresses is how the loans were SOURCED. Declared as evidence so a
        # question asking about a different cohort — seasoning, vintage —
        # cannot be marked answered merely because two books were compared.
        "cohortConcept": "sourcing",
        "requestedMetric": requested.get("field"),
        "requestedMetrics": [r.get("field") for r in requested_set],
        "requestedMetricsCompared": [r.get("field") for r in compared_ok],
        "requestedMetricsNotCompared": [r.get("field") for r in not_compared],
        "requestedMetricCompared": bool(compared_ok) if requested_set else None,
        "measuresCompared": [c.get("field") for c in metric_comparisons],
        "measureLabels": [c.get("display_name") for c in metric_comparisons],
        "aggregations": [c.get("aggregation") for c in metric_comparisons],
        "dimensionsCompared": [c.get("field") for c in distribution_comparisons],
        "portfolioA": sides[0]["label"],
        "portfolioB": sides[1]["label"],
        "reportingDate": result.get("reporting_date"),
    }
    return envelope


# --------------------------------------------------------------------------- #
# Concentration Analysis — adapter for the governed workflow package.
#
# The workflow (mi_workflows.concentration_analysis) owns recognition
# predicates, scope resolution, Business Semantics Registry consumption and
# every calculation. This adapter only resolves the collaborators the workflow
# needs (frame, portfolio registry, BSR, the workspace scope) and re-keys the
# result contract into the chat envelope — it performs no calculations and
# takes no decisions.
# --------------------------------------------------------------------------- #
#: The key this workflow's pre-claim reading is carried under.
CONCENTRATION_READING_KEY = "concentration"


def _lens_from_contract(interpretation):
    """The source-portfolio lens the contract states. See `contract_scope`."""
    from . import contract_scope as _scope
    return _scope.lens_from_contract(interpretation)


def _recognise_concentration(request: RouteRequest) -> Recognition:
    if _is_aggregate_contribution_question(request.question):
        # Concentration measures how exposure is DISTRIBUTED at one date; it
        # does not decompose a weighted average across groups.
        return Recognition.no("aggregate_contribution_question")
    matched, reason = conc_mod.is_concentration_question(
        request.question, request.spec)
    if matched:
        # THE READING IS KEPT. This recogniser already reads the question; the
        # workflow used to read it AGAIN, after the route was claimed, for the
        # concept and the single-name framing. Reading it once here and
        # carrying the result is what removes those two post-claim decisions
        # without inventing a governed concept for either.
        remember = getattr(request, "remember_recognition", None)
        if remember is not None:
            remember(CONCENTRATION_READING_KEY,
                     conc_mod.read_question(request.question))
    return (Recognition.yes(_WORKFLOW_CONFIDENCE, reason) if matched
            else Recognition.no(reason))


def _concentration_rows(categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "rank": c["rank"],
        "category": c["category"],
        "exposure": "—" if c["exposure"] is None else _gbp(c["exposure"]),
        "exposure_share": _share_pct(c["exposure_share"]),
        "count": c["count"],
        "count_share": _share_pct(c["count_share"]),
        "cumulative_share": _share_pct(c["cumulative_share"]),
    } for c in categories]


_CONCENTRATION_COLUMNS = [
    {"key": "rank", "label": "Rank", "align": "right", "format": "number"},
    {"key": "category", "label": "Category", "align": "left", "format": "text"},
    {"key": "exposure", "label": "Exposure", "align": "right", "format": "text"},
    {"key": "exposure_share", "label": "Exposure share", "align": "right",
     "format": "text"},
    {"key": "count", "label": "Loans", "align": "right", "format": "number"},
    {"key": "count_share", "label": "Loan share", "align": "right",
     "format": "text"},
    {"key": "cumulative_share", "label": "Cumulative", "align": "right",
     "format": "text"},
]


def _concentration_tables(result: Dict[str, Any], *, spec_dict, portfolio_id,
                          as_of) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    scope_label = (result.get("portfolio_scope") or {}).get("label") or "Total"
    for dim in result.get("dimension_results") or []:
        unknown = dim["unknown"]
        rows = _concentration_rows(dim["categories"])
        if unknown["count"]:
            rows.append({
                "rank": None, "category": conc_mod.engine.UNKNOWN_CATEGORY,
                "exposure": ("—" if unknown["exposure"] is None
                             else _gbp(unknown["exposure"])),
                "exposure_share": _share_pct(unknown["exposure_share"]),
                "count": unknown["count"],
                "count_share": _share_pct(unknown["count_share"]),
                "cumulative_share": "—",
            })
        tables.append(_table_artifact(
            f"{dim['display_name']} concentration — {scope_label}",
            columns=_CONCENTRATION_COLUMNS, rows=rows, spec=spec_dict,
            portfolio_id=portfolio_id, as_of=as_of,
            description=(f"{dim['category_count']} categor(ies) ranked by "
                         f"{dim['basis']}; unknown values reported explicitly.")))
    for sn in result.get("single_name_results") or []:
        rows = _concentration_rows(sn["categories"])
        tables.append(_table_artifact(
            f"Largest {sn['kind']} exposures — {scope_label}",
            columns=_CONCENTRATION_COLUMNS, rows=rows, spec=spec_dict,
            portfolio_id=portfolio_id, as_of=as_of,
            description=(f"Top {sn['listed']} of {sn['distinct_names']} "
                         f"governed {sn['kind']} identifier(s) by "
                         f"{sn['basis']}; remainder share "
                         f"{_share_pct(sn['remainder']['share'])}.")))
    return tables


def _route_concentration(request: RouteRequest) -> Optional[Dict[str, Any]]:
    """Adapter: collaborators in, workflow result out, envelope re-keying only."""
    route = conc_mod.WORKFLOW_ID
    if request.frame_resolver is None:
        return _undeliverable(
            question=request.question, spec=request.spec_dict,
            route=route, lens_applied=True,
            answer="I can't resolve the governed funded book to measure concentration here.",
            warnings=["insufficient-data: no funded frame available."])
    try:
        df = request.frame_resolver(request.client_id, request.run_id)
    except Exception:  # noqa: BLE001 - a resolution hiccup degrades, never 500s
        df = None
    if df is None or not len(df):
        return _undeliverable(
            question=request.question, spec=request.spec_dict,
            route=route, lens_applied=True,
            answer="I couldn't load the governed funded book to measure concentration.",
            warnings=["insufficient-data: no funded frame available."])
    try:
        bsr = load_business_semantics()
    except Exception as exc:  # noqa: BLE001 - a controlled outcome, never a 500
        _logger.warning("business semantics registry unavailable: %s", exc)
        return _undeliverable(
            question=request.question, spec=request.spec_dict,
            route=route, lens_applied=True,
            answer=("Concentration analysis is unavailable: the Business "
                    "Semantics Registry could not be loaded, and concentration "
                    "is only measured over governed semantics."),
            warnings=["insufficient-data: business semantics registry unavailable."])
    try:
        from . import portfolio_context as _ctx
        registry = _ctx.build_registry(df, client_id=request.client_id)
    except Exception:  # noqa: BLE001 - the workflow builds its own from the frame
        registry = None

    # THE SCOPE COMES FROM THE CONTRACT, not from a second reading of the
    # sentence. `source_scope` already carries the owner's answer and the
    # provenance that decides precedence against a workspace selection —
    # measured equivalent to `_resolve_lens` on all 882 corpus questions, and
    # again with a workspace selection present.
    # UNRESOLVABLE IS NOT ABSENT. A book the registry does not hold must reach
    # the workflow as the name the reader used, so it refuses by that name —
    # not as "no scope", which is the whole book.
    try:
        from . import contract_scope as _scope
        context_id = _scope.requested_context_id(request.resolve_interpretation())
    except Exception:  # noqa: BLE001 - an identity fault must not fail the route
        context_id = None

    result = conc_mod.run_concentration_analysis(
        df, question=request.question, bsr=bsr, mi_semantics=request.semantics,
        registry=registry, client_id=request.client_id, as_of=request.as_of,
        spec=request.spec, parse_meta=request.parse_meta,
        context_id=context_id,
        reading=request.recalled_recognition(CONCENTRATION_READING_KEY))

    warnings = list(result.get("warnings") or [])
    warnings.extend(f"limitation: {note}" for note in result.get("limitations") or [])

    if not result.get("available"):
        envelope = _undeliverable(
            question=request.question, spec=request.spec_dict,
            route=route, lens_applied=True,
            answer=f"I can't measure concentration here: {result.get('reason')}.",
            warnings=warnings)
        envelope["workflow"] = result
        return envelope

    scope = result["portfolio_scope"]
    summary_lines = result.get("summary") or []
    if summary_lines:
        answer = " ".join(summary_lines)
    else:
        answer = (f"Measured concentration for {scope['label']} at "
                  f"{result.get('reporting_date') or 'the current reporting date'} "
                  f"on the {result.get('concentration_basis')} basis.")

    artifacts = _concentration_tables(
        result, spec_dict=request.spec_dict, portfolio_id=request.portfolio_id,
        as_of=request.as_of)

    audit = result.get("audit") or {}
    notes = [{"field": "governance",
              "note": (f"Business Semantics Registry v{audit.get('bsr_version')} "
                       f"(schema {audit.get('bsr_schema_version')}); calculation "
                       f"version {audit.get('calculation_version')}; only "
                       "governed concentration dimensions were measured, on the "
                       f"{result.get('concentration_basis')} basis.")}]
    recon = {"dataset": conc_mod.DATASET,
             "reporting_date": result.get("reporting_date"),
             "rows_in_scope": scope["row_count"]}
    envelope = _envelope(
        ok=True, question=request.question, answer=answer,
        spec=request.spec_dict, artifacts=artifacts, reconciliation=recon,
        source_notes=notes, route=route, warnings=warnings, lens_applied=True)
    envelope["workflow"] = result
    # Structured evidence of a SINGLE-NAME analysis, so the receipt and the P0
    # share facet can be settled from what executed rather than from the answer
    # text. Reading a percentage out of prose proves only that prose mentions
    # one; this states the grain, the measure and both sides of the share.
    single = _single_name_evidence(result)
    if single:
        envelope.setdefault("metadata", {})["concentration"] = single
    # THE SCOPE THIS ANALYSIS WAS NARROWED TO, from the workflow's own record.
    # It titles its table "— Direct" and reports 441 of 640 loans; until this
    # was published, nothing a consumer could read said the narrowing had
    # happened, which is the same silence that hid a wrong movement figure.
    if scope.get("context_id") not in (None, _portfolio_lens.LENS_TOTAL):
        _declare_scope(envelope,
                       {"detail": ", ".join(scope.get("portfolio_ids") or ()),
                        "rowsBefore": int(len(df)),
                        "rowsAfter": scope.get("row_count")},
                       context=scope.get("context_id"),
                       label=scope.get("label"))
    return envelope


def _single_name_evidence(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """What the concentration workflow declares about a single-name result.

    ``None`` when it measured dimensions rather than individual names — a
    regional breakdown is not evidence that the largest single loan was found.
    """
    for entry in (result.get("single_name_results") or []):
        cats = entry.get("categories") or []
        if not cats:
            continue
        top = cats[0]
        return {
            "kind": entry.get("kind"),
            "grainField": entry.get("field"),
            "basis": entry.get("basis"),
            "population": entry.get("population"),
            "distinctNames": entry.get("distinct_names"),
            "topExposure": top.get("exposure"),
            "topShare": top.get("exposure_share"),
            "totalExposure": entry.get("total_exposure"),
            "reportingDate": result.get("reporting_date"),
        }
    return None


#: Human wording per route for the lens disclosure below.
_ROUTE_NOUN = {
    "temporal_compare": "period comparison", "evolution": "trend",
    "evolution_funnel": "funnel trend", "evolution_pipeline_stage": "pipeline trend",
    "forecast_extrapolation": "forecast", "scenario": "scenario",
    "risk_limits": "risk-limit", "cohort_conversion": "conversion",
    "portfolio_risk_comparison": "portfolio comparison",
    "concentration_analysis": "concentration",
}


def _lens_aware_routes() -> frozenset:
    """Routes that genuinely narrow their figures to the portfolio lens.

    Derived from the registry so the fact is declared once, on the recogniser,
    rather than duplicated in a set that can drift out of step with it.
    """
    return frozenset(r.name for r in REGISTRY.ordered() if r.lens_aware)


def _disclose_lens_scope(envelope: Optional[Dict[str, Any]], question: str,
                         source_lens: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Make every routed answer state, truthfully, what scope it covers.

    A routed answer that was not narrowed to the requested lens is whole-book.
    Saying nothing let the governed envelope stamp it with the narrow scope
    anyway (``mi_service._stamp_routed_scope``), so the one control that exists
    to prevent misattribution was performing it. This marks the answer instead:
    ``metadata.lensApplied`` becomes an explicit boolean, and a non-total lens
    that could not be applied is disclosed to the user in plain words.
    """
    if not isinstance(envelope, dict):
        return envelope
    meta = envelope.setdefault("metadata", {})
    if not isinstance(meta, dict):
        return envelope
    route = meta.get("route") or ""
    if meta.get("lensApplied") is None:
        meta["lensApplied"] = route in _lens_aware_routes()
    if meta["lensApplied"]:
        return envelope
    try:
        lens = _resolve_lens(question, source_lens)
    except Exception:  # noqa: BLE001 - disclosure must never break an answer
        return envelope
    if not lens.filters:          # Total was requested; whole-book IS the scope.
        meta["lensApplied"] = True
        return envelope
    noun = _ROUTE_NOUN.get(route, "routed")
    disclosure = (
        f"Scope not narrowed: this {noun} answer is computed across the whole "
        f"platform book. It is sourced from a governed run artefact that carries "
        f"no source-portfolio provenance, so it could not be scoped to "
        f"'{lens.label}' — these figures are NOT {lens.label}-only.")
    warnings = envelope.setdefault("warnings", [])
    if isinstance(warnings, list) and disclosure not in warnings:
        warnings.append(disclosure)
    meta["lensRequested"] = lens.label
    return envelope


def _capability_unavailable_envelope(req: RouteRequest, recogniser,
                                     state: Any) -> Dict[str, Any]:
    """The governed 'this capability does not apply here' answer.

    Built from the SAME ``CapabilityState`` the React dashboard renders, so a
    scope that cannot support an analysis is explained identically on every
    channel — rather than each surface inventing its own data-availability
    error. ``controlledUnsupported`` makes ``mi_service`` classify it as
    ``UNSUPPORTED_QUESTION`` (HTTP 200, ``ok:false``), which is the existing
    governed contract for "I will not answer that".
    """
    detail = (getattr(state, "detail", None)
              or "This analysis is not available for the selected portfolios.")
    envelope = _envelope(
        ok=False, question=req.question, answer=detail, spec=req.spec_dict,
        artifacts=[], route=recogniser.name, error=detail,
        lens_applied=True,
        warnings=[f"capability unavailable: {recogniser.capability}"])
    meta = envelope["metadata"]
    meta["controlledUnsupported"] = True
    meta["capability"] = recogniser.capability
    meta["capabilityReason"] = getattr(state, "reason_code", None)
    meta["capabilityExcluded"] = list(getattr(state, "excluded_portfolios", ()) or ())
    return envelope


def _execution_failure_envelope(req: RouteRequest, recogniser) -> Dict[str, Any]:
    """The governed 'this analysis failed' answer for a claimed route.

    NO NEW PUBLIC TAXONOMY. An `ok:false` envelope that is neither
    `controlledUnsupported` nor `unmappedQuestion` nor a no-rows case is already
    classified `ErrorCode.CALCULATION_FAILED` by
    `mi_service._classify_analytical_failure` — the existing governed code for
    "the calculation broke", as distinct from "I will not answer that".

    NOTHING INTERNAL REACHES THE READER. The exception class, its message and
    its traceback are logged and never published; the caller is told which
    analysis failed and that nothing was substituted for it. The claimed route
    stays on `metadata.route`, so an answer produced by a different route after
    this one failed is detectable rather than indistinguishable.
    """
    detail = ("I could not complete this analysis: it failed while running. I "
              "have not answered your question with a different analysis "
              "instead.")
    envelope = _envelope(
        ok=False, question=req.question, answer=detail, spec=req.spec_dict,
        artifacts=[], route=recogniser.name, error=detail, lens_applied=True,
        warnings=[f"execution failed in {recogniser.name}"])
    meta = envelope["metadata"]
    meta["executionFailure"] = True
    meta["claimedRoute"] = recogniser.name
    meta["claimBoundaryCrossed"] = True
    if recogniser.capability:
        meta["capability"] = recogniser.capability
    return envelope


def _register_default_recognisers(registry: RecogniserRegistry) -> RecogniserRegistry:
    """Declare the governed capability routes.

    ``priority`` reproduces the historical chain order exactly for the eleven
    routes migrated from the old ``if/elif`` chain; ``period_change_analysis``
    was added afterwards and takes a position between them without moving any
    of them relative to each other. A capability gate is declared ONLY where an
    unavailable capability is a genuine, explainable outcome (pipeline /
    origination / cohort / risk); funded-book routes are ungated because every
    scope with rows supports them, and gating them would cost a context
    resolution on the common path for no decision.
    """
    registry.extend([
        # 0. The ANALYTICAL CAPABILITY LAYER. Registered first, and at a higher
        #    confidence, because a question whose answer is two or more governed
        #    capabilities must not be answered in part by whichever single
        #    capability below happens to match it first. Safe to put here only
        #    because its recognition is strict: it matches ONLY when the
        #    deterministic planner composes a multi-capability plan, and the
        #    planner declines every question a route below already owns
        #    (mi_workflows/analytical/planner.py). Its handler also returns None
        #    when a plan produces nothing computable, so a book missing the data
        #    a plan needs falls through to exactly the behaviour it had before.
        analytical_mod.recogniser(),

        # 1. What-if / scenario perturbs the run-rate and re-solves the
        #    milestone. Returns None when the magnitude cannot be quantified, so
        #    it falls through to forecast / conversion.
        Recogniser(
            name="scenario", priority=10, capability=CAP_PIPELINE, lens_aware=False,
            description="Deterministic what-if on the completion run-rate.",
            recognise=lambda r: _is_scenario(r.question),
            handle=lambda r: _route_scenario(
                r.question, r.spec, r.spec_dict, client_id=r.client_id,
                run_id=r.run_id, output_root=r.output_root,
                pipeline_root=r.pipeline_root, history_model=r.resolve_history_model(),
                portfolio_id=r.portfolio_id, as_of=r.as_of)),

        # 2. Cohort-tracked conversion is the canonical "conversion" answer and
        #    is checked before forecast so a bare "conversion rate" resolves to
        #    the single cumulative-cohort definition.
        Recogniser(
            name="cohort_conversion", priority=20, capability=CAP_PIPELINE,
            description="Cumulative cohort conversion KFI → Funded.",
            recognise=lambda r: _is_conversion(r.question),
            handle=lambda r: _route_conversion(
                r.question, r.spec_dict, history_model=r.resolve_history_model(),
                portfolio_id=r.portfolio_id, as_of=r.as_of)),

        # 3. Run-rate / scale-up extrapolation.
        Recogniser(
            name="forecast_extrapolation", priority=30,
            capability=CAP_ORIGINATION_FORECAST,
            description="Completion run-rate forecast and milestone solving.",
            recognise=lambda r: getattr(r.spec, "forecast_mode", None) == "extrapolation",
            handle=lambda r: _route_forecast(
                r.question, r.spec, r.spec_dict, client_id=r.client_id,
                run_id=r.run_id, output_root=r.output_root,
                pipeline_root=r.pipeline_root, history_model=r.resolve_history_model(),
                portfolio_id=r.portfolio_id, as_of=r.as_of)),

        # 4. Funded-balance attribution bridge.
        Recogniser(
            name="funded_bridge", priority=40, lens_aware=True,
            description="Governed funded-balance attribution waterfall.",
            recognise=lambda r: (bool(getattr(r.spec, "bridge_query", False))
                                 and not _is_aggregate_contribution_question(r.question)),
            handle=lambda r: _route_bridge(
                r.question, r.spec, r.spec_dict, client_id=r.client_id,
                run_id=r.run_id, output_root=r.output_root,
                portfolio_id=r.portfolio_id, as_of=r.as_of, semantics=r.semantics,
                source_lens=r.source_lens,
                interpretation=r.resolve_interpretation())),

        # 5. Static-pool cohort progression.
        Recogniser(
            name="cohort_progression", priority=50, capability=CAP_COHORTS,
            lens_aware=True,
            description="Metric progression across reporting dates by vintage.",
            recognise=lambda r: bool(getattr(r.spec, "cohort_progression", False)),
            handle=lambda r: _route_cohort_progression(
                r.question, r.spec, r.spec_dict, client_id=r.client_id,
                run_id=r.run_id, output_root=r.output_root,
                portfolio_id=r.portfolio_id, as_of=r.as_of,
                source_lens=r.source_lens)),

        # 6. ITL3 geographic concentration. Reads a dataframe, so it narrows.
        Recogniser(
            name="geo_exposure", priority=60, lens_aware=True,
            description="Funded exposure by UK ITL3 area.",
            recognise=lambda r: _is_geo_exposure(
                r.question, spec=r.spec, view=r.view,
                interpretation_provider=r.resolve_interpretation),
            handle=lambda r: _route_geo(
                r.question, r.spec_dict, client_id=r.client_id, run_id=r.run_id,
                frame_resolver=r.frame_resolver, portfolio_id=r.portfolio_id,
                as_of=r.as_of, source_lens=r.source_lens,
                interpretation=r.resolve_interpretation())),

        # 6b. Portfolio Risk Comparison — the governed workflow layer's second
        #     workflow. Recognition, scope resolution and every calculation
        #     live in mi_workflows.portfolio_risk_comparison; this entry only
        #     declares the route and its Business Semantics Registry terms.
        #     Ungated like the other funded-book routes: "there are not two
        #     governed scopes to compare" is a workflow-level controlled
        #     failure with its own explanation, not a CapabilityState.
        Recogniser(
            name=prc_mod.WORKFLOW_ID, priority=65, lens_aware=True,
            description=("Deterministic comparison of governed portfolio "
                         "scopes at one reporting date."),
            metadata={
                "workflow": prc_mod.WORKFLOW_ID,
                "bsr_workflow_tag": "portfolio_comparison",
                "bsr_axes_consumed": (
                    "analytical_role", "analytical_concept", "categories",
                    "default_aggregation", "weight_field", "share_basis",
                    "directionality", "portfolio_comparability", "confidence",
                    "rationale", "asset_applicability"),
                "comparison_basis": "portfolio_scopes_at_one_reporting_date",
            },
            recognise=_recognise_portfolio_comparison,
            handle=_route_portfolio_comparison),

        # 6c. Concentration Analysis — the governed workflow layer's third
        #     workflow. Recognition, scope resolution and every calculation
        #     live in mi_workflows.concentration_analysis; this entry only
        #     declares the route and its Business Semantics Registry terms.
        #     Registered AFTER geo_exposure and the risk-limit monitor's
        #     territory by construction: the workflow's own recognition defers
        #     ITL3 location questions, limit/covenant framings, grouped
        #     rankings and cross-portfolio comparisons to their owners.
        Recogniser(
            name=conc_mod.WORKFLOW_ID, priority=66, lens_aware=True,
            description=("Deterministic measurement of exposure distribution "
                         "across governed dimensions at one reporting date."),
            metadata={
                "workflow": conc_mod.WORKFLOW_ID,
                "bsr_dimension_category": conc_mod.CONCENTRATION_CATEGORY,
                "bsr_axes_consumed": (
                    "analytical_role", "analytical_concept", "categories",
                    "asset_applicability", "portfolio_comparability",
                    "confidence"),
                "concentration_bases": ("exposure", "count"),
            },
            recognise=_recognise_concentration,
            handle=_route_concentration),

        # 7-8. Governed composite answers. Checked before compare/evolution
        #      because both are narrower intents than a single-metric
        #      comparison; each returns None when it cannot answer.
        Recogniser(
            name="period_movement", priority=70, lens_aware=True,
            description="Month-on-month movement with attribution.",
            recognise=lambda r: _is_period_movement(r.question),
            handle=lambda r: _route_period_movement(
                r.question, r.spec, r.spec_dict, client_id=r.client_id,
                run_id=r.run_id, output_root=r.output_root,
                portfolio_id=r.portfolio_id, as_of=r.as_of,
                source_lens=r.source_lens,
                interpretation=r.resolve_interpretation())),
        Recogniser(
            name="portfolio_summary", priority=80, lens_aware=True,
            description="Current governed headline position.",
            recognise=lambda r: _is_portfolio_summary(r.question, r.spec),
            handle=lambda r: _route_portfolio_summary(
                r.question, r.spec, r.spec_dict, client_id=r.client_id,
                run_id=r.run_id, output_root=r.output_root,
                portfolio_id=r.portfolio_id, as_of=r.as_of,
                source_lens=r.source_lens,
                interpretation=r.resolve_interpretation())),

        # 8b. Governed Period Change Analysis — the first workflow layer built on
        #     the Business Semantics Registry. It sits AFTER the two composite
        #     routes above (which keep every question they already answer) and
        #     BEFORE temporal_compare, whose single-metric named-period
        #     comparison its recogniser explicitly declines. See
        #     ``mi_agent.period_change.recognition`` for the full deference rules
        #     and docs/period_change_analysis_workflow.md for why they are drawn
        #     where they are.
        Recogniser(
            name=_period_change.ROUTE_NAME, priority=85, lens_aware=True,
            description=("Governed period-change analysis across two portfolio "
                         "snapshots, driven by the Business Semantics Registry."),
            metadata={"business_semantics_workflow_tag": "period_change",
                      "registry": "config/business_semantics_registry.yaml",
                      "selection_policy": "config/period_change_selection.yaml"},
            recognise=_period_change.recognise_request,
            handle=lambda r: _period_change.route_period_change(
                r.question, r.spec, r.spec_dict, client_id=r.client_id,
                run_id=r.run_id, output_root=r.output_root,
                portfolio_id=r.portfolio_id, as_of=r.as_of,
                source_lens=r.source_lens,
                semantics_context=dict(r.semantics_context or {}),
                semantics=dict(r.semantics or {}),
                view=r.view,
                recognition=r.recalled_recognition(
                    _period_change.RECOGNITION_KEY),
                interpretation=r.resolve_interpretation())),

        # 9. Cross-period comparison.
        Recogniser(
            name="temporal_compare", priority=90,
            description="Governed comparison of two reporting periods.",
            recognise=lambda r: getattr(r.spec, "temporal_mode", None) == "compare",
            # NO `view=`. The dataset is the question's, and the route asks
            # `workspace.resolve_dataset` for it. Leaving the parameter here
            # would be a live wire back to the tab.
            handle=lambda r: _route_compare(
                r.question, r.spec_dict, client_id=r.client_id,
                run_id=r.run_id, output_root=r.output_root,
                pipeline_root=r.pipeline_root,
                portfolio_id=r.portfolio_id, as_of=r.as_of,
                interpretation=r.resolve_interpretation())),

        # 10. Contractual risk limits.
        Recogniser(
            name="risk_limits", priority=100, capability=CAP_RISK,
            description="Contractual concentration limits and headroom tests.",
            recognise=lambda r: bool(getattr(r.spec, "risk_limit_query", None)),
            handle=lambda r: _route_risk(
                r.question, r.spec, r.spec_dict, client_id=r.client_id,
                run_id=r.run_id, output_root=r.output_root,
                portfolio_id=r.portfolio_id, as_of=r.as_of)),

        # 11. Time series / evolution.
        Recogniser(
            name="evolution", priority=110,
            description="Metric evolution across governed reporting periods.",
            recognise=lambda r: _is_evolution(r.question, r.spec),
            handle=lambda r: _route_evolution(
                r.question, r.spec, r.spec_dict, client_id=r.client_id,
                run_id=r.run_id, output_root=r.output_root,
                pipeline_root=r.pipeline_root,
                portfolio_id=r.portfolio_id, as_of=r.as_of,
                semantics=r.semantics,
                interpretation=r.resolve_interpretation())),
    ])
    return registry


_register_default_recognisers(REGISTRY)


def try_route(question: str, *, portfolio_id: Optional[str], view: str,
              output_root: Optional[str], pipeline_root: Optional[str],
              semantics: Dict[str, Any], history_model: Optional[Dict[str, Any]] = None,
              history_model_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
              as_of: Optional[str] = None,
              source_lens: Optional[Any] = None,
              frame_resolver: Optional[Callable[[str, Optional[str]], Any]] = None,
              extra_filters: Optional[Dict[str, Any]] = None,
              parsed: Optional[Any] = None,
              registry: Optional[RecogniserRegistry] = None,
              capability_resolver: Optional[Callable[[Optional[str]], Any]] = None,
              base_frame_resolver: Optional[Callable[[str, Optional[str]], Any]] = None,
              ) -> Optional[Dict[str, Any]]:
    """Route a question to a governed capability, or return ``None`` to defer to
    the point-in-time MI Agent path.

    ``history_model_provider`` defers the historical completion model until a
    handler that genuinely needs it asks for it (scenario / cohort conversion /
    run-rate forecast). Building that model replays every retained weekly
    extract, so passing it eagerly — as the serving path used to — charged every
    MI and Copilot question for analysis almost none of them use. Pass
    ``history_model`` instead to supply an already-built model; it still wins.

    ``parsed`` is the SINGLE :class:`~mi_agent.parsed_question.ParsedQuestion`
    for this request. The caller (``mi_service``) parses once and passes it here
    and to the workflow, so routing and execution can never disagree about the
    spec. It is optional only so existing direct callers and tests keep working:
    when omitted this parses once itself, which is still one parse per request.

    Never raises for analytics issues — the caller wraps this defensively.
    """
    client_id, run_id = _split_portfolio(portfolio_id)
    reg = registry if registry is not None else REGISTRY

    if parsed is None:
        try:
            parsed = ParsedQuestion.parse(question, semantics)
        except Exception:  # noqa: BLE001 - never block the normal path on a parse hiccup
            return None
    # Caller-supplied filters (UI drill-through / req.filters) are merged onto
    # the shared parse, so a routed answer is scoped identically to a
    # within-snapshot one — and only ever merged once.
    parsed.merge_filters(extra_filters)
    spec = parsed.spec

    # THE ANALYTICAL INTENT BOUNDARY.
    #
    # Before any recogniser is consulted, read the question for its governed
    # analytical FAMILY and OPERATIONS, and settle the governed intent flags the
    # parser left open. This is routing, not analysis: every flag it can set is
    # one an existing route already recognises, and it never overrides a flag the
    # parser has already settled.
    #
    # It exists because the same governed question reached two different answers
    # depending on wording — "which limits have the least headroom?" reached the
    # limits route; "where are we closest to our limits?" did not, and the funded
    # executor answered it with weighted average LTV by region. The family is the
    # same; only the phrasing differed. See
    # ``mi_workflows/analytical/intent.py`` for the six families.
    # GOVERNED SPAN OWNERSHIP, resolved ONCE for everything on this path that
    # reads the sentence for a vocabulary of its own. The book's categorical
    # values are the same catalogue the parser was handed; a span already
    # claimed as one of them may not create a second semantic claim from the
    # tokens inside it. Measured on brokers named "Growth Partners" and "London
    # Bridge Loans": one was read as a movement question, the other as a funded
    # bridge, and both refused. `mi_agent.categorical_spans` owns the rule.
    def _values_for_recognition() -> Any:
        if "value" not in _ownership_memo:
            from mi_agent import execution_receipt as _receipt

            value = None
            try:
                frame = None
                if base_frame_resolver is not None:
                    frame = base_frame_resolver(view, portfolio_id)
                elif frame_resolver is not None:
                    frame = frame_resolver(view, portfolio_id)
                if frame is not None:
                    value = _receipt.book_values(frame, semantics)
            except Exception as exc:  # noqa: BLE001 - no catalogue, old routing
                _logger.info("book value catalogue unavailable: %s", exc)
            _ownership_memo["value"] = value
        return _ownership_memo["value"]

    _ownership_memo: Dict[str, Any] = {}

    def _owned_question() -> str:
        values = _values_for_recognition()
        if not values:
            return question
        try:
            from mi_agent.categorical_spans import mask_value_spans

            return mask_value_spans(question, values)
        except Exception:  # noqa: BLE001
            return question

    try:
        # The INTENT boundary owns no book field: every family word it matches
        # that lies inside a claimed value span belongs to the value.
        analytical_reading, analytical_flags = analytical_intent.settle(
            _owned_question(), spec)
    except Exception as exc:  # noqa: BLE001 - the boundary must never break routing
        _logger.warning("analytical intent boundary failed: %s", exc)
        analytical_reading, analytical_flags = None, {}

    # PHASE 1G §9 — THE SEMANTIC HANDOFF for the routed path.
    #
    # Phase 1F found that a routed question never builds a
    # `QuestionInterpretation`: the single production construction site is on
    # the point-in-time path, which routing bypasses. A compositional plan may
    # read the contract and nothing else, so the contract has to exist here
    # before a handler can be converted onto it.
    #
    # Assembled from the SAME spec and facets the receipt layer already
    # produces, and from the SAME owners — it re-interprets nothing and decides
    # nothing. The registry and the caller's workspace selection go in, so the
    # claim carries the governed identity and the provenance that decides
    # precedence, rather than the pre-1E/pre-1G readings.
    #
    # NOTHING READS IT YET. It is carried so the first route conversion has a
    # contract to plan from; every handler below is byte-for-byte unaffected.
    def _build_interpretation() -> Any:
        from question_interpretation.projection import from_parts as _qi_build

        from mi_agent import execution_receipt as _receipt

        frame = None
        if base_frame_resolver is not None:
            frame = base_frame_resolver(view, portfolio_id)
        elif frame_resolver is not None:
            frame = frame_resolver(view, portfolio_id)
        # `frame.columns` is a pandas Index, and `Index or []` RAISES rather
        # than falling back — "The truth value of a Index is ambiguous". The
        # first cut of this wiring wrote exactly that, so the provider raised on
        # every routed question and the try/except around it returned None
        # silently: a construction site that never constructed. Tested only
        # against a lambda, it looked wired. `test_the_routed_path_really_builds
        # _a_contract` now exercises it against a real frame.
        cols = getattr(frame, "columns", None)
        columns = list(cols) if cols is not None else None
        dim_terms = _receipt.requested_dimension_terms(question, semantics, columns)
        facets = _receipt.detect_requested_facets(
            question, semantics, frame=frame, requested_dimensions=dim_terms)
        registry_for_scope = None
        try:
            from . import portfolio_context as _ctx

            registry_for_scope = _ctx.build_registry(frame)
        except Exception as exc:  # noqa: BLE001 - identity never breaks routing
            _logger.info("governed registry unavailable for interpretation: %s", exc)
        # GOVERNED SPAN OWNERSHIP — the book's own category values, so the
        # contract's SourceScopeClaim cannot re-read a span already claimed as a
        # categorical value. Same catalogue the parser was handed; the rule is
        # `mi_agent.categorical_spans`'s, not this site's.
        try:
            values_for_scope = _receipt.book_values(frame, semantics)
        except Exception as exc:  # noqa: BLE001 - no catalogue, old reading
            _logger.info("book value catalogue unavailable for scope: %s", exc)
            values_for_scope = None
        return _qi_build(question, spec=spec, facets=list(facets),
                         dim_terms=dim_terms, semantics=semantics,
                         registry=registry_for_scope, caller_scope=source_lens,
                         available_values=values_for_scope)

    # THE CONCEPT-MERGE ARM, off by default and independent of the free-form
    # parser. It runs HERE — after the deterministic contract exists and before
    # any recogniser has seen the spec — so a concept the model recovers is
    # routed on, rather than being added to a contract routing has already
    # decided against. The interpretation it merges into is the DETERMINISTIC
    # one, built from the spec before any fill.
    concept_merge_evidence = None
    try:
        from . import concept_merge_arm as _merge_arm

        if _merge_arm.enabled():
            concept_merge_evidence = _merge_arm.apply(
                question, spec, semantics,
                interpretation=_build_interpretation(),
                available_values=_values_for_recognition(),
                available_columns=getattr(parsed, "available_columns", None))
    except Exception as exc:  # noqa: BLE001 - the arm never fails a request
        _logger.info("concept merge arm skipped: %s: %s", type(exc).__name__, exc)
    if concept_merge_evidence is not None:
        # CARRIED ON THE PARSE METADATA, which is already the channel for "how
        # was this contract arrived at". The point-in-time path returns no
        # routed envelope to stamp, and the spec it executes is the same object
        # this arm just changed — so the evidence has to travel with the parse,
        # not with a route.
        try:
            parsed.meta["conceptMerge"] = concept_merge_evidence
        except Exception:  # noqa: BLE001 - evidence never fails a request
            pass

    request = RouteRequest(
        question=question, spec=spec, spec_dict=spec.to_dict(),
        available_values=_values_for_recognition(),
        semantics=semantics, view=view, client_id=client_id, run_id=run_id,
        portfolio_id=portfolio_id, output_root=output_root,
        pipeline_root=pipeline_root, history_model=history_model,
        history_model_provider=history_model_provider, as_of=as_of,
        source_lens=source_lens, frame_resolver=frame_resolver,
        base_frame_resolver=base_frame_resolver,
        interpretation_provider=_build_interpretation,
        parse_meta=parsed.meta, semantics_context=parsed.semantics_context)

    # The governed scope this request runs in, resolved lazily and AT MOST ONCE
    # per request — capability resolution can touch storage, so it must not run
    # for a question no gated recogniser matched.
    gate_cache: Dict[str, Any] = {}

    def _capability_state(capability: str) -> Any:
        if capability not in gate_cache:
            try:
                lens = _resolve_lens(question, source_lens)
                context_id = _portfolio_lens.context_id(lens)
            except Exception:  # noqa: BLE001
                context_id = None
            gate_cache[capability] = resolve_capability_state(
                capability, context_id, resolver=capability_resolver)
        return gate_cache[capability]

    for recogniser, _verdict in reg.candidates(request):
        # Governed capability gate — the SAME resolution React uses. A capability
        # that genuinely does not apply to this scope is explained, not attempted.
        if recogniser.capability:
            state = _capability_state(recogniser.capability)
            # ``None`` means the gate could not be resolved (an infrastructure
            # problem). Attempting the route is the honest fallback: claiming the
            # capability does not apply would be a different, false statement.
            if state is not None and not getattr(state, "enabled", True):
                return _disclose_lens_scope(
                    _capability_unavailable_envelope(request, recogniser, state),
                    question, source_lens)
        try:
            envelope = recogniser.handle(request)
        except Exception as exc:  # noqa: BLE001 - fails CLOSED, never 500s
            # THE CLAIM BOUNDARY. This used to `continue`, so a route that broke
            # partway through its own analysis handed the question to the next
            # candidate, which answered it as a different analysis. Measured: a
            # fault injected after `period_change_analysis` had already run the
            # governed period-change produced a `temporal_compare` refusal about
            # ranking, with no receipt and no trace that the claimed route had
            # ever run.
            #
            # The distinction is the registry's own, not a list of route names.
            # `recognise` is pre-claim and still fails open — a recogniser that
            # raises is skipped inside `RecogniserRegistry.candidates`, and a
            # route that does not apply says so by returning None from `handle`,
            # which still falls through. Entering `handle` is the claim: the
            # registry has selected this route to answer, so its failure is a
            # failure of the answer, not an offer to let something else try.
            #
            # Measured before changing it: across all 882 corpus questions, zero
            # handlers raise. Nothing on the normal path relies on this.
            _logger.exception("route %s failed after claiming the question",
                              recogniser.name)
            return _disclose_lens_scope(
                _execution_failure_envelope(request, recogniser), question,
                source_lens)
        if envelope is not None:
            _stamp_analytical_intent(envelope, analytical_reading, analytical_flags)
            return _disclose_lens_scope(envelope, question, source_lens)
    return None


def _stamp_analytical_intent(envelope: Dict[str, Any], reading, flags) -> None:
    """Publish what the boundary recognised, on the answer it shaped.

    Evidence, not decoration: a reader (and the fail-closed check downstream)
    must be able to see which family a question was read as, and which governed
    flag — if any — was settled on its behalf.
    """
    if not isinstance(envelope, dict) or reading is None or not reading.recognised:
        return
    meta = envelope.setdefault("metadata", {})
    if isinstance(meta, dict):
        block = reading.to_dict()
        if flags:
            block["flagsApplied"] = dict(flags)
        meta["analyticalIntent"] = block
