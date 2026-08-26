"""mi_agent_api/period_change_route — the Period Change Analysis adapter.

The seam between the governed workflow (``mi_agent.period_change``, which knows
nothing about datasets, HTTP or presentation) and the platform (which knows
nothing about business semantics). It does three things and nothing else:

1. **supplies snapshots.** ``evolution.funded_frames`` is the EXISTING governed
   per-period frame service — the same one the Evolution tab, the funded bridge,
   the temporal-compare route and the movement summary already read. Nothing new
   walks storage, and the resolved portfolio lens is applied with the same
   ``chat_routing._apply_lens_filter`` those routes use, so a period-change
   answer covers exactly the portfolios every other routed answer would;
2. **runs the workflow** and maps a controlled ``PeriodChangeFailure`` onto the
   repository's existing error taxonomy (``trakt_core.errors.ErrorCode``);
3. **renders** the structured result through the existing chat envelope and
   artifact builders. The full governed result travels intact under the
   additive ``periodChange`` key, so a channel that wants the structure gets the
   structure and one that wants a table gets a table — from the same numbers.

The adapter never calculates. Every figure it renders was computed by the
workflow; if a number is not in the result, it is not in the answer.
"""

from __future__ import annotations

from dataclasses import replace

from mi_agent import period_request as _period_request

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from trakt_core.errors import ErrorCode

from mi_agent.period_change import (
    PeriodChangeRequest,
    recognise,
    run_period_change_analysis,
)
from mi_agent.period_change.models import (
    BRIDGE_STATUS_AVAILABLE,
    FAIL_AMBIGUOUS_PERIOD_RANGE,
    FAIL_CROSS_TENANT_ACCESS,
    FAIL_IDENTICAL_SNAPSHOTS,
    FAIL_INSUFFICIENT_SNAPSHOTS,
    FAIL_NO_ELIGIBLE_FIELDS,
    FAIL_PORTFOLIO_ABSENT_AT_PERIOD,
    FAIL_REGISTRY_UNAVAILABLE,
    FAIL_REVERSED_PERIOD_RANGE,
    UNIT_COUNT,
    UNIT_CURRENCY,
    UNIT_PERCENTAGE_POINT,
    UNIT_RATIO,
    WORKFLOW_ID,
    PeriodChangeFailure,
    PeriodChangeResult,
    PortfolioScopeRef,
    SnapshotFrame,
)

logger = logging.getLogger("mi_agent_api.period_change_route")

#: The route id. Matches the workflow identifier so a routed answer, an audit
#: line and the workflow all name the same capability.
ROUTE_NAME = WORKFLOW_ID

#: Controlled failure reason → the repository's existing error taxonomy. The
#: workflow keeps its own vocabulary (it must not import the API layer); the
#: mapping lives here, once.
FAILURE_ERROR_CODES: Dict[str, str] = {
    FAIL_INSUFFICIENT_SNAPSHOTS: ErrorCode.NO_MATCHING_RECORDS,
    FAIL_PORTFOLIO_ABSENT_AT_PERIOD: ErrorCode.NO_MATCHING_RECORDS,
    FAIL_NO_ELIGIBLE_FIELDS: ErrorCode.NO_MATCHING_RECORDS,
    FAIL_AMBIGUOUS_PERIOD_RANGE: ErrorCode.AMBIGUOUS_QUESTION,
    FAIL_REVERSED_PERIOD_RANGE: ErrorCode.INVALID_INPUT,
    FAIL_IDENTICAL_SNAPSHOTS: ErrorCode.UNSUPPORTED_QUESTION,
    FAIL_CROSS_TENANT_ACCESS: ErrorCode.PORTFOLIO_NOT_AUTHORISED,
    FAIL_REGISTRY_UNAVAILABLE: ErrorCode.INTERNAL_ERROR,
}


# --------------------------------------------------------------------------- #
# Snapshot supply — the EXISTING governed per-period frame service
# --------------------------------------------------------------------------- #
class PopulationNotApplied(Exception):
    """A governed row predicate could not be applied to every snapshot.

    Raised, never returned, and never swallowed into an unnarrowed frame: a
    comparison whose population failed on one of its two dates is not a
    comparison with a missing filter, it is no comparison at all. The same
    fail-closed shape `_filtered_funded_evo` already raises for the funded
    series, for the same reason.
    """

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def build_snapshots(output_root: Optional[str], client_id: str, *,
                    to_run_id: Optional[str] = None,
                    lens: Any = None,
                    population: Sequence[Any] = (),
                    semantics: Optional[Dict[str, Any]] = None,
                    evidence_out: Optional[List[Any]] = None,
                    ) -> Tuple[SnapshotFrame, ...]:
    """Governed portfolio snapshots, oldest → newest, for a client and lens.

    ``lens`` is a RESOLVED portfolio lens (``chat_routing._resolve_lens``), whose
    filters carry the registry's explicit portfolio-id list. Narrowing uses the
    same ``_apply_lens_filter`` the funded-bridge and movement routes use, so a
    period-change answer covers exactly the portfolios every other routed answer
    would cover for the same lens.

    ``population`` are the governed `Predicate` objects the compositional plan
    selected — `SELECT_POPULATION(kind=row_predicates)` from `RowPredicateClaim`,
    the same objects `_filtered_funded_evo` receives. THE POPULATION IS APPLIED
    HERE, in the one place every snapshot this route compares is built, so a
    predicate cannot reach one date and miss the other. Applying it in the
    caller, per date, is how a filtered comparison comes to open on one
    population and close on another.

    Execution goes through `population.apply_population` — the single governed
    meaning of a predicate — and FAILS CLOSED on any snapshot it cannot narrow.
    """
    from . import chat_routing
    from . import contract_scope as _scope
    from . import evolution as evolution_mod
    from mi_agent import portfolio_scope as scope_mod

    frames = evolution_mod.funded_frames(output_root, client_id, to_run_id)
    snapshots: List[SnapshotFrame] = []
    for frame in frames:
        source_df = frame.get("df")
        if source_df is None:
            continue
        df = (chat_routing._apply_lens_filter(source_df, lens)
              if lens is not None else source_df)
        if population:
            from mi_agent import population as _population

            df, evidence = _population.apply_population(df, population, semantics)
            if df is None or not evidence.is_usable:
                raise PopulationNotApplied(
                    "; ".join(evidence.unavailable
                              or [evidence.blocked_reason or "unknown reason"]))
            if evidence_out is not None:
                # KEYED BY SNAPSHOT, because the receipt publishes counts for
                # the TWO snapshots the comparison resolved to, not for every
                # snapshot the book has. An unkeyed list would pair a filtered
                # count for two dates with unfiltered counts for five.
                evidence_out.append((str(frame.get("run_id")), evidence))
        source = frame.get("source")
        snapshots.append(SnapshotFrame(
            snapshot_id=str(frame.get("run_id")),
            reporting_date=frame.get("reporting_date"),
            frame=df,
            # A label, never a path: a result and a log must not carry storage
            # layout (the discipline trakt_core.envelope.SnapshotRef applies).
            dataset_label=_dataset_label(source),
            dataset_reference=str(frame.get("run_id")),
            row_count=int(len(df)),
            # Presence is read from the UNNARROWED snapshot: a portfolio that
            # the lens selected but that this reporting date never contained
            # must be reported as absent, not as a snapshot full of nulls.
            portfolio_ids=_portfolio_ids(source_df, scope_mod)))
    return tuple(snapshots)


def _dataset_label(source: Any) -> Optional[str]:
    if not source:
        return None
    text = str(source)
    return text.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def _portfolio_ids(df: Any, scope_mod: Any) -> Tuple[str, ...]:
    """Source portfolios present in a frame, via the governed scope helper."""
    from trakt_core.portfolio import FIELD_PORTFOLIO_ID

    try:
        records = scope_mod.portfolio_records(df)
    except Exception:  # noqa: BLE001 - absent provenance is not an error
        return ()
    return tuple(sorted({str(r.get(FIELD_PORTFOLIO_ID)) for r in records
                         if r.get(FIELD_PORTFOLIO_ID)}))


def _registry_for(snapshots: Sequence[SnapshotFrame],
                  client_id: Optional[str]) -> Any:
    """The governed portfolio registry for the closing snapshot, or None.

    Read from the CLOSING frame because that is the book the analysis reports
    on; a portfolio onboarded since the opening date is registered there.
    """
    if not snapshots:
        return None
    try:
        from mi_agent.portfolio_scope import registry_for_frame

        return registry_for_frame(snapshots[-1].frame, client_id=client_id)
    except Exception:  # noqa: BLE001 - metadata must never break an answer
        logger.exception("portfolio registry unavailable for client %r", client_id)
        return None


def scope_ref_from_lens(lens: Any, *, tenant_id: Optional[str] = None,
                        asset_classes: Sequence[str] = (),
                        registry: Any = None) -> PortfolioScopeRef:
    """The workflow's scope reference for a RESOLVED portfolio lens.

    A resolved lens carries the registry's explicit portfolio-id list in its
    filters (``chat_routing._resolve_lens``), never a type string, so the scope
    the workflow records is exactly the scope the frames were narrowed to.

    The asset classes come from the GOVERNED PORTFOLIO REGISTRY for those ids —
    the single source of truth onboarding writes. An explicit ``asset_classes``
    argument still wins, so a caller that knows its book can state it, but the
    registry means no caller HAS to: previously this argument had no production
    caller at all, so every analysis ran with an unknown asset class and only
    cross-asset semantics were ever admitted.
    """
    from mi_agent import portfolio_lens as lens_mod

    filters = dict(getattr(lens, "filters", None) or {})
    ids = filters.get(lens_mod.SOURCE_ID_FIELD) or ()
    if isinstance(ids, str):
        ids = (ids,)
    context = None
    try:
        context = lens_mod.context_id(lens) if lens is not None else None
    except Exception:  # noqa: BLE001 - a lens fault must not fail the analysis
        context = None
    resolved = resolve_asset_classes(asset_classes)
    if not resolved and registry is not None:
        try:
            resolved = resolve_asset_classes(
                registry.asset_classes(tuple(str(i) for i in ids) or None))
        except Exception:  # noqa: BLE001 - metadata must never break an answer
            logger.exception("asset-class resolution failed for scope %r", context)
            resolved = ()
    return PortfolioScopeRef(
        tenant_id=tenant_id, context_id=context,
        label=getattr(lens, "label", None),
        portfolio_ids=tuple(str(i) for i in ids),
        asset_classes=resolved)


#: BSR asset-applicability values, so a caller-supplied class is validated
#: against the registry's controlled vocabulary rather than passed through.
KNOWN_ASSET_CLASSES: Tuple[str, ...] = (
    "commercial_real_estate", "corporate", "equipment_leasing", "equity_release",
    "residential_mortgage", "sme",
)


def resolve_asset_classes(explicit: Sequence[str] = ()) -> Tuple[str, ...]:
    """The BSR asset classes a period-change analysis may use.

    Deliberately conservative: an asset class is used ONLY when a caller states
    it. The MI layer has no governed asset-class signal at query time —
    ``snapshots.portfolio_risk_type`` exists but *defaults* to ``erm`` with no
    evidence, so treating its output as an asset classification would admit
    equity-release fields onto every book, which is precisely what §9 forbids.

    With no stated class the result is ``()``, and the registry then admits only
    ``cross_asset`` entries (92 of the 106 period-change entries). The extension
    point is this function's argument: a governed portfolio-metadata asset class,
    or an API caller that knows its book, supplies it and the asset-specific
    entries become eligible with no other change.
    """
    return tuple(a for a in explicit if a in KNOWN_ASSET_CLASSES)


# --------------------------------------------------------------------------- #
# Direct invocation — the governed capability, callable without the chat path
# --------------------------------------------------------------------------- #
def analyse_period_change(*, client_id: str, output_root: Optional[str],
                          question: str = "",
                          mode: Optional[str] = None,
                          requested_fields: Sequence[str] = (),
                          requested_concepts: Sequence[str] = (),
                          period_request: Any = None,
                          to_run_id: Optional[str] = None,
                          scope: Any = None,
                          tenant_id: Optional[str] = None,
                          authorised_portfolio_ids: Sequence[str] = (),
                          include_bridge: bool = True,
                          asset_classes: Sequence[str] = (),
                          source_id: Optional[str] = None,
                          ) -> PeriodChangeResult:
    """Run the workflow for a client and scope. Raises ``PeriodChangeFailure``.

    The entry point for a direct Python caller, an API route or a scheduled job.
    The chat route below is one caller among several, not the only way in.
    """
    from mi_agent.period_change.periods import PeriodRequest
    from mi_agent.period_change.models import MODE_PORTFOLIO_OVERVIEW

    snapshots = build_snapshots(output_root, client_id, to_run_id=to_run_id,
                                lens=scope)
    scope_ref = scope_ref_from_lens(scope, tenant_id=tenant_id,
                                    asset_classes=asset_classes,
                                    registry=_registry_for(snapshots, client_id))
    request = PeriodChangeRequest(
        question=question, mode=mode or MODE_PORTFOLIO_OVERVIEW,
        period_request=period_request or PeriodRequest(),
        requested_fields=tuple(requested_fields),
        requested_concepts=tuple(requested_concepts),
        scope=scope_ref, include_bridge=include_bridge,
        authorised_portfolio_ids=tuple(authorised_portfolio_ids),
        source_id=source_id)
    return run_period_change_analysis(request, snapshots)


# --------------------------------------------------------------------------- #
# Recognition — reads the SINGLE parse on the route request
# --------------------------------------------------------------------------- #
#: The key this route's pre-claim reading is carried under.
RECOGNITION_KEY = "period_change"


def recognise_request(req: Any) -> Any:
    """``Recognition`` for the governed recogniser registry.

    THE READING IS KEPT. This is the one place the question is read to decide
    whether the route owns it, and the `PeriodChangeIntent` it produces carries
    the mode, the requested fields and the period request the handler needs.
    It used to be discarded and rebuilt inside the handler from the same
    sentence — the same function, run twice, the second time after the route
    had already claimed the question.
    """
    from .recogniser_registry import Recognition

    intent = recognise(req.question, spec=req.spec, view=req.view,
                       semantics_context=req.semantics_context)
    remember = getattr(req, "remember_recognition", None)
    if remember is not None:
        remember(RECOGNITION_KEY, intent)
    if not intent.matched:
        return Recognition.no(intent.reason)
    return Recognition.yes(reason=f"{intent.reason}:{intent.mode}")


# --------------------------------------------------------------------------- #
# Chat route
# --------------------------------------------------------------------------- #
def route_period_change(question: str, spec: Any, spec_dict: Dict[str, Any], *,
                        client_id: str, run_id: Optional[str],
                        output_root: Optional[str],
                        portfolio_id: Optional[str], as_of: Optional[str],
                        source_lens: Any = None,
                        semantics_context: Optional[Dict[str, Any]] = None,
                        semantics: Optional[Dict[str, Any]] = None,
                        view: str = "funded",
                        lens: Any = None,
                        interpretation: Any = None,
                        recognition: Any = None) -> Optional[Dict[str, Any]]:
    """The governed period-change answer, or ``None`` to defer to the next route.

    `recognition` is THIS ROUTE'S OWN PRE-CLAIM READING, produced by
    `recognise_request` before the registry entered the handler and carried in
    on the request. It is not recomputed here: once the registry has claimed
    the question, the handler does not go back to the sentence to find out what
    it owns.

    NO READING, NO ANSWER FROM THIS ROUTE — the rule Conversion 1 set for the
    population and Conversion 2 for the window. A caller that skipped
    recognition gets a deferral, not a second recogniser hidden in the handler.
    """
    from . import chat_routing
    from . import contract_scope as _scope

    intent = recognition
    if intent is None or not intent.matched:
        return None

    # THE POPULATION COMES FROM THE CONTRACT. This called
    # `chat_routing._resolve_lens(question, source_lens)` — a second reading of
    # the sentence for a scope the contract had already claimed, and the last
    # population owner left downstream of interpretation in this estate.
    # Measured over 882 corpus questions before the switch, and again with a
    # workspace selection present so caller precedence was actually exercised:
    # the contract-derived lens and the resolver agree every time.
    resolved_lens = (lens if lens is not None
                     else _scope.lens_from_contract(interpretation))
    if resolved_lens is None:
        # NO SCOPE CLAIM, NO ANSWER FROM THIS ROUTE. Conversion 1's rule, and
        # for its reason: keeping the resolver here as a fallback would leave a
        # second population owner reachable exactly when the first one failed.
        return None
    # THE POPULATION THE READER STATED, PLANNED FROM THE CONTRACT.
    #
    # Same chain the funded evolution route uses and the same objects:
    # `RowPredicateClaim` -> `SELECT_POPULATION(kind=row_predicates)` ->
    # `Predicate` -> `governed_predicate_mask`. This route reads no filter
    # meaning of its own; it asks the plan what rows the question selected, and
    # a route-local reading of `spec.filters` would be a second owner of the
    # answer the contract already carries.
    from . import analytical_plan as _plan

    _population = _plan.row_predicates(_plan.row_predicate_step(interpretation))
    _pop_evidence: List[Any] = []
    try:
        snapshots = build_snapshots(output_root, client_id, to_run_id=run_id,
                                    lens=resolved_lens,
                                    population=_population,
                                    # THE MI SEMANTICS, not `semantics_context`.
                                    # `governed_predicate_mask` resolves the
                                    # field, its domain and its scale from this
                                    # dict; handed the registry context (empty
                                    # today) it compares a stored LTV RATIO
                                    # against 50 and narrows every snapshot to
                                    # nothing. Same object the funded evolution
                                    # route passes to the same call.
                                    semantics=semantics,
                                    evidence_out=_pop_evidence)
    except PopulationNotApplied as exc:
        # THE REQUESTED POPULATION WAS NOT APPLIED, SO NOTHING IS ANSWERED.
        # Naming it is the point: a reader who asked about one population must
        # never be handed the whole book with the difference left unsaid.
        message = (f"I understood the population you asked about, but it could "
                   f"not be applied to both snapshots of this comparison "
                   f"({exc.detail}). I have not compared the whole book "
                   f"instead.")
        return chat_routing._envelope(
            ok=False, question=question, spec=spec_dict, artifacts=[],
            answer=message, error=message, route="period_change",
            warnings=[message])
    scope_ref = scope_ref_from_lens(
        resolved_lens, registry=_registry_for(snapshots, client_id))

    # The bridge is computed for every broad question, and for a narrow one only
    # when the question actually asked what drove the movement. Reconciling the
    # book to answer "how did LTV change?" is work nobody asked for.
    from mi_agent.period_change.models import MODE_REQUESTED_METRIC

    # P1C. The ranking intent is resolved BEFORE the analysis runs, because it
    # decides what the analysis must cover: asking "which region grew the most?"
    # and receiving an analysis that never looked at geography cannot produce a
    # ranked answer, and re-running afterwards would compute the book twice.
    rank_intent = resolve_rank_intent(interpretation)
    if rank_intent.refusal is not None:
        return _rank_refusal_envelope(question, spec_dict, rank_intent)

    # NO IMPLICIT COMPARISON PERIOD.
    #
    # A ranked MOVEMENT needs two dates. The recogniser will happily supply
    # "latest versus previous" when the question names none, and while that
    # default was masked by a false dimension refusal it never surfaced; with
    # the refusal fixed it would have started answering "which region grew the
    # most?" over a window the reader never asked for.
    #
    # The governed ruling is that a missing element is CLARIFIED, not invented.
    # The contract says whether a period was named — `comparison_periods` for an
    # explicit pair, `window_periods` for a span — and neither is guessed here.
    #
    # THE RULE IS THE ROUTE'S, NOT THE RANKED SUB-CASE'S. It was gated on
    # `rank_intent.requested`, which left the NARRATIVE half of the same route
    # inventing the same default: "What changed?" names no period, and the
    # recogniser's latest-versus-previous silently became the reader's window.
    # Every answer this route gives is a comparison between two dates, so the
    # question of which two dates is never optional here. Nothing about the
    # rule changes — only the false premise that it applied to ranking alone.
    if interpretation is not None:
        _time = getattr(interpretation, "time", None)
        _named = bool(getattr(_time, "comparison_periods", None)
                      or getattr(_time, "window_periods", None))
        if not _named:
            message = (
                (f"I can rank {rank_intent.term} by movement, but this question "
                 if rank_intent.requested else
                 "I can report what changed, but this question ")
                + f"names no period to compare over, and I have not chosen one "
                f"for you. Tell me the window — for example “since last month”, "
                f"“over the last 3 months”, or two named months.")
            return chat_routing._envelope(
                ok=False, question=question, spec=spec_dict, artifacts=[],
                answer=message, error=message, route="period_change",
                warnings=[message])

    mode, requested_fields = intent.mode, intent.requested_fields
    if rank_intent.requested:
        # Requested-metric mode is the governed way to say "analyse THIS field".
        # The ranked dimension is exactly that: a field the reader named.
        mode = MODE_REQUESTED_METRIC
        # EVERY governed field the term could bind to, not the primary alone.
        # Passing only the primary is what made a book carrying the ALTERNATE
        # answer "region is not a governed period-change dimension for this
        # book" — a false statement, and canary defect D1.
        requested_fields = (rank_intent.field,) + tuple(rank_intent.alt_fields)

    # HONOUR THE STATED PERIOD, OR CLARIFY. A question naming "this year" that
    # is answered over the latest month has had a declared element replaced.
    # Disclosing the narrower window in the prose is not honouring it: the
    # reader asked about a year. Where the snapshots DO reach back far enough
    # the span is honoured by opening the comparison at that snapshot; where
    # they do not, the question is clarified and no shorter window is
    # substituted. The same code answers once more history is loaded.
    period_request = intent.period_request
    # THE SPAN COMES FROM THE CONTRACT. `TimeClaim.window_periods` exists for
    # this exact read — its own docstring names `requested_span(question)` here
    # as "a second read of the sentence for a fact the contract had already
    # claimed". Conversion 2 closed against it; this route had not.
    # Measured over 882 corpus questions before the switch: the two agree 882
    # times and disagree none.
    span = _period_request.span_from_claim(getattr(interpretation, "time", None))
    if span is not None and not (period_request.requested_start
                                 or period_request.requested_end):
        if len(snapshots) > span.periods:
            # HONOUR: open the comparison at the snapshot the span reaches back
            # to. The resolver takes a calendar token, and a reporting date is
            # the governed identity of a snapshot, so its year-month is that
            # token.
            opening_token = str(snapshots[-1 - span.periods].reporting_date or "")[:7]
            if opening_token:
                period_request = replace(period_request,
                                         requested_start=opening_token,
                                         relative_mode=None)
        elif len(snapshots) >= 2:
            # CLARIFY: there is history, but not as far back as the question
            # asked. Only a SPAN problem belongs to this guard — a book with
            # fewer than two snapshots cannot compare anything at all, and the
            # existing controlled failure already says so with its error
            # taxonomy attached. Firing here would replace a classified failure
            # with an unclassified one.
            message = _period_request.clarification(span, len(snapshots))
            return chat_routing._envelope(
                ok=False, question=question, spec=spec_dict, artifacts=[],
                answer=message, error=message, route="period_change",
                warnings=[message])

    request = PeriodChangeRequest(
        question=question, mode=mode,
        period_request=period_request,
        requested_fields=requested_fields,
        requested_concepts=() if rank_intent.requested else intent.requested_concepts,
        scope=scope_ref,
        include_bridge=(intent.include_bridge or mode != MODE_REQUESTED_METRIC),
        composition_focus=intent.composition_focus or rank_intent.requested)

    try:
        result = run_period_change_analysis(request, snapshots)
    except PeriodChangeFailure as failure:
        if rank_intent.requested and failure.reason == FAIL_NO_ELIGIBLE_FIELDS:
            # The reader named a dimension the governed registry does not carry
            # for this book. Say that, rather than the generic period-change
            # message, and rank nothing in its place.
            return _rank_refusal_envelope(question, spec_dict, RankingOutcome(
                requested=True, request=rank_intent.request,
                refusal_reason="dimension_not_governed",
                refusal=(f"I could not rank movement by {rank_intent.term}: it "
                         f"is not a governed period-change dimension for this "
                         f"book — none of "
                         f"{', '.join((rank_intent.field,) + tuple(rank_intent.alt_fields))}"
                         f" is eligible here. I have not ranked a different "
                         f"dimension instead.")))
        return _failure_envelope(question, spec_dict, failure)

    # A ranked question is answered by RANKING the governed period-change
    # output, never by recalculating it.
    ranking = apply_ranking(rank_intent, result)
    if ranking.refusal is not None:
        return _rank_refusal_envelope(question, spec_dict, ranking)

    # THE ONE EVIDENCE RECORD, built here because this is the only place where
    # both the intent (which fields the term could have bound to) and the
    # outcome (which one it did) are in scope. Everything downstream — prose,
    # table, metadata — reads it. Nothing downstream re-derives it.
    receipt = (movement_receipt_for(result, rank_intent, ranking,
                                    population=_population,
                                    pop_evidence=_pop_evidence)
               if ranking.applied else None)
    out = _render(result, question, spec_dict, portfolio_id, as_of,
                  receipt=receipt)
    if _population and _pop_evidence:
        # THE ROUTE DECLARES WHAT IT APPLIED. The population ledger accepts
        # execution evidence only and treats a silent route as having widened,
        # so an answer that genuinely narrowed every snapshot says so — in the
        # same ledger shape, and with the same per-period wording, the funded
        # evolution route already publishes.
        _last = _pop_evidence[-1][1]
        out.setdefault("metadata", {})["populationApplied"] = {
            "applied": [f"{p.field} (applied to every snapshot compared)"
                        for p in _population],
            "unavailable": [],
            "rowsBefore": _last.rows_before,
            "rowsAfter": _last.rows_after,
        }
    return out


# --------------------------------------------------------------------------- #
# P1C — ranked period-over-period movement
# --------------------------------------------------------------------------- #
# The sequence, and nothing else:
#
#     governed period-change engine  →  CategoryShift per category
#                                    →  deterministic ranking (mi_agent.period_change.ranking)
#                                    →  answer + table + receipt evidence
#
# No figure is recalculated here, no ordering is asked of a model, and a ranking
# that cannot be produced is REFUSED rather than replaced by the narrative — a
# reader who asked "which region grew the most" and received a portfolio summary
# has been answered a different question.

# THE ROUTE-LOCAL RANKING VOCABULARY IS DELETED.
#
# `_NARRATIVE_RANK_SUBJECTS` (36 nouns), `_RANK_SUBJECT_LEAD_RE` and
# `_RANK_SUBJECT_SKIP` (30 words) lived here, and `_rank_subject` read the raw
# question against them. `question_interpretation.lexical` now owns direction,
# basis and limit for the whole estate and the contract carries its answers, so
# this route reads a claim instead of a sentence.
#
# The narrative test went with them. It asked whether the noun after "which"
# was one of 36 words; the contract answers the same question structurally —
# a ranking with no dimension claim is the governed narrative, and one with a
# dimension claim is a ranking of that dimension. That reading does not depend
# on which interrogative opened the sentence, which is why
# "show me the drivers that grew the most" used to miss the guard entirely.


@dataclass(frozen=True)
class RankingOutcome:
    """What the route resolved for a ranked question.

    Exactly one of three states, so the caller never has to infer:

    ``requested`` False               not a ranked question — narrative as before
    ``refusal`` set                   ranked, but the ranking could not be produced
    ``movement`` set and ``ok``       ranked, and these are the ranked results
    """

    requested: bool = False
    request: Any = None
    #: The canonical dimension field to rank, and the term the reader used.
    field: Optional[str] = None
    term: Optional[str] = None
    #: Other fields the same term resolves to once dataset availability is
    #: known; carried so an availability difference is never read as a
    #: substitution.
    alt_fields: Tuple[str, ...] = ()
    movement: Any = None
    distribution: Any = None
    refusal: Optional[str] = None
    #: Machine-readable reason for the refusal, for the audit line.
    refusal_reason: Optional[str] = None

    @property
    def applied(self) -> bool:
        return bool(self.movement is not None and getattr(self.movement, "ok", False))


def _distribution_for(result: PeriodChangeResult, keys: Sequence[str]) -> Any:
    """The governed distribution whose field is one of ``keys``, or None."""
    wanted = [str(k) for k in keys if k]
    for key in wanted:
        for dist in result.distribution_changes:
            if dist.field == key:
                return dist
    return None


def resolve_rank_intent(interpretation: Any) -> RankingOutcome:
    """What the question asks to rank, READ FROM THE CONTRACT.

    The question is no longer a parameter, and that absence is the reduction's
    real result: there is nothing left for this route to read from a sentence.
    Direction, basis and limit arrive on `operation.ordering_*`; the dimension
    and every governed field it could bind to arrive on the dimension claim.

    THE ALTERNATES ARE CARRIED THROUGH. The old reader resolved a primary plus
    alternates and this route passed only the primary into the analysis, so a
    book carrying the alternate was told it carried neither — canary defect D1.
    `candidate_concepts` is the primary followed by every alternate, and all of
    them go to the workflow.
    """
    op = getattr(interpretation, "operation", None)
    if op is None or op.type != "ranking":
        return RankingOutcome(requested=False)
    if not (op.ordering_direction and op.ordering_basis):
        # A ranking the contract cannot describe is not ranked here on a guess.
        return RankingOutcome(requested=False)

    dims = [d for d in (getattr(interpretation, "dimensions", None) or [])
            if d.candidate_concept]
    if not dims:
        # No dimension claim: the governed narrative, which is the answer this
        # route already gives for "what were the largest movements".
        return RankingOutcome(requested=False)

    claim = dims[0]
    concepts = list(claim.candidate_concepts)
    request = _rank_request_from_contract(op, claim.raw_text or concepts[0])
    return RankingOutcome(requested=True, request=request, field=concepts[0],
                          term=(claim.raw_text or concepts[0]),
                          alt_fields=tuple(concepts[1:]))


def _rank_request_from_contract(op: Any, term: str) -> Any:
    """The engine's `RankRequest`, built from contract values only.

    A mapping, not a decision: the contract's basis vocabulary is the governed
    one and the engine's is its own, and translating between them in one place
    is what stops a second basis vocabulary appearing here.
    """
    from mi_agent.period_change import rank_request as rank_mod
    from mi_agent.period_change import ranking as rk
    from question_interpretation.schema import (
        ORDER_BASIS_COUNT, ORDER_BASIS_PERCENT, ORDER_BASIS_SHARE,
        ORDER_DECREASE, ORDER_EITHER)

    basis = {ORDER_BASIS_SHARE: rk.BASIS_BALANCE_SHARE,
             ORDER_BASIS_PERCENT: rk.BASIS_BALANCE_PERCENT,
             ORDER_BASIS_COUNT: rk.BASIS_COUNT_ABSOLUTE}.get(
                 op.ordering_basis, rk.BASIS_BALANCE_ABSOLUTE)
    direction = {ORDER_DECREASE: rk.DIRECTION_DECREASE,
                 ORDER_EITHER: rk.DIRECTION_ANY}.get(
                     op.ordering_direction, rk.DIRECTION_INCREASE)
    return rank_mod.RankRequest(basis=basis, direction=direction,
                                top_n=op.ordering_limit,
                                dimension_term=str(term).lower())


def apply_ranking(intent: RankingOutcome, result: PeriodChangeResult
                  ) -> RankingOutcome:
    """Rank the governed period-change output for an already-resolved intent."""
    from mi_agent.period_change import ranking as rk
    from mi_agent.period_change.models import (
        STATUS_AVAILABLE, STATUS_PARTIALLY_AVAILABLE)

    if not intent.requested:
        return intent
    request, term = intent.request, intent.term
    distribution = _distribution_for(result,
                                     (intent.field,) + tuple(intent.alt_fields))
    if distribution is None:
        return RankingOutcome(
            requested=True, request=request,
            refusal_reason="dimension_not_analysed",
            refusal=(f"I could not rank movement by {term}: that dimension was "
                     f"not part of the governed period-change analysis for this "
                     f"book. I have not ranked a different dimension instead."))
    if distribution.status not in (STATUS_AVAILABLE, STATUS_PARTIALLY_AVAILABLE):
        return RankingOutcome(
            requested=True, request=request, distribution=distribution,
            refusal_reason="dimension_not_comparable",
            refusal=(f"I could not rank movement by {term}: it is not comparable "
                     f"across both reporting dates ({distribution.status}). I "
                     f"have not ranked a different dimension instead."))

    movement = rk.rank_movement(distribution.categories, basis=request.basis,
                                direction=request.direction,
                                top_n=request.top_n)
    if not movement.ok:
        return RankingOutcome(
            requested=True, request=request, distribution=distribution,
            movement=movement,
            refusal_reason=("no_category_moved_that_way"
                            if movement.reason
                            and movement.reason.startswith("no category")
                            else "ranking_not_producible"),
            refusal=_unrankable_message(result, term, movement))

    return RankingOutcome(requested=True, request=request, movement=movement,
                          distribution=distribution)


def movement_receipt_for(result: PeriodChangeResult, intent: RankingOutcome,
                         ranking: RankingOutcome,
                         population: Sequence[Any] = (),
                         pop_evidence: Sequence[Any] = ()) -> Any:
    """The governed `MovementReceipt` for a delivered ranked movement.

    EVERY FIGURE IS CARRIED, NONE IS RECOMPUTED. The ranking engine has already
    stated each row's opening, closing, absolute and percentage movement and the
    value it sorted on; this hands those to the receipt verbatim. Recomputing a
    percentage here would create a second calculation of a published fact, and
    the point of the receipt is that there is exactly one.

    Nothing in this function reads the question, a chart column, an artifact
    title or the route's own identity.
    """
    from .movement_receipt import (PopulationEvidence, RankedElement,
                                   build_movement_receipt)

    payload = result.to_dict()
    period = payload["summary"]["period"]
    distribution, movement = ranking.distribution, ranking.movement
    provenance = list(payload.get("dataset_provenance") or [])
    scope = payload.get("portfolio_scope") or {}
    # Per-period row counts, from the governed provenance. WHERE THE ROUTE
    # NARROWED BY ROW PREDICATE, THE RECEIPT SAYS SO AND SHOWS THE EFFECT.
    #
    # This block used to publish `predicates=()` unconditionally, on the stated
    # ground that "this route selects a population by scope, not by row
    # predicate". That was true of the route and is no longer: it now applies
    # the contract's governed population to every snapshot it compares. An
    # empty tuple would now assert that no narrowing happened while one did,
    # which is the one thing a receipt must never do.
    #
    # Nothing is recomputed. The predicates are the SAME objects execution ran,
    # and both row counts come from the evidence `population.apply_population`
    # returned per snapshot: `rows_after` is what was measured, `rows_before`
    # what it was measured out of. Where no predicate ran, the two are equal
    # and the block states exactly what it always did.
    counts = tuple(int(ref.get("row_count") or 0) for ref in provenance)
    before_by_snapshot = {sid: e.rows_before for sid, e in (pop_evidence or ())
                          if getattr(e, "rows_before", None) is not None}
    unfiltered = tuple(int(before_by_snapshot[sid])
                       for sid in (str(ref.get("snapshot_id")) for ref in provenance)
                       if sid in before_by_snapshot)
    population = PopulationEvidence(
        dataset=(payload.get("request_interpretation") or {}).get("mode"),
        portfolio_ids=tuple(str(p) for p in (scope.get("portfolio_ids") or ())),
        predicates=tuple((p.field, p.op, p.value) for p in (population or ())),
        row_counts=counts,
        unfiltered_row_counts=(unfiltered if population and unfiltered else counts))
    elements = tuple(
        RankedElement(rank=position, group_value=str(row.category),
                      start_value=row.start_value, end_value=row.end_value,
                      absolute_movement=row.absolute_movement,
                      percentage_movement=row.percent_movement,
                      rank_value=row.rank_value, presence=row.presence,
                      note=row.note)
        for position, row in enumerate(movement.rows, start=1))
    candidates = tuple(f for f in ((intent.field,) + tuple(intent.alt_fields)) if f)
    # The aggregation as the GOVERNED RESULT publishes it for this dimension —
    # not a word chosen here. A dimension the payload does not carry leaves it
    # unstated rather than guessed.
    published = next((d for d in (payload.get("distribution_changes") or [])
                      if d.get("canonical_field") == distribution.field), {})
    return build_movement_receipt(
        measure=distribution.balance_field,
        aggregation=published.get("aggregation"),
        grouping_dimension=distribution.field,
        grouping_display_name=distribution.display_name,
        grouping_candidates=candidates,
        periods=(period["start"], period["end"]),
        levels=(), ranked=(), elements=elements,
        analysed_groups=tuple(str(c.category) for c in distribution.categories),
        basis=movement.basis, basis_label=movement.basis_label,
        direction=movement.direction, limit=movement.top_n,
        direction_excluded=movement.direction_excluded,
        exclusions=tuple(movement.excluded), population=population)


def _unrankable_message(result: PeriodChangeResult, term: str, movement: Any
                        ) -> str:
    """Why no ranking was produced, as a finding rather than an apology.

    "No region declined" IS the answer to "which region declined the most" — it
    is stated as a fact about the two periods. What must never happen is
    quietly answering with the increases instead, so the message says that too.
    """
    from . import chat_routing

    period = result.summary.get("period") or {}
    start = chat_routing._date_label(period.get("start"))
    end = chat_routing._date_label(period.get("end"))
    if (movement.reason or "").startswith("no category"):
        moved = ("decreased" if movement.direction == "decrease" else "increased")
        return (f"Between {start} and {end}, no {term} {moved} on "
                f"{movement.basis_label}. I have not answered with the "
                f"movements in the other direction instead.")
    return (f"I could not rank {term} by {movement.basis_label} between {start} "
            f"and {end}: {movement.reason}.")


def _rank_refusal_envelope(question: str, spec_dict: Dict[str, Any],
                           ranking: RankingOutcome) -> Dict[str, Any]:
    """A controlled refusal for a ranking that could not be produced.

    Fail-closed by construction: the alternative is returning the period-change
    narrative, which answers a materially different question from the one asked.
    """
    from . import chat_routing

    envelope = chat_routing._envelope(
        ok=False, question=question, answer=ranking.refusal, spec=spec_dict,
        artifacts=[], route=ROUTE_NAME, error=ranking.refusal,
        lens_applied=True,
        warnings=[f"ranking unavailable: {ranking.refusal_reason}"])
    meta = envelope["metadata"]
    meta["controlledUnsupported"] = True
    meta["controlledRefusal"] = True
    meta["workflowId"] = WORKFLOW_ID
    meta["rankedMovement"] = {"applied": False,
                              "reason": ranking.refusal_reason}
    meta["errorCode"] = ErrorCode.UNSUPPORTED_QUESTION
    envelope["controlledRefusal"] = True
    return envelope


#: How many runners-up the prose names when the reader did not ask for a Top N.
#: The table always carries every ranked row.
_PROSE_RUNNERS_UP = 3

_RANK_COLUMNS = [
    {"key": "rank", "label": "Rank"},
    {"key": "category", "label": "Category"},
    {"key": "start_value", "label": "Opening"},
    {"key": "end_value", "label": "Closing"},
    {"key": "movement", "label": "Movement"},
    {"key": "percent_movement", "label": "Relative"},
    {"key": "presence", "label": "Presence"},
]

#: The unit each ranking basis is expressed in, so the table formats the figure
#: the ranking actually used rather than assuming currency.
_BASIS_UNITS = {
    "balance_absolute": UNIT_CURRENCY,
    "balance_percent": UNIT_CURRENCY,
    "count_absolute": UNIT_COUNT,
    "balance_share": UNIT_RATIO,
    "count_share": UNIT_RATIO,
}


def _share_pp(value: Optional[float]) -> str:
    """A governed share is a ratio; a reader wants percentage points."""
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def _rank_rows(receipt: Any) -> List[Dict[str, Any]]:
    """The ranked table, read from the receipt.

    THE POSITION IS READ, NOT RE-COUNTED. This used to number the rows with its
    own `enumerate`, so the rank a reader saw in the table was a second
    derivation of the rank the evidence recorded. They agreed only because the
    same list was iterated twice.
    """
    basis = receipt.ranking_basis
    unit = _BASIS_UNITS.get(basis, UNIT_CURRENCY)
    share = basis in ("balance_share", "count_share")
    rows: List[Dict[str, Any]] = []
    for row in receipt.elements:
        rows.append({
            "rank": row.rank,
            "category": row.group_value,
            "start_value": (_share_pp(row.start_value) if share
                            else _format_value(row.start_value, unit)),
            "end_value": (_share_pp(row.end_value) if share
                          else _format_value(row.end_value, unit)),
            "movement": (f"{row.absolute_movement * 100:+.2f} pp" if share
                         else _format_movement(row.absolute_movement, unit)),
            "percent_movement": ("—" if row.percentage_movement is None
                                 else f"{row.percentage_movement:+.1f}%"),
            "presence": row.presence,
        })
    return rows


def build_rank_answer(receipt: Any) -> str:
    """The ranked answer, stated FROM THE RECEIPT and nothing else.

    Every semantic fact this sentence asserts — which two dates, which
    dimension, which basis, which direction, which group came first, and what
    its opening and closing figures were — is read from the governed evidence
    record. The result object, the ranking outcome and the question are not
    parameters, so there is nothing here to re-derive them from.
    """
    from . import chat_routing

    start = chat_routing._date_label(receipt.start_period)
    end = chat_routing._date_label(receipt.end_period)
    dimension = receipt.grouping_display_name
    basis = receipt.basis_label
    share = receipt.ranking_basis in ("balance_share", "count_share")
    unit = _BASIS_UNITS.get(receipt.ranking_basis, UNIT_CURRENCY)
    top_n = receipt.ordering_limit

    def _describe(row) -> str:
        if share:
            figure = (f"{_share_pp(row.start_value)} → {_share_pp(row.end_value)} "
                      f"({row.absolute_movement * 100:+.2f} pp)")
        else:
            relative = ("" if row.percentage_movement is None
                        else f", {row.percentage_movement:+.1f}%")
            figure = (f"{_format_value(row.start_value, unit)} → "
                      f"{_format_value(row.end_value, unit)} "
                      f"({_format_movement(row.absolute_movement, unit)}"
                      f"{relative})")
        return f"{row.group_value} {figure}"

    direction_word = {"increase": "increased", "decrease": "decreased"}.get(
        receipt.ranking_direction, "moved")
    lead = receipt.elements[0]
    parts = [f"Between {start} and {end}, ranked by {basis} across "
             f"{dimension}, {lead.group_value} {direction_word} the most: "
             f"{_describe(lead)}."]

    # Prose names the leader and a few runners-up; the full ranking is the table.
    # A twelve-category ranking read out as a sentence is not an answer anybody
    # can use, and truncating it silently would hide rows — so when the prose
    # stops short it says how many rows the table carries.
    rows = list(receipt.elements)
    named = rows[1:] if top_n else rows[1:_PROSE_RUNNERS_UP + 1]
    if named:
        parts.append("Then " + "; ".join(_describe(r) for r in named) + ".")
    if top_n:
        parts.append(f"Showing the top {top_n} of "
                     f"{receipt.groups_analysed} categories.")
    elif len(rows) > len(named) + 1:
        parts.append(f"The full ranking of all {len(rows)} ranked "
                     f"{'category' if len(rows) == 1 else 'categories'} "
                     f"is in the table below.")
    if receipt.direction_excluded:
        count = receipt.direction_excluded
        moved = {"increase": "increase", "decrease": "decrease"}.get(
            receipt.ranking_direction, "move")
        parts.append(f"{count} further "
                     f"{'category' if count == 1 else 'categories'} did not "
                     f"{moved} on this basis and "
                     f"{'is' if count == 1 else 'are'} not listed.")
    if receipt.exclusions:
        parts.append("Not ranked: " + "; ".join(
            f"{category} ({reason})" for category, reason in receipt.exclusions)
            + ".")
    return " ".join(parts)


def _failure_envelope(question: str, spec_dict: Dict[str, Any],
                      failure: PeriodChangeFailure) -> Dict[str, Any]:
    """A controlled refusal. Explicit, never a fabricated or substituted answer."""
    from . import chat_routing

    envelope = chat_routing._envelope(
        ok=False, question=question, answer=failure.message, spec=spec_dict,
        artifacts=[], route=ROUTE_NAME, error=failure.message,
        lens_applied=True,
        warnings=[f"period-change unavailable: {failure.reason}"])
    meta = envelope["metadata"]
    # ``controlledUnsupported`` is the existing contract for "I will not answer
    # that": HTTP 200 with ok:false, rather than a data error.
    meta["controlledUnsupported"] = True
    meta["workflowId"] = WORKFLOW_ID
    meta["periodChangeFailure"] = failure.to_dict()
    meta["errorCode"] = FAILURE_ERROR_CODES.get(failure.reason,
                                                ErrorCode.UNSUPPORTED_QUESTION)
    return envelope


# --------------------------------------------------------------------------- #
# Presentation — deterministic, from the result only
# --------------------------------------------------------------------------- #
#: Reader-facing wording for a unit, used where the answer states the scope a
#: ranking was made within.
_UNIT_WORDS: Dict[str, str] = {
    UNIT_CURRENCY: "currency", UNIT_COUNT: "counts",
    UNIT_PERCENTAGE_POINT: "percentage points", UNIT_RATIO: "ratios",
}


def _format_value(value: Optional[float], unit: str) -> str:
    from . import chat_routing

    if value is None:
        return "—"
    if unit == UNIT_CURRENCY:
        return chat_routing._gbp(value)
    if unit == UNIT_PERCENTAGE_POINT:
        return f"{float(value):.2f}%"
    if unit == UNIT_COUNT:
        return f"{int(round(float(value))):,}"
    return f"{float(value):,.4f}"


def _format_movement(value: Optional[float], unit: str) -> str:
    if value is None:
        return "—"
    sign = "+" if value >= 0 else "−"
    if unit == UNIT_PERCENTAGE_POINT:
        return f"{sign}{abs(float(value)):.2f} pp"
    return f"{sign}{_format_value(abs(float(value)), unit)}"


def describe_basis(metric: Any) -> str:
    """How the figure was calculated, in words a reader can act on.

    Without this the table showed "Interest In Arrears 4.00% → 6.60%" with
    nothing to say whether that is a share of LOANS or of BALANCE — and the
    governed default for every v2 flag is a count share, which a reader
    unfamiliar with the registry would not assume.
    """
    aggregation = (metric.aggregation or "").lower()
    if aggregation == "share":
        basis = (metric.share_basis or "count").lower()
        return ("share of loan count" if basis == "count"
                else f"share of {basis}")
    if aggregation == "weighted_average":
        return (f"weighted average by {metric.weight_field}"
                if metric.weight_field else "weighted average")
    if aggregation == "average":
        return "simple average"
    if aggregation == "sum":
        return "sum"
    return aggregation or "—"


def _metric_rows(result: PeriodChangeResult) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric in result.metric_changes:
        rows.append({
            "metric": metric.display_name,
            "canonical_field": metric.field,
            "concept": metric.analytical_concept,
            "temporality": metric.temporality,
            "basis": describe_basis(metric),
            "aggregation": metric.aggregation,
            "share_basis": metric.share_basis,
            "weight_field": metric.weight_field,
            "rank_scope": metric.movement_unit,
            "start_value": _format_value(metric.start_value, metric.movement_unit),
            "end_value": _format_value(metric.end_value, metric.movement_unit),
            "movement": _format_movement(metric.movement_value, metric.movement_unit),
            "relative_change": ("—" if metric.relative_change is None
                                else f"{metric.relative_change * 100:+.1f}%"),
            "interpretation": metric.interpretation,
            "significance": metric.significance,
            "status": metric.status,
            "confidence": metric.confidence,
        })
    return rows


_METRIC_COLUMNS = [
    {"key": "metric", "label": "Metric"},
    {"key": "concept", "label": "Concept"},
    {"key": "temporality", "label": "Temporality"},
    # The basis is displayed, not buried in the payload: a share of loan count
    # and a share of balance are different answers to "the arrears share".
    {"key": "basis", "label": "Basis"},
    {"key": "start_value", "label": "Opening"},
    {"key": "end_value", "label": "Closing"},
    {"key": "movement", "label": "Movement"},
    {"key": "relative_change", "label": "Relative"},
    {"key": "interpretation", "label": "Interpretation"},
    {"key": "significance", "label": "Observed significance"},
    {"key": "rank_scope", "label": "Ranked within"},
    {"key": "status", "label": "Status"},
]

_DISTRIBUTION_COLUMNS = [
    {"key": "dimension", "label": "Dimension"},
    {"key": "category", "label": "Category"},
    {"key": "start_count", "label": "Opening count"},
    {"key": "end_count", "label": "Closing count"},
    {"key": "count_share_movement", "label": "Count-share movement"},
    {"key": "balance_share_movement", "label": "Balance-share movement"},
    {"key": "presence", "label": "Presence"},
]

_BRIDGE_COLUMNS = [
    {"key": "component", "label": "Component"},
    {"key": "amount", "label": "Amount"},
    {"key": "loans", "label": "Loans"},
]


def _distribution_rows(result: PeriodChangeResult) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dist in result.distribution_changes:
        for category in dist.categories:
            rows.append({
                "dimension": dist.display_name,
                "canonical_field": dist.field,
                "category": category.category,
                "start_count": category.start_count,
                "end_count": category.end_count,
                "count_share_movement": (
                    "—" if category.count_share_movement is None
                    else f"{category.count_share_movement * 100:+.2f} pp"),
                "balance_share_movement": (
                    "—" if category.balance_share_movement is None
                    else f"{category.balance_share_movement * 100:+.2f} pp"),
                "presence": category.presence,
            })
    return rows


def _bridge_rows(result: PeriodChangeResult) -> List[Dict[str, Any]]:
    bridge = result.balance_bridge
    if bridge is None or bridge.status != BRIDGE_STATUS_AVAILABLE:
        return []
    fmt = lambda v: _format_value(v, UNIT_CURRENCY)  # noqa: E731
    return [
        {"component": "Opening balance", "amount": fmt(bridge.opening_balance),
         "loans": bridge.continuing_loan_count + bridge.exited_loan_count},
        {"component": "New loans (closing balance)",
         "amount": fmt(bridge.new_loan_balance), "loans": bridge.new_loan_count},
        {"component": "Exited loans (opening balance)",
         "amount": f"−{fmt(bridge.exited_loan_balance)}",
         "loans": bridge.exited_loan_count},
        {"component": "Movement on continuing loans",
         "amount": _format_movement(bridge.continuing_movement, UNIT_CURRENCY),
         "loans": bridge.continuing_loan_count},
        {"component": "Closing balance", "amount": fmt(bridge.closing_balance),
         "loans": bridge.continuing_loan_count + bridge.new_loan_count},
    ]


def build_answer(result: PeriodChangeResult) -> str:
    """Plain-language rendering of the deterministic summary.

    Every clause restates a value from ``result.summary``; no fact is added, no
    cause is asserted, and no movement is described as material.
    """
    from . import chat_routing

    summary = dict(result.summary)
    period = summary.get("period") or {}
    start = chat_routing._date_label(period.get("start"))
    end = chat_routing._date_label(period.get("end"))
    parts = [f"Between {start} and {end}, "
             f"{summary.get('metrics_comparable', 0)} of "
             f"{summary.get('metrics_analysed', 0)} governed metrics could be "
             f"compared across both snapshots."]

    # Reported per unit, and said so: a currency movement and a percentage-point
    # movement are not ranked against each other, so presenting them in one
    # "largest movements" list would assert a comparison that was not made.
    by_unit = summary.get("top_movements_by_unit") or {}
    for unit in sorted(by_unit):
        rows = by_unit[unit] or []
        if not rows:
            continue
        described = "; ".join(
            f"{row['display_name']} {_format_movement(row['movement_value'], unit)} "
            f"({_format_value(row['start_value'], unit)} → "
            f"{_format_value(row['end_value'], unit)})"
            for row in rows)
        parts.append(
            f"Largest observed movements measured in "
            f"{_UNIT_WORDS.get(unit, unit)}: {described}.")

    improvements = summary.get("improvements") or []
    deteriorations = summary.get("deteriorations") or []
    if improvements or deteriorations:
        parts.append(
            f"Against the registry's directionality, {len(improvements)} metric(s) "
            f"moved in the improving direction and {len(deteriorations)} in the "
            f"deteriorating direction.")

    shifts = summary.get("largest_composition_shifts") or []
    if shifts:
        described = "; ".join(
            f"{row['display_name']} — {row['category']} "
            f"{row['count_share_movement'] * 100:+.2f} pp of count share"
            for row in shifts)
        parts.append(f"The largest observed composition shifts were {described}.")

    bridge = summary.get("balance_bridge") or {}
    if bridge.get("status") == BRIDGE_STATUS_AVAILABLE:
        parts.append(
            f"The balance bridge reconciles: opening "
            f"{_format_value(bridge['opening_balance'], UNIT_CURRENCY)} "
            f"{_format_movement(bridge['new_loan_closing_balance'], UNIT_CURRENCY)} "
            f"new lending, "
            f"−{_format_value(bridge['exited_loan_opening_balance'], UNIT_CURRENCY)} "
            f"exits, "
            f"{_format_movement(bridge['movement_on_continuing_loans'], UNIT_CURRENCY)} "
            f"on continuing loans, closing "
            f"{_format_value(bridge['closing_balance'], UNIT_CURRENCY)}.")

    parts.append(summary.get("materiality") or "")
    return " ".join(p for p in parts if p)


def _render(result: PeriodChangeResult, question: str, spec_dict: Dict[str, Any],
            portfolio_id: Optional[str], as_of: Optional[str],
            receipt: Any = None) -> Dict[str, Any]:
    """Render the governed answer.

    THE RANKING OUTCOME IS NOT A PARAMETER. It used to be, and every ranked
    fact in this function was read from it or from `result.summary` — so the
    prose, the table and the metadata were three independent derivations that
    happened to agree. `receipt` is the only ranked input now, which is what
    makes "narration does not infer this independently" a structural property
    rather than a promise.
    """
    from . import chat_routing

    payload = result.to_dict()
    artifacts: List[Dict[str, Any]] = []
    ranked = receipt is not None and bool(receipt.elements)

    summary = payload["summary"]
    kpis = [
        {"label": "Opening period",
         "value": chat_routing._date_label(summary["period"]["start"])},
        {"label": "Closing period",
         "value": chat_routing._date_label(summary["period"]["end"])},
        {"label": "Metrics compared",
         "value": f"{summary['metrics_comparable']}/{summary['metrics_analysed']}"},
        {"label": "Improved / deteriorated",
         "value": f"{len(summary['improvements'])} / {len(summary['deteriorations'])}"},
    ]
    artifacts.append(chat_routing._summary_kpi_artifact(
        "Period change — governed overview", kpis, spec=spec_dict,
        portfolio_id=portfolio_id, as_of=as_of,
        description=(f"Resolved by {summary['period']['resolution_method']} "
                     f"from the governed Business Semantics Registry.")))

    if ranked:
        # The ranking the question asked for leads, before the governed
        # narrative's own tables: the reader asked which category moved most,
        # and that answer must not be the fourth table down.
        # Title, columns and description all state what the RECEIPT records.
        # The table used to name its own dimension and dates from the result
        # while the prose named them from the ranking outcome — two readings of
        # one fact, in one answer.
        artifacts.append(chat_routing._table_artifact(
            f"Ranked movement by {receipt.grouping_display_name}",
            columns=_RANK_COLUMNS, rows=_rank_rows(receipt), spec=spec_dict,
            portfolio_id=portfolio_id, as_of=as_of,
            description=(f"Ranked on {receipt.basis_label} between "
                         f"{chat_routing._date_label(receipt.start_period)} "
                         f"and "
                         f"{chat_routing._date_label(receipt.end_period)}"
                         f", from the governed period-change result.")))

    metric_rows = _metric_rows(result)
    if metric_rows:
        artifacts.append(chat_routing._table_artifact(
            "Metric movements", columns=_METRIC_COLUMNS, rows=metric_rows,
            spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
            description=(f"{len(metric_rows)} governed metrics, selected by "
                         f"policy {payload['field_selection']['policy_name']} "
                         f"v{payload['field_selection']['policy_version']}.")))

    distribution_rows = _distribution_rows(result)
    if distribution_rows:
        artifacts.append(chat_routing._table_artifact(
            "Composition shifts", columns=_DISTRIBUTION_COLUMNS,
            rows=distribution_rows, spec=spec_dict, portfolio_id=portfolio_id,
            as_of=as_of,
            description="Count and balance share movement by category."))

    bridge_rows = _bridge_rows(result)
    if bridge_rows:
        artifacts.append(chat_routing._table_artifact(
            "Balance bridge", columns=_BRIDGE_COLUMNS, rows=bridge_rows,
            spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
            description=("Opening to closing balance over a stable loan "
                         "identifier.")))

    envelope = chat_routing._envelope(
        ok=True, question=question,
        answer=(build_rank_answer(receipt) if ranked
                else build_answer(result)),
        spec=spec_dict,
        artifacts=artifacts, route=ROUTE_NAME, lens_applied=True,
        warnings=list(payload["warnings"]) + list(payload["limitations"]),
        source_notes=[{
            "label": "Business Semantics Registry",
            "detail": (f"v{payload['audit']['business_semantics']['registry_version']} "
                       f"(schema {payload['audit']['business_semantics']['schema_version']})"),
        }, {
            "label": "Governed snapshots",
            "detail": " → ".join(
                str(ref.get("reporting_date") or ref.get("snapshot_id"))
                for ref in payload["dataset_provenance"]),
        }])

    # Additive: the complete governed result travels intact alongside the
    # existing envelope keys. No existing field is renamed or removed.
    envelope["periodChange"] = payload
    meta = envelope["metadata"]
    meta["workflowId"] = WORKFLOW_ID
    meta["periodResolution"] = payload["period_resolution"]["resolution_method"]
    meta["periodChangeMode"] = payload["request_interpretation"]["mode"]
    if ranked:
        # The EVIDENCE the P0 guard verifies against. It states what was ranked,
        # on which basis, in which direction and over which two dates — so the
        # guard can prove the ranking the question asked for is the ranking that
        # ran, rather than taking the route's word for it.
        #
        # EVERY VALUE IS NOW A PROJECTION OF THE RECEIPT. The keys and the
        # figures are unchanged, because the receipt carries the engine's own
        # rows verbatim; what changed is that this dict no longer reads the
        # ranking outcome and the result summary a second time. The receipt
        # itself travels alongside it, additively, so a consumer can audit the
        # answer without reconstructing it from these keys.
        meta["rankedMovement"] = {
            "applied": True,
            "canonicalField": receipt.grouping_dimension,
            "displayName": receipt.grouping_display_name,
            "basis": receipt.ranking_basis,
            "basisLabel": receipt.basis_label,
            "direction": receipt.ranking_direction,
            "topN": receipt.ordering_limit,
            "rankedCategories": len(receipt.elements),
            "categoriesAnalysed": receipt.groups_analysed,
            "excluded": [{"category": c, "reason": r}
                         for c, r in receipt.exclusions],
            "openingPeriod": receipt.start_period,
            "closingPeriod": receipt.end_period,
            "rows": [{"category": e.group_value, "start_value": e.start_value,
                      "end_value": e.end_value,
                      "absolute_movement": e.absolute_movement,
                      "percent_movement": e.percentage_movement,
                      "rank_value": e.rank_value, "presence": e.presence,
                      "note": e.note} for e in receipt.elements],
        }
        meta["movementReceipt"] = receipt.to_dict()
    return envelope
