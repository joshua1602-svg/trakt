"""mi_agent_api/contract_scope — the source-portfolio scope THE CONTRACT states.

WHY ITS OWN MODULE. Two routes need to turn a `SourceScopeClaim` into the lens
and the context id their executors take, and neither `analytical_plan` nor
`period_change_route` is the right home: the plan module is guarded against
importing a question resolver at all (`test_the_plan_module_never_imports_a_
question_resolver`, and putting this there broke it), and a shared helper that
lives inside one route is a helper the next route copies.

A MAPPING, NOT A DECISION. `mi_agent.portfolio_lens` remains the only thing
that decides what "the acquired book" MEANS; this hands the contract's answer
back to that owner's own constructors. Nothing here reads a question.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

_logger = logging.getLogger(__name__)


def lens_from_contract(interpretation: Any) -> Any:
    """The RESOLVED source-portfolio lens the contract states, or ``None``.

    ``None`` for every state except FILLED, and that is deliberate: a scope the
    owner was never consulted about (EMPTY) is NOT Total, and one it could not
    resolve (UNRESOLVABLE) is not Total either. A caller that needs to know a
    scope was ASKED FOR but could not be resolved must use
    :func:`requested_context_id`, not the absence of a lens.

    Source lens and row predicates stay different axes: this reads
    `source_scope` and never `row_predicates`.
    """
    from mi_agent import portfolio_lens as lens_mod

    scope = getattr(interpretation, "source_scope", None)
    if scope is None or getattr(scope, "state", None) != "filled":
        return None
    name = getattr(scope, "scope", None)
    ids = tuple(str(i) for i in (getattr(scope, "portfolio_ids", ()) or ()))
    if name == "total":
        return lens_mod.total_lens()
    if name in ("direct", "acquired"):
        return _through_the_registry(lens_mod.lens_from_term(name))
    if name == "cohort" and ids:
        return _through_the_registry(lens_mod._selection_lens(list(ids)))
    return None


def _through_the_registry(lens: Any) -> Any:
    """``lens`` re-expressed as the explicit portfolio-id list it names.

    THE LENS A ROUTE APPLIES MUST CARRY IDS, NOT A TYPE STRING.
    `portfolio_lens._type_lens` builds ``{source_portfolio_type: "direct"}``,
    and every consumer that narrows a frame — `chat_routing._apply_lens_filter`
    and the point-in-time executor alike — filters on
    ``source_portfolio_id``. Handed a type lens, the filter matched nothing to
    narrow BY and returned the frame unchanged.

    That is what "Summarise the month-on-month movement in the Direct book"
    answered from: five snapshots, 520 rows in and 520 rows out of every one,
    a whole-book movement of £22.6m reported for a book that moved £12.4m, and
    a receipt that declared the Direct scope it had not applied. The same
    question routed to `period_movement` — which resolves its lens through
    `chat_routing._resolve_lens` — answered £12.4m. The two envelopes were
    identical in every published field.

    This is the same registry resolution `_resolve_lens` performs, and it is
    performed HERE so that a lens derived from the contract and a lens derived
    from the sentence are the same object by construction rather than by
    coincidence. Best-effort, exactly as there: an unavailable registry returns
    the lens unresolved — and `_apply_lens_filter` then REFUSES it rather than
    quietly widening, which is the half of this fix that keeps the class shut.
    """
    from mi_agent import portfolio_lens as lens_mod

    try:
        from . import portfolio_context as _ctx

        scope = _ctx.resolve_context(lens_mod.context_id(lens),
                                     discover_pipeline=False).scope
        filters = dict(getattr(scope, "filters", None) or {})
        if not filters.get(lens_mod.SOURCE_ID_FIELD):
            return lens
        return lens_mod.PortfolioLens(name=lens.name, label=lens.label,
                                      filters=filters, cohort_id=lens.cohort_id)
    except Exception:  # noqa: BLE001 - an unavailable registry is not a decision
        _logger.info("contract lens %r could not be resolved through the "
                     "registry; it stays unresolved and any consumer that "
                     "cannot apply it must refuse", getattr(lens, "name", None),
                     exc_info=True)
        return lens


def requested_context_id(interpretation: Any) -> Optional[str]:
    """The scope the question ASKED FOR, resolvable or not.

    UNRESOLVABLE IS NOT ABSENT, and conflating them is a silent widening.
    Measured: with the lens alone, "show product concentration for acquired_009"
    — a book the registry does not hold — produced ``None``, the workflow ran
    over the WHOLE book, and the answer named neither the requested scope nor
    the fact that it had not been applied. The name the reader used comes back
    here so the executor can refuse by it.
    """
    from mi_agent import portfolio_lens as lens_mod

    scope = getattr(interpretation, "source_scope", None)
    if scope is None:
        return None
    state = getattr(scope, "state", None)
    if state == "unresolvable":
        ids = tuple(str(i) for i in (getattr(scope, "portfolio_ids", ()) or ()))
        return (ids[0] if ids else (getattr(scope, "raw_text", None) or None))
    lens = lens_from_contract(interpretation)
    if lens is None:
        return None
    try:
        return lens_mod.context_id(lens)
    except Exception:  # noqa: BLE001 - an identity fault must not fail a route
        return None
