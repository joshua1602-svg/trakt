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

from typing import Any, Optional


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
        return lens_mod.lens_from_term(name)
    if name == "cohort" and ids:
        return lens_mod._selection_lens(list(ids))
    return None


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
