"""mi_agent_api/recogniser_registry — governed capability routing, declaratively.

What this replaces
------------------
``chat_routing.try_route`` used to be a hand-ordered ``if/elif`` chain: a
capability was four things in four places (a predicate, a position in the chain,
a handler, and membership of a lens set), and its precedence was **source-code
line order**. Adding a capability meant editing the chain; two recognisers that
both matched were resolved by whichever was written first, and that could not be
inspected or reported.

This module makes routing a *registry*:

    Recogniser(name, priority, recognise, handle, …)  ──► RecogniserRegistry
                                                            │
    RouteRequest ──────────────────────────────────────────►│
                                                            ▼
                                              ordered candidates → first
                                              handler returning an answer

Guarantees
----------
* **Deterministic registration.** A duplicate name is refused, not silently
  overwritten. Registration order is recorded and is part of the sort key.
* **Deterministic ordering.** Candidates sort by ``(-confidence, priority,
  registration_index)``. Every component is total and stable, so the same
  registry with the same request always yields the same order — asserted by
  ``test_recogniser_registry.py``.
* **Behaviour preservation.** Every recogniser migrated from the old chain
  declares ``DEFAULT_CONFIDENCE``, so ordering degenerates to priority order,
  which is the historical chain order. A handler returning ``None`` falls
  through to the next candidate exactly as the ``if`` chain did.
* **Capability gating in one place.** A recogniser may declare the governed
  capability it needs. The router resolves it through the SAME
  ``portfolio_context.resolve_context`` the React dashboard uses, so an
  unavailable capability produces the same governed explanation on every
  channel instead of a per-route data error.

Extension points for the Business Semantics Registry
----------------------------------------------------
The forthcoming Business Semantics Registry becomes the semantic foundation for
workflow recognisers (period change, portfolio risk comparison, covenant
headroom, driver attribution). Three seams exist so it can be plugged in without
restructuring:

1. ``RouteRequest.semantics_context`` — governed semantic metadata resolved at
   parse time (see ``mi_agent.parsed_question``). A future recogniser reads it
   from the request it already receives; no signature changes.
2. ``Recogniser.metadata`` — a free-form declarative slot. A workflow recogniser
   can declare the business terms, comparison bases or materiality rules it
   consumes, and a registry-aware loader can validate those against the BSR
   without this module knowing the schema.
3. ``RecogniserRegistry.register`` is public and additive — a BSR-driven loader
   can register recognisers at startup from configuration rather than code.

Deliberately NOT here: workflow recognisers, multi-capability orchestration, a
workflow planner, comparison engines or materiality logic. This module routes to
ONE deterministic capability and stops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from trakt_core.portfolio import REASON_NO_PORTFOLIOS_IN_SCOPE

logger = logging.getLogger("mi_agent_api.recogniser_registry")

#: The confidence every migrated recogniser declares. Equal confidence across
#: the board makes ordering collapse to priority order — i.e. the historical
#: chain order — so this refactor is behaviour-preserving by construction.
#: A future recogniser that is *more* specific may declare a higher value to win
#: regardless of position; one that is a guess may declare lower.
DEFAULT_CONFIDENCE = 0.5


# --------------------------------------------------------------------------- #
# Request / recognition contracts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RouteRequest:
    """Everything a recogniser may inspect and a handler may need.

    One object, so adding an input to routing does not change eleven handler
    signatures. ``spec``/``spec_dict`` come from the SINGLE parse
    (``mi_agent.parsed_question.ParsedQuestion``) — a recogniser must never
    re-parse the question.
    """

    question: str
    spec: Any
    spec_dict: Dict[str, Any]
    semantics: Mapping[str, Any]
    view: str
    client_id: str
    run_id: Optional[str]
    portfolio_id: Optional[str]
    output_root: Optional[str] = None
    pipeline_root: Optional[str] = None
    #: An ALREADY-BUILT historical completion model. Direct callers and tests
    #: still pass one; the serving path passes ``history_model_provider``
    #: instead so the (expensive) build is deferred. When both are supplied the
    #: eager value wins, so an existing caller's behaviour is unchanged.
    history_model: Optional[Mapping[str, Any]] = None
    #: Builds the historical completion model ON DEMAND. Called at most once per
    #: request, and ONLY by a handler that actually needs it — see
    #: :meth:`resolve_history_model`.
    history_model_provider: Optional[Callable[[], Optional[Mapping[str, Any]]]] = None
    as_of: Optional[str] = None
    source_lens: Optional[Any] = None
    frame_resolver: Optional[Callable[[str, Optional[str]], Any]] = None
    #: Parser metadata for the single parse (mode, confidence, note).
    parse_meta: Mapping[str, Any] = field(default_factory=dict)
    #: Governed semantic metadata from the Business Semantics Registry, when one
    #: is wired in. Empty today; recognisers may read it without a signature
    #: change once it is populated.
    semantics_context: Mapping[str, Any] = field(default_factory=dict)
    #: Memo for :meth:`resolve_history_model`. A mutable default on a frozen
    #: dataclass is fine — ``frozen`` prevents rebinding the attribute, not
    #: mutating the object it points at. Excluded from equality/repr so two
    #: otherwise-identical requests still compare equal.
    _history_memo: Dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False)

    def resolve_history_model(self) -> Optional[Mapping[str, Any]]:
        """The historical completion model for this request, built on first use.

        Building it reads and replays EVERY retained weekly extract, so it must
        never happen for a question that does not need it. Recognition never
        touches this — only a matched handler calls it — which is what keeps an
        ordinary MI or Copilot question off that path entirely.

        Memoised per request: a handler that asks twice pays once. A provider
        that raises is treated as "no model available", exactly as a ``None``
        eager value already was, so a history fault degrades the answer's
        precision rather than failing the request.
        """
        if self.history_model is not None:
            return self.history_model
        if self.history_model_provider is None:
            return None
        if "value" not in self._history_memo:
            try:
                self._history_memo["value"] = self.history_model_provider()
            except Exception:  # noqa: BLE001 - history is additive, never fatal
                self._history_memo["value"] = None
        return self._history_memo["value"]


@dataclass(frozen=True)
class Recognition:
    """A recogniser's verdict on one request."""

    matched: bool
    confidence: float = DEFAULT_CONFIDENCE
    reason: str = ""

    @classmethod
    def no(cls, reason: str = "") -> "Recognition":
        return cls(False, 0.0, reason)

    @classmethod
    def yes(cls, confidence: float = DEFAULT_CONFIDENCE,
            reason: str = "") -> "Recognition":
        return cls(True, confidence, reason)


def _as_recognition(value: Any) -> Recognition:
    """Accept a bare bool from a simple predicate, or a full Recognition."""
    if isinstance(value, Recognition):
        return value
    return Recognition.yes() if value else Recognition.no()


@dataclass(frozen=True)
class Recogniser:
    """One declaratively-registered capability route."""

    #: Stable route id. Appears as ``metadata.route`` on the answer.
    name: str
    #: Lower runs earlier. Ties break on registration order.
    priority: int
    #: ``f(RouteRequest) -> bool | Recognition``. Pure; no side effects.
    recognise: Callable[[RouteRequest], Any]
    #: ``f(RouteRequest) -> dict | None``. ``None`` falls through to the next
    #: candidate, exactly as the old chain did.
    handle: Callable[[RouteRequest], Optional[Dict[str, Any]]]
    #: True when the handler genuinely narrows its figures to the portfolio
    #: lens. False (the safe default) means whole-book WITH disclosure.
    lens_aware: bool = False
    #: Governed capability this route needs (``trakt_core.portfolio.CAP_*``).
    #: ``None`` means "no gate". Resolved through the shared context service.
    capability: Optional[str] = None
    description: str = ""
    #: Free-form declarative slot. Reserved for Business Semantics Registry
    #: metadata (business terms consumed, comparison bases, materiality rules).
    metadata: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class DuplicateRecogniserError(ValueError):
    """Raised when a name is registered twice. Never silently overwritten."""


class RecogniserRegistry:
    """An ordered, deterministic collection of recognisers."""

    def __init__(self) -> None:
        self._items: Dict[str, Tuple[int, Recogniser]] = {}
        self._next_index = 0

    # -- registration ------------------------------------------------------ #
    def register(self, recogniser: Recogniser) -> Recogniser:
        """Register one recogniser. Refuses a duplicate name.

        Silent replacement would make ordering depend on import order, which is
        precisely the non-determinism this registry exists to remove.
        """
        if not recogniser.name:
            raise ValueError("a recogniser must have a name")
        if recogniser.name in self._items:
            raise DuplicateRecogniserError(
                f"recogniser {recogniser.name!r} is already registered")
        self._items[recogniser.name] = (self._next_index, recogniser)
        self._next_index += 1
        return recogniser

    def extend(self, recognisers: Sequence[Recogniser]) -> None:
        for rec in recognisers:
            self.register(rec)

    def get(self, name: str) -> Optional[Recogniser]:
        entry = self._items.get(name)
        return entry[1] if entry else None

    def names(self) -> Tuple[str, ...]:
        return tuple(r.name for r in self.ordered())

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    # -- ordering ---------------------------------------------------------- #
    def ordered(self) -> Tuple[Recogniser, ...]:
        """Every recogniser in deterministic evaluation order."""
        return tuple(rec for _, rec in
                     sorted(self._items.values(), key=lambda e: (e[1].priority, e[0])))

    def candidates(self, request: RouteRequest) -> Tuple[Tuple[Recogniser, Recognition], ...]:
        """The recognisers that matched, in deterministic dispatch order.

        Sorted by ``(-confidence, priority, registration_index)``. With every
        migrated recogniser on ``DEFAULT_CONFIDENCE`` this is exactly priority
        order — the historical chain — so behaviour is preserved. A recogniser
        that raises is skipped and logged: a faulty recogniser must not be able
        to take the whole chat path down.
        """
        scored: List[Tuple[float, int, int, Recogniser, Recognition]] = []
        for index, rec in sorted(self._items.values(), key=lambda e: (e[1].priority, e[0])):
            try:
                verdict = _as_recognition(rec.recognise(request))
            except Exception as exc:  # noqa: BLE001 - one bad recogniser must not break routing
                logger.warning("recogniser %s raised during recognition: %s", rec.name, exc)
                continue
            if verdict.matched:
                scored.append((-verdict.confidence, rec.priority, index, rec, verdict))
        scored.sort(key=lambda s: (s[0], s[1], s[2]))
        return tuple((rec, verdict) for _, _, _, rec, verdict in scored)


#: The process-wide registry the chat path uses. Populated by
#: ``chat_routing`` at import time.
REGISTRY = RecogniserRegistry()


# --------------------------------------------------------------------------- #
# Capability gate — the SAME resolution the React dashboard uses
# --------------------------------------------------------------------------- #
#: ``f(context_id) -> ResolvedContext``. Injectable so tests (and a future
#: channel) can supply their own resolution without importing the HTTP layer.
CapabilityResolver = Callable[[Optional[str]], Any]


def default_capability_resolver(context_id: Optional[str]) -> Any:
    """Resolve capabilities through ``portfolio_context.resolve_context``.

    This is the single governed resolution React's ``/mi/portfolio-context``,
    ``/mi/pipeline/*`` and ``/mi/forecast/*`` endpoints already use, so chat and
    dashboard cannot disagree about whether a capability applies to a scope.
    """
    from . import portfolio_context as ctx_mod
    return ctx_mod.resolve_context(context_id)


def resolve_capability_state(capability: str, context_id: Optional[str], *,
                             resolver: Optional[CapabilityResolver] = None) -> Any:
    """The governed :class:`CapabilityState` for one capability and scope.

    Returns ``None`` when the gate is **not answerable**, which is deliberately
    distinct from "the capability is disabled". Two cases:

    * resolution failed (storage, config) — an infrastructure problem must never
      be reported to a user as "this analysis does not apply to your portfolio",
      which is a materially different and false statement;
    * the deployment has **no governed portfolio registry at all** — a canonical
      tape without source-portfolio provenance yields an empty registry, so
      every capability resolves ``NO_PORTFOLIOS_IN_SCOPE``. That means
      "provenance is unavailable here", not "this analysis is inapplicable", and
      gating on it would disable every routed capability for a single-portfolio
      deployment. Such a deployment must keep behaving exactly as before.
    """
    try:
        resolved = (resolver or default_capability_resolver)(context_id)
    except Exception as exc:  # noqa: BLE001 - never fail a question on gate resolution
        logger.info("capability resolution unavailable for %s: %s", capability, exc)
        return None
    try:
        state = resolved.capability(capability)
    except Exception as exc:  # noqa: BLE001
        logger.info("capability lookup failed for %s: %s", capability, exc)
        return None
    if state is None:
        return None
    if getattr(state, "reason_code", None) == REASON_NO_PORTFOLIOS_IN_SCOPE:
        return None
    return state
