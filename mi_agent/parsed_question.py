"""mi_agent.parsed_question — the single semantic parse of one question.

Why this exists
---------------
A chat request used to be parsed **twice**: once in
``mi_agent_api.chat_routing.try_route`` (always deterministic, to detect the
intent) and again in ``mi_agent.mi_agent_workflow.run_mi_agent_query`` (possibly
via the LLM, to execute). That cost double parse time, and — worse — meant the
*routing* decision could be taken on a different spec from the one *executed*.

``ParsedQuestion`` is the single parsed representation. It is produced once, as
early as the dataframe is known, and threaded through routing and execution. A
component that needs the spec receives it; nothing re-derives it.

    question ──► ParsedQuestion.parse() ──► ParsedQuestion
                                             │
                                             ├──► recogniser registry (routing)
                                             └──► workflow (execution)

Extension point — Business Semantics Registry
---------------------------------------------
``ParsedQuestion.semantics_context`` is a free-form, additive slot for governed
semantic metadata resolved alongside the spec (business term → canonical metric,
materiality thresholds, comparison bases, …). It is populated by
``semantics_resolver`` when one is supplied, and is ``{}`` otherwise, so the
forthcoming Business Semantics Registry can be injected at the single existing
parse site rather than threaded through every recogniser and handler.

Nothing in this module interprets ``semantics_context``; it is carried, not
consumed. Recognisers that need it read it from the request they already
receive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Set

#: Signature of an optional Business Semantics Registry resolver:
#: ``f(question, spec, semantics) -> Mapping[str, Any]``. Injected, never
#: imported — this module has no dependency on any registry implementation.
SemanticsResolver = Callable[[str, Any, Mapping[str, Any]], Mapping[str, Any]]


@dataclass
class ParsedQuestion:
    """One question, parsed once.

    ``spec`` is the governed :class:`~mi_agent.mi_query_spec.MIQuerySpec` the
    parser proposed. ``meta`` is the parser metadata (mode, confidence, note,
    LLM usage) that the workflow reports and the recogniser registry may weigh.
    """

    question: str
    spec: Any
    meta: Dict[str, Any] = field(default_factory=dict)
    #: Columns the parse was resolved against; ``None`` when no frame was
    #: available (the parse is then registry-only, exactly as routing's
    #: historical parse was).
    available_columns: Optional[Set[str]] = None
    #: Additive governed semantic metadata. See the module docstring.
    semantics_context: Dict[str, Any] = field(default_factory=dict)

    # -- parser metadata a caller commonly branches on ---------------------- #
    @property
    def note(self) -> str:
        return str(self.meta.get("note") or "")

    @property
    def parser_mode(self) -> str:
        return str(self.meta.get("parser_mode") or "deterministic")

    @property
    def confidence(self) -> str:
        return str(self.meta.get("parser_confidence") or "")

    @property
    def is_unmapped(self) -> bool:
        """True when the parser could not map the question to any analytic."""
        return self.note in ("unmapped", "unresolved_metric")

    def merge_filters(self, extra: Optional[Mapping[str, Any]]) -> None:
        """Merge caller-supplied (drill-through) filters into the spec, in place.

        Done once, here, so routing and execution cannot disagree about which
        filters are attached — they share this object.
        """
        if not extra:
            return
        try:
            self.spec.filters = {**(getattr(self.spec, "filters", None) or {}),
                                 **dict(extra)}
        except Exception:  # noqa: BLE001 - a bad filter dict must not block parsing
            pass

    @classmethod
    def parse(cls, question: str, semantics: Mapping[str, Any], *,
              available_columns: Optional[Set[str]] = None,
              llm_enabled: bool = False,
              model: Optional[str] = None,
              max_attempts: int = 1,
              llm_callable: Optional[Callable[..., Any]] = None,
              provider: str = "anthropic",
              catalog_mode: str = "core",
              zero_cost_first: bool = True,
              semantics_resolver: Optional[SemanticsResolver] = None,
              ) -> "ParsedQuestion":
        """THE parse entry point for the chat path.

        Wraps ``parse_with_repair`` so there is exactly one place a question
        becomes a spec. Tests assert this is called once per request.
        """
        from .llm_query_parser import parse_with_repair

        spec, meta = parse_with_repair(
            question, semantics, available_columns=available_columns,
            llm_enabled=llm_enabled, model=model, max_attempts=max_attempts,
            llm_callable=llm_callable, provider=provider,
            catalog_mode=catalog_mode, zero_cost_first=zero_cost_first)

        context: Dict[str, Any] = {}
        if semantics_resolver is not None:
            try:
                context = dict(semantics_resolver(question, spec, semantics) or {})
            except Exception:  # noqa: BLE001 - a registry fault must not fail a query
                context = {}
        return cls(question=question, spec=spec, meta=dict(meta or {}),
                   available_columns=set(available_columns)
                   if available_columns is not None else None,
                   semantics_context=context)
