"""mi_workflows.analytical.populations — governed populations for a plan.

A capability call names the population it runs over. This module is the only
place a population is created, and every one it creates comes from a governed
resolver that already exists:

* **seasoning** — :mod:`mi_agent.seasoning` decides what "front book" means and
  what the boundary is. The front/back split is a governed BINARY PARTITION, so
  naming one side in a comparative question makes the other side available by
  construction rather than by interpretation.
* **provenance** — :mod:`mi_agent.portfolio_lens` decides what "direct" and
  "acquired" mean, and the governed portfolio registry resolves each to its
  current member portfolios.
* **dimension value** — a category is accepted only when it is literally
  present in the governed frame for a governed dimension. A value the book does
  not carry is refused, never approximated.

Every synthesised population is then put back through
:func:`mi_agent.population.fabricated_concepts` — the safety fix's own guard —
so this layer cannot do the thing that guard exists to prevent. A plan that
would execute a governed population concept the question never requested is
rejected here, before any data is read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mi_agent import population as _population
from mi_agent import portfolio_lens as _portfolio_lens
from mi_agent import seasoning as _seasoning
from mi_workflows import engine

from .contract import PopulationRef

#: Population kinds. Each maps to one governed resolver above.
KIND_TOTAL = "total"
KIND_SEASONING = "seasoning"
KIND_PROVENANCE = "provenance"
KIND_DIMENSION_VALUE = "dimension_value"


class FabricatedPopulation(ValueError):
    """A plan asked for a governed population the question never requested."""


@dataclass(frozen=True)
class PopulationSpec:
    """A population a plan intends to run over, before it is resolved."""

    key: str
    label: str
    kind: str = KIND_TOTAL
    #: Row predicate, as ``{field: value}``. Empty for the total scope and for
    #: a provenance population (which travels on the lens, not on filters).
    filters: Mapping[str, Any] = field(default_factory=dict)
    #: Portfolio lens name for a provenance population (``direct``/``acquired``).
    lens_term: Optional[str] = None

    @property
    def is_total(self) -> bool:
        return self.kind == KIND_TOTAL and not self.filters and not self.lens_term

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "kind": self.kind,
                "filters": dict(self.filters), "lens": self.lens_term}


TOTAL = PopulationSpec(key="total", label="the whole funded book", kind=KIND_TOTAL)


# --------------------------------------------------------------------------- #
# Governed construction
# --------------------------------------------------------------------------- #
def seasoning_population(segment: str) -> PopulationSpec:
    """One side of the governed seasoning partition.

    The label carries the configured boundary (``Front Book (0-12 months)``)
    because the cutoff is configurable and a reader must be able to see which
    one produced the population in front of them.
    """
    config = _seasoning.load_seasoning_config()
    return PopulationSpec(
        key="front_book" if segment == _seasoning.FRONT_BOOK else "back_book",
        label=config.describe_segment(segment),
        kind=KIND_SEASONING,
        filters={_seasoning.SEASONING_SEGMENT_FIELD: segment})


#: A lending-window population is still a SEASONING population — same axis, same
#: governed config, same concept as far as the fabrication guard is concerned.
#: It is not a fourth kind.
def lending_window_population(key: str) -> Optional[PopulationSpec]:
    """One governed lending window (``new`` / ``recent`` / front / back).

    The predicate comes from :mod:`mi_agent.seasoning`, which owns the ruling and
    the thresholds; this function only wraps it. Front and back keep the
    ``seasoning_segment`` predicate they already had, so a question that resolved
    to one of them before resolves to exactly the same rows now.
    """
    config = _seasoning.load_seasoning_config()
    window = config.lending_window(key)
    if window is None:
        return None
    return PopulationSpec(key=window.key, label=window.label,
                          kind=KIND_SEASONING, filters=window.predicate())


def provenance_population(lens_term: str) -> PopulationSpec:
    """A direct or acquired population, resolved through the portfolio lens."""
    lens = _portfolio_lens.resolve_lens(lens_term)
    return PopulationSpec(key=lens.name, label=lens.label,
                          kind=KIND_PROVENANCE, lens_term=lens_term)


def dimension_population(field_name: str, value: str, label: str) -> PopulationSpec:
    return PopulationSpec(key=f"{field_name}={value}", label=label,
                          kind=KIND_DIMENSION_VALUE,
                          filters={field_name: value})


def other_seasoning_segment(segment: str) -> str:
    """The other side of the governed binary seasoning partition."""
    return (_seasoning.BACK_BOOK if segment == _seasoning.FRONT_BOOK
            else _seasoning.FRONT_BOOK)


# --------------------------------------------------------------------------- #
# The fabrication guard, applied to this layer's own plans
# --------------------------------------------------------------------------- #
def assert_not_fabricated(specs: Sequence[PopulationSpec], question: str) -> None:
    """Refuse a plan that invents a governed population concept.

    The union of every population a plan will execute is checked against the
    question with the SAME function the parse seam uses, so the analytical layer
    is held to the rule the fabricated-population safety fix established rather
    than being trusted to respect it.

    Both sides of a governed binary partition pass, and pass for the right
    reason rather than by exemption: the guard compares CONCEPTS, so a question
    naming "the front book" has requested the seasoning concept, and the other
    side of that same split is the concept's other value. A question that names
    no seasoning concept at all is refused a seasoning population in either
    direction, exactly as the safety fix requires.
    """
    filters: Dict[str, Any] = {}
    for spec in specs:
        filters.update(dict(spec.filters or {}))
    if not filters:
        return
    fabricated = _population.fabricated_concepts(filters, question)
    if fabricated:
        raise FabricatedPopulation(
            "the analytical plan would execute the governed population "
            f"concept(s) {', '.join(sorted(fabricated))}, which this question "
            "did not request")


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def resolve_lens(spec: PopulationSpec):
    """The governed portfolio lens for a provenance population.

    Resolved through ``chat_routing._resolve_lens``, so the lens carries the
    registry's explicit portfolio-id list rather than a provenance TYPE string.
    That matters here more than anywhere: a comparative question names BOTH
    sides, so resolving from the question would pick one of them (or neither)
    and both sides of the comparison would end up covering the same book.
    """
    from mi_agent_api import chat_routing as _routing

    return _routing._resolve_lens(f"the {spec.lens_term} book", None)


def apply(frame, spec: PopulationSpec, semantics: Optional[Mapping[str, Any]] = None,
          *, lens_filter=None) -> Tuple[Any, PopulationRef, Dict[str, Any]]:
    """Narrow ``frame`` to ``spec``. Returns ``(frame, ref, evidence)``.

    Row predicates go through :func:`mi_agent.population.apply_population`, so
    this layer and the point-in-time executor cannot disagree about what a
    predicate means. Provenance goes through the caller-supplied lens filter,
    which is the narrowing every other route performs.
    """
    evidence: Dict[str, Any] = {"applied": [], "unavailable": [],
                                "rowsBefore": None, "rowsAfter": None}
    if frame is None:
        return None, PopulationRef(key=spec.key, label=spec.label,
                                   is_total=spec.is_total), evidence

    rows_before = int(len(frame))
    work = frame
    predicate_text: Optional[str] = None

    if spec.lens_term and lens_filter is None:
        # Nothing to narrow with. Recording it as unavailable is what keeps the
        # population honest: a scope that was not applied must never be
        # presented as one that was.
        evidence["unavailable"].append(f"portfolio lens = {spec.lens_term}")
    elif spec.lens_term:
        lens = resolve_lens(spec)
        work = lens_filter(work, lens)
        ids = (lens.filters or {}).get(_portfolio_lens.SOURCE_ID_FIELD) or []
        predicate_text = (f"portfolio lens = {lens.label}"
                          + (f" ({', '.join(sorted(str(i) for i in ids))})"
                             if ids else ""))

    if spec.filters:
        predicates = _population.material_predicates(spec.filters, semantics)
        work, ev = _population.apply_population(work, predicates, semantics)
        evidence["applied"] = list(ev.applied)
        evidence["unavailable"] = list(ev.unavailable)
        joined = "; ".join(ev.applied)
        predicate_text = f"{predicate_text}; {joined}" if predicate_text else joined

    rows_after = int(len(work)) if work is not None else 0
    evidence["rowsBefore"] = rows_before
    evidence["rowsAfter"] = rows_after

    # The population's exposure is the governed engine's SUM aggregate, not a
    # second one computed here — a population label and an answer's balance must
    # never be able to disagree.
    exposure = None
    balance_col = "current_outstanding_balance"
    if work is not None and balance_col in getattr(work, "columns", []):
        exposure = engine.aggregate(work, balance_col, engine.AGG_SUM,
                                    unit=engine.UNIT_CURRENCY).value

    ref = PopulationRef(key=spec.key, label=spec.label,
                        predicate=predicate_text or None,
                        rows=rows_after, rows_before=rows_before,
                        exposure=exposure, is_total=spec.is_total)
    return work, ref, evidence
