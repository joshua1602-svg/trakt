"""GovernedQueryPlan — what Trakt is AUTHORISED to execute.

The plan may carry physical bindings. That is not a contradiction of the safety
boundary, it is the point of it: every binding on this object was chosen by the
deterministic compiler against the governed registries, and none of them came
from the model. The plan is where physical facts are allowed to exist because it
is downstream of the decision that produced them.

Nothing here executes. This sprint ends at the plan.

Immutability is load-bearing. A plan that could be edited after compilation
would let a caller re-open a decision the compiler closed, and the provenance
would still claim the compiler made it. Every object is frozen, every collection
is a tuple, and ``plan_id`` is a content hash — so a mutated copy is a different
plan and says so.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

PLAN_SCHEMA_VERSION = "governed_query_plan/1.0"


def _stable(value: Any) -> str:
    """A total order over heterogeneous plan fragments, for canonicalisation."""
    return json.dumps(value, sort_keys=True, default=str)


@dataclass(frozen=True)
class MeasureBinding:
    """A semantic measure, the binding the compiler chose, and the statistic.

    ``canonical_field`` is None for a specialist measure: the borrowing base is
    not a column, it is a methodology, and ``capability_owner`` names the
    deterministic owner that holds it.
    """

    concept: str                                # semantic, from the intent
    statistic: str                              # governed, resolved
    canonical_field: Optional[str] = None       # compiler-chosen binding
    weight_concept: Optional[str] = None
    weight_field: Optional[str] = None
    capability_owner: Optional[str] = None
    #: True when the compiler applied the registry's governed default rather
    #: than a statistic the question named. Recorded so an answer can disclose
    #: it; never silent.
    statistic_defaulted: bool = False


@dataclass(frozen=True)
class FilterBinding:
    """A row predicate bound to a governed field, or to a capability.

    ``canonical_field`` is None only for a predicate on a specialist dimension
    that has no column — a pipeline stage, an ineligibility reason — where
    ``capability_owner`` names the deterministic owner that resolves it.
    """

    concept: str
    comparator: str
    canonical_field: Optional[str] = None
    value: Any = None
    capability_owner: Optional[str] = None


@dataclass(frozen=True)
class DimensionBinding:
    """A grouping dimension bound to a governed field or a capability."""

    concept: str
    canonical_field: Optional[str] = None
    capability_owner: Optional[str] = None


@dataclass(frozen=True)
class GeographyBinding:
    """The resolved geography contract.

    Carries the request as well as the resolution, because "which region did
    this measure?" is a question a reader is entitled to ask of the answer, and
    ``mi_agent.region_basis`` exists precisely because three field families wear
    the word "Region".
    """

    requested_basis: Optional[str]
    requested_level: Optional[str]
    resolved_level: str
    canonical_field: str
    #: Whether the plan groups by geography, restricts to places, or both.
    group_by: bool = True
    values: Tuple[Any, ...] = ()
    #: Set when the compiler supplied a governed default the question did not
    #: state (a bare "by region" resolves to the harmonised reporting taxonomy).
    defaulted: bool = False
    default_reason: str = ""


@dataclass(frozen=True)
class PeriodBinding:
    """The resolved period contract request.

    Deliberately NOT a snapshot id. This sprint ends before execution, and the
    governed period contract resolves a semantic span against a book's actual
    history at run time — pinning a date here would be this package deciding
    something it has no data to decide.
    """

    form: str
    labels: Tuple[str, ...] = ()
    grain: Optional[str] = None
    periods_back: Optional[int] = None
    #: The governed contract that will resolve it, named so the caller knows
    #: which deterministic owner to hand the plan to.
    contract: str = ""
    resolved: bool = False
    #: True when the capability owns its own window — a stage transition or a
    #: bridge defines the period it spans, and the question does not have to.
    owned_by_capability: bool = False


@dataclass(frozen=True)
class TargetBinding:
    """A governed threshold a milestone or limit question is asking about."""

    concept: str
    comparator: str
    value: Any
    canonical_field: Optional[str] = None
    capability_owner: Optional[str] = None


@dataclass(frozen=True)
class PopulationBinding:
    """The governed population: the book state, the lens, the seasoning."""

    base: str
    lens: str
    seasoning: str
    #: Predicates the population itself implies, separate from the question's
    #: own filters, so the two channels stay distinguishable (see
    #: mi_agent.population — scope and row predicates are not one thing).
    scope_predicates: Tuple[FilterBinding, ...] = ()


@dataclass(frozen=True)
class OutputPlan:
    """One authorised figure or table."""

    id: str
    measures: Tuple[MeasureBinding, ...] = ()
    dimensions: Tuple[DimensionBinding, ...] = ()
    filters: Tuple[FilterBinding, ...] = ()
    geography: Optional[GeographyBinding] = None


@dataclass(frozen=True)
class PlanProvenance:
    """Who decided what.

    Split three ways on purpose. ``intent_claims`` is what the model said;
    ``compiler_bindings`` is what the compiler chose. An audit that cannot tell
    those apart cannot answer the only question that matters about this
    architecture — did the model pick the field?
    """

    question: str = ""
    model_id: str = ""
    interpreter_version: str = ""
    vocabulary_version: str = ""
    compiler_version: str = ""
    intent_schema_version: str = ""
    intent_claims: Mapping[str, Any] = field(default_factory=dict)
    compiler_bindings: Mapping[str, Any] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()
    usage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GovernedQueryPlan:
    """An immutable, versioned, execution-ready-but-unexecuted plan."""

    schema_version: str
    capability: str
    operation: str
    population: PopulationBinding
    outputs: Tuple[OutputPlan, ...]
    period: PeriodBinding
    comparison_kind: str = "none"
    comparison_left: Optional[str] = None
    comparison_right: Optional[str] = None
    filters: Tuple[FilterBinding, ...] = ()
    geography: Optional[GeographyBinding] = None
    target: Optional[TargetBinding] = None
    provenance: PlanProvenance = field(default_factory=PlanProvenance)

    @property
    def plan_id(self) -> str:
        """A content hash over the AUTHORISED content, excluding provenance.

        Provenance carries the question and the model id, which differ between
        two paraphrases that mean the same thing. Hashing them in would make
        every plan unique and the identity useless for the one thing it is for:
        telling whether two questions compiled to the same authorised work.
        """
        return "plan_" + hashlib.sha256(
            json.dumps(self._authorised_content(), sort_keys=True,
                       default=str).encode("utf-8")).hexdigest()[:16]

    #: Keys that record HOW a value was arrived at rather than WHAT was
    #: authorised. Stripped from the identity: "the total balance" and "the
    #: balance" authorise the same sum, and one of them reached it through the
    #: registry default. A hash that noticed would report two paraphrases as
    #: divergent for a difference that changes no computation.
    _DERIVATION_KEYS = ("statistic_defaulted", "defaulted", "default_reason")

    def _authorised_content(self) -> Dict[str, Any]:
        """The authorised work, canonicalised so ORDER is not part of identity.

        A grouping is a set, a conjunction of filters is a set, and the figures
        in an output are a set. Which order the model happened to list them in
        is presentation, and two paraphrases routinely differ in it — "by LTV
        band and product type" against "by product type and LTV band" is one
        analysis. The plan still CARRIES the requested order, for whoever
        renders it; the identity just does not depend on it.
        """
        body = asdict(self)
        body.pop("provenance", None)
        body = self._strip_derivation(body)
        body["filters"] = sorted(body.get("filters") or [], key=_stable)
        for output in body.get("outputs") or []:
            output["measures"] = sorted(output.get("measures") or [], key=_stable)
            output["dimensions"] = sorted(output.get("dimensions") or [],
                                          key=_stable)
            output["filters"] = sorted(output.get("filters") or [], key=_stable)
        body["outputs"] = sorted(body.get("outputs") or [],
                                 key=lambda o: str(o.get("id")))
        population = body.get("population") or {}
        population["scope_predicates"] = sorted(
            population.get("scope_predicates") or [], key=_stable)
        return body

    @classmethod
    def _strip_derivation(cls, node: Any) -> Any:
        if isinstance(node, dict):
            return {k: cls._strip_derivation(v) for k, v in node.items()
                    if k not in cls._DERIVATION_KEYS}
        if isinstance(node, (list, tuple)):
            return [cls._strip_derivation(v) for v in node]
        return node

    def to_dict(self) -> Dict[str, Any]:
        body = asdict(self)
        body["plan_id"] = self.plan_id
        return body

    def bound_fields(self) -> Tuple[str, ...]:
        """Every canonical field this plan authorises. For audit and tests."""
        found = []
        for f in self.filters:
            found.append(f.canonical_field)
        for predicate in self.population.scope_predicates:
            found.append(predicate.canonical_field)
        if self.geography is not None:
            found.append(self.geography.canonical_field)
        if self.target is not None and self.target.canonical_field:
            found.append(self.target.canonical_field)
        for output in self.outputs:
            for measure in output.measures:
                if measure.canonical_field:
                    found.append(measure.canonical_field)
                if measure.weight_field:
                    found.append(measure.weight_field)
            for dimension in output.dimensions:
                if dimension.canonical_field:
                    found.append(dimension.canonical_field)
            for f in output.filters:
                found.append(f.canonical_field)
            if output.geography is not None:
                found.append(output.geography.canonical_field)
        return tuple(sorted(set(x for x in found if x)))
