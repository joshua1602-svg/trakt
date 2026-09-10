"""The authoritative governed concept index — one index, two readers.

WHAT CHANGED, AND WHY
---------------------
The first build projected ONE registry into business names and hid every
canonical identifier from the model. Measurement showed the cost: the
interpreter could not verify that "drawdown" is a product or "Offer" a stage,
and clarified on sixteen of twenty questions for want of metadata that was
governed all along.

So the index is now keyed by CANONICAL CONCEPT IDENTIFIER, built from the whole
authoritative estate via :mod:`mi_agent.interpretation_v2.metadata`, and Opus
reads it through read-only retrieval tools rather than a prompt dump. Opus may
see an identifier and name it.

That does not make Opus authoritative. This same index is what the compiler
validates against — deliberately the same object, so that what the model is
shown and what the compiler will accept cannot drift apart — and the compiler
re-derives existence, asset applicability, portfolio availability, permitted
operation and physical binding for every concept before anything enters a plan.

Business names survive as ALIASES. "balance" still resolves to
``current_outstanding_balance``; "region" resolves to nothing, because seven
governed fields wear that word, and :meth:`GovernedVocabulary.candidates` names
all seven rather than picking one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import (Any, Dict, FrozenSet, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

from .metadata import (
    applies_to_asset_class,
    business_semantics,
    canonical_fields,
    governed_values_for_field,
    portfolio_semantic_context,
)
from .metadata import _load as _load_source
from .metadata import _slug

VOCABULARY_VERSION = "2.0.0"


# --------------------------------------------------------------------------- #
# Closed enumerations — the axes the model chooses ON.
# --------------------------------------------------------------------------- #

CAPABILITIES: FrozenSet[str] = frozenset({
    "generic_analysis",
    "portfolio_summary",
    "concentration",
    "limit_assessment",
    "period_movement",
    "borrowing_base",
    "funded_bridge",
    "pipeline",
    "pipeline_stage_movement",
    "forecast",
})

OPERATIONS: FrozenSet[str] = frozenset({
    "point_in_time", "breakdown", "rank", "distribution", "series", "compare",
    "movement", "bridge", "summary",
    # specialist capability operations
    "headroom", "utilisation", "eligibility", "transition", "arrivals",
    "departures", "stayers", "reconciliation", "forecast_milestone",
    "forecast_projection",
})

CAPABILITY_OPERATIONS: Mapping[str, FrozenSet[str]] = {
    "generic_analysis": frozenset({
        "point_in_time", "breakdown", "rank", "distribution", "series",
        "compare", "movement"}),
    "portfolio_summary": frozenset({"summary", "compare"}),
    "concentration": frozenset({"summary", "rank", "breakdown", "point_in_time"}),
    "limit_assessment": frozenset({"summary", "rank", "point_in_time",
                                   "forecast_projection", "headroom"}),
    "period_movement": frozenset({"movement", "rank", "breakdown", "compare",
                                  "series"}),
    "borrowing_base": frozenset({"point_in_time", "headroom", "utilisation",
                                 "eligibility", "breakdown", "movement",
                                 "bridge", "series"}),
    "funded_bridge": frozenset({"bridge", "movement"}),
    "pipeline": frozenset({"summary", "breakdown", "point_in_time", "series"}),
    "pipeline_stage_movement": frozenset({
        "transition", "arrivals", "departures", "stayers", "reconciliation",
        "movement", "breakdown"}),
    "forecast": frozenset({"forecast_milestone", "forecast_projection",
                           "series", "point_in_time"}),
}

STATISTICS: FrozenSet[str] = frozenset({
    "count", "count_distinct", "sum", "average", "weighted_average",
    "median", "min", "max", "share", "contribution",
})

#: ANALYTIC MODES rather than statistics of a field. A share is a ratio of two
#: populations and a contribution decomposes an aggregate across groups; both
#: are governed elsewhere in this repository (``mi_agent.statistic``, P1A, P1D)
#: and neither appears in a field's ``allowed_aggregations``, because neither is
#: an aggregation OF the field.
ANALYTIC_MODES: FrozenSet[str] = frozenset({"share", "contribution"})
ADDITIVE_STATISTICS: FrozenSet[str] = frozenset({"sum", "count", "count_distinct"})

#: Only a weighted average NEEDS a weight named. A contribution takes one
#: optionally: decomposing a weighted average needs it, decomposing an additive
#: total does not — there the measure is its own weight.
STATISTICS_REQUIRING_WEIGHT: FrozenSet[str] = frozenset({"weighted_average"})
STATISTICS_FORBIDDING_WEIGHT: FrozenSet[str] = frozenset({
    "count", "count_distinct", "sum", "median", "min", "max", "share"})

COMPARATORS: FrozenSet[str] = frozenset({
    "gt", "gte", "lt", "lte", "eq", "ne", "between", "in", "not_in"})

TIME_FORMS: FrozenSet[str] = frozenset({
    "current", "previous_reporting_period", "relative_pair", "explicit_period",
    "range", "series", "forward_looking"})
TIME_GRAINS: FrozenSet[str] = frozenset({"daily", "weekly", "monthly",
                                         "quarterly", "annual"})

GEOGRAPHY_BASES: FrozenSet[str] = frozenset({"obligor", "collateral",
                                             "reporting_taxonomy"})
GEOGRAPHY_LEVELS: FrozenSet[str] = frozenset({"reporting", "nuts3", "itl3",
                                              "postcode"})

POPULATION_BASES: FrozenSet[str] = frozenset({"funded", "pipeline", "forecast",
                                              "whole_book"})
POPULATION_LENSES: FrozenSet[str] = frozenset({"direct", "acquired", "all"})
SEASONING_SEGMENTS: FrozenSet[str] = frozenset({"front_book", "back_book", "any"})

COMPARISON_KINDS: FrozenSet[str] = frozenset({
    "none", "population_pair", "period_pair", "dimension_pair"})

#: Specialist measures OWNED by a capability, not composed from fields. Opus may
#: name them; it is never shown how they are built.
SPECIALIST_MEASURES: Mapping[str, Tuple[str, ...]] = {
    "borrowing_base": (
        "borrowing_base", "borrowing_base_headroom", "borrowing_base_utilisation",
        "facility_utilisation", "facility_drawn", "facility_commitment",
        "eligible_balance", "ineligible_balance", "ineligible_loan_count",
        "ineligible_loan_share", "ineligible_balance_share"),
    "funded_bridge": ("funded_balance_movement", "bridge_component"),
    "pipeline": ("pipeline_amount", "pipeline_case_count"),
    "pipeline_stage_movement": (
        "cases_moved", "amount_moved", "cases_arrived", "cases_departed",
        "cases_stayed", "stayer_amount_change", "stage_opening", "stage_closing"),
    "forecast": ("forecast_funded_balance", "forecast_completion_rate",
                 "forecast_milestone_date"),
    "concentration": ("concentration_exposure", "concentration_share"),
    "limit_assessment": ("limit_headroom", "limit_utilisation",
                         "limit_breach_status"),
    "portfolio_summary": ("portfolio_overview",),
}

SPECIALIST_DIMENSIONS: Mapping[str, Tuple[str, ...]] = {
    "borrowing_base": ("ineligibility_reason",),
    "pipeline_stage_movement": ("origin_stage", "destination_stage"),
    "limit_assessment": ("concentration_test",),
    "concentration": ("concentration_test",),
    "funded_bridge": ("bridge_component",),
}

#: The row itself. "How many loans?" counts rows and needs no field; without a
#: concept for it the model would have to nominate some arbitrary column to
#: count, which is a binding decision over an irrelevant field.
_BASE_CONCEPTS: Tuple[Dict[str, Any], ...] = (
    {"concept_id": "loan", "label": "Loan",
     "description": "The loan itself. Count it to answer 'how many loans'.",
     "role": "measure",
     "allowed_statistics": ("count", "count_distinct", "share", "contribution"),
     "default_statistic": "count", "aliases": ("loans", "case_count", "cases")},
    {"concept_id": "case", "label": "Pipeline case",
     "description": "A pipeline case. Count it to answer 'how many cases'.",
     "role": "measure",
     "allowed_statistics": ("count", "count_distinct", "share", "contribution"),
     "default_statistic": "count", "aliases": ("pipeline_case",)},
)

#: The seven governed region fields, each with the (basis, level) it represents.
#: Named here because ``mi_agent.region_basis`` governs the three families and
#: this is the projection of that ruling into the concept index — a geography
#: concept the model names as a dimension is routed into the geography contract
#: by the compiler rather than grouped as a bare column.
GEOGRAPHY_CONCEPTS: Mapping[str, Tuple[str, str]] = {
    "canonical_region_reporting": ("reporting_taxonomy", "reporting"),
    "canonical_region_detail": ("reporting_taxonomy", "reporting"),
    "collateral_geography": ("collateral", "reporting"),
    "geographic_region_obligor": ("obligor", "nuts3"),
    "geographic_region_collateral": ("collateral", "nuts3"),
    "geographic_region_obligor_itl3": ("obligor", "itl3"),
    "geographic_region_collateral_itl3": ("collateral", "itl3"),
    "postcode": ("collateral", "postcode"),
}

#: Internal snapshot/reporting mechanics. Excluded because they are handles on
#: a physical extract, not concepts a reader asks about, and offering one would
#: hand the model a snapshot selector by another name.
_MECHANIC_FIELDS = frozenset({
    "reporting_date", "cut_off_date", "upload_timestamp",
    "pipeline_snapshot_date", "portfolio_id", "spv_id", "acquired_portfolio_id"})

_AGG_TRANSLATION = {
    "sum": "sum", "avg": "average", "average": "average",
    "weighted_avg": "weighted_average", "weighted_average": "weighted_average",
    "median": "median", "min": "min", "max": "max", "count": "count",
    "count_distinct": "count_distinct", "share": "share",
    "contribution": "contribution", "distribution": None, "loan_level": None,
    "balance_sum": "sum",
}
_ROLE_TRANSLATION = {"metric": "measure", "dimension": "dimension",
                     "date": "date", "flag": "flag"}


# --------------------------------------------------------------------------- #
# Concepts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SemanticConcept:
    """One governed concept, as both the model and the compiler see it.

    ``concept_id`` IS the canonical identifier. Opus may read it and name it;
    the compiler still re-derives everything about it from the registry before a
    plan can carry it.
    """

    concept_id: str
    label: str
    description: str
    role: str                                   # measure | dimension | date | flag
    aliases: Tuple[str, ...] = ()
    allowed_statistics: Tuple[str, ...] = ()
    default_statistic: Optional[str] = None
    default_weight_concept: Optional[str] = None
    unit: Optional[str] = None
    data_type: Optional[str] = None
    values: Tuple[str, ...] = ()
    values_source: str = ""
    analytical_concept: Optional[str] = None
    temporality: Optional[str] = None
    categories: Tuple[str, ...] = ()
    workflow_tags: Tuple[str, ...] = ()
    asset_applicability: Tuple[str, ...] = ()
    bucket_concept: Optional[str] = None
    geography_basis: Optional[str] = None
    geography_level: Optional[str] = None
    canonical_field: Optional[str] = None
    owning_capability: Optional[str] = None

    @property
    def is_specialist(self) -> bool:
        return self.owning_capability is not None

    @property
    def is_geography(self) -> bool:
        return self.geography_level is not None

    def search_view(self) -> Dict[str, Any]:
        """The compact row a search returns."""
        view: Dict[str, Any] = {"concept_id": self.concept_id,
                                "label": self.label, "role": self.role}
        if self.description:
            view["definition"] = self.description
        if self.aliases:
            view["aliases"] = list(self.aliases[:8])
        if self.unit:
            view["unit"] = self.unit
        if self.owning_capability:
            view["owned_by_capability"] = self.owning_capability
        if self.is_geography:
            view["geography"] = {"basis": self.geography_basis,
                                 "level": self.geography_level}
        return view

    def metadata_view(self) -> Dict[str, Any]:
        """Everything governed about this concept."""
        view = self.search_view()
        view.update({
            "data_type": self.data_type,
            "temporality": self.temporality,
            "analytical_concept": self.analytical_concept,
            "categories": list(self.categories),
            "workflow_tags": list(self.workflow_tags),
            "asset_applicability": list(self.asset_applicability) or ["cross_asset"],
            "has_governed_values": bool(self.values),
        })
        if self.allowed_statistics:
            view["permitted_statistics"] = list(self.allowed_statistics)
        if self.default_statistic:
            view["default_statistic"] = self.default_statistic
        if self.default_weight_concept:
            view["default_weight_concept"] = self.default_weight_concept
        if self.bucket_concept:
            view["banded_as"] = self.bucket_concept
        if self.is_specialist:
            view["note"] = ("Owned by a capability. Name it; do not impose a "
                            "statistic or weight and do not decompose it.")
        return view


@dataclass(frozen=True)
class GovernedVocabulary:
    """The authoritative concept index for one governed context. Immutable."""

    concepts: Mapping[str, SemanticConcept]
    alias_index: Mapping[str, Tuple[str, ...]]
    capabilities: FrozenSet[str]
    asset_class: str = ""
    version: str = VOCABULARY_VERSION
    book_id: Optional[str] = None
    available_fields: FrozenSet[str] = frozenset()

    # -- lookups ------------------------------------------------------------ #

    def resolve(self, term: Optional[str]) -> Optional[SemanticConcept]:
        """A term -> the one concept it names, or None.

        Exact identifier first, then a UNIQUE alias. Never fuzzy, and never
        first-wins on an ambiguous alias: "region" is claimed by seven governed
        fields and resolving it to any one of them would be the compiler
        deciding what the reader meant.
        """
        if not isinstance(term, str):
            return None
        key = term.strip().lower()
        if key in self.concepts:
            return self.concepts[key]
        matches = self.alias_index.get(key, ())
        if len(matches) == 1:
            return self.concepts.get(matches[0])
        return None

    def candidates(self, term: Optional[str]) -> Tuple[SemanticConcept, ...]:
        """Every concept a term could mean. Non-empty only when ambiguous."""
        if not isinstance(term, str):
            return ()
        key = term.strip().lower()
        if key in self.concepts:
            return (self.concepts[key],)
        return tuple(self.concepts[c] for c in self.alias_index.get(key, ())
                     if c in self.concepts)

    def terms(self, role: Optional[str] = None) -> Tuple[str, ...]:
        if role is None:
            return tuple(sorted(self.concepts))
        return tuple(sorted(c for c, v in self.concepts.items() if v.role == role))

    def measures(self) -> Tuple[str, ...]:
        return self.terms("measure")

    def dimensions(self) -> Tuple[str, ...]:
        return self.terms("dimension")

    def applies_here(self, concept: SemanticConcept) -> bool:
        """Whether this concept applies to the configured asset class."""
        return applies_to_asset_class(concept.asset_applicability, self.asset_class)

    def is_available(self, concept: SemanticConcept) -> bool:
        if concept.is_specialist or concept.canonical_field is None:
            return True
        if not self.available_fields:
            return True
        return concept.canonical_field in self.available_fields

    def for_book(self, available_fields: Iterable[str], *,
                 book_id: Optional[str] = None,
                 capabilities: Optional[Iterable[str]] = None
                 ) -> "GovernedVocabulary":
        """A view narrowed to what this book can actually answer."""
        present = frozenset(str(f).strip() for f in available_fields if f)
        caps = (frozenset(capabilities) & self.capabilities
                if capabilities is not None else self.capabilities)
        kept = {k: c for k, c in self.concepts.items()
                if (c.is_specialist or c.canonical_field is None
                    or not present or c.canonical_field in present)
                and (not c.is_specialist or c.owning_capability in caps)}
        aliases = {a: tuple(c for c in ids if c in kept)
                   for a, ids in self.alias_index.items()}
        aliases = {a: ids for a, ids in aliases.items() if ids}
        return replace(self, concepts=kept, alias_index=aliases, capabilities=caps,
                       available_fields=present, book_id=book_id or self.book_id)

    # -- the standing context shown to the model ---------------------------- #

    def orientation_payload(self) -> Dict[str, Any]:
        """The small standing block the interpreter is given up front.

        Deliberately NOT the registry. It is the closed enumerations, the
        capability names, and the counts — enough to orient, after which the
        model RETRIEVES what it needs. Dumping 150 concepts into every prompt
        was the previous design's other mistake: it cost 24k tokens a question
        and still under-described every one of them.
        """
        return {
            "vocabulary_version": self.version,
            "asset_class": self.asset_class,
            "capabilities": sorted(self.capabilities),
            "capability_operations": {k: sorted(v) for k, v in
                                      sorted(CAPABILITY_OPERATIONS.items())
                                      if k in self.capabilities},
            "operations": sorted(OPERATIONS),
            "statistics": sorted(STATISTICS),
            "comparators": sorted(COMPARATORS),
            "time_forms": sorted(TIME_FORMS),
            "time_grains": sorted(TIME_GRAINS),
            "geography_bases": sorted(GEOGRAPHY_BASES),
            "geography_levels": sorted(GEOGRAPHY_LEVELS),
            "population_bases": sorted(POPULATION_BASES),
            "population_lenses": sorted(POPULATION_LENSES),
            "seasoning_segments": sorted(SEASONING_SEGMENTS),
            "comparison_kinds": sorted(COMPARISON_KINDS),
            "governed_defaults": dict(GOVERNED_DEFAULTS),
            "concept_counts": {
                "total": len(self.concepts),
                "measures": len(self.measures()),
                "dimensions": len(self.dimensions()),
            },
            "how_to_find_a_concept": (
                "Call search_concepts to find the governed identifier for a "
                "business word, get_concept_metadata for its full definition, "
                "and get_allowed_values before asserting any filter value."),
        }


#: Slots the compiler fills from a governed default when the question leaves
#: them empty. Declared so the model knows an empty slot is SAFE — the first
#: live run had the interpreter blocking on every bare "region" because nothing
#: told it a governed default exists.
GOVERNED_DEFAULTS: Mapping[str, str] = {
    "geography.basis": (
        "at level 'reporting' an empty basis resolves to the client's "
        "harmonised reporting taxonomy. At 'nuts3' and 'itl3' there is NO "
        "default — the borrower's region and the property's region are "
        "different answers — so leaving basis empty there asks the user."),
    "geography.level": "empty resolves to 'reporting'.",
    "measures[].statistic": (
        "empty resolves to the governed registry default for that concept "
        "(a sum for balances, an exposure-weighted average for LTV and rates)."),
    "measures[].weight": (
        "empty resolves to the governed registry weight for that concept."),
    "population.base": "empty resolves to 'funded'.",
    "time.form": "empty resolves to 'current'.",
}


# --------------------------------------------------------------------------- #
# Building the index from the authoritative estate
# --------------------------------------------------------------------------- #

def _concept_from_registry(canonical_field: str, mi_entry: Mapping[str, Any],
                           business: Mapping[str, Any],
                           canonical: Mapping[str, Any],
                           values: Mapping[str, Tuple[str, ...]]
                           ) -> Optional[SemanticConcept]:
    role = _ROLE_TRANSLATION.get(str(mi_entry.get("role") or "").strip().lower())
    if role is None:
        return None

    stats: List[str] = []
    for raw in mi_entry.get("allowed_aggregations") or ():
        mapped = _AGG_TRANSLATION.get(str(raw).strip().lower())
        if mapped and mapped in STATISTICS and mapped not in stats:
            stats.append(mapped)
    default = _AGG_TRANSLATION.get(
        str(mi_entry.get("default_aggregation") or "").strip().lower())
    if default is not None and default not in stats:
        stats.append(default)
    # Additivity is the REGISTRY's statement, read before the universal
    # statistics are added — every measure can be counted, and letting that
    # count make everything look additive would permit a share of an average.
    additive = not ADDITIVE_STATISTICS.isdisjoint(stats)
    if role == "measure":
        for extra in ("count", "min", "max"):
            if extra not in stats:
                stats.append(extra)
        if additive:
            stats.extend(m for m in sorted(ANALYTIC_MODES) if m not in stats)

    aliases = {_slug(mi_entry.get("business_name"))}
    aliases.update(_slug(s) for s in (mi_entry.get("synonyms") or ()))
    aliases.update(_slug(s) for s in (business.get("aliases") or ()))
    aliases.discard("")
    aliases.discard(canonical_field)

    basis, level = GEOGRAPHY_CONCEPTS.get(canonical_field, (None, None))
    return SemanticConcept(
        concept_id=canonical_field,
        label=str(mi_entry.get("business_name")
                  or mi_entry.get("display_name") or canonical_field),
        description=str(mi_entry.get("business_description")
                        or business.get("rationale") or "").strip(),
        role=role,
        aliases=tuple(sorted(aliases)),
        allowed_statistics=tuple(sorted(stats)),
        default_statistic=default,
        unit=mi_entry.get("format"),
        data_type=str(canonical.get("format") or mi_entry.get("format") or ""),
        values=values.get(canonical_field, ()),
        values_source="governed registry" if canonical_field in values else "",
        analytical_concept=business.get("analytical_concept"),
        temporality=business.get("temporality"),
        categories=tuple(str(c) for c in (business.get("categories") or ())),
        workflow_tags=tuple(str(t) for t in (business.get("workflow_tags") or ())),
        asset_applicability=tuple(str(a) for a in
                                  (business.get("asset_applicability") or ())),
        bucket_concept=mi_entry.get("bucket_field"),
        geography_basis=basis,
        geography_level=level,
        canonical_field=canonical_field,
    )


def _specialist_concepts(capabilities: Iterable[str]) -> Dict[str, SemanticConcept]:
    out: Dict[str, SemanticConcept] = {}
    caps = set(capabilities)
    for cap, ids in SPECIALIST_MEASURES.items():
        if cap not in caps:
            continue
        for concept_id in ids:
            out[concept_id] = SemanticConcept(
                concept_id=concept_id,
                label=concept_id.replace("_", " ").title(),
                description=(f"Owned by the {cap} capability. Its methodology is "
                             f"deterministic and is not composed by the "
                             f"interpreter."),
                role="measure", allowed_statistics=(), owning_capability=cap)
    for cap, ids in SPECIALIST_DIMENSIONS.items():
        if cap not in caps:
            continue
        for concept_id in ids:
            if concept_id in out:
                continue
            out[concept_id] = SemanticConcept(
                concept_id=concept_id,
                label=concept_id.replace("_", " ").title(),
                description=f"A governed dimension of the {cap} capability.",
                role="dimension", owning_capability=cap,
                values=governed_values_for_field().get("pipeline_stage", ())
                if concept_id.endswith("_stage") else ())
    return out


@lru_cache(maxsize=1)
def load_governed_vocabulary() -> GovernedVocabulary:
    """Build the authoritative concept index from the committed estate."""
    mi_fields = (_load_source("mi_semantics") or {}).get("fields") or {}
    business = business_semantics()
    canonical = canonical_fields()
    values = governed_values_for_field()

    concepts: Dict[str, SemanticConcept] = {}
    for canonical_field, entry in sorted(mi_fields.items()):
        if canonical_field in _MECHANIC_FIELDS:
            continue
        concept = _concept_from_registry(
            canonical_field, entry, business.get(canonical_field) or {},
            canonical.get(canonical_field) or {}, values)
        if concept is not None:
            concepts[concept.concept_id] = concept

    # Default weights, now that every identifier exists.
    for concept_id, concept in list(concepts.items()):
        weight_field = (mi_fields.get(concept.canonical_field) or {}).get("weight_field")
        if weight_field and weight_field in concepts:
            concepts[concept_id] = replace(concept,
                                           default_weight_concept=weight_field)

    for base in _BASE_CONCEPTS:
        payload = dict(base)
        payload["aliases"] = tuple(payload.get("aliases", ()))
        concepts[payload["concept_id"]] = SemanticConcept(**payload)

    caps = CAPABILITIES
    concepts.update(_specialist_concepts(caps))

    #: alias -> every concept that claims it. A word claimed by more than one
    #: governed concept resolves to NONE and reports all the candidates, which
    #: is how "region" becomes a question rather than a guess.
    alias_index: Dict[str, List[str]] = {}
    for concept in concepts.values():
        for alias in concept.aliases:
            if alias and alias not in concepts:
                alias_index.setdefault(alias, []).append(concept.concept_id)

    return GovernedVocabulary(
        concepts=concepts,
        alias_index={a: tuple(sorted(ids)) for a, ids in alias_index.items()},
        capabilities=caps,
        asset_class=str(portfolio_semantic_context().get("asset_class") or ""),
    )


def canonical_field_names() -> FrozenSet[str]:
    """Every canonical field the MI semantics registry knows."""
    return frozenset(((_load_source("mi_semantics") or {}).get("fields") or {}).keys())
