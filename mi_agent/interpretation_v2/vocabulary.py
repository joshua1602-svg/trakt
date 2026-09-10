"""The governed SEMANTIC vocabulary — the only language the model may speak.

Why this module exists
----------------------
``mi_agent/mi_semantics_field_registry.yaml`` is keyed by CANONICAL FIELD:
``current_outstanding_balance``, ``current_loan_to_value``,
``canonical_region_reporting``. That is a physical schema. Handing it to a
language model as its output vocabulary would hand the model the binding
decision, which is precisely what the interpretation/compiler split exists to
prevent.

So this module is a boundary in front of it. It projects the registry into
SEMANTIC CONCEPTS — the words a lender uses — and keeps the canonical field on
the compiler's side of the wall:

    registry (physical)  ──►  GovernedVocabulary  ──►  prompt_payload()  ──►  Opus
                                     │                  (no canonical fields)
                                     └──►  resolve(term).canonical_field  ──►  compiler only

``prompt_payload`` is asserted by test to contain no canonical field name that
is not also the concept's own business term, so the model cannot learn the
physical schema from what it is shown.

Book scoping
------------
The model should not be offered every concept Trakt has ever known when the
governed context can narrow it. :meth:`GovernedVocabulary.for_book` returns a
view restricted to concepts the book actually carries, which is also what lets
the compiler tell CONCEPT_UNAVAILABLE (governed, absent here) apart from
UNREGISTERED_CONCEPT (not governed anywhere) — two different answers that must
never be collapsed into one.

Nothing here is a redesign of the registry, OCC, or the global field registry.
It is a read-only projection, built once and cached.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

VOCABULARY_VERSION = "1.0.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]
MI_SEMANTICS_PATH = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
CAPABILITY_REGISTRY_PATH = (
    _REPO_ROOT / "config" / "system" / "mi_capability_registry.yaml")
#: Which governed enum domain each canonical field draws its values from.
FIELDS_REGISTRY_PATH = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
#: The business spellings of each governed enum domain.
ENUM_SYNONYMS_PATH = _REPO_ROOT / "config" / "system" / "enum_synonyms.yaml"


# --------------------------------------------------------------------------- #
# Closed enumerations — the axes the model chooses ON, never the values it
# invents. Every one of these is validated at intent-parse time.
# --------------------------------------------------------------------------- #

#: WHAT KIND of analysis is being asked for. A specialist capability names the
#: deterministic owner of a methodology; the model identifies WHICH, never HOW.
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

#: The SHAPE of the answer. Distinct from the statistic, which says how a
#: measure is reduced. "balance by region" is a breakdown whose statistic is a
#: sum; "the median LTV" is a point_in_time whose statistic is a median.
OPERATIONS: FrozenSet[str] = frozenset({
    "point_in_time",
    "breakdown",
    "rank",
    "distribution",
    "series",
    "compare",
    "movement",
    "bridge",
    "summary",
    # specialist capability operations
    "headroom",
    "utilisation",
    "eligibility",
    "transition",
    "arrivals",
    "departures",
    "stayers",
    "reconciliation",
    "forecast_milestone",
    "forecast_projection",
})

#: Which operations belong to which capability. A capability/operation pair
#: outside this map is an UNSUPPORTED_COMPOSITION — the compiler does not
#: reach for the nearest supported neighbour.
CAPABILITY_OPERATIONS: Mapping[str, FrozenSet[str]] = {
    "generic_analysis": frozenset({
        "point_in_time", "breakdown", "rank", "distribution", "series",
        "compare", "movement"}),
    "portfolio_summary": frozenset({"summary", "compare"}),
    "concentration": frozenset({"summary", "rank", "breakdown", "point_in_time"}),
    "limit_assessment": frozenset({"summary", "rank", "point_in_time",
                                   "forecast_projection"}),
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

#: How a measure is reduced over the population.
STATISTICS: FrozenSet[str] = frozenset({
    "count", "count_distinct", "sum", "average", "weighted_average",
    "median", "min", "max", "share", "contribution",
})

#: ANALYTIC MODES rather than statistics of a field. A share is a ratio of two
#: populations and a contribution decomposes a weighted aggregate across groups;
#: both are governed elsewhere in this repository (see
#: ``mi_agent.statistic.ANALYTIC_MODES``, P1A and P1D) and neither appears in a
#: field's ``allowed_aggregations``, because they are not aggregations OF the
#: field. They are permitted on any additive measure.
ANALYTIC_MODES: FrozenSet[str] = frozenset({"share", "contribution"})

#: The statistics an analytic mode may decompose. A mode over a non-additive
#: measure is meaningless — a share of an average is not a quantity.
ADDITIVE_STATISTICS: FrozenSet[str] = frozenset({"sum", "count", "count_distinct"})

#: Statistics that require a weight term, and those for which one is meaningless.
#: Only a weighted average NEEDS a weight named. A contribution takes one
#: optionally: decomposing a weighted average needs the weight, but decomposing
#: an additive total does not — there the measure is its own weight.
STATISTICS_REQUIRING_WEIGHT: FrozenSet[str] = frozenset({"weighted_average"})
STATISTICS_FORBIDDING_WEIGHT: FrozenSet[str] = frozenset({
    "count", "count_distinct", "sum", "median", "min", "max", "share"})

COMPARATORS: FrozenSet[str] = frozenset({
    "gt", "gte", "lt", "lte", "eq", "ne", "between", "in", "not_in",
})

#: SEMANTIC time forms. No dates, no snapshot identifiers, ever.
TIME_FORMS: FrozenSet[str] = frozenset({
    "current",
    "previous_reporting_period",
    "relative_pair",
    "explicit_period",
    "range",
    "series",
    "forward_looking",
})
TIME_GRAINS: FrozenSet[str] = frozenset({"daily", "weekly", "monthly",
                                         "quarterly", "annual"})

#: Geography is TWO axes and the model states both. `basis` is whose geography
#: (the borrower's or the property's); `level` is how fine. The compiler owns
#: which column each pair lands on — see mi_agent.region_basis, which is where
#: the three families are governed.
GEOGRAPHY_BASES: FrozenSet[str] = frozenset({"obligor", "collateral",
                                             "reporting_taxonomy"})
GEOGRAPHY_LEVELS: FrozenSet[str] = frozenset({"reporting", "nuts3", "itl3",
                                              "postcode"})

#: Which rows are in scope, stated semantically. `base` is the book state,
#: `lens` the provenance scope, `seasoning` the vintage partition.
POPULATION_BASES: FrozenSet[str] = frozenset({"funded", "pipeline", "forecast",
                                              "whole_book"})
POPULATION_LENSES: FrozenSet[str] = frozenset({"direct", "acquired", "all"})
SEASONING_SEGMENTS: FrozenSet[str] = frozenset({"front_book", "back_book", "any"})

COMPARISON_KINDS: FrozenSet[str] = frozenset({
    "none", "population_pair", "period_pair", "dimension_pair",
})

#: Specialist measures that are OWNED by a capability, not composed from fields.
#: Opus may name them; it may never be asked how they are calculated. The
#: capability's registered deterministic owner holds the methodology.
SPECIALIST_MEASURES: Mapping[str, Tuple[str, ...]] = {
    "borrowing_base": (
        "borrowing_base", "borrowing_base_headroom", "borrowing_base_utilisation",
        "facility_utilisation", "facility_drawn", "facility_commitment",
        "eligible_balance", "ineligible_balance", "ineligible_loan_count",
        "ineligible_loan_share", "ineligible_balance_share",
        "ineligibility_reason",
    ),
    "funded_bridge": ("funded_balance_movement", "bridge_component"),
    "pipeline": ("pipeline_amount", "pipeline_case_count"),
    "pipeline_stage_movement": (
        "cases_moved", "amount_moved", "cases_arrived", "cases_departed",
        "cases_stayed", "stayer_amount_change", "stage_opening", "stage_closing",
    ),
    "forecast": ("forecast_funded_balance", "forecast_completion_rate",
                 "forecast_milestone_date"),
    "concentration": ("concentration_exposure", "concentration_share"),
    "limit_assessment": ("limit_headroom", "limit_utilisation", "limit_breach_status"),
    "portfolio_summary": ("portfolio_overview",),
}

#: Dimensions a specialist capability owns which have no canonical field in the
#: MI semantics registry — a stage, an ineligibility reason, a limit. They are
#: governed dimensions of the capability, bound by it and not by a column.
SPECIALIST_DIMENSIONS: Mapping[str, Tuple[str, ...]] = {
    "borrowing_base": ("ineligibility_reason",),
    "pipeline_stage_movement": ("stage", "destination_stage", "origin_stage"),
    "pipeline": ("stage",),
    "limit_assessment": ("concentration_test",),
    "concentration": ("concentration_test",),
    "funded_bridge": ("bridge_component",),
}


#: Concepts that are not columns and not specialist methodologies: the ROW
#: itself. "How many loans?" counts rows and needs no field, and without a term
#: for it the model would have to nominate some arbitrary column to count —
#: which is a binding decision, made by the model, over an irrelevant field.
_BASE_CONCEPTS: Tuple[Dict[str, Any], ...] = (
    {"term": "loan", "label": "Loan",
     "description": "The loan itself. Count it to answer 'how many loans'.",
     "role": "measure",
     "allowed_statistics": ("count", "count_distinct", "share", "contribution"),
     "default_statistic": "count"},
    {"term": "case", "label": "Case",
     "description": "A pipeline case. Count it to answer 'how many cases'.",
     "role": "measure",
     "allowed_statistics": ("count", "count_distinct", "share", "contribution"),
     "default_statistic": "count"},
)


def _slug(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")


# --------------------------------------------------------------------------- #
# Concepts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SemanticConcept:
    """One governed business concept, and the binding the COMPILER may use.

    ``canonical_field`` is deliberately on this object and deliberately not in
    :meth:`GovernedVocabulary.prompt_payload`. It is the compiler's half of the
    contract; the model never sees it and never supplies it.
    """

    term: str
    label: str
    description: str
    role: str                                   # measure | dimension | date | flag
    synonyms: Tuple[str, ...] = ()
    allowed_statistics: Tuple[str, ...] = ()
    default_statistic: Optional[str] = None
    default_weight_term: Optional[str] = None
    value_domain: Optional[str] = None
    unit: Optional[str] = None
    #: The governed business spellings this dimension's values may take, where a
    #: governed enum domain declares them. Empty means NO governed enum exists —
    #: which the interpreter must be able to see, because a categorical filter
    #: whose value cannot be checked against anything is a guess, and the
    #: honest answer is to say so rather than assert it.
    values: Tuple[str, ...] = ()
    # ---- compiler-side binding; never exposed to the model ---------------- #
    canonical_field: Optional[str] = None
    owning_capability: Optional[str] = None     # set for specialist concepts

    @property
    def is_specialist(self) -> bool:
        return self.owning_capability is not None

    def public_view(self) -> Dict[str, Any]:
        """What the model is allowed to see about this concept."""
        view: Dict[str, Any] = {"term": self.term, "label": self.label,
                                "role": self.role}
        if self.description:
            view["description"] = self.description
        if self.synonyms:
            view["synonyms"] = list(self.synonyms)
        if self.allowed_statistics:
            view["statistics"] = list(self.allowed_statistics)
        if self.unit:
            view["unit"] = self.unit
        if self.values:
            view["values"] = list(self.values)
        elif self.role == "dimension" and not self.owning_capability:
            view["values"] = "NO GOVERNED VALUE LIST — do not assert a filter " \
                             "value on this dimension; flag it instead"
        if self.owning_capability:
            view["owned_by_capability"] = self.owning_capability
        return view


#: Registry aggregation spellings -> governed statistic names. The registry
#: predates this vocabulary and uses its own words; translating here keeps the
#: model's language stable while the registry evolves.
_AGG_TRANSLATION = {
    "sum": "sum", "avg": "average", "average": "average",
    "weighted_avg": "weighted_average", "weighted_average": "weighted_average",
    "median": "median", "min": "min", "max": "max",
    "count": "count", "count_distinct": "count_distinct",
    "share": "share", "contribution": "contribution",
    # registry modes that are not statistics of a measure
    "distribution": None, "loan_level": None, "balance_sum": "sum",
}

#: Registry roles -> vocabulary roles.
_ROLE_TRANSLATION = {"metric": "measure", "dimension": "dimension",
                     "date": "date", "flag": "flag"}

#: Concepts the geography contract owns. They are removed from the general
#: dimension vocabulary so a question about geography must travel through the
#: geography slot, where basis and level are stated and governed, instead of
#: arriving as a bare column-shaped dimension.
_GEOGRAPHY_FIELDS = frozenset({
    "canonical_region_reporting", "canonical_region_detail",
    "collateral_geography", "geographic_region_obligor",
    "geographic_region_collateral", "geographic_region_obligor_itl3",
    "geographic_region_collateral_itl3",
})

#: Fields whose meaning is an internal snapshot/reporting mechanic rather than
#: something a lender asks about. Excluded so the model is never offered a
#: physical snapshot handle dressed as a business concept.
_MECHANIC_FIELDS = frozenset({
    "reporting_date", "cut_off_date", "upload_timestamp",
    "pipeline_snapshot_date", "portfolio_id", "spv_id",
    "acquired_portfolio_id",
})


@dataclass(frozen=True)
class GovernedVocabulary:
    """The closed vocabulary one interpretation runs against.

    Immutable. ``for_book`` returns a new instance rather than mutating, so a
    vocabulary handed to an interpreter cannot change under it.
    """

    concepts: Mapping[str, SemanticConcept]
    capabilities: FrozenSet[str]
    version: str = VOCABULARY_VERSION
    book_id: Optional[str] = None
    #: Canonical fields the book carries. Empty means "not scoped" — every
    #: governed concept is offered.
    available_fields: FrozenSet[str] = frozenset()

    # -- lookups ------------------------------------------------------------ #

    def resolve(self, term: Optional[str]) -> Optional[SemanticConcept]:
        """The concept a term names, or None. Exact match only.

        Deliberately NOT fuzzy. Approximate field-name matching is how a model's
        near-miss becomes a confidently wrong binding; a term that is not the
        governed spelling is not a governed term.
        """
        if not isinstance(term, str):
            return None
        return self.concepts.get(term.strip().lower())

    def terms(self, role: Optional[str] = None) -> Tuple[str, ...]:
        if role is None:
            return tuple(sorted(self.concepts))
        return tuple(sorted(t for t, c in self.concepts.items() if c.role == role))

    def measures(self) -> Tuple[str, ...]:
        return self.terms("measure")

    def dimensions(self) -> Tuple[str, ...]:
        return self.terms("dimension")

    def is_available(self, concept: SemanticConcept) -> bool:
        """Whether this book carries the data behind a concept.

        A specialist concept has no canonical field: its availability is the
        capability's, checked separately by the compiler.
        """
        if concept.is_specialist or concept.canonical_field is None:
            return True
        if not self.available_fields:
            return True
        return concept.canonical_field in self.available_fields

    def for_book(self, available_fields: Iterable[str], *,
                 book_id: Optional[str] = None,
                 capabilities: Optional[Iterable[str]] = None) -> "GovernedVocabulary":
        """A view narrowed to what this book can actually answer."""
        present = frozenset(str(f).strip() for f in available_fields if f)
        kept = {t: c for t, c in self.concepts.items()
                if c.is_specialist or c.canonical_field is None
                or c.canonical_field in present}
        caps = (frozenset(capabilities) & self.capabilities
                if capabilities is not None else self.capabilities)
        # A specialist concept whose capability is gone goes with it.
        kept = {t: c for t, c in kept.items()
                if not c.is_specialist or c.owning_capability in caps}
        return replace(self, concepts=kept, capabilities=caps,
                       available_fields=present, book_id=book_id or self.book_id)

    # -- what the model is shown -------------------------------------------- #

    def prompt_payload(self, *, include_roles: Sequence[str] = ("measure",
                                                                "dimension",
                                                                "flag")) -> Dict[str, Any]:
        """The vocabulary as the interpreter presents it to the model.

        Contains SEMANTIC TERMS and closed enumerations only. It carries no
        canonical field, no snapshot, no portfolio value and no loan data —
        asserted by ``tests/interpretation_v2/test_model_sees_no_data.py``.
        """
        concepts: Dict[str, List[Dict[str, Any]]] = {}
        for role in include_roles:
            rows = [c.public_view() for c in self.concepts.values() if c.role == role]
            rows.sort(key=lambda r: r["term"])
            if rows:
                concepts[role] = rows
        return {
            "vocabulary_version": self.version,
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
            "concepts": concepts,
        }


# --------------------------------------------------------------------------- #
# Building the vocabulary from the governed registries
# --------------------------------------------------------------------------- #

def _concept_from_registry_entry(canonical_field: str,
                                 entry: Mapping[str, Any]) -> Optional[SemanticConcept]:
    role = _ROLE_TRANSLATION.get(str(entry.get("role") or "").strip().lower())
    if role is None:
        return None
    term = _slug(entry.get("business_name") or canonical_field)
    if not term:
        return None
    stats: List[str] = []
    for raw in entry.get("allowed_aggregations") or ():
        mapped = _AGG_TRANSLATION.get(str(raw).strip().lower())
        if mapped and mapped in STATISTICS and mapped not in stats:
            stats.append(mapped)
    default = _AGG_TRANSLATION.get(
        str(entry.get("default_aggregation") or "").strip().lower())
    if default is not None and default not in stats:
        stats.append(default)
    # Additivity is the REGISTRY's statement about the field, read before the
    # universal statistics are added below — every measure can be counted, and
    # letting that count make everything look additive would permit a share of
    # an average, which is not a quantity.
    additive = not ADDITIVE_STATISTICS.isdisjoint(stats)
    if role == "measure":
        # Every measure can be counted over and can bound a population.
        for extra in ("count", "min", "max"):
            if extra not in stats:
                stats.append(extra)
        if additive:
            stats.extend(m for m in sorted(ANALYTIC_MODES) if m not in stats)
    weight_field = entry.get("weight_field")
    return SemanticConcept(
        term=term,
        label=str(entry.get("business_name") or entry.get("display_name") or term),
        description=str(entry.get("business_description") or "").strip(),
        role=role,
        synonyms=tuple(str(s) for s in (entry.get("synonyms") or ())),
        allowed_statistics=tuple(sorted(stats)),
        default_statistic=default,
        default_weight_term=None,          # filled in once all terms are known
        value_domain=entry.get("value_domain"),
        unit=entry.get("format"),
        canonical_field=canonical_field,
    )


def _specialist_concepts(capabilities: Iterable[str]) -> Dict[str, SemanticConcept]:
    out: Dict[str, SemanticConcept] = {}
    caps = set(capabilities)
    for cap, terms in SPECIALIST_MEASURES.items():
        if cap not in caps:
            continue
        for term in terms:
            out[term] = SemanticConcept(
                term=term,
                label=term.replace("_", " ").title(),
                description=f"Owned by the {cap} capability; its methodology is "
                            f"deterministic and is not composed by the interpreter.",
                role="measure",
                allowed_statistics=(),   # the capability decides; not a free choice
                owning_capability=cap,
            )
    for cap, terms in SPECIALIST_DIMENSIONS.items():
        if cap not in caps:
            continue
        for term in terms:
            if term in out:
                continue
            out[term] = SemanticConcept(
                term=term,
                label=term.replace("_", " ").title(),
                description=f"A governed dimension of the {cap} capability.",
                role="dimension",
                owning_capability=cap,
            )
    return out


def _registered_capabilities() -> FrozenSet[str]:
    """The capabilities this build offers the model.

    Grounded in ``config/system/mi_capability_registry.yaml`` where a semantic
    capability has a registered deterministic owner there, and otherwise in the
    specialist routes this repository already ships. The map is explicit rather
    than inferred, because "which deterministic owner answers this" is a
    governance statement, not a naming coincidence.
    """
    return CAPABILITIES


@lru_cache(maxsize=4)
def _load_registry(path: str) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def _governed_enum_values() -> Mapping[str, Tuple[str, ...]]:
    """canonical_field -> the governed BUSINESS spellings of its values.

    Two committed registries, joined: ``fields_registry.yaml`` says which enum
    domain a field draws from, and ``enum_synonyms.yaml`` holds that domain's
    business spellings. Neither is redesigned here and neither is guessed at —
    a field whose ``allowed_values`` is null simply has no governed enum, and
    the vocabulary says so rather than inventing one.

    The ESMA codes on the right of the synonym map are deliberately NOT
    exposed: they are a physical encoding, and the model speaks business words.
    """
    try:
        fields = (_load_registry(str(FIELDS_REGISTRY_PATH)).get("fields") or {})
        domains = _load_registry(str(ENUM_SYNONYMS_PATH)) or {}
    except FileNotFoundError:  # pragma: no cover - registries are committed
        return {}

    spellings: Dict[str, Tuple[str, ...]] = {}
    for domain, block in domains.items():
        words: List[str] = []
        for mapping in (block or {}).values():
            if not isinstance(mapping, Mapping):
                continue
            for spelling in mapping:
                text = str(spelling).strip()
                # Skip the codes themselves — a key that maps to itself is the
                # canonical code, not a business word.
                if text and text != mapping[spelling] and text not in words:
                    words.append(text)
        if words:
            spellings[domain] = tuple(sorted(words))

    out: Dict[str, Tuple[str, ...]] = {}
    for canonical_field, entry in fields.items():
        domain = (entry or {}).get("allowed_values")
        if domain and domain in spellings:
            out[canonical_field] = spellings[domain]
    return out


#: Slots the compiler fills from a governed default when the question leaves
#: them empty. Declared to the model so it knows an empty slot is SAFE — the
#: first live run had the interpreter blocking on every bare "region" because
#: nothing told it that a governed default exists. Leaving a slot empty and
#: refusing to answer are different acts, and the model can only tell them
#: apart if it is told which slots have defaults.
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


@lru_cache(maxsize=1)
def load_governed_vocabulary() -> GovernedVocabulary:
    """Build the semantic vocabulary from the committed governed registries."""
    registry = _load_registry(str(MI_SEMANTICS_PATH))
    fields = registry.get("fields") or {}

    concepts: Dict[str, SemanticConcept] = {}
    collisions: Dict[str, List[str]] = {}
    for canonical_field, entry in sorted(fields.items()):
        if canonical_field in _GEOGRAPHY_FIELDS or canonical_field in _MECHANIC_FIELDS:
            continue
        concept = _concept_from_registry_entry(canonical_field, entry)
        if concept is None:
            continue
        if concept.term in concepts:
            collisions.setdefault(concept.term, [concepts[concept.term].canonical_field])
            collisions[concept.term].append(canonical_field)
            continue
        concepts[concept.term] = concept

    # A term that two governed fields both claim is AMBIGUOUS, not first-wins.
    # Dropping it means the compiler answers UNREGISTERED_CONCEPT rather than
    # silently binding one of the two — which is the whole rule this package
    # enforces, applied to its own construction.
    for term in collisions:
        concepts.pop(term, None)

    # Resolve default weights and governed value lists now that every term
    # exists.
    weight_terms = {c.canonical_field: c.term for c in concepts.values()}
    enum_values = _governed_enum_values()
    resolved: Dict[str, SemanticConcept] = {}
    for term, concept in concepts.items():
        entry = fields.get(concept.canonical_field) or {}
        wf = entry.get("weight_field")
        resolved[term] = replace(
            concept, default_weight_term=weight_terms.get(wf) if wf else None,
            values=enum_values.get(concept.canonical_field, ()))

    for base in _BASE_CONCEPTS:
        resolved[base["term"]] = SemanticConcept(**base)

    caps = _registered_capabilities()
    resolved.update(_specialist_concepts(caps))
    return GovernedVocabulary(concepts=resolved, capabilities=caps)


def canonical_field_names() -> FrozenSet[str]:
    """Every canonical field the MI semantics registry knows.

    Used by the intent parser to reject a physical column name arriving in a
    semantic slot, and by tests to prove the model was never shown one.
    """
    registry = _load_registry(str(MI_SEMANTICS_PATH))
    return frozenset((registry.get("fields") or {}).keys())
