"""CandidateIntent — what the user MEANT, as the model understood it.

This is not a query. It is a reading of a sentence.

The distinction this module exists to hold is the whole architecture::

    CandidateIntent      describes WHAT THE USER MEANS
    GovernedQueryPlan    describes WHAT TRAKT IS ALLOWED TO EXECUTE

So the model may say ``measure = loan_to_value``, ``weight = balance``,
``geography.basis = collateral``, ``geography.level = itl3``,
``time.form = previous_reporting_period``. It may not say
``field = current_ltv``, ``snapshot = 2026-06-30``, or anything that is a
dataframe expression, a SQL string or Python.

That is enforced two ways, and both matter:

1. **Structurally.** There is no slot on any object in this module that takes a
   physical field, a snapshot identifier, a column, or code. A slot that does
   not exist cannot be filled, whatever the model returns.
   ``tests/interpretation_v2/test_intent_has_no_physical_slots.py`` walks the
   dataclass tree and fails if one is ever added.

2. **On parse.** :func:`parse_candidate_intent` is fail-closed: an unknown key,
   an out-of-vocabulary enum, a code marker, an ISO date or a snapshot-shaped
   token anywhere in the payload rejects the whole intent. Nothing is stripped
   and salvaged — a payload that tried to reach past the boundary does not get a
   second, tidier chance.

Free text survives in exactly two places, and neither binds anything: the
``label`` on a period the question named in words ("April", "last month"), and
``evidence`` / ``ambiguity`` spans quoted from the question. Both are validated
against the same code, date and snapshot guards as everything else.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .vocabulary import (
    COMPARATORS,
    COMPARISON_KINDS,
    GEOGRAPHY_BASES,
    GEOGRAPHY_LEVELS,
    OPERATIONS,
    POPULATION_BASES,
    POPULATION_LENSES,
    SEASONING_SEGMENTS,
    SEASONING_SEGMENTS as _SEASONING,
    STATISTICS,
    TIME_FORMS,
    TIME_GRAINS,
    CAPABILITIES,
)

#: The CandidateIntent contract version. A payload declaring anything else is
#: refused rather than read optimistically — a schema drift that parses is worse
#: than one that fails.
INTENT_SCHEMA_VERSION = "candidate_intent/1.0"


class IntentParseError(ValueError):
    """A payload that cannot become a CandidateIntent. Carries governed codes."""

    def __init__(self, code: str, subject: str = "", detail: str = "") -> None:
        super().__init__(f"{code}: {subject} {detail}".strip())
        self.code = code
        self.subject = subject
        self.detail = detail


# --------------------------------------------------------------------------- #
# Guards on every string the model produces
# --------------------------------------------------------------------------- #

#: Markers that are code in ANY slot, including the two free-text provenance
#: slots. None of these is a thing an English sentence contains.
_HARD_CODE_MARKERS: Tuple[str, ...] = (
    "```", "import ", "def ", "lambda ", "pd.", "df[", "df.", "np.",
    "print(", "eval(", "exec(", ".query(", ".groupby(", ".agg(",
    ".loc[", ".iloc[", "${", "<script",
)

#: Additionally rejected in every BINDING slot. These are SQL words, and SQL
#: words are also ordinary English — "count the loans WHERE current LTV exceeds
#: 50%" is a question, not a query. They are refused where a value would bind
#: something and allowed where the model is quoting the user back at us, which
#: is the only place they can legitimately appear.
_SQL_CODE_MARKERS: Tuple[str, ...] = (
    "select ", " from ", "where ", "group by", "order by", "inner join",
    "left join", " union ", "insert into", "drop table",
)

_CODE_MARKERS: Tuple[str, ...] = _HARD_CODE_MARKERS + _SQL_CODE_MARKERS

#: Payload keys whose content is the model QUOTING THE QUESTION. They bind
#: nothing, are carried into provenance, and are read by no decision — so they
#: are guarded against code, and not against the user's own words.
_FREE_TEXT_KEYS = frozenset({"evidence", "ambiguity"})

#: A snapshot identifier or an explicit calendar date. The model states time
#: SEMANTICALLY; resolving "the previous reporting period" to a snapshot is the
#: compiler's job and needs the governed period contract to do it.
_DATE_LIKE = re.compile(
    r"""(
        \b\d{4}-\d{2}-\d{2}\b            # 2026-06-30
      | \b\d{2}/\d{2}/\d{4}\b            # 30/06/2026
      | \b\d{4}\d{2}\d{2}\b              # 20260630
      | \bsnapshot[\s_\-:]*\w+\b         # snapshot_17, snapshot: abc
      | \bas[_ ]of[_ ]\d                 # as_of_2026
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def _guard_string(value: str, *, slot: str) -> None:
    """Reject code and physical bindings wherever a BINDING string appears."""
    lowered = value.lower()
    for marker in _CODE_MARKERS:
        if marker in lowered:
            raise IntentParseError("MODEL_OUTPUT_CONTAINS_CODE", slot,
                                   f"code marker {marker!r}")
    if _DATE_LIKE.search(value):
        raise IntentParseError("MODEL_OUTPUT_CONTAINS_PHYSICAL_BINDING", slot,
                               "an explicit date or snapshot identifier")


def _guard_free_text(value: str, *, slot: str) -> None:
    """Guard a quoted-provenance string: code only, not the user's own words."""
    lowered = value.lower()
    for marker in _HARD_CODE_MARKERS:
        if marker in lowered:
            raise IntentParseError("MODEL_OUTPUT_CONTAINS_CODE", slot,
                                   f"code marker {marker!r}")


def _guard_payload(node: Any, *, slot: str = "root", free_text: bool = False) -> None:
    """Walk the raw payload and guard every string in it, at any depth."""
    guard = _guard_free_text if free_text else _guard_string
    if isinstance(node, str):
        guard(node, slot=slot)
    elif isinstance(node, Mapping):
        for key, value in node.items():
            if isinstance(key, str):
                _guard_string(key, slot=f"{slot}.<key>")
            _guard_payload(value, slot=f"{slot}.{key}",
                           free_text=free_text or key in _FREE_TEXT_KEYS)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _guard_payload(value, slot=f"{slot}[{index}]", free_text=free_text)


def _term(value: Any, *, slot: str) -> str:
    """A semantic term as the model spelled it — lowercased, never resolved.

    Resolution is the compiler's. This only normalises case and whitespace so an
    exact-match lookup is not defeated by capitalisation.
    """
    if not isinstance(value, str) or not value.strip():
        raise IntentParseError("INTENT_SCHEMA_INVALID", slot, "expected a term")
    text = value.strip().lower()
    _guard_string(text, slot=slot)
    if len(text) > 80:
        raise IntentParseError("INTENT_SCHEMA_INVALID", slot, "term too long")
    return text


def _enum(value: Any, allowed, *, slot: str, optional: bool = False) -> Optional[str]:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise IntentParseError("INTENT_SCHEMA_INVALID", slot, "expected a string")
    text = value.strip().lower()
    if text not in allowed:
        raise IntentParseError("INTENT_SCHEMA_INVALID", slot,
                               f"{text!r} is not one of {sorted(allowed)}")
    return text


def _known_keys(payload: Mapping[str, Any], allowed: Sequence[str], *, slot: str) -> None:
    """Fail closed on any key the contract does not define.

    An unknown key is how a physical binding would arrive if one ever could —
    ``{"measure": "balance", "field": "current_outstanding_balance"}``. Ignoring
    it would let the model believe it had been heard.
    """
    extra = sorted(set(payload) - set(allowed))
    if extra:
        raise IntentParseError("MODEL_OUTPUT_CONTAINS_PHYSICAL_BINDING"
                               if any(k in _PHYSICAL_KEY_NAMES for k in extra)
                               else "INTENT_SCHEMA_INVALID",
                               slot, f"unknown keys {extra}")


#: Key names that are unambiguously an attempt to supply a binding rather than a
#: meaning. Reported distinctly so the telemetry can tell a schema slip from a
#: boundary breach.
_PHYSICAL_KEY_NAMES = frozenset({
    "field", "field_key", "fields", "column", "columns", "canonical_field",
    "snapshot", "snapshot_id", "as_of", "as_of_date", "reporting_date",
    "sql", "query", "expression", "pandas", "dataframe", "code", "python",
    "spec", "mi_query_spec", "filters_sql", "predicate_expression",
})


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SourceSpan:
    """A fragment of the original question supporting a claim.

    Provenance only. The compiler may record a span; it may never read one to
    decide what the question meant — that decision is already made, upstream, by
    the model, and re-reading the English here would rebuild the very thing this
    architecture replaced.
    """

    claim: str
    text: str


@dataclass(frozen=True)
class Ambiguity:
    """Something the model noticed about the question that a reader should know.

    Two different things, and the difference decides an outcome:

    ``blocking=True``   the model could NOT choose, and left the slot empty.
                        The compiler clarifies: asking is the honest answer.
    ``blocking=False``  the model chose, and is disclosing the reading it took.
                        The compiler plans, and carries the disclosure into
                        provenance.

    Collapsing the two was tried first and made almost every question a
    clarification, because a careful reader flags an assumption on nearly every
    sentence. A disclosed assumption is a feature; a refusal to answer because
    somebody was careful is not.
    """

    slot: str
    note: str
    options: Tuple[str, ...] = ()
    blocking: bool = False


@dataclass(frozen=True)
class SemanticFilter:
    """A row predicate stated in business terms.

    ``concept`` is a semantic term. ``value`` is a literal lifted from the
    question ("55", "London", "drawdown") — the user's own data, not a binding.
    """

    concept: str
    comparator: str
    value: Any = None

    def key(self) -> Tuple[Any, ...]:
        value = tuple(sorted(map(str, self.value))) if isinstance(self.value, (list, tuple)) \
            else self.value
        return (self.concept, self.comparator, value)


@dataclass(frozen=True)
class SemanticMeasure:
    """One thing to measure, and how to reduce it.

    ``weight`` is a semantic term too — "weighted by balance", never
    "weight_field: current_outstanding_balance".
    """

    concept: str
    statistic: Optional[str] = None
    weight: Optional[str] = None

    def key(self) -> Tuple[Any, ...]:
        return (self.concept, self.statistic, self.weight)


@dataclass(frozen=True)
class SemanticGeography:
    """A geography REQUEST: whose geography, and how fine.

    Neither axis is a column. ``mi_agent.region_basis`` governs which of the
    seven region fields a (basis, level) pair lands on, and the compiler applies
    it — see :mod:`mi_agent.interpretation_v2.compiler`.
    """

    requested: bool = False
    basis: Optional[str] = None
    level: Optional[str] = None
    #: Break the answer down by geography ("balance BY region").
    group_by: bool = False
    #: Restrict the answer to named places ("balance IN London"). The place
    #: names are the question's own words; which column they are matched
    #: against is the compiler's decision, not the model's.
    values: Tuple[Any, ...] = ()

    def key(self) -> Tuple[Any, ...]:
        return (self.requested, self.basis, self.level, self.group_by,
                tuple(sorted(map(str, self.values))))


@dataclass(frozen=True)
class SemanticTime:
    """The time the question named, in the only form the model may state it.

    ``labels`` carries the question's own words for a named period ("April",
    "last month", "the quarter"). It is a LABEL, not a resolution: the compiler
    hands it to the governed period contract, which decides whether the book
    reaches that far, and refuses if it does not.
    """

    form: str = "current"
    labels: Tuple[str, ...] = ()
    grain: Optional[str] = None
    periods_back: Optional[int] = None

    def key(self) -> Tuple[Any, ...]:
        return (self.form, tuple(self.labels), self.grain, self.periods_back)


@dataclass(frozen=True)
class SemanticPopulation:
    """Which rows the question is about, stated semantically."""

    base: str = "funded"
    lens: str = "all"
    seasoning: str = "any"

    def key(self) -> Tuple[Any, ...]:
        return (self.base, self.lens, self.seasoning)


@dataclass(frozen=True)
class SemanticComparison:
    """Two things held against each other, named semantically."""

    kind: str = "none"
    left: Optional[str] = None
    right: Optional[str] = None

    def key(self) -> Tuple[Any, ...]:
        return (self.kind, self.left, self.right)


@dataclass(frozen=True)
class RequestedOutput:
    """One figure or table the question asks for.

    Multi-output is first-class from day one. "How many joint-borrower loans are
    there, what balance do they represent, and how much of that balance is above
    40% LTV?" is ONE population with THREE outputs, the third carrying an
    output-local predicate — not three unrelated questions whose populations
    might quietly differ.
    """

    id: str
    measures: Tuple[SemanticMeasure, ...] = ()
    dimensions: Tuple[str, ...] = ()
    filters: Tuple[SemanticFilter, ...] = ()
    geography: SemanticGeography = field(default_factory=SemanticGeography)

    def key(self) -> Tuple[Any, ...]:
        return (
            tuple(sorted(m.key() for m in self.measures)),
            tuple(sorted(self.dimensions)),
            tuple(sorted(f.key() for f in self.filters)),
            self.geography.key(),
        )


@dataclass(frozen=True)
class IntentProvenance:
    """Where this reading came from. Never an input to any decision."""

    question: str = ""
    model_id: str = ""
    interpreter_version: str = ""
    vocabulary_version: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateIntent:
    """A machine-readable reading of one question. Not executable.

    Every slot on this object is semantic. There is no field, column, snapshot,
    expression or code slot, and there is no place to add a value that the
    compiler would use verbatim as a binding.
    """

    schema_version: str
    capability: str
    operation: str
    population: SemanticPopulation = field(default_factory=SemanticPopulation)
    measures: Tuple[SemanticMeasure, ...] = ()
    dimensions: Tuple[str, ...] = ()
    filters: Tuple[SemanticFilter, ...] = ()
    geography: SemanticGeography = field(default_factory=SemanticGeography)
    time: SemanticTime = field(default_factory=SemanticTime)
    comparison: SemanticComparison = field(default_factory=SemanticComparison)
    outputs: Tuple[RequestedOutput, ...] = ()
    ambiguity: Tuple[Ambiguity, ...] = ()
    evidence: Tuple[SourceSpan, ...] = ()
    provenance: IntentProvenance = field(default_factory=IntentProvenance)

    # -- derived views ------------------------------------------------------ #

    def effective_outputs(self) -> Tuple[RequestedOutput, ...]:
        """The outputs to compile.

        A single-figure question needs no ``outputs`` block, so one is
        synthesised from the top-level measures. Everything downstream then has
        exactly one shape to handle, and the one-question/one-metric limitation
        is never baked in anywhere.
        """
        if self.outputs:
            return self.outputs
        return (RequestedOutput(id="primary", measures=self.measures,
                                dimensions=self.dimensions, filters=(),
                                geography=SemanticGeography()),)

    def semantic_key(self) -> Tuple[Any, ...]:
        """An order-insensitive fingerprint of the MEANING.

        Excludes provenance and evidence: two paraphrases that mean the same
        thing quote different words, and a fingerprint that noticed would be
        measuring wording, not meaning.
        """
        return (
            self.capability,
            self.operation,
            self.population.key(),
            tuple(sorted(m.key() for m in self.measures)),
            tuple(sorted(self.dimensions)),
            tuple(sorted(f.key() for f in self.filters)),
            self.geography.key(),
            self.time.key(),
            self.comparison.key(),
            tuple(sorted(o.key() for o in self.effective_outputs())),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Fail-closed parsing
# --------------------------------------------------------------------------- #

_INTENT_KEYS = ("schema_version", "capability", "operation", "population",
                "measures", "dimensions", "filters", "geography", "time",
                "comparison", "outputs", "ambiguity", "evidence")
_MEASURE_KEYS = ("concept", "statistic", "weight")
_FILTER_KEYS = ("concept", "comparator", "value")
_GEO_KEYS = ("requested", "basis", "level", "group_by", "values")
_TIME_KEYS = ("form", "labels", "grain", "periods_back")
_POP_KEYS = ("base", "lens", "seasoning")
_CMP_KEYS = ("kind", "left", "right")
_OUTPUT_KEYS = ("id", "measures", "dimensions", "filters", "geography")
_AMBIG_KEYS = ("slot", "note", "options", "blocking")
_EVIDENCE_KEYS = ("claim", "text")

#: A filter value may only be a scalar or a flat list of scalars. A nested
#: object here would be a predicate tree — a query, not a meaning.
_SCALAR = (str, int, float, bool)


def _parse_value(raw: Any, *, slot: str) -> Any:
    if raw is None or isinstance(raw, _SCALAR):
        if isinstance(raw, str):
            _guard_string(raw, slot=slot)
        return raw
    if isinstance(raw, (list, tuple)):
        out = []
        for index, item in enumerate(raw):
            if not isinstance(item, _SCALAR):
                raise IntentParseError("INTENT_SCHEMA_INVALID", f"{slot}[{index}]",
                                       "filter values must be scalars")
            if isinstance(item, str):
                _guard_string(item, slot=f"{slot}[{index}]")
            out.append(item)
        return out
    raise IntentParseError("INTENT_SCHEMA_INVALID", slot,
                           "filter value must be a scalar or a list of scalars")


def _parse_measure(raw: Any, *, slot: str) -> SemanticMeasure:
    if isinstance(raw, str):
        return SemanticMeasure(concept=_term(raw, slot=slot))
    if not isinstance(raw, Mapping):
        raise IntentParseError("INTENT_SCHEMA_INVALID", slot, "expected an object")
    _known_keys(raw, _MEASURE_KEYS, slot=slot)
    return SemanticMeasure(
        concept=_term(raw.get("concept"), slot=f"{slot}.concept"),
        statistic=_enum(raw.get("statistic"), STATISTICS, slot=f"{slot}.statistic",
                        optional=True),
        weight=(_term(raw["weight"], slot=f"{slot}.weight")
                if raw.get("weight") is not None else None),
    )


def _parse_filter(raw: Any, *, slot: str) -> SemanticFilter:
    if not isinstance(raw, Mapping):
        raise IntentParseError("INTENT_SCHEMA_INVALID", slot, "expected an object")
    _known_keys(raw, _FILTER_KEYS, slot=slot)
    return SemanticFilter(
        concept=_term(raw.get("concept"), slot=f"{slot}.concept"),
        comparator=_enum(raw.get("comparator"), COMPARATORS,
                         slot=f"{slot}.comparator"),
        value=_parse_value(raw.get("value"), slot=f"{slot}.value"),
    )


def _parse_geography(raw: Any, *, slot: str) -> SemanticGeography:
    if raw is None:
        return SemanticGeography()
    if not isinstance(raw, Mapping):
        raise IntentParseError("INTENT_SCHEMA_INVALID", slot, "expected an object")
    _known_keys(raw, _GEO_KEYS, slot=slot)
    basis = _enum(raw.get("basis"), GEOGRAPHY_BASES, slot=f"{slot}.basis",
                  optional=True)
    level = _enum(raw.get("level"), GEOGRAPHY_LEVELS, slot=f"{slot}.level",
                  optional=True)
    raw_values = raw.get("values") or ()
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    values = _parse_value(list(raw_values), slot=f"{slot}.values")
    requested = bool(raw.get("requested",
                             bool(basis or level or values
                                  or raw.get("group_by"))))
    # A geography that names no places must be a grouping, or it says nothing.
    group_by = bool(raw.get("group_by", requested and not values))
    return SemanticGeography(requested=requested, basis=basis, level=level,
                             group_by=group_by, values=tuple(values))


def _parse_time(raw: Any, *, slot: str) -> SemanticTime:
    if raw is None:
        return SemanticTime()
    if not isinstance(raw, Mapping):
        raise IntentParseError("INTENT_SCHEMA_INVALID", slot, "expected an object")
    _known_keys(raw, _TIME_KEYS, slot=slot)
    labels_raw = raw.get("labels") or ()
    if isinstance(labels_raw, str):
        labels_raw = [labels_raw]
    if not isinstance(labels_raw, (list, tuple)):
        raise IntentParseError("INTENT_SCHEMA_INVALID", f"{slot}.labels",
                               "expected a list of period labels")
    labels = []
    for index, label in enumerate(labels_raw):
        if not isinstance(label, str):
            raise IntentParseError("INTENT_SCHEMA_INVALID", f"{slot}.labels[{index}]",
                                   "expected a string")
        text = label.strip()
        _guard_string(text, slot=f"{slot}.labels[{index}]")
        if text:
            labels.append(text)
    periods_back = raw.get("periods_back")
    if periods_back is not None:
        if not isinstance(periods_back, int) or isinstance(periods_back, bool) \
                or not 0 <= periods_back <= 120:
            raise IntentParseError("INTENT_SCHEMA_INVALID", f"{slot}.periods_back",
                                   "expected an integer between 0 and 120")
    return SemanticTime(
        form=_enum(raw.get("form", "current"), TIME_FORMS, slot=f"{slot}.form"),
        labels=tuple(labels),
        grain=_enum(raw.get("grain"), TIME_GRAINS, slot=f"{slot}.grain",
                    optional=True),
        periods_back=periods_back,
    )


def _parse_population(raw: Any, *, slot: str) -> SemanticPopulation:
    if raw is None:
        return SemanticPopulation()
    if not isinstance(raw, Mapping):
        raise IntentParseError("INTENT_SCHEMA_INVALID", slot, "expected an object")
    _known_keys(raw, _POP_KEYS, slot=slot)
    return SemanticPopulation(
        base=_enum(raw.get("base", "funded"), POPULATION_BASES, slot=f"{slot}.base"),
        lens=_enum(raw.get("lens", "all"), POPULATION_LENSES, slot=f"{slot}.lens"),
        seasoning=_enum(raw.get("seasoning", "any"), _SEASONING,
                        slot=f"{slot}.seasoning"),
    )


def _parse_comparison(raw: Any, *, slot: str) -> SemanticComparison:
    if raw is None:
        return SemanticComparison()
    if not isinstance(raw, Mapping):
        raise IntentParseError("INTENT_SCHEMA_INVALID", slot, "expected an object")
    _known_keys(raw, _CMP_KEYS, slot=slot)
    left, right = raw.get("left"), raw.get("right")
    return SemanticComparison(
        kind=_enum(raw.get("kind", "none"), COMPARISON_KINDS, slot=f"{slot}.kind"),
        left=_term(left, slot=f"{slot}.left") if left is not None else None,
        right=_term(right, slot=f"{slot}.right") if right is not None else None,
    )


def _parse_output(raw: Any, *, slot: str, index: int) -> RequestedOutput:
    if not isinstance(raw, Mapping):
        raise IntentParseError("INTENT_SCHEMA_INVALID", slot, "expected an object")
    _known_keys(raw, _OUTPUT_KEYS, slot=slot)
    raw_id = raw.get("id")
    output_id = _term(raw_id, slot=f"{slot}.id") if raw_id else f"output_{index + 1}"
    return RequestedOutput(
        id=output_id,
        measures=tuple(_parse_measure(m, slot=f"{slot}.measures[{i}]")
                       for i, m in enumerate(raw.get("measures") or ())),
        dimensions=tuple(_term(d, slot=f"{slot}.dimensions[{i}]")
                         for i, d in enumerate(raw.get("dimensions") or ())),
        filters=tuple(_parse_filter(f, slot=f"{slot}.filters[{i}]")
                      for i, f in enumerate(raw.get("filters") or ())),
        geography=_parse_geography(raw.get("geography"), slot=f"{slot}.geography"),
    )


def parse_candidate_intent(payload: Any, *,
                           provenance: Optional[IntentProvenance] = None
                           ) -> CandidateIntent:
    """Turn a model payload into a CandidateIntent, or raise IntentParseError.

    Fail-closed throughout. There is no partial parse and no repair loop: a
    payload that does not satisfy the contract produces no intent, and therefore
    no plan.
    """
    if not isinstance(payload, Mapping):
        raise IntentParseError("MODEL_OUTPUT_MALFORMED", "root",
                               "expected a JSON object")
    _guard_payload(payload)
    _known_keys(payload, _INTENT_KEYS, slot="root")

    version = payload.get("schema_version")
    if version != INTENT_SCHEMA_VERSION:
        raise IntentParseError("INTENT_SCHEMA_VERSION_UNSUPPORTED", "schema_version",
                               f"{version!r} != {INTENT_SCHEMA_VERSION!r}")

    ambiguity = []
    for index, raw in enumerate(payload.get("ambiguity") or ()):
        slot = f"ambiguity[{index}]"
        if not isinstance(raw, Mapping):
            raise IntentParseError("INTENT_SCHEMA_INVALID", slot, "expected an object")
        _known_keys(raw, _AMBIG_KEYS, slot=slot)
        options = raw.get("options") or ()
        if isinstance(options, str):
            options = [options]
        ambiguity.append(Ambiguity(
            slot=str(raw.get("slot") or "").strip(),
            note=str(raw.get("note") or "").strip(),
            options=tuple(str(o).strip() for o in options if str(o).strip()),
            blocking=bool(raw.get("blocking", False)),
        ))

    evidence = []
    for index, raw in enumerate(payload.get("evidence") or ()):
        slot = f"evidence[{index}]"
        if not isinstance(raw, Mapping):
            raise IntentParseError("INTENT_SCHEMA_INVALID", slot, "expected an object")
        _known_keys(raw, _EVIDENCE_KEYS, slot=slot)
        evidence.append(SourceSpan(claim=str(raw.get("claim") or "").strip(),
                                   text=str(raw.get("text") or "").strip()))

    return CandidateIntent(
        schema_version=INTENT_SCHEMA_VERSION,
        capability=_enum(payload.get("capability"), CAPABILITIES, slot="capability"),
        operation=_enum(payload.get("operation"), OPERATIONS, slot="operation"),
        population=_parse_population(payload.get("population"), slot="population"),
        measures=tuple(_parse_measure(m, slot=f"measures[{i}]")
                       for i, m in enumerate(payload.get("measures") or ())),
        dimensions=tuple(_term(d, slot=f"dimensions[{i}]")
                         for i, d in enumerate(payload.get("dimensions") or ())),
        filters=tuple(_parse_filter(f, slot=f"filters[{i}]")
                      for i, f in enumerate(payload.get("filters") or ())),
        geography=_parse_geography(payload.get("geography"), slot="geography"),
        time=_parse_time(payload.get("time"), slot="time"),
        comparison=_parse_comparison(payload.get("comparison"), slot="comparison"),
        outputs=tuple(_parse_output(o, slot=f"outputs[{i}]", index=i)
                      for i, o in enumerate(payload.get("outputs") or ())),
        ambiguity=tuple(ambiguity),
        evidence=tuple(evidence),
        provenance=provenance or IntentProvenance(),
    )


# --------------------------------------------------------------------------- #
# The JSON schema the model generates against
# --------------------------------------------------------------------------- #

def candidate_intent_json_schema() -> Dict[str, Any]:
    """A strict JSON schema for structured generation.

    ``additionalProperties: false`` everywhere: the schema itself refuses a
    ``field`` or ``snapshot`` key, so a compliant generation cannot produce one
    and a non-compliant one is caught by :func:`parse_candidate_intent`. Two
    layers, because a schema is a request and a parser is a guarantee.
    """
    term = {"type": "string", "maxLength": 80,
            "description": "A governed SEMANTIC term from the supplied "
                           "vocabulary. Never a database column."}
    measure = {
        "type": "object", "additionalProperties": False,
        "required": ["concept"],
        "properties": {
            "concept": term,
            "statistic": {"type": "string", "enum": sorted(STATISTICS)},
            "weight": dict(term, description="Semantic term to weight by, e.g. "
                                             "'balance'. Only for a weighted "
                                             "average or a contribution."),
        },
    }
    filt = {
        "type": "object", "additionalProperties": False,
        "required": ["concept", "comparator"],
        "properties": {
            "concept": term,
            "comparator": {"type": "string", "enum": sorted(COMPARATORS)},
            "value": {"description": "A literal from the question — a number, a "
                                     "string, or a flat list of them."},
        },
    }
    geography = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "requested": {"type": "boolean"},
            "basis": {"type": "string", "enum": sorted(GEOGRAPHY_BASES),
                      "description": "Whose geography: the borrower's "
                                     "(obligor), the property's (collateral), "
                                     "or the client's reporting taxonomy. Leave "
                                     "empty if the question does not say."},
            "level": {"type": "string", "enum": sorted(GEOGRAPHY_LEVELS)},
            "group_by": {"type": "boolean",
                         "description": "True for 'balance BY region'."},
            "values": {"type": "array", "items": {"type": "string"},
                       "description": "Place names the question restricts to, "
                                      "e.g. ['London']. The question's own "
                                      "words."},
        },
    }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "capability", "operation"],
        "properties": {
            "schema_version": {"type": "string", "const": INTENT_SCHEMA_VERSION},
            "capability": {"type": "string", "enum": sorted(CAPABILITIES)},
            "operation": {"type": "string", "enum": sorted(OPERATIONS)},
            "population": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "base": {"type": "string", "enum": sorted(POPULATION_BASES)},
                    "lens": {"type": "string", "enum": sorted(POPULATION_LENSES)},
                    "seasoning": {"type": "string", "enum": sorted(SEASONING_SEGMENTS)},
                },
            },
            "measures": {"type": "array", "items": measure},
            "dimensions": {"type": "array", "items": term},
            "filters": {"type": "array", "items": filt},
            "geography": geography,
            "time": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "form": {"type": "string", "enum": sorted(TIME_FORMS)},
                    "labels": {"type": "array", "items": {
                        "type": "string",
                        "description": "The question's own words for a period "
                                       "('last month', 'April'). NEVER a date "
                                       "or a snapshot id."}},
                    "grain": {"type": "string", "enum": sorted(TIME_GRAINS)},
                    "periods_back": {"type": "integer", "minimum": 0, "maximum": 120},
                },
            },
            "comparison": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": sorted(COMPARISON_KINDS)},
                    "left": term,
                    "right": term,
                },
            },
            "outputs": {
                "type": "array",
                "description": "One entry per figure or table the question asks "
                               "for. Use this whenever a question asks for more "
                               "than one thing about the SAME population.",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string", "maxLength": 80},
                        "measures": {"type": "array", "items": measure},
                        "dimensions": {"type": "array", "items": term},
                        "filters": {"type": "array", "items": filt,
                                    "description": "Predicates that apply to "
                                                   "THIS output only."},
                        "geography": geography,
                    },
                },
            },
            "ambiguity": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["slot", "note", "blocking"],
                    "properties": {
                        "slot": {"type": "string"},
                        "note": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "blocking": {
                            "type": "boolean",
                            "description": "true ONLY if you could not choose "
                                           "and left the slot empty. false if "
                                           "you chose and are disclosing the "
                                           "reading you took."},
                    },
                },
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["claim", "text"],
                    "properties": {
                        "claim": {"type": "string"},
                        "text": {"type": "string",
                                 "description": "The words from the question "
                                                "that support the claim."},
                    },
                },
            },
        },
    }
