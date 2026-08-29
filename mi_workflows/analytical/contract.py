"""mi_workflows.analytical.contract — the analytical capability contract.

ONE contract, describing capabilities that already exist. Nothing here computes
anything: a :class:`AnalyticalCapability` is a *declaration* that binds a stable
capability id to a deterministic executor the repository already owns, and says
what that executor needs, what it produces and what it cannot do.

Why a contract at all
---------------------
The Recogniser Registry answers "which ONE governed capability answers this
question?". A CFO question like *"what is the current value of outstanding
offers, how much do we expect to complete, and when?"* has no single answer —
it is a pipeline stock reading and a completion forecast, composed. Composition
needs three things the registry deliberately does not carry
(``recogniser_registry`` module docstring: *"routes to ONE deterministic
capability and stops"*):

* a **name** for each analytical operation, independent of the route that
  happens to expose it;
* the **inputs** each operation requires, so a plan can be validated before any
  data is touched;
* a **findings** shape all of them share, so a narrative can be composed from
  results rather than from prose.

What this is NOT
----------------
Not a second field registry and not a second business-semantics registry.
Measure identity, statistic identity, units, weighting and directionality all
stay where they already live (``mi_agent/mi_semantics_field_registry.yaml``,
``config/business_semantics_registry.yaml``, ``mi_workflows.engine``). A
capability declaration references them; it never restates them.

Not a capability *availability* model either — that is
``config/system/mi_capability_registry.yaml`` + ``trakt_core.capability``, which
answers "can this measure be produced for this portfolio?". This contract
answers the orthogonal question: "what analytical operations can be composed,
and by which deterministic executor?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

#: Version of the analytical composition semantics. Recorded on every plan so a
#: historical answer can be tied to the composition rules that produced it. The
#: deterministic calculations themselves carry their OWN versions (see
#: ``mi_workflows.engine.CALCULATION_VERSION`` and
#: ``mi_agent.period_change.CALCULATION_VERSION``) — this layer never restates
#: them, it records which of them ran.
COMPOSITION_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Scope references — what a finding was measured over
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PopulationRef:
    """The rows a finding was measured over, and the proof that they were.

    ``rows`` is execution evidence, not intent: a population that claims to have
    narrowed must be able to say to what. ``predicate`` is the governed
    row-predicate description produced by :mod:`mi_agent.population`, or the
    portfolio-lens label — never a phrase re-derived from the question.
    """

    key: str
    label: str
    predicate: Optional[str] = None
    rows: Optional[int] = None
    rows_before: Optional[int] = None
    #: Total exposure of the population, where the measure basis is monetary.
    exposure: Optional[float] = None
    #: True when this population is the whole governed scope (no narrowing).
    is_total: bool = False
    #: Which governed dataset these rows come from. Load-bearing: the P1L row
    #: population ledger is about the FUNDED book, and a pipeline-stage
    #: predicate published into it would be a predicate on a different set of
    #: rows entirely.
    dataset: str = "funded"
    #: Membership of this population at the START of a compared period, where a
    #: period was compared. NOT the same as ``rows_before``, which is the count
    #: before the predicate narrowed the CURRENT snapshot.
    #:
    #: This is the count the layer already computed per snapshot and discarded;
    #: it is surfaced rather than recomputed, so nothing new is calculated.
    rows_prior: Optional[int] = None
    #: True when membership of this population is defined by a TIME-RELATIVE
    #: predicate — a months-on-book window or a seasoning segment. Such a
    #: population is a rolling cohort: a loan joins or leaves it by the passage
    #: of time alone, so the same label denotes a different row set at each
    #: reporting date.
    #:
    #: Provenance and dimension-value populations are NOT time-relative. Their
    #: counts also move between snapshots, but only because loans are originated
    #: or exit — which is genuine book movement, and reads correctly as such.
    time_relative: bool = False
    #: The governed ``(field, value)`` pairs this population narrowed on that
    #: ``predicate`` does not name a field for.
    #:
    #: ``predicate`` is written for a READER, and one narrowing this layer
    #: applies is written in a form no reader of the contract can parse: a
    #: portfolio lens describes itself as ``"portfolio lens = Direct
    #: (direct_001)"``, which names a lens and never the field the lens scopes.
    #: Every other predicate here arrives from ``mi_agent.population`` already
    #: field-named, so every other narrowing is legible downstream and that one
    #: was not — the plan performed it, and no consumer could see that it had.
    #:
    #: This is the machine-readable twin, not a second narrowing: it records
    #: what the SAME filter scoped the rows to. A consumer reconciling "was this
    #: answer narrowed to Direct?" reads it here; nothing is re-derived, and the
    #: reader-facing predicate is unchanged.
    narrowed_on: Tuple[Tuple[str, str], ...] = ()

    @property
    def membership_changed(self) -> bool:
        """Whether this population denotes a different row set across the period."""
        return (self.time_relative and self.rows_prior is not None
                and self.rows is not None and self.rows_prior != self.rows)

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "predicate": self.predicate,
                "rows": self.rows, "rowsBefore": self.rows_before,
                "rowsPrior": self.rows_prior, "timeRelative": self.time_relative,
                "membershipChanged": self.membership_changed,
                "exposure": self.exposure, "isTotal": self.is_total,
                "dataset": self.dataset}


@dataclass(frozen=True)
class PeriodRef:
    """The reporting date(s) a finding covers.

    A point-in-time finding carries ``end`` only. A movement carries both, and
    ``method`` names the governed resolution that chose them
    (``mi_agent.period_change.models.ALL_RESOLUTION_METHODS``) so a reader can
    see WHICH two snapshots were compared and why.
    """

    end: Optional[str] = None
    start: Optional[str] = None
    method: Optional[str] = None
    #: Every governed snapshot available to the resolution, for disclosure.
    available: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"start": self.start, "end": self.end, "method": self.method,
                "available": list(self.available)}


# --------------------------------------------------------------------------- #
# Findings — the ONLY currency between deterministic execution and narrative
# --------------------------------------------------------------------------- #
#: Finding kinds. Closed vocabulary: a narrative renders a kind it knows, and
#: refuses to describe one it does not, rather than improvising.
KIND_MEASURE = "measure"              # one governed aggregate at one date
KIND_MOVEMENT = "movement"            # one governed aggregate across two dates
KIND_COMPOSITION = "composition"      # one category's share of a population
KIND_COMPOSITION_SHIFT = "composition_shift"   # that share, across two dates
KIND_COMPARISON = "comparison"        # two populations' values for one measure
KIND_FORECAST = "forecast"            # a projected value, with its basis
KIND_TIMING = "timing"                # an expected amount landing in a period
KIND_THRESHOLD = "threshold"          # the date a projection crosses a level
KIND_LIMIT = "limit"                  # actual vs limit, with headroom
KIND_COHORT = "cohort"                # one origination vintage's metrics

ALL_KINDS: Tuple[str, ...] = (
    KIND_MEASURE, KIND_MOVEMENT, KIND_COMPOSITION, KIND_COMPOSITION_SHIFT,
    KIND_COMPARISON, KIND_FORECAST, KIND_TIMING, KIND_THRESHOLD, KIND_LIMIT,
    KIND_COHORT,
)

#: Finding statuses.
STATUS_OK = "ok"
#: Computed, but something about it must be said before it is read.
STATUS_QUALIFIED = "qualified"
#: Not computed. ``note`` says why, and ``value`` is None — never zero.
STATUS_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Finding:
    """One deterministic result, in the shape a narrative can be built from.

    Every field is optional except the ones that identify it, because a finding
    is a *record of what was computed*, not a form to be filled. What matters is
    the discipline the dataclass enforces: a number always arrives attached to
    the measure it is, the population it covers, the period it covers and the
    evidence that produced it. A narrative that wants to state a figure has to
    read one of these; it can never reach past them to the data.
    """

    capability: str
    kind: str
    label: str
    #: Canonical/semantic field key, where the finding is about a governed
    #: measure. ``None`` for findings about a population rather than a measure
    #: (a case count, an expected completion month).
    metric: Optional[str] = None
    population: Optional[PopulationRef] = None
    period: Optional[PeriodRef] = None
    unit: Optional[str] = None
    aggregation: Optional[str] = None

    value: Optional[float] = None
    prior_value: Optional[float] = None
    change: Optional[float] = None
    relative_change: Optional[float] = None

    #: Comparison findings: the other side.
    comparand: Optional[PopulationRef] = None
    comparand_value: Optional[float] = None

    #: Ranking within a set the capability declares (``rank`` of ``rank_of``).
    rank: Optional[int] = None
    rank_of: Optional[int] = None
    #: Share of the population's basis, where the finding is compositional.
    share: Optional[float] = None
    prior_share: Optional[float] = None

    #: Limit findings.
    limit: Optional[float] = None
    headroom: Optional[float] = None
    limit_status: Optional[str] = None

    #: Forecast / timing / threshold findings.
    forecast_value: Optional[float] = None
    forecast_date: Optional[str] = None
    probability_basis: Optional[str] = None

    status: str = STATUS_OK
    #: Plain-English qualification. Always present when status is not ``ok``.
    note: Optional[str] = None
    #: Receipt-grade evidence: executor id, calculation version, denominators,
    #: valid/excluded populations, source dates. Never prose.
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ALL_KINDS:
            raise ValueError(f"unknown finding kind: {self.kind!r}")
        if self.status != STATUS_OK and not self.note:
            raise ValueError(
                f"a {self.status!r} finding must say why ({self.label!r})")

    @property
    def ok(self) -> bool:
        return self.status in (STATUS_OK, STATUS_QUALIFIED)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "capability": self.capability, "kind": self.kind,
            "label": self.label, "metric": self.metric,
            "unit": self.unit, "aggregation": self.aggregation,
            "value": self.value, "priorValue": self.prior_value,
            "change": self.change, "relativeChange": self.relative_change,
            "status": self.status, "note": self.note,
            "evidence": dict(self.evidence or {}),
        }
        if self.population is not None:
            out["population"] = self.population.to_dict()
        if self.period is not None:
            out["period"] = self.period.to_dict()
        if self.comparand is not None:
            out["comparand"] = self.comparand.to_dict()
            out["comparandValue"] = self.comparand_value
        for key, value in (("rank", self.rank), ("rankOf", self.rank_of),
                           ("share", self.share), ("priorShare", self.prior_share),
                           ("limit", self.limit), ("headroom", self.headroom),
                           ("limitStatus", self.limit_status),
                           ("forecastValue", self.forecast_value),
                           ("forecastDate", self.forecast_date),
                           ("probabilityBasis", self.probability_basis)):
            if value is not None:
                out[key] = value
        return out


# --------------------------------------------------------------------------- #
# Capability declarations
# --------------------------------------------------------------------------- #
#: Datasets a capability reads. A capability declaring ``pipeline`` cannot run
#: on a deployment with no governed pipeline source, and says so rather than
#: returning zero.
DATASET_FUNDED = "funded"
DATASET_PIPELINE = "pipeline"
DATASET_FUNDED_HISTORY = "funded_history"
DATASET_LIMITS = "limits"


@dataclass(frozen=True)
class AnalyticalCapability:
    """One deterministic analytical operation, declared.

    ``executor`` is the callable this layer invokes. It is a *thin adapter* over
    an engine that already exists (see ``mi_workflows.analytical.executors``);
    the adapter's only jobs are to marshal declared inputs into the engine's own
    signature and to shape its output into :class:`Finding` objects. No
    capability adapter may contain a financial calculation.
    """

    id: str
    #: What analytical question shape this capability serves, in one line.
    intent: str
    #: Input names the planner MUST supply. Validated before execution.
    required_inputs: Tuple[str, ...] = ()
    #: Input names the planner MAY supply.
    optional_inputs: Tuple[str, ...] = ()
    #: Population/scope kinds this capability can be narrowed to.
    supported_scopes: Tuple[str, ...] = ()
    #: Governed datasets it reads.
    datasets: Tuple[str, ...] = (DATASET_FUNDED,)
    #: The deterministic engine it delegates to, named for the record.
    engine: str = ""
    #: The finding kinds it can emit.
    produces: Tuple[str, ...] = ()
    #: What it cannot do. Read by the planner and printed in the catalogue, so a
    #: limitation is a declared property rather than folklore.
    limitations: Tuple[str, ...] = ()
    #: The existing recogniser that already owns questions of this shape, when
    #: one does. The planner NEVER claims a question a listed route answers —
    #: this is the deference rule that makes the layer additive.
    route_owner: Optional[str] = None
    #: Bound at registration.
    executor: Optional[Callable[..., Sequence[Finding]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "intent": self.intent,
            "requiredInputs": list(self.required_inputs),
            "optionalInputs": list(self.optional_inputs),
            "supportedScopes": list(self.supported_scopes),
            "datasets": list(self.datasets),
            "engine": self.engine,
            "produces": list(self.produces),
            "limitations": list(self.limitations),
            "routeOwner": self.route_owner,
        }

    def validate_inputs(self, inputs: Mapping[str, Any]) -> List[str]:
        """Errors in a proposed input set, or ``[]``.

        Runs BEFORE any data is touched, so an unusable plan — including one an
        LLM proposed — is rejected rather than half-executed.
        """
        supplied = set(inputs or {})
        allowed = set(self.required_inputs) | set(self.optional_inputs)
        errors = [f"{self.id}: missing required input {name!r}"
                  for name in self.required_inputs if name not in supplied]
        errors += [f"{self.id}: unknown input {name!r}"
                   for name in sorted(supplied - allowed)]
        return errors


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CapabilityCall:
    """One deterministic capability invocation within a plan."""

    capability: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    #: Why this call is in the plan. Shown in the answer's execution evidence,
    #: so a reader can see the analytical reasoning without reading code.
    because: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"capability": self.capability, "inputs": dict(self.inputs),
                "because": self.because}


@dataclass(frozen=True)
class AnalyticalPlan:
    """An ordered set of deterministic capability calls, and why.

    A plan is data. It can be produced deterministically (the default) or
    proposed by a language model, and either way it is validated against the
    registry before anything executes — the same discipline the MI Query parser
    already applies to an LLM-proposed ``MIQuerySpec``.
    """

    intent: str
    calls: Tuple[CapabilityCall, ...]
    #: ``deterministic`` | ``llm`` | ``llm_rejected_deterministic``
    origin: str = "deterministic"
    rationale: str = ""
    composition_version: str = COMPOSITION_VERSION
    #: Finding kinds without which this plan has not answered the question it
    #: was built for. A plan that runs end to end but produces none of them has
    #: produced a REFUSAL, and must say so in the words of the capability that
    #: could not run — never present the half it did compute as the answer.
    required_kinds: Tuple[str, ...] = ()

    @property
    def is_composite(self) -> bool:
        """Does answering this question need more than one capability?

        The layer engages only for composite work; a single-capability question
        stays on the route that already answers it.
        """
        return len(self.calls) > 1

    def to_dict(self) -> Dict[str, Any]:
        return {"intent": self.intent, "origin": self.origin,
                "rationale": self.rationale,
                "compositionVersion": self.composition_version,
                "requiredKinds": list(self.required_kinds),
                "calls": [c.to_dict() for c in self.calls]}
