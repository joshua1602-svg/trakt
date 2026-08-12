"""trakt_core.capability — what Trakt can tell you about a portfolio, and why not.

Trakt should know what it knows, and it should know why it cannot answer
something. Those are two different pieces of knowledge and only the first was
ever written down. This module holds the second.

The distinction that is load-bearing
------------------------------------
An autonomous consumer asking for a weighted average life needs to tell four
answers apart, and ``null`` collapses all of them:

    "this portfolio has no WAL"          NOT_APPLICABLE
    "Trakt lacks a field"                UNAVAILABLE
    "WAL needs an unknown future rate"   ASSUMPTION_REQUIRED
    "WAL needs behavioural modelling"    MODEL_REQUIRED

A fifth is needed and is deliberately *not* folded into UNAVAILABLE:

    "we have the data and have not settled the definition"
                                         METHODOLOGY_NOT_APPROVED

"We lack a field" and "we have not decided what this means" call for entirely
different responses — one is a data request to the client, the other is a
decision Trakt owes itself — and Sprint 2.5C found that conflating them is how
a plausible formula gets published as though it were owned.

Asset-agnostic by construction
------------------------------
There is no asset class in this file. A capability declares the economic
conditions it needs and any portfolio meeting them gets it. Where a restriction
is real it is economic: a loan-to-*value* needs a value, and a *contractual*
life needs a contractual repayment date. Neither is "a mortgage metric".
Onboarding a new asset class requires no change here — if its canonical tape
carries the concepts, the capabilities light up.

Cheap by construction
---------------------
Discovery must never compute the metrics it describes. Every condition is a
column-presence check, a snapshot count, or a single ``value_counts`` over an
enum column. *Can calculate* is not *calculate now*, and an agent surveying a
portfolio should not pay for a cash-flow enumeration to be told one exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import config_cache

DEFAULT_REGISTRY = "config/system/mi_capability_registry.yaml"
REGISTRY_ENV = "TRAKT_MI_CAPABILITY_REGISTRY"

# -- availability states ---------------------------------------------------- #
AVAILABLE = "AVAILABLE"
UNAVAILABLE = "UNAVAILABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"
ASSUMPTION_REQUIRED = "ASSUMPTION_REQUIRED"
MODEL_REQUIRED = "MODEL_REQUIRED"
METHODOLOGY_NOT_APPROVED = "METHODOLOGY_NOT_APPROVED"

STATES = (AVAILABLE, UNAVAILABLE, NOT_APPLICABLE, ASSUMPTION_REQUIRED,
          MODEL_REQUIRED, METHODOLOGY_NOT_APPROVED)

# -- reason codes ----------------------------------------------------------- #
REASON_MISSING_INPUT = "MISSING_REQUIRED_INPUT"
REASON_INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
REASON_NO_COLLATERAL = "NO_COLLATERAL"
REASON_NO_GEOGRAPHY = "NO_GEOGRAPHY"
REASON_CONTINGENT_REPAYMENT = "CONTINGENT_REPAYMENT"
REASON_FUTURE_RATE_PATH = "FUTURE_RATE_PATH_REQUIRED"
REASON_BEHAVIOURAL_MODEL = "BEHAVIOURAL_MODEL_REQUIRED"
REASON_METHODOLOGY = "METHODOLOGY_NOT_APPROVED"

#: Canonical fields that constitute collateral evidence. Any one of them makes
#: a loan-to-value meaningful; none of them makes it unsecured.
COLLATERAL_FIELDS = (
    "current_valuation_amount", "original_valuation_amount", "indexed_value",
    "collateral_value", "market_value", "current_loan_to_value",
    "original_loan_to_value", "indexed_loan_to_value",
)

#: RREL35 codes whose principal series the contract fixes directly, so it does
#: not depend on the interest rate.
RATE_INDEPENDENT_PRINCIPAL = frozenset({"BLLT", "FIXE"})
#: RREL35 codes where the contract fixes the TOTAL instalment, so principal is
#: the residual after interest and inherits the rate's uncertainty.
RATE_DEPENDENT_PRINCIPAL = frozenset({"FRXX", "DEXX"})
#: RREL42 codes whose rate is contractually fixed for the life of the exposure.
RATE_FIXED_FOR_LIFE = frozenset({"FXRL"})


@dataclass(frozen=True)
class Condition:
    """One cheap, named predicate a capability depends on."""

    type: str
    any_of: Tuple[str, ...] = ()
    minimum: int = 0
    reason_code: str = ""
    status_when_absent: str = ""
    explanation: str = ""


@dataclass(frozen=True)
class Capability:
    """One analytical capability Trakt publishes."""

    id: str
    name: str
    category: str
    description: str = ""
    unit: str = ""
    #: ``observed`` | ``contractual`` | ``modelled``. The vocabulary boundary
    #: that keeps a measured CPR from being read as a projected one.
    basis: str = "observed"
    time_basis: str = ""
    aggregation: str = ""
    weighting: str = ""
    methodology: Optional[str] = None
    calculation_source: Optional[str] = None
    metric_id: Optional[str] = None
    consumers: Tuple[str, ...] = ()
    #: Phrases a human might use for this capability. Used only to recognise
    #: a question Trakt cannot answer, so that it can say WHY rather than
    #: "that measure does not exist" — never to resolve a value.
    synonyms: Tuple[str, ...] = ()
    conditions: Tuple[Condition, ...] = ()
    #: Set where the answer never depends on the portfolio — a metric whose
    #: methodology Trakt has not settled, or one that needs a model Trakt does
    #: not build. Short-circuits resolution entirely.
    status_override: Optional[str] = None
    reason_code: str = ""
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.id, "name": self.name, "category": self.category,
            "description": self.description, "unit": self.unit,
            "basis": self.basis, "time_basis": self.time_basis or None,
            "aggregation": self.aggregation or None,
            "weighting": self.weighting or None,
            "methodology": self.methodology,
            "calculation_source": self.calculation_source,
            "metric_id": self.metric_id,
            "consumers": list(self.consumers),
        }


@dataclass(frozen=True)
class Availability:
    """Whether one capability can be produced for one portfolio, and why not."""

    metric: str
    status: str
    reason_code: str = ""
    explanation: str = ""
    missing_inputs: Tuple[str, ...] = ()
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == AVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"metric": self.metric, "status": self.status}
        if self.status != AVAILABLE:
            payload["reason_code"] = self.reason_code or None
            payload["explanation"] = self.explanation or None
            if self.missing_inputs:
                payload["missing_inputs"] = list(self.missing_inputs)
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass(frozen=True)
class CapabilityRegistry:
    version: int
    categories: Dict[str, str]
    capabilities: Tuple[Capability, ...]

    def get(self, metric_id: str) -> Optional[Capability]:
        return next((c for c in self.capabilities if c.id == metric_id), None)

    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(c.id for c in self.capabilities)


def _registry_path() -> str:
    return os.environ.get(REGISTRY_ENV, DEFAULT_REGISTRY)


def load_registry(path: Optional[str] = None) -> CapabilityRegistry:
    """The published capability catalogue. Content-identity cached, like every
    other governed configuration — the catalogue is a property of the
    deployment, not of a request."""
    from pathlib import Path

    resolved = Path(path or _registry_path())

    def _load() -> "CapabilityRegistry":
        import yaml

        with resolved.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return _build(raw)

    return config_cache.get_or_load("mi_capability_registry", resolved, _load)


def _build(raw: Mapping[str, Any]) -> CapabilityRegistry:
    capabilities: List[Capability] = []
    for entry in (raw.get("capabilities") or []):
        conditions = tuple(
            Condition(
                type=str(c.get("type") or ""),
                any_of=tuple(str(f) for f in (c.get("any_of") or ())),
                minimum=int(c.get("minimum") or 0),
                reason_code=str(c.get("reason_code") or ""),
                status_when_absent=str(c.get("status_when_absent") or ""),
                explanation=str(c.get("explanation") or "").strip(),
            )
            for c in (entry.get("conditions") or []))
        capabilities.append(Capability(
            id=str(entry.get("id") or ""),
            name=str(entry.get("name") or entry.get("id") or ""),
            category=str(entry.get("category") or "uncategorised"),
            description=str(entry.get("description") or "").strip(),
            unit=str(entry.get("unit") or ""),
            basis=str(entry.get("basis") or "observed"),
            time_basis=str(entry.get("time_basis") or ""),
            aggregation=str(entry.get("aggregation") or ""),
            weighting=str(entry.get("weighting") or ""),
            methodology=entry.get("methodology") or None,
            calculation_source=entry.get("calculation_source") or None,
            metric_id=entry.get("metric_id") or None,
            consumers=tuple(str(c) for c in (entry.get("consumers") or ())),
            synonyms=tuple(str(x).strip().lower()
                           for x in (entry.get("synonyms") or ())),
            conditions=conditions,
            status_override=entry.get("status_override") or None,
            reason_code=str(entry.get("reason_code") or ""),
            explanation=str(entry.get("explanation") or "").strip(),
        ))

    return CapabilityRegistry(
        version=int(raw.get("registry_version") or 1),
        categories=dict(raw.get("categories") or {}),
        capabilities=tuple(c for c in capabilities if c.id))


# --------------------------------------------------------------------------- #
# The portfolio's side of the question
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PortfolioShape:
    """What a portfolio carries, at the cost of reading its columns.

    Deliberately small. Building this must not require the analytics it will
    be used to describe — the amortisation and rate mixes are a
    ``value_counts`` over two enum columns, which is the most expensive thing
    here and is still a single pass over two columns rather than a schedule
    enumeration over every loan.
    """

    columns: frozenset
    history_periods: int = 1
    row_count: int = 0
    #: RREL35 code -> balance share (0-1). Empty when the column is absent.
    amortisation_mix: Mapping[str, float] = field(default_factory=dict)
    #: RREL42 code -> balance share (0-1).
    rate_type_mix: Mapping[str, float] = field(default_factory=dict)

    def has(self, names: Iterable[str]) -> bool:
        return any(n in self.columns for n in names)


def describe_portfolio(frame: Any, *, history_periods: int = 1
                       ) -> PortfolioShape:
    """Read a governed frame's shape without measuring anything in it."""
    import pandas as pd

    columns = frozenset(str(c) for c in getattr(frame, "columns", ()))
    balance_column = next(
        (c for c in ("current_principal_balance", "current_outstanding_balance")
         if c in columns), None)

    def mix(column: str, normaliser) -> Dict[str, float]:
        """Balance share by normalised enum code.

        Grouped on the RAW column and normalised afterwards, over the distinct
        values only. Mapping the normaliser across every row instead calls a
        Python function per loan — 2.4 seconds at 100k rows, for a column with
        five distinct values in it. Capability discovery has to stay cheap or
        agents stop surveying and start guessing.
        """
        if column not in columns or not len(frame):
            return {}
        if balance_column is not None:
            weights = pd.to_numeric(frame[balance_column],
                                    errors="coerce").fillna(0.0)
        else:
            weights = pd.Series(1.0, index=frame.index)
        total = float(weights.sum())
        if total <= 0:
            return {}
        by_raw = weights.groupby(frame[column], dropna=False).sum()
        out: Dict[str, float] = {}
        for raw_value, weight in by_raw.items():
            code = normaliser(raw_value)
            if code is None:
                continue
            out[str(code)] = out.get(str(code), 0.0) + float(weight) / total
        return out

    from analytics_lib.contractual import (
        normalise_amortisation,
        normalise_rate_type,
    )

    return PortfolioShape(
        columns=columns,
        history_periods=max(1, int(history_periods)),
        row_count=int(len(frame)) if hasattr(frame, "__len__") else 0,
        amortisation_mix=mix("amortisation_type", normalise_amortisation),
        rate_type_mix=mix("interest_rate_type", normalise_rate_type),
    )


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def _principal_determinism(shape: PortfolioShape) -> Availability:
    """Can a contractual principal series be enumerated for this book?

    The 2.5D result, applied at portfolio level. BLLT and FIXE fix principal
    directly, so they survive a floating rate. FRXX and DEXX fix the *total*
    instalment, so their principal moves with the rate and needs it fixed.
    Everything else — including the ``OTHR`` that a lifetime mortgage reports
    — has no contractual principal profile at all.
    """
    mix = shape.amortisation_mix
    if not mix:
        return Availability(
            metric="", status=UNAVAILABLE, reason_code=REASON_MISSING_INPUT,
            missing_inputs=("amortisation_type",),
            explanation=("No amortisation type (RREL35) is carried, so what "
                         "the contract does with principal is unknown."))

    rate_independent = sum(v for k, v in mix.items()
                           if k in RATE_INDEPENDENT_PRINCIPAL)
    rate_dependent = sum(v for k, v in mix.items()
                         if k in RATE_DEPENDENT_PRINCIPAL)
    fixed_rate = sum(v for k, v in mix.items()
                     if k in RATE_DEPENDENT_PRINCIPAL) and sum(
                         v for k, v in shape.rate_type_mix.items()
                         if k in RATE_FIXED_FOR_LIFE)

    deterministic = rate_independent
    if rate_dependent:
        # The rate-dependent share only counts where the rate is fixed. Mixes
        # are measured at portfolio level, so this is the honest bound rather
        # than a per-loan answer — the per-loan answer is the schedule's job.
        deterministic += min(rate_dependent,
                             sum(v for k, v in shape.rate_type_mix.items()
                                 if k in RATE_FIXED_FOR_LIFE))

    detail = {"amortisation_mix": {k: round(v, 6) for k, v in mix.items()},
              "deterministic_principal_share": round(deterministic, 6)}

    if deterministic > 0:
        return Availability(metric="", status=AVAILABLE, detail=detail)

    if rate_dependent:
        return Availability(
            metric="", status=ASSUMPTION_REQUIRED,
            reason_code=REASON_FUTURE_RATE_PATH, detail=detail,
            explanation=("Every exposure amortises on a fixed TOTAL instalment "
                         "(RREL35 FRXX/DEXX) at a rate that is not "
                         "contractually fixed, so the principal share of each "
                         "payment depends on an unknown future reference rate. "
                         "No rate path is assumed."))

    return Availability(
        metric="", status=NOT_APPLICABLE,
        reason_code=REASON_CONTINGENT_REPAYMENT, detail=detail,
        explanation=("No exposure carries a contractual principal profile — "
                     "the book reports RREL35 = OTHR. For a lifetime mortgage "
                     "that is correct rather than missing: repayment is "
                     "contingent on death, sale or long-term care, so no "
                     "contractual repayment date exists. A legal long-stop "
                     "maturity is not a contractual life and is not used as "
                     "one."))


def _interest_determinism(shape: PortfolioShape) -> Availability:
    """Is the contractual interest series determined? Needed by yield, and by
    nothing else — which is why a floating bullet keeps its WAL."""
    mix = shape.rate_type_mix
    if not mix:
        return Availability(
            metric="", status=UNAVAILABLE, reason_code=REASON_MISSING_INPUT,
            missing_inputs=("interest_rate_type",),
            explanation=("No interest rate type (RREL42) is carried, so "
                         "whether the rate is contractually fixed is unknown. "
                         "It is not assumed to be fixed."))

    fixed = sum(v for k, v in mix.items() if k in RATE_FIXED_FOR_LIFE)
    detail = {"rate_type_mix": {k: round(v, 6) for k, v in mix.items()},
              "fixed_for_life_share": round(fixed, 6)}
    if fixed > 0:
        return Availability(metric="", status=AVAILABLE, detail=detail)
    return Availability(
        metric="", status=ASSUMPTION_REQUIRED,
        reason_code=REASON_FUTURE_RATE_PATH, detail=detail,
        explanation=("No exposure has a rate contractually fixed for life "
                     "(RREL42 = FXRL), so future interest cash flows depend on "
                     "a reference-rate path Trakt does not hold. RREL48/RREL49 "
                     "bound the rate; they do not determine it."))


def _evaluate_condition(condition: Condition, shape: PortfolioShape
                        ) -> Optional[Availability]:
    """None means satisfied. Otherwise the reason it is not."""
    if condition.type == "fields_present":
        if shape.has(condition.any_of):
            return None
        return Availability(
            metric="",
            status=condition.status_when_absent or UNAVAILABLE,
            reason_code=condition.reason_code or REASON_MISSING_INPUT,
            missing_inputs=condition.any_of,
            explanation=condition.explanation or (
                "The portfolio carries none of the required inputs: "
                + ", ".join(condition.any_of) + "."))

    if condition.type == "history_periods":
        if shape.history_periods >= condition.minimum:
            return None
        return Availability(
            metric="", status=UNAVAILABLE,
            reason_code=condition.reason_code or REASON_INSUFFICIENT_HISTORY,
            explanation=condition.explanation or (
                f"This metric compares periods and needs at least "
                f"{condition.minimum} governed snapshots; "
                f"{shape.history_periods} "
                f"{'is' if shape.history_periods == 1 else 'are'} available."),
            detail={"required_periods": condition.minimum,
                    "available_periods": shape.history_periods})

    if condition.type == "collateral_present":
        if shape.has(COLLATERAL_FIELDS):
            return None
        return Availability(
            metric="",
            status=condition.status_when_absent or NOT_APPLICABLE,
            reason_code=condition.reason_code or REASON_NO_COLLATERAL,
            explanation=condition.explanation or (
                "The portfolio carries no collateral valuation."))

    if condition.type == "principal_series_deterministic":
        outcome = _principal_determinism(shape)
        return None if outcome.status == AVAILABLE else outcome

    if condition.type == "interest_series_deterministic":
        outcome = _interest_determinism(shape)
        return None if outcome.status == AVAILABLE else outcome

    # An unknown condition must not silently pass: a typo in configuration
    # would otherwise publish a capability nothing checks.
    return Availability(
        metric="", status=UNAVAILABLE, reason_code="UNKNOWN_CONDITION",
        explanation=(f"The registry declares condition type "
                     f"{condition.type!r}, which this resolver does not "
                     "implement."))


def resolve_capability(capability: Capability, shape: PortfolioShape
                       ) -> Availability:
    """One capability against one portfolio. No calculation is performed."""
    if capability.status_override:
        return Availability(
            metric=capability.id, status=capability.status_override,
            reason_code=capability.reason_code or REASON_METHODOLOGY,
            explanation=capability.explanation)

    detail: Dict[str, Any] = {}
    for condition in capability.conditions:
        outcome = _evaluate_condition(condition, shape)
        if outcome is not None:
            merged = dict(outcome.detail)
            merged.update(detail)
            return Availability(
                metric=capability.id, status=outcome.status,
                reason_code=outcome.reason_code,
                explanation=outcome.explanation,
                missing_inputs=outcome.missing_inputs, detail=merged)
        # Determinism checks carry useful shape detail even when satisfied.
        if condition.type == "principal_series_deterministic":
            detail.update(_principal_determinism(shape).detail)
        elif condition.type == "interest_series_deterministic":
            detail.update(_interest_determinism(shape).detail)

    return Availability(metric=capability.id, status=AVAILABLE, detail=detail)


def resolve_all(shape: PortfolioShape, *,
                registry: Optional[CapabilityRegistry] = None,
                categories: Optional[Sequence[str]] = None,
                ) -> List[Availability]:
    """Every published capability against one portfolio."""
    catalogue = registry or load_registry()
    wanted = set(categories or ())
    return [resolve_capability(c, shape) for c in catalogue.capabilities
            if not wanted or c.category in wanted]


def summarise(results: Sequence[Availability]) -> Dict[str, int]:
    """Counts by state. Totalled nowhere else, deliberately: NOT_APPLICABLE and
    UNAVAILABLE must not be added together into a single 'missing' figure."""
    counts: Dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def match_capability(text: str, *, registry: Optional[CapabilityRegistry] = None
                     ) -> Optional[Capability]:
    """The capability a question is asking about, if any.

    Deliberately narrow. This exists so that "what is the contractual WAL of
    this ERM portfolio?" can be answered with *why there is no contractual
    WAL*, instead of the generic "that measure does not exist here" — which is
    both wrong and unhelpful, because the measure exists and the portfolio is
    the reason it does not apply.

    It never resolves a value and never substitutes a different measure. The
    longest matching phrase wins, so "expected weighted average life" is not
    swallowed by "weighted average life".
    """
    if not text:
        return None
    import re as _re

    # Punctuation is stripped rather than split on: "what is the YTM?" must
    # match, and "ytm?" is not the token "ytm".
    normalised = _re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    haystack = " " + " ".join(normalised.split()) + " "
    catalogue = registry or load_registry()

    best: Optional[Capability] = None
    best_length = 0
    for capability in catalogue.capabilities:
        for phrase in (capability.id.replace("_", " "),) + capability.synonyms:
            token = " " + phrase.strip().lower() + " "
            if token in haystack and len(token) > best_length:
                best, best_length = capability, len(token)
    return best
