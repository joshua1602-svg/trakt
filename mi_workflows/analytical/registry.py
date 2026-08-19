"""mi_workflows.analytical.registry — the analytical capability catalogue.

Ten declarations. Every one names a deterministic engine that existed before
this layer did; none of them introduces a calculation. The catalogue is the
answer to two questions a planner and a report both need:

* *"what analytical operations can be composed?"* — the ids below;
* *"which of them does an existing route already own?"* — ``route_owner``.

``route_owner`` is the deference rule that makes this layer additive. A
capability that names one is still fully implemented and callable — it is used
by a composite plan and covered by tests — but the planner never claims a
question whose whole answer is that one capability, because a route already
answers it correctly and replacing a working answer is not an improvement.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from .contract import (
    AnalyticalCapability,
    DATASET_FUNDED,
    DATASET_FUNDED_HISTORY,
    DATASET_LIMITS,
    DATASET_PIPELINE,
    KIND_COHORT,
    KIND_COMPOSITION,
    KIND_COMPOSITION_SHIFT,
    KIND_FORECAST,
    KIND_LIMIT,
    KIND_MEASURE,
    KIND_MOVEMENT,
    KIND_THRESHOLD,
    KIND_TIMING,
)
from . import executors


class DuplicateCapabilityError(ValueError):
    """A capability id was declared twice. Never silently overwritten."""


class UnknownCapabilityError(KeyError):
    """A plan named a capability that is not declared."""


_DECLARATIONS: Tuple[AnalyticalCapability, ...] = (
    AnalyticalCapability(
        id="portfolio_snapshot",
        intent="Governed measures for one population at the current reporting date.",
        required_inputs=("measures",),
        optional_inputs=("population",),
        supported_scopes=("total", "seasoning", "provenance", "dimension_value"),
        datasets=(DATASET_FUNDED,),
        engine="mi_workflows.engine.aggregate (Business Semantics Registry)",
        produces=(KIND_MEASURE,),
        limitations=(
            "Point-in-time only — it states a level, never a movement.",
            "A measure the Business Semantics Registry does not govern is "
            "reported unavailable rather than aggregated with a default rule.",
        ),
        executor=executors.portfolio_snapshot),

    AnalyticalCapability(
        id="population_profile",
        intent="How one population is composed across governed dimensions.",
        optional_inputs=("population", "dimensions", "basis", "top_n"),
        supported_scopes=("total", "seasoning", "provenance", "dimension_value"),
        datasets=(DATASET_FUNDED,),
        engine="mi_workflows.engine.ranked_distribution",
        produces=(KIND_COMPOSITION,),
        limitations=(
            "A dimension the book does not carry is skipped and disclosed, "
            "never substituted with a neighbouring one.",
            "Unknown/missing values are their own disclosed bucket and stay in "
            "the denominator, so shares cannot flatter a sparse dimension.",
        ),
        executor=executors.population_profile),

    AnalyticalCapability(
        id="period_movement",
        intent="Governed movement of measures and composition for one population "
               "between two governed snapshots.",
        optional_inputs=("population", "measures", "dimensions", "window"),
        supported_scopes=("total", "seasoning", "provenance", "dimension_value"),
        datasets=(DATASET_FUNDED_HISTORY,),
        engine="mi_agent.period_change.run_period_change_analysis",
        produces=(KIND_MOVEMENT, KIND_COMPOSITION_SHIFT),
        limitations=(
            "Requires at least two retained governed snapshots.",
            "Movements are ranked within a unit of measurement only; no "
            "materiality verdict is produced without a configured threshold.",
        ),
        executor=executors.period_movement),

    AnalyticalCapability(
        id="vintage_analysis",
        intent="Static-pool metrics per origination vintage.",
        optional_inputs=("population", "dimension"),
        supported_scopes=("total", "seasoning", "provenance"),
        datasets=(DATASET_FUNDED,),
        engine="mi_agent_api.cohorts.cohort_analysis",
        produces=(KIND_COHORT,),
        limitations=(
            "Point-in-time static-pool aggregates only — redemption, "
            "completion and performance curves are not computed in the MI path.",
            "Only the metrics the tape supports are emitted; the set is "
            "declared per answer.",
        ),
        executor=executors.vintage_analysis),

    AnalyticalCapability(
        id="pipeline_stock",
        intent="Case count and gross amount of open pipeline cases, by stage.",
        optional_inputs=("stages",),
        supported_scopes=("total",),
        datasets=(DATASET_PIPELINE,),
        engine="mi_agent_api.pipeline_contract.compute_pipeline_snapshot",
        produces=(KIND_MEASURE,),
        limitations=(
            "Requires a governed weekly pipeline extract; absent one the "
            "capability reports unavailable rather than zero.",
            "Pipeline exposure carries no portfolio provenance, so it cannot be "
            "narrowed to a portfolio lens.",
        ),
        executor=executors.pipeline_stock),

    AnalyticalCapability(
        id="pipeline_completion_forecast",
        intent="Expected completion amount and timing for open pipeline cases.",
        optional_inputs=("stages", "horizon_months"),
        supported_scopes=("total",),
        datasets=(DATASET_PIPELINE,),
        engine="mi_agent_api.pipeline_contract (governed completion "
               "probabilities from the pipeline preparation layer)",
        produces=(KIND_FORECAST, KIND_TIMING),
        limitations=(
            "Completion probabilities are the governed stage rates — empirical "
            "over the retained weekly window above the sufficiency floor, the "
            "configured stage assumption below it. This layer never assigns one.",
            "Expected completion timing exists only where the extract carries "
            "completion-date evidence.",
        ),
        executor=executors.pipeline_completion_forecast),

    AnalyticalCapability(
        id="funded_balance_forecast",
        intent="Current funded balance plus probability-weighted expected "
               "completions.",
        datasets=(DATASET_FUNDED, DATASET_PIPELINE),
        supported_scopes=("total",),
        engine="mi_agent_api.forecast_bridge.compute_forecast_bridge",
        produces=(KIND_MEASURE, KIND_FORECAST),
        limitations=(
            "An aggregate composition — pipeline rows are never merged into "
            "the funded book.",
            "Without a governed pipeline the forecast is reported unavailable "
            "and the funded balance is explicitly not labelled a forecast.",
        ),
        executor=executors.funded_balance_forecast),

    AnalyticalCapability(
        id="completion_run_rate",
        intent="The governed completion run-rate from month-on-month funded growth.",
        datasets=(DATASET_FUNDED_HISTORY,),
        supported_scopes=("total",),
        engine="mi_agent_api.forecast_extrapolation.run_rate_model",
        produces=(KIND_FORECAST,),
        limitations=(
            "Downside/base/upside are indicative scenario bands, not "
            "statistically validated confidence intervals.",
            "Needs enough retained funded history to observe growth.",
        ),
        route_owner="forecast_extrapolation",
        executor=executors.completion_run_rate),

    AnalyticalCapability(
        id="threshold_projection",
        intent="The date a projected funded balance crosses a threshold.",
        required_inputs=("threshold",),
        datasets=(DATASET_FUNDED_HISTORY,),
        supported_scopes=("total",),
        engine="mi_agent_api.forecast_extrapolation (milestone solver)",
        produces=(KIND_THRESHOLD,),
        limitations=(
            "Bounded by the projection horizon; beyond it the answer is "
            "'not within the horizon', never an extrapolated date.",
        ),
        route_owner="forecast_extrapolation",
        executor=executors.threshold_projection),

    AnalyticalCapability(
        id="concentration_limits",
        intent="Governed limits, actuals and headroom, ranked by proximity.",
        optional_inputs=("category", "top_n"),
        datasets=(DATASET_FUNDED, DATASET_LIMITS),
        supported_scopes=("total",),
        engine="mi_agent_api.concentration_tests_api.compute_concentration_tests",
        produces=(KIND_LIMIT,),
        limitations=(
            "Forward-looking (expected / full-pipeline) limit states require an "
            "APPROVED concentration configuration and a governed pipeline; "
            "without both, only the funded position is measurable.",
            "A test whose input field the book lacks is reported unavailable, "
            "never as a passing test.",
        ),
        route_owner="risk_limits",
        executor=executors.concentration_limits),
)


class CapabilityRegistry:
    """The declared analytical capabilities, in a stable order."""

    def __init__(self, declarations: Iterable[AnalyticalCapability] = ()) -> None:
        self._items: Dict[str, AnalyticalCapability] = {}
        self._order: List[str] = []
        for declaration in declarations:
            self.register(declaration)

    def register(self, capability: AnalyticalCapability) -> AnalyticalCapability:
        if not capability.id:
            raise ValueError("an analytical capability must have an id")
        if capability.id in self._items:
            raise DuplicateCapabilityError(
                f"analytical capability {capability.id!r} is already declared")
        if capability.executor is None:
            raise ValueError(
                f"analytical capability {capability.id!r} declares no executor; "
                "a described capability that cannot run is documentation, not a "
                "capability")
        self._items[capability.id] = capability
        self._order.append(capability.id)
        return capability

    def get(self, capability_id: str) -> Optional[AnalyticalCapability]:
        return self._items.get(capability_id)

    def require(self, capability_id: str) -> AnalyticalCapability:
        capability = self._items.get(capability_id)
        if capability is None:
            raise UnknownCapabilityError(capability_id)
        return capability

    def ids(self) -> Tuple[str, ...]:
        return tuple(self._order)

    def all(self) -> Tuple[AnalyticalCapability, ...]:
        return tuple(self._items[i] for i in self._order)

    def owned_routes(self) -> Tuple[str, ...]:
        return tuple(sorted({c.route_owner for c in self.all() if c.route_owner}))

    def catalogue(self) -> List[dict]:
        """The catalogue, as data. Published on every analytical answer's
        metadata so a consumer can see what was available to the plan."""
        return [c.to_dict() for c in self.all()]

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, capability_id: object) -> bool:
        return capability_id in self._items


#: The process-wide analytical capability registry.
CAPABILITIES = CapabilityRegistry(_DECLARATIONS)
