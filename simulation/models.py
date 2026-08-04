"""simulation.models — typed contracts for the hardening framework.

Plain dataclasses, no pandas, no production imports: the manifest and the loan
state model can be inspected, serialised and validated without loading the
pipeline. Vocabularies are closed on purpose — an unknown asset class, dialect
or failure class is a caller bug, not a state the framework quietly learns.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

# --------------------------------------------------------------------------- #
# Closed vocabularies
# --------------------------------------------------------------------------- #

#: Asset families the framework generates. Equipment finance is deliberately NOT
#: here: it is an asset-finance SUBTYPE (see ``ASSET_SUBTYPES``).
ASSET_EQUITY_RELEASE = "equity_release"
ASSET_BRIDGE = "bridge"
ASSET_FINANCE = "asset_finance"
ASSET_CLASSES: Tuple[str, ...] = (ASSET_EQUITY_RELEASE, ASSET_BRIDGE, ASSET_FINANCE)

#: Recognised subtypes per asset class. The subtype drives contract and
#: collateral configuration only; it never selects a different code path.
ASSET_SUBTYPES: Dict[str, Tuple[str, ...]] = {
    ASSET_EQUITY_RELEASE: ("lump_sum", "drawdown", "mixed"),
    ASSET_BRIDGE: ("residential_bridge", "commercial_bridge", "mixed_bridge"),
    ASSET_FINANCE: ("equipment_finance", "vehicle_finance", "mixed_asset_finance"),
}

#: The production ``--portfolio-type`` each asset family runs under. This is the
#: existing registry-selection extension point; nothing new was invented.
PORTFOLIO_TYPE_BY_ASSET: Dict[str, str] = {
    ASSET_EQUITY_RELEASE: "equity_release",
    ASSET_BRIDGE: "cre",
    ASSET_FINANCE: "equipment",
}

#: Source dialects. Registered renderers live in :mod:`simulation.dialects`.
DIALECT_CLEAN_CSV = "clean_csv"
DIALECT_LENDER_EXCEL = "lender_excel"
DIALECT_LOCALE_CSV = "european_locale_csv"
DIALECTS: Tuple[str, ...] = (DIALECT_CLEAN_CSV, DIALECT_LENDER_EXCEL, DIALECT_LOCALE_CSV)

#: Regime applicability, as the manifest must state it.
REGIME_APPLICABLE = "applicable"
REGIME_NOT_APPLICABLE = "not_applicable"
REGIME_NOT_SUPPORTED = "not_supported"
REGIME_AWAITING_DEAL_CONFIG = "awaiting_deal_configuration"
REGIME_APPLICABILITY: Tuple[str, ...] = (
    REGIME_APPLICABLE, REGIME_NOT_APPLICABLE, REGIME_NOT_SUPPORTED,
    REGIME_AWAITING_DEAL_CONFIG,
)

#: Intended risk state for a case, in the governed limit vocabulary.
RISK_GREEN = "green"
RISK_AMBER = "amber"
RISK_RED = "red"
RISK_STATES: Tuple[str, ...] = (RISK_GREEN, RISK_AMBER, RISK_RED)

#: Expected outcome of a case.
EXPECT_SUCCESS = "success"
EXPECT_FAILURE = "failure"
EXPECTATIONS: Tuple[str, ...] = (EXPECT_SUCCESS, EXPECT_FAILURE)


class FailureClass:
    """Closed classification for every way a case may stop.

    ``No unhandled exception is a valid expected outcome`` — an uncaught error is
    classified :data:`UNCLASSIFIED_EXCEPTION`, which the runner always treats as
    an unexpected failure.
    """

    #: The framework generated a source the pipeline correctly rejected.
    INTENDED_VALIDATION_FAILURE = "intended_validation_failure"
    #: The source schema is genuinely outside what ingestion supports.
    UNSUPPORTED_SOURCE_SCHEMA = "unsupported_source_schema"
    #: A mandatory canonical input was deliberately withheld.
    MISSING_MANDATORY_DATA = "missing_mandatory_data"
    #: The regime does not apply to this book.
    NON_APPLICABLE_REGIME = "non_applicable_regime"
    #: A defect in the platform under test.
    PRODUCTION_DEFECT = "production_defect"
    #: A defect in the simulator itself.
    SIMULATION_DEFECT = "simulation_generation_defect"
    #: Deterministic MI disagreed with independent expected truth.
    MI_ASSERTION_FAILURE = "mi_assertion_failure"
    #: The governed MI Agent answered without the required grounded evidence.
    AGENT_GROUNDING_FAILURE = "agent_grounding_failure"
    #: The regulatory artefact could not be produced or did not validate.
    REGULATORY_ARTEFACT_FAILURE = "regulatory_artefact_failure"
    #: Anything that escaped as a raw exception.
    UNCLASSIFIED_EXCEPTION = "unclassified_exception"

    ALL: Tuple[str, ...] = (
        INTENDED_VALIDATION_FAILURE, UNSUPPORTED_SOURCE_SCHEMA,
        MISSING_MANDATORY_DATA, NON_APPLICABLE_REGIME, PRODUCTION_DEFECT,
        SIMULATION_DEFECT, MI_ASSERTION_FAILURE, AGENT_GROUNDING_FAILURE,
        REGULATORY_ARTEFACT_FAILURE, UNCLASSIFIED_EXCEPTION,
    )

    #: Classes a case may legitimately DECLARE as its expected outcome. A case
    #: can never expect a production defect, a simulator defect or an
    #: unclassified exception — those are always failures of the run.
    DECLARABLE: Tuple[str, ...] = (
        INTENDED_VALIDATION_FAILURE, UNSUPPORTED_SOURCE_SCHEMA,
        MISSING_MANDATORY_DATA, NON_APPLICABLE_REGIME,
    )


#: Loan lifecycle states. Deliberately funded-book only: there is no pipeline,
#: application or decisioning state anywhere in this framework.
STATUS_PERFORMING = "performing"
STATUS_ARREARS = "arrears"
STATUS_DEFAULT = "default"
STATUS_ENFORCEMENT = "enforcement"
STATUS_REDEEMED = "redeemed"
STATUS_REPOSSESSED = "repossessed"
STATUS_WRITTEN_OFF = "written_off"
LOAN_STATUSES: Tuple[str, ...] = (
    STATUS_PERFORMING, STATUS_ARREARS, STATUS_DEFAULT, STATUS_ENFORCEMENT,
    STATUS_REDEEMED, STATUS_REPOSSESSED, STATUS_WRITTEN_OFF,
)

#: Statuses that mean the loan is still ON the funded book at the reporting date.
OPEN_STATUSES: Tuple[str, ...] = (
    STATUS_PERFORMING, STATUS_ARREARS, STATUS_DEFAULT, STATUS_ENFORCEMENT,
)

#: Statuses that mean the loan LEFT the book during the period.
CLOSED_STATUSES: Tuple[str, ...] = (
    STATUS_REDEEMED, STATUS_REPOSSESSED, STATUS_WRITTEN_OFF,
)

#: Movement kinds recorded in the history. Opening + these = closing, exactly.
MOVE_FUNDED = "funded_addition"
MOVE_ACQUIRED = "acquired_addition"
MOVE_INTEREST = "rolled_up_interest"
MOVE_AMORTISATION = "principal_amortisation"
MOVE_REPAYMENT = "repayment"
MOVE_REDEMPTION = "redemption"
MOVE_WRITE_OFF = "write_off"
MOVE_RECOVERY = "recovery"
MOVE_DISPOSAL = "asset_disposal"
MOVE_FURTHER_ADVANCE = "further_advance"
MOVE_RESTATEMENT = "restatement"
MOVEMENT_KINDS: Tuple[str, ...] = (
    MOVE_FUNDED, MOVE_ACQUIRED, MOVE_INTEREST, MOVE_AMORTISATION,
    MOVE_REPAYMENT, MOVE_REDEMPTION, MOVE_WRITE_OFF, MOVE_RECOVERY,
    MOVE_DISPOSAL, MOVE_FURTHER_ADVANCE, MOVE_RESTATEMENT,
)

#: Movements that ADD to the closing balance. Every other movement kind
#: subtracts. ``restatement`` is additive and SIGNED — a correction may move the
#: balance in either direction, and recording it with a sign keeps one rule
#: rather than two movement kinds that mean the same thing.
POSITIVE_MOVEMENTS: Tuple[str, ...] = (
    MOVE_FUNDED, MOVE_ACQUIRED, MOVE_INTEREST, MOVE_FURTHER_ADVANCE,
    MOVE_RESTATEMENT,
)


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
#: The closed set of DECLARED source-schema variations. Uncontrolled drift is
#: not allowed anywhere in the framework, so an unrecognised kind is an error.
EVOLVE_REORDER = "reorder_columns"
EVOLVE_ADD_OPTIONAL = "add_optional_column"
EVOLVE_DROP_OPTIONAL = "drop_optional_column"
EVOLVE_CHANGE_ALIAS = "change_alias"
EVOLVE_CHANGE_ENUMS = "change_enum_synonyms"
EVOLVE_SWITCH_FORMAT = "switch_format"
EVOLVE_RESTATE = "restate_prior"
EVOLVE_AMBIGUOUS = "ambiguous_schema"
EVOLVE_DROP_MANDATORY = "drop_mandatory_column"
SCHEMA_EVOLUTION_KINDS: Tuple[str, ...] = (
    EVOLVE_REORDER, EVOLVE_ADD_OPTIONAL, EVOLVE_DROP_OPTIONAL,
    EVOLVE_CHANGE_ALIAS, EVOLVE_CHANGE_ENUMS, EVOLVE_SWITCH_FORMAT,
    EVOLVE_RESTATE, EVOLVE_AMBIGUOUS, EVOLVE_DROP_MANDATORY,
)


@dataclass(frozen=True)
class SchemaEvolution:
    """A DECLARED source-schema variation for one reporting period.

    Every variation names the period it applies to and why it exists, so a
    failing run points at an intent rather than at randomness.
    """

    #: 0-based index of the reporting period the variation starts at.
    from_period: int
    #: One of :data:`SCHEMA_EVOLUTION_KINDS`.
    kind: str
    purpose: str
    #: Kind-specific detail (column name, alias, target format...).
    detail: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in SCHEMA_EVOLUTION_KINDS:
            raise ValueError(f"unknown schema-evolution kind {self.kind!r}; "
                             f"known: {SCHEMA_EVOLUTION_KINDS}")


@dataclass(frozen=True)
class RiskIntent:
    """The risk state a case is engineered to produce, and on which limits."""

    intended_status: str = RISK_GREEN
    targeted_limits: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.intended_status not in RISK_STATES:
            raise ValueError(f"unknown intended risk status "
                             f"{self.intended_status!r}")


#: Whether the repository can actually DELIVER the regulatory artefact for a
#: regime, as distinct from whether the regime applies to the book. Annex 2 has a
#: committed builder + XSD; the other annexes project correctly but have no
#: committed delivery template, so their artefact awaits deployment
#: configuration. Conflating the two would either fabricate an artefact or
#: mis-state applicability.
ARTEFACT_SUPPORTED = "supported"
ARTEFACT_AWAITING_CONFIG = "awaiting_configuration"
ARTEFACT_NOT_APPLICABLE = "not_applicable"
ARTEFACT_STATUSES: Tuple[str, ...] = (
    ARTEFACT_SUPPORTED, ARTEFACT_AWAITING_CONFIG, ARTEFACT_NOT_APPLICABLE)


@dataclass(frozen=True)
class RegimeIntent:
    """What the case asserts about regulatory applicability.

    ``regime_id`` must be one of the repository's genuinely supported regimes
    (validated against ``engine.orchestrator.trakt_run.VALID_REGULATORY_REGIMES``
    by :func:`simulation.manifests.validate_manifest`). Applicability is never
    invented: a case that does not attract a regime says so and is skipped with
    a reason.
    """

    applicability: str = REGIME_NOT_APPLICABLE
    regime_id: Optional[str] = None
    reason: str = ""
    artefact_status: str = ARTEFACT_NOT_APPLICABLE

    def __post_init__(self) -> None:
        if self.applicability not in REGIME_APPLICABILITY:
            raise ValueError(f"unknown regime applicability "
                             f"{self.applicability!r}")
        if self.artefact_status not in ARTEFACT_STATUSES:
            raise ValueError(f"unknown artefact status {self.artefact_status!r}")


@dataclass(frozen=True)
class CaseManifest:
    """One versioned, seed-controlled simulation case.

    Everything needed to reproduce the case exactly is here. The runner writes
    this object verbatim into every run directory, so a failing generated case
    becomes a permanent regression fixture by copying one JSON file.
    """

    case_id: str
    simulation_version: int
    seed: int
    purpose: str

    asset_class: str
    asset_subtype: str
    client_id: str
    portfolio_id: str
    #: Funding structure. Carried as the portfolio/cohort segmentation because
    #: ``spv_id`` is not a canonical registry field (see the framework doc).
    spv_id: str
    source_portfolio_type: str          # direct | acquired
    currency: str
    jurisdiction: str

    start_reporting_date: str           # ISO month-end
    months: int
    opening_loan_count: int

    dialects: Tuple[str, ...]
    economic_intent: Tuple[str, ...]
    risk_intent: RiskIntent
    regime: RegimeIntent

    #: Mandatory for an ACQUIRED book: ``engine.provenance`` fails closed
    #: without it rather than onboarding a back book with an unknown
    #: acquisition date, so the manifest has to carry it.
    acquisition_date: Optional[str] = None
    seller_name: Optional[str] = None
    expectation: str = EXPECT_SUCCESS
    expected_failure_class: Optional[str] = None
    #: Declared, purposeful schema variation across the history.
    schema_evolution: Tuple[SchemaEvolution, ...] = ()
    #: ``small`` | ``standard`` | ``large`` — see :data:`SCALES`.
    scale: str = "small"
    #: Profiles this case belongs to.
    profiles: Tuple[str, ...] = ("standard",)
    #: Validation warnings the run is expected to emit (substring match).
    expected_validation_warnings: Tuple[str, ...] = ()

    # -- identity ---------------------------------------------------------- #
    @property
    def run_key(self) -> str:
        """Stable identity: case, seed, asset class and simulator version."""
        return (f"{self.case_id}|{self.asset_class}|seed={self.seed}"
                f"|v{self.simulation_version}")

    def content_hash(self) -> str:
        return "sha256:" + hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                       default=str).encode("utf-8")).hexdigest()[:32]

    @property
    def portfolio_type(self) -> str:
        """The production ``--portfolio-type`` this case runs under."""
        return PORTFOLIO_TYPE_BY_ASSET[self.asset_class]

    def reporting_dates(self) -> List[str]:
        from .dates import month_end_series
        return month_end_series(self.start_reporting_date, self.months)

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        # dataclasses.asdict turns tuples into lists already; normalise nested.
        out["risk_intent"] = {"intended_status": self.risk_intent.intended_status,
                              "targeted_limits": list(self.risk_intent.targeted_limits)}
        out["regime"] = {"applicability": self.regime.applicability,
                         "regime_id": self.regime.regime_id,
                         "reason": self.regime.reason,
                         "artefact_status": self.regime.artefact_status}
        out["schema_evolution"] = [
            {"from_period": s.from_period, "kind": s.kind,
             "purpose": s.purpose, "detail": dict(s.detail)}
            for s in self.schema_evolution]
        out["portfolio_type"] = self.portfolio_type
        out["run_key"] = self.run_key
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseManifest":
        payload = dict(data)
        payload.pop("portfolio_type", None)
        payload.pop("run_key", None)
        ri = payload.get("risk_intent") or {}
        payload["risk_intent"] = RiskIntent(
            intended_status=ri.get("intended_status", RISK_GREEN),
            targeted_limits=tuple(ri.get("targeted_limits") or ()))
        rg = payload.get("regime") or {}
        payload["regime"] = RegimeIntent(
            applicability=rg.get("applicability", REGIME_NOT_APPLICABLE),
            regime_id=rg.get("regime_id"), reason=rg.get("reason", ""),
            artefact_status=rg.get("artefact_status", ARTEFACT_NOT_APPLICABLE))
        payload["schema_evolution"] = tuple(
            SchemaEvolution(from_period=int(s["from_period"]), kind=s["kind"],
                            purpose=s.get("purpose", ""),
                            detail=dict(s.get("detail") or {}))
            for s in (payload.get("schema_evolution") or ()))
        for key in ("dialects", "economic_intent", "profiles",
                    "expected_validation_warnings"):
            if key in payload and payload[key] is not None:
                payload[key] = tuple(payload[key])
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})


#: Loan-count scales. ``opening_loan_count`` on a manifest is the SMALL figure;
#: the runner multiplies it for the larger profiles so one manifest serves all.
SCALES: Dict[str, int] = {"small": 100, "standard": 5_000, "large": 100_000}


# --------------------------------------------------------------------------- #
# Loan state + movements
# --------------------------------------------------------------------------- #
@dataclass
class LoanState:
    """One loan at one reporting date.

    Deliberately mutable: :mod:`simulation.history` rolls a loan forward month by
    month in place and snapshots a frozen copy per period. Asset-specific
    attributes live in ``attributes`` so the core roll-forward stays
    asset-agnostic.
    """

    loan_id: str
    status: str
    #: Balance at the START of the current period.
    opening_balance: float
    #: Balance at the END of the current period (0.0 once closed).
    closing_balance: float
    origination_date: str
    funding_date: str
    original_principal: float
    interest_rate: float                       # annual percentage points
    #: Collateral / asset value at the reporting date.
    current_value: float
    original_value: float
    borrower_id: str
    region: str
    postcode: str
    #: Period-scoped movement amounts, keyed by :data:`MOVEMENT_KINDS`.
    movements: Dict[str, float] = field(default_factory=dict)
    #: Asset-class-specific fields (charge rank, balloon, manufacturer, ...).
    attributes: Dict[str, Any] = field(default_factory=dict)
    #: Cohort tag: ``direct`` originations vs an ``acquired`` book.
    cohort: str = "direct"
    #: Set on the period the loan left the book.
    exit_date: Optional[str] = None
    exit_reason: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES

    def record(self, kind: str, amount: float) -> None:
        if kind not in MOVEMENT_KINDS:
            raise ValueError(f"unknown movement kind {kind!r}")
        self.movements[kind] = round(self.movements.get(kind, 0.0) + amount, 2)

    def reset_period(self) -> None:
        self.opening_balance = self.closing_balance
        self.movements = {}


@dataclass(frozen=True)
class PeriodSnapshot:
    """The frozen funded book at one reporting date, plus its movements."""

    reporting_date: str
    period_index: int
    #: Loans ON the book at the reporting date (open statuses only).
    loans: Tuple[LoanState, ...]
    #: Loans that LEFT during the period — needed to reconcile the movement.
    exits: Tuple[LoanState, ...]
    opening_balance: float
    closing_balance: float
    movement_totals: Dict[str, float]

    @property
    def loan_count(self) -> int:
        return len(self.loans)

    def reconciles(self, tolerance: float = 0.05) -> bool:
        """Opening + every movement == closing, under transparent rules."""
        delta = sum(v if k in POSITIVE_MOVEMENTS else -v
                    for k, v in self.movement_totals.items())
        return abs((self.opening_balance + delta) - self.closing_balance) <= tolerance


@dataclass(frozen=True)
class EconomicHistory:
    """The complete generated history for one case: ONE economic truth."""

    case_id: str
    seed: int
    asset_class: str
    periods: Tuple[PeriodSnapshot, ...]

    def period(self, index: int) -> PeriodSnapshot:
        return self.periods[index]

    def reporting_dates(self) -> List[str]:
        return [p.reporting_date for p in self.periods]


# --------------------------------------------------------------------------- #
# Assertion + stage results
# --------------------------------------------------------------------------- #
STAGE_GENERATE = "generate"
STAGE_RENDER = "render"
STAGE_INGEST = "ingest"
STAGE_CANONICAL = "canonical"
STAGE_EQUIVALENCE = "canonical_equivalence"
STAGE_HISTORY = "history_integration"
STAGE_MI = "mi"
STAGE_RISK = "risk"
STAGE_AGENT = "agent"
STAGE_REGIME = "regime"
STAGES: Tuple[str, ...] = (
    STAGE_GENERATE, STAGE_RENDER, STAGE_INGEST, STAGE_CANONICAL,
    STAGE_EQUIVALENCE, STAGE_HISTORY, STAGE_MI, STAGE_RISK, STAGE_AGENT,
    STAGE_REGIME,
)


@dataclass
class Assertion:
    """One verifiable claim about a production output."""

    stage: str
    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    detail: str = ""
    #: Present only on failure.
    failure_class: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "name": self.name, "passed": self.passed,
                "expected": self.expected, "actual": self.actual,
                "detail": self.detail, "failure_class": self.failure_class}


@dataclass
class StageResult:
    """Outcome of one pipeline stage for one case."""

    stage: str
    ok: bool
    skipped: bool = False
    skip_reason: str = ""
    failure_class: Optional[str] = None
    message: str = ""
    duration_seconds: float = 0.0
    assertions: List[Assertion] = field(default_factory=list)
    artefacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def failed_assertions(self) -> List[Assertion]:
        return [a for a in self.assertions if not a.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "ok": self.ok, "skipped": self.skipped,
                "skip_reason": self.skip_reason,
                "failure_class": self.failure_class, "message": self.message,
                "duration_seconds": round(self.duration_seconds, 3),
                "assertions": [a.to_dict() for a in self.assertions],
                "artefacts": self.artefacts}


@dataclass
class CaseResult:
    """The complete outcome of running one case."""

    case_id: str
    seed: int
    asset_class: str
    simulation_version: int
    expectation: str
    stages: List[StageResult] = field(default_factory=list)
    run_dir: str = ""
    manifest_hash: str = ""
    production_modules: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    peak_memory_mb: Optional[float] = None

    # -- outcome ----------------------------------------------------------- #
    @property
    def failures(self) -> List[StageResult]:
        return [s for s in self.stages if not s.ok and not s.skipped]

    @property
    def observed_failure_class(self) -> Optional[str]:
        for s in self.stages:
            if not s.ok and not s.skipped:
                return s.failure_class or FailureClass.UNCLASSIFIED_EXCEPTION
        return None

    @property
    def ok(self) -> bool:
        """True when the case ended the way its manifest said it would."""
        observed = self.observed_failure_class
        if self.expectation == EXPECT_SUCCESS:
            return observed is None
        return observed is not None and observed == self.expected_failure_class

    #: Set by the runner from the manifest so :attr:`ok` can compare.
    expected_failure_class: Optional[str] = None

    def assertion_counts(self) -> Dict[str, int]:
        total = sum(len(s.assertions) for s in self.stages)
        passed = sum(1 for s in self.stages for a in s.assertions if a.passed)
        return {"total": total, "passed": passed, "failed": total - passed}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id, "seed": self.seed,
            "asset_class": self.asset_class,
            "simulation_version": self.simulation_version,
            "expectation": self.expectation,
            "expected_failure_class": self.expected_failure_class,
            "observed_failure_class": self.observed_failure_class,
            "ok": self.ok, "run_dir": self.run_dir,
            "manifest_hash": self.manifest_hash,
            "assertions": self.assertion_counts(),
            "production_modules": sorted(set(self.production_modules)),
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "peak_memory_mb": self.peak_memory_mb,
            "stages": [s.to_dict() for s in self.stages],
        }


__all__ = [
    "ARTEFACT_AWAITING_CONFIG", "ARTEFACT_NOT_APPLICABLE", "ARTEFACT_STATUSES",
    "ARTEFACT_SUPPORTED",
    "ASSET_CLASSES", "ASSET_SUBTYPES", "ASSET_BRIDGE", "ASSET_EQUITY_RELEASE",
    "ASSET_FINANCE", "Assertion", "CLOSED_STATUSES", "CaseManifest",
    "CaseResult", "DIALECTS", "DIALECT_CLEAN_CSV", "DIALECT_LENDER_EXCEL",
    "DIALECT_LOCALE_CSV", "EXPECTATIONS", "EXPECT_FAILURE", "EXPECT_SUCCESS",
    "EconomicHistory", "FailureClass", "LOAN_STATUSES", "LoanState",
    "MOVEMENT_KINDS", "MOVE_ACQUIRED", "MOVE_AMORTISATION", "MOVE_DISPOSAL",
    "MOVE_FUNDED", "MOVE_FURTHER_ADVANCE", "MOVE_INTEREST", "MOVE_RECOVERY",
    "MOVE_REDEMPTION", "MOVE_REPAYMENT", "MOVE_RESTATEMENT", "MOVE_WRITE_OFF",
    "OPEN_STATUSES", "PORTFOLIO_TYPE_BY_ASSET", "POSITIVE_MOVEMENTS",
    "PeriodSnapshot", "REGIME_APPLICABILITY", "REGIME_APPLICABLE",
    "REGIME_AWAITING_DEAL_CONFIG", "REGIME_NOT_APPLICABLE",
    "REGIME_NOT_SUPPORTED", "RISK_AMBER", "RISK_GREEN", "RISK_RED",
    "RISK_STATES", "RegimeIntent", "RiskIntent", "SCALES", "STAGES",
    "STAGE_AGENT", "STAGE_CANONICAL", "STAGE_EQUIVALENCE", "STAGE_GENERATE",
    "STAGE_HISTORY", "STAGE_INGEST", "STAGE_MI", "STAGE_REGIME", "STAGE_RENDER",
    "STAGE_RISK", "STATUS_ARREARS", "STATUS_DEFAULT", "STATUS_ENFORCEMENT",
    "STATUS_PERFORMING", "STATUS_REDEEMED", "STATUS_REPOSSESSED",
    "STATUS_WRITTEN_OFF", "SchemaEvolution", "StageResult",
]
