"""mi_workflows.analytical.orchestrator — run a plan, collect findings.

The whole of the orchestration contract:

1. every call in the plan is executed by its registered deterministic executor;
2. the findings each returns are collected in plan order;
3. where a plan ran the same capability over two populations, the comparison
   between them is derived with ``mi_workflows.engine.compare_values`` and
   ``directionality_verdict`` — the platform's single definitions — and emitted
   as further findings;
4. an executor that raises degrades to an ``unavailable`` finding naming the
   capability, never to a missing number presented as a complete answer.

Nothing here decides what a figure means. The comparison arithmetic in step 3
is the engine's; this module only decides which pairs of findings are the same
measure over two populations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mi_workflows import engine

from .contract import (
    AnalyticalPlan,
    DATASET_FUNDED,
    Finding,
    KIND_COMPARISON,
    KIND_MEASURE,
    KIND_MOVEMENT,
    STATUS_OK,
    STATUS_UNAVAILABLE,
)
from .context import AnalyticalContext
from .registry import CAPABILITIES, CapabilityRegistry

logger = logging.getLogger("mi_workflows.analytical.orchestrator")


@dataclass
class ExecutionRecord:
    """What one capability call actually did."""

    capability: str
    because: str
    findings: int
    ok: bool
    engine: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {"capability": self.capability, "because": self.because,
               "findings": self.findings, "ok": self.ok, "engine": self.engine}
        if self.error:
            out["error"] = self.error
        return out


@dataclass
class AnalyticalResult:
    """The deterministic output of one plan. The narrative reads only this."""

    plan: AnalyticalPlan
    findings: Tuple[Finding, ...]
    execution: Tuple[ExecutionRecord, ...]
    warnings: Tuple[str, ...] = ()
    source_notes: Tuple[Dict[str, Any], ...] = ()
    #: Population evidence, merged across every capability that narrowed rows.
    population_evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Did anything at all compute?

        A plan whose every finding is unavailable has produced a refusal, not an
        answer, and the route hands the question back rather than presenting a
        page of blanks.
        """
        return any(f.ok and (f.value is not None or f.forecast_value is not None)
                   for f in self.findings)

    @property
    def satisfied(self) -> bool:
        """Did the plan produce what its own contract says it must?

        ``usable`` asks whether anything computed; this asks whether the thing
        the question needed computed. A funded-balance outlook that resolved the
        funded balance but no forecast is ``usable`` and NOT ``satisfied`` —
        presenting it would offer a point-in-time figure to someone who asked
        where the book is going.
        """
        for kind in self.plan.required_kinds:
            if not any(f.kind == kind and f.ok
                       and (f.value is not None or f.forecast_value is not None)
                       for f in self.findings):
                return False
        return True

    def unmet_reasons(self) -> Tuple[str, ...]:
        """Why the plan was not satisfied, in the capabilities' own words."""
        seen: List[str] = []
        for finding in self.findings:
            if finding.ok or not finding.note:
                continue
            if finding.note not in seen:
                seen.append(finding.note)
        return tuple(seen)

    def of_kind(self, *kinds: str) -> Tuple[Finding, ...]:
        wanted = set(kinds)
        return tuple(f for f in self.findings if f.kind in wanted)

    def by_capability(self, capability: str) -> Tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.capability == capability)

    def to_dict(self) -> Dict[str, Any]:
        return {"plan": self.plan.to_dict(),
                "findings": [f.to_dict() for f in self.findings],
                "execution": [e.to_dict() for e in self.execution],
                "warnings": list(self.warnings)}


def run(plan: AnalyticalPlan, ctx: AnalyticalContext, *,
        registry: CapabilityRegistry = CAPABILITIES) -> AnalyticalResult:
    """Execute a validated plan."""
    findings: List[Finding] = []
    records: List[ExecutionRecord] = []
    for call in plan.calls:
        capability = registry.require(call.capability)
        try:
            produced = list(capability.executor(ctx, **dict(call.inputs)) or [])
            records.append(ExecutionRecord(
                capability=call.capability, because=call.because,
                findings=len(produced), engine=capability.engine,
                ok=any(f.ok for f in produced)))
        except Exception as exc:  # noqa: BLE001 - a failed capability is a gap
            logger.warning("analytical capability %s failed: %s",
                           call.capability, exc)
            produced = [Finding(
                capability=call.capability, kind=KIND_MEASURE,
                label=call.capability.replace("_", " ").title(),
                status=STATUS_UNAVAILABLE,
                note=(f"The {call.capability.replace('_', ' ')} capability could "
                      f"not complete for this book ({type(exc).__name__})."))]
            records.append(ExecutionRecord(
                capability=call.capability, because=call.because,
                findings=0, ok=False, engine=capability.engine, error=str(exc)))
        findings.extend(produced)

    findings.extend(_comparisons(findings, ctx))
    return AnalyticalResult(
        plan=plan, findings=tuple(findings), execution=tuple(records),
        warnings=tuple(ctx.warnings), source_notes=tuple(ctx.source_notes),
        population_evidence=_population_evidence(findings))


# --------------------------------------------------------------------------- #
# Derived comparisons — the engine's arithmetic, this module's pairing
# --------------------------------------------------------------------------- #
def _comparisons(findings: Sequence[Finding], ctx: AnalyticalContext
                 ) -> List[Finding]:
    """Pair the same measure across two populations and compare it.

    Only pairs are compared: with three or more populations there is no single
    "the other side", and inventing one would be a judgement rather than a
    calculation.
    """
    bsr = ctx.business_semantics()
    out: List[Finding] = []
    for kind in (KIND_MEASURE, KIND_MOVEMENT):
        groups: Dict[str, List[Finding]] = {}
        for finding in findings:
            if finding.kind != kind or not finding.metric or not finding.ok:
                continue
            if finding.population is None or finding.value is None:
                continue
            groups.setdefault(finding.metric, []).append(finding)
        for metric, group in groups.items():
            populations = {f.population.key for f in group}
            if len(populations) != 2 or len(group) != 2:
                continue
            left, right = group[0], group[1]
            comparison = engine.compare_values(left.value, right.value)
            entry = bsr.get(metric) if bsr is not None else None
            verdict = engine.directionality_verdict(
                entry.directionality if entry else "neutral",
                left.value, right.value)
            out.append(Finding(
                capability="analytical_comparison", kind=KIND_COMPARISON,
                label=left.label, metric=metric,
                population=left.population, comparand=right.population,
                period=left.period, unit=left.unit, aggregation=left.aggregation,
                value=left.value, comparand_value=right.value,
                change=comparison["absolute"],
                relative_change=comparison["relative"],
                status=STATUS_OK,
                evidence={
                    "engine": "mi_workflows.engine.compare_values",
                    "calculationVersion": engine.CALCULATION_VERSION,
                    "directionality": entry.directionality if entry else "neutral",
                    "higher": verdict.get("higher"),
                    "interpretation": verdict.get("interpretation"),
                    "comparisonBasis": ("two governed populations at one "
                                        "reporting date" if kind == KIND_MEASURE
                                        else "two governed populations across "
                                             "the same reporting window"),
                    "leftPopulationRows": left.population.rows,
                    "rightPopulationRows": right.population.rows,
                }))
    return out


def _population_evidence(findings: Sequence[Finding]) -> Dict[str, Any]:
    """The merged row-population evidence this plan can prove.

    Feeds ``metadata.populationApplied``, which the P1L execution receipt
    reconciles a requested population against. A predicate that no finding
    proves is simply absent — the receipt then treats the population as lost and
    refuses, which is the correct outcome.
    """
    applied: List[str] = []
    unavailable: List[str] = []
    rows_before: Optional[int] = None
    rows_after: Optional[int] = None
    for finding in findings:
        # A capability that refused a population it could not apply records the
        # loss, so the receipt sees UNAVAILABLE rather than a silently absent
        # predicate. Only findings that carry a governed row predicate are read.
        if (finding.status == STATUS_UNAVAILABLE and finding.population is not None
                and not finding.population.rows and finding.population.label
                and finding.population.label not in unavailable
                and finding.population.dataset == DATASET_FUNDED
                and not finding.population.is_total):
            unavailable.append(finding.population.label)
        ref = finding.population
        if ref is None or ref.is_total or not ref.predicate:
            continue
        if ref.dataset != DATASET_FUNDED:
            # P1L governs the FUNDED row population. A pipeline-stage predicate
            # narrows a different set of rows entirely, and publishing it here
            # would let it stand as proof that a funded population was applied.
            continue
        for part in str(ref.predicate).split(";"):
            text = part.strip()
            if text and not text.startswith("portfolio lens") and text not in applied:
                applied.append(text)
        if ref.rows_before is not None and rows_before is None:
            rows_before = ref.rows_before
        if ref.rows is not None and rows_after is None:
            rows_after = ref.rows
    if not applied and not unavailable:
        return {}
    return {"applied": applied, "unavailable": unavailable,
            "rowsBefore": rows_before, "rowsAfter": rows_after}
