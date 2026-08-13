"""trakt_core.readiness — the readiness framework, and one fact under many rulebooks.

The separation this module exists to enforce
--------------------------------------------
A portfolio fact and a rule about that fact are different things, and Trakt must
never conflate them::

    FACT       London share of balance = 28.7%          measured by Trakt
    EXTERNAL   warehouse limit         <= 35%    PASS   from an agreement
    EXTERNAL   proposed transaction    <= 30%    PASS   from a term sheet
    SCREENING  Trakt attention         <= 25%    FLAG   Trakt's own threshold

One measurement. Three rulebooks. Three different answers, none of which
recalculates anything, and each of which carries the authority it came from.

That last part is the load-bearing one. A screening FLAG and a warehouse BREACH
look similar in a report and mean entirely different things: one is Trakt
suggesting somebody look, the other is a contractual failure. Every result here
carries ``authority`` and ``authority_label`` so a downstream agent physically
cannot present the first as the second.

Why the fact is computed once
-----------------------------
:func:`evaluate_packs` groups every rule across every pack by the
``(metric, parameters)`` it needs, computes each distinct fact **once**, and then
applies all the rules that reference it. Two consequences, and the second is the
important one:

* it is cheaper — N packs over M shared metrics cost M computations, not N×M;
* **the packs cannot disagree about the fact.** If each pack computed its own
  London share, a rounding difference or a stale parameter would let the
  warehouse pass and the securitisation criteria fail on the same portfolio for
  no reason a reviewer could explain. Sharing the computation makes that
  impossible rather than unlikely.

What this module does NOT do
----------------------------
It computes nothing. Facts arrive from the existing deterministic engines —
``mi_agent.concentration_tests.metrics`` for the 39 registered metric evaluators,
``analytics_lib`` for the shared maths, the governed tools for the rest. There is
no securitisation-specific calculation anywhere in Trakt, and this file is where
that would most easily have crept in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from . import config_cache

DEFAULT_FRAMEWORK_CONFIG = "config/system/securitisation_readiness_framework.yaml"
FRAMEWORK_CONFIG_ENV = "TRAKT_READINESS_FRAMEWORK_CONFIG"

DEFAULT_RULE_PACK_DIR = "config/system/rule_packs"
RULE_PACK_DIR_ENV = "TRAKT_RULE_PACK_DIR"

# -- metric classification -------------------------------------------------- #
MEASURE_ONLY = "MEASURE_ONLY"
TRAKT_SCREEN = "TRAKT_SCREEN"
EXTERNAL_CRITERIA = "EXTERNAL_CRITERIA"
HYBRID = "HYBRID"
METRIC_TYPES = frozenset({MEASURE_ONLY, TRAKT_SCREEN, EXTERNAL_CRITERIA, HYBRID})

# -- gap classification ----------------------------------------------------- #
READY = "READY"
EXPOSE = "EXPOSE"
SMALL_GAP = "SMALL_GAP"
DATA_GAP = "DATA_GAP"
JUDGEMENT_ONLY = "JUDGEMENT_ONLY"
OUT_OF_SCOPE = "OUT_OF_SCOPE"
STATUSES = frozenset({READY, EXPOSE, SMALL_GAP, DATA_GAP, JUDGEMENT_ONLY,
                      OUT_OF_SCOPE})

# -- authority -------------------------------------------------------------- #
#: Where a rule came from. The distinction a report must never lose.
AUTHORITY_TRAKT = "trakt_internal"
AUTHORITY_WAREHOUSE = "warehouse_agreement"
AUTHORITY_SECURITISATION = "securitisation_criteria"
AUTHORITY_RATING = "rating_methodology"
AUTHORITY_CLIENT = "client_mandate"
AUTHORITY_REGULATORY = "regulatory"
EXTERNAL_AUTHORITIES = frozenset({
    AUTHORITY_WAREHOUSE, AUTHORITY_SECURITISATION, AUTHORITY_RATING,
    AUTHORITY_CLIENT, AUTHORITY_REGULATORY})

# -- outcomes --------------------------------------------------------------- #
#: Internal packs FLAG or CLEAR. External packs PASS or BREACH. Deliberately
#: different vocabularies, so the two can never be confused in a report.
OUTCOME_FLAG = "flag"
OUTCOME_CLEAR = "clear"
OUTCOME_PASS = "pass"
OUTCOME_BREACH = "breach"
OUTCOME_WARNING = "warning"
OUTCOME_UNAVAILABLE = "unavailable"

OPERATOR_MAX = "max"
OPERATOR_MIN = "min"
OPERATORS = frozenset({OPERATOR_MAX, OPERATOR_MIN})


# --------------------------------------------------------------------------- #
# The framework
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrameworkMetric:
    """One thing a first-pass readiness review would look at."""

    id: str
    name: str
    category: str
    description: str = ""
    type: str = MEASURE_ONLY
    status: str = READY
    fact_tool: Optional[str] = None
    calculation_source: Optional[str] = None
    metric_id: Optional[str] = None
    #: The shared MI capability this readiness requirement consumes, from
    #: ``config/system/mi_capability_registry.yaml``. Readiness owns no
    #: alternative calculation: the requirement names a capability, the
    #: capability names the one implementation, and its availability state
    #: tells a reviewer whether an absence is a data gap or simply a metric
    #: that does not apply to this book.
    capability: Optional[str] = None
    #: Set when ``metric_id`` names a concentration-library metric that is
    #: still unimplemented, but a *tool* now serves the same economic concept
    #: by a different route. Prepayment is the example: a rate needs two
    #: periods, so no single-snapshot library evaluator can produce one, and
    #: ``prepayment_analysis`` supersedes it. Without this marker the two
    #: sources drift silently — which is exactly how PERF_PREPAYMENT and
    #: PERF_LOSSES stayed published as gaps for a sprint after their tools
    #: were built.
    supersedes_library_metric: bool = False
    screening_rule: Optional[str] = None
    external_criteria_supported: bool = True
    severity_if_flagged: Optional[str] = None
    asset_class_applicability: Tuple[str, ...] = ("cross_asset",)
    evidence_required: Tuple[str, ...] = ()
    agent_investigation_guidance: str = ""

    @property
    def deterministic(self) -> bool:
        """Can Trakt measure this today, through a governed tool?"""
        return self.status == READY and bool(self.fact_tool)

    def applies_to(self, asset_class: str) -> bool:
        if not asset_class:
            return True
        return ("cross_asset" in self.asset_class_applicability
                or asset_class in self.asset_class_applicability)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "category": self.category,
            "description": self.description, "type": self.type,
            "status": self.status, "fact_tool": self.fact_tool,
            "calculation_source": self.calculation_source,
            "metric_id": self.metric_id,
            "capability": self.capability,
            "supersedes_library_metric": self.supersedes_library_metric,
            "screening_rule": self.screening_rule,
            "external_criteria_supported": self.external_criteria_supported,
            "severity_if_flagged": self.severity_if_flagged,
            "asset_class_applicability": list(self.asset_class_applicability),
            "evidence_required": list(self.evidence_required),
            "agent_investigation_guidance": self.agent_investigation_guidance,
        }


@dataclass(frozen=True)
class ReadinessFramework:
    """The registry, indexed for lookup."""

    framework_id: str = "SECURITISATION_READINESS"
    framework_name: str = ""
    version: int = 0
    authority: str = AUTHORITY_TRAKT
    authority_note: str = ""
    categories: Dict[str, str] = field(default_factory=dict)
    metrics: Tuple[FrameworkMetric, ...] = ()
    configured: bool = False

    @property
    def reference(self) -> str:
        return f"{self.framework_id}@v{self.version}"

    def get(self, metric_id: str) -> Optional[FrameworkMetric]:
        return next((m for m in self.metrics if m.id == metric_id), None)

    def by_category(self, category: str) -> List[FrameworkMetric]:
        return [m for m in self.metrics if m.category == category]

    def coverage(self, asset_class: str = "") -> Dict[str, Any]:
        """How much of the framework Trakt can evaluate deterministically today.

        Counted honestly: only ``READY`` metrics with a governed tool count as
        covered. ``JUDGEMENT_ONLY`` is excluded from the denominator rather than
        counted as covered — no deterministic calculation *should* exist for it,
        so scoring it either way would misrepresent the number.
        """
        applicable = [m for m in self.metrics if m.applies_to(asset_class)]
        scored = [m for m in applicable if m.status != JUDGEMENT_ONLY]
        by_category: Dict[str, Dict[str, Any]] = {}
        for metric in scored:
            bucket = by_category.setdefault(
                metric.category, {"total": 0, "deterministic": 0, "metrics": []})
            bucket["total"] += 1
            bucket["deterministic"] += 1 if metric.deterministic else 0
            bucket["metrics"].append(metric.id)
        for bucket in by_category.values():
            bucket["coverage_pct"] = (round(bucket["deterministic"]
                                            / bucket["total"] * 100, 1)
                                      if bucket["total"] else None)
        covered = sum(1 for m in scored if m.deterministic)
        return {
            "framework": self.reference,
            "asset_class": asset_class or "all",
            "scored_metrics": len(scored),
            "deterministic": covered,
            "coverage_pct": (round(covered / len(scored) * 100, 1)
                             if scored else None),
            "excluded_judgement_only": [m.id for m in applicable
                                        if m.status == JUDGEMENT_ONLY],
            "by_category": dict(sorted(by_category.items())),
            "by_status": {
                status: sorted(m.id for m in applicable if m.status == status)
                for status in sorted({m.status for m in applicable})},
        }


def load_framework(config_path: Optional[str | Path] = None) -> ReadinessFramework:
    """Load the framework registry, cached on content identity."""
    path = Path(config_path
                or os.environ.get(FRAMEWORK_CONFIG_ENV)
                or DEFAULT_FRAMEWORK_CONFIG)

    def _load() -> ReadinessFramework:
        if not path.exists():
            return ReadinessFramework(configured=False)
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        metrics: List[FrameworkMetric] = []
        seen: set = set()
        for entry in (raw.get("metrics") or ()):
            entry = entry or {}
            metric_key = str(entry.get("id") or "").strip()
            if not metric_key or metric_key in seen:
                # A duplicate id would make `get` ambiguous and let two
                # different definitions answer to one name.
                continue
            seen.add(metric_key)
            declared_type = str(entry.get("type") or MEASURE_ONLY)
            declared_status = str(entry.get("status") or READY)
            metrics.append(FrameworkMetric(
                id=metric_key,
                name=str(entry.get("name") or metric_key),
                category=str(entry.get("category") or "uncategorised"),
                description=str(entry.get("description") or "").strip(),
                type=declared_type if declared_type in METRIC_TYPES else MEASURE_ONLY,
                status=declared_status if declared_status in STATUSES else READY,
                fact_tool=entry.get("fact_tool") or None,
                calculation_source=entry.get("calculation_source") or None,
                metric_id=entry.get("metric_id") or None,
                capability=entry.get("capability") or None,
                supersedes_library_metric=bool(
                    entry.get("supersedes_library_metric", False)),
                screening_rule=entry.get("screening_rule") or None,
                external_criteria_supported=bool(
                    entry.get("external_criteria_supported", True)),
                severity_if_flagged=entry.get("severity_if_flagged") or None,
                asset_class_applicability=tuple(
                    str(a) for a in (entry.get("asset_class_applicability")
                                     or ("cross_asset",))),
                evidence_required=tuple(
                    str(e) for e in (entry.get("evidence_required") or ())),
                agent_investigation_guidance=str(
                    entry.get("agent_investigation_guidance") or "").strip()))

        return ReadinessFramework(
            framework_id=str(raw.get("framework_id") or "SECURITISATION_READINESS"),
            framework_name=str(raw.get("framework_name") or ""),
            version=int(raw.get("framework_version") or 1),
            authority=str(raw.get("authority") or AUTHORITY_TRAKT),
            authority_note=str(raw.get("authority_note") or "").strip(),
            categories={str(k): str(v)
                        for k, v in (raw.get("categories") or {}).items()},
            metrics=tuple(metrics), configured=True)

    return config_cache.get_or_load("readiness_framework", path, _load)


# --------------------------------------------------------------------------- #
# Rule packs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    """One threshold applied to one fact, from one rulebook."""

    rule_id: str
    framework_metric: str
    operator: str = OPERATOR_MAX
    threshold: Optional[float] = None
    unit: str = "percent"
    severity: str = "medium"
    rationale: str = ""
    #: How the fact is obtained: a registered concentration metric, or a named
    #: key in a fact block a tool produced. Exactly one is used.
    metric_id: Optional[str] = None
    fact_key: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def fact_signature(self) -> Tuple:
        """What identifies the fact this rule needs.

        Two rules from different packs with the same signature share one
        computation — which is what makes it impossible for the packs to
        disagree about the underlying number.
        """
        return (self.metric_id or "", self.fact_key or "",
                tuple(sorted((k, _hashable(v))
                             for k, v in (self.parameters or {}).items())))


def _hashable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


@dataclass(frozen=True)
class RulePack:
    """One rulebook. Trakt's own, or somebody else's."""

    pack_id: str
    pack_name: str = ""
    version: int = 1
    authority: str = AUTHORITY_TRAKT
    authority_label: str = ""
    purpose: str = "attention"
    description: str = ""
    rules: Tuple[Rule, ...] = ()
    source_document: str = ""
    effective_date: str = ""
    expiry_date: str = ""

    @property
    def reference(self) -> str:
        return f"{self.pack_id}@v{self.version}"

    @property
    def is_external(self) -> bool:
        return self.authority in EXTERNAL_AUTHORITIES

    def outcome_for(self, breached: bool) -> str:
        """FLAG/CLEAR for an internal pack, BREACH/PASS for an external one.

        Different words on purpose. A screening flag and a contractual breach
        read alike in a summary and mean entirely different things, and giving
        them the same vocabulary is how that distinction gets lost.
        """
        if self.is_external:
            return OUTCOME_BREACH if breached else OUTCOME_PASS
        return OUTCOME_FLAG if breached else OUTCOME_CLEAR


def _rule_from(entry: Mapping[str, Any]) -> Optional[Rule]:
    rule_id = str(entry.get("rule_id") or "").strip()
    if not rule_id:
        return None
    operator = str(entry.get("operator") or OPERATOR_MAX)
    threshold = entry.get("threshold")
    return Rule(
        rule_id=rule_id,
        framework_metric=str(entry.get("framework_metric") or ""),
        operator=operator if operator in OPERATORS else OPERATOR_MAX,
        threshold=None if threshold is None else float(threshold),
        unit=str(entry.get("unit") or "percent"),
        severity=str(entry.get("severity") or "medium"),
        rationale=str(entry.get("rationale") or "").strip(),
        metric_id=entry.get("metric_id") or None,
        fact_key=entry.get("fact_key") or None,
        parameters=dict(entry.get("parameters") or {}))


def load_rule_packs(directory: Optional[str | Path] = None) -> List[RulePack]:
    """Every rule pack on this deployment, Trakt's own and any supplied.

    Packs are files, so adding a client's warehouse criteria is a deployment
    action rather than a code change — which is the whole point of the pack
    being a pack.
    """
    root = Path(directory
                or os.environ.get(RULE_PACK_DIR_ENV)
                or DEFAULT_RULE_PACK_DIR)
    if not root.exists():
        return []

    packs: List[RulePack] = []
    for path in sorted(root.glob("*.yaml")):
        pack = load_rule_pack(path)
        if pack is not None:
            packs.append(pack)
    return packs


def load_rule_pack(path: str | Path) -> Optional[RulePack]:
    """One pack, cached on content identity."""
    path = Path(path)

    def _load() -> Optional[RulePack]:
        if not path.exists():
            return None
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        pack_id = str(raw.get("pack_id") or "").strip()
        if not pack_id:
            return None
        rules = [r for r in (_rule_from(e or {})
                             for e in (raw.get("rules") or ())) if r]
        authority = str(raw.get("authority") or AUTHORITY_TRAKT)
        return RulePack(
            pack_id=pack_id,
            pack_name=str(raw.get("pack_name") or pack_id),
            version=int(raw.get("version") or 1),
            authority=authority,
            authority_label=str(raw.get("authority_label") or "").strip(),
            purpose=str(raw.get("purpose") or "attention"),
            description=str(raw.get("description") or "").strip(),
            rules=tuple(rules),
            source_document=str(raw.get("source_document") or ""),
            effective_date=str(raw.get("effective_date") or ""),
            expiry_date=str(raw.get("expiry_date") or ""))

    return config_cache.get_or_load(f"rule_pack:{path.name}", path, _load)


# --------------------------------------------------------------------------- #
# Evaluation: compute each fact once, apply every rulebook to it
# --------------------------------------------------------------------------- #
@dataclass
class Fact:
    """One measured value, and where it came from."""

    signature: Tuple
    value: Optional[float]
    unit: str = ""
    available: bool = True
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "unit": self.unit,
                "available": self.available,
                **({"reason": self.reason} if self.reason else {}),
                **({"detail": self.detail} if self.detail else {})}


def _breached(value: float, operator: str, threshold: float) -> bool:
    return value > threshold if operator == OPERATOR_MAX else value < threshold


def evaluate_packs(
    packs: Iterable[RulePack],
    compute_fact: Callable[[Rule], Fact],
    *,
    framework: Optional[ReadinessFramework] = None,
    warning_fraction: float = 0.9,
) -> Dict[str, Any]:
    """Apply every pack to facts computed ONCE per distinct signature.

    ``compute_fact`` is injected rather than imported so this stays free of
    pandas and of any particular engine — the same reason ``trakt_core`` holds no
    dataframe code anywhere else. The caller supplies the deterministic
    measurement; this decides nothing except pass/fail against declared
    thresholds.
    """
    packs = list(packs)
    facts: Dict[Tuple, Fact] = {}

    # ---- one computation per distinct (metric, parameters) --------------- #
    for pack in packs:
        for rule in pack.rules:
            if rule.fact_signature not in facts:
                facts[rule.fact_signature] = compute_fact(rule)

    results: List[Dict[str, Any]] = []
    for pack in packs:
        for rule in pack.rules:
            fact = facts[rule.fact_signature]
            metric = framework.get(rule.framework_metric) if framework else None

            entry: Dict[str, Any] = {
                "rule_id": rule.rule_id,
                "pack": pack.reference,
                "pack_name": pack.pack_name,
                # The three fields that keep a screening flag from being read as
                # a contractual breach.
                "authority": pack.authority,
                "authority_label": pack.authority_label,
                "is_external_criterion": pack.is_external,
                "framework_metric": rule.framework_metric,
                "metric_name": metric.name if metric else rule.framework_metric,
                "category": metric.category if metric else None,
                "operator": rule.operator,
                "threshold": rule.threshold,
                "unit": rule.unit,
                "severity": rule.severity,
                "parameters": dict(rule.parameters),
                "fact": fact.to_dict(),
                "rationale": rule.rationale,
            }

            if not fact.available or fact.value is None or rule.threshold is None:
                entry["outcome"] = OUTCOME_UNAVAILABLE
                entry["reason"] = (fact.reason
                                   or "the fact could not be measured")
                results.append(entry)
                continue

            breached = _breached(float(fact.value), rule.operator,
                                 float(rule.threshold))
            entry["outcome"] = pack.outcome_for(breached)
            entry["value"] = float(fact.value)
            entry["headroom"] = (float(rule.threshold) - float(fact.value)
                                 if rule.operator == OPERATOR_MAX
                                 else float(fact.value) - float(rule.threshold))
            if not breached and rule.threshold:
                near = (float(fact.value) >= float(rule.threshold) * warning_fraction
                        if rule.operator == OPERATOR_MAX
                        else float(fact.value) <= float(rule.threshold)
                        / max(warning_fraction, 1e-9))
                if near:
                    entry["approaching"] = True
            results.append(entry)

    return {
        "framework": framework.reference if framework else None,
        "packs": [{"pack": p.reference, "pack_name": p.pack_name,
                   "authority": p.authority,
                   "authority_label": p.authority_label,
                   "is_external_criterion": p.is_external,
                   "rule_count": len(p.rules),
                   "source_document": p.source_document} for p in packs],
        "results": results,
        "facts_computed": len(facts),
        "rules_evaluated": len(results),
        "summary": summarise(results),
    }


def summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts, kept separate by authority.

    External breaches and internal flags are never summed into one number: a
    total that mixed them would be the exact conflation this module exists to
    prevent.
    """
    external = [r for r in results if r.get("is_external_criterion")]
    internal = [r for r in results if not r.get("is_external_criterion")]
    return {
        "external_criteria": {
            "evaluated": len(external),
            "breaches": sum(1 for r in external if r["outcome"] == OUTCOME_BREACH),
            "passes": sum(1 for r in external if r["outcome"] == OUTCOME_PASS),
            "unavailable": sum(1 for r in external
                               if r["outcome"] == OUTCOME_UNAVAILABLE),
        },
        "trakt_screening": {
            "evaluated": len(internal),
            "flags": sum(1 for r in internal if r["outcome"] == OUTCOME_FLAG),
            "clear": sum(1 for r in internal if r["outcome"] == OUTCOME_CLEAR),
            "unavailable": sum(1 for r in internal
                               if r["outcome"] == OUTCOME_UNAVAILABLE),
        },
        "note": ("Screening flags are Trakt's internal attention thresholds and "
                 "are NOT breaches of any external requirement. External "
                 "breaches and screening flags are counted separately and must "
                 "not be added together."),
    }
