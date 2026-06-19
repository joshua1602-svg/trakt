"""
capability_readiness.py
=======================

Capability-based onboarding readiness.

Base MI must not depend on every possible MI / risk / regulatory field. This
module emits readiness **per capability** so a run can be promoted for the
capabilities it actually supports, while missing optional fields stay visible as
non-blocking gaps.

Capabilities (config-driven vocabulary, see ``config/asset/product_profiles.yaml``):
    base_mi, pipeline_mi, risk_migration, risk_monitor, spv_segmentation,
    mna_segmentation, regulatory_reporting.

A capability is ``ready`` when every field in its contract is *satisfied* —
mapped from source, derived, defaulted, or marked not-applicable by the active
product profile — and none of its required fields is unresolved-missing. The
artefact-role distinction is honoured: ``pipeline_mi`` only activates when a
pipeline artefact is present, and ``regulatory_reporting`` is never relaxed by a
product profile (it keeps its own regime contract).

Safety
------
* Risk capabilities (risk_migration / risk_monitor) are reported ``unavailable``
  when their risk fields are missing — so downstream must not run risk-migration
  queries. This is explicit, never silent.
* ``promotion_decision`` allows an ``mi_only`` run to promote when ``base_mi``
  (and ``pipeline_mi`` where a pipeline artefact exists) is ready, even if
  ``risk_migration`` is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Set

from .product_profile import ProductProfile

# Readiness states.
READY = "ready"
NOT_READY = "not_ready"
UNAVAILABLE = "unavailable"        # required inputs absent -> capability disabled
OUT_OF_SCOPE = "out_of_scope"      # capability not requested for this run


def _norm(s: Any) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", str(s or "").strip().lower()).strip("_")


@dataclass
class CapabilityResult:
    capability: str
    status: str
    ready: bool = False
    required_fields: List[str] = dc_field(default_factory=list)
    satisfied_fields: List[str] = dc_field(default_factory=list)
    missing_fields: List[str] = dc_field(default_factory=list)
    non_blocking_gaps: List[str] = dc_field(default_factory=list)
    rationale: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability,
            "status": self.status,
            "ready": self.ready,
            "required_fields": list(self.required_fields),
            "satisfied_fields": list(self.satisfied_fields),
            "missing_fields": list(self.missing_fields),
            "non_blocking_gaps": list(self.non_blocking_gaps),
            "rationale": self.rationale,
        }


def _equivalent_satisfied(
    group: Sequence[str], satisfied: Set[str]
) -> Optional[str]:
    for f in group:
        if _norm(f) in satisfied:
            return _norm(f)
    return None


def compute_capability_readiness(
    *,
    profile: ProductProfile,
    satisfied_fields: Sequence[str],
    artefact_roles: Optional[Sequence[str]] = None,
    requested_capabilities: Optional[Sequence[str]] = None,
) -> Dict[str, CapabilityResult]:
    """Compute readiness for each capability the profile defines a contract for.

    ``satisfied_fields`` are canonical fields that are mapped, derived, defaulted,
    or not-applicable (i.e. resolved, not blocking). ``artefact_roles`` is the set
    of file classifications present (e.g. ``current_loan_report``,
    ``pipeline_report``). ``requested_capabilities`` optionally restricts which
    capabilities are in scope (others are reported ``out_of_scope``).
    """
    satisfied: Set[str] = {_norm(f) for f in satisfied_fields}
    roles: Set[str] = {_norm(r) for r in (artefact_roles or [])}
    requested = {_norm(c) for c in requested_capabilities} if requested_capabilities else None

    results: Dict[str, CapabilityResult] = {}
    for cap, spec in profile.capability_fields.items():
        cap_norm = _norm(cap)
        required = [_norm(f) for f in (spec.get("required_fields", []) or [])]
        groups = [list(g) for g in (spec.get("equivalent_field_groups", []) or [])]
        role_required = spec.get("artefact_role_required")

        # Out-of-scope short circuit.
        if requested is not None and cap_norm not in requested:
            results[cap] = CapabilityResult(
                capability=cap, status=OUT_OF_SCOPE, ready=False,
                required_fields=required,
                rationale="Capability not requested for this run.")
            continue

        # Artefact-role gating (e.g. pipeline_mi needs a pipeline artefact).
        if role_required and _norm(role_required) not in roles:
            results[cap] = CapabilityResult(
                capability=cap, status=UNAVAILABLE, ready=False,
                required_fields=required,
                rationale=(f"Requires a '{role_required}' artefact, which is not "
                           f"present; capability disabled."))
            continue

        # Resolve which required fields are satisfied, honouring equivalence groups.
        grouped: Dict[str, List[str]] = {}
        for g in groups:
            for member in g:
                grouped.setdefault(_norm(member), [_norm(x) for x in g])

        sat: List[str] = []
        missing: List[str] = []
        for f in required:
            if f in grouped:
                hit = _equivalent_satisfied(grouped[f], satisfied)
                (sat if hit else missing).append(hit or f)
            elif f in satisfied:
                sat.append(f)
            else:
                missing.append(f)

        # Non-blocking excusal applies ONLY to base MI: the profile marks fields
        # not_applicable / derived / defaulted / optional *for base MI*. For every
        # other capability a missing required field is a genuine gap for THAT
        # capability (e.g. ifrs9_stage is non-blocking for base MI but is required
        # for risk_migration).
        if cap_norm == "base_mi":
            non_blocking = [f for f in missing if profile.is_non_blocking_for_base_mi(f)]
        else:
            non_blocking = []
        hard_missing = [f for f in missing if f not in non_blocking]

        is_risk = cap_norm in ("risk_migration", "risk_monitor")
        if not hard_missing:
            status, ready = READY, True
            rationale = "All required fields satisfied."
        elif is_risk:
            # Risk capabilities are *disabled* (not merely not-ready) when their
            # risk inputs are absent, so downstream cannot run risk queries.
            status, ready = UNAVAILABLE, False
            rationale = ("Risk fields missing: " + ", ".join(hard_missing) +
                         "; risk capability disabled (no risk-migration queries).")
        else:
            status, ready = NOT_READY, False
            rationale = "Missing required fields: " + ", ".join(hard_missing)

        results[cap] = CapabilityResult(
            capability=cap, status=status, ready=ready,
            required_fields=required, satisfied_fields=sat,
            missing_fields=hard_missing, non_blocking_gaps=non_blocking,
            rationale=rationale,
        )
    return results


def promotion_decision(
    readiness: Dict[str, CapabilityResult], *, mode: str = "mi_only"
) -> Dict[str, Any]:
    """Decide promotability from capability readiness.

    For ``mi_only`` a run is promotable when ``base_mi`` is ready and, where a
    pipeline artefact exists, ``pipeline_mi`` is ready — even if ``risk_migration``
    is unavailable. Missing optional capabilities stay visible, never block.
    """
    def status_of(cap: str) -> str:
        r = readiness.get(cap)
        return r.status if r else OUT_OF_SCOPE

    base_ready = status_of("base_mi") == READY
    pipeline_status = status_of("pipeline_mi")
    # pipeline_mi only blocks when it is in scope (a pipeline artefact present and
    # thus NOT_READY); UNAVAILABLE/OUT_OF_SCOPE never blocks promotion.
    pipeline_ok = pipeline_status in (READY, UNAVAILABLE, OUT_OF_SCOPE)

    promotable = base_ready and pipeline_ok
    if mode == "regulatory_mi":
        # Profiles never relax regulatory readiness; require it explicitly.
        promotable = promotable and status_of("regulatory_reporting") == READY

    available = [c for c, r in readiness.items() if r.ready]
    unavailable = [c for c, r in readiness.items() if r.status == UNAVAILABLE]
    blocking = []
    if not base_ready:
        blocking.append("base_mi")
    if mode != "regulatory_mi" and pipeline_status == NOT_READY:
        blocking.append("pipeline_mi")

    return {
        "mode": mode,
        "promotable": promotable,
        "ready_capabilities": available,
        "unavailable_capabilities": unavailable,
        "blocking_capabilities": blocking,
        "non_blocking_gaps": {
            c: r.non_blocking_gaps for c, r in readiness.items() if r.non_blocking_gaps
        },
        "rationale": (
            "Promotable: base_mi ready"
            + ("; pipeline_mi ready" if pipeline_status == READY else "")
            + ("; risk capabilities unavailable but non-blocking"
               if unavailable else "")
            if promotable else
            "Not promotable: " + ", ".join(blocking or ["unmet capability requirements"])
        ),
    }


def readiness_summary(readiness: Dict[str, CapabilityResult]) -> Dict[str, Any]:
    """Flat, artefact-friendly summary of capability readiness."""
    return {cap: r.as_dict() for cap, r in readiness.items()}
