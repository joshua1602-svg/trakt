"""apps.blob_trigger_app.gate_diagnostics — generic per-gate observability.

One standard diagnostics object per orchestrator gate (onboarding, transform,
validation, stamp/provenance, assembler, MI projection), so a halt at ANY gate is
inspectable the same way — no more fixing observability one gate at a time.

Each gate object is JSON-safe and best-effort (never raises):

    gate_name, status, ready_flag_name, ready_flag_value, halt_reason,
    issue_count, blocking_issue_count, warning_count, issues[:20],
    affected_fields, severity_counts, source_artifact_paths,
    persisted_artifact_uris (filled by persistence), next_recommended_operator_action,
    payload (gate-specific rich readiness dict)

``collect_gates(state)`` returns them in execution order; ``build_run_summary``
reduces them to the run-level summary (failed gate, per-gate status, the specific
next operator action).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HALTED = "halted"
_FAILED = "failed"
_DONE = "done"
_PENDING = "pending"
_SKIPPED = "skipped"

#: Ordered gate → orchestrator step accessor. Portfolio gates read the first
#: portfolio's step; run-level gates read the RunState attribute.
_PORTFOLIO_GATES = (
    ("onboarding", "onboard"),
    ("transform", "transform"),
    ("validation", "validate"),
    ("stamp", "stamp"),
)
_RUN_GATES = (
    ("assembler", "assemble"),
    ("projection", "project"),
)

#: Gate → the specific operator action when that gate is the blocking one.
GATE_ACTION = {
    "onboarding": "inspect_onboarding",
    "transform": "inspect_transform",
    "validation": "inspect_validation",
    "stamp": "inspect_onboarding",      # provenance config — closest inspect surface
    "assembler": "inspect_assembler",
    "projection": "inspect_projection",
}

#: Severities treated as blocking across gates.
_BLOCKING_SEVERITIES = ("error", "blocking", "critical", "fatal", "fail", "failure")
_WARNING_SEVERITIES = ("warn", "warning")


# --------------------------------------------------------------------------- #
# Low-level readers (never raise)
# --------------------------------------------------------------------------- #

def read_json_maybe(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read a JSON artefact; if given a ``.csv`` path, try the ``.json`` sibling."""
    if not path:
        return None
    p = Path(path)
    for cand in (p, p.with_suffix(".json")):
        try:
            if cand.exists():
                return json.loads(cand.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    return None


def _rows_of(doc: Any) -> List[Dict[str, Any]]:
    if doc is None:
        return []
    if isinstance(doc, list):
        return [r for r in doc if isinstance(r, dict)]
    return [r for r in (doc.get("rows") or doc.get("issues") or []) if isinstance(r, dict)]


def normalise_issues(doc: Any) -> List[Dict[str, Any]]:
    """Normalise an issue document into ``[{target_field, issue_type, severity,
    message}]``. Severity is lower-cased; a ``blocking`` flag implies ``error``."""
    out: List[Dict[str, Any]] = []
    for r in _rows_of(doc):
        sev = r.get("severity") or ("error" if r.get("blocking") else "warning")
        out.append({
            "target_field": (r.get("target_field") or r.get("field")
                             or r.get("canonical_field") or ""),
            "issue_type": (r.get("issue_type") or r.get("type")
                           or r.get("validation_classification") or r.get("category") or ""),
            "severity": str(sev).lower(),
            "message": (r.get("message") or r.get("detail") or r.get("reason")
                        or r.get("recommended_action") or ""),
        })
    return out


def _severity_split(issues: List[Dict[str, Any]]) -> Tuple[int, int, Dict[str, int]]:
    blocking = warning = 0
    counts: Dict[str, int] = {}
    for i in issues:
        sev = i["severity"]
        counts[sev] = counts.get(sev, 0) + 1
        if sev in _BLOCKING_SEVERITIES:
            blocking += 1
        elif sev in _WARNING_SEVERITIES:
            warning += 1
    return blocking, warning, counts


def _reason(step: Any) -> str:
    return ("; ".join(getattr(step, "blockers", None) or [])
            or getattr(step, "message", "") or "")


# --------------------------------------------------------------------------- #
# Gate-specific rich payloads (embedded in the gate object; also reused by the
# specific show-handoff / show-transform / show-validation commands)
# --------------------------------------------------------------------------- #

def onboarding_payload(step: Any) -> Dict[str, Any]:
    """Onboarding handoff readiness — which readiness gate failed, blocking
    operator decisions, missing/unresolved fields, artefact paths."""
    m = read_json_maybe(getattr(step, "manifest_path", None))
    if not m:
        return {}
    rj = read_json_maybe(m.get("readiness_path")) or {}
    blocking_count = int(m.get("blocking_decision_count", rj.get("blocking_decision_count", 0)) or 0)
    gap = int(m.get("registry_gap_count", rj.get("registry_gap_count", 0)) or 0)
    central_present = bool(rj.get("central_tape_present",
                                  int(m.get("central_tape_row_count", 0) or 0) > 0))
    coverage_present = bool(rj.get("coverage_matrix_present", True))
    universe_loaded = bool(rj.get("target_universe_loaded", True))
    failed_gates: List[str] = []
    if not central_present:
        failed_gates.append("central_tape_present=false")
    if not coverage_present:
        failed_gates.append("coverage_matrix_present=false")
    if not universe_loaded:
        failed_gates.append("target_universe_loaded=false")
    if gap > 0:
        failed_gates.append(f"registry_gap_count={gap}")
    if blocking_count > 0:
        failed_gates.append(f"blocking_decision_count={blocking_count}")

    blocking_decisions, unresolved_decisions = _decision_queue(m.get("decision_queue_path"))
    missing_fields, unresolved_fields = _coverage_gaps(m.get("target_coverage_matrix_path"))
    return {
        "ready_for_transformation_validation": bool(m.get("ready_for_transformation_validation")),
        "ready_for_projection": bool(m.get("ready_for_projection")),
        "ready_for_xml_delivery": bool(m.get("ready_for_xml_delivery")),
        "failed_readiness_gates": failed_gates,
        "blocking_decision_count": blocking_count,
        "non_blocking_decision_count": int(m.get("non_blocking_decision_count", 0) or 0),
        "operator_decision_pending_count": int(m.get("operator_decision_pending_count", 0) or 0),
        "registry_gap_count": gap,
        "issue_count": blocking_count + gap + len(missing_fields),
        "blocking_decisions": blocking_decisions,
        "unresolved_decisions": unresolved_decisions,
        "unresolved_fields": sorted(set(unresolved_fields)),
        "missing_target_fields": sorted(set(missing_fields)),
        "source_absent_count": int(m.get("source_absent_count", 0) or 0),
        "target_field_count": int(m.get("target_field_count", 0) or 0),
        "source_mapped_count": int(m.get("source_mapped_count", 0) or 0),
        "handoff_manifest_path": getattr(step, "manifest_path", None),
        "readiness_path": m.get("readiness_path"),
        "target_coverage_matrix_path": m.get("target_coverage_matrix_path"),
        "decision_queue_path": m.get("decision_queue_path"),
        "field_contract_path": m.get("field_contract_path"),
        "handoff_manifest": m,
    }


def _decision_queue(dq_path: Optional[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    doc = read_json_maybe(dq_path) or {}
    blocking: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    for r in _rows_of(doc):
        entry = {"target_field": r.get("target_field", ""), "esma_code": r.get("esma_code", ""),
                 "reason": (r.get("operator_question") or r.get("question")
                            or r.get("decision_reason") or r.get("coverage_status") or ""),
                 "decision_status": r.get("decision_status")}
        if r.get("blocking"):
            blocking.append(entry)
        elif r.get("requires_user_decision") or not r.get("decision_status"):
            unresolved.append(entry)
    return blocking, unresolved


def _coverage_gaps(cov_path: Optional[str]) -> Tuple[List[str], List[str]]:
    doc = read_json_maybe(cov_path) or {}
    missing: List[str] = []
    unresolved: List[str] = []
    for r in _rows_of(doc):
        tf = r.get("target_field", "")
        if not tf:
            continue
        if r.get("coverage_status") == "missing_required":
            missing.append(tf)
        if r.get("requires_user_decision") or r.get("blocking"):
            unresolved.append(tf)
    return missing, unresolved


def transform_payload(step: Any) -> Dict[str, Any]:
    """Gate 2 transform readiness — readiness flags, issue tally, first 20 issues,
    affected fields, artefact paths."""
    m = read_json_maybe(getattr(step, "manifest_path", None))
    if not m:
        return {}
    issues = normalise_issues(read_json_maybe(m.get("transformation_issues_json")))
    blocking, warning, sev_counts = _severity_split(issues)
    return {
        "ready_for_validation": bool(m.get("ready_for_validation")),
        "ready_for_projection": bool(m.get("ready_for_projection")),
        "ready_for_xml_delivery": bool(m.get("ready_for_xml_delivery")),
        "issue_count": int(m.get("issue_count", len(issues)) or 0),
        "blocking_issue_count": blocking,
        "warning_count": warning,
        "issue_type_counts": m.get("issue_type_counts") or {},
        "severity_counts": sev_counts,
        "affected_fields": sorted({i["target_field"] for i in issues if i["target_field"]}),
        "issues": issues[:20],
        "blocking_issues": [i for i in issues if i["severity"] in _BLOCKING_SEVERITIES][:20],
        "transformation_manifest_path": getattr(step, "manifest_path", None),
        "transformed_canonical_path": getattr(step, "output_path", None),
        "transformation_issues_csv": m.get("transformation_issues_csv"),
        "transformation_issues_json": m.get("transformation_issues_json"),
        "transformation_manifest": m,
    }


def validation_payload(step: Any) -> Dict[str, Any]:
    """Gate 3 validation readiness — mandatory / type / numeric / date failures,
    first 20 issues, ready_for_publish flag, artefact paths."""
    m = read_json_maybe(getattr(step, "manifest_path", None))
    if not m:
        return {}
    issues = normalise_issues(read_json_maybe(m.get("output_validation_issues_path")))
    blocking, warning, sev_counts = _severity_split(issues)

    def _of_type(*types: str) -> List[Dict[str, Any]]:
        return [i for i in issues if any(t in i["issue_type"] for t in types)]

    ready = bool(m.get("ready_for_validation_complete"))
    return {
        "ready_for_validation_complete": ready,
        "ready_for_publish": ready,               # publish is gated by validation
        "ready_for_projection": bool(m.get("ready_for_projection")),
        "issue_count": int(m.get("issue_count", len(issues)) or 0),
        "blocking_issue_count": int(m.get("blocking_for_validation_count", blocking) or 0),
        "warning_count": int(m.get("validation_warning_count", warning) or 0),
        "validation_failure_count": int(m.get("validation_failure_count", 0) or 0),
        "issue_type_counts": m.get("issue_type_counts") or {},
        "severity_counts": sev_counts,
        "mandatory_field_failures": [i["target_field"] for i in _of_type("mandatory", "missing")],
        "type_failures": [i["target_field"] for i in _of_type("type")],
        "numeric_parse_failures": [i["target_field"] for i in _of_type("numeric_parse")],
        "date_parse_failures": [i["target_field"] for i in _of_type("date_parse")],
        "affected_fields": sorted({i["target_field"] for i in issues if i["target_field"]}),
        "issues": issues[:20],
        "blocking_issues": [i for i in issues if i["severity"] in _BLOCKING_SEVERITIES][:20],
        "validation_manifest_path": getattr(step, "manifest_path", None),
        "validation_issues_csv": m.get("output_validation_issues_path"),
        "validation_readiness_path": m.get("output_validation_readiness_path"),
        "validation_manifest": m,
    }


_GATE_PAYLOAD = {
    "onboarding": (onboarding_payload, "ready_for_transformation_validation"),
    "transform": (transform_payload, "ready_for_validation"),
    "validation": (validation_payload, "ready_for_validation_complete"),
}


# --------------------------------------------------------------------------- #
# Standard gate object + collection
# --------------------------------------------------------------------------- #

def _gate_object(gate_name: str, step: Any) -> Dict[str, Any]:
    status = getattr(step, "status", _PENDING) or _PENDING
    extractor = _GATE_PAYLOAD.get(gate_name)
    payload: Dict[str, Any] = {}
    if extractor is not None:
        try:
            payload = extractor[0](step) or {}
        except Exception:  # noqa: BLE001 — diagnostics never raise
            payload = {}
    ready_name = extractor[1] if extractor else None
    ready_value = payload.get(ready_name) if ready_name else None

    issues = payload.get("issues") or []
    issue_count = payload.get("issue_count", len(issues))
    blocking_issue_count = payload.get("blocking_issue_count",
                                       payload.get("blocking_decision_count", 0))
    warning_count = payload.get("warning_count", 0)
    affected = payload.get("affected_fields") or payload.get("missing_target_fields") or []
    severity_counts = payload.get("severity_counts") or {}

    # source artefact paths (scratch) surfaced from the payload.
    src_paths = {k: v for k, v in payload.items()
                 if k.endswith("_path") and v}

    return {
        "gate_name": gate_name,
        "status": status,
        "ready_flag_name": ready_name,
        "ready_flag_value": ready_value,
        "halt_reason": _reason(step) if status in (_HALTED, _FAILED) else "",
        "issue_count": int(issue_count or 0),
        "blocking_issue_count": int(blocking_issue_count or 0),
        "warning_count": int(warning_count or 0),
        "issues": issues[:20],
        "affected_fields": affected,
        "severity_counts": severity_counts,
        "source_artifact_paths": src_paths,
        "persisted_artifact_uris": {},          # filled by persistence
        "next_recommended_operator_action": GATE_ACTION.get(gate_name, "rerun"),
        "payload": payload,
    }


def collect_gates(state: Any) -> List[Dict[str, Any]]:
    """Standard diagnostics for every gate, in execution order."""
    gates: List[Dict[str, Any]] = []
    portfolios = getattr(state, "portfolios", []) or []
    p0 = portfolios[0] if portfolios else None
    for gate_name, step_name in _PORTFOLIO_GATES:
        if p0 is None:
            continue
        try:
            step = p0.step(step_name)
        except Exception:  # noqa: BLE001
            continue
        gates.append(_gate_object(gate_name, step))
    for gate_name, attr in _RUN_GATES:
        step = getattr(state, attr, None)
        if step is not None:
            gates.append(_gate_object(gate_name, step))
    return gates


def first_failed_gate(gates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for g in gates:
        if g["status"] in (_HALTED, _FAILED):
            return g
    return None


def build_run_summary(state: Any, gates: List[Dict[str, Any]],
                      central_canonical_path: Optional[str]) -> Dict[str, Any]:
    """Run-level summary: the failed gate, per-gate status, central canonical
    availability, and the specific next operator action for the failed gate."""
    failed = first_failed_gate(gates)
    # A transform/validation step that halted at its GUARD (it never ran, so it has
    # no readiness payload) because the upstream onboarding handoff is not ready →
    # attribute the failure to ONBOARDING, which carries the real blocking decisions.
    if failed and failed["gate_name"] in ("transform", "validation") and not failed["payload"]:
        onb = next((g for g in gates if g["gate_name"] == "onboarding"), None)
        if onb and (onb.get("payload") or {}).get("ready_for_transformation_validation") is False:
            onb = {**onb, "halt_reason": failed["halt_reason"] or onb.get("halt_reason", "")}
            failed = onb
    gate_status = {g["gate_name"]: g["status"] for g in gates}
    summary: Dict[str, Any] = {
        "failed_gate": failed["gate_name"] if failed else None,
        "failed_gate_status": failed["status"] if failed else None,
        "gate_status": gate_status,
        "central_canonical_path": central_canonical_path,
        "central_canonical_unavailable_reason": None,
        "next_action_key": failed["next_recommended_operator_action"] if failed else None,
    }
    if not central_canonical_path:
        if failed:
            summary["central_canonical_unavailable_reason"] = (
                f"orchestrator halted at gate '{failed['gate_name']}' "
                f"({failed['status']}) before the central canonical was assembled"
                + (f": {failed['halt_reason']}" if failed["halt_reason"] else ""))
        else:
            summary["central_canonical_unavailable_reason"] = (
                "central canonical was not produced")
    return summary
