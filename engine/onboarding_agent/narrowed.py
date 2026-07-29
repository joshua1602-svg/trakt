"""engine.onboarding_agent.narrowed — the narrowed Onboarding Agent contract.

ADDITIVE facade (no existing agent module is modified). The agent's
authoritative execution interface becomes:

    run_onboarding(input_files, effective_configuration, execution_context)
        -> OnboardingResult

The agent consumes ONE immutable effective configuration (hash-verified
snapshot files materialised by the Operations Control Centre) and owns only
deterministic execution: file/schema recognition, mapping, aliases, enums,
type conversion, validation, provenance and canonical outputs. It does NOT
resolve configuration layers, persist decisions, mutate shared rule stores,
infer decision scope, or interact with users — unresolved items are returned
as structured decision requests for the OCC.

A compatibility wrapper (:func:`run_onboarding_legacy`) resolves legacy
path-style calls into an ad-hoc effective configuration and emits a
deprecation warning. The authoritative path is OCC resolution ->
EffectiveConfiguration -> this facade.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings as _warnings
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

STATUS_READY = "ready"
STATUS_NEEDS_REVIEW = "needs_review"
STATUS_BLOCKED = "blocked"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class ExecutionContext:
    """Tenant-bound execution context supplied by the caller (the OCC)."""

    run_id: str
    client_id: str
    tenant_id: str
    output_root: str                       # run-scoped project directory
    portfolio_id: str = ""
    reporting_period: str = ""
    mode: str = "mi_only"
    # Materialised, hash-verified snapshot paths (from the OCC resolver).
    config_snapshot: Dict[str, str] = field(default_factory=dict)
    # Optional approved run decisions file (deterministic re-apply).
    approved_decisions_path: str = ""
    managed_service: bool = True


@dataclass
class DecisionRequest:
    """Structured decision request returned to the OCC (never acted on by
    the agent itself)."""

    decision_id: str
    category: str          # unmapped_source_field | unknown_enum | ambiguous_schema | ...
    severity: str          # blocking | warning | info
    field: str = ""
    source_file: str = ""
    observed_values: List[str] = dataclasses.field(default_factory=list)
    affected_record_count: int = 0
    suggested_resolution: str = ""
    alternatives: List[str] = dataclasses.field(default_factory=list)
    confidence: Optional[float] = None
    evidence: Dict[str, Any] = dataclasses.field(default_factory=dict)
    allowed_scopes: List[str] = dataclasses.field(
        default_factory=lambda: ["file", "run", "portfolio", "client"])
    blocking: bool = False
    plain_english_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class OnboardingResult:
    run_id: str
    status: str
    input_files: List[str] = field(default_factory=list)
    recognised_file_types: List[Dict[str, Any]] = field(default_factory=list)
    recognised_schemas: List[Dict[str, Any]] = field(default_factory=list)
    mapping_summary: Dict[str, Any] = field(default_factory=dict)
    alias_summary: Dict[str, Any] = field(default_factory=dict)
    enum_summary: Dict[str, Any] = field(default_factory=dict)
    canonical_output_reference: str = ""
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    provenance_reference: str = ""
    warnings: List[str] = field(default_factory=list)
    blocking_decisions: List[DecisionRequest] = field(default_factory=list)
    suggested_decisions: List[DecisionRequest] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    audit_reference: str = ""
    effective_config_hash: str = ""
    handoff_manifest_reference: str = ""


class ImmutableConfigError(Exception):
    """A snapshot file does not match the effective configuration's pinned
    hash — the configuration mutated after run start."""


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_snapshot(effective_configuration: Any,
                     snapshot: Dict[str, str]) -> None:
    """Verify every pinned layer file against the materialised snapshot."""
    layer_files = getattr(effective_configuration, "layer_files", None) or {}
    root = snapshot.get("snapshot_root")
    if not root or not layer_files:
        return
    root_p = Path(root)
    for layer, files in layer_files.items():
        for rel, digest in files.items():
            p = root_p / rel
            if not p.exists():
                raise ImmutableConfigError(f"missing snapshot file: {rel}")
            actual = "sha256:" + hashlib.sha256(
                p.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            if actual != digest:
                raise ImmutableConfigError(
                    f"snapshot file mutated after resolution: {rel}")


def _stage_inputs(input_files: Union[str, Sequence[str]],
                  output_root: Path) -> Path:
    """Accept a directory or a list of file references; list inputs are staged
    into a run-local input directory (the agent never mutates sources)."""
    if isinstance(input_files, (str, Path)):
        p = Path(input_files)
        if p.is_dir():
            return p
        input_files = [str(p)]
    dest = output_root / "input_files"
    dest.mkdir(parents=True, exist_ok=True)
    for f in input_files:
        shutil.copyfile(str(f), dest / Path(str(f)).name)
    return dest


def run_onboarding(input_files: Union[str, Sequence[str]],
                   effective_configuration: Any,
                   execution_context: ExecutionContext) -> OnboardingResult:
    """Deterministic canonicalisation against one immutable configuration."""
    from engine.onboarding_agent import workflow as _wf

    ctx = execution_context
    out_root = Path(ctx.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    snapshot = dict(ctx.config_snapshot or {})
    _verify_snapshot(effective_configuration, snapshot)

    registry = snapshot.get("registry", "config/system/fields_registry.yaml")
    aliases_dir = snapshot.get("aliases_dir", "config/system")
    input_dir = _stage_inputs(input_files, out_root)

    _wf.run_operator_workflow(
        input_dir=str(input_dir),
        client_name=ctx.client_id, client_id=ctx.client_id,
        run_id=ctx.run_id or "run", project_dir=str(out_root),
        mode=ctx.mode, registry=registry, aliases_dir=aliases_dir,
        enable_mapping_review=True,
        reporting_date=ctx.reporting_period,
        reporting_period=ctx.reporting_period,
        managed_service=ctx.managed_service,
        target_first_decisions=ctx.approved_decisions_path or "")

    return _collect_result(out_root, ctx, effective_configuration)


def run_onboarding_legacy(*, input_dir: str, client_id: str,
                          registry: str = "config/system/fields_registry.yaml",
                          aliases_dir: str = "config/system",
                          output_root: str = "",
                          reporting_period: str = "",
                          mode: str = "mi_only") -> OnboardingResult:
    """DEPRECATED compatibility wrapper: resolves a legacy path-style call
    into an ad-hoc (unpinned) effective configuration and invokes the
    narrowed interface. New callers must use OCC resolution ->
    EffectiveConfiguration -> :func:`run_onboarding`."""
    _warnings.warn(
        "run_onboarding_legacy is a compatibility path; migrate to the OCC "
        "EffectiveConfiguration flow (run_onboarding).",
        DeprecationWarning, stacklevel=2)

    class _AdHoc:                       # unpinned: hash checks are skipped
        layer_files: Dict[str, Dict[str, str]] = {}
        content_hash = "unpinned-legacy-call"

    ctx = ExecutionContext(
        run_id="run", client_id=client_id, tenant_id=client_id,
        output_root=output_root or f".ops_state/legacy/{client_id}",
        reporting_period=reporting_period, mode=mode,
        config_snapshot={"registry": registry, "aliases_dir": aliases_dir})
    return run_onboarding(input_dir, _AdHoc(), ctx)


# --------------------------------------------------------------------------- #
# Result collection (read-only over the run's own artefacts)
# --------------------------------------------------------------------------- #

def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _first(root: Path, name: str) -> Optional[Path]:
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


_CATEGORY_BY_TYPE = {
    "missing_required_target": "unmapped_source_field",
    "mapping_gap": "unmapped_source_field",
    "mapping_conflict": "conflicting_field_mapping",
    "enum_gap": "unknown_enum",
    "type_mismatch": "invalid_type",
}


def _collect_result(out_root: Path, ctx: ExecutionContext,
                    effective_configuration: Any) -> OnboardingResult:
    summary = _read_json(_first(out_root, "40_operator_workflow_summary.json")
                         or Path("/nonexistent"))
    status_map = {"READY": STATUS_READY,
                  "NEEDS_CONFIRMATION": STATUS_NEEDS_REVIEW,
                  "NEEDS_CONFIGURATION": STATUS_BLOCKED,
                  "BLOCKED": STATUS_BLOCKED, "FAILED": STATUS_FAILED}
    status = status_map.get(str(summary.get("status", "")).upper(),
                            STATUS_NEEDS_REVIEW if summary else STATUS_FAILED)

    # File / schema recognition evidence from the run's own artefacts.
    file_types: List[Dict[str, Any]] = []
    classify = _first(out_root, "02_file_classification.json") or \
        _first(out_root, "01_file_profile.json")
    if classify:
        doc = _read_json(classify)
        entries = doc if isinstance(doc, list) else \
            doc.get("files") or doc.get("classifications") or []
        for e in entries if isinstance(entries, list) else []:
            if isinstance(e, dict):
                file_types.append({
                    "file": e.get("file") or e.get("name", ""),
                    "classification": e.get("classification")
                    or e.get("role", ""),
                    "confidence": e.get("confidence"),
                    "ambiguous": bool(e.get("ambiguous"))})

    # Decision requests (structured; the agent never acts on them).
    blocking: List[DecisionRequest] = []
    suggested: List[DecisionRequest] = []
    decisions_file = _first(out_root, "34_target_first_decisions.yaml")
    if decisions_file:
        doc = yaml.safe_load(decisions_file.read_text(encoding="utf-8")) or {}
        for d in doc.get("decisions") or []:
            if not isinstance(d, dict) or \
                    str(d.get("status", "pending")).lower() == "approved":
                continue
            req = DecisionRequest(
                decision_id=str(d.get("decision_id", "")),
                category=_CATEGORY_BY_TYPE.get(
                    str(d.get("decision_type", "")), "unmapped_source_field"),
                severity="blocking" if d.get("blocking", True) else "warning",
                field=str(d.get("target_field") or ""),
                source_file=str(d.get("source_file") or ""),
                suggested_resolution=str(d.get("recommended_action") or ""),
                alternatives=[str(a) for a in (d.get("options") or [])],
                evidence={"issue": d.get("issue", ""),
                          "detail": d.get("evidence_summary", "")},
                blocking=bool(d.get("blocking", True)),
                plain_english_message=str(d.get("operator_question") or ""))
            (blocking if req.blocking else suggested).append(req)
    queue = _first(out_root, "33_mapping_review_queue.json")
    if queue:
        q = _read_json(queue)
        for group, cat in (("needs_user_decision_risky_conflicting",
                            "conflicting_field_mapping"),
                           ("missing_target_propose_schema_extension",
                            "unmapped_source_field")):
            for item in q.get(group) or []:
                if not isinstance(item, dict):
                    continue
                req = DecisionRequest(
                    decision_id=f"map_{item.get('source_column', '')}",
                    category=cat, severity="blocking",
                    field=str(item.get("canonical_field") or ""),
                    source_file=str(item.get("source_file") or ""),
                    observed_values=[str(v) for v in
                                     (item.get("sample_values") or [])][:5],
                    suggested_resolution=str(item.get("canonical_field")
                                             or ""),
                    alternatives=[str(a.get("canonical_field")
                                      if isinstance(a, dict) else a)
                                  for a in (item.get("candidates") or [])][:5],
                    confidence=item.get("confidence"),
                    blocking=True,
                    plain_english_message=f"The file column "
                    f"'{item.get('source_column')}' needs confirmation.")
                blocking.append(req)

    tape = _first(out_root, "18_central_lender_tape.csv")
    handoff = _first(out_root, "24_onboarding_handoff_manifest.json")
    lineage = _first(out_root, "18b_central_tape_lineage.csv") or \
        _first(out_root, "field_lineage.json")

    return OnboardingResult(
        run_id=ctx.run_id, status=status,
        input_files=sorted(str(p) for p in
                           Path(_stage_dir_of(out_root)).glob("*")
                           ) if _stage_dir_of(out_root) else [],
        recognised_file_types=file_types,
        recognised_schemas=[],
        mapping_summary={k: summary.get(k) for k in summary
                         if "mapping" in k or "coverage" in k},
        alias_summary={"aliases_source": "effective configuration snapshot"},
        enum_summary={k: summary.get(k) for k in summary if "enum" in k},
        canonical_output_reference=str(tape) if tape else "",
        validation_summary={k: summary.get(k) for k in summary
                            if "validation" in k or "blocking" in k},
        provenance_reference=str(lineage) if lineage else "",
        warnings=[],
        blocking_decisions=blocking, suggested_decisions=suggested,
        metrics={"summary_status": summary.get("status", "")},
        audit_reference=str(_first(out_root,
                                   "40_operator_workflow_summary.json") or ""),
        effective_config_hash=str(getattr(effective_configuration,
                                          "content_hash", "")),
        handoff_manifest_reference=str(handoff) if handoff else "")


def _stage_dir_of(out_root: Path) -> Optional[Path]:
    d = out_root / "input_files"
    return d if d.exists() else None
