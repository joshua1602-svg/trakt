"""
promotion_planner.py
====================

PART 10 — Azure-ready, DRY-RUN handoff planner.

Builds a review-first handoff plan from approved onboarding artefacts and the
consolidated central tapes. It writes manifests that a future Azure Blob /
Event-Grid trigger can consume, but performs NO live Azure orchestration, NO
SDK upload, and never runs Gates 1–5.

Outputs (under output/manifests/)::
    19_promotion_plan.yaml
    20_pipeline_handoff_manifest.yaml
    21_pipeline_handoff_readiness.json
    23_pipeline_trigger.json

When ``storage_backend == azure_blob_compatible`` and Azure-style URIs are
supplied, every manifest path is mirrored with a blob-compatible URI; files are
still only written locally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import domain_coverage as dc

# Where promotion would hand off to (downstream Trakt pipeline entrypoint).
_TARGET_ENTRYPOINT = "engine.orchestrator.run_pipeline"


def _exists(path: Path) -> bool:
    return Path(path).exists()


def _ref(run_paths: Any, path: Optional[str]) -> Dict[str, Optional[str]]:
    if not path or not _exists(Path(path)):
        return {"path": None, "uri": None}
    return {
        "path": run_paths.to_manifest_path(path),
        "uri": run_paths.to_manifest_uri(path),
    }


def _readiness(
    mode: str,
    regulatory_reporting_enabled: bool,
    tape_result: Dict[str, Any],
    coverage: List[Any],
    approved_config_present: bool,
) -> Dict[str, Any]:
    cov_by_domain = {c.domain: c for c in coverage}

    def covered(domain: str) -> bool:
        c = cov_by_domain.get(domain)
        return bool(c and c.status == dc.COVERED)

    def not_missing(domain: str) -> bool:
        c = cov_by_domain.get(domain)
        return bool(c and c.status in (dc.COVERED, dc.PARTIAL))

    lender_created = bool(tape_result.get("central_lender_tape_created"))
    loan_count = int(tape_result.get("loan_count", 0) or 0)
    conflicts = int(tape_result.get("conflict_count", 0) or 0)

    summary = tape_result.get("lender_summary", {}) or {}
    gaps = summary.get("gap_count", 0)

    blocking_items: List[str] = []
    warnings: List[str] = []

    # Blocking domains missing.
    for c in coverage:
        if c.blocking and c.status == dc.MISSING:
            blocking_items.append(f"{dc.DOMAIN_LABELS.get(c.domain, c.domain)} domain missing")
        elif c.status == dc.PARTIAL:
            warnings.append(f"{dc.DOMAIN_LABELS.get(c.domain, c.domain)} domain partially covered")
    if conflicts:
        blocking_items.append(f"{conflicts} unresolved value conflict(s) in central tape")
    if not lender_created or loan_count == 0:
        blocking_items.append("central lender tape is empty")

    ready_for_mi_agent = lender_created and loan_count > 0 and not blocking_items
    ready_for_gate1 = ready_for_mi_agent and approved_config_present
    ready_for_regulatory = (
        ready_for_gate1
        and (mode == "regulatory_mi" or regulatory_reporting_enabled)
        and not_missing(dc.COLLATERAL)
    )
    ready_for_warehouse = (
        lender_created and loan_count > 0
        and mode == "warehouse_securitisation"
        and not_missing(dc.CASHFLOW) and not_missing(dc.WAREHOUSE)
        and not blocking_items
    )

    return {
        "central_lender_tape_created": lender_created,
        "central_pipeline_tape_created": bool(tape_result.get("central_pipeline_tape_created")),
        "loan_count": loan_count,
        "pipeline_count": int(tape_result.get("pipeline_count", 0) or 0),
        "mapped_field_count": int(tape_result.get("mapped_field_count", 0) or 0),
        "domain_coverage": {c.domain: c.status for c in coverage},
        "unresolved_required_fields": gaps,
        "unresolved_conflicts": conflicts,
        "blocking_items": blocking_items,
        "warnings": warnings,
        "ready_for_mi_agent": ready_for_mi_agent,
        "ready_for_gate1_handoff": ready_for_gate1,
        "ready_for_regulatory_projection": ready_for_regulatory,
        "ready_for_warehouse_analysis": ready_for_warehouse,
    }


def build_promotion_plan(
    project_dir: str | Path,
    run_paths: Any,
    tape_result: Dict[str, Any],
    coverage: List[Any],
    mode: str,
    regulatory_reporting_enabled: bool,
    client_name: str = "",
    project_id: str = "",
) -> Dict[str, Any]:
    """Write the dry-run promotion plan, handoff manifest, readiness + trigger."""
    project_dir = Path(project_dir)
    manifests_dir = Path(run_paths.manifests_dir)
    run_paths.guard(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    client_id = run_paths.client_id or project_id or project_dir.name
    run_id = run_paths.run_id or "run"

    # Artefact references.
    approved_config = project_dir / "11_approved_config.yaml"
    mapping_overrides = project_dir / "12_approved_mapping_overrides.yaml"
    source_precedence = project_dir / "13_source_precedence_rules.yaml"
    enum_decisions = project_dir / "14_enum_review_decisions.yaml"
    lender_tape = tape_result.get("central_lender_tape_path", "")
    pipeline_tape = tape_result.get("central_pipeline_tape_path", "")
    lineage_path = tape_result.get("central_tape_lineage_path", "")
    gaps_path = tape_result.get("central_tape_gaps_path", "")

    approved_config_present = _exists(approved_config)
    readiness = _readiness(
        mode, regulatory_reporting_enabled, tape_result, coverage, approved_config_present
    )
    # Separated readiness concepts (MI runtime vs governance vs XML delivery) so an
    # MI-only run is not reported as wholesale "blocked" by non-blocking confirmations.
    from . import readiness as _readiness_mod
    readiness_breakdown = _readiness_mod.compute_readiness_breakdown(
        project_dir, tape_result, coverage, mode, regulatory_reporting_enabled)

    domains_detected = sorted({
        c.domain for c in coverage if c.status in (dc.COVERED, dc.PARTIAL)
    })
    out_of_scope_items = [c.domain for c in coverage if c.status == dc.OUT_OF_SCOPE]

    ready = not readiness["blocking_items"]
    readiness_status = "ready_for_pipeline" if ready else "blocked"

    cref = _ref(run_paths, str(approved_config) if approved_config_present else None)
    lref = _ref(run_paths, lender_tape)
    pref = _ref(run_paths, pipeline_tape)
    linref = _ref(run_paths, lineage_path)
    gref = _ref(run_paths, gaps_path)
    mref = _ref(run_paths, str(mapping_overrides) if _exists(mapping_overrides) else None)
    spref = _ref(run_paths, str(source_precedence) if _exists(source_precedence) else None)
    eref = _ref(run_paths, str(enum_decisions) if _exists(enum_decisions) else None)

    # ---- 19 promotion plan ----
    promotion_plan = {
        "_warning": "DRY-RUN promotion plan. Review-only. Does not run Gates 1–5 "
                    "and does not upload to Azure.",
        "project_id": project_id or client_id,
        "client_id": client_id,
        "client_name": client_name,
        "run_id": run_id,
        "onboarding_mode": mode,
        "storage_backend": run_paths.storage_backend,
        "input_uri": run_paths.input_uri,
        "output_uri": run_paths.output_uri,
        "domains_detected": domains_detected,
        "domain_coverage_status": {c.domain: c.status for c in coverage},
        "approved_config_path": cref["path"],
        "approved_config_uri": cref["uri"],
        "central_lender_tape_path": lref["path"],
        "central_lender_tape_uri": lref["uri"],
        "central_pipeline_tape_path": pref["path"],
        "central_pipeline_tape_uri": pref["uri"],
        "lineage_path": linref["path"],
        "lineage_uri": linref["uri"],
        "gaps_path": gref["path"],
        "gaps_uri": gref["uri"],
        "approved_mapping_overrides_path": mref["path"],
        "approved_mapping_overrides_uri": mref["uri"],
        "source_precedence_rules_path": spref["path"],
        "source_precedence_rules_uri": spref["uri"],
        "enum_review_decisions_path": eref["path"],
        "enum_review_decisions_uri": eref["uri"],
        "target_pipeline_entrypoint": _TARGET_ENTRYPOINT,
        "recommended_next_command": (
            "Review manifests under output/manifests, then (when approved) trigger "
            "the downstream Trakt pipeline with 23_pipeline_trigger.json."
        ),
        "readiness_status": readiness_status,
        "readiness_breakdown": readiness_breakdown,
        "blocking_items": readiness["blocking_items"],
        "warnings": readiness["warnings"],
        "out_of_scope_items": out_of_scope_items,
    }

    # ---- 20 handoff manifest ----
    handoff_manifest = {
        "client_id": client_id,
        "run_id": run_id,
        "input_tape": lref["path"],
        "input_tape_uri": lref["uri"],
        "pipeline_tape": pref["path"],
        "pipeline_tape_uri": pref["uri"],
        "client_config": cref["path"],
        "client_config_uri": cref["uri"],
        "mapping_overrides": mref["path"],
        "mapping_overrides_uri": mref["uri"],
        "source_precedence": spref["path"],
        "source_precedence_uri": spref["uri"],
        "enum_decisions": eref["path"],
        "enum_decisions_uri": eref["uri"],
        "mode": mode,
        "regulatory_reporting_enabled": regulatory_reporting_enabled,
        "run_gates": "none",
        "dry_run_only": True,
    }

    # ---- 21 readiness JSON ----
    readiness_json = {
        "client_id": client_id,
        "run_id": run_id,
        **readiness,
        "readiness_breakdown": readiness_breakdown,
    }

    # ---- 23 pipeline trigger JSON (Azure / Event-Grid friendly) ----
    if ready:
        trigger = {
            "event_type": "trakt.onboarding.handoff.ready",
            "client_id": client_id,
            "run_id": run_id,
            "status": "ready_for_pipeline",
            "storage_backend": run_paths.storage_backend,
            "mode": mode,
            "regulatory_reporting_enabled": regulatory_reporting_enabled,
            "central_lender_tape_uri": lref["uri"] or lref["path"],
            "central_pipeline_tape_uri": pref["uri"] or pref["path"],
            "handoff_manifest_uri": None,  # filled below
            "readiness_uri": None,         # filled below
            "ready_for_mi_agent": readiness["ready_for_mi_agent"],
            "ready_for_gate1_handoff": readiness["ready_for_gate1_handoff"],
            "ready_for_regulatory_projection": readiness["ready_for_regulatory_projection"],
            "ready_for_warehouse_analysis": readiness["ready_for_warehouse_analysis"],
            "blocking_items": readiness["blocking_items"],
            "warnings": readiness["warnings"],
        }
    else:
        trigger = {
            "event_type": "trakt.onboarding.handoff.blocked",
            "client_id": client_id,
            "run_id": run_id,
            "status": "blocked",
            "storage_backend": run_paths.storage_backend,
            "mode": mode,
            "blocking_items": readiness["blocking_items"],
            "warnings": readiness["warnings"],
        }

    # Write files.
    plan_path = manifests_dir / "19_promotion_plan.yaml"
    manifest_path = manifests_dir / "20_pipeline_handoff_manifest.yaml"
    readiness_path = manifests_dir / "21_pipeline_handoff_readiness.json"
    trigger_path = manifests_dir / "23_pipeline_trigger.json"

    # Now that manifest/readiness paths exist, fill their URIs in the trigger.
    trigger["handoff_manifest_uri"] = (
        run_paths.to_manifest_uri(manifest_path) or run_paths.to_manifest_path(manifest_path)
    )
    trigger["readiness_uri"] = (
        run_paths.to_manifest_uri(readiness_path) or run_paths.to_manifest_path(readiness_path)
    )

    plan_path.write_text(yaml.safe_dump(promotion_plan, sort_keys=False), encoding="utf-8")
    manifest_path.write_text(yaml.safe_dump(handoff_manifest, sort_keys=False), encoding="utf-8")
    readiness_path.write_text(json.dumps(readiness_json, indent=2, default=str), encoding="utf-8")
    trigger_path.write_text(json.dumps(trigger, indent=2, default=str), encoding="utf-8")

    return {
        "promotion_plan_path": str(plan_path),
        "handoff_manifest_path": str(manifest_path),
        "readiness_path": str(readiness_path),
        "pipeline_trigger_path": str(trigger_path),
        "readiness_status": readiness_status,
        "ready": ready,
        "readiness": readiness_json,
        "readiness_breakdown": readiness_breakdown,
        "trigger": trigger,
    }
