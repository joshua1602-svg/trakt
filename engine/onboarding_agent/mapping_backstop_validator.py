"""
mapping_backstop_validator.py
=============================

PART 6 — deterministic backstop validation of proposed mappings.

Every proposed mapping (deterministic candidate OR LLM proposal) must pass these
deterministic checks before it can become active. The LLM can never finalise a
mapping; this validator is the gate. Auto-approval is deliberately conservative.

Artefacts:
    32_mapping_backstop_validation.csv / .json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pipeline_field_contract import _FIELD_META as _PIPE_META

# Statuses (PART 6).
AUTO_APPROVED = "auto_approved_candidate"
REVIEW_REQUIRED = "review_required"
BLOCKED = "blocked"
REGISTRY_TARGET_MISSING = "registry_target_missing"
PIPELINE_TARGET_MISSING = "pipeline_contract_target_missing"
UNSAFE = "unsafe_mapping"
OUT_OF_SCOPE = "out_of_scope"
CONFLICTS_MEMORY = "conflicts_with_memory"
CONFLICTS_MAPPING = "conflicts_with_existing_mapping"


def _norm_conf(value: Any) -> str:
    if isinstance(value, (int, float)):
        return "high" if value >= 0.95 else ("medium" if value >= 0.8 else "low")
    v = str(value or "").strip().lower()
    return v if v in ("high", "medium", "low", "no_match") else "low"


def _is_material(target: str, registry_fields: Dict[str, Any]) -> bool:
    meta = registry_fields.get(target, {}) or {}
    return meta.get("category") == "regulatory" or meta.get("core_canonical") is True


def _type_compatible_proposal(prop: Dict[str, Any]) -> bool:
    # Shortlist proposals carry type_compatible; LLM proposals default to True
    # unless an explicit incompatibility flag is present.
    if "type_compatible" in prop:
        return bool(prop["type_compatible"])
    return "type_incompatible" not in (prop.get("validation_risks") or [])


def validate_mappings(
    proposals: List[Dict[str, Any]],
    registry_fields: Optional[Dict[str, Any]] = None,
    field_scope: Any = None,
    memory_store: Any = None,
    evidence_by_col: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Validate proposed mappings. One row per proposal."""
    registry_fields = registry_fields or {}
    evidence_by_col = evidence_by_col or {}
    included = getattr(field_scope, "included_fields", set()) or set()

    # Memory target lookup (client memory says column X -> field Y).
    memory_targets: Dict[str, str] = {}
    if memory_store is not None:
        from .mapping_memory import DECISION_MAPPING_OVERRIDE
        for e in memory_store.by_type(DECISION_MAPPING_OVERRIDE):
            memory_targets[e.normalized_source_column or e.source_column] = e.canonical_field

    # Detect two source columns assigned the same target.
    target_assignees: Dict[str, List[str]] = {}
    for p in proposals:
        tgt = p.get("proposed_target_field") or p.get("candidate_target_field") or ""
        if tgt:
            target_assignees.setdefault(tgt, []).append(p["source_column"])

    rows: List[Dict[str, Any]] = []
    for p in proposals:
        col = p["source_column"]
        target = p.get("proposed_target_field") or p.get("candidate_target_field") or ""
        source = p.get("proposed_target_source") or p.get("candidate_source") or "unknown"
        conf = _norm_conf(p.get("confidence", p.get("candidate_confidence", 0)))
        is_pipeline = bool(p.get("is_pipeline_field")) or target in _PIPE_META
        reasons: List[str] = []

        target_exists = target in registry_fields
        proposed_new = source == "proposed_new_field" or bool(p.get("proposed_new_field"))

        # 1. missing target.
        if not target:
            status = (PIPELINE_TARGET_MISSING if source in ("pipeline_contract", "value_profile")
                      and is_pipeline else REGISTRY_TARGET_MISSING)
            reasons.append("no canonical target")
            rows.append(_row(p, col, target, source, conf, status, False, reasons, is_pipeline))
            continue
        if not target_exists and not proposed_new and not is_pipeline:
            status = REGISTRY_TARGET_MISSING
            reasons.append("target not in registry and not a proposed_new_field")
            rows.append(_row(p, col, target, source, conf, status, False, reasons, is_pipeline))
            continue

        # 2. field scope.
        out_of_scope = (field_scope is not None and target_exists
                        and getattr(field_scope, "is_excluded", None)
                        and field_scope.is_excluded(target))
        if out_of_scope:
            reasons.append(f"target out of scope for mode {getattr(field_scope,'mode_name','')}")
            rows.append(_row(p, col, target, source, conf, OUT_OF_SCOPE, False, reasons, is_pipeline))
            continue

        # 3. type / value-profile compatibility.
        if not _type_compatible_proposal(p):
            reasons.append("type/value-profile incompatible")
            rows.append(_row(p, col, target, source, conf, UNSAFE, False, reasons, is_pipeline))
            continue

        # 4. conflict with client memory.
        from .mapping_memory import normalize_column
        mem_target = memory_targets.get(normalize_column(col))
        if mem_target and mem_target != target:
            reasons.append(f"client memory maps this column to '{mem_target}'")
            rows.append(_row(p, col, target, source, conf, CONFLICTS_MEMORY, False, reasons, is_pipeline))
            continue

        # 5. conflict with another source column mapped to same target.
        assignees = [a for a in target_assignees.get(target, []) if a != col]
        if assignees:
            reasons.append(f"target also claimed by: {', '.join(sorted(set(assignees)))}")
            rows.append(_row(p, col, target, source, conf, CONFLICTS_MAPPING, False, reasons, is_pipeline))
            continue

        # 6. materiality + ambiguity -> always review.
        material = _is_material(target, registry_fields)
        ambiguous = bool(p.get("ambiguity_flags")) or conf in ("low", "no_match")
        competitors = len(set(p.get("alternative_targets") or []))

        # Material regulatory/economic fields never auto-approve from a fuzzy
        # semantic-alignment or LLM source — only from deterministic exact
        # sources (client memory / alias).
        material_ok = (not material) or source in ("client_memory", "alias")
        auto_ok = (
            conf == "high" and target and (target_exists or is_pipeline)
            and not out_of_scope and _type_compatible_proposal(p)
            and competitors == 0 and not (material and ambiguous) and material_ok
            and not mem_target and not assignees
            and source in ("client_memory", "alias", "pipeline_contract", "semantic_alignment")
        )
        if auto_ok:
            status = AUTO_APPROVED
        else:
            status = REVIEW_REQUIRED
            if material:
                reasons.append("material regulatory/economic field — needs approval")
            if conf != "high":
                reasons.append(f"confidence={conf}")
            if source == "llm_suggested":
                reasons.append("LLM-only suggestion needs deterministic + user approval")
        rows.append(_row(p, col, target, source, conf, status, status == AUTO_APPROVED,
                         reasons, is_pipeline))
    return rows


def _row(p, col, target, source, conf, status, approvable, reasons, is_pipeline):
    return {
        "source_file": p.get("source_file", ""),
        "source_column": col,
        "proposed_target_field": target,
        "candidate_source": source,
        "confidence": conf,
        "validation_status": status,
        "auto_approvable": bool(approvable),
        "requires_user_approval": not bool(approvable),
        "is_pipeline_field": bool(is_pipeline),
        "validation_reasons": "; ".join(reasons),
    }


_VAL_COLUMNS = [
    "source_file", "source_column", "proposed_target_field", "candidate_source",
    "confidence", "validation_status", "auto_approvable", "requires_user_approval",
    "is_pipeline_field", "validation_reasons",
]


def write_validation_artifacts(rows: List[Dict[str, Any]], output_dir: str | Path) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "32_mapping_backstop_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_VAL_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _VAL_COLUMNS})
    json_path = out_dir / "32_mapping_backstop_validation.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}
