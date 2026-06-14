"""
mapping_review_queue.py
=======================

PART 7 — build a CONCISE, prioritised review queue from validation results.

The risk is overwhelming the user with a wall of mapping questions. This module
turns validation rows into compact review cards, grouped into a handful of
queues, prioritised so material/high-impact items surface first and noisy
low-value columns are collapsed.

Artefacts:
    33_mapping_review_queue.csv / .json
    34_mapping_review_decisions.yaml   (decision template)
    35_mapping_review_action_log.json
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .mapping_backstop_validator import (
    AUTO_APPROVED, BLOCKED, CONFLICTS_MAPPING, CONFLICTS_MEMORY, OUT_OF_SCOPE,
    PIPELINE_TARGET_MISSING, REGISTRY_TARGET_MISSING, REVIEW_REQUIRED, UNSAFE,
)

# Review groups (PART 7).
G_APPROVALS = "high_confidence_approvals"
G_DECISIONS = "needs_your_decision"
G_MISSING = "missing_trakt_fields"
G_CONFLICTS = "conflicts_or_risky"
G_IGNORED = "ignored_or_out_of_scope"

_STATUS_GROUP = {
    AUTO_APPROVED: G_APPROVALS,
    REVIEW_REQUIRED: G_DECISIONS,
    REGISTRY_TARGET_MISSING: G_MISSING,
    PIPELINE_TARGET_MISSING: G_MISSING,
    BLOCKED: G_CONFLICTS,
    UNSAFE: G_CONFLICTS,
    CONFLICTS_MEMORY: G_CONFLICTS,
    CONFLICTS_MAPPING: G_CONFLICTS,
    OUT_OF_SCOPE: G_IGNORED,
}

# High-priority evidence types / concepts surface first.
_HIGH_PRIORITY_TYPES = {"identifier", "date", "amount", "rate", "percentage", "enum"}
_HIGH_PRIORITY_TOKENS = ("id", "date", "amount", "loan", "facility", "rate", "status",
                          "stage", "region", "broker", "product", "value", "balance")
_LOW_PRIORITY_TOKENS = ("detail", "comment", "note", "rejection reason", "dob",
                         "gender", "youngest")

_DECISION_OPTIONS = ["approve", "choose_different_field", "mark_not_needed",
                     "create_new_field", "ask_later"]


def _priority(col: str, etype: str, status: str) -> int:
    """Lower number = higher priority (surfaces first)."""
    low = col.lower()
    if status in (OUT_OF_SCOPE,):
        return 9
    if any(t in low for t in _LOW_PRIORITY_TOKENS):
        return 7
    if etype in _HIGH_PRIORITY_TYPES or any(t in low for t in _HIGH_PRIORITY_TOKENS):
        return 1
    return 4


def _evidence_summary(ev: Dict[str, Any]) -> str:
    bits = [f"{ev.get('data_type_guess','?')} values"]
    if ev.get("chronology_relationships"):
        bits.append("chronology: " + ev["chronology_relationships"].split(";")[0])
    if ev.get("amount_relationships"):
        bits.append("amounts: " + ev["amount_relationships"].split(";")[0])
    if ev.get("null_rate"):
        bits.append(f"{ev['null_rate']:.0%} null")
    if ev.get("candidate_value_profile_matches"):
        bits.append(ev["candidate_value_profile_matches"])
    return "; ".join(bits)


def _ekey(row: Dict[str, Any]):
    return (row.get("source_file", ""), row.get("source_sheet", ""),
            row.get("source_column", ""))


def build_review_queue(
    validation_rows: List[Dict[str, Any]],
    evidence_by_key: Optional[Dict[Any, Dict[str, Any]]] = None,
    llm_by_key: Optional[Dict[Any, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the grouped, prioritised, MULTI-FILE review queue + a top summary.

    ``evidence_by_key`` / ``llm_by_key`` are keyed by the composite
    (source_file, source_sheet, source_column) so columns with the same name in
    different files never collide.
    """
    evidence_by_key = evidence_by_key or {}
    llm_by_key = llm_by_key or {}
    items: List[Dict[str, Any]] = []
    for v in validation_rows:
        col = v["source_column"]
        status = v["validation_status"]
        key = _ekey(v)
        ev = evidence_by_key.get(key, {})
        llm = llm_by_key.get(key, {})
        etype = ev.get("data_type_guess", "")
        group = _STATUS_GROUP.get(status, G_DECISIONS)
        meaning = llm.get("proposed_business_meaning", "") or ev.get(
            "candidate_value_profile_matches", "") or f"{etype} field"
        risk = "none/low"
        if status in (UNSAFE, BLOCKED, CONFLICTS_MEMORY, CONFLICTS_MAPPING):
            risk = "high — " + v.get("validation_reasons", "")
        elif status == REVIEW_REQUIRED and v.get("validation_reasons"):
            risk = "medium — " + v["validation_reasons"]
        items.append({
            "source_file": v.get("source_file", ""),
            "source_sheet": v.get("source_sheet", "") or ev.get("source_sheet", ""),
            "source_column": col,
            "domain_guess": ev.get("domain_guess", ""),
            "file_domain_guess": ev.get("file_domain_guess", ""),
            "group": group,
            "priority": _priority(col, etype, status),
            "likely_meaning": meaning,
            "suggested_mapping": v.get("proposed_target_field", ""),
            "candidate_source": v.get("candidate_source", ""),
            "confidence": v.get("confidence", ""),
            "validation_status": status,
            "is_pipeline_field": v.get("is_pipeline_field", False),
            "evidence_summary": _evidence_summary(ev),
            "risk": risk,
            "decision_options": _DECISION_OPTIONS,
            "llm_reasoning": llm.get("reasoning_summary", ""),
            "requires_user_approval": v.get("requires_user_approval", True),
        })
    items.sort(key=lambda x: (x["source_file"], x["priority"], x["source_column"]))

    counts: Dict[str, int] = {}
    by_file: Dict[str, int] = {}
    for it in items:
        counts[it["group"]] = counts.get(it["group"], 0) + 1
        by_file[it["source_file"]] = by_file.get(it["source_file"], 0) + 1
    high_priority = sum(1 for it in items if it["priority"] <= 1
                        and it["group"] == G_DECISIONS)
    needs_review = counts.get(G_DECISIONS, 0) + counts.get(G_CONFLICTS, 0)
    summary = {
        "total_columns_reviewed": len(items),
        "auto_approved": counts.get(G_APPROVALS, 0),
        "needs_review": needs_review,
        "blocked_or_missing_target": counts.get(G_MISSING, 0) + counts.get(G_CONFLICTS, 0),
        "high_priority_decisions": high_priority,
        "ignored_out_of_scope": counts.get(G_IGNORED, 0),
        # ~30s per decision item; approvals/ignored are near-zero effort.
        "estimated_review_minutes": round(needs_review * 0.5 + high_priority * 0.5, 1),
        "group_counts": counts,
        "files_in_review_queue": len(by_file),
        "review_items_by_file": by_file,
    }
    return {"summary": summary, "items": items}


_QUEUE_COLUMNS = [
    "source_file", "source_sheet", "source_column", "domain_guess",
    "file_domain_guess", "group", "priority", "likely_meaning", "suggested_mapping",
    "candidate_source", "confidence", "validation_status", "is_pipeline_field",
    "evidence_summary", "risk", "requires_user_approval",
]


def write_queue_artifacts(queue: Dict[str, Any], output_dir: str | Path) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = queue["items"]
    csv_path = out_dir / "33_mapping_review_queue.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_QUEUE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for it in items:
            w.writerow({c: it.get(c, "") for c in _QUEUE_COLUMNS})
    json_path = out_dir / "33_mapping_review_queue.json"
    json_path.write_text(json.dumps(queue, indent=2, default=str), encoding="utf-8")

    # 34 — decision template (pre-filled with safe defaults).
    decisions = {"_doc": "Workbench mapping decisions. Edit 'decision' per column.",
                 "generated_at": _now(), "decisions": {}}
    for it in items:
        default = ("approve" if it["validation_status"] == AUTO_APPROVED
                   else ("mark_not_needed" if it["group"] == G_IGNORED else "ask_later"))
        decisions["decisions"][it["source_column"]] = {
            "decision": default,
            "target_field": it["suggested_mapping"],
            "save_to_memory": False,
            "save_as_alias": False,
        }
    dec_path = out_dir / "34_mapping_review_decisions.yaml"
    dec_path.write_text(yaml.safe_dump(decisions, sort_keys=False), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "decisions": str(dec_path)}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_review_action_log(
    output_dir: str | Path, client_id: str, run_id: str, action: str,
    inputs: Optional[Dict[str, Any]] = None, outputs: Optional[List[str]] = None,
    status: str = "ok",
) -> Path:
    out_dir = Path(output_dir)
    path = out_dir / "35_mapping_review_action_log.json"
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if not isinstance(log, list):
        log = []
    log.append({"timestamp": _now(), "client_id": client_id, "run_id": run_id,
                "action": action, "inputs": inputs or {}, "outputs_written": outputs or [],
                "status": status})
    path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    return path
