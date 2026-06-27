"""
target_first_decisions.py
=========================

Gate 4 decision capture + deterministic re-application for target-first
onboarding.

The managed-service loop is:

    run onboarding -> review Gate 4 (28c compact decision queue) -> operator
    edits an exported YAML template -> approved decisions are supplied to the
    next run -> approved decisions are applied deterministically to 28a/28c ->
    Gate 4 shrinks / becomes ready.

This module is deliberately separate from the source-column-level
``34_mapping_review_decisions.yaml`` / ``35_mapping_review_action_log.json``
artefacts (which are untouched). It owns:

    34_target_first_decisions.yaml                  (operator-editable template)
    35_target_first_decision_application_log.json    (what was applied this run)
    35_target_first_decision_application_log.csv

Application is deterministic and auditable: only ``status: approved`` decisions
are applied, every approved decision yields exactly one application-log row, and
invalid / missing / deferred decisions are logged without ever crashing the run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Recognised operator actions (selected_action values).
SUPPORTED_ACTIONS = [
    "confirm_selected",
    "choose_alternative",
    "merge_or_reconcile",
    "configure_static_value",
    "confirm_default_or_nd",
    "mark_not_applicable",
    "provide_source_mapping",
    "defer",
    "reject_recommendation",
]

# Only this status is ever applied.
_APPROVED = "approved"

# Application-log statuses.
APPLIED = "applied"
PENDING = "pending"
IGNORED_DEFERRED = "ignored_deferred"
INVALID = "invalid"
TARGET_FIELD_NOT_FOUND = "target_field_not_found"
DECISION_ID_NOT_FOUND = "decision_id_not_found"
REQUIRES_OPERATOR_REVIEW = "requires_operator_review"

_TEMPLATE_NAME = "34_target_first_decisions.yaml"
_LOG_JSON_NAME = "35_target_first_decision_application_log.json"
_LOG_CSV_NAME = "35_target_first_decision_application_log.csv"

_LOG_COLUMNS = [
    "decision_id", "target_field", "decision_type", "selected_action",
    "application_status", "applied_to_28a", "applied_to_28c",
    "selected_source_file", "selected_source_column", "configured_value",
    "default_confirmed", "not_applicable_confirmed", "operator_note", "message",
]


# ---------------------------------------------------------------------------
# Template generation (from 28c)
# ---------------------------------------------------------------------------

# Decision types that are source-priority / conflicting-source oriented.
_SOURCE_PRIORITY_TYPES = {"source_priority_confirmation", "conflicting_source_candidates"}


def build_decision_template(
    decision_rows: List[Dict[str, Any]],
    mode: str,
    client_id: str = "",
    run_id: str = "",
    target_contract_id: str = "",
) -> Dict[str, Any]:
    """Build the operator-editable YAML template from the 28c decision rows.

    One entry per Gate 4 decision; every entry starts ``status: pending``. For
    source-priority / conflicting-source decisions the deterministic recommended
    source is pre-populated (but the entry still requires explicit approval).
    """
    decisions: List[Dict[str, Any]] = []
    for d in decision_rows:
        dtype = d.get("decision_type", "")
        is_source_priority = dtype in _SOURCE_PRIORITY_TYPES
        decisions.append({
            # --- context echoed from 28c (read-only for the operator) ---
            "decision_id": d.get("decision_id", ""),
            "decision_type": dtype,
            "priority": d.get("priority", ""),
            "mode": mode,
            "target_contract_id": d.get("target_contract_id", "") or target_contract_id,
            "target_field": d.get("target_field", ""),
            "source_file": d.get("source_file", ""),
            "source_column": d.get("source_column", ""),
            "issue": d.get("issue", ""),
            "recommendation": d.get("recommendation", ""),
            "options": list(d.get("options", []) or []),
            "blocking": bool(d.get("blocking", False)),
            "operator_question": d.get("operator_question", ""),
            "evidence_summary": d.get("evidence_summary", ""),
            # --- operator-editable fields ---
            "status": "pending",
            "selected_action": None,
            # Pre-populate the deterministic recommended source for priority calls.
            "selected_source_file": (d.get("source_file", "") or None) if is_source_priority else None,
            "selected_source_column": (d.get("source_column", "") or None) if is_source_priority else None,
            "configured_value": None,
            "default_confirmed": None,
            "not_applicable_confirmed": None,
            "operator_note": None,
            "approved_by": None,
            "approved_at": None,
        })
    return {
        "version": 1,
        "client_id": client_id,
        "run_id": run_id,
        "mode": mode,
        "decision_source": "28c_human_decision_queue",
        "supported_actions": list(SUPPORTED_ACTIONS),
        "_doc": ("Set 'status: approved' and a 'selected_action' per decision, then "
                 "re-run onboarding with --target-first-decisions <this file>. Only "
                 "approved decisions are applied."),
        "decisions": decisions,
    }


def write_decision_template(template: Dict[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / _TEMPLATE_NAME
    path.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    return path


def load_decisions(path: str | Path) -> Optional[Dict[str, Any]]:
    """Load a target-first decisions YAML (None when absent / unreadable)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def approved_decisions(decisions_doc: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only the decisions whose status is exactly 'approved'."""
    if not decisions_doc:
        return []
    out = []
    for d in decisions_doc.get("decisions", []) or []:
        if isinstance(d, dict) and str(d.get("status", "")).strip().lower() == _APPROVED:
            out.append(d)
    return out


def is_real_approval(decisions_doc: Optional[Dict[str, Any]]) -> bool:
    """True iff at least one decision is actually ``status: approved``.

    A copied pending template (every decision still ``pending``) is NOT a real
    approval — governance must not treat it as one.
    """
    return bool(approved_decisions(decisions_doc))


# ---------------------------------------------------------------------------
# Deterministic application
# ---------------------------------------------------------------------------

def _log_entry(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "decision_id": d.get("decision_id", ""),
        "target_field": d.get("target_field", ""),
        "decision_type": d.get("decision_type", ""),
        "selected_action": (d.get("selected_action") or ""),
        "application_status": "",
        "applied_to_28a": False,
        "applied_to_28c": False,
        "selected_source_file": d.get("selected_source_file") or "",
        "selected_source_column": d.get("selected_source_column") or "",
        "configured_value": d.get("configured_value") or "",
        "default_confirmed": d.get("default_confirmed") if d.get("default_confirmed") is not None else "",
        "not_applicable_confirmed": d.get("not_applicable_confirmed")
        if d.get("not_applicable_confirmed") is not None else "",
        "operator_note": d.get("operator_note") or "",
        "message": "",
    }


def apply_decisions(
    coverage_rows: List[Dict[str, Any]],
    decision_rows: List[Dict[str, Any]],
    approved: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply approved decisions to 28a (coverage) and 28c (decision queue).

    Returns ``(coverage_rows, remaining_decision_rows, application_log)``.
    Mutates coverage rows in place; resolved 28c decisions are dropped from the
    returned queue. Never raises — every problem becomes a log row.
    """
    cov_by_field: Dict[str, Dict[str, Any]] = {}
    for r in coverage_rows:
        cov_by_field.setdefault(r.get("target_field", ""), r)
    dec_by_id = {r.get("decision_id", ""): r for r in decision_rows}
    resolved_ids: set = set()
    log: List[Dict[str, Any]] = []

    for d in approved:
        e = _log_entry(d)
        did = d.get("decision_id", "")
        tfield = d.get("target_field", "")
        action = str(d.get("selected_action") or "").strip()
        dec_row = dec_by_id.get(did)
        cov_row = cov_by_field.get(tfield)

        if not did or dec_row is None:
            e["application_status"] = DECISION_ID_NOT_FOUND
            e["message"] = f"decision_id '{did}' not present in current 28c queue"
            log.append(e)
            continue
        if action not in SUPPORTED_ACTIONS:
            e["application_status"] = INVALID
            e["message"] = f"unsupported selected_action '{action}'"
            log.append(e)
            continue

        # Actions that mutate 28a require the target field to exist.
        needs_cov = action in {
            "confirm_selected", "choose_alternative", "configure_static_value",
            "confirm_default_or_nd", "mark_not_applicable", "provide_source_mapping",
        }
        if needs_cov and cov_row is None:
            e["application_status"] = TARGET_FIELD_NOT_FOUND
            e["message"] = f"target_field '{tfield}' not present in current 28a matrix"
            log.append(e)
            continue

        if action == "defer":
            e["application_status"] = IGNORED_DEFERRED
            e["message"] = "deferred by operator; kept in 28c"
            log.append(e)
            continue

        if action == "merge_or_reconcile":
            e["application_status"] = REQUIRES_OPERATOR_REVIEW
            e["message"] = ("merge/reconcile needs an explicit reconciliation rule; "
                            "kept pending in 28c")
            log.append(e)
            continue

        if action == "reject_recommendation":
            e["application_status"] = REQUIRES_OPERATOR_REVIEW
            e["message"] = ("recommendation rejected without an alternative "
                            "action/source/config; kept in 28c")
            log.append(e)
            continue

        if action == "confirm_selected":
            cov_row["requires_user_decision"] = False
            cov_row["blocking"] = False
            cov_row["decision_reason"] = "operator confirmed selected source"
            e["applied_to_28a"] = True
            e["applied_to_28c"] = True
            e["application_status"] = APPLIED
            e["message"] = (f"confirmed selected source "
                            f"{cov_row.get('selected_source_column','')}")
            resolved_ids.add(did)

        elif action == "choose_alternative":
            new_file = d.get("selected_source_file") or ""
            new_col = d.get("selected_source_column") or ""
            if not new_col:
                e["application_status"] = INVALID
                e["message"] = "choose_alternative requires selected_source_column"
                log.append(e)
                continue
            prev = (f"{cov_row.get('selected_source_file','')}::"
                    f"{cov_row.get('selected_source_column','')}")
            cov_row["selected_source_file"] = new_file
            cov_row["selected_source_column"] = new_col
            cov_row["coverage_status"] = "source_mapped"
            cov_row["coverage_basis"] = "operator_override_alternative_source"
            cov_row["requires_user_decision"] = False
            cov_row["blocking"] = False
            cov_row["decision_reason"] = (f"operator chose alternative source "
                                          f"{new_file}::{new_col} (was {prev})")
            e["applied_to_28a"] = True
            e["applied_to_28c"] = True
            e["application_status"] = APPLIED
            e["message"] = f"selected source changed {prev} -> {new_file}::{new_col}"
            resolved_ids.add(did)

        elif action == "configure_static_value":
            value = d.get("configured_value")
            cov_row["coverage_status"] = "configured_static"
            cov_row["coverage_basis"] = "operator_configured_static"
            cov_row["configured_value_source"] = (
                d.get("configured_value_source") or "operator_config")
            cov_row["requires_user_decision"] = False
            cov_row["blocking"] = False
            cov_row["decision_reason"] = (
                f"operator configured static value: {value}" if value not in (None, "")
                else "operator configured static value")
            e["applied_to_28a"] = True
            e["applied_to_28c"] = True
            e["application_status"] = APPLIED
            e["message"] = "configured static value applied"
            resolved_ids.add(did)

        elif action == "confirm_default_or_nd":
            base = cov_row.get("default_rule", "") or cov_row.get("nd_rule_applied", "")
            cov_row["default_rule"] = (f"{base} (operator confirmed)" if base
                                       else "operator confirmed default/ND")
            cov_row["coverage_basis"] = (cov_row.get("coverage_basis", "")
                                         or "default_or_nd") + "; operator_confirmed"
            cov_row["requires_user_decision"] = False
            cov_row["blocking"] = False
            cov_row["decision_reason"] = "default / ND rule confirmed by operator"
            e["applied_to_28a"] = True
            e["applied_to_28c"] = True
            e["default_confirmed"] = True
            e["application_status"] = APPLIED
            e["message"] = "default/ND confirmed"
            resolved_ids.add(did)

        elif action == "mark_not_applicable":
            cov_row["applicability_status"] = "not_applicable"
            cov_row["coverage_status"] = "not_applicable"
            cov_row["coverage_basis"] = "operator_not_applicable"
            cov_row["requires_user_decision"] = False
            cov_row["blocking"] = False
            cov_row["decision_reason"] = "marked not applicable by operator"
            e["applied_to_28a"] = True
            e["applied_to_28c"] = True
            e["not_applicable_confirmed"] = True
            e["application_status"] = APPLIED
            e["message"] = "target field marked not applicable"
            resolved_ids.add(did)

        elif action == "provide_source_mapping":
            new_file = d.get("selected_source_file") or ""
            new_col = d.get("selected_source_column") or ""
            if not new_col:
                e["application_status"] = INVALID
                e["message"] = "provide_source_mapping requires selected_source_column"
                log.append(e)
                continue
            cov_row["selected_source_file"] = new_file
            cov_row["selected_source_column"] = new_col
            cov_row["coverage_status"] = "source_mapped"
            cov_row["coverage_basis"] = "operator_source_mapping"
            cov_row["requires_user_decision"] = False
            cov_row["blocking"] = False
            cov_row["decision_reason"] = (
                f"operator provided source mapping {new_file}::{new_col}")
            e["applied_to_28a"] = True
            e["applied_to_28c"] = True
            e["application_status"] = APPLIED
            e["message"] = f"source mapping provided {new_file}::{new_col}"
            resolved_ids.add(did)

        log.append(e)

    remaining = [r for r in decision_rows if r.get("decision_id", "") not in resolved_ids]
    return coverage_rows, remaining, log


# ---------------------------------------------------------------------------
# Application-log writers + summary
# ---------------------------------------------------------------------------

def application_summary(log: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for e in log:
        counts[e["application_status"]] = counts.get(e["application_status"], 0) + 1
    return {
        "decisions_supplied": len(log),
        "applied": counts.get(APPLIED, 0),
        "deferred": counts.get(IGNORED_DEFERRED, 0),
        "requires_operator_review": counts.get(REQUIRES_OPERATOR_REVIEW, 0),
        "invalid": (counts.get(INVALID, 0) + counts.get(TARGET_FIELD_NOT_FOUND, 0)
                    + counts.get(DECISION_ID_NOT_FOUND, 0)),
        "application_status_counts": counts,
    }


def write_application_log(
    log: List[Dict[str, Any]], out_dir: str | Path,
    client_id: str = "", run_id: str = "", decisions_source: str = "",
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / _LOG_JSON_NAME
    json_path.write_text(json.dumps({
        "client_id": client_id, "run_id": run_id, "decisions_source": decisions_source,
        "summary": application_summary(log), "rows": log,
    }, indent=2, default=str), encoding="utf-8")
    csv_path = out / _LOG_CSV_NAME
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_LOG_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for e in log:
            w.writerow({c: e.get(c, "") for c in _LOG_COLUMNS})
    return {"json": str(json_path), "csv": str(csv_path)}
