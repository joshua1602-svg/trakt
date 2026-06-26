"""
non_blocking_approval.py
========================

Deterministic approval of NON-BLOCKING Gate 4 confirmations for MI-only demo /
test runs.

The target-first Gate 4 template (``34_target_first_decisions.yaml``) starts
every decision ``status: pending``. For a clean MI-only run the remaining Gate 4
items are typically *non-blocking confirmations* — confirm the deterministic
selected source, confirm an ND/default, or mark a field not-applicable — none of
which need a human to make a substantive choice. This module approves ONLY those,
deterministically and with a full audit trail, and writes a REAL approved
decision file (``34_target_first_decisions_approved.yaml``) with ``approved``
statuses — never a copied pending template.

Safety properties:
  * BLOCKING decisions are never auto-approved (kept pending).
  * Only confirmation-style decision types are auto-approved; anything requiring a
    substantive operator choice (missing required target, value-compatibility
    conflict, parse/header blocker, config value we do not already have) is left
    pending.
  * Nothing here touches 28a / 28c; the approved 34 file is the single artefact
    the workflow consumes on rerun.
  * Every auto-approved decision records ``auto_approved: true`` + a reason, and a
    top-level ``non_blocking_approval`` audit block lists exactly what was approved
    and what was skipped (and why).
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import target_first_decisions as tfd

_DECISIONS_NAME = "34_target_first_decisions.yaml"
_DEFAULT_OUT = "34_target_first_decisions_approved.yaml"

# Non-blocking confirmation decision types -> the deterministic operator action.
# These resolve a decision WITHOUT a substantive operator choice.
_CONFIRM_ACTION_BY_TYPE = {
    "source_priority_confirmation": "confirm_selected",
    "conflicting_source_candidates": "confirm_selected",
    "nd_default_confirmation": "confirm_default_or_nd",
    "default_or_nd_confirmation": "confirm_default_or_nd",
    "not_applicable_confirmation": "mark_not_applicable",
    # config_confirmation / config_value_required only when a value already exists
    # on the decision (handled in _action_for — never fabricated here).
}


def _action_for(dec: Dict[str, Any]) -> Optional[str]:
    """The deterministic confirmation action for a non-blocking decision, or None
    when the decision needs a substantive operator choice (left pending)."""
    dtype = str(dec.get("decision_type", "") or "")
    action = _CONFIRM_ACTION_BY_TYPE.get(dtype)
    if action == "confirm_selected":
        # Need a concrete selected source to confirm.
        src = dec.get("selected_source_column") or dec.get("source_column")
        return "confirm_selected" if src else None
    if action:
        return action
    if dtype in ("config_confirmation", "config_value_required"):
        # Only confirm a config value that is ALREADY present — never fabricate one.
        if dec.get("configured_value") not in (None, ""):
            return "configure_static_value"
    return None


def _apply_patch(dec: Dict[str, Any], action: str) -> None:
    """Set the operator-editable fields for a deterministic confirmation action."""
    dec["selected_action"] = action
    if action == "confirm_selected":
        dec["selected_source_file"] = dec.get("selected_source_file") or dec.get("source_file") or ""
        dec["selected_source_column"] = dec.get("selected_source_column") or dec.get("source_column") or ""
    elif action == "confirm_default_or_nd":
        dec["default_confirmed"] = True
    elif action == "mark_not_applicable":
        dec["not_applicable_confirmed"] = True
    elif action == "configure_static_value":
        # configured_value already present (guaranteed by _action_for).
        pass


def approve_non_blocking_decisions(
    project_dir: str | Path,
    *,
    decisions_path: Optional[str | Path] = None,
    out_path: Optional[str | Path] = None,
    approved_by: str = "",
    now: Optional[str] = None,
) -> Dict[str, Any]:
    """Approve all non-blocking Gate 4 confirmations into a real approved 34 file.

    Returns a summary dict ``{out_path, approved, already_approved, skipped_blocking,
    skipped_not_confirmable, ...}``. Never auto-approves blocking decisions; never
    mutates 28a/28c.
    """
    pdir = Path(project_dir)
    dec_path = Path(decisions_path) if decisions_path else pdir / _DECISIONS_NAME
    out = Path(out_path) if out_path else pdir / _DEFAULT_OUT
    stamp = now or datetime.now(timezone.utc).isoformat()
    who = approved_by or "approve-non-blocking-decisions"

    if not dec_path.exists():
        return {"error": f"decisions file not found: {dec_path}", "approved": 0,
                "skipped_blocking": [], "skipped_not_confirmable": []}

    doc = tfd.load_decisions(dec_path) or {}
    out_doc = copy.deepcopy(doc)

    approved_ids: List[str] = []
    already_approved: List[str] = []
    skipped_blocking: List[Dict[str, str]] = []
    skipped_not_confirmable: List[Dict[str, str]] = []

    for dec in out_doc.get("decisions", []) or []:
        dec_id = dec.get("decision_id", "")
        field = dec.get("target_field", "")
        dtype = dec.get("decision_type", "")
        if str(dec.get("status", "")).strip().lower() == "approved":
            already_approved.append(dec_id)
            continue
        if bool(dec.get("blocking", False)):
            skipped_blocking.append({"decision_id": dec_id, "target_field": field,
                                     "decision_type": dtype})
            continue
        action = _action_for(dec)
        if not action:
            skipped_not_confirmable.append({"decision_id": dec_id, "target_field": field,
                                            "decision_type": dtype})
            continue
        reason = (f"non-blocking {dtype or 'confirmation'} auto-approved via "
                  f"approve-non-blocking-decisions ({action})")
        _apply_patch(dec, action)
        dec["status"] = "approved"
        dec["auto_approved"] = True
        dec["auto_approved_reason"] = reason
        dec["operator_note"] = reason
        dec["approved_by"] = who
        dec["approved_at"] = stamp
        approved_ids.append(dec_id)

    # Top-level audit trail (provenance for every auto-approval / skip).
    out_doc["non_blocking_approval"] = {
        "approved_by": who,
        "approved_at": stamp,
        "source_decisions": str(dec_path),
        "approved": len(approved_ids),
        "approved_ids": approved_ids,
        "already_approved": already_approved,
        "skipped_blocking": skipped_blocking,
        "skipped_not_confirmable": skipped_not_confirmable,
        "policy": ("Only non-blocking confirmation decisions are auto-approved. "
                   "Blocking decisions and decisions needing a substantive operator "
                   "choice are left pending."),
    }
    out_doc["_doc"] = (
        "Generated by approve-non-blocking-decisions. Non-blocking Gate 4 "
        "confirmations were approved deterministically; blocking decisions remain "
        "pending. Re-run onboarding with --target-first-decisions <this file>.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(out_doc, sort_keys=False), encoding="utf-8")

    return {
        "out_path": str(out),
        "approved": len(approved_ids),
        "approved_ids": approved_ids,
        "already_approved": already_approved,
        "skipped_blocking": skipped_blocking,
        "skipped_not_confirmable": skipped_not_confirmable,
        "decisions_total": len(out_doc.get("decisions", []) or []),
        "source_decisions": str(dec_path),
        "approved_by": who,
        "approved_at": stamp,
    }


def format_summary(summary: Dict[str, Any]) -> str:
    """Human-readable console summary."""
    if summary.get("error"):
        return f"approve-non-blocking-decisions ERROR: {summary['error']}"
    lines = [
        "=" * 64,
        "Approve non-blocking Gate 4 confirmations (deterministic)",
        f"  Source decisions : {summary['source_decisions']}",
        f"  Approved file    : {summary['out_path']}",
        "",
        f"  Auto-approved (non-blocking) : {summary['approved']}",
        f"  Already approved             : {len(summary['already_approved'])}",
        f"  Skipped (blocking)           : {len(summary['skipped_blocking'])}",
        f"  Skipped (needs operator)     : {len(summary['skipped_not_confirmable'])}",
    ]
    if summary["skipped_blocking"]:
        lines.append("")
        lines.append("  Blocking decisions left pending (never auto-approved):")
        for s in summary["skipped_blocking"][:50]:
            lines.append(f"    - {s['decision_id']} [{s['target_field']}] ({s['decision_type']})")
    lines.append("")
    lines.append("  Next: re-run onboarding with "
                 f"--target-first-decisions {summary['out_path']}")
    lines.append("=" * 64)
    return "\n".join(lines)
