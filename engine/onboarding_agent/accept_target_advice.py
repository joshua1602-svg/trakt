"""
accept_target_advice.py
=======================

Operator helper: accept the target-first LLM advisor's recommendations
(``36_target_first_llm_recommendations.json``) into an **approved** Gate-4
decision file (``34_target_first_decisions_approved.yaml``) that the onboarding
workflow consumes on rerun via ``--target-first-decisions``.

The advisor is advisory only — it never mutates the deterministic 28a/28c
artefacts and it leaves ``34_target_first_decisions.yaml`` pending by design.
This module turns *advised* recommendations into approved decisions, so an
operator does not have to hand-edit the template, while preserving every safety
property:

* only rows with ``llm_advice_status == advised`` are applied (other statuses —
  ``invalid_response`` / ``parse_failed`` / ``skipped_budget`` / ``no_advice`` /
  ``requires_operator_review`` — are skipped unless explicitly allowed);
* ``requires_operator_review`` / ``merge_or_reconcile`` / ``reject_recommendation``
  / ``defer`` actions are never auto-approved (operator-review actions);
* a recommended source column must be within the field's allowed candidates
  (from 28a) — an invented column is skipped, never approved;
* decisions without usable advice stay ``pending``;
* nothing here touches 28a / 28c — the approved 34 file remains the single thing
  the workflow consumes on rerun.

It maps the advisor's ``llm_recommended_action`` to the ``selected_action``
vocabulary that ``target_first_decisions.apply_decisions`` understands:

    provide_source_mapping  -> provide_source_mapping   (map a source column)
    choose_alternative      -> choose_alternative       (map a source column)
    confirm_selected        -> confirm_selected         (confirm the source)
    configure_static_value  -> configure_static_value   (set a derivation/default)
    confirm_default_or_nd   -> confirm_default_or_nd    (confirm an ND/default)
    mark_not_applicable     -> mark_not_applicable      (mark not applicable)
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from . import target_first_decisions as tfd

_DECISIONS_NAME = "34_target_first_decisions.yaml"
_RECS_NAME = "36_target_first_llm_recommendations.json"
_RAW_RESPONSE_NAME = "36_target_first_llm_raw_response.json"
_USAGE_NAME = "36_target_first_llm_usage_summary.json"
_DEFAULT_OUT = "34_target_first_decisions_approved.yaml"
_COVERAGE_NAME = "28a_target_coverage_matrix.json"

# Advice statuses the advisor emits; only ADVISED is applied by default.
ADVISED = "advised"
_NON_APPLY_STATUSES = {
    "invalid_response", "parse_failed", "skipped_budget", "no_advice",
    "requires_operator_review",
}

# Advisor actions that resolve a decision (map straight onto selected_action).
_AUTO_APPROVABLE_ACTIONS = {
    "confirm_selected", "choose_alternative", "provide_source_mapping",
    "configure_static_value", "confirm_default_or_nd", "mark_not_applicable",
}
# Actions that always need a human (never auto-approved).
_OPERATOR_REVIEW_ACTIONS = {
    "requires_operator_review", "merge_or_reconcile", "reject_recommendation",
    "defer",
}
# Actions that resolve a decision by mapping/choosing a source column.
_SOURCE_MAPPING_ACTIONS = {"choose_alternative", "provide_source_mapping"}


def _norm_action(value: Any) -> str:
    return re.sub(r"\s+", "_", str(value or "").strip().lower())


def _norm_col(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _load_recommendations(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("rows") or data.get("recommendations") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    return [r for r in rows if isinstance(r, dict)]


def _parse_diagnostics(pdir: Path) -> Dict[str, Any]:
    """Surface the advisor's parse status / error and the raw-response artefact
    path so a parse failure is explained, not silently swallowed."""
    raw_path = pdir / _RAW_RESPONSE_NAME
    usage_path = pdir / _USAGE_NAME
    parse_status = ""
    parse_error = ""
    recs_parsed: Optional[int] = None
    for p in (raw_path, usage_path):
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict):
            parse_status = parse_status or str(data.get("parse_status", "") or "")
            parse_error = parse_error or str(data.get("parse_error", "") or "")
            if recs_parsed is None and data.get("recommendations_parsed") is not None:
                try:
                    recs_parsed = int(data.get("recommendations_parsed"))
                except (TypeError, ValueError):
                    recs_parsed = None
    return {
        "parse_status": parse_status or None,
        "parse_error": parse_error or None,
        "recommendations_parsed": recs_parsed,
        "raw_response_path": str(raw_path) if raw_path.exists() else None,
        "usage_path": str(usage_path) if usage_path.exists() else None,
    }


def _allowed_columns_by_field(coverage_path: Path) -> Dict[str, set]:
    """Map target_field -> set of allowed source-column tokens, from 28a.

    Combines the deterministic ``selected_source_column`` with every column named
    in ``alternative_source_candidates`` (``file::sheet::col (conf); ...``). An
    empty map means "28a unavailable" — callers then skip column validation
    rather than wrongly rejecting.
    """
    if not coverage_path.exists():
        return {}
    try:
        data = json.loads(coverage_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: Dict[str, set] = {}
    for row in data.get("rows", []) or []:
        field = row.get("target_field", "")
        cols: set = set()
        sel = row.get("selected_source_column", "")
        if sel:
            cols.add(_norm_col(sel))
        for chunk in str(row.get("alternative_source_candidates", "") or "").split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            # "file::sheet::col (0.97)" -> col
            head = chunk.split(" (")[0]
            col = head.split("::")[-1] if "::" in head else head
            if col:
                cols.add(_norm_col(col))
        out[field] = cols
    return out


# --------------------------------------------------------------------------- #
# Acceptance
# --------------------------------------------------------------------------- #

def accept_target_advice(
    project_dir: str | Path,
    *,
    recommendations_path: Optional[str | Path] = None,
    decisions_path: Optional[str | Path] = None,
    coverage_path: Optional[str | Path] = None,
    out_path: Optional[str | Path] = None,
    approved_by: str = "",
    allow_statuses: Sequence[str] = (),
    allow_actions: Sequence[str] = (),
    min_confidence: float = 0.0,
    now: Optional[str] = None,
) -> Dict[str, Any]:
    """Accept advised recommendations into an approved 34 decision file.

    Returns a summary dict: ``{out_path, approved, pending, skipped:[{decision_id,
    target_field, reason}], counts, ...}``. Never mutates 28a/28c.
    """
    pdir = Path(project_dir)
    dec_path = Path(decisions_path) if decisions_path else pdir / _DECISIONS_NAME
    rec_path = Path(recommendations_path) if recommendations_path else pdir / _RECS_NAME
    cov_path = Path(coverage_path) if coverage_path else pdir / _COVERAGE_NAME
    out = Path(out_path) if out_path else pdir / _DEFAULT_OUT
    stamp = now or datetime.now(timezone.utc).isoformat()

    if not dec_path.exists():
        return {"error": f"decisions file not found: {dec_path}", "approved": 0,
                "pending": 0, "skipped": []}
    if not rec_path.exists():
        return {"error": f"recommendations file not found: {rec_path}", "approved": 0,
                "pending": 0, "skipped": []}

    doc = tfd.load_decisions(dec_path) or {}
    decisions = list(doc.get("decisions", []) or [])
    recs = _load_recommendations(rec_path)
    rec_by_id = {r.get("decision_id", ""): r for r in recs if r.get("decision_id")}
    allowed_cols = _allowed_columns_by_field(cov_path)

    allow_status_set = {ADVISED} | {str(s).strip().lower() for s in allow_statuses}
    allow_action_set = {_norm_action(a) for a in allow_actions}

    out_doc = copy.deepcopy(doc)
    approved = 0
    pending = 0
    skipped: List[Dict[str, str]] = []

    def _skip(dec_id: str, field: str, reason: str) -> None:
        nonlocal pending
        pending += 1
        skipped.append({"decision_id": dec_id, "target_field": field, "reason": reason})

    for dec in out_doc.get("decisions", []) or []:
        dec_id = dec.get("decision_id", "")
        field = dec.get("target_field", "")
        # Never override a decision an operator already approved.
        if str(dec.get("status", "")).strip().lower() == "approved":
            approved += 1
            continue

        rec = rec_by_id.get(dec_id)
        if rec is None:
            _skip(dec_id, field, "no_recommendation")
            continue

        status = str(rec.get("llm_advice_status", "")).strip().lower()
        if status not in allow_status_set:
            _skip(dec_id, field, f"advice_status={status or 'unknown'}")
            continue

        action = _norm_action(rec.get("llm_recommended_action"))
        if action in _OPERATOR_REVIEW_ACTIONS and action not in allow_action_set:
            _skip(dec_id, field, f"action={action}_requires_operator_review")
            continue
        if action not in _AUTO_APPROVABLE_ACTIONS:
            _skip(dec_id, field, f"action={action or 'none'}_not_applicable")
            continue

        try:
            conf = float(rec.get("llm_confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < float(min_confidence):
            _skip(dec_id, field, f"confidence={conf}<min_confidence={min_confidence}")
            continue

        # Resolve the action into 34 fields; validate before approving.
        note = (rec.get("llm_rationale", "") or rec.get("llm_operator_note", "")
                or "accepted from LLM target advisor")
        ok, reason, patch = _resolve_action(action, rec, field, allowed_cols)
        if not ok:
            _skip(dec_id, field, reason)
            continue

        dec.update(patch)
        dec["status"] = "approved"
        dec["operator_note"] = (note + " (auto-accepted via accept-target-advice)")[:500]
        dec["approved_by"] = approved_by or "accept-target-advice"
        dec["approved_at"] = stamp
        dec["llm_advice_status"] = status
        dec["llm_confidence"] = conf
        approved += 1

    # Record provenance at the top of the approved doc.
    out_doc["acceptance"] = {
        "accepted_from": str(rec_path),
        "source_decisions": str(dec_path),
        "approved_by": approved_by or "accept-target-advice",
        "approved_at": stamp,
        "approved": approved,
        "pending": pending,
        "skipped": len(skipped),
        "min_confidence": float(min_confidence),
        "allowed_statuses": sorted(allow_status_set),
        "allowed_actions": sorted(allow_action_set),
    }
    out_doc["_doc"] = (
        "Generated by accept-target-advice from the LLM target advisor. Only "
        "'advised' recommendations were applied; others remain pending. Re-run "
        "onboarding with --target-first-decisions <this file> to apply.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(out_doc, sort_keys=False), encoding="utf-8")

    # If anything was skipped because the advice could not be parsed, surface the
    # parse status/error and the raw-response artefact path (never a silent skip).
    parse_skips = sum(1 for s in skipped if str(s["reason"]).startswith("advice_status=parse_failed"))
    diagnostics = _parse_diagnostics(pdir) if parse_skips else {}

    return {
        "out_path": str(out),
        "approved": approved,
        "pending": pending,
        "skipped": skipped,
        "parse_failed_skips": parse_skips,
        "parse_diagnostics": diagnostics,
        "decisions_total": len(decisions),
        "recommendations_total": len(recs),
        "source_decisions": str(dec_path),
        "accepted_from": str(rec_path),
        "approved_by": approved_by or "accept-target-advice",
        "approved_at": stamp,
    }


def _resolve_action(
    action: str, rec: Dict[str, Any], field: str, allowed_cols: Dict[str, set],
) -> Tuple[bool, str, Dict[str, Any]]:
    """Map an advised action to 34 fields. Returns (ok, skip_reason, field_patch)."""
    patch: Dict[str, Any] = {
        "selected_action": None, "selected_source_file": None,
        "selected_source_column": None, "configured_value": None,
        "default_confirmed": None, "not_applicable_confirmed": None,
    }

    if action == "mark_not_applicable":
        patch["selected_action"] = "mark_not_applicable"
        patch["not_applicable_confirmed"] = True
        return True, "", patch

    if action == "confirm_default_or_nd":
        patch["selected_action"] = "confirm_default_or_nd"
        patch["default_confirmed"] = True
        return True, "", patch

    if action == "configure_static_value":
        value = rec.get("llm_recommended_configured_value", "")
        if value in (None, ""):
            return False, "configure_static_value_missing_configured_value", patch
        patch["selected_action"] = "configure_static_value"
        patch["configured_value"] = value
        return True, "", patch

    if action == "confirm_selected":
        patch["selected_action"] = "confirm_selected"
        # Carry the source through for the log (cov already holds the selection).
        patch["selected_source_file"] = rec.get("llm_recommended_source_file") or None
        patch["selected_source_column"] = rec.get("llm_recommended_source_column") or None
        return True, "", patch

    if action in _SOURCE_MAPPING_ACTIONS:
        col = rec.get("llm_recommended_source_column", "")
        if not col:
            return False, f"{action}_missing_source_column", patch
        # Validate the column is within the field's allowed candidates (28a).
        field_allowed = allowed_cols.get(field)
        if field_allowed and _norm_col(col) not in field_allowed:
            return False, f"{action}_source_not_in_candidates", patch
        patch["selected_action"] = action
        patch["selected_source_file"] = rec.get("llm_recommended_source_file") or ""
        patch["selected_source_column"] = col
        return True, "", patch

    return False, f"unhandled_action={action}", patch


def _humanise_skip(reason: str) -> str:
    """Operator-facing wording for a skip reason. A parse failure must NOT imply
    the field is missing from MI output — only that the advice could not be parsed."""
    r = str(reason or "")
    if r.startswith("advice_status=parse_failed"):
        return ("Target decision pending because LLM advice could not be parsed. "
                "Field may still be present in MI output.")
    if r.startswith("advice_status="):
        return f"Pending — advisor status {r.split('=', 1)[1]} is not auto-approvable."
    if r == "no_recommendation":
        return "Pending — no advisor recommendation for this decision."
    if r.endswith("_requires_operator_review"):
        return "Pending — advisor action needs operator review."
    if r.endswith("_source_not_in_candidates"):
        return "Pending — advised source is not among the allowed candidates."
    return f"Pending — {r}."


def format_summary(summary: Dict[str, Any]) -> str:
    """Human-readable console summary."""
    if summary.get("error"):
        return f"accept-target-advice ERROR: {summary['error']}"
    lines = [
        "=" * 64,
        "Accept target-first LLM advice",
        f"  Source decisions : {summary['source_decisions']}",
        f"  Recommendations  : {summary['accepted_from']}",
        f"  Approved file    : {summary['out_path']}",
        "",
        f"  Decisions approved : {summary['approved']}",
        f"  Decisions pending  : {summary['pending']}",
        f"  Decisions skipped  : {len(summary['skipped'])}",
    ]
    diag = summary.get("parse_diagnostics") or {}
    if summary.get("parse_failed_skips"):
        lines.append("")
        lines.append(f"  ! {summary['parse_failed_skips']} decision(s) pending because the LLM "
                     "advice could not be parsed into recommendations.")
        if diag.get("parse_status"):
            lines.append(f"    parse_status : {diag['parse_status']}")
        if diag.get("parse_error"):
            lines.append(f"    parse_error  : {diag['parse_error']}")
        if diag.get("raw_response_path"):
            lines.append(f"    raw response : {diag['raw_response_path']}")
        lines.append("    These fields may still be present in the prepared MI output.")
    if summary["skipped"]:
        lines.append("")
        lines.append("  Skipped (kept pending):")
        for s in summary["skipped"][:50]:
            lines.append(f"    - {s['decision_id']} [{s['target_field']}]: {_humanise_skip(s['reason'])}")
    lines.append("")
    lines.append("  Next: re-run onboarding with "
                 f"--target-first-decisions {summary['out_path']}")
    lines.append("=" * 64)
    return "\n".join(lines)
