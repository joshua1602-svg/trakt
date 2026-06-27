"""
blocker_diagnostics.py
======================

Operator-first "Why is the MI pipeline blocked?" diagnostics.

This is a DIAGNOSTICS / UX layer. It does not change mapping economics, funded MI,
pipeline MI or forecast logic — it only reads the deterministic Gate 4 decision
queue (28c) and the classified source-pack inventory (01) and explains, in plain
English, what is blocked, why, which file/domain is missing and what to do next.

It is the single source of truth for BOTH the HTML review pack "Why blocked?" box
and the ``explain-blockers`` CLI command, so the two never disagree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# --------------------------------------------------------------------------- #
# Source-pack composition: map the deterministic file classifications onto the
# operator-facing buckets used in the diagnostics.
# --------------------------------------------------------------------------- #
_FUNDED_LOAN = {"current_loan_report", "historical_loan_report"}
_PROPERTY = {"collateral_report"}
_CASHFLOW_FUNDER = {"cashflow_report", "warehouse_agreement"}
_PIPELINE = {"pipeline_report"}
_DOCS = {"data_dictionary", "securitisation_document", "investor_report"}

# Operator-facing bucket order (also the display order).
PACK_BUCKETS = ["pipeline_report", "funded_loan_tape", "property_tape",
                "cashflow_funder_tape", "docs", "unknown"]
# Human labels for the buckets (mirrors the requirement wording).
PACK_LABELS = {
    "pipeline_report": "pipeline_report",
    "funded_loan_tape": "funded_loan_tape",
    "property_tape": "property_tape",
    "cashflow_funder_tape": "cashflow/funder tape",
    "docs": "docs",
    "unknown": "unknown",
}
# The buckets that make up the funded book.
_FUNDED_BUCKETS = ("funded_loan_tape", "property_tape", "cashflow_funder_tape")
# Always displayed (even at 0) so a funded gap is obvious; unknown only if present.
_ALWAYS_SHOWN = ("pipeline_report", "funded_loan_tape", "property_tape",
                 "cashflow_funder_tape", "docs")


def _DISPLAY_BUCKETS(comp: Dict[str, Any]) -> List[str]:
    out = list(_ALWAYS_SHOWN)
    if comp["buckets"].get("unknown"):
        out.append("unknown")
    return out

# Funded-book canonical fields that typically cannot be resolved without a funded
# source file (loan/property/funder), so a pipeline-only pack legitimately blocks.
FUNDED_BOOK_FIELDS = {
    "origination_date", "maturity_date", "current_outstanding_balance",
    "original_balance", "current_loan_to_value", "original_loan_to_value",
    "current_valuation_amount", "original_valuation_amount",
    "current_interest_rate", "funded_date", "completion_date",
}
# Pipeline dates an operator MIGHT intentionally use to derive a funded date.
_PIPELINE_DATE_HINTS = ("Date Funds Released", "Application Submitted Date",
                        "KFI Submitted Date")


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def classify_source_pack(classifications: Iterable[str]) -> Dict[str, Any]:
    """Bucket the classified source files into operator-facing pack composition.

    Returns ``{buckets, total, funded_source_count, funded_present, pipeline_only,
    mostly_pipeline}``.
    """
    buckets = {b: 0 for b in PACK_BUCKETS}
    for raw in classifications:
        c = str(raw or "").strip().lower()
        if c in _PIPELINE:
            buckets["pipeline_report"] += 1
        elif c in _FUNDED_LOAN:
            buckets["funded_loan_tape"] += 1
        elif c in _PROPERTY:
            buckets["property_tape"] += 1
        elif c in _CASHFLOW_FUNDER:
            buckets["cashflow_funder_tape"] += 1
        elif c in _DOCS:
            buckets["docs"] += 1
        else:
            buckets["unknown"] += 1
    total = sum(buckets.values())
    funded = sum(buckets[b] for b in _FUNDED_BUCKETS)
    return {
        "buckets": buckets,
        "total": total,
        "funded_source_count": funded,
        "funded_present": funded > 0,
        "pipeline_only": total > 0 and buckets["pipeline_report"] > 0 and funded == 0,
        "mostly_pipeline": total > 0 and (buckets["pipeline_report"] / total) >= 0.5,
    }


def _pack_context_phrase(comp: Dict[str, Any]) -> str:
    """A short phrase describing the funded-source situation in the pack."""
    if comp["funded_present"]:
        return (f"{comp['funded_source_count']} funded source file(s) detected "
                f"({comp['buckets']['funded_loan_tape']} loan, "
                f"{comp['buckets']['property_tape']} property, "
                f"{comp['buckets']['cashflow_funder_tape']} cashflow/funder)")
    if comp["buckets"]["pipeline_report"]:
        return ("no funded loan tape detected; only pipeline_report files found")
    return "no funded source files detected"


def _explain_blocking(row: Dict[str, Any], comp: Dict[str, Any]) -> Dict[str, str]:
    """Explain one blocking Gate 4 decision: reason + pack context + action."""
    field = row.get("target_field", "") or "(unnamed target field)"
    dtype = str(row.get("decision_type", "") or "")
    is_funded = field in FUNDED_BOOK_FIELDS

    if dtype == "missing_required_target":
        reason = (f"required {'funded ' if is_funded else ''}MI field has no source, "
                  "derivation, default or ND rule.")
    else:
        reason = (row.get("operator_question") or row.get("recommendation")
                  or row.get("issue") or "operator decision required before MI handoff.")

    pack_context = _pack_context_phrase(comp)

    if is_funded and not comp["funded_present"]:
        action = ("add funded LoanExtract/PropertyExtract/Funder files, or run "
                  "pipeline-only mode (mark the funded field not applicable), or "
                  "intentionally map it to a pipeline date "
                  f"({', '.join(_PIPELINE_DATE_HINTS)}).")
    elif row.get("recommendation"):
        action = f"apply the deterministic recommendation: {row.get('recommendation')}"
    else:
        action = ("review the Gate 4 decision and approve a source, derivation, "
                  "default or ND rule.")

    return {
        "target_field": field,
        "decision_type": dtype,
        "reason": reason,
        "source_pack_context": pack_context,
        "suggested_action": action,
    }


def analyze_blockers(
    decision_rows: List[Dict[str, Any]],
    classifications: Iterable[str],
    *,
    tapes_present: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build the structured operator blocker report from the 28c decision rows and
    the classified source pack. Pure function — no I/O.
    """
    comp = classify_source_pack(classifications)
    rows = [r for r in (decision_rows or []) if isinstance(r, dict)]
    blocking = [r for r in rows if _as_bool(r.get("blocking"))]
    non_blocking = [r for r in rows if not _as_bool(r.get("blocking"))]

    items = [_explain_blocking(r, comp) for r in blocking]
    is_blocked = bool(blocking)
    status = "BLOCKED" if is_blocked else "READY_FOR_HANDOFF"

    # Central artifacts only render once no blocking Gate 4 decision remains.
    if is_blocked:
        central_rendered = False
        reason_not_rendered = "blocking Gate 4 decision remains."
    elif tapes_present is None:
        central_rendered = True
        reason_not_rendered = ""
    else:
        central_rendered = bool(tapes_present)
        reason_not_rendered = "" if tapes_present else "central tapes not yet built — run promote."

    # Plain-English "blocked because" bullets (deduplicated, operator-first).
    because: List[str] = []
    missing_fields = [it["target_field"] for it in items
                      if it["decision_type"] == "missing_required_target"]
    for f in missing_fields:
        because.append(f"Missing required target field: {f}")
    if missing_fields and not comp["funded_present"] and any(
            it["target_field"] in FUNDED_BOOK_FIELDS for it in items):
        because.append("No funded-book source file appears to be present")
    if comp["pipeline_only"]:
        because.append("Current source pack is classified as pipeline-only "
                       "(only pipeline_report files)")
    elif comp["mostly_pipeline"] and not comp["funded_present"]:
        because.append("Current source pack is mostly pipeline_report with no "
                       "funded-book files")
    # Any blocking decisions that are not missing-required still need a bullet.
    for it in items:
        if it["decision_type"] != "missing_required_target":
            because.append(f"Blocking decision on {it['target_field']}: {it['reason']}")

    return {
        "status": status,
        "is_blocked": is_blocked,
        "blocking_count": len(blocking),
        "non_blocking_count": len(non_blocking),
        "blocking_items": items,
        "because": because,
        "composition": comp,
        "central_artifacts_rendered": central_rendered,
        "reason_not_rendered": reason_not_rendered,
    }


# --------------------------------------------------------------------------- #
# Disk loaders (for the CLI) — read a finished run's artefacts.
# --------------------------------------------------------------------------- #
def _candidate_dirs(project_dir: Path, output_root: Optional[Path]) -> List[Path]:
    dirs = [project_dir, project_dir / "output"]
    if output_root:
        dirs += [output_root, output_root.parent]
    seen, out = set(), []
    for d in dirs:
        if d and str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    return out


def _find(project_dir: Path, output_root: Optional[Path], name: str) -> Optional[Path]:
    for d in _candidate_dirs(project_dir, output_root):
        p = d / name
        if p.exists():
            return p
    return None


def _load_json(path: Optional[Path]) -> Any:
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _decision_rows(project_dir: Path, output_root: Optional[Path]) -> List[Dict[str, Any]]:
    data = _load_json(_find(project_dir, output_root, "28c_human_decision_queue.json"))
    if isinstance(data, dict):
        return list(data.get("rows", []) or [])
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    return []


def _classifications(project_dir: Path, output_root: Optional[Path]) -> List[str]:
    data = _load_json(_find(project_dir, output_root, "01_file_inventory.json"))
    rows = data if isinstance(data, list) else (data or {}).get("rows", []) if data else []
    return [str((r or {}).get("classification", "")) for r in rows if isinstance(r, dict)]


def _tapes_present(project_dir: Path, output_root: Optional[Path]) -> bool:
    return bool(_find(project_dir, output_root, "18_central_lender_tape.csv"))


def load_blocker_report(project_dir: str | Path,
                        output_root: str | Path | None = None) -> Dict[str, Any]:
    """Load a finished run's 28c + 01 inventory and build the blocker report."""
    pdir = Path(project_dir)
    oroot = Path(output_root) if output_root else None
    rows = _decision_rows(pdir, oroot)
    classifications = _classifications(pdir, oroot)
    report = analyze_blockers(rows, classifications,
                              tapes_present=_tapes_present(pdir, oroot))
    report["project_dir"] = str(pdir)
    report["decision_queue_present"] = bool(
        _find(pdir, oroot, "28c_human_decision_queue.json"))
    report["inventory_present"] = bool(_find(pdir, oroot, "01_file_inventory.json"))
    return report


# --------------------------------------------------------------------------- #
# Compact CLI rendering
# --------------------------------------------------------------------------- #
def format_cli(report: Dict[str, Any]) -> str:
    """The short, machine-readable-ish console summary for ``explain-blockers``."""
    lines: List[str] = []
    lines.append(f"Status: {report['status']}")
    if not report.get("decision_queue_present", True):
        lines.append("  (no 28c_human_decision_queue found — run onboarding first)")
    if report["blocking_items"]:
        lines.append("Blocking target decisions:")
        for i, it in enumerate(report["blocking_items"], 1):
            lines.append(f"{i}. {it['target_field']}")
            lines.append(f"   Reason: {it['reason']}")
            lines.append(f"   Source-pack context: {it['source_pack_context']}")
            lines.append(f"   Suggested action: {it['suggested_action']}")
    else:
        lines.append("Blocking target decisions: none")
    lines.append(f"Non-blocking confirmations: {report['non_blocking_count']}")
    comp = report["composition"]
    bucket_str = ", ".join(
        f"{PACK_LABELS[b]}: {comp['buckets'][b]}" for b in _DISPLAY_BUCKETS(comp))
    lines.append(f"Source-pack composition: {bucket_str}")
    if not comp["funded_present"]:
        lines.append("Warning: no funded source files detected; funded MI central "
                     "tape may not be renderable.")
    lines.append("Central artifacts rendered: "
                 + ("yes" if report["central_artifacts_rendered"] else "no"))
    if report["reason_not_rendered"]:
        lines.append(f"Reason artifacts not rendered: {report['reason_not_rendered']}")
    return "\n".join(lines)
