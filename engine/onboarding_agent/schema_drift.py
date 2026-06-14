"""
schema_drift.py
===============

PART 9 — schema drift handling for repeat lender runs.

Compares the current file's per-column evidence against a stored signature from
the client's previous run, and routes ONLY changed/new items to review so repeat
onboarding does not re-ask the same questions.

Artefacts:
    37_schema_drift_report.csv / .json
A signature snapshot is stored per client under the client_memory dir.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Statuses (PART 9).
UNCHANGED = "unchanged_known_mapping"
VALUE_PROFILE_CHANGED = "known_mapping_value_profile_changed"
NEW_COLUMN = "new_column"
MISSING_COLUMN = "missing_expected_column"
ENUM_CHANGED = "enum_values_changed"
STAGE_CHANGED = "stage_values_changed"
REQUIRES_REVIEW = "requires_review"

SIGNATURE_FILE = "schema_signature.json"


def build_signature(evidence_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A compact, comparable signature of the current schema."""
    cols: Dict[str, Any] = {}
    for e in evidence_rows:
        cols[e["source_column"]] = {
            "data_type_guess": e.get("data_type_guess", ""),
            "distinct_count": e.get("distinct_count", 0),
            "null_rate": e.get("null_rate", 0.0),
            "distinct_values": sorted(
                str(e.get("sample_values_distinct_redacted", "")).split("; ")) if
                e.get("data_type_guess") in ("enum",) else [],
            "stage_like_score": e.get("stage_like_score", 0.0),
        }
    return {"columns": cols}


def load_signature(memory_dir: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(memory_dir) / SIGNATURE_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_signature(signature: Dict[str, Any], memory_dir: str | Path) -> Path:
    d = Path(memory_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / SIGNATURE_FILE
    p.write_text(json.dumps(signature, indent=2, default=str), encoding="utf-8")
    return p


def _value_profile_changed(prev: Dict[str, Any], cur: Dict[str, Any]) -> bool:
    if prev.get("data_type_guess") != cur.get("data_type_guess"):
        return True
    pn, cn = prev.get("null_rate", 0), cur.get("null_rate", 0)
    if abs(float(pn) - float(cn)) > 0.4:
        return True
    pd_, cd = prev.get("distinct_count", 0), cur.get("distinct_count", 0)
    if max(pd_, cd) and abs(pd_ - cd) / max(pd_, cd, 1) > 0.5:
        return True
    return False


def detect_drift(
    current_evidence: List[Dict[str, Any]],
    previous_signature: Optional[Dict[str, Any]],
    known_mapped_columns: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Compare the current schema to the previous signature; one row per change."""
    known_mapped_columns = known_mapped_columns or set()
    cur_sig = build_signature(current_evidence)
    cur_cols = cur_sig["columns"]
    rows: List[Dict[str, Any]] = []

    if not previous_signature:
        # First run for this client: nothing to compare; everything is "new".
        for col, c in cur_cols.items():
            rows.append(_drift_row(col, NEW_COLUMN, "first run — no prior signature",
                                   requires_review=col not in known_mapped_columns))
        return rows

    prev_cols = previous_signature.get("columns", {})
    for col, c in cur_cols.items():
        if col not in prev_cols:
            rows.append(_drift_row(col, NEW_COLUMN, "column not seen in previous run",
                                   requires_review=True))
            continue
        p = prev_cols[col]
        if c.get("stage_like_score", 0) >= 0.6 and sorted(c.get("distinct_values", [])) \
                != sorted(p.get("distinct_values", [])):
            rows.append(_drift_row(col, STAGE_CHANGED, "stage/status values changed",
                                   requires_review=True))
        elif c.get("data_type_guess") == "enum" and sorted(c.get("distinct_values", [])) \
                != sorted(p.get("distinct_values", [])):
            rows.append(_drift_row(col, ENUM_CHANGED, "enum values changed",
                                   requires_review=True))
        elif _value_profile_changed(p, c):
            rows.append(_drift_row(col, VALUE_PROFILE_CHANGED,
                                   "value profile changed for a known column",
                                   requires_review=True))
        else:
            rows.append(_drift_row(col, UNCHANGED, "no material change",
                                   requires_review=False))
    for col in prev_cols:
        if col not in cur_cols:
            rows.append(_drift_row(col, MISSING_COLUMN,
                                   "previously mapped column missing", requires_review=True))
    return rows


def _drift_row(col: str, status: str, reason: str, requires_review: bool) -> Dict[str, Any]:
    return {"source_column": col, "drift_status": status, "reason": reason,
            "requires_review": bool(requires_review)}


def columns_needing_review(drift_rows: List[Dict[str, Any]]) -> set:
    return {r["source_column"] for r in drift_rows if r["requires_review"]}


_DRIFT_COLUMNS = ["source_column", "drift_status", "reason", "requires_review"]


def write_drift_artifacts(rows: List[Dict[str, Any]], output_dir: str | Path) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "37_schema_drift_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_DRIFT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _DRIFT_COLUMNS})
    json_path = out_dir / "37_schema_drift_report.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}
