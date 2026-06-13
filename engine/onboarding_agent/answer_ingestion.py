"""
answer_ingestion.py
===================

PART 4/5 — v1 answer-ingestion loop for the Onboarding Agent.

A reviewer answers the generated ``07_gap_questions.yaml`` (using the
``example_answers.yaml`` template) and feeds the answers back in. This module
deterministically validates the answers and emits *approved* onboarding
artefacts:

  10_approved_onboarding_project.yaml
  11_approved_config.yaml
  12_approved_mapping_overrides.yaml
  13_source_precedence_rules.yaml
  14_enum_review_decisions.yaml
  15_answer_ingestion_report.json

It is review-first and deterministic: no LLMs, no Gates 1–5, and it never
mutates production config. Approval flips to *ready_for_handoff* only when every
blocking question has a valid answer.

Answers file schema (simple, documented)::

    answers:
      Q1:
        answer: 2026-01-31
        approved_by: user
        note: month-end reporting date
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .gap_analyzer import ENUM_DECISION_ACTIONS

# Approval statuses (also used as the updated review status).
STATUS_READY = "ready_for_handoff"
STATUS_BLOCKED = "still_blocking"
STATUS_INVALID = "invalid_answer"
STATUS_ANSWERED = "answered"

_VALID_GEOGRAPHY_ANSWERS = {"GBZZZ", "ITL3", "other"}
_DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class ProjectContext:
    """Everything answer ingestion needs, loaded from an onboarding output dir."""

    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.run_summary = _load_json(self.project_dir / "09_onboarding_run_summary.json") or {}
        self.questions: List[Dict[str, Any]] = _load_yaml(self.project_dir / "07_gap_questions.yaml") or []
        cfg = _load_yaml(self.project_dir / "06_config_suggestions.yaml") or {}
        self.config_suggestions: List[Dict[str, Any]] = cfg.get("suggestions", []) if isinstance(cfg, dict) else []
        self.mapping_candidates: List[Dict[str, Any]] = _load_json(self.project_dir / "05_mapping_candidates.json") or []
        self.overlap: List[Dict[str, str]] = _load_csv(self.project_dir / "04_source_overlap_analysis.csv")
        self.inventory: List[Dict[str, Any]] = _load_json(self.project_dir / "01_file_inventory.json") or []

    @property
    def source_file_names(self) -> set:
        return {i.get("file_name", "") for i in self.inventory}

    def question_by_id(self) -> Dict[str, Dict[str, Any]]:
        return {q.get("question_id"): q for q in self.questions}

    def config_by_field(self) -> Dict[str, Dict[str, Any]]:
        return {c.get("field"): c for c in self.config_suggestions}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _parse_date(value: str) -> Optional[str]:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(str(value).strip(), fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return None


def _validate_answer(question: Dict[str, Any], answer_obj: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (is_valid, reason)."""
    if not isinstance(answer_obj, dict) or "answer" not in answer_obj:
        return False, "answer object missing 'answer' field"
    answer = answer_obj["answer"]
    if answer in (None, ""):
        return False, "empty answer"

    answer_s = str(answer).strip()
    category = question.get("category", "")
    candidates = question.get("candidate_answers", []) or []

    if category == "date":
        if _parse_date(answer_s) is None:
            return False, f"answer '{answer_s}' is not a valid date"
        return True, ""

    if category == "enum":
        if answer_s not in ENUM_DECISION_ACTIONS:
            return False, f"enum decision '{answer_s}' is not a valid action"
        return True, ""

    if category == "geography":
        if answer_s not in _VALID_GEOGRAPHY_ANSWERS:
            return False, f"geography answer '{answer_s}' is not a known option"
        return True, ""

    if category == "source_of_truth":
        # Must refer to a candidate (a valid source file).
        if candidates and answer_s not in candidates:
            return False, f"source '{answer_s}' is not one of the candidate sources"
        return True, ""

    # Generic / config: must match candidates where they exist.
    if candidates and answer_s not in candidates:
        return False, f"answer '{answer_s}' is not among candidate answers"
    return True, ""


# ---------------------------------------------------------------------------
# Artefact builders
# ---------------------------------------------------------------------------


def _overlap_columns_for(ctx: ProjectContext, canonical: str, primary_file: str):
    """Return (primary_col, secondary_file, secondary_col, match_rate) for a field."""
    for row in ctx.overlap:
        if row.get("canonical_candidate") != canonical:
            continue
        fa, ca = row.get("source_file_a", ""), row.get("source_column_a", "")
        fb, cb = row.get("source_file_b", ""), row.get("source_column_b", "")
        try:
            match_rate = float(row.get("sample_match_rate", 0) or 0)
        except ValueError:
            match_rate = 0.0
        if fa == primary_file:
            return ca, fb, cb, match_rate
        if fb == primary_file:
            return cb, fa, ca, match_rate
    return "", "", "", 0.0


def _build_approved_config(ctx: ProjectContext, answered: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cfg = ctx.config_by_field()

    def val(field, default=""):
        return cfg.get(field, {}).get("suggested_value", default)

    # Reporting date from an answered date question if present.
    reporting_date = val("reporting_date")
    uk_geo_mode = "GBZZZ"
    for q in ctx.questions:
        ans = answered.get(q.get("question_id"))
        if not ans:
            continue
        a = str(ans["answer"]).strip()
        if q.get("category") == "date" and q.get("subject") == "reporting_date":
            parsed = _parse_date(a)
            if parsed:
                reporting_date = parsed
        elif q.get("category") == "geography" and q.get("subject") == "uk_geography_mode":
            uk_geo_mode = a
        elif q.get("category") == "config" and q.get("subject"):
            cfg.setdefault(q["subject"], {})["suggested_value"] = a

    # classification_year is sourced from policy — NEVER from the reporting date.
    classification_year = val("classification_year", "2021")

    return {
        "_warning": "Approved (review-first) onboarding config. Not a production config.",
        "client_name": val("client_name", ctx.run_summary.get("client_name", "")),
        "asset_class": val("asset_class"),
        "portfolio_type": val("portfolio_type", val("asset_class")),
        "currency": val("currency"),
        "jurisdiction": val("jurisdiction"),
        "reporting_date": reporting_date,
        "data_cut_off_date": val("data_cut_off_date", reporting_date),
        "regime": val("regime"),
        "classification_year": classification_year,
        "geography_policy": {
            "ESMA_Annex2": {"uk_geography_mode": uk_geo_mode},
            "MI": {"region_display_field": "collateral_geography"},
        },
    }


def _build_source_precedence(ctx: ProjectContext, answered: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rules: Dict[str, Any] = {}
    for q in ctx.questions:
        if q.get("category") != "source_of_truth":
            continue
        ans = answered.get(q.get("question_id"))
        if not ans:
            continue
        canonical = q.get("subject", "")
        primary_file = str(ans["answer"]).strip()
        primary_col, sec_file, sec_col, match_rate = _overlap_columns_for(ctx, canonical, primary_file)
        rules[canonical] = {
            "primary_source_file": primary_file,
            "primary_source_column": primary_col,
            "secondary_source_file": sec_file,
            "secondary_source_column": sec_col,
            "reconciliation_status": "matched" if match_rate >= 0.999 else "review_required",
            "approved_by": ans.get("approved_by", ""),
        }
    return rules


def _build_enum_decisions(ctx: ProjectContext, answered: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    decisions: Dict[str, Any] = {}
    for q in ctx.questions:
        if q.get("category") != "enum":
            continue
        ans = answered.get(q.get("question_id"))
        if not ans:
            continue
        field = q.get("subject", "")
        raw_value = q.get("subject_value", "")
        decisions.setdefault(field, {})[raw_value] = {
            "decision": str(ans["answer"]).strip(),
            "approved_by": ans.get("approved_by", ""),
            "note": ans.get("note", ""),
        }
    return decisions


def _build_mapping_overrides(ctx: ProjectContext, answered: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    # v1: carry the high-confidence deterministic mappings as the approved
    # baseline plus any explicit user mapping corrections (none expected yet).
    approved = [
        {
            "source_file": m.get("source_file"),
            "source_column": m.get("source_column"),
            "canonical_field": m.get("candidate_canonical_field"),
            "confidence": m.get("confidence"),
            "method": m.get("method"),
        }
        for m in ctx.mapping_candidates
        if m.get("candidate_canonical_field") and float(m.get("confidence", 0) or 0) >= 0.92
    ]
    return {
        "_warning": "Approved (review-first) mapping overrides. Not applied to production.",
        "user_overrides": [],
        "approved_high_confidence_mappings": approved,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def ingest_answers(project_dir: str | Path, answers_path: str | Path) -> Dict[str, Any]:
    """Validate answers and write the approved onboarding artefacts."""
    project_dir = Path(project_dir)
    ctx = ProjectContext(project_dir)

    raw = _load_yaml(Path(answers_path)) or {}
    answers: Dict[str, Dict[str, Any]] = raw.get("answers", {}) if isinstance(raw, dict) else {}

    questions = ctx.questions
    q_by_id = ctx.question_by_id()
    blocking_ids = [q["question_id"] for q in questions if q.get("severity") == "blocking"]

    answered: Dict[str, Dict[str, Any]] = {}
    invalid: List[Dict[str, str]] = []
    unanswered: List[str] = []

    for q in questions:
        qid = q.get("question_id")
        ans = answers.get(qid)
        if ans is None:
            unanswered.append(qid)
            continue
        ok, reason = _validate_answer(q, ans)
        if ok:
            answered[qid] = ans
        else:
            invalid.append({"question_id": qid, "reason": reason})

    blocking_answered = [qid for qid in blocking_ids if qid in answered]
    blocking_unanswered = [qid for qid in blocking_ids if qid not in answered]
    blocking_invalid = [i for i in invalid if i["question_id"] in blocking_ids]

    if invalid:
        approval_status = STATUS_INVALID
    elif blocking_unanswered or blocking_invalid:
        approval_status = STATUS_BLOCKED
    elif len(answered) == len(questions):
        approval_status = STATUS_READY
    else:
        # All blocking resolved, some non-blocking remain.
        approval_status = STATUS_READY if not blocking_unanswered else STATUS_ANSWERED

    # ---- Build & write approved artefacts ----
    generated_at = datetime.utcnow().isoformat() + "Z"
    approved_by = next(
        (a.get("approved_by") for a in answered.values() if a.get("approved_by")), ""
    )

    approved_project = {
        "project_id": ctx.run_summary.get("project_id", project_dir.name),
        "client_name": ctx.run_summary.get("client_name", ""),
        "approval_status": approval_status,
        "answered_questions": sorted(answered.keys()),
        "unresolved_questions": sorted(unanswered + [i["question_id"] for i in invalid]),
        "blocking_status": {
            "total": len(blocking_ids),
            "answered": len(blocking_answered),
            "unanswered": sorted(blocking_unanswered),
        },
        "generated_at": generated_at,
        "approved_by": approved_by,
    }

    approved_config = _build_approved_config(ctx, answered)
    mapping_overrides = _build_mapping_overrides(ctx, answered)
    precedence = _build_source_precedence(ctx, answered)
    enum_decisions = _build_enum_decisions(ctx, answered)

    artefacts = {
        "10_approved_onboarding_project.yaml": approved_project,
        "11_approved_config.yaml": approved_config,
        "12_approved_mapping_overrides.yaml": mapping_overrides,
        "13_source_precedence_rules.yaml": precedence,
        "14_enum_review_decisions.yaml": enum_decisions,
    }
    written: List[str] = []
    for name, payload in artefacts.items():
        (project_dir / name).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        written.append(name)

    report = {
        "questions_total": len(questions),
        "blocking_total": len(blocking_ids),
        "blocking_answered": len(blocking_answered),
        "answers_invalid": len(invalid),
        "invalid_detail": invalid,
        "unanswered": sorted(unanswered),
        "approval_status": approval_status,
        "artefacts_written": written + ["15_answer_ingestion_report.json"],
        "generated_at": generated_at,
    }
    (project_dir / "15_answer_ingestion_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # PART 7 — reflect the approval status back into the static review pack.
    try:
        from .review_pack_builder import refresh_review_pack_approval

        refresh_review_pack_approval(project_dir)
    except Exception:
        pass

    return report
