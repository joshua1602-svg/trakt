"""
gap_analyzer.py
==============

PART 8 — turn detected gaps and ambiguities into user-facing questions.

It compares available mapped / configured data against the minimum requirements
for canonical loan reporting, the MI Agent, ESMA Annex 2 projection, warehouse
funding analytics and pipeline MI, and emits :class:`GapQuestion` objects.

Question categories produced:
  * date            — conflicting reporting dates
  * source_of_truth — ambiguous authoritative source (from overlap findings)
  * enum            — values that are not valid canonical enums
  * config          — missing / unresolved mandatory config
  * geography       — ESMA UK geography policy confirmation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .onboarding_models import (
    ColumnProfile,
    ConfigSuggestion,
    FileInventoryItem,
    GapQuestion,
    OverlapFinding,
)

# Mandatory config that must be resolved before a clean pipeline handoff.
_MANDATORY_CONFIG = ["client_name", "asset_class", "currency", "reporting_date", "regime"]

# Enum-like columns we actively quality-check, with valid canonical tokens.
_KNOWN_ENUM_FIELDS = {
    "employment_status": {
        "EMPLOYED", "SELF_EMPLOYED", "UNEMPLOYED", "RETIRED", "STUDENT",
        "ND1", "ND5", "OTHR",
    },
}
_ENUM_NAME_HINTS = ("status", "type", "purpose", "channel", "flag")


def _detect_conflicting_dates(profiles: List[ColumnProfile]) -> List[str]:
    dates = set()
    for p in profiles:
        if p.likely_reporting_date:
            if p.date_min:
                dates.add(p.date_min)
            if p.date_max:
                dates.add(p.date_max)
    return sorted(dates)


def _enum_questions(
    inventory: List[FileInventoryItem],
    dataframes: Dict[str, pd.DataFrame],
    start_idx: int,
) -> List[GapQuestion]:
    questions: List[GapQuestion] = []
    idx = start_idx
    seen: set = set()
    for item in inventory:
        df = dataframes.get(item.file_path)
        if df is None:
            continue
        for col in df.columns:
            col_norm = str(col).lower().replace(" ", "_")
            if col_norm not in _KNOWN_ENUM_FIELDS and not any(
                h in col_norm for h in _ENUM_NAME_HINTS
            ):
                continue
            series = df[col].dropna().astype(str)
            if series.empty or series.nunique() > 20:
                continue
            values = set(series.unique())
            valid = _KNOWN_ENUM_FIELDS.get(col_norm)
            for val in sorted(values):
                key = (col_norm, val)
                if key in seen:
                    continue
                suspicious = False
                if valid is not None:
                    suspicious = val.strip().upper() not in valid
                else:
                    # Heuristic: lowercase value among otherwise upper-cased codes.
                    upper_siblings = [v for v in values if v.isupper()]
                    suspicious = (
                        val.islower()
                        and len(upper_siblings) >= 1
                        and val.strip().upper() not in {v.upper() for v in upper_siblings}
                    )
                if suspicious:
                    seen.add(key)
                    idx += 1
                    questions.append(
                        GapQuestion(
                            question_id=f"Q{idx}",
                            category="enum",
                            severity="high",
                            question=(
                                f'Should "{val}" in {col_norm} map to a canonical enum '
                                f"value, or be treated as missing?"
                            ),
                            reason=f'Value "{val}" is not a recognised {col_norm} enum.',
                            candidate_answers=["ND1", "OTHR", "treat as missing", "add as new alias"],
                            default_recommendation="OTHR",
                            blocking_for=["ESMA_Annex2"],
                            source_evidence=f"{item.file_name}:{col}",
                        )
                    )
    return questions


def analyze_gaps(
    inventory: List[FileInventoryItem],
    profiles: List[ColumnProfile],
    overlap: List[OverlapFinding],
    config_suggestions: List[ConfigSuggestion],
    dataframes: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[GapQuestion]:
    """Generate gap questions from the accumulated onboarding evidence."""
    dataframes = dataframes or {}
    questions: List[GapQuestion] = []
    idx = 0

    # --- Conflicting reporting dates ---
    dates = _detect_conflicting_dates(profiles)
    idx += 1
    if len(dates) > 1:
        questions.append(
            GapQuestion(
                question_id=f"Q{idx}",
                category="date",
                severity="blocking",
                question="What is the authoritative reporting date for this onboarding run?",
                reason=f"Multiple reporting dates were detected: {', '.join(dates)}.",
                candidate_answers=dates,
                default_recommendation=dates[-1],
                blocking_for=["canonical_loan_reporting", "ESMA_Annex2", "MI_Agent"],
                source_evidence="reporting-date columns across sources",
            )
        )
    elif len(dates) == 1:
        questions.append(
            GapQuestion(
                question_id=f"Q{idx}",
                category="date",
                severity="info",
                question=f"Confirm the reporting date for this run is {dates[0]}.",
                reason="A single reporting date was detected across sources.",
                candidate_answers=[dates[0]],
                default_recommendation=dates[0],
                blocking_for=[],
                source_evidence="reporting-date columns",
            )
        )

    # --- Ambiguous authoritative source (overlap) ---
    seen_canon: set = set()
    for f in overlap:
        if f.canonical_candidate in seen_canon:
            continue
        seen_canon.add(f.canonical_candidate)
        idx += 1
        questions.append(
            GapQuestion(
                question_id=f"Q{idx}",
                category="source_of_truth",
                severity="high",
                question=f"Which source should be authoritative for {f.canonical_candidate}?",
                reason=(
                    f"{f.source_file_a} ({f.source_column_a}) and "
                    f"{f.source_file_b} ({f.source_column_b}) both contain "
                    f"{f.canonical_candidate}-like fields with "
                    f"{f.sample_match_rate:.1%} value match."
                ),
                candidate_answers=[f.source_file_a, f.source_file_b],
                default_recommendation=f.recommended_primary_source,
                blocking_for=["canonical_loan_reporting"],
                source_evidence=f"{f.source_file_a}:{f.source_column_a} | {f.source_file_b}:{f.source_column_b}",
            )
        )

    # --- Enum quality ---
    questions.extend(_enum_questions(inventory, dataframes, idx))
    idx = len(questions)  # keep numbering monotonic

    # --- Missing / unresolved mandatory config ---
    by_field: Dict[str, ConfigSuggestion] = {}
    for s in config_suggestions:
        # Keep the highest-confidence suggestion per field.
        if s.field not in by_field or s.confidence > by_field[s.field].confidence:
            by_field[s.field] = s
    for fld in _MANDATORY_CONFIG:
        s = by_field.get(fld)
        if s is None or s.review_status == "missing" or s.confidence < 0.5:
            idx += 1
            questions.append(
                GapQuestion(
                    question_id=f"Q{idx}",
                    category="config",
                    severity="blocking" if s is None or s.review_status == "missing" else "medium",
                    question=f"Please confirm the value for mandatory config '{fld}'.",
                    reason=(
                        "Required config could not be inferred."
                        if s is None
                        else f"Inferred value '{s.suggested_value}' has low confidence ({s.confidence})."
                    ),
                    candidate_answers=[s.suggested_value] if s and s.suggested_value else [],
                    default_recommendation=s.suggested_value if s else "",
                    blocking_for=["pipeline_handoff"],
                    source_evidence=s.evidence if s else "",
                )
            )

    # --- ESMA UK geography policy confirmation ---
    idx += 1
    questions.append(
        GapQuestion(
            question_id=f"Q{idx}",
            category="geography",
            severity="medium",
            question="Confirm ESMA Annex 2 UK geography policy: should RREL11/RREC6 use GBZZZ?",
            reason="Current policy is GBZZZ for ESMA; ITL3 retained for MI/FCA display.",
            candidate_answers=["GBZZZ (ESMA) + ITL3 (MI)", "ITL3 everywhere", "other"],
            default_recommendation="GBZZZ (ESMA) + ITL3 (MI)",
            blocking_for=["ESMA_Annex2"],
            source_evidence="geography projection policy",
        )
    )

    return questions
