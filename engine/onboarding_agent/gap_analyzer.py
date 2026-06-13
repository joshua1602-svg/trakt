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
from typing import Any, Dict, List, Optional

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

# Placeholder / process tokens that are never a meaningful enum value. They must
# be surfaced as unresolved (requires_review) and must NOT be auto-defaulted to a
# canonical code such as OTHR without an explicit human decision.
_UNRESOLVED_ENUM_PLACEHOLDERS = {
    "manual", "unknown", "n/a", "na", "none", "null", "tbc", "tbd",
    "to be confirmed", "requires review", "requires_review", "?", "-", "",
}

# Allowed decision actions an answer may take on an unresolved enum value.
ENUM_DECISION_ACTIONS = [
    "treat_as_missing",
    "map_to_ND1",
    "map_to_OTHR",
    "provide_custom_mapping",
    "exclude_field_from_regulatory_delivery",
]


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
    out_of_scope_columns: Optional[set] = None,
) -> List[GapQuestion]:
    """Generate enum-quality gaps.

    Columns diverted out of scope by the mode field scope
    (``out_of_scope_columns`` = set of ``(file_name, column)``) are skipped: a
    regulatory non-core field that is out of scope for MI-only must not be typed
    as an active enum, validated, or produce an actionable enum gap.
    """
    out_of_scope_columns = out_of_scope_columns or set()
    questions: List[GapQuestion] = []
    idx = start_idx
    seen: set = set()
    for item in inventory:
        df = dataframes.get(item.file_path)
        if df is None:
            continue
        for col in df.columns:
            if (item.file_name, str(col)) in out_of_scope_columns:
                continue  # field is out of scope for this mode — no enum validation
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
                placeholder = val.strip().lower() in _UNRESOLVED_ENUM_PLACEHOLDERS
                if valid is not None:
                    suspicious = placeholder or val.strip().upper() not in valid
                else:
                    # Heuristic: lowercase value among otherwise upper-cased codes,
                    # or an explicit unresolved placeholder token.
                    upper_siblings = [v for v in values if v.isupper()]
                    suspicious = placeholder or (
                        val.islower()
                        and len(upper_siblings) >= 1
                        and val.strip().upper() not in {v.upper() for v in upper_siblings}
                    )
                if not suspicious:
                    continue
                seen.add(key)
                idx += 1
                if placeholder:
                    # Unresolved placeholder: never auto-default to a canonical code.
                    question = (
                        f'The source value "{val}" appears in {col_norm} but is not a '
                        f"recognised {col_norm} enum. How should it be treated?"
                    )
                    reason = (
                        f'"{val}" is a placeholder / process token, not a semantically '
                        f"meaningful {col_norm} value."
                    )
                    candidates = list(ENUM_DECISION_ACTIONS)
                    default = "requires_review"
                else:
                    question = (
                        f'Should "{val}" in {col_norm} map to a canonical enum '
                        f"value, or be treated as missing?"
                    )
                    reason = f'Value "{val}" is not a recognised {col_norm} enum.'
                    candidates = list(ENUM_DECISION_ACTIONS)
                    default = "requires_review"
                questions.append(
                    GapQuestion(
                        question_id=f"Q{idx}",
                        category="enum",
                        severity="high",
                        question=question,
                        reason=reason,
                        candidate_answers=candidates,
                        default_recommendation=default,
                        blocking_for=["ESMA_Annex2"],
                        source_evidence=f"{item.file_name}:{col}",
                        subject=col_norm,
                        subject_value=val,
                    )
                )
    return questions


# Warehouse facility core terms checked for completeness.
_WAREHOUSE_CORE_TERMS = [
    "warehouse_facility_present",
    "advance_rate",
    "margin",
    "warehouse_limit",
    "warehouse_lender_name",
    "interest_index",
]


def _warehouse_gap_questions(
    inventory: List[FileInventoryItem],
    config_suggestions: List[ConfigSuggestion],
    mode: str,
    start_idx: int,
) -> List[GapQuestion]:
    """Flag missing / unresolved warehouse facility terms.

    Only raised when a warehouse objective applies: a warehouse agreement is in
    the data room, or the mode is warehouse_securitisation.
    """
    warehouse_present = any(i.classification == "warehouse_agreement" for i in inventory)
    if not warehouse_present and mode != "warehouse_securitisation":
        return []

    by_field: Dict[str, ConfigSuggestion] = {}
    for s in config_suggestions:
        if s.field not in by_field or s.confidence > by_field[s.field].confidence:
            by_field[s.field] = s

    questions: List[GapQuestion] = []
    idx = start_idx
    for term in _WAREHOUSE_CORE_TERMS:
        s = by_field.get(term)
        missing = s is None or s.review_status == "missing" or s.suggested_value in ("", "unknown")
        if not missing and s.review_status == "suggested" and s.confidence >= 0.7:
            continue  # confidently extracted — no gap
        idx += 1
        base_severity = "blocking" if missing else "high"
        questions.append(
            GapQuestion(
                question_id=f"Q{idx}",
                category="warehouse",
                severity=base_severity,
                question=f"Confirm the warehouse facility term '{term}'.",
                reason=(
                    "Warehouse core term is missing from the data room."
                    if missing
                    else f"Extracted value '{s.suggested_value}' requires review."
                ),
                candidate_answers=[s.suggested_value] if s and s.suggested_value not in ("", "unknown") else [],
                default_recommendation=s.suggested_value if s and not missing else "requires_review",
                blocking_for=["warehouse_analytics", "securitisation_readiness"],
                source_evidence=s.source_file if s else "warehouse agreement",
                subject=term,
            )
        )
    return questions


# Actions a reviewer can take on a missing / unmapped core field.
CORE_FIELD_ACTIONS = [
    "select_source_column",
    "provide_mapping_override",
    "mark_unavailable",
    "not_applicable",
]


def _missing_core_field_questions(field_scope, mapping_candidates, start_idx: int):
    """Emit answerable gaps for in-scope core_canonical fields with no mapping.

    Severity is ``blocking`` when the field is in the mode's blocking set, else
    ``high`` (visible but non-blocking — e.g. non-structural core fields in
    mna_dd). Out-of-scope / regulatory non-core fields never produce these.
    """
    if field_scope is None:
        return []
    covered = {
        m.candidate_canonical_field for m in (mapping_candidates or [])
        if m.candidate_canonical_field
    }
    in_scope_core = field_scope.included_fields & field_scope.core_canonical_fields
    questions: List[GapQuestion] = []
    idx = start_idx
    for fname in sorted(in_scope_core - covered):
        idx += 1
        blocking = fname in field_scope.blocking_fields
        questions.append(
            GapQuestion(
                question_id=f"Q{idx}",
                category="core_field",
                severity="blocking" if blocking else "high",
                question=(
                    f"The core canonical field {fname} is missing or unmapped. "
                    f"Which source column should be used, or should it be marked unavailable?"
                ),
                reason=f"Core canonical field '{fname}' has no mapping candidate.",
                candidate_answers=list(CORE_FIELD_ACTIONS),
                default_recommendation="mark_unavailable",
                blocking_for=["canonical_loan_reporting"] if blocking else [],
                source_evidence="field scope (core_canonical=true, unmapped)",
                subject=fname,
            )
        )
    return questions


def analyze_gaps(
    inventory: List[FileInventoryItem],
    profiles: List[ColumnProfile],
    overlap: List[OverlapFinding],
    config_suggestions: List[ConfigSuggestion],
    dataframes: Optional[Dict[str, pd.DataFrame]] = None,
    mode_policy=None,
    field_scope=None,
    out_of_scope_fields: Optional[List[Dict[str, Any]]] = None,
    mapping_candidates: Optional[List[Any]] = None,
) -> List[GapQuestion]:
    """Generate gap questions, then re-rank severity for the onboarding mode.

    ``mode_policy`` is an optional :class:`mode_policy.ModePolicy`. When omitted
    the legacy (regulatory) severities are returned unchanged. ``field_scope``
    (a :class:`field_scope.FieldScopeResult`) drives out-of-scope summaries and
    mode-scoped regulatory gap suppression. ``out_of_scope_fields`` is the list of
    diverted (file, column) records so their enums are not validated.
    """
    dataframes = dataframes or {}
    out_of_scope_columns = {
        (o.get("source_file"), o.get("source_column"))
        for o in (out_of_scope_fields or [])
    }
    mode = getattr(mode_policy, "name", "regulatory_mi")
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
                subject="reporting_date",
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
                subject="reporting_date",
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
                subject=f.canonical_candidate,
            )
        )

    # --- Enum quality (skip out-of-scope fields) ---
    questions.extend(_enum_questions(inventory, dataframes, idx, out_of_scope_columns))
    idx = len(questions)  # keep numbering monotonic

    # --- Missing / unresolved mandatory config (mode-scoped) ---
    mandatory_config = (
        list(mode_policy.required_config_fields)
        if mode_policy is not None and mode_policy.required_config_fields
        else _MANDATORY_CONFIG
    )
    by_field: Dict[str, ConfigSuggestion] = {}
    for s in config_suggestions:
        # Keep the highest-confidence suggestion per field.
        if s.field not in by_field or s.confidence > by_field[s.field].confidence:
            by_field[s.field] = s
    for fld in mandatory_config:
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
                    subject=fld,
                )
            )

    # --- ESMA UK geography policy confirmation (only when regime is in scope) ---
    regime_in_scope = mode_policy is None or mode_policy.regime_config_required
    if regime_in_scope:
        idx += 1
        questions.append(
            GapQuestion(
                question_id=f"Q{idx}",
                category="geography",
                severity="medium",
                question="Confirm ESMA Annex 2 UK geography policy: should RREL11/RREC6 use GBZZZ?",
                reason="Current policy is GBZZZ for ESMA; ITL3 retained for MI/FCA display.",
                candidate_answers=["GBZZZ", "ITL3", "other"],
                default_recommendation="GBZZZ",
                blocking_for=["ESMA_Annex2"],
                source_evidence="geography projection policy",
                subject="uk_geography_mode",
            )
        )

    # --- Warehouse facility terms ---
    questions.extend(_warehouse_gap_questions(inventory, config_suggestions, mode, len(questions)))

    # --- Missing in-scope core canonical fields (answerable) ---
    questions.extend(_missing_core_field_questions(field_scope, mapping_candidates, len(questions)))

    # --- Out-of-scope summary (regulatory fields excluded by mode field scope) ---
    if field_scope is not None:
        excluded_reg = field_scope.excluded_fields & field_scope.regulatory_fields
        if excluded_reg and "regulatory" in getattr(mode_policy, "exclude_categories", []):
            idx = len(questions) + 1
            questions.append(
                GapQuestion(
                    question_id=f"Q{idx}",
                    category="scope",
                    severity="info",
                    question=(
                        f"{len(excluded_reg)} regulatory fields are out of scope for "
                        f"mode '{mode}' and were excluded from mapping/type requirements."
                    ),
                    reason="Regulatory category fields are not required in this mode.",
                    candidate_answers=[],
                    default_recommendation="acknowledged",
                    blocking_for=[],
                    source_evidence="field scope (registry category=regulatory)",
                    subject="out_of_scope_regulatory_fields",
                    subject_value=str(len(excluded_reg)),
                )
            )

    # --- Mode-aware severity re-ranking (PART 2) ---
    if mode_policy is not None:
        for q in questions:
            q.severity = mode_policy.severity_for(q.category, q.severity)

    # Drop questions whose mode severity marks them out of scope.
    questions = [q for q in questions if q.severity != "out_of_scope"]
    return questions
