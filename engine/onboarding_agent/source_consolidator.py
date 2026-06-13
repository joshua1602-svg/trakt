"""
source_consolidator.py
======================

PART 5 — candidate keys + source overlap / duplication analysis.

Two responsibilities:

  1. :func:`detect_candidate_keys` — find columns that look like join / business
     keys (loan identifier, collateral identifier, reporting date, ...).

  2. :func:`analyze_overlap` — detect where the *same business field* appears in
     more than one source extract (e.g. ``current_balance`` in the loan report
     and ``principal_outstanding`` in the cashflow report), using the canonical
     mapping candidates plus value-level sample matching on a shared key.

Nothing is merged or deleted — recommendations only.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from engine.gate_1_alignment.semantic_alignment import normalise_name, tokenize
from .onboarding_models import (
    CandidateKey,
    ColumnProfile,
    FileInventoryItem,
    MappingCandidate,
    OverlapFinding,
)

# Candidate key concepts and the tokens that identify them.
_KEY_CONCEPTS: Dict[str, List[str]] = {
    "loan_identifier": ["loan_id", "loan_identifier", "loanid", "account_number", "account_id"],
    "underlying_exposure_identifier": ["underlying_exposure_identifier", "exposure_id", "exposure_identifier"],
    "account_number": ["account_number", "account_no", "account"],
    "contract_id": ["contract_id", "contract_number", "agreement_id"],
    "collateral_identifier": ["collateral_id", "collateral_identifier", "security_id", "property_id"],
    "borrower_identifier": ["borrower_id", "obligor_id", "borrower_identifier", "obligor_identifier", "customer_id"],
    "reporting_date": ["reporting_date", "report_date", "data_cut_off_date", "as_of_date", "as_at_date"],
}


def _match_key_concept(col_name: str) -> Optional[str]:
    norm = str(col_name).lower().replace(" ", "_")
    norm_tokens = normalise_name(col_name)
    for concept, signals in _KEY_CONCEPTS.items():
        for sig in signals:
            if sig == norm or sig in norm or normalise_name(sig) == norm_tokens:
                return concept
    return None


def detect_candidate_keys(profiles: List[ColumnProfile]) -> List[CandidateKey]:
    """Identify candidate join / business keys from column profiles."""
    keys: List[CandidateKey] = []
    for p in profiles:
        concept = _match_key_concept(p.source_column)
        is_id_like = p.likely_identifier or p.likely_reporting_date
        if concept is None and not is_id_like:
            continue

        denom = p.non_null_count or 1
        uniqueness = round(p.unique_count / denom, 3) if denom else 0.0

        if concept is None:
            # Generic identifier with no named concept — only keep reporting dates
            # or columns whose name strongly implies an identifier key.
            name = p.source_column.lower()
            strong_id = name.endswith(("_id", "_identifier", "_number")) or any(
                t in name for t in ("account", "contract")
            )
            if p.likely_reporting_date:
                concept = "reporting_date"
            elif strong_id:
                concept = "other_identifier"
            else:
                continue
            confidence = 0.4
        else:
            # Named concept; confidence higher when also unique / id-like.
            confidence = 0.9 if (is_id_like or concept == "reporting_date") else 0.6

        keys.append(
            CandidateKey(
                candidate_key=concept,
                file_path=p.file_path,
                file_name=p.file_name,
                source_column=p.source_column,
                unique_count=p.unique_count,
                null_rate=p.null_rate,
                uniqueness_ratio=uniqueness,
                confidence=confidence,
                notes="reporting date" if p.likely_reporting_date else "",
            )
        )
    return keys


def _loan_key_column(file_name: str, candidate_keys: List[CandidateKey]) -> Optional[str]:
    """Return the loan-identifier column name for a file, if any."""
    for k in candidate_keys:
        if k.file_name == file_name and k.candidate_key in (
            "loan_identifier",
            "account_number",
            "underlying_exposure_identifier",
        ):
            return k.source_column
    return None


def _sample_match_rate(
    df_a: pd.DataFrame,
    col_a: str,
    df_b: pd.DataFrame,
    col_b: str,
    key_a: Optional[str],
    key_b: Optional[str],
) -> float:
    """Fraction of rows where the two columns agree, joined on a shared key."""
    if key_a is None or key_b is None or key_a not in df_a.columns or key_b not in df_b.columns:
        return 0.0
    if col_a == key_a or col_b == key_b:
        # The "value" column is the join key itself — nothing meaningful to compare.
        return 0.0
    try:
        left = df_a[[key_a, col_a]].dropna().drop_duplicates(subset=[key_a])
        left.columns = ["_key", col_a]
        right = df_b[[key_b, col_b]].dropna().drop_duplicates(subset=[key_b])
        right.columns = ["_key", col_b]
        merged = left.merge(right, on="_key", how="inner")
        if merged.empty:
            return 0.0

        a_num = pd.to_numeric(merged[col_a], errors="coerce")
        b_num = pd.to_numeric(merged[col_b], errors="coerce")
        numeric_mask = a_num.notna() & b_num.notna()
        matches = 0
        for i in range(len(merged)):
            if numeric_mask.iloc[i]:
                av, bv = a_num.iloc[i], b_num.iloc[i]
                denom = max(abs(av), abs(bv), 1.0)
                if abs(av - bv) / denom <= 0.01:
                    matches += 1
            else:
                if str(merged[col_a].iloc[i]).strip() == str(merged[col_b].iloc[i]).strip():
                    matches += 1
        return round(matches / len(merged), 3)
    except Exception:
        return 0.0


def _header_similarity(col_a: str, col_b: str) -> float:
    ta, tb = set(tokenize(col_a)), set(tokenize(col_b))
    if not ta or not tb:
        return 0.0
    return round(len(ta & tb) / len(ta | tb), 3)


# Source-of-truth priority by classification for a given business field.
_PRIMARY_PRIORITY = {
    "current_principal_balance": ["current_loan_report", "cashflow_report", "collateral_report"],
    "current_valuation_amount": ["collateral_report", "current_loan_report"],
}


def _recommend_sources(canonical: str, cls_a: str, cls_b: str, file_a: str, file_b: str):
    priority = _PRIMARY_PRIORITY.get(canonical, [])
    rank = {cls: i for i, cls in enumerate(priority)}
    ra = rank.get(cls_a, 99)
    rb = rank.get(cls_b, 99)
    if ra <= rb:
        return file_a, file_b
    return file_b, file_a


def analyze_overlap(
    inventory: List[FileInventoryItem],
    mapping_candidates: List[MappingCandidate],
    candidate_keys: List[CandidateKey],
    dataframes: Dict[str, pd.DataFrame],
) -> List[OverlapFinding]:
    """Detect columns across different files that map to the same canonical field."""
    cls_by_file = {i.file_name: i.classification for i in inventory}
    path_by_file = {i.file_name: i.file_path for i in inventory}

    # Group mapped columns by canonical field.
    by_canonical: Dict[str, List[MappingCandidate]] = {}
    for mc in mapping_candidates:
        if not mc.candidate_canonical_field:
            continue
        by_canonical.setdefault(mc.candidate_canonical_field, []).append(mc)

    # Keys / identifiers are expected to recur across files (join keys) — they
    # are handled by candidate-key detection, not treated as duplicate fields.
    _KEY_CANONICALS = {
        "loan_identifier", "collateral_identifier", "borrower_identifier",
        "obligor_identifier", "account_number", "contract_id",
        "underlying_exposure_identifier", "reporting_date", "data_cut_off_date",
    }

    findings: List[OverlapFinding] = []
    for canonical, cands in by_canonical.items():
        if canonical in _KEY_CANONICALS:
            continue
        # Only interesting when the field appears in >1 distinct file.
        files = {c.source_file for c in cands}
        if len(files) < 2:
            continue
        # Pairwise across different files.
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                if a.source_file == b.source_file:
                    continue
                df_a = dataframes.get(path_by_file.get(a.source_file, ""))
                df_b = dataframes.get(path_by_file.get(b.source_file, ""))
                key_a = _loan_key_column(a.source_file, candidate_keys)
                key_b = _loan_key_column(b.source_file, candidate_keys)
                match_rate = 0.0
                if df_a is not None and df_b is not None:
                    match_rate = _sample_match_rate(
                        df_a, a.source_column, df_b, b.source_column, key_a, key_b
                    )
                header_sim = _header_similarity(a.source_column, b.source_column)
                # Combined similarity: agreement on canonical + header + values.
                similarity = round(max(header_sim, 0.6) * 0.4 + match_rate * 0.6, 3)

                primary, secondary = _recommend_sources(
                    canonical,
                    cls_by_file.get(a.source_file, ""),
                    cls_by_file.get(b.source_file, ""),
                    a.source_file,
                    b.source_file,
                )
                findings.append(
                    OverlapFinding(
                        canonical_candidate=canonical,
                        source_file_a=a.source_file,
                        source_column_a=a.source_column,
                        source_file_b=b.source_file,
                        source_column_b=b.source_column,
                        similarity_score=similarity,
                        sample_match_rate=match_rate,
                        recommended_primary_source=primary,
                        recommended_secondary_source=secondary,
                        review_required=True,
                        reason=(
                            f"Both sources map to '{canonical}'. "
                            f"Value match rate on shared key = {match_rate:.1%}."
                            if match_rate
                            else f"Both sources map to '{canonical}' (no shared key to value-check)."
                        ),
                    )
                )
    return findings
