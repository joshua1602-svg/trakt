"""
column_evidence.py
==================

PART 3 — deterministic per-column evidence packs.

For every uploaded source column we build a compact, redacted evidence object:
type/like-scores, null/distinct stats, numeric/date ranges, cross-column
chronology + amount relationships, and deterministic candidate target matches
(client memory, alias, semantic alignment, pipeline contract, registry,
value-profile). These packs are the ONLY thing the LLM mapping reviewer ever
sees — never the full raw file.

Artefacts:
    29_column_evidence.csv / .json / _summary.md
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from engine.gate_1_alignment.semantic_alignment import normalise_name
from . import domain_coverage as dc
from .file_profiler import redact_value
from .pipeline_field_contract import RAW_TO_PIPELINE_FIELD, pipeline_contract_field_names

_UK_POSTCODE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$", re.I)
_STAGE_TOKENS = {"kfi", "application", "offer", "completed", "complete", "funded",
                 "withdrawn", "underwriting", "rejected", "pending", "approved",
                 "declined", "referred", "live", "in progress"}
_GENDER_TOKENS = {"m", "f", "male", "female", "man", "woman", "nonbinary", "other"}
_BOOL_TOKENS = {"y", "n", "yes", "no", "true", "false", "0", "1", "t", "f"}


def _num(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[£$€,\s%]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        # ISO first (the common case); coerce handles non-dates and overflow.
        out = pd.to_datetime(s, errors="coerce", format="ISO8601")
        if out.notna().mean() < 0.5:
            out = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return out


def _frac(mask: pd.Series, denom: int) -> float:
    return round(float(mask.sum()) / denom, 4) if denom else 0.0


def _like_scores(series: pd.Series) -> Dict[str, float]:
    s = series.dropna().astype(str).str.strip()
    n = len(s)
    if n == 0:
        return {k: 0.0 for k in (
            "currency_like_score", "percentage_like_score", "identifier_like_score",
            "enum_like_score", "free_text_like_score", "boolean_like_score",
            "postcode_like_score", "gender_like_score", "amount_like_score",
            "rate_like_score", "stage_like_score")}
    low = s.str.lower()
    nums = _num(series).dropna()
    num_frac = len(nums) / n
    distinct = s.nunique()
    has_pct = _frac(s.str.contains("%"), n)
    has_ccy = _frac(s.str.contains(r"[£$€]"), n)
    amount_like = num_frac * _frac(nums.abs() > 100, len(nums)) if len(nums) else 0.0
    rate_like = (num_frac * _frac((nums >= 0) & (nums <= 15) & (nums != nums.round()),
                                  len(nums))) if len(nums) else 0.0
    pct_like = max(has_pct, (num_frac * _frac((nums >= 0) & (nums <= 100), len(nums))
                             if len(nums) else 0.0))
    # Code-like identifier: alphanumeric AND contains a digit (ACC0001, KFI0001),
    # which separates keys from pure-word enums/stages (Completed, Offer).
    code_like = s.str.match(r"^[A-Za-z0-9_\-/]+$") & s.str.contains(r"\d")
    ident_like = (distinct / n) * _frac(code_like, n)
    bool_like = _frac(low.isin(_BOOL_TOKENS), n)
    gender_like = _frac(low.isin(_GENDER_TOKENS), n)
    postcode_like = _frac(s.str.match(_UK_POSTCODE), n)
    stage_like = _frac(low.apply(lambda v: any(t in v for t in _STAGE_TOKENS)), n)
    enum_like = 1.0 if (distinct <= 20 and num_frac < 0.5) else (0.5 if distinct <= 20 else 0.0)
    avg_len = s.str.len().mean()
    free_text_like = min(1.0, (avg_len / 40.0)) * _frac(s.str.contains(" "), n) \
        if distinct > max(20, 0.6 * n) else 0.0
    return {
        "currency_like_score": round(max(has_ccy, amount_like if has_ccy else 0.0), 4),
        "percentage_like_score": round(pct_like, 4),
        "identifier_like_score": round(min(1.0, ident_like), 4),
        "enum_like_score": round(enum_like, 4),
        "free_text_like_score": round(free_text_like, 4),
        "boolean_like_score": round(bool_like, 4),
        "postcode_like_score": round(postcode_like, 4),
        "gender_like_score": round(gender_like, 4),
        "amount_like_score": round(amount_like, 4),
        "rate_like_score": round(rate_like, 4),
        "stage_like_score": round(stage_like, 4),
    }


def _type_guess(series: pd.Series, like: Dict[str, float]) -> str:
    nums = _num(series).dropna()
    dates = _date(series).dropna()
    n = max(1, series.notna().sum())
    if len(dates) / n >= 0.8 and len(nums) / n < 0.8:
        return "date"
    if like["boolean_like_score"] >= 0.8:
        return "boolean"
    if like["postcode_like_score"] >= 0.6:
        return "postcode"
    # Stage/status enums win over identifier — but a code-like identifier
    # (e.g. KFI0001, which contains the 'kfi' stage token as a substring) must
    # not be misread as a stage.
    if (like["stage_like_score"] >= 0.6 and len(nums) / n < 0.5
            and like["identifier_like_score"] < 0.7):
        return "enum"
    if len(nums) / n >= 0.8:
        if like["rate_like_score"] >= 0.5:
            return "rate"
        if like["percentage_like_score"] >= 0.6 and like["amount_like_score"] < 0.4:
            return "percentage"
        if like["amount_like_score"] >= 0.4:
            return "amount"
        return "numeric"
    if like["identifier_like_score"] >= 0.7:
        return "identifier"
    if like["enum_like_score"] >= 1.0:
        return "enum"
    if like["free_text_like_score"] >= 0.4:
        return "free_text"
    return "string"


def _chronology(df: pd.DataFrame, date_cols: List[str]) -> Dict[str, List[str]]:
    """For each date col, list other date cols it precedes (>=80% of populated rows)."""
    parsed = {c: _date(df[c]) for c in date_cols}
    rel: Dict[str, List[str]] = {c: [] for c in date_cols}
    for a in date_cols:
        for b in date_cols:
            if a == b:
                continue
            both = parsed[a].notna() & parsed[b].notna()
            denom = int(both.sum())
            if denom < 2:
                continue
            le = ((parsed[a] <= parsed[b]) & both).sum()
            if le / denom >= 0.8:
                rel[a].append(f"{a}<= {b} ({le}/{denom})")
    return rel


def _amount_relationships(df: pd.DataFrame, amount_cols: List[str]) -> Dict[str, List[str]]:
    nums = {c: _num(df[c]) for c in amount_cols}
    rel: Dict[str, List[str]] = {c: [] for c in amount_cols}
    for a in amount_cols:
        for b in amount_cols:
            if a == b:
                continue
            both = nums[a].notna() & nums[b].notna()
            denom = int(both.sum())
            if denom < 2:
                continue
            le = ((nums[a] <= nums[b]) & both).sum()
            if le / denom >= 0.8:
                rel[a].append(f"{a}<= {b} ({le}/{denom})")
    return rel


def build_column_evidence(
    df: pd.DataFrame,
    source_file: str,
    registry_fields: Optional[Dict[str, Any]] = None,
    field_scope: Any = None,
    semantic_mapper: Any = None,
    alias_index: Any = None,
    memory_store: Any = None,
    sheet_name: str = "",
) -> List[Dict[str, Any]]:
    """Build a compact evidence pack for every column in ``df``."""
    registry_fields = registry_fields or {}
    cols = [str(c) for c in df.columns]

    # Pre-classify columns for cross-column relationships.
    like_by_col = {c: _like_scores(df[c]) for c in cols}
    type_by_col = {c: _type_guess(df[c], like_by_col[c]) for c in cols}
    date_cols = [c for c in cols if type_by_col[c] == "date"]
    amount_cols = [c for c in cols if type_by_col[c] in ("amount", "numeric")]
    chrono = _chronology(df, date_cols)
    amt_rel = _amount_relationships(df, amount_cols)
    pipeline_fields = set(pipeline_contract_field_names())

    rows: List[Dict[str, Any]] = []
    for c in cols:
        series = df[c]
        non_null = series.dropna()
        n = len(series)
        like = like_by_col[c]
        tguess = type_by_col[c]
        nums = _num(series).dropna()
        dates = _date(series).dropna()
        distinct = int(non_null.astype(str).nunique())
        samples = [redact_value(v) for v in non_null.astype(str).head(5).tolist()]
        distinct_samples = [redact_value(v) for v in
                            non_null.astype(str).drop_duplicates().head(8).tolist()]

        # Deterministic candidate matches.
        norm = normalise_name(c)
        sem_match = ""
        if semantic_mapper is not None:
            from .semantic_alignment_adapter import align_header
            r = align_header(semantic_mapper, c, field_scope=field_scope)
            if r.candidate:
                sem_match = f"{r.candidate} ({r.method} {r.confidence:.2f})"
        alias_match = ""
        if alias_index is not None:
            hit = alias_index.lookup(c)
            if hit:
                alias_match = f"{hit[0]} (alias:{hit[2]})"
        pipeline_target = RAW_TO_PIPELINE_FIELD.get(c, "")
        if not pipeline_target and norm.replace(" ", "_") in pipeline_fields:
            pipeline_target = norm.replace(" ", "_")
        registry_match = c if c in registry_fields else (
            norm.replace(" ", "_") if norm.replace(" ", "_") in registry_fields else "")
        memory_match = ""
        if memory_store is not None:
            from .mapping_memory import file_matches, normalize_column
            for e in memory_store.entries:
                if (e.canonical_field and file_matches(e.source_file_pattern, source_file)
                        and normalize_column(c) == (e.normalized_source_column
                                                    or normalize_column(e.source_column))):
                    memory_match = f"{e.canonical_field} ({e.decision_type})"
                    break

        rows.append({
            "source_file": source_file,
            "source_sheet": sheet_name,
            "source_column": c,
            "normalized_column": norm,
            "domain_guess": "; ".join(dc._column_domains(c)) or "unknown",
            "file_domain_guess": "pipeline" if pipeline_target else "unknown",
            "sample_values_redacted": "; ".join(samples),
            "sample_values_distinct_redacted": "; ".join(distinct_samples),
            "data_type_guess": tguess,
            "null_count": int(n - len(non_null)),
            "null_rate": round((n - len(non_null)) / n, 4) if n else 0.0,
            "distinct_count": distinct,
            "uniqueness_ratio": round(distinct / len(non_null), 4) if len(non_null) else 0.0,
            "min_value": str(non_null.min()) if len(non_null) else "",
            "max_value": str(non_null.max()) if len(non_null) else "",
            "mean_value_if_numeric": round(float(nums.mean()), 4) if len(nums) else "",
            "median_value_if_numeric": round(float(nums.median()), 4) if len(nums) else "",
            "date_parse_rate": round(len(dates) / len(non_null), 4) if len(non_null) else 0.0,
            "min_date": str(dates.min().date()) if len(dates) else "",
            "max_date": str(dates.max().date()) if len(dates) else "",
            **like,
            "nearby_columns": "; ".join([x for x in cols if x != c][:6]),
            "chronology_relationships": "; ".join(chrono.get(c, [])),
            "amount_relationships": "; ".join(amt_rel.get(c, [])),
            "candidate_existing_registry_fields": registry_match,
            "candidate_existing_pipeline_contract_fields": pipeline_target,
            "candidate_alias_matches": alias_match,
            "candidate_semantic_alignment_matches": sem_match,
            "candidate_value_profile_matches": _value_profile_hint(tguess, like),
            "out_of_scope_candidates": "",
            "known_client_memory_matches": memory_match,
        })
    return rows


def _value_profile_hint(tguess: str, like: Dict[str, float]) -> str:
    hints = []
    if like["rate_like_score"] >= 0.5:
        hints.append("rate (interest)")
    if like["percentage_like_score"] >= 0.6 and like["amount_like_score"] < 0.3:
        hints.append("percentage (servicing/ratio)")
    if like["amount_like_score"] >= 0.4:
        hints.append("monetary amount")
    if like["stage_like_score"] >= 0.5:
        hints.append("pipeline stage/status")
    if like["postcode_like_score"] >= 0.5:
        hints.append("postcode")
    if like["gender_like_score"] >= 0.5:
        hints.append("gender")
    if tguess == "date":
        hints.append("date/milestone")
    if tguess == "identifier":
        hints.append("identifier/key")
    return "; ".join(hints)


_EVIDENCE_COLUMNS = [
    "source_file", "source_sheet", "source_column", "normalized_column",
    "domain_guess", "file_domain_guess", "sample_values_redacted",
    "sample_values_distinct_redacted", "data_type_guess", "null_count", "null_rate",
    "distinct_count", "uniqueness_ratio", "min_value", "max_value",
    "mean_value_if_numeric", "median_value_if_numeric", "date_parse_rate",
    "min_date", "max_date", "currency_like_score", "percentage_like_score",
    "identifier_like_score", "enum_like_score", "free_text_like_score",
    "boolean_like_score", "postcode_like_score", "gender_like_score",
    "amount_like_score", "rate_like_score", "stage_like_score", "nearby_columns",
    "chronology_relationships", "amount_relationships",
    "candidate_existing_registry_fields", "candidate_existing_pipeline_contract_fields",
    "candidate_alias_matches", "candidate_semantic_alignment_matches",
    "candidate_value_profile_matches", "out_of_scope_candidates",
    "known_client_memory_matches",
]


def write_evidence_artifacts(rows: List[Dict[str, Any]], output_dir: str | Path) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "29_column_evidence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_EVIDENCE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _EVIDENCE_COLUMNS})
    json_path = out_dir / "29_column_evidence.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    md_path = out_dir / "29_column_evidence_summary.md"
    md_path.write_text(_render_md(rows), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "summary_md": str(md_path)}


def _render_md(rows: List[Dict[str, Any]]) -> str:
    lines = ["# Column evidence summary", "",
             f"{len(rows)} source columns profiled (deterministic, redacted).", ""]
    lines.append("| column | type | distinct | null% | profile hints | chronology |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(f"| {r['source_column']} | {r['data_type_guess']} | {r['distinct_count']} "
                     f"| {r['null_rate']:.0%} | {r['candidate_value_profile_matches'] or '—'} "
                     f"| {r['chronology_relationships'] or '—'} |")
    return "\n".join(lines) + "\n"
