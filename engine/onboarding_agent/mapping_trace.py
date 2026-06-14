"""
mapping_trace.py
================

Explainability / audit for the Onboarding Agent's DETERMINISTIC-FIRST mapping
path. For every source column it records exactly how the mapping result was
reached, so it is obvious that the agent leverages existing Trakt pipeline
assets (Python profiling → field registry → alias libraries → deterministic
scoring → value matching → source precedence) and only falls back to the LLM
for unresolved ambiguity.

It does not perform any new mapping. It re-reads the SAME deterministic engine
(``MappingProposer`` / Gate-1 ``HeaderMapper`` / the loaded alias libraries) and
the artefacts the run already produced (05 candidates, 05b ambiguities, overlap
analysis, field scope, LLM suggestions) and renders the decision trail.

Artefacts:
    05c_mapping_trace.csv / .json   one row per source column
    05d_mapping_explanation.md      short, human-readable summary

Selection reasons (how the decision was reached):
    approved_override · alias_match · registry_match · header_similarity ·
    domain_context · value_match · regulatory_preference_rule ·
    source_precedence · llm_suggestion · user_gap_required
"""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from engine.gate_1_alignment.semantic_alignment import normalise_name
from . import domain_coverage as dc
from .file_profiler import redact_value

# method (from MappingCandidate) -> selection reason category.
_METHOD_REASON = {
    "exact": "registry_match",
    "normalized": "registry_match",
    "alias": "alias_match",
    "token_set": "header_similarity",
    "fuzz_token_set": "header_similarity",
    "fuzz_ratio_norm": "header_similarity",
    "context_hint": "domain_context",
    "geography_display_guard": "domain_context",
    "unmapped": "user_gap_required",
    "empty": "user_gap_required",
}

_TRACE_COLUMNS = [
    "source_file", "source_sheet", "source_column", "normalized_column",
    "detected_domain", "profile_type", "sample_values_redacted",
    "alias_files_loaded", "alias_hit", "alias_source_file", "alias_matched_text",
    "alias_target_field", "registry_field_exists", "candidate_fields",
    "candidate_scores", "field_scope_status", "selected_candidate",
    "selection_reason", "ambiguity_rule_applied", "value_match_evidence",
    "source_precedence_evidence", "llm_used", "llm_suggestion_used",
    "final_status", "unmapped_reason",
]


@dataclass
class AliasIndex:
    """Per-file alias provenance (which alias library a synonym came from)."""

    files_loaded: List[str] = field(default_factory=list)
    # normalised alias text -> (canonical_field, source_file, original_text)
    by_norm: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, aliases_dir: str | Path) -> "AliasIndex":
        aliases_dir = Path(aliases_dir)
        idx = cls()
        if not aliases_dir.exists():
            return idx
        for yaml_file in sorted(aliases_dir.glob("aliases_*.yaml")):
            try:
                data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            idx.files_loaded.append(yaml_file.name)
            for canon, meta in data.items():
                aliases = meta if isinstance(meta, list) else (meta or {}).get("aliases", []) or []
                for a in aliases:
                    norm = normalise_name(a)
                    if norm and norm not in idx.by_norm:
                        idx.by_norm[norm] = (canon, yaml_file.name, str(a))
        return idx

    def lookup(self, column: str):
        """Return (canonical, source_file, matched_text) or None for a column."""
        return self.by_norm.get(normalise_name(column))


# ---------------------------------------------------------------------------


def _overlap_value_evidence(canonical: str, overlap: List[Any]) -> str:
    """Summarise value-match evidence for a canonical field from overlap analysis."""
    rows = [o for o in overlap if getattr(o, "canonical_candidate", "") == canonical]
    if not rows:
        return ""
    parts = []
    for o in rows:
        rate = getattr(o, "sample_match_rate", 0.0)
        primary = getattr(o, "recommended_primary_source", "")
        sec = getattr(o, "recommended_secondary_source", "")
        conflict = " conflict_gap_unless_precedence" if rate < 0.999 else ""
        parts.append(
            f"value_match_rate={rate:.0%}; primary={primary}; validation={sec}{conflict}"
        )
    return " | ".join(parts)


def _precedence_evidence(canonical: str, precedence: Dict[str, Any]) -> str:
    rule = (precedence or {}).get(canonical)
    if not rule:
        return ""
    return (f"approved primary={rule.get('primary_source_file','')}:"
            f"{rule.get('primary_source_column','')} "
            f"({rule.get('reconciliation_status','')})")


def build_trace(
    inventory: List[Any],
    dataframes: Dict[str, pd.DataFrame],
    mapping_candidates: List[Any],
    out_of_scope_fields: List[Dict[str, Any]],
    mapping_ambiguities: List[Any],
    overlap_analysis: List[Any],
    field_scope: Any,
    registry_fields: Dict[str, Any],
    aliases_dir: str | Path,
    llm_suggestions: Optional[List[Dict[str, Any]]] = None,
    precedence: Optional[Dict[str, Any]] = None,
    profiles: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Build the per-column mapping trace + a summary. Pure read/reconstruct."""
    alias_index = AliasIndex.load(aliases_dir)
    included = getattr(field_scope, "included_fields", set()) or set()

    cand_by_key = {(m.source_file, m.source_column): m for m in mapping_candidates}
    oos_by_key = {(o.get("source_file"), o.get("source_column")): o for o in out_of_scope_fields}
    amb_by_key = {(a.source_file, a.source_column): a for a in mapping_ambiguities}
    inv_by_name = {i.file_name: i for i in inventory}
    profile_by_key = {}
    for p in (profiles or []):
        profile_by_key[(p.file_name, p.source_column)] = p

    llm_columns = set()
    for s in (llm_suggestions or []):
        sf, sc = s.get("source_file"), s.get("source_column")
        if sf and sc:
            llm_columns.add((sf, sc))

    # Map canonical field -> domains for detected_domain.
    rows: List[Dict[str, Any]] = []
    summary = {
        "alias_files_loaded": alias_index.files_loaded,
        "registry_fields_count": len(registry_fields),
        "mapped_by_alias": 0,
        "mapped_by_registry_header": 0,
        "mapped_by_value_or_context": 0,
        "out_of_scope": 0,
        "ambiguous_needs_review": 0,
        "unmapped": 0,
        "sent_to_llm": 0,
    }

    for item in inventory:
        df = dataframes.get(item.file_path)
        if df is None:
            continue
        for col in df.columns:
            col = str(col)
            key = (item.file_name, col)
            cand = cand_by_key.get(key)
            oos = oos_by_key.get(key)
            amb = amb_by_key.get(key)
            prof = profile_by_key.get(key)

            alias_hit_info = alias_index.lookup(col)
            samples = []
            if prof is not None:
                samples = list(prof.sample_values_redacted)[:3]
            elif col in df.columns:
                samples = [redact_value(v) for v in df[col].dropna().drop_duplicates().head(3).tolist()]

            selected = cand.candidate_canonical_field if cand else ""
            method = cand.method if cand else ("unmapped" if not oos else "")
            reason = _METHOD_REASON.get(method, "")
            ambiguity_rule = (cand.ambiguity_rule_applied if cand else "") or (
                amb.ambiguity_rule_applied if amb else "")
            if ambiguity_rule:
                reason = "regulatory_preference_rule"

            # Candidate ranking evidence (deterministic).
            candidate_fields, candidate_scores = [], []
            if cand:
                candidate_fields.append(cand.candidate_canonical_field or "")
                candidate_scores.append(round(cand.confidence, 4))
                for alt in (cand.alternative_candidates or []):
                    candidate_fields.append(alt.get("field", ""))
                    candidate_scores.append(alt.get("confidence", ""))

            # Field-scope status + final status.
            value_ev = _overlap_value_evidence(selected, overlap_analysis) if selected else ""
            prec_ev = _precedence_evidence(selected, precedence or {}) if selected else ""
            if value_ev and reason in ("", "user_gap_required"):
                reason = reason or "value_match"

            if oos is not None:
                scope_status = "out_of_scope"
                final_status = "out_of_scope"
                unmapped_reason = oos.get("reason", "excluded by mode field scope")
                reason = reason or "user_gap_required"
                summary["out_of_scope"] += 1
            elif selected:
                scope_status = "in_scope" if (not included or selected in included) else "out_of_scope"
                if cand and cand.requires_review and (ambiguity_rule or (cand.confidence < 0.92)):
                    final_status = "ambiguous_needs_review"
                    summary["ambiguous_needs_review"] += 1
                else:
                    final_status = "mapped"
                    if reason == "alias_match":
                        summary["mapped_by_alias"] += 1
                    elif reason == "registry_match" or reason == "header_similarity":
                        summary["mapped_by_registry_header"] += 1
                    else:
                        summary["mapped_by_value_or_context"] += 1
                unmapped_reason = ""
            else:
                scope_status = "in_scope"
                final_status = "unmapped"
                summary["unmapped"] += 1
                if alias_hit_info is None:
                    unmapped_reason = "unmapped_alias_missing: no alias/registry/context match"
                else:
                    unmapped_reason = "unmapped_low_confidence"
                reason = "user_gap_required"

            llm_used = key in llm_columns
            if llm_used:
                summary["sent_to_llm"] += 1

            rows.append({
                "source_file": item.file_name,
                "source_sheet": item.sheet_name,
                "source_column": col,
                "normalized_column": normalise_name(col),
                "detected_domain": "; ".join(
                    dc.field_domains(selected, registry_fields.get(selected, {})) if selected
                    else dc._column_domains(col)
                ),
                "profile_type": getattr(prof, "inferred_type", "") if prof else "",
                "sample_values_redacted": "; ".join(str(s) for s in samples),
                "alias_files_loaded": "; ".join(alias_index.files_loaded),
                "alias_hit": bool(alias_hit_info),
                "alias_source_file": alias_hit_info[1] if alias_hit_info else "",
                "alias_matched_text": alias_hit_info[2] if alias_hit_info else "",
                "alias_target_field": alias_hit_info[0] if alias_hit_info else "",
                "registry_field_exists": bool(selected and selected in registry_fields),
                "candidate_fields": "; ".join(str(c) for c in candidate_fields),
                "candidate_scores": "; ".join(str(s) for s in candidate_scores),
                "field_scope_status": scope_status,
                "selected_candidate": selected,
                "selection_reason": reason,
                "ambiguity_rule_applied": ambiguity_rule,
                "value_match_evidence": value_ev,
                "source_precedence_evidence": prec_ev,
                "llm_used": llm_used,
                "llm_suggestion_used": False,  # suggestions are NEVER promoted to final
                "final_status": final_status,
                "unmapped_reason": unmapped_reason,
            })

    summary["columns_total"] = len(rows)
    return {"rows": rows, "summary": summary, "alias_index": alias_index}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_trace_artifacts(trace: Dict[str, Any], out_dir: str | Path) -> List[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = trace["rows"]

    csv_path = out_dir / "05c_mapping_trace.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_TRACE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in _TRACE_COLUMNS})

    json_path = out_dir / "05c_mapping_trace.json"
    json_path.write_text(
        json.dumps({"summary": trace["summary"], "rows": rows}, indent=2, default=str),
        encoding="utf-8",
    )
    return [str(csv_path), str(json_path)]


def write_explanation_report(
    trace: Dict[str, Any], out_dir: str | Path, mode: str, client_name: str = "",
) -> str:
    out_dir = Path(out_dir)
    s = trace["summary"]
    rows = trace["rows"]

    def example(filter_fn, limit=4):
        out = []
        for r in rows:
            if filter_fn(r):
                out.append(r)
            if len(out) >= limit:
                break
        return out

    alias_examples = example(lambda r: r["selection_reason"] == "alias_match")
    value_examples = example(lambda r: r["value_match_evidence"] and r["final_status"] == "mapped")
    review_examples = example(lambda r: r["final_status"] == "ambiguous_needs_review")
    unmapped_examples = example(lambda r: r["final_status"] == "unmapped")

    lines: List[str] = []
    lines.append(f"# Onboarding mapping explanation — {client_name or 'run'}")
    lines.append("")
    lines.append("_Deterministic-first mapping. Python profiling → field registry → alias "
                 "libraries → deterministic scoring → value matching → source precedence. "
                 "The LLM only reviews unresolved ambiguity and never writes final mappings._")
    lines.append("")
    lines.append(
        f"The Onboarding Agent loaded {len(s['alias_files_loaded'])} alias files "
        f"({', '.join(s['alias_files_loaded']) or 'none'}) and "
        f"{s['registry_fields_count']} registry fields.")
    lines.append(
        f"It mapped {s['mapped_by_alias']} source columns by alias, "
        f"{s['mapped_by_registry_header']} by registry/header similarity, and "
        f"{s['mapped_by_value_or_context']} by value matching/context.")
    lines.append(
        f"{s['out_of_scope']} columns were excluded as out-of-scope for `{mode}` mode, "
        f"{s['ambiguous_needs_review']} are ambiguous and require review, and "
        f"{s['unmapped']} remained unmapped.")
    lines.append(f"{s['sent_to_llm']} columns were sent to the LLM.")
    lines.append("")

    def block(title, examples, render):
        if not examples:
            return
        lines.append(f"## {title}")
        for r in examples:
            lines.append(f"- {render(r)}")
        lines.append("")

    block("Mapped by existing alias", alias_examples,
          lambda r: (f'Source column "{r["source_column"]}" → '
                     f'`{r["selected_candidate"]}` (alias in {r["alias_source_file"]}, '
                     f'matched "{r["alias_matched_text"]}").'))
    block("Supported by value matching / validation sources", value_examples,
          lambda r: (f'`{r["selected_candidate"]}` from "{r["source_column"]}" — '
                     f'{r["value_match_evidence"]}.'))
    block("Ambiguous — require review", review_examples,
          lambda r: (f'Source column "{r["source_column"]}" → candidate '
                     f'`{r["selected_candidate"]}` '
                     f'({r["ambiguity_rule_applied"] or "low confidence"}); flagged for review.'))
    block("Unmapped", unmapped_examples,
          lambda r: f'Source column "{r["source_column"]}" — {r["unmapped_reason"]}.')

    path = out_dir / "05d_mapping_explanation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
