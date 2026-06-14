"""
compare_semantic_alignment.py
=============================

PART 2 — semantic-alignment parity audit.

Runs the SAME source headers through both mapping paths and produces a
column-by-column delta so we can *prove* whether the Onboarding Agent regressed
versus the existing Gate 1 semantic alignment engine:

    OLD path : engine.gate_1_alignment.semantic_alignment.HeaderMapper.map_one
    NEW path : engine.onboarding_agent.mapping_proposer.propose_mappings

Outputs (under ``--output-dir``):
    27_semantic_alignment_parity.csv
    27_semantic_alignment_parity.json
    27_semantic_alignment_parity_summary.md

It is read-only/diagnostic: it never mutates production config, aliases or the
registry, and never runs the LLM.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from engine.gate_1_alignment.semantic_alignment import normalise_name
from .field_scope import resolve_field_scope
from .mapping_proposer import propose_mappings
from .mode_policy import load_mode_policy
from .onboarding_models import FileInventoryItem
from .semantic_alignment_adapter import build_header_mapper, is_semantic_tier

# Documented, deliberately-ambiguous terms (see aliases_onboarding_lending.yaml):
# these are intentionally NOT aliased and are routed to review / value matching.
_INTENTIONALLY_AMBIGUOUS = {
    normalise_name(t) for t in ("loan amount", "initial advance", "principal outstanding")
}

# Header concepts that have NO canonical target in the registry yet (KFI /
# application / pipeline fields). For these the correct outcome is to recommend a
# new pipeline canonical field — never to invent a target silently.
_REGISTRY_TARGET_MISSING_HINTS = (
    "kfi", "offer date", "application submitted", "application date", "funds released",
    "gender", "dob", "date of birth", "product", "dpr", "rejection", "peg",
    "facility", "entitlement", "contracted payment", "fees added", "company",
    "loan plan", "loan purpose detail",
)


def _read_headers_and_series(input_file: str | Path) -> Tuple[List[str], Dict[str, pd.Series], pd.DataFrame]:
    p = Path(input_file)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p, low_memory=False)
    headers = [str(c) for c in df.columns]
    return headers, {h: df[h] for h in headers}, df


def _classify_both_unmapped(normalized: str, raw: str) -> Tuple[str, str]:
    """Return (reason_for_difference, recommended_fix) for a both-unmapped column."""
    low = raw.lower()
    if normalized in _INTENTIONALLY_AMBIGUOUS:
        return ("both_unmapped_intentionally_ambiguous",
                "leave to review/value-matching (documented ambiguous term)")
    if any(h in low for h in _REGISTRY_TARGET_MISSING_HINTS):
        return ("both_unmapped_registry_target_missing",
                "registry_target_missing: add a pipeline/KFI canonical field, then alias")
    return ("both_unmapped", "review: confirm whether a canonical target exists")


def run_parity(
    input_file: str | Path,
    registry: str | Path,
    aliases_dir: str | Path,
    mode: str = "regulatory_mi",
    regulatory_reporting_enabled: bool = False,
    source_file_name: str = "",
    portfolio_type: str = "equity_release",
) -> Dict[str, Any]:
    """Compute the old-vs-new parity rows + a summary. Pure (writes no files)."""
    headers, series_by_header, df = _read_headers_and_series(input_file)
    source_file_name = source_file_name or Path(input_file).name

    # --- OLD path: the Gate 1 deterministic semantic alignment engine ---
    mapper, _ = build_header_mapper(registry, aliases_dir, portfolio_type)
    old: Dict[str, Tuple[str, str, float]] = {}
    for h in headers:
        canon, method, conf = mapper.map_one(h)
        old[h] = (canon or "", method, float(conf or 0.0))

    # --- NEW path: the full Onboarding Agent mapping pipeline ---
    policy = load_mode_policy(mode)
    field_scope = resolve_field_scope(
        str(registry), policy, regulatory_reporting_enabled=regulatory_reporting_enabled
    )
    inv = [FileInventoryItem(file_path=str(input_file), file_name=source_file_name,
                             file_type=Path(input_file).suffix.lstrip(".") or "csv",
                             classification="loan_report")]
    cands, oos, _amb = propose_mappings(
        inv, {str(input_file): df}, Path(registry), Path(aliases_dir),
        field_scope=field_scope, regulatory_reporting_enabled=regulatory_reporting_enabled,
    )
    new_by_col = {c.source_column: c for c in cands}
    oos_by_col = {o.get("source_column"): o for o in oos}

    rows: List[Dict[str, Any]] = []
    counts = {"mapped_both": 0, "old_mapped_new_unmapped": 0, "old_mapped_new_different": 0,
              "new_mapped_old_unmapped": 0, "unmapped_both": 0,
              "field_scope_excluded": 0, "semantic_alignment_used": 0}

    for h in headers:
        old_canon, old_method, old_conf = old[h]
        new_cand = new_by_col.get(h)
        oos = oos_by_col.get(h)
        new_canon = new_cand.candidate_canonical_field if new_cand else ""
        new_method = new_cand.method if new_cand else ("out_of_scope" if oos else "unmapped")
        new_conf = float(new_cand.confidence) if new_cand else 0.0
        if oos is not None and not new_canon:
            new_field_scope_status = "out_of_scope"
            new_canon = oos.get("candidate_field", "")
            counts["field_scope_excluded"] += 1
        elif new_cand is not None:
            new_field_scope_status = "in_scope"
        else:
            new_field_scope_status = "unmapped"

        same = bool(old_canon) and old_canon == new_canon and new_field_scope_status == "in_scope"
        old_mapped_new_unmapped = bool(old_canon) and not (new_cand and new_canon and new_field_scope_status == "in_scope")
        old_mapped_new_different = (bool(old_canon) and bool(new_canon)
                                    and old_canon != new_canon)
        new_mapped_old_unmapped = (not old_canon) and bool(new_canon) and new_field_scope_status == "in_scope"

        # Reason + recommended fix.
        if same:
            reason, fix = "same_result", "ok"
            counts["mapped_both"] += 1
        elif old_mapped_new_unmapped and new_field_scope_status == "out_of_scope":
            # NOT a regression: the target is deliberately excluded by the mode
            # field scope. Counted under field_scope_excluded, not regressions.
            reason = "old_mapped_new_out_of_scope"
            fix = "expected: target excluded by mode field scope (mode-safe)"
        elif old_mapped_new_unmapped:
            reason = "old_mapped_new_unmapped"
            fix = "integrate Gate 1 semantic alignment adapter (real gap)"
            counts["old_mapped_new_unmapped"] += 1
        elif old_mapped_new_different:
            reason = "old_mapped_new_different"
            fix = "review mapping difference (context/geography/ambiguity rule)"
            counts["old_mapped_new_different"] += 1
        elif new_mapped_old_unmapped:
            reason = "new_mapped_old_unmapped"
            fix = "ok: new path is stronger here"
            counts["new_mapped_old_unmapped"] += 1
        else:
            reason, fix = _classify_both_unmapped(normalise_name(h), h)
            counts["unmapped_both"] += 1

        sem_used = bool(new_canon) and is_semantic_tier(new_method)
        if sem_used:
            counts["semantic_alignment_used"] += 1

        rows.append({
            "source_file": source_file_name,
            "source_column": h,
            "normalized_column": normalise_name(h),
            "old_semantic_alignment_candidate": old_canon,
            "old_semantic_alignment_method": old_method,
            "old_semantic_alignment_confidence": round(old_conf, 4),
            "new_onboarding_candidate": new_canon,
            "new_onboarding_method": new_method,
            "new_onboarding_confidence": round(new_conf, 4),
            "new_field_scope_status": new_field_scope_status,
            "semantic_alignment_used": sem_used,
            "same_result": same,
            "old_mapped_new_unmapped": old_mapped_new_unmapped and new_field_scope_status != "out_of_scope",
            "old_mapped_new_different": old_mapped_new_different,
            "new_mapped_old_unmapped": new_mapped_old_unmapped,
            "reason_for_difference": reason,
            "recommended_fix": fix,
        })

    summary = {
        "input_file": str(input_file),
        "source_file": source_file_name,
        "mode": mode,
        "columns_total": len(headers),
        "llm_used": False,
        **counts,
    }
    return {"rows": rows, "summary": summary}


_CSV_COLUMNS = [
    "source_file", "source_column", "normalized_column",
    "old_semantic_alignment_candidate", "old_semantic_alignment_method",
    "old_semantic_alignment_confidence", "new_onboarding_candidate",
    "new_onboarding_method", "new_onboarding_confidence", "new_field_scope_status",
    "semantic_alignment_used", "same_result", "old_mapped_new_unmapped",
    "old_mapped_new_different", "new_mapped_old_unmapped", "reason_for_difference",
    "recommended_fix",
]


def write_parity_artifacts(result: Dict[str, Any], output_dir: str | Path) -> Dict[str, str]:
    """Write the 27_* parity artefacts (CSV, JSON, markdown summary)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = result["rows"], result["summary"]

    csv_path = out_dir / "27_semantic_alignment_parity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _CSV_COLUMNS})

    json_path = out_dir / "27_semantic_alignment_parity.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    md_path = out_dir / "27_semantic_alignment_parity_summary.md"
    md_path.write_text(_render_summary_md(rows, summary), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "summary_md": str(md_path)}


def _render_summary_md(rows: List[Dict[str, Any]], s: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Semantic alignment parity audit")
    lines.append("")
    lines.append(f"Input file: `{s['source_file']}` · mode: `{s['mode']}` · "
                 f"columns: {s['columns_total']} · LLM used: {s['llm_used']}")
    lines.append("")
    lines.append("## Headline")
    lines.append(f"- {s['mapped_both']} columns mapped the same by both paths.")
    lines.append(f"- {s['old_mapped_new_unmapped']} mapped by old semantic alignment but "
                 f"NOT by new onboarding (true regressions).")
    lines.append(f"- {s['old_mapped_new_different']} mapped differently by the two paths.")
    lines.append(f"- {s['new_mapped_old_unmapped']} mapped by new onboarding but not old "
                 f"(new is stronger).")
    lines.append(f"- {s['unmapped_both']} unmapped by both.")
    lines.append(f"- {s['field_scope_excluded']} diverted out of scope by the mode field "
                 f"scope (expected, mode-safe).")
    lines.append(f"- {s['semantic_alignment_used']} mapped via a fuzzy semantic tier "
                 f"(token_set / fuzz).")
    lines.append("")

    def table(title: str, predicate) -> None:
        sel = [r for r in rows if predicate(r)]
        if not sel:
            return
        lines.append(f"## {title} ({len(sel)})")
        lines.append("| column | old → | new → | reason | recommended fix |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in sel:
            lines.append(
                f"| {r['source_column']} | {r['old_semantic_alignment_candidate'] or '—'} "
                f"| {r['new_onboarding_candidate'] or '—'} | {r['reason_for_difference']} "
                f"| {r['recommended_fix']} |")
        lines.append("")

    table("True regressions — old mapped, new unmapped",
          lambda r: r["old_mapped_new_unmapped"])
    table("Mapped differently", lambda r: r["old_mapped_new_different"])
    table("Registry target missing (need a new canonical field)",
          lambda r: r["reason_for_difference"] == "both_unmapped_registry_target_missing")
    table("Intentionally ambiguous (left to review by design)",
          lambda r: r["reason_for_difference"] == "both_unmapped_intentionally_ambiguous")
    table("New path stronger", lambda r: r["new_mapped_old_unmapped"])

    lines.append("## Main causes of mismatch")
    if s["old_mapped_new_unmapped"] == 0:
        lines.append("- No true regression: the new onboarding path uses the same Gate 1 "
                     "`HeaderMapper` tier chain as the old path, so it maps everything the "
                     "old path maps (subject to mode field scope).")
    else:
        lines.append("- Some columns mapped by the old engine are not active in the new "
                     "path — integrate the semantic alignment adapter for those.")
    lines.append("- Most unmapped KFI/application/pipeline columns have **no canonical "
                 "target in the registry yet** (registry_target_missing) — add pipeline "
                 "canonical fields rather than inventing targets.")
    lines.append("- A few unmapped columns are **deliberately ambiguous** and routed to "
                 "review/value-matching by design.")
    lines.append("")
    return "\n".join(lines)


def run_and_write(
    input_file: str | Path,
    registry: str | Path,
    aliases_dir: str | Path,
    output_dir: str | Path,
    mode: str = "regulatory_mi",
    regulatory_reporting_enabled: bool = False,
) -> Dict[str, Any]:
    """Convenience: compute parity and write the 27_* artefacts. Returns paths+summary."""
    result = run_parity(input_file, registry, aliases_dir, mode,
                        regulatory_reporting_enabled=regulatory_reporting_enabled)
    paths = write_parity_artifacts(result, output_dir)
    return {"paths": paths, "summary": result["summary"], "rows": result["rows"]}
