"""
llm_assisted_mapping.py
=======================

Orchestrates the controlled LLM-assisted mapping workbench (PARTS 2–9):

    pipeline contract (28) -> column evidence (29) -> candidate shortlist (30)
      -> schema drift (37) -> [optional] LLM reviewer (31)
      -> deterministic backstop (32) -> concise review queue (33/34)

Deterministic-first and LLM-off by default: the LLM is only invoked when an
``llm_callable`` is supplied AND ``enable_llm`` is set. Everything the LLM sees
is a compact, redacted evidence pack — never the raw file. The LLM never
finalises a mapping; the backstop validator and user approval do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from engine.gate_1_alignment.semantic_alignment import load_field_registry

from . import (
    column_evidence as ce,
    mapping_backstop_validator as backstop,
    mapping_candidate_finder as finder,
    mapping_review_queue as queue,
    pipeline_field_contract as contract,
    schema_drift as drift,
)
from .field_scope import resolve_field_scope
from .llm_mapping_controller import LLMMappingController, write_llm_review_artifacts
from .mapping_trace import AliasIndex
from .mode_policy import load_mode_policy
from .semantic_alignment_adapter import build_header_mapper

# Priority order for choosing the single deterministic candidate per column.
# pipeline_contract outranks alias/semantic so a pipeline/KFI column (e.g.
# "Status") maps to its pipeline field (status_raw) rather than being polluted
# into a coincidental regulatory/funded alias (account_status).
_SOURCE_RANK = {"client_memory": 0, "pipeline_contract": 1, "alias": 2,
                "semantic_alignment": 3, "registry_description": 4, "value_profile": 5}


def _best_candidate(cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    have = [c for c in cands if c.get("candidate_target_field")]
    if not have:
        return None
    return sorted(have, key=lambda c: (_SOURCE_RANK.get(c["candidate_source"], 9),
                                       -float(c["candidate_confidence"])))[0]


def _ek(row: Dict[str, Any]):
    return (row.get("source_file", ""), row.get("source_sheet", ""),
            row.get("source_column", ""))


def run_llm_assisted_mapping(
    input_file: Optional[str | Path] = None,
    df: Optional[pd.DataFrame] = None,
    dataframes: Optional[Dict[str, pd.DataFrame]] = None,
    inventory: Optional[List[Dict[str, Any]]] = None,
    output_dir: str | Path = ".",
    registry_path: str | Path = "config/system/fields_registry.yaml",
    aliases_dir: str | Path = "config/system",
    mode: str = "regulatory_mi",
    regulatory_reporting_enabled: bool = False,
    client_id: str = "",
    run_id: str = "",
    source_file_name: str = "",
    enable_llm: bool = False,
    llm_callable: Optional[Callable[[str], str]] = None,
    only_unresolved: bool = False,
    memory_store: Any = None,
    memory_dir: Optional[str | Path] = None,
    max_llm_items: int = 60,
    max_cost_gbp: float = 1.0,
    enable_file_conversion_fallback: bool = False,
) -> Dict[str, Any]:
    """Run the full controlled mapping workbench pipeline and write artefacts.

    Input precedence: ``inventory`` (a full data room — every file is parsed via
    the robust multi-sheet loader and a coverage record is emitted) > ``dataframes``
    > single ``df``/``input_file``. Evidence + relationships are built per
    (file, sheet); the shortlist/backstop/queue operate across ALL columns using
    composite (file, sheet, column) keys so same-named columns never collide.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the input table(s) + coverage.
    tables: List[Tuple[str, str, pd.DataFrame]] = []
    coverage: List[Dict[str, Any]] = []
    sheet_coverage: List[Dict[str, Any]] = []
    if inventory is not None:
        from . import source_table_loader as stl
        loaded, cov, sheets = stl.load_source_tables(
            inventory, enable_conversion=enable_file_conversion_fallback)
        tables = [(t.file_name, t.sheet_name, t.df) for t in loaded]
        coverage = [c.__dict__ for c in cov]
        sheet_coverage = [s.__dict__ for s in sheets]
    elif dataframes:
        tables = [(str(k), "", v) for k, v in dataframes.items()]
    elif df is not None:
        tables = [(source_file_name or "uploaded.csv", "", df)]
    else:
        if input_file is None:
            raise ValueError("provide inventory, dataframes, df, or input_file")
        p = Path(input_file)
        loaded_df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") \
            else pd.read_csv(p, low_memory=False)
        tables = [(source_file_name or p.name, "", loaded_df)]

    policy = load_mode_policy(mode)
    field_scope = resolve_field_scope(str(registry_path), policy,
                                      regulatory_reporting_enabled=regulatory_reporting_enabled)
    registry_fields = load_field_registry(Path(registry_path)).get("fields", {}) or {}
    mapper, _ = build_header_mapper(registry_path, aliases_dir)
    alias_index = AliasIndex.load(aliases_dir)

    # 28 — existing Pipeline MI field contract.
    contract_rows = contract.build_pipeline_field_contract(registry_path)
    contract.write_contract_artifacts(contract_rows, out_dir)

    # 29 — deterministic column evidence packs (per file AND per sheet).
    evidence_rows: List[Dict[str, Any]] = []
    for fname, sheet, fdf in tables:
        evidence_rows += ce.build_column_evidence(
            fdf, fname, registry_fields=registry_fields, field_scope=field_scope,
            semantic_mapper=mapper, alias_index=alias_index, memory_store=memory_store,
            sheet_name=sheet)
    ce.write_evidence_artifacts(evidence_rows, out_dir)
    evidence_by_key = {_ek(e): e for e in evidence_rows}

    # MI-relevant regulatory fields stay in scope for the REVIEW under mi_only /
    # mna_dd (they feed MI/static-pools). The regulatory funded-tape scope is
    # unchanged — this only affects how the review classifies them.
    extra_in_scope = (finder.MI_RELEVANT_FIELDS
                      if mode in ("mi_only", "mna_dd") else set())

    # 30 — deterministic candidate shortlists (composite-keyed).
    shortlist_rows = finder.build_candidate_shortlist(
        evidence_rows, registry_fields, field_scope, extra_in_scope=extra_in_scope)
    finder.write_shortlist_artifacts(shortlist_rows, out_dir)
    shortlist_by_key = finder.shortlist_by_key(shortlist_rows)

    # 37 — schema drift vs the client's previous signature.
    prev_sig = drift.load_signature(memory_dir) if memory_dir else None
    drift_rows = drift.detect_drift(evidence_rows, prev_sig)
    drift.write_drift_artifacts(drift_rows, out_dir)
    if memory_dir:
        drift.save_signature(drift.build_signature(evidence_rows), memory_dir)
    drift_review_cols = drift.columns_needing_review(drift_rows)

    # Columns lacking a confident deterministic mapping (LLM focus set).
    unresolved_cols = set()
    for e in evidence_rows:
        best = _best_candidate(shortlist_by_key.get(_ek(e), []))
        if best is None or float(best["candidate_confidence"]) < 0.95:
            unresolved_cols.add(e["source_column"])
    if only_unresolved:
        unresolved_cols |= drift_review_cols

    # 31 — controlled LLM reviewer (off unless enabled + callable provided). The
    # controller keys proposals by source_column; we map them back per (file,sheet).
    llm_result = {"proposals": [], "usage": {"llm_enabled": False, "calls_completed": 0,
                                             "estimated_cost_gbp": 0.0}}
    if enable_llm and llm_callable is not None:
        controller = LLMMappingController(
            llm_callable=llm_callable, registry_fields=registry_fields,
            field_scope=field_scope, max_items=max_llm_items, max_cost_gbp=max_cost_gbp)
        llm_result = controller.review(
            evidence_rows, finder.shortlist_by_column(shortlist_rows),
            only_unresolved=unresolved_cols if only_unresolved else None)
    if llm_result["usage"].get("llm_enabled") or llm_result["proposals"]:
        write_llm_review_artifacts(llm_result, out_dir)
    llm_by_col = {p["source_column"]: p for p in llm_result["proposals"]}
    llm_by_key = {_ek(e): llm_by_col[e["source_column"]]
                  for e in evidence_rows if e["source_column"] in llm_by_col}

    # Merge into one proposal per (file, sheet, column): deterministic best wins;
    # fall back to the LLM proposal where deterministic found nothing.
    proposals: List[Dict[str, Any]] = []
    for e in evidence_rows:
        key = _ek(e)
        col, src_file, src_sheet = e["source_column"], e.get("source_file", ""), e.get("source_sheet", "")
        best = _best_candidate(shortlist_by_key.get(key, []))
        llm = llm_by_key.get(key)
        if best is not None:
            proposals.append({**best, "source_file": src_file, "source_sheet": src_sheet})
        elif llm is not None and llm.get("proposed_target_field"):
            proposals.append({
                "source_file": src_file, "source_sheet": src_sheet, "source_column": col,
                "proposed_target_field": llm["proposed_target_field"],
                "candidate_source": "llm_suggested", "confidence": llm["confidence"],
                "ambiguity_flags": llm.get("ambiguity_flags", []),
                "alternative_targets": llm.get("alternative_targets", []),
                "is_pipeline_field": False})
        else:
            proposals.append({
                "source_file": src_file, "source_sheet": src_sheet, "source_column": col,
                "proposed_target_field": "", "candidate_source":
                ("pipeline_contract" if e.get("candidate_existing_pipeline_contract_fields")
                 else "value_profile"), "confidence": "no_match"})

    # 32 — deterministic backstop validation.
    validation_rows = backstop.validate_mappings(
        proposals, registry_fields=registry_fields, field_scope=field_scope,
        memory_store=memory_store, evidence_by_col=evidence_by_key,
        extra_in_scope=extra_in_scope)
    backstop.write_validation_artifacts(validation_rows, out_dir)

    # 33/34 — concise multi-file review queue.
    review = queue.build_review_queue(validation_rows, evidence_by_key, llm_by_key)
    queue.write_queue_artifacts(review, out_dir)

    # 29a/29b — per-file coverage + per-sheet parse coverage (explicit diagnostics).
    coverage = _finalise_coverage(coverage, evidence_rows, shortlist_rows,
                                  validation_rows, review["items"])
    _write_coverage_artifacts(coverage, out_dir)
    _write_sheet_coverage(sheet_coverage, out_dir)

    queue.append_review_action_log(
        out_dir, client_id, run_id, "run_llm_assisted_mapping",
        inputs={"files": len(coverage) or len(tables), "mode": mode,
                "llm_enabled": bool(llm_result["usage"].get("llm_enabled"))},
        outputs=["28..37 artefacts"], status="ok")

    return {
        "contract": contract_rows,
        "evidence": evidence_rows,
        "shortlist": shortlist_rows,
        "drift": drift_rows,
        "llm": llm_result,
        "validation": validation_rows,
        "review_queue": review,
        "file_coverage": coverage,
        "sheet_coverage": sheet_coverage,
        "summary": {**review["summary"], "file_coverage": _coverage_summary(coverage)},
    }


_COVERAGE_COLUMNS = [
    "file_name", "file_path", "file_type", "classification", "domains_detected",
    "declared_extension", "detected_container_type", "detected_excel_format",
    "extension_mismatch_detected", "parser_attempted", "engine_used",
    "attempted_column_evidence", "column_evidence_rows", "included_in_candidate_shortlist",
    "candidate_shortlist_rows", "included_in_backstop_validation", "backstop_rows",
    "included_in_review_queue", "review_queue_rows", "parse_status", "parse_error",
    "reason_excluded", "conversion_available", "conversion_tool", "conversion_attempted",
    "conversion_status", "conversion_error", "converted_file_path",
    "recommended_next_action",
]

_PASS_THROUGH = [
    "file_name", "file_path", "file_type", "classification", "domains_detected",
    "declared_extension", "detected_container_type", "detected_excel_format",
    "extension_mismatch_detected", "parser_attempted", "engine_used",
    "attempted_column_evidence", "parse_status", "parse_error", "reason_excluded",
    "conversion_available", "conversion_tool", "conversion_attempted",
    "conversion_status", "conversion_error", "converted_file_path",
    "recommended_next_action",
]


def _finalise_coverage(coverage, evidence_rows, shortlist_rows, validation_rows, queue_items):
    """Fill per-file counts into the coverage records (29a). Never blanks a reason."""
    def counts(rows):
        out: Dict[str, int] = {}
        for r in rows:
            out[r.get("source_file", "")] = out.get(r.get("source_file", ""), 0) + 1
        return out
    ev_c, sl_c, val_c, q_c = (counts(evidence_rows), counts(shortlist_rows),
                              counts(validation_rows), counts(queue_items))
    out = []
    for c in coverage:
        name = c.get("file_name", "")
        row = {k: c.get(k, "") for k in _PASS_THROUGH}
        # A parse_error row must never have a blank error or next action.
        if row.get("parse_status") == "parse_error" and not row.get("parse_error"):
            row["parse_error"] = "unknown parse failure"
        if row.get("parse_status") in ("parse_error", "unsupported_file_type",
                                       "dependency_missing") and not row.get("recommended_next_action"):
            row["recommended_next_action"] = "open in Excel/LibreOffice and resave as .xlsx or .csv"
        row.update({
            "column_evidence_rows": ev_c.get(name, 0),
            "included_in_candidate_shortlist": sl_c.get(name, 0) > 0,
            "candidate_shortlist_rows": sl_c.get(name, 0),
            "included_in_backstop_validation": val_c.get(name, 0) > 0,
            "backstop_rows": val_c.get(name, 0),
            "included_in_review_queue": q_c.get(name, 0) > 0,
            "review_queue_rows": q_c.get(name, 0),
        })
        out.append(row)
    return out


def _write_coverage_artifacts(rows, out_dir: Path):
    import csv as _csv
    import json as _json
    csv_path = out_dir / "29a_column_evidence_file_coverage.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=_COVERAGE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _COVERAGE_COLUMNS})
    (out_dir / "29a_column_evidence_file_coverage.json").write_text(
        _json.dumps(rows, indent=2, default=str), encoding="utf-8")


_SHEET_COLUMNS = ["file_name", "declared_extension", "detected_container_type",
                  "sheet_name", "parse_status", "rows", "columns", "engine_used",
                  "parse_error"]


def _write_sheet_coverage(rows, out_dir: Path):
    import csv as _csv
    import json as _json
    csv_path = out_dir / "29b_excel_sheet_parse_coverage.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=_SHEET_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _SHEET_COLUMNS})
    (out_dir / "29b_excel_sheet_parse_coverage.json").write_text(
        _json.dumps(rows, indent=2, default=str), encoding="utf-8")


def _coverage_summary(coverage) -> Dict[str, Any]:
    inventoried = len(coverage)
    attempted = sum(1 for c in coverage if c.get("attempted_column_evidence"))
    with_evidence = sum(1 for c in coverage if c.get("column_evidence_rows", 0) > 0)
    in_queue = sum(1 for c in coverage if c.get("included_in_review_queue"))
    excluded = inventoried - with_evidence
    return {
        "files_inventoried": inventoried,
        "files_attempted_for_evidence": attempted,
        "files_with_column_evidence": with_evidence,
        "files_in_review_queue": in_queue,
        "files_excluded": excluded,
    }
