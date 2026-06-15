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
    enable_context_resolver: bool = False,
    context_llm_callable: Optional[Callable[[str], str]] = None,
    target_first_decisions_path: Optional[str] = None,
    enable_llm_target_advisor: bool = False,
    llm_target_advisor_callable: Optional[Callable[[str], str]] = None,
    llm_target_advisor_max_calls: int = 1,
    llm_target_advisor_model: str = "",
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

    # 27 — context resolution: deterministic guess -> optional LLM resolver ->
    # deterministic backstop -> final context (27a/27b/27). LLM is used only when a
    # context callable is supplied; the final context is always backstopped.
    from . import onboarding_context as octx
    from . import required_target_contract as rtc
    from . import llm_mapping_resolver as resolver
    ctx_inventory = inventory or [{"file_name": t[0], "classification": "",
                                   "domains_detected": []} for t in tables]
    ctx_out = octx.resolve_onboarding_context(
        ctx_inventory, evidence_rows, mode=mode, client_name=client_id,
        llm_callable=(context_llm_callable if enable_context_resolver else None))
    context = ctx_out["final"]
    context_usage = ctx_out["usage"]
    octx.write_context_artifacts(context, out_dir, deterministic=ctx_out["deterministic"],
                                 llm=ctx_out["llm"])

    # 28b — required target data contract SELECTED from the final context.
    required_contract = rtc.build_required_contract(context)
    rtc.write_contract_artifacts(required_contract, out_dir)

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

    # 31 — asset/regime-aware mapping RESOLVER against the required contract
    # (LLM is the semantic resolver when enabled; deterministic fallback otherwise).
    res = resolver.resolve_mappings(
        evidence_rows, shortlist_by_key, context, required_contract,
        llm_callable=(llm_callable if enable_llm else None),
        only_unresolved=only_unresolved, max_items=max_llm_items, max_cost_gbp=max_cost_gbp)
    resolver.write_resolver_artifacts(res, out_dir)
    # Persist the raw field LLM response for inspection (change 4).
    import json as _json
    fu = res["usage"]
    if fu.get("field_raw_response") is not None:
        (out_dir / "31_llm_field_raw_response.json").write_text(
            _json.dumps({"llm_batch_id": fu.get("llm_batch_id", ""),
                         "raw_response": fu.get("field_raw_response", "")},
                        indent=2, default=str), encoding="utf-8")
    # Combined usage summary separating context vs field LLM calls (auditable).
    cu = context_usage
    resolver_usage = {
        "llm_enabled": bool(cu.get("llm_enabled") or fu.get("llm_enabled")),
        "context_calls_completed": int(cu.get("calls_completed", 0)),
        "field_calls_completed": int(fu.get("calls_completed", 0)),
        "total_calls_completed": int(cu.get("calls_completed", 0)) + int(fu.get("calls_completed", 0)),
        "field_llm_callable_present": fu.get("field_llm_callable_present", False),
        "eligible_field_rows": fu.get("eligible_field_rows", 0),
        "eligible_reason_counts": fu.get("eligible_reason_counts", {}),
        "field_rows_requested": fu.get("field_rows_requested", 0),
        "field_rows_selected_for_llm": fu.get("field_rows_selected_for_llm", 0),
        "field_rows_reviewed": fu.get("rows_llm_reviewed", 0),
        "field_rows_skipped_due_to_cap": fu.get("field_rows_skipped_due_to_cap", 0),
        "field_rows_skipped_reason_counts": fu.get("field_rows_skipped_reason_counts", {}),
        "field_parse_status": fu.get("parse_status", ""),
        "field_parse_error": fu.get("parse_error", ""),
        # Row-level join diagnostics (change 5).
        "field_response_chars": fu.get("field_response_chars", 0),
        "field_response_shape": fu.get("field_response_shape", ""),
        "field_results_parsed": fu.get("field_results_parsed", 0),
        "field_results_matched": fu.get("field_results_matched", 0),
        "field_results_applied": fu.get("field_results_applied", 0),
        "field_results_rejected_invalid_decision": fu.get("field_results_rejected_invalid_decision", 0),
        "field_results_rejected_missing_required_fields": fu.get("field_results_rejected_missing_required_fields", 0),
        "field_result_decision_counts_raw": fu.get("field_result_decision_counts_raw", {}),
        "field_result_decision_counts_normalised": fu.get("field_result_decision_counts_normalised", {}),
        "field_results_unmatched": fu.get("field_results_unmatched", 0),
        "field_results_missing_row_id": fu.get("field_results_missing_row_id", 0),
        "field_incomplete_response": fu.get("field_incomplete_response", False),
        "estimated_cost_gbp": round(float(cu.get("estimated_cost_gbp", 0))
                                    + float(fu.get("estimated_cost_gbp", 0)), 6),
    }
    (out_dir / "31_llm_resolver_usage_summary.json").write_text(
        _json.dumps(resolver_usage, indent=2, default=str), encoding="utf-8")
    resolved_by_key = {(r["source_file"], r["source_sheet"], r["source_column"]): r
                       for r in res["resolved"]}
    llm_result = {"proposals": [], "usage": res["usage"]}
    # Surface LLM-resolved rows in the queue's LLM columns (per (file,sheet,col)).
    llm_by_key = {k: {"proposed_target_field": r["resolved_target_field"],
                      "proposed_business_meaning": r["rationale"],
                      "reasoning_summary": r["rationale"], "confidence": r["confidence"],
                      "llm_batch_id": r.get("llm_batch_id", "")}
                  for k, r in resolved_by_key.items() if r.get("llm_used")}

    # Proposals for the backstop: deterministic best wins (preserves existing
    # behaviour); where deterministic found nothing, use the resolver's mapping.
    cfields = rtc.contract_field_set(required_contract)
    proposals: List[Dict[str, Any]] = []
    for e in evidence_rows:
        key = _ek(e)
        col, src_file, src_sheet = e["source_column"], e.get("source_file", ""), e.get("source_sheet", "")
        best = _best_candidate(shortlist_by_key.get(key, []))
        rr = resolved_by_key.get(key, {})
        if best is not None:
            proposals.append({**best, "source_file": src_file, "source_sheet": src_sheet})
        elif rr.get("resolved_target_field") and rr.get("decision") in (
                resolver.MAP_EXISTING, resolver.PROPOSE_NEW):
            # An LLM-resolved row that becomes the ACTIVE proposal is tagged
            # llm_suggested (so the queue shows used_as_active_proposal, not
            # superseded); a deterministic cashflow extension stays cashflow_ledger.
            src = ("llm_suggested" if rr.get("llm_used")
                   else ("cashflow_ledger" if rr["decision"] == resolver.PROPOSE_NEW
                         else "contract_resolver"))
            proposals.append({
                "source_file": src_file, "source_sheet": src_sheet, "source_column": col,
                "proposed_target_field": rr["resolved_target_field"],
                "candidate_source": src, "confidence": rr["confidence"],
                "proposed_new_field": rr["decision"] == resolver.PROPOSE_NEW,
                "is_pipeline_field": False})
        else:
            proposals.append({
                "source_file": src_file, "source_sheet": src_sheet, "source_column": col,
                "proposed_target_field": "", "candidate_source":
                ("pipeline_contract" if e.get("candidate_existing_pipeline_contract_fields")
                 else "value_profile"), "confidence": "no_match"})

    # 32 — deterministic backstop validation (contract-aware coverage).
    validation_rows = backstop.validate_mappings(
        proposals, registry_fields=registry_fields, field_scope=field_scope,
        memory_store=memory_store, evidence_by_col=evidence_by_key,
        extra_in_scope=extra_in_scope, contract_fields=cfields)
    # Attach LLM/resolver audit columns to the backstop rows.
    for v in validation_rows:
        rr = resolved_by_key.get((v.get("source_file", ""), v.get("source_sheet", ""),
                                  v.get("source_column", "")), {})
        v["llm_suggested_mapping"] = rr.get("resolved_target_field", "") if rr.get("llm_used") else ""
        v["llm_confidence"] = rr.get("confidence", "") if rr.get("llm_used") else ""
        v["llm_rationale"] = rr.get("rationale", "") if rr.get("llm_used") else ""
        v["final_mapping"] = v.get("proposed_target_field", "")
        v["final_status"] = v.get("validation_status", "")
        v["backstop_decision"] = ("accepted" if v.get("auto_approvable")
                                  else ("rejected" if v.get("validation_status") in
                                        (backstop.UNSAFE, backstop.OUT_OF_SCOPE,
                                         backstop.CONFLICTS_MEMORY, backstop.CONFLICTS_MAPPING)
                                        else "review"))
        v["backstop_rejection_reason"] = (v.get("validation_reasons", "")
                                          if v["backstop_decision"] != "accepted" else "")
    backstop.write_validation_artifacts(validation_rows, out_dir)

    # Required-contract coverage (which mandatory/required fields are covered).
    mapped_targets = {v.get("proposed_target_field") for v in validation_rows
                      if v.get("proposed_target_field")
                      and v.get("validation_status") not in (backstop.UNSAFE,
                          backstop.OUT_OF_SCOPE, backstop.REGISTRY_TARGET_MISSING)}
    mandatory = rtc.mandatory_fields(required_contract)
    contract_coverage = {
        "required_fields_total": len(required_contract),
        "mandatory_total": len(mandatory),
        "mandatory_covered": len(mandatory & mapped_targets),
        "mandatory_missing": sorted(mandatory - mapped_targets),
        "covered_fields": sorted(cfields & mapped_targets),
    }

    # 28a/28b/28c — TARGET-CONTRACT-FIRST coverage. The compact human decision
    # queue is driven by target coverage gaps/conflicts, not by every source
    # column. The 33/34 source-column queue (below) remains as audit detail but
    # is no longer the primary gate artefact.
    from . import target_coverage as tcov
    target_first = tcov.run_target_first_coverage(
        mode=policy.name, context=context, evidence_rows=evidence_rows,
        resolved_rows=res["resolved"], output_dir=out_dir,
        client_id=client_id, run_id=run_id,
        decisions_path=target_first_decisions_path)

    # 36 — OPTIONAL target-first LLM ADVISOR. Operates on the 28c decisions +
    # 28a/28b evidence (NOT the raw source-column universe). Advisory only: it
    # never mutates the deterministic target-first state.
    if enable_llm_target_advisor:
        from . import target_first_llm_advisor as adv
        try:
            adv_res = adv.run_target_advisor(
                decision_rows=target_first["decision_queue"],
                coverage_rows=target_first["coverage"],
                residual_rows=target_first["residual"],
                file_inventory=inventory,
                evidence_rows=evidence_rows,
                llm_callable=llm_target_advisor_callable,
                max_items=max_llm_items, max_calls=llm_target_advisor_max_calls,
                max_cost_gbp=max_cost_gbp, model=llm_target_advisor_model)
            adv.write_advisor_artifacts(adv_res, out_dir)
            target_first["llm_advisor"] = {"usage": adv_res["usage"],
                                           "summary": adv.advisor_summary(adv_res["recommendations"])}
        except Exception as exc:  # never break onboarding on advisor failure
            target_first["llm_advisor"] = {"error": str(exc)}

    # 33/34 — concise multi-file review queue (source-column audit detail).
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
        "context": context,
        "context_deterministic": ctx_out["deterministic"],
        "context_llm": ctx_out["llm"],
        "context_usage": context_usage,
        "resolver_usage": resolver_usage,
        "required_contract": required_contract,
        "resolver": res,
        "evidence": evidence_rows,
        "shortlist": shortlist_rows,
        "drift": drift_rows,
        "llm": llm_result,
        "validation": validation_rows,
        "review_queue": review,
        "target_first_coverage": target_first,
        "file_coverage": coverage,
        "sheet_coverage": sheet_coverage,
        "contract_coverage": contract_coverage,
        "summary": {**review["summary"], "file_coverage": _coverage_summary(coverage),
                    "target_coverage": target_first["coverage_summary"],
                    "source_residual": target_first["residual_summary"],
                    "human_decision_queue": target_first["decision_summary"],
                    "target_contract_id": target_first["target_contract_id"],
                    "context": {k: context[k] for k in ("asset_class", "reporting_regime",
                                                        "required_domains", "confidence")},
                    "contract_coverage": contract_coverage},
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
