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
from typing import Any, Callable, Dict, List, Optional

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


def run_llm_assisted_mapping(
    input_file: Optional[str | Path] = None,
    df: Optional[pd.DataFrame] = None,
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
) -> Dict[str, Any]:
    """Run the full controlled mapping workbench pipeline and write artefacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df is None:
        if input_file is None:
            raise ValueError("provide df or input_file")
        p = Path(input_file)
        df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") \
            else pd.read_csv(p, low_memory=False)
        source_file_name = source_file_name or p.name
    source_file_name = source_file_name or "uploaded.csv"

    policy = load_mode_policy(mode)
    field_scope = resolve_field_scope(str(registry_path), policy,
                                      regulatory_reporting_enabled=regulatory_reporting_enabled)
    registry_fields = load_field_registry(Path(registry_path)).get("fields", {}) or {}
    mapper, _ = build_header_mapper(registry_path, aliases_dir)
    alias_index = AliasIndex.load(aliases_dir)

    # 28 — existing Pipeline MI field contract.
    contract_rows = contract.build_pipeline_field_contract(registry_path)
    contract.write_contract_artifacts(contract_rows, out_dir)

    # 29 — deterministic column evidence packs.
    evidence_rows = ce.build_column_evidence(
        df, source_file_name, registry_fields=registry_fields, field_scope=field_scope,
        semantic_mapper=mapper, alias_index=alias_index, memory_store=memory_store)
    ce.write_evidence_artifacts(evidence_rows, out_dir)
    evidence_by_col = {e["source_column"]: e for e in evidence_rows}

    # 30 — deterministic candidate shortlists.
    shortlist_rows = finder.build_candidate_shortlist(evidence_rows, registry_fields, field_scope)
    finder.write_shortlist_artifacts(shortlist_rows, out_dir)
    shortlist_by_col = finder.shortlist_by_column(shortlist_rows)

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
        best = _best_candidate(shortlist_by_col.get(e["source_column"], []))
        if best is None or float(best["candidate_confidence"]) < 0.95:
            unresolved_cols.add(e["source_column"])
    if only_unresolved:
        unresolved_cols |= drift_review_cols

    # 31 — controlled LLM reviewer (off unless enabled + callable provided).
    llm_result = {"proposals": [], "usage": {"llm_enabled": False, "calls_completed": 0,
                                             "estimated_cost_gbp": 0.0}}
    if enable_llm and llm_callable is not None:
        controller = LLMMappingController(
            llm_callable=llm_callable, registry_fields=registry_fields,
            field_scope=field_scope, max_items=max_llm_items, max_cost_gbp=max_cost_gbp)
        llm_result = controller.review(
            evidence_rows, shortlist_by_col,
            only_unresolved=unresolved_cols if only_unresolved else None)
    write_llm_review_artifacts(llm_result, out_dir)
    llm_by_col = {p["source_column"]: p for p in llm_result["proposals"]}

    # Merge into one proposal per column for validation: deterministic best wins;
    # fall back to the LLM proposal where deterministic found nothing.
    proposals: List[Dict[str, Any]] = []
    for e in evidence_rows:
        col = e["source_column"]
        best = _best_candidate(shortlist_by_col.get(col, []))
        llm = llm_by_col.get(col)
        if best is not None:
            proposals.append({**best, "source_file": source_file_name})
        elif llm is not None and llm.get("proposed_target_field"):
            proposals.append({
                "source_file": source_file_name, "source_column": col,
                "proposed_target_field": llm["proposed_target_field"],
                "candidate_source": "llm_suggested", "confidence": llm["confidence"],
                "ambiguity_flags": llm.get("ambiguity_flags", []),
                "alternative_targets": llm.get("alternative_targets", []),
                "is_pipeline_field": False})
        else:
            proposals.append({
                "source_file": source_file_name, "source_column": col,
                "proposed_target_field": "", "candidate_source":
                ("pipeline_contract" if e.get("candidate_existing_pipeline_contract_fields")
                 else "value_profile"), "confidence": "no_match"})

    # 32 — deterministic backstop validation.
    validation_rows = backstop.validate_mappings(
        proposals, registry_fields=registry_fields, field_scope=field_scope,
        memory_store=memory_store, evidence_by_col=evidence_by_col)
    backstop.write_validation_artifacts(validation_rows, out_dir)

    # 33/34 — concise review queue.
    review = queue.build_review_queue(validation_rows, evidence_by_col, llm_by_col)
    queue.write_queue_artifacts(review, out_dir)
    queue.append_review_action_log(
        out_dir, client_id, run_id, "run_llm_assisted_mapping",
        inputs={"file": source_file_name, "mode": mode, "llm_enabled": bool(
            llm_result["usage"].get("llm_enabled"))},
        outputs=["28..37 artefacts"], status="ok")

    return {
        "contract": contract_rows,
        "evidence": evidence_rows,
        "shortlist": shortlist_rows,
        "drift": drift_rows,
        "llm": llm_result,
        "validation": validation_rows,
        "review_queue": review,
        "summary": review["summary"],
    }
