"""
onboarding_orchestrator.py
=========================

Drives the onboarding workflow end to end and writes the reviewable pack.

Stages (each delegated to its own module):
  classify -> profile -> candidate keys -> mappings -> overlap -> config ->
  gaps -> review pack -> optional handoff

All outputs land under ``output_dir`` with numbered filenames. Nothing here
mutates production config or canonical data; the handoff artefacts are drafts.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from engine.gate_1_alignment.semantic_alignment import load_field_registry

from . import (
    config_suggester,
    domain_coverage,
    file_classifier,
    file_profiler,
    gap_analyzer,
    source_consolidator,
)
from .document_extractor import (
    extract_documents,
    load_document_policy,
    write_document_extraction_summary,
)
from .field_scope import resolve_field_scope
from .llm_mapping_reviewer import run_llm_mapping_review
from .llm_policy import resolve_llm_policy
from .mapping_proposer import propose_mappings
from .mode_policy import ModePolicy, default_mode, load_mode_policy, severity_rank
from .onboarding_models import OnboardingProject
from .review_pack_builder import build_review_pack


def compute_readiness(gap_questions, policy: ModePolicy) -> str:
    """Mode-aware readiness status from the (already re-ranked) gap questions.

    blocked            - any blocking question remains
    requires_review    - no blockers but high/medium questions remain
    ready_for_<mode>   - clean (policy.readiness_status_label)
    """
    max_rank = max((severity_rank(q.severity) for q in gap_questions), default=0)
    if max_rank >= severity_rank("blocking"):
        return "blocked"
    if max_rank >= severity_rank("medium"):
        return "requires_review"
    return policy.readiness_status_label

# Confidence at/above which a mapping counts as "high confidence" for handoff.
HANDOFF_CONFIDENCE = 0.92


def _write_csv(path: Path, items: List, fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for it in items:
            row = dataclasses.asdict(it) if dataclasses.is_dataclass(it) else dict(it)
            out = {}
            for k in fieldnames:
                v = row.get(k, "")
                if isinstance(v, (list, dict)):
                    v = "; ".join(str(x) for x in v) if isinstance(v, list) else json.dumps(v)
                out[k] = v
            writer.writerow(out)


def _write_json(path: Path, items: List) -> None:
    data = [dataclasses.asdict(it) if dataclasses.is_dataclass(it) else it for it in items]
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _load_structured_dataframes(inventory) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for item in inventory:
        if item.file_type not in ("csv", "xlsx", "xls"):
            continue
        try:
            if item.file_type in ("xlsx", "xls"):
                frames[item.file_path] = pd.read_excel(item.file_path)
            else:
                frames[item.file_path] = pd.read_csv(item.file_path, low_memory=False)
        except Exception:
            continue
    return frames


def run_onboarding(
    input_dir: str,
    client_name: str,
    output_dir: str,
    registry_path: str,
    aliases_dir: str,
    project_id: str = "",
    enable_handoff: bool = True,
    mode: str = "",
    regulatory_reporting_enabled: bool = False,
    enable_llm_review: bool = False,
    llm_budget_profile: str = "",
    llm_max_calls: int | None = None,
    llm_max_items_per_call: int | None = None,
    llm_callable=None,
    client_id: str = "",
    run_id: str = "",
    storage_backend: str = "local",
    input_uri: str = "",
    output_uri: str = "",
    client_memory_dir: str = "",
    apply_client_memory: bool | None = None,
    enable_mapping_review: bool = False,
    enable_llm_mapping_review: bool = False,
    llm_mapping_callable=None,
    llm_mapping_profile: str = "",
    llm_mapping_only_unresolved: bool = False,
    llm_max_mapping_items: int = 60,
    llm_max_cost_gbp: float = 1.0,
    enable_file_conversion_fallback: bool = False,
    enable_context_resolver: bool = False,
    context_llm_callable=None,
    target_first_decisions_path: str = "",
    enable_llm_target_advisor: bool = False,
    llm_target_advisor_callable=None,
    llm_target_advisor_model: str = "",
    target_contract: str = "",
    regime_config_path: str = "",
    asset_config_path: str = "",
) -> OnboardingProject:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = mode or default_mode()
    policy = load_mode_policy(mode)

    # Low-cost LLM policy (off by default; opt-in via CLI/config). Resolved here
    # so the whole run shares one budget.
    # Only force-enable when --enable-llm-review is explicitly set; otherwise let
    # the budget profile / config decide (so `--llm-budget-profile low` enables
    # without the store_true default clobbering it back to off).
    llm_policy = resolve_llm_policy(
        enable_llm_review=True if enable_llm_review else None,
        budget_profile=llm_budget_profile,
        max_calls=llm_max_calls,
        max_items_per_call=llm_max_items_per_call,
    )

    project = OnboardingProject(
        project_id=project_id or client_name.lower().replace(" ", "_"),
        client_name=client_name,
        input_dir=str(in_dir),
        output_dir=str(out_dir),
        onboarding_mode=policy.name,
        registry_path=str(registry_path),
        aliases_dir=str(aliases_dir),
        client_id=client_id,
        run_id=run_id,
        storage_backend=storage_backend or "local",
        input_uri=input_uri,
        output_uri=output_uri,
    )

    # --- PART 3: classify ---
    inventory = file_classifier.classify_directory(in_dir)
    project.file_inventory = inventory
    project.source_files = [i.file_path for i in inventory]

    dataframes = _load_structured_dataframes(inventory)

    # --- PART 4: profile ---
    profiles = []
    for item in inventory:
        if item.file_type in ("csv", "xlsx", "xls"):
            file_profiles = file_profiler.profile_file(Path(item.file_path))
            profiles.extend(file_profiles)
            # Surface a single reporting date onto the inventory item.
            rep_dates = sorted(
                {p.date_max for p in file_profiles if p.likely_reporting_date and p.date_max}
            )
            if rep_dates:
                item.detected_reporting_date = rep_dates[-1]
    project.field_profiles = profiles

    # --- Field scope (registry category + core_canonical, driven by mode) ---
    field_scope = resolve_field_scope(
        str(registry_path), policy, regulatory_reporting_enabled=regulatory_reporting_enabled
    )
    project.field_scope_summary = field_scope.counts()

    # --- PART 5a: candidate keys ---
    project.candidate_keys = source_consolidator.detect_candidate_keys(profiles)

    # --- PART 6: mapping candidates (mode field scope diverts out-of-scope targets) ---
    (project.mapping_candidates, project.out_of_scope_fields,
     project.mapping_ambiguities) = propose_mappings(
        inventory, dataframes, Path(registry_path), Path(aliases_dir),
        field_scope=field_scope,
        regulatory_reporting_enabled=regulatory_reporting_enabled,
    )

    # Mapping coverage by registry category (for the review pack).
    by_cat = {"regulatory": 0, "analytics": 0, "core": 0, "other": 0}
    for m in project.mapping_candidates:
        t = m.candidate_canonical_field
        if not t:
            continue
        if t in field_scope.core_canonical_fields:
            by_cat["core"] += 1
        elif t in field_scope.regulatory_fields:
            by_cat["regulatory"] += 1
        elif t in field_scope.analytics_fields:
            by_cat["analytics"] += 1
        else:
            by_cat["other"] += 1
    project.field_scope_summary["mapping_candidates_by_category"] = by_cat
    project.field_scope_summary["out_of_scope_fields_count"] = len(project.out_of_scope_fields)

    # --- PART 5b: overlap analysis (uses mappings + keys) ---
    project.overlap_analysis = source_consolidator.analyze_overlap(
        inventory, project.mapping_candidates, project.candidate_keys, dataframes
    )

    # --- PART 9/10: client mapping memory (applied AFTER deterministic mapping,
    # before gap generation). Mode-aware, field-scope-safe, never silently
    # overrides a material conflict. ---
    memory_resolved_enums: set = set()
    memory_ignored_columns: set = set()
    memory_resolved_source_fields: set = set()
    memory_gap_questions: List = []
    project.client_memory_summary = {"client_mapping_memory_loaded": False,
                                     "memory_entries_applied": 0}
    if client_id and apply_client_memory is not False:
        from . import mapping_memory as _mm
        try:
            mem_dir = _mm.resolve_memory_dir(
                memory_dir=client_memory_dir or None,
                output_dir=str(out_dir.parent), client_id=client_id,
            )
        except ValueError:
            mem_dir = None
        store = _mm.MappingMemoryStore(mem_dir, client_id=client_id) if mem_dir else None
        # When apply was not explicitly requested, only apply if memory exists.
        if store is not None and (apply_client_memory or not store.is_empty):
            conflict_signals = {
                o.canonical_candidate: float(o.sample_match_rate or 0)
                for o in project.overlap_analysis if o.canonical_candidate
            }
            mem_result = _mm.apply_mapping_memory(
                project.mapping_candidates, store, field_scope=field_scope,
                mode=policy.name, conflict_signals=conflict_signals,
            )
            memory_ignored_columns = set(mem_result["ignored_columns"])
            memory_gap_questions = mem_result["gap_questions"]
            memory_resolved_enums = _mm.resolved_enum_keys(store)
            memory_resolved_source_fields = _mm.resolved_precedence_fields(store)
            inv_dicts = [dataclasses.asdict(i) for i in inventory]
            memory_ignored_columns |= _mm.ignored_column_keys(store, inv_dicts)
            project.client_memory_summary = _mm.summarize_application(mem_result, store)

    # --- PART 6 (docs): extract config-relevant facts under minimisation policy ---
    doc_policy = load_document_policy()
    project.document_extractions = extract_documents(inventory, doc_policy)

    # --- PART 7: config suggestions (mode-scoped regulatory config) ---
    project.config_suggestions = config_suggester.suggest_config(
        client_name, in_dir, inventory, profiles, project.document_extractions,
        regulatory_config_in_scope=policy.regime_config_required,
        regime_optional=(policy.name == "mna_dd"),
    )

    # --- PART 8: gap questions (mode-aware severity + field scope) ---
    project.gap_questions = gap_analyzer.analyze_gaps(
        inventory, profiles, project.overlap_analysis, project.config_suggestions,
        dataframes, mode_policy=policy, field_scope=field_scope,
        out_of_scope_fields=project.out_of_scope_fields,
        mapping_candidates=project.mapping_candidates,
        memory_resolved_enums=memory_resolved_enums,
        memory_ignored_columns=memory_ignored_columns,
        memory_resolved_source_fields=memory_resolved_source_fields,
    )
    # Warning gaps from materially-conflicting client memory (PART 10).
    if memory_gap_questions:
        from .onboarding_models import GapQuestion as _GQ
        for gq in memory_gap_questions:
            project.gap_questions.append(_GQ(**{
                k: v for k, v in gq.items() if k in _GQ.__dataclass_fields__
            }))

    # --- PART 4/5/6: targeted, bounded LLM mapping review (off by default) ---
    registry_fields = load_field_registry(Path(registry_path)).get("fields", {}) or {}
    suggestions, llm_usage, llm_gap_questions = run_llm_mapping_review(
        mapping_candidates=project.mapping_candidates,
        mapping_ambiguities=project.mapping_ambiguities,
        field_scope=field_scope,
        registry_fields=registry_fields,
        mode=policy.name,
        policy=llm_policy,
        regulatory_reporting_enabled=regulatory_reporting_enabled,
        column_profiles=profiles,
        llm_callable=llm_callable,
        gap_question_start_index=len(project.gap_questions) + 1,
    )
    project.llm_mapping_suggestions = suggestions
    project.llm_usage_summary = llm_usage
    # Excess uncertainty becomes user gap questions instead of token spend.
    project.gap_questions.extend(llm_gap_questions)

    # --- PART 5/6: domain detection + coverage (domain-based, not file-based) ---
    registry_fields = load_field_registry(Path(registry_path)).get("fields", {}) or {}
    column_index: Dict[str, List[str]] = {}
    for item in inventory:
        df = dataframes.get(item.file_path)
        if df is not None:
            column_index[item.file_name] = [str(c) for c in df.columns]
    domain_coverage.annotate_inventory_domains(
        inventory, project.mapping_candidates, registry_fields, column_index
    )
    project.domain_coverage = domain_coverage.assess_domain_coverage(
        inventory, project.mapping_candidates, field_scope, policy.name,
        registry_fields, document_extractions=project.document_extractions,
        column_index=column_index,
    )
    domain_coverage.write_domain_coverage_artifacts(project.domain_coverage, out_dir)
    project.generated_artifacts.append(str(out_dir / "17_domain_coverage.csv"))
    project.generated_artifacts.append(str(out_dir / "17_domain_coverage.json"))

    # --- Deterministic-first mapping trace (explainability/audit) ---
    from . import mapping_trace
    trace = mapping_trace.build_trace(
        inventory=inventory,
        dataframes=dataframes,
        mapping_candidates=project.mapping_candidates,
        out_of_scope_fields=project.out_of_scope_fields,
        mapping_ambiguities=project.mapping_ambiguities,
        overlap_analysis=project.overlap_analysis,
        field_scope=field_scope,
        registry_fields=registry_fields,
        aliases_dir=aliases_dir,
        llm_suggestions=project.llm_mapping_suggestions,
        precedence={},  # approved precedence is decided later (answer ingestion)
        profiles=profiles,
    )
    project.mapping_trace_summary = trace["summary"]
    mapping_trace.write_trace_artifacts(trace, out_dir)
    mapping_trace.write_explanation_report(trace, out_dir, policy.name, project.client_name)
    for name in ("05c_mapping_trace.csv", "05c_mapping_trace.json", "05d_mapping_explanation.md"):
        project.generated_artifacts.append(str(out_dir / name))

    # --- PARTS 2-9: controlled LLM-assisted mapping review (deterministic-first;
    # LLM is off unless explicitly enabled AND a callable is available). Writes
    # artefacts 28-37 next to the numbered onboarding pack. ---
    if enable_mapping_review or enable_llm_mapping_review or enable_llm_target_advisor:
        from . import llm_assisted_mapping as _lam
        from . import mapping_memory as _mm
        # Pass the FULL inventory so the review's robust loader parses every file
        # (all sheets) and emits explicit per-file coverage (29a) — never silently
        # limited to one file.
        inventory_dicts = [dataclasses.asdict(i) for i in inventory]
        mr_store = None
        mr_memory_dir = None
        if client_id:
            try:
                mr_memory_dir = _mm.resolve_memory_dir(
                    output_dir=str(out_dir.parent), client_id=client_id)
                mr_store = _mm.MappingMemoryStore(mr_memory_dir, client_id=client_id)
            except ValueError:
                mr_store = None
        enable_llm_map = bool(
            enable_llm_mapping_review and (llm_mapping_profile or "low") != "off")
        try:
            mr = _lam.run_llm_assisted_mapping(
                inventory=inventory_dicts, output_dir=str(out_dir), registry_path=registry_path,
                aliases_dir=aliases_dir, mode=policy.name,
                regulatory_reporting_enabled=regulatory_reporting_enabled,
                client_id=client_id, run_id=run_id,
                enable_llm=enable_llm_map, llm_callable=llm_mapping_callable,
                only_unresolved=llm_mapping_only_unresolved,
                memory_store=mr_store,
                memory_dir=str(mr_memory_dir) if mr_memory_dir else None,
                max_llm_items=llm_max_mapping_items, max_cost_gbp=llm_max_cost_gbp,
                enable_file_conversion_fallback=enable_file_conversion_fallback,
                enable_context_resolver=enable_context_resolver,
                context_llm_callable=context_llm_callable,
                target_first_decisions_path=(target_first_decisions_path or None),
                enable_llm_target_advisor=enable_llm_target_advisor,
                llm_target_advisor_callable=llm_target_advisor_callable,
                llm_target_advisor_max_calls=(llm_max_calls or 1),
                llm_target_advisor_model=llm_target_advisor_model,
                target_contract=target_contract,
                regime_config_path=(regime_config_path or None),
                asset_config_path=(asset_config_path or None),
            )
            ru = mr.get("resolver_usage", {})
            project.mapping_review_summary = {
                **mr["summary"],
                "llm_enabled": bool(ru.get("llm_enabled")),
                "context_calls_completed": ru.get("context_calls_completed", 0),
                "field_calls_completed": ru.get("field_calls_completed", 0),
                "field_rows_reviewed": ru.get("field_rows_reviewed", 0),
                "eligible_field_rows": ru.get("eligible_field_rows", 0),
                "field_rows_selected_for_llm": ru.get("field_rows_selected_for_llm", 0),
                "llm_estimated_cost_gbp": ru.get("estimated_cost_gbp", 0.0),
            }
        except Exception as exc:  # never break the onboarding run on review failure
            project.mapping_review_summary = {"error": str(exc)}
        for name in (
            "27a_deterministic_context_guess.json", "27b_llm_context_resolution.json",
            "27_onboarding_context.json", "27_onboarding_context_summary.md",
            "28_required_target_contract.csv", "28_required_target_contract.json",
            "28_required_target_contract_summary.md",
            "28a_target_coverage_matrix.csv", "28a_target_coverage_matrix.json",
            "28a_target_coverage_summary.md",
            "28b_source_residual_register.csv", "28b_source_residual_register.json",
            "28b_source_residual_summary.md",
            "28c_human_decision_queue.csv", "28c_human_decision_queue.json",
            "28c_human_decision_summary.md",
            "34_target_first_decisions.yaml",
            "35_target_first_decision_application_log.json",
            "35_target_first_decision_application_log.csv",
            "36_target_first_llm_recommendations.csv",
            "36_target_first_llm_recommendations.json",
            "36_target_first_llm_recommendations_summary.md",
            "36_target_first_llm_raw_response.json",
            "36_target_first_llm_usage_summary.json",
            "42_annex2_config_validation.csv",
            "42_annex2_config_validation.json",
            "42_annex2_config_validation_summary.md",
            "43_annex2_field_universe_reconciliation.csv",
            "43_annex2_field_universe_reconciliation.json",
            "43_annex2_field_universe_reconciliation_summary.md",
            "44_annex2_nd_eligibility_reconciliation.csv",
            "44_annex2_nd_eligibility_reconciliation.json",
            "44_annex2_nd_eligibility_reconciliation_summary.md",
            "46_annex2_enum_coverage_reconciliation.csv",
            "46_annex2_enum_coverage_reconciliation.json",
            "46_annex2_enum_coverage_reconciliation_summary.md",
            "47_annex2_semantic_mapping_reconciliation.csv",
            "47_annex2_semantic_mapping_reconciliation.json",
            "47_annex2_semantic_mapping_reconciliation_summary.md",
            "45_annex2_config_alignment_review.csv",
            "45_annex2_config_alignment_review.json",
            "45_annex2_config_alignment_review_summary.md",
            "31_llm_mapping_resolver.csv", "31_llm_mapping_resolver.json",
            "31_llm_mapping_resolver_summary.md", "31_llm_usage_summary.json",
            "31_llm_resolver_usage_summary.json", "31_llm_field_raw_response.json",
            "28_existing_pipeline_field_contract.csv",
            "28_existing_pipeline_field_contract.json",
            "28_existing_pipeline_field_contract_summary.md",
            "29_column_evidence.csv", "29_column_evidence.json",
            "29_column_evidence_summary.md",
            "29a_column_evidence_file_coverage.csv",
            "29a_column_evidence_file_coverage.json",
            "29b_excel_sheet_parse_coverage.csv",
            "29b_excel_sheet_parse_coverage.json",
            "30_mapping_candidate_shortlist.csv",
            "30_mapping_candidate_shortlist.json", "31_llm_mapping_review.csv",
            "31_llm_mapping_review.json", "31_llm_mapping_review_summary.md",
            "31_llm_usage_summary.json", "32_mapping_backstop_validation.csv",
            "32_mapping_backstop_validation.json", "33_mapping_review_queue.csv",
            "33_mapping_review_queue.json", "34_mapping_review_decisions.yaml",
            "35_mapping_review_action_log.json", "37_schema_drift_report.csv",
            "37_schema_drift_report.json",
        ):
            ap = out_dir / name
            if ap.exists():
                project.generated_artifacts.append(str(ap))

    # --- mode-aware review status / readiness ---
    project.review_status = compute_readiness(project.gap_questions, policy)

    # --- write artefacts (incl. 17 document extraction summary, client-scoped) ---
    write_document_extraction_summary(project.document_extractions, out_dir, doc_policy)
    project.generated_artifacts.append(str(out_dir / "17_document_extraction_summary.yaml"))
    _write_artifacts(project)

    # --- PART 5: LLM usage / cost summary (always written; off => zero cost) ---
    _write_llm_artifacts(project, out_dir)

    # --- PART 9: review pack ---
    pack_path = out_dir / "08_onboarding_review_pack.html"
    build_review_pack(project, pack_path)
    project.generated_artifacts.append(str(pack_path))

    # --- PART 10: optional handoff ---
    if enable_handoff:
        _write_handoff(project)

    # run summary
    summary_path = out_dir / "09_onboarding_run_summary.json"
    summary_path.write_text(
        json.dumps(project.to_summary_dict(), indent=2, default=str), encoding="utf-8"
    )
    project.generated_artifacts.append(str(summary_path))

    return project


def _write_llm_artifacts(project: OnboardingProject, out_dir: Path) -> None:
    """PART 5 — write 22_llm_usage_summary.json (+ suggestions when present).

    When the LLM is off the summary is the minimal zero-cost record.
    """
    usage = project.llm_usage_summary or {}
    if not usage.get("llm_enabled"):
        usage = {"llm_enabled": False, "calls_completed": 0, "estimated_cost": 0,
                 **{k: v for k, v in usage.items() if k not in
                    ("llm_enabled", "calls_completed", "estimated_cost")}}
    summary_path = out_dir / "22_llm_usage_summary.json"
    summary_path.write_text(json.dumps(usage, indent=2, default=str), encoding="utf-8")
    project.generated_artifacts.append(str(summary_path))

    if project.llm_mapping_suggestions:
        sugg_path = out_dir / "22_llm_mapping_suggestions.json"
        sugg_path.write_text(
            json.dumps(
                {"_warning": "SUGGESTION-ONLY — never promoted to final mappings.",
                 "llm_mapping_suggestions": project.llm_mapping_suggestions},
                indent=2, default=str),
            encoding="utf-8",
        )
        project.generated_artifacts.append(str(sugg_path))


def _write_artifacts(project: OnboardingProject) -> None:
    out = Path(project.output_dir)

    # 01 file inventory
    inv_fields = [f.name for f in dataclasses.fields(project.file_inventory[0])] if project.file_inventory else []
    if project.file_inventory:
        _write_csv(out / "01_file_inventory.csv", project.file_inventory, inv_fields)
    _write_json(out / "01_file_inventory.json", project.file_inventory)

    # 02 column profiles
    if project.field_profiles:
        prof_fields = [f.name for f in dataclasses.fields(project.field_profiles[0])]
        _write_csv(out / "02_column_profiles.csv", project.field_profiles, prof_fields)
    _write_json(out / "02_column_profiles.json", project.field_profiles)

    # 03 candidate keys
    if project.candidate_keys:
        key_fields = [f.name for f in dataclasses.fields(project.candidate_keys[0])]
        _write_csv(out / "03_candidate_keys.csv", project.candidate_keys, key_fields)
    else:
        _write_csv(out / "03_candidate_keys.csv", [], ["candidate_key", "file_name", "source_column"])

    # 04 overlap analysis
    overlap_fields = [
        "canonical_candidate", "source_file_a", "source_column_a", "source_file_b",
        "source_column_b", "similarity_score", "sample_match_rate",
        "recommended_primary_source", "recommended_secondary_source", "review_required", "reason",
    ]
    _write_csv(out / "04_source_overlap_analysis.csv", project.overlap_analysis, overlap_fields)

    # 05 mapping candidates
    map_fields = [
        "source_file", "source_file_classification", "source_column",
        "candidate_canonical_field", "confidence", "method",
        "sample_values_redacted", "requires_review", "reason",
        "ambiguity_rule_applied", "alternative_candidates",
    ]
    _write_csv(out / "05_mapping_candidates.csv", project.mapping_candidates, map_fields)
    _write_json(out / "05_mapping_candidates.json", project.mapping_candidates)

    # 05a out-of-scope fields (excluded by mode field scope)
    oos_fields = ["source_file", "source_column", "candidate_field", "category", "reason", "mode"]
    _write_csv(out / "05a_out_of_scope_fields.csv", project.out_of_scope_fields, oos_fields)

    # 05b mapping ambiguities (resolved by the regulatory-preference rule)
    amb_fields = [
        "source_file", "source_column", "selected_canonical_field",
        "selected_category", "selected_core_canonical", "selected_confidence",
        "alternative_canonical_field", "alternative_category",
        "alternative_core_canonical", "alternative_confidence", "confidence_delta",
        "ambiguity_rule_applied", "review_required", "reason", "mode",
    ]
    _write_csv(out / "05b_mapping_ambiguities.csv", project.mapping_ambiguities, amb_fields)
    _write_json(out / "05b_mapping_ambiguities.json", project.mapping_ambiguities)

    # 06 config suggestions
    cfg_fields = [
        "field", "suggested_value", "confidence", "source_file",
        "source_column_or_document_reference", "evidence", "review_status",
    ]
    _write_csv(out / "06_config_suggestions.csv", project.config_suggestions, cfg_fields)
    cfg_yaml = {
        "client_name": project.client_name,
        "generated_by": "trakt_onboarding_agent_v2",
        "review_status": project.review_status,
        "suggestions": [dataclasses.asdict(s) for s in project.config_suggestions],
    }
    (out / "06_config_suggestions.yaml").write_text(
        yaml.safe_dump(cfg_yaml, sort_keys=False), encoding="utf-8"
    )

    # 07 gap questions
    gap_fields = [
        "question_id", "category", "severity", "question", "reason",
        "candidate_answers", "default_recommendation", "blocking_for", "source_evidence",
        "subject", "subject_value",
    ]
    _write_csv(out / "07_gap_questions.csv", project.gap_questions, gap_fields)
    (out / "07_gap_questions.yaml").write_text(
        yaml.safe_dump([dataclasses.asdict(q) for q in project.gap_questions], sort_keys=False),
        encoding="utf-8",
    )

    # example_answers.yaml — a pre-filled answer template the user can edit and
    # feed back via `--answers` (answer ingestion, PART 4).
    _write_example_answers(project, out / "example_answers.yaml")
    project.generated_artifacts.append(str(out / "example_answers.yaml"))

    for name in [
        "01_file_inventory.csv", "01_file_inventory.json",
        "02_column_profiles.csv", "02_column_profiles.json",
        "03_candidate_keys.csv", "04_source_overlap_analysis.csv",
        "05_mapping_candidates.csv", "05_mapping_candidates.json",
        "05a_out_of_scope_fields.csv",
        "05b_mapping_ambiguities.csv", "05b_mapping_ambiguities.json",
        "06_config_suggestions.csv", "06_config_suggestions.yaml",
        "07_gap_questions.csv", "07_gap_questions.yaml",
    ]:
        project.generated_artifacts.append(str(out / name))


def _example_answer_for(q) -> str:
    """Pick a sensible pre-filled answer for the example template."""
    if q.default_recommendation and q.default_recommendation in q.candidate_answers:
        return q.default_recommendation
    if q.candidate_answers:
        return q.candidate_answers[0]
    return q.default_recommendation or ""


def _write_example_answers(project: OnboardingProject, path: Path) -> None:
    answers = {}
    for q in project.gap_questions:
        answers[q.question_id] = {
            "answer": _example_answer_for(q),
            "approved_by": "user",
            "note": q.question,
        }
    payload = {
        "_doc": (
            "Answer the gap questions below, then run answer ingestion with "
            "`--answers <this file>`. 'answer' must be one of the candidate answers "
            "where candidates exist; dates must be ISO; source answers must be a "
            "source file name."
        ),
        "project_id": project.project_id,
        "answers": answers,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_handoff(project: OnboardingProject) -> None:
    """PART 10 — draft, review-only handoff artefacts (never production config)."""
    out = Path(project.output_dir)

    high_conf = [
        m for m in project.mapping_candidates
        if m.candidate_canonical_field and m.confidence >= HANDOFF_CONFIDENCE
    ]

    # 09 draft client config
    cfg = {f.field: f.suggested_value for f in project.config_suggestions}
    draft_config = {
        "_warning": "DRAFT — review-only. Do not deploy to production without approval.",
        "client_name": project.client_name,
        "portfolio": {
            "asset_class": cfg.get("asset_class", ""),
            "base_currency": cfg.get("currency", ""),
            "country": cfg.get("jurisdiction", ""),
        },
        "default_regime": cfg.get("regime", ""),
        "reporting_date": cfg.get("reporting_date", ""),
        "warehouse": {
            "present": cfg.get("warehouse_facility_present", "unknown"),
            "lender_name": cfg.get("warehouse_lender_name", ""),
            "advance_rate": cfg.get("advance_rate", ""),
            "margin": cfg.get("margin", ""),
            "interest_index": cfg.get("interest_index", ""),
        },
        "geography_policy": cfg.get("geography_policy", ""),
    }
    (out / "09_draft_client_config.yaml").write_text(
        yaml.safe_dump(draft_config, sort_keys=False), encoding="utf-8"
    )

    # 09 draft mapping overrides (high-confidence only)
    overrides = {
        "_warning": "DRAFT — review-only. High-confidence mappings for the existing pipeline.",
        "mappings": [
            {
                "source_file": m.source_file,
                "source_column": m.source_column,
                "canonical_field": m.candidate_canonical_field,
                "confidence": m.confidence,
                "method": m.method,
            }
            for m in high_conf
        ],
    }
    (out / "09_draft_mapping_overrides.yaml").write_text(
        yaml.safe_dump(overrides, sort_keys=False), encoding="utf-8"
    )

    for name in ["09_draft_client_config.yaml", "09_draft_mapping_overrides.yaml"]:
        project.generated_artifacts.append(str(out / name))
