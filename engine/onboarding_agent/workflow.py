"""
workflow.py
===========

Thin managed-service OPERATOR WORKFLOW wrapper for the Onboarding Agent.

This is a wrapper only — it orchestrates the existing, working pieces and never
reimplements onboarding logic:

    deterministic target-first onboarding  (run_onboarding)
      -> 28a coverage / 28b residual / 28c decision queue / 34 template
    optional target-first LLM advisor       (advisory only -> 36_*)
    optional approved-decision application   (--target-first-decisions -> 35_*)

It then reads the resulting artefacts and emits an operator-readable workflow
summary plus a conservative, non-mutating legacy-file audit:

    40_operator_workflow_summary.json / .md
    41_onboarding_legacy_file_audit.json / .md

Run it with::

    python -m engine.onboarding_agent.workflow --input-dir ... --client-name ...

The existing lower-level CLI (``engine.onboarding_agent.cli``) is unchanged and
remains available for development / debugging.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent.mode_policy import VALID_MODES  # noqa: E402

# Workflow status vocabulary (target-first state only — never legacy gaps).
READY = "READY"
NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
# Annex 2: the target universe is known but not fully configured (codes pending a
# regime rule, or missing from 28a). Never READY in this state.
NEEDS_CONFIGURATION = "NEEDS_CONFIGURATION"
BLOCKED = "BLOCKED"
FAILED = "FAILED"

FIRST_PASS = "first_pass_review_generation"
SECOND_PASS = "second_pass_decision_application"


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_yaml(path: Path) -> Optional[dict]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Status derivation (pure — unit-testable)
# ---------------------------------------------------------------------------

def derive_status(decisions_after: int, blocking_after: int,
                  required_artifacts_missing: bool) -> str:
    """Workflow status from the resulting target-first 28c state."""
    if required_artifacts_missing:
        return FAILED
    if blocking_after > 0:
        return BLOCKED
    if decisions_after > 0:
        return NEEDS_CONFIRMATION
    return READY


def _next_action(stage: str, status: str, advisor_enabled: bool) -> str:
    if status == FAILED:
        return ("Workflow failed to produce the required target-first artefacts — "
                "re-run with --enable-mapping-review and check the logs.")
    if status == BLOCKED:
        advice = (" LLM recommendations are advisory: to apply them run "
                  "`python -m engine.onboarding_agent.cli accept-target-advice` "
                  "(writes 34_target_first_decisions_approved.yaml), or approve "
                  "manually." if advisor_enabled else "")
        return ("Resolve the remaining BLOCKING Gate 4 decisions (provide a source, "
                "configure a value, or mark not applicable) in a copy of "
                "34_target_first_decisions.yaml and rerun with "
                "--target-first-decisions." + advice)
    if status == NEEDS_CONFIGURATION:
        return ("The Annex 2 target universe is loaded but not fully configured — some "
                "codes are pending a regime field rule (or missing from 28a). Complete the "
                "regime config (config/regime/annex2_delivery_rules.yaml) against the "
                "workbook universe; see 43_annex2_field_universe_reconciliation.")
    if status == READY:
        return ("All Gate 4 decisions are resolved — the pack is ready for MI handoff / "
                "promotion.")
    # NEEDS_CONFIRMATION
    advisor_hint = (" Use the LLM advisor column as optional guidance." if advisor_enabled
                    else "")
    if stage == FIRST_PASS:
        return ("Review Gate 4 in the HTML review pack." + advisor_hint +
                " Then approve decisions in a copy of 34_target_first_decisions.yaml and "
                "rerun with --target-first-decisions.")
    return ("Confirm the remaining non-blocking Gate 4 decisions (approve them in a copy "
            "of 34_target_first_decisions.yaml) and rerun with --target-first-decisions.")


# ---------------------------------------------------------------------------
# Lifecycle-aware handoff messaging (messaging only — no logic change)
# ---------------------------------------------------------------------------

# Suggested human-readable readiness statements.
HANDOFF_READY_MSG = "Onboarding handoff: READY for Transformation & Validation."
HANDOFF_NOT_READY_MSG = (
    "Onboarding handoff: NOT READY — resolve blocking decisions / registry gaps "
    "before handing off to Transformation & Validation.")
PROJECTION_NOT_READY_MSG = (
    "Projection readiness: NOT READY — pending regime/projection rules remain.")
PROJECTION_READY_MSG = "Projection readiness: READY."
XML_NOT_READY_MSG = (
    "XML delivery readiness: NOT READY — Transformation/Validation and Projection "
    "have not yet produced an XML-ready target frame.")
XML_READY_MSG = "XML delivery readiness: READY."


def _handoff_next_actions(
    *,
    ready_tv: bool,
    ready_proj: bool,
    ready_xml: bool,
    manifest_path: str,
    contract_path: str,
    base_action: str,
) -> Dict[str, str]:
    """Lifecycle-aware next-action wording for a run that produced a handoff.

    ``base_action`` is the existing (onboarding-centric) ``next_operator_action``;
    it is retained as the *downstream/projection* action — never as the primary
    Onboarding next step once the handoff is ready.
    """
    if ready_tv:
        onboarding_next = HANDOFF_READY_MSG
        downstream_next = (
            base_action if (not ready_proj and base_action) else (
                "Projection is ready." if ready_proj else
                "Complete the downstream regime/projection rules and materialise "
                "downstream defaults; see 43_annex2_field_universe_reconciliation."))
        primary = (
            "Onboarding handoff is ready for Transformation & Validation. "
            "Projection/XML delivery is not yet ready: pending regime/projection "
            "rules, downstream defaults, or unresolved derivations remain. "
            f"The next agent should consume {manifest_path} and {contract_path}."
        ) if not (ready_proj and ready_xml) else (
            "Onboarding handoff is ready for Transformation & Validation. "
            f"The next agent should consume {manifest_path} and {contract_path}.")
    else:
        # Handoff not ready: the onboarding-centric action remains primary.
        onboarding_next = HANDOFF_NOT_READY_MSG
        downstream_next = base_action
        primary = base_action
    return {
        "onboarding_next_action": onboarding_next,
        "downstream_next_action": downstream_next,
        "next_operator_action": primary,
    }



# ---------------------------------------------------------------------------
# Summary assembly (reads existing artefacts — never mutates them)
# ---------------------------------------------------------------------------

_CORE_ARTIFACTS = [
    "08_onboarding_review_pack.html",
    "28a_target_coverage_matrix.csv",
    "28c_human_decision_queue.csv",
    "34_target_first_decisions.yaml",
]
_DECISION_ARTIFACTS = [
    "35_target_first_decision_application_log.json",
    "35_target_first_decision_application_log.csv",
]
_ADVISOR_ARTIFACTS = [
    "36_target_first_llm_recommendations.csv",
    "36_target_first_llm_usage_summary.json",
]


def build_workflow_summary(
    project_dir: Path,
    output_root: Path,
    *,
    client_id: str,
    client_name: str,
    run_id: str,
    mode: str,
    decisions_supplied_file: str,
    advisor_enabled: bool,
    input_source_files_count: int,
    run_error: str = "",
    regime_config_path: str = "",
    asset_config_path: str = "",
) -> Dict[str, Any]:
    """Read the run's artefacts and build the 40 workflow summary dict."""
    stage = SECOND_PASS if decisions_supplied_file else FIRST_PASS

    cov = _read_json(project_dir / "28a_target_coverage_matrix.json") or {}
    cov_sum = cov.get("summary", {}) or {}
    dec = _read_json(project_dir / "28c_human_decision_queue.json") or {}
    dec_sum = dec.get("summary", {}) or {}
    cfgval = _read_json(project_dir / "42_annex2_config_validation.json") or {}
    cfgval_sum = cfgval.get("summary", {}) or {}
    recon = _read_json(project_dir / "43_annex2_field_universe_reconciliation.json") or {}
    recon_sum = recon.get("summary", {}) or {}
    nd_recon = _read_json(project_dir / "44_annex2_nd_eligibility_reconciliation.json") or {}
    nd_sum = nd_recon.get("summary", {}) or {}
    align = _read_json(project_dir / "45_annex2_config_alignment_review.json") or {}
    align_sum = align.get("summary", {}) or {}
    enumc = _read_json(project_dir / "46_annex2_enum_coverage_reconciliation.json") or {}
    enum_sum = enumc.get("summary", {}) or {}
    semm = _read_json(project_dir / "47_annex2_semantic_mapping_reconciliation.json") or {}
    sem_sum = semm.get("summary", {}) or {}
    prop = _read_json(project_dir / "48_annex2_mapping_correction_proposals.json") or {}
    prop_sum = prop.get("summary", {}) or {}
    app = _read_json(project_dir / "35_target_first_decision_application_log.json")
    app_sum = (app or {}).get("summary", {}) or {}
    advu = _read_json(project_dir / "36_target_first_llm_usage_summary.json")
    advr = _read_json(project_dir / "36_target_first_llm_recommendations.json")
    adv_sum = (advr or {}).get("summary", {}) or {}

    decisions_after = int(dec_sum.get("human_decision_rows_total", len(dec.get("rows", []) or [])))
    blocking_after = int(dec_sum.get("blocking_decisions", 0))
    non_blocking_after = max(0, decisions_after - blocking_after)
    applied = int(app_sum.get("applied", 0))
    # Deterministic reconstruction: applied decisions were removed from 28c.
    decisions_before = decisions_after + applied if stage == SECOND_PASS else decisions_after

    # Required-artefact validation (warnings; FAILED only when CORE is missing).
    warnings: List[str] = []
    core_missing = [n for n in _CORE_ARTIFACTS if not (project_dir / n).exists()]
    for n in core_missing:
        warnings.append(f"required artefact missing: {n}")
    if decisions_supplied_file:
        for n in _DECISION_ARTIFACTS:
            if not (project_dir / n).exists():
                warnings.append(f"expected decision-application artefact missing: {n}")
    if advisor_enabled:
        for n in _ADVISOR_ARTIFACTS:
            if not (project_dir / n).exists():
                warnings.append(f"expected LLM advisor artefact missing: {n}")

    failed = bool(run_error) or bool(core_missing)
    status = FAILED if failed else derive_status(decisions_after, blocking_after, False)

    def _p(name: str) -> str:
        p = project_dir / name
        return str(p) if p.exists() else ""

    summary = {
        "workflow_run_id": run_id,
        "client_id": client_id,
        "client_name": client_name,
        "mode": mode,
        "project_dir": str(project_dir),
        "output_root": str(output_root),
        "workflow_stage": stage,
        "status": status,
        "generated_at": _now(),
        "input_source_files_count": input_source_files_count,
        "target_fields_count": int(cov_sum.get("target_fields_total", len(cov.get("rows", []) or []))),
        "target_coverage_file": _p("28a_target_coverage_matrix.csv"),
        "human_decision_queue_file": _p("28c_human_decision_queue.csv"),
        "human_decision_queue_count_before": decisions_before,
        "human_decision_queue_count_after": decisions_after,
        "blocking_decisions_count": blocking_after,
        "non_blocking_confirmations_count": non_blocking_after,
        "target_first_decisions_template_file": _p("34_target_first_decisions.yaml"),
        "target_first_decisions_supplied_file": decisions_supplied_file or "",
        "approved_decisions_supplied_count": int(app_sum.get("decisions_supplied", 0)),
        "applied_decisions_count": applied,
        "deferred_decisions_count": int(app_sum.get("deferred", 0)),
        "invalid_decisions_count": int(app_sum.get("invalid", 0)),
        "requires_operator_review_count": int(app_sum.get("requires_operator_review", 0)),
        "application_log_json": _p("35_target_first_decision_application_log.json"),
        "application_log_csv": _p("35_target_first_decision_application_log.csv"),
        "llm_target_advisor_enabled": bool(advisor_enabled),
        "llm_target_advisor_file": _p("36_target_first_llm_recommendations.csv"),
        "llm_target_advisor_rows": int(adv_sum.get("recommendations_total",
                                                   (advu or {}).get("decision_rows_available", 0))),
        "llm_target_advisor_advised_count": int((advu or {}).get("decision_rows_advised",
                                                                 adv_sum.get("advised", 0))),
        "llm_target_advisor_parse_failed_count": int((advu or {}).get("decision_rows_parse_failed", 0)),
        "llm_target_advisor_estimated_cost_gbp": float((advu or {}).get("estimated_cost_gbp", 0.0)),
        "review_pack_html": _p("08_onboarding_review_pack.html"),
        "warnings": warnings,
        "error": run_error,
    }

    # --- Target contract identity + Annex 2 (ESMA) config layers ---
    target_contract_id = cov.get("target_contract_id", "")
    summary["target_contract_id"] = target_contract_id
    summary["target_contract_source"] = cov.get("target_contract_source", "")
    summary["regime_config_path"] = regime_config_path or cfgval.get("regime_config_source", "")
    summary["asset_config_path"] = asset_config_path or cfgval.get("asset_config_source", "")
    if target_contract_id == "esma_annex_2":
        dtc = dec_sum.get("decision_type_counts", {}) or {}
        missing_from_28a = int(recon_sum.get("missing_from_28a_count", 0))
        # Codes in the authoritative universe with no full regime rule yet — a
        # config-completeness gap regardless of whether a source was matched.
        pending_rule = int(recon_sum.get("missing_from_regime_rules_count",
                                         cov_sum.get("pending_regime_rule_fields", 0)))
        # Subset with neither a regime rule nor a source (genuinely uncovered).
        pending_uncovered = int(cov_sum.get("pending_regime_rule_fields", 0))
        summary.update({
            # Total authoritative target universe (workbook ∪ regime ∪ deferred).
            "annex2_authoritative_field_count": int(
                recon_sum.get("authoritative_field_count",
                              cov_sum.get("target_fields_total", 0))),
            "annex2_coverage_field_count": int(
                recon_sum.get("coverage_field_count",
                              cov_sum.get("target_fields_total", 0))),
            "annex2_regime_rule_count": int(recon_sum.get("regime_rule_count", 0)),
            "annex2_config_validation_count": int(recon_sum.get("config_validation_count", 0)),
            "annex2_missing_from_28a_count": missing_from_28a,
            "annex2_deferred_field_count": int(recon_sum.get("deferred_field_count", 0)),
            "annex2_pending_regime_rule_count": pending_rule,
            "annex2_pending_uncovered_count": pending_uncovered,
            "annex2_deliverable_field_count": int(recon_sum.get("deliverable_field_count", 0)),
            # Back-compat: 28a field count (now the full universe).
            "annex2_field_count": int(cov_sum.get("target_fields_total", 0)),
            "annex2_source_mapped_count": int(cov_sum.get("source_mapped_fields", 0)),
            "annex2_derived_count": int(cov_sum.get("derived_fields", 0)),
            "annex2_defaulted_nd_count": int(cov_sum.get("defaulted_nd_fields", 0)),
            "annex2_defaulted_value_count": int(cov_sum.get("defaulted_value_fields", 0)),
            "annex2_configured_static_count": int(cov_sum.get("configured_static_fields", 0)),
            "annex2_missing_required_count": int(cov_sum.get("missing_required_fields", 0)),
            "annex2_gate4_decision_count": decisions_after,
            "annex2_invalid_default_count": int(
                cfgval_sum.get("invalid_default_not_allowed",
                               dtc.get("invalid_default_value", 0))),
            # Registry mapping coverage (fields_registry ESMA_Annex2 mappings).
            "annex2_registry_mapped_count": int(recon_sum.get("registry_mapped_count", 0)),
            "annex2_registry_gap_count": int(recon_sum.get("registry_gap_count", 0)),
            # Active phantom deferred fields (codes outside the workbook universe).
            "annex2_active_phantom_deferred_count": int(
                recon_sum.get("not_in_authoritative_universe_count", 0)),
            "annex2_config_validation_summary": _p("42_annex2_config_validation_summary.md"),
            "annex2_field_universe_reconciliation_summary": _p(
                "43_annex2_field_universe_reconciliation_summary.md"),
            # ND-eligibility reconciliation (regime nd_allowed vs workbook).
            "annex2_nd_match_count": int(nd_sum.get("match", 0)),
            "annex2_nd_regime_stricter_count": int(nd_sum.get("regime_stricter", 0)),
            "annex2_nd_regime_broader_count": int(nd_sum.get("regime_broader", 0)),
            "annex2_nd_divergent_count": int(nd_sum.get("divergent", 0)),
            "annex2_nd_compliance_risk_count": int(nd_sum.get("nd_compliance_risk_count", 0)),
            "annex2_nd_eligibility_reconciliation_summary": _p(
                "44_annex2_nd_eligibility_reconciliation_summary.md"),
            # Config-alignment review (45).
            "annex2_alignment_tightened_count": int(align_sum.get("tightened_to_workbook", 0)),
            "annex2_alignment_left_stricter_count": int(align_sum.get("left_stricter_by_policy", 0)),
            "annex2_alignment_divergent_count": int(align_sum.get("divergent_requires_review", 0)),
            "annex2_alignment_phantom_removed_count": int(align_sum.get("phantom_deferred_removed", 0)),
            "annex2_alignment_registry_added_count": int(align_sum.get("registry_mapping_added", 0)),
            "annex2_asset_default_conflict_count": int(align_sum.get("asset_default_conflict", 0)),
            "annex2_alignment_manual_review_count": int(
                align_sum.get("requires_manual_review_count", 0)),
            "annex2_config_alignment_review_summary": _p(
                "45_annex2_config_alignment_review_summary.md"),
            # Enum-coverage reconciliation (46).
            "annex2_enum_constrained_count": int(enum_sum.get("constrained_within_workbook", 0)),
            "annex2_enum_unconstrained_count": int(enum_sum.get("unconstrained_no_enum_map", 0)),
            "annex2_enum_targets_outside_workbook_count": int(
                enum_sum.get("targets_outside_workbook", 0)),
            "annex2_enum_semantic_mismatch_count": int(enum_sum.get("semantic_mismatch", 0)),
            "annex2_enum_no_rule_count": int(enum_sum.get("no_regime_rule", 0)),
            "annex2_enum_coverage_reconciliation_summary": _p(
                "46_annex2_enum_coverage_reconciliation_summary.md"),
            # Semantic-mapping reconciliation (47).
            "annex2_semantic_aligned_count": int(sem_sum.get("aligned", 0)),
            "annex2_semantic_mismatch_count": int(sem_sum.get("semantic_mismatch", 0)),
            "annex2_semantic_mapping_reconciliation_summary": _p(
                "47_annex2_semantic_mapping_reconciliation_summary.md"),
            # Mapping-correction proposals (48) — report-only.
            "annex2_mapping_proposals_total": int(prop_sum.get("proposal_rows_total", 0)),
            "annex2_mapping_proposals_repoint_only": int(prop_sum.get("re_point_source_only", 0)),
            "annex2_mapping_proposals_need_mechanics": int(
                prop_sum.get("needs_rule_mechanics_changes", 0)),
            "annex2_mapping_correction_proposals_summary": _p(
                "48_annex2_mapping_correction_proposals_summary.md"),
        })
        sem_mismatch = int(sem_sum.get("semantic_mismatch", 0))
        if sem_mismatch > 0:
            warnings.append(
                f"{sem_mismatch} Annex 2 regime rule(s) map a source field that does not match "
                "the workbook field for that code (suspected code↔field mismap) — manual "
                "review required; see 47_annex2_semantic_mapping_reconciliation.")
        enum_outside = int(enum_sum.get("targets_outside_workbook", 0))
        if enum_outside > 0:
            warnings.append(
                f"{enum_outside} Annex 2 {{LIST}} field(s) map to enum codes the workbook FORBIDS "
                "(must be zero) — see 46_annex2_enum_coverage_reconciliation.")
        enum_semantic = int(enum_sum.get("semantic_mismatch", 0))
        if enum_semantic > 0:
            warnings.append(
                f"{enum_semantic} Annex 2 {{LIST}} field(s) have a regime rule whose source "
                "field does not match the workbook field — enum left unconstrained pending "
                "mapping review; see 46_annex2_enum_coverage_reconciliation.")
        # regime_broader (regime permits ND the workbook FORBIDS) must be zero.
        nd_broader = int(nd_sum.get("regime_broader", 0))
        if nd_broader > 0:
            warnings.append(
                f"{nd_broader} Annex 2 code(s) where the regime nd_allowed set is BROADER than "
                "the authoritative workbook ND eligibility (compliance risk — must be zero) — "
                "see 45_annex2_config_alignment_review.")
        nd_divergent = int(nd_sum.get("divergent", 0))
        if nd_divergent > 0:
            warnings.append(
                f"{nd_divergent} Annex 2 code(s) with ND sets divergent from the workbook "
                "require manual review (not auto-fixed) — see "
                "45_annex2_config_alignment_review.")
        # The Annex 2 universe is known but not fully configured — never READY.
        if status in (READY, NEEDS_CONFIRMATION) and (missing_from_28a > 0 or pending_rule > 0):
            status = NEEDS_CONFIGURATION
            summary["status"] = status
            if missing_from_28a > 0:
                warnings.append(
                    f"{missing_from_28a} authoritative Annex 2 code(s) missing from 28a "
                    "coverage — see 43_annex2_field_universe_reconciliation.")
            if pending_rule > 0:
                warnings.append(
                    f"{pending_rule} Annex 2 code(s) in the authoritative universe have no "
                    "full regime field rule yet (pending_regime_rule) — regime config is "
                    "incomplete relative to the workbook universe.")

    summary["next_operator_action"] = _next_action(stage, status, advisor_enabled)
    return summary


def _write_summary_md(path: Path, s: Dict[str, Any]) -> None:
    supplied = (Path(s["target_first_decisions_supplied_file"]).name
                if s["target_first_decisions_supplied_file"] else "none")
    md = [
        "# Operator workflow summary", "",
        f"Client: {s['client_name']}  ",
        f"Mode: {s['mode']}  ",
        f"Run: {s['workflow_run_id']}  ",
        f"Status: {s['status']}", "",
    ]
    if s.get("target_contract_id"):
        md += [
            "## Target contract", "",
            f"- target_contract_id: {s.get('target_contract_id', '')}",
        ]
        if s.get("regime_config_path"):
            md.append(f"- regime_config: {s.get('regime_config_path', '')}")
        if s.get("asset_config_path"):
            md.append(f"- asset_config: {s.get('asset_config_path', '')}")
        md.append("")
    if s.get("target_contract_id") == "esma_annex_2":
        md += [
            "## ESMA Annex 2 field universe", "",
            f"- Authoritative Annex 2 universe (workbook): {s.get('annex2_authoritative_field_count', 0)}",
            f"- Registry mapped: {s.get('annex2_registry_mapped_count', 0)}",
            f"- Registry gaps: {s.get('annex2_registry_gap_count', 0)}",
            f"- Fields covered in 28a: {s.get('annex2_coverage_field_count', 0)}",
            f"- Fields with full regime rules: {s.get('annex2_regime_rule_count', 0)}",
            f"- Config-validation rows: {s.get('annex2_config_validation_count', 0)}",
            f"- Deferred / pending reconciliation: {s.get('annex2_deferred_field_count', 0)}",
            f"- Pending regime rule (config gap): {s.get('annex2_pending_regime_rule_count', 0)}",
            f"- Active phantom deferred fields: {s.get('annex2_active_phantom_deferred_count', 0)}",
            f"- Missing from 28a: {s.get('annex2_missing_from_28a_count', 0)}",
            f"- Deliverable (rule + coverage): {s.get('annex2_deliverable_field_count', 0)}", "",
            "## ESMA Annex 2 ND eligibility & alignment", "",
            f"- ND broader than workbook (compliance risk): {s.get('annex2_nd_regime_broader_count', 0)}",
            f"- ND divergent (manual review): {s.get('annex2_nd_divergent_count', 0)}",
            f"- ND stricter than workbook (policy): {s.get('annex2_nd_regime_stricter_count', 0)}",
            f"- ND matches workbook: {s.get('annex2_nd_match_count', 0)}",
            f"- Tightened to workbook: {s.get('annex2_alignment_tightened_count', 0)}",
            f"- Phantom deferred removed: {s.get('annex2_alignment_phantom_removed_count', 0)}",
            f"- Registry mappings added: {s.get('annex2_alignment_registry_added_count', 0)}",
            f"- Asset-default conflicts: {s.get('annex2_asset_default_conflict_count', 0)}",
            f"- Items requiring manual review: {s.get('annex2_alignment_manual_review_count', 0)}", "",
            "## ESMA Annex 2 coverage", "",
            f"- Annex 2 fields in 28a: {s.get('annex2_field_count', 0)}",
            f"- Source mapped: {s.get('annex2_source_mapped_count', 0)}",
            f"- Derived: {s.get('annex2_derived_count', 0)}",
            f"- ND defaulted: {s.get('annex2_defaulted_nd_count', 0)}",
            f"- Value defaulted: {s.get('annex2_defaulted_value_count', 0)}",
            f"- Configured / static: {s.get('annex2_configured_static_count', 0)}",
            f"- Missing required: {s.get('annex2_missing_required_count', 0)}",
            f"- Gate 4 decisions: {s.get('annex2_gate4_decision_count', 0)}",
            f"- Invalid asset defaults: {s.get('annex2_invalid_default_count', 0)}", "",
        ]
    md += [
        "## Target-first status", "",
        f"- Target fields: {s['target_fields_count']}",
        f"- Gate 4 decisions remaining: {s['human_decision_queue_count_after']}",
        f"- Blocking decisions: {s['blocking_decisions_count']}",
        f"- Non-blocking confirmations: {s['non_blocking_confirmations_count']}", "",
        "## Decisions", "",
        f"- Decision template: {Path(s['target_first_decisions_template_file']).name or '—'}",
        f"- Approved decisions supplied: {supplied}",
        f"- Applied decisions: {s['applied_decisions_count']}",
        f"- Deferred: {s['deferred_decisions_count']} · "
        f"invalid: {s['invalid_decisions_count']} · "
        f"requires review: {s['requires_operator_review_count']}", "",
        "## LLM target advisor", "",
        f"- Enabled: {'yes' if s['llm_target_advisor_enabled'] else 'no'}",
        f"- Decisions reviewed: {s['llm_target_advisor_rows']}",
        f"- Decisions advised: {s['llm_target_advisor_advised_count']}",
        f"- Estimated cost: £{s['llm_target_advisor_estimated_cost_gbp']}", "",
    ]
    if s.get("handoff_created"):
        ready_tv = bool(s.get("ready_for_transformation_validation"))
        ready_proj = bool(s.get("ready_for_projection"))
        ready_xml = bool(s.get("ready_for_xml_delivery"))
        md += [
            "## Onboarding handoff", "",
            (HANDOFF_READY_MSG if ready_tv else HANDOFF_NOT_READY_MSG),
            "",
            (PROJECTION_READY_MSG if ready_proj else PROJECTION_NOT_READY_MSG),
            (XML_READY_MSG if ready_xml else XML_NOT_READY_MSG), "",
            f"- next_agent: {s.get('next_agent', 'transformation_validation')}",
            f"- handoff_manifest_path: {s.get('handoff_manifest_path', '')}",
            f"- ready_for_transformation_validation: {ready_tv}",
            f"- ready_for_projection: {ready_proj}",
            f"- ready_for_xml_delivery: {ready_xml}", "",
            "## Onboarding next action", "",
            s.get("onboarding_next_action", ""), "",
            "## Downstream / projection next action", "",
            s.get("downstream_next_action", ""), "",
        ]
    md += [
        "## Next operator action", "",
        s["next_operator_action"], "",
    ]
    if s.get("warnings"):
        md += ["## Warnings", ""] + [f"- {w}" for w in s["warnings"]] + [""]
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Legacy file audit (41) — conservative, NON-MUTATING
# ---------------------------------------------------------------------------

# (basename, kind, classification, reason). Existence is checked at write time.
_AUDIT_CATALOG = [
    # --- target-first artefacts that MUST be kept ---
    ("28a_target_coverage_matrix.csv", "artefact", "keep_core", "Target-first coverage matrix (primary gate)."),
    ("28b_source_residual_register.csv", "artefact", "keep_core", "Target-first residual register (suppressed source columns)."),
    ("28c_human_decision_queue.csv", "artefact", "keep_core", "Target-first compact Gate 4 decision queue (primary operator workflow)."),
    ("34_target_first_decisions.yaml", "artefact", "keep_core", "Target-first operator decision template / persistence."),
    ("35_target_first_decision_application_log.json", "artefact", "keep_core", "Target-first decision application log."),
    ("36_target_first_llm_recommendations.csv", "artefact", "keep_core", "Target-first LLM advisory recommendations (advisory only)."),
    ("36_target_first_llm_usage_summary.json", "artefact", "keep_core", "Target-first LLM advisor usage summary."),
    ("08_onboarding_review_pack.html", "artefact", "keep_core", "Operator-first review pack (primary deliverable)."),
    # --- workflow artefacts that MUST be kept ---
    ("40_operator_workflow_summary.json", "artefact", "keep_core", "Operator workflow summary (this command)."),
    ("41_onboarding_legacy_file_audit.json", "artefact", "keep_core", "Legacy-file audit (this command)."),
    ("42_annex2_config_validation.csv", "artefact", "keep_core", "ESMA Annex 2 regime/asset config validation (Annex 2 mode only)."),
    ("43_annex2_field_universe_reconciliation.csv", "artefact", "keep_core", "ESMA Annex 2 field-universe reconciliation (Annex 2 mode only)."),
    ("44_annex2_nd_eligibility_reconciliation.csv", "artefact", "keep_core", "ESMA Annex 2 ND-eligibility reconciliation: regime nd_allowed vs workbook (Annex 2 mode only)."),
    ("45_annex2_config_alignment_review.csv", "artefact", "keep_core", "ESMA Annex 2 config-alignment review: actions taken + manual-review items (Annex 2 mode only)."),
    ("46_annex2_enum_coverage_reconciliation.csv", "artefact", "keep_core", "ESMA Annex 2 enum-coverage reconciliation: regime enum_map vs workbook allowed codes (Annex 2 mode only)."),
    ("47_annex2_semantic_mapping_reconciliation.csv", "artefact", "keep_core", "ESMA Annex 2 semantic-mapping reconciliation: regime source field vs workbook field per code (Annex 2 mode only)."),
    ("48_annex2_mapping_correction_proposals.csv", "artefact", "keep_core", "ESMA Annex 2 mapping-correction proposals: proposed source/ND/mechanics fixes for mismapped codes, report-only (Annex 2 mode only)."),
    # --- source-column legacy decision artefacts RETAINED FOR AUDIT ---
    ("33_mapping_review_queue.csv", "artefact", "keep_legacy_audit", "Source-column review queue; retained as audit detail, no longer the primary gate."),
    ("34_mapping_review_decisions.yaml", "artefact", "keep_legacy_audit", "Source-column decision template; superseded by 34_target_first_decisions.yaml; kept for audit."),
    ("35_mapping_review_action_log.json", "artefact", "keep_legacy_audit", "Source-column action log; superseded by 35_target_first_*; kept for audit."),
    # --- old LLM source-column resolver artefacts ---
    ("31_llm_mapping_resolver.csv", "artefact", "keep_legacy_audit", "Source-column LLM resolver output; separate layer from the target-first advisor; kept for audit."),
    ("31_llm_mapping_resolver_summary.md", "artefact", "keep_legacy_audit", "Source-column LLM resolver summary; kept for audit."),
    ("31_llm_resolver_usage_summary.json", "artefact", "keep_legacy_audit", "Source-column LLM resolver usage; kept for audit."),
    ("31_llm_field_raw_response.json", "artefact", "keep_legacy_audit", "Source-column LLM raw response; kept for audit."),
    # --- old gap-question readiness artefacts ---
    ("07_gap_questions.csv", "artefact", "keep_legacy_audit", "Legacy gap questions; superseded by target-first Gate 4; do NOT drive workflow status."),
    ("07_gap_questions.yaml", "artefact", "candidate_for_deprecation", "Legacy gap-question readiness; superseded by 28c/40 status; deprecate after migration."),
    # --- supporting deterministic artefacts (kept) ---
    ("28_required_target_contract.csv", "artefact", "keep_core", "Required target contract that 28a is built against."),
    ("32_mapping_backstop_validation.csv", "artefact", "keep_legacy_audit", "Source-column backstop validation; audit detail."),
    # --- modules ---
    ("engine/onboarding_agent/target_coverage.py", "module", "keep_core", "Deterministic target-first coverage."),
    ("engine/onboarding_agent/target_first_decisions.py", "module", "keep_core", "Target-first decision capture / application."),
    ("engine/onboarding_agent/target_first_llm_advisor.py", "module", "keep_core", "Target-first LLM advisor (advisory only)."),
    ("engine/onboarding_agent/workflow.py", "module", "keep_core", "Operator workflow wrapper (this command)."),
    ("engine/onboarding_agent/mapping_review_queue.py", "module", "keep_legacy_audit", "Source-column 33 queue generation; retained for audit detail."),
    ("engine/onboarding_agent/llm_mapping_resolver.py", "module", "keep_legacy_audit", "Source-column LLM resolver; separate layer; retained."),
]


def build_legacy_audit(project_dir: Path) -> Dict[str, Any]:
    entries = []
    counts: Dict[str, int] = {}
    for name, kind, classification, reason in _AUDIT_CATALOG:
        present = ((project_dir / name).exists() if kind == "artefact"
                   else (_REPO_ROOT / name).exists())
        entries.append({
            "name": name, "kind": kind, "classification": classification,
            "reason": reason, "present_in_run": bool(present),
            "retained": classification != "candidate_for_removal_after_migration",
        })
        counts[classification] = counts.get(classification, 0) + 1
    return {
        "generated_at": _now(),
        "scope": "onboarding-related files only",
        "non_mutating": True,
        "note": ("Audit only — nothing is deleted or modified. Conservative: when "
                 "unsure, classify as unknown_review_required."),
        "classification_counts": counts,
        "entries": entries,
    }


def _write_audit_md(path: Path, audit: Dict[str, Any]) -> None:
    md = ["# Onboarding legacy file audit", "",
          "Non-mutating audit — nothing is deleted in this PR.", "",
          "## Classification counts", ""]
    for cls, c in sorted(audit["classification_counts"].items()):
        md.append(f"- `{cls}`: {c}")
    md += ["", "## Entries", "",
           "| Name | Kind | Classification | Present | Reason |",
           "|---|---|---|---|---|"]
    for e in audit["entries"]:
        md.append(f"| `{e['name']}` | {e['kind']} | {e['classification']} | "
                  f"{'yes' if e['present_in_run'] else 'no'} | {e['reason']} |")
    md.append("")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_operator_workflow(
    *,
    input_dir: str,
    client_name: str,
    client_id: str = "",
    run_id: str = "",
    project_dir: str = "",
    output_root: str = "",
    mode: str = "mi_only",
    registry: str = "config/system/fields_registry.yaml",
    aliases_dir: str = "config/system",
    enable_mapping_review: bool = True,
    enable_llm_target_advisor: bool = False,
    target_first_decisions: str = "",
    llm_max_cost_gbp: float = 1.0,
    llm_max_calls: Optional[int] = None,
    llm_max_items_per_call: Optional[int] = None,
    target_contract: str = "",
    regime_config: str = "",
    asset_config: str = "",
    reporting_date: str = "",
    override_reporting_date: bool = False,
    product_profile: str = "",
    enable_context_resolver: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run the managed-service operator workflow; returns the 40 summary dict."""
    client_id = client_id or client_name.lower().replace(" ", "_")
    run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    pdir = Path(project_dir) if project_dir else (
        _REPO_ROOT / "onboarding_output" / client_id / run_id)
    oroot = Path(output_root) if output_root else (pdir / "output")
    pdir.mkdir(parents=True, exist_ok=True)

    # --- Annex 2 (ESMA) target-contract resolution -------------------------
    # regulatory_mi already means "ESMA Annex 2 delivery". When the run targets
    # the Annex 2 contract we load TWO config layers: the regime rules
    # (config/regime/annex2_delivery_rules.yaml) and the ERM asset defaults
    # (config/asset/product_defaults_ERM.yaml) — both default automatically and
    # can be overridden by explicit flags.
    from engine.onboarding_agent.mode_policy import resolve_mode_alias
    from engine.onboarding_agent import target_coverage as _tcov
    resolved_mode, _ = resolve_mode_alias(mode)
    explicit_annex2 = str(target_contract or "").strip().lower() in (
        "esma_annex2", "esma_annex_2", "annex2", "annex_2")
    is_annex2 = (explicit_annex2
                 or _tcov.target_contract_kind(resolved_mode) == "esma_annex_2")
    if is_annex2:
        regime_config = regime_config or str(_tcov._ANNEX2_REGIME_DEFAULT)
        asset_config = asset_config or str(_tcov._ASSET_CONFIG_DEFAULT)

    # Build a shared LLM callable only when the advisor is enabled (None when no
    # ANTHROPIC_API_KEY — the advisor then records a deterministic no_advice).
    advisor_callable = None
    advisor_model = ""
    if enable_llm_target_advisor:
        from engine.onboarding_agent.cli import _build_mapping_llm_callable
        advisor_callable = _build_mapping_llm_callable("low")
        advisor_model = "claude-haiku-4-5-20251001"
        if advisor_callable is None:
            print("[workflow] No ANTHROPIC_API_KEY — LLM target advisor will record a "
                  "deterministic no_advice fallback.")

    run_error = ""
    input_files = 0
    # Use the (low-cost) LLM also for onboarding-context resolution so the asset
    # class / product profile can be DETECTED when deterministic file/column tokens
    # are weak. Defaults to following the advisor flag; an explicit
    # ``enable_context_resolver`` overrides. Reuses the same callable (no extra
    # configuration / key).
    use_context_resolver = (bool(advisor_callable) if enable_context_resolver is None
                            else bool(enable_context_resolver))
    context_callable = advisor_callable if use_context_resolver else None
    try:
        from engine.onboarding_agent.onboarding_orchestrator import run_onboarding
        project = run_onboarding(
            input_dir=input_dir, client_name=client_name, output_dir=str(pdir),
            registry_path=registry, aliases_dir=aliases_dir, mode=mode,
            client_id=client_id, run_id=run_id, output_uri="",
            enable_mapping_review=enable_mapping_review,
            target_first_decisions_path=(target_first_decisions or ""),
            enable_llm_target_advisor=enable_llm_target_advisor,
            llm_target_advisor_callable=advisor_callable,
            llm_target_advisor_model=advisor_model,
            llm_max_cost_gbp=llm_max_cost_gbp,
            llm_max_calls=llm_max_calls,
            llm_max_items_per_call=llm_max_items_per_call,
            target_contract=("ESMA_Annex2" if is_annex2 else target_contract),
            regime_config_path=regime_config,
            asset_config_path=asset_config,
            product_profile=product_profile,
            enable_context_resolver=use_context_resolver,
            context_llm_callable=context_callable,
            reporting_date=reporting_date,
        )
        input_files = len(project.file_inventory)
    except Exception as exc:  # produce a FAILED summary instead of crashing
        run_error = f"{type(exc).__name__}: {exc}"

    summary = build_workflow_summary(
        pdir, oroot, client_id=client_id, client_name=client_name, run_id=run_id,
        mode=mode, decisions_supplied_file=(target_first_decisions or ""),
        advisor_enabled=enable_llm_target_advisor,
        input_source_files_count=input_files, run_error=run_error,
        regime_config_path=(regime_config if is_annex2 else ""),
        asset_config_path=(asset_config if is_annex2 else ""))

    (pdir / "40_operator_workflow_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_summary_md(pdir / "40_operator_workflow_summary.md", summary)

    # --- Onboarding → Transformation & Validation handoff package (24–27) ---
    # Governed canonical handoff. Additive; never mutates existing outputs and
    # never re-canonicalises the generic central tape. Built for the Annex 2
    # (regulatory) target contract; skipped on failed runs (no 28a).
    if not run_error and is_annex2:
        try:
            from engine.onboarding_agent import onboarding_handoff
            handoff = onboarding_handoff.build_handoff_package(
                pdir, oroot, client_id=client_id, client_name=client_name,
                run_id=run_id, mode=mode, registry=registry, aliases_dir=aliases_dir,
                regime_config_path=(regime_config if is_annex2 else ""),
                asset_config_path=(asset_config if is_annex2 else ""),
                decisions_supplied_file=(target_first_decisions or ""),
                reporting_date=reporting_date,
                override_reporting_date=override_reporting_date)
            if handoff:
                m = handoff["manifest"]
                summary["onboarding_handoff_manifest_json"] = handoff["manifest_json_path"]
                summary["onboarding_handoff_readiness_json"] = handoff["readiness_json_path"]
                summary["onboarding_handoff_field_contract_csv"] = (
                    handoff["field_contract_csv_path"])
                summary["onboarding_handoff_lineage_json"] = handoff["lineage_path"]
                summary["onboarding_handoff_type"] = m.get("handoff_type", "")
                summary["onboarding_handoff_next_agent"] = m.get("next_agent", "")
                summary["ready_for_transformation_validation"] = bool(
                    m.get("ready_for_transformation_validation", False))
                summary["ready_for_projection"] = bool(m.get("ready_for_projection", False))
                summary["ready_for_xml_delivery"] = bool(m.get("ready_for_xml_delivery", False))
                # Lifecycle-aware messaging (messaging only — no logic change).
                summary["handoff_created"] = True
                summary["handoff_manifest_path"] = handoff["manifest_json_path"]
                summary["next_agent"] = m.get("next_agent", "transformation_validation")
                summary.update(_handoff_next_actions(
                    ready_tv=summary["ready_for_transformation_validation"],
                    ready_proj=summary["ready_for_projection"],
                    ready_xml=summary["ready_for_xml_delivery"],
                    manifest_path="output/handoff/24_onboarding_handoff_manifest.json",
                    contract_path="output/handoff/26_onboarding_handoff_field_contract.csv",
                    base_action=summary.get("next_operator_action", "")))
                # Re-write the 40 summary (json + md) so it carries the handoff
                # references and lifecycle-aware next actions.
                (pdir / "40_operator_workflow_summary.json").write_text(
                    json.dumps(summary, indent=2, default=str), encoding="utf-8")
                _write_summary_md(pdir / "40_operator_workflow_summary.md", summary)
                # Inject the handoff section into the static review pack.
                try:
                    from engine.onboarding_agent.review_pack_builder import (
                        refresh_review_pack_handoff)
                    refresh_review_pack_handoff(pdir, oroot)
                except Exception:
                    pass
        except Exception as exc:  # never break the workflow on handoff issues
            summary.setdefault("warnings", []).append(
                f"onboarding handoff package not generated: {type(exc).__name__}: {exc}")

    audit = build_legacy_audit(pdir)
    (pdir / "41_onboarding_legacy_file_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8")
    _write_audit_md(pdir / "41_onboarding_legacy_file_audit.md", audit)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m engine.onboarding_agent.workflow",
        description="Managed-service operator workflow for the Onboarding Agent "
        "(thin wrapper around the existing target-first pipeline).")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--client-name", required=True)
    p.add_argument("--client-id", default="")
    p.add_argument("--run-id", default="")
    p.add_argument("--project-dir", default="")
    p.add_argument("--output-root", default="")
    p.add_argument("--mode", choices=list(VALID_MODES) + ["mi_mna"], default="mi_only")
    p.add_argument("--registry", default="config/system/fields_registry.yaml")
    p.add_argument("--aliases-dir", default="config/system")
    # Target-first artefacts are required by this workflow, so mapping review is
    # ON by default; --no-mapping-review disables it for debugging.
    p.add_argument("--enable-mapping-review", action="store_true", default=True,
                   help="(default on) generate the deterministic target-first artefacts.")
    p.add_argument("--no-mapping-review", dest="enable_mapping_review",
                   action="store_false", help="disable target-first artefact generation.")
    p.add_argument("--enable-llm-target-advisor", action="store_true",
                   help="run the target-first LLM advisor (advisory only; writes 36_*).")
    p.add_argument("--no-llm-target-advisor", dest="enable_llm_target_advisor",
                   action="store_false", help="explicitly disable the LLM target advisor.")
    p.set_defaults(enable_llm_target_advisor=False)
    p.add_argument("--target-first-decisions", default="",
                   help="approved 34_target_first_decisions.yaml to apply (second pass).")
    p.add_argument("--llm-max-cost-gbp", type=float, default=1.0)
    p.add_argument("--llm-max-calls", type=int, default=None)
    p.add_argument("--llm-max-items-per-call", type=int, default=None)
    # Explicit target-contract / two-layer config selection. regulatory_mi already
    # means ESMA Annex 2; these flags make the selection explicit and override the
    # default config paths. Backwards compatible (all optional).
    p.add_argument("--target-contract", default="",
                   help="explicit target contract id, e.g. ESMA_Annex2 (default: mode-derived).")
    p.add_argument("--regime-config", default="",
                   help="regime rules config (default: config/regime/annex2_delivery_rules.yaml "
                        "for Annex 2 mode).")
    p.add_argument("--asset-config", default="",
                   help="asset-class defaults config (default: "
                        "config/asset/product_defaults_ERM.yaml for Annex 2 mode).")
    p.add_argument("--reporting-date", default="",
                   help="OPTIONAL fallback reporting cut-off date (YYYY-MM-DD). "
                        "Source-derived data_cut_off_date is preferred; this is used "
                        "only when none is found in the source pack/config.")
    p.add_argument("--product-profile", default="",
                   help="explicitly select a config-driven product profile (e.g. "
                        "equity_release_lifetime_mortgage) from config/asset/"
                        "product_profiles.yaml. Trusted outright; otherwise the "
                        "profile is detected from evidence. Controls which missing "
                        "fields are base-MI blocking vs visible/non-blocking.")
    p.add_argument("--enable-context-resolver", dest="enable_context_resolver",
                   action="store_true", default=None,
                   help="use the LLM to resolve the onboarding context (asset class / "
                        "product profile) when deterministic tokens are weak. Defaults "
                        "to following --enable-llm-target-advisor.")
    p.add_argument("--no-context-resolver", dest="enable_context_resolver",
                   action="store_false",
                   help="disable LLM onboarding-context resolution (deterministic only).")
    p.add_argument("--override-reporting-date", action="store_true",
                   help="force --reporting-date to win even when a source-derived "
                        "data_cut_off_date exists (recorded as cli_override).")
    return p


def _print_console(s: Dict[str, Any]) -> None:
    line = "=" * 64
    print(line)
    print(f"Operator workflow completed for: {s['client_name']}")
    print(f"Status: {s['status']}")
    print(f"Run ID: {s['workflow_run_id']}")
    if s.get("target_contract_id"):
        print(f"Target contract: {s['target_contract_id']}")
    if s.get("target_contract_id") == "esma_annex_2":
        print(f"Regime config: {s.get('regime_config_path', '')}")
        print(f"Asset config: {s.get('asset_config_path', '')}")
        print(f"Annex 2 target universe: {s.get('annex2_authoritative_field_count', 0)} "
              f"(28a coverage: {s.get('annex2_coverage_field_count', 0)}, "
              f"full regime rules: {s.get('annex2_regime_rule_count', 0)}, "
              f"deferred/pending: {s.get('annex2_deferred_field_count', 0)})")
        print(f"Annex 2 registry mapped: {s.get('annex2_registry_mapped_count', 0)} "
              f"(gaps: {s.get('annex2_registry_gap_count', 0)})")
        print(f"Annex 2 codes missing from 28a: {s.get('annex2_missing_from_28a_count', 0)}")
        print(f"Annex 2 active phantom deferred: {s.get('annex2_active_phantom_deferred_count', 0)}")
        print(f"Annex 2 ND broader (compliance risk): {s.get('annex2_nd_regime_broader_count', 0)} · "
              f"divergent (review): {s.get('annex2_nd_divergent_count', 0)} · "
              f"stricter (policy): {s.get('annex2_nd_regime_stricter_count', 0)}")
        print(f"Annex 2 asset-default conflicts: {s.get('annex2_asset_default_conflict_count', 0)}")
        print(f"Annex 2 invalid asset defaults surfaced: {s.get('annex2_invalid_default_count', 0)}")
    print("")
    print(f"Target fields: {s['target_fields_count']}")
    print(f"Gate 4 decisions remaining: {s['human_decision_queue_count_after']}")
    print(f"Blocking decisions: {s['blocking_decisions_count']}")
    print(f"Non-blocking confirmations: {s['non_blocking_confirmations_count']}")
    if s["workflow_stage"] == SECOND_PASS:
        print("")
        print(f"Approved decisions supplied: {s['approved_decisions_supplied_count']}")
        print(f"Applied decisions: {s['applied_decisions_count']}")
        print(f"Remaining Gate 4 decisions: {s['human_decision_queue_count_after']}")
    print("")
    print("Decision template:")
    print(f"  {s['target_first_decisions_template_file'] or '—'}")
    print("")
    print("LLM target advisor:")
    print(f"  enabled: {str(s['llm_target_advisor_enabled']).lower()}")
    print(f"  advised: {s['llm_target_advisor_advised_count']}")
    print(f"  cost: £{s['llm_target_advisor_estimated_cost_gbp']}")
    if s.get("llm_target_advisor_advised_count"):
        print("  note: recommendations are advisory. To apply them, run "
              "`python -m engine.onboarding_agent.cli accept-target-advice "
              f"--project-dir {s.get('project_dir', '<project-dir>')}` or approve manually.")
    print("")
    print("Review pack:")
    print(f"  {s['review_pack_html'] or '—'}")
    print("")
    if s.get("handoff_created"):
        ready_tv = bool(s.get("ready_for_transformation_validation"))
        ready_proj = bool(s.get("ready_for_projection"))
        ready_xml = bool(s.get("ready_for_xml_delivery"))
        print("Handoff lifecycle:")
        print(f"  {HANDOFF_READY_MSG if ready_tv else HANDOFF_NOT_READY_MSG}")
        print(f"  {PROJECTION_READY_MSG if ready_proj else PROJECTION_NOT_READY_MSG}")
        print(f"  {XML_READY_MSG if ready_xml else XML_NOT_READY_MSG}")
        print("")
        print("Downstream / projection action:")
        print(f"  {s.get('downstream_next_action', '')}")
        print("")
    print("Next action:")
    print(f"  {s['next_operator_action']}")
    print(line)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_operator_workflow(
        input_dir=args.input_dir, client_name=args.client_name,
        client_id=args.client_id, run_id=args.run_id, project_dir=args.project_dir,
        output_root=args.output_root, mode=args.mode, registry=args.registry,
        aliases_dir=args.aliases_dir, enable_mapping_review=args.enable_mapping_review,
        enable_llm_target_advisor=args.enable_llm_target_advisor,
        target_first_decisions=args.target_first_decisions,
        llm_max_cost_gbp=args.llm_max_cost_gbp, llm_max_calls=args.llm_max_calls,
        llm_max_items_per_call=args.llm_max_items_per_call,
        target_contract=args.target_contract, regime_config=args.regime_config,
        asset_config=args.asset_config, reporting_date=args.reporting_date,
        override_reporting_date=args.override_reporting_date,
        product_profile=args.product_profile,
        enable_context_resolver=args.enable_context_resolver)
    _print_console(summary)
    return 0 if summary["status"] != FAILED else 2


if __name__ == "__main__":
    raise SystemExit(main())
