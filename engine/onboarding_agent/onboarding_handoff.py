"""
onboarding_handoff.py
=====================

PART 11 — Onboarding Agent → Transformation & Validation HANDOFF package.

The Onboarding Agent's job ends with a *governed canonical handoff package*. It
must NOT:

  * own Annex 2 XML input generation;
  * create bespoke MI / regulatory / investor delivery tapes;
  * re-run raw Gate 1 source canonicalisation on the central tape.

It SHOULD stop after producing a controlled, lineage-rich, approved canonical
handoff package that the downstream Transformation & Validation Agent can consume
without rerunning raw source discovery or fuzzy mapping.

This module reads the artefacts the onboarding workflow already produced
(28a coverage, 28c decision queue, 34/35 decisions, 36 LLM advisory, 43 universe
reconciliation, the central lender tape) and emits, under
``output/handoff/``::

    24_onboarding_handoff_manifest.json
    24_onboarding_handoff_manifest.yaml
    25_onboarding_handoff_readiness.json
    25_onboarding_handoff_readiness.md
    26_onboarding_handoff_field_contract.csv
    26_onboarding_handoff_field_contract.json
    27_onboarding_handoff_lineage.json

It is ADDITIVE — it never mutates the existing onboarding outputs, never changes
the generic central lender tape, and never emits an XML/XSD verdict.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from engine.onboarding_agent import target_coverage as tcov
from engine.onboarding_agent import target_contract_completion as tcc

# --------------------------------------------------------------------------- #
# Handoff vocabulary
# --------------------------------------------------------------------------- #

HANDOFF_TYPE = "canonical_onboarding_package"
HANDOFF_STAGE = "post_onboarding_pre_transformation_validation"
NEXT_AGENT = "transformation_validation"

# Controlled handoff_classification values.
HC_SOURCE_MAPPED = "source_mapped"
HC_SOURCE_MAPPED_ALT = "source_mapped_with_alternatives"
HC_OPERATOR_DECISION_PENDING = "operator_decision_pending"
HC_APPROVED_DECISION_APPLIED = "approved_decision_applied"
HC_CONFIGURED_STATIC = "configured_static"
HC_DEFAULT_DOWNSTREAM = "default_downstream"
HC_ND_DEFAULT_DOWNSTREAM = "nd_default_downstream"
HC_PENDING_REGIME_RULE = "pending_regime_rule"
HC_SOURCE_ABSENT = "source_absent"
HC_ALIAS_MISMATCH = "alias_mismatch"
HC_SEMANTIC_DERIVATION_REQUIRED = "semantic_derivation_required"
HC_TRANSFORMATION_REQUIRED = "transformation_required"
HC_PROJECTION_REQUIRED = "projection_required"
HC_DELIVERY_REQUIRED = "delivery_required"
HC_NOT_APPLICABLE = "not_applicable"
# Portfolio-level run / source context fields (one value for the whole tape,
# e.g. data_cut_off_date / RREL6) resolved from the source pack rather than a
# per-loan source column or a static config default.
HC_SOURCE_CONTEXT_MAPPED = "source_context_mapped"
HC_RUN_CONTEXT_MAPPED = "run_context_mapped"

# Controlled downstream_owner values.
OWN_ONBOARDING = "onboarding"
OWN_TRANSFORMATION = "transformation_validation"
OWN_PROJECTION = "projection"
OWN_DELIVERY = "delivery"
OWN_OPERATOR = "operator"

_FIELD_CONTRACT_COLUMNS = [
    "target_contract_id", "esma_code", "target_field", "canonical_field", "domain",
    "coverage_status", "selected_source_file", "selected_source_column",
    "selected_confidence", "selected_value_sample", "alternative_sources",
    "decision_id", "decision_status", "blocking_decision", "lineage_status",
    "handoff_status", "handoff_classification", "next_agent_action",
    "downstream_owner",
    # Target-field disposition carried from the Onboarding completion checklist
    # (29_*). Downstream agents EXECUTE these dispositions; they do not rediscover
    # them. Empty when no checklist row exists for the field.
    "field_disposition", "disposition_source", "requires_client_input",
    "requires_operator_review", "requires_config", "requires_projection_rule",
    "requires_derivation", "blocking_for_validation", "blocking_for_projection",
    "notes",
]


# --------------------------------------------------------------------------- #
# Semantic-derivation rules (do NOT silently alias)
# --------------------------------------------------------------------------- #
#
# Equity-release outstanding balance may include rolled-up interest, fees,
# advances and product-specific accrual mechanics, so an
# ``current_outstanding_balance -> current_principal_balance`` move is a
# semantic derivation decision, never a silent alias.
_SEMANTIC_DERIVATION_RULES: Dict[str, Dict[str, Any]] = {
    "current_principal_balance": {
        "trigger_source_tokens": ("outstanding",),
        "next_agent_action": "define_approved_ERM_balance_derivation_or_operator_decision",
        "note": ("equity-release outstanding balance may include rolled-up interest, "
                 "fees, advances and product-specific accrual mechanics — do not "
                 "silently alias current_outstanding_balance to current_principal_balance"),
    },
}


def _norm(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "")).strip("_")


def is_semantic_derivation(canonical_field: str, selected_source_column: str) -> bool:
    """True when this canonical/source pair must go through an approved derivation.

    Detects the ERM ``current_outstanding_balance -> current_principal_balance``
    case (and any future rule) so it is surfaced rather than silently aliased.
    """
    rule = _SEMANTIC_DERIVATION_RULES.get(str(canonical_field or "").strip())
    if not rule:
        return False
    src = _norm(selected_source_column)
    return any(tok in src for tok in rule["trigger_source_tokens"])


# --------------------------------------------------------------------------- #
# Per-field classification
# --------------------------------------------------------------------------- #

def _canonical_of(row: Dict[str, Any]) -> str:
    return (row.get("canonical_field")
            or row.get("projected_source_field")
            or row.get("match_field") or "")


def classify_field(row: Dict[str, Any], *, decision_applied: bool = False) -> Dict[str, str]:
    """Classify one 28a coverage row for the downstream agent.

    Returns ``handoff_classification`` / ``downstream_owner`` /
    ``next_agent_action`` / ``handoff_status`` using the controlled vocabulary.
    Pure + unit-testable (no IO).
    """
    status = row.get("coverage_status", "")
    required = str(row.get("required_status", "")).lower()
    requires_decision = bool(row.get("requires_user_decision"))
    blocking = bool(row.get("blocking"))
    default_src = row.get("default_rule_source", "")
    canonical = _canonical_of(row)
    sel_col = row.get("selected_source_column", "")
    nd_default = tcov._is_nd(row.get("default_value"))

    def out(classification: str, owner: str, action: str,
            handoff_status: str = "classified") -> Dict[str, str]:
        return {
            "handoff_classification": classification,
            "downstream_owner": owner,
            "next_agent_action": action,
            "handoff_status": handoff_status,
        }

    # 1) Semantic-derivation guard takes precedence — never silently aliased.
    if canonical and is_semantic_derivation(canonical, sel_col):
        rule = _SEMANTIC_DERIVATION_RULES[canonical]
        return out(HC_SEMANTIC_DERIVATION_REQUIRED, OWN_TRANSFORMATION,
                   rule["next_agent_action"], "needs_semantic_derivation")

    # 2) An operator decision still pending blocks the onboarding handoff item.
    if requires_decision and blocking:
        return out(HC_OPERATOR_DECISION_PENDING, OWN_OPERATOR,
                   "resolve_blocking_operator_decision", "blocking")
    if requires_decision and not decision_applied:
        return out(HC_OPERATOR_DECISION_PENDING, OWN_OPERATOR,
                   "confirm_non_blocking_operator_decision", "needs_confirmation")

    if status == tcov.SOURCE_MAPPED:
        cls = HC_APPROVED_DECISION_APPLIED if decision_applied else HC_SOURCE_MAPPED
        return out(cls, OWN_TRANSFORMATION, "consume_mapped_canonical_source", "resolved")
    if status == tcov.SOURCE_MAPPED_ALT:
        return out(HC_SOURCE_MAPPED_ALT, OWN_TRANSFORMATION,
                   "consume_selected_source_retain_alternatives", "resolved")
    if status == tcov.DERIVED:
        return out(HC_TRANSFORMATION_REQUIRED, OWN_TRANSFORMATION,
                   "apply_derivation_rule", "needs_transformation")
    if status == tcov.CONFIGURED_STATIC:
        return out(HC_CONFIGURED_STATIC, OWN_TRANSFORMATION,
                   "materialise_configured_static_value", "needs_transformation")
    if status == tcov.DEFAULTED_VALUE or (status == tcov.DEFAULTED and not nd_default):
        action = ("materialise_default_from_asset_config" if default_src == "asset_config"
                  else "materialise_default_from_regime_config")
        return out(HC_DEFAULT_DOWNSTREAM, OWN_TRANSFORMATION, action, "needs_transformation")
    if status == tcov.DEFAULTED_ND or (status == tcov.DEFAULTED and nd_default):
        return out(HC_ND_DEFAULT_DOWNSTREAM, OWN_TRANSFORMATION,
                   "materialise_nd_default_if_still_unmapped", "needs_transformation")
    if status in (tcov.PENDING_REGIME_RULE, tcov.DEFERRED):
        return out(HC_PENDING_REGIME_RULE, OWN_PROJECTION,
                   "implement_or_defer_regime_rule", "needs_projection")
    if status == tcov.MISSING_REQUIRED:
        if blocking:
            return out(HC_OPERATOR_DECISION_PENDING, OWN_OPERATOR,
                       "resolve_blocking_missing_required", "blocking")
        return out(HC_SOURCE_ABSENT, OWN_TRANSFORMATION,
                   "materialise_default_or_flag_source_absent", "needs_transformation")
    if status == tcov.NOT_APPLICABLE:
        return out(HC_NOT_APPLICABLE, OWN_ONBOARDING, "none", "resolved")
    if status == tcov.NEEDS_CONFIRMATION:
        return out(HC_SOURCE_ABSENT, OWN_TRANSFORMATION,
                   "materialise_default_or_flag_source_absent", "needs_transformation")
    if status == tcov.OPTIONAL_FOR_MI:
        return out(HC_NOT_APPLICABLE, OWN_ONBOARDING, "none", "resolved")

    # Fallback — never silently drop a field.
    return out(HC_SOURCE_ABSENT, OWN_TRANSFORMATION,
               "review_unclassified_coverage_status", "needs_transformation")


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _p(project_dir: Path, name: str) -> str:
    p = project_dir / name
    return str(p) if p.exists() else ""


def _rel(path: str, base: Path) -> str:
    if not path:
        return ""
    try:
        return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return str(path)


def _contract_name(contract_id: str) -> str:
    return {
        "esma_annex_2": "ESMA Annex 2 — non-ABCP underlying exposures",
        "mi_semantics_field_registry": "MI semantics field registry",
    }.get(contract_id, contract_id or "")


# --------------------------------------------------------------------------- #
# Central tape resolution (build if absent — never re-canonicalise it)
# --------------------------------------------------------------------------- #

def _resolve_central_tape(
    project_dir: Path,
    output_root: Path,
    *,
    client_id: str,
    run_id: str,
    mode: str,
    registry: str,
) -> Dict[str, Any]:
    """Locate the central lender tape; build it (dry-run) if not yet present.

    Returns ``{path, exists, row_count, field_count, lineage_path}``. Building it
    consolidates the already-approved mapping candidates — it never re-runs raw
    Gate 1 canonicalisation.
    """
    central = output_root / "central" / "18_central_lender_tape.csv"
    lineage = output_root / "lineage" / "18b_central_tape_lineage.csv"
    if central.exists():
        return _central_stats(central, lineage)

    try:
        from engine.onboarding_agent import central_tape_builder, storage_paths
        run_paths = storage_paths.resolve_run_paths(
            project_dir=str(project_dir), output_root=str(output_root),
            client_id=client_id, run_id=run_id)
        tape_result = central_tape_builder.build_central_tapes(
            project_dir, run_paths, registry, mode=mode)
        cpath = Path(tape_result.get("central_lender_tape_path", central))
        lpath = Path(tape_result.get("central_tape_lineage_path", lineage))
        stats = _central_stats(cpath, lpath)
        stats["row_count"] = int(tape_result.get("loan_count", stats["row_count"]))
        stats["field_count"] = int(tape_result.get("mapped_field_count", stats["field_count"]))
        return stats
    except Exception:
        return {"path": str(central), "exists": False, "row_count": 0,
                "field_count": 0, "lineage_path": str(lineage)}


def _central_stats(central: Path, lineage: Path) -> Dict[str, Any]:
    row_count = 0
    field_count = 0
    if Path(central).exists():
        with open(central, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            field_count = len(header)
            row_count = sum(1 for _ in reader)
    return {
        "path": str(central),
        "exists": Path(central).exists(),
        "row_count": row_count,
        "field_count": field_count,
        "lineage_path": str(lineage) if Path(lineage).exists() else "",
    }


# --------------------------------------------------------------------------- #
# Field contract + lineage
# --------------------------------------------------------------------------- #

def build_field_contract(
    coverage_rows: List[Dict[str, Any]],
    decisions_by_field: Dict[str, Dict[str, Any]],
    applied_fields: set,
) -> List[Dict[str, Any]]:
    """One row-level contract entry per target field in 28a."""
    contract: List[Dict[str, Any]] = []
    for row in coverage_rows:
        tf = row.get("target_field", "")
        decision = decisions_by_field.get(tf, {})
        applied = tf in applied_fields
        cls = classify_field(row, decision_applied=applied)
        sel_file = row.get("selected_source_file", "")
        lineage_status = "source_linked" if sel_file else (
            "configured" if row.get("selected_value") else "unlinked")
        contract.append({
            "target_contract_id": row.get("target_contract_id", ""),
            "esma_code": row.get("esma_code", ""),
            "target_field": tf,
            "canonical_field": _canonical_of(row),
            "domain": row.get("target_domain", row.get("domain", "")),
            "coverage_status": row.get("coverage_status", ""),
            "selected_source_file": sel_file,
            "selected_source_column": row.get("selected_source_column", ""),
            "selected_confidence": row.get("selected_source_confidence", ""),
            "selected_value_sample": row.get("selected_value", ""),
            "alternative_sources": row.get("alternative_source_candidates", ""),
            "decision_id": decision.get("decision_id", ""),
            "decision_status": ("applied" if applied else
                                ("pending" if decision else "")),
            "blocking_decision": bool(decision.get("blocking", row.get("blocking", False))),
            "lineage_status": lineage_status,
            "handoff_status": cls["handoff_status"],
            "handoff_classification": cls["handoff_classification"],
            "next_agent_action": cls["next_agent_action"],
            "downstream_owner": cls["downstream_owner"],
            "notes": row.get("decision_reason", "") or row.get("default_reason", ""),
        })
    return contract


def _apply_dispositions_to_contract(
    contract: List[Dict[str, Any]],
    disposition_by_code: Dict[str, Dict[str, Any]],
) -> None:
    """Carry the onboarding disposition columns onto the handoff field contract.

    Additive: fields without a checklist row keep empty disposition columns and
    their existing classification is untouched.
    """
    keys = ("field_disposition", "disposition_source", "requires_client_input",
            "requires_operator_review", "requires_config", "requires_projection_rule",
            "requires_derivation", "blocking_for_validation", "blocking_for_projection")
    for c in contract:
        disp = disposition_by_code.get(c.get("esma_code", "")) or \
            disposition_by_code.get(c.get("target_field", ""))
        if not disp:
            continue
        for k in keys:
            c[k] = disp.get(k, "")


def _build_completion_checklist(
    project_dir: Path,
    coverage_rows: List[Dict[str, Any]],
    *,
    contract_id: str,
    client_id: str,
    run_id: str,
    registry: str,
    regime_config_path: str,
    asset_config_path: str,
) -> Dict[str, Any]:
    """Build + write the 29_* completion checklist and 29a_* review bench.

    Returns ``{checklist_rows, review_rows, disposition_by_code, paths, summary}``.
    Never raises — a config/load failure yields an empty checklist so the handoff
    remains additive and robust.
    """
    empty = {"checklist_rows": [], "review_rows": [], "disposition_by_code": {},
             "paths": {}, "summary": {}}
    try:
        repo_root = Path(__file__).resolve().parents[2]
        regime_path = regime_config_path or str(repo_root / "config" / "regime" / "annex2_delivery_rules.yaml")
        asset_path = asset_config_path or str(repo_root / "config" / "asset" / "product_defaults_ERM.yaml")
        registry_path = registry or str(repo_root / "config" / "system" / "fields_registry.yaml")
        if registry_path and not Path(registry_path).is_absolute() and not Path(registry_path).exists():
            registry_path = str(repo_root / registry_path)
        universe_path = str(tcov._ANNEX2_UNIVERSE_DEFAULT)

        cfgs = tcc.load_target_contract_configs(
            regime_config_path=regime_path, field_universe_path=universe_path,
            registry_path=registry_path, asset_config_path=asset_path)
        asset_cfg = cfgs["asset_cfg"]
        asset_class = str(asset_cfg.get("asset_class", "") or "equity_release")

        coverage_by_code = {}
        for row in coverage_rows:
            code = row.get("esma_code", "")
            if code and code not in coverage_by_code:
                coverage_by_code[code] = row

        checklist_rows = tcc.build_completion_checklist(
            contract_id=contract_id or "ESMA_Annex2",
            field_universe=cfgs["field_universe"], registry_fields=cfgs["registry_fields"],
            regime_index=cfgs["regime_index"], asset_cfg=asset_cfg,
            asset_class=asset_class, coverage_by_code=coverage_by_code,
            client_policy=cfgs["client_policy"])
        review_rows = tcc.build_review_bench(checklist_rows)
        paths = tcc.write_checklist_artefacts(
            project_dir, checklist_rows, review_rows,
            contract_id=contract_id, client_id=client_id, run_id=run_id)
        disposition_by_code = {r["esma_code"]: r for r in checklist_rows if r.get("esma_code")}
        return {"checklist_rows": checklist_rows, "review_rows": review_rows,
                "disposition_by_code": disposition_by_code, "paths": paths,
                "summary": paths.get("summary", {})}
    except Exception:
        return empty


def build_lineage(coverage_rows: List[Dict[str, Any]],
                  contract_by_field: Dict[str, Dict[str, Any]],
                  decisions_by_field: Dict[str, Dict[str, Any]],
                  llm_by_field: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Material candidate/selected lineage for each target field."""
    lineage: List[Dict[str, Any]] = []
    for row in coverage_rows:
        tf = row.get("target_field", "")
        c = contract_by_field.get(tf, {})
        sel_file = row.get("selected_source_file", "")
        sel_value = row.get("selected_value", "")
        # Only record material rows (a source, a value, or a classification of note).
        if not (sel_file or sel_value or c.get("handoff_classification") not in (
                HC_NOT_APPLICABLE,)):
            continue
        decision = decisions_by_field.get(tf, {})
        llm = llm_by_field.get(tf, {})
        lineage.append({
            "target_field": tf,
            "esma_code": row.get("esma_code", ""),
            "canonical_field": _canonical_of(row),
            "source_file": sel_file,
            "source_column": row.get("selected_source_column", ""),
            "mapping_confidence": row.get("selected_source_confidence", ""),
            "classification": c.get("handoff_classification", ""),
            "downstream_owner": c.get("downstream_owner", ""),
            "operator_decision_id": decision.get("decision_id", ""),
            "operator_decision_status": c.get("decision_status", ""),
            "llm_recommendation_id": llm.get("recommendation_id", llm.get("decision_id", "")),
            "lineage_note": (row.get("decision_reason", "")
                             or row.get("default_reason", "")
                             or row.get("coverage_basis", "")),
        })
    return lineage


# --------------------------------------------------------------------------- #
# Counts + readiness
# --------------------------------------------------------------------------- #

def _counts(contract: List[Dict[str, Any]]) -> Dict[str, int]:
    by_cls: Dict[str, int] = {}
    by_owner: Dict[str, int] = {}
    for c in contract:
        by_cls[c["handoff_classification"]] = by_cls.get(c["handoff_classification"], 0) + 1
        by_owner[c["downstream_owner"]] = by_owner.get(c["downstream_owner"], 0) + 1
    g = by_cls.get
    source_mapped = (g(HC_SOURCE_MAPPED, 0) + g(HC_APPROVED_DECISION_APPLIED, 0)
                     + g(HC_SOURCE_CONTEXT_MAPPED, 0) + g(HC_RUN_CONTEXT_MAPPED, 0))
    default_required = g(HC_DEFAULT_DOWNSTREAM, 0) + g(HC_ND_DEFAULT_DOWNSTREAM, 0)
    transformation_required = (g(HC_TRANSFORMATION_REQUIRED, 0)
                               + g(HC_CONFIGURED_STATIC, 0)
                               + default_required
                               + g(HC_SEMANTIC_DERIVATION_REQUIRED, 0)
                               + g(HC_SOURCE_ABSENT, 0))
    projection_required = g(HC_PENDING_REGIME_RULE, 0) + g(HC_PROJECTION_REQUIRED, 0)
    return {
        "target_field_count": len(contract),
        "source_mapped_count": source_mapped,
        "source_mapped_with_alternatives_count": g(HC_SOURCE_MAPPED_ALT, 0),
        "derived_count": g(HC_TRANSFORMATION_REQUIRED, 0),
        "configured_static_count": g(HC_CONFIGURED_STATIC, 0),
        "defaulted_value_count": g(HC_DEFAULT_DOWNSTREAM, 0),
        "defaulted_nd_count": g(HC_ND_DEFAULT_DOWNSTREAM, 0),
        "pending_regime_rule_count": g(HC_PENDING_REGIME_RULE, 0),
        "source_absent_count": g(HC_SOURCE_ABSENT, 0),
        "operator_decision_pending_count": g(HC_OPERATOR_DECISION_PENDING, 0),
        "semantic_derivation_required_count": g(HC_SEMANTIC_DERIVATION_REQUIRED, 0),
        "downstream_default_required_count": default_required,
        "downstream_transformation_required_count": transformation_required,
        "downstream_projection_required_count": projection_required,
        "approved_decision_applied_count": g(HC_APPROVED_DECISION_APPLIED, 0),
        "not_applicable_count": g(HC_NOT_APPLICABLE, 0),
        "owner_counts": by_owner,
        "classification_counts": by_cls,
    }


def compute_readiness(
    *,
    central_exists: bool,
    coverage_present: bool,
    target_universe_loaded: bool,
    registry_gap_count: int,
    blocking_decision_count: int,
    counts: Dict[str, int],
) -> Dict[str, bool]:
    """Onboarding handoff readiness — separate from Annex 2 XML readiness."""
    unresolved_classified = True  # every field carries a controlled classification

    ready_for_transformation_validation = bool(
        central_exists
        and coverage_present
        and target_universe_loaded
        and registry_gap_count == 0
        and blocking_decision_count == 0
        and unresolved_classified
    )

    # Onboarding never claims projection-readiness: pending rules, unresolved
    # semantic derivations, or not-yet-materialised transformation outputs all
    # keep this false.
    ready_for_projection = bool(
        ready_for_transformation_validation
        and counts.get("pending_regime_rule_count", 0) == 0
        and counts.get("semantic_derivation_required_count", 0) == 0
        and counts.get("downstream_transformation_required_count", 0) == 0
        and counts.get("operator_decision_pending_count", 0) == 0
    )

    # XML delivery is only ready once Transformation/Validation/Projection have
    # produced an XML-ready target frame — never at the onboarding stage.
    ready_for_xml_delivery = False

    return {
        "ready_for_transformation_validation": ready_for_transformation_validation,
        "ready_for_projection": ready_for_projection,
        "ready_for_xml_delivery": ready_for_xml_delivery,
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Run / source context fields (portfolio-level, e.g. data_cut_off_date)
# --------------------------------------------------------------------------- #

# Canonical fields treated as portfolio-level run/source context (one value for
# the whole tape) rather than per-loan source columns.
_RUN_CONTEXT_FIELDS = ("data_cut_off_date",)


def _resolve_run_context(
    project_dir: Path, output_root: Path, central_path: str, *,
    asset_config_path: str = "", regime_config_path: str = "",
    reporting_date: str = "", override_reporting_date: bool = False,
    reporting_period: str = "", run_id: str = "", managed_service: bool = False,
) -> Dict[str, Any]:
    """Resolve portfolio-level context fields (currently data_cut_off_date).

    In managed-service mode the blob-event ``reporting_period`` (the folder
    period) is consumed as the first-class ``folder_period`` source and all CLI
    fallback semantics are disabled — there is no path to a ``cli_fallback``
    provenance. In interactive CLI mode ``reporting_date`` remains an accepted
    fallback/override.
    """
    try:
        from engine.onboarding_agent import run_context as rc
        return rc.extract_data_cut_off_date(
            project_dir, central_path,
            asset_config_path=asset_config_path,
            regime_config_path=regime_config_path,
            cli_reporting_date=reporting_date,
            override_reporting_date=override_reporting_date,
            folder_period=reporting_period,
            run_id=run_id,
            managed_service=managed_service)
    except Exception as exc:  # never break the handoff on context extraction
        return {"value": "", "source": "", "source_file": "", "source_location": "",
                "confidence": 0.0, "candidates": [], "conflict": False,
                "conflict_detail": f"extraction_error: {exc}", "missing": True}


def _apply_run_context_to_contract(
    contract: List[Dict[str, Any]],
    lineage: List[Dict[str, Any]],
    run_context: Dict[str, Any],
) -> bool:
    """Stamp the resolved run-context date onto the data_cut_off_date contract row.

    Returns True when the context is in conflict (a blocking operator item).
    """
    value = run_context.get("value", "")
    source = run_context.get("source", "")
    conflict = bool(run_context.get("conflict"))
    rows = [c for c in contract if c.get("canonical_field") in _RUN_CONTEXT_FIELDS]
    for c in rows:
        if conflict:
            c["handoff_classification"] = HC_OPERATOR_DECISION_PENDING
            c["downstream_owner"] = OWN_OPERATOR
            c["handoff_status"] = "blocking"
            c["next_agent_action"] = "resolve_conflicting_run_context_date"
            c["blocking_decision"] = True
            c["notes"] = (run_context.get("conflict_detail", "")
                          or "conflicting run-context date candidates")
            continue
        if not value:
            # Missing — leave the row's coverage-based classification so Validation
            # can still fail later; record the gap in notes.
            c["notes"] = (c.get("notes", "")
                          or "run-context date not found in source pack/config")
            continue
        cls = (HC_SOURCE_CONTEXT_MAPPED if source in ("source_column",)
               else HC_RUN_CONTEXT_MAPPED)
        c["handoff_classification"] = cls
        c["downstream_owner"] = OWN_TRANSFORMATION
        c["handoff_status"] = "resolved"
        c["next_agent_action"] = "materialise_run_context_value"
        c["selected_value_sample"] = value
        c["lineage_status"] = "source_context_linked"
        c["notes"] = (f"portfolio-level {source} "
                      f"({run_context.get('source_location', '')})")
        # extend lineage with explicit context evidence
        lineage.append({
            "target_field": c.get("target_field", ""),
            "esma_code": c.get("esma_code", ""),
            "canonical_field": c.get("canonical_field", ""),
            "source_file": run_context.get("source_file", ""),
            "source_column": run_context.get("source_location", ""),
            "mapping_confidence": run_context.get("confidence", ""),
            "classification": cls,
            "downstream_owner": OWN_TRANSFORMATION,
            "operator_decision_id": "",
            "operator_decision_status": "",
            "llm_recommendation_id": "",
            "lineage_note": f"run_context_resolved value={value} source={source}",
        })
    return conflict


def build_handoff_package(
    project_dir: str | Path,
    output_root: str | Path | None = None,
    *,
    client_id: str = "",
    client_name: str = "",
    run_id: str = "",
    mode: str = "regulatory_mi",
    registry: str = "config/system/fields_registry.yaml",
    aliases_dir: str = "config/system",
    regime_config_path: str = "",
    asset_config_path: str = "",
    decisions_supplied_file: str = "",
    reporting_date: str = "",
    override_reporting_date: bool = False,
    reporting_period: str = "",
    managed_service: bool = False,
) -> Optional[Dict[str, Any]]:
    """Build the governed canonical onboarding handoff package (24–27).

    Returns the handoff manifest dict (and paths), or ``None`` if the required
    coverage matrix (28a) is not present (e.g. a failed run).
    """
    project_dir = Path(project_dir)
    output_root = Path(output_root) if output_root else (project_dir / "output")
    handoff_dir = output_root / "handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)

    cov = _read_json(project_dir / "28a_target_coverage_matrix.json")
    if not cov:
        return None
    coverage_rows = cov.get("rows", []) or []
    contract_id = cov.get("target_contract_id", "")

    dec = _read_json(project_dir / "28c_human_decision_queue.json") or {}
    decision_rows = dec.get("rows", []) or []
    dec_sum = dec.get("summary", {}) or {}
    decisions_by_field = {d.get("target_field", ""): d
                          for d in decision_rows if d.get("target_field")}

    # Applied approved decisions (35) — fields whose decisions were applied.
    app = _read_json(project_dir / "35_target_first_decision_application_log.json") or {}
    applied_fields = set()
    for r in (app.get("rows", []) or []):
        if str(r.get("result", r.get("status", ""))).lower() in ("applied", "ok", "success"):
            applied_fields.add(r.get("target_field", ""))

    # LLM advisory recommendations (36) — advisory only; never alter handoff state.
    llm = _read_json(project_dir / "36_target_first_llm_recommendations.json") or {}
    llm_by_field = {r.get("target_field", ""): r
                    for r in (llm.get("rows", []) or []) if r.get("target_field")}
    llm_present = bool(llm_by_field)

    # Universe reconciliation (43) for registry-gap classification.
    recon = _read_json(project_dir / "43_annex2_field_universe_reconciliation.json") or {}
    registry_gap_count = int((recon.get("summary", {}) or {}).get("registry_gap_count", 0))

    central = _resolve_central_tape(
        project_dir, output_root, client_id=client_id, run_id=run_id,
        mode=mode, registry=registry)

    # --- field contract + lineage ---
    contract = build_field_contract(coverage_rows, decisions_by_field, applied_fields)
    contract_by_field = {c["target_field"]: c for c in contract}

    # --- target contract completion checklist / disposition layer (29 / 29a) ---
    # Onboarding owns target-field disposition; downstream agents execute it.
    completion = _build_completion_checklist(
        project_dir, coverage_rows, contract_id=contract_id,
        client_id=client_id, run_id=run_id, registry=registry,
        regime_config_path=regime_config_path, asset_config_path=asset_config_path)
    _apply_dispositions_to_contract(contract, completion["disposition_by_code"])
    lineage = build_lineage(coverage_rows, contract_by_field, decisions_by_field, llm_by_field)

    # --- run / source context fields (portfolio-level, e.g. data_cut_off_date) --
    run_context = _resolve_run_context(
        project_dir, output_root, central["path"],
        asset_config_path=asset_config_path, regime_config_path=regime_config_path,
        reporting_date=reporting_date, override_reporting_date=override_reporting_date,
        reporting_period=reporting_period, run_id=run_id, managed_service=managed_service)
    context_conflict = _apply_run_context_to_contract(contract, lineage, run_context)

    counts = _counts(contract)

    blocking_decision_count = int(dec_sum.get("blocking_decisions",
                                              sum(1 for d in decision_rows if d.get("blocking"))))
    # A conflicting (non-silently resolvable) run-context date is a blocking
    # operator item — surfaced, never auto-picked.
    if context_conflict:
        blocking_decision_count += 1
    non_blocking_decision_count = max(0, len(decision_rows) - blocking_decision_count)

    readiness = compute_readiness(
        central_exists=central["exists"],
        coverage_present=bool(coverage_rows),
        target_universe_loaded=bool(coverage_rows),
        registry_gap_count=registry_gap_count,
        blocking_decision_count=blocking_decision_count,
        counts=counts,
    )

    # --- artefact paths (written below) ---
    manifest_json = handoff_dir / "24_onboarding_handoff_manifest.json"
    manifest_yaml = handoff_dir / "24_onboarding_handoff_manifest.yaml"
    readiness_json = handoff_dir / "25_onboarding_handoff_readiness.json"
    readiness_md = handoff_dir / "25_onboarding_handoff_readiness.md"
    contract_csv = handoff_dir / "26_onboarding_handoff_field_contract.csv"
    contract_json = handoff_dir / "26_onboarding_handoff_field_contract.json"
    lineage_path = handoff_dir / "27_onboarding_handoff_lineage.json"

    manifest: Dict[str, Any] = {
        "client_id": client_id,
        "client_name": client_name,
        "run_id": run_id,
        "created_at": _now(),
        "workflow_mode": mode,
        "target_contract_id": contract_id,
        "target_contract_name": _contract_name(contract_id),

        # Governance flags — what this package IS and IS NOT.
        "handoff_type": HANDOFF_TYPE,
        "handoff_stage": HANDOFF_STAGE,
        "next_agent": NEXT_AGENT,
        "not_raw_source": True,
        "not_xml_ready": not readiness["ready_for_xml_delivery"],
        "do_not_rerun_gate1_on_central_tape": True,

        # Central canonical tape (consume — do not re-canonicalise).
        "central_tape_path": _rel(central["path"], output_root),
        "central_tape_row_count": central["row_count"],
        "central_tape_field_count": central["field_count"],

        # Portfolio-level run / source context fields (e.g. data_cut_off_date).
        "run_context_fields": list(_RUN_CONTEXT_FIELDS),
        "source_context_fields": (["data_cut_off_date"]
                                  if run_context.get("source") in ("source_column",)
                                  else []),
        "data_cut_off_date": run_context.get("value", ""),
        "data_cut_off_date_source": run_context.get("source", ""),
        "data_cut_off_date_source_file": run_context.get("source_file", ""),
        "data_cut_off_date_source_column_or_location": run_context.get("source_location", ""),
        "data_cut_off_date_confidence": run_context.get("confidence", 0.0),
        "data_cut_off_date_conflict": bool(run_context.get("conflict", False)),
        "data_cut_off_date_missing": bool(run_context.get("missing", False)),
        "data_cut_off_date_candidates": run_context.get("candidates", []),

        # Onboarding artefact references the next agent should read.
        "target_coverage_matrix_path": _p(project_dir, "28a_target_coverage_matrix.csv"),
        "target_contract_completion_checklist_path": _p(
            project_dir, "29_target_contract_completion_checklist.csv"),
        "target_contract_review_bench_path": _p(
            project_dir, "29a_target_contract_review_bench.csv"),
        "target_contract_completion_summary": completion.get("summary", {}),
        "decision_queue_path": _p(project_dir, "28c_human_decision_queue.csv"),
        "approved_decisions_path": (decisions_supplied_file
                                    or _p(project_dir, "34_target_first_decisions.yaml")),
        "decision_application_log_path": _p(
            project_dir, "35_target_first_decision_application_log.json"),
        "lineage_path": str(lineage_path),
        "field_contract_path": str(contract_csv),
        "readiness_path": str(readiness_json),
        "review_pack_path": _p(project_dir, "08_onboarding_review_pack.html"),
        "summary_path": _p(project_dir, "40_operator_workflow_summary.json"),
        "regime_config_path": regime_config_path,
        "asset_config_path": asset_config_path,
        "registry_path": registry,
        "aliases_dir": aliases_dir,

        # Counts (downstream planning).
        **counts,
        "blocking_decision_count": blocking_decision_count,
        "non_blocking_decision_count": non_blocking_decision_count,
        "registry_gap_count": registry_gap_count,

        # LLM advisory is advisory-only and never required for the handoff.
        "llm_recommendations_present": llm_present,
        "llm_recommendations_advisory_only": True,

        # Readiness (handoff readiness != XML readiness).
        **readiness,
        "ready_for_transformation_validation": readiness["ready_for_transformation_validation"],
    }
    # Drop verbose nested count maps from the flat manifest top-level note.
    manifest.pop("owner_counts", None)
    manifest.pop("classification_counts", None)

    readiness_doc = {
        "client_id": client_id,
        "run_id": run_id,
        "target_contract_id": contract_id,
        "handoff_type": HANDOFF_TYPE,
        "next_agent": NEXT_AGENT,
        "central_tape_present": central["exists"],
        "coverage_matrix_present": bool(coverage_rows),
        "target_universe_loaded": bool(coverage_rows),
        "registry_gap_count": registry_gap_count,
        "blocking_decision_count": blocking_decision_count,
        "operator_decision_pending_count": counts["operator_decision_pending_count"],
        "downstream_default_required_count": counts["downstream_default_required_count"],
        "pending_regime_rule_count": counts["pending_regime_rule_count"],
        "semantic_derivation_required_count": counts["semantic_derivation_required_count"],
        "source_absent_count": counts["source_absent_count"],
        "llm_recommendations_present": llm_present,
        "llm_recommendations_advisory_only": True,
        **readiness,
    }

    # --- write artefacts ---
    manifest_json.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    manifest_yaml.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    readiness_json.write_text(json.dumps(readiness_doc, indent=2, default=str), encoding="utf-8")
    readiness_md.write_text(_readiness_md(manifest, readiness_doc, counts), encoding="utf-8")

    contract_json.write_text(json.dumps(
        {"target_contract_id": contract_id,
         "summary": {"classification_counts": counts["classification_counts"],
                     "owner_counts": counts["owner_counts"]},
         "rows": contract}, indent=2, default=str), encoding="utf-8")
    with open(contract_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELD_CONTRACT_COLUMNS)
        w.writeheader()
        for c in contract:
            w.writerow({k: c.get(k, "") for k in _FIELD_CONTRACT_COLUMNS})

    lineage_path.write_text(json.dumps(
        {"target_contract_id": contract_id, "rows": lineage},
        indent=2, default=str), encoding="utf-8")

    # --- governed risk-limits config contract (output/risk/risk_limits_config.yaml) ---
    # Discover the client's Schedule 8 doc, parse it deterministically and emit the
    # production config the MI Risk Limits panel reads. Best-effort + self-describing
    # (source_type / extraction_status / is_placeholder); never fabricates limits and
    # never breaks the handoff.
    risk_config_path = ""
    try:
        from mi_agent.risk_monitor import risk_limits_contract as _rlc
        _risk_cfg = _rlc.build_config(
            client_id or "client",
            search_roots=[str(project_dir), str(project_dir / "docs"),
                          str(project_dir / "input"), str(project_dir / "input" / "docs")],
            extracted_at=_now())
        risk_config_path = str(_rlc.write_config_to_output_dir(output_root, _risk_cfg))
    except Exception:  # noqa: BLE001 — additive, must never fail the handoff
        risk_config_path = ""

    return {
        "manifest": manifest,
        "risk_limits_config_path": risk_config_path,
        "readiness": readiness_doc,
        "manifest_json_path": str(manifest_json),
        "manifest_yaml_path": str(manifest_yaml),
        "readiness_json_path": str(readiness_json),
        "readiness_md_path": str(readiness_md),
        "field_contract_csv_path": str(contract_csv),
        "field_contract_json_path": str(contract_json),
        "lineage_path": str(lineage_path),
        "handoff_dir": str(handoff_dir),
        "completion_checklist_csv_path": completion.get("paths", {}).get("checklist_csv_path", ""),
        "completion_checklist_json_path": completion.get("paths", {}).get("checklist_json_path", ""),
        "review_bench_csv_path": completion.get("paths", {}).get("review_bench_csv_path", ""),
        "target_contract_completion_summary": completion.get("summary", {}),
    }


def _readiness_md(manifest: Dict[str, Any], r: Dict[str, Any],
                  counts: Dict[str, int]) -> str:
    def yn(v: bool) -> str:
        return "✅ yes" if v else "❌ no"

    md = [
        "# Onboarding handoff readiness", "",
        f"Client: {manifest.get('client_name', '')} ({manifest.get('client_id', '')})  ",
        f"Run: {manifest.get('run_id', '')}  ",
        f"Target contract: {manifest.get('target_contract_id', '')}  ",
        f"Handoff type: **{manifest.get('handoff_type', '')}**  ",
        f"Next agent: **{manifest.get('next_agent', '')}**", "",
        "> The central lender tape is a canonical onboarding handoff artefact. It is "
        "not raw source input and not an XML-ready regulatory delivery tape. "
        "Downstream agents must consume this through the Transformation & Validation "
        "handoff path and must NOT re-run raw Gate 1 canonicalisation on it.", "",
        "## Readiness flags", "",
        f"- ready_for_transformation_validation: {yn(r['ready_for_transformation_validation'])}",
        f"- ready_for_projection: {yn(r['ready_for_projection'])}",
        f"- ready_for_xml_delivery: {yn(r['ready_for_xml_delivery'])}", "",
        "## Gate inputs", "",
        f"- central canonical tape present: {yn(r['central_tape_present'])}",
        f"- 28a coverage matrix present: {yn(r['coverage_matrix_present'])}",
        f"- target universe loaded: {yn(r['target_universe_loaded'])}",
        f"- registry gaps: {r['registry_gap_count']}",
        f"- blocking onboarding decisions: {r['blocking_decision_count']}", "",
        "## Classified for downstream", "",
        f"- operator decisions pending: {counts['operator_decision_pending_count']}",
        f"- downstream defaults required: {counts['downstream_default_required_count']}",
        f"- ND defaults: {counts['defaulted_nd_count']}",
        f"- pending regime rules (projection): {counts['pending_regime_rule_count']}",
        f"- semantic derivations required: {counts['semantic_derivation_required_count']}",
        f"- source absent: {counts['source_absent_count']}",
        f"- source mapped: {counts['source_mapped_count']}", "",
        "## LLM advisory", "",
        f"- recommendations present: {yn(r['llm_recommendations_present'])}",
        "- advisory only (never required for handoff): yes", "",
    ]
    return "\n".join(md) + "\n"
