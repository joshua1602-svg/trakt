"""
target_coverage.py
==================

Target-contract-first onboarding coverage (v1).

The historical mapping review queue (artefact 33) is *source-column-led*: it emits
one review card per source column, which produces a large, noisy operator queue
(172 source columns -> 119 approval rows for the ERE pack). That is not the
desired operating model.

This module flips the workflow to be *target-contract-first*:

    1. Identify mode / asset / regime (already resolved into ``context``).
    2. Load the relevant TARGET CONTRACT for the mode:
         - MI modes      -> the MI semantics field registry
                            (``mi_agent/mi_semantics_field_registry.yaml``)
         - Regulatory    -> the relevant ESMA annex, primarily ANNEX 2
                            (``config/regime/annex2_delivery_rules.yaml``).
                            Annex 12 is deliberately NOT mixed in here.
    3. For each TARGET field, find the best source candidate(s), or a
       derivation / default / ND / configured-static rule, or mark it missing.
    4. Suppress non-target residual source columns from the main approval queue
       (they go to a residual register instead).
    5. Produce a COMPACT human decision queue driven by target coverage
       gaps / conflicts, not by every source column.

Artefacts (numbered after the 28 required-target-contract artefact, before the
source-column 33 review queue):

    28a_target_coverage_matrix.csv / .json / _summary.md
    28b_source_residual_register.csv / .json / _summary.md
    28c_human_decision_queue.csv / .json / _summary.md

This module does NOT redesign the LLM resolver, does NOT batch LLM calls, and
does NOT prune the MI semantics registry. It reuses the existing deterministic
evidence + resolver output and re-frames it around the target contract.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Modes whose target contract is the MI semantics registry (vs an ESMA annex).
_MI_MODES = {"mi_only", "mna_dd"}

# Confidence floors for deterministically-matched source candidates.
_CANONICAL_MATCH_CONF = 0.85
_NAME_MATCH_CONF = 0.65
# Two candidates within this confidence delta are treated as a priority conflict.
_PRIORITY_CONFLICT_DELTA = 0.15

# --- coverage statuses ---
SOURCE_MAPPED = "source_mapped"
SOURCE_MAPPED_ALT = "source_mapped_with_alternatives"
DERIVED = "derived"
CONFIGURED_STATIC = "configured_static"
DEFAULTED = "defaulted"
DEFAULTED_ND = "defaulted_ND"
NOT_APPLICABLE = "not_applicable"
MISSING_REQUIRED = "missing_required"
NEEDS_CONFIRMATION = "needs_confirmation"
OPTIONAL_FOR_MI = "optional_for_mi"

# Overlay coverage statuses that are valid as `coverage_status_if_no_source`.
_OVERLAY_STATUSES = {NOT_APPLICABLE, OPTIONAL_FOR_MI, NEEDS_CONFIRMATION,
                     CONFIGURED_STATIC, DEFAULTED, DERIVED, DEFAULTED_ND}

# --- residual classes ---
R_DUP_ALT = "duplicate_or_alternative_source"
R_RECON = "reconciliation_or_audit_only"
R_CASHFLOW = "cashflow_ledger_support"
R_FUTURE = "useful_future_extension"
R_NULL = "ignored_null_empty"
R_NOT_RELEVANT = "not_relevant_to_current_mode"
R_HEADER = "header_or_parse_issue"

# --- decision types ---
D_MISSING = "missing_required_target"
D_CONFLICT = "conflicting_source_candidates"
D_PRIORITY = "source_priority_confirmation"
D_VALUE = "value_compatibility_conflict"
D_CONFIG = "config_value_required"
D_ND = "nd_default_confirmation"
D_EXTENSION = "reporting_extension_candidate"
D_PARSE = "parse_or_header_blocker"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _norm_field(s: Any) -> str:
    """Canonical comparison key: lowercase, non-alnum -> underscore, trimmed."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(s or "").lower())).strip("_")


def _ek(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (row.get("source_file", ""), row.get("source_sheet", ""),
            row.get("source_column", ""))


def _is_unnamed(col: str) -> bool:
    return bool(re.match(r"^Unnamed: ?\d+$", str(col)) or str(col).strip() == "")


def _is_nd(value: Any) -> bool:
    return bool(re.match(r"^ND[0-9]$", str(value or "").strip().upper()))


def _cashflow_like(ev: Dict[str, Any]) -> bool:
    blob = (str(ev.get("source_column", "")) + " " + str(ev.get("domain_guess", ""))
            + " " + str(ev.get("file_domain_guess", ""))).lower()
    return any(t in blob for t in ("b/f", "c/f", "payment_allocation", "payment allocation",
                                   "ledger", "redemption", "cash paid", "arrears balance",
                                   "cashflow", "cash flow", "brought forward", "carried forward"))


# ---------------------------------------------------------------------------
# Target contract loading (MI semantics registry / ESMA Annex 2)
# ---------------------------------------------------------------------------

def target_contract_kind(mode: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Return 'mi_semantics' or 'esma_annex_2' for the run."""
    if mode in _MI_MODES:
        return "mi_semantics"
    return "esma_annex_2"


def _mi_domain(entry: Dict[str, Any]) -> str:
    crit = (entry.get("source_criteria") or [""])[0]
    if not crit:
        return entry.get("role", "unknown") or "unknown"
    if crit.startswith("layer:"):
        return crit.split(":", 1)[1]
    return {"core_canonical": "core", "derived_bucket": "derived_dimension"}.get(crit, crit)


def load_mi_target_contract(
    registry_path: Optional[str | Path] = None,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Load the MI semantics registry as the MI target contract.

    Returns ``(contract_id, contract_source, target_fields)``. We do NOT invent a
    separate field list and we do NOT prune the registry — it is the source of
    truth for MI target coverage at this stage.
    """
    path = Path(registry_path) if registry_path else (
        _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fields = data.get("fields", {}) or {}
    rows: List[Dict[str, Any]] = []
    for name, entry in fields.items():
        tier = entry.get("mi_tier", "extended")
        derived = bool(entry.get("derived"))
        rows.append({
            "target_field": name,
            "target_domain": _mi_domain(entry),
            "target_label": entry.get("business_name", "") or entry.get("display_name", name),
            "required_status": "required" if tier == "core" else "optional",
            "applicability_status": "applicable",
            "match_field": entry.get("canonical_field", name),
            "synonyms": list(entry.get("synonyms", []) or []),
            "derived": derived,
            "derivation_rule": (f"derived from {entry.get('derived_from')}"
                                if derived and entry.get("derived_from") else ""),
            "default_rule": "",
            "default_value": "",
            "nd_allowed": [],
            "configured_value_source": "",
        })
    return "mi_semantics_field_registry", str(path), rows


def _annex2_domain(code: str) -> str:
    code = str(code or "").upper()
    if code.startswith("RREC"):
        return "collateral"
    if code.startswith("RREL"):
        return "loan"
    return "annex2"


def load_annex2_target_contract(
    config_path: Optional[str | Path] = None,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Load ESMA Annex 2 delivery rules as the regulatory target contract.

    Annex 12 is deliberately excluded — it is heavily driven by non-data-tape
    (report-level / issuer / SPV / waterfall / note / investor-report) metadata
    and is handled separately. This loader only reads the Annex 2 rules file.
    """
    path = Path(config_path) if config_path else (
        _REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    field_rules = data.get("field_rules", {}) or {}
    deferred = set(data.get("reconciliation_scope", {}).get("deferred_fields", []) or [])
    rows: List[Dict[str, Any]] = []
    for code, rule in field_rules.items():
        transform = rule.get("transform", {}) or {}
        derive = rule.get("derive", {}) or {}
        default_value = rule.get("default_value", "") if rule.get("default_allowed") else ""
        nd_allowed = list(rule.get("nd_allowed", []) or [])
        configured = ""
        if "enum_map" in transform:
            configured = "enum_map"
        elif "geography_map" in transform:
            configured = "geography_map"
        elif "boolean" in transform:
            configured = "boolean_transform"
        rows.append({
            "target_field": code,
            "target_domain": _annex2_domain(code),
            "target_label": rule.get("workbook_semantic", "") or code,
            "required_status": "mandatory" if rule.get("mandatory") else "optional",
            "applicability_status": "deferred_reconciliation" if code in deferred else "applicable",
            "match_field": rule.get("projected_source_field", ""),
            "synonyms": [s for s in (rule.get("projected_source_field", ""),
                                     rule.get("workbook_semantic", "")) if s],
            "derived": bool(derive),
            "derivation_rule": (f"{derive.get('type')}" if derive else ""),
            "default_rule": (f"default_value={default_value}" if default_value else ""),
            "default_value": default_value,
            "nd_allowed": nd_allowed,
            "configured_value_source": configured,
        })
    return "esma_annex_2", str(path), rows


# ---------------------------------------------------------------------------
# MI applicability / default overlay (asset/regime-aware, config-driven)
# ---------------------------------------------------------------------------

_MI_OVERLAY_PATH = _REPO_ROOT / "config" / "mi" / "mi_equity_release_uk_applicability.yaml"


def _matches(rule_val: Any, ctx_val: Any) -> bool:
    """An overlay scope value matches when it is absent / '*' / equal to context."""
    rv = str(rule_val or "").strip().lower()
    if rv in ("", "*", "any"):
        return True
    return rv == str(ctx_val or "").strip().lower()


def load_mi_applicability_overlay(
    mode: str,
    context: Optional[Dict[str, Any]] = None,
    overlay_path: Optional[str | Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load the MI applicability/default overlay for the run context.

    Returns ``{target_field: rule}``. The overlay only applies to MI modes and is
    scoped by asset_class / jurisdiction / mode (both file-level ``meta`` and
    per-rule overrides). Returns ``{}`` when the overlay does not apply or is
    missing — the generic registry behaviour is then used unchanged.
    """
    if target_contract_kind(mode, context) != "mi_semantics":
        return {}
    path = Path(overlay_path) if overlay_path else _MI_OVERLAY_PATH
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    meta = data.get("meta", {}) or {}
    ctx = context or {}
    asset = ctx.get("asset_class", "")
    juris = ctx.get("jurisdiction", "")
    # File-level scope gate.
    if not (_matches(meta.get("asset_class"), asset)
            and _matches(meta.get("jurisdiction"), juris)
            and _matches(meta.get("mode"), mode)):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for rule in data.get("rules", []) or []:
        field = rule.get("field")
        if not field:
            continue
        # Per-rule scope overrides (default to file-level meta when omitted).
        if not (_matches(rule.get("asset_class", meta.get("asset_class")), asset)
                and _matches(rule.get("jurisdiction", meta.get("jurisdiction")), juris)
                and _matches(rule.get("mode", meta.get("mode")), mode)):
            continue
        status = str(rule.get("coverage_status_if_no_source", "")).strip()
        if status and status not in _OVERLAY_STATUSES:
            # Unknown status -> safest is to require confirmation, never silently drop.
            status = NEEDS_CONFIRMATION
        out[field] = {**rule, "coverage_status_if_no_source": status}
    return out


def load_target_contract(
    mode: str, context: Optional[Dict[str, Any]] = None,
    mi_registry_path: Optional[str | Path] = None,
    annex2_config_path: Optional[str | Path] = None,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Load the mode-appropriate target contract."""
    if target_contract_kind(mode, context) == "mi_semantics":
        return load_mi_target_contract(mi_registry_path)
    return load_annex2_target_contract(annex2_config_path)


# ---------------------------------------------------------------------------
# Source candidate matching (per target field)
# ---------------------------------------------------------------------------

def _evidence_field_hints(ev: Dict[str, Any], resolved: Dict[str, Any]) -> set:
    """Canonical target/registry fields this source column points at."""
    hints: set = set()
    for k in ("candidate_existing_registry_fields",
              "candidate_existing_pipeline_contract_fields"):
        v = ev.get(k, "")
        if v:
            hints.add(_norm_field(v))
    for k in ("candidate_alias_matches", "candidate_semantic_alignment_matches",
              "known_client_memory_matches"):
        v = ev.get(k, "")
        if v:
            hints.add(_norm_field(str(v).split(" (")[0]))
    rt = resolved.get("resolved_target_field", "")
    if rt:
        hints.add(_norm_field(rt))
    hints.discard("")
    return hints


def _match_candidates(
    tf: Dict[str, Any],
    evidence_rows: List[Dict[str, Any]],
    resolved_by_key: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return ranked source candidates for one target field (best first)."""
    key_set = {_norm_field(tf["target_field"]), _norm_field(tf["match_field"])}
    key_set |= {_norm_field(s) for s in tf.get("synonyms", [])}
    key_set.discard("")
    cands: List[Dict[str, Any]] = []
    for ev in evidence_rows:
        col = ev.get("source_column", "")
        if _is_unnamed(col):
            continue
        rr = resolved_by_key.get(_ek(ev), {})
        field_hints = _evidence_field_hints(ev, rr)
        name_keys = {_norm_field(col), _norm_field(ev.get("normalized_column", ""))}
        name_keys.discard("")
        by_field = bool(field_hints & key_set)
        by_name = bool(name_keys & key_set)
        if not (by_field or by_name):
            continue
        base = _CANONICAL_MATCH_CONF if by_field else _NAME_MATCH_CONF
        rconf = 0.0
        if _norm_field(rr.get("resolved_target_field", "")) in key_set:
            try:
                rconf = float(rr.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                rconf = 0.0
        conf = round(max(base, rconf), 4)
        cands.append({
            "source_file": ev.get("source_file", ""),
            "source_sheet": ev.get("source_sheet", ""),
            "source_column": col,
            "confidence": conf,
            "basis": "canonical_field_match" if by_field else "name_synonym_match",
            "null_rate": ev.get("null_rate", 0),
            "data_type_guess": ev.get("data_type_guess", ""),
        })
    cands.sort(key=lambda c: (-c["confidence"], c["source_file"], c["source_column"]))
    return cands


# ---------------------------------------------------------------------------
# Coverage classification (artefact 28a)
# ---------------------------------------------------------------------------

def _classify(
    tf: Dict[str, Any],
    cands: List[Dict[str, Any]],
    overlay_rule: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Decide coverage_status + basis + decision flags for one target field.

    When there is no source candidate, an MI applicability/default ``overlay_rule``
    (if present) is consulted BEFORE the generic required/optional fallback, so a
    field that is genuinely not-applicable / configured / defaulted / derived for
    the asset class & jurisdiction is not reported as ``missing_required``.
    """
    required = tf["required_status"] in ("mandatory", "required")
    applicable = tf["applicability_status"] not in ("not_applicable",)
    nd_default = _is_nd(tf.get("default_value"))
    has_default = bool(tf.get("default_value"))
    has_derive = bool(tf.get("derived")) or bool(tf.get("derivation_rule"))
    has_config = bool(tf.get("configured_value_source"))

    requires_decision = False
    blocking = False
    decision_reason = ""
    operator_question = ""
    value_compat = ""
    # Optional overrides written back onto the coverage row by the overlay.
    overrides: Dict[str, Any] = {}

    if cands:
        primary = cands[0]
        if len(cands) > 1:
            status = SOURCE_MAPPED_ALT
            second = cands[1]
            different_files = any(c["source_file"] != primary["source_file"] for c in cands[1:])
            close = (primary["confidence"] - second["confidence"]) < _PRIORITY_CONFLICT_DELTA
            if close or different_files:
                requires_decision = True
                decision_reason = (
                    f"{len(cands)} source candidates for one target field; "
                    f"top two within {_PRIORITY_CONFLICT_DELTA} confidence"
                    if close else
                    f"{len(cands)} source candidates across multiple files")
                operator_question = (
                    f"Which source column is the authoritative source for "
                    f"'{tf['target_field']}'?")
        else:
            status = SOURCE_MAPPED
        coverage_basis = primary["basis"]
        if has_config:
            value_compat = "enum_mapping_required"
        else:
            value_compat = "compatible"
    elif overlay_rule and overlay_rule.get("coverage_status_if_no_source"):
        # Asset/regime-aware overlay: reclassify a no-source field by config rule.
        status = overlay_rule["coverage_status_if_no_source"]
        coverage_basis = "mi_applicability_overlay"
        blocking = bool(overlay_rule.get("blocking", False))
        requires_decision = blocking or bool(overlay_rule.get("requires_confirmation", False))
        decision_reason = overlay_rule.get("reason", "")
        operator_question = overlay_rule.get("operator_question", "")
        if overlay_rule.get("applicability_status"):
            overrides["applicability_status"] = overlay_rule["applicability_status"]
        if overlay_rule.get("default_rule"):
            overrides["default_rule"] = overlay_rule["default_rule"]
        if overlay_rule.get("configured_value_source"):
            overrides["configured_value_source"] = overlay_rule["configured_value_source"]
        if status == DERIVED and overlay_rule.get("default_rule"):
            overrides["derivation_rule"] = overlay_rule["default_rule"]
    elif has_derive:
        status, coverage_basis = DERIVED, "derivation_rule"
    elif nd_default:
        status, coverage_basis = DEFAULTED_ND, "nd_default_rule"
    elif has_default:
        status, coverage_basis = CONFIGURED_STATIC, "configured_static_value"
    elif has_config:
        status, coverage_basis = CONFIGURED_STATIC, "configured_transform"
    elif not applicable:
        status, coverage_basis = NOT_APPLICABLE, "not_applicable"
    elif required:
        status, coverage_basis = MISSING_REQUIRED, "no_source_no_rule"
        requires_decision = True
        blocking = True
        decision_reason = "required target field has no source, derivation, default or ND rule"
        operator_question = (
            f"'{tf['target_field']}' is required but unmapped — provide a source "
            f"column, a derivation/default, or confirm an ND code.")
    else:
        status, coverage_basis = NEEDS_CONFIRMATION, "no_source_optional"

    return {
        "coverage_status": status,
        "coverage_basis": coverage_basis,
        "requires_user_decision": requires_decision,
        "blocking": blocking,
        "decision_reason": decision_reason,
        "operator_question": operator_question,
        "value_compatibility_status": value_compat,
        "overrides": overrides,
    }


def build_target_coverage(
    mode: str,
    context: Dict[str, Any],
    target_contract_id: str,
    target_contract_source: str,
    target_fields: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    resolved_rows: List[Dict[str, Any]],
    overlay: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str], List[Tuple[str, float, bool]]]]:
    """Build the target coverage matrix (28a) — one row per TARGET field.

    Returns ``(coverage_rows, matched_targets_by_key)`` where the second value
    maps each source (file, sheet, column) to the list of
    ``(target_field, confidence, is_primary)`` it matched (used to build the
    residual register).

    ``overlay`` is the optional MI applicability/default overlay (asset/regime
    aware); it is applied only to fields with no source candidate.
    """
    overlay = overlay or {}
    resolved_by_key = {_ek(r): r for r in resolved_rows}
    matched_by_key: Dict[Tuple[str, str, str], List[Tuple[str, float, bool]]] = {}
    rows: List[Dict[str, Any]] = []
    for tf in target_fields:
        cands = _match_candidates(tf, evidence_rows, resolved_by_key)
        for i, c in enumerate(cands):
            k = (c["source_file"], c["source_sheet"], c["source_column"])
            matched_by_key.setdefault(k, []).append(
                (tf["target_field"], c["confidence"], i == 0))
        overlay_rule = overlay.get(tf["target_field"]) if not cands else None
        cls = _classify(tf, cands, overlay_rule)
        ov = cls.get("overrides", {})
        primary = cands[0] if cands else {}
        alts = cands[1:]
        nd_applied = (cls["coverage_status"] == DEFAULTED_ND
                      or (not cands and _is_nd(tf.get("default_value"))))
        rows.append({
            "mode": mode,
            "target_contract_id": target_contract_id,
            "target_contract_source": target_contract_source,
            "target_field": tf["target_field"],
            "target_domain": tf["target_domain"],
            "target_label": tf["target_label"],
            "required_status": tf["required_status"],
            "applicability_status": ov.get("applicability_status", tf["applicability_status"]),
            "coverage_status": cls["coverage_status"],
            "coverage_basis": cls["coverage_basis"],
            "selected_source_file": primary.get("source_file", ""),
            "selected_source_sheet": primary.get("source_sheet", ""),
            "selected_source_column": primary.get("source_column", ""),
            "selected_source_confidence": primary.get("confidence", ""),
            "alternative_source_candidates": "; ".join(
                f"{c['source_file']}::{c['source_sheet']}::{c['source_column']} "
                f"({c['confidence']})" for c in alts),
            "overlap_evidence": (
                f"{len(cands)} candidate source columns" if len(cands) > 1 else ""),
            "value_compatibility_status": cls["value_compatibility_status"],
            "derivation_rule": ov.get("derivation_rule", tf.get("derivation_rule", "")),
            "default_rule": ov.get("default_rule", tf.get("default_rule", "")),
            "nd_rule_applied": ("; ".join(tf.get("nd_allowed", [])) if nd_applied else ""),
            "configured_value_source": ov.get("configured_value_source",
                                              tf.get("configured_value_source", "")),
            "requires_user_decision": cls["requires_user_decision"],
            "blocking": cls["blocking"],
            "decision_reason": cls["decision_reason"],
            "operator_question": cls["operator_question"],
        })
    return rows, matched_by_key


# ---------------------------------------------------------------------------
# Source residual register (artefact 28b)
# ---------------------------------------------------------------------------

def build_source_residual_register(
    mode: str,
    evidence_rows: List[Dict[str, Any]],
    matched_by_key: Dict[Tuple[str, str, str], List[Tuple[str, float, bool]]],
) -> List[Dict[str, Any]]:
    """One row per source column NOT selected as the PRIMARY source of a target.

    Non-target residual columns are suppressed from the main approval queue.
    """
    primary_keys = {
        k for k, matches in matched_by_key.items()
        if any(is_primary for (_t, _c, is_primary) in matches)
    }
    rows: List[Dict[str, Any]] = []
    for ev in evidence_rows:
        k = _ek(ev)
        if k in primary_keys:
            continue  # this column IS a primary target source — not residual
        col = ev.get("source_column", "")
        matches = matched_by_key.get(k, [])  # only alternative matches reach here
        dup_targets = sorted({t for (t, _c, _p) in matches})
        if matches:
            residual_class = R_DUP_ALT
            dup_field = "; ".join(dup_targets)
            dup_evidence = "; ".join(f"{t} ({c})" for (t, c, _p) in matches)
            residual_reason = ("alternative/duplicate source for an already-mapped "
                               "target field")
            future_use = "reconciliation against the primary source"
            visible = True
        elif _is_unnamed(col):
            residual_class = R_HEADER
            dup_field = dup_evidence = ""
            residual_reason = "blank / Unnamed header — header detection or parse issue"
            future_use = "fix headers and re-profile if a target depends on it"
            visible = False
        elif ev.get("null_rate", 0) >= 0.999 or ev.get("distinct_count", 1) == 0:
            residual_class = R_NULL
            dup_field = dup_evidence = ""
            residual_reason = "empty / 100% null column"
            future_use = "none"
            visible = False
        elif _cashflow_like(ev):
            residual_class = R_CASHFLOW
            dup_field = dup_evidence = ""
            residual_reason = ("cashflow / ledger support column — not part of the "
                               "current target contract")
            future_use = "cashflow monitoring / ledger reconciliation"
            visible = True
        elif ev.get("data_type_guess") == "identifier" or "id" in _norm_field(col).split("_"):
            residual_class = R_RECON
            dup_field = dup_evidence = ""
            residual_reason = "identifier / audit column not required by the target contract"
            future_use = "join / reconciliation key"
            visible = True
        elif ev.get("candidate_value_profile_matches") or ev.get("data_type_guess") not in ("", "unknown"):
            residual_class = R_FUTURE
            dup_field = dup_evidence = ""
            residual_reason = ("meaningful source column with no current target field "
                               "— potential reporting extension")
            future_use = "candidate for a future target-contract extension"
            visible = True
        else:
            residual_class = R_NOT_RELEVANT
            dup_field = dup_evidence = ""
            residual_reason = "not relevant to the current mode's target contract"
            future_use = "none"
            visible = False
        rows.append({
            "source_file": ev.get("source_file", ""),
            "source_sheet": ev.get("source_sheet", ""),
            "source_column": col,
            "residual_class": residual_class,
            # Non-target residuals are suppressed from the main approval queue.
            "suppressed_from_main_queue": True,
            "duplicate_of_target_field": dup_field,
            "duplicate_or_overlap_evidence": dup_evidence,
            "residual_reason": residual_reason,
            "possible_future_use": future_use,
            "operator_visible": visible,
        })
    return rows


# ---------------------------------------------------------------------------
# Compact human decision queue (artefact 28c)
# ---------------------------------------------------------------------------

def build_human_decision_queue(
    mode: str,
    coverage_rows: List[Dict[str, Any]],
    residual_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compact, target-decision-led queue.

    Built from target coverage gaps/conflicts (28a) and *blocking* residuals
    (28b) only — never from every source column. Residual rows are not included
    unless they are genuinely blocking.
    """
    decisions: List[Dict[str, Any]] = []
    seq = 0

    def _add(decision_type, priority, target_field, source_file, source_column,
             issue, recommendation, options, blocking, operator_question,
             evidence_summary):
        nonlocal seq
        seq += 1
        decisions.append({
            "decision_id": f"DQ-{mode}-{seq:03d}",
            "decision_type": decision_type,
            "priority": priority,
            "mode": mode,
            "target_contract_id": coverage_rows[0]["target_contract_id"] if coverage_rows else "",
            "target_field": target_field,
            "source_file": source_file,
            "source_column": source_column,
            "issue": issue,
            "recommendation": recommendation,
            "options": options,
            "blocking": blocking,
            "operator_question": operator_question,
            "evidence_summary": evidence_summary,
        })

    for cov in coverage_rows:
        status = cov["coverage_status"]
        if status == MISSING_REQUIRED:
            _add(D_MISSING, "high", cov["target_field"], "", "",
                 cov["decision_reason"] or "required target field unmapped",
                 "Provide a source column, a derivation/default, or confirm an ND code.",
                 ["map_source_column", "set_derivation_or_default", "confirm_ND_code",
                  "mark_not_applicable"],
                 True, cov["operator_question"],
                 f"required={cov['required_status']}; domain={cov['target_domain']}")
        elif status == SOURCE_MAPPED_ALT and cov["requires_user_decision"]:
            dtype = (D_VALUE if cov["value_compatibility_status"] == "value_compatibility_conflict"
                     else (D_PRIORITY if "confidence" in cov["decision_reason"] else D_CONFLICT))
            _add(dtype, "medium", cov["target_field"],
                 cov["selected_source_file"], cov["selected_source_column"],
                 cov["decision_reason"] or "multiple source candidates for one target field",
                 f"Confirm '{cov['selected_source_column']}' as the authoritative source "
                 f"(or choose an alternative).",
                 ["confirm_selected", "choose_alternative", "merge_or_reconcile"],
                 False, cov["operator_question"],
                 cov["overlap_evidence"] + "; alts: " + cov["alternative_source_candidates"])
        elif cov["requires_user_decision"] and cov["blocking"]:
            _add(D_CONFIG, "high", cov["target_field"], cov["selected_source_file"],
                 cov["selected_source_column"], cov["decision_reason"],
                 "Supply the required config / static value.",
                 ["set_config_value", "confirm_ND_code", "mark_not_applicable"],
                 True, cov["operator_question"],
                 f"status={status}; basis={cov['coverage_basis']}")
        elif cov["requires_user_decision"]:
            # Non-blocking operator confirmation requested by the overlay/config.
            dtype = D_ND if status == DEFAULTED_ND else D_CONFIG
            _add(dtype, "medium", cov["target_field"], cov["selected_source_file"],
                 cov["selected_source_column"], cov["decision_reason"],
                 "Confirm the configured / default / ND value.",
                 ["confirm_value", "provide_source", "mark_not_applicable"],
                 False, cov["operator_question"],
                 f"status={status}; basis={cov['coverage_basis']}")

    # Residuals: only escalate genuinely-blocking ones (suppressed otherwise).
    for r in residual_rows:
        if r.get("suppressed_from_main_queue", True) and not r.get("blocking"):
            continue
        if r["residual_class"] == R_HEADER:
            _add(D_PARSE, "high", "", r["source_file"], r["source_column"],
                 r["residual_reason"], "Fix headers / re-parse the file.",
                 ["fix_headers_and_reload", "ignore"], True, "", r["residual_reason"])
        else:
            _add(D_EXTENSION, "low", "", r["source_file"], r["source_column"],
                 r["residual_reason"], "Decide whether to extend the target contract.",
                 ["propose_extension", "ignore"], False, "", r["residual_reason"])

    order = {"high": 0, "medium": 1, "low": 2}
    decisions.sort(key=lambda d: (order.get(d["priority"], 9), d["decision_type"],
                                  d["target_field"], d["source_column"]))
    return decisions


# ---------------------------------------------------------------------------
# Summaries + artefact writers
# ---------------------------------------------------------------------------

def coverage_summary(coverage_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    for r in coverage_rows:
        status_counts[r["coverage_status"]] = status_counts.get(r["coverage_status"], 0) + 1
    source_mapped = status_counts.get(SOURCE_MAPPED, 0) + status_counts.get(SOURCE_MAPPED_ALT, 0)
    derived_config_defaulted = (status_counts.get(DERIVED, 0)
                                + status_counts.get(CONFIGURED_STATIC, 0)
                                + status_counts.get(DEFAULTED, 0)
                                + status_counts.get(DEFAULTED_ND, 0))
    return {
        "target_fields_total": len(coverage_rows),
        "coverage_status_counts": status_counts,
        "source_mapped_fields": source_mapped,
        "derived_config_defaulted_fields": derived_config_defaulted,
        "missing_required_fields": status_counts.get(MISSING_REQUIRED, 0),
        "needs_confirmation_fields": status_counts.get(NEEDS_CONFIRMATION, 0),
        "not_applicable_fields": status_counts.get(NOT_APPLICABLE, 0),
        "optional_for_mi_fields": status_counts.get(OPTIONAL_FOR_MI, 0),
    }


def residual_summary(residual_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    class_counts: Dict[str, int] = {}
    for r in residual_rows:
        class_counts[r["residual_class"]] = class_counts.get(r["residual_class"], 0) + 1
    return {
        "residual_source_columns_total": len(residual_rows),
        "suppressed_from_main_queue": sum(1 for r in residual_rows
                                          if r.get("suppressed_from_main_queue")),
        "operator_visible": sum(1 for r in residual_rows if r.get("operator_visible")),
        "residual_class_counts": class_counts,
    }


def decision_summary(decision_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    type_counts: Dict[str, int] = {}
    for d in decision_rows:
        type_counts[d["decision_type"]] = type_counts.get(d["decision_type"], 0) + 1
    return {
        "human_decision_rows_total": len(decision_rows),
        "blocking_decisions": sum(1 for d in decision_rows if d.get("blocking")),
        "decision_type_counts": type_counts,
    }


_COVERAGE_COLUMNS = [
    "mode", "target_contract_id", "target_contract_source", "target_field",
    "target_domain", "target_label", "required_status", "applicability_status",
    "coverage_status", "coverage_basis", "selected_source_file",
    "selected_source_sheet", "selected_source_column", "selected_source_confidence",
    "alternative_source_candidates", "overlap_evidence", "value_compatibility_status",
    "derivation_rule", "default_rule", "nd_rule_applied", "configured_value_source",
    "requires_user_decision", "blocking", "decision_reason", "operator_question",
]

_RESIDUAL_COLUMNS = [
    "source_file", "source_sheet", "source_column", "residual_class",
    "suppressed_from_main_queue", "duplicate_of_target_field",
    "duplicate_or_overlap_evidence", "residual_reason", "possible_future_use",
    "operator_visible",
]

_DECISION_COLUMNS = [
    "decision_id", "decision_type", "priority", "mode", "target_contract_id",
    "target_field", "source_file", "source_column", "issue", "recommendation",
    "options", "blocking", "operator_question", "evidence_summary",
]


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for c in columns:
                v = r.get(c, "")
                if isinstance(v, (list, dict)):
                    v = "; ".join(str(x) for x in v) if isinstance(v, list) else json.dumps(v)
                out[c] = v
            w.writerow(out)


def write_artifacts(
    out_dir: str | Path,
    coverage_rows: List[Dict[str, Any]],
    residual_rows: List[Dict[str, Any]],
    decision_rows: List[Dict[str, Any]],
    target_contract_id: str,
    target_contract_source: str,
    mode: str,
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cov_sum = coverage_summary(coverage_rows)
    res_sum = residual_summary(residual_rows)
    dec_sum = decision_summary(decision_rows)

    # 28a — target coverage matrix
    _write_csv(out / "28a_target_coverage_matrix.csv", coverage_rows, _COVERAGE_COLUMNS)
    (out / "28a_target_coverage_matrix.json").write_text(
        json.dumps({"mode": mode, "target_contract_id": target_contract_id,
                    "target_contract_source": target_contract_source,
                    "summary": cov_sum, "rows": coverage_rows}, indent=2, default=str),
        encoding="utf-8")
    _write_coverage_md(out / "28a_target_coverage_summary.md", mode, target_contract_id,
                       target_contract_source, cov_sum, coverage_rows)

    # 28b — source residual register
    _write_csv(out / "28b_source_residual_register.csv", residual_rows, _RESIDUAL_COLUMNS)
    (out / "28b_source_residual_register.json").write_text(
        json.dumps({"mode": mode, "summary": res_sum, "rows": residual_rows},
                   indent=2, default=str), encoding="utf-8")
    _write_residual_md(out / "28b_source_residual_summary.md", mode, res_sum)

    # 28c — compact human decision queue
    _write_csv(out / "28c_human_decision_queue.csv", decision_rows, _DECISION_COLUMNS)
    (out / "28c_human_decision_queue.json").write_text(
        json.dumps({"mode": mode, "summary": dec_sum, "rows": decision_rows},
                   indent=2, default=str), encoding="utf-8")
    _write_decision_md(out / "28c_human_decision_summary.md", mode, dec_sum, decision_rows)

    return {
        "coverage_csv": str(out / "28a_target_coverage_matrix.csv"),
        "coverage_json": str(out / "28a_target_coverage_matrix.json"),
        "coverage_md": str(out / "28a_target_coverage_summary.md"),
        "residual_csv": str(out / "28b_source_residual_register.csv"),
        "residual_json": str(out / "28b_source_residual_register.json"),
        "residual_md": str(out / "28b_source_residual_summary.md"),
        "decision_csv": str(out / "28c_human_decision_queue.csv"),
        "decision_json": str(out / "28c_human_decision_queue.json"),
        "decision_md": str(out / "28c_human_decision_summary.md"),
    }


def _write_coverage_md(path: Path, mode, cid, csrc, summary, rows) -> None:
    md = [f"# Target coverage matrix ({mode})", "",
          f"- **Target contract:** `{cid}`",
          f"- **Source:** `{csrc}`",
          f"- **Target fields:** {summary['target_fields_total']}",
          f"- **Source-mapped:** {summary['source_mapped_fields']}",
          f"- **Derived / configured / defaulted:** {summary['derived_config_defaulted_fields']}",
          f"- **Missing required:** {summary['missing_required_fields']}", "",
          "## Coverage status counts", ""]
    for status, count in sorted(summary["coverage_status_counts"].items(),
                                key=lambda kv: -kv[1]):
        md.append(f"- `{status}`: {count}")
    md += ["", "## Missing required target fields", ""]
    missing = [r for r in rows if r["coverage_status"] == MISSING_REQUIRED]
    if missing:
        for r in missing:
            md.append(f"- `{r['target_field']}` ({r['target_domain']}) — {r['decision_reason']}")
    else:
        md.append("_None._")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def _write_residual_md(path: Path, mode, summary) -> None:
    md = [f"# Source residual register ({mode})", "",
          f"- **Residual source columns:** {summary['residual_source_columns_total']}",
          f"- **Suppressed from main queue:** {summary['suppressed_from_main_queue']}",
          f"- **Operator-visible:** {summary['operator_visible']}", "",
          "## Residual class counts", ""]
    for cls, count in sorted(summary["residual_class_counts"].items(), key=lambda kv: -kv[1]):
        md.append(f"- `{cls}`: {count}")
    md += ["", "Non-target residual columns are suppressed from the main approval "
           "queue; only genuinely-blocking residuals reach the human decision queue."]
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def _write_decision_md(path: Path, mode, summary, rows) -> None:
    md = [f"# Human decision queue ({mode})", "",
          f"- **Decisions:** {summary['human_decision_rows_total']}",
          f"- **Blocking:** {summary['blocking_decisions']}", "",
          "This compact queue is generated from TARGET coverage gaps/conflicts and "
          "blocking residuals — not from every source column.", "",
          "## Decision type counts", ""]
    for dtype, count in sorted(summary["decision_type_counts"].items(), key=lambda kv: -kv[1]):
        md.append(f"- `{dtype}`: {count}")
    md += ["", "## Decisions", ""]
    for d in rows:
        md.append(f"- **{d['decision_id']}** [{d['priority']}] `{d['decision_type']}` "
                  f"{('· ' + d['target_field']) if d['target_field'] else ''} — {d['issue']}")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_target_first_coverage(
    mode: str,
    context: Dict[str, Any],
    evidence_rows: List[Dict[str, Any]],
    resolved_rows: List[Dict[str, Any]],
    output_dir: str | Path,
    mi_registry_path: Optional[str | Path] = None,
    annex2_config_path: Optional[str | Path] = None,
    mi_overlay_path: Optional[str | Path] = None,
    client_id: str = "",
    run_id: str = "",
    decisions_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Build + write the target-first coverage artefacts (28a/28b/28c).

    When approved Gate 4 decisions are available (``decisions_path``, else an
    auto-discovered ``34_target_first_decisions.yaml`` in ``output_dir``), they
    are applied deterministically to 28a/28c BEFORE the artefacts are written,
    and a 35 application log is emitted. A fresh 34 template is (re)generated
    from the resulting decision queue for the next loop.
    """
    from . import target_first_decisions as tfd

    cid, csrc, target_fields = load_target_contract(
        mode, context, mi_registry_path=mi_registry_path,
        annex2_config_path=annex2_config_path)
    overlay = load_mi_applicability_overlay(mode, context, overlay_path=mi_overlay_path)
    coverage_rows, matched_by_key = build_target_coverage(
        mode, context, cid, csrc, target_fields, evidence_rows, resolved_rows,
        overlay=overlay)
    residual_rows = build_source_residual_register(mode, evidence_rows, matched_by_key)
    decision_rows = build_human_decision_queue(mode, coverage_rows, residual_rows)

    # --- Gate 4 decision application (deterministic, auditable) ---
    out = Path(output_dir)
    template_path = out / "34_target_first_decisions.yaml"
    decisions_file: Optional[Path] = None
    if decisions_path:
        p = Path(decisions_path)
        if p.exists():
            decisions_file = p
    if decisions_file is None and template_path.exists():
        decisions_file = template_path  # auto-discover an in-project approved file

    decision_application: Optional[Dict[str, Any]] = None
    if decisions_file is not None:
        doc = tfd.load_decisions(decisions_file)
        approved = tfd.approved_decisions(doc)
        coverage_rows, decision_rows, app_log = tfd.apply_decisions(
            coverage_rows, decision_rows, approved)
        tfd.write_application_log(app_log, out, client_id=client_id, run_id=run_id,
                                  decisions_source=str(decisions_file))
        decision_application = {
            "decisions_source": str(decisions_file),
            "log": app_log,
            "summary": tfd.application_summary(app_log),
        }

    paths = write_artifacts(output_dir, coverage_rows, residual_rows, decision_rows,
                            cid, csrc, mode)

    # (Re)generate the operator-editable 34 template from the resulting 28c —
    # unless we just applied that very file in place (never clobber approvals).
    template = tfd.build_decision_template(decision_rows, mode, client_id=client_id,
                                           run_id=run_id, target_contract_id=cid)
    if decisions_file is None or Path(decisions_file).resolve() != template_path.resolve():
        tfd.write_decision_template(template, out)

    return {
        "target_contract_id": cid,
        "target_contract_source": csrc,
        "target_contract_kind": target_contract_kind(mode, context),
        "overlay_applied": bool(overlay),
        "overlay_fields": sorted(overlay.keys()),
        "coverage": coverage_rows,
        "residual": residual_rows,
        "decision_queue": decision_rows,
        "coverage_summary": coverage_summary(coverage_rows),
        "residual_summary": residual_summary(residual_rows),
        "decision_summary": decision_summary(decision_rows),
        "decision_application": decision_application,
        "decision_template_path": str(template_path),
        "paths": paths,
    }
