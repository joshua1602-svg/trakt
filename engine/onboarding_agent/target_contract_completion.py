"""
target_contract_completion.py
=============================

Target Contract Completion Checklist / **Target Field Disposition** layer.

This is the architectural keystone of the agentic pipeline:

    Onboarding   decides the expected *disposition* for each target field.
    Transformation materialises source / default / context values.
    Validation   validates values and whether dispositions are blocking.
    Projection   constructs the target frame using approved dispositions.
    Delivery/XML applies XSD/XML structure.

For a chosen target contract (e.g. ``ESMA_Annex2``) this module produces **one
disposition row per target field / ESMA code**, answering — up front, at
onboarding — whether the field has source data, an asset default, a client/lender
policy default, a valid ND treatment, a derivation rule, a formal client
onboarding requirement, an operator review decision, a projection-only rule, or a
true unresolved gap.

Design rules (enforced here):

  * **One disposition per field** from a controlled vocabulary, plus secondary
    boolean flags.
  * **"ND allowed" ≠ "ND selected"** — a field is only *completed* by ND when an
    asset/client/config policy has actually *selected* a valid ND, never merely
    because the regulator permits one.
  * **Asset-specific, never generic** — ERM behaviour (DTI not captured, lifetime
    mortgage maturity = ND5) comes from the asset config / registry applicability
    layer, not from hard-coded code paths here.
  * **Pure / IO-free** core (``build_completion_checklist`` /
    ``build_review_bench``) so it is unit-testable; a thin writer emits the
    ``29_*`` and ``29a_*`` artefacts.

This module does NOT generate XML, build UI, or mutate upstream artefacts.
"""

from __future__ import annotations

import csv
import json
import math
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Controlled disposition vocabulary
# --------------------------------------------------------------------------- #

D_SOURCE_SUPPLIED = "source_supplied"
D_SOURCE_MAPPED_REVIEW = "source_mapped_with_review"
D_ASSET_DEFAULT = "asset_default_supplied"
D_CLIENT_POLICY_DEFAULT = "client_policy_default_supplied"
D_CONFIGURED_STATIC = "configured_static_supplied"
D_ND_POLICY_SELECTED = "nd_policy_selected"
D_DERIVATION = "derivation_configured"
D_CALCULATION = "calculation_configured"
D_PROJECTION_RULE_REQUIRED = "projection_rule_required"
D_CLIENT_ONBOARDING_REQUIRED = "client_onboarding_required"
D_OPERATOR_REVIEW_REQUIRED = "operator_review_required"
D_CONFIG_MAPPING_REQUIRED = "config_mapping_required"
D_NOT_APPLICABLE = "not_applicable"
D_UNRESOLVED_GAP = "unresolved_gap"

DISPOSITIONS = (
    D_SOURCE_SUPPLIED, D_SOURCE_MAPPED_REVIEW, D_ASSET_DEFAULT,
    D_CLIENT_POLICY_DEFAULT, D_CONFIGURED_STATIC, D_ND_POLICY_SELECTED,
    D_DERIVATION, D_CALCULATION, D_PROJECTION_RULE_REQUIRED,
    D_CLIENT_ONBOARDING_REQUIRED, D_OPERATOR_REVIEW_REQUIRED,
    D_CONFIG_MAPPING_REQUIRED, D_NOT_APPLICABLE, D_UNRESOLVED_GAP,
)

# A disposition is "completed" (does not block downstream) if it is one of these.
_COMPLETED = {
    D_SOURCE_SUPPLIED, D_ASSET_DEFAULT, D_CLIENT_POLICY_DEFAULT,
    D_CONFIGURED_STATIC, D_ND_POLICY_SELECTED, D_DERIVATION, D_CALCULATION,
    D_NOT_APPLICABLE,
}

# Review-bench categories.
RB_OPERATOR = "operator_decision"
RB_CLIENT_INPUT = "client_required_input"
RB_CONFIG = "config_required"
RB_ASSET_POLICY = "asset_policy_required"
RB_DERIVATION = "derivation_required"
RB_PROJECTION_RULE = "projection_rule_required"

# Owners.
OWN_TRANSFORMATION = "transformation_validation"
OWN_PROJECTION = "projection"
OWN_OPERATOR = "operator"
OWN_CONFIG = "config_policy"
OWN_CLIENT = "client_onboarding"

CHECKLIST_COLUMNS = [
    "target_contract_id", "esma_code", "target_field", "canonical_field",
    "record_group", "business_label", "required_status", "data_type",
    "allowed_values", "nd_allowed", "default_allowed", "asset_default_value",
    "client_policy_default_value", "configured_default_value", "source_status",
    "selected_source_file", "selected_source_column", "selected_source_confidence",
    "derivation_rule", "calculation_rule", "applicability_status",
    "field_disposition", "disposition_source", "disposition_confidence",
    "requires_client_input", "requires_operator_review", "requires_config",
    "requires_projection_rule", "requires_derivation",
    "blocking_for_onboarding_handoff", "blocking_for_transformation",
    "blocking_for_validation", "blocking_for_projection",
    "recommended_action", "owner", "notes",
]

REVIEW_BENCH_COLUMNS = [
    "review_id", "esma_code", "target_field", "canonical_field", "record_group",
    "review_category", "field_disposition", "required_status", "blocking",
    "recommended_action", "owner", "notes",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        if isinstance(v, float) and math.isnan(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    if s.lower() in ("nan", "<na>", "none"):
        return ""
    return s


def _is_nd(v: Any) -> bool:
    s = _to_str(v).upper()
    return len(s) in (3, 4) and s.startswith("ND") and s[2:].isdigit()


def _norm(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "")).strip("_")


def _record_group(esma_code: str) -> str:
    c = _to_str(esma_code).upper()
    if c.startswith("RREL"):
        return "RREL"
    if c.startswith("RREC"):
        return "RREC"
    return "other"


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return _to_str(v).lower() in ("true", "1", "yes", "y")


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [_to_str(x).upper() for x in v if _to_str(x)]
    s = _to_str(v)
    if not s:
        return []
    return [p.strip().upper() for p in s.replace(",", ";").split(";") if p.strip()]


# --------------------------------------------------------------------------- #
# Config indexing
# --------------------------------------------------------------------------- #

def build_registry_index(registry_fields: Dict[str, Any], contract_id: str) -> Dict[str, Dict[str, Any]]:
    """Map ESMA code -> registry metadata for the chosen target contract.

    Reads the per-field ``regime_mapping.<contract>.code`` plus ``format`` /
    ``allowed_values`` and the **asset-specific** ``applicability`` block.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for canonical, meta in (registry_fields or {}).items():
        meta = meta or {}
        rm = (meta.get("regime_mapping") or {}).get(contract_id) or {}
        code = _to_str(rm.get("code"))
        if not code:
            continue
        out[code] = {
            "canonical_field": canonical,
            "priority": _to_str(rm.get("priority")),
            "data_type": _to_str(meta.get("format")),
            "allowed_values": meta.get("allowed_values"),
            "applicability": meta.get("applicability") or {},
        }
    return out


def _policy_block(cfg: Optional[dict]) -> Dict[str, Any]:
    """Extract the optional ``reporting_policy`` layer from an asset/client config."""
    if not isinstance(cfg, dict):
        return {}
    return cfg.get("reporting_policy") or {}


def build_regime_index(regime_cfg: Optional[dict]) -> Dict[str, Dict[str, Any]]:
    """Index the regime ``field_rules`` by ESMA code (projection envelope)."""
    out: Dict[str, Dict[str, Any]] = {}
    cfg = regime_cfg or {}
    deferred = set((cfg.get("reconciliation_scope", {}) or {}).get("deferred_fields", []) or [])
    for code, rule in (cfg.get("field_rules", {}) or {}).items():
        rule = rule or {}
        transform = rule.get("transform") if isinstance(rule.get("transform"), dict) else {}
        out[str(code)] = {
            "canonical_field": rule.get("projected_source_field", "") or "",
            "nd_allowed": [str(x).upper() for x in (rule.get("nd_allowed") or [])],
            "default_allowed": bool(rule.get("default_allowed", False)),
            "default_value": "" if rule.get("default_value") is None else str(rule.get("default_value")),
            "enum_map": transform.get("enum_map") if isinstance(transform.get("enum_map"), dict) else {},
            "derive": rule.get("derive"),
            "mandatory": bool(rule.get("mandatory", False)),
            "enforce_presence": bool(rule.get("enforce_presence", False)),
            "deferred": str(code) in deferred,
        }
    return out


def load_target_contract_configs(
    *,
    regime_config_path: str = "",
    field_universe_path: str = "",
    registry_path: str = "",
    asset_config_path: str = "",
    client_policy_path: str = "",
) -> Dict[str, Any]:
    """Load (and index) the configs the completion checklist needs.

    Returns ``{field_universe, registry_fields, regime_index, asset_cfg,
    client_policy}``. Missing files yield empty structures (never raises).
    """
    import yaml
    from pathlib import Path

    def _y(path: str) -> Dict[str, Any]:
        if not path:
            return {}
        try:
            return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    regime_cfg = _y(regime_config_path)
    registry = _y(registry_path)
    return {
        "field_universe": _y(field_universe_path),
        "registry_fields": registry.get("fields", {}) or {},
        "regime_index": build_regime_index(regime_cfg),
        "asset_cfg": _y(asset_config_path),
        "client_policy": _y(client_policy_path),
    }


# --------------------------------------------------------------------------- #
# ND / default selection (ND allowed vs ND selected)
# --------------------------------------------------------------------------- #

def _select_default(
    canonical: str,
    nd_allowed: List[str],
    *,
    client_pol: Dict[str, Any],
    asset_defaults: Dict[str, Any],
    asset_nd_defaults: Dict[str, Any],
    asset_policy: Dict[str, Any],
    registry_applicability: Dict[str, Any],
    asset_class: str,
    regime_default_value: str,
) -> Optional[Tuple[str, str, str, str]]:
    """Resolve whether a *policy* has actually selected an ND/default value.

    Returns ``(disposition, value, disposition_source, note)`` or ``None`` if no
    layer selected a value. ``nd_allowed`` being non-empty does NOT count — only
    an explicit selection by a config/policy layer does.

    Layer precedence: client policy → asset policy/defaults → registry asset
    applicability → regime configured default.
    """
    nd_allowed = [x.upper() for x in (nd_allowed or [])]
    conflicts: List[str] = []

    def _consider(value: str, source: str, *, is_policy_nd: bool) -> Optional[Tuple[str, str, str, str]]:
        val = _to_str(value)
        if not val:
            return None
        if _is_nd(val):
            if not nd_allowed or val.upper() in nd_allowed:
                return (D_ND_POLICY_SELECTED, val.upper(), source,
                        f"ND policy {val.upper()} selected by {source}")
            conflicts.append(f"{source}:{val.upper()} not in nd_allowed {nd_allowed}")
            return None
        # non-ND configured value.
        disp = {
            "client_policy": D_CLIENT_POLICY_DEFAULT,
            "asset_config": D_ASSET_DEFAULT,
        }.get(source, D_CONFIGURED_STATIC)
        return (disp, val, source, f"default {val} selected by {source}")

    # 1) client policy layer (reporting_policy.nd_policy / defaults).
    cp_nd = (client_pol.get("nd_policy") or {})
    cp_def = (client_pol.get("defaults") or {})
    for table, src in ((cp_nd, "client_policy"), (cp_def, "client_policy")):
        if canonical in table:
            r = _consider(table[canonical], src, is_policy_nd=True)
            if r:
                return r

    # 2) asset config (explicit defaults / nd_defaults) + asset reporting_policy.
    ap_nd = (asset_policy.get("nd_policy") or {})
    if canonical in ap_nd:
        r = _consider(ap_nd[canonical], "asset_config", is_policy_nd=True)
        if r:
            return r
    if canonical in (asset_nd_defaults or {}):
        r = _consider(asset_nd_defaults[canonical], "asset_config", is_policy_nd=True)
        if r:
            return r
    if canonical in (asset_defaults or {}):
        r = _consider(asset_defaults[canonical], "asset_config", is_policy_nd=False)
        if r:
            return r

    # 3) registry asset-specific applicability (e.g. maturity_date.equity_release.nd_default).
    appl = (registry_applicability or {}).get(asset_class) or {}
    if appl.get("nd_default"):
        r = _consider(appl["nd_default"], "asset_applicability", is_policy_nd=True)
        if r:
            note = r[3]
            if appl.get("reason"):
                note = f"{note} — {appl['reason']}"
            return (r[0], r[1], r[2], note)

    # 4) regime configured default (a config-level selection, generic envelope).
    if regime_default_value:
        r = _consider(regime_default_value, "regime_config", is_policy_nd=True)
        if r:
            note = r[3]
            if conflicts:
                note = f"{note}; overrides {', '.join(conflicts)}"
            return (r[0], r[1], r[2], note)

    if conflicts:
        return (D_CONFIG_MAPPING_REQUIRED, "", "config_conflict",
                "selected ND/default outside regime envelope: " + "; ".join(conflicts))
    return None


# --------------------------------------------------------------------------- #
# Per-field disposition
# --------------------------------------------------------------------------- #

# Source coverage statuses (mirrors target_coverage vocabulary, defensively).
_SOURCE_MAPPED = {"source_mapped", "mapped", "source_linked"}
_SOURCE_MAPPED_ALT = {"source_mapped_alt", "source_mapped_alternative"}
_DERIVED = {"derived"}
_PENDING_RULE = {"pending_regime_rule", "deferred", "projection_required"}
_ABSENT = {"source_absent", "missing", "unmapped", "missing_required",
           "missing_not_required", ""}


def decide_disposition(
    *,
    canonical: str,
    esma_code: str,
    nd_allowed: List[str],
    default_allowed: bool,
    regime_default_value: str,
    has_enum_map: bool,
    enum_complete: Optional[bool],
    is_enum_field: bool,
    has_derivation: bool,
    is_deferred: bool,
    coverage_status: str,
    requires_user_decision: bool,
    blocking_decision: bool,
    selected_source_confidence: float,
    applicability_status: str,
    mandatory: bool,
    formal_client_field: bool,
    not_applicable_field: bool,
    client_pol: Dict[str, Any],
    asset_defaults: Dict[str, Any],
    asset_nd_defaults: Dict[str, Any],
    asset_policy: Dict[str, Any],
    registry_applicability: Dict[str, Any],
    asset_class: str,
) -> Dict[str, Any]:
    """Return the single disposition + secondary flags for one target field.

    Pure and deterministic. See module docstring for the design rules.
    """
    cov = _to_str(coverage_status).lower()
    appl = _to_str(applicability_status).lower()

    def result(disposition: str, *, source: str, owner: str, action: str,
               confidence: float = 0.0, note: str = "", value: str = "",
               req_client=False, req_op=False, req_cfg=False,
               req_proj=False, req_deriv=False,
               b_tx=False, b_val=False, b_proj=False) -> Dict[str, Any]:
        return {
            "field_disposition": disposition,
            "disposition_source": source,
            "disposition_confidence": round(float(confidence), 4),
            "selected_value": value,
            "requires_client_input": req_client,
            "requires_operator_review": req_op,
            "requires_config": req_cfg,
            "requires_projection_rule": req_proj,
            "requires_derivation": req_deriv,
            "blocking_for_onboarding_handoff": False,
            "blocking_for_transformation": b_tx,
            "blocking_for_validation": b_val,
            "blocking_for_projection": b_proj,
            "recommended_action": action,
            "owner": owner,
            "notes": note,
        }

    # 0) Not applicable for this asset class / contract.
    if not_applicable_field or appl in ("not_applicable", "na", "not_required_na"):
        return result(D_NOT_APPLICABLE, source="applicability_config",
                      owner=OWN_TRANSFORMATION, confidence=1.0,
                      action="exclude from delivery (not applicable for this asset)",
                      note="field marked not applicable for this asset class")

    source_present = cov in _SOURCE_MAPPED or (
        cov not in _ABSENT and cov not in _PENDING_RULE
        and selected_source_confidence > 0)

    # 1) Formal client-onboarding identifiers that are not confidently sourced.
    if formal_client_field and not (
            cov in _SOURCE_MAPPED and selected_source_confidence >= 0.9
            and not requires_user_decision):
        return result(D_CLIENT_ONBOARDING_REQUIRED, source="formal_onboarding_policy",
                      owner=OWN_CLIENT, confidence=0.0, req_client=True,
                      b_val=True, b_proj=True,
                      action="request formal regulatory exposure identifiers from client",
                      note="formal onboarding field not present / not inferable from ordinary loan IDs")

    # 2) Confidently source-mapped.
    if cov in _SOURCE_MAPPED:
        if is_enum_field and enum_complete is False:
            return result(D_CONFIG_MAPPING_REQUIRED, source="enum_mapping",
                          owner=OWN_CONFIG, confidence=selected_source_confidence,
                          req_cfg=True, b_proj=True,
                          action="complete the enum/code mapping for this field",
                          note="source values present but enum mapping incomplete")
        if requires_user_decision:
            return result(D_SOURCE_MAPPED_REVIEW, source="source_match",
                          owner=OWN_OPERATOR, confidence=selected_source_confidence,
                          req_op=True,
                          action="confirm the selected source mapping",
                          note="source mapped but flagged for operator confirmation")
        return result(D_SOURCE_SUPPLIED, source="source_match",
                      owner=OWN_TRANSFORMATION, confidence=selected_source_confidence,
                      action="materialise the mapped source value",
                      note="confident source mapping")

    # 3) Multiple plausible sources / mapping ambiguity.
    if cov in _SOURCE_MAPPED_ALT or requires_user_decision:
        return result(D_OPERATOR_REVIEW_REQUIRED, source="source_ambiguity",
                      owner=OWN_OPERATOR, confidence=selected_source_confidence,
                      req_op=True, b_proj=bool(mandatory),
                      action="operator to choose between candidate source columns",
                      note="multiple plausible source columns / ambiguous mapping")

    # 4) Derivation configured.
    if has_derivation or cov in _DERIVED:
        return result(D_DERIVATION, source="derivation_rule",
                      owner=OWN_TRANSFORMATION, confidence=0.9, req_deriv=True,
                      action="apply the configured derivation rule",
                      note="derivation rule configured for this field")

    # 5) Source absent → resolve a configured ND/default policy.
    sel = _select_default(
        canonical, nd_allowed, client_pol=client_pol, asset_defaults=asset_defaults,
        asset_nd_defaults=asset_nd_defaults, asset_policy=asset_policy,
        registry_applicability=registry_applicability, asset_class=asset_class,
        regime_default_value=regime_default_value)
    if sel is not None:
        disposition, value, source, note = sel
        if disposition == D_CONFIG_MAPPING_REQUIRED:
            return result(D_CONFIG_MAPPING_REQUIRED, source=source, owner=OWN_CONFIG,
                          req_cfg=True, b_proj=True,
                          action="align the selected ND/default with the regime envelope",
                          note=note)
        owner = OWN_TRANSFORMATION
        return result(disposition, source=source, owner=owner, confidence=1.0,
                      value=value,
                      action=f"materialise configured value '{value}'", note=note)

    # 6) Projection-only rule still pending.
    if is_deferred or cov in _PENDING_RULE:
        return result(D_PROJECTION_RULE_REQUIRED, source="regime_rule",
                      owner=OWN_PROJECTION, confidence=0.0, req_proj=True,
                      b_proj=True,
                      action="implement or defer the projection rule for this field",
                      note="no source/default; deferred to a projection rule")

    # 7) ND allowed but NO policy selected, or a genuine gap.
    if nd_allowed:
        return result(D_UNRESOLVED_GAP, source="policy_gap", owner=OWN_CONFIG,
                      req_cfg=True, b_val=bool(mandatory), b_proj=bool(mandatory),
                      action="configure/client-confirm an ND/default reporting policy",
                      note="ND is permitted by the regulator but no policy has selected one")

    # 8) Mandatory with nothing at all → hard gap; optional → low-severity gap.
    return result(D_UNRESOLVED_GAP, source="gap",
                  owner=OWN_OPERATOR if mandatory else OWN_CONFIG,
                  req_op=bool(mandatory), req_cfg=not mandatory,
                  b_val=bool(mandatory), b_proj=bool(mandatory),
                  action="supply a source value, derivation or policy for this field",
                  note="no source, no default/ND policy, no derivation")


# --------------------------------------------------------------------------- #
# Checklist builder
# --------------------------------------------------------------------------- #

def build_completion_checklist(
    *,
    contract_id: str,
    field_universe: Dict[str, Any],
    registry_fields: Dict[str, Any],
    regime_index: Dict[str, Dict[str, Any]],
    asset_cfg: Dict[str, Any],
    asset_class: str,
    coverage_by_code: Optional[Dict[str, Dict[str, Any]]] = None,
    client_policy: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Produce one disposition row per target field / ESMA code.

    The spine is the authoritative **field universe** (every Annex 2 code), so
    the checklist is complete. Each code is enriched with registry metadata
    (canonical + asset applicability), the regime projection envelope, the asset
    config + policy layers, and any source/mapping signal from the 28a coverage
    matrix (``coverage_by_code``).
    """
    coverage_by_code = coverage_by_code or {}
    client_policy = client_policy or {}
    asset_defaults = (asset_cfg or {}).get("defaults") or {}
    asset_nd_defaults = (asset_cfg or {}).get("nd_defaults") or {}
    asset_policy = _policy_block(asset_cfg)
    client_pol = _policy_block(client_policy)

    registry_index = build_registry_index(registry_fields, contract_id)

    formal_set = {_norm(x) for x in (
        list(asset_policy.get("formal_client_onboarding_required", []) or [])
        + list(client_pol.get("formal_client_onboarding_required", []) or []))}
    not_applicable_set = {_norm(x) for x in (
        list(asset_policy.get("not_applicable", []) or [])
        + list(client_pol.get("not_applicable", []) or []))}

    universe_fields = (field_universe or {}).get("fields") or {}
    # spine = union of universe codes and any regime/registry codes (defensive).
    codes = list(universe_fields.keys())
    for code in list(regime_index.keys()) + list(registry_index.keys()):
        if code not in universe_fields:
            codes.append(code)

    rows: List[Dict[str, Any]] = []
    for code in codes:
        wb = universe_fields.get(code, {}) or {}
        reg = registry_index.get(code, {})
        rule = regime_index.get(code, {})
        cov = coverage_by_code.get(code, {})

        canonical = (rule.get("canonical_field") or reg.get("canonical_field")
                     or _to_str(cov.get("canonical_field")))
        canon_norm = _norm(canonical)

        # ND envelope: prefer the regime projection rule, else the workbook universe.
        nd_allowed = rule.get("nd_allowed") or _as_list(cov.get("nd_allowed"))
        if not nd_allowed:
            nd_allowed = _nd_from_universe(wb)
        default_allowed = bool(rule.get("default_allowed")) or _truthy(cov.get("regime_default_allowed"))
        regime_default_value = _to_str(rule.get("default_value")) or _to_str(cov.get("regime_default_value"))

        is_enum_field = bool(rule.get("enum_map")) or bool(reg.get("allowed_values")) or bool(wb.get("allowed_values"))
        has_enum_map = bool(rule.get("enum_map")) or _truthy(cov.get("has_enum_map"))
        enum_status = _to_str(cov.get("enum_coverage_status")).lower()
        enum_complete: Optional[bool]
        if enum_status:
            enum_complete = enum_status in ("complete", "covered", "ok", "full")
        else:
            enum_complete = None
        has_derivation = bool(rule.get("derive")) or bool(_to_str(cov.get("derivation_rule")))
        is_deferred = bool(rule.get("deferred"))

        priority = reg.get("priority", "")
        mandatory = (str(priority).lower() == "mandatory"
                     or bool(rule.get("mandatory")) or bool(rule.get("enforce_presence"))
                     or _to_str(cov.get("required_status")).lower() == "mandatory")
        required_status = priority or ("mandatory" if mandatory else "optional")

        try:
            sel_conf = float(cov.get("selected_source_confidence") or 0.0)
        except (TypeError, ValueError):
            sel_conf = 0.0

        appl_status = _to_str(cov.get("applicability_status"))
        formal = canon_norm in formal_set
        not_applicable = canon_norm in not_applicable_set

        disp = decide_disposition(
            canonical=canonical, esma_code=code, nd_allowed=nd_allowed,
            default_allowed=default_allowed, regime_default_value=regime_default_value,
            has_enum_map=has_enum_map, enum_complete=enum_complete,
            is_enum_field=is_enum_field, has_derivation=has_derivation,
            is_deferred=is_deferred, coverage_status=_to_str(cov.get("coverage_status")),
            requires_user_decision=_truthy(cov.get("requires_user_decision")),
            blocking_decision=_truthy(cov.get("blocking")),
            selected_source_confidence=sel_conf, applicability_status=appl_status,
            mandatory=mandatory, formal_client_field=formal,
            not_applicable_field=not_applicable, client_pol=client_pol,
            asset_defaults=asset_defaults, asset_nd_defaults=asset_nd_defaults,
            asset_policy=asset_policy,
            registry_applicability=reg.get("applicability") or {},
            asset_class=asset_class)

        # resolve the concrete configured default value (if any) for transparency.
        configured_default = ""
        if disp["field_disposition"] in (
                D_ND_POLICY_SELECTED, D_ASSET_DEFAULT, D_CLIENT_POLICY_DEFAULT,
                D_CONFIGURED_STATIC):
            configured_default = _to_str(disp.get("selected_value")) or (
                _to_str(asset_nd_defaults.get(canonical))
                or _to_str(asset_defaults.get(canonical)) or regime_default_value)

        rows.append({
            "target_contract_id": contract_id,
            "esma_code": code,
            "target_field": code,
            "canonical_field": canonical,
            "record_group": _record_group(code),
            "business_label": _to_str(wb.get("field_name")) or _to_str(cov.get("business_label")),
            "required_status": required_status,
            "data_type": _to_str(reg.get("data_type")) or _to_str(wb.get("format"))
                         or _to_str(cov.get("data_type_guess")),
            "allowed_values": _allowed_values_str(reg.get("allowed_values"), wb.get("allowed_values")),
            "nd_allowed": ";".join(nd_allowed),
            "default_allowed": default_allowed,
            "asset_default_value": _to_str(asset_nd_defaults.get(canonical))
                                   or _to_str(asset_defaults.get(canonical)),
            "client_policy_default_value": _to_str((client_pol.get("nd_policy") or {}).get(canonical))
                                           or _to_str((client_pol.get("defaults") or {}).get(canonical)),
            "configured_default_value": configured_default,
            "source_status": _to_str(cov.get("coverage_status")) or "unknown",
            "selected_source_file": _to_str(cov.get("selected_source_file")),
            "selected_source_column": _to_str(cov.get("selected_source_column")),
            "selected_source_confidence": sel_conf,
            "derivation_rule": _to_str(cov.get("derivation_rule"))
                               or ("configured" if rule.get("derive") else ""),
            "calculation_rule": "",
            "applicability_status": appl_status,
            **disp,
        })
    return rows


def _nd_from_universe(wb: Dict[str, Any]) -> List[str]:
    nd: List[str] = []
    if _truthy(wb.get("nd1_4_allowed")):
        nd += ["ND1", "ND2", "ND3", "ND4"]
    if _truthy(wb.get("nd5_allowed")):
        nd += ["ND5"]
    return nd


def _allowed_values_str(reg_av: Any, wb_av: Any) -> str:
    for av in (reg_av, wb_av):
        if av in (None, "", "null"):
            continue
        if isinstance(av, (list, tuple)):
            return ";".join(_to_str(x) for x in av)
        return _to_str(av)
    return ""


# --------------------------------------------------------------------------- #
# Review bench
# --------------------------------------------------------------------------- #

_DISPOSITION_TO_REVIEW = {
    D_OPERATOR_REVIEW_REQUIRED: RB_OPERATOR,
    D_SOURCE_MAPPED_REVIEW: RB_OPERATOR,
    D_CLIENT_ONBOARDING_REQUIRED: RB_CLIENT_INPUT,
    D_CONFIG_MAPPING_REQUIRED: RB_CONFIG,
    D_PROJECTION_RULE_REQUIRED: RB_PROJECTION_RULE,
}


def build_review_bench(checklist_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn every unresolved/review checklist row into a review-bench item.

    Distinguishes: human mapping ambiguity, client onboarding missing field,
    asset policy missing, enum/config mapping missing and projection rule missing.
    """
    bench: List[Dict[str, Any]] = []
    n = 0
    for r in checklist_rows:
        disp = r["field_disposition"]
        if disp in _COMPLETED:
            continue
        if disp == D_UNRESOLVED_GAP:
            category = RB_ASSET_POLICY if r.get("requires_config") else RB_OPERATOR
        else:
            category = _DISPOSITION_TO_REVIEW.get(disp)
        if category is None:
            continue
        n += 1
        bench.append({
            "review_id": f"RB-{n:04d}",
            "esma_code": r["esma_code"],
            "target_field": r["target_field"],
            "canonical_field": r["canonical_field"],
            "record_group": r["record_group"],
            "review_category": category,
            "field_disposition": disp,
            "required_status": r["required_status"],
            "blocking": bool(r["blocking_for_validation"] or r["blocking_for_projection"]),
            "recommended_action": r["recommended_action"],
            "owner": r["owner"],
            "notes": r["notes"],
        })
    return bench


# --------------------------------------------------------------------------- #
# Counts + artefact writers
# --------------------------------------------------------------------------- #

def disposition_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {d: 0 for d in DISPOSITIONS}
    for r in rows:
        d = r["field_disposition"]
        counts[d] = counts.get(d, 0) + 1
    return counts


def checklist_summary(rows: List[Dict[str, Any]], bench: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = disposition_counts(rows)
    completed = sum(1 for r in rows if r["field_disposition"] in _COMPLETED)
    return {
        "target_field_count": len(rows),
        "completed_disposition_count": completed,
        "unresolved_disposition_count": len(rows) - completed,
        "review_bench_item_count": len(bench),
        "requires_client_input_count": sum(1 for r in rows if r["requires_client_input"]),
        "requires_operator_review_count": sum(1 for r in rows if r["requires_operator_review"]),
        "requires_config_count": sum(1 for r in rows if r["requires_config"]),
        "requires_projection_rule_count": sum(1 for r in rows if r["requires_projection_rule"]),
        "blocking_for_validation_count": sum(1 for r in rows if r["blocking_for_validation"]),
        "blocking_for_projection_count": sum(1 for r in rows if r["blocking_for_projection"]),
        "disposition_counts": counts,
    }


def write_checklist_artefacts(
    out_dir,
    rows: List[Dict[str, Any]],
    bench: List[Dict[str, Any]],
    *,
    contract_id: str = "",
    client_id: str = "",
    run_id: str = "",
) -> Dict[str, Any]:
    """Write 29_* checklist (csv/json/md) and 29a_* review bench (csv)."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = checklist_summary(rows, bench)

    csv_path = out_dir / "29_target_contract_completion_checklist.csv"
    json_path = out_dir / "29_target_contract_completion_checklist.json"
    md_path = out_dir / "29_target_contract_completion_checklist.md"
    bench_csv = out_dir / "29a_target_contract_review_bench.csv"
    bench_json = out_dir / "29a_target_contract_review_bench.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CHECKLIST_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CHECKLIST_COLUMNS})

    json_path.write_text(json.dumps({
        "target_contract_id": contract_id, "client_id": client_id, "run_id": run_id,
        "summary": summary, "rows": rows,
    }, indent=2, default=str), encoding="utf-8")

    with open(bench_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_BENCH_COLUMNS)
        w.writeheader()
        for r in bench:
            w.writerow({k: r.get(k, "") for k in REVIEW_BENCH_COLUMNS})
    bench_json.write_text(json.dumps({
        "target_contract_id": contract_id, "review_item_count": len(bench),
        "rows": bench,
    }, indent=2, default=str), encoding="utf-8")

    md_path.write_text(_checklist_md(rows, bench, summary, contract_id, client_id, run_id),
                       encoding="utf-8")

    return {
        "checklist_csv_path": str(csv_path),
        "checklist_json_path": str(json_path),
        "checklist_md_path": str(md_path),
        "review_bench_csv_path": str(bench_csv),
        "review_bench_json_path": str(bench_json),
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# Disposition execution (consumed by downstream agents)
# --------------------------------------------------------------------------- #
#
# Onboarding *decides* the disposition; the downstream agents *execute* it.
# These pure mappers let Transformation / Validation / Projection translate an
# onboarding disposition into their own action / classification vocabulary,
# instead of rediscovering the field's treatment from scratch.

def transformation_action_for_disposition(disposition: str) -> str:
    """How the Transformation Agent should execute an onboarding disposition."""
    return {
        D_SOURCE_SUPPLIED: "materialise_mapped_source_value",
        D_SOURCE_MAPPED_REVIEW: "materialise_mapped_source_value",
        D_ASSET_DEFAULT: "materialise_asset_default",
        D_CLIENT_POLICY_DEFAULT: "materialise_client_policy_default",
        D_CONFIGURED_STATIC: "materialise_configured_static",
        D_ND_POLICY_SELECTED: "materialise_selected_nd",
        D_DERIVATION: "apply_derivation_rule",
        D_CALCULATION: "apply_calculation_rule",
        D_PROJECTION_RULE_REQUIRED: "carry_forward_for_projection",
        D_CLIENT_ONBOARDING_REQUIRED: "carry_forward_client_input_required",
        D_OPERATOR_REVIEW_REQUIRED: "carry_forward_operator_decision",
        D_CONFIG_MAPPING_REQUIRED: "carry_forward_config_required",
        D_NOT_APPLICABLE: "skip_not_applicable",
        D_UNRESOLVED_GAP: "carry_forward_unresolved_gap",
    }.get(disposition, "carry_forward_unresolved_gap")


def validation_classification_for_disposition(disposition: str) -> str:
    """How the Validation Agent should classify an onboarding disposition.

    Crucially, a ``config_mapping_required`` disposition is a *config* blocker —
    never a data failure — and ``client_onboarding_required`` is owned by the
    client, not surfaced as a generic validation failure.
    """
    return {
        D_SOURCE_SUPPLIED: "validation_pass",
        D_SOURCE_MAPPED_REVIEW: "validation_warning",
        D_ASSET_DEFAULT: "validation_pass",
        D_CLIENT_POLICY_DEFAULT: "validation_pass",
        D_CONFIGURED_STATIC: "validation_pass",
        D_ND_POLICY_SELECTED: "validation_pass",
        D_DERIVATION: "validation_pass",
        D_CALCULATION: "validation_pass",
        D_PROJECTION_RULE_REQUIRED: "projection_required",
        D_CLIENT_ONBOARDING_REQUIRED: "client_onboarding_required",
        D_OPERATOR_REVIEW_REQUIRED: "operator_required",
        D_CONFIG_MAPPING_REQUIRED: "config_required",
        D_NOT_APPLICABLE: "acceptable_downstream_gap",
        D_UNRESOLVED_GAP: "config_required",
    }.get(disposition, "config_required")


def projection_status_for_disposition(disposition: str) -> str:
    """How the Projection Agent should treat an onboarding disposition."""
    return {
        D_SOURCE_SUPPLIED: "projected_from_transformed",
        D_SOURCE_MAPPED_REVIEW: "projected_from_transformed",
        D_ASSET_DEFAULT: "projected_asset_default",
        D_CLIENT_POLICY_DEFAULT: "projected_asset_default",
        D_CONFIGURED_STATIC: "projected_asset_default",
        D_ND_POLICY_SELECTED: "projected_nd_default",
        D_DERIVATION: "projected_from_transformed",
        D_CALCULATION: "projected_from_transformed",
        D_PROJECTION_RULE_REQUIRED: "projection_rule_required",
        D_CLIENT_ONBOARDING_REQUIRED: "blocked_client_onboarding_dependency",
        D_OPERATOR_REVIEW_REQUIRED: "blocked_operator_or_config_dependency",
        D_CONFIG_MAPPING_REQUIRED: "blocked_operator_or_config_dependency",
        D_NOT_APPLICABLE: "not_applicable",
        D_UNRESOLVED_GAP: "unresolved_source_mapping",
    }.get(disposition, "unresolved_source_mapping")


def _checklist_md(rows, bench, summary, contract_id, client_id, run_id) -> str:
    counts = summary["disposition_counts"]
    lines = [
        "# Target Contract Completion Checklist", "",
        f"Client: {client_id}  ",
        f"Run: {run_id}  ",
        f"Target contract: {contract_id}  ",
        f"Target fields: **{summary['target_field_count']}** "
        f"(completed: {summary['completed_disposition_count']}, "
        f"unresolved: {summary['unresolved_disposition_count']})", "",
        "> Onboarding owns the target-field disposition. Downstream agents "
        "execute these dispositions; they do not rediscover them. "
        "**ND allowed ≠ ND selected** — a field is only completed when a "
        "config/policy has actually selected an ND/default treatment.", "",
        "## Disposition mix", "",
        "| Disposition | Count |", "| --- | --- |",
    ]
    for d in DISPOSITIONS:
        if counts.get(d):
            lines.append(f"| `{d}` | {counts[d]} |")
    lines += ["", "## Secondary requirements", "",
              f"- requires client input: {summary['requires_client_input_count']}",
              f"- requires operator review: {summary['requires_operator_review_count']}",
              f"- requires config: {summary['requires_config_count']}",
              f"- requires projection rule: {summary['requires_projection_rule_count']}",
              f"- blocking for validation: {summary['blocking_for_validation_count']}",
              f"- blocking for projection: {summary['blocking_for_projection_count']}", "",
              "## Review bench", "",
              f"Items needing human / config / client action: **{len(bench)}**", ""]
    by_cat: Dict[str, int] = {}
    for b in bench:
        by_cat[b["review_category"]] = by_cat.get(b["review_category"], 0) + 1
    if by_cat:
        lines += ["| Category | Count |", "| --- | --- |"]
        for cat in sorted(by_cat):
            lines.append(f"| `{cat}` | {by_cat[cat]} |")
    else:
        lines.append("_None — every field has a completed disposition._")
    lines.append("")
    return "\n".join(lines) + "\n"
