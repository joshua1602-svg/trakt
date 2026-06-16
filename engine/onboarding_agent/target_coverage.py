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
# Explicit non-ND regulatory value applied from the regime contract (e.g. RREC8
# lien = "1"). Distinct from configured_static (transaction/client/transform
# value) and from defaulted_ND (an explicit ND code).
DEFAULTED_VALUE = "defaulted_value"
NOT_APPLICABLE = "not_applicable"
MISSING_REQUIRED = "missing_required"
NEEDS_CONFIRMATION = "needs_confirmation"
OPTIONAL_FOR_MI = "optional_for_mi"
# An authoritative Annex 2 code that is in the target universe but has no full
# regime field rule yet (config completeness gap, not a data gap).
PENDING_REGIME_RULE = "pending_regime_rule"
# An Annex 2 code that is explicitly deferred for reconciliation.
DEFERRED = "deferred"

# Overlay coverage statuses that are valid as `coverage_status_if_no_source`.
_OVERLAY_STATUSES = {NOT_APPLICABLE, OPTIONAL_FOR_MI, NEEDS_CONFIRMATION,
                     CONFIGURED_STATIC, DEFAULTED, DERIVED, DEFAULTED_ND,
                     DEFAULTED_VALUE}

# --- Annex 2 config-validation statuses (42 artefact) ---
VS_VALID = "valid"
VS_MISSING_NOT_REQ = "missing_asset_default_but_not_required"
VS_MISSING_REQ = "missing_asset_default_and_required"
VS_INVALID = "invalid_default_not_allowed"
VS_UNKNOWN = "unknown_asset_field_mapping"
VS_NA = "not_applicable"

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
# An asset-config default that is NOT allowed by the regime field rule (e.g. an
# ND code outside nd_allowed). Surfaced as a non-blocking confirmation when a
# valid regime fallback exists; never silently applied.
D_INVALID_DEFAULT = "invalid_default_value"

# Default asset-class config layer for ESMA Annex 2 (UK Equity Release).
_ASSET_CONFIG_DEFAULT = _REPO_ROOT / "config" / "asset" / "product_defaults_ERM.yaml"
_ANNEX2_REGIME_DEFAULT = _REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml"


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
            "esma_code": "",
            "projected_source_field": "",
            "target_domain": _mi_domain(entry),
            "target_label": entry.get("business_name", "") or entry.get("display_name", name),
            "required_status": "required" if tier == "core" else "optional",
            "enforce_presence": False,
            "applicability_status": "applicable",
            "match_field": entry.get("canonical_field", name),
            "synonyms": list(entry.get("synonyms", []) or []),
            "derived": derived,
            "derivation_rule": (f"derived from {entry.get('derived_from')}"
                                if derived and entry.get("derived_from") else ""),
            "default_rule": "",
            "default_value": "",
            "default_rule_source": "",
            "default_reason": "",
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


def _annex2_family(code: str) -> str:
    code = str(code or "").upper()
    if code.startswith("RREC"):
        return "RREC"
    if code.startswith("RREL"):
        return "RREL"
    return "other"


def _code_sort_key(code: str) -> Tuple[str, int, str]:
    """Sort ESMA codes by family then numeric suffix (RREL2 < RREL10)."""
    m = re.match(r"^([A-Za-z]+)(\d+)$", str(code or ""))
    if m:
        return (m.group(1), int(m.group(2)), "")
    return (str(code or ""), 1 << 30, str(code or ""))


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
            "esma_code": code,
            "projected_source_field": rule.get("projected_source_field", ""),
            "target_domain": _annex2_domain(code),
            "target_label": rule.get("workbook_semantic", "") or code,
            "required_status": "mandatory" if rule.get("mandatory") else "optional",
            "enforce_presence": bool(rule.get("enforce_presence")),
            "applicability_status": "deferred_reconciliation" if code in deferred else "applicable",
            "match_field": rule.get("projected_source_field", ""),
            "synonyms": [s for s in (rule.get("projected_source_field", ""),
                                     rule.get("workbook_semantic", "")) if s],
            "derived": bool(derive),
            "derivation_rule": (f"{derive.get('type')}" if derive else ""),
            "default_rule": (f"default_value={default_value}" if default_value else ""),
            "default_value": default_value,
            # The regime contract is the default source until an asset-config
            # layer overrides it (see apply_asset_overlay).
            "default_rule_source": ("regime_config" if default_value else ""),
            "default_reason": ("ESMA Annex 2 regime default" if default_value else ""),
            "nd_allowed": nd_allowed,
            "configured_value_source": configured,
        })
    return "esma_annex_2", str(path), rows


# ---------------------------------------------------------------------------
# Authoritative Annex 2 target universe (workbook-derived registry + regime)
# ---------------------------------------------------------------------------

def load_annex2_authoritative_universe(
    registry_path: Optional[str | Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return the canonical-field map for Annex 2 codes from ``fields_registry``.

    Reads ``fields_registry.yaml`` and extracts every canonical field carrying a
    ``regime_mapping.ESMA_Annex2.code``. Used to resolve a source-matchable
    canonical field name per ESMA code. Returns
    ``{esma_code: {canonical_field, priority, synonyms}}``.
    """
    if not registry_path:
        registry_path = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
    path = Path(registry_path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for fname, meta in (data.get("fields", {}) or {}).items():
        rm = ((meta or {}).get("regime_mapping", {}) or {}).get("ESMA_Annex2", {}) or {}
        code = rm.get("code")
        if code and code not in out:
            out[code] = {
                "canonical_field": fname,
                "priority": str(rm.get("priority", "") or ""),
                "synonyms": list(meta.get("synonyms", []) or []),
            }
    return out


# Workbook-derived authoritative Annex 2 field universe (preferred source).
_ANNEX2_UNIVERSE_DEFAULT = _REPO_ROOT / "config" / "regime" / "annex2_field_universe.yaml"


def load_annex2_workbook_universe(
    universe_path: Optional[str | Path] = None,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Load the authoritative workbook-derived Annex 2 field universe.

    This is the COMPLETE ESMA Annex 2 field set (every field code in the
    template workbook), derived by ``scripts/build_annex2_universe.py``. Returns
    ``({esma_code: {field_name, section, content, nd1_4_allowed, nd5_allowed,
    format}}, source_path)``. Empty when the config is absent (caller falls back
    to the registry mapping and warns).
    """
    path = Path(universe_path) if universe_path else _ANNEX2_UNIVERSE_DEFAULT
    if not path.exists():
        return {}, str(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}, str(path)
    return dict(data.get("fields", {}) or {}), str(path)


def _annex2_nd_allowed_from_workbook(meta: Dict[str, Any]) -> List[str]:
    nd: List[str] = []
    if meta.get("nd1_4_allowed"):
        nd += ["ND1", "ND2", "ND3", "ND4"]
    if meta.get("nd5_allowed"):
        nd += ["ND5"]
    return nd


def build_annex2_full_contract(
    annex2_config_path: Optional[str | Path] = None,
    registry_path: Optional[str | Path] = None,
    universe_path: Optional[str | Path] = None,
) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any]]:
    """Build the FULL Annex 2 target contract over the authoritative universe.

    Authoritative universe source preference:
      1. the workbook-derived field universe (``annex2_field_universe.yaml``);
      2. otherwise the ``fields_registry`` ESMA_Annex2 mapping ∪ regime rules.

    Codes with a full regime rule keep their rich rule. Authoritative codes
    without one are included as ``pending_regime_rule`` (or ``deferred`` when in
    ``reconciliation_scope.deferred_fields``), enriched with workbook metadata
    and the registry canonical field (so they can still be source-mapped).
    Codes declared deferred in the regime config that are NOT in the
    authoritative universe are reported in 43 as ``not_in_authoritative_universe``
    and are not added to 28a.

    Returns ``(contract_id, contract_source, target_fields, universe_meta)``.
    """
    cid, csrc, ruled = load_annex2_target_contract(annex2_config_path)
    ruled_by_code = {r["target_field"]: r for r in ruled}
    try:
        regime = yaml.safe_load(Path(csrc).read_text(encoding="utf-8")) or {}
    except Exception:
        regime = {}
    deferred = set(regime.get("reconciliation_scope", {}).get("deferred_fields", []) or [])

    workbook, workbook_src = load_annex2_workbook_universe(universe_path)
    registry = load_annex2_authoritative_universe(registry_path)

    warnings: List[str] = []
    if workbook:
        authoritative = set(workbook)            # the 107-code workbook universe
        authoritative_source = workbook_src
    else:
        authoritative = set(registry) | set(ruled_by_code)
        authoritative_source = str(registry_path) if registry_path else ""
        warnings.append(
            "workbook-derived Annex 2 universe (annex2_field_universe.yaml) not "
            "found — falling back to the fields_registry mapping, which may be "
            "incomplete relative to the ESMA template.")
    if registry and not (set(ruled_by_code) <= set(registry) | set(workbook)):
        warnings.append("some regime field_rules are absent from the authoritative "
                        "Annex 2 universe.")

    # Contract codes = authoritative universe ∪ implemented regime rules.
    contract_codes = authoritative | set(ruled_by_code)
    rows: List[Dict[str, Any]] = list(ruled)
    for code in sorted(contract_codes - set(ruled_by_code), key=_code_sort_key):
        wb = workbook.get(code, {})
        reg = registry.get(code, {})
        canon = reg.get("canonical_field", "")
        is_deferred = code in deferred
        nd_allowed = _annex2_nd_allowed_from_workbook(wb)
        priority = str(reg.get("priority", "")).strip().lower()
        # Mandatory when the registry says so, or when the workbook permits no ND
        # value at all (strictly required); informational only (never blocking).
        required = (priority == "mandatory"
                    or (bool(wb) and not wb.get("nd1_4_allowed")
                        and not wb.get("nd5_allowed")))
        rows.append({
            "target_field": code,
            "esma_code": code,
            "projected_source_field": canon,
            "target_domain": _annex2_domain(code),
            "target_label": wb.get("field_name", "") or canon or code,
            "required_status": "mandatory" if required else "optional",
            "enforce_presence": False,
            "applicability_status": ("deferred_reconciliation" if is_deferred
                                     else "applicable"),
            "match_field": canon,
            "synonyms": [s for s in ([canon] + list(reg.get("synonyms", []) or [])) if s],
            "derived": False,
            "derivation_rule": "",
            "default_rule": "",
            "default_value": "",
            "default_rule_source": "",
            "default_reason": "",
            "nd_allowed": nd_allowed,
            "configured_value_source": "",
            "pending_regime_rule": (not is_deferred),
            "deferred_reconciliation": is_deferred,
            "in_workbook_universe": code in authoritative,
            "workbook_format": wb.get("format", ""),
        })

    # 43 considers every code we know about, incl. phantom deferred codes that
    # are NOT in the authoritative universe (a config-quality finding).
    all_codes = authoritative | set(ruled_by_code) | set(registry) | deferred
    universe_meta = {
        "workbook_codes": sorted(authoritative, key=_code_sort_key),
        "regime_rule_codes": sorted(ruled_by_code.keys(), key=_code_sort_key),
        "registry_codes": sorted(registry.keys(), key=_code_sort_key),
        "deferred_codes": sorted(deferred, key=_code_sort_key),
        "all_codes": sorted(all_codes, key=_code_sort_key),
        "registry_has_annex2": bool(registry),
        "workbook_universe_present": bool(workbook),
        "registry_source": str(registry_path) if registry_path else "",
        "authoritative_source": authoritative_source,
        "warnings": warnings,
    }
    return cid, csrc, rows, universe_meta


# ---------------------------------------------------------------------------
# Asset-class config layer (LAYER 2) + regime/asset config validation (42)
# ---------------------------------------------------------------------------

def _validate_value_against_rule(value: Any, rule: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a chosen default value against a regime field rule.

    The regime rule defines the *allowed envelope* (nd_allowed / enum map /
    validator regex). An asset-config chosen default must fall inside it.
    """
    v = str(value or "").strip()
    if not v:
        return False, "empty value"
    nd_allowed = [str(x).strip().upper() for x in (rule.get("nd_allowed") or [])]
    if _is_nd(v):
        if v.upper() in nd_allowed:
            return True, f"ND value {v} within nd_allowed {nd_allowed or '[]'}"
        return False, f"ND value {v} not in regime nd_allowed {nd_allowed or '[]'}"
    transform = rule.get("transform", {}) or {}
    enum_map = transform.get("enum_map")
    if isinstance(enum_map, dict) and enum_map:
        keys = {str(k) for k in enum_map.keys()} | {str(x) for x in enum_map.values()}
        if v in keys:
            return True, "value present in regime enum map"
        return False, f"value '{v}' not present in regime enum map"
    validators = rule.get("validators", {}) or {}
    rx = validators.get("regex")
    if rx:
        try:
            if re.match(rx, v):
                return True, "value matches regime validator regex"
            return False, f"value '{v}' fails regime validator regex"
        except re.error:
            return True, "regime validator regex could not be compiled — accepted"
    return True, "no regime constraint on non-ND value — accepted"


def load_asset_defaults(
    asset_config_path: Optional[str | Path] = None,
) -> Tuple[Dict[str, Tuple[str, str, str]], str]:
    """Load the asset-class default layer (``product_defaults_ERM.yaml``).

    Returns ``({normalized_field: (original_field, value, section)}, source_path)``
    where ``section`` is ``defaults`` (static) or ``nd_defaults`` (ND choices).
    """
    path = Path(asset_config_path) if asset_config_path else _ASSET_CONFIG_DEFAULT
    if not path.exists():
        return {}, str(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: Dict[str, Tuple[str, str, str]] = {}
    for section in ("defaults", "nd_defaults"):
        for name, value in (data.get(section, {}) or {}).items():
            norm = _norm_field(name)
            if norm and norm not in out:
                out[norm] = (str(name), str(value), section)
    return out, str(path)


def build_annex2_config_validation(
    regime_config_path: str | Path,
    asset_config_path: Optional[str | Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], str]:
    """Cross-check the asset-config chosen defaults against the regime envelope.

    Returns ``(validation_rows, asset_overlay, asset_config_source)``. The overlay
    is keyed by ESMA code and records, per field, the value to apply (only when
    valid), its source layer and the validation outcome — it never silently
    applies an invalid default.
    """
    regime = yaml.safe_load(Path(regime_config_path).read_text(encoding="utf-8")) or {}
    field_rules = regime.get("field_rules", {}) or {}
    deferred = set(regime.get("reconciliation_scope", {}).get("deferred_fields", []) or [])
    asset_defaults, asset_src = load_asset_defaults(asset_config_path)

    # Reverse index: normalized projected_source_field -> esma_code.
    proj_to_code: Dict[str, str] = {}
    for code, rule in field_rules.items():
        psf = _norm_field(rule.get("projected_source_field", ""))
        if psf and psf not in proj_to_code:
            proj_to_code[psf] = code

    rows: List[Dict[str, Any]] = []
    overlay: Dict[str, Dict[str, Any]] = {}
    consumed: set = set()

    for code, rule in field_rules.items():
        psf = _norm_field(rule.get("projected_source_field", ""))
        nd_allowed = list(rule.get("nd_allowed", []) or [])
        default_allowed = bool(rule.get("default_allowed"))
        default_value = rule.get("default_value", "") if default_allowed else ""
        mandatory = bool(rule.get("mandatory")) or bool(rule.get("enforce_presence"))
        asset_entry = asset_defaults.get(psf)
        if asset_entry:
            consumed.add(psf)
            _orig, asset_val, asset_section = asset_entry
        else:
            asset_val, asset_section = "", ""

        if code in deferred:
            status, msg = VS_NA, "deferred reconciliation field — not enforced in this pass"
        elif asset_entry:
            ok, why = _validate_value_against_rule(asset_val, rule)
            if ok:
                status, msg = VS_VALID, why
                overlay[code] = {
                    "default_value": asset_val, "default_rule_source": "asset_config",
                    "asset_default_value": asset_val, "asset_default_source": asset_section,
                    "validation_status": status, "valid": True, "message": why}
            else:
                status, msg = VS_INVALID, why
                overlay[code] = {
                    "default_value": default_value,
                    "default_rule_source": ("regime_config" if default_value else ""),
                    "asset_default_value": asset_val, "asset_default_source": asset_section,
                    "validation_status": status, "valid": False, "message": why}
        elif mandatory and not default_allowed:
            status, msg = VS_MISSING_REQ, ("mandatory/enforce_presence field with no asset "
                                           "default and no regime default")
        else:
            status = VS_MISSING_NOT_REQ
            msg = ("regime default present" if default_allowed
                   else "field not mandatory; no asset default supplied")

        rows.append({
            "esma_code": code,
            "projected_source_field": rule.get("projected_source_field", ""),
            "regime_nd_allowed": "; ".join(nd_allowed),
            "regime_default_allowed": default_allowed,
            "regime_default_value": default_value,
            "asset_default_value": asset_val,
            "asset_default_source": asset_section,
            "validation_status": status,
            "message": msg,
        })

    # Asset defaults that do not map to any Annex 2 ESMA code.
    for norm, (orig, val, section) in sorted(asset_defaults.items()):
        if norm in consumed:
            continue
        rows.append({
            "esma_code": "",
            "projected_source_field": orig,
            "regime_nd_allowed": "",
            "regime_default_allowed": "",
            "regime_default_value": "",
            "asset_default_value": val,
            "asset_default_source": section,
            "validation_status": VS_UNKNOWN,
            "message": "asset default field does not map to any Annex 2 ESMA code",
        })
    return rows, overlay, asset_src


def apply_asset_overlay(target_fields: List[Dict[str, Any]],
                        overlay: Dict[str, Dict[str, Any]]) -> None:
    """Merge the validated asset-config overlay into the regime target fields.

    Valid asset defaults override the regime default (the asset-specific chosen
    value); invalid asset defaults are NOT applied — the regime default is kept
    and the field is flagged so a non-blocking Gate 4 confirmation is raised.
    """
    by_code = {tf.get("target_field"): tf for tf in target_fields}
    for code, ov in overlay.items():
        tf = by_code.get(code)
        if tf is None:
            continue
        tf["asset_default_value"] = ov.get("asset_default_value", "")
        tf["asset_default_source"] = ov.get("asset_default_source", "")
        tf["config_validation_status"] = ov.get("validation_status", "")
        tf["config_validation_message"] = ov.get("message", "")
        if ov.get("valid") and str(ov.get("default_value", "")).strip():
            tf["default_value"] = ov["default_value"]
            tf["default_rule"] = f"default_value={ov['default_value']}"
            tf["default_rule_source"] = "asset_config"
            tf["default_reason"] = ov.get("message", "")
        else:
            # Invalid asset default — keep the regime default, flag for review.
            tf["asset_default_invalid"] = True
            if not tf.get("default_rule_source"):
                tf["default_rule_source"] = ov.get("default_rule_source", "")
            tf["default_reason"] = ov.get("message", "")


def annex2_validation_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["validation_status"]] = counts.get(r["validation_status"], 0) + 1
    return {
        "annex2_config_rows_total": len(rows),
        "valid": counts.get(VS_VALID, 0),
        "invalid_default_not_allowed": counts.get(VS_INVALID, 0),
        "missing_asset_default_and_required": counts.get(VS_MISSING_REQ, 0),
        "missing_asset_default_but_not_required": counts.get(VS_MISSING_NOT_REQ, 0),
        "unknown_asset_field_mapping": counts.get(VS_UNKNOWN, 0),
        "not_applicable": counts.get(VS_NA, 0),
        "validation_status_counts": counts,
    }


_ANNEX2_VALIDATION_COLUMNS = [
    "esma_code", "projected_source_field", "regime_nd_allowed",
    "regime_default_allowed", "regime_default_value", "asset_default_value",
    "asset_default_source", "validation_status", "message",
]


def write_annex2_config_validation(
    out_dir: str | Path,
    rows: List[Dict[str, Any]],
    *,
    regime_config_source: str = "",
    asset_config_source: str = "",
) -> Dict[str, str]:
    """Write the 42 Annex 2 config-validation artefacts (csv / json / md)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = annex2_validation_summary(rows)
    _write_csv(out / "42_annex2_config_validation.csv", rows, _ANNEX2_VALIDATION_COLUMNS)
    (out / "42_annex2_config_validation.json").write_text(
        json.dumps({"regime_config_source": regime_config_source,
                    "asset_config_source": asset_config_source,
                    "summary": summary, "rows": rows}, indent=2, default=str),
        encoding="utf-8")
    md = ["# ESMA Annex 2 config validation", "",
          f"- **Regime config:** `{regime_config_source}`",
          f"- **Asset config:** `{asset_config_source}`",
          f"- **Fields checked:** {summary['annex2_config_rows_total']}",
          f"- **Valid:** {summary['valid']}",
          f"- **Invalid (default not allowed):** {summary['invalid_default_not_allowed']}",
          f"- **Missing required:** {summary['missing_asset_default_and_required']}",
          f"- **Unknown asset field mapping:** {summary['unknown_asset_field_mapping']}", "",
          "## Validation status counts", ""]
    for st, c in sorted(summary["validation_status_counts"].items(), key=lambda kv: -kv[1]):
        md.append(f"- `{st}`: {c}")
    invalid = [r for r in rows if r["validation_status"] == VS_INVALID]
    md += ["", "## Invalid / conflicting defaults (surfaced, not applied)", ""]
    if invalid:
        for r in invalid:
            md.append(f"- `{r['esma_code']}` ({r['projected_source_field']}): "
                      f"asset='{r['asset_default_value']}' vs regime nd_allowed="
                      f"[{r['regime_nd_allowed']}] — {r['message']}")
    else:
        md.append("_None._")
    (out / "42_annex2_config_validation_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    return {
        "csv": str(out / "42_annex2_config_validation.csv"),
        "json": str(out / "42_annex2_config_validation.json"),
        "summary_md": str(out / "42_annex2_config_validation_summary.md"),
    }


# ---------------------------------------------------------------------------
# Annex 2 field-universe reconciliation (43)
# ---------------------------------------------------------------------------

# Coverage statuses that count as genuinely deliverable Annex 2 output.
_DELIVERABLE_STATUSES = {SOURCE_MAPPED, SOURCE_MAPPED_ALT, DERIVED,
                         CONFIGURED_STATIC, DEFAULTED_VALUE, DEFAULTED_ND}

_ANNEX2_RECON_COLUMNS = [
    "esma_code", "field_family", "in_workbook_reconciliation",
    "in_registry_mapping", "in_regime_field_rules", "in_config_validation",
    "in_28a_coverage", "coverage_status", "reconciliation_status",
    "registry_mapping_status", "message",
]


def build_annex2_field_universe_reconciliation(
    universe_meta: Dict[str, Any],
    coverage_rows: List[Dict[str, Any]],
    config_validation_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Reconcile the Annex 2 code universes: workbook registry vs regime rules vs
    config validation vs 28a coverage. One row per known ESMA code.
    """
    workbook = set(universe_meta.get("workbook_codes", []) or [])
    regime = set(universe_meta.get("regime_rule_codes", []) or [])
    registry = set(universe_meta.get("registry_codes", []) or [])
    deferred = set(universe_meta.get("deferred_codes", []) or [])
    cov_by: Dict[str, Dict[str, Any]] = {}
    for r in coverage_rows:
        code = r.get("esma_code") or r.get("target_field")
        if code:
            cov_by[code] = r
    validation_codes = {r.get("esma_code") for r in (config_validation_rows or [])
                        if r.get("esma_code")}

    all_codes = (set(universe_meta.get("all_codes", []) or [])
                 | set(cov_by) | validation_codes)
    rows: List[Dict[str, Any]] = []
    for code in sorted(all_codes, key=_code_sort_key):
        in_wb = code in workbook
        in_regime = code in regime
        in_val = code in validation_codes
        in_28a = code in cov_by
        cstatus = cov_by.get(code, {}).get("coverage_status", "")
        if not in_wb:
            rstatus = "not_in_authoritative_universe"
            msg = ("declared in regime/registry config but NOT in the authoritative "
                   "Annex 2 workbook universe (config-quality issue)")
        elif not in_28a:
            rstatus = "missing_from_28a"
            msg = "authoritative Annex 2 code is missing from 28a target coverage"
        elif code in deferred:
            rstatus = "deferred_in_regime"
            msg = "explicitly deferred for reconciliation"
        elif not in_regime:
            rstatus = "missing_from_regime_rules"
            msg = ("in the authoritative universe but has no full regime field "
                   "rule yet (pending configuration)")
        elif not in_val:
            rstatus = "missing_from_validation"
            msg = "regime rule present but no config-validation row"
        else:
            rstatus = "present_everywhere"
            msg = "present in workbook universe, regime rules, validation and 28a"
        in_registry = code in registry
        # Registry-mapping coverage: an authoritative workbook code should carry
        # a fields_registry ESMA_Annex2 mapping; if not, it is a registry gap.
        if not in_wb:
            registry_status = "not_in_workbook"
        elif in_registry:
            registry_status = "registry_mapped"
        else:
            registry_status = "registry_gap"
        rows.append({
            "esma_code": code,
            "field_family": _annex2_family(code),
            "in_workbook_reconciliation": in_wb,
            "in_registry_mapping": in_registry,
            "in_regime_field_rules": in_regime,
            "in_config_validation": in_val,
            "in_28a_coverage": in_28a,
            "coverage_status": cstatus,
            "reconciliation_status": rstatus,
            "registry_mapping_status": registry_status,
            "message": msg,
        })
    return rows


def annex2_reconciliation_summary(
    rows: List[Dict[str, Any]],
    universe_meta: Dict[str, Any],
    coverage_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rstatus_counts: Dict[str, int] = {}
    for r in rows:
        rstatus_counts[r["reconciliation_status"]] = (
            rstatus_counts.get(r["reconciliation_status"], 0) + 1)
    deliverable = sum(1 for r in coverage_rows
                      if r.get("coverage_status") in _DELIVERABLE_STATUSES)
    cov_codes = {r.get("esma_code") or r.get("target_field") for r in coverage_rows}
    cov_codes.discard(None)
    return {
        # The authoritative universe is the workbook field set (or registry fallback).
        "authoritative_field_count": len(universe_meta.get("workbook_codes", []) or []),
        "known_codes_count": len(universe_meta.get("all_codes", []) or []),
        "workbook_reconciliation_count": len(universe_meta.get("workbook_codes", []) or []),
        "regime_rule_count": len(universe_meta.get("regime_rule_codes", []) or []),
        "registry_mapped_count": sum(
            1 for r in rows if r.get("registry_mapping_status") == "registry_mapped"),
        "registry_gap_count": sum(
            1 for r in rows if r.get("registry_mapping_status") == "registry_gap"),
        "config_validation_count": sum(1 for r in rows if r["in_config_validation"]),
        "coverage_field_count": len(cov_codes),
        "deferred_field_count": len(universe_meta.get("deferred_codes", []) or []),
        "missing_from_28a_count": rstatus_counts.get("missing_from_28a", 0),
        "missing_from_regime_rules_count": rstatus_counts.get("missing_from_regime_rules", 0),
        "not_in_authoritative_universe_count": rstatus_counts.get(
            "not_in_authoritative_universe", 0),
        "deliverable_field_count": deliverable,
        "workbook_universe_present": bool(universe_meta.get("workbook_universe_present")),
        "registry_has_annex2": bool(universe_meta.get("registry_has_annex2")),
        "reconciliation_status_counts": rstatus_counts,
    }


def write_annex2_field_universe_reconciliation(
    out_dir: str | Path,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    *,
    registry_source: str = "",
    regime_config_source: str = "",
    warnings: Optional[List[str]] = None,
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings = warnings or []
    _write_csv(out / "43_annex2_field_universe_reconciliation.csv", rows,
               _ANNEX2_RECON_COLUMNS)
    (out / "43_annex2_field_universe_reconciliation.json").write_text(
        json.dumps({"registry_source": registry_source,
                    "regime_config_source": regime_config_source,
                    "warnings": warnings, "summary": summary, "rows": rows},
                   indent=2, default=str), encoding="utf-8")
    md = ["# ESMA Annex 2 field-universe reconciliation", "",
          f"- **Authoritative target universe (workbook):** {summary['authoritative_field_count']}",
          f"- **Registry-mapped (fields_registry ESMA_Annex2):** "
          f"{summary.get('registry_mapped_count', 0)}",
          f"- **Registry gaps (workbook code, no registry mapping):** "
          f"{summary.get('registry_gap_count', 0)}",
          f"- **Regime field rules:** {summary['regime_rule_count']}",
          f"- **Config-validation rows:** {summary['config_validation_count']}",
          f"- **Present in 28a coverage:** {summary['coverage_field_count']}",
          f"- **Deferred / pending reconciliation:** {summary['deferred_field_count']}",
          f"- **Missing from 28a:** {summary['missing_from_28a_count']}",
          f"- **Declared but not in authoritative universe:** "
          f"{summary.get('not_in_authoritative_universe_count', 0)}",
          f"- **Deliverable (rule + coverage):** {summary['deliverable_field_count']}", ""]
    if warnings:
        md += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]
    md += ["## Reconciliation status counts", ""]
    for st, c in sorted(summary["reconciliation_status_counts"].items(),
                        key=lambda kv: -kv[1]):
        md.append(f"- `{st}`: {c}")
    missing = [r for r in rows if r["reconciliation_status"] == "missing_from_28a"]
    md += ["", "## Codes missing from 28a (must be zero)", ""]
    if missing:
        for r in missing:
            md.append(f"- `{r['esma_code']}` — {r['message']}")
    else:
        md.append("_None — 28a covers the full authoritative universe._")
    phantom = [r for r in rows
               if r["reconciliation_status"] == "not_in_authoritative_universe"]
    md += ["", "## Codes declared in config but not in the authoritative universe", ""]
    if phantom:
        for r in phantom:
            md.append(f"- `{r['esma_code']}` — {r['message']}")
    else:
        md.append("_None._")
    gaps = [r for r in rows if r.get("registry_mapping_status") == "registry_gap"]
    md += ["", "## Registry gaps (workbook codes with no fields_registry mapping)", ""]
    if gaps:
        for r in gaps:
            md.append(f"- `{r['esma_code']}` — authoritative workbook code with no "
                      "ESMA_Annex2 mapping in fields_registry.yaml")
    else:
        md.append("_None — every authoritative workbook code is registry-mapped._")
    md += ["", "## Relationship between artefacts", "",
           "- **42 config validation** applies to fields with regime/default "
           "validation metadata (regime field rules).",
           "- **28a coverage** applies to the FULL authoritative target universe; "
           "codes without a full regime rule appear as `pending_regime_rule` / "
           "`deferred`, never dropped.", ""]
    (out / "43_annex2_field_universe_reconciliation_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    return {
        "csv": str(out / "43_annex2_field_universe_reconciliation.csv"),
        "json": str(out / "43_annex2_field_universe_reconciliation.json"),
        "summary_md": str(out / "43_annex2_field_universe_reconciliation_summary.md"),
    }


# ---------------------------------------------------------------------------
# Annex 2 ND-eligibility reconciliation (44) — regime nd_allowed vs workbook
# ---------------------------------------------------------------------------

_ANNEX2_ND_COLUMNS = [
    "esma_code", "field_family", "has_regime_rule", "in_workbook",
    "regime_nd_allowed", "workbook_nd1_4_allowed", "workbook_nd5_allowed",
    "workbook_nd_allowed", "nd_alignment_status", "message",
]


def build_annex2_nd_eligibility_reconciliation(
    regime_config_path: Optional[str | Path] = None,
    universe_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Compare each regime rule's ``nd_allowed`` set against the workbook's
    authoritative ND1-4 / ND5 eligibility. Report-only — never mutates the
    regime validation behaviour.

    Statuses:
      match            - regime nd_allowed equals the workbook ND eligibility
      regime_stricter  - regime forbids ND value(s) the workbook permits
      regime_broader   - regime permits ND value(s) the workbook FORBIDS (risk)
      divergent        - sets overlap only partially / are disjoint
      no_regime_rule   - workbook code with no regime rule to compare
      not_in_workbook  - regime code absent from the authoritative workbook
    """
    cid, csrc, _ruled = load_annex2_target_contract(regime_config_path)
    try:
        regime = yaml.safe_load(Path(csrc).read_text(encoding="utf-8")) or {}
    except Exception:
        regime = {}
    field_rules = regime.get("field_rules", {}) or {}
    workbook, _src = load_annex2_workbook_universe(universe_path)

    all_codes = set(field_rules) | set(workbook)
    rows: List[Dict[str, Any]] = []
    for code in sorted(all_codes, key=_code_sort_key):
        rule = field_rules.get(code)
        wb = workbook.get(code)
        in_wb = wb is not None
        has_rule = rule is not None
        regime_nd = {str(x).strip().upper() for x in (rule.get("nd_allowed") or [])} if has_rule else set()
        wb_nd = set(_annex2_nd_allowed_from_workbook(wb or {}))
        if not in_wb:
            status = "not_in_workbook"
            msg = "regime code is absent from the authoritative Annex 2 workbook"
        elif not has_rule:
            status = "no_regime_rule"
            msg = "workbook code has no regime field rule to compare"
        elif regime_nd == wb_nd:
            status = "match"
            msg = "regime nd_allowed matches workbook ND eligibility"
        elif regime_nd < wb_nd:
            status = "regime_stricter"
            msg = ("regime forbids ND value(s) the workbook permits: "
                   + ", ".join(sorted(wb_nd - regime_nd)))
        elif regime_nd > wb_nd:
            status = "regime_broader"
            msg = ("regime permits ND value(s) the workbook FORBIDS (compliance "
                   "risk): " + ", ".join(sorted(regime_nd - wb_nd)))
        else:
            status = "divergent"
            msg = (f"regime/workbook ND sets diverge: regime="
                   f"[{', '.join(sorted(regime_nd))}] workbook=[{', '.join(sorted(wb_nd))}]")
        rows.append({
            "esma_code": code,
            "field_family": _annex2_family(code),
            "has_regime_rule": has_rule,
            "in_workbook": in_wb,
            "regime_nd_allowed": "; ".join(sorted(regime_nd)),
            "workbook_nd1_4_allowed": bool((wb or {}).get("nd1_4_allowed")),
            "workbook_nd5_allowed": bool((wb or {}).get("nd5_allowed")),
            "workbook_nd_allowed": "; ".join(sorted(wb_nd)),
            "nd_alignment_status": status,
            "message": msg,
        })
    return rows


def annex2_nd_eligibility_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["nd_alignment_status"]] = counts.get(r["nd_alignment_status"], 0) + 1
    return {
        "nd_rows_total": len(rows),
        "match": counts.get("match", 0),
        "regime_stricter": counts.get("regime_stricter", 0),
        "regime_broader": counts.get("regime_broader", 0),
        "divergent": counts.get("divergent", 0),
        "no_regime_rule": counts.get("no_regime_rule", 0),
        "not_in_workbook": counts.get("not_in_workbook", 0),
        # regime_broader + divergent are the compliance-relevant mismatches.
        "nd_compliance_risk_count": counts.get("regime_broader", 0) + counts.get("divergent", 0),
        "nd_alignment_status_counts": counts,
    }


def write_annex2_nd_eligibility_reconciliation(
    out_dir: str | Path,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    *,
    regime_config_source: str = "",
    universe_source: str = "",
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "44_annex2_nd_eligibility_reconciliation.csv", rows,
               _ANNEX2_ND_COLUMNS)
    (out / "44_annex2_nd_eligibility_reconciliation.json").write_text(
        json.dumps({"regime_config_source": regime_config_source,
                    "universe_source": universe_source,
                    "summary": summary, "rows": rows}, indent=2, default=str),
        encoding="utf-8")
    md = ["# ESMA Annex 2 ND-eligibility reconciliation", "",
          "Report only — compares each regime rule's `nd_allowed` against the "
          "authoritative workbook ND1-4 / ND5 eligibility. The regime validation "
          "behaviour is unchanged.", "",
          f"- **Codes compared:** {summary['nd_rows_total']}",
          f"- **Match:** {summary['match']}",
          f"- **Regime stricter than workbook:** {summary['regime_stricter']}",
          f"- **Regime broader than workbook (compliance risk):** {summary['regime_broader']}",
          f"- **Divergent ND sets:** {summary['divergent']}",
          f"- **Workbook codes without a regime rule:** {summary['no_regime_rule']}", "",
          "## ND-eligibility status counts", ""]
    for st, c in sorted(summary["nd_alignment_status_counts"].items(), key=lambda kv: -kv[1]):
        md.append(f"- `{st}`: {c}")
    risk = [r for r in rows if r["nd_alignment_status"] in ("regime_broader", "divergent")]
    md += ["", "## Compliance-relevant mismatches (regime_broader / divergent)", ""]
    if risk:
        for r in risk:
            md.append(f"- `{r['esma_code']}` ({r['nd_alignment_status']}): "
                      f"regime=[{r['regime_nd_allowed']}] workbook=[{r['workbook_nd_allowed']}] "
                      f"— {r['message']}")
    else:
        md.append("_None._")
    (out / "44_annex2_nd_eligibility_reconciliation_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    return {
        "csv": str(out / "44_annex2_nd_eligibility_reconciliation.csv"),
        "json": str(out / "44_annex2_nd_eligibility_reconciliation.json"),
        "summary_md": str(out / "44_annex2_nd_eligibility_reconciliation_summary.md"),
    }


# ---------------------------------------------------------------------------
# Annex 2 config-alignment review (45) — record of alignment actions vs workbook
# ---------------------------------------------------------------------------

# Audit trail of the alignment actions taken to bring the Annex 2 configs into
# line with the authoritative workbook universe. Records the BEFORE state so 45
# can show before/after truthfully; the builder verifies the live config matches
# the recorded AFTER state (else the row is flagged for manual review).
_ANNEX2_ALIGNMENT_ACTIONS: Dict[str, Any] = {
    # regime nd_allowed sets that were broader than the workbook ND envelope and
    # were tightened to it (compliance-risk fixes). before -> after.
    "nd_tightened_to_workbook": {
        "RREL1": {"before": ["ND1", "ND2", "ND3"], "after": []},
        "RREL2": {"before": ["ND5"], "after": []},
        "RREL6": {"before": ["ND5"], "after": []},
        "RREL69": {"before": ["ND5"], "after": []},
        "RREL83": {"before": ["ND5"], "after": []},
    },
    # registry mappings added to close a fields_registry gap.
    "registry_mapping_added": {
        "RREC1": {"canonical_field": "collateral_unique_identifier",
                  "reason": "workbook defines RREC1 as the RREL1 unique identifier "
                            "echoed into the collateral section"},
    },
    # codes removed from the active deferred list because they are not in the
    # authoritative workbook universe (phantom). Moved to an audit-only list.
    "phantom_deferred_removed": [
        "RREC24", "RREC25", "RREC26", "RREC27", "RREC30", "RREC31", "RREC32",
        "RREC33", "RREC34", "RREC36", "RREC39",
    ],
    # asset-default enum maps added so a plain-English ERM default resolves to a
    # valid ESMA enum code (not OTHR).
    "asset_enum_map_added": {
        "RREL42": {"value": "Fixed", "code": "FXRL",
                   "reason": "ERM fixed-for-life rate -> ESMA FXRL"},
    },
    # {LIST} fields constrained to the workbook's allowed enum codes (identity
    # maps derived from the authoritative template). Semantic mapping verified.
    "enum_constrained_from_workbook": {
        "RREL19": 6, "RREL56": 30, "RREL57": 13, "RREC10": 8, "RREC18": 9,
    },
}

_ANNEX2_ALIGN_COLUMNS = [
    "esma_code", "workbook_field_name", "workbook_nd_allowed",
    "regime_nd_allowed_before", "regime_nd_allowed_after", "alignment_status",
    "action_taken", "requires_manual_review", "message",
]


def build_annex2_config_alignment_review(
    regime_config_path: Optional[str | Path] = None,
    asset_config_path: Optional[str | Path] = None,
    universe_path: Optional[str | Path] = None,
    registry_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Record every Annex 2 config-alignment action and unresolved review item.

    Combines: the workbook universe, the (post-alignment) regime rules, the
    registry mapping, the ND reconciliation and the asset-config validation, plus
    the recorded alignment actions, into one auditable review. Report-only.
    """
    cid, csrc, _ruled = load_annex2_target_contract(regime_config_path)
    try:
        regime = yaml.safe_load(Path(csrc).read_text(encoding="utf-8")) or {}
    except Exception:
        regime = {}
    field_rules = regime.get("field_rules", {}) or {}
    workbook, _src = load_annex2_workbook_universe(universe_path)
    registry = load_annex2_authoritative_universe(registry_path)
    val_rows, _overlay, _asrc = build_annex2_config_validation(
        csrc, asset_config_path)
    invalid_by_code = {r["esma_code"]: r for r in val_rows
                       if r["validation_status"] == VS_INVALID}

    nd_tightened = _ANNEX2_ALIGNMENT_ACTIONS["nd_tightened_to_workbook"]
    enum_added = _ANNEX2_ALIGNMENT_ACTIONS["asset_enum_map_added"]

    def _wb_nd(code: str) -> List[str]:
        return _annex2_nd_allowed_from_workbook(workbook.get(code, {}))

    def _fmt(nd) -> str:
        return "; ".join(nd)

    rows: List[Dict[str, Any]] = []

    # 1. ND alignment for every regime-ruled workbook code.
    for code in sorted(set(field_rules) & set(workbook), key=_code_sort_key):
        wb = workbook.get(code, {})
        wb_nd = set(_wb_nd(code))
        after = {str(x).strip().upper()
                 for x in (field_rules[code].get("nd_allowed") or [])}
        before = after
        manual = False
        if code in nd_tightened:
            before = {str(x).upper() for x in nd_tightened[code]["before"]}
            status = "tightened_to_workbook"
            action = f"nd_allowed {sorted(before)} -> {sorted(after)} (workbook envelope)"
            msg = "regime ND was broader than the workbook; tightened to ESMA envelope"
            if after != wb_nd:
                manual = True
                msg += " (WARNING: live config does not match workbook — review)"
        elif after == wb_nd:
            status = "aligned"
            action = "none"
            msg = "regime nd_allowed matches workbook ND eligibility"
        elif after < wb_nd:
            status = "left_stricter_by_policy"
            action = "none (kept stricter)"
            msg = ("regime is intentionally stricter than the workbook envelope: "
                   f"workbook permits {sorted(wb_nd - after)} that the regime omits")
        elif after > wb_nd:
            status = "divergent_requires_review"
            action = "none"
            manual = True
            msg = ("regime permits ND value(s) the workbook FORBIDS: "
                   f"{sorted(after - wb_nd)} (compliance risk — review)")
        else:
            status = "divergent_requires_review"
            action = "none"
            manual = True
            msg = (f"regime/workbook ND sets diverge: regime={sorted(after)} "
                   f"workbook={sorted(wb_nd)} — manual review")
        rows.append({
            "esma_code": code,
            "workbook_field_name": wb.get("field_name", ""),
            "workbook_nd_allowed": _fmt(sorted(wb_nd)),
            "regime_nd_allowed_before": _fmt(sorted(before)),
            "regime_nd_allowed_after": _fmt(sorted(after)),
            "alignment_status": status,
            "action_taken": action,
            "requires_manual_review": manual,
            "message": msg,
        })

    # 2. Registry mapping additions / remaining gaps for workbook codes.
    for code, info in _ANNEX2_ALIGNMENT_ACTIONS["registry_mapping_added"].items():
        wb = workbook.get(code, {})
        mapped = code in registry
        rows.append({
            "esma_code": code,
            "workbook_field_name": wb.get("field_name", ""),
            "workbook_nd_allowed": _fmt(_wb_nd(code)),
            "regime_nd_allowed_before": "",
            "regime_nd_allowed_after": "",
            "alignment_status": "registry_mapping_added" if mapped else "registry_gap",
            "action_taken": (f"added fields_registry mapping -> {info['canonical_field']}"
                             if mapped else "registry mapping still missing"),
            "requires_manual_review": not mapped,
            "message": (info["reason"] if mapped
                        else "authoritative workbook code has no registry mapping"),
        })
    # Any other workbook code missing from the registry is a surfaced gap.
    for code in sorted(set(workbook) - set(registry), key=_code_sort_key):
        if code in _ANNEX2_ALIGNMENT_ACTIONS["registry_mapping_added"]:
            continue
        wb = workbook.get(code, {})
        rows.append({
            "esma_code": code,
            "workbook_field_name": wb.get("field_name", ""),
            "workbook_nd_allowed": _fmt(_wb_nd(code)),
            "regime_nd_allowed_before": "",
            "regime_nd_allowed_after": "",
            "alignment_status": "registry_gap",
            "action_taken": "none",
            "requires_manual_review": True,
            "message": "authoritative workbook code has no fields_registry ESMA_Annex2 mapping",
        })

    # 3. Phantom deferred codes removed from the active runtime list.
    active_deferred = set(regime.get("reconciliation_scope", {})
                          .get("deferred_fields", []) or [])
    for code in _ANNEX2_ALIGNMENT_ACTIONS["phantom_deferred_removed"]:
        still_active = code in active_deferred
        rows.append({
            "esma_code": code,
            "workbook_field_name": workbook.get(code, {}).get("field_name", "(not in workbook)"),
            "workbook_nd_allowed": "",
            "regime_nd_allowed_before": "deferred",
            "regime_nd_allowed_after": ("deferred (STILL ACTIVE)" if still_active
                                        else "removed (audit-only)"),
            "alignment_status": "phantom_deferred_removed",
            "action_taken": ("moved out of active deferred_fields to audit-only"
                             if not still_active else "NOT removed — review"),
            "requires_manual_review": still_active,
            "message": ("code is not in the authoritative workbook universe; removed "
                        "from active runtime deferral" if not still_active
                        else "phantom code still active in deferred_fields"),
        })

    # 4. Asset-default conflicts: resolved via enum map, or still unresolved.
    for code, info in enum_added.items():
        wb = workbook.get(code, {})
        resolved = code not in invalid_by_code
        rows.append({
            "esma_code": code,
            "workbook_field_name": wb.get("field_name", ""),
            "workbook_nd_allowed": _fmt(_wb_nd(code)),
            "regime_nd_allowed_before": "",
            "regime_nd_allowed_after": "",
            "alignment_status": "aligned" if resolved else "asset_default_conflict",
            "action_taken": (f"added enum map '{info['value']}' -> {info['code']} "
                             f"({info['reason']})"),
            "requires_manual_review": not resolved,
            "message": ("asset default now resolves to a valid ESMA enum"
                        if resolved else "enum map added but asset default still invalid"),
        })
    for code, r in sorted(invalid_by_code.items(), key=lambda kv: _code_sort_key(kv[0])):
        if code in enum_added:
            continue
        wb = workbook.get(code, {})
        rows.append({
            "esma_code": code,
            "workbook_field_name": wb.get("field_name", ""),
            "workbook_nd_allowed": _fmt(_wb_nd(code)),
            "regime_nd_allowed_before": r.get("regime_nd_allowed", ""),
            "regime_nd_allowed_after": r.get("regime_nd_allowed", ""),
            "alignment_status": "asset_default_conflict",
            "action_taken": "none — invalid asset default surfaced, not applied",
            "requires_manual_review": True,
            "message": (f"asset default '{r.get('asset_default_value', '')}' is invalid "
                        f"against the active regime rule: {r.get('message', '')}"),
        })

    # 5. {LIST} fields constrained to the workbook's allowed enum codes.
    for code, n in _ANNEX2_ALIGNMENT_ACTIONS["enum_constrained_from_workbook"].items():
        wb = workbook.get(code, {})
        emap = ((field_rules.get(code, {}).get("transform") or {}).get("enum_map")) or {}
        applied = bool(emap)
        rows.append({
            "esma_code": code,
            "workbook_field_name": wb.get("field_name", ""),
            "workbook_nd_allowed": _fmt(_wb_nd(code)),
            "regime_nd_allowed_before": "",
            "regime_nd_allowed_after": "",
            "alignment_status": "enum_constrained_to_workbook" if applied else "registry_gap",
            "action_taken": (f"added enum_map of {n} workbook allowed codes"
                             if applied else "enum_map missing — review"),
            "requires_manual_review": not applied,
            "message": ("enum values constrained to the workbook's allowed set"
                        if applied else "expected enum_map not present"),
        })
    return rows


def annex2_config_alignment_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["alignment_status"]] = counts.get(r["alignment_status"], 0) + 1
    return {
        "alignment_rows_total": len(rows),
        "aligned": counts.get("aligned", 0),
        "tightened_to_workbook": counts.get("tightened_to_workbook", 0),
        "left_stricter_by_policy": counts.get("left_stricter_by_policy", 0),
        "divergent_requires_review": counts.get("divergent_requires_review", 0),
        "registry_mapping_added": counts.get("registry_mapping_added", 0),
        "registry_gap": counts.get("registry_gap", 0),
        "phantom_deferred_removed": counts.get("phantom_deferred_removed", 0),
        "asset_default_conflict": counts.get("asset_default_conflict", 0),
        "enum_constrained_to_workbook": counts.get("enum_constrained_to_workbook", 0),
        "requires_manual_review_count": sum(1 for r in rows if r["requires_manual_review"]),
        "alignment_status_counts": counts,
    }


def write_annex2_config_alignment_review(
    out_dir: str | Path,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    *,
    regime_config_source: str = "",
    registry_source: str = "",
    asset_config_source: str = "",
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "45_annex2_config_alignment_review.csv", rows,
               _ANNEX2_ALIGN_COLUMNS)
    (out / "45_annex2_config_alignment_review.json").write_text(
        json.dumps({"regime_config_source": regime_config_source,
                    "registry_source": registry_source,
                    "asset_config_source": asset_config_source,
                    "summary": summary, "rows": rows}, indent=2, default=str),
        encoding="utf-8")
    md = ["# ESMA Annex 2 config-alignment review", "",
          "Record of every config-alignment action taken to align with the "
          "authoritative workbook universe, plus unresolved review items. "
          "Report only — regime validation behaviour is unchanged.", "",
          f"- **Aligned:** {summary['aligned']}",
          f"- **Tightened to workbook (compliance-risk fixes):** {summary['tightened_to_workbook']}",
          f"- **Left stricter by policy:** {summary['left_stricter_by_policy']}",
          f"- **Divergent (manual review):** {summary['divergent_requires_review']}",
          f"- **Registry mappings added:** {summary['registry_mapping_added']}",
          f"- **Registry gaps remaining:** {summary['registry_gap']}",
          f"- **Phantom deferred removed:** {summary['phantom_deferred_removed']}",
          f"- **Asset-default conflicts:** {summary['asset_default_conflict']}",
          f"- **Requires manual review:** {summary['requires_manual_review_count']}", ""]
    review = [r for r in rows if r["requires_manual_review"]]
    md += ["## Items requiring manual review", ""]
    if review:
        for r in review:
            md.append(f"- `{r['esma_code']}` ({r['alignment_status']}): {r['message']}")
    else:
        md.append("_None._")
    md += ["", "## Actions taken", ""]
    for r in rows:
        if r["action_taken"] not in ("none", "none (kept stricter)"):
            md.append(f"- `{r['esma_code']}` ({r['alignment_status']}): {r['action_taken']}")
    (out / "45_annex2_config_alignment_review_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    return {
        "csv": str(out / "45_annex2_config_alignment_review.csv"),
        "json": str(out / "45_annex2_config_alignment_review.json"),
        "summary_md": str(out / "45_annex2_config_alignment_review_summary.md"),
    }


# ---------------------------------------------------------------------------
# Annex 2 enum-coverage reconciliation (46) — regime enum_map vs workbook codes
# ---------------------------------------------------------------------------

_ANNEX2_ENUM_COLUMNS = [
    "esma_code", "workbook_field_name", "regime_source_field", "has_regime_rule",
    "has_enum_map", "workbook_allowed_count", "workbook_allowed_codes",
    "regime_enum_targets", "targets_outside_workbook", "regime_semantic_aligned",
    "enum_coverage_status", "requires_manual_review", "message",
]


def _annex2_workbook_enum_codes(content: str) -> List[str]:
    """Parse the allowed enum codes from a workbook field's content prose.

    Codes appear parenthesised, e.g. ``Employed - Private Sector (EMRS)``. Only
    3-4 char upper tokens are treated as codes, to avoid catching regulatory
    citations such as ``(EU)``.
    """
    seen: List[str] = []
    for m in re.findall(r'\(([A-Z]{3,4}\d?)\)', content or ""):
        if m not in seen:
            seen.append(m)
    return seen


def _tokens(s: str) -> set:
    return set(re.findall(r'[a-z0-9]+', (s or "").lower()))


def build_annex2_enum_coverage_reconciliation(
    regime_config_path: Optional[str | Path] = None,
    universe_path: Optional[str | Path] = None,
    registry_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Reconcile regime ``enum_map`` coverage against the workbook's allowed
    enum codes for every ``{LIST}`` Annex 2 field. Report-only.

    Statuses:
      constrained_within_workbook  - enum_map present, all targets are workbook codes
      targets_outside_workbook     - enum_map maps to code(s) the workbook forbids
      unconstrained_no_enum_map    - regime rule present but no enum_map (backlog)
      semantic_mismatch            - regime rule's source field does not match the
                                     workbook field for this code (enum not trusted)
      no_regime_rule               - {LIST} field with no regime rule yet
    """
    cid, csrc, _ruled = load_annex2_target_contract(regime_config_path)
    try:
        regime = yaml.safe_load(Path(csrc).read_text(encoding="utf-8")) or {}
    except Exception:
        regime = {}
    field_rules = regime.get("field_rules", {}) or {}
    workbook, _src = load_annex2_workbook_universe(universe_path)
    registry = load_annex2_authoritative_universe(registry_path)

    rows: List[Dict[str, Any]] = []
    list_codes = [c for c, m in workbook.items()
                  if str(m.get("format", "")).strip().upper() == "{LIST}"]
    for code in sorted(list_codes, key=_code_sort_key):
        wb = workbook[code]
        allowed = _annex2_workbook_enum_codes(wb.get("content", ""))
        rule = field_rules.get(code)
        has_rule = rule is not None
        emap = ((rule.get("transform") or {}).get("enum_map") if has_rule else None) or {}
        targets = sorted(set(emap.values()))
        outside = sorted(set(targets) - set(allowed)) if allowed else []
        # Semantic check: regime source field vs the registry canonical field for
        # this code (registry canonical == workbook field name, verified).
        canon = registry.get(code, {}).get("canonical_field", "")
        psf = str(rule.get("projected_source_field", "")) if has_rule else ""
        if canon and psf:
            ov = _tokens(canon) & _tokens(psf)
            sem_aligned = len(ov) / max(1, len(_tokens(canon) | _tokens(psf))) >= 0.34
        else:
            sem_aligned = True
        manual = False
        if not has_rule:
            status = "no_regime_rule"; manual = True
            msg = "workbook {LIST} field has no regime rule yet (enum backlog)"
        elif not sem_aligned:
            status = "semantic_mismatch"; manual = True
            msg = (f"regime rule source '{psf}' does not match the workbook field "
                   f"'{wb.get('field_name','')}' (registry canonical '{canon}') — "
                   "enum not constrained pending mapping review")
        elif outside:
            status = "targets_outside_workbook"; manual = True
            msg = f"enum_map targets not in the workbook allowed set: {outside}"
        elif emap:
            status = "constrained_within_workbook"
            msg = "enum_map constrains values to the workbook's allowed codes"
        else:
            status = "unconstrained_no_enum_map"; manual = True
            msg = "regime rule present but enum values are not constrained to the workbook set"
        rows.append({
            "esma_code": code,
            "workbook_field_name": wb.get("field_name", ""),
            "regime_source_field": psf,
            "has_regime_rule": has_rule,
            "has_enum_map": bool(emap),
            "workbook_allowed_count": len(allowed),
            "workbook_allowed_codes": "; ".join(allowed),
            "regime_enum_targets": "; ".join(targets),
            "targets_outside_workbook": "; ".join(outside),
            "regime_semantic_aligned": sem_aligned,
            "enum_coverage_status": status,
            "requires_manual_review": manual,
            "message": msg,
        })
    return rows


def annex2_enum_coverage_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["enum_coverage_status"]] = counts.get(r["enum_coverage_status"], 0) + 1
    return {
        "enum_rows_total": len(rows),
        "constrained_within_workbook": counts.get("constrained_within_workbook", 0),
        "unconstrained_no_enum_map": counts.get("unconstrained_no_enum_map", 0),
        "targets_outside_workbook": counts.get("targets_outside_workbook", 0),
        "semantic_mismatch": counts.get("semantic_mismatch", 0),
        "no_regime_rule": counts.get("no_regime_rule", 0),
        "requires_manual_review_count": sum(1 for r in rows if r["requires_manual_review"]),
        "enum_coverage_status_counts": counts,
    }


def write_annex2_enum_coverage_reconciliation(
    out_dir: str | Path,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    *,
    regime_config_source: str = "",
    universe_source: str = "",
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "46_annex2_enum_coverage_reconciliation.csv", rows,
               _ANNEX2_ENUM_COLUMNS)
    (out / "46_annex2_enum_coverage_reconciliation.json").write_text(
        json.dumps({"regime_config_source": regime_config_source,
                    "universe_source": universe_source,
                    "summary": summary, "rows": rows}, indent=2, default=str),
        encoding="utf-8")
    md = ["# ESMA Annex 2 enum-coverage reconciliation", "",
          "Compares each regime rule's `enum_map` against the authoritative "
          "workbook's allowed enum codes for every `{LIST}` field. Report only — "
          "no regime values were widened.", "",
          f"- **`{{LIST}}` fields:** {summary['enum_rows_total']}",
          f"- **Constrained to workbook codes:** {summary['constrained_within_workbook']}",
          f"- **Unconstrained (no enum_map, backlog):** {summary['unconstrained_no_enum_map']}",
          f"- **Targets outside workbook (risk):** {summary['targets_outside_workbook']}",
          f"- **Semantic mismatch (mapping review):** {summary['semantic_mismatch']}",
          f"- **No regime rule yet:** {summary['no_regime_rule']}", "",
          "## Status counts", ""]
    for st, c in sorted(summary["enum_coverage_status_counts"].items(), key=lambda kv: -kv[1]):
        md.append(f"- `{st}`: {c}")
    review = [r for r in rows if r["requires_manual_review"]]
    md += ["", "## Fields requiring manual review", ""]
    if review:
        for r in review:
            md.append(f"- `{r['esma_code']}` {r['workbook_field_name']} "
                      f"({r['enum_coverage_status']}): {r['message']}")
    else:
        md.append("_None._")
    (out / "46_annex2_enum_coverage_reconciliation_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    return {
        "csv": str(out / "46_annex2_enum_coverage_reconciliation.csv"),
        "json": str(out / "46_annex2_enum_coverage_reconciliation.json"),
        "summary_md": str(out / "46_annex2_enum_coverage_reconciliation_summary.md"),
    }


# ---------------------------------------------------------------------------
# Annex 2 semantic-mapping reconciliation (47) — regime source field vs workbook
# ---------------------------------------------------------------------------

_ANNEX2_SEMANTIC_COLUMNS = [
    "esma_code", "workbook_field_name", "registry_canonical_field",
    "regime_source_field", "regime_workbook_semantic", "token_overlap",
    "semantic_status", "requires_manual_review", "message",
]

# Below this token-overlap between the regime source field and the workbook
# field name (via the registry canonical), the rule likely targets a different
# field than the code denotes.
_ANNEX2_SEMANTIC_THRESHOLD = 0.34


def build_annex2_semantic_mapping_reconciliation(
    regime_config_path: Optional[str | Path] = None,
    universe_path: Optional[str | Path] = None,
    registry_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Check, for every regime-ruled Annex 2 code, whether the rule's
    ``projected_source_field`` actually corresponds to the workbook field for
    that code (via the verified registry canonical field). Report-only — does
    NOT change any rule; flags suspected code↔field mismaps for manual review.
    """
    cid, csrc, _ruled = load_annex2_target_contract(regime_config_path)
    try:
        regime = yaml.safe_load(Path(csrc).read_text(encoding="utf-8")) or {}
    except Exception:
        regime = {}
    field_rules = regime.get("field_rules", {}) or {}
    workbook, _src = load_annex2_workbook_universe(universe_path)
    registry = load_annex2_authoritative_universe(registry_path)

    rows: List[Dict[str, Any]] = []
    for code in sorted(set(field_rules) & set(workbook), key=_code_sort_key):
        wb = workbook[code]
        canon = registry.get(code, {}).get("canonical_field", "")
        rule = field_rules[code]
        psf = str(rule.get("projected_source_field", ""))
        sem = str(rule.get("workbook_semantic", ""))
        if canon and psf:
            ct, pt = _tokens(canon), _tokens(psf)
            overlap = len(ct & pt) / max(1, len(ct | pt))
        else:
            overlap = 1.0
        aligned = overlap >= _ANNEX2_SEMANTIC_THRESHOLD
        status = "aligned" if aligned else "semantic_mismatch"
        msg = ("regime source field matches the workbook field" if aligned else
               f"regime maps source '{psf}' to {code}, but the workbook field for "
               f"{code} is '{wb.get('field_name','')}' (registry canonical "
               f"'{canon}') — verify the code↔field mapping")
        rows.append({
            "esma_code": code,
            "workbook_field_name": wb.get("field_name", ""),
            "registry_canonical_field": canon,
            "regime_source_field": psf,
            "regime_workbook_semantic": sem,
            "token_overlap": round(overlap, 2),
            "semantic_status": status,
            "requires_manual_review": not aligned,
            "message": msg,
        })
    return rows


def annex2_semantic_mapping_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["semantic_status"]] = counts.get(r["semantic_status"], 0) + 1
    return {
        "semantic_rows_total": len(rows),
        "aligned": counts.get("aligned", 0),
        "semantic_mismatch": counts.get("semantic_mismatch", 0),
        "requires_manual_review_count": sum(1 for r in rows if r["requires_manual_review"]),
        "semantic_status_counts": counts,
    }


def write_annex2_semantic_mapping_reconciliation(
    out_dir: str | Path,
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    *,
    regime_config_source: str = "",
    registry_source: str = "",
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "47_annex2_semantic_mapping_reconciliation.csv", rows,
               _ANNEX2_SEMANTIC_COLUMNS)
    (out / "47_annex2_semantic_mapping_reconciliation.json").write_text(
        json.dumps({"regime_config_source": regime_config_source,
                    "registry_source": registry_source,
                    "summary": summary, "rows": rows}, indent=2, default=str),
        encoding="utf-8")
    md = ["# ESMA Annex 2 semantic-mapping reconciliation", "",
          "Checks whether each regime rule's source field matches the workbook "
          "field for that code (via the verified registry canonical name). "
          "Report only — no rules were changed. Mismatches likely indicate the "
          "regime config was authored against a different Annex 2 code layout and "
          "need a human mapping review.", "",
          f"- **Ruled codes checked:** {summary['semantic_rows_total']}",
          f"- **Aligned:** {summary['aligned']}",
          f"- **Semantic mismatch (manual review):** {summary['semantic_mismatch']}", "",
          "## Codes whose regime source does not match the workbook field", ""]
    mism = [r for r in rows if r["semantic_status"] == "semantic_mismatch"]
    if mism:
        for r in mism:
            md.append(f"- `{r['esma_code']}` workbook='{r['workbook_field_name']}' "
                      f"but regime source='{r['regime_source_field']}' "
                      f"(semantic '{r['regime_workbook_semantic']}')")
    else:
        md.append("_None — every regime rule maps the workbook's field for its code._")
    (out / "47_annex2_semantic_mapping_reconciliation_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    return {
        "csv": str(out / "47_annex2_semantic_mapping_reconciliation.csv"),
        "json": str(out / "47_annex2_semantic_mapping_reconciliation.json"),
        "summary_md": str(out / "47_annex2_semantic_mapping_reconciliation_summary.md"),
    }


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
        # Explicit non-ND regulatory value (e.g. RREC8 lien = "1"): the default
        # carries a concrete regulatory value, distinct from a configured/static
        # transform value.
        src = tf.get("default_rule_source", "")
        if src == "asset_config":
            status, coverage_basis = CONFIGURED_STATIC, "asset_config_static_value"
        else:
            status, coverage_basis = DEFAULTED_VALUE, "regime_default_value"
    elif has_config:
        status, coverage_basis = CONFIGURED_STATIC, "configured_transform"
    elif tf.get("deferred_reconciliation"):
        # Authoritative code explicitly deferred for reconciliation — included in
        # 28a (never dropped), not a blocking data gap.
        status, coverage_basis = DEFERRED, "deferred_reconciliation"
    elif tf.get("pending_regime_rule"):
        # In the authoritative Annex 2 universe but no full regime rule yet — a
        # config-completeness gap; surfaced, never silently dropped or blocked.
        status, coverage_basis = PENDING_REGIME_RULE, "no_regime_rule"
        decision_reason = ("authoritative Annex 2 code has no full regime field "
                           "rule yet (pending configuration)")
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

    # Asset-config default that conflicts with the regime envelope: surface a
    # non-blocking confirmation for mandatory fields (a valid regime default is
    # kept). Never silently apply the invalid asset default.
    if (not cands and tf.get("asset_default_invalid")
            and tf.get("config_validation_status") == VS_INVALID and required):
        requires_decision = True
        decision_reason = (tf.get("config_validation_message", "")
                           or "asset-config default not allowed by the regime rule")
        operator_question = (
            f"Asset default '{tf.get('asset_default_value', '')}' for "
            f"'{tf['target_field']}' is not allowed by the regime rule "
            f"(nd_allowed={tf.get('nd_allowed', [])}). Confirm the regime default "
            f"'{tf.get('default_value', '')}' or supply a valid value.")

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
        status = cls["coverage_status"]
        nd_applied = (status == DEFAULTED_ND
                      or (not cands and _is_nd(tf.get("default_value"))))
        # The value actually placed into the target field when there is no source.
        selected_value = ""
        if not cands and status in (DEFAULTED_ND, DEFAULTED_VALUE, CONFIGURED_STATIC):
            selected_value = tf.get("default_value", "")
        rows.append({
            "mode": mode,
            "target_contract_id": target_contract_id,
            "target_contract_source": target_contract_source,
            "target_field": tf["target_field"],
            "esma_code": tf.get("esma_code", ""),
            "projected_source_field": tf.get("projected_source_field", "") or tf.get("match_field", ""),
            "target_domain": tf["target_domain"],
            "target_label": tf["target_label"],
            "required_status": tf["required_status"],
            "applicability_status": ov.get("applicability_status", tf["applicability_status"]),
            "coverage_status": status,
            "coverage_basis": cls["coverage_basis"],
            "selected_source_file": primary.get("source_file", ""),
            "selected_source_sheet": primary.get("source_sheet", ""),
            "selected_source_column": primary.get("source_column", ""),
            "selected_source_confidence": primary.get("confidence", ""),
            "selected_value": selected_value,
            "alternative_source_candidates": "; ".join(
                f"{c['source_file']}::{c['source_sheet']}::{c['source_column']} "
                f"({c['confidence']})" for c in alts),
            "overlap_evidence": (
                f"{len(cands)} candidate source columns" if len(cands) > 1 else ""),
            "value_compatibility_status": cls["value_compatibility_status"],
            "derivation_rule": ov.get("derivation_rule", tf.get("derivation_rule", "")),
            "default_rule": ov.get("default_rule", tf.get("default_rule", "")),
            "default_value": tf.get("default_value", ""),
            "default_rule_source": tf.get("default_rule_source", ""),
            "default_reason": tf.get("default_reason", ""),
            "nd_allowed": "; ".join(tf.get("nd_allowed", [])),
            "nd_rule_applied": ("; ".join(tf.get("nd_allowed", [])) if nd_applied else ""),
            "configured_value_source": ov.get("configured_value_source",
                                              tf.get("configured_value_source", "")),
            "config_validation_status": tf.get("config_validation_status", ""),
            "config_validation_message": tf.get("config_validation_message", ""),
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
             evidence_summary, esma_code=""):
        nonlocal seq
        seq += 1
        decisions.append({
            "decision_id": f"DQ-{mode}-{seq:03d}",
            "decision_type": decision_type,
            "priority": priority,
            "mode": mode,
            "target_contract_id": coverage_rows[0]["target_contract_id"] if coverage_rows else "",
            "target_field": target_field,
            "esma_code": esma_code,
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
        ecode = cov.get("esma_code", "")
        invalid_default = cov.get("config_validation_status") == VS_INVALID
        if status == MISSING_REQUIRED:
            _add(D_MISSING, "high", cov["target_field"], "", "",
                 cov["decision_reason"] or "required target field unmapped",
                 "Provide a source column, a derivation/default, or confirm an ND code.",
                 ["map_source_column", "set_derivation_or_default", "confirm_ND_code",
                  "mark_not_applicable"],
                 True, cov["operator_question"],
                 f"required={cov['required_status']}; domain={cov['target_domain']}", ecode)
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
                 cov["overlap_evidence"] + "; alts: " + cov["alternative_source_candidates"], ecode)
        elif cov["requires_user_decision"] and invalid_default:
            # Asset-config default outside the regime envelope: non-blocking
            # confirmation (a valid regime fallback is kept), unless the field is
            # also a hard blocker (no valid fallback).
            _add(D_INVALID_DEFAULT, "high" if cov["blocking"] else "medium",
                 cov["target_field"], cov["selected_source_file"],
                 cov["selected_source_column"],
                 cov["decision_reason"] or "asset default not allowed by regime rule",
                 "Confirm the regime default, or supply a value within the allowed envelope.",
                 ["confirm_default_or_nd", "configure_static_value", "mark_not_applicable"],
                 bool(cov["blocking"]), cov["operator_question"],
                 f"asset_default={cov.get('asset_default_value','')}; "
                 f"nd_allowed={cov.get('nd_allowed','')}; regime_default={cov.get('default_value','')}",
                 ecode)
        elif cov["requires_user_decision"] and cov["blocking"]:
            _add(D_CONFIG, "high", cov["target_field"], cov["selected_source_file"],
                 cov["selected_source_column"], cov["decision_reason"],
                 "Supply the required config / static value.",
                 ["set_config_value", "confirm_ND_code", "mark_not_applicable"],
                 True, cov["operator_question"],
                 f"status={status}; basis={cov['coverage_basis']}", ecode)
        elif cov["requires_user_decision"]:
            # Non-blocking operator confirmation requested by the overlay/config.
            dtype = D_ND if status == DEFAULTED_ND else D_CONFIG
            _add(dtype, "medium", cov["target_field"], cov["selected_source_file"],
                 cov["selected_source_column"], cov["decision_reason"],
                 "Confirm the configured / default / ND value.",
                 ["confirm_value", "provide_source", "mark_not_applicable"],
                 False, cov["operator_question"],
                 f"status={status}; basis={cov['coverage_basis']}", ecode)

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
                                + status_counts.get(DEFAULTED_VALUE, 0)
                                + status_counts.get(DEFAULTED_ND, 0))
    return {
        "target_fields_total": len(coverage_rows),
        "coverage_status_counts": status_counts,
        "source_mapped_fields": source_mapped,
        "derived_config_defaulted_fields": derived_config_defaulted,
        "derived_fields": status_counts.get(DERIVED, 0),
        "defaulted_nd_fields": status_counts.get(DEFAULTED_ND, 0),
        "defaulted_value_fields": status_counts.get(DEFAULTED_VALUE, 0),
        "configured_static_fields": status_counts.get(CONFIGURED_STATIC, 0),
        "missing_required_fields": status_counts.get(MISSING_REQUIRED, 0),
        "needs_confirmation_fields": status_counts.get(NEEDS_CONFIRMATION, 0),
        "not_applicable_fields": status_counts.get(NOT_APPLICABLE, 0),
        "optional_for_mi_fields": status_counts.get(OPTIONAL_FOR_MI, 0),
        "pending_regime_rule_fields": status_counts.get(PENDING_REGIME_RULE, 0),
        "deferred_fields": status_counts.get(DEFERRED, 0),
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
    "esma_code", "projected_source_field",
    "target_domain", "target_label", "required_status", "applicability_status",
    "coverage_status", "coverage_basis", "selected_source_file",
    "selected_source_sheet", "selected_source_column", "selected_source_confidence",
    "selected_value", "alternative_source_candidates", "overlap_evidence",
    "value_compatibility_status", "derivation_rule", "default_rule", "default_value",
    "default_rule_source", "default_reason", "nd_allowed", "nd_rule_applied",
    "configured_value_source", "config_validation_status", "config_validation_message",
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
    "target_field", "esma_code", "source_file", "source_column", "issue",
    "recommendation", "options", "blocking", "operator_question", "evidence_summary",
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
    asset_config_path: Optional[str | Path] = None,
    registry_path: Optional[str | Path] = None,
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

    # Annex 2 loads the FULL authoritative target universe (workbook registry ∪
    # regime field_rules ∪ deferred); MI keeps the MI semantics registry.
    universe_meta: Optional[Dict[str, Any]] = None
    universe_warnings: List[str] = []
    if target_contract_kind(mode, context) == "esma_annex_2":
        cid, csrc, target_fields, universe_meta = build_annex2_full_contract(
            annex2_config_path=annex2_config_path, registry_path=registry_path)
        universe_warnings = list(universe_meta.get("warnings", []) or [])
    else:
        cid, csrc, target_fields = load_target_contract(
            mode, context, mi_registry_path=mi_registry_path,
            annex2_config_path=annex2_config_path)

    # --- LAYER 2: asset-class config (Annex 2 only) ---
    # Validate the ERM asset defaults against the regime envelope, apply the
    # valid ones (asset-chosen defaults override regime defaults) and surface
    # any conflict — never silently apply an invalid default.
    config_validation: Optional[Dict[str, Any]] = None
    if cid == "esma_annex_2":
        asset_path = Path(asset_config_path) if asset_config_path else _ASSET_CONFIG_DEFAULT
        if Path(asset_path).exists():
            val_rows, asset_overlay, asset_src = build_annex2_config_validation(
                csrc, asset_path)
            apply_asset_overlay(target_fields, asset_overlay)
            config_validation = {
                "rows": val_rows,
                "summary": annex2_validation_summary(val_rows),
                "regime_config_source": csrc,
                "asset_config_source": asset_src,
            }

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

    # 42 — Annex 2 regime/asset config validation (config-driven, source
    # independent). Written only for the Annex 2 target contract.
    if config_validation is not None:
        v_paths = write_annex2_config_validation(
            out, config_validation["rows"],
            regime_config_source=config_validation["regime_config_source"],
            asset_config_source=config_validation["asset_config_source"])
        config_validation["paths"] = v_paths
        paths.update({f"config_validation_{k}": v for k, v in v_paths.items()})

    # 43 — Annex 2 field-universe reconciliation (workbook vs regime vs
    # validation vs 28a coverage). Written only for the Annex 2 target contract.
    field_universe: Optional[Dict[str, Any]] = None
    if universe_meta is not None:
        recon_rows = build_annex2_field_universe_reconciliation(
            universe_meta, coverage_rows,
            (config_validation or {}).get("rows", []))
        recon_sum = annex2_reconciliation_summary(
            recon_rows, universe_meta, coverage_rows)
        r_paths = write_annex2_field_universe_reconciliation(
            out, recon_rows, recon_sum,
            registry_source=universe_meta.get("registry_source", ""),
            regime_config_source=csrc, warnings=universe_warnings)
        field_universe = {"rows": recon_rows, "summary": recon_sum,
                          "warnings": universe_warnings, "paths": r_paths,
                          "universe_meta": universe_meta}
        paths.update({f"field_universe_{k}": v for k, v in r_paths.items()})

        # 44 — ND-eligibility reconciliation (regime nd_allowed vs workbook).
        nd_rows = build_annex2_nd_eligibility_reconciliation(
            regime_config_path=annex2_config_path)
        nd_sum = annex2_nd_eligibility_summary(nd_rows)
        nd_paths = write_annex2_nd_eligibility_reconciliation(
            out, nd_rows, nd_sum, regime_config_source=csrc,
            universe_source=universe_meta.get("authoritative_source", ""))
        field_universe["nd_eligibility"] = {"rows": nd_rows, "summary": nd_sum,
                                            "paths": nd_paths}
        paths.update({f"nd_eligibility_{k}": v for k, v in nd_paths.items()})

        # 45 — config-alignment review (actions taken + manual-review items).
        align_rows = build_annex2_config_alignment_review(
            regime_config_path=annex2_config_path,
            asset_config_path=asset_config_path,
            registry_path=registry_path)
        align_sum = annex2_config_alignment_summary(align_rows)
        align_paths = write_annex2_config_alignment_review(
            out, align_rows, align_sum, regime_config_source=csrc,
            registry_source=str(registry_path or ""),
            asset_config_source=str(asset_config_path or ""))
        field_universe["config_alignment"] = {"rows": align_rows, "summary": align_sum,
                                              "paths": align_paths}
        paths.update({f"config_alignment_{k}": v for k, v in align_paths.items()})

        # 46 — enum-coverage reconciliation (regime enum_map vs workbook codes).
        enum_rows = build_annex2_enum_coverage_reconciliation(
            regime_config_path=annex2_config_path, registry_path=registry_path)
        enum_sum = annex2_enum_coverage_summary(enum_rows)
        enum_paths = write_annex2_enum_coverage_reconciliation(
            out, enum_rows, enum_sum, regime_config_source=csrc,
            universe_source=universe_meta.get("authoritative_source", ""))
        field_universe["enum_coverage"] = {"rows": enum_rows, "summary": enum_sum,
                                           "paths": enum_paths}
        paths.update({f"enum_coverage_{k}": v for k, v in enum_paths.items()})

        # 47 — semantic-mapping reconciliation (regime source field vs workbook).
        sem_rows = build_annex2_semantic_mapping_reconciliation(
            regime_config_path=annex2_config_path, registry_path=registry_path)
        sem_sum = annex2_semantic_mapping_summary(sem_rows)
        sem_paths = write_annex2_semantic_mapping_reconciliation(
            out, sem_rows, sem_sum, regime_config_source=csrc,
            registry_source=str(registry_path or ""))
        field_universe["semantic_mapping"] = {"rows": sem_rows, "summary": sem_sum,
                                              "paths": sem_paths}
        paths.update({f"semantic_mapping_{k}": v for k, v in sem_paths.items()})

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
        "config_validation": config_validation,
        "field_universe": field_universe,
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
