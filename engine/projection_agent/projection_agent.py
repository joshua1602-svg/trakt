"""
projection_agent.py
===================

Trakt Projection Agent v1 — the stage after Validation.

It consumes the Validation Agent package (40 manifest, 41 results, 42 readiness,
43 issues, 44 lineage, 46 projection-blocker diagnostics) + the transformed
canonical tape (31) and the transformation field contract (32), and emits a
governed **projection package** under ``output/projection/`` (artefacts 50..56):
a long, explicit Annex 2 *target frame* plus projection readiness, issues and a
blocker-resolution report.

Guardrails (enforced by construction):
  * never re-runs Gate 1 / Transformation / Validation, never mutates their
    artefacts (writes only under ``output/projection/``);
  * never invokes the Gate 5 XML builder, never produces XML, never claims XML
    readiness (``ready_for_xml_delivery`` is always ``False``);
  * never runs Gate 4b delivery normalisation (precision / regex / boolean-XSD /
    preflight) — that is deferred to the Delivery Agent;
  * never invents ND-values / defaults / source mappings — unresolved items are
    carried forward with an explicit owner + recommended action.

It reuses the frozen Gate 4 ESMA-code ordering primitives and the authoritative
``annex2_delivery_rules.yaml`` regime contract through
:mod:`engine.projection_agent.gate4_adapter`.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from engine.projection_agent import gate4_adapter as g4
from engine.validation_agent.projection_blocker_diagnostics import (
    PB_MATERIALISED,
    PB_NOT_MATERIALISED,
    PB_ND_OR_DEFAULT,
    PB_SOURCE_MAPPING,
    PB_OP_CONFIG_DEP,
    PB_UNKNOWN,
)

# --------------------------------------------------------------------------- #
# Identity / vocabulary
# --------------------------------------------------------------------------- #

AGENT = "projection_agent"
AGENT_VERSION = "1.0"
STAGE = "post_validation_projection"

REQUIRED_VAL_AGENT = "validation_agent"

NEXT_DELIVERY = "delivery_normalisation"
NEXT_REMEDIATION = "operator_config_projection_remediation"

# projection_status vocabulary.
ST_FROM_TRANSFORMED = "projected_from_transformed"
ST_ND_DEFAULT = "projected_nd_default"
ST_ASSET_DEFAULT = "projected_asset_default"
ST_UNRESOLVED_SOURCE = "unresolved_source_mapping"
ST_BLOCKED_OP_CONFIG = "blocked_operator_or_config_dependency"
ST_UNRESOLVED_NOT_MATERIALISED = "unresolved_not_materialised"
ST_UNRESOLVED_ND = "invalid_nd_for_field"
ST_CARRIED_BLANK = "not_projected_blank"
ST_BLOCKED_CLIENT = "blocked_client_onboarding_dependency"

# downstream owners.
OWN_PROJECTION = "projection"
OWN_CONFIG = "config_policy"
OWN_OPERATOR = "operator"
OWN_TRANSFORMATION = "transformation_validation"
OWN_DELIVERY = "delivery_xml"

# projection issue types (per task spec).
IT_OPERATOR = "operator_dependency_unresolved"
IT_CONFIG = "config_dependency_unresolved"
IT_SOURCE_MAPPING = "source_mapping_unresolved"
IT_ND_DEFAULT_MISSING = "nd_default_rule_missing"
IT_INVALID_ND = "invalid_nd_for_field"
IT_INVALID_VALUE = "invalid_projected_value"
IT_FIELD_UNIVERSE = "field_universe_missing"
IT_LEGACY = "legacy_projector_incompatible"
IT_DELIVERY_DEFERRED = "delivery_structure_deferred"
IT_CLIENT_ONBOARDING = "client_onboarding_dependency_unresolved"

# Onboarding dispositions (carried via the transformation contract) that the
# Projection Agent must execute coherently rather than rediscovering as generic
# source-mapping gaps.
DISP_CLIENT_ONBOARDING = "client_onboarding_required"
DISP_FORMAL_IDENTIFIER = "formal_identifier_policy_required"
DISP_CONFIG_MAPPING = "config_mapping_required"
DISP_ASSET_POLICY = "asset_policy_required"
DISP_OPERATOR_REVIEW = "operator_review_required"
DISP_SOURCE_MAPPED_REVIEW = "source_mapped_with_review"
_DISP_CLIENT = {DISP_CLIENT_ONBOARDING, DISP_FORMAL_IDENTIFIER}
_DISP_CONFIG = {DISP_CONFIG_MAPPING, DISP_ASSET_POLICY}
# A field the onboarding agent flagged for operator review (e.g. an ambiguous
# valuation/rate source) must NOT be auto-resolved by applying a blind ND/default
# — it is carried as an operator dependency until the operator confirms.
_DISP_OPERATOR = {DISP_OPERATOR_REVIEW, DISP_SOURCE_MAPPED_REVIEW}

# Onboarding dispositions for which the TARGET FRAME must NOT be filled with a
# generic ND/default (or an unconfirmed source value). The field is carried as
# unresolved per its disposition instead of being silently defaulted.
DISP_CLIENT_POLICY_REQUIRED = "client_policy_required"
_DISP_SUPPRESS_FRAME_DEFAULT = (
    _DISP_CLIENT | _DISP_CONFIG | _DISP_OPERATOR | {DISP_CLIENT_POLICY_REQUIRED}
)


def _frame_status_for_disposition(disposition: str) -> str:
    """Map a suppressed onboarding disposition to a blocked target-frame status."""
    if disposition in _DISP_CLIENT:
        return ST_BLOCKED_CLIENT
    return ST_BLOCKED_OP_CONFIG

_FRAME_COLUMNS = [
    "row_id", "loan_identifier", "record_group", "esma_code", "canonical_field",
    "projected_value", "projected_value_type", "value_source", "projection_status",
    "nd_applied", "default_applied", "source_field", "source_value_sample",
    "rule_id", "blocking_for_delivery",
]

_CONTRACT_COLUMNS = [
    "esma_code", "canonical_field", "record_group", "mandatory", "enforce_presence",
    "nd_allowed", "default_allowed", "deferred", "rows_total",
    "rows_projected_from_transformed", "rows_nd_default", "rows_asset_default",
    "rows_unresolved", "field_projection_status", "blocking_for_delivery", "notes",
]

_ISSUE_COLUMNS = [
    "issue_id", "source_issue_id", "esma_code", "canonical_field", "record_group",
    "issue_type", "projection_status", "severity", "blocking_for_delivery",
    "blocking_for_xml_delivery", "recommended_action", "downstream_owner", "description",
]

_RESOLUTION_COLUMNS = [
    "validation_issue_id", "esma_code", "canonical_field", "diagnostic_subtype",
    "onboarding_disposition", "projection_action", "projection_status", "resolved",
    "resolution_source", "projected_value_sample", "remaining_issue_id", "notes",
]


class ValidationHandoffError(RuntimeError):
    """Raised when the validation package is missing or not consumable by projection."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_yaml(path: Path) -> Optional[dict]:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _to_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        if isinstance(v, float) and math.isnan(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    if s.lower() in ("nan", "<na>"):
        return ""
    return s


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "y")


# --------------------------------------------------------------------------- #
# Validate the validation manifest
# --------------------------------------------------------------------------- #

def validate_validation_manifest(manifest: dict) -> None:
    """Fail loudly unless the validation package was produced by the Validation
    Agent and is in a state the Projection Agent can consume.

    Required:
        agent == validation_agent (or equivalent);
        ready_for_xml_delivery == false;
        governance flags (performed_projection / performed_xml_delivery) respected.

    Note: ``ready_for_projection == false`` is **allowed** — the Projection Agent
    is explicitly built to consume a package that still carries unresolved
    projection blockers and to reduce / carry them forward.
    """
    if not isinstance(manifest, dict):
        raise ValidationHandoffError("Validation manifest is not a JSON object.")

    problems: List[str] = []
    agent = str(manifest.get("agent", "")).lower()
    if REQUIRED_VAL_AGENT not in agent and "validation" not in agent:
        problems.append(f"agent must be a validation agent, got {manifest.get('agent')!r}")
    if manifest.get("ready_for_xml_delivery") is True:
        problems.append("ready_for_xml_delivery must be false at the projection stage")
    if manifest.get("performed_xml_delivery") is True:
        problems.append("validation package must not have performed XML delivery")
    if manifest.get("consumes_transformation_package") is False:
        problems.append("validation package must consume the transformation package")
    if problems:
        raise ValidationHandoffError(
            "Validation package is not consumable by the Projection Agent:\n  - "
            + "\n  - ".join(problems))


def _resolve_inputs(manifest_path: Path) -> Dict[str, Path]:
    val_dir = manifest_path.parent                 # .../output/validation
    output_root = val_dir.parent                   # .../output
    tx_dir = output_root / "transformation"
    return {
        "val_dir": val_dir,
        "output_root": output_root,
        "tx_dir": tx_dir,
        "validation_results": val_dir / "41_validation_results.csv",
        "validation_readiness": val_dir / "42_validation_readiness.json",
        "validation_issues": val_dir / "43_validation_issues.csv",
        "validation_lineage": val_dir / "44_validation_lineage.json",
        "blocker_diagnostics": val_dir / "46_projection_blocker_diagnostics.csv",
        "transformed_tape": tx_dir / "31_transformed_canonical_tape.csv",
        "tx_contract": tx_dir / "32_transformation_field_contract.csv",
        "tx_lineage": tx_dir / "34_transformation_lineage.json",
        "handoff_manifest": output_root / "handoff" / "24_onboarding_handoff_manifest.json",
    }


# --------------------------------------------------------------------------- #
# Per-cell projection
# --------------------------------------------------------------------------- #

def _project_cell(
    raw: Any,
    rule: Dict[str, Any],
    asset_defaults: Dict[str, Any],
    asset_nd_defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Project a single (loan, field) cell into an Annex 2 value, conservatively.

    Never invents a value: ND/defaults are applied only where explicitly allowed
    by the regime rule or configured in the asset config; enum values are mapped
    only via the rule's explicit ``enum_map`` / ``geography_map``.
    """
    canonical = rule["canonical_field"]
    nd_allowed = rule["nd_allowed"]
    res: Dict[str, Any] = {
        "projected_value": "", "projected_value_type": "blank", "value_source": "none",
        "projection_status": ST_CARRIED_BLANK, "nd_applied": False,
        "default_applied": False, "blocking_for_delivery": False, "problem": None,
    }

    v = _to_str(raw)

    # (1) materialised value present in the transformed tape.
    if v != "":
        pv, vtype, mapped = g4.apply_safe_transform(v, rule)
        res.update(
            projected_value=pv, projected_value_type=vtype,
            value_source=("enum_map" if mapped else "transformed_tape"),
            projection_status=ST_FROM_TRANSFORMED,
        )
        if vtype == "unmapped_enum":
            res["blocking_for_delivery"] = True
            res["problem"] = {
                "issue_type": IT_SOURCE_MAPPING, "severity": "warn",
                "owner": OWN_CONFIG, "status": ST_UNRESOLVED_SOURCE,
                "action": "add an explicit enum mapping for this materialised value",
                "desc": f"materialised value '{v}' not present in enum_map for {rule['esma_code']}",
            }
        return res

    # (2) field is blank/absent — try an explicitly-allowed ND/default only.

    # 2a) asset config nd_defaults (e.g. primary_income: ND1).
    if canonical in asset_nd_defaults:
        ndv = _to_str(asset_nd_defaults[canonical]).upper()
        if g4.is_nd_value(ndv):
            if ndv in nd_allowed:
                res.update(
                    projected_value=ndv, projected_value_type="nd",
                    value_source="asset_nd_default", projection_status=ST_ND_DEFAULT,
                    nd_applied=True)
            else:
                res.update(
                    projection_status=ST_UNRESOLVED_ND, blocking_for_delivery=True,
                    problem={
                        "issue_type": IT_INVALID_ND, "severity": "error",
                        "owner": OWN_CONFIG, "status": ST_UNRESOLVED_ND,
                        "action": f"asset ND default {ndv} not allowed for {rule['esma_code']}; fix config",
                        "desc": f"asset nd_default {ndv} not in nd_allowed {nd_allowed}",
                    })
            return res
        # A non-ND value sitting under nd_defaults — treat it as a static default.
        return _apply_static_default(res, _to_str(asset_nd_defaults[canonical]),
                                     rule, "asset_default")

    # 2b) asset config static defaults (e.g. exposure_currency_denomination: GBP).
    if canonical in asset_defaults:
        return _apply_static_default(res, _to_str(asset_defaults[canonical]),
                                     rule, "asset_default")

    # 2c) regime rule default.
    if rule["default_allowed"] and rule["default_value"]:
        dv = _to_str(rule["default_value"])
        if g4.is_nd_value(dv):
            if dv.upper() in nd_allowed:
                res.update(
                    projected_value=dv.upper(), projected_value_type="nd",
                    value_source="regime_default_nd", projection_status=ST_ND_DEFAULT,
                    nd_applied=True)
            else:
                res.update(
                    projection_status=ST_UNRESOLVED_ND, blocking_for_delivery=True,
                    problem={
                        "issue_type": IT_INVALID_ND, "severity": "error",
                        "owner": OWN_CONFIG, "status": ST_UNRESOLVED_ND,
                        "action": f"regime default {dv} not allowed for {rule['esma_code']}; fix config",
                        "desc": f"regime default_value {dv} not in nd_allowed {nd_allowed}",
                    })
            return res
        return _apply_static_default(res, dv, rule, "regime_default")

    # (3) nothing materialised and no allowed ND/default.
    if rule["mandatory"] and rule["enforce_presence"]:
        res.update(
            projection_status=ST_UNRESOLVED_NOT_MATERIALISED, blocking_for_delivery=True,
            problem={
                "issue_type": IT_ND_DEFAULT_MISSING, "severity": "warn",
                "owner": OWN_PROJECTION, "status": ST_UNRESOLVED_NOT_MATERIALISED,
                "action": "supply a source value or configure an allowed ND/default",
                "desc": f"mandatory field {rule['esma_code']} absent/blank with no allowed ND/default",
            })
        return res

    # non-mandatory blank — carried, not blocking.
    return res


def _apply_static_default(
    res: Dict[str, Any], default_val: str, rule: Dict[str, Any], source: str
) -> Dict[str, Any]:
    """Apply a non-ND configured default value through the safe transform."""
    pv, vtype, _ = g4.apply_safe_transform(default_val, rule)
    if vtype == "unmapped_enum":
        res.update(
            projection_status=ST_UNRESOLVED_SOURCE, blocking_for_delivery=True,
            problem={
                "issue_type": IT_SOURCE_MAPPING, "severity": "warn",
                "owner": OWN_CONFIG, "status": ST_UNRESOLVED_SOURCE,
                "action": "add an enum mapping for the configured default value",
                "desc": f"configured default '{default_val}' not mappable for {rule['esma_code']}",
            })
        return res
    res.update(
        projected_value=pv, projected_value_type=vtype, value_source=source,
        projection_status=ST_ASSET_DEFAULT, default_applied=True)
    return res


# --------------------------------------------------------------------------- #
# Core build
# --------------------------------------------------------------------------- #

def build_projection_package(
    validation_manifest_path: str | Path,
    *,
    registry_path: str = "",
    regime_config_path: str = "",
    asset_config_path: str = "",
    esma_code_order_path: str = "",
) -> Dict[str, Any]:
    """Consume the validation package and emit the projection package.

    Returns a dict of artefact paths + the projection manifest. Raises
    :class:`ValidationHandoffError` if the validation manifest is missing/invalid
    or not consumable.
    """
    manifest_path = Path(validation_manifest_path)
    if not manifest_path.exists():
        raise ValidationHandoffError(f"Validation manifest not found: {manifest_path}")
    val_manifest = _read_json(manifest_path)
    if val_manifest is None:
        raise ValidationHandoffError(f"Validation manifest is not valid JSON: {manifest_path}")

    # 1) validate readiness / governance.
    validate_validation_manifest(val_manifest)

    paths = _resolve_inputs(manifest_path)
    client_id = val_manifest.get("client_id", "")
    run_id = val_manifest.get("run_id", "")
    target_contract_id = val_manifest.get("target_contract_id", "")

    repo_root = Path(__file__).resolve().parents[2]
    regime_config_path = regime_config_path or val_manifest.get("regime_config_path", "") or str(
        repo_root / "config" / "regime" / "annex2_delivery_rules.yaml")
    asset_config_path = asset_config_path or val_manifest.get("asset_config_path", "") or str(
        repo_root / "config" / "asset" / "product_defaults_ERM.yaml")
    registry_path = registry_path or val_manifest.get("registry_path", "") or str(
        repo_root / "config" / "system" / "fields_registry.yaml")
    esma_code_order_path = esma_code_order_path or str(
        repo_root / "config" / "system" / "esma_code_order.yaml")

    # 2) load the transformed canonical tape (NO source discovery / re-run).
    if not paths["transformed_tape"].exists():
        raise ValidationHandoffError(
            f"Transformed canonical tape not found at {paths['transformed_tape']}")
    df = pd.read_csv(paths["transformed_tape"], dtype=str, low_memory=False).fillna("")
    row_count = int(len(df))

    # 3) load validation issues + projection blocker diagnostics.
    val_issues = _read_csv_rows(paths["validation_issues"])
    blocker_rows = _read_csv_rows(paths["blocker_diagnostics"])

    # Onboarding target-field dispositions, carried through the transformation
    # field contract (32). The Projection Agent EXECUTES these dispositions —
    # e.g. a client_onboarding_required field is carried as a client/onboarding
    # blocker, not rediscovered as a generic source-mapping gap.
    tx_contract_rows = _read_csv_rows(paths["tx_contract"])
    disposition_by_field: Dict[str, str] = {}
    for r in tx_contract_rows:
        disp = _to_str(r.get("field_disposition"))
        if not disp:
            continue
        for key in (r.get("canonical_field"), r.get("esma_code")):
            key = _to_str(key)
            if key and key not in disposition_by_field:
                disposition_by_field[key] = disp

    # 4) regime / asset / ordering config.
    regime_cfg = g4.load_regime_rules(regime_config_path)
    proj_index = g4.build_projection_index(regime_cfg)
    asset_cfg = _read_yaml(Path(asset_config_path)) or {}
    asset_defaults = (asset_cfg.get("defaults") or {}) if isinstance(asset_cfg, dict) else {}
    asset_nd_defaults = (asset_cfg.get("nd_defaults") or {}) if isinstance(asset_cfg, dict) else {}

    # 4b) Supplement the regime field set with target fields that are present in
    #     the transformation contract (32) but lack a full regime field_rules entry
    #     (e.g. RREL24 maturity_date, mapped only via the registry). Include them so
    #     an approved/materialised value is not silently dropped from the target
    #     frame. We only add a code that has a canonical field and either a
    #     materialised value in the transformed tape or an onboarding disposition.
    for r in tx_contract_rows:
        code = _to_str(r.get("esma_code"))
        canonical = _to_str(r.get("canonical_field"))
        if not code or code in proj_index or not canonical:
            continue
        has_value = canonical in df.columns and df[canonical].map(
            lambda v: _to_str(v) != "").any()
        if not (has_value or disposition_by_field.get(canonical)
                or disposition_by_field.get(code)):
            continue
        proj_index[code] = {
            "esma_code": code, "canonical_field": canonical,
            "mandatory": False, "enforce_presence": False, "nd_allowed": [],
            "default_allowed": False, "default_value": "",
            "enum_map": {}, "geography_map": {}, "boolean": "", "regex": "",
            "precision": {}, "deferred": False, "supplementary": True,
        }

    record_order = g4.load_record_order(esma_code_order_path)
    ordered_codes = g4.order_esma_codes(list(proj_index.keys()), record_order)

    # 5) build the projected Annex 2 target frame (one row per loan x field).
    #    Honour the onboarding disposition so a field flagged for operator/config/
    #    client review is NOT filled with a generic ND/default.
    frame_rows, field_summary, field_problems = _build_target_frame(
        df, proj_index, ordered_codes, asset_defaults, asset_nd_defaults,
        disposition_by_field=disposition_by_field)

    # 6) build the blocker-resolution report from 46 + the field summary.
    resolution_rows, issues = _resolve_blockers(
        blocker_rows, field_summary, field_problems, proj_index,
        disposition_by_field=disposition_by_field)

    # 7) readiness.
    counts = _count(frame_rows, issues, resolution_rows)
    readiness = compute_readiness(
        projection_ran=True,
        frame_nonempty=len(frame_rows) > 0,
        blocking_for_delivery_issue_count=counts["blocking_for_delivery_issue_count"],
        required_fields_unresolved_count=counts["required_fields_unresolved_count"],
        remaining_blocker_count=counts["remaining_blocker_count"],
    )

    # 8) write artefacts.
    out_dir = paths["output_root"] / "projection"
    out_dir.mkdir(parents=True, exist_ok=True)
    return _write_artefacts(
        out_dir=out_dir, frame_rows=frame_rows, field_summary=field_summary,
        ordered_codes=ordered_codes, proj_index=proj_index, issues=issues,
        resolution_rows=resolution_rows, readiness=readiness, counts=counts,
        val_manifest=val_manifest, manifest_path=manifest_path, paths=paths,
        client_id=client_id, run_id=run_id, target_contract_id=target_contract_id,
        row_count=row_count, blocker_rows=blocker_rows,
        config_paths={
            "registry_path": registry_path,
            "regime_config_path": regime_config_path,
            "asset_config_path": asset_config_path,
            "esma_code_order_path": esma_code_order_path,
        })


# --------------------------------------------------------------------------- #
# Target frame + field summary
# --------------------------------------------------------------------------- #

def _loan_identifier(df: pd.DataFrame, idx: int) -> str:
    for col in ("loan_identifier", "unique_identifier", "original_underlying_exposure_identifier"):
        if col in df.columns:
            v = _to_str(df.at[idx, col])
            if v:
                return v
    return f"row_{idx}"


def _build_target_frame(
    df: pd.DataFrame,
    proj_index: Dict[str, Dict[str, Any]],
    ordered_codes: List[str],
    asset_defaults: Dict[str, Any],
    asset_nd_defaults: Dict[str, Any],
    disposition_by_field: Optional[Dict[str, str]] = None,
):
    """Return (frame_rows, field_summary, field_problems).

    Honours the onboarding disposition: for fields the onboarding agent flagged as
    operator/config/client-policy/formal-identifier unresolved, the target frame is
    NOT populated with a generic ND/default (or an unconfirmed source value) — the
    cell is carried as a blocked dependency instead.
    """
    disposition_by_field = disposition_by_field or {}
    frame_rows: List[Dict[str, Any]] = []
    # field_summary[esma_code] = aggregate counts + representative sample/status.
    field_summary: Dict[str, Dict[str, Any]] = {}
    # field_problems[esma_code] = first problem dict seen for that field.
    field_problems: Dict[str, Dict[str, Any]] = {}

    for code in ordered_codes:
        rule = proj_index[code]
        field_summary[code] = {
            "esma_code": code, "canonical_field": rule["canonical_field"],
            "record_group": g4.record_group_for_code(code),
            "mandatory": rule["mandatory"], "enforce_presence": rule["enforce_presence"],
            "nd_allowed": rule["nd_allowed"], "default_allowed": rule["default_allowed"],
            "deferred": rule["deferred"], "rows_total": 0,
            "rows_projected_from_transformed": 0, "rows_nd_default": 0,
            "rows_asset_default": 0, "rows_unresolved": 0, "blocking": False,
            "sample": "",
        }

    rid = 0
    for idx in range(len(df)):
        loan_id = _loan_identifier(df, idx)
        for code in ordered_codes:
            rule = proj_index[code]
            canonical = rule["canonical_field"]
            raw = df.at[idx, canonical] if canonical in df.columns else ""
            disposition = (disposition_by_field.get(canonical)
                           or disposition_by_field.get(code) or "")
            if disposition in _DISP_SUPPRESS_FRAME_DEFAULT:
                # Do NOT default/project — carry as a blocked dependency per the
                # onboarding disposition (e.g. RREC17 operator_review must not be
                # silently filled with ND1).
                cell = {
                    "projected_value": "", "projected_value_type": "blocked",
                    "value_source": "onboarding_disposition",
                    "projection_status": _frame_status_for_disposition(disposition),
                    "nd_applied": False, "default_applied": False,
                    "blocking_for_delivery": True, "problem": None,
                }
            else:
                cell = _project_cell(raw, rule, asset_defaults, asset_nd_defaults)

            rid += 1
            frame_rows.append({
                "row_id": f"PR-{rid:06d}",
                "loan_identifier": loan_id,
                "record_group": g4.record_group_for_code(code),
                "esma_code": code,
                "canonical_field": canonical,
                "projected_value": cell["projected_value"],
                "projected_value_type": cell["projected_value_type"],
                "value_source": cell["value_source"],
                "projection_status": cell["projection_status"],
                "nd_applied": cell["nd_applied"],
                "default_applied": cell["default_applied"],
                "source_field": canonical,
                "source_value_sample": _to_str(raw)[:64],
                "rule_id": code,
                "blocking_for_delivery": cell["blocking_for_delivery"],
            })

            # update field summary.
            s = field_summary[code]
            s["rows_total"] += 1
            st = cell["projection_status"]
            if st == ST_FROM_TRANSFORMED and not cell["blocking_for_delivery"]:
                s["rows_projected_from_transformed"] += 1
            elif st == ST_ND_DEFAULT:
                s["rows_nd_default"] += 1
            elif st == ST_ASSET_DEFAULT:
                s["rows_asset_default"] += 1
            elif cell["blocking_for_delivery"] or st in (
                ST_UNRESOLVED_SOURCE, ST_UNRESOLVED_NOT_MATERIALISED, ST_UNRESOLVED_ND):
                s["rows_unresolved"] += 1
            if cell["blocking_for_delivery"]:
                s["blocking"] = True
            if not s["sample"] and cell["projected_value"]:
                s["sample"] = cell["projected_value"]

            if cell["problem"] and code not in field_problems:
                field_problems[code] = {**cell["problem"], "esma_code": code,
                                        "canonical_field": canonical,
                                        "record_group": g4.record_group_for_code(code)}

    # derive a representative field_projection_status per field.
    for code, s in field_summary.items():
        s["field_projection_status"] = _field_status(s, field_problems.get(code))
        s["blocking_for_delivery"] = s["blocking"]
    return frame_rows, field_summary, field_problems


def _field_status(s: Dict[str, Any], problem: Optional[Dict[str, Any]]) -> str:
    if s["rows_total"] == 0:
        return ST_CARRIED_BLANK
    if s["rows_unresolved"] > 0 and problem:
        return problem.get("status", ST_UNRESOLVED_SOURCE)
    if s["rows_projected_from_transformed"] > 0:
        return ST_FROM_TRANSFORMED
    if s["rows_nd_default"] > 0:
        return ST_ND_DEFAULT
    if s["rows_asset_default"] > 0:
        return ST_ASSET_DEFAULT
    return ST_CARRIED_BLANK


# --------------------------------------------------------------------------- #
# Blocker resolution + projection issues
# --------------------------------------------------------------------------- #

def _resolve_blockers(
    blocker_rows: List[Dict[str, Any]],
    field_summary: Dict[str, Dict[str, Any]],
    field_problems: Dict[str, Dict[str, Any]],
    proj_index: Dict[str, Dict[str, Any]],
    *,
    disposition_by_field: Optional[Dict[str, str]] = None,
):
    """Map each 46_projection_blocker diagnostic into a resolution row, and emit
    the surviving projection issues (55).

    Returns (resolution_rows, issues).
    """
    # index field_summary by canonical_field too (46 esma_code may be blank).
    by_canonical: Dict[str, Dict[str, Any]] = {}
    for code, s in field_summary.items():
        cf = s["canonical_field"]
        if cf and cf not in by_canonical:
            by_canonical[cf] = s

    resolution_rows: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    issue_n = 0
    covered_codes: set = set()

    def _new_issue(*, esma_code, canonical, record_group, issue_type, status, severity,
                   blocking_delivery, action, owner, desc, source_issue_id="") -> str:
        nonlocal issue_n
        issue_n += 1
        iid = f"PROJ-{issue_n:04d}"
        issues.append({
            "issue_id": iid, "source_issue_id": source_issue_id,
            "esma_code": esma_code, "canonical_field": canonical,
            "record_group": record_group, "issue_type": issue_type,
            "projection_status": status, "severity": severity,
            "blocking_for_delivery": bool(blocking_delivery),
            "blocking_for_xml_delivery": True,
            "recommended_action": action, "downstream_owner": owner,
            "description": desc,
        })
        return iid

    disposition_by_field = disposition_by_field or {}

    for b in blocker_rows:
        subtype = b.get("projection_blocker_subtype", "")
        canonical = b.get("canonical_field", "")
        esma = b.get("esma_code", "")
        classification = b.get("validation_classification", "")
        val_issue_id = b.get("issue_id", "")
        onboarding_disposition = (disposition_by_field.get(canonical)
                                  or disposition_by_field.get(esma) or "")

        # locate the field projection outcome.
        summary = field_summary.get(esma) if esma in field_summary else by_canonical.get(canonical)
        if summary is not None:
            esma = summary["esma_code"]
            record_group = summary["record_group"]
            covered_codes.add(esma)
        else:
            record_group = g4.record_group_for_code(esma)

        problem = field_problems.get(esma) if summary is not None else None
        sample = summary["sample"] if summary else ""

        action = ""
        status = ""
        resolved = False
        resolution_source = ""
        remaining_issue_id = ""
        notes = ""

        # EXECUTE the onboarding disposition first so downstream instructions are
        # never contradictory: a field marked client_onboarding/formal-identifier
        # is carried as a client dependency; a config_mapping/asset_policy field is
        # carried as a config dependency — never rediscovered as a source gap.
        if onboarding_disposition in _DISP_CLIENT:
            action = "carry forward client/onboarding dependency (per onboarding disposition)"
            status = ST_BLOCKED_CLIENT
            remaining_issue_id = _new_issue(
                esma_code=esma, canonical=canonical, record_group=record_group,
                issue_type=IT_CLIENT_ONBOARDING, status=ST_BLOCKED_CLIENT,
                severity="warn", blocking_delivery=True,
                action="request/approve the formal identifier/value from the client, then re-project",
                owner="client_onboarding",
                desc=f"{canonical or esma} requires client onboarding per the onboarding disposition "
                     f"({onboarding_disposition})",
                source_issue_id=val_issue_id)
            notes = f"executed onboarding disposition: {onboarding_disposition}"

        elif onboarding_disposition in _DISP_CONFIG:
            action = "carry forward config dependency (per onboarding disposition)"
            status = ST_BLOCKED_OP_CONFIG
            remaining_issue_id = _new_issue(
                esma_code=esma, canonical=canonical, record_group=record_group,
                issue_type=IT_CONFIG, status=ST_BLOCKED_OP_CONFIG,
                severity="warn", blocking_delivery=True,
                action="complete the enum/config mapping or asset/client policy, then re-project",
                owner=OWN_CONFIG,
                desc=f"{canonical or esma} requires config/policy per the onboarding disposition "
                     f"({onboarding_disposition}) — not a source-mapping gap",
                source_issue_id=val_issue_id)
            notes = f"executed onboarding disposition: {onboarding_disposition}"

        elif onboarding_disposition in _DISP_OPERATOR:
            # Operator must confirm an ambiguous source — do NOT silently apply a
            # blind ND/default (e.g. valuation/rate fields). Carried, not resolved.
            action = "carry forward operator decision (per onboarding disposition)"
            status = ST_BLOCKED_OP_CONFIG
            remaining_issue_id = _new_issue(
                esma_code=esma, canonical=canonical, record_group=record_group,
                issue_type=IT_OPERATOR, status=ST_BLOCKED_OP_CONFIG,
                severity="warn", blocking_delivery=True,
                action="operator to confirm the source/treatment, then re-project "
                       "(do not apply a blind ND/default over a real source candidate)",
                owner=OWN_OPERATOR,
                desc=f"{canonical or esma} flagged operator_review by the onboarding disposition "
                     f"— ambiguous source must not be auto-ND-defaulted",
                source_issue_id=val_issue_id)
            notes = f"executed onboarding disposition: {onboarding_disposition}"

        elif subtype == PB_OP_CONFIG_DEP:
            # never resolved by the projection agent — carried forward.
            action = "carry forward operator/config dependency (not resolvable at projection)"
            status = ST_BLOCKED_OP_CONFIG
            is_operator = classification == "operator_required"
            remaining_issue_id = _new_issue(
                esma_code=esma, canonical=canonical, record_group=record_group,
                issue_type=IT_OPERATOR if is_operator else IT_CONFIG,
                status=ST_BLOCKED_OP_CONFIG, severity="warn", blocking_delivery=True,
                action="resolve the upstream operator/config decision, then re-project",
                owner=OWN_OPERATOR if is_operator else OWN_CONFIG,
                desc=f"projection blocked by upstream {classification} dependency on {canonical or esma}",
                source_issue_id=val_issue_id)
            notes = "operator/config decision required upstream"

        elif subtype == PB_MATERIALISED:
            action = "project the transformed canonical value into the target frame"
            if summary and summary["rows_projected_from_transformed"] > 0 and not summary["blocking"]:
                resolved, status = True, ST_FROM_TRANSFORMED
                resolution_source = "transformed_tape"
                notes = "materialised value projected from the transformed tape"
            else:
                status = (problem or {}).get("status", ST_UNRESOLVED_SOURCE)
                remaining_issue_id = _new_issue(
                    esma_code=esma, canonical=canonical, record_group=record_group,
                    issue_type=(problem or {}).get("issue_type", IT_SOURCE_MAPPING),
                    status=status, severity=(problem or {}).get("severity", "warn"),
                    blocking_delivery=True,
                    action=(problem or {}).get("action", "implement an explicit projection rule"),
                    owner=(problem or {}).get("owner", OWN_PROJECTION),
                    desc=(problem or {}).get("desc", f"materialised field {esma} needs an explicit projection rule"),
                    source_issue_id=val_issue_id)
                notes = "materialised but no explicit safe mapping available in v1"

        elif subtype == PB_ND_OR_DEFAULT:
            action = "apply an allowed ND-value or configured default"
            if summary and (summary["rows_nd_default"] > 0 or summary["rows_asset_default"] > 0) \
                    and not summary["blocking"]:
                resolved = True
                status = ST_ND_DEFAULT if summary["rows_nd_default"] > 0 else ST_ASSET_DEFAULT
                resolution_source = "regime_config/asset_config"
                notes = "ND/default applied from regime/asset config"
            else:
                status = (problem or {}).get("status", ST_UNRESOLVED_ND)
                remaining_issue_id = _new_issue(
                    esma_code=esma, canonical=canonical, record_group=record_group,
                    issue_type=(problem or {}).get("issue_type", IT_ND_DEFAULT_MISSING),
                    status=status, severity=(problem or {}).get("severity", "warn"),
                    blocking_delivery=True,
                    action=(problem or {}).get("action", "configure an allowed ND/default for this field"),
                    owner=(problem or {}).get("owner", OWN_CONFIG),
                    desc=(problem or {}).get("desc", f"no allowed ND/default applied for {esma}"),
                    source_issue_id=val_issue_id)
                notes = "ND/default not applicable or not allowed — carried forward"

        elif subtype == PB_SOURCE_MAPPING:
            action = "use an explicit existing mapping only (no generic guessing in v1)"
            if summary and not summary["blocking"] and (
                summary["rows_projected_from_transformed"] > 0
                or summary["rows_nd_default"] > 0
                or summary["rows_asset_default"] > 0):
                resolved = True
                status = summary["field_projection_status"]
                resolution_source = "explicit_regime_rule"
                notes = "resolved via an explicit existing regime/config rule"
            else:
                status = ST_UNRESOLVED_SOURCE
                remaining_issue_id = _new_issue(
                    esma_code=esma, canonical=canonical, record_group=record_group,
                    issue_type=IT_SOURCE_MAPPING, status=ST_UNRESOLVED_SOURCE,
                    severity="warn", blocking_delivery=True,
                    action="define an explicit source-mapping rule (deferred — no generic guessing)",
                    owner=OWN_PROJECTION,
                    desc=f"no explicit safe source mapping for {canonical or esma} in v1",
                    source_issue_id=val_issue_id)
                notes = "no explicit mapping rule — carried forward"

        elif subtype == PB_NOT_MATERIALISED:
            action = "no safe rule available — requires a source value or config"
            if summary and not summary["blocking"] and (
                summary["rows_nd_default"] > 0 or summary["rows_asset_default"] > 0
                or summary["rows_projected_from_transformed"] > 0):
                resolved = True
                status = summary["field_projection_status"]
                resolution_source = "regime_config/asset_config"
                notes = "unexpectedly resolvable via config — applied"
            else:
                status = ST_UNRESOLVED_NOT_MATERIALISED
                remaining_issue_id = _new_issue(
                    esma_code=esma, canonical=canonical, record_group=record_group,
                    issue_type=IT_ND_DEFAULT_MISSING, status=ST_UNRESOLVED_NOT_MATERIALISED,
                    severity="warn", blocking_delivery=True,
                    action="supply a source value or configure an allowed ND/default",
                    owner=OWN_TRANSFORMATION,
                    desc=f"field {canonical or esma} is absent/blank with no source and no allowed ND/default",
                    source_issue_id=val_issue_id)
                notes = "genuine source gap — carried forward"

        else:  # PB_UNKNOWN or anything unexpected.
            action = "investigate field origin and projection requirements"
            status = ST_UNRESOLVED_SOURCE
            remaining_issue_id = _new_issue(
                esma_code=esma, canonical=canonical, record_group=record_group,
                issue_type=IT_SOURCE_MAPPING, status=ST_UNRESOLVED_SOURCE,
                severity="warn", blocking_delivery=True,
                action="investigate field origin and projection requirements manually",
                owner=OWN_PROJECTION,
                desc=f"unknown projection dependency for {canonical or esma}",
                source_issue_id=val_issue_id)
            notes = "unknown projection dependency"

        resolution_rows.append({
            "validation_issue_id": val_issue_id,
            "esma_code": esma,
            "canonical_field": canonical,
            "diagnostic_subtype": subtype,
            "onboarding_disposition": onboarding_disposition,
            "projection_action": action,
            "projection_status": status,
            "resolved": resolved,
            "resolution_source": resolution_source,
            "projected_value_sample": sample,
            "remaining_issue_id": remaining_issue_id,
            "notes": notes,
        })

    # additional projection issues for field-level problems NOT linked to a 46 row
    # (e.g. an unmapped materialised enum that validation did not flag).
    for code, problem in field_problems.items():
        if code in covered_codes:
            continue
        _new_issue(
            esma_code=code, canonical=problem.get("canonical_field", ""),
            record_group=problem.get("record_group", g4.record_group_for_code(code)),
            issue_type=problem.get("issue_type", IT_SOURCE_MAPPING),
            status=problem.get("status", ST_UNRESOLVED_SOURCE),
            severity=problem.get("severity", "warn"), blocking_delivery=True,
            action=problem.get("action", "implement an explicit projection rule"),
            owner=problem.get("owner", OWN_PROJECTION),
            desc=problem.get("desc", f"projection problem on {code}"))

    # one explicit, non-delivery-blocking note that delivery/XML shaping is deferred.
    issue_n += 1
    issues.append({
        "issue_id": f"PROJ-{issue_n:04d}", "source_issue_id": "",
        "esma_code": "", "canonical_field": "", "record_group": "",
        "issue_type": IT_DELIVERY_DEFERRED, "projection_status": "deferred",
        "severity": "info", "blocking_for_delivery": False,
        "blocking_for_xml_delivery": True,
        "recommended_action": "Delivery/XML Agent owns record shaping, precision, "
                              "regex/boolean XSD formatting, RREL/RREC nesting and XML build",
        "downstream_owner": OWN_DELIVERY,
        "description": "Annex 2 delivery normalisation and XML record structure are "
                       "deliberately deferred from Projection Agent v1 "
                       "(see docs/projection_agent_v1_review.md).",
    })

    return resolution_rows, issues


# --------------------------------------------------------------------------- #
# Counts + readiness
# --------------------------------------------------------------------------- #

def _count(frame_rows, issues, resolution_rows) -> Dict[str, int]:
    blocking_delivery = sum(1 for i in issues if i["blocking_for_delivery"])
    remaining = sum(1 for r in resolution_rows if not r["resolved"])
    resolved = sum(1 for r in resolution_rows if r["resolved"])
    required_unresolved = sum(
        1 for r in frame_rows
        if r["blocking_for_delivery"])
    issue_type_counts: Dict[str, int] = {}
    for i in issues:
        issue_type_counts[i["issue_type"]] = issue_type_counts.get(i["issue_type"], 0) + 1
    status_counts: Dict[str, int] = {}
    for r in frame_rows:
        status_counts[r["projection_status"]] = status_counts.get(r["projection_status"], 0) + 1
    return {
        "frame_row_count": len(frame_rows),
        "issue_count": len(issues),
        "blocking_for_delivery_issue_count": blocking_delivery,
        "resolution_row_count": len(resolution_rows),
        "blockers_resolved_count": resolved,
        "remaining_blocker_count": remaining,
        "required_fields_unresolved_count": required_unresolved,
        "issue_type_counts": issue_type_counts,
        "projection_status_counts": status_counts,
    }


def compute_readiness(
    *,
    projection_ran: bool,
    frame_nonempty: bool,
    blocking_for_delivery_issue_count: int,
    required_fields_unresolved_count: int,
    remaining_blocker_count: int,
) -> Dict[str, bool]:
    """Four distinct flags — never collapse them.

    * ``projection_ran``                — the agent produced a target frame;
    * ``projection_complete``           — frame produced AND no blocker remains;
    * ``ready_for_delivery_normalisation`` — frame produced, no delivery-blocking
      issues, all required projected fields present/valid/ND/defaulted/deferred;
    * ``ready_for_xml_delivery``        — always False in this PR.
    """
    ready_for_delivery_normalisation = bool(
        projection_ran
        and frame_nonempty
        and blocking_for_delivery_issue_count == 0
        and required_fields_unresolved_count == 0
    )
    projection_complete = bool(
        ready_for_delivery_normalisation and remaining_blocker_count == 0
    )
    return {
        "projection_ran": bool(projection_ran),
        "projection_complete": projection_complete,
        "ready_for_delivery_normalisation": ready_for_delivery_normalisation,
        "ready_for_xml_delivery": False,
    }


# --------------------------------------------------------------------------- #
# Artefact writers
# --------------------------------------------------------------------------- #

def _write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _write_artefacts(
    *, out_dir: Path, frame_rows, field_summary, ordered_codes, proj_index, issues,
    resolution_rows, readiness, counts, val_manifest, manifest_path, paths,
    client_id, run_id, target_contract_id, row_count, blocker_rows, config_paths,
) -> Dict[str, Any]:

    # 51 — projected Annex 2 target frame (csv + json).
    frame_csv = out_dir / "51_projected_annex2_target_frame.csv"
    frame_json = out_dir / "51_projected_annex2_target_frame.json"
    _write_csv(frame_csv, _FRAME_COLUMNS, frame_rows)
    frame_json.write_text(json.dumps({
        "row_count": len(frame_rows), "shape": "long_one_row_per_loan_field",
        "esma_code_order": ordered_codes, "rows": frame_rows,
    }, indent=2, default=str), encoding="utf-8")

    # 52 — projection field contract (csv + json).
    contract_rows = [field_summary[c] for c in ordered_codes]
    contract_csv = out_dir / "52_projection_field_contract.csv"
    contract_json = out_dir / "52_projection_field_contract.json"
    _write_csv(contract_csv, _CONTRACT_COLUMNS, contract_rows)
    contract_json.write_text(json.dumps({
        "field_count": len(contract_rows), "rows": contract_rows,
    }, indent=2, default=str), encoding="utf-8")

    # 55 — projection issues (csv + json).
    issues_csv = out_dir / "55_projection_issues.csv"
    issues_json = out_dir / "55_projection_issues.json"
    _write_csv(issues_csv, _ISSUE_COLUMNS, issues)
    issues_json.write_text(json.dumps({
        "issue_count": len(issues), "issue_type_counts": counts["issue_type_counts"],
        "rows": issues,
    }, indent=2, default=str), encoding="utf-8")

    # 56 — projection blocker resolution (csv + json).
    resolution_csv = out_dir / "56_projection_blocker_resolution.csv"
    resolution_json = out_dir / "56_projection_blocker_resolution.json"
    _write_csv(resolution_csv, _RESOLUTION_COLUMNS, resolution_rows)
    resolution_json.write_text(json.dumps({
        "blocker_count": len(resolution_rows),
        "blockers_resolved_count": counts["blockers_resolved_count"],
        "remaining_blocker_count": counts["remaining_blocker_count"],
        "rows": resolution_rows,
    }, indent=2, default=str), encoding="utf-8")

    # 54 — projection lineage (extend validation/transformation lineage).
    lineage_path = out_dir / "54_projection_lineage.json"
    val_lineage = _read_json(paths["validation_lineage"]) or {}
    proj_lineage = [{
        "esma_code": s["esma_code"], "canonical_field": s["canonical_field"],
        "record_group": s["record_group"],
        "field_projection_status": s["field_projection_status"],
        "input_artifact": "31_transformed_canonical_tape.csv",
        "output_artifact": "51_projected_annex2_target_frame.csv",
    } for s in (field_summary[c] for c in ordered_codes)]
    lineage_path.write_text(json.dumps({
        "client_id": client_id, "run_id": run_id, "target_contract_id": target_contract_id,
        "transformation_lineage_source": "34_transformation_lineage.json",
        "validation_lineage_source": "44_validation_lineage.json",
        "onboarding_lineage": val_lineage.get("onboarding_lineage", []),
        "transformation_lineage": val_lineage.get("transformation_lineage", []),
        "validation_lineage": val_lineage.get("validation_lineage", []),
        "projection_lineage": proj_lineage,
    }, indent=2, default=str), encoding="utf-8")

    # 53 — readiness (json + md).
    readiness_json = out_dir / "53_projection_readiness.json"
    readiness_md = out_dir / "53_projection_readiness.md"
    next_agent = NEXT_DELIVERY if readiness["ready_for_delivery_normalisation"] else NEXT_REMEDIATION
    readiness_doc = {
        "agent": AGENT, "agent_version": AGENT_VERSION,
        "client_id": client_id, "run_id": run_id,
        "target_contract_id": target_contract_id, "created_at": _now(),
        "row_count": row_count, "next_agent": next_agent,
        **{k: counts[k] for k in (
            "frame_row_count", "issue_count", "blocking_for_delivery_issue_count",
            "resolution_row_count", "blockers_resolved_count", "remaining_blocker_count",
            "required_fields_unresolved_count")},
        "projection_status_counts": counts["projection_status_counts"],
        "issue_type_counts": counts["issue_type_counts"],
        **readiness,
    }
    readiness_json.write_text(json.dumps(readiness_doc, indent=2, default=str), encoding="utf-8")
    readiness_md.write_text(_readiness_md(readiness_doc, readiness, counts, next_agent),
                            encoding="utf-8")

    # 50 — projection manifest (json + yaml).
    manifest_json = out_dir / "50_projection_manifest.json"
    manifest_yaml = out_dir / "50_projection_manifest.yaml"
    manifest = {
        "agent": AGENT, "agent_version": AGENT_VERSION, "stage": STAGE,
        "created_at": _now(),
        "client_id": client_id, "run_id": run_id, "target_contract_id": target_contract_id,

        # governance — what this package IS and IS NOT.
        "consumes_validation_package": True,
        "not_raw_source": True, "did_not_rerun_gate1": True,
        "did_not_rerun_transformation": True, "did_not_rerun_validation": True,
        "did_not_mutate_upstream_artefacts": True,
        "performed_delivery_normalisation": False,
        "invoked_gate5_xml_builder": False,
        "performed_xml_delivery": False,

        # inputs.
        "input_validation_manifest_path": str(manifest_path),
        "input_validation_issues_path": str(paths["validation_issues"]),
        "input_blocker_diagnostics_path": str(paths["blocker_diagnostics"]),
        "input_transformed_tape_path": str(paths["transformed_tape"]),
        "input_transformation_contract_path": str(paths["tx_contract"]),
        "input_validation_lineage_path": str(paths["validation_lineage"]),
        "input_handoff_manifest_path": str(paths["handoff_manifest"]),
        **config_paths,

        # outputs.
        "output_target_frame_csv": str(frame_csv),
        "output_field_contract_csv": str(contract_csv),
        "output_issues_csv": str(issues_csv),
        "output_blocker_resolution_csv": str(resolution_csv),
        "output_readiness_json": str(readiness_json),
        "output_lineage_json": str(lineage_path),

        # counts.
        "row_count": row_count,
        "projected_field_count": len(ordered_codes),
        **{k: counts[k] for k in (
            "frame_row_count", "issue_count", "blocking_for_delivery_issue_count",
            "resolution_row_count", "blockers_resolved_count", "remaining_blocker_count",
            "required_fields_unresolved_count")},
        "input_projection_blocker_count": len(blocker_rows),
        "issue_type_counts": counts["issue_type_counts"],
        "projection_status_counts": counts["projection_status_counts"],

        # readiness.
        **readiness,
        "next_agent": next_agent,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    manifest_yaml.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    return {
        "manifest": manifest,
        "readiness": readiness_doc,
        "projection_dir": str(out_dir),
        "manifest_json_path": str(manifest_json),
        "manifest_yaml_path": str(manifest_yaml),
        "target_frame_csv_path": str(frame_csv),
        "target_frame_json_path": str(frame_json),
        "field_contract_csv_path": str(contract_csv),
        "field_contract_json_path": str(contract_json),
        "readiness_json_path": str(readiness_json),
        "readiness_md_path": str(readiness_md),
        "lineage_path": str(lineage_path),
        "issues_csv_path": str(issues_csv),
        "issues_json_path": str(issues_json),
        "blocker_resolution_csv_path": str(resolution_csv),
        "blocker_resolution_json_path": str(resolution_json),
    }


def _readiness_md(r, readiness, counts, next_agent) -> str:
    def yn(v: bool) -> str:
        return "✅ yes" if v else "❌ no"

    if readiness["projection_complete"]:
        verdict = ("Projection completed and every projection blocker was resolved. "
                   "The target frame is ready for delivery normalisation.")
    elif readiness["ready_for_delivery_normalisation"]:
        verdict = ("Projection ran and produced a target frame with no "
                   "delivery-blocking issues, though some blockers were carried "
                   "forward. It can proceed to delivery normalisation.")
    else:
        verdict = ("Projection ran and produced an Annex 2 target frame, but "
                   "unresolved operator/config/source-mapping blockers remain. "
                   "It is NOT yet ready for delivery normalisation.")

    lines = [
        "# Projection Agent result", "",
        f"Client: {r.get('client_id', '')}  ",
        f"Run: {r.get('run_id', '')}  ",
        f"Target contract: {r.get('target_contract_id', '')}  ",
        f"Agent: **{AGENT} v{AGENT_VERSION}**", "",
        f"> {verdict}", "",
        "## Readiness flags", "",
        f"- projection_ran: {yn(readiness['projection_ran'])}",
        f"- projection_complete: {yn(readiness['projection_complete'])}",
        f"- ready_for_delivery_normalisation: {yn(readiness['ready_for_delivery_normalisation'])}",
        f"- ready_for_xml_delivery: {yn(readiness['ready_for_xml_delivery'])} "
        "(always false in Projection Agent v1)", "",
        "## Projection", "",
        f"- target frame rows: {counts['frame_row_count']}",
        f"- projection issues: {counts['issue_count']} "
        f"(delivery-blocking: {counts['blocking_for_delivery_issue_count']})",
        f"- input projection blockers: {counts['resolution_row_count']}",
        f"- blockers resolved: {counts['blockers_resolved_count']}",
        f"- blockers carried forward: {counts['remaining_blocker_count']}", "",
        "## Projection status mix", "",
    ]
    for k in sorted(counts["projection_status_counts"]):
        lines.append(f"- {k}: {counts['projection_status_counts'][k]}")
    lines += ["", "## Issue type mix", ""]
    if counts["issue_type_counts"]:
        for k in sorted(counts["issue_type_counts"]):
            lines.append(f"- {k}: {counts['issue_type_counts'][k]}")
    else:
        lines.append("- none")
    lines += ["", "## Recommended next action", "",
              f"- next agent: **{next_agent}**"]
    if readiness["ready_for_delivery_normalisation"]:
        lines.append("- Hand the projected target frame to the Delivery/XML Agent for "
                     "delivery normalisation (precision/regex/boolean-XSD/preflight) "
                     "and XML building. **No XML is produced in this PR.**")
    else:
        lines.append("- Resolve carried-forward operator/config/source-mapping "
                     "blockers, then re-run projection. Delivery normalisation and "
                     "XML building remain deferred to the Delivery/XML Agent.")
    lines.append("")
    return "\n".join(lines) + "\n"
