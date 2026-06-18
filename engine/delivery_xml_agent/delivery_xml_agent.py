"""
delivery_xml_agent.py
=====================

Trakt Delivery/XML Agent v1 — the stage after Projection.

It consumes the Projection package (50 manifest, 51 target frame, 52 field
contract, 55 issues, 56 blocker resolution) and emits a governed **delivery
package** under ``output/delivery_xml/`` (artefacts 60..64): a delivery-facing
view of the projected target frame, a delivery-readiness report, delivery issues
by category, and lineage.

Guardrails (enforced by construction — see ``docs/delivery_xml_agent_v1_review.md``):

  * never re-runs / mutates Onboarding / Transformation / Validation / Projection
    artefacts (writes only under ``output/delivery_xml/``);
  * never silently fills a blocked or missing value — blocked rows are carried,
    never rewritten or promoted;
  * never lets an XML builder override an upstream decision (the frozen Gate 5
    builder and Gate 4b mutator are **not** invoked);
  * **refuses** XML unless every delivery-readiness gate passes; XML preview is
    additionally guarded behind ``--allow-xml-preview`` and still honours the gates.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from engine.delivery_xml_agent import gate5_adapter as g5
from engine.delivery_xml_agent.delivery_readiness import compute_delivery_readiness

# --------------------------------------------------------------------------- #
# Identity / vocabulary
# --------------------------------------------------------------------------- #

AGENT = "delivery_xml_agent"
AGENT_VERSION = "1.0"
STAGE = "post_projection_delivery"

REQUIRED_PROJECTION_AGENT = "projection_agent"

NEXT_XML = "xml_generation"
NEXT_REMEDIATION = "operator_config_projection_remediation"

# Projection target-frame statuses treated as candidate deliverable values
# (subject to delivery validation).
ST_FROM_TRANSFORMED = "projected_from_transformed"
ST_ND_DEFAULT = "projected_nd_default"
ST_ASSET_DEFAULT = "projected_asset_default"
CANDIDATE_DELIVERABLE = {ST_FROM_TRANSFORMED, ST_ND_DEFAULT, ST_ASSET_DEFAULT}

# Projection target-frame statuses always treated as delivery blockers.
ST_BLOCKED_CLIENT = "blocked_client_onboarding_dependency"
ST_BLOCKED_OP_CONFIG = "blocked_operator_or_config_dependency"
ST_UNRESOLVED_NOT_MATERIALISED = "unresolved_not_materialised"
ST_UNRESOLVED_SOURCE = "unresolved_source_mapping"
ST_INVALID_ND = "invalid_nd_for_field"
ALWAYS_BLOCKED = {
    ST_BLOCKED_CLIENT, ST_BLOCKED_OP_CONFIG, ST_UNRESOLVED_NOT_MATERIALISED,
    ST_UNRESOLVED_SOURCE, ST_INVALID_ND,
}
# Conditionally blocked: blank only blocks when the field is mandatory and no
# allowed/selected ND/default exists.
ST_CARRIED_BLANK = "not_projected_blank"

# Delivery statuses (62 frame).
DS_DELIVERABLE = "deliverable"
DS_BLOCKED = "blocked"
DS_INVALID = "delivery_invalid"
DS_NOT_REQUIRED_BLANK = "not_required_blank"

# Delivery blocker-type categories (also the 63 issue categories).
BT_CLIENT = "client_onboarding_dependency"
BT_OPERATOR_OR_CONFIG = "operator_or_config_dependency"
BT_CONFIG = "config_dependency"
BT_SOURCE_MAPPING = "source_mapping_unresolved"
BT_ND_DEFAULT_MISSING = "nd_default_rule_missing"
BT_FORMAT = "delivery_format_invalid"
BT_STRUCTURE_DEFERRED = "delivery_structure_deferred"
BT_TEMPLATE_ORDER = "template_order_incomplete"

# Map a blocked projection status to a coarse blocker type (refined per-code
# using the projection issue types when available).
_STATUS_BLOCKER_TYPE = {
    ST_BLOCKED_CLIENT: BT_CLIENT,
    ST_BLOCKED_OP_CONFIG: BT_OPERATOR_OR_CONFIG,
    ST_UNRESOLVED_SOURCE: BT_SOURCE_MAPPING,
    ST_UNRESOLVED_NOT_MATERIALISED: BT_ND_DEFAULT_MISSING,
    ST_INVALID_ND: BT_ND_DEFAULT_MISSING,
    ST_CARRIED_BLANK: BT_ND_DEFAULT_MISSING,
}

# Projection issue types (55) → delivery blocker type, for refinement + carry.
_PROJ_ISSUE_TYPE_BLOCKER = {
    "client_onboarding_dependency_unresolved": BT_CLIENT,
    "operator_dependency_unresolved": BT_OPERATOR_OR_CONFIG,
    "config_dependency_unresolved": BT_CONFIG,
    "source_mapping_unresolved": BT_SOURCE_MAPPING,
    "nd_default_rule_missing": BT_ND_DEFAULT_MISSING,
    "invalid_nd_for_field": BT_ND_DEFAULT_MISSING,
    "delivery_structure_deferred": BT_STRUCTURE_DEFERRED,
}

# Downstream owners.
OWN_CLIENT = "client_onboarding"
OWN_OPERATOR = "operator"
OWN_CONFIG = "config_policy"
OWN_PROJECTION = "projection"
OWN_DELIVERY = "delivery_xml"

_BLOCKER_OWNER = {
    BT_CLIENT: OWN_CLIENT,
    BT_OPERATOR_OR_CONFIG: OWN_OPERATOR,
    BT_CONFIG: OWN_CONFIG,
    BT_SOURCE_MAPPING: OWN_PROJECTION,
    BT_ND_DEFAULT_MISSING: OWN_CONFIG,
    BT_FORMAT: OWN_DELIVERY,
    BT_STRUCTURE_DEFERRED: OWN_DELIVERY,
    BT_TEMPLATE_ORDER: OWN_DELIVERY,
}

# Required report/header metadata codes that must be deliverable before XML.
REQUIRED_HEADER_METADATA_CODES = ["RREL1"]

_FRAME_COLUMNS = [
    "row_id", "loan_identifier", "record_group", "esma_code", "canonical_field",
    "projected_value", "projection_status",
    # delivery-facing columns:
    "delivery_status", "delivery_value", "delivery_blocker_type", "delivery_rule_id",
    "xml_path", "xml_record_group", "xsd_type", "is_mandatory", "is_nd_value",
    "nd_allowed", "format_valid", "enum_valid", "delivery_issue_id",
]

_ISSUE_COLUMNS = [
    "delivery_issue_id", "esma_code", "canonical_field", "record_group",
    "xml_record_group", "delivery_blocker_type", "delivery_status", "severity",
    "source_projection_issue_id", "blocking_for_delivery", "blocking_for_xml",
    "recommended_action", "downstream_owner", "description",
]


class ProjectionHandoffError(RuntimeError):
    """Raised when the projection package is missing or not consumable by delivery."""


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
# Validate the projection manifest
# --------------------------------------------------------------------------- #

def validate_projection_manifest(manifest: dict) -> None:
    """Fail loudly unless the package was produced by the Projection Agent and is
    consumable by the Delivery/XML Agent.

    Required:
        agent == projection_agent (or equivalent);
        performed_xml_delivery == false (delivery has not already happened);
        consumes_validation_package == true (it is a real projection package).

    Note: ``projection_complete == false`` is **allowed** — the Delivery Agent is
    built to consume an incomplete projection and refuse XML cleanly.
    """
    if not isinstance(manifest, dict):
        raise ProjectionHandoffError("Projection manifest is not a JSON object.")

    problems: List[str] = []
    agent = str(manifest.get("agent", "")).lower()
    if REQUIRED_PROJECTION_AGENT not in agent and "projection" not in agent:
        problems.append(f"agent must be a projection agent, got {manifest.get('agent')!r}")
    if manifest.get("performed_xml_delivery") is True:
        problems.append("projection package must not have performed XML delivery")
    if manifest.get("consumes_validation_package") is False:
        problems.append("projection package must consume the validation package")
    if problems:
        raise ProjectionHandoffError(
            "Projection package is not consumable by the Delivery/XML Agent:\n  - "
            + "\n  - ".join(problems))


def _resolve_inputs(manifest_path: Path) -> Dict[str, Path]:
    proj_dir = manifest_path.parent              # .../output/projection
    output_root = proj_dir.parent                # .../output
    return {
        "proj_dir": proj_dir,
        "output_root": output_root,
        "target_frame_csv": proj_dir / "51_projected_annex2_target_frame.csv",
        "target_frame_json": proj_dir / "51_projected_annex2_target_frame.json",
        "field_contract": proj_dir / "52_projection_field_contract.csv",
        "projection_readiness": proj_dir / "53_projection_readiness.json",
        "projection_lineage": proj_dir / "54_projection_lineage.json",
        "projection_issues": proj_dir / "55_projection_issues.csv",
        "blocker_resolution": proj_dir / "56_projection_blocker_resolution.csv",
    }


# --------------------------------------------------------------------------- #
# Regime contract access (raw field_rules — never a mutation pass)
# --------------------------------------------------------------------------- #

def _load_field_rules(regime_config_path: str | Path) -> Dict[str, Dict[str, Any]]:
    cfg = _read_yaml(Path(regime_config_path)) or {}
    rules = cfg.get("field_rules")
    if not isinstance(rules, dict):
        return {}
    return {str(k): (v or {}) for k, v in rules.items()}


def _is_mandatory(code: str, rule: Dict[str, Any], universe: Dict[str, Dict[str, Any]]) -> bool:
    if rule:
        return bool(rule.get("mandatory", False))
    # Fall back to the workbook universe: a field that allows no ND at all is
    # treated as mandatory-presence for the readiness gate.
    entry = universe.get(code) or {}
    if entry:
        return not (bool(entry.get("nd1_4_allowed")) or bool(entry.get("nd5_allowed")))
    return False


def _nd_allowed_list(rule: Dict[str, Any], universe_entry: Dict[str, Any]) -> List[str]:
    if rule and isinstance(rule.get("nd_allowed"), list):
        return [str(x).upper() for x in rule["nd_allowed"]]
    out: List[str] = []
    if universe_entry.get("nd1_4_allowed"):
        out += ["ND1", "ND2", "ND3", "ND4"]
    if universe_entry.get("nd5_allowed"):
        out += ["ND5"]
    return out


# --------------------------------------------------------------------------- #
# Build the delivery-normalised frame (62) + delivery issues (63)
# --------------------------------------------------------------------------- #

def _build_delivery_frame(
    *,
    frame_rows: List[Dict[str, Any]],
    field_rules: Dict[str, Dict[str, Any]],
    universe: Dict[str, Dict[str, Any]],
    proj_issue_types_by_code: Dict[str, set],
) -> Dict[str, Any]:
    """Return delivery frame rows + aggregate counters used by the readiness gates.

    Each row is classified — deliverable / blocked / delivery_invalid /
    not_required_blank — without ever rewriting or promoting a blocked value.
    """
    out_rows: List[Dict[str, Any]] = []
    blocked_codes: Dict[str, Dict[str, Any]] = {}     # esma_code -> sample blocked row info
    blocked_row_count = 0
    mandatory_blank_without_nd = 0
    format_violations = 0
    rows_without_record_group = 0

    for r in frame_rows:
        code = _to_str(r.get("esma_code"))
        canonical = _to_str(r.get("canonical_field"))
        record_group = _to_str(r.get("record_group"))
        status = _to_str(r.get("projection_status"))
        projected = _to_str(r.get("projected_value"))

        rule = field_rules.get(code, {})
        universe_entry = universe.get(code, {})
        mandatory = _is_mandatory(code, rule, universe)
        nd_allowed = _nd_allowed_list(rule, universe_entry)
        is_nd = g5.is_nd_value(projected)
        xml_group = g5.record_group_to_xml_group(record_group)
        xsd_type = g5.xsd_type_for_code(code, universe)
        xml_path = _to_str(rule.get("workbook_semantic"))  # best-effort; blank => deferred

        if not record_group:
            rows_without_record_group += 1

        fmt_ok = g5.format_valid(projected, rule)
        enum_ok = g5.enum_valid(projected, rule)

        # Classify.
        delivery_status = DS_DELIVERABLE
        delivery_value = ""
        blocker_type = ""

        conditionally_blocked = (
            status == ST_CARRIED_BLANK and mandatory and not nd_allowed and projected == ""
        )
        if status in ALWAYS_BLOCKED or conditionally_blocked:
            delivery_status = DS_BLOCKED
            blocker_type = _refine_blocker_type(status, code, proj_issue_types_by_code)
            blocked_row_count += 1
            blocked_codes.setdefault(code, {
                "esma_code": code, "canonical_field": canonical,
                "record_group": record_group, "xml_record_group": xml_group,
                "blocker_type": blocker_type, "projection_status": status,
            })
            # delivery_value left blank: never promote a blocked projected value.
        elif status in CANDIDATE_DELIVERABLE:
            if projected == "":
                if mandatory and not nd_allowed:
                    delivery_status = DS_BLOCKED
                    blocker_type = BT_ND_DEFAULT_MISSING
                    blocked_row_count += 1
                    mandatory_blank_without_nd += 1
                    blocked_codes.setdefault(code, {
                        "esma_code": code, "canonical_field": canonical,
                        "record_group": record_group, "xml_record_group": xml_group,
                        "blocker_type": blocker_type, "projection_status": status,
                    })
                else:
                    delivery_status = DS_NOT_REQUIRED_BLANK
            elif not (fmt_ok and enum_ok):
                delivery_status = DS_INVALID
                blocker_type = BT_FORMAT
                format_violations += 1
            else:
                delivery_status = DS_DELIVERABLE
                delivery_value = projected
        else:
            # Unknown / non-mandatory blank statuses — carry, do not deliver.
            if projected == "" and not (mandatory and not nd_allowed):
                delivery_status = DS_NOT_REQUIRED_BLANK
            else:
                delivery_status = DS_BLOCKED
                blocker_type = _refine_blocker_type(status, code, proj_issue_types_by_code)
                blocked_row_count += 1
                blocked_codes.setdefault(code, {
                    "esma_code": code, "canonical_field": canonical,
                    "record_group": record_group, "xml_record_group": xml_group,
                    "blocker_type": blocker_type, "projection_status": status,
                })

        out_rows.append({
            "row_id": _to_str(r.get("row_id")),
            "loan_identifier": _to_str(r.get("loan_identifier")),
            "record_group": record_group,
            "esma_code": code,
            "canonical_field": canonical,
            "projected_value": projected,
            "projection_status": status,
            "delivery_status": delivery_status,
            "delivery_value": delivery_value,
            "delivery_blocker_type": blocker_type,
            "delivery_rule_id": f"annex2_delivery_rules::{code}" if rule else "",
            "xml_path": xml_path,
            "xml_record_group": xml_group,
            "xsd_type": xsd_type,
            "is_mandatory": mandatory,
            "is_nd_value": is_nd,
            "nd_allowed": "|".join(nd_allowed),
            "format_valid": fmt_ok,
            "enum_valid": enum_ok,
            "delivery_issue_id": "",   # linked below
        })

    return {
        "rows": out_rows,
        "blocked_codes": blocked_codes,
        "blocked_row_count": blocked_row_count,
        "mandatory_blank_without_nd": mandatory_blank_without_nd,
        "format_violations": format_violations,
        "rows_without_record_group": rows_without_record_group,
    }


def _refine_blocker_type(status: str, code: str, proj_issue_types_by_code: Dict[str, set]) -> str:
    """Refine a coarse status-derived blocker type using the projection issue
    types observed for the same code (so config vs operator is distinguished)."""
    types = proj_issue_types_by_code.get(code, set())
    for it in ("config_dependency_unresolved", "client_onboarding_dependency_unresolved",
               "operator_dependency_unresolved", "source_mapping_unresolved"):
        if it in types:
            return _PROJ_ISSUE_TYPE_BLOCKER[it]
    return _STATUS_BLOCKER_TYPE.get(status, BT_OPERATOR_OR_CONFIG)


def _build_delivery_issues(
    *,
    delivery_frame: Dict[str, Any],
    proj_issues: List[Dict[str, Any]],
    rows_by_code: Dict[str, List[Dict[str, Any]]],
    missing_required_order_codes: List[str],
) -> List[Dict[str, Any]]:
    """One delivery issue per blocked code + the carried delivery-blocking
    projection issues + the structural (deferred / template-order) issues."""
    issues: List[Dict[str, Any]] = []
    seq = 0

    def _new_id() -> str:
        nonlocal seq
        seq += 1
        return f"DEL-{seq:04d}"

    # 1. one delivery issue per blocked code (carries the blocked rows into 63).
    seen_proj_issue_for_code: Dict[str, str] = {}
    for pi in proj_issues:
        code = _to_str(pi.get("esma_code"))
        if code and code not in seen_proj_issue_for_code:
            seen_proj_issue_for_code[code] = _to_str(pi.get("issue_id"))

    for code, info in delivery_frame["blocked_codes"].items():
        issue_id = _new_id()
        # link every blocked frame row of this code to the issue.
        for row in rows_by_code.get(code, []):
            row["delivery_issue_id"] = issue_id
        blocker_type = info["blocker_type"]
        issues.append({
            "delivery_issue_id": issue_id,
            "esma_code": code,
            "canonical_field": info["canonical_field"],
            "record_group": info["record_group"],
            "xml_record_group": info["xml_record_group"],
            "delivery_blocker_type": blocker_type,
            "delivery_status": DS_BLOCKED,
            "severity": "error",
            "source_projection_issue_id": seen_proj_issue_for_code.get(code, ""),
            "blocking_for_delivery": True,
            "blocking_for_xml": True,
            "recommended_action": _recommended_action(blocker_type),
            "downstream_owner": _BLOCKER_OWNER.get(blocker_type, OWN_PROJECTION),
            "description": (f"Projection carried {code} as {info['projection_status']}; "
                           f"delivery refuses to fill it."),
        })

    # 1b. format-invalid rows (present value that fails delivery validation).
    fmt_codes = {}
    for row in delivery_frame["rows"]:
        if row["delivery_status"] == DS_INVALID:
            fmt_codes.setdefault(row["esma_code"], row)
    for code, row in fmt_codes.items():
        issue_id = _new_id()
        for r2 in rows_by_code.get(code, []):
            if r2["delivery_status"] == DS_INVALID and not r2["delivery_issue_id"]:
                r2["delivery_issue_id"] = issue_id
        issues.append({
            "delivery_issue_id": issue_id,
            "esma_code": code,
            "canonical_field": row["canonical_field"],
            "record_group": row["record_group"],
            "xml_record_group": row["xml_record_group"],
            "delivery_blocker_type": BT_FORMAT,
            "delivery_status": DS_INVALID,
            "severity": "error",
            "source_projection_issue_id": "",
            "blocking_for_delivery": True,
            "blocking_for_xml": True,
            "recommended_action": "fix value to satisfy regime format/enum rules",
            "downstream_owner": OWN_DELIVERY,
            "description": f"Value for {code} violates delivery format/enum rules.",
        })

    # 2. structural — RREL/RREC nesting / collateral cardinality deferred to v2.
    issues.append({
        "delivery_issue_id": _new_id(),
        "esma_code": "", "canonical_field": "", "record_group": "",
        "xml_record_group": "", "delivery_blocker_type": BT_STRUCTURE_DEFERRED,
        "delivery_status": "", "severity": "info",
        "source_projection_issue_id": "",
        "blocking_for_delivery": False, "blocking_for_xml": True,
        "recommended_action": "decide RREL/RREC nesting & collateral cardinality (v2)",
        "downstream_owner": OWN_DELIVERY,
        "description": ("Long target frame tags record_group only; RREL↔RREC nesting "
                       "is deferred to Delivery/XML Agent v2."),
    })

    # 3. template/code order incomplete for required XML fields.
    if missing_required_order_codes:
        issues.append({
            "delivery_issue_id": _new_id(),
            "esma_code": ",".join(missing_required_order_codes[:20]),
            "canonical_field": "", "record_group": "", "xml_record_group": "",
            "delivery_blocker_type": BT_TEMPLATE_ORDER, "delivery_status": "",
            "severity": "error", "source_projection_issue_id": "",
            "blocking_for_delivery": False, "blocking_for_xml": True,
            "recommended_action": "add missing required codes to esma_code_order.yaml Record list",
            "downstream_owner": OWN_DELIVERY,
            "description": (f"{len(missing_required_order_codes)} required code(s) absent from "
                           "the ESMA Record order; XML ordering cannot be guaranteed."),
        })

    return issues


def _recommended_action(blocker_type: str) -> str:
    return {
        BT_CLIENT: "resolve client onboarding identifier policy (upstream)",
        BT_OPERATOR_OR_CONFIG: "resolve operator/config dependency (upstream)",
        BT_CONFIG: "supply config mapping (upstream)",
        BT_SOURCE_MAPPING: "resolve source mapping in projection (upstream)",
        BT_ND_DEFAULT_MISSING: "define an allowed ND/default rule (upstream)",
        BT_FORMAT: "fix value to satisfy regime format/enum rules",
        BT_STRUCTURE_DEFERRED: "decide RREL/RREC nesting & collateral cardinality (v2)",
        BT_TEMPLATE_ORDER: "complete esma_code_order Record list",
    }.get(blocker_type, "resolve upstream blocker")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def build_delivery_package(
    projection_manifest_path: str | Path,
    *,
    regime_config_path: str = "",
    esma_code_order_path: str = "",
    registry_path: str = "",
    field_universe_path: str = "",
    allow_xml_preview: bool = False,
) -> Dict[str, Any]:
    """Consume the Projection package and produce the delivery package (60..64).

    Returns a dict of artefact paths + the delivery manifest. Raises
    :class:`ProjectionHandoffError` if the projection manifest is missing/invalid
    or not consumable. **Never** generates production XML; an XML *preview* is
    written only if every readiness gate passes **and** ``allow_xml_preview`` is
    set (the flag never bypasses the gates).
    """
    manifest_path = Path(projection_manifest_path)
    if not manifest_path.exists():
        raise ProjectionHandoffError(f"Projection manifest not found: {manifest_path}")
    proj_manifest = _read_json(manifest_path)
    if proj_manifest is None:
        raise ProjectionHandoffError(f"Projection manifest is not valid JSON: {manifest_path}")
    validate_projection_manifest(proj_manifest)

    paths = _resolve_inputs(manifest_path)
    if not paths["target_frame_csv"].exists():
        raise ProjectionHandoffError(
            f"Projected target frame not found: {paths['target_frame_csv']}")

    client_id = proj_manifest.get("client_id", "")
    run_id = proj_manifest.get("run_id", "")
    target_contract_id = proj_manifest.get("target_contract_id", "")

    repo_root = Path(__file__).resolve().parents[2]
    regime_config_path = (regime_config_path
                          or proj_manifest.get("regime_config_path", "")
                          or str(repo_root / "config" / "regime" / "annex2_delivery_rules.yaml"))
    esma_code_order_path = (esma_code_order_path
                            or proj_manifest.get("esma_code_order_path", "")
                            or str(repo_root / "config" / "system" / "esma_code_order.yaml"))
    registry_path = (registry_path
                     or proj_manifest.get("registry_path", "")
                     or str(repo_root / "config" / "system" / "fields_registry.yaml"))
    field_universe_path = (field_universe_path
                           or str(repo_root / "config" / "regime" / "annex2_field_universe.yaml"))

    # Inputs.
    frame_rows = _read_csv_rows(paths["target_frame_csv"])
    proj_issues = _read_csv_rows(paths["projection_issues"])
    field_rules = _load_field_rules(regime_config_path)
    universe = g5.field_universe_index(field_universe_path)
    record_order = g5.load_record_order(esma_code_order_path)
    record_order_set = set(record_order)

    # Projection issue types observed per code (for blocker-type refinement/carry).
    proj_issue_types_by_code: Dict[str, set] = {}
    delivery_blocking_proj_issue_count = 0
    for pi in proj_issues:
        code = _to_str(pi.get("esma_code"))
        it = _to_str(pi.get("issue_type"))
        if code and it:
            proj_issue_types_by_code.setdefault(code, set()).add(it)
        if _truthy(pi.get("blocking_for_delivery")):
            delivery_blocking_proj_issue_count += 1

    # Delivery frame.
    delivery_frame = _build_delivery_frame(
        frame_rows=frame_rows, field_rules=field_rules, universe=universe,
        proj_issue_types_by_code=proj_issue_types_by_code)
    delivery_rows = delivery_frame["rows"]
    rows_by_code: Dict[str, List[Dict[str, Any]]] = {}
    for row in delivery_rows:
        rows_by_code.setdefault(row["esma_code"], []).append(row)

    # Required header/report metadata present & deliverable?
    deliverable_codes = {
        row["esma_code"] for row in delivery_rows if row["delivery_status"] == DS_DELIVERABLE}
    missing_header_metadata = [
        c for c in REQUIRED_HEADER_METADATA_CODES if c not in deliverable_codes]

    # Template/code order completeness for required (mandatory) frame codes.
    frame_codes = {row["esma_code"] for row in delivery_rows if row["esma_code"]}
    mandatory_frame_codes = {
        c for c in frame_codes if _is_mandatory(c, field_rules.get(c, {}), universe)}
    missing_required_order_codes = sorted(
        c for c in mandatory_frame_codes if record_order_set and c not in record_order_set)

    # Delivery issues (carries blocked rows + structural issues).
    issues = _build_delivery_issues(
        delivery_frame=delivery_frame, proj_issues=proj_issues,
        rows_by_code=rows_by_code,
        missing_required_order_codes=missing_required_order_codes)

    # Readiness gates.
    proj_complete = bool(proj_manifest.get("projection_complete", False))
    proj_ready_norm = bool(proj_manifest.get("ready_for_delivery_normalisation", False))
    proj_ready_xml = bool(proj_manifest.get("ready_for_xml_delivery", False))
    readiness = compute_delivery_readiness(
        projection_complete=proj_complete,
        ready_for_delivery_normalisation=proj_ready_norm,
        ready_for_xml_delivery=proj_ready_xml,
        delivery_blocking_projection_issue_count=delivery_blocking_proj_issue_count,
        blocked_frame_row_count=delivery_frame["blocked_row_count"],
        mandatory_blank_without_nd_count=delivery_frame["mandatory_blank_without_nd"],
        format_violation_count=delivery_frame["format_violations"],
        missing_header_metadata=missing_header_metadata,
        rows_without_record_group=delivery_frame["rows_without_record_group"],
        missing_required_order_codes=missing_required_order_codes,
    )

    counts = _counts(delivery_rows, issues, readiness)

    out_dir = paths["output_root"] / "delivery_xml"
    out_dir.mkdir(parents=True, exist_ok=True)

    return _write_artefacts(
        out_dir=out_dir, delivery_rows=delivery_rows, issues=issues,
        readiness=readiness, counts=counts, proj_manifest=proj_manifest,
        manifest_path=manifest_path, paths=paths, client_id=client_id,
        run_id=run_id, target_contract_id=target_contract_id,
        config_paths={
            "regime_config_path": regime_config_path,
            "esma_code_order_path": esma_code_order_path,
            "registry_path": registry_path,
            "field_universe_path": field_universe_path,
        },
        allow_xml_preview=allow_xml_preview,
        missing_required_order_codes=missing_required_order_codes,
    )


def _counts(delivery_rows, issues, readiness) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    for r in delivery_rows:
        status_counts[r["delivery_status"]] = status_counts.get(r["delivery_status"], 0) + 1
    blocker_counts: Dict[str, int] = {}
    for i in issues:
        bt = i["delivery_blocker_type"]
        blocker_counts[bt] = blocker_counts.get(bt, 0) + 1
    return {
        "frame_row_count": len(delivery_rows),
        "deliverable_row_count": status_counts.get(DS_DELIVERABLE, 0),
        "blocked_row_count": status_counts.get(DS_BLOCKED, 0),
        "invalid_row_count": status_counts.get(DS_INVALID, 0),
        "not_required_blank_count": status_counts.get(DS_NOT_REQUIRED_BLANK, 0),
        "issue_count": len(issues),
        "delivery_status_counts": status_counts,
        "delivery_blocker_type_counts": blocker_counts,
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
    *, out_dir, delivery_rows, issues, readiness, counts, proj_manifest,
    manifest_path, paths, client_id, run_id, target_contract_id, config_paths,
    allow_xml_preview, missing_required_order_codes,
) -> Dict[str, Any]:

    xml_generation_allowed = readiness["xml_generation_allowed"]
    delivery_norm_complete = readiness["delivery_normalisation_complete"]
    next_agent = NEXT_XML if xml_generation_allowed else NEXT_REMEDIATION

    # 62 — delivery-normalised frame (csv + json).
    frame_csv = out_dir / "62_delivery_normalised_frame.csv"
    frame_json = out_dir / "62_delivery_normalised_frame.json"
    _write_csv(frame_csv, _FRAME_COLUMNS, delivery_rows)
    frame_json.write_text(json.dumps({
        "row_count": len(delivery_rows),
        "shape": "long_one_row_per_loan_field_delivery_view",
        "columns": _FRAME_COLUMNS, "rows": delivery_rows,
    }, indent=2, default=str), encoding="utf-8")

    # 63 — delivery issues (csv + json).
    issues_csv = out_dir / "63_delivery_issues.csv"
    issues_json = out_dir / "63_delivery_issues.json"
    _write_csv(issues_csv, _ISSUE_COLUMNS, issues)
    issues_json.write_text(json.dumps({
        "issue_count": len(issues),
        "delivery_blocker_type_counts": counts["delivery_blocker_type_counts"],
        "rows": issues,
    }, indent=2, default=str), encoding="utf-8")

    # 64 — delivery lineage (extend projection lineage).
    lineage_path = out_dir / "64_delivery_lineage.json"
    proj_lineage = _read_json(paths["projection_lineage"]) or {}
    lineage_path.write_text(json.dumps({
        "client_id": client_id, "run_id": run_id, "target_contract_id": target_contract_id,
        "projection_lineage_source": "54_projection_lineage.json",
        "onboarding_lineage": proj_lineage.get("onboarding_lineage", []),
        "transformation_lineage": proj_lineage.get("transformation_lineage", []),
        "validation_lineage": proj_lineage.get("validation_lineage", []),
        "projection_lineage": proj_lineage.get("projection_lineage", []),
        "delivery_lineage": [{
            "input_artifact": "51_projected_annex2_target_frame.csv",
            "output_artifact": "62_delivery_normalised_frame.csv",
            "deliverable_rows": counts["deliverable_row_count"],
            "blocked_rows": counts["blocked_row_count"],
        }],
    }, indent=2, default=str), encoding="utf-8")

    # 61 — delivery readiness (json + md).
    readiness_json = out_dir / "61_delivery_readiness.json"
    readiness_md = out_dir / "61_delivery_readiness.md"
    readiness_doc = {
        "agent": AGENT, "agent_version": AGENT_VERSION,
        "client_id": client_id, "run_id": run_id,
        "target_contract_id": target_contract_id, "created_at": _now(),
        "delivery_xml_ran": True,
        "delivery_normalisation_complete": delivery_norm_complete,
        "xml_generation_allowed": xml_generation_allowed,
        "xml_generated": False,
        "ready_for_xml_delivery": False,
        "next_agent": next_agent,
        "gates": readiness["gates"],
        **{k: counts[k] for k in (
            "frame_row_count", "deliverable_row_count", "blocked_row_count",
            "invalid_row_count", "not_required_blank_count", "issue_count")},
        "delivery_status_counts": counts["delivery_status_counts"],
        "delivery_blocker_type_counts": counts["delivery_blocker_type_counts"],
    }
    readiness_json.write_text(json.dumps(readiness_doc, indent=2, default=str), encoding="utf-8")
    readiness_md.write_text(_readiness_md(readiness_doc, readiness, counts, next_agent),
                            encoding="utf-8")

    # 65/66 — XML preview (guarded; never produced unless gates pass AND flag set).
    xml_preview_path = out_dir / "65_xml_preview.xml"
    xml_report_path = out_dir / "66_xml_validation_report.json"
    xml_generated = False
    xml_preview_written = False
    if allow_xml_preview and xml_generation_allowed:
        # v1 still does NOT build a production tree here; it writes a guarded,
        # clearly-labelled preview placeholder so downstream tooling can detect
        # that the gates passed. Real tree building is deferred to v2.
        xml_preview_path.write_text(
            "<!-- Delivery/XML Agent v1 preview placeholder. Production XML "
            "generation is deferred to v2. -->\n", encoding="utf-8")
        xml_report_path.write_text(json.dumps({
            "xml_preview": True, "production_xml": False,
            "note": "preview only; gates passed but production XML deferred to v2",
        }, indent=2), encoding="utf-8")
        xml_preview_written = True
        # xml_generated stays False — no PRODUCTION XML is ever generated in v1.

    # 60 — delivery manifest (json + yaml).
    manifest_json = out_dir / "60_delivery_manifest.json"
    manifest_yaml = out_dir / "60_delivery_manifest.yaml"
    manifest = {
        "agent": AGENT, "agent_version": AGENT_VERSION, "stage": STAGE,
        "created_at": _now(),
        "client_id": client_id, "run_id": run_id, "target_contract_id": target_contract_id,

        # governance — what this package IS and IS NOT.
        "consumes_projection_package": True,
        "did_not_rerun_projection": True,
        "did_not_mutate_upstream_artefacts": True,
        "invoked_gate5_xml_builder": False,
        "invoked_gate4b_normalizer": False,
        "silently_filled_blocked_values": False,

        # inputs.
        "input_projection_manifest_path": str(manifest_path),
        "input_target_frame_path": str(paths["target_frame_csv"]),
        "input_field_contract_path": str(paths["field_contract"]),
        "input_projection_issues_path": str(paths["projection_issues"]),
        "input_blocker_resolution_path": str(paths["blocker_resolution"]),
        "input_projection_lineage_path": str(paths["projection_lineage"]),
        **config_paths,

        # outputs (all artefacts linked).
        "output_readiness_json": str(readiness_json),
        "output_readiness_md": str(readiness_md),
        "output_delivery_frame_csv": str(frame_csv),
        "output_delivery_frame_json": str(frame_json),
        "output_delivery_issues_csv": str(issues_csv),
        "output_delivery_issues_json": str(issues_json),
        "output_lineage_json": str(lineage_path),
        "output_xml_preview": str(xml_preview_path) if xml_preview_written else "",
        "output_xml_validation_report": str(xml_report_path) if xml_preview_written else "",

        # counts.
        **{k: counts[k] for k in (
            "frame_row_count", "deliverable_row_count", "blocked_row_count",
            "invalid_row_count", "not_required_blank_count", "issue_count")},
        "delivery_status_counts": counts["delivery_status_counts"],
        "delivery_blocker_type_counts": counts["delivery_blocker_type_counts"],
        "missing_required_order_code_count": len(missing_required_order_codes),

        # readiness verdict.
        "delivery_xml_ran": True,
        "delivery_normalisation_complete": delivery_norm_complete,
        "xml_generation_allowed": xml_generation_allowed,
        "allow_xml_preview_flag": bool(allow_xml_preview),
        "xml_preview_written": xml_preview_written,
        "xml_generated": xml_generated,
        "ready_for_xml_delivery": False,
        "next_agent": next_agent,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    manifest_yaml.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    return {
        "manifest": manifest,
        "readiness": readiness_doc,
        "delivery_dir": str(out_dir),
        "manifest_json_path": str(manifest_json),
        "manifest_yaml_path": str(manifest_yaml),
        "readiness_json_path": str(readiness_json),
        "readiness_md_path": str(readiness_md),
        "delivery_frame_csv_path": str(frame_csv),
        "delivery_frame_json_path": str(frame_json),
        "delivery_issues_csv_path": str(issues_csv),
        "delivery_issues_json_path": str(issues_json),
        "lineage_path": str(lineage_path),
        "xml_preview_path": str(xml_preview_path) if xml_preview_written else "",
        "xml_validation_report_path": str(xml_report_path) if xml_preview_written else "",
    }


def _readiness_md(r, readiness, counts, next_agent) -> str:
    def yn(v: bool) -> str:
        return "✅ yes" if v else "❌ no"

    if readiness["xml_generation_allowed"]:
        verdict = ("All delivery-readiness gates passed. XML generation is allowed "
                   "(production XML is still deferred to v2; only a guarded preview "
                   "may be written).")
    elif readiness["delivery_normalisation_complete"]:
        verdict = ("Delivery normalisation completed for controlled data, but some "
                   "structural gates remain — XML generation is refused.")
    else:
        verdict = ("Projection is not complete and delivery blockers remain. Delivery "
                   "normalisation is incomplete and XML generation is refused.")

    lines = [
        "# Delivery/XML Agent result", "",
        f"Client: {r.get('client_id', '')}  ",
        f"Run: {r.get('run_id', '')}  ",
        f"Target contract: {r.get('target_contract_id', '')}  ",
        f"Agent: **{AGENT} v{AGENT_VERSION}**", "",
        f"> {verdict}", "",
        "## Readiness flags", "",
        f"- delivery_xml_ran: {yn(r['delivery_xml_ran'])}",
        f"- delivery_normalisation_complete: {yn(r['delivery_normalisation_complete'])}",
        f"- xml_generation_allowed: {yn(r['xml_generation_allowed'])}",
        f"- xml_generated: {yn(r['xml_generated'])} (always false in v1)",
        f"- ready_for_xml_delivery: {yn(r['ready_for_xml_delivery'])}", "",
        "## Delivery-readiness gates", "",
    ]
    for g in readiness["gates"]:
        lines.append(f"- {yn(g['passed'])} **{g['gate']}** — {g['reason']}")
    lines += [
        "", "## Delivery frame", "",
        f"- rows: {counts['frame_row_count']}",
        f"- deliverable: {counts['deliverable_row_count']}",
        f"- blocked: {counts['blocked_row_count']}",
        f"- delivery-invalid: {counts['invalid_row_count']}",
        f"- not-required-blank: {counts['not_required_blank_count']}", "",
        "## Delivery issue categories", "",
    ]
    if counts["delivery_blocker_type_counts"]:
        for k in sorted(counts["delivery_blocker_type_counts"]):
            lines.append(f"- {k}: {counts['delivery_blocker_type_counts'][k]}")
    else:
        lines.append("- none")
    lines += ["", "## Recommended next action", "",
              f"- next agent: **{next_agent}**"]
    if readiness["xml_generation_allowed"]:
        lines.append("- Proceed to XML generation (deferred to v2). No production XML "
                     "is produced in this PR.")
    else:
        lines.append("- Resolve the failing gates upstream (operator / config / "
                     "client / projection), then re-run delivery. **No XML is "
                     "produced in this PR.**")
    lines.append("")
    return "\n".join(lines) + "\n"
