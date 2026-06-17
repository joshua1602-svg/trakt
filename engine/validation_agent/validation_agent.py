"""
validation_agent.py
===================

Trakt Validation Agent v1 — the control gate after Transformation.

It consumes the Transformation Agent output package (30 manifest, 31 tape, 32
field contract, 34 lineage, 35 issues), validates transformed canonical values +
the transformation issue classifications, and emits a governed validation
readiness package under ``output/validation/`` (artefacts 40..45) for the
Projection Agent.

Guardrails (enforced by construction):
  * never re-runs raw Gate 1 / source discovery / fuzzy matching;
  * never mutates Onboarding or Transformation artefacts (writes only under
    output/validation/);
  * never projects / produces XML / claims XML readiness;
  * never silently resolves operator decisions or adds enum mappings / defaults
    — unresolved items are carried forward with explicit owner + action.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from engine.validation_agent import rules_adapter as ra

# --------------------------------------------------------------------------- #
# Identity / vocabulary
# --------------------------------------------------------------------------- #

AGENT = "validation_agent"
AGENT_VERSION = "1.0"
STAGE = "post_transformation_validation"

REQUIRED_TX_AGENT = "transformation_agent"

# Validation classifications (per task spec).
VC_PASS = "validation_pass"
VC_WARNING = "validation_warning"
VC_FAILURE = "validation_failure"
VC_OPERATOR = "operator_required"
VC_CONFIG = "config_required"
VC_PROJECTION = "projection_required"
VC_ACCEPTABLE_GAP = "acceptable_downstream_gap"
VC_SEMANTIC = "semantic_derivation_required"

# Downstream owners.
OWN_VALIDATION = "validation"
OWN_TRANSFORMATION = "transformation_validation"
OWN_PROJECTION = "projection"
OWN_OPERATOR = "operator"
OWN_CONFIG = "config_policy"

# Next-agent values.
NEXT_PROJECTION = "projection"
NEXT_REMEDIATION = "operator_config_projection_remediation"

_ISSUE_COLUMNS = [
    "issue_id", "source_issue_id", "severity", "field", "canonical_field",
    "esma_code", "validation_rule_id", "issue_type", "validation_classification",
    "source_value_sample", "transformed_value_sample", "description",
    "blocking_for_validation", "blocking_for_projection", "blocking_for_xml_delivery",
    "recommended_action", "downstream_owner",
]

_RESULT_COLUMNS = [
    "validation_rule_id", "field", "canonical_field", "esma_code", "check_type",
    "status", "severity", "row_count_checked", "failure_count", "warning_count",
    "sample_failures", "blocking_for_validation", "blocking_for_projection", "notes",
]


class TransformationHandoffError(RuntimeError):
    """Raised when the transformation package is missing or not ready for validation."""


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


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    try:
        if isinstance(v, float) and math.isnan(v):
            return True
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s == "" or s.lower() == "nan" or s == "<NA>"


# --------------------------------------------------------------------------- #
# Validate the transformation manifest
# --------------------------------------------------------------------------- #

def validate_transformation_manifest(manifest: dict) -> None:
    """Fail loudly unless the transformation package is ready for validation.

    Required:
        agent == transformation_agent (or equivalent);
        ready_for_validation == true;
        ready_for_xml_delivery == false;
        governance flags (not_raw_source / did_not_rerun_gate1) respected.
    """
    if not isinstance(manifest, dict):
        raise TransformationHandoffError("Transformation manifest is not a JSON object.")

    problems: List[str] = []
    agent = str(manifest.get("agent", "")).lower()
    if REQUIRED_TX_AGENT not in agent and "transformation" not in agent:
        problems.append(f"agent must be a transformation agent, got {manifest.get('agent')!r}")
    if manifest.get("ready_for_validation") is not True:
        problems.append("ready_for_validation must be true — refusing to validate a "
                        "transformation package that is not ready for validation")
    if manifest.get("ready_for_xml_delivery") is True:
        problems.append("ready_for_xml_delivery must be false at the validation stage")
    if manifest.get("not_raw_source") is False:
        problems.append("not_raw_source must not be false (canonical input expected)")
    if manifest.get("did_not_rerun_gate1") is False:
        problems.append("did_not_rerun_gate1 governance flag must be respected")
    if problems:
        raise TransformationHandoffError(
            "Transformation package is not consumable by the Validation Agent:\n  - "
            + "\n  - ".join(problems))


def _resolve_inputs(manifest_path: Path) -> Dict[str, Path]:
    tx_dir = manifest_path.parent                  # .../output/transformation
    output_root = tx_dir.parent                    # .../output
    return {
        "tx_dir": tx_dir,
        "output_root": output_root,
        "transformed_tape": tx_dir / "31_transformed_canonical_tape.csv",
        "tx_contract": tx_dir / "32_transformation_field_contract.csv",
        "tx_lineage": tx_dir / "34_transformation_lineage.json",
        "tx_issues": tx_dir / "35_transformation_issues.csv",
        "handoff_manifest": output_root / "handoff" / "24_onboarding_handoff_manifest.json",
    }


# --------------------------------------------------------------------------- #
# Transformation issue carry-forward + reclassification
# --------------------------------------------------------------------------- #

def classify_transformation_issue(
    tx_issue: Dict[str, Any],
    *,
    regime_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate + reclassify a carried-forward transformation issue.

    Returns the classification fields for a validation issue row. Never silently
    resolves an operator/config/projection item — it is carried forward with an
    explicit downstream owner + recommended action.
    """
    it = str(tx_issue.get("issue_type", "")).strip()
    canonical = tx_issue.get("canonical_field", "")
    rule = regime_index.get(canonical, {})

    # Defaults.
    classification = VC_WARNING
    severity = "warn"
    b_val = False
    b_proj = True
    owner = OWN_TRANSFORMATION
    action = tx_issue.get("recommended_action", "")
    out_type = it

    if it == "pending_projection_rule":
        classification, owner = VC_PROJECTION, OWN_PROJECTION
        severity, action = "info", "implement or defer regime rule at projection"

    elif it == "operator_decision_pending":
        classification, owner = VC_OPERATOR, OWN_OPERATOR
        severity, action = "warn", "resolve operator decision before projection"

    elif it == "semantic_derivation_required":
        classification, owner, out_type = VC_SEMANTIC, OWN_TRANSFORMATION, "semantic_derivation_required"
        severity, action = "warn", "define approved semantic derivation or operator decision"

    elif it == "source_absent":
        # Warning by default; config_required when a default can/should be configured.
        if rule.get("default_allowed"):
            classification, owner = VC_CONFIG, OWN_CONFIG
            action = "configure asset/regime default for this field"
        else:
            classification, owner = VC_WARNING, OWN_TRANSFORMATION
            action = "confirm source absent or supply value"
        severity = "warn"

    elif it == "enum_unmapped":
        # config_required (extend mapping) — failure only if a mandatory field.
        if rule.get("mandatory") and rule.get("enforce_presence"):
            classification, severity, b_val = VC_FAILURE, "error", True
            owner, action = OWN_CONFIG, "add enum mapping for mandatory field"
        else:
            classification, owner = VC_CONFIG, OWN_CONFIG
            severity, action = "warn", "extend enum mapping or confirm at projection"

    elif it in ("invalid_default", "invalid_nd_default"):
        classification, severity, b_val = VC_FAILURE, "error", True
        owner, action = OWN_CONFIG, "fix invalid default in asset/regime config"

    elif it in ("date_parse_failed", "numeric_parse_failed", "boolean_parse_failed"):
        classification, severity, b_val = VC_FAILURE, "error", True
        owner, action = OWN_TRANSFORMATION, "correct source values or parse rule"

    return {
        "validation_classification": classification,
        "issue_type": out_type,
        "severity": severity,
        "blocking_for_validation": b_val,
        "blocking_for_projection": b_proj,
        "downstream_owner": owner,
        "recommended_action": action,
    }


# --------------------------------------------------------------------------- #
# Core build
# --------------------------------------------------------------------------- #

def build_validation_package(
    transformation_manifest_path: str | Path,
    *,
    registry_path: str = "",
    regime_config_path: str = "",
    asset_config_path: str = "",
    enum_config_dir: str = "",
) -> Dict[str, Any]:
    """Consume the transformation package and emit the validation package.

    Returns a dict of artefact paths + the validation manifest. Raises
    :class:`TransformationHandoffError` if the transformation manifest is
    missing/invalid or not ready for validation.
    """
    manifest_path = Path(transformation_manifest_path)
    if not manifest_path.exists():
        raise TransformationHandoffError(
            f"Transformation manifest not found: {manifest_path}")
    tx_manifest = _read_json(manifest_path)
    if tx_manifest is None:
        raise TransformationHandoffError(
            f"Transformation manifest is not valid JSON: {manifest_path}")

    # 1) validate readiness / governance
    validate_transformation_manifest(tx_manifest)

    paths = _resolve_inputs(manifest_path)
    client_id = tx_manifest.get("client_id", "")
    run_id = tx_manifest.get("run_id", "")
    target_contract_id = tx_manifest.get("target_contract_id", "")

    repo_root = Path(__file__).resolve().parents[2]
    registry_path = registry_path or tx_manifest.get("registry_path", "") or str(
        repo_root / "config" / "system" / "fields_registry.yaml")
    if registry_path and not Path(registry_path).is_absolute() and not Path(registry_path).exists():
        registry_path = str(repo_root / registry_path)
    regime_config_path = regime_config_path or tx_manifest.get("regime_config_path", "") or str(
        repo_root / "config" / "regime" / "annex2_delivery_rules.yaml")
    asset_config_path = asset_config_path or tx_manifest.get("asset_config_path", "") or str(
        repo_root / "config" / "asset" / "product_defaults_ERM.yaml")
    enum_config_dir = enum_config_dir or str(repo_root / "config" / "system")

    # 2) load transformed canonical tape (NO source discovery / fuzzy mapping)
    if not paths["transformed_tape"].exists():
        raise TransformationHandoffError(
            f"Transformed canonical tape not found at {paths['transformed_tape']}")
    df = pd.read_csv(paths["transformed_tape"], dtype=str, low_memory=False)
    row_count = int(len(df))
    field_count = int(len(df.columns))

    # 3) load transformation field contract (field-level control layer)
    tx_contract = _read_csv_rows(paths["tx_contract"])

    # 4) load transformation issues
    tx_issues = _read_csv_rows(paths["tx_issues"])

    # config layers
    registry_fields = ra.load_registry_fields(registry_path)
    enum_lib = ra.load_enum_lib(enum_config_dir)
    regime_cfg = _read_yaml(Path(regime_config_path)) or {}
    regime_index = ra.build_regime_index(regime_cfg)

    # 5/6) value-level + cross-field validation -------------------------------
    results: List[Dict[str, Any]] = []
    uncontrolled_error = ""
    try:
        for crow in tx_contract:
            canonical = (crow.get("canonical_field") or "").strip()
            if not canonical or canonical not in df.columns:
                continue
            esma = crow.get("esma_code", "")
            meta = registry_fields.get(canonical, {}) or {}
            fmt = str(meta.get("format", "")).lower()
            enum_name = meta.get("allowed_values", "") or ""
            rule = regime_index.get(canonical, {})
            mandatory = bool(rule.get("mandatory") and rule.get("enforce_presence"))
            results.extend(ra.validate_field(
                df, canonical, fmt, esma_code=esma, regime_rule=rule,
                enum_name=enum_name, enum_lib=enum_lib, mandatory=mandatory))

        results.extend(ra.validate_uniqueness(
            df, ["loan_identifier", "unique_identifier"]))
        results.extend(ra.validate_business_rules(df))
    except Exception as exc:  # uncontrolled parser/type/enum exception
        uncontrolled_error = f"{type(exc).__name__}: {exc}"

    # de-dup results by rule id (a canonical field can appear on >1 contract row)
    seen: set = set()
    dedup_results: List[Dict[str, Any]] = []
    for r in results:
        if r["validation_rule_id"] in seen:
            continue
        seen.add(r["validation_rule_id"])
        dedup_results.append(r)
    results = dedup_results

    # --- build validation issues ---------------------------------------------
    issues: List[Dict[str, Any]] = []
    n = 0

    # 7) carry-forward + reclassify transformation issues
    for ti in tx_issues:
        n += 1
        cls = classify_transformation_issue(ti, regime_index=regime_index)
        issues.append({
            "issue_id": f"VAL-{n:04d}",
            "source_issue_id": ti.get("issue_id", ""),
            "severity": cls["severity"],
            "field": ti.get("field", ""),
            "canonical_field": ti.get("canonical_field", ""),
            "esma_code": ti.get("esma_code", ""),
            "validation_rule_id": "",
            "issue_type": cls["issue_type"],
            "validation_classification": cls["validation_classification"],
            "source_value_sample": ti.get("source_value_sample", ""),
            "transformed_value_sample": ti.get("transformed_value_sample", ""),
            "description": ti.get("description", ""),
            "blocking_for_validation": cls["blocking_for_validation"],
            "blocking_for_projection": cls["blocking_for_projection"],
            "blocking_for_xml_delivery": True,
            "recommended_action": cls["recommended_action"],
            "downstream_owner": cls["downstream_owner"],
        })

    # value-level failures/warnings -> new validation issues
    _check_to_issue_type = {
        "presence": "missing_required_value",
        "type_date": "invalid_date",
        "type_numeric": "invalid_number",
        "type_boolean": "invalid_boolean",
        "enum": "invalid_enum",
        "country_code": "invalid_country_code",
        "lei": "invalid_lei",
        "identifier_uniqueness": "duplicate_identifier",
        "rate_bounds": "invalid_rate",
        "regex": "invalid_type",
        "cross_field_rule": "cross_field_rule_failed",
    }
    for r in results:
        if r["status"] in ("pass", "not_checked"):
            continue
        n += 1
        is_fail = r["status"] == "fail"
        issues.append({
            "issue_id": f"VAL-{n:04d}",
            "source_issue_id": "",
            "severity": r["severity"],
            "field": r["field"],
            "canonical_field": r["canonical_field"],
            "esma_code": r["esma_code"],
            "validation_rule_id": r["validation_rule_id"],
            "issue_type": _check_to_issue_type.get(r["check_type"], "cross_field_rule_failed"),
            "validation_classification": VC_FAILURE if is_fail else VC_WARNING,
            "source_value_sample": "",
            "transformed_value_sample": r["sample_failures"],
            "description": (r["notes"] or r["check_type"]) +
                           f" — {r['failure_count']} failure(s), {r['warning_count']} warning(s)",
            "blocking_for_validation": bool(r["blocking_for_validation"]),
            "blocking_for_projection": bool(r["blocking_for_projection"]),
            "blocking_for_xml_delivery": True,
            "recommended_action": ("correct values / config" if is_fail
                                   else "review warning before projection"),
            "downstream_owner": OWN_VALIDATION if is_fail else OWN_PROJECTION,
        })

    # --- readiness -----------------------------------------------------------
    counts = _count(issues, results)
    readiness = compute_readiness(
        tape_loaded=row_count > 0,
        checks_ran=not uncontrolled_error,
        uncontrolled_error=bool(uncontrolled_error),
        blocking_validation_issue_count=counts["blocking_for_validation_count"],
        operator_required_count=counts["operator_required_count"],
        config_required_count=counts["config_required_count"],
        semantic_required_count=counts["semantic_required_count"],
        projection_required_count=counts["projection_required_count"],
    )

    # --- write artefacts -----------------------------------------------------
    out_dir = paths["output_root"] / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    return _write_artefacts(
        out_dir=out_dir, df=df, results=results, issues=issues, readiness=readiness,
        counts=counts, tx_manifest=tx_manifest, manifest_path=manifest_path,
        paths=paths, client_id=client_id, run_id=run_id,
        target_contract_id=target_contract_id, row_count=row_count,
        field_count=field_count, uncontrolled_error=uncontrolled_error,
        config_paths={
            "registry_path": registry_path,
            "regime_config_path": regime_config_path,
            "asset_config_path": asset_config_path,
        })


# --------------------------------------------------------------------------- #
# Counts + readiness
# --------------------------------------------------------------------------- #

def _count(issues: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, int]:
    by_cls: Dict[str, int] = {}
    for i in issues:
        c = i["validation_classification"]
        by_cls[c] = by_cls.get(c, 0) + 1
    pass_count = sum(1 for r in results if r["status"] == "pass")
    warn_count = sum(1 for r in results if r["status"] == "warning")
    fail_count = sum(1 for r in results if r["status"] == "fail")
    return {
        "validation_rule_count": len(results),
        "validation_pass_count": pass_count,
        "validation_warning_count": warn_count,
        "validation_failure_count": fail_count,
        "blocking_for_validation_count": sum(1 for i in issues if i["blocking_for_validation"]),
        "blocking_for_projection_count": sum(1 for i in issues if i["blocking_for_projection"]),
        "operator_required_count": by_cls.get(VC_OPERATOR, 0),
        "config_required_count": by_cls.get(VC_CONFIG, 0),
        "projection_required_count": by_cls.get(VC_PROJECTION, 0),
        "semantic_required_count": by_cls.get(VC_SEMANTIC, 0),
        "validation_warning_issue_count": by_cls.get(VC_WARNING, 0),
        "validation_failure_issue_count": by_cls.get(VC_FAILURE, 0),
        "acceptable_downstream_gap_count": by_cls.get(VC_ACCEPTABLE_GAP, 0),
        "classification_counts": by_cls,
        "issue_count": len(issues),
    }


def compute_readiness(
    *,
    tape_loaded: bool,
    checks_ran: bool,
    uncontrolled_error: bool,
    blocking_validation_issue_count: int,
    operator_required_count: int,
    config_required_count: int,
    semantic_required_count: int,
    projection_required_count: int,
    projection_deferral_allowed: bool = False,
) -> Dict[str, bool]:
    """Three distinct readiness flags (validation / projection / xml)."""
    ready_for_validation_complete = bool(
        tape_loaded
        and checks_ran
        and not uncontrolled_error
        and blocking_validation_issue_count == 0
    )
    ready_for_projection = bool(
        ready_for_validation_complete
        and operator_required_count == 0
        and config_required_count == 0
        and semantic_required_count == 0
        and (projection_required_count == 0 or projection_deferral_allowed)
    )
    return {
        "ready_for_validation_complete": ready_for_validation_complete,
        "ready_for_projection": ready_for_projection,
        "ready_for_xml_delivery": False,
    }


# --------------------------------------------------------------------------- #
# Artefact writers
# --------------------------------------------------------------------------- #

def _write_artefacts(
    *, out_dir: Path, df: pd.DataFrame, results: List[Dict[str, Any]],
    issues: List[Dict[str, Any]], readiness: Dict[str, bool], counts: Dict[str, int],
    tx_manifest: dict, manifest_path: Path, paths: Dict[str, Path], client_id: str,
    run_id: str, target_contract_id: str, row_count: int, field_count: int,
    uncontrolled_error: str, config_paths: Dict[str, str],
) -> Dict[str, Any]:

    # 41 — validation results (csv + json)
    results_csv = out_dir / "41_validation_results.csv"
    results_json = out_dir / "41_validation_results.json"
    with open(results_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_RESULT_COLUMNS)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in _RESULT_COLUMNS})
    results_json.write_text(json.dumps(
        {"row_count": row_count, "rule_count": len(results), "rows": results},
        indent=2, default=str), encoding="utf-8")

    # 43 — validation issues (csv + json)
    issues_csv = out_dir / "43_validation_issues.csv"
    issues_json = out_dir / "43_validation_issues.json"
    with open(issues_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_ISSUE_COLUMNS)
        w.writeheader()
        for r in issues:
            w.writerow({k: r.get(k, "") for k in _ISSUE_COLUMNS})
    issue_type_counts: Dict[str, int] = {}
    for i in issues:
        issue_type_counts[i["issue_type"]] = issue_type_counts.get(i["issue_type"], 0) + 1
    issues_json.write_text(json.dumps(
        {"issue_count": len(issues),
         "classification_counts": counts["classification_counts"],
         "issue_type_counts": issue_type_counts,
         "rows": issues}, indent=2, default=str), encoding="utf-8")

    # 44 — validation lineage (extend transformation lineage)
    lineage_path = out_dir / "44_validation_lineage.json"
    tx_lineage = _read_json(paths["tx_lineage"]) or {}
    issues_by_field: Dict[str, List[str]] = {}
    for i in issues:
        key = i["canonical_field"] or i["field"]
        issues_by_field.setdefault(key, []).append(i["issue_id"])
    val_rows = []
    for r in results:
        key = r["canonical_field"] or r["field"]
        val_rows.append({
            "validation_rule_id": r["validation_rule_id"],
            "field": r["field"],
            "canonical_field": r["canonical_field"],
            "check_type": r["check_type"],
            "status": r["status"],
            "issue_ids": issues_by_field.get(key, []),
            "input_artifact": "31_transformed_canonical_tape.csv",
            "output_artifact": "41_validation_results.csv",
        })
    lineage_path.write_text(json.dumps({
        "client_id": client_id, "run_id": run_id,
        "target_contract_id": target_contract_id,
        "transformation_lineage": tx_lineage.get("transformation_lineage", []),
        "transformation_lineage_source": "34_transformation_lineage.json",
        "onboarding_lineage": tx_lineage.get("onboarding_lineage", []),
        "validation_lineage": val_rows,
    }, indent=2, default=str), encoding="utf-8")

    # 42 — readiness (json + md)
    readiness_json = out_dir / "42_validation_readiness.json"
    readiness_md = out_dir / "42_validation_readiness.md"
    next_agent = NEXT_PROJECTION if readiness["ready_for_projection"] else NEXT_REMEDIATION
    readiness_doc = {
        "agent": AGENT, "agent_version": AGENT_VERSION,
        "client_id": client_id, "run_id": run_id,
        "target_contract_id": target_contract_id, "created_at": _now(),
        "tape_loaded": row_count > 0, "row_count": row_count, "field_count": field_count,
        "uncontrolled_error": uncontrolled_error,
        "next_agent": next_agent,
        **counts, **readiness,
    }
    readiness_doc.pop("classification_counts", None)
    readiness_json.write_text(json.dumps(readiness_doc, indent=2, default=str), encoding="utf-8")
    readiness_md.write_text(_readiness_md(readiness_doc, readiness, counts, next_agent),
                            encoding="utf-8")

    # 45 — validation summary (md)
    summary_md = out_dir / "45_validation_summary.md"
    summary_md.write_text(_summary_md(
        client_id, run_id, target_contract_id, row_count, field_count,
        results, counts, readiness, next_agent), encoding="utf-8")

    # 40 — validation manifest (json + yaml)
    manifest_json = out_dir / "40_validation_manifest.json"
    manifest_yaml = out_dir / "40_validation_manifest.yaml"
    manifest = {
        "agent": AGENT, "agent_version": AGENT_VERSION, "stage": STAGE,
        "created_at": _now(),
        "client_id": client_id, "run_id": run_id,
        "target_contract_id": target_contract_id,

        # governance
        "consumes_transformation_package": True,
        "not_raw_source": True, "did_not_rerun_gate1": True,
        "did_not_source_match": True, "did_not_mutate_upstream_artefacts": True,
        "performed_projection": False, "performed_xml_delivery": False,

        # inputs
        "input_transformation_manifest_path": str(manifest_path),
        "input_transformed_tape_path": str(paths["transformed_tape"]),
        "input_transformation_contract_path": str(paths["tx_contract"]),
        "input_transformation_issues_path": str(paths["tx_issues"]),
        "input_transformation_lineage_path": str(paths["tx_lineage"]),
        **config_paths,

        # outputs
        "output_validation_results_path": str(results_csv),
        "output_validation_issues_path": str(issues_csv),
        "output_validation_readiness_path": str(readiness_json),
        "output_validation_lineage_path": str(lineage_path),
        "output_validation_summary_path": str(summary_md),

        # counts
        "row_count": row_count, "field_count": field_count,
        "validation_rule_count": counts["validation_rule_count"],
        "validation_pass_count": counts["validation_pass_count"],
        "validation_warning_count": counts["validation_warning_count"],
        "validation_failure_count": counts["validation_failure_count"],
        "blocking_for_validation_count": counts["blocking_for_validation_count"],
        "blocking_for_projection_count": counts["blocking_for_projection_count"],
        "operator_required_count": counts["operator_required_count"],
        "config_required_count": counts["config_required_count"],
        "projection_required_count": counts["projection_required_count"],
        "semantic_required_count": counts["semantic_required_count"],
        "issue_count": counts["issue_count"],
        "issue_type_counts": issue_type_counts,
        "validation_classification_counts": counts["classification_counts"],

        # readiness
        **readiness,
        "next_agent": next_agent,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    manifest_yaml.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    return {
        "manifest": manifest,
        "readiness": readiness_doc,
        "validation_dir": str(out_dir),
        "manifest_json_path": str(manifest_json),
        "manifest_yaml_path": str(manifest_yaml),
        "results_csv_path": str(results_csv),
        "results_json_path": str(results_json),
        "readiness_json_path": str(readiness_json),
        "readiness_md_path": str(readiness_md),
        "issues_csv_path": str(issues_csv),
        "issues_json_path": str(issues_json),
        "lineage_path": str(lineage_path),
        "summary_md_path": str(summary_md),
    }


def _readiness_md(r: Dict[str, Any], readiness: Dict[str, bool],
                  counts: Dict[str, int], next_agent: str) -> str:
    def yn(v: bool) -> str:
        return "✅ yes" if v else "❌ no"

    if readiness["ready_for_validation_complete"] and not readiness["ready_for_projection"]:
        verdict = ("Validation completed successfully for the transformed canonical "
                   "tape. There are no validation-blocking issues, but projection is "
                   "not ready because operator/config/projection items remain "
                   "unresolved.")
    elif readiness["ready_for_projection"]:
        verdict = ("Validation completed successfully. The transformed canonical tape "
                   "is ready for the Projection Agent.")
    else:
        verdict = ("Validation found blocking failures in the transformed canonical "
                   "tape. It is NOT ready to pass to Projection until they are fixed.")

    lines = [
        "# Validation Agent result", "",
        f"Client: {r.get('client_id', '')}  ",
        f"Run: {r.get('run_id', '')}  ",
        f"Target contract: {r.get('target_contract_id', '')}  ",
        f"Agent: **{AGENT} v{AGENT_VERSION}**", "",
        f"> {verdict}", "",
        "## Readiness flags", "",
        f"- ready_for_validation_complete: {yn(readiness['ready_for_validation_complete'])}",
        f"- ready_for_projection: {yn(readiness['ready_for_projection'])}",
        f"- ready_for_xml_delivery: {yn(readiness['ready_for_xml_delivery'])}", "",
        "## Issues", "",
        f"- total issues: {counts['issue_count']}",
        f"- blocking validation issues: {counts['blocking_for_validation_count']}",
        f"- blocking projection issues: {counts['blocking_for_projection_count']}",
        f"- operator required: {counts['operator_required_count']}",
        f"- config required: {counts['config_required_count']}",
        f"- projection required: {counts['projection_required_count']}",
        f"- semantic derivation required: {counts['semantic_required_count']}", "",
        "## Recommended next action", "",
        f"- next agent: **{next_agent}**",
    ]
    if readiness["ready_for_projection"]:
        lines.append("- Hand the transformed tape to the Projection Agent.")
    elif readiness["ready_for_validation_complete"]:
        lines.append("- Resolve operator / config / projection items, then re-run "
                     "projection readiness. Validation itself is complete.")
    else:
        lines.append("- Fix validation-blocking failures, re-run Transformation, "
                     "then re-validate.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _summary_md(client_id, run_id, target_contract_id, row_count, field_count,
                results, counts, readiness, next_agent) -> str:
    fails = [r for r in results if r["status"] == "fail"]
    warns = [r for r in results if r["status"] == "warning"]
    lines = [
        "# Validation summary", "",
        f"Client: {client_id} · Run: {run_id} · Contract: {target_contract_id}  ",
        f"Rows validated: {row_count} · Fields: {field_count}  ",
        f"Checks run: {counts['validation_rule_count']} "
        f"(pass {counts['validation_pass_count']}, "
        f"warning {counts['validation_warning_count']}, "
        f"fail {counts['validation_failure_count']})", "",
        "## Readiness", "",
        f"- ready_for_validation_complete: {readiness['ready_for_validation_complete']}",
        f"- ready_for_projection: {readiness['ready_for_projection']}",
        f"- ready_for_xml_delivery: {readiness['ready_for_xml_delivery']}",
        f"- next agent: {next_agent}", "",
        "## Issue classification mix", "",
    ]
    for k in sorted(counts["classification_counts"]):
        lines.append(f"- {k}: {counts['classification_counts'][k]}")
    lines += ["", "## Validation failures (blocking)", ""]
    if fails:
        for r in fails[:50]:
            lines.append(f"- `{r['validation_rule_id']}` {r['check_type']} on "
                         f"`{r['canonical_field'] or r['field']}` — "
                         f"{r['failure_count']} failure(s); e.g. {r['sample_failures']}")
    else:
        lines.append("- none")
    lines += ["", "## Validation warnings", ""]
    if warns:
        for r in warns[:50]:
            lines.append(f"- `{r['validation_rule_id']}` {r['check_type']} on "
                         f"`{r['canonical_field'] or r['field']}` — "
                         f"{r['warning_count']} warning(s)")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines) + "\n"
