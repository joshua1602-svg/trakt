"""
transformation_agent.py
=======================

Trakt Transformation Agent v1 — the deterministic bridge between the Onboarding
Agent handoff package and the Validation Agent.

It consumes the governed canonical onboarding handoff package (24 manifest, 26
field contract, 27 lineage) and the central canonical lender tape (18), applies
deterministic transformations by reusing the existing ``engine.gate_2_transform``
logic (through :mod:`engine.transformation_agent.gate2_adapter`), and emits a
normalized, validation-ready transformed canonical package under
``output/transformation/``::

    30_transformation_manifest.json / .yaml
    31_transformed_canonical_tape.csv / .json
    32_transformation_field_contract.csv / .json
    33_transformation_readiness.json / .md
    34_transformation_lineage.json
    35_transformation_issues.csv / .json

Guardrails (enforced by construction):
  * never re-runs raw Gate 1 on the central tape;
  * never performs fuzzy source matching / source discovery;
  * never mutates Onboarding Agent artefacts (writes only under transformation/);
  * never projects to Annex 2 XML and never claims XML readiness — projection
    gaps are classified ``pending_projection_rule`` for the Projection Agent.
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

from engine.onboarding_agent import target_contract_completion as _tcc


def _disposition_action(disposition: str) -> str:
    """Map an onboarding disposition to the Transformation execution action."""
    return _tcc.transformation_action_for_disposition(disposition)

from engine.onboarding_agent import onboarding_handoff as oh
from engine.transformation_agent import gate2_adapter as g2

# --------------------------------------------------------------------------- #
# Identity / vocabulary
# --------------------------------------------------------------------------- #

AGENT = "transformation_agent"
AGENT_VERSION = "1.0"
STAGE = "post_onboarding_transformation"
NEXT_AGENT = "validation"

REQUIRED_HANDOFF_TYPE = "canonical_onboarding_package"
REQUIRED_NEXT_AGENT = "transformation_validation"

# Per-field transformation status vocabulary.
TS_TRANSFORMED = "transformed"
TS_COPIED = "copied"
TS_DEFAULT = "default_materialised"
TS_ND_DEFAULT = "nd_default_materialised"
TS_CONFIGURED_STATIC = "configured_static_materialised"
TS_SOURCE_CONTEXT = "source_context_materialised"
TS_RUN_CONTEXT = "run_context_materialised"
TS_ENUM_NORMALIZED = "enum_normalized"
TS_TYPE_NORMALIZED = "type_normalized"
TS_DERIVED = "derived"
TS_PENDING_PROJECTION = "pending_projection_rule"
TS_SOURCE_ABSENT = "source_absent"
TS_SEMANTIC_DERIVATION = "semantic_derivation_required"
TS_OPERATOR_PENDING = "operator_decision_pending"
TS_VALIDATION_REQUIRED = "validation_required"
TS_NOT_APPLICABLE = "not_applicable"
TS_FAILED = "failed_transformation"

# Issue types.
IT_MISSING_REQUIRED = "missing_required_value"
IT_INVALID_DEFAULT = "invalid_default"
IT_INVALID_ND_DEFAULT = "invalid_nd_default"
IT_ENUM_UNMAPPED = "enum_unmapped"
IT_DATE_PARSE = "date_parse_failed"
IT_NUMERIC_PARSE = "numeric_parse_failed"
IT_BOOLEAN_PARSE = "boolean_parse_failed"
IT_SEMANTIC_DERIVATION = "semantic_derivation_required"
IT_OPERATOR_PENDING = "operator_decision_pending"
IT_PENDING_PROJECTION = "pending_projection_rule"
IT_SOURCE_ABSENT = "source_absent"

# Downstream owners (mirror the onboarding handoff vocabulary).
OWN_TRANSFORMATION = "transformation_validation"
OWN_VALIDATION = "validation"
OWN_PROJECTION = "projection"
OWN_OPERATOR = "operator"

_ISSUE_COLUMNS = [
    "issue_id", "severity", "field", "canonical_field", "esma_code",
    "issue_type", "source_value_sample", "transformed_value_sample",
    "description", "blocking_for_validation", "blocking_for_projection",
    "recommended_action", "downstream_owner",
]

_CONTRACT_COLUMNS = [
    "target_contract_id", "esma_code", "target_field", "canonical_field", "domain",
    "coverage_status", "handoff_classification", "handoff_downstream_owner",
    # Onboarding target-field disposition carried from 26/29 — Transformation
    # EXECUTES the disposition rather than rediscovering the field treatment.
    "field_disposition", "disposition_source", "disposition_action",
    "transformation_status", "transformed_value_sample", "value_source",
    "type_cast", "enum_map_used", "parse_rule", "issue_id",
    "blocking_for_validation", "blocking_for_projection", "downstream_owner", "notes",
]


class HandoffValidationError(RuntimeError):
    """Raised when the onboarding handoff manifest is missing or not consumable."""


# --------------------------------------------------------------------------- #
# Small IO helpers
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


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    if value is pd.NA:
        return True
    s = str(value).strip()
    return s == "" or s.lower() == "nan" or s == "<NA>"


def _clean(value: Any) -> str:
    """Render a cell value for CSV/JSON output (NA / NaN -> empty string)."""
    if _is_blank(value):
        return ""
    return str(value)


# --------------------------------------------------------------------------- #
# 1) Load + validate the onboarding handoff manifest
# --------------------------------------------------------------------------- #

def validate_handoff_manifest(manifest: dict) -> None:
    """Fail loudly unless the manifest is a consumable canonical handoff package.

    Required:
        handoff_type == canonical_onboarding_package
        not_raw_source == true
        ready_for_transformation_validation == true
        next_agent == transformation_validation
    """
    if not isinstance(manifest, dict):
        raise HandoffValidationError("Handoff manifest is not a JSON object.")

    problems: List[str] = []
    if manifest.get("handoff_type") != REQUIRED_HANDOFF_TYPE:
        problems.append(
            f"handoff_type must be '{REQUIRED_HANDOFF_TYPE}', "
            f"got {manifest.get('handoff_type')!r}")
    if manifest.get("not_raw_source") is not True:
        problems.append("not_raw_source must be true (central tape is canonical, "
                        "not raw source input)")
    if manifest.get("next_agent") != REQUIRED_NEXT_AGENT:
        problems.append(
            f"next_agent must be '{REQUIRED_NEXT_AGENT}', "
            f"got {manifest.get('next_agent')!r}")
    if manifest.get("ready_for_transformation_validation") is not True:
        problems.append("ready_for_transformation_validation must be true — refusing "
                        "to consume an onboarding package that is not ready for "
                        "transformation & validation")
    if problems:
        raise HandoffValidationError(
            "Onboarding handoff manifest is not consumable by the Transformation "
            "Agent:\n  - " + "\n  - ".join(problems))


def _resolve_inputs(manifest_path: Path) -> Dict[str, Path]:
    """Resolve the handoff/central paths relative to the manifest location.

    Robust to absolute paths recorded in the manifest by always preferring the
    on-disk artefacts next to the manifest (handoff dir) and the sibling
    ``central`` directory under the run output root.
    """
    handoff_dir = manifest_path.parent
    output_root = handoff_dir.parent  # .../output

    return {
        "handoff_dir": handoff_dir,
        "output_root": output_root,
        "central_tape": output_root / "central" / "18_central_lender_tape.csv",
        "field_contract_csv": handoff_dir / "26_onboarding_handoff_field_contract.csv",
        "field_contract_json": handoff_dir / "26_onboarding_handoff_field_contract.json",
        "lineage": handoff_dir / "27_onboarding_handoff_lineage.json",
    }


# --------------------------------------------------------------------------- #
# 3) Field contract
# --------------------------------------------------------------------------- #

def _load_field_contract(paths: Dict[str, Path]) -> List[Dict[str, Any]]:
    j = _read_json(paths["field_contract_json"])
    if j and isinstance(j, dict) and j.get("rows"):
        return j["rows"]
    rows: List[Dict[str, Any]] = []
    csv_path = paths["field_contract_csv"]
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    return rows


# --------------------------------------------------------------------------- #
# Value resolution for downstream defaults / static / ND values
# --------------------------------------------------------------------------- #

def _build_regime_defaults(regime_cfg: Optional[dict]) -> Dict[str, Dict[str, Any]]:
    """Map esma_code -> {default_value, nd_allowed, projected_source_field, has_enum_map}."""
    out: Dict[str, Dict[str, Any]] = {}
    for code, rule in ((regime_cfg or {}).get("field_rules", {}) or {}).items():
        rule = rule or {}
        out[str(code)] = {
            "default_value": rule.get("default_value", ""),
            "default_allowed": bool(rule.get("default_allowed", False)),
            "nd_allowed": rule.get("nd_allowed", []) or [],
            "projected_source_field": rule.get("projected_source_field", ""),
            "has_enum_map": bool((rule.get("transform") or {}).get("enum_map")),
        }
    return out


def _resolve_value(
    canonical: str,
    esma_code: str,
    contract_value: Any,
    asset_defaults: Dict[str, Any],
    asset_nd_defaults: Dict[str, Any],
    regime_defaults: Dict[str, Dict[str, Any]],
) -> Tuple[str, str]:
    """Resolve a deterministic value for a default/static/ND field.

    Precedence (deterministic, no inference):
      1. value already carried in the handoff contract (selected_value_sample);
      2. asset-class config default (product_defaults_ERM.yaml);
      3. asset-class ND default;
      4. regime config default_value (annex2_delivery_rules.yaml).

    Returns ``(value, value_source)``; value is "" when nothing resolves.
    """
    if not _is_blank(contract_value):
        return str(contract_value).strip(), "handoff_contract"
    if canonical in asset_defaults and not _is_blank(asset_defaults[canonical]):
        return str(asset_defaults[canonical]).strip(), "asset_config_default"
    if canonical in asset_nd_defaults and not _is_blank(asset_nd_defaults[canonical]):
        return str(asset_nd_defaults[canonical]).strip(), "asset_config_nd_default"
    rd = regime_defaults.get(str(esma_code), {})
    if not _is_blank(rd.get("default_value")):
        return str(rd["default_value"]).strip(), "regime_config_default"
    return "", ""


# --------------------------------------------------------------------------- #
# Issue helpers
# --------------------------------------------------------------------------- #

class _IssueLog:
    def __init__(self) -> None:
        self._rows: List[Dict[str, Any]] = []
        self._n = 0

    def add(self, *, severity: str, field: str, canonical_field: str, esma_code: str,
            issue_type: str, source_value_sample: str = "", transformed_value_sample: str = "",
            description: str = "", blocking_for_validation: bool = False,
            blocking_for_projection: bool = False, recommended_action: str = "",
            downstream_owner: str = OWN_TRANSFORMATION) -> str:
        self._n += 1
        issue_id = f"TX-{self._n:04d}"
        self._rows.append({
            "issue_id": issue_id,
            "severity": severity,
            "field": field,
            "canonical_field": canonical_field,
            "esma_code": esma_code,
            "issue_type": issue_type,
            "source_value_sample": _clean(source_value_sample),
            "transformed_value_sample": _clean(transformed_value_sample),
            "description": description,
            "blocking_for_validation": bool(blocking_for_validation),
            "blocking_for_projection": bool(blocking_for_projection),
            "recommended_action": recommended_action,
            "downstream_owner": downstream_owner,
        })
        return issue_id

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return self._rows


# --------------------------------------------------------------------------- #
# Core build
# --------------------------------------------------------------------------- #

def build_transformation_package(
    handoff_manifest_path: str | Path,
    *,
    asset_config_path: str = "",
    regime_config_path: str = "",
    registry_path: str = "",
    enum_mapping_path: str = "",
    dayfirst: bool = True,
) -> Dict[str, Any]:
    """Consume the onboarding handoff package and emit the transformation package.

    Returns a dict of artefact paths + the transformation manifest. Raises
    :class:`HandoffValidationError` if the handoff manifest is missing/invalid or
    refuses transformation/validation.
    """
    manifest_path = Path(handoff_manifest_path)
    if not manifest_path.exists():
        raise HandoffValidationError(
            f"Onboarding handoff manifest not found: {manifest_path}")

    handoff = _read_json(manifest_path)
    if handoff is None:
        raise HandoffValidationError(
            f"Onboarding handoff manifest is not valid JSON: {manifest_path}")

    # --- 1) validate handoff identity / readiness ---
    validate_handoff_manifest(handoff)

    paths = _resolve_inputs(manifest_path)
    client_id = handoff.get("client_id", "")
    run_id = handoff.get("run_id", "")
    target_contract_id = handoff.get("target_contract_id", "")

    # --- resolve config inputs (default to the handoff-recorded configs) ---
    repo_root = Path(__file__).resolve().parents[2]
    asset_config_path = asset_config_path or handoff.get("asset_config_path", "") or str(
        repo_root / "config" / "asset" / "product_defaults_ERM.yaml")
    regime_config_path = regime_config_path or handoff.get("regime_config_path", "") or str(
        repo_root / "config" / "regime" / "annex2_delivery_rules.yaml")
    registry_path = registry_path or handoff.get("registry_path", "") or str(
        repo_root / "config" / "system" / "fields_registry.yaml")
    if registry_path and not Path(registry_path).is_absolute() and not Path(registry_path).exists():
        registry_path = str(repo_root / registry_path)
    enum_mapping_path = enum_mapping_path or str(
        repo_root / "config" / "system" / "enum_mapping.yaml")

    asset_cfg = _read_yaml(Path(asset_config_path)) or {}
    asset_defaults = asset_cfg.get("defaults", {}) or {}
    asset_nd_defaults = asset_cfg.get("nd_defaults", {}) or {}
    regime_cfg = _read_yaml(Path(regime_config_path)) or {}
    regime_defaults = _build_regime_defaults(regime_cfg)

    # --- 2) load the central canonical tape (NO Gate 1 re-run) ---
    if not paths["central_tape"].exists():
        raise HandoffValidationError(
            f"Central canonical tape not found at {paths['central_tape']} — the "
            "Transformation Agent consumes the onboarding central tape and never "
            "re-runs raw source discovery.")
    df = pd.read_csv(paths["central_tape"], dtype=str, low_memory=False)
    central_row_count = int(len(df))

    # --- 3) load the handoff field contract (the control layer) ---
    contract_rows = _load_field_contract(paths)

    # --- 4/5/6/7) apply deterministic Gate 2 transformations ---------------- #
    registry_fields = g2.load_registry_fields(registry_path)
    issues = _IssueLog()

    # 6a. Type-normalise the source columns already present in the tape. Gate 2
    #     treats ND codes as missing here; ND defaults are (re)materialised below.
    type_report = g2.normalize_types(df, registry_fields, dayfirst=dayfirst)

    # 6b/5. Materialise asset-class + ND defaults (contract-driven), filling only
    #       blanks. This adds canonical columns the tape does not yet carry.
    field_results = _materialise_fields(
        df, contract_rows, asset_defaults, asset_nd_defaults, regime_defaults,
        registry_fields, issues)

    # 7. Apply Gate 2 config-driven asset defaults (reuse) as a backstop fill.
    g2.apply_defaults(df, asset_defaults)

    # 8. Canonical enum normalisation (internal standardisation; not projection).
    enum_report = g2.normalize_enums(df, asset_cfg)

    # --- surface uncontrolled type/parse failures as issues ---
    _record_parse_issues(type_report, registry_fields, contract_rows, issues)

    # --- finalise per-field contract + transformation status ---
    contract_out, status_counts = _finalise_contract(
        df, contract_rows, field_results, enum_report, type_report,
        registry_fields, regime_defaults, issues)

    # --- readiness ---
    blocking_validation_issues = sum(
        1 for r in issues.rows if r["blocking_for_validation"])
    blocking_decision_count = int(handoff.get("blocking_decision_count", 0) or 0)
    pending_projection = status_counts.get(TS_PENDING_PROJECTION, 0)
    semantic_required = status_counts.get(TS_SEMANTIC_DERIVATION, 0)
    operator_pending = status_counts.get(TS_OPERATOR_PENDING, 0)

    readiness = compute_readiness(
        central_loaded=central_row_count > 0,
        handoff_valid=True,
        blocking_decision_count=blocking_decision_count,
        blocking_validation_issue_count=blocking_validation_issues,
        pending_projection_count=pending_projection,
        semantic_derivation_count=semantic_required,
        operator_pending_count=operator_pending,
    )

    # --- write artefacts ---
    out_dir = paths["output_root"] / "transformation"
    out_dir.mkdir(parents=True, exist_ok=True)

    artefacts = _write_artefacts(
        out_dir=out_dir,
        df=df,
        contract_out=contract_out,
        issues=issues.rows,
        readiness=readiness,
        status_counts=status_counts,
        handoff=handoff,
        paths=paths,
        client_id=client_id,
        run_id=run_id,
        target_contract_id=target_contract_id,
        central_row_count=central_row_count,
        type_report=type_report,
        enum_report=enum_report,
        config_paths={
            "asset_config_path": asset_config_path,
            "regime_config_path": regime_config_path,
            "registry_path": registry_path,
        },
    )
    return artefacts


# --------------------------------------------------------------------------- #
# Field materialisation
# --------------------------------------------------------------------------- #

def _materialise_run_context(
    df: pd.DataFrame, tf: str, canonical: str, esma_code: str,
    contract_value: Any, cls: str, issues: _IssueLog,
) -> Dict[str, Any]:
    """Materialise a portfolio-level run/source context value into every row.

    The value is normalised to ISO ``YYYY-MM-DD``; an unparseable value is a
    controlled date_parse_failed issue (Validation will then fail), never guessed.
    """
    from engine.onboarding_agent import run_context as rc

    raw = "" if _is_blank(contract_value) else str(contract_value).strip()
    iso = rc.normalize_to_iso(raw) if raw else None
    source = ("source_context" if cls == oh.HC_SOURCE_CONTEXT_MAPPED else "run_context")

    if not iso:
        # No usable context value — surface, never fabricate.
        issue_id = issues.add(
            severity="warning", field=tf, canonical_field=canonical, esma_code=esma_code,
            issue_type=IT_SOURCE_ABSENT,
            description="run/source context value missing or not ISO-parseable",
            blocking_for_validation=False, blocking_for_projection=True,
            recommended_action="supply a valid reporting cut-off date in source/config",
            downstream_owner=OWN_TRANSFORMATION)
        return {"value_source": "", "materialised": False, "value": "",
                "issue_id": issue_id, "absent": True, "run_context": True, "source": source}

    if canonical not in df.columns:
        df[canonical] = pd.NA
    df[canonical] = iso  # one portfolio-level value for every row
    return {"value_source": f"handoff_{source}", "materialised": True,
            "value": iso, "run_context": True, "source": source}


def _materialise_asset_override(
    df: pd.DataFrame,
    tf: str,
    canonical: str,
    asset_defaults: Dict[str, Any],
    asset_nd_defaults: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Materialise an ASSET-config default into a field the handoff did not own.

    Only fires when the asset config explicitly defines a value for ``canonical``
    (``defaults`` or ``nd_defaults``). Fills blanks only; never overwrites an
    existing source value. Returns ``None`` when the asset config has nothing for
    this field, so non-ERM / traditional configs are completely unaffected.
    """
    if canonical in asset_defaults and not _is_blank(asset_defaults[canonical]):
        value = str(asset_defaults[canonical]).strip()
        source, is_nd = "asset_config_default", False
    elif canonical in asset_nd_defaults and not _is_blank(asset_nd_defaults[canonical]):
        value = str(asset_nd_defaults[canonical]).strip()
        source, is_nd = "asset_config_nd_default", True
    else:
        return None  # asset config does not define this field — leave as-is

    if canonical not in df.columns:
        df[canonical] = pd.NA
    col = df[canonical].astype("string")
    blank_mask = col.isna() | (col.str.strip() == "") | (col.str.strip() == "<NA>")
    if blank_mask.any():
        df.loc[blank_mask, canonical] = value
    return {"value_source": source, "materialised": True, "value": value,
            "asset_override": True, "nd": is_nd}


def _materialise_fields(
    df: pd.DataFrame,
    contract_rows: List[Dict[str, Any]],
    asset_defaults: Dict[str, Any],
    asset_nd_defaults: Dict[str, Any],
    regime_defaults: Dict[str, Dict[str, Any]],
    registry_fields: Dict[str, Any],
    issues: _IssueLog,
) -> Dict[str, Dict[str, Any]]:
    """Materialise downstream default / static / ND values onto the tape.

    Returns ``{target_field: {value_source, materialised, value}}``. Only blanks
    are filled; existing source values are never overwritten.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for row in contract_rows:
        tf = row.get("target_field", "")
        canonical = (row.get("canonical_field") or "").strip()
        esma_code = row.get("esma_code", "")
        cls = row.get("handoff_classification", "")
        contract_value = row.get("selected_value_sample", "")

        # Portfolio-level run/source context fields (e.g. data_cut_off_date) are a
        # single value for the WHOLE tape — materialise into every row.
        if cls in (oh.HC_SOURCE_CONTEXT_MAPPED, oh.HC_RUN_CONTEXT_MAPPED) and canonical:
            results[tf] = _materialise_run_context(
                df, tf, canonical, esma_code, contract_value, cls, issues)
            continue

        materialise = cls in (
            oh.HC_DEFAULT_DOWNSTREAM, oh.HC_ND_DEFAULT_DOWNSTREAM,
            oh.HC_CONFIGURED_STATIC,
        )
        if not materialise:
            # Asset-config-driven override: a field that the handoff left as
            # pending_regime_rule / source_absent can still be materialised when
            # the ASSET config explicitly defines a default/ND default for it
            # (e.g. ERM product_defaults_ERM.yaml: maturity_date: ND5). This is
            # strictly asset-specific — traditional asset configs that do not
            # define the field are untouched, so they keep sourcing/validating it
            # normally. Generic regime defaults are NOT used here.
            if canonical and cls in (oh.HC_PENDING_REGIME_RULE, oh.HC_SOURCE_ABSENT):
                res = _materialise_asset_override(
                    df, tf, canonical, asset_defaults, asset_nd_defaults)
                if res is not None:
                    results[tf] = res
            continue
        if not canonical:
            continue

        value, source = _resolve_value(
            canonical, esma_code, contract_value,
            asset_defaults, asset_nd_defaults, regime_defaults)

        if _is_blank(value):
            # A default/static-owned field with no resolvable value anywhere is a
            # controlled source-absent item: surfaced with clear ownership (never
            # silently dropped, never fabricated, never a hard validation failure).
            issue_id = issues.add(
                severity="warning",
                field=tf, canonical_field=canonical, esma_code=esma_code,
                issue_type=IT_SOURCE_ABSENT,
                description=("no resolvable default/static value from handoff "
                             "contract, asset config or regime config"),
                blocking_for_validation=False,
                blocking_for_projection=True,
                recommended_action="add default to asset/regime config or supply value",
                downstream_owner=OWN_TRANSFORMATION)
            results[tf] = {"value_source": "", "materialised": False,
                           "value": "", "issue_id": issue_id, "absent": True}
            continue

        # Ensure the canonical column exists, then fill blanks only.
        if canonical not in df.columns:
            df[canonical] = pd.NA
        col = df[canonical].astype("string")
        blank_mask = col.isna() | (col.str.strip() == "") | (col.str.strip() == "<NA>")
        if blank_mask.any():
            df.loc[blank_mask, canonical] = value
        results[tf] = {"value_source": source, "materialised": True, "value": value}
    return results


# --------------------------------------------------------------------------- #
# Parse-failure issues
# --------------------------------------------------------------------------- #

def _record_parse_issues(
    type_report: Dict[str, Any],
    registry_fields: Dict[str, Any],
    contract_rows: List[Dict[str, Any]],
    issues: _IssueLog,
) -> None:
    """Turn Gate 2 ``apply_types`` parse failures into controlled issues."""
    canon_to_code = {}
    for r in contract_rows:
        c = (r.get("canonical_field") or "").strip()
        if c and c not in canon_to_code:
            canon_to_code[c] = r.get("esma_code", "")

    for col, info in (type_report.get("fields", {}) or {}).items():
        failures = int(info.get("parse_failures", 0) or 0)
        if failures <= 0:
            continue
        fmt = str(info.get("format", "")).lower()
        if fmt == "date":
            it, rule = IT_DATE_PARSE, "iso_then_dayfirst"
        elif fmt in {"decimal", "number", "float", "integer", "int"}:
            it, rule = IT_NUMERIC_PARSE, "numeric_strip"
        elif fmt in {"boolean", "bool", "y/n"}:
            it, rule = IT_BOOLEAN_PARSE, "y_n"
        else:
            continue
        sample = info.get("sample_failures", []) or []
        issues.add(
            severity="error",
            field=col, canonical_field=col, esma_code=canon_to_code.get(col, ""),
            issue_type=it,
            source_value_sample="; ".join(str(s) for s in sample[:5]),
            description=f"{failures} value(s) failed deterministic {rule} parsing",
            blocking_for_validation=True,
            blocking_for_projection=True,
            recommended_action="correct source values or add explicit parse rule",
            downstream_owner=OWN_TRANSFORMATION)


# --------------------------------------------------------------------------- #
# Finalise the per-field contract + transformation status
# --------------------------------------------------------------------------- #

def _col_sample(df: pd.DataFrame, canonical: str) -> str:
    if canonical and canonical in df.columns:
        for v in df[canonical].tolist():
            if not _is_blank(v):
                return str(v)
    return ""


def _finalise_contract(
    df: pd.DataFrame,
    contract_rows: List[Dict[str, Any]],
    field_results: Dict[str, Dict[str, Any]],
    enum_report: Dict[str, Any],
    type_report: Dict[str, Any],
    registry_fields: Dict[str, Any],
    regime_defaults: Dict[str, Dict[str, Any]],
    issues: _IssueLog,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    enum_fields = set(
        (enum_report.get("canonical_enum_normalization", {}) or {}).get("fields", {}).keys())
    enum_map_fields = set(enum_report.get("_normalization_map_fields", []) or [])
    type_fields = type_report.get("fields", {}) or {}

    out: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}

    for row in contract_rows:
        tf = row.get("target_field", "")
        canonical = (row.get("canonical_field") or "").strip()
        esma_code = row.get("esma_code", "")
        cls = row.get("handoff_classification", "")
        handoff_owner = row.get("downstream_owner", "")
        fmt = str((registry_fields.get(canonical, {}) or {}).get("format", "")).lower()

        status = TS_COPIED
        value_source = ""
        type_cast = fmt if (canonical in type_fields) else ""
        enum_map_used = ""
        parse_rule = ""
        issue_id = ""
        owner = OWN_VALIDATION
        b_val = False
        b_proj = False
        note = row.get("notes", "")

        res = field_results.get(tf, {})

        # Asset-config override: the handoff left this field as
        # pending_regime_rule / source_absent, but the ASSET config explicitly
        # materialised a default/ND default for it (e.g. ERM maturity_date=ND5).
        # The materialised value means it is no longer a missing-materialisation
        # projection blocker. This is asset-specific by construction.
        if res.get("asset_override") and res.get("materialised"):
            status = TS_ND_DEFAULT if res.get("nd") else TS_DEFAULT
            value_source = res.get("value_source", "")
            owner = OWN_VALIDATION
            note = (note + " | asset-config default materialised "
                    f"({value_source})").strip(" |")
            counts[status] = counts.get(status, 0) + 1
            out.append({
                "target_contract_id": row.get("target_contract_id", ""),
                "esma_code": esma_code,
                "target_field": tf,
                "canonical_field": canonical,
                "domain": row.get("domain", ""),
                "coverage_status": row.get("coverage_status", ""),
                "handoff_classification": cls,
                "handoff_downstream_owner": handoff_owner,
                "transformation_status": status,
                "transformed_value_sample": _col_sample(df, canonical),
                "value_source": value_source,
                "type_cast": type_cast,
                "enum_map_used": enum_map_used,
                "parse_rule": parse_rule,
                "issue_id": "",
                "blocking_for_validation": False,
                "blocking_for_projection": False,
                "downstream_owner": owner,
                "notes": note,
            })
            continue

        if cls == oh.HC_SEMANTIC_DERIVATION_REQUIRED or (
                canonical and oh.is_semantic_derivation(canonical, row.get("selected_source_column", ""))):
            status = TS_SEMANTIC_DERIVATION
            owner = OWN_TRANSFORMATION
            b_proj = True
            issue_id = issues.add(
                severity="warning", field=tf, canonical_field=canonical, esma_code=esma_code,
                issue_type=IT_SEMANTIC_DERIVATION,
                description=("approved semantic derivation required — not silently "
                             "aliased (e.g. ERM outstanding vs principal balance)"),
                blocking_for_validation=False, blocking_for_projection=True,
                recommended_action="define approved derivation or operator decision",
                downstream_owner=OWN_TRANSFORMATION)

        elif cls == oh.HC_OPERATOR_DECISION_PENDING:
            status = TS_OPERATOR_PENDING
            owner = OWN_OPERATOR
            b_proj = True
            issue_id = issues.add(
                severity="warning", field=tf, canonical_field=canonical, esma_code=esma_code,
                issue_type=IT_OPERATOR_PENDING,
                description="operator decision pending from onboarding handoff",
                blocking_for_validation=False, blocking_for_projection=True,
                recommended_action="resolve operator decision",
                downstream_owner=OWN_OPERATOR)

        elif cls == oh.HC_PENDING_REGIME_RULE:
            status = TS_PENDING_PROJECTION
            owner = OWN_PROJECTION
            b_proj = True
            issue_id = issues.add(
                severity="info", field=tf, canonical_field=canonical, esma_code=esma_code,
                issue_type=IT_PENDING_PROJECTION,
                description="pending regime/projection rule — handed to Projection Agent",
                blocking_for_validation=False, blocking_for_projection=True,
                recommended_action="implement or defer regime rule at projection",
                downstream_owner=OWN_PROJECTION)

        elif cls == oh.HC_SOURCE_ABSENT:
            status = TS_SOURCE_ABSENT
            owner = OWN_TRANSFORMATION
            b_proj = True
            issue_id = issues.add(
                severity="warning", field=tf, canonical_field=canonical, esma_code=esma_code,
                issue_type=IT_SOURCE_ABSENT,
                description="no source value available; surfaced rather than fabricated",
                blocking_for_validation=False, blocking_for_projection=True,
                recommended_action="materialise default or confirm source absent",
                downstream_owner=OWN_TRANSFORMATION)

        elif cls == oh.HC_NOT_APPLICABLE:
            status = TS_NOT_APPLICABLE
            owner = OWN_VALIDATION

        elif cls in (oh.HC_SOURCE_CONTEXT_MAPPED, oh.HC_RUN_CONTEXT_MAPPED):
            value_source = res.get("value_source", "")
            if res.get("materialised"):
                status = (TS_SOURCE_CONTEXT if cls == oh.HC_SOURCE_CONTEXT_MAPPED
                          else TS_RUN_CONTEXT)
                owner = OWN_VALIDATION
                parse_rule = "iso_date"
            else:
                status = TS_SOURCE_ABSENT
                owner = OWN_TRANSFORMATION
                b_proj = True
                issue_id = res.get("issue_id", "")

        elif cls == oh.HC_CONFIGURED_STATIC:
            value_source = res.get("value_source", "")
            if res.get("materialised"):
                status = TS_CONFIGURED_STATIC
                owner = OWN_VALIDATION
            else:
                # No static value configured anywhere — controlled source_absent
                # (issue already logged in _materialise_fields).
                status = TS_SOURCE_ABSENT
                owner = OWN_TRANSFORMATION
                b_proj = True
                issue_id = res.get("issue_id", "")

        elif cls == oh.HC_ND_DEFAULT_DOWNSTREAM:
            value_source = res.get("value_source", "")
            if res.get("materialised"):
                status = TS_ND_DEFAULT
                owner = OWN_VALIDATION
            else:
                status = TS_SOURCE_ABSENT
                owner = OWN_TRANSFORMATION
                b_proj = True
                issue_id = res.get("issue_id", "")

        elif cls == oh.HC_DEFAULT_DOWNSTREAM:
            value_source = res.get("value_source", "")
            if res.get("materialised"):
                status = TS_DEFAULT
                owner = OWN_VALIDATION
            else:
                status = TS_SOURCE_ABSENT
                owner = OWN_TRANSFORMATION
                b_proj = True
                issue_id = res.get("issue_id", "")

        elif cls == oh.HC_TRANSFORMATION_REQUIRED:
            # Derivation owned by transformation. Where a deterministic regime
            # default exists we materialise it; otherwise we surface, never guess.
            owner = OWN_TRANSFORMATION
            rd = regime_defaults.get(str(esma_code), {})
            if not _is_blank(rd.get("default_value")) and not _col_sample(df, canonical):
                if canonical and canonical not in df.columns:
                    df[canonical] = pd.NA
                if canonical:
                    col = df[canonical].astype("string")
                    blank = col.isna() | (col.str.strip() == "") | (col.str.strip() == "<NA>")
                    if blank.any():
                        df.loc[blank, canonical] = str(rd["default_value"]).strip()
                status = TS_DERIVED
                value_source = "regime_config_default"
            elif _col_sample(df, canonical):
                status = TS_DERIVED
            else:
                status = TS_SEMANTIC_DERIVATION
                b_proj = True
                issue_id = issues.add(
                    severity="warning", field=tf, canonical_field=canonical, esma_code=esma_code,
                    issue_type=IT_SEMANTIC_DERIVATION,
                    description="derivation required but no deterministic rule/default available",
                    blocking_for_validation=False, blocking_for_projection=True,
                    recommended_action="define approved derivation rule",
                    downstream_owner=OWN_TRANSFORMATION)

        else:
            # source_mapped / source_mapped_with_alternatives / approved_decision
            owner = OWN_VALIDATION
            normalized = canonical in enum_fields and (
                (enum_report.get("canonical_enum_normalization", {}) or {})
                .get("fields", {}).get(canonical, {}).get("rows_changed", 0) > 0)
            if normalized:
                status = TS_ENUM_NORMALIZED
                enum_map_used = "canonical_enum_normalization"
            elif type_cast and type_cast not in ("string", "none", ""):
                status = TS_TYPE_NORMALIZED
                parse_rule = type_cast
            else:
                status = TS_COPIED

        # enum map availability annotation
        if canonical in enum_map_fields and not enum_map_used:
            enum_map_used = "canonical_enum_normalization"
            # surface unmapped enum values (do not guess)
            unmapped = ((enum_report.get("canonical_enum_normalization", {}) or {})
                        .get("fields", {}).get(canonical, {}).get("unmapped_examples", []))
            if unmapped and status in (TS_COPIED, TS_TYPE_NORMALIZED, TS_ENUM_NORMALIZED):
                eid = issues.add(
                    severity="warning", field=tf, canonical_field=canonical, esma_code=esma_code,
                    issue_type=IT_ENUM_UNMAPPED,
                    source_value_sample="; ".join(str(s) for s in unmapped[:5]),
                    description="enum value(s) not in configured normalization map",
                    blocking_for_validation=False, blocking_for_projection=True,
                    recommended_action="extend enum mapping or confirm value at projection",
                    downstream_owner=OWN_PROJECTION)
                issue_id = issue_id or eid

        counts[status] = counts.get(status, 0) + 1
        out.append({
            "target_contract_id": row.get("target_contract_id", ""),
            "esma_code": esma_code,
            "target_field": tf,
            "canonical_field": canonical,
            "domain": row.get("domain", ""),
            "coverage_status": row.get("coverage_status", ""),
            "handoff_classification": cls,
            "handoff_downstream_owner": handoff_owner,
            "field_disposition": row.get("field_disposition", ""),
            "disposition_source": row.get("disposition_source", ""),
            "disposition_action": (
                _disposition_action(row.get("field_disposition", ""))
                if row.get("field_disposition") else ""),
            "transformation_status": status,
            "transformed_value_sample": _col_sample(df, canonical),
            "value_source": value_source,
            "type_cast": type_cast,
            "enum_map_used": enum_map_used,
            "parse_rule": parse_rule,
            "issue_id": issue_id,
            "blocking_for_validation": b_val,
            "blocking_for_projection": b_proj,
            "downstream_owner": owner,
            "notes": note,
        })

    return out, counts


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #

def compute_readiness(
    *,
    central_loaded: bool,
    handoff_valid: bool,
    blocking_decision_count: int,
    blocking_validation_issue_count: int,
    pending_projection_count: int,
    semantic_derivation_count: int,
    operator_pending_count: int,
) -> Dict[str, bool]:
    """Distinct validation / projection / xml readiness flags.

    ``ready_for_validation``  — central tape loaded, handoff valid, no blocking
    onboarding decisions, and no uncontrolled type/enum/date parsing failures
    (every field carries a controlled transformation status).

    ``ready_for_projection``  — false while pending projection/regime rules,
    semantic derivations, or operator decisions remain.

    ``ready_for_xml_delivery`` — always false at this stage.
    """
    ready_for_validation = bool(
        central_loaded
        and handoff_valid
        and blocking_decision_count == 0
        and blocking_validation_issue_count == 0
    )
    ready_for_projection = bool(
        ready_for_validation
        and pending_projection_count == 0
        and semantic_derivation_count == 0
        and operator_pending_count == 0
    )
    ready_for_xml_delivery = False
    return {
        "ready_for_validation": ready_for_validation,
        "ready_for_projection": ready_for_projection,
        "ready_for_xml_delivery": ready_for_xml_delivery,
    }


# --------------------------------------------------------------------------- #
# Artefact writers
# --------------------------------------------------------------------------- #

def _df_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        records.append({c: _clean(r[c]) for c in df.columns})
    return records


def _write_artefacts(
    *, out_dir: Path, df: pd.DataFrame, contract_out: List[Dict[str, Any]],
    issues: List[Dict[str, Any]], readiness: Dict[str, bool],
    status_counts: Dict[str, int], handoff: dict, paths: Dict[str, Path],
    client_id: str, run_id: str, target_contract_id: str, central_row_count: int,
    type_report: Dict[str, Any], enum_report: Dict[str, Any],
    config_paths: Dict[str, str],
) -> Dict[str, Any]:

    # 31 — transformed canonical tape (csv + json)
    tape_csv = out_dir / "31_transformed_canonical_tape.csv"
    tape_json = out_dir / "31_transformed_canonical_tape.json"
    clean_df = df.copy()
    for c in clean_df.columns:
        clean_df[c] = clean_df[c].map(_clean)
    clean_df.to_csv(tape_csv, index=False)
    records = _df_records(df)
    tape_json.write_text(json.dumps(
        {"row_count": len(records), "field_count": len(df.columns),
         "columns": list(df.columns), "rows": records}, indent=2), encoding="utf-8")

    # 32 — transformation field contract (csv + json)
    contract_csv = out_dir / "32_transformation_field_contract.csv"
    contract_json = out_dir / "32_transformation_field_contract.json"
    with open(contract_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CONTRACT_COLUMNS)
        w.writeheader()
        for r in contract_out:
            w.writerow({k: r.get(k, "") for k in _CONTRACT_COLUMNS})
    contract_json.write_text(json.dumps(
        {"target_contract_id": target_contract_id,
         "summary": {"transformation_status_counts": status_counts},
         "rows": contract_out}, indent=2, default=str), encoding="utf-8")

    # 35 — transformation issues (csv + json)
    issues_csv = out_dir / "35_transformation_issues.csv"
    issues_json = out_dir / "35_transformation_issues.json"
    with open(issues_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_ISSUE_COLUMNS)
        w.writeheader()
        for r in issues:
            w.writerow({k: r.get(k, "") for k in _ISSUE_COLUMNS})
    issue_type_counts: Dict[str, int] = {}
    for r in issues:
        issue_type_counts[r["issue_type"]] = issue_type_counts.get(r["issue_type"], 0) + 1
    issues_json.write_text(json.dumps(
        {"issue_count": len(issues),
         "blocking_for_validation": sum(1 for r in issues if r["blocking_for_validation"]),
         "blocking_for_projection": sum(1 for r in issues if r["blocking_for_projection"]),
         "issue_type_counts": issue_type_counts,
         "rows": issues}, indent=2, default=str), encoding="utf-8")

    # 34 — transformation lineage (carry forward onboarding lineage + extend)
    lineage_path = out_dir / "34_transformation_lineage.json"
    onboarding_lineage = _read_json(paths["lineage"]) or {}
    tx_rows = []
    contract_by_field = {r["target_field"]: r for r in contract_out}
    for r in contract_out:
        tx_rows.append({
            "target_field": r["target_field"],
            "esma_code": r["esma_code"],
            "source_canonical_field": r["canonical_field"],
            "transformed_field": r["canonical_field"],
            "transformation_applied": r["transformation_status"],
            "default_source": r["value_source"],
            "enum_map_used": r["enum_map_used"],
            "type_cast": r["type_cast"],
            "parse_rule": r["parse_rule"],
            "issue_id": r["issue_id"],
        })
    lineage_path.write_text(json.dumps({
        "client_id": client_id,
        "run_id": run_id,
        "target_contract_id": target_contract_id,
        "onboarding_lineage": onboarding_lineage.get("rows", []),
        "onboarding_lineage_source": "27_onboarding_handoff_lineage.json",
        "transformation_lineage": tx_rows,
    }, indent=2, default=str), encoding="utf-8")

    # 33 — readiness (json + md)
    readiness_json = out_dir / "33_transformation_readiness.json"
    readiness_md = out_dir / "33_transformation_readiness.md"
    readiness_doc = {
        "agent": AGENT,
        "agent_version": AGENT_VERSION,
        "client_id": client_id,
        "run_id": run_id,
        "target_contract_id": target_contract_id,
        "created_at": _now(),
        "central_tape_loaded": central_row_count > 0,
        "central_tape_row_count": central_row_count,
        "transformed_field_count": len(df.columns),
        "transformation_status_counts": status_counts,
        "issue_count": len(issues),
        "blocking_for_validation_count": sum(1 for r in issues if r["blocking_for_validation"]),
        "blocking_for_projection_count": sum(1 for r in issues if r["blocking_for_projection"]),
        "next_agent": NEXT_AGENT,
        **readiness,
    }
    readiness_json.write_text(json.dumps(readiness_doc, indent=2, default=str), encoding="utf-8")
    readiness_md.write_text(_readiness_md(readiness_doc), encoding="utf-8")

    # 30 — transformation manifest (json + yaml)
    manifest_json = out_dir / "30_transformation_manifest.json"
    manifest_yaml = out_dir / "30_transformation_manifest.yaml"
    manifest = {
        "agent": AGENT,
        "agent_version": AGENT_VERSION,
        "stage": STAGE,
        "next_agent": NEXT_AGENT,
        "created_at": _now(),
        "client_id": client_id,
        "run_id": run_id,
        "target_contract_id": target_contract_id,

        # Governance flags — what this package IS and IS NOT.
        "input_handoff_type": handoff.get("handoff_type", ""),
        "consumes_canonical_handoff_package": True,
        "not_raw_source": True,
        "did_not_rerun_gate1": True,
        "did_not_fuzzy_match_sources": True,
        "did_not_mutate_onboarding_artefacts": True,
        "performed_projection": False,
        "performed_xml_delivery": False,

        # Inputs consumed.
        "onboarding_handoff_manifest": str(paths["handoff_dir"] / "24_onboarding_handoff_manifest.json"),
        "central_tape_path": str(paths["central_tape"]),
        "central_tape_row_count": central_row_count,
        "field_contract_path": str(paths["field_contract_csv"]),
        "onboarding_lineage_path": str(paths["lineage"]),
        **config_paths,

        # Outputs produced.
        "transformed_tape_csv": str(tape_csv),
        "transformed_tape_json": str(tape_json),
        "transformation_field_contract_csv": str(contract_csv),
        "transformation_field_contract_json": str(contract_json),
        "transformation_readiness_json": str(readiness_json),
        "transformation_readiness_md": str(readiness_md),
        "transformation_lineage_json": str(lineage_path),
        "transformation_issues_csv": str(issues_csv),
        "transformation_issues_json": str(issues_json),

        # Counts.
        "transformed_field_count": len(df.columns),
        "transformation_status_counts": status_counts,
        "issue_count": len(issues),
        "issue_type_counts": issue_type_counts,

        # Readiness.
        **readiness,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    manifest_yaml.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    return {
        "manifest": manifest,
        "readiness": readiness_doc,
        "transformation_dir": str(out_dir),
        "manifest_json_path": str(manifest_json),
        "manifest_yaml_path": str(manifest_yaml),
        "transformed_tape_csv_path": str(tape_csv),
        "transformed_tape_json_path": str(tape_json),
        "field_contract_csv_path": str(contract_csv),
        "field_contract_json_path": str(contract_json),
        "readiness_json_path": str(readiness_json),
        "readiness_md_path": str(readiness_md),
        "lineage_path": str(lineage_path),
        "issues_csv_path": str(issues_csv),
        "issues_json_path": str(issues_json),
    }


def _readiness_md(r: Dict[str, Any]) -> str:
    def yn(v: bool) -> str:
        return "✅ yes" if v else "❌ no"

    counts = r.get("transformation_status_counts", {}) or {}
    lines = [
        "# Transformation readiness", "",
        f"Client: {r.get('client_id', '')}  ",
        f"Run: {r.get('run_id', '')}  ",
        f"Target contract: {r.get('target_contract_id', '')}  ",
        f"Agent: **{r.get('agent', '')} v{r.get('agent_version', '')}**  ",
        f"Next agent: **{r.get('next_agent', '')}**", "",
        "> The transformed canonical tape is a deterministic, validation-ready "
        "artefact. It is NOT projected to Annex 2 XML and makes NO XML-delivery "
        "claim. Projection-specific gaps are classified `pending_projection_rule` "
        "for the Projection Agent.", "",
        "## Readiness flags", "",
        f"- ready_for_validation: {yn(r['ready_for_validation'])}",
        f"- ready_for_projection: {yn(r['ready_for_projection'])}",
        f"- ready_for_xml_delivery: {yn(r['ready_for_xml_delivery'])}", "",
        "## Inputs", "",
        f"- central tape loaded: {yn(r['central_tape_loaded'])} "
        f"({r['central_tape_row_count']} rows)",
        f"- transformed field count: {r['transformed_field_count']}", "",
        "## Issues", "",
        f"- total: {r['issue_count']}",
        f"- blocking for validation: {r['blocking_for_validation_count']}",
        f"- blocking for projection: {r['blocking_for_projection_count']}", "",
        "## Transformation status counts", "",
    ]
    for k in sorted(counts):
        lines.append(f"- {k}: {counts[k]}")
    lines.append("")
    return "\n".join(lines) + "\n"
