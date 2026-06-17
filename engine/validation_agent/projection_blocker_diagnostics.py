"""
projection_blocker_diagnostics.py
==================================

Refines coarse ``projection_required`` / ``blocking_for_projection`` items
from the Validation Agent into six precise projection-blocker subtypes.

This module is called *after* the main validation pass and writes the
diagnostic artefact ``46_projection_blocker_diagnostics.csv/.json/.md`` under
``output/validation/``.  It never mutates upstream artefacts and never
produces projection or XML output.

Subtypes
--------
materialised_projection_pending
    Field IS in the transformed tape with at least one non-blank value.
    The projection rule is needed to map / validate / output those values.

not_materialised_projection_pending
    Field is absent from the tape or entirely blank.  No ND-value or
    regime default is allowed and no related source fields are present.
    A genuine source gap — the value must come from somewhere.

nd_or_default_rule_pending
    Field is absent / blank but the regime or asset config DOES allow a
    ND-value or default.  The projection rule just needs to apply that.

source_mapping_pending
    Field is absent / blank, but other canonical fields sharing the same
    conceptual tokens ARE present and non-blank in the tape.  A derivation
    or mapping rule can likely produce the value.

operator_or_config_dependency
    The blocker depends on an existing ``operator_required`` or
    ``config_required`` issue for the same field — or the issue itself IS
    one of those classifications.  Resolving the upstream dependency will
    unblock projection.

unknown_projection_dependency
    Fallback for edge cases not covered by the subtypes above.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Validation classification constants (mirrors validation_agent.py vocabulary).
_VC_OPERATOR = "operator_required"
_VC_CONFIG = "config_required"
_VC_PROJECTION = "projection_required"
_VC_SEMANTIC = "semantic_derivation_required"

# Projection-blocker subtypes.
PB_MATERIALISED = "materialised_projection_pending"
PB_NOT_MATERIALISED = "not_materialised_projection_pending"
PB_ND_OR_DEFAULT = "nd_or_default_rule_pending"
PB_SOURCE_MAPPING = "source_mapping_pending"
PB_OP_CONFIG_DEP = "operator_or_config_dependency"
PB_UNKNOWN = "unknown_projection_dependency"

SUBTYPES = (
    PB_MATERIALISED,
    PB_NOT_MATERIALISED,
    PB_ND_OR_DEFAULT,
    PB_SOURCE_MAPPING,
    PB_OP_CONFIG_DEP,
    PB_UNKNOWN,
)

_SUBTYPE_DESCRIPTION = {
    PB_MATERIALISED: (
        "Field is present and non-blank in the transformed tape. "
        "A projection rule is needed to map/validate these values for Annex 2 output."
    ),
    PB_NOT_MATERIALISED: (
        "Field is absent or entirely blank in the transformed tape. "
        "No ND-value/default is allowed and no candidate source fields were found. "
        "A source value must be supplied."
    ),
    PB_ND_OR_DEFAULT: (
        "Field is absent or entirely blank, but the regime or asset config "
        "permits a ND-value or default.  The projection rule only needs to apply it."
    ),
    PB_SOURCE_MAPPING: (
        "Field is absent or entirely blank, but related canonical fields sharing "
        "the same conceptual tokens are present and non-blank in the tape.  "
        "A derivation or mapping rule can likely produce the value."
    ),
    PB_OP_CONFIG_DEP: (
        "Depends on an existing operator_required or config_required issue for "
        "the same field.  Resolving that upstream dependency will unblock projection."
    ),
    PB_UNKNOWN: (
        "Fallback subtype — does not match any of the other projection-blocker patterns."
    ),
}

_SUBTYPE_ACTION = {
    PB_MATERIALISED: "implement the regime projection rule to map/validate this field",
    PB_NOT_MATERIALISED: "supply a source value or configure a ND/default if regime allows",
    PB_ND_OR_DEFAULT: "implement the ND-value or default rule at the projection layer",
    PB_SOURCE_MAPPING: "implement a derivation rule using the related source fields",
    PB_OP_CONFIG_DEP: "resolve the linked operator/config issue first, then re-validate",
    PB_UNKNOWN: "investigate field origin and projection requirements manually",
}

_DIAGNOSTIC_COLUMNS = [
    "issue_id", "canonical_field", "esma_code",
    "validation_classification", "issue_type",
    "projection_blocker_subtype", "projection_blocker_rationale",
    "has_materialised_value", "nd_or_default_allowed", "nd_or_default_source",
    "nd_default_possible", "asset_default_possible",
    "operator_dependency_present", "config_dependency_present",
    "related_fields_in_tape", "recommended_action", "downstream_owner",
    "blocking_for_projection",
]


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

def _has_non_blank_values(df: pd.DataFrame, canonical: str) -> bool:
    if canonical not in df.columns:
        return False
    col = df[canonical].fillna("").astype(str)
    return bool((col.str.strip().ne("") & col.str.lower().ne("nan")).any())


def _nd_or_default_evidence(
    canonical: str,
    esma_code: str,
    regime_index: Dict[str, Any],
    field_universe: Dict[str, Any],
    asset_defaults: Dict[str, Any],
    asset_nd_defaults: Dict[str, Any],
    tx_row: Dict[str, Any],
) -> Dict[str, Any]:
    """Gather ND/default eligibility evidence from every available source.

    ND/default eligibility is no longer read from ``field_rules`` alone. It is
    resolved (in priority order, recording the first source that confirms it)
    from:

      1. the runtime regime index (``field_rules`` + merged universe fallback);
      2. the authoritative workbook universe ND envelope (by ESMA code);
      3. the asset config (``defaults`` / ``nd_defaults``) — a concrete default;
      4. the transformation field contract columns
         (``default_rule`` / ``default_value`` / ``nd_allowed``).

    Returns a dict with booleans + the confirming ``source`` for diagnostics.
    """
    rule = regime_index.get(canonical, {}) or {}
    wb = field_universe.get(str(esma_code), {}) or {}
    tx_row = tx_row or {}

    # ND eligibility (the regulatory envelope permits an ND value).
    nd_from_rule = bool(rule.get("nd_allowed"))
    nd_from_universe = bool(wb.get("nd_allowed"))
    nd_from_tx = _truthy_cell(tx_row.get("nd_allowed"))
    nd_possible = nd_from_rule or nd_from_universe or nd_from_tx

    # Concrete default available (regime default_allowed, or an asset-config value).
    asset_default_possible = (
        (canonical in asset_defaults and not _blank_cell(asset_defaults.get(canonical)))
        or (canonical in asset_nd_defaults and not _blank_cell(asset_nd_defaults.get(canonical)))
    )
    default_from_rule = bool(rule.get("default_allowed"))
    default_from_tx = (
        _truthy_cell(tx_row.get("default_rule"))
        or not _blank_cell(tx_row.get("default_value"))
    )
    default_possible = asset_default_possible or default_from_rule or default_from_tx

    allowed = nd_possible or default_possible

    # First confirming source (for evidence / auditability). The regime index
    # records whether its ND envelope came from field_rules or the merged
    # field_universe fallback (``nd_source``), so honour that attribution.
    source = ""
    if nd_from_rule or default_from_rule:
        source = rule.get("nd_source", "field_rules") or "field_rules"
        if default_from_rule and not nd_from_rule:
            source = "field_rules"
    elif nd_from_universe:
        source = "field_universe"
    elif asset_default_possible:
        source = "asset_config"
    elif nd_from_tx or default_from_tx:
        source = "transformation_contract"

    return {
        "allowed": allowed,
        "source": source,
        "nd_default_possible": nd_possible,
        "asset_default_possible": asset_default_possible,
    }


def _related_fields(canonical: str, df: pd.DataFrame) -> List[str]:
    tokens = set(canonical.lower().split("_")) - {"", "date", "type", "code", "flag"}
    if not tokens:
        return []
    return [
        c for c in df.columns
        if c != canonical
        and set(c.lower().split("_")) & tokens
        and _has_non_blank_values(df, c)
    ]


def _dependency_flags(
    issue: Dict[str, Any],
    issues_by_field: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, bool]:
    """Whether this issue (or a peer for the same field) is operator/config-owned."""
    classification = issue.get("validation_classification", "")
    canonical = issue.get("canonical_field", "")
    issue_id = issue.get("issue_id", "")
    op = classification == _VC_OPERATOR
    cfg = classification == _VC_CONFIG
    for peer in issues_by_field.get(canonical, []):
        if peer.get("issue_id") == issue_id:
            continue
        pc = peer.get("validation_classification", "")
        op = op or pc == _VC_OPERATOR
        cfg = cfg or pc == _VC_CONFIG
    return {"operator_dependency_present": op, "config_dependency_present": cfg}


def _classify_subtype(
    issue: Dict[str, Any],
    df: pd.DataFrame,
    issues_by_field: Dict[str, List[Dict[str, Any]]],
    evidence: Dict[str, Any],
    dep: Dict[str, bool],
) -> tuple[str, str]:
    """Return (subtype, rationale) for a single projection-blocking issue.

    Branch ordering note: a direct operator_required/config_required issue is
    still reported as ``operator_or_config_dependency`` (it must be resolved by a
    human/config first). But a *peer-linked* dependency no longer masks clear
    ND/default eligibility — if the field is ND/default eligible we report
    ``nd_or_default_rule_pending`` and surface the dependency via the evidence
    columns instead. ND eligibility is always visible in the evidence regardless.
    """
    classification = issue.get("validation_classification", "")
    canonical = issue.get("canonical_field", "")

    # Direct operator/config decisions: this issue IS the dependency.
    if classification in (_VC_OPERATOR, _VC_CONFIG):
        return PB_OP_CONFIG_DEP, (
            f"{classification} issue — resolve before projection can proceed"
        )

    has_values = _has_non_blank_values(df, canonical)
    nd_default = bool(evidence.get("allowed"))

    # Materialised values present → projection rule needed to map/validate.
    if has_values:
        return PB_MATERIALISED, (
            "field has non-blank values in the transformed tape; "
            "projection rule needed to map/validate for Annex 2 output"
        )

    # Field absent/blank but ND/default eligible (from any source) → this is the
    # primary, most actionable classification; it is NOT masked by a peer
    # operator/config dependency (that is recorded in the evidence columns).
    if nd_default:
        src = evidence.get("source", "")
        return PB_ND_OR_DEFAULT, (
            "field is absent/blank but ND-value or default is permitted "
            f"(source: {src or 'unknown'}); projection rule can apply it"
        )

    # No ND/default eligibility: a peer operator/config dependency is the blocker.
    if dep.get("operator_dependency_present") or dep.get("config_dependency_present"):
        return PB_OP_CONFIG_DEP, (
            "no ND/default eligibility; depends on a linked operator/config issue "
            "for the same field"
        )

    related = _related_fields(canonical, df)
    if related:
        return PB_SOURCE_MAPPING, (
            f"field is absent/blank but related tape fields are present: "
            f"{', '.join(related[:5])}{'…' if len(related) > 5 else ''}"
        )

    return PB_NOT_MATERIALISED, (
        "field is absent/blank in the tape, no ND/default permitted, "
        "and no related source fields found"
    )


def classify_projection_blockers(
    issues: List[Dict[str, Any]],
    df: pd.DataFrame,
    tx_contract: List[Dict[str, Any]],
    regime_index: Dict[str, Any],
    *,
    field_universe: Optional[Dict[str, Any]] = None,
    asset_defaults: Optional[Dict[str, Any]] = None,
    asset_nd_defaults: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Classify every projection-blocking issue into a precise subtype.

    Args:
        issues:        All validation issues (from 43_validation_issues).
        df:            Transformed canonical tape (31_transformed_canonical_tape).
        tx_contract:   Transformation field contract rows (32_*).
        regime_index:  Dict keyed by canonical_field (field_rules + universe merge).
        field_universe: Workbook ND envelope keyed by ESMA code (fallback metadata).
        asset_defaults / asset_nd_defaults: asset config default maps.

    ND/default eligibility is resolved from regime_index, the field universe, the
    asset config and the transformation contract — never from ``field_rules``
    alone.

    Returns:
        List of diagnostic dicts, one per projection-blocking issue.
    """
    field_universe = field_universe or {}
    asset_defaults = asset_defaults or {}
    asset_nd_defaults = asset_nd_defaults or {}

    # Index issues by canonical field for cross-issue dependency checks.
    issues_by_field: Dict[str, List[Dict[str, Any]]] = {}
    for iss in issues:
        cf = iss.get("canonical_field", "")
        issues_by_field.setdefault(cf, []).append(iss)

    # Index tx_contract by canonical_field for default_rule/value/nd_allowed lookup.
    tx_by_field: Dict[str, Dict[str, Any]] = {}
    for row in tx_contract:
        cf = row.get("canonical_field", "")
        if cf:
            tx_by_field[cf] = row

    diagnostic_rows: List[Dict[str, Any]] = []
    for iss in issues:
        if not _truthy(iss.get("blocking_for_projection", False)):
            continue

        canonical = iss.get("canonical_field", "")
        esma = iss.get("esma_code", "")

        evidence = _nd_or_default_evidence(
            canonical, esma, regime_index, field_universe,
            asset_defaults, asset_nd_defaults, tx_by_field.get(canonical, {}))
        dep = _dependency_flags(iss, issues_by_field)

        subtype, rationale = _classify_subtype(
            iss, df, issues_by_field, evidence, dep)

        has_values = _has_non_blank_values(df, canonical)
        related = _related_fields(canonical, df)

        diagnostic_rows.append({
            "issue_id": iss.get("issue_id", ""),
            "canonical_field": canonical,
            "esma_code": esma,
            "validation_classification": iss.get("validation_classification", ""),
            "issue_type": iss.get("issue_type", ""),
            "projection_blocker_subtype": subtype,
            "projection_blocker_rationale": rationale,
            "has_materialised_value": has_values,
            "nd_or_default_allowed": bool(evidence.get("allowed")),
            "nd_or_default_source": evidence.get("source", ""),
            "nd_default_possible": bool(evidence.get("nd_default_possible")),
            "asset_default_possible": bool(evidence.get("asset_default_possible")),
            "operator_dependency_present": dep["operator_dependency_present"],
            "config_dependency_present": dep["config_dependency_present"],
            "related_fields_in_tape": ", ".join(related[:5]),
            "recommended_action": _SUBTYPE_ACTION[subtype],
            "downstream_owner": iss.get("downstream_owner", ""),
            "blocking_for_projection": True,
        })

    return diagnostic_rows


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "y")


def _truthy_cell(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("", "nan", "<na>", "none", "[]", "false", "0", "no"):
        return False
    return True


def _blank_cell(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in ("nan", "<na>", "none")


# --------------------------------------------------------------------------- #
# Counts
# --------------------------------------------------------------------------- #

def subtype_counts(diagnostic_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return a dict of {subtype: count} for all known subtypes."""
    counts: Dict[str, int] = {s: 0 for s in SUBTYPES}
    for r in diagnostic_rows:
        st = r.get("projection_blocker_subtype", PB_UNKNOWN)
        counts[st] = counts.get(st, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Artefact writer
# --------------------------------------------------------------------------- #

def write_blocker_diagnostics(
    out_dir: Path,
    diagnostic_rows: List[Dict[str, Any]],
    *,
    client_id: str = "",
    run_id: str = "",
    target_contract_id: str = "",
) -> Dict[str, Any]:
    """Write 46_projection_blocker_diagnostics.csv/.json/.md.

    Returns a dict with paths and subtype counts.
    """
    out_dir = Path(out_dir)
    counts = subtype_counts(diagnostic_rows)
    total = len(diagnostic_rows)

    csv_path = out_dir / "46_projection_blocker_diagnostics.csv"
    json_path = out_dir / "46_projection_blocker_diagnostics.json"
    md_path = out_dir / "46_projection_blocker_diagnostics.md"

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_DIAGNOSTIC_COLUMNS)
        w.writeheader()
        for r in diagnostic_rows:
            w.writerow({k: r.get(k, "") for k in _DIAGNOSTIC_COLUMNS})

    # JSON
    json_path.write_text(json.dumps({
        "client_id": client_id,
        "run_id": run_id,
        "target_contract_id": target_contract_id,
        "projection_blocker_count": total,
        "projection_blocker_subtype_counts": counts,
        "subtype_descriptions": _SUBTYPE_DESCRIPTION,
        "rows": diagnostic_rows,
    }, indent=2, default=str), encoding="utf-8")

    # Markdown
    md_path.write_text(_blocker_md(
        diagnostic_rows, counts, total, client_id, run_id, target_contract_id
    ), encoding="utf-8")

    return {
        "projection_blocker_count": total,
        "projection_blocker_subtype_counts": counts,
        "diagnostic_csv_path": str(csv_path),
        "diagnostic_json_path": str(json_path),
        "diagnostic_md_path": str(md_path),
    }


def _blocker_md(
    rows: List[Dict[str, Any]],
    counts: Dict[str, int],
    total: int,
    client_id: str,
    run_id: str,
    target_contract_id: str,
) -> str:
    lines = [
        "# Projection Blocker Diagnostics", "",
        f"Client: {client_id}  ",
        f"Run: {run_id}  ",
        f"Target contract: {target_contract_id}  ",
        f"Total projection blockers: **{total}**", "",
        "## Subtype summary", "",
        "| Subtype | Count | Description |",
        "| --- | --- | --- |",
    ]
    for st in SUBTYPES:
        n = counts.get(st, 0)
        if n:
            lines.append(f"| `{st}` | {n} | {_SUBTYPE_DESCRIPTION[st]} |")
    lines += ["", "## Blocker detail", ""]
    for st in SUBTYPES:
        subset = [r for r in rows if r.get("projection_blocker_subtype") == st]
        if not subset:
            continue
        lines.append(f"### {st} ({len(subset)})")
        lines.append("")
        for r in subset:
            lines.append(
                f"- **{r['issue_id']}** `{r['canonical_field']}` ({r['esma_code']})  "
            )
            lines.append(
                f"  _{r['projection_blocker_rationale']}_  "
            )
            lines.append(
                f"  action: {r['recommended_action']}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
