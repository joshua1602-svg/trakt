"""Administrator read models over the governed configuration packages.

The lifecycle itself (draft -> validate -> activate -> rollback) lives in
:mod:`operations_control.configuration.packages` and stays authoritative. This
module only *describes* what the administrator area needs to render:

* per-layer overview (active, draft, package hash, validation state);
* the asset and regime catalogues, derived from the real package contents;
* the asset-to-regime compatibility matrix (backend-defined relationships only);
* semantic-where-possible comparison between two versions;
* impact analysis (which clients / portfolios / workflows are pinned);
* the administrator audit trail in plain English.

Nothing here mutates configuration, and no business meaning is inferred in the
browser — every sentence the administrator reads is produced here.
"""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..stores import OpsStore
from .packages import (
    ADMIN_LAYERS,
    ASSET_MODEL,
    ConfigPackageStore,
    LAYER_ASSET,
    LAYER_FILES,
    LAYER_REGIME,
    LAYER_SYSTEM,
    STATUS_ACTIVE,
    STATUS_DRAFT,
)

#: The audit chain for administrator configuration events (see app.py).
ADMIN_AUDIT_CLIENT = "_admin"

#: How many diff lines a single file contributes before it is truncated. The
#: administrator can always open the file itself under Files.
MAX_DIFF_LINES = 400

LAYER_LABELS: Dict[str, str] = {
    LAYER_SYSTEM: "System package",
    LAYER_REGIME: "Regime packages",
    LAYER_ASSET: "Asset packages",
}

LAYER_DESCRIPTIONS: Dict[str, str] = {
    LAYER_SYSTEM: "The platform-wide field registry, wording, and run policy "
                  "every client shares.",
    LAYER_REGIME: "The regulatory reporting packages, one per annex.",
    LAYER_ASSET: "The product packages that describe how each asset type "
                 "behaves.",
}

#: Regulatory packages Trakt is configured for. Regime and asset are RELATED
#: entities — an asset SUPPORTS a regime; it never "is" one. Everything that
#: can be read from the package contents is read, not restated here.
REGIME_MODEL: Dict[str, Dict[str, Any]] = {
    "ESMA_Annex2": {
        "label": "ESMA Annex 2",
        "annex": "Annex 2",
        "regulator": "ESMA",
        "description": "Underlying exposure reporting for non-ABCP "
                       "securitisations.",
        "field_universe_file": "config/regime/annex2_field_universe.yaml",
        # The delivery contract is DERIVED from the field universe, the fields
        # registry, the mapping workbook and the XSD — there is no file.
        "delivery_rules_file": "",
        "code_order_file": "config/system/esma_code_order.yaml",
        "schema_contract_file": "config/delivery/annex2_xml_structure_contract.yaml",
    },
}

#: Files outside a package that a package still depends upon, with the plain
#: sentence the administrator sees.
LAYER_DEPENDENCIES: Dict[str, List[Dict[str, str]]] = {
    LAYER_SYSTEM: [
        {"name": "Client configuration",
         "relationship": "Client and portfolio settings are layered on top of "
                         "this package during onboarding."},
    ],
    LAYER_REGIME: [
        {"name": "System package",
         "relationship": "Regime reporting reads its field names and code "
                         "order from the system package."},
    ],
    LAYER_ASSET: [
        {"name": "System package",
         "relationship": "Asset defaults are expressed in system field names."},
    ],
}


#: Plain names for the files inside a package. Administrators see these; the
#: repository path stays behind the technical-details panel.
FILE_LABELS: Dict[str, str] = {
    "config/system/fields_registry.yaml": "Field registry",
    "config/system/aliases_mandatory.yaml": "Required field names",
    "config/system/aliases_optional.yaml": "Optional field names",
    "config/system/aliases_analytics.yaml": "Analytics field names",
    "config/system/aliases_onboarding_kfi.yaml": "Onboarding key figures",
    "config/system/aliases_onboarding_lending.yaml": "Onboarding lending names",
    "config/system/enum_mapping.yaml": "Accepted values",
    "config/system/esma_code_order.yaml": "Regulatory field order",
    "config/system/onboarding_modes.yaml": "Onboarding modes",
    "config/system/run_context_policy.yaml": "Run policy",
    "config/regime/annex2_field_universe.yaml": "Annex 2 field list",
    "config/regime/onboarding_standing_fields.yaml": "Standing field definitions",
    "config/asset/product_profiles.yaml": "Product profiles",
    "config/asset/product_defaults_ERM.yaml": "Equity release defaults",
    "config/asset/issue_policy.yaml": "Issue handling policy",
}


def friendly_file_name(rel: str) -> str:
    """A plain name for a package file — never a repository path."""
    if rel in FILE_LABELS:
        return FILE_LABELS[rel]
    stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.replace("_", " ").strip().capitalize()


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def package_hash(doc: Optional[Dict[str, Any]]) -> str:
    """A stable hash over every file hash in the package."""
    if not doc:
        return ""
    files = doc.get("files") or {}
    joined = "\n".join(f"{rel}:{f.get('sha256', '')}" for rel, f in
                       sorted(files.items()))
    return _sha(joined)


def _yaml_of(doc: Optional[Dict[str, Any]], rel: str) -> Dict[str, Any]:
    f = ((doc or {}).get("files") or {}).get(rel)
    if not f:
        return {}
    try:
        parsed = yaml.safe_load(f.get("content") or "")
    except yaml.YAMLError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def changed_files(base: Optional[Dict[str, Any]],
                   draft: Optional[Dict[str, Any]]) -> List[str]:
    if not base or not draft:
        return []
    old = {rel: f.get("sha256") for rel, f in (base.get("files") or {}).items()}
    new = {rel: f.get("sha256") for rel, f in (draft.get("files") or {}).items()}
    return sorted(set(
        [rel for rel, h in new.items() if old.get(rel) != h]
        + [rel for rel in old if rel not in new]))


# --------------------------------------------------------------------------- #
# Layer overview
# --------------------------------------------------------------------------- #

def describe_draft(pkgs: ConfigPackageStore, layer: str,
                   versions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The newest draft for a layer, if one is awaiting action."""
    drafts = [v for v in versions if v.get("status") == STATUS_DRAFT]
    if not drafts:
        return None
    newest = max(drafts, key=lambda v: v["version"])
    doc = pkgs.get_version(layer, newest["version"]) or {}
    base = (pkgs.get_version(layer, doc.get("based_on_version"))
            if doc.get("based_on_version") else None)
    return {
        "version": doc.get("version"),
        "based_on_version": doc.get("based_on_version"),
        "created_by": doc.get("created_by") or "",
        "created_at": doc.get("created_at") or "",
        "notes": doc.get("notes") or "",
        "validated": bool(doc.get("validated")),
        "validation": doc.get("validation"),
        "validation_summary": explain_validation(layer, doc.get("validation")),
        "changed_files": [{"path": rel, "label": friendly_file_name(rel)}
                          for rel in changed_files(base, doc)],
        "package_hash": package_hash(doc),
        "file_count": len(doc.get("files") or {}),
    }


def describe_layer(pkgs: ConfigPackageStore, layer: str, *,
                   by: str = "system") -> Dict[str, Any]:
    active = pkgs.ensure_seeded(layer, by=by)
    versions = pkgs.list_versions(layer)
    return {
        "layer": layer,
        "label": LAYER_LABELS.get(layer, layer.title()),
        "description": LAYER_DESCRIPTIONS.get(layer, ""),
        "active_version": active["version"],
        "activated_at": active.get("activated_at"),
        "activated_by": active.get("activated_by"),
        "status": active.get("status", STATUS_ACTIVE),
        "validated": bool(active.get("validated")),
        "validation": active.get("validation"),
        "validation_summary": explain_validation(layer,
                                                 active.get("validation")),
        "package_hash": package_hash(active),
        "file_count": len(active.get("files") or {}),
        "files": describe_files(active),
        "draft": describe_draft(pkgs, layer, versions),
        "versions": versions,
        "dependencies": LAYER_DEPENDENCIES.get(layer, []),
    }


def describe_files(doc: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Read-only file inventory: plain name, path, hash and size."""
    return [{"path": rel,
             "label": friendly_file_name(rel),
             "sha256": f.get("sha256", ""),
             "size": len(f.get("content", "") or "")}
            for rel, f in sorted(((doc or {}).get("files") or {}).items())]


# --------------------------------------------------------------------------- #
# Asset & regime catalogues
# --------------------------------------------------------------------------- #

def describe_regimes(pkgs: ConfigPackageStore,
                     repo: Path) -> List[Dict[str, Any]]:
    """Regime entities, enriched from the ACTIVE regime package contents."""
    active = pkgs.ensure_seeded(LAYER_REGIME)
    versions = pkgs.list_versions(LAYER_REGIME)
    draft = describe_draft(pkgs, LAYER_REGIME, versions)
    out: List[Dict[str, Any]] = []
    for regime_id, spec in REGIME_MODEL.items():
        universe = _yaml_of(active, spec["field_universe_file"])
        from engine.regime_contract.annex2_contract import (
            as_delivery_rules, contract)
        rules = as_delivery_rules()
        field_rules = rules.get("field_rules") or {}
        nd_allowed = sorted({nd for r in field_rules.values()
                             if isinstance(r, dict)
                             for nd in (r.get("nd_allowed") or [])})
        schema = _schema_metadata(repo, spec.get("schema_contract_file", ""))
        code_order = _code_order_metadata(pkgs, spec.get("code_order_file", ""))
        compatible = [{"id": aid, "label": a["label"]}
                      for aid, a in ASSET_MODEL.items()
                      if regime_id in a.get("supports_regimes", [])]
        out.append({
            "id": regime_id,
            "label": spec["label"],
            "annex": spec["annex"],
            "regulator": spec["regulator"],
            "description": spec["description"],
            "layer": LAYER_REGIME,
            "active_version": active["version"],
            "package_version": None,     # derived, so it has no version of its own
            "status": active.get("status", STATUS_ACTIVE),
            "validated": bool(active.get("validated")),
            "draft_version": (draft or {}).get("version"),
            "schema": schema,
            "field_universe": {
                "count": universe.get("field_count")
                         or len(universe.get("fields") or {}),
                "source": universe.get("source") or "",
            },
            "code_order": code_order,
            "validation_policy": {
                # A value the regulator's own code list does not contain is
                # always rejected — an invariant of the delivery, not a setting.
                "unknown_values": "reject",
                "apply_defaults_only_if_explicit": True,
                "attribute_only_fields": contract().non_emitting_codes(),
                "rule_count": len(field_rules),
            },
            "no_data_policy": {
                "codes_allowed": nd_allowed,
                "fields_allowing_no_data": sum(
                    1 for r in field_rules.values()
                    if isinstance(r, dict) and r.get("nd_allowed")),
            },
            "compatible_assets": compatible,
            "dependencies": LAYER_DEPENDENCIES.get(LAYER_REGIME, []),
            "last_activation": {"at": active.get("activated_at"),
                                "by": active.get("activated_by")},
        })
    return out


def _schema_metadata(repo: Path, rel: str) -> Dict[str, Any]:
    """Schema facts as declared by the delivery contract in the repository."""
    if not rel:
        return {}
    path = Path(repo) / rel
    if not path.exists():
        return {}
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    schema = ((doc.get("xml_contract") or {}).get("schema") or {})
    if not schema:
        return {}
    return {
        "file": schema.get("xsd_file") or "",
        "namespace": schema.get("target_namespace") or "",
        "is_draft": bool(schema.get("is_draft")),
        "state": schema.get("status") or "",
    }


def _code_order_metadata(pkgs: ConfigPackageStore,
                         rel: str) -> Dict[str, Any]:
    if not rel:
        return {}
    system = pkgs.ensure_seeded(LAYER_SYSTEM)
    doc = _yaml_of(system, rel)
    meta = doc.get("metadata") or {}
    counted = 0
    for value in doc.values():
        if isinstance(value, list):
            counted += len(value)
        elif isinstance(value, dict) and value is not meta:
            counted += len([k for k, v in value.items()
                            if not isinstance(v, (dict, list))])
    return {"version": str(meta.get("version") or ""),
            "entry_count": counted,
            "source": rel}


def describe_assets(pkgs: ConfigPackageStore) -> List[Dict[str, Any]]:
    """Asset entities, enriched from the ACTIVE asset package contents."""
    active = pkgs.ensure_seeded(LAYER_ASSET)
    versions = pkgs.list_versions(LAYER_ASSET)
    draft = describe_draft(pkgs, LAYER_ASSET, versions)
    profiles = _yaml_of(active, "config/asset/product_profiles.yaml")
    defaults = _yaml_of(active, "config/asset/product_defaults_ERM.yaml")
    issue_policy = _yaml_of(active, "config/asset/issue_policy.yaml")
    out: List[Dict[str, Any]] = []
    for asset_id, spec in ASSET_MODEL.items():
        supported = [{"id": rid,
                      "label": REGIME_MODEL.get(rid, {}).get("label", rid)}
                     for rid in spec.get("supports_regimes", [])]
        out.append({
            "id": asset_id,
            "label": spec["label"],
            "layer": LAYER_ASSET,
            "active_version": active["version"],
            "status": active.get("status", STATUS_ACTIVE),
            "validated": bool(active.get("validated")),
            "draft_version": (draft or {}).get("version"),
            "supported_regimes": supported,
            "source_semantics": {
                "profile_count": len(profiles.get("profiles") or []),
                "capabilities": profiles.get("capabilities") or [],
                "profile_version": str(profiles.get("version") or ""),
            },
            "mapping_defaults": {
                "setting_count": _leaf_count(defaults),
                "source": "config/asset/product_defaults_ERM.yaml",
            },
            "taxonomies": sorted(
                k for k in defaults
                if isinstance(defaults.get(k), (dict, list))),
            "issue_policy": {
                "rule_count": _leaf_count(issue_policy),
            },
            "dependencies": LAYER_DEPENDENCIES.get(LAYER_ASSET, []),
            "last_activation": {"at": active.get("activated_at"),
                                "by": active.get("activated_by")},
        })
    return out


def _leaf_count(node: Any) -> int:
    if isinstance(node, dict):
        return sum(_leaf_count(v) for v in node.values())
    if isinstance(node, list):
        return sum(_leaf_count(v) for v in node)
    return 1


#: Readiness states, most to least usable. The operational question an asset
#: package must answer at a glance is "can this safely process a delivery?".
READINESS_READY = "ready"
READINESS_ATTENTION = "attention"
READINESS_BLOCKED = "blocked"

READINESS_LABELS = {
    READINESS_READY: "Ready for use",
    READINESS_ATTENTION: "Usable, with things to know",
    READINESS_BLOCKED: "Not ready for use",
}


def _sentence_list(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def describe_readiness(asset: Dict[str, Any],
                       regimes: List[Dict[str, Any]],
                       issues: Optional[List[Dict[str, Any]]] = None
                       ) -> Dict[str, Any]:
    """The operational answer for one asset package, in plain language.

    Everything an operator needs to decide whether a delivery can be processed
    with this configuration — and nothing an operator does not. The
    administrative detail stays in the package read models beside it.
    """
    by_id = {r["id"]: r for r in regimes}
    warnings: List[str] = []
    for issue in issues or []:
        if issue.get("asset") in (None, asset["id"]):
            warnings.append(issue["message"])

    supported = asset.get("supported_regimes") or []
    for ref in supported:
        regime = by_id.get(ref["id"]) or {}
        if (regime.get("schema") or {}).get("is_draft"):
            warnings.append(
                f"{ref['label']} uses a draft schema published by the "
                "regulator.")
        if regime and not regime.get("validated"):
            warnings.append(f"{ref['label']} has not passed its checks.")

    active = bool(asset.get("status") == STATUS_ACTIVE)
    valid = bool(asset.get("validated"))
    if not active or not valid:
        state = READINESS_BLOCKED
    elif warnings:
        state = READINESS_ATTENTION
    else:
        state = READINESS_READY

    uses = ["MI reporting"] + [r["label"] for r in supported]
    version = f"v{asset.get('active_version')}"
    active_and_valid = (f"{asset['label']} {version} is active and valid for "
                        f"{_sentence_list(uses)}.")
    if state == READINESS_BLOCKED:
        statement = (f"{asset['label']} {version} is not active and valid, so "
                     "it cannot be used to process a delivery yet.")
    elif state == READINESS_ATTENTION:
        statement = (f"{active_and_valid} There is something to know before "
                     "you rely on it.")
    else:
        statement = f"Ready for use. {active_and_valid}"
    return {
        "state": state,
        "label": READINESS_LABELS[state],
        "statement": statement,
        "can_process": state != READINESS_BLOCKED,
        "warnings": warnings,
    }


def with_readiness(assets: List[Dict[str, Any]],
                   regimes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach the operator readiness view to every asset package."""
    issues = compatibility_issues(assets, regimes)
    for asset in assets:
        asset["readiness"] = describe_readiness(asset, regimes, issues)
    return assets


def compatibility_matrix(assets: List[Dict[str, Any]],
                         regimes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Only real, backend-declared relationships. Nothing is inferred from a
    filename, and nothing is described as 'planned' unless declared."""
    rows = []
    for asset in assets:
        supported = {r["id"] for r in asset["supported_regimes"]}
        cells = []
        for regime in regimes:
            ok = regime["id"] in supported
            cells.append({
                "regime": regime["id"],
                "regime_label": regime["label"],
                "supported": ok,
                "label": "Supported" if ok else "Not supported",
            })
        rows.append({"asset": asset["id"], "asset_label": asset["label"],
                     "cells": cells})
    return {
        "regimes": [{"id": r["id"], "label": r["label"]} for r in regimes],
        "rows": rows,
    }


def compatibility_issues(assets: List[Dict[str, Any]],
                         regimes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Backend-detected blockers an administrator must see before activating."""
    issues: List[Dict[str, Any]] = []
    regime_ids = {r["id"] for r in regimes}
    for asset in assets:
        for supported in asset["supported_regimes"]:
            if supported["id"] not in regime_ids:
                issues.append({
                    "layer": LAYER_ASSET,
                    "asset": asset["id"],
                    "regime": supported["id"],
                    "message": f"{asset['label']} is set up to report under "
                               f"{supported['label']}, but that regulatory "
                               "package is not configured.",
                })
    for regime in regimes:
        if not regime["compatible_assets"]:
            issues.append({
                "layer": LAYER_REGIME,
                "regime": regime["id"],
                "message": f"No asset type is set up to report under "
                           f"{regime['label']} yet.",
            })
        schema = regime.get("schema") or {}
        if schema.get("is_draft"):
            issues.append({
                "layer": LAYER_REGIME,
                "regime": regime["id"],
                "message": f"{regime['label']} uses a draft regulatory schema. "
                           "Confirm the published schema before relying on "
                           "this for a real submission.",
            })
    return issues


# --------------------------------------------------------------------------- #
# Activation blockers (dependency + compatibility)
# --------------------------------------------------------------------------- #

def declared_regimes(regime_doc: Optional[Dict[str, Any]]) -> set:
    """The regime identifiers a regime package version actually declares."""
    out = set()
    for rel in ((regime_doc or {}).get("files") or {}):
        parsed = _yaml_of(regime_doc, rel)
        regime_id = parsed.get("regime")
        if regime_id:
            out.add(str(regime_id))
    return out


def activation_blockers(pkgs: ConfigPackageStore, layer: str,
                        version: int) -> List[Dict[str, str]]:
    """Reasons the backend will not let this version become active.

    Checked against the CANDIDATE version's own contents, so a draft that drops
    a governed file, or that would leave an asset reporting under a regulatory
    package the platform no longer carries, is refused before activation.
    """
    candidate = pkgs.get_version(layer, version)
    if candidate is None:
        return [{"kind": "not_found",
                 "message": "That version could not be found."}]
    blockers: List[Dict[str, str]] = []
    present = set((candidate.get("files") or {}).keys())
    for rel in LAYER_FILES.get(layer, []):
        if rel not in present:
            blockers.append({
                "kind": "missing_file",
                "message": f"{friendly_file_name(rel)} is missing from this "
                           "version. Packages that depend on it would stop "
                           "working."})
    regime_doc = (candidate if layer == LAYER_REGIME
                  else pkgs.ensure_seeded(LAYER_REGIME))
    declared = declared_regimes(regime_doc)
    for spec in ASSET_MODEL.values():
        for regime_id in spec.get("supports_regimes", []):
            if regime_id not in declared:
                label = REGIME_MODEL.get(regime_id, {}).get("label", regime_id)
                blockers.append({
                    "kind": "incompatible_dependency",
                    "message": f"{spec['label']} reports under {label}, but "
                               "this version does not carry that regulatory "
                               "package."})
    return blockers


# --------------------------------------------------------------------------- #
# Validation presentation
# --------------------------------------------------------------------------- #

def explain_validation(layer: str, validation: Optional[Dict[str, Any]],
                       *, warnings: Optional[List[str]] = None
                       ) -> Dict[str, Any]:
    """Turn the validator's output into something an administrator can read.

    This does NOT re-run or second-guess validation — the backend validator
    stays authoritative. It only adds wording and keeps the raw problem text
    behind a technical-details field.
    """
    warnings = list(warnings or [])
    if validation is None:
        return {
            "state": "not_checked",
            "ok": False,
            "headline": "Not checked yet",
            "detail": "This version has not been checked. Check it before "
                      "activating.",
            "problems": [], "warnings": warnings,
            "dependencies": LAYER_DEPENDENCIES.get(layer, []),
            "checked_at": None,
        }
    problems = validation.get("problems") or []
    if validation.get("ok"):
        return {
            "state": "passed",
            "ok": True,
            "headline": "Checks passed",
            "detail": "Every check passed. This version can be activated.",
            "problems": [], "warnings": warnings,
            "dependencies": LAYER_DEPENDENCIES.get(layer, []),
            "checked_at": validation.get("validated_at"),
        }
    count = len(problems)
    noun = "check" if count == 1 else "checks"
    return {
        "state": "failed",
        "ok": False,
        "headline": "Validation failed",
        "detail": f"This draft cannot be activated because {count} "
                  f"configuration {noun} failed.",
        "problems": [_explain_problem(p) for p in problems],
        "warnings": warnings,
        "dependencies": LAYER_DEPENDENCIES.get(layer, []),
        "checked_at": validation.get("validated_at"),
    }


def _explain_problem(problem: str) -> Dict[str, str]:
    rel, _, rest = problem.partition(": ")
    name = friendly_file_name(rel) if "/" in rel else rel
    reason = rest or problem
    if "not valid" in reason.lower():
        summary = f"{name} could not be read. Its formatting is broken."
    elif "empty" in reason.lower():
        summary = f"{name} is empty, or does not contain any settings."
    elif "missing" in reason.lower():
        summary = f"{name} is missing something it must contain."
    else:
        summary = f"{name} did not pass its check."
    return {"summary": summary, "file": name, "technical": problem}


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

def compare(pkgs: ConfigPackageStore, layer: str, from_version: int,
            to_version: int) -> Dict[str, Any]:
    left = pkgs.get_version(layer, from_version)
    right = pkgs.get_version(layer, to_version)
    if left is None or right is None:
        raise ValueError("one of those versions could not be found")

    old_files = left.get("files") or {}
    new_files = right.get("files") or {}
    files: List[Dict[str, Any]] = []
    added = changed = removed = 0
    for rel in sorted(set(old_files) | set(new_files)):
        before = old_files.get(rel)
        after = new_files.get(rel)
        if before and after and before.get("sha256") == after.get("sha256"):
            files.append({"path": rel, "label": friendly_file_name(rel),
                          "change": "unchanged",
                          "from_hash": before.get("sha256"),
                          "to_hash": after.get("sha256"), "lines": []})
            continue
        if before is None:
            added += 1
            change = "added"
        elif after is None:
            removed += 1
            change = "removed"
        else:
            changed += 1
            change = "changed"
        lines, truncated = _diff_lines(
            (before or {}).get("content", ""),
            (after or {}).get("content", ""))
        files.append({
            "path": rel,
            "label": friendly_file_name(rel),
            "change": change,
            "from_hash": (before or {}).get("sha256", ""),
            "to_hash": (after or {}).get("sha256", ""),
            "lines": lines,
            "truncated": truncated,
        })

    return {
        "layer": layer,
        "from": _version_ref(left),
        "to": _version_ref(right),
        "summary": {"added": added, "changed": changed, "removed": removed,
                    "unchanged": sum(1 for f in files
                                     if f["change"] == "unchanged")},
        "files": files,
        "dependencies": {
            "shared": LAYER_DEPENDENCIES.get(layer, []),
            "note": "Both versions depend on the same layers; only the file "
                    "contents differ.",
        },
        "validation": {
            "from": left.get("validation"),
            "to": right.get("validation"),
        },
    }


def _version_ref(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {"version": doc.get("version"), "status": doc.get("status"),
            "created_by": doc.get("created_by") or "",
            "created_at": doc.get("created_at") or "",
            "activated_at": doc.get("activated_at"),
            "activated_by": doc.get("activated_by"),
            "notes": doc.get("notes") or "",
            "package_hash": package_hash(doc)}


def _diff_lines(before: str, after: str) -> Tuple[List[Dict[str, str]], bool]:
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(),
                                lineterm="", n=3)
    out: List[Dict[str, str]] = []
    for raw in diff:
        if raw.startswith("---") or raw.startswith("+++"):
            continue
        if raw.startswith("@@"):
            kind = "hunk"
        elif raw.startswith("+"):
            kind = "added"
        elif raw.startswith("-"):
            kind = "removed"
        else:
            kind = "context"
        out.append({"kind": kind, "text": raw})
        if len(out) >= MAX_DIFF_LINES:
            return out, True
    return out, False


# --------------------------------------------------------------------------- #
# Impact analysis
# --------------------------------------------------------------------------- #

RUNNING_STATUSES = ("received", "running", "needs_review", "blocked",
                    "awaiting_publication")


def impact(store: OpsStore, layer: str, version: int) -> Dict[str, Any]:
    """Which clients, portfolios and workflows are pinned to this version.

    Existing runs stay pinned to the version they resolved with; activation
    only changes what FUTURE runs resolve. Nothing here rewrites history.
    """
    clients: set = set()
    portfolios: set = set()
    pinned: List[Dict[str, Any]] = []
    running = 0
    key = f"v{version}"
    for client_id in store.known_clients():
        for row in store.list_workflows(client_id):
            run = store.load_workflow(client_id, row["workflow_id"])
            if run is None:
                continue
            versions = (run.effective_config or {}).get("package_versions") or {}
            if versions.get(layer) != key:
                continue
            clients.add(client_id)
            portfolios.add((client_id, run.portfolio_id))
            is_running = run.status in RUNNING_STATUSES
            running += 1 if is_running else 0
            pinned.append({
                "workflow_id": run.workflow_id,
                "client_id": client_id,
                "portfolio_id": run.portfolio_id,
                "status": run.status,
                "reporting_period": run.reporting_period,
                "pinned_version": key,
                "still_running": is_running,
            })
    return {
        "layer": layer,
        "version": version,
        "clients": len(clients),
        "portfolios": len(portfolios),
        "pinned_workflows": len(pinned),
        "running_pinned_workflows": running,
        "future_workflows_affected": True,
        "workflows": sorted(pinned, key=lambda w: w["workflow_id"])[:50],
        "explanation": [
            "Workflows that have already started stay on the version they "
            "began with.",
            "Only workflows started after activation will use the new "
            "version.",
            "Reports that have already been published are unchanged.",
        ],
    }


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #

_LAYER_WORD = {LAYER_SYSTEM: "System", LAYER_REGIME: "Regime",
               LAYER_ASSET: "Asset"}

_EVENT_LABELS = {
    "config_draft_created": "Draft created",
    "config_validated": "Validation completed",
    "config_activated": "Activated",
    "config_rolled_back": "Rolled back",
}


def audit_sentence(record: Dict[str, Any]) -> str:
    detail = record.get("detail") or {}
    layer = _LAYER_WORD.get(detail.get("layer", ""), "Configuration")
    actor = record.get("actor") or "an administrator"
    version = detail.get("version")
    event = record.get("event")
    if event == "config_draft_created":
        base = detail.get("based_on")
        edited = detail.get("edited_files") or []
        changed = (f" {len(edited)} file changed."
                   if len(edited) == 1 else
                   f" {len(edited)} files changed." if edited else "")
        return (f"{layer} package draft v{version} created by {actor}, based "
                f"on v{base}.{changed} A draft changes nothing until it is "
                "validated and activated.")
    if event == "config_validated":
        problems = detail.get("problems") or []
        if detail.get("ok"):
            return (f"{layer} package v{version} passed its checks when "
                    f"{actor} asked Trakt to check it.")
        count = len(problems)
        noun = "check" if count == 1 else "checks"
        return (f"{layer} package v{version} did not pass: {count} {noun} "
                f"failed. It cannot be activated until they are fixed.")
    if event == "config_activated":
        return (f"{layer} package v{version} activated by {actor}. Future "
                "workflows will use this version. Existing workflows remain "
                "pinned.")
    if event == "config_rolled_back":
        return (f"{layer} package rolled back to v{detail.get('to_version')} "
                f"by {actor}. The version it replaced has been kept.")
    return f"{layer} configuration event recorded by {actor}."


def audit_trail(store: OpsStore, *, limit: int = 200) -> Dict[str, Any]:
    rows = []
    for record in reversed(store.list_audit(ADMIN_AUDIT_CLIENT)):
        detail = record.get("detail") or {}
        event = record.get("event") or ""
        if not event.startswith("config_"):
            continue
        rows.append({
            "seq": record.get("seq"),
            "event": event,
            "event_label": _EVENT_LABELS.get(event, "Configuration change"),
            "actor": record.get("actor") or "",
            "at": record.get("at") or "",
            "layer": detail.get("layer") or "",
            "layer_label": _LAYER_WORD.get(detail.get("layer", ""), ""),
            "version": detail.get("version") or detail.get("to_version"),
            "notes": detail.get("notes") or "",
            "outcome": ("failed" if event == "config_validated"
                        and not detail.get("ok") else "ok"),
            "description": audit_sentence(record),
        })
        if len(rows) >= limit:
            break
    return {"entries": rows,
            "chain_intact": store.verify_audit_chain(ADMIN_AUDIT_CLIENT)}


# --------------------------------------------------------------------------- #
# Overview assembly
# --------------------------------------------------------------------------- #

def overview(store: OpsStore, pkgs: ConfigPackageStore, *,
             by: str = "system", repo: Optional[Path] = None
             ) -> Dict[str, Any]:
    layers = {layer: describe_layer(pkgs, layer, by=by)
              for layer in ADMIN_LAYERS}
    regimes = describe_regimes(pkgs, repo or pkgs.repo)
    assets = with_readiness(describe_assets(pkgs), regimes)
    issues = compatibility_issues(assets, regimes)

    drafts = [{"layer": layer, "label": info["label"],
               "version": info["draft"]["version"],
               "validated": info["draft"]["validated"],
               "created_by": info["draft"]["created_by"],
               "created_at": info["draft"]["created_at"]}
              for layer, info in layers.items() if info["draft"]]
    failures = [{"layer": layer, "label": info["label"],
                 "version": info["draft"]["version"],
                 "problems": (info["draft"]["validation"] or {}).get(
                     "problems") or []}
                for layer, info in layers.items()
                if info["draft"] and info["draft"]["validation"]
                and not (info["draft"]["validation"] or {}).get("ok")]
    needs_review = [{"layer": layer, "label": info["label"],
                     "version": info["draft"]["version"]}
                    for layer, info in layers.items()
                    if info["draft"] and info["draft"]["validation"] is None]

    audit = audit_trail(store, limit=50)
    recent_activations = [e for e in audit["entries"]
                          if e["event"] == "config_activated"][:5]
    recent_rollbacks = [e for e in audit["entries"]
                        if e["event"] == "config_rolled_back"][:5]

    return {
        "layers": layers,
        "assets": assets,
        "regimes": regimes,
        "compatibility": compatibility_matrix(assets, regimes),
        "attention": {
            "drafts_awaiting_action": drafts,
            "validation_failures": failures,
            "packages_requiring_review": needs_review,
            "compatibility_issues": issues,
        },
        "recent_activations": recent_activations,
        "recent_rollbacks": recent_rollbacks,
    }
