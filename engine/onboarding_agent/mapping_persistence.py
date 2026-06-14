"""
mapping_persistence.py
======================

PART 8 — controlled persistence of APPROVED mappings so future runs become
deterministic.

Three controlled sinks, each with its own safety rules:

A) Client memory     — always safe, client-scoped (reuses mapping_memory).
B) Alias library     — only on explicit confirmation; written to a dedicated
                       pipeline alias file + a companion metadata file.
C) Registry patches  — a new canonical field is NEVER silently added to the core
                       registry. We first emit a proposed patch (36_*), then, only
                       on confirmation, apply it to a controlled *pipeline*
                       registry extension (config/system/fields_registry_pipeline.yaml).
                       A new REGULATORY field is never created from LLM alone.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import mapping_memory as mm

ALIAS_PIPELINE_FILE = "config/system/aliases_pipeline.yaml"
ALIAS_PIPELINE_META = "config/system/aliases_pipeline_meta.yaml"
PIPELINE_REGISTRY_FILE = "config/system/fields_registry_pipeline.yaml"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _evidence_hash(evidence: Dict[str, Any]) -> str:
    blob = json.dumps(evidence or {}, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# A) Client memory
# ---------------------------------------------------------------------------


def persist_to_client_memory(
    approved: List[Dict[str, Any]],
    client_id: str,
    output_dir: Optional[str | Path] = None,
    memory_dir: Optional[str | Path] = None,
    run_id: str = "",
    approved_by: str = "workbench",
) -> Dict[str, Any]:
    """Write approved column->field mappings to client-scoped mapping memory.

    Each ``approved`` item: {source_file_pattern, source_column, canonical_field,
    mode, domain, decision_type, evidence}. Always safe.
    """
    mem_dir = mm.resolve_memory_dir(memory_dir=memory_dir, output_dir=output_dir,
                                    client_id=client_id)
    store = mm.MappingMemoryStore(mem_dir, client_id=client_id)
    saved = 0
    for a in approved:
        ev = dict(a.get("evidence", {}) or {})
        if run_id:
            ev.setdefault("reviewed_in_run_id", run_id)
        store.save_entry(mm.MemoryEntry(
            client_id=client_id,
            decision_type=a.get("decision_type", mm.DECISION_MAPPING_OVERRIDE),
            source_file_pattern=a.get("source_file_pattern", "*"),
            source_column=a.get("source_column", ""),
            canonical_field=a.get("canonical_field", ""),
            mode=a.get("mode", ""), domain=a.get("domain", ""),
            approved_by=approved_by, evidence=ev,
        ))
        saved += 1
    return {"saved": saved, "memory_dir": str(mem_dir)}


# ---------------------------------------------------------------------------
# B) Alias library (only on explicit confirmation)
# ---------------------------------------------------------------------------


def persist_aliases(
    approved_aliases: List[Dict[str, Any]],
    confirm: bool = False,
    repo_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Append approved synonyms to the pipeline alias file (+ metadata).

    Each item: {alias, canonical_field, approved_by, source_client_id,
    source_run_id, evidence, llm_used, approval_method}. Writes ONLY when
    ``confirm`` is True. Returns what was (or would be) written.
    """
    root = Path(repo_root) if repo_root else Path.cwd()
    alias_path = root / ALIAS_PIPELINE_FILE
    meta_path = root / ALIAS_PIPELINE_META
    planned = [a for a in approved_aliases if a.get("alias") and a.get("canonical_field")]
    if not confirm:
        return {"written": False, "planned": planned, "alias_file": str(alias_path)}

    data: Dict[str, Any] = {}
    if alias_path.exists():
        data = yaml.safe_load(alias_path.read_text(encoding="utf-8")) or {}
    meta: List[Dict[str, Any]] = []
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or []

    written = 0
    for a in planned:
        canon = a["canonical_field"]
        entry = data.setdefault(canon, {"aliases": []})
        if isinstance(entry, list):
            entry = {"aliases": entry}
            data[canon] = entry
        if a["alias"] not in entry["aliases"]:
            entry["aliases"].append(a["alias"])
            written += 1
        meta.append({
            "alias": a["alias"], "canonical_field": canon,
            "approved_by": a.get("approved_by", ""), "approved_at": _now(),
            "source_client_id": a.get("source_client_id", ""),
            "source_run_id": a.get("source_run_id", ""),
            "evidence_hash": _evidence_hash(a.get("evidence", {})),
            "llm_used": bool(a.get("llm_used", False)),
            "approval_method": a.get("approval_method", "user_confirmed"),
            "review_required_if_seen_in_new_domain": True,
        })
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    alias_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return {"written": True, "count": written, "alias_file": str(alias_path),
            "meta_file": str(meta_path)}


# ---------------------------------------------------------------------------
# C) Registry patches
# ---------------------------------------------------------------------------


def propose_registry_patch(
    new_fields: List[Dict[str, Any]],
    output_dir: str | Path,
    client_id: str = "",
    run_id: str = "",
) -> Path:
    """Write a PROPOSED registry patch (36_*). Never edits the core registry.

    Each new field item must provide at least field_name + format. Regulatory
    fields are refused (never created from LLM-assisted onboarding).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields: Dict[str, Any] = {}
    refused: List[str] = []
    for nf in new_fields:
        name = nf.get("field_name")
        if not name:
            continue
        if str(nf.get("category", "pipeline")).lower() == "regulatory":
            refused.append(name)
            continue
        fields[name] = {
            "display_name": nf.get("display_name", name.replace("_", " ").title()),
            "description": nf.get("description", ""),
            "category": "pipeline",
            "domain": nf.get("domain", "pipeline"),
            "portfolio_type": nf.get("portfolio_type", "common"),
            "format": nf.get("format", "string"),
            "core_canonical": bool(nf.get("core_canonical", False)),
            "required_in_modes": nf.get("required_in_modes", []),
            "used_by_pipeline_mi": bool(nf.get("used_by_pipeline_mi", True)),
            "source": "llm_assisted_user_approved",
            "created_at": _now(),
            "created_from_client_id": client_id,
            "created_from_run_id": run_id,
        }
    patch = {
        "_warning": "PROPOSED registry patch. Not applied until explicitly confirmed. "
                    "Targets the PIPELINE registry extension, never the core/regulatory registry.",
        "target_registry": PIPELINE_REGISTRY_FILE,
        "refused_regulatory_fields": refused,
        "fields": fields,
    }
    path = out_dir / "36_registry_patch_proposed.yaml"
    path.write_text(yaml.safe_dump(patch, sort_keys=False), encoding="utf-8")
    return path


def apply_registry_patch(
    patch_path: str | Path,
    confirm: bool = False,
    repo_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Merge a proposed patch into the controlled pipeline registry extension.

    Writes ONLY when ``confirm`` is True. Never touches the core registry.
    """
    patch = yaml.safe_load(Path(patch_path).read_text(encoding="utf-8")) or {}
    new_fields = patch.get("fields", {}) or {}
    root = Path(repo_root) if repo_root else Path.cwd()
    pipeline_reg = root / PIPELINE_REGISTRY_FILE
    if not confirm:
        return {"applied": False, "would_add": sorted(new_fields.keys()),
                "pipeline_registry": str(pipeline_reg)}
    data: Dict[str, Any] = {"fields": {}}
    if pipeline_reg.exists():
        data = yaml.safe_load(pipeline_reg.read_text(encoding="utf-8")) or {"fields": {}}
    data.setdefault("fields", {})
    added = 0
    for name, meta in new_fields.items():
        if name not in data["fields"]:
            data["fields"][name] = meta
            added += 1
    pipeline_reg.parent.mkdir(parents=True, exist_ok=True)
    pipeline_reg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return {"applied": True, "added": added, "pipeline_registry": str(pipeline_reg)}
