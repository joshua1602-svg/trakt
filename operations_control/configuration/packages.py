"""Config packages — versioned, admin-governed system/regime/asset configuration.

Repository YAML files remain the authoritative storage format; version 1 of
each package is seeded from them. New versions follow Draft -> Validate ->
Activate (never in-place mutation of an active version); rollback re-activates
a prior version. Running workflows stay pinned to the version they resolved.

Storage: operations-control container, `_global/config-packages/{layer}/…`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..contracts import now_iso
from ..stores import OpsStore, _read_json, _write_json

REPO = Path(__file__).resolve().parents[2]

LAYER_SYSTEM = "system"
LAYER_REGIME = "regime"
LAYER_ASSET = "asset"
ADMIN_LAYERS = (LAYER_SYSTEM, LAYER_REGIME, LAYER_ASSET)

#: The repository files each package layer governs (relative to repo root).
LAYER_FILES: Dict[str, List[str]] = {
    LAYER_SYSTEM: [
        "config/system/fields_registry.yaml",
        "config/system/aliases_mandatory.yaml",
        "config/system/aliases_optional.yaml",
        "config/system/aliases_analytics.yaml",
        "config/system/aliases_onboarding_kfi.yaml",
        "config/system/aliases_onboarding_lending.yaml",
        "config/system/enum_mapping.yaml",
        "config/system/esma_code_order.yaml",
        "config/system/onboarding_modes.yaml",
        "config/system/run_context_policy.yaml",
    ],
    LAYER_REGIME: [
        "config/regime/annex2_field_universe.yaml",
        "config/regime/annex2_delivery_rules.yaml",
        # Which fields of each reporting product are STANDING (client
        # onboarding) rather than delivery-specific. Governed here so a future
        # regime becomes available to onboarding as a configuration change.
        "config/regime/onboarding_standing_fields.yaml",
    ],
    LAYER_ASSET: [
        "config/asset/product_profiles.yaml",
        "config/asset/product_defaults_ERM.yaml",
        "config/asset/issue_policy.yaml",
    ],
}

#: Asset and regime are RELATED entities — an asset SUPPORTS regimes; it never
#: "is" one. Seeded from repository reality; extensible without code change
#: via the system package metadata in a later pass.
ASSET_MODEL: Dict[str, Dict[str, Any]] = {
    "equity_release": {
        "label": "Equity Release",
        "supports_regimes": ["ESMA_Annex2"],
    },
}

STATUS_DRAFT = "draft"
STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class ConfigPackageStore:
    """Versioned package persistence over the operations-control container."""

    def __init__(self, store: OpsStore, repo_root: Path = REPO):
        self.store = store
        self.repo = Path(repo_root)

    # -- paths -------------------------------------------------------------- #
    def _base(self, layer: str) -> str:
        return (f"blob://{self.store.layout.container}/_global/"
                f"config-packages/{layer}")

    def _version_uri(self, layer: str, version: int) -> str:
        return f"{self._base(layer)}/{version:04d}.json"

    def _active_uri(self, layer: str) -> str:
        return f"{self._base(layer)}/active.json"

    # -- seeding ------------------------------------------------------------ #
    def ensure_seeded(self, layer: str, *, by: str = "system") -> Dict[str, Any]:
        """Version 1 = snapshot of the repository files, auto-activated."""
        active = self.active_version(layer)
        if active is not None:
            return active
        files: Dict[str, Dict[str, str]] = {}
        for rel in LAYER_FILES.get(layer, []):
            p = self.repo / rel
            if p.exists():
                text = p.read_text(encoding="utf-8")
                files[rel] = {"content": text, "sha256": _sha(text)}
        doc = {"layer": layer, "version": 1, "status": STATUS_ACTIVE,
               "files": files, "notes": "seeded from repository files",
               "created_by": by, "created_at": now_iso(),
               "validated": True, "validation": {"ok": True, "problems": []},
               "activated_by": by, "activated_at": now_iso()}
        _write_json(self.store.storage, self._version_uri(layer, 1), doc)
        _write_json(self.store.storage, self._active_uri(layer),
                    {"version": 1, "activated_at": doc["activated_at"],
                     "activated_by": by})
        return doc

    # -- queries ------------------------------------------------------------ #
    def active_version(self, layer: str) -> Optional[Dict[str, Any]]:
        ptr = _read_json(self.store.storage, self._active_uri(layer))
        if not ptr:
            return None
        return _read_json(self.store.storage,
                          self._version_uri(layer, int(ptr["version"])))

    def get_version(self, layer: str, version: int) -> Optional[Dict[str, Any]]:
        return _read_json(self.store.storage,
                          self._version_uri(layer, version))

    def list_versions(self, layer: str) -> List[Dict[str, Any]]:
        out = []
        for uri in self.store.storage.list(self._base(layer)):
            if uri.endswith("active.json") or not uri.endswith(".json"):
                continue
            doc = _read_json(self.store.storage, uri)
            if doc:
                out.append({k: doc.get(k) for k in
                            ("layer", "version", "status", "notes",
                             "created_by", "created_at", "validated",
                             "activated_by", "activated_at")})
        return sorted(out, key=lambda d: d["version"])

    def file_hashes(self, layer: str) -> Dict[str, str]:
        pkg = self.ensure_seeded(layer)
        return {rel: f["sha256"] for rel, f in (pkg.get("files") or {}).items()}

    # -- admin lifecycle: draft -> validate -> activate -> rollback ---------- #
    def create_draft(self, layer: str, *, by: str,
                     from_version: Optional[int] = None,
                     edits: Optional[Dict[str, str]] = None,
                     notes: str = "") -> Dict[str, Any]:
        base = (self.get_version(layer, from_version) if from_version
                else self.ensure_seeded(layer))
        if base is None:
            raise ValueError(f"no such version to draft from: {from_version}")
        versions = self.list_versions(layer)
        next_v = (max(v["version"] for v in versions) + 1) if versions else 1
        files = {rel: dict(f) for rel, f in (base.get("files") or {}).items()}
        for rel, content in (edits or {}).items():
            files[rel] = {"content": content, "sha256": _sha(content)}
        doc = {"layer": layer, "version": next_v, "status": STATUS_DRAFT,
               "files": files, "notes": notes,
               "based_on_version": base["version"],
               "created_by": by, "created_at": now_iso(),
               "validated": False, "validation": None,
               "activated_by": None, "activated_at": None}
        _write_json(self.store.storage, self._version_uri(layer, next_v), doc)
        return doc

    def validate_version(self, layer: str, version: int) -> Dict[str, Any]:
        doc = self.get_version(layer, version)
        if doc is None:
            raise ValueError(f"no such version: {layer} v{version}")
        problems: List[str] = []
        for rel, f in (doc.get("files") or {}).items():
            try:
                parsed = yaml.safe_load(f["content"])
                if parsed is None or not isinstance(parsed, (dict, list)):
                    problems.append(f"{rel}: empty or non-mapping document")
            except yaml.YAMLError as exc:
                problems.append(f"{rel}: not valid YAML ({type(exc).__name__})")
        if layer == LAYER_SYSTEM:
            reg = doc["files"].get("config/system/fields_registry.yaml")
            if reg and "fields" not in (yaml.safe_load(reg["content"]) or {}):
                problems.append("fields_registry.yaml: missing 'fields' root")
        doc["validated"] = not problems
        doc["validation"] = {"ok": not problems, "problems": problems,
                             "validated_at": now_iso()}
        _write_json(self.store.storage, self._version_uri(layer, version), doc)
        return doc

    def activate_version(self, layer: str, version: int, *,
                         by: str) -> Dict[str, Any]:
        doc = self.get_version(layer, version)
        if doc is None:
            raise ValueError(f"no such version: {layer} v{version}")
        if doc["status"] == STATUS_ACTIVE:
            return doc
        if not doc.get("validated"):
            raise ValueError("version must be validated before activation")
        current = self.active_version(layer)
        if current is not None and current["version"] != version:
            current["status"] = STATUS_SUPERSEDED
            _write_json(self.store.storage,
                        self._version_uri(layer, current["version"]), current)
        doc["status"] = STATUS_ACTIVE
        doc["activated_by"] = by
        doc["activated_at"] = now_iso()
        _write_json(self.store.storage, self._version_uri(layer, version), doc)
        _write_json(self.store.storage, self._active_uri(layer),
                    {"version": version, "activated_at": doc["activated_at"],
                     "activated_by": by})
        return doc

    def rollback(self, layer: str, to_version: int, *, by: str) -> Dict[str, Any]:
        doc = self.get_version(layer, to_version)
        if doc is None:
            raise ValueError(f"no such version: {layer} v{to_version}")
        doc["validated"] = True   # a previously active version is proven
        _write_json(self.store.storage,
                    self._version_uri(layer, to_version), doc)
        return self.activate_version(layer, to_version, by=by)

    # -- materialisation ---------------------------------------------------- #
    def materialise(self, layer: str, version: int, dest: Path) -> Dict[str, Path]:
        """Write the pinned file contents under ``dest`` (read-only snapshot
        source for a run); returns relpath -> written path."""
        doc = self.get_version(layer, version)
        if doc is None:
            raise ValueError(f"no such version: {layer} v{version}")
        out: Dict[str, Path] = {}
        for rel, f in (doc.get("files") or {}).items():
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f["content"], encoding="utf-8")
            if _sha(f["content"]) != f["sha256"]:
                raise ValueError(f"hash mismatch materialising {rel}")
            out[rel] = p
        return out
