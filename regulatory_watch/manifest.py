"""regulatory_watch.manifest — the explicit ESMA source allowlist.

Loads and validates ``config/regulatory_watch/esma_annex2_sources.yaml``.

The allowlist is the only set of URLs the watch will ever touch, and the only
set of local artefacts it will ever hash. There is no discovery, no crawling and
no wildcard. A malformed manifest raises; it is never partially honoured,
because a silently-dropped artefact would look exactly like "that source did not
change".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .contracts import ARTEFACT_TYPES, SourceArtefact, UNKNOWN

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST_PATH = (REPO_ROOT / "config" / "regulatory_watch"
                         / "esma_annex2_sources.yaml")

#: Stage 1 is ESMA Annex 2 only. Anything else in the manifest is a
#: configuration error, not a feature.
SUPPORTED_AUTHORITIES = {"ESMA"}
SUPPORTED_REGIMES = {"ESMA_Annex2"}


class ManifestError(ValueError):
    """The source manifest is unusable. Never downgraded to a warning."""


@dataclass
class SourceManifest:
    manifest_id: str
    regime: str
    authority: str
    version: int
    retrieval_enabled: bool
    allowlist_only: bool
    artefacts: List[SourceArtefact] = field(default_factory=list)
    notes: str = ""
    path: str = ""

    def by_id(self, artefact_id: str) -> SourceArtefact:
        for a in self.artefacts:
            if a.artefact_id == artefact_id:
                return a
        raise KeyError(artefact_id)

    def of_type(self, artefact_type: str) -> List[SourceArtefact]:
        return [a for a in self.artefacts if a.artefact_type == artefact_type]

    @property
    def allowed_urls(self) -> List[str]:
        return sorted({a.source_url for a in self.artefacts if a.source_url})

    def local_artefacts(self) -> List[SourceArtefact]:
        """Artefacts with a vendored local copy declared."""
        return [a for a in self.artefacts if a.local_path]

    def declared_without_local_copy(self) -> List[SourceArtefact]:
        """Artefacts in the allowlist with no vendored copy to hash.

        These are reported as unchecked sources, never as unchanged ones.
        """
        return [a for a in self.artefacts if not a.local_path]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "regime": self.regime,
            "authority": self.authority,
            "version": self.version,
            "retrieval_enabled": self.retrieval_enabled,
            "allowlist_only": self.allowlist_only,
            "path": self.path,
            "artefacts": [a.to_dict() for a in self.artefacts],
        }


def _require(data: Dict[str, Any], key: str, where: str) -> Any:
    if key not in data or data[key] in (None, ""):
        raise ManifestError(f"{where}: missing required key '{key}'")
    return data[key]


def load_manifest(path: Optional[Path] = None) -> SourceManifest:
    """Load and validate the allowlist. Raises :class:`ManifestError`."""
    p = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    if not p.exists():
        raise ManifestError(f"source manifest not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ManifestError(f"source manifest is not valid YAML: {p}: {exc}")
    if not isinstance(raw, dict):
        raise ManifestError(f"source manifest must be a mapping: {p}")

    head = raw.get("manifest") or {}
    if not isinstance(head, dict):
        raise ManifestError(f"'manifest' block must be a mapping: {p}")

    authority = str(_require(head, "authority", "manifest"))
    regime = str(_require(head, "regime", "manifest"))
    if authority not in SUPPORTED_AUTHORITIES:
        raise ManifestError(
            f"manifest: unsupported authority '{authority}' "
            f"(Stage 1 supports {sorted(SUPPORTED_AUTHORITIES)})")
    if regime not in SUPPORTED_REGIMES:
        raise ManifestError(
            f"manifest: unsupported regime '{regime}' "
            f"(Stage 1 supports {sorted(SUPPORTED_REGIMES)})")

    entries = raw.get("artefacts")
    if not isinstance(entries, list) or not entries:
        raise ManifestError(f"'artefacts' must be a non-empty list: {p}")

    artefacts: List[SourceArtefact] = []
    seen: set = set()
    for i, entry in enumerate(entries):
        where = f"artefacts[{i}]"
        if not isinstance(entry, dict):
            raise ManifestError(f"{where}: must be a mapping")
        artefact_id = str(_require(entry, "artefact_id", where))
        if artefact_id in seen:
            raise ManifestError(f"{where}: duplicate artefact_id "
                                f"'{artefact_id}'")
        seen.add(artefact_id)
        a_authority = str(_require(entry, "authority", where))
        a_regime = str(_require(entry, "regime", where))
        a_type = str(_require(entry, "artefact_type", where))
        if a_authority not in SUPPORTED_AUTHORITIES:
            raise ManifestError(f"{where}: unsupported authority "
                                f"'{a_authority}'")
        if a_regime not in SUPPORTED_REGIMES:
            raise ManifestError(f"{where}: unsupported regime '{a_regime}'")
        if a_type not in ARTEFACT_TYPES:
            raise ManifestError(
                f"{where}: unsupported artefact_type '{a_type}' "
                f"(allowed: {sorted(ARTEFACT_TYPES)})")
        url = str(entry.get("source_url") or "")
        if url and not url.startswith("https://"):
            raise ManifestError(f"{where}: source_url must be https://: {url}")

        artefacts.append(SourceArtefact(
            artefact_id=artefact_id,
            authority=a_authority,
            regime=a_regime,
            artefact_type=a_type,
            title=str(entry.get("title") or ""),
            external_version=str(entry.get("external_version") or UNKNOWN),
            source_url=url,
            publication_date=str(entry.get("publication_date") or UNKNOWN),
            effective_date=str(entry.get("effective_date") or UNKNOWN),
            source_status=str(entry.get("source_status") or UNKNOWN),
            local_path=str(entry.get("local_path") or ""),
            parser=str(entry.get("parser") or ""),
            notes=str(entry.get("notes") or "").strip(),
        ))

    return SourceManifest(
        manifest_id=str(head.get("manifest_id") or "unnamed"),
        regime=regime,
        authority=authority,
        version=int(head.get("version") or 1),
        retrieval_enabled=bool(head.get("retrieval_enabled", False)),
        allowlist_only=bool(head.get("allowlist_only", True)),
        artefacts=artefacts,
        notes=str(head.get("notes") or "").strip(),
        path=str(p),
    )
