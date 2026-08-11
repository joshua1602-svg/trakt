"""regulatory_watch.discovery.sources — the upstream ESMA publication allowlist.

Loads and validates ``config/regulatory_watch/esma_annex2_publication_sources.yaml``.

The allowlist is the only set of URLs discovery will ever fetch. There is no
crawling, no link-following, no search-engine use and no wildcard. A malformed
allowlist raises rather than being partially honoured, because a
silently-dropped source looks exactly like a source with nothing new.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import yaml

from .contracts import (
    CRITICALITIES,
    GATING,
    PARSER_TYPES,
    PublicationSource,
    SOURCE_TYPES,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCES_PATH = (REPO_ROOT / "config" / "regulatory_watch"
                        / "esma_annex2_publication_sources.yaml")

#: Stage 1B is ESMA Annex 2 only. Anything else is a configuration error.
SUPPORTED_AUTHORITIES = {"ESMA"}
SUPPORTED_REGIMES = {"ESMA_Annex2"}

#: Every allowlisted URL must live on an official ESMA host. This is the
#: structural guard against the allowlist quietly becoming a crawler seed.
ALLOWED_HOSTS = {"www.esma.europa.eu", "esma.europa.eu"}


class SourceConfigError(ValueError):
    """The publication allowlist is unusable. Never downgraded to a warning."""


@dataclass
class PublicationSourceSet:
    manifest_id: str
    regime: str
    authority: str
    version: int
    retrieval_enabled: bool
    allowlist_only: bool
    default_frequency: str
    sources: List[PublicationSource] = field(default_factory=list)
    notes: str = ""
    path: str = ""

    def by_id(self, source_id: str) -> PublicationSource:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(source_id)

    @property
    def enabled_sources(self) -> List[PublicationSource]:
        return [s for s in self.sources if s.enabled]

    @property
    def gating_sources(self) -> List[PublicationSource]:
        return [s for s in self.sources if s.criticality == GATING]

    @property
    def allowed_urls(self) -> List[str]:
        return sorted({s.url for s in self.sources})

    def is_allowlisted(self, url: str) -> bool:
        return url in set(self.allowed_urls)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "regime": self.regime,
            "authority": self.authority,
            "version": self.version,
            "retrieval_enabled": self.retrieval_enabled,
            "allowlist_only": self.allowlist_only,
            "default_frequency": self.default_frequency,
            "path": self.path,
            "sources": [s.to_dict() for s in self.sources],
        }


def _require(data: Dict[str, Any], key: str, where: str) -> Any:
    if key not in data or data[key] in (None, ""):
        raise SourceConfigError(f"{where}: missing required key '{key}'")
    return data[key]


def load_publication_sources(path: Optional[Path] = None
                             ) -> PublicationSourceSet:
    """Load and validate the upstream allowlist. Raises on any defect."""
    p = Path(path) if path is not None else DEFAULT_SOURCES_PATH
    if not p.exists():
        raise SourceConfigError(f"publication source config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SourceConfigError(f"publication source config is not valid "
                                f"YAML: {p}: {exc}")
    if not isinstance(raw, dict):
        raise SourceConfigError(f"publication source config must be a "
                                f"mapping: {p}")

    head = raw.get("manifest") or {}
    if not isinstance(head, dict):
        raise SourceConfigError(f"'manifest' block must be a mapping: {p}")

    authority = str(_require(head, "authority", "manifest"))
    regime = str(_require(head, "regime", "manifest"))
    if authority not in SUPPORTED_AUTHORITIES:
        raise SourceConfigError(
            f"manifest: unsupported authority '{authority}' "
            f"(Stage 1B supports {sorted(SUPPORTED_AUTHORITIES)})")
    if regime not in SUPPORTED_REGIMES:
        raise SourceConfigError(
            f"manifest: unsupported regime '{regime}' "
            f"(Stage 1B supports {sorted(SUPPORTED_REGIMES)})")

    entries = raw.get("sources")
    if not isinstance(entries, list) or not entries:
        raise SourceConfigError(f"'sources' must be a non-empty list: {p}")

    sources: List[PublicationSource] = []
    seen: set = set()
    for i, entry in enumerate(entries):
        where = f"sources[{i}]"
        if not isinstance(entry, dict):
            raise SourceConfigError(f"{where}: must be a mapping")
        source_id = str(_require(entry, "source_id", where))
        if source_id in seen:
            raise SourceConfigError(f"{where}: duplicate source_id "
                                    f"'{source_id}'")
        seen.add(source_id)

        source_type = str(_require(entry, "source_type", where))
        if source_type not in SOURCE_TYPES:
            raise SourceConfigError(
                f"{where}: unsupported source_type '{source_type}' "
                f"(allowed: {sorted(SOURCE_TYPES)})")

        parser = str(_require(entry, "parser", where))
        if parser not in PARSER_TYPES:
            raise SourceConfigError(
                f"{where}: unsupported parser '{parser}' "
                f"(allowed: {sorted(PARSER_TYPES)})")

        criticality = str(_require(entry, "criticality", where))
        if criticality not in CRITICALITIES:
            raise SourceConfigError(
                f"{where}: criticality must be one of {sorted(CRITICALITIES)}, "
                f"got '{criticality}'")

        url = str(_require(entry, "url", where))
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise SourceConfigError(f"{where}: url must be https://: {url}")
        if parsed.netloc not in ALLOWED_HOSTS:
            raise SourceConfigError(
                f"{where}: url host '{parsed.netloc}' is not an official ESMA "
                f"host (allowed: {sorted(ALLOWED_HOSTS)})")

        sources.append(PublicationSource(
            source_id=source_id,
            source_type=source_type,
            url=url,
            parser=parser,
            criticality=criticality,
            enabled=bool(entry.get("enabled", True)),
            title=str(entry.get("title") or "").strip(),
            retrieval_frequency=str(entry.get("retrieval_frequency")
                                    or head.get("default_frequency")
                                    or "weekly"),
            last_known_identifier=(str(entry["last_known_identifier"])
                                   if entry.get("last_known_identifier")
                                   else None),
            verified_reachable=bool(entry.get("verified_reachable", False)),
            notes=str(entry.get("notes") or "").strip(),
        ))

    return PublicationSourceSet(
        manifest_id=str(head.get("manifest_id") or "unnamed"),
        regime=regime,
        authority=authority,
        version=int(head.get("version") or 1),
        retrieval_enabled=bool(head.get("retrieval_enabled", False)),
        allowlist_only=bool(head.get("allowlist_only", True)),
        default_frequency=str(head.get("default_frequency") or "weekly"),
        sources=sources,
        notes=str(head.get("notes") or "").strip(),
        path=str(p),
    )
