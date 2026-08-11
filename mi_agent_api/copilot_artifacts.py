"""copilot_artifacts.py — the generic Copilot artifact registry.

ONE registry maps an ``artifact_type`` slug to a resolver. Adding a new
downloadable artifact to the Microsoft 365 Copilot surface is a SERVER change
only — register a resolver here — never a Copilot package/manifest redesign:
the package exposes a single generic ``getArtifact(artifactType)`` action whose
``artifactType`` is an OPEN string the server validates against this registry
(no enumerated per-artifact function, no per-artifact OpenAPI path).

Each resolver reuses the EXISTING latest-output resolution conventions (the very
paths the dashboard / pipeline already publish to). This module owns NO new
storage layout and NO analytical behaviour — it is a thin lookup table over
capabilities Trakt already produces.

Registering a new artifact type
-------------------------------
Add one ``register(ArtifactSpec(...))`` call with a resolver returning a
``ResolvedArtifact`` (or ``None`` when not generated yet; raise on a genuine
storage fault so the route can surface a 503). That is the whole change — the
shipped Copilot package does not move.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("mi_agent_api.copilot")

# Media types (kept here so a resolver registration is fully self-describing).
PPTX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation")
CSV_MEDIA_TYPE = "text/csv"
XML_MEDIA_TYPE = "application/xml"
JSON_MEDIA_TYPE = "application/json"


# --------------------------------------------------------------------------- #
# Registry data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResolvedArtifact:
    """A resolved, servable artifact: a local file plus UI-safe metadata."""

    path: Path
    file_name: str
    reporting_period: Optional[str] = None
    generated_at: Optional[str] = None


@dataclass(frozen=True)
class ArtifactSpec:
    """One registered artifact type.

    ``resolve(client_id)`` returns a :class:`ResolvedArtifact` for the current
    latest artifact, ``None`` when none has been generated yet, and MAY raise on
    a genuine storage fault (the route maps that to a 503).
    """

    artifact_type: str            # canonical slug, e.g. "investor_deck"
    content_type: str
    label: str                    # human phrase for error messages
    resolve: Callable[[str], Optional[ResolvedArtifact]]
    aliases: Tuple[str, ...] = ()


_REGISTRY: Dict[str, ArtifactSpec] = {}
_ALIAS_INDEX: Dict[str, str] = {}


def _normalise(raw: Optional[str]) -> str:
    """Lower-case and fold spaces/hyphens to underscores so the model may pass
    ``"investor deck"``, ``"investor-deck"`` or ``"investor_deck"`` alike."""
    if not raw:
        return ""
    key = str(raw).strip().lower()
    for ch in (" ", "-", "/"):
        key = key.replace(ch, "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_")


def register(spec: ArtifactSpec) -> ArtifactSpec:
    """Register (or replace) an artifact type. Server-only extension point."""
    _REGISTRY[spec.artifact_type] = spec
    _ALIAS_INDEX[_normalise(spec.artifact_type)] = spec.artifact_type
    for alias in spec.aliases:
        _ALIAS_INDEX[_normalise(alias)] = spec.artifact_type
    return spec


def lookup(raw: Optional[str]) -> Optional[ArtifactSpec]:
    """Resolve a model-supplied ``artifactType`` (canonical slug or alias) to a
    registered spec, or ``None`` when it is not a known artifact type."""
    canonical = _ALIAS_INDEX.get(_normalise(raw))
    return _REGISTRY.get(canonical) if canonical else None


def available_types() -> List[str]:
    """Canonical artifact-type slugs currently registered (sorted, stable)."""
    return sorted(_REGISTRY)


def registry() -> Dict[str, ArtifactSpec]:
    """The live registry (read-only view for tests / diagnostics)."""
    return dict(_REGISTRY)


# --------------------------------------------------------------------------- #
# Shared resolution helpers (reuse the EXISTING latest-output conventions)
# --------------------------------------------------------------------------- #
def _client_matches_platform_uri(client_id: Optional[str]) -> bool:
    """False when the configured platform URI names a DIFFERENT client than the
    deployment — a misconfiguration we fail closed on rather than serve another
    tenant's artifact (mirrors the tape client-isolation guard)."""
    if not client_id:
        return True
    from .app import _client_from_platform_uri  # local: app includes the router
    uri_client = _client_from_platform_uri()
    if uri_client and uri_client != client_id:
        logger.error("copilot artifact client mismatch: deployment client %r vs "
                     "platform URI client %r — refusing to serve.",
                     client_id, uri_client)
        return False
    return True


def _tape_reporting_period() -> Optional[str]:
    """Best-effort reporting period from the dated platform cuts alongside
    ``latest/`` (blob deployments). ``None`` when undeterminable."""
    uri = os.environ.get("MI_AGENT_PLATFORM_URI") or ""
    if "/platform/" not in uri:
        return None
    try:
        from apps.blob_trigger_app.storage import open_storage
        from . import platform_snapshots_blob as psb
        root = uri.split("/latest", 1)[0]
        dated = psb.list_dated_platform_canonicals(root, open_storage())
        dates = sorted(d.get("date") for d in dated if d.get("date"))
        return dates[-1] if dates else None
    except Exception:  # noqa: BLE001 - metadata is best-effort, never blocks
        return None


def _platform_latest_dir() -> Optional[Path]:
    """The local ``latest/`` directory that holds the platform canonical when the
    deployment resolves platform outputs from a local directory. Blob
    deployments instead set the per-artifact explicit override env var below."""
    pdir = os.environ.get("MI_AGENT_PLATFORM_DIR")
    if pdir:
        p = Path(pdir)
        return p if p.exists() else None
    from .data_source import _resolve_platform_canonical
    canonical = _resolve_platform_canonical()
    return canonical.parent if canonical is not None else None


def _resolve_published_file(explicit_env: str, filename: str,
                            download_name: str,
                            client_id: str) -> Optional[ResolvedArtifact]:
    """Resolve a pipeline-published artifact by the SAME conventions as the tape:
    an explicit per-artifact override env var, else the conventional filename in
    the platform ``latest/`` directory. Returns ``None`` when absent."""
    if not _client_matches_platform_uri(client_id):
        return None
    explicit = os.environ.get(explicit_env)
    if explicit:
        p = Path(explicit)
        return (ResolvedArtifact(p, download_name, _tape_reporting_period())
                if p.exists() else None)
    latest = _platform_latest_dir()
    if latest is None:
        return None
    p = latest / filename
    return (ResolvedArtifact(p, download_name, _tape_reporting_period())
            if p.exists() else None)


def _resolve_latest_tape(client_id: Optional[str] = None) -> Optional[Path]:
    """The latest canonical loan tape via the EXISTING platform-canonical
    resolution (blob ``…/platform/{client}/latest/platform_canonical_typed.csv``
    or its explicit/local equivalents). Deliberately does NOT continue down the
    wider data-source chain — no central-tape substitute, no demo fallback."""
    if not _client_matches_platform_uri(client_id):
        return None
    from .data_source import _resolve_platform_canonical
    path = _resolve_platform_canonical()
    if path is None:
        return None
    p = Path(path)
    return p if p.exists() else None


# --------------------------------------------------------------------------- #
# First-class resolvers (behaviour-preserving migration of deck + tape)
# --------------------------------------------------------------------------- #
def _resolve_investor_deck(client_id: str) -> Optional[ResolvedArtifact]:
    from . import decks as decks_mod
    resolved = decks_mod.resolve_deck_local(client_id, None)
    if resolved is None:
        return None
    path, download_name = resolved
    latest = (decks_mod.list_decks(client_id) or {}).get("latest") or {}
    return ResolvedArtifact(
        path=path, file_name=download_name,
        reporting_period=latest.get("period"),
        generated_at=latest.get("generatedAt"))


def _resolve_canonical_tape(client_id: str) -> Optional[ResolvedArtifact]:
    path = _resolve_latest_tape(client_id)
    if path is None:
        return None
    return ResolvedArtifact(
        path=path,
        file_name=f"{client_id}_canonical_loan_tape_latest.csv",
        reporting_period=_tape_reporting_period())


def _resolve_mapping_report(client_id: str) -> Optional[ResolvedArtifact]:
    return _resolve_published_file(
        "MI_AGENT_MAPPING_REPORT", "mapping_report.csv",
        f"{client_id}_mapping_report_latest.csv", client_id)


def _resolve_validation_report(client_id: str) -> Optional[ResolvedArtifact]:
    return _resolve_published_file(
        "MI_AGENT_VALIDATION_REPORT", "xml_validation_report.json",
        f"{client_id}_validation_report_latest.json", client_id)


def _resolve_esma_xml(client_id: str) -> Optional[ResolvedArtifact]:
    return _resolve_published_file(
        "MI_AGENT_ESMA_XML", "annex2_final.xml",
        f"{client_id}_esma_annex2_latest.xml", client_id)


# --------------------------------------------------------------------------- #
# Registrations — the ONLY place an artifact type is declared. Add a line here
# (server deploy) to ship a new downloadable artifact; the package never moves.
# --------------------------------------------------------------------------- #
register(ArtifactSpec(
    # The governed capability owns this slug (``artefacts.ARTEFACT_INVESTOR_DECK``)
    # and the Copilot route serves this type THROUGH it, so the two must agree.
    artifact_type="investor_deck", content_type=PPTX_MEDIA_TYPE,
    label="investor deck", resolve=_resolve_investor_deck,
    aliases=("investor deck", "investor pack", "deck", "presentation", "slides")))

register(ArtifactSpec(
    artifact_type="canonical_tape", content_type=CSV_MEDIA_TYPE,
    label="canonical loan tape", resolve=_resolve_canonical_tape,
    aliases=("canonical tape", "canonical loan tape", "loan tape",
             "normalised dataset", "normalized dataset")))

register(ArtifactSpec(
    artifact_type="mapping_report", content_type=CSV_MEDIA_TYPE,
    label="mapping report", resolve=_resolve_mapping_report,
    aliases=("mapping report", "field mapping", "mapping reconciliation",
             "mapping")))

register(ArtifactSpec(
    artifact_type="validation_report", content_type=JSON_MEDIA_TYPE,
    label="validation report", resolve=_resolve_validation_report,
    aliases=("validation report", "data quality report", "validation",
             "xml validation report")))

register(ArtifactSpec(
    artifact_type="esma_xml", content_type=XML_MEDIA_TYPE,
    label="ESMA Annex 2 XML", resolve=_resolve_esma_xml,
    aliases=("esma xml", "esma", "annex 2", "annex2", "annex 2 xml",
             "regulatory xml", "annex2 xml")))
