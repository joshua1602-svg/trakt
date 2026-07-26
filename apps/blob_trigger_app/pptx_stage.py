"""apps.blob_trigger_app.pptx_stage — final orchestration stage: investor PPTX.

Wires the existing MI Agent-native PowerPoint generator (``mi_agent_pptx``) into
the Azure blob-triggered orchestration as the **final artifact of every
successful run**. It is a thin orchestration seam only:

* it does NOT contain any deck / chart / analytics / registry logic — it invokes
  the completed generator through its internal Python entrypoint
  (``mi_agent_pptx.cli.run``), never a duplicated implementation;
* it writes to the existing run directory (``<run_dir>/reports/investor_pack.pptx``),
  overwriting any older pack so a run never accumulates multiple decks;
* it records the artifact in the run manifest (``run_state.json``);
* it never fails the overall run unless the investor pack is explicitly
  configured as a mandatory artifact.

The generator consumes the already-completed canonical / analytics / risk /
manifest artifacts; when an optional artifact is absent it degrades to branded
placeholders (handled inside the generator, not here).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

GENERATOR_NAME = "mi_agent_pptx"
ARTIFACT_KEY = "investor_pack_pptx"
DEFAULT_DECK_CONFIG_REL = "configs/pptx/investor_pack.yaml"
# Output is always this fixed path within the run directory (idempotent — a
# replay overwrites it; a run never accumulates multiple investor packs).
OUTPUT_REL = "reports/investor_pack.pptx"
MANIFEST_NAME = "run_state.json"


# --------------------------------------------------------------------------- #
# Configuration flags (operational; default = generate, non-mandatory).
# --------------------------------------------------------------------------- #

def pptx_enabled() -> bool:
    """Whether the investor PPTX stage runs (default: enabled)."""
    val = os.environ.get("TRAKT_INVESTOR_PPTX_ENABLED", "true").strip().lower()
    return val not in ("0", "false", "no", "off")


def pptx_mandatory() -> bool:
    """Whether a PPTX failure should fail the overall run (default: no).

    Only honoured when the operator has explicitly opted in — the investor pack
    is an output artifact and, by default, its failure is recorded in the
    manifest without failing an otherwise-successful pipeline run.
    """
    val = os.environ.get("TRAKT_INVESTOR_PPTX_MANDATORY", "false").strip().lower()
    return val in ("1", "true", "yes", "on")


def pptx_persist_enabled() -> bool:
    """Whether the generated deck is uploaded to durable storage (so the MI API
    can serve it). Defaults ON in Azure (a blob connection is configured) and OFF
    in local dev / tests where the scratch deck is enough. Force with
    ``TRAKT_INVESTOR_PPTX_PERSIST=true|false``."""
    override = os.environ.get("TRAKT_INVESTOR_PPTX_PERSIST")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes", "on")
    # Auto: only when durable Azure blob storage is configured.
    return bool(os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
                or os.environ.get("TRAKT_BLOB_CONNECTION"))


# --------------------------------------------------------------------------- #
# Internals.
# --------------------------------------------------------------------------- #

def _repo_root() -> Path:
    # Reuse the generator's own repo-root resolution so registry/config paths
    # resolve identically regardless of the orchestration's working directory.
    from mi_agent_pptx.registry_loader import REPO_ROOT
    return Path(REPO_ROOT)


def default_deck_config() -> str:
    return str(_repo_root() / DEFAULT_DECK_CONFIG_REL)


def _rel_config(deck_config: str) -> str:
    """Express the deck config relative to the repo root when possible."""
    try:
        return str(Path(deck_config).resolve().relative_to(_repo_root()))
    except Exception:
        return os.path.basename(deck_config)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _invoke_generator(argv) -> int:
    """Call the existing generator via its internal Python entrypoint.

    Isolated behind a single function so orchestration tests can mock it without
    running matplotlib/python-pptx. Returns the CLI return code.
    """
    from mi_agent_pptx.cli import run as _run
    return _run(argv)


def _update_manifest(run_dir: Path, artifact: Dict[str, Any]) -> None:
    """Add/replace the investor-pack artifact in the run manifest.

    Updates ``run_state.json`` in place as raw JSON (NOT via ``RunState``, whose
    ``to_dict`` would drop the extra ``artifacts`` key). Never raises — a
    manifest write failure must not break the run.
    """
    manifest_path = run_dir / MANIFEST_NAME
    data: Dict[str, Any] = {}
    try:
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Investor PPTX: could not read manifest %s: %s",
                       manifest_path, exc)
        data = {}

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    artifacts[ARTIFACT_KEY] = artifact
    data["artifacts"] = artifacts

    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Investor PPTX: manifest update failed for %s: %s",
                       manifest_path, exc)


def period_key(period: str) -> Optional[str]:
    """Normalise an as-of/reporting value to a ``YYYY-MM`` period key for the
    durable dated deck path, or None when no month can be parsed."""
    import re
    m = re.search(r"(\d{4})[-_/]?(\d{2})", str(period or ""))
    if not m:
        return None
    month = int(m.group(2))
    return f"{m.group(1)}-{m.group(2)}" if 1 <= month <= 12 else None


#: Back-compat alias (the private name was used before the backfill was added).
_period_key = period_key

PPTX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation")
#: A .pptx is an OOXML zip — every valid deck starts with the local file header.
_ZIP_MAGIC = b"PK\x03\x04"


def deck_checksum(deck_path: Path) -> str:
    """``sha256:<hex>`` over the deck bytes, streamed so a large deck is cheap."""
    import hashlib
    digest = hashlib.sha256()
    with open(deck_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def validate_deck_file(deck_path: Path) -> Optional[str]:
    """``None`` when *deck_path* is a plausible investor deck, else the reason.

    Deliberately cheap and structural: the file must exist, be non-empty and
    carry the OOXML zip magic. It does NOT open the presentation — publication
    must not depend on python-pptx being importable in the publishing process.
    """
    try:
        if not deck_path.is_file():
            return "file does not exist"
        size = deck_path.stat().st_size
        if size == 0:
            return "file is empty"
        with open(deck_path, "rb") as fh:
            if fh.read(4) != _ZIP_MAGIC:
                return "not a PPTX (OOXML) file"
    except OSError as exc:
        return f"unreadable: {exc}"
    return None


def _generator_version() -> str:
    try:
        from mi_agent_pptx import __version__ as ver
        return str(ver)
    except Exception:  # noqa: BLE001 — version metadata is best-effort
        return "unknown"


def build_deck_pointer(deck_path: Path, *, client_id: str,
                       period: Optional[str], latest_uri: str,
                       period_uri: Optional[str] = None,
                       as_of_date: Optional[str] = None,
                       orchestration_run_id: Optional[str] = None,
                       source_run_id: Optional[str] = None) -> Dict[str, Any]:
    """The durable ``latest_investor_pack.json`` payload.

    Carries everything a consumer needs to identify and verify the deck without
    reaching into storage: client, reporting period, generation timestamp, the
    originating orchestration/source run, the governed relative key, checksum,
    size, content type and generator version. No credentials, no account names.
    """
    return {
        "deck_name": DECK_NAME_DEFAULT,
        "blob_name": latest_uri,
        "period_blob_name": period_uri,
        "client_id": client_id,
        # Normalised YYYY-MM period key — the same value the dated deck path uses,
        # so a pointer and its dated copy can never disagree.
        "reporting_period": period,
        # The finer as-of value the run reported (e.g. a month-end date), kept
        # for display; never used for path or ordering decisions.
        "as_of_date": as_of_date or None,
        "generated_at": _now_iso(),
        "orchestration_run_id": orchestration_run_id,
        "source_run_id": source_run_id,
        "checksum": deck_checksum(deck_path),
        "size_bytes": deck_path.stat().st_size,
        "content_type": PPTX_CONTENT_TYPE,
        "generator": GENERATOR_NAME,
        "generator_version": _generator_version(),
    }


#: Kept as a module constant so the pointer builder does not need a Layout.
DECK_NAME_DEFAULT = "investor_pack.pptx"


def _pointer_period(storage, layout, client: str) -> Optional[str]:
    """The reporting period recorded in the client's current latest pointer."""
    try:
        uri = layout.deck_latest_pointer_uri(client)
        if not storage.exists(uri):
            return None
        return (json.loads(storage.read_text(uri)) or {}).get("reporting_period")
    except Exception:  # noqa: BLE001 — an unreadable pointer is treated as absent
        return None


def persist_investor_deck(
    deck_path: str | Path,
    *,
    client_id: str,
    period: str = "",
    orchestration_run_id: Optional[str] = None,
    source_run_id: Optional[str] = None,
    force: bool = False,
    log: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """Publish a generated deck to durable storage so the MI API can serve it.

    Publication contract:

      1. the immutable run artifact (``<run_dir>/reports/investor_pack.pptx``)
         is left untouched — this only ever reads it;
      2. the dated copy ``decks/{client}/{YYYY-MM}/investor_pack.pptx`` is
         write-once. An existing dated deck is NOT overwritten by a different
         deck for the same period (the publication contract has no versioning);
         re-publishing byte-identical content is a no-op;
      3. ``decks/{client}/latest/investor_pack.pptx`` is replaced, EXCEPT when
         the current latest pointer names a NEWER reporting period — a late or
         replayed older run must never regress the client's latest deck;
      4. ``decks/{client}/latest/latest_investor_pack.json`` is written last, so
         a consumer never sees a pointer describing bytes that are not yet there.

    The whole operation is idempotent and safe to retry. Fully guarded: returns
    ``None`` (never raises) when persistence is disabled, the file is not a valid
    deck, or any storage op fails — the run is unaffected.

    ``force`` bypasses the ``pptx_persist_enabled()`` gate. It exists for the
    explicit admin backfill (``deck_backfill``), whose entire purpose is
    publication; the orchestration stage never sets it.
    """
    log = log or logger
    if not (force or pptx_persist_enabled()):
        return None
    deck_path = Path(deck_path)
    invalid = validate_deck_file(deck_path)
    if invalid is not None:
        log.warning("Investor PPTX not published (%s): %s", invalid, deck_path)
        return None
    try:
        from .storage import open_storage
        from .layout import Layout
        storage = open_storage()
        layout = Layout.from_env()
        client = client_id or "Client"
        key = period_key(period)
        published: Dict[str, Any] = {"client_id": client}

        # (2) write-once dated copy.
        period_uri: Optional[str] = None
        if key:
            period_uri = layout.deck_period_uri(client, key)
            if storage.exists(period_uri):
                published["period_skipped"] = "already published for this period"
                log.info("Investor PPTX dated copy already exists, not "
                         "overwriting: %s", period_uri)
            else:
                storage.upload_file(deck_path, period_uri)
            published["period_uri"] = period_uri
            published["period"] = key

        # (3) latest — never regress to an older reporting period.
        current = _pointer_period(storage, layout, client)
        current_key = period_key(current or "")
        if key and current_key and current_key > key:
            published["latest_skipped"] = (
                f"latest already holds a newer period ({current_key})")
            log.info("Investor PPTX latest not replaced: current period %s is "
                     "newer than %s", current_key, key)
            return published

        latest_uri = layout.deck_latest_uri(client)
        storage.upload_file(deck_path, latest_uri)
        published["latest_uri"] = latest_uri

        # (4) pointer last.
        pointer = build_deck_pointer(
            deck_path, client_id=client, period=key or (period or None),
            latest_uri=latest_uri, period_uri=period_uri,
            as_of_date=period or None,
            orchestration_run_id=orchestration_run_id, source_run_id=source_run_id)
        storage.write_text(layout.deck_latest_pointer_uri(client),
                           json.dumps(pointer, indent=2))
        published["checksum"] = pointer["checksum"]
        published["size_bytes"] = pointer["size_bytes"]
        log.info("Investor PPTX published to durable storage: %s", latest_uri)
        return published
    except Exception as exc:  # noqa: BLE001 — publish is best-effort, never fatal
        log.warning("Investor PPTX durable publish failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Public stage entry point.
# --------------------------------------------------------------------------- #

def generate_investor_pptx(
    run_dir: str | Path,
    *,
    client_name: str,
    as_of_date: str = "",
    deck_config: Optional[str] = None,
    mandatory: bool = False,
    source_run_id: Optional[str] = None,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Generate the investor PPTX for a completed run and update the manifest.

    Returns the artifact record written to the manifest. Never raises unless
    *mandatory* is set and generation fails (in which case the caller is
    expected to fail the run).
    """
    log = log or logger
    run_dir = Path(run_dir)
    deck_config = deck_config or default_deck_config()
    output = run_dir / OUTPUT_REL
    # Ensure the reports directory exists; overwrite any older investor pack.
    output.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        "--run-dir", str(run_dir),
        "--deck-config", str(deck_config),
        "--client-name", client_name or "Client",
        "--output", str(output),
    ]
    if as_of_date:
        argv += ["--as-of-date", str(as_of_date)]

    log.info("Starting Investor PPTX generation...")
    try:
        rc = _invoke_generator(argv)
        # Success is defined by the deck existing on disk; the CLI may return a
        # non-zero validation code while still having written a (placeholder)
        # deck. A missing file is a hard failure.
        if not output.exists():
            raise RuntimeError(
                f"generator returned rc={rc} but no deck was written to {output}")
        artifact = {
            "type": "pptx",
            "status": "available",
            "path": OUTPUT_REL,
            "generated_at": _now_iso(),
            "generator": GENERATOR_NAME,
            "generator_version": _generator_version(),
            "deck_config": _rel_config(deck_config),
            # Recorded so a later backfill can resolve the reporting period from
            # the manifest alone, without re-deriving it from the tape.
            "reporting_period": period_key(as_of_date) or (as_of_date or None),
            "client_id": client_name or None,
            "source_run_id": source_run_id,
        }
        # Durable publish (guarded): upload the deck so the MI API can serve it.
        # A publish failure never fails the run — it is recorded on the artifact.
        published = persist_investor_deck(
            output, client_id=client_name, period=as_of_date,
            orchestration_run_id=run_dir.name, source_run_id=source_run_id,
            log=log)
        if published:
            artifact["published"] = published
        _update_manifest(run_dir, artifact)
        log.info("Investor PPTX successfully generated:\n%s", output)
        return artifact
    except Exception as exc:  # noqa: BLE001
        artifact = {
            "type": "pptx",
            "status": "failed",
            "error": str(exc),
            "generated_at": _now_iso(),
            "generator": GENERATOR_NAME,
        }
        _update_manifest(run_dir, artifact)
        log.error("Investor PPTX generation failed:\n%s", exc)
        if mandatory:
            raise
        return artifact
