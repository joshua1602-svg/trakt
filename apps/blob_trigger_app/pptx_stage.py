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


def _period_key(period: str) -> Optional[str]:
    """Normalise an as-of/reporting value to a ``YYYY-MM`` period key for the
    durable dated deck path, or None when no month can be parsed."""
    import re
    m = re.search(r"(\d{4})[-_/]?(\d{2})", str(period or ""))
    return f"{m.group(1)}-{m.group(2)}" if m else None


def persist_investor_deck(
    deck_path: str | Path,
    *,
    client_id: str,
    period: str = "",
    log: Optional[logging.Logger] = None,
) -> Optional[Dict[str, Any]]:
    """Upload a generated deck to durable storage so the MI API can serve it.

    Writes the client's ``latest`` deck (stable pointer, overwritten each run)
    and, when a reporting month can be parsed, a dated period copy for history.
    Fully guarded: returns ``None`` (never raises) when persistence is disabled
    or any storage op fails — the scratch deck and the run are unaffected.
    """
    log = log or logger
    if not pptx_persist_enabled():
        return None
    deck_path = Path(deck_path)
    if not deck_path.exists():
        return None
    try:
        from .storage import open_storage
        from .layout import Layout
        storage = open_storage()
        layout = Layout.from_env()
        client = client_id or "Client"
        latest_uri = layout.deck_latest_uri(client)
        storage.upload_file(deck_path, latest_uri)
        published: Dict[str, Any] = {"latest_uri": latest_uri}
        pointer = {
            "deck_name": Layout.DECK_NAME,
            "blob_name": latest_uri,
            "client_id": client,
            "reporting_period": period or None,
            "generated_at": _now_iso(),
        }
        period_key = _period_key(period)
        if period_key:
            period_uri = layout.deck_period_uri(client, period_key)
            storage.upload_file(deck_path, period_uri)
            published["period_uri"] = period_uri
            published["period"] = period_key
            pointer["period_blob_name"] = period_uri
        storage.write_text(layout.deck_latest_pointer_uri(client),
                           json.dumps(pointer, indent=2))
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
            "deck_config": _rel_config(deck_config),
        }
        # Durable publish (guarded): upload the deck so the MI API can serve it.
        # A publish failure never fails the run — it is recorded on the artifact.
        published = persist_investor_deck(
            output, client_id=client_name, period=as_of_date, log=log)
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
