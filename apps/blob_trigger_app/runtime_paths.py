"""apps.blob_trigger_app.runtime_paths — Azure-safe runtime output paths.

Azure Functions runs from a **read-only** package mount
(``/home/site/wwwroot``), so the trigger must never write its runtime output
(event manifests, orchestrator state/run dirs, downloaded packs, temporary
canonicals, assembler outputs) under the repo-root ``out/``. This module
resolves a single **writable output root** that everything else derives from.

Resolution order (``TRAKT_TRIGGER_OUT`` is respected everywhere):
    1. an explicit value passed in code (highest precedence);
    2. the ``TRAKT_TRIGGER_OUT`` app setting / env var;
    3. ``/tmp/trakt/blob_trigger`` when running in Azure (writable, ephemeral);
    4. ``out/blob_trigger`` locally (unchanged behaviour).
"""

from __future__ import annotations

import os
from pathlib import Path

#: Writable, ephemeral default when running inside Azure (no env override).
AZURE_DEFAULT_OUTPUT_ROOT = "/tmp/trakt/blob_trigger"
#: Repo-relative default for local dev / tests (unchanged behaviour).
LOCAL_DEFAULT_OUTPUT_ROOT = "out/blob_trigger"

#: App-setting / env var name honoured everywhere.
ENV_TRIGGER_OUT = "TRAKT_TRIGGER_OUT"


def running_in_azure() -> bool:
    """True when executing in the Azure App Service / Functions sandbox.

    Azure sets ``WEBSITE_INSTANCE_ID`` / ``WEBSITE_SITE_NAME`` in the cloud; a
    local ``func start`` does not, so local behaviour is unaffected.
    """
    return bool(os.environ.get("WEBSITE_INSTANCE_ID")
                or os.environ.get("WEBSITE_SITE_NAME"))


def resolve_output_root(explicit: str | os.PathLike | None = None) -> str:
    """Resolve the writable runtime output root (see module docstring)."""
    if explicit:
        return str(explicit)
    env = os.environ.get(ENV_TRIGGER_OUT)
    if env:
        return env
    return AZURE_DEFAULT_OUTPUT_ROOT if running_in_azure() else LOCAL_DEFAULT_OUTPUT_ROOT


def ensure_output_root(explicit: str | os.PathLike | None = None) -> str:
    """Resolve the output root and make sure it exists (fail fast if unwritable)."""
    root = resolve_output_root(explicit)
    Path(root).mkdir(parents=True, exist_ok=True)
    return root
