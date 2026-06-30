"""Azure Functions deployment entrypoint (root) — thin shim.

Azure Functions loads the Function App from the **root** ``function_app.py``.
This module deliberately contains NO logic: it re-exports the real app from
:mod:`apps.blob_trigger_app.function_app`, which owns the blob trigger that
routes uploads to the Orchestrator Agent (new-source onboarding vs deterministic
processing, pack-completion gating, etc.).

Why a shim:
    The previous root ``function_app.py`` was a legacy Event Grid trigger bound
    to the ``inbound`` container. While it was the deployed entrypoint, uploads
    to the current ``raw-v2`` container were silently skipped. Delegating to the
    blob-trigger app makes the watched container configurable (``%TRAKT_BLOB_CONTAINER%``)
    and routes through the Orchestrator.

Do not add logic here. All routing/inference lives in
``apps/blob_trigger_app/function_app.py`` and its (Azure-free) decision core in
``apps/blob_trigger_app/router.py``.
"""

from __future__ import annotations

from apps.blob_trigger_app.function_app import app

__all__ = ["app"]
