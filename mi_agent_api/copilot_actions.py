"""copilot_actions.py — the three Microsoft 365 Copilot v1 actions.

A THIN action layer over the existing Trakt capabilities — no business logic,
no calculations, no new pipeline:

  POST /v1/copilot/mi/query
      askTraktMi — delegates to the EXISTING ``/mi/query`` handler
      (``mi_agent_api.app.query``: deterministic-first parser → MIQuerySpec →
      deterministic executor → adapter envelope) and reshapes the envelope into
      a compact, typed answer Copilot can ground on. Fails closed (503) when the
      active data source is the synthetic demo dataset or unavailable — the
      Copilot surface never answers from demo data.

  GET /v1/copilot/artifacts/latest/investor-deck
      getLatestInvestorDeck — resolves the CURRENT latest deck exactly as the
      dashboard does (``mi_agent_api.decks``: MI_AGENT_DECK_ROOT local mode, else
      the durable blob store ``decks/{client}/latest/investor_pack.pptx``) and
      returns metadata plus a short-lived, HMAC-signed download URL.

  GET /v1/copilot/artifacts/latest/canonical-tape
      getLatestCanonicalTape — resolves the CURRENT latest platform canonical
      exactly as the API's data layer does (``data_source._resolve_platform_canonical``:
      MI_AGENT_PLATFORM_URI → …/platform/{client}/latest/platform_canonical_typed.csv,
      or the explicit/local equivalents) and returns metadata plus a short-lived
      signed download URL. Never falls back to the synthetic demo dataset or any
      substitute artifact.

  GET /v1/copilot/artifacts/download?token=…
      Redeems a signed token for the file bytes. The token is the ONLY
      credential in the URL — blob credentials never leave the server, and raw
      blob paths are never exposed. Not part of the Copilot plugin surface (it
      is the link a user clicks in the Copilot response).

Authentication: every action route requires a validated Entra ID bearer token
(``copilot_auth.copilot_auth_guard``); the download route is authenticated by
its signed, expiring token. The Easy-Auth header guard in ``auth.py`` does not
apply to this prefix (see ``auth.COPILOT_PATH_PREFIX``).

Environment (in addition to copilot_auth.py):

  ``TRAKT_COPILOT_DOWNLOAD_SIGNING_KEY``   HMAC key for download tokens. REQUIRED
                                           in production with >1 worker (each
                                           process otherwise generates its own
                                           ephemeral key and tokens won't
                                           validate across workers).
  ``TRAKT_COPILOT_DOWNLOAD_TTL_SECONDS``   Token lifetime (default 300, 60–3600).
  ``TRAKT_COPILOT_PUBLIC_BASE_URL``        External base URL used to build the
                                           download link (default: request base).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .copilot_auth import copilot_auth_guard
from .data_source import (
    KIND_SYNTHETIC_DEMO,
    KIND_UNAVAILABLE,
    data_source_kind,
    data_source_label,
)
from . import decks as decks_mod

logger = logging.getLogger("mi_agent_api.copilot")

router = APIRouter(prefix="/v1/copilot", tags=["copilot"])

_PPTX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation")
_CSV_MEDIA_TYPE = "text/csv"

#: Data-source kinds the Copilot MI action refuses to answer from (fail closed).
_BLOCKED_SOURCE_KINDS = {KIND_SYNTHETIC_DEMO, KIND_UNAVAILABLE}

_MAX_SUPPORTING_ROWS = 50


# --------------------------------------------------------------------------- #
# Schemas (explicit request/response models for the Copilot OpenAPI contract)
# --------------------------------------------------------------------------- #
class CopilotMiQueryRequest(BaseModel):
    """One standalone portfolio MI question. Copilot rewrites follow-ups into
    standalone questions before calling (the API itself is stateless, matching
    the existing MI Agent contract)."""

    question: str = Field(..., min_length=1,
                          description="The portfolio MI question, in natural language.")
    portfolioId: Optional[str] = Field(
        None, description="Optional portfolio/run selector '{client_id}/{run_id}'. "
                          "Omit to query the current active reporting dataset.")
    asOfDate: Optional[str] = Field(
        None, description="Optional as-of date label (YYYY-MM-DD).")


class CopilotSupportingArtifact(BaseModel):
    """Compact, typed extraction of one MI result artifact so Copilot can ground
    its narrative on the numbers the deterministic executor produced."""

    kind: str = Field(..., description="kpi | table | chart")
    title: Optional[str] = None
    kpis: Optional[List[Dict[str, Any]]] = Field(
        None, description="For kind=kpi: [{label, value}] as formatted by the MI Agent.")
    columns: Optional[List[str]] = None
    rows: Optional[List[Dict[str, Any]]] = Field(
        None, description="Result rows (capped; see truncated).")
    truncated: bool = False


class CopilotMiAnswer(BaseModel):
    ok: bool
    question: str
    answer: str = Field(..., description="The MI Agent's answer text. Compose any "
                                         "narrative ONLY from this and supportingValues.")
    error: Optional[str] = None
    interpreted: Optional[str] = Field(
        None, description="How the deterministic parser interpreted the question.")
    reportingDate: Optional[str] = None
    datasetContext: Optional[str] = Field(None, description="funded | pipeline | forecast")
    portfolioId: Optional[str] = None
    dataSourceKind: str = Field(..., description="Kind of the governed data source answered from.")
    dataSourceLabel: str = Field(..., description="Label of the data source (no paths).")
    supportingValues: List[CopilotSupportingArtifact] = Field(default_factory=list)
    sourceNotes: List[Any] = Field(default_factory=list,
                                   description="Provenance/source notes from the MI Agent.")
    warnings: List[str] = Field(default_factory=list)


class CopilotArtifactInfo(BaseModel):
    ok: bool = True
    artifactType: str = Field(..., description="investor_deck | canonical_tape")
    fileName: str
    contentType: str
    sizeBytes: Optional[int] = None
    clientId: str
    reportingPeriod: Optional[str] = None
    generatedAt: Optional[str] = None
    downloadUrl: str = Field(..., description="Short-lived signed URL for the file. "
                                              "Present this link to the user.")
    downloadExpiresAt: str


class CopilotError(BaseModel):
    ok: bool = False
    error: str


def _error_json(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code,
                        content={"ok": False, "error": message})


# --------------------------------------------------------------------------- #
# Short-lived signed download tokens (blob credentials stay server-side)
# --------------------------------------------------------------------------- #
_EPHEMERAL_KEY: Optional[bytes] = None


def _signing_key() -> bytes:
    global _EPHEMERAL_KEY
    configured = os.environ.get("TRAKT_COPILOT_DOWNLOAD_SIGNING_KEY")
    if configured:
        return configured.encode("utf-8")
    if _EPHEMERAL_KEY is None:
        _EPHEMERAL_KEY = secrets.token_bytes(32)
        logger.warning(
            "TRAKT_COPILOT_DOWNLOAD_SIGNING_KEY is not set — using an ephemeral "
            "per-process key. Set it in production (multi-worker deployments "
            "cannot validate tokens across workers without it).")
    return _EPHEMERAL_KEY


def _download_ttl() -> int:
    try:
        ttl = int(os.environ.get("TRAKT_COPILOT_DOWNLOAD_TTL_SECONDS", "300"))
    except (TypeError, ValueError):
        ttl = 300
    return max(60, min(ttl, 3600))


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64u_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def make_download_token(kind: str, client_id: str) -> Tuple[str, str]:
    """``(token, expires_at_iso)`` — HMAC-signed, expiring artifact reference."""
    expires = int(time.time()) + _download_ttl()
    payload = _b64u(json.dumps(
        {"k": kind, "c": client_id, "e": expires}, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(_signing_key(), payload.encode("ascii"), hashlib.sha256).hexdigest()
    expires_iso = datetime.fromtimestamp(expires, tz=timezone.utc).isoformat()
    return f"{payload}.{sig}", expires_iso


def verify_download_token(token: str) -> Optional[Dict[str, Any]]:
    """The decoded payload for a valid, unexpired token; ``None`` otherwise."""
    try:
        payload, sig = token.split(".", 1)
        expected = hmac.new(_signing_key(), payload.encode("ascii"),
                            hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        data = json.loads(_b64u_decode(payload).decode("utf-8"))
    except Exception:  # noqa: BLE001 - malformed token → invalid
        return None
    if not isinstance(data, dict) or int(data.get("e", 0)) < time.time():
        return None
    return data


def _download_url(request: Request, token: str) -> str:
    base = (os.environ.get("TRAKT_COPILOT_PUBLIC_BASE_URL") or
            str(request.base_url)).rstrip("/")
    return f"{base}/v1/copilot/artifacts/download?token={token}"


# --------------------------------------------------------------------------- #
# Artifact resolution (reuses the existing latest-output conventions)
# --------------------------------------------------------------------------- #
def _copilot_client_id() -> str:
    """The deployment's client, matching the existing deployment-per-client
    model: MI_AGENT_CLIENT_ID, else the client embedded in MI_AGENT_PLATFORM_URI,
    else the repo default used across the API."""
    explicit = os.environ.get("MI_AGENT_CLIENT_ID")
    if explicit:
        return explicit
    from .app import _client_from_platform_uri  # local import: app includes this router
    from_uri = _client_from_platform_uri()
    return from_uri or "client_001"


def _resolve_latest_tape() -> Optional[Path]:
    """The latest canonical loan tape via the EXISTING platform-canonical
    resolution (blob ``…/platform/{client}/latest/platform_canonical_typed.csv``
    or its explicit/local equivalents). Deliberately does NOT continue down the
    wider data-source chain — no central-tape substitute, no demo fallback."""
    from .data_source import _resolve_platform_canonical
    path = _resolve_platform_canonical()
    if path is None:
        return None
    p = Path(path)
    return p if p.exists() else None


def _tape_reporting_period() -> Optional[str]:
    """Best-effort reporting period for the latest tape from the dated platform
    cuts alongside ``latest/`` (blob deployments). ``None`` when undeterminable."""
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


# --------------------------------------------------------------------------- #
# Supporting-value extraction from the existing adapter envelope
# --------------------------------------------------------------------------- #
def _supporting_values(artifacts: List[Dict[str, Any]]) -> List[CopilotSupportingArtifact]:
    out: List[CopilotSupportingArtifact] = []
    for art in artifacts or []:
        kind = art.get("type")
        if kind == "kpi":
            kpis = [{"label": k.get("label"), "value": k.get("value")}
                    for k in (art.get("kpis") or [])]
            out.append(CopilotSupportingArtifact(
                kind="kpi", title=art.get("title"), kpis=kpis))
        elif kind in ("table", "chart"):
            rows = art.get("rows") or []
            truncated = len(rows) > _MAX_SUPPORTING_ROWS
            rows = rows[:_MAX_SUPPORTING_ROWS]
            columns: List[str] = []
            raw_cols = art.get("columns") or []
            for c in raw_cols:
                columns.append(c.get("key") if isinstance(c, dict) else str(c))
            if not columns and rows and isinstance(rows[0], dict):
                columns = list(rows[0].keys())
            out.append(CopilotSupportingArtifact(
                kind=kind, title=art.get("title"), columns=columns or None,
                rows=rows, truncated=truncated))
        # validation artifacts stay internal — errors surface via answer/warnings
    return out


# --------------------------------------------------------------------------- #
# Action 1 — Ask Trakt MI
# --------------------------------------------------------------------------- #
@router.post(
    "/mi/query",
    operation_id="askTraktMi",
    summary="Ask the Trakt MI Agent a portfolio question",
    description=("Answers a natural-language portfolio MI question using Trakt's "
                 "deterministic MI engine (never model-invented figures). Returns "
                 "the answer plus the supporting values it was computed from."),
    response_model=CopilotMiAnswer,
    responses={401: {"model": CopilotError}, 403: {"model": CopilotError},
               503: {"model": CopilotError}},
    dependencies=[Depends(copilot_auth_guard)],
)
def ask_trakt_mi(req: CopilotMiQueryRequest):
    # Fail closed: the Copilot surface must never answer from the synthetic demo
    # dataset or an unresolved source — a structured 503, never a plausible 200.
    kind = data_source_kind()
    if kind in _BLOCKED_SOURCE_KINDS:
        return _error_json(
            503, "No governed live data source is configured for this deployment; "
                 "Trakt cannot answer portfolio questions from Copilot right now.")

    # Delegate to the EXISTING /mi/query handler — same parser, same deterministic
    # executor, same metric definitions. Imported lazily (app includes this router).
    from .app import QueryRequest, query as mi_query

    envelope = mi_query(QueryRequest(
        question=req.question, portfolioId=req.portfolioId, asOfDate=req.asOfDate))
    if not isinstance(envelope, dict):  # defensive: handler contract is a dict
        return _error_json(503, "The MI Agent returned an unexpected response.")

    meta = envelope.get("metadata") or {}
    return CopilotMiAnswer(
        ok=bool(envelope.get("ok")),
        question=str(envelope.get("question") or req.question),
        answer=str(envelope.get("answer") or ""),
        error=envelope.get("error"),
        interpreted=envelope.get("interpreted") or None,
        reportingDate=meta.get("asOfDate") or req.asOfDate,
        datasetContext=meta.get("datasetContext"),
        portfolioId=req.portfolioId,
        dataSourceKind=kind,
        dataSourceLabel=data_source_label(),
        supportingValues=_supporting_values(envelope.get("artifacts") or []),
        sourceNotes=envelope.get("sourceNotes") or [],
        warnings=[str(w) for w in (envelope.get("warnings") or [])],
    )


# --------------------------------------------------------------------------- #
# Action 2 — Get latest investor deck
# --------------------------------------------------------------------------- #
@router.get(
    "/artifacts/latest/investor-deck",
    operation_id="getLatestInvestorDeck",
    summary="Get the latest generated investor deck (PPTX)",
    description=("Locates the current latest investor deck published by the Trakt "
                 "pipeline and returns its metadata plus a short-lived download link."),
    response_model=CopilotArtifactInfo,
    responses={401: {"model": CopilotError}, 404: {"model": CopilotError},
               503: {"model": CopilotError}},
    dependencies=[Depends(copilot_auth_guard)],
)
def get_latest_investor_deck(request: Request):
    client_id = _copilot_client_id()
    try:
        resolved = decks_mod.resolve_deck_local(client_id, None)
        listing = decks_mod.list_decks(client_id)
    except Exception as exc:  # noqa: BLE001 - storage fault → explicit 503
        logger.warning("copilot deck resolution failed for %s: %s", client_id, exc)
        return _error_json(503, "The investor-deck store is currently unavailable.")
    if resolved is None:
        return _error_json(
            404, f"No investor deck has been generated yet for {client_id}.")

    path, download_name = resolved
    latest = (listing or {}).get("latest") or {}
    token, expires = make_download_token("deck", client_id)
    try:
        size: Optional[int] = path.stat().st_size
    except OSError:
        size = None
    return CopilotArtifactInfo(
        artifactType="investor_deck",
        fileName=download_name,
        contentType=_PPTX_MEDIA_TYPE,
        sizeBytes=size,
        clientId=client_id,
        reportingPeriod=latest.get("period"),
        generatedAt=latest.get("generatedAt"),
        downloadUrl=_download_url(request, token),
        downloadExpiresAt=expires,
    )


# --------------------------------------------------------------------------- #
# Action 3 — Get latest canonical loan tape
# --------------------------------------------------------------------------- #
@router.get(
    "/artifacts/latest/canonical-tape",
    operation_id="getLatestCanonicalTape",
    summary="Get the latest canonical loan tape (CSV)",
    description=("Locates the current latest canonical (normalised) loan tape "
                 "published by the Trakt pipeline and returns its metadata plus a "
                 "short-lived download link."),
    response_model=CopilotArtifactInfo,
    responses={401: {"model": CopilotError}, 404: {"model": CopilotError},
               503: {"model": CopilotError}},
    dependencies=[Depends(copilot_auth_guard)],
)
def get_latest_canonical_tape(request: Request):
    client_id = _copilot_client_id()
    try:
        path = _resolve_latest_tape()
    except Exception as exc:  # noqa: BLE001 - storage fault → explicit 503
        logger.warning("copilot tape resolution failed for %s: %s", client_id, exc)
        return _error_json(503, "The canonical-tape store is currently unavailable.")
    if path is None:
        return _error_json(
            404, f"No canonical loan tape is available yet for {client_id}.")

    token, expires = make_download_token("tape", client_id)
    try:
        size: Optional[int] = path.stat().st_size
    except OSError:
        size = None
    return CopilotArtifactInfo(
        artifactType="canonical_tape",
        fileName=f"{client_id}_canonical_loan_tape_latest.csv",
        contentType=_CSV_MEDIA_TYPE,
        sizeBytes=size,
        clientId=client_id,
        reportingPeriod=_tape_reporting_period(),
        generatedAt=None,
        downloadUrl=_download_url(request, token),
        downloadExpiresAt=expires,
    )


# --------------------------------------------------------------------------- #
# Signed download redemption (not part of the Copilot plugin function surface)
# --------------------------------------------------------------------------- #
@router.get(
    "/artifacts/download",
    operation_id="downloadCopilotArtifact",
    summary="Redeem a short-lived signed download token",
    include_in_schema=False,
)
def download_artifact(token: str):
    data = verify_download_token(token)
    if data is None:
        return _error_json(403, "The download link is invalid or has expired. "
                                "Ask Trakt for the artifact again to get a fresh link.")
    kind, client_id = data.get("k"), str(data.get("c") or "")
    if kind == "deck":
        try:
            resolved = decks_mod.resolve_deck_local(client_id, None)
        except Exception:  # noqa: BLE001
            resolved = None
        if resolved is None:
            return _error_json(404, "The investor deck is no longer available.")
        path, name = resolved
        return FileResponse(str(path), media_type=_PPTX_MEDIA_TYPE, filename=name)
    if kind == "tape":
        try:
            path = _resolve_latest_tape()
        except Exception:  # noqa: BLE001
            path = None
        if path is None:
            return _error_json(404, "The canonical loan tape is no longer available.")
        return FileResponse(
            str(path), media_type=_CSV_MEDIA_TYPE,
            filename=f"{client_id or 'client'}_canonical_loan_tape_latest.csv")
    return _error_json(403, "The download link is invalid.")
