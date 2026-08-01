"""FastAPI app exposing the existing MI Agent to the React UI.

Endpoints:
  GET  /health         - liveness + data-source status
  GET  /mi/catalogue   - real semantic layer (states/dimensions/measures/...)
  POST /mi/query       - run one MI question through run_mi_agent_query

Run:
  uvicorn mi_agent_api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .auth import auth_guard, principal_from_request
from . import gateway

from mi_agent.mi_agent_config import get_llm_config
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_validator import load_mi_semantics

from .adapters import adapt_workflow_result
from .catalogue import build_catalogue
from .data_source import (
    KIND_PLATFORM_CANONICAL,
    data_source_info,
    data_source_kind,
    data_source_label,
    get_dataframe,
    semantics_path,
)
from . import snapshots as snapshots_mod
from . import currency as currency_mod
from . import platform_snapshots_blob as platform_blob_mod
from . import pipeline_contract as pipeline_mod
from . import pipeline_history
from . import forecast_bridge as forecast_mod
from . import workspace as workspace_mod
from . import evolution as evolution_mod
from . import chat_routing as chat_routing_mod
from . import pipeline_timing as timing_mod
from . import decks as decks_mod
from . import cohorts as cohorts_mod
from . import geo as geo_mod
from . import portfolio_context as portfolio_ctx_mod
# The shared governed MI application service both channels (React + Copilot)
# call. It imports this module's data resolvers lazily, so there is no cycle.
from . import mi_service

# Dataset resolution now lives in an interface-neutral module (no FastAPI), so the
# governed capability, the deck generator and any future adapter can resolve data
# without importing this HTTP module. Re-exported here under their original names
# so the routes below — and existing callers/tests that reference
# ``mi_agent_api.app._<helper>`` — are unchanged.
from trakt_core.context import CHANNEL_REACT
from trakt_core.portfolio import (
    CAP_CONSOLIDATED_FORECAST,
    CAP_ORIGINATION_FORECAST,
    CAP_PIPELINE,
    CAP_RUNOFF_FORECAST,
    REASON_NON_ORIGINATING,
)
from trakt_core.errors import TraktError
from trakt_core.runtime import runtime_mode, validate_runtime_mode

from . import artefacts as artefacts_mod
from . import identity as identity_mod
from . import presenters
from .dependencies import default_tenant_id

from .datasets import (  # noqa: F401  (re-exported for backward compatibility)
    # The mutable resolution caches are re-exported as the SAME objects, so an
    # existing caller that clears ``app._PIPELINE_MIRROR_CACHE`` still clears the
    # cache ``datasets`` actually reads.
    _CLIENT_CURRENCY_CACHE,
    _PIPELINE_MIRROR_CACHE,
    _PIPELINE_URI_CACHE,
    _apply_request_currency,
    _blob_dated_snapshots,
    _blob_funded_evolution,
    _blob_platform_index,
    _clean_provenance_value,
    _client_from_platform_uri,
    _evo_ids,
    _funded_date_from_run,
    _kfi_completion_lag_weeks,
    _kfi_lag_weeks_from_model,
    _latest_pipeline_extract_date,
    _materialise_pipeline_root,
    _mi_llm_config,
    _onboarding_output_root,
    _period_from_platform_uri,
    _pid_label,
    _pipeline_discovery_root,
    _pipeline_history,
    _pipeline_root,
    _pipeline_root_from_uri,
    _platform_client_id,
    _platform_reporting_date,
    _platform_snapshot_index,
    _resolve_pipeline_source,
    _resolve_pipeline_uri_local,
    _resolve_query_frame,
    _resolve_run_dataframe,
    _scan_any_date_column,
    _weekly_files_window,
)

logger = logging.getLogger("mi_agent_api")

# Fail closed at import time on an unsafe runtime mode: a non-production mode is
# refused outright inside Azure, so a stray app setting cannot turn a deployed
# API into one that answers from fixture or synthetic data. Raising here means a
# misconfigured deployment does not start, rather than starting and quietly
# serving unapproved answers.
validate_runtime_mode()


def _execution_context(request: "Request", *, channel: str):
    """The trusted context for this request.

    The tenant is deployment configuration, the actor is the authenticated
    principal, and the request id honours an inbound ``X-Request-Id`` /
    ``X-Correlation-Id`` so a caller can correlate its own trace with the audit
    event. Raises :class:`~trakt_core.errors.TraktError` when identity cannot be
    established — the caller maps that to its status code.
    """
    principal = (getattr(request.state, "principal", None)
                 or principal_from_request(request))
    return identity_mod.context_from_principal(
        principal,
        tenant_id=default_tenant_id(),
        channel=channel,
        request_id=request.headers.get("x-request-id") or None,
        correlation_id=request.headers.get("x-correlation-id") or None,
    )


def _warm_caches() -> None:
    """Best-effort warm so the FIRST user request isn't cold. Loads the active
    dataset (populating the signature cache) and parses the semantics registry
    (populating its mtime cache). Never fatal: a deploy with no data source yet
    still starts; the first request simply pays the cold cost as before."""
    try:
        get_dataframe()
    except Exception as exc:  # noqa: BLE001 - warming must never block startup
        logger.info("startup dataset warm skipped: %s", exc)
    try:
        load_mi_semantics(semantics_path())
    except Exception as exc:  # noqa: BLE001
        logger.info("startup semantics warm skipped: %s", exc)


@asynccontextmanager
async def _lifespan(_app: "FastAPI"):
    _warm_caches()
    yield


# Global authentication guard: every /mi/* route requires an authenticated
# principal carrying an MI role (client|operator). Probe/index/docs routes stay
# open. Enforcement is toggled by MI_AGENT_AUTH_ENABLED (default on); see auth.py.
app = FastAPI(title="Trakt MI Agent API", version="1.0.0",
              dependencies=[Depends(auth_guard)], lifespan=_lifespan)

# CORS. With the SWA linked-backend deployment the UI calls the API same-origin,
# so CORS is not relied on for security. We still restrict it: allowed origins
# come from MI_AGENT_CORS_ORIGINS (comma-separated) and default to the local dev
# servers only. There is deliberately NO "*" fallback — an unset/empty value
# denies cross-origin browser calls rather than opening to any origin.
_origins = os.environ.get(
    "MI_AGENT_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:4173",
).split(",")
# The deployed front end's origin, when the UI is served cross-origin rather than
# through the linked-backend proxy. Additive; there is still deliberately no "*".
_origins += gateway.extra_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Accept the gateway prefix this deployment sits behind (default "/api") as well
# as the bare paths. Installed AFTER CORS so it is the outermost layer and every
# downstream component — CORS, the auth guard, the router — sees one normalised
# path regardless of which front door the request came through. See gateway.py
# for why: without it, the Static Web Apps linked-backend topology forwards
# /api/mi/query to an app that only serves /mi/query, and every question 404s.
API_PREFIX = gateway.install_gateway_prefix(app)


@app.exception_handler(TraktError)
async def _trakt_error_handler(request: Request, exc: TraktError) -> JSONResponse:
    """Map a governed error onto its stable HTTP status and machine-readable body.

    The status comes from the shared code table in ``trakt_core.errors``, so a
    given code produces the same classification here and in the Copilot adapter.
    """
    logger.info("governed error on %s %s: %s", request.method, request.url.path, exc.code)
    return JSONResponse(status_code=exc.http_status,
                        content={"ok": False, "error": exc.message,
                                 "errorCode": exc.code, "retryable": exc.retryable,
                                 "category": exc.category})


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fail safe: never leak a stack trace / internal path to a client. Unhandled
    errors (e.g. from the /mi/query workflow) become a generic 500 payload; the
    detail is logged server-side only."""
    logger.exception("unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": "An internal error occurred processing the request."},
    )


class PortfolioContext(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    entity: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    portfolio: Optional[PortfolioContext] = None
    portfolioId: Optional[str] = None
    asOfDate: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    # Active workspace view the question runs against (funded | pipeline |
    # forecast). Explicit wording in the question overrides this. ``context`` may
    # also carry ``{"activeView": ...}``.
    datasetContext: Optional[str] = None
    context: Optional[Any] = None
    # Selected source-portfolio lens: "total" | "direct" | "acquired" | a cohort
    # id ("direct_001" / "acquired_001"). Acts as the default scope; a portfolio
    # named in the question overrides it. Realised as a provenance filter.
    sourcePortfolioLens: Optional[str] = None


@app.get("/")
def root() -> Dict[str, Any]:
    """Friendly index so the bare URL isn't a confusing 404."""
    return {
        "service": "mi_agent_api",
        "version": app.version,
        "endpoints": ["/health", "/mi/catalogue", "/mi/snapshots", "/mi/snapshot",
                      "/mi/pipeline/snapshots", "/mi/pipeline/snapshot",
                      "/mi/forecast/snapshot", "/mi/workspace/view", "/mi/query"],
        "hint": "GET /health for data-source status; POST /mi/query to ask a question.",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    csv = data_source_label()
    info = data_source_info()
    return {
        "ok": True,
        "service": "mi_agent_api",
        "version": app.version,
        "dataSource": csv,
        "dataSourceKind": info.get("kind"),
        "preparationApplied": info.get("preparation_applied", False),
        "dimensionsAvailable": info.get("dimensions_available", []),
        "missingDimensions": info.get("missing_dimensions", []),
        "missingDimensionNames": [
            m["dimension"] if isinstance(m, dict) else m
            for m in info.get("missing_dimensions", [])
        ],
        # The single MI dataset contract: per-field metadata + display hints
        # (format + storage scale) so React never guesses field meaning or scale.
        "datasetContract": info.get("dataset_contract", {}),
        # NOTE: the full ``info`` dict is intentionally NOT echoed here — it
        # carries the server-side dataset file path. Expose only non-sensitive
        # summary fields above.
        "dataAvailable": csv != "unavailable",
        "semantics": semantics_path().name,
        # LLM parser availability (ENABLE_LLM_MI_AGENT + key). The chat runs
        # deterministically when unavailable — surface which mode is live.
        "llm": get_llm_config().to_dict(),
        # Governance posture. Surfaced so a deployment check can assert it
        # without reading Azure config by hand: which runtime mode is live
        # (production refuses fixture/synthetic sources), whether the injected
        # client-principal header can be trusted, and which tenant this
        # deployment serves. No secrets, no paths.
        "governance": {
            "runtimeMode": runtime_mode(),
            "tenantId": default_tenant_id(),
            "platformAuth": identity_mod.platform_auth_status(),
        },
        # How this deployment is reachable. Surfaced so a 404 can be diagnosed
        # with one request: it states the gateway prefix the API accepts and the
        # exact path forms the chat endpoint answers on, instead of leaving the
        # caller to infer the topology from a build log.
        "routing": {
            "apiPrefix": API_PREFIX or None,
            "queryPaths": [p for p in ("/mi/query",
                                       f"{API_PREFIX}/mi/query" if API_PREFIX else None)
                           if p],
        },
    }


@app.get("/me")
def me(request: Request) -> Dict[str, Any]:
    """The authenticated caller as the API resolved them (identity + MI roles).

    Useful for the UI to show who is signed in and whether they hold the
    operator role. Requires authentication (via the global guard)."""
    principal = getattr(request.state, "principal", None) or principal_from_request(request)
    if principal is None:
        return {"authenticated": False}
    return {"authenticated": True, **principal.to_public()}


@app.get("/mi/catalogue")
def catalogue() -> Dict[str, Any]:
    return build_catalogue()


def _resolve_portfolio_context(portfolio_context: Optional[str],
                               client_id: Optional[str] = None, df=None):
    """The governed portfolio context for a request. Never raises.

    Every portfolio-scoped route resolves through THIS — one registry, one scope
    resolver, one capability resolver — so a route can never invent its own idea
    of what "Direct" contains or what an acquired book may do."""
    try:
        return portfolio_ctx_mod.resolve_context(
            portfolio_context, df=df, client_id=client_id)
    except Exception as exc:  # noqa: BLE001 - scope resolution must never 500
        logger.warning("portfolio context resolution failed (%r): %s",
                       portfolio_context, exc)
        return None


def _scoped_frame(df, resolved):
    """Narrow a frame to a resolved context's scope (no-op for Total / None)."""
    if df is None or resolved is None:
        return df
    from mi_agent.portfolio_scope import apply_scope
    try:
        return apply_scope(df, resolved.scope)
    except Exception as exc:  # noqa: BLE001 - filtering must never 500
        logger.warning("portfolio scope filtering failed: %s", exc)
        return df


def _scope_block(df, resolved, *, fields: Optional[List[str]] = None
                 ) -> Optional[Dict[str, Any]]:
    """The governed scope + coverage block a portfolio-scoped response carries.

    ``df`` is the UNSCOPED frame, so the block can state which portfolios in
    scope had no rows rather than silently omitting them."""
    if resolved is None:
        return None
    try:
        return portfolio_ctx_mod.scope_metadata(
            df, resolved.scope, capabilities=resolved.capabilities, fields=fields)
    except Exception as exc:  # noqa: BLE001 - disclosure must never 500
        logger.warning("portfolio scope disclosure failed: %s", exc)
        return None


def _pipeline_scope_gate(portfolio_context: Optional[str], client_id: Optional[str],
                         dataset: str, **extra) -> tuple:
    """``(resolved, refusal)`` for a pipeline-family route.

    The governed capability resolver decides whether an origination pipeline
    APPLIES to an explicitly selected portfolio scope. When it does not, the
    route returns a controlled NOT-APPLICABLE response carrying the business
    reason — never an empty pipeline that reads as "no cases this week".

    Two deliberate limits keep this additive:

    * With no ``portfolioContext`` the route behaves exactly as before. A caller
      that never asked for a scope must not be refused by one.
    * Only a BUSINESS refusal gates the route: a scope where nothing originates.
      "No extract published yet" is a data-availability condition the route
      already reports through its own no-source path, and gating on it here
      would turn a transient gap into a hard refusal.
    """
    if not portfolio_context:
        return None, None
    resolved = _resolve_portfolio_context(portfolio_context, client_id)
    if resolved is None or not resolved.registry:
        # No governed registry (e.g. a dataset with no source provenance) —
        # there is no portfolio applicability question to answer.
        return resolved, None
    state = resolved.capability(CAP_PIPELINE)
    if (state is not None and not state.enabled
            and state.reason_code == REASON_NON_ORIGINATING):
        refusal = {
            "ok": False, "dataset": dataset, "applicable": False,
            "reason": state.detail or "Pipeline does not apply to this portfolio scope.",
            "reasonCode": state.reason_code,
            "portfolioScope": resolved.scope.to_dict(),
            "pipelineCapability": state.to_dict(),
        }
        refusal.update(extra)
        return resolved, refusal
    return resolved, None


def _originates(resolved, portfolio_context: Optional[str]) -> bool:
    """Should the pipeline contribute to this request?

    Only an EXPLICIT scope whose governed capability says nothing in it
    originates suppresses the pipeline. An unscoped request, an unknown
    registry, or a merely-undiscovered extract all keep the existing behaviour.
    """
    if not portfolio_context or resolved is None or not resolved.registry:
        return True
    state = resolved.capability(CAP_ORIGINATION_FORECAST)
    return not (state is not None and not state.enabled
                and state.reason_code == REASON_NON_ORIGINATING)


@app.get("/mi/portfolio-context")
def portfolio_context() -> Dict[str, Any]:
    """THE governed portfolio contract for the workspace.

    The dynamic hierarchy (Total → type groups → every source portfolio), each
    portfolio's governed metadata (type, origination capability, forecast
    treatment, runoff profile, reporting-date coverage), and the resolved
    capability set for every selectable context.

    This is the single source every client-facing channel gates on. A channel
    renders what it finds here; it never infers applicability from a portfolio
    name, a type string or the presence of a field. Adding ``direct_002`` to the
    platform changes this response — and therefore the whole workspace — with no
    code or configuration change in any channel.
    """
    try:
        return portfolio_ctx_mod.context_index()
    except Exception as exc:  # noqa: BLE001 - the selector must never 500
        logger.warning("portfolio context index failed: %s", exc)
        return {"available": False, "contexts": [], "portfolios": [],
                "portfolio_types": [], "default_context_id": None,
                "pipeline_portfolios": None, "error": str(exc)}


@app.get("/mi/source-portfolios")
def source_portfolios() -> Dict[str, Any]:
    """Discover the source-portfolio lenses present in the active dataset.

    Returns Total, Direct / Acquired (when present), and one entry per source
    cohort (direct_001 / acquired_001 / …) — the options for the UI dropdown.
    Each lens carries ``funded_only`` so the UI hides Pipeline / Forecast for
    acquired-only scopes. When the active dataset carries no provenance, only
    Total is returned (``available=false``).
    """
    from mi_agent import portfolio_lens as plens
    try:
        df = get_dataframe()
    except Exception as exc:  # never 500 the dropdown
        return {"available": False, "lenses": plens.available_lenses([]),
                "source": "unavailable", "error": str(exc)}

    cols = set(df.columns)
    if "source_portfolio_id" not in cols and "source_portfolio_type" not in cols:
        return {"available": False, "lenses": plens.available_lenses([]),
                "source": data_source_label()}

    keep = [c for c in ("source_portfolio_id", "source_portfolio_type",
                        "source_portfolio_label") if c in cols]
    records = (df[keep].drop_duplicates().to_dict("records")) if keep else []
    records = [{k: _clean_provenance_value(v) for k, v in r.items()} for r in records]
    lenses = plens.available_lenses(records)
    return {
        "available": len(lenses) > 1,
        "lenses": lenses,
        "source": data_source_label(),
    }


@app.get("/mi/snapshots")
def snapshots() -> Dict[str, Any]:
    """Data-driven discovery of available funded portfolios and reporting runs.

    The portfolio / reporting-date dropdowns are built from THIS — only real
    output appears (no hardcoded prototype options). A ``blob://`` onboarding
    output root enumerates the dated platform canonicals (one run per funded cut);
    an on-disk root uses the onboarding-tape walk; and either way, when nothing is
    discovered, it falls back to the loaded platform canonical (latest).
    """
    root = _onboarding_output_root()
    if root and platform_blob_mod.is_blob_root(root):
        idx = _blob_platform_index(root)
        if idx and idx.get("portfolios"):
            return idx
        # Nothing dated under the blob root → the loaded latest canonical.
        platform = _platform_snapshot_index()
        if platform is not None:
            return platform
        return {"portfolios": [], "source": root}
    if root:
        try:
            result = snapshots_mod.discover_snapshots(root)
        except Exception as exc:  # noqa: BLE001 - discovery must never 500
            logger.warning("snapshot discovery failed: %s", exc)
            return {"portfolios": [], "source": "error", "error": str(exc)}
        if result.get("portfolios"):
            result["source"] = root
            return result
        # On-disk root discovered nothing → loaded platform canonical, if any.
        platform = _platform_snapshot_index()
        if platform is not None:
            return platform
        result["source"] = root
        return result
    # No on-disk root: derive portfolios from the loaded platform canonical.
    platform = _platform_snapshot_index()
    if platform is not None:
        return platform
    return {"portfolios": [], "source": "unavailable"}


@app.get("/mi/snapshot")
def snapshot(portfolioId: Optional[str] = None,
             client_id: Optional[str] = None,
             run_id: Optional[str] = None,
             portfolioContext: Optional[str] = None) -> Dict[str, Any]:
    """Deterministic funded-book snapshot (KPIs + month-on-month change) for a run.

    ``portfolioId`` is ``"<client_id>/<run_id>"`` (matching the /mi/query contract);
    ``client_id`` + ``run_id`` may be passed separately instead.
    ``portfolioContext`` is the governed workspace scope (``total`` / a type group
    / a ``source_portfolio_id``); the KPIs, stratifications and the prior-period
    comparison are all computed over the SAME scoped rows, so the figures can
    never disagree with the context the workspace is displaying.
    """
    if portfolioId and "/" in portfolioId:
        client_id, run_id = portfolioId.split("/", 1)
    if not client_id or not run_id:
        return {"ok": False, "error": "portfolioId (client_id/run_id) is required",
                "kpis": [], "warnings": [], "diagnostics": []}

    root = _onboarding_output_root()
    df, prep_report = _resolve_run_dataframe(client_id, run_id, root)
    if df is None:
        return {"ok": False,
                "error": f"No funded dataset found for {client_id}/{run_id}.",
                "portfolio": {"client_id": client_id, "run_id": run_id},
                "kpis": [], "warnings": ["No funded data available for this run."],
                "diagnostics": []}

    semantics = load_mi_semantics(semantics_path())

    # Governed portfolio scope. Resolved once, applied to BOTH the current and
    # the prior frame so the month-on-month change compares like with like.
    resolved = _resolve_portfolio_context(portfolioContext, client_id, df)
    unscoped_df = df
    df = _scoped_frame(df, resolved)

    # Resolve the prior available run for month-on-month change.
    prior_df = prior_run_id = prior_reporting_date = None
    reporting_date = snapshots_mod.infer_reporting_date(run_id, df)
    if root and platform_blob_mod.is_blob_root(root):
        try:
            index = _blob_platform_index(root) or {"portfolios": []}
            prior = snapshots_mod.find_prior_run(index, client_id, run_id)
            if prior:
                prior_run_id = prior["run_id"]
                prior_reporting_date = prior["reporting_date"]
                prior_df, _ = _resolve_run_dataframe(client_id, prior_run_id, root)
        except Exception as exc:  # noqa: BLE001 - prior comparison is additive
            logger.warning("blob prior-run resolution failed: %s", exc)
    elif root:
        try:
            index = snapshots_mod.discover_snapshots(root)
            prior = snapshots_mod.find_prior_run(index, client_id, run_id)
            if prior:
                prior_run_id = prior["run_id"]
                prior_reporting_date = prior["reporting_date"]
                prior_tape = snapshots_mod.resolve_tape_path(root, client_id, prior_run_id)
                if prior_tape is not None:
                    prior_df, _ = snapshots_mod.load_prepared_run(prior_tape)
        except Exception as exc:  # noqa: BLE001 - prior comparison is additive
            logger.warning("prior-run resolution failed: %s", exc)

    result = snapshots_mod.compute_funded_snapshot(
        df, semantics, client_id=client_id, run_id=run_id,
        reporting_date=reporting_date, prep_report=prep_report,
        prior_df=_scoped_frame(prior_df, resolved), prior_run_id=prior_run_id,
        prior_reporting_date=prior_reporting_date,
        scope=resolved.scope if resolved else None,
    )
    block = _scope_block(unscoped_df, resolved)
    if block is not None:
        result["portfolioScope"] = block
    for d in result.get("diagnostics", []):
        logger.info("snapshot diagnostic [%s/%s]: %s", client_id, run_id, d)
    return result


@app.get("/mi/pipeline/snapshots")
def pipeline_snapshots(portfolioId: Optional[str] = None) -> Dict[str, Any]:
    """Data-driven discovery of governed pipeline sources and reporting dates."""
    configured = os.environ.get("MI_AGENT_PIPELINE_ROOT") or _pipeline_root()
    root = _materialise_pipeline_root(configured)
    if not root:
        return {"sources": [], "source": "unavailable"}
    client_id = portfolioId.split("/", 1)[0] if portfolioId else None
    try:
        sources = pipeline_mod.discover_pipeline_sources(root, client_id=client_id)
    except Exception as exc:  # noqa: BLE001 - discovery must never 500
        logger.warning("pipeline discovery failed: %s", exc)
        return {"sources": [], "source": "error", "error": str(exc)}
    # Report the ORIGINAL configured root (the blob:// URI), not the local mirror.
    return {"sources": sources, "source": configured}


@app.get("/mi/pipeline/snapshot")
def pipeline_snapshot(portfolioId: Optional[str] = None,
                      client_id: Optional[str] = None,
                      runId: Optional[str] = None,
                      run_id: Optional[str] = None,
                      portfolioContext: Optional[str] = None) -> Dict[str, Any]:
    """Deterministic pipeline single-source snapshot for the latest weekly cut.

    ``portfolioId`` is ``"<client_id>/<run_id>"`` (matching the funded contract);
    ``client_id`` + ``runId``/``run_id`` may be passed separately instead. The
    pipeline as-of/extract dates are exposed distinctly from the funded run date.
    """
    run_id = runId or run_id
    if portfolioId and "/" in portfolioId:
        client_id, run_id = portfolioId.split("/", 1)
    client_id = client_id or "client_001"

    resolved, refusal = _pipeline_scope_gate(
        portfolioContext, client_id, "pipeline",
        recordType="pipeline", portfolioId=f"{client_id}/{run_id or ''}",
        pipelineRowCount=0, stageBreakdown=[], availableMetrics=[],
        availableDimensions=[], dataQuality=[])
    if refusal is not None:
        return refusal

    source = _resolve_pipeline_source(client_id, run_id)
    if source is None:
        return {"ok": False, "recordType": "pipeline",
                "error": f"No governed pipeline source found for {client_id}.",
                "portfolioId": f"{client_id}/{run_id or ''}",
                "pipelineRowCount": 0, "stageBreakdown": [],
                "availableMetrics": [], "availableDimensions": [], "dataQuality": []}

    history = _pipeline_history(source.get("client_id", client_id))
    df, report = pipeline_mod.load_prepared_pipeline(source, historical_model=history)
    semantics = load_mi_semantics(semantics_path())
    prior_week = pipeline_mod.compute_prior_week_aggregates(source, historical_model=history)
    result = pipeline_mod.compute_pipeline_snapshot(
        df, report, semantics, client_id=source.get("client_id", client_id),
        run_id=run_id or source.get("run_id", ""), source=source, prior_week=prior_week)
    # Disclose the funded-vs-pipeline timing (never truncate): funded anchor = the
    # selected run's reporting date; pipeline anchor = the latest weekly extract.
    result["pipelineTiming"] = timing_mod.timing_disclosure(
        _funded_date_from_run(run_id), result.get("pipelineAsOfDate"))
    if resolved is not None:
        # Total stays Total: the workspace context is preserved and the response
        # states which portfolios the pipeline is actually sourced from.
        result["portfolioScope"] = resolved.scope.to_dict()
        state = resolved.capability(CAP_PIPELINE)
        if state is not None:
            result["pipelineCapability"] = state.to_dict()
    return result


@app.get("/mi/forecast/snapshot")
def forecast_snapshot(portfolioId: Optional[str] = None,
                      client_id: Optional[str] = None,
                      runId: Optional[str] = None,
                      run_id: Optional[str] = None,
                      portfolioContext: Optional[str] = None) -> Dict[str, Any]:
    """Deterministic funded + pipeline forecast bridge for a selected run.

    Composes the funded snapshot balance/count, the Phase 1 pipeline snapshot, and
    the config stage probabilities into ``forecastBridge`` (+ embedded
    ``pipelineSnapshot`` + ``watchlist``). Never 500s on a missing pipeline — it
    returns the funded balance with a blocked forecast-readiness status.
    """
    run_id = runId or run_id
    if portfolioId and "/" in portfolioId:
        client_id, run_id = portfolioId.split("/", 1)
    client_id = client_id or "client_001"
    if not run_id:
        return {"ok": False, "error": "portfolioId (client_id/run_id) is required",
                "forecastBridge": None, "pipelineSnapshot": None, "watchlist": []}

    semantics = load_mi_semantics(semantics_path())

    # Funded side (reuse the funded resolution; never merged with pipeline).
    root = _onboarding_output_root()
    funded_df, _funded_report = _resolve_run_dataframe(client_id, run_id, root)
    funded_reporting_date = snapshots_mod.infer_reporting_date(run_id, funded_df)

    # Governed portfolio scope: the funded side is narrowed to the selected
    # context, and the pipeline side is included ONLY where the capability
    # resolver says a portfolio in scope originates. An acquired-only scope
    # therefore contributes funded actuals with zero new originations — it never
    # silently inherits another book's pipeline.
    resolved = _resolve_portfolio_context(portfolioContext, client_id, funded_df)
    unscoped_funded = funded_df
    funded_df = _scoped_frame(funded_df, resolved)
    pipeline_applies = _originates(resolved, portfolioContext)

    # Pipeline side (Phase 1 prep + contract): the LATEST weekly extract for the
    # run's source scope. Its as-of/extract dates stay distinct from the funded date.
    pipeline_df = pipeline_report = pipeline_snap = None
    source = _resolve_pipeline_source(client_id, run_id) if pipeline_applies else None
    if source is not None:
        try:
            history = _pipeline_history(source.get("client_id", client_id))
            pipeline_df, pipeline_report = pipeline_mod.load_prepared_pipeline(
                source, historical_model=history)
            prior_week = pipeline_mod.compute_prior_week_aggregates(
                source, historical_model=history)
            pipeline_snap = pipeline_mod.compute_pipeline_snapshot(
                pipeline_df, pipeline_report, semantics,
                client_id=source.get("client_id", client_id),
                run_id=run_id, source=source, prior_week=prior_week)
        except Exception as exc:  # noqa: BLE001 - a bad pipeline must not 500
            logger.warning("pipeline load failed for forecast [%s/%s]: %s",
                           client_id, run_id, exc)
            pipeline_df = pipeline_report = pipeline_snap = None

    envelope = forecast_mod.compute_forecast_bridge(
        client_id=client_id, run_id=run_id, funded_reporting_date=funded_reporting_date,
        funded_df=funded_df, pipeline_df=pipeline_df,
        pipeline_report=pipeline_report, pipeline_snapshot=pipeline_snap,
        pipeline_source=source)
    # Forecast-by-dimension breakdowns (funded actual + weighted pipeline), derived
    # by aggregate composition — never a row merge.
    envelope["forecastBreakdowns"] = workspace_mod.forecast_breakdowns(funded_df, pipeline_df)
    basis = (pipeline_report or {}).get("completion_probability_basis")
    evidence = pipeline_history.historical_model_evidence(
        (pipeline_report or {}).get("historical_completion_model"), basis)
    envelope["historicalModelEvidence"] = evidence
    envelope["completionProbabilityBasis"] = basis
    envelope["lineage"] = workspace_mod.lineage_for(
        "forecast", funded_reporting_date=funded_reporting_date,
        pipeline_as_of_date=(source or {}).get("pipeline_as_of_date"),
        pipeline_source_folder_date=(source or {}).get("pipeline_source_folder_date"),
        current_pipeline_snapshot_date=(source or {}).get("current_pipeline_snapshot_date"),
        current_pipeline_source_file=(source or {}).get("current_pipeline_source_file"),
        completion_probability_basis=basis, historical_model_evidence=evidence)
    # Both anchors + non-blocking timing disclosure (funded actuals vs latest
    # pipeline). The forecast bridge composes funded actuals with the LATEST
    # pipeline; when the pipeline extract is later than the funded cut we disclose
    # it rather than hide the pipeline.
    pipeline_as_of = ((source or {}).get("pipeline_as_of_date")
                      or _latest_pipeline_extract_date(client_id))
    envelope["pipelineTiming"] = timing_mod.timing_disclosure(
        funded_reporting_date or _funded_date_from_run(run_id), pipeline_as_of)

    # Portfolio-aware consolidated projection: every portfolio in scope under its
    # OWN governed forecast treatment, with per-portfolio runoff disclosure.
    if resolved is not None:
        envelope["portfolioScope"] = _scope_block(unscoped_funded, resolved) \
            or resolved.scope.to_dict()
        for cap in (CAP_PIPELINE, CAP_ORIGINATION_FORECAST, CAP_RUNOFF_FORECAST,
                    CAP_CONSOLIDATED_FORECAST):
            state = resolved.capability(cap)
            if state is not None:
                envelope.setdefault("capabilities", {})[cap] = state.to_dict()
        try:
            weighted = float((envelope.get("forecastBridge") or {})
                             .get("weightedExpectedFundedAmount") or 0.0)
            envelope["portfolioProjections"] = forecast_mod.portfolio_projections(
                funded_df, resolved.registry, resolved.scope,
                weighted_pipeline=weighted if pipeline_applies else 0.0,
                pipeline_portfolios=resolved.pipeline_portfolios)
        except Exception as exc:  # noqa: BLE001 - projection must never 500
            logger.warning("portfolio projections failed for %s/%s: %s",
                           client_id, run_id, exc)
    return envelope


@app.get("/mi/evolution/funded")
def funded_evolution(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                     toRunId: Optional[str] = None, to_run_id: Optional[str] = None,
                     portfolioContext: Optional[str] = None
                     ) -> Dict[str, Any]:
    """Funded time series across monthly runs up to ``toRunId`` (per-period
    reconciliation + lineage), narrowed to the governed ``portfolioContext``.
    Never 500s — returns an empty series on no data."""
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    if not root:
        return {"dataset": "funded", "portfolioId": cid, "toRunId": trid,
                "periods": [], "breakdowns": {}, "singlePeriod": True,
                "error": "no onboarding output root configured"}
    resolved = _resolve_portfolio_context(portfolioContext, cid)
    scope = resolved.scope if resolved else None
    try:
        if platform_blob_mod.is_blob_root(root):
            result = _blob_funded_evolution(root, cid, trid, scope=scope)
        else:
            result = evolution_mod.funded_evolution(root, cid, trid, scope=scope)
        if scope is not None:
            result["portfolioScope"] = scope.to_dict()
        return result
    except Exception as exc:  # noqa: BLE001 - evolution must never 500
        logger.warning("funded evolution failed: %s", exc)
        return {"dataset": "funded", "portfolioId": cid, "toRunId": trid,
                "periods": [], "breakdowns": {}, "singlePeriod": True, "error": str(exc)}


@app.get("/mi/evolution/pipeline")
def pipeline_evolution(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                       toRunId: Optional[str] = None, to_run_id: Optional[str] = None,
                       portfolioContext: Optional[str] = None
                       ) -> Dict[str, Any]:
    """Pipeline time series across governed weekly extracts (amount / cases / by
    stage over time), with per-period reconciliation + lineage."""
    cid, _funded_trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    # The pipeline is a continuous weekly operational view — the selected FUNDED
    # reporting date (carried in portfolioId) must NOT truncate it. Only an EXPLICIT
    # pipeline toRunId query param caps it (rare; no UI toggle needed by default).
    pipeline_cut = toRunId or to_run_id
    resolved, refusal = _pipeline_scope_gate(
        portfolioContext, cid, "pipeline", portfolioId=cid, toRunId=pipeline_cut,
        periods=[], byStage=[], singlePeriod=True)
    if refusal is not None:
        return refusal
    root = _pipeline_discovery_root()
    if not root:
        return {"dataset": "pipeline", "portfolioId": cid, "toRunId": pipeline_cut,
                "periods": [], "byStage": [], "singlePeriod": True,
                "error": "no pipeline root configured"}
    try:
        result = evolution_mod.pipeline_evolution(
            root, cid, pipeline_cut, historical_model=_pipeline_history(cid))
        # Disclose the funded-vs-pipeline timing on the evolution response too, so
        # the pipeline evolution view can surface the non-blocking banner.
        latest = None
        dates = [p.get("extract_date") for p in result.get("periods", [])]
        dates = [d for d in dates if d]
        if dates:
            latest = max(dates)
        result["pipelineTiming"] = timing_mod.timing_disclosure(
            _funded_date_from_run(_funded_trid), latest)
        if resolved is not None:
            result["portfolioScope"] = resolved.scope.to_dict()
            state = resolved.capability(CAP_PIPELINE)
            if state is not None:
                result["pipelineCapability"] = state.to_dict()
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("pipeline evolution failed: %s", exc)
        return {"dataset": "pipeline", "portfolioId": cid, "toRunId": pipeline_cut,
                "periods": [], "byStage": [], "singlePeriod": True, "error": str(exc)}


@app.get("/mi/evolution/funnel")
def funnel_evolution(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                     toRunId: Optional[str] = None, to_run_id: Optional[str] = None,
                     portfolioContext: Optional[str] = None
                     ) -> Dict[str, Any]:
    """Weekly origination funnel trends (KFI / Application / Offer / Completion
    value + count, 5-week average, latest week, delta vs prior week). Never 500s."""
    cid, _funded_trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    # Origination funnel is weekly-pipeline data — the funded reporting date must
    # NOT truncate it; only an explicit pipeline toRunId caps it.
    pipeline_cut = toRunId or to_run_id
    resolved, refusal = _pipeline_scope_gate(
        portfolioContext, cid, "pipeline_funnel", portfolioId=cid,
        toRunId=pipeline_cut, stages=[], weeks=[], series={}, summary={},
        singlePeriod=True)
    if refusal is not None:
        return refusal
    root = _pipeline_discovery_root()
    if not root:
        return {"dataset": "pipeline_funnel", "portfolioId": cid, "toRunId": pipeline_cut,
                "stages": [], "weeks": [], "series": {}, "summary": {},
                "singlePeriod": True, "error": "no pipeline root configured"}
    try:
        model = _pipeline_history(cid)  # built once: feeds both the lag and the cohort funnel
        lag_weeks = _kfi_lag_weeks_from_model(model)
        result = evolution_mod.pipeline_funnel_evolution(
            root, cid, pipeline_cut, lag_weeks=lag_weeks)
        # Canonical conversion = cumulative cohort progression (% of the KFI
        # cohort reaching each milestone to date). The weekly-flow rate on each
        # stage stays available as an operational velocity, not "conversion".
        if model:
            result["cohortProgression"] = model.get("cohortProgression")
            result["cumulativeCohortConversion"] = model.get("cumulativeCohortConversion")
        if resolved is not None:
            result["portfolioScope"] = resolved.scope.to_dict()
            state = resolved.capability(CAP_PIPELINE)
            if state is not None:
                result["pipelineCapability"] = state.to_dict()
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("funnel evolution failed: %s", exc)
        return {"dataset": "pipeline_funnel", "portfolioId": cid, "toRunId": pipeline_cut,
                "stages": [], "weeks": [], "series": {}, "summary": {},
                "singlePeriod": True, "error": str(exc)}


@app.get("/mi/cohorts")
def cohorts(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
            runId: Optional[str] = None, run_id: Optional[str] = None,
            grain: str = "Y", dimension: str = "vintage",
            portfolioContext: Optional[str] = None) -> Dict[str, Any]:
    """Funded origination-vintage (static-pool) cohort analysis for a run.

    Balance / loan count / book share and balance-weighted LTV, rate and
    months-on-book by origination year — computed from the governed funded tape.
    Returns ``available=false`` (with a reason) when the tape carries no vintage,
    so the UI never fabricates cohort metrics. Never 500s.
    """
    run_id = runId or run_id
    if portfolioId and "/" in portfolioId:
        client_id, run_id = portfolioId.split("/", 1)
    client_id = client_id or "client_001"
    pid = f"{client_id}/{run_id or ''}"
    if not run_id:
        return {"dataset": "cohorts", "portfolioId": pid, "available": False,
                "reason": "portfolioId (client_id/run_id) is required",
                "cohorts": [], "metricsAvailable": []}
    try:
        root = _onboarding_output_root()
        df, _report = _resolve_run_dataframe(client_id, run_id, root)
        reporting_date = snapshots_mod.infer_reporting_date(run_id, df)
        resolved = _resolve_portfolio_context(portfolioContext, client_id, df)
        result = cohorts_mod.cohort_analysis(
            _scoped_frame(df, resolved), client_id=client_id, portfolio_id=pid,
            reporting_date=reporting_date, grain=grain, dimension=dimension)
        block = _scope_block(df, resolved)
        if block is not None:
            result["portfolioScope"] = block
        return result
    except Exception as exc:  # noqa: BLE001 - cohort analysis must never 500
        logger.warning("cohort analysis failed for %s: %s", pid, exc)
        return {"dataset": "cohorts", "portfolioId": pid, "available": False,
                "reason": str(exc), "cohorts": [], "metricsAvailable": []}


@app.get("/mi/geo/exposure")
def geo_exposure(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                 runId: Optional[str] = None, run_id: Optional[str] = None,
                 portfolioContext: Optional[str] = None) -> Dict[str, Any]:
    """Funded exposure per UK ITL3 area (e.g. Bristol) for a run — the DATA layer
    for the geographic view. ITL3 comes from the tape's ITL3 field or is derived
    from the property postcode via the in-repo master lookup. Returns
    ``available=false`` (with a reason) when neither is present. Never 500s."""
    run_id = runId or run_id
    if portfolioId and "/" in portfolioId:
        client_id, run_id = portfolioId.split("/", 1)
    client_id = client_id or "client_001"
    pid = f"{client_id}/{run_id or ''}"
    if not run_id:
        return {"dataset": "geo_itl3", "portfolioId": pid, "available": False,
                "reason": "portfolioId (client_id/run_id) is required", "areas": []}
    try:
        df, _report = _resolve_run_dataframe(client_id, run_id, _onboarding_output_root())
        currency_mod.resolve_and_set(df)
        resolved = _resolve_portfolio_context(portfolioContext, client_id, df)
        result = geo_mod.exposure_by_itl3(_scoped_frame(df, resolved))
        result.update({"dataset": "geo_itl3", "portfolioId": pid,
                       "currencyCode": currency_mod.current_code()})
        block = _scope_block(df, resolved)
        if block is not None:
            result["portfolioScope"] = block
        return result
    except Exception as exc:  # noqa: BLE001 - geo view must never 500
        logger.warning("geo exposure failed for %s: %s", pid, exc)
        return {"dataset": "geo_itl3", "portfolioId": pid, "available": False,
                "reason": str(exc), "areas": []}


@app.get("/mi/cohorts/progression")
def cohort_progression(portfolioId: Optional[str] = None,
                       client_id: Optional[str] = None,
                       lens: Optional[str] = None,
                       portfolioContext: Optional[str] = None,
                       vintage: Optional[str] = None,
                       grain: str = "Y") -> Dict[str, Any]:
    """Static-pool cohort PROGRESSION: how a cohort's funded metrics (balance,
    loan count, WA LTV / rate, NNEG exposure / headroom) evolve ACROSS reporting
    periods. The cohort is a source-portfolio ``lens`` (total | direct | acquired
    | a cohort id such as ``acquired_001``) optionally narrowed to an origination
    ``vintage`` at ``grain`` (Y|Q|M) — e.g. acquired_001 loans originated in 2023.
    Never 500s; returns ``available=false`` with a reason when the cohort is empty.
    """
    cid = "client_001"
    if portfolioId and "/" in portfolioId:
        cid = portfolioId.split("/", 1)[0]
    elif portfolioId:
        cid = portfolioId
    cid = client_id or cid
    try:
        # ``portfolioContext`` is the governed workspace scope; ``lens`` is the
        # pre-existing parameter name and is accepted as its alias so existing
        # callers keep working. Either way the scope is RESOLVED through the
        # registry, so a group cohort spans every current member.
        resolved = _resolve_portfolio_context(portfolioContext or lens, cid)
        scope = resolved.scope if resolved else None
        result = evolution_mod.funded_cohort_progression(
            _onboarding_output_root(), cid,
            lens_filters=(scope.filters or None) if scope else None,
            lens_label=scope.label if scope else "Total",
            vintage=vintage, grain=grain)
        if scope is not None:
            result["portfolioScope"] = scope.to_dict()
        return result
    except Exception as exc:  # noqa: BLE001 - progression must never 500
        logger.warning("cohort progression failed for %s: %s", cid, exc)
        return {"dataset": "cohort_progression", "portfolioId": cid,
                "available": False, "reason": str(exc), "periods": [],
                "metricsAvailable": []}


_PPTX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation")


@app.get("/mi/decks")
def list_decks(portfolioId: Optional[str] = None,
               client_id: Optional[str] = None) -> Dict[str, Any]:
    """Discover investor PPTX decks published by the orchestration for a client.

    UI-safe: returns the ``latest`` deck pointer and the dated reporting-period
    decks available (never raw blob paths). Empty listing when none exist — the
    UI then shows a disabled 'No deck available' state. Never 500s.
    """
    cid, _trid = _evo_ids(portfolioId, client_id, None, None)
    try:
        return decks_mod.list_decks(cid)
    except Exception as exc:  # noqa: BLE001 - discovery must never 500
        logger.warning("deck discovery failed for %s: %s", cid, exc)
        return {"available": False, "latest": None, "decks": [], "client_id": cid,
                "error": str(exc)}


@app.get("/mi/decks/download")
def download_deck(request: Request, portfolioId: Optional[str] = None,
                  client_id: Optional[str] = None, period: Optional[str] = None):
    """Serve an investor PPTX deck (the latest, or a specific reporting period).

    The deck is selected by the AUTHENTICATED tenant, not by ``client_id``.
    ``client_id`` is retained for backward compatibility and is DEPRECATED: it is
    accepted when it matches the trusted tenant and refused (403) when it names
    another one. Previously it selected the deck outright, which allowed any
    authenticated user to fetch another tenant's investor pack from the shared
    deck container.
    """
    context = _execution_context(request, channel=CHANNEL_REACT)
    result = artefacts_mod.get_investor_pack(
        context, portfolio_id=portfolioId, period=period,
        requested_client_id=client_id)
    if not result.ok:
        err = result.error
        return JSONResponse(
            status_code=result.http_status,
            content={"ok": False, "error": err.message if err else "Unavailable.",
                     "errorCode": err.code if err else None,
                     "retryable": err.retryable if err else False,
                     "requestId": result.request_id})
    artefact = result.result
    return FileResponse(str(artefact.local_path), media_type=artefact.content_type,
                        filename=artefact.download_name)


@app.get("/mi/evolution/forecast")
def forecast_evolution(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                       toRunId: Optional[str] = None, to_run_id: Optional[str] = None,
                       portfolioContext: Optional[str] = None
                       ) -> Dict[str, Any]:
    """Forecast bridge over time (funded balance + weighted pipeline per run),
    narrowed to the governed ``portfolioContext``. The pipeline contribution is
    included only where the capability resolver says a portfolio in scope
    originates — a non-originating book contributes funded actuals alone."""
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    proot = _pipeline_discovery_root()
    if not root:
        return {"dataset": "forecast", "portfolioId": cid, "toRunId": trid,
                "periods": [], "singlePeriod": True,
                "error": "no onboarding output root configured"}
    try:
        # Weight the pipeline by the governed historical stage rates (same basis
        # as the point-in-time bridge and the scale-up forecast) so every forecast
        # surface shows ONE consistent 'weighted expected pipeline'.
        resolved = _resolve_portfolio_context(portfolioContext, cid)
        scope = resolved.scope if resolved else None
        result = evolution_mod.forecast_evolution(
            root, proot or root, cid, trid, historical_model=_pipeline_history(cid),
            scope=scope,
            include_pipeline=_originates(resolved, portfolioContext))
        if resolved is not None:
            result["portfolioScope"] = _scope_block(None, resolved) or scope.to_dict()
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("forecast evolution failed: %s", exc)
        return {"dataset": "forecast", "portfolioId": cid, "toRunId": trid,
                "periods": [], "singlePeriod": True, "error": str(exc)}


@app.get("/mi/forecast/extrapolation")
def forecast_extrapolation(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                           toRunId: Optional[str] = None, to_run_id: Optional[str] = None,
                           portfolioContext: Optional[str] = None
                           ) -> Dict[str, Any]:
    """Securitisation scale-up forecast: completion run-rate + KFI-conversion
    extrapolation with downside/base/upside bands and milestone dates to funding
    thresholds, plus the existing point-in-time weighted-pipeline forecast.
    Never 500s — returns controlled insufficient-history caveats."""
    from . import forecast_extrapolation as fx_mod
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    proot = _pipeline_discovery_root()
    if not root:
        return {"portfolioId": cid, "toRunId": trid, "currentFundedBalance": 0.0,
                "completionRunRateForecast": {"available": False,
                                              "status": "insufficient_data",
                                              "caveat": "no onboarding output root configured"},
                "dataSufficiency": "insufficient_data"}
    try:
        history = _pipeline_history(cid)
        resolved = _resolve_portfolio_context(portfolioContext, cid)
        result = fx_mod.build_extrapolation(root, proot or root, cid, trid,
                                            history_model=history,
                                            scope=resolved.scope if resolved else None)
        if resolved is not None:
            result["portfolioScope"] = resolved.scope.to_dict()
        return result
    except Exception as exc:  # noqa: BLE001 - forecast must never 500
        logger.warning("forecast extrapolation failed: %s", exc)
        return {"portfolioId": cid, "toRunId": trid, "currentFundedBalance": 0.0,
                "completionRunRateForecast": {"available": False,
                                              "status": "insufficient_data",
                                              "caveat": str(exc)},
                "dataSufficiency": "insufficient_data", "error": str(exc)}


@app.get("/mi/risk-limits")
def risk_limits(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                toRunId: Optional[str] = None, to_run_id: Optional[str] = None,
                portfolioContext: Optional[str] = None
                ) -> Dict[str, Any]:
    """Governed risk-limit / concentration monitor: Schedule 8 extracted limits
    vs funded actual exposure, headroom, pass/warn/fail status, source, confidence
    and movement vs the prior run. Never 500s — returns controlled
    unavailable / needs-review states when limits or fields are missing."""
    from . import risk_limits as risk_mod
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    try:
        resolved = _resolve_portfolio_context(portfolioContext, cid)
        result = risk_mod.compute_risk_limits(
            root, cid, trid, scope=resolved.scope if resolved else None)
        if resolved is not None:
            result["portfolioScope"] = resolved.scope.to_dict()
        return result
    except Exception as exc:  # noqa: BLE001 - risk monitor must never 500
        logger.warning("risk-limits failed: %s", exc)
        return {"portfolioId": cid, "toRunId": trid, "available": False,
                "limitsStatus": "unavailable", "limitsSource": "error",
                "summary": {"testsPassed": 0, "warnings": 0, "breaches": 0,
                            "needsReview": 0, "unavailable": 0, "total": 0,
                            "closestHeadroom": None, "largestConcentration": None},
                "testsByCategory": {}, "tests": [], "observations": [],
                "error": str(exc)}


@app.get("/mi/concentration-tests")
def concentration_tests(portfolioId: Optional[str] = None,
                        client_id: Optional[str] = None,
                        toRunId: Optional[str] = None,
                        to_run_id: Optional[str] = None,
                        portfolioContext: Optional[str] = None
                        ) -> Dict[str, Any]:
    """Governed concentration-test monitor: the operator-APPROVED, versioned
    configuration evaluated deterministically against the funded book (current
    + prior period, headroom, utilization, status transitions), falling back to
    the legacy extracted limits — explicitly marked unapproved — where no
    approved configuration exists. Never 500s."""
    from . import concentration_tests_api as conc_mod
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    try:
        resolved = _resolve_portfolio_context(portfolioContext, cid)
        result = conc_mod.compute_concentration_tests(
            root, cid, trid, scope=resolved.scope if resolved else None)
        if resolved is not None:
            result["portfolioScope"] = resolved.scope.to_dict()
        return result
    except Exception as exc:  # noqa: BLE001 - the monitor must never 500
        logger.warning("concentration-tests failed: %s", exc)
        return {"portfolioId": cid, "toRunId": trid, "available": False,
                "source": "none", "approvalStatus": None, "tests": [],
                "summary": {"overallStatus": "unavailable", "activeTests": 0,
                            "breaches": 0, "warnings": 0, "passes": 0,
                            "unavailable": 0, "deteriorations": 0,
                            "closestToLimit": None, "priorAvailable": False},
                "error": str(exc)}


@app.get("/mi/concentration-tests/drillthrough")
def concentration_drillthrough(testId: str,
                               portfolioId: Optional[str] = None,
                               client_id: Optional[str] = None,
                               toRunId: Optional[str] = None,
                               to_run_id: Optional[str] = None,
                               portfolioContext: Optional[str] = None
                               ) -> Dict[str, Any]:
    """Contributing-loan population for one approved concentration test —
    the SAME mask the evaluator used, so it reconciles exactly to the
    numerator. Never 500s."""
    from . import concentration_tests_api as conc_mod
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    try:
        resolved = _resolve_portfolio_context(portfolioContext, cid)
        return conc_mod.compute_drillthrough(
            root, cid, trid, testId,
            scope=resolved.scope if resolved else None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("concentration drill-through failed: %s", exc)
        return {"available": False, "reason": str(exc), "rows": [],
                "columns": []}


@app.get("/mi/concentration-tests/drivers")
def concentration_drivers(testId: str,
                          portfolioId: Optional[str] = None,
                          client_id: Optional[str] = None,
                          toRunId: Optional[str] = None,
                          to_run_id: Optional[str] = None,
                          portfolioContext: Optional[str] = None
                          ) -> Dict[str, Any]:
    """Pipeline cases driving one approved test's Expected Forecast movement —
    the forecast engine's own probabilities and contributions, reconciling to
    the expected numerator. Never 500s."""
    from . import concentration_tests_api as conc_mod
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    try:
        resolved = _resolve_portfolio_context(portfolioContext, cid)
        return conc_mod.compute_pipeline_drivers(
            root, cid, trid, testId,
            scope=resolved.scope if resolved else None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("concentration drivers failed: %s", exc)
        return {"available": False, "reason": str(exc), "drivers": []}


@app.get("/mi/concentration-tests/history")
def concentration_history(portfolioId: Optional[str] = None,
                          client_id: Optional[str] = None,
                          toRunId: Optional[str] = None,
                          to_run_id: Optional[str] = None,
                          testId: Optional[str] = None,
                          portfolioContext: Optional[str] = None
                          ) -> Dict[str, Any]:
    """Metric history for the approved tests across real governed snapshots
    (no fabricated history). Never 500s."""
    from . import concentration_tests_api as conc_mod
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    try:
        resolved = _resolve_portfolio_context(portfolioContext, cid)
        return conc_mod.compute_history(
            root, cid, trid, test_id=testId,
            scope=resolved.scope if resolved else None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("concentration history failed: %s", exc)
        return {"available": False, "reason": str(exc), "series": []}


@app.get("/mi/evolution/compare")
def evolution_compare(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                      toRunId: Optional[str] = None, to_run_id: Optional[str] = None,
                      dataset: str = "funded", metric: Optional[str] = None,
                      aggregation: str = "sum", periodA: str = "prior",
                      periodB: str = "latest",
                      portfolioContext: Optional[str] = None) -> Dict[str, Any]:
    """Governed cross-period comparison (period A vs period B) over the evolution
    series: value A/B, absolute + % delta, source periods, reconciliation, and a
    controlled insufficient-data response. Never 500s."""
    from . import temporal_compare as compare_mod
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    proot = _pipeline_discovery_root()
    try:
        resolved = _resolve_portfolio_context(portfolioContext, cid)
        result = compare_mod.run_temporal_compare(
            root, proot or root, cid, trid, dataset=dataset, metric=metric,
            aggregation=aggregation, period_a=periodA, period_b=periodB,
            scope=resolved.scope if resolved else None)
        if resolved is not None:
            result["portfolioScope"] = resolved.scope.to_dict()
        return result
    except Exception as exc:  # noqa: BLE001 - comparison must never 500
        logger.warning("evolution compare failed: %s", exc)
        return {"available": False, "status": "insufficient_data", "dataset": dataset,
                "portfolioId": cid, "toRunId": trid, "reason": str(exc)}


@app.get("/mi/workspace/view")
def workspace_view(portfolioId: Optional[str] = None,
                   client_id: Optional[str] = None,
                   runId: Optional[str] = None,
                   run_id: Optional[str] = None,
                   view: Optional[str] = None,
                   portfolioContext: Optional[str] = None) -> Dict[str, Any]:
    """Unified workspace view-model composing the funded snapshot + pipeline
    snapshot + forecast bridge for one portfolio/run. ``view`` (optional) marks the
    active/foregrounded view; all three blocks are returned so the UI can switch
    tabs without refetching. Composes existing endpoints — no duplicated logic.
    """
    run_id = runId or run_id
    if portfolioId and "/" in portfolioId:
        client_id, run_id = portfolioId.split("/", 1)
    client_id = client_id or "client_001"
    active = (view or workspace_mod.DEFAULT_VIEW).strip().lower()
    if active not in workspace_mod.VIEWS:
        active = workspace_mod.DEFAULT_VIEW
    pid = f"{client_id}/{run_id}" if run_id else client_id

    funded = snapshot(portfolioId=pid, portfolioContext=portfolioContext)
    pipeline = pipeline_snapshot(portfolioId=pid, portfolioContext=portfolioContext)
    forecast = forecast_snapshot(portfolioId=pid, portfolioContext=portfolioContext)

    resolved = _resolve_portfolio_context(portfolioContext, client_id)
    pipe_ok = bool(pipeline.get("ok"))
    return {
        "ok": True,
        "portfolioId": pid,
        "client_id": client_id,
        "runId": run_id,
        "activeView": active,
        "views": list(workspace_mod.VIEWS),
        # One governed context for the whole workspace view-model.
        "portfolioScope": resolved.scope.to_dict() if resolved else None,
        "capabilities": (portfolio_ctx_mod.capabilities_to_dict(resolved.capabilities)
                         if resolved else None),
        "funded": funded,
        "pipeline": pipeline,
        "forecast": forecast,
        "lineage": {
            "funded": workspace_mod.lineage_for(
                "funded", funded_reporting_date=(funded.get("portfolio") or {}).get("reporting_date")),
            "pipeline": workspace_mod.lineage_for(
                "pipeline", pipeline_as_of_date=pipeline.get("pipelineAsOfDate"),
                pipeline_source_folder_date=pipeline.get("pipelineSourceFolderDate"),
                current_pipeline_snapshot_date=pipeline.get("currentPipelineSnapshotDate"),
                current_pipeline_source_file=pipeline.get("currentPipelineSourceFile"),
                completion_probability_basis=pipeline.get("completionProbabilityBasis"),
                source_file=pipeline.get("sourceFile"),
                historical_model_evidence=pipeline.get("historicalModelEvidence"),
            ) if pipe_ok else workspace_mod.lineage_for("pipeline"),
            "forecast": forecast.get("lineage", workspace_mod.lineage_for("forecast")),
        },
    }


@app.post("/mi/query")
def query(req: QueryRequest, request: Request) -> Any:
    """React MI Agent channel — a THIN adapter over the governed MI capability.

    The adapter does exactly three things: turn the authenticated principal into
    an :class:`~trakt_core.context.ExecutionContext`, translate the HTTP body
    into an ``MiQueryRequest``, and present the ``GovernedResult``. No parsing,
    routing, dataset resolution, calculation, validation, policy or provenance
    logic lives here — all of it is owned by ``mi_service`` and is therefore
    shared with Copilot and with any future adapter.

    The response body is the existing React envelope plus an additive
    ``governance`` block; no pre-existing field changed.
    """
    context = _execution_context(request, channel=CHANNEL_REACT)
    result = mi_service.execute_governed_mi_query(
        mi_service.MiQueryRequest(
            question=req.question,
            portfolio_id=req.portfolioId or (req.portfolio.id if req.portfolio else None),
            as_of_date=req.asOfDate,
            filters=req.filters,
            dataset_context=req.datasetContext,
            context=req.context,
            source_portfolio_lens=req.sourcePortfolioLens,
        ),
        context,
    )
    payload = presenters.to_react_payload(result)
    # Governance refusals (unauthorised portfolio, unapproved data source) carry
    # their mapped HTTP status. Analytical outcomes stay 200 with ok=false, which
    # is the contract the React client already implements.
    status = result.http_status
    if status == 200:
        return payload
    return JSONResponse(status_code=status, content=payload)


# Microsoft 365 Copilot actions (askTraktMi / getArtifact) — a thin,
# bearer-token-authenticated action layer over the handlers above. Imported last
# (the module calls back into this one lazily).
from .copilot_actions import router as _copilot_router  # noqa: E402

app.include_router(_copilot_router)
