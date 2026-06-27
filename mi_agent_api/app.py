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
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_validator import load_mi_semantics

from .adapters import adapt_workflow_result
from .catalogue import build_catalogue
from .data_source import (
    data_source_info,
    data_source_label,
    get_dataframe,
    semantics_path,
)
from . import snapshots as snapshots_mod
from . import pipeline_contract as pipeline_mod
from . import pipeline_history
from . import forecast_bridge as forecast_mod
from . import workspace as workspace_mod
from . import evolution as evolution_mod

logger = logging.getLogger("mi_agent_api")

app = FastAPI(title="Trakt MI Agent API", version="1.0.0")

# CORS for the React dev server (and configurable origins in deployment).
_origins = os.environ.get(
    "MI_AGENT_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:4173",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()] or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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


def _onboarding_output_root() -> Optional[str]:
    """The local onboarding output root used for run/portfolio discovery."""
    root = os.environ.get("MI_AGENT_ONBOARDING_OUTPUT_ROOT")
    if root:
        return root
    # Fall back to inferring a root from an explicit central tape path so a
    # single configured run is still discoverable (`.../output` above /central).
    tape = os.environ.get("MI_AGENT_CENTRAL_TAPE")
    if tape:
        from pathlib import Path
        p = Path(tape)
        # .../<client>/<run>/output/central/18_central_lender_tape.csv -> climb to a
        # root that still contains the client/run components for inference.
        parents = list(p.parents)
        return str(parents[3]) if len(parents) > 3 else str(p.parent.parent)
    return None


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
        "dataSourceInfo": info,
        "dataAvailable": csv != "unavailable",
        "semantics": semantics_path().name,
    }


@app.get("/mi/catalogue")
def catalogue() -> Dict[str, Any]:
    return build_catalogue()


@app.get("/mi/snapshots")
def snapshots() -> Dict[str, Any]:
    """Data-driven discovery of available funded portfolios and reporting runs.

    The portfolio / reporting-date dropdowns are built from THIS — only real
    onboarding output appears (no hardcoded prototype options).
    """
    root = _onboarding_output_root()
    if not root:
        return {"portfolios": [], "source": "unavailable"}
    try:
        result = snapshots_mod.discover_snapshots(root)
    except Exception as exc:  # noqa: BLE001 - discovery must never 500
        logger.warning("snapshot discovery failed: %s", exc)
        return {"portfolios": [], "source": "error", "error": str(exc)}
    result["source"] = root
    return result


def _resolve_run_dataframe(client_id: str, run_id: str, root: Optional[str]):
    """``(df, prep_report)`` for a specific run, preferring on-disk discovery and
    falling back to the active env-configured dataframe for the active run."""
    if root:
        tape = snapshots_mod.resolve_tape_path(root, client_id, run_id)
        if tape is not None:
            return snapshots_mod.load_prepared_run(tape)
    # Fall back to the active data source if it matches the requested run.
    info = data_source_info()
    if info.get("client_id") == client_id and info.get("run_id") == run_id:
        return get_dataframe(), info
    return None, None


@app.get("/mi/snapshot")
def snapshot(portfolioId: Optional[str] = None,
             client_id: Optional[str] = None,
             run_id: Optional[str] = None) -> Dict[str, Any]:
    """Deterministic funded-book snapshot (KPIs + month-on-month change) for a run.

    ``portfolioId`` is ``"<client_id>/<run_id>"`` (matching the /mi/query contract);
    ``client_id`` + ``run_id`` may be passed separately instead.
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

    # Resolve the prior available run for month-on-month change.
    prior_df = prior_run_id = prior_reporting_date = None
    reporting_date = snapshots_mod.infer_reporting_date(run_id, df)
    if root:
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
        prior_df=prior_df, prior_run_id=prior_run_id,
        prior_reporting_date=prior_reporting_date,
    )
    for d in result.get("diagnostics", []):
        logger.info("snapshot diagnostic [%s/%s]: %s", client_id, run_id, d)
    return result


def _pipeline_root() -> Optional[str]:
    """Root to discover governed pipeline sources (18a tape / M2L KFI extracts)."""
    for key in ("MI_AGENT_PIPELINE_ROOT", "MI_AGENT_ONBOARDING_OUTPUT_ROOT"):
        root = os.environ.get(key)
        if root:
            return root
    return _onboarding_output_root()


@app.get("/mi/pipeline/snapshots")
def pipeline_snapshots(portfolioId: Optional[str] = None) -> Dict[str, Any]:
    """Data-driven discovery of governed pipeline sources and reporting dates."""
    root = os.environ.get("MI_AGENT_PIPELINE_ROOT") or _pipeline_root()
    if not root:
        return {"sources": [], "source": "unavailable"}
    client_id = portfolioId.split("/", 1)[0] if portfolioId else None
    try:
        sources = pipeline_mod.discover_pipeline_sources(root, client_id=client_id)
    except Exception as exc:  # noqa: BLE001 - discovery must never 500
        logger.warning("pipeline discovery failed: %s", exc)
        return {"sources": [], "source": "error", "error": str(exc)}
    return {"sources": sources, "source": root}


def _resolve_pipeline_source(client_id: str, run_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The governed pipeline scope for a client/run (explicit env or discovery).

    Returns a scope dict with the separated date concepts (folder / extract /
    as-of), never a single ambiguous reporting date.
    """
    explicit = os.environ.get("MI_AGENT_PIPELINE_SOURCE")
    if explicit:
        from pathlib import Path as _Path
        p = _Path(explicit)
        if p.exists():
            folder_date = pipeline_mod._folder_date(p.parent)
            extract_date = pipeline_mod._extract_date(p)
            return {"client_id": client_id, "source_file": str(p),
                    "run_id": run_id or pipeline_mod._run_id_for(folder_date, extract_date, p),
                    "pipeline_source_folder": str(p.parent),
                    "pipeline_source_folder_date": folder_date,
                    "pipeline_extract_date": extract_date,
                    "pipeline_as_of_date": extract_date or folder_date}
    root = os.environ.get("MI_AGENT_PIPELINE_ROOT") or _pipeline_root()
    if root:
        return pipeline_mod.resolve_pipeline_source(root, client_id, run_id)
    return None


def _pipeline_history(client_id: str) -> Optional[Dict[str, Any]]:
    """The historical completion-rate model from a client's weekly pipeline files
    (None when only a single explicit source / no discovery root is configured)."""
    if os.environ.get("MI_AGENT_PIPELINE_SOURCE"):
        return None
    root = os.environ.get("MI_AGENT_PIPELINE_ROOT") or _pipeline_root()
    if not root:
        return None
    try:
        return pipeline_mod.build_pipeline_history(root, client_id)
    except Exception as exc:  # noqa: BLE001 - history is additive; never 500
        logger.warning("pipeline history build failed for %s: %s", client_id, exc)
        return None


@app.get("/mi/pipeline/snapshot")
def pipeline_snapshot(portfolioId: Optional[str] = None,
                      client_id: Optional[str] = None,
                      runId: Optional[str] = None,
                      run_id: Optional[str] = None) -> Dict[str, Any]:
    """Deterministic pipeline single-source snapshot for the latest weekly cut.

    ``portfolioId`` is ``"<client_id>/<run_id>"`` (matching the funded contract);
    ``client_id`` + ``runId``/``run_id`` may be passed separately instead. The
    pipeline as-of/extract dates are exposed distinctly from the funded run date.
    """
    run_id = runId or run_id
    if portfolioId and "/" in portfolioId:
        client_id, run_id = portfolioId.split("/", 1)
    client_id = client_id or "client_001"

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
    return pipeline_mod.compute_pipeline_snapshot(
        df, report, semantics, client_id=source.get("client_id", client_id),
        run_id=run_id or source.get("run_id", ""), source=source, prior_week=prior_week)


@app.get("/mi/forecast/snapshot")
def forecast_snapshot(portfolioId: Optional[str] = None,
                      client_id: Optional[str] = None,
                      runId: Optional[str] = None,
                      run_id: Optional[str] = None) -> Dict[str, Any]:
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

    # Pipeline side (Phase 1 prep + contract): the LATEST weekly extract for the
    # run's source scope. Its as-of/extract dates stay distinct from the funded date.
    pipeline_df = pipeline_report = pipeline_snap = None
    source = _resolve_pipeline_source(client_id, run_id)
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
    return envelope


def _evo_ids(portfolioId, client_id, toRunId, to_run_id):
    """Resolve (client_id, to_run_id) from a portfolioId or explicit params."""
    if portfolioId and "/" in portfolioId:
        client_id, to_run_id = portfolioId.split("/", 1)
    elif portfolioId:
        client_id = portfolioId
    return (client_id or "client_001"), (toRunId or to_run_id)


@app.get("/mi/evolution/funded")
def funded_evolution(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                     toRunId: Optional[str] = None, to_run_id: Optional[str] = None
                     ) -> Dict[str, Any]:
    """Funded time series across monthly runs up to ``toRunId`` (per-period
    reconciliation + lineage). Never 500s — returns an empty series on no data."""
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    if not root:
        return {"dataset": "funded", "portfolioId": cid, "toRunId": trid,
                "periods": [], "breakdowns": {}, "singlePeriod": True,
                "error": "no onboarding output root configured"}
    try:
        return evolution_mod.funded_evolution(root, cid, trid)
    except Exception as exc:  # noqa: BLE001 - evolution must never 500
        logger.warning("funded evolution failed: %s", exc)
        return {"dataset": "funded", "portfolioId": cid, "toRunId": trid,
                "periods": [], "breakdowns": {}, "singlePeriod": True, "error": str(exc)}


@app.get("/mi/evolution/pipeline")
def pipeline_evolution(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                       toRunId: Optional[str] = None, to_run_id: Optional[str] = None
                       ) -> Dict[str, Any]:
    """Pipeline time series across governed weekly extracts (amount / cases / by
    stage over time), with per-period reconciliation + lineage."""
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = os.environ.get("MI_AGENT_PIPELINE_ROOT") or _pipeline_root()
    if not root:
        return {"dataset": "pipeline", "portfolioId": cid, "toRunId": trid,
                "periods": [], "byStage": [], "singlePeriod": True,
                "error": "no pipeline root configured"}
    try:
        return evolution_mod.pipeline_evolution(root, cid, trid)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pipeline evolution failed: %s", exc)
        return {"dataset": "pipeline", "portfolioId": cid, "toRunId": trid,
                "periods": [], "byStage": [], "singlePeriod": True, "error": str(exc)}


@app.get("/mi/evolution/funnel")
def funnel_evolution(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                     toRunId: Optional[str] = None, to_run_id: Optional[str] = None
                     ) -> Dict[str, Any]:
    """Weekly origination funnel trends (KFI / Application / Offer / Completion
    value + count, 5-week average, latest week, delta vs prior week). Never 500s."""
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = os.environ.get("MI_AGENT_PIPELINE_ROOT") or _pipeline_root()
    if not root:
        return {"dataset": "pipeline_funnel", "portfolioId": cid, "toRunId": trid,
                "stages": [], "weeks": [], "series": {}, "summary": {},
                "singlePeriod": True, "error": "no pipeline root configured"}
    try:
        return evolution_mod.pipeline_funnel_evolution(root, cid, trid)
    except Exception as exc:  # noqa: BLE001
        logger.warning("funnel evolution failed: %s", exc)
        return {"dataset": "pipeline_funnel", "portfolioId": cid, "toRunId": trid,
                "stages": [], "weeks": [], "series": {}, "summary": {},
                "singlePeriod": True, "error": str(exc)}


@app.get("/mi/evolution/forecast")
def forecast_evolution(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                       toRunId: Optional[str] = None, to_run_id: Optional[str] = None
                       ) -> Dict[str, Any]:
    """Forecast bridge over time (funded balance + weighted pipeline per run)."""
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    proot = os.environ.get("MI_AGENT_PIPELINE_ROOT") or _pipeline_root()
    if not root:
        return {"dataset": "forecast", "portfolioId": cid, "toRunId": trid,
                "periods": [], "singlePeriod": True,
                "error": "no onboarding output root configured"}
    try:
        return evolution_mod.forecast_evolution(root, proot or root, cid, trid)
    except Exception as exc:  # noqa: BLE001
        logger.warning("forecast evolution failed: %s", exc)
        return {"dataset": "forecast", "portfolioId": cid, "toRunId": trid,
                "periods": [], "singlePeriod": True, "error": str(exc)}


@app.get("/mi/forecast/extrapolation")
def forecast_extrapolation(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                           toRunId: Optional[str] = None, to_run_id: Optional[str] = None
                           ) -> Dict[str, Any]:
    """Securitisation scale-up forecast: completion run-rate + KFI-conversion
    extrapolation with downside/base/upside bands and milestone dates to funding
    thresholds, plus the existing point-in-time weighted-pipeline forecast.
    Never 500s — returns controlled insufficient-history caveats."""
    from . import forecast_extrapolation as fx_mod
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    proot = os.environ.get("MI_AGENT_PIPELINE_ROOT") or _pipeline_root()
    if not root:
        return {"portfolioId": cid, "toRunId": trid, "currentFundedBalance": 0.0,
                "completionRunRateForecast": {"available": False,
                                              "status": "insufficient_data",
                                              "caveat": "no onboarding output root configured"},
                "dataSufficiency": "insufficient_data"}
    try:
        history = _pipeline_history(cid)
        return fx_mod.build_extrapolation(root, proot or root, cid, trid,
                                          history_model=history)
    except Exception as exc:  # noqa: BLE001 - forecast must never 500
        logger.warning("forecast extrapolation failed: %s", exc)
        return {"portfolioId": cid, "toRunId": trid, "currentFundedBalance": 0.0,
                "completionRunRateForecast": {"available": False,
                                              "status": "insufficient_data",
                                              "caveat": str(exc)},
                "dataSufficiency": "insufficient_data", "error": str(exc)}


@app.get("/mi/risk-limits")
def risk_limits(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                toRunId: Optional[str] = None, to_run_id: Optional[str] = None
                ) -> Dict[str, Any]:
    """Governed risk-limit / concentration monitor: Schedule 8 extracted limits
    vs funded actual exposure, headroom, pass/warn/fail status, source, confidence
    and movement vs the prior run. Never 500s — returns controlled
    unavailable / needs-review states when limits or fields are missing."""
    from . import risk_limits as risk_mod
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    try:
        return risk_mod.compute_risk_limits(root, cid, trid)
    except Exception as exc:  # noqa: BLE001 - risk monitor must never 500
        logger.warning("risk-limits failed: %s", exc)
        return {"portfolioId": cid, "toRunId": trid, "available": False,
                "limitsStatus": "unavailable", "limitsSource": "error",
                "summary": {"testsPassed": 0, "warnings": 0, "breaches": 0,
                            "needsReview": 0, "unavailable": 0, "total": 0,
                            "closestHeadroom": None, "largestConcentration": None},
                "testsByCategory": {}, "tests": [], "observations": [],
                "error": str(exc)}


@app.get("/mi/evolution/compare")
def evolution_compare(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                      toRunId: Optional[str] = None, to_run_id: Optional[str] = None,
                      dataset: str = "funded", metric: Optional[str] = None,
                      aggregation: str = "sum", periodA: str = "prior",
                      periodB: str = "latest") -> Dict[str, Any]:
    """Governed cross-period comparison (period A vs period B) over the evolution
    series: value A/B, absolute + % delta, source periods, reconciliation, and a
    controlled insufficient-data response. Never 500s."""
    from . import temporal_compare as compare_mod
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    proot = os.environ.get("MI_AGENT_PIPELINE_ROOT") or _pipeline_root()
    try:
        return compare_mod.run_temporal_compare(
            root, proot or root, cid, trid, dataset=dataset, metric=metric,
            aggregation=aggregation, period_a=periodA, period_b=periodB)
    except Exception as exc:  # noqa: BLE001 - comparison must never 500
        logger.warning("evolution compare failed: %s", exc)
        return {"available": False, "status": "insufficient_data", "dataset": dataset,
                "portfolioId": cid, "toRunId": trid, "reason": str(exc)}


@app.get("/mi/workspace/view")
def workspace_view(portfolioId: Optional[str] = None,
                   client_id: Optional[str] = None,
                   runId: Optional[str] = None,
                   run_id: Optional[str] = None,
                   view: Optional[str] = None) -> Dict[str, Any]:
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

    funded = snapshot(portfolioId=pid)
    pipeline = pipeline_snapshot(portfolioId=pid)
    forecast = forecast_snapshot(portfolioId=pid)

    pipe_ok = bool(pipeline.get("ok"))
    return {
        "ok": True,
        "portfolioId": pid,
        "client_id": client_id,
        "runId": run_id,
        "activeView": active,
        "views": list(workspace_mod.VIEWS),
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


def _query_dataset_context(req: QueryRequest) -> str:
    ctx = req.datasetContext
    if not ctx and isinstance(req.context, dict):
        ctx = req.context.get("activeView") or req.context.get("datasetContext")
    return workspace_mod.resolve_active_view(req.question, ctx)


def _resolve_query_frame(view: str, portfolio_id: Optional[str]):
    """``(df, error)`` for a tab-aware query. Funded keeps the existing active
    dataset (unchanged); pipeline / forecast resolve the governed pipeline (and,
    for forecast, a derived funded + weighted-pipeline frame)."""
    client_id, run_id = "client_001", None
    if portfolio_id and "/" in portfolio_id:
        client_id, run_id = portfolio_id.split("/", 1)
    elif portfolio_id:
        client_id = portfolio_id

    if view == "funded":
        return get_dataframe(), None  # existing behaviour, unchanged

    pipeline_df = None
    source = _resolve_pipeline_source(client_id, run_id)
    if source is not None:
        try:
            pipeline_df, _ = pipeline_mod.load_prepared_pipeline(
                source, historical_model=_pipeline_history(source.get("client_id", client_id)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("pipeline frame load failed for query: %s", exc)

    if view == "pipeline":
        if pipeline_df is None or not len(pipeline_df):
            return None, "No governed pipeline data is available for the pipeline view."
        return pipeline_df, None

    # forecast — derived funded + weighted pipeline frame.
    funded_df = None
    if run_id:
        funded_df, _ = _resolve_run_dataframe(client_id, run_id, _onboarding_output_root())
    if funded_df is None:
        try:
            funded_df = get_dataframe()
        except FileNotFoundError:
            funded_df = None
    frame = workspace_mod.build_forecast_view_frame(funded_df, pipeline_df)
    if not len(frame):
        return None, "No forecast data is available for the forecast view."
    return frame, None


@app.post("/mi/query")
def query(req: QueryRequest) -> Dict[str, Any]:
    portfolio_id = req.portfolioId or (req.portfolio.id if req.portfolio else None)
    view = _query_dataset_context(req)

    def _error(msg: str) -> Dict[str, Any]:
        return {
            "ok": False, "error": msg, "question": req.question,
            "answer": msg, "interpreted": "", "spec": {},
            "validation": {"ok": False, "errors": [msg], "warnings": [], "resolved_fields": {}},
            "artifacts": [], "warnings": [], "assumptions": [],
            "metadata": {"engine": "mi_agent", "source": "python", "mock": False,
                         "datasetContext": view},
        }

    try:
        df, frame_error = _resolve_query_frame(view, portfolio_id)
    except FileNotFoundError as exc:
        return _error(str(exc))
    if frame_error:
        return _error(frame_error)

    workflow = run_mi_agent_query(
        req.question, df, str(semantics_path()), parser_mode="deterministic",
        extra_filters=req.filters or None)
    result = adapt_workflow_result(workflow, portfolio_id=portfolio_id, as_of=req.asOfDate)
    # Surface which dataset/view answered (funded | pipeline | forecast).
    meta = result.setdefault("metadata", {}) if isinstance(result, dict) else {}
    if isinstance(meta, dict):
        meta["datasetContext"] = view
    return result
