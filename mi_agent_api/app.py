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
    # Optional prior conversation turns (accepted, not yet used for context).
    context: Optional[List[Dict[str, Any]]] = None


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
        "endpoints": ["/health", "/mi/catalogue", "/mi/snapshots", "/mi/snapshot", "/mi/query"],
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


@app.post("/mi/query")
def query(req: QueryRequest) -> Dict[str, Any]:
    portfolio_id = req.portfolioId or (req.portfolio.id if req.portfolio else None)

    try:
        df = get_dataframe()
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "question": req.question,
            "answer": "No portfolio data is available to query.",
            "interpreted": "",
            "spec": {},
            "validation": {"ok": False, "errors": [str(exc)], "warnings": [], "resolved_fields": {}},
            "artifacts": [],
            "warnings": [],
            "assumptions": [],
            "metadata": {"engine": "mi_agent", "source": "python", "mock": False},
        }

    workflow = run_mi_agent_query(
        req.question,
        df,
        str(semantics_path()),
        parser_mode="deterministic",
    )
    return adapt_workflow_result(
        workflow,
        portfolio_id=portfolio_id,
        as_of=req.asOfDate,
    )
