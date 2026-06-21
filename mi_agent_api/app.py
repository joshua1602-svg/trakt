"""FastAPI app exposing the existing MI Agent to the React UI.

Endpoints:
  GET  /health         - liveness + data-source status
  GET  /mi/catalogue   - real semantic layer (states/dimensions/measures/...)
  POST /mi/query       - run one MI question through run_mi_agent_query

Run:
  uvicorn mi_agent_api.app:app --reload --port 8000
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mi_agent.mi_agent_workflow import run_mi_agent_query

from .adapters import adapt_workflow_result
from .catalogue import build_catalogue
from .data_source import (
    data_source_info,
    data_source_label,
    get_dataframe,
    semantics_path,
)

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
        "dataSourceInfo": info,
        "dataAvailable": csv != "unavailable",
        "semantics": semantics_path().name,
    }


@app.get("/mi/catalogue")
def catalogue() -> Dict[str, Any]:
    return build_catalogue()


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
