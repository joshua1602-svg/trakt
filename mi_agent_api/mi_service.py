"""mi_agent_api/mi_service.py — THE shared governed MI application service.

One analytical implementation, two presentation channels:

    React MI Agent  ──┐
                      ├──►  execute_governed_mi_query()  ──► governed envelope
    M365 Copilot    ──┘

``POST /mi/query`` (React) and ``POST /v1/copilot/mi/query`` (Copilot) are both
THIN adapters over :func:`execute_governed_mi_query`. Neither owns any analytical
behaviour. Everything below the channel boundary lives here (or in the modules
this service composes):

  * question parsing + intent classification (``chat_routing`` / ``mi_agent``);
  * portfolio / dataset / active-dataframe resolution;
  * metric, dimension and filter resolution;
  * deterministic calculation, top-N, point-in-time analysis;
  * temporal comparison, evolution, cohort, forecast and risk-limit routing;
  * validation, reconciliation, provenance, warnings, governed error handling;
  * analytical artifact creation.

Channels may differ ONLY in presentation: React renders charts / drill-through /
workspace state; Copilot reshapes the same envelope into its OpenAPI schema,
truncates oversized row sets and normalises text for its renderer.

Data-resolution note
--------------------
The concrete data resolvers (``_resolve_query_frame``, ``_resolve_run_dataframe``,
``_pipeline_history``, ``_onboarding_output_root``, …) still live in
``mi_agent_api.app`` alongside the dashboard endpoints that share them. This
module imports them lazily (:func:`_resolvers`) so there is no import cycle and
no duplication — ``app`` remains the data-resolution layer, this module is the
analytical application layer, and neither channel reaches past it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("mi_agent_api.mi_service")

#: Client used when neither the caller nor the deployment names one. Matches the
#: historical React default so the existing contract is unchanged.
DEFAULT_CLIENT_ID = "client_001"


# --------------------------------------------------------------------------- #
# Channel-neutral request contract
# --------------------------------------------------------------------------- #
@dataclass
class MiQueryRequest:
    """One governed MI question, expressed without any channel/HTTP concepts.

    ``portfolio_id`` is the existing ``"{client_id}/{run_id}"`` selector (or a
    bare client id). ``client_id`` is the authenticated tenant/client context of
    the calling deployment and is used only when the caller supplied no explicit
    portfolio — it never overrides an explicit selection.
    """

    question: str
    portfolio_id: Optional[str] = None
    as_of_date: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    dataset_context: Optional[str] = None
    context: Optional[Any] = None
    source_portfolio_lens: Optional[str] = None
    #: Authenticated tenant/client context (deployment-per-client).
    client_id: Optional[str] = None
    #: Channel-neutral execution options (reserved; no analytical effect).
    options: Dict[str, Any] = field(default_factory=dict)

    def effective_portfolio_id(self) -> Optional[str]:
        """The portfolio context the analysis runs against: the caller's explicit
        selection, else the authenticated client, else nothing (active dataset)."""
        return self.portfolio_id or self.client_id or None


def split_portfolio(portfolio_id: Optional[str],
                    default_client: str = DEFAULT_CLIENT_ID) -> tuple[str, Optional[str]]:
    """``("{client}/{run}" | "{client}" | None) -> (client_id, run_id | None)``."""
    if not portfolio_id:
        return default_client, None
    if "/" in portfolio_id:
        client_id, run_id = portfolio_id.split("/", 1)
        return (client_id or default_client), (run_id or None)
    return portfolio_id, None


def _resolvers():
    """The data-resolution layer (see the module docstring). Imported lazily to
    keep ``app`` (HTTP) → ``mi_service`` (application) the only import direction."""
    from . import app as _app
    return _app


# --------------------------------------------------------------------------- #
# Governed response envelope
# --------------------------------------------------------------------------- #
def _governed_context(envelope: Dict[str, Any], *, req: MiQueryRequest,
                      client_id: str, run_id: Optional[str], view: str,
                      run_required: bool) -> Dict[str, Any]:
    """Stamp the channel-neutral governance block onto the envelope metadata.

    Additive only — every pre-existing React key is left exactly as the adapter
    produced it, so the React contract is preserved byte-for-byte.
    """
    from .data_source import data_source_kind, data_source_label

    meta = envelope.setdefault("metadata", {})
    if not isinstance(meta, dict):
        return envelope
    meta["datasetContext"] = view
    meta["selectedClient"] = client_id
    meta["selectedPortfolio"] = req.effective_portfolio_id()
    # A run is reported ONLY where the analysis genuinely used one; a
    # point-in-time question against the active governed dataset has none.
    meta["selectedRun"] = run_id if run_required else None
    meta["runRequired"] = bool(run_required)
    try:
        meta["dataSourceKind"] = data_source_kind()
        meta["dataSourceLabel"] = data_source_label()
    except Exception as exc:  # noqa: BLE001 - labelling must never fail a query
        logger.warning("data-source labelling failed: %s", exc)
        meta.setdefault("dataSourceKind", "unknown")
        meta.setdefault("dataSourceLabel", "unknown")
    envelope.setdefault("assumptions", [])
    envelope.setdefault("warnings", [])
    envelope.setdefault("diagnostics", [])
    envelope.setdefault("sourceNotes", [])
    return envelope


def _error_envelope(msg: str, *, req: MiQueryRequest, view: str) -> Dict[str, Any]:
    return {
        "ok": False, "error": msg, "question": req.question,
        "answer": msg, "interpreted": "", "spec": {},
        "validation": {"ok": False, "errors": [msg], "warnings": [],
                       "resolved_fields": {}},
        "artifacts": [], "warnings": [], "assumptions": [], "diagnostics": [],
        "sourceNotes": [],
        "metadata": {"engine": "mi_agent", "source": "python", "mock": False,
                     "datasetContext": view},
    }


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #
def execute_governed_mi_query(req: MiQueryRequest) -> Dict[str, Any]:
    """Answer one governed MI question and return the canonical envelope.

    The envelope carries ``ok``/``error``, the original and interpreted question,
    the parsed query spec, the selected client / portfolio / run / dataset, the
    reporting date, the answer, analytical artifacts (supporting values),
    validation, reconciliation, provenance (``sourceNotes``), warnings,
    diagnostics, assumptions, and the governed data-source kind + label.

    Never raises for an analytical fault — a controlled ``ok=false`` envelope is
    returned instead, so a channel can never turn a governed failure into a
    plausible narrative.
    """
    app = _resolvers()
    portfolio_id = req.effective_portfolio_id()
    view = app.workspace_mod.resolve_active_view(
        req.question,
        req.dataset_context
        or (req.context.get("activeView") or req.context.get("datasetContext")
            if isinstance(req.context, dict) else None))
    client_id, run_id = split_portfolio(portfolio_id)

    # ---- governed-intent routing (compare / evolution / forecast / risk /
    # geo / cohort / bridge / scenario). Point-in-time questions return None and
    # fall through to the deterministic executor below. -------------------- #
    routed = None
    try:
        # Request-scoped display currency BEFORE routing so routed answers format
        # in the book's currency too (tape -> config -> GBP; cached per client).
        app._apply_request_currency(client_id, portfolio_id)

        def _routed_frame(cli: str, rid: Optional[str]):
            """The funded frame for a routed intent, resolved by exactly the same
            governed resolver the point-in-time path uses. With no run selected
            this is the ACTIVE governed dataset — a point-in-time question (e.g.
            regional concentration) must never require a run id."""
            pid = f"{cli}/{rid}" if rid else (cli or None)
            frame, err = app._resolve_query_frame("funded", pid)
            return None if err else frame

        routed = app.chat_routing_mod.try_route(
            req.question, portfolio_id=portfolio_id, view=view,
            output_root=app._onboarding_output_root(),
            pipeline_root=app._pipeline_discovery_root(),
            semantics=app.load_mi_semantics(app.semantics_path()),
            history_model=app._pipeline_history(client_id), as_of=req.as_of_date,
            source_lens=req.source_portfolio_lens or None,
            frame_resolver=_routed_frame,
            extra_filters=req.filters or None)
    except Exception as exc:  # noqa: BLE001 - routing must never break the chat
        logger.warning("chat routing failed; using point-in-time path: %s", exc)
        routed = None
    if routed is not None:
        route = (routed.get("metadata") or {}).get("route") if isinstance(routed, dict) else None
        return _governed_context(routed, req=req, client_id=client_id, run_id=run_id,
                                 view=view, run_required=_route_requires_run(route))

    # ---- point-in-time: active governed dataset (or the selected run) ----- #
    try:
        df, frame_error = app._resolve_query_frame(view, portfolio_id)
    except FileNotFoundError as exc:
        return _error_envelope(str(exc), req=req, view=view)
    except Exception as exc:  # noqa: BLE001 - data load/prep must not raw-500
        logger.exception("MI query frame resolution failed for portfolio=%r view=%r",
                         portfolio_id, view)
        return _error_envelope(
            f"Could not load the data for this query: {type(exc).__name__}: {exc}",
            req=req, view=view)
    if frame_error:
        return _error_envelope(frame_error, req=req, view=view)

    app.currency_mod.resolve_and_set(df)

    llm_cfg = app._mi_llm_config()
    llm_enabled, llm_model = llm_cfg.enabled, llm_cfg.model
    try:
        workflow = app.run_mi_agent_query(
            req.question, df, str(app.semantics_path()),
            parser_mode="llm" if llm_enabled else "deterministic",
            llm_enabled=llm_enabled, model=llm_model,
            extra_filters=req.filters or None,
            source_portfolio_lens=req.source_portfolio_lens or None)
        result = app.adapt_workflow_result(
            workflow, portfolio_id=portfolio_id, as_of=req.as_of_date)
    except Exception as exc:  # noqa: BLE001 - surface, don't 500
        logger.exception("MI query failed for question=%r portfolio=%r",
                         req.question, portfolio_id)
        return _error_envelope(
            f"The MI Agent could not complete this query: {type(exc).__name__}: {exc}",
            req=req, view=view)

    meta = result.setdefault("metadata", {}) if isinstance(result, dict) else {}
    if isinstance(meta, dict):
        meta["llm"] = {"enabled": llm_cfg.enabled, "available": llm_cfg.available,
                       "model": llm_cfg.model if llm_cfg.available else None,
                       "status": llm_cfg.status}
        if workflow.get("portfolio_lens"):
            meta["portfolioLens"] = workflow["portfolio_lens"]
    # An LLM that was requested but is unusable is a configuration fault the
    # operator must see, not a silent downgrade.
    if llm_cfg.enabled and not llm_cfg.available and isinstance(result, dict):
        result.setdefault("warnings", []).extend(llm_cfg.warnings)
    # A point-in-time answer is run-scoped only when a run was explicitly selected.
    return _governed_context(result, req=req, client_id=client_id, run_id=run_id,
                             view=view, run_required=bool(run_id))


#: Routes whose analytical intent GENUINELY needs a specific run / history:
#: dated comparison, evolution, cohort progression, forecast, run-scoped risk.
#: Everything else (notably geographic exposure) is point-in-time and answers
#: from the active governed dataset.
_RUN_SCOPED_ROUTES = {
    "temporal_compare", "evolution", "evolution_funnel",
    "evolution_pipeline_stage", "forecast_extrapolation", "scenario",
    "cohort_progression", "cohort_conversion", "risk_limits", "funded_bridge",
}


def _route_requires_run(route: Optional[str]) -> bool:
    return bool(route) and route in _RUN_SCOPED_ROUTES
