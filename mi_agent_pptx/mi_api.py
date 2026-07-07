"""mi_agent_pptx.mi_api — headless bridge to the MI Agent API computations.

Produces the SAME payloads the React dashboard renders, by calling the exact
compute functions behind the ``/mi/*`` endpoints in-process (no HTTP server, no
LLM). The deck then renders those payloads verbatim, so every number in the pack
equals the dashboard for the same ``portfolioId = "<client_id>/<run_id>"``:

* ``/mi/snapshot``            → ``snapshots.compute_funded_snapshot``
* ``/mi/forecast/snapshot``   → ``pipeline_contract`` + ``forecast_bridge`` + ``workspace``
* ``/mi/evolution/{funded,pipeline,funnel,forecast}`` → ``evolution.*``
* ``/mi/cohorts``             → ``cohorts.cohort_analysis``
* ``/mi/geo/exposure``        → ``geo.exposure_by_itl3``
* ``/mi/risk-limits``         → ``risk_limits.compute_risk_limits``
* ``/mi/forecast/extrapolation`` → ``forecast_extrapolation.build_extrapolation``

Every call is individually guarded: a payload that can't be computed comes back
empty (``{}``) and its slide degrades to a branded placeholder — the deck never
fails, and a slide is a placeholder only when the dashboard would also have no
data for it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


@dataclass
class DashboardData:
    """The dashboard payloads for one run, plus resolution provenance."""

    client_id: str
    run_id: str
    reporting_date: Optional[str] = None
    funded: Dict[str, Any] = field(default_factory=dict)          # /mi/snapshot
    pipeline: Dict[str, Any] = field(default_factory=dict)        # pipelineSnapshot
    forecast: Dict[str, Any] = field(default_factory=dict)        # /mi/forecast/snapshot
    funded_evolution: Dict[str, Any] = field(default_factory=dict)
    pipeline_evolution: Dict[str, Any] = field(default_factory=dict)
    funnel: Dict[str, Any] = field(default_factory=dict)
    forecast_evolution: Dict[str, Any] = field(default_factory=dict)
    cohorts: Dict[str, Any] = field(default_factory=dict)
    geo: Dict[str, Any] = field(default_factory=dict)
    risk: Dict[str, Any] = field(default_factory=dict)
    extrapolation: Dict[str, Any] = field(default_factory=dict)
    source_files: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        if msg and msg not in self.notes:
            self.notes.append(msg)


def _guard(note_target: DashboardData, label: str, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """Run *fn*, returning ``{}`` and recording a note on failure."""
    try:
        out = fn()
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001 — a missing payload must not fail the deck
        note_target.note(f"{label}: {type(exc).__name__}: {exc}")
        return {}


def _run_ids(run_dir: Path, client_id: Optional[str], run_id: Optional[str]):
    """Resolve (client_id, run_id) from run_state.json / the run dir name."""
    rs: Dict[str, Any] = {}
    p = run_dir / "run_state.json"
    if p.exists():
        try:
            rs = json.loads(p.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            rs = {}
    cid = client_id or rs.get("client_id") or "client"
    rid = run_id or rs.get("run_id") or run_dir.name
    return str(cid), str(rid), rs


def build_dashboard_data(
    run_dir: str | Path,
    *,
    client_id: Optional[str] = None,
    run_id: Optional[str] = None,
    as_of: Optional[str] = None,
    output_root: Optional[str] = None,
    pipeline_root: Optional[str] = None,
    prior_run_dir: Optional[str] = None,
) -> DashboardData:
    """Compute the full set of dashboard payloads for *run_dir*, headless."""
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    from mi_agent_api import snapshots as snap

    run_path = Path(run_dir)
    cid, rid, rs = _run_ids(run_path, client_id, run_id)
    data = DashboardData(client_id=cid, run_id=rid)

    try:
        semantics = load_mi_semantics(semantics_path())
    except Exception as exc:  # noqa: BLE001
        data.note(f"semantics: {exc}")
        semantics = {}

    # -- funded frame (the prepared MI dataset the dashboard uses) --------
    funded_df = _resolve_funded_df(run_path)
    reporting_date = as_of or rs.get("reporting_date")
    if funded_df is not None and not funded_df.empty:
        try:
            reporting_date = reporting_date or snap.infer_reporting_date(rid, funded_df)
        except Exception:  # noqa: BLE001
            pass
    data.reporting_date = reporting_date

    # Roots for the multi-run (evolution / risk) endpoints. Pipeline & funded
    # history live across sibling runs, so default to the run dir's parent.
    out_root = (output_root or os.environ.get("MI_AGENT_ONBOARDING_OUTPUT_ROOT")
                or str(run_path.parent))
    prow = _pipeline_root(run_path, pipeline_root)

    # -- prior funded frame (for month-on-month KPI deltas) --------------
    prior_df = _resolve_funded_df(Path(prior_run_dir)) if prior_run_dir else None

    # -- FUNDED snapshot (KPIs + stratifications) ------------------------
    if funded_df is not None and not funded_df.empty:
        data.funded = _guard(data, "funded_snapshot", lambda: snap.compute_funded_snapshot(
            funded_df, semantics, client_id=cid, run_id=rid,
            reporting_date=reporting_date, prior_df=prior_df))
        data.cohorts = _guard(data, "cohorts", lambda: _cohorts(funded_df, cid, rid, reporting_date))
        data.geo = _guard(data, "geo", lambda: _geo(funded_df, cid, rid))
    else:
        data.note("No funded dataset resolved for this run — funded slides "
                  "render as branded placeholders.")

    # -- PIPELINE + FORECAST (rich weekly source across runs) ------------
    pipe_df, pipe_report, pipe_snap, source = _pipeline(run_path, prow, cid, rid,
                                                        semantics, data)
    if pipe_snap:
        data.pipeline = pipe_snap
    data.forecast = _guard(data, "forecast", lambda: _forecast(
        cid, rid, reporting_date, funded_df, pipe_df, pipe_report, pipe_snap, source))

    # -- EVOLUTION time series (multi-run) -------------------------------
    data.funded_evolution = _guard(data, "funded_evolution",
                                   lambda: _funded_evolution(out_root, cid, rid))
    data.pipeline_evolution = _guard(data, "pipeline_evolution",
                                     lambda: _pipeline_evolution(prow, cid, rid))
    data.funnel = _guard(data, "funnel", lambda: _funnel(prow, cid, rid))
    data.forecast_evolution = _guard(data, "forecast_evolution",
                                     lambda: _forecast_evolution(out_root, prow, cid, rid))

    # -- RISK limits (multi-run) -----------------------------------------
    data.risk = _guard(data, "risk", lambda: _risk(out_root, cid, rid))

    # -- FORECAST extrapolation (scale-up curve) -------------------------
    data.extrapolation = _guard(data, "extrapolation",
                                lambda: _extrapolation(out_root, prow, cid, rid))

    if source and source.get("source_file"):
        data.source_files.append(Path(source["source_file"]).name)
    return data


# --------------------------------------------------------------------------- #
# Resolution helpers (mirror app.py, filesystem-only).
# --------------------------------------------------------------------------- #

def _resolve_funded_df(run_path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Resolve + prepare the funded frame for a run dir (funded_prep)."""
    if run_path is None or not run_path.exists():
        return None
    from .artifact_loader import load_run_artifacts
    from .cli import _prep_funded
    art = load_run_artifacts(run_path)
    if not art.has_tape:
        return None
    df = _prep_funded(art.tape)
    return df if (df is not None and not df.empty) else None


def _pipeline_root(run_path: Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    env = os.environ.get("MI_AGENT_PIPELINE_ROOT")
    if env:
        return env
    return str(run_path.parent)


def _pipeline(run_path, prow, cid, rid, semantics, data):
    """(df, report, snapshot, source) for the latest weekly pipeline extract."""
    from mi_agent_api import pipeline_contract as pc
    from .cli import _pipeline_roots, _filter_client
    from .artifact_loader import load_run_artifacts

    art = load_run_artifacts(run_path)
    source = None
    for root in _pipeline_roots(art, prow):
        try:
            srcs = _filter_client(pc.discover_pipeline_sources(root), cid)
        except Exception:  # noqa: BLE001
            continue
        if srcs:
            source = srcs[-1]
            break
    if not source:
        data.note("No pipeline source resolved — pipeline & forecast slides "
                  "render as branded placeholders.")
        return None, None, None, None
    try:
        history = pc.build_pipeline_history(
            str(Path(source["source_file"]).parent.parent), source.get("client_id", cid))
    except Exception:  # noqa: BLE001
        history = None
    try:
        pdf, report = pc.load_prepared_pipeline(source, historical_model=history)
        prior_week = None
        try:
            prior_week = pc.compute_prior_week_aggregates(source, historical_model=history)
        except Exception:  # noqa: BLE001
            prior_week = None
        snap = pc.compute_pipeline_snapshot(
            pdf, report, semantics, client_id=source.get("client_id", cid),
            run_id=rid, source=source, prior_week=prior_week)
        return pdf, report, snap, source
    except Exception as exc:  # noqa: BLE001
        data.note(f"pipeline_snapshot: {exc}")
        return None, None, None, source


def _forecast(cid, rid, reporting_date, funded_df, pipe_df, pipe_report, pipe_snap, source):
    from mi_agent_api import forecast_bridge as fb
    from mi_agent_api import workspace
    env = fb.compute_forecast_bridge(
        client_id=cid, run_id=rid, funded_reporting_date=reporting_date,
        funded_df=funded_df, pipeline_df=pipe_df, pipeline_report=pipe_report,
        pipeline_snapshot=pipe_snap, pipeline_source=source)
    try:
        env["forecastBreakdowns"] = workspace.forecast_breakdowns(funded_df, pipe_df)
    except Exception:  # noqa: BLE001
        env.setdefault("forecastBreakdowns", {})
    return env


def _cohorts(funded_df, cid, rid, reporting_date):
    from mi_agent_api import cohorts
    return cohorts.cohort_analysis(
        funded_df, client_id=cid, portfolio_id=f"{cid}/{rid}",
        reporting_date=reporting_date, grain="Y", dimension="vintage")


def _geo(funded_df, cid, rid):
    from mi_agent_api import geo
    out = dict(geo.exposure_by_itl3(funded_df))
    out.update({"dataset": "geo_itl3", "portfolioId": f"{cid}/{rid}"})
    return out


def _funded_evolution(out_root, cid, rid):
    from mi_agent_api import evolution
    return evolution.funded_evolution(out_root, cid, rid)


def _pipeline_evolution(prow, cid, rid):
    from mi_agent_api import evolution
    return evolution.pipeline_evolution(prow, cid, rid)


def _funnel(prow, cid, rid):
    from mi_agent_api import evolution
    return evolution.pipeline_funnel_evolution(prow, cid, rid)


def _forecast_evolution(out_root, prow, cid, rid):
    from mi_agent_api import evolution
    return evolution.forecast_evolution(out_root, prow, cid, rid)


def _risk(out_root, cid, rid):
    from mi_agent_api import risk_limits
    return risk_limits.compute_risk_limits(out_root, cid, rid)


def _extrapolation(out_root, prow, cid, rid):
    from mi_agent_api import forecast_extrapolation as fx
    return fx.build_extrapolation(out_root, prow, cid, rid)
