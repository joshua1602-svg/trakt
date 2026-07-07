"""mi_agent_pptx.mi_api — headless bridge to the MI Agent API computations.

Produces the SAME payloads the React dashboard renders, by calling the exact
compute functions behind the ``/mi/*`` endpoints in-process (no HTTP server, no
LLM, and — deliberately — no FastAPI import, so the deck runs anywhere the
compute modules ship, including the Azure Functions PPTX stage).

The endpoint HANDLERS live in ``mi_agent_api.app`` (which pulls in FastAPI); this
module instead mirrors the *thin resolution helpers* those handlers use — the
onboarding output root, the materialised pipeline discovery root, and the
multi-week historical completion model — and then calls the identical underlying
compute functions:

* ``/mi/snapshot``               → ``snapshots.compute_funded_snapshot``
* ``/mi/pipeline/snapshot``      → ``pipeline_contract`` prep + ``compute_pipeline_snapshot``
* ``/mi/forecast/snapshot``      → ``forecast_bridge`` + ``workspace.forecast_breakdowns``
* ``/mi/evolution/{funded,pipeline,funnel,forecast}`` → ``evolution.*`` (with the
  SAME ``historical_model`` / KFI-lag the dashboard passes)
* ``/mi/cohorts`` · ``/mi/geo/exposure`` · ``/mi/risk-limits`` · ``/mi/forecast/extrapolation``

Because resolution matches the dashboard, the deck populates exactly where the
dashboard populates: against a root with reporting history the evolution / funnel /
forecast surfaces fill in; against a single run they degrade to the same
single-period state and the deck renders a branded placeholder — never a failure.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
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
    pipeline: Dict[str, Any] = field(default_factory=dict)        # /mi/pipeline/snapshot
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


def _guard(note_target: DashboardData, label: str,
           fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
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


def _funded_canonical(run_path: Path) -> Optional[str]:
    """The run's funded platform canonical (the funded book the dashboard serves)."""
    try:
        from .artifact_loader import load_run_artifacts
        art = load_run_artifacts(run_path)
        if art.has_tape and art.tape_path is not None:
            return str(art.tape_path)
    except Exception:  # noqa: BLE001
        pass
    conventional = run_path / "out_platform" / "platform_canonical_typed.csv"
    return str(conventional) if conventional.exists() else None


@contextmanager
def _api_env(overrides: Dict[str, Optional[str]]):
    """Temporarily set the env the MI Agent API reads, then restore it — so the
    deck resolves a run exactly as the dashboard does, without leaking config."""
    saved = {k: os.environ.get(k) for k in overrides}

    def _reset_cache():
        try:
            from mi_agent_api import data_source
            data_source.reset_cache()
        except Exception:  # noqa: BLE001
            pass

    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _reset_cache()
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _reset_cache()


# --------------------------------------------------------------------------- #
# Resolution helpers — mirror the like-named private helpers in mi_agent_api.app
# (which cannot be imported here because that module pulls in FastAPI).
# --------------------------------------------------------------------------- #

def _pipeline_discovery_root(out_root: str) -> str:
    """The pipeline discovery root (``app._pipeline_discovery_root``).

    Precedence: ``MI_AGENT_PIPELINE_ROOT`` → the onboarding output root. Filesystem
    roots (the deck's runtime — local downloaded runs) are used unchanged; blob
    mirroring is an app-server concern and never applies to a local run dir."""
    return os.environ.get("MI_AGENT_PIPELINE_ROOT") or out_root


def _pipeline_history(root: str, client_id: str) -> Optional[Dict[str, Any]]:
    """The multi-week historical completion model (``app._pipeline_history``):
    None unless ≥2 weekly extracts exist (a single extract is not a history)."""
    from mi_agent_api import pipeline_contract as pc
    try:
        model = pc.build_pipeline_history(root, client_id)
    except Exception:  # noqa: BLE001
        return None
    if int((model or {}).get("uniqueWeeklyExtractsUsed", 0)) < 2:
        return None
    return model


def _kfi_lag_weeks(model: Optional[Dict[str, Any]]) -> Optional[int]:
    """Median KFI→completion lag in whole weeks (``app._kfi_lag_weeks_from_model``)."""
    timing = ((model or {}).get("historicalCompletionTimingByStage") or {}).get("KFI") or {}
    median_days = timing.get("medianDays")
    return max(1, round(float(median_days) / 7.0)) if median_days else None


def _funded_frame(cid: str) -> Optional[pd.DataFrame]:
    """The prepared funded frame for the active run (the platform canonical set via
    ``MI_AGENT_PLATFORM_CANONICAL``), scoped to the client when the canonical
    carries multiple source portfolios — mirrors ``app._resolve_run_dataframe``."""
    from mi_agent_api import data_source
    df = data_source.get_dataframe()
    if df is None or df.empty:
        return None
    if cid and "source_portfolio_id" in df.columns:
        ids = df["source_portfolio_id"].astype(str).str.strip()
        if (ids == cid).any():
            return df[ids == cid]
    return df


def build_dashboard_data(
    run_dir: str | Path,
    *,
    client_id: Optional[str] = None,
    run_id: Optional[str] = None,
    as_of: Optional[str] = None,
    output_root: Optional[str] = None,
    pipeline_root: Optional[str] = None,
    prior_run_dir: Optional[str] = None,  # accepted for CLI compatibility (unused)
) -> DashboardData:
    """Compute the full set of dashboard payloads for *run_dir*, headless."""
    run_path = Path(run_dir)
    cid, rid, rs = _run_ids(run_path, client_id, run_id)
    data = DashboardData(client_id=cid, run_id=rid)
    pid = f"{cid}/{rid}"

    funded_uri = _funded_canonical(run_path)
    out_root = (output_root or os.environ.get("MI_AGENT_ONBOARDING_OUTPUT_ROOT")
                or str(run_path.parent))
    pipe_root_env = pipeline_root or os.environ.get("MI_AGENT_PIPELINE_ROOT")
    overrides: Dict[str, Optional[str]] = {
        "MI_AGENT_ONBOARDING_OUTPUT_ROOT": out_root,
        "MI_AGENT_PIPELINE_ROOT": pipe_root_env or str(run_path.parent),
        "MI_AGENT_CLIENT_ID": cid,
    }
    if funded_uri:
        overrides["MI_AGENT_PLATFORM_CANONICAL"] = funded_uri

    with _api_env(overrides):
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.data_source import semantics_path
        from mi_agent_api import snapshots as snap

        try:
            semantics = load_mi_semantics(semantics_path())
        except Exception as exc:  # noqa: BLE001
            data.note(f"semantics: {exc}")
            semantics = {}

        prow = _pipeline_discovery_root(out_root)
        # The pipeline is keyed by its governed source client, which is the funded
        # client in production (the pipeline root carries the client in its path) but
        # can differ under a local run layout (funded ERE vs a direct_001 / run-folder
        # pipeline tree). Resolve the client the sources actually live under so the
        # dashboard-exact resolver finds them, then use it for every pipeline-side call.
        pipe_cid = _pipeline_client(prow, cid)
        history = _pipeline_history(prow, pipe_cid)

        funded_df = _funded_frame(cid)
        reporting_date = as_of or rs.get("reporting_date")
        if funded_df is not None and not funded_df.empty:
            try:
                reporting_date = reporting_date or snap.infer_reporting_date(rid, funded_df)
            except Exception:  # noqa: BLE001
                pass
        data.reporting_date = reporting_date

        # -- FUNDED snapshot (KPIs + stratifications) --------------------
        if funded_df is not None and not funded_df.empty:
            data.funded = _guard(data, "funded_snapshot", lambda: snap.compute_funded_snapshot(
                funded_df, semantics, client_id=cid, run_id=rid,
                reporting_date=reporting_date))
            data.cohorts = _guard(data, "cohorts",
                                  lambda: _cohorts(funded_df, cid, pid, reporting_date))
            data.geo = _guard(data, "geo", lambda: _geo(funded_df, cid, rid))
        else:
            data.note("No funded dataset resolved for this run — funded slides "
                      "render as branded placeholders.")

        # -- PIPELINE snapshot (latest governed weekly extract) ----------
        pipe_df, pipe_report, source = _pipeline(prow, pipe_cid, rid, semantics, history, data)

        # -- FORECAST snapshot (funded + weighted pipeline bridge) -------
        data.forecast = _guard(data, "forecast", lambda: _forecast(
            cid, rid, reporting_date, funded_df, pipe_df, pipe_report,
            data.pipeline, source))

        # -- Multi-run EVOLUTION / FUNNEL / FORECAST (same history model)-
        data.funded_evolution = _guard(data, "funded_evolution",
                                       lambda: _funded_evo(out_root, cid, rid))
        data.pipeline_evolution = _guard(data, "pipeline_evolution",
                                         lambda: _pipeline_evo(prow, pipe_cid, history))
        data.funnel = _guard(data, "funnel",
                             lambda: _funnel(prow, pipe_cid, history))
        data.forecast_evolution = _guard(data, "forecast_evolution",
                                         lambda: _forecast_evo(out_root, prow, cid, rid, history))

        # -- RISK limits / FORECAST extrapolation (multi-run) ------------
        data.risk = _guard(data, "risk", lambda: _risk(out_root, cid, rid))
        data.extrapolation = _guard(data, "extrapolation",
                                    lambda: _extrapolation(out_root, prow, cid, rid, history))

    if not data.pipeline:
        data.note("No pipeline source resolved — pipeline & forecast slides "
                  "render as branded placeholders.")
    if source and source.get("source_file"):
        data.source_files.append(Path(source["source_file"]).name)
    return data


# --------------------------------------------------------------------------- #
# Per-endpoint compute wrappers (call the SAME functions app.py's handlers call).
# --------------------------------------------------------------------------- #

def _pipeline_client(prow, cid: str) -> str:
    """The client the governed pipeline sources actually live under.

    Prefer the funded client (``app._resolve_pipeline_source`` passes it, and in
    production the pipeline root carries the client in its path). When strict
    path-inferred matching finds nothing under that client — a local run layout
    where the pipeline tree is keyed by ``direct_001`` / the run folder rather than
    the funded client — fall back to the client discovery infers from the tree so
    the same governed source the dashboard uses is still found."""
    from mi_agent_api import pipeline_contract as pc
    try:
        if pc.resolve_pipeline_source(prow, cid, None):
            return cid
    except Exception:  # noqa: BLE001
        pass
    try:
        srcs = pc.discover_pipeline_sources(prow)  # client_id=None → all, inferred
        if srcs:
            return srcs[-1].get("client_id") or cid
    except Exception:  # noqa: BLE001
        pass
    return cid


def _pipeline(prow, cid, rid, semantics, history, data: DashboardData):
    """Resolve + snapshot the latest governed weekly pipeline extract."""
    from mi_agent_api import pipeline_contract as pc
    try:
        source = pc.resolve_pipeline_source(prow, cid, rid)
    except Exception as exc:  # noqa: BLE001
        data.note(f"pipeline_source: {exc}")
        source = None
    if not source:
        return None, None, None
    try:
        pdf, report = pc.load_prepared_pipeline(source, historical_model=history)
        try:
            prior_week = pc.compute_prior_week_aggregates(source, historical_model=history)
        except Exception:  # noqa: BLE001
            prior_week = None
        snap_out = pc.compute_pipeline_snapshot(
            pdf, report, semantics, client_id=source.get("client_id", cid),
            run_id=rid, source=source, prior_week=prior_week)
        if snap_out.get("pipelineRowCount"):
            data.pipeline = snap_out
        return pdf, report, source
    except Exception as exc:  # noqa: BLE001
        data.note(f"pipeline_snapshot: {exc}")
        return None, None, source


def _forecast(cid, rid, reporting_date, funded_df, pipe_df, pipe_report, pipe_snap, source):
    from mi_agent_api import forecast_bridge as fb
    from mi_agent_api import workspace
    env = fb.compute_forecast_bridge(
        client_id=cid, run_id=rid, funded_reporting_date=reporting_date,
        funded_df=funded_df, pipeline_df=pipe_df, pipeline_report=pipe_report,
        pipeline_snapshot=(pipe_snap or None), pipeline_source=source)
    try:
        env["forecastBreakdowns"] = workspace.forecast_breakdowns(funded_df, pipe_df)
    except Exception:  # noqa: BLE001
        env.setdefault("forecastBreakdowns", {})
    return env


def _cohorts(funded_df, cid, pid, reporting_date):
    from mi_agent_api import cohorts
    return cohorts.cohort_analysis(
        funded_df, client_id=cid, portfolio_id=pid,
        reporting_date=reporting_date, grain="Y", dimension="vintage")


def _geo(funded_df, cid, rid):
    from mi_agent_api import geo
    out = dict(geo.exposure_by_itl3(funded_df))
    out.update({"dataset": "geo_itl3", "portfolioId": f"{cid}/{rid}"})
    return out


def _funded_evo(out_root, cid, rid):
    from mi_agent_api import evolution
    return evolution.funded_evolution(out_root, cid, rid)


def _pipeline_evo(prow, cid, history):
    from mi_agent_api import evolution
    return evolution.pipeline_evolution(prow, cid, None, historical_model=history)


def _funnel(prow, cid, history):
    from mi_agent_api import evolution
    return evolution.pipeline_funnel_evolution(
        prow, cid, None, lag_weeks=_kfi_lag_weeks(history))


def _forecast_evo(out_root, prow, cid, rid, history):
    from mi_agent_api import evolution
    return evolution.forecast_evolution(out_root, prow or out_root, cid, rid,
                                        historical_model=history)


def _risk(out_root, cid, rid):
    from mi_agent_api import risk_limits
    return risk_limits.compute_risk_limits(out_root, cid, rid)


def _extrapolation(out_root, prow, cid, rid, history):
    from mi_agent_api import forecast_extrapolation as fx
    return fx.build_extrapolation(out_root, prow or out_root, cid, rid,
                                  history_model=history)
