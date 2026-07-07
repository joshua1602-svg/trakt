"""mi_agent_pptx.mi_api — headless bridge to the MI Agent API computations.

Produces the SAME payloads the React dashboard renders, by calling the exact
compute functions behind the ``/mi/*`` endpoints in-process (no HTTP server, no
LLM, and — deliberately — no FastAPI import, so the deck runs anywhere the
compute modules ship, including the Azure Functions PPTX stage).

Resolution parity is the point: rather than a PPTX-only guesser, the deck resolves
a run exactly as the dashboard does and then calls the identical compute functions.

Historical (multi-period) resolution is covered for both deployments:

* **Azure / blob roots** — ``MI_AGENT_ONBOARDING_OUTPUT_ROOT`` = a ``blob://``
  platform root: funded evolution loads the dated platform canonicals
  (``evolution.funded_frames`` blob branch); ``MI_AGENT_PIPELINE_ROOT`` = a
  ``blob://`` pipeline root is mirrored locally (``_materialise_pipeline_root``, the
  same mirror ``app._pipeline_discovery_root`` performs) so pipeline evolution /
  funnel / run-rate projection discover every dated weekly snapshot.
* **Local downloaded history** — a filesystem root carrying dated cuts
  (``…/{YYYY-MM-DD}/platform_canonical_typed.csv`` for funded,
  ``…/{YYYY-MM-DD}/…pipeline…`` for weekly extracts) is discovered directly; the
  historical cuts do NOT need to live inside the current run directory.
* **Single local run** — one funded cut / one weekly extract: the time-series
  surfaces report ``singlePeriod`` and the deck renders an *insufficient history*
  placeholder (not "data unavailable").

Every resolution is recorded in :attr:`DashboardData.diagnostics` for the deck's
data-coverage appendix (current sources, history roots checked, dated-cut counts,
and the placeholder reason per time-series slide).
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_PLATFORM_CANONICAL_NAME = "platform_canonical_typed.csv"


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
    multidim: Dict[str, Any] = field(default_factory=dict)
    cohort_progression: Dict[str, Any] = field(default_factory=dict)
    source_files: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
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
    conventional = run_path / "out_platform" / _PLATFORM_CANONICAL_NAME
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

def _materialise_pipeline_root(root: Optional[str]) -> Optional[str]:
    """Return a LOCAL discovery root for ``root`` (``app._materialise_pipeline_root``).

    Filesystem roots are returned unchanged. A ``blob://`` root is mirrored to a
    local scratch tree holding ONLY the dated ``{date}/pipeline_snapshot.csv``
    snapshots (``latest/`` excluded) so filesystem discovery / evolution / history
    see the same set of dated sources the dashboard does."""
    if not root or not str(root).startswith("blob://"):
        return root
    try:
        from apps.blob_trigger_app.storage import open_storage, split_blob_uri
        storage = open_storage()
        try:
            uris = storage.list(root)
        except Exception:  # noqa: BLE001
            return root
        dated = [u for u in uris if "/latest/" not in u
                 and re.search(r"/\d{4}-\d{2}-\d{2}/pipeline_snapshot\.csv$", u)]
        if not dated:
            return root
        scratch = os.environ.get("MI_AGENT_SCRATCH", "/tmp/trakt/mi_platform")
        base = Path(scratch) / "pipeline_root"
        _c, key = split_blob_uri(root)
        prefix = key.rstrip("/")
        for uri in dated:
            _cc, ukey = split_blob_uri(uri)
            rel = ukey[len(prefix):].lstrip("/") if ukey.startswith(prefix) else \
                "/".join(uri.rstrip("/").split("/")[-2:])
            dest = base / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            storage.download_file(uri, dest)
        return str(base)
    except Exception:  # noqa: BLE001 — never break discovery on a mirror error
        return root


def _pipeline_discovery_root(out_root: str) -> str:
    """The pipeline discovery root (``app._pipeline_discovery_root``): the pipeline
    root (or the onboarding output root), blob-mirrored to local where needed."""
    root = os.environ.get("MI_AGENT_PIPELINE_ROOT") or out_root
    return _materialise_pipeline_root(root) or root


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


def _pipeline_client(prow, cid: str) -> str:
    """The client the governed pipeline sources actually live under.

    Prefer the funded client (``app._resolve_pipeline_source`` passes it, and in
    production the pipeline root carries the client in its path). When strict
    path-inferred matching finds nothing under that client — a local run layout
    where the pipeline tree is keyed by ``direct_001`` / the run folder rather than
    the funded client — fall back to the client discovery infers from the tree."""
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


# --------------------------------------------------------------------------- #
# Dated-cut discovery for the diagnostics + local (downloaded) history support.
# --------------------------------------------------------------------------- #

def _dated_funded_cuts(out_root: str, cid: str) -> List[Tuple[str, str]]:
    """``[(date, uri_or_path)]`` for every dated funded platform canonical under
    *out_root* — the dashboard's blob cuts, or a local downloaded history tree
    (``…/{YYYY-MM-DD}/platform_canonical_typed.csv``). Oldest → newest."""
    root = str(out_root)
    if str(root).startswith("blob://"):
        try:
            from apps.blob_trigger_app.storage import open_storage
            from mi_agent_api import platform_snapshots_blob as pb
            dated = pb.list_dated_platform_canonicals(root, open_storage())
            return [(d["date"], d["uri"]) for d in dated]
        except Exception:  # noqa: BLE001
            return []
    cuts: Dict[str, str] = {}
    base = Path(root)
    if base.exists():
        for p in base.glob(f"**/{_PLATFORM_CANONICAL_NAME}"):
            date = p.parent.name if _DATE_RE.fullmatch(p.parent.name) else None
            if not date:
                m = _DATE_RE.search(str(p))
                date = m.group(1) if m else None
            if date:
                cuts.setdefault(date, str(p))
    return sorted(cuts.items())


def _local_funded_frames(cuts: List[Tuple[str, str]], cid: str) -> List[Dict[str, Any]]:
    """Prepared funded frames from LOCAL dated platform canonicals, scoped to the
    client — the local analogue of ``platform_snapshots_blob.build_funded_evolution_frames``."""
    from mi_agent_api.funded_prep import prepare_funded_mi_dataset
    frames: List[Dict[str, Any]] = []
    for date, path in cuts:
        try:
            raw = pd.read_csv(path, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        if cid and "source_portfolio_id" in raw.columns:
            ids = raw["source_portfolio_id"].astype(str).str.strip()
            if (ids == cid).any():
                raw = raw[ids == cid]
        try:
            df, _rep = prepare_funded_mi_dataset(raw)
        except Exception:  # noqa: BLE001
            continue
        frames.append({"run_id": date, "reporting_date": date, "df": df, "source": path})
    return frames


def _prior_funded(cuts: List[Tuple[str, str]], cid: str, reporting_date: Optional[str]):
    """The prepared funded frame for the reporting period BEFORE *reporting_date*
    (the most recent dated cut strictly earlier), for month-on-month KPI deltas."""
    if not cuts:
        return None, None, None
    earlier = [(d, p) for d, p in cuts if not reporting_date or d < str(reporting_date)]
    if not earlier:
        return None, None, None
    frames = _local_funded_frames([earlier[-1]], cid)
    if not frames:
        return None, None, None
    d, _p = earlier[-1]
    return frames[0]["df"], d, d


# --------------------------------------------------------------------------- #
# Deck-specific funded enrichments (same stratify / banding engine as the API).
# --------------------------------------------------------------------------- #

_BALANCE = "current_outstanding_balance"


def _titled(series):
    """Title-case category labels (South West, Single, Joint) for readable axes."""
    if series is None:
        return None
    return series.astype("string").str.strip().str.title()


def _borrower_type_series(df: pd.DataFrame):
    if "borrower_type" in df.columns and df["borrower_type"].notna().any():
        return _titled(df["borrower_type"])
    for col in ("borrower_2_DOB", "borrower_2_dob", "second_borrower_dob",
                "borrower_2_date_of_birth", "borrower_2_date_of_death"):
        if col in df.columns:
            joint = df[col].notna() & (df[col].astype(str).str.strip().isin(("", "nan", "NaT")) == False)  # noqa: E712
            return joint.map({True: "Joint", False: "Single"}).astype("string")
    return None


def _ticket_series(df: pd.DataFrame):
    if "ticket_bucket" in df.columns and df["ticket_bucket"].notna().any():
        return df["ticket_bucket"].astype("string")
    if _BALANCE not in df.columns:
        return None
    from analytics_lib.numeric import coerce_numeric
    bal = coerce_numeric(df[_BALANCE])
    bins = [0, 100_000, 150_000, 200_000, 250_000, 300_000, 400_000, 1e12]
    labels = ["<£100K", "£100–150K", "£150–200K", "£200–250K", "£250–300K",
              "£300–400K", "£400K+"]
    return pd.cut(bal, bins, labels=labels, right=False).astype("string")


def _broker_series(df: pd.DataFrame):
    for col in ("broker_channel", "broker_name", "broker", "origination_channel"):
        if col in df.columns and df[col].notna().any():
            return df[col].astype("string")
    return None


def _region_series(df: pd.DataFrame):
    # Prefer READABLE region names (South West, North East) over ITL3 codes.
    for col in ("collateral_geography", "property_region", "region_name",
                "geographic_region_collateral", "geographic_region_obligor", "region"):
        if col in df.columns and df[col].notna().any():
            vals = df[col].astype("string")
            # If the column is ITL3 codes (e.g. TLI3), keep as-is; else title-case.
            sample = vals.dropna().head(20)
            is_code = bool(len(sample)) and sample.str.match(r"^TL[A-Z0-9]{2,3}$").mean() > 0.5
            return vals if is_code else _titled(vals)
    return None


def _ltv_series(df: pd.DataFrame):
    from mi_agent_api import cohorts as _c
    series, _h = _c._dimension_series(df, "ltv", "Y")
    return series


def _age_series(df: pd.DataFrame):
    from mi_agent_api import cohorts as _c
    series, _h = _c._dimension_series(df, "age", "Y")
    return series


def _stratify_dim(df: pd.DataFrame, series, key: str, label: str):
    """One stratification ``{key,label,bars:[{label,balance,count,sharePct}]}`` —
    the same shape ``snapshots._funded_stratifications`` emits."""
    if series is None or _BALANCE not in df.columns:
        return None
    from analytics_lib.stratify import stratify as _stratify
    work = df.assign(__dim=series)
    if work["__dim"].notna().sum() == 0:
        return None
    try:
        tbl = _stratify(work, "__dim", balance_col=_BALANCE)
    except Exception:  # noqa: BLE001
        return None
    if tbl.empty:
        return None
    bars = [{"label": str(r["__dim"]), "balance": round(float(r["balance_sum"]), 2),
             "count": int(r["loan_count"]),
             "sharePct": round(float(r["balance_share"]) * 100.0, 1)}
            for _, r in tbl.iterrows()]
    bars.sort(key=lambda b: b["balance"], reverse=True)
    return {"key": key, "label": label, "bars": bars[:12]}


def _extra_stratifications(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Broker / borrower-type / ticket-size funded stratifications (each skipped
    when the source column is absent)."""
    out = []
    for series, key, label in (
        (_broker_series(df), "broker", "By broker / channel"),
        (_borrower_type_series(df), "borrower_type", "By borrower type"),
        (_ticket_series(df), "ticket", "By ticket size"),
    ):
        st = _stratify_dim(df, series, key, label)
        if st:
            out.append(st)
    return out


def _matrix(df: pd.DataFrame, x_series, y_series):
    """``(x_labels, y_labels, matrix, points)`` of summed balance for two banded
    dimensions — feeds the multi-dimension bubble / heatmap charts."""
    from analytics_lib.numeric import coerce_numeric
    if x_series is None or y_series is None or _BALANCE not in df.columns:
        return None
    work = pd.DataFrame({"x": x_series.astype("string"), "y": y_series.astype("string"),
                         "bal": coerce_numeric(df[_BALANCE])}).dropna(subset=["x", "y"])
    if work.empty:
        return None
    x_labels = [str(v) for v in sorted(work["x"].dropna().unique())]
    y_labels = [str(v) for v in sorted(work["y"].dropna().unique())]
    xi = {v: i for i, v in enumerate(x_labels)}
    yi = {v: i for i, v in enumerate(y_labels)}
    matrix = [[0.0] * len(x_labels) for _ in y_labels]
    points = []
    for (xv, yv), sub in work.groupby(["x", "y"]):
        b = float(sub["bal"].sum())
        matrix[yi[str(yv)]][xi[str(xv)]] = round(b, 2)
        points.append({"x": xi[str(xv)], "y": yi[str(yv)], "value": round(b, 2)})
    return {"xLabels": x_labels, "yLabels": y_labels, "matrix": matrix, "points": points}


def _loan_points(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Per-LOAN (LTV%, youngest age, balance) points for the bubble scatter."""
    from analytics_lib.numeric import coerce_numeric
    if not {_BALANCE, "current_loan_to_value", "youngest_borrower_age"} <= set(df.columns):
        return []
    ltv = coerce_numeric(df["current_loan_to_value"])
    ltv = ltv.where(ltv.abs() > 1.5, ltv * 100.0)  # fraction → LTV points
    age = coerce_numeric(df["youngest_borrower_age"])
    bal = coerce_numeric(df[_BALANCE])
    work = pd.DataFrame({"ltv": ltv, "age": age, "bal": bal}).dropna()
    return [{"ltv": round(float(r.ltv), 1), "age": round(float(r.age), 1),
             "balance": round(float(r.bal), 2)} for r in work.itertuples()]


def _multidim(df: pd.DataFrame) -> Dict[str, Any]:
    """LTV×Age (per-loan bubble), LTV×BorrowerType (heatmap), LTV×Region (heatmap)."""
    ltv = _ltv_series(df)
    out: Dict[str, Any] = {}
    loans = _loan_points(df)
    if loans:
        out["ltv_age_loans"] = loans
    m = _matrix(df, ltv, _borrower_type_series(df))
    if m:
        out["ltv_borrower_type"] = m
    m = _matrix(df, ltv, _region_series(df))
    if m:
        out["ltv_region"] = m
    return out


def _cohort_progression(out_root: str, cid: str) -> Dict[str, Any]:
    """Static-pool cohort progression across reporting periods (line curves per
    vintage) — the dashboard's /mi/cohorts/progression."""
    from mi_agent_api import evolution
    return evolution.funded_cohort_progression(out_root, cid, grain="Y")


def _pipeline_extract_count(root: str, cid: str) -> int:
    """Number of dated weekly pipeline extracts discoverable under *root* for the
    client (the dashboard's ``weekly_extract_inventory``)."""
    from mi_agent_api import pipeline_contract as pc
    try:
        return int(len(pc.weekly_extract_inventory(root, cid).get("extracts", [])))
    except Exception:  # noqa: BLE001
        return 0


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #

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

        funded_cuts = _dated_funded_cuts(out_root, cid)
        prior_df, prior_rid, prior_rd = _prior_funded(funded_cuts, cid, reporting_date)

        # -- FUNDED snapshot (KPIs + stratifications) --------------------
        if funded_df is not None and not funded_df.empty:
            data.funded = _guard(data, "funded_snapshot", lambda: snap.compute_funded_snapshot(
                funded_df, semantics, client_id=cid, run_id=rid,
                reporting_date=reporting_date, prior_df=prior_df,
                prior_run_id=prior_rid, prior_reporting_date=prior_rd))
            # Extra funded stratifications the deck shows (broker / borrower type /
            # ticket size) — computed with the same stratify engine as the snapshot,
            # appended so the deck renders one consistent BarList visual.
            extra = _extra_stratifications(funded_df)
            if extra and isinstance(data.funded.get("stratifications"), list):
                have = {s.get("key") for s in data.funded["stratifications"]}
                data.funded["stratifications"] += [s for s in extra if s["key"] not in have]
            data.multidim = _guard(data, "multidim", lambda: _multidim(funded_df))
            data.cohorts = _guard(data, "cohorts",
                                  lambda: _cohorts(funded_df, cid, pid, reporting_date))
            data.cohort_progression = _guard(data, "cohort_progression",
                                             lambda: _cohort_progression(out_root, cid))
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

        # -- Multi-run EVOLUTION / FUNNEL / FORECAST ---------------------
        data.funded_evolution = _guard(data, "funded_evolution",
                                       lambda: _funded_evo(out_root, cid, rid, funded_cuts))
        data.pipeline_evolution = _guard(data, "pipeline_evolution",
                                         lambda: _pipeline_evo(prow, pipe_cid, history))
        data.funnel = _guard(data, "funnel", lambda: _funnel(prow, pipe_cid, history))
        data.forecast_evolution = _guard(data, "forecast_evolution",
                                         lambda: _forecast_evo(out_root, prow, cid, rid, history))

        # -- RISK limits / FORECAST extrapolation (multi-run) ------------
        data.risk = _guard(data, "risk", lambda: _risk(out_root, cid, rid))
        data.extrapolation = _guard(data, "extrapolation",
                                    lambda: _extrapolation(out_root, prow, cid, rid, history,
                                                           data.funded_evolution, reporting_date))

        pipe_snapshots = _pipeline_extract_count(prow, pipe_cid)

    # Provenance + diagnostics ------------------------------------------
    if source and source.get("source_file"):
        data.source_files.append(Path(source["source_file"]).name)
    data.diagnostics = _diagnostics(data, out_root, prow, funded_uri, source,
                                    funded_cuts, pipe_snapshots)

    if not data.pipeline:
        data.note("No pipeline source resolved — pipeline & forecast slides "
                  "render as branded placeholders.")
    return data


def _diagnostics(data, out_root, prow, funded_uri, source, funded_cuts, pipe_snapshots):
    """The data-coverage provenance the appendix renders (requirement #4)."""
    def _pph(payload, min_periods=2):
        periods = len(payload.get("periods", []))
        single = bool(payload.get("singlePeriod")) or periods < min_periods
        return periods, single

    f_periods, f_single = _pph(data.funded_evolution)
    p_periods, p_single = _pph(data.pipeline_evolution)
    proj = ((data.extrapolation.get("completionRunRateForecast") or {}).get("available")
            or (data.extrapolation.get("kfiConversionForecast") or {}).get("available"))
    risk_ok = bool(data.risk.get("available", False)) or bool(data.risk.get("tests"))
    pipeline_ok = bool(data.pipeline)
    return {
        "fundedCurrentSource": Path(funded_uri).name if funded_uri else None,
        "pipelineCurrentSource": (Path(source["source_file"]).name
                                  if source and source.get("source_file") else None),
        "fundedHistoryRoot": out_root,
        "fundedCutsFound": len(funded_cuts),
        "pipelineHistoryRoot": prow,
        "pipelineSnapshotsFound": pipe_snapshots,
        "timeSeries": {
            "funded_evolution": {
                "placeholder": f_single,
                "reason": (f"insufficient history — {len(funded_cuts)} funded cut(s) "
                           f"found, need ≥2" if f_single else None),
                "periods": f_periods},
            "pipeline_evolution": {
                "placeholder": p_single,
                "reason": (f"insufficient history — {pipe_snapshots} weekly extract(s) "
                           f"found, need ≥2" if p_single else None),
                "periods": p_periods},
            "funnel": {
                "placeholder": not pipeline_ok,
                "reason": ("current-week funnel shown (single weekly extract)"
                           if pipeline_ok and pipe_snapshots < 2 else
                           (None if pipeline_ok else "no pipeline source resolved"))},
            "forecast_projection": {
                "placeholder": not proj,
                "reason": (None if proj else
                           f"insufficient run-rate history — {pipe_snapshots} weekly "
                           f"extract(s) found")},
            "risk": {
                "placeholder": not risk_ok,
                "reason": (None if risk_ok else "no Schedule 8 risk-limit extract")},
        },
    }


# --------------------------------------------------------------------------- #
# Per-endpoint compute wrappers (call the SAME functions app.py's handlers call).
# --------------------------------------------------------------------------- #

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


def _funded_evo(out_root, cid, rid, funded_cuts):
    """Funded evolution: the dashboard's resolver first (blob dated cuts / local
    central-tape cuts); when that yields <2 periods, supplement from LOCAL dated
    platform canonicals so downloaded history renders too (requirement #3)."""
    from mi_agent_api import evolution
    result = evolution.funded_evolution(out_root, cid, rid)
    if len(result.get("periods", [])) >= 2:
        return result
    frames = _local_funded_frames(funded_cuts, cid)
    if len(frames) >= 2:
        return evolution.assemble_funded_evolution(frames, cid, rid, lineage={
            "source": "dated platform canonicals (platform_canonical_typed.csv)",
            "metric": "funded book actuals per reporting cut",
            "note": "One period per dated funded cut under the onboarding output root."})
    return result


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


def _extrapolation(out_root, prow, cid, rid, history, funded_evo, reporting_date):
    """The scale-up extrapolation. ``build_extrapolation`` reads funded completion
    history from ``evolution.funded_evolution`` (which misses local dated platform
    canonicals); when it comes back insufficient but our resolved funded evolution
    has ≥2 periods, recompute the completion run-rate model from those periods with
    the endpoint's OWN public helpers — so it renders locally exactly as the
    dashboard renders it against a blob root."""
    from mi_agent_api import forecast_extrapolation as fx
    result = fx.build_extrapolation(out_root, prow or out_root, cid, rid,
                                    history_model=history)
    crf = result.get("completionRunRateForecast") or {}
    if not crf.get("available"):
        periods = (funded_evo or {}).get("periods", [])
        comp = fx.completion_history(periods) if len(periods) >= 2 else []
        if comp:
            latest = (periods[-1].get("metrics") or {}).get("funded_balance")
            current = float(latest if latest is not None else result.get("currentFundedBalance") or 0)
            rp = str(reporting_date)[:7] if reporting_date else None
            model = fx.run_rate_model(current, [c["completion_amount"] for c in comp],
                                      reporting_period=rp)
            model["completionHistory"] = comp
            result["completionRunRateForecast"] = model
            result["dataSufficiency"] = model.get("status", result.get("dataSufficiency"))
    return result
