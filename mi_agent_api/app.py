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
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .auth import auth_guard, principal_from_request

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

logger = logging.getLogger("mi_agent_api")

# Global authentication guard: every /mi/* route requires an authenticated
# principal carrying an MI role (client|operator). Probe/index/docs routes stay
# open. Enforcement is toggled by MI_AGENT_AUTH_ENABLED (default on); see auth.py.
app = FastAPI(title="Trakt MI Agent API", version="1.0.0",
              dependencies=[Depends(auth_guard)])

# CORS. With the SWA linked-backend deployment the UI calls the API same-origin,
# so CORS is not relied on for security. We still restrict it: allowed origins
# come from MI_AGENT_CORS_ORIGINS (comma-separated) and default to the local dev
# servers only. There is deliberately NO "*" fallback — an unset/empty value
# denies cross-origin browser calls rather than opening to any origin.
_origins = os.environ.get(
    "MI_AGENT_CORS_ORIGINS",
    "http://localhost:5173,http://localhost:4173",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


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
        # NOTE: the full ``info`` dict is intentionally NOT echoed here — it
        # carries the server-side dataset file path. Expose only non-sensitive
        # summary fields above.
        "dataAvailable": csv != "unavailable",
        "semantics": semantics_path().name,
        # LLM parser availability (ENABLE_LLM_MI_AGENT + key). The chat runs
        # deterministically when unavailable — surface which mode is live.
        "llm": get_llm_config().to_dict(),
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


def _clean_provenance_value(v: Any) -> Optional[str]:
    """Normalise a provenance cell: pandas NaN / blank / 'nan' → None, so blank
    labels fall back to the source_portfolio_id rather than the string 'nan'."""
    if v is None:
        return None
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return None
    except Exception:  # noqa: BLE001
        pass
    s = str(v).strip()
    return None if s.lower() in ("", "nan", "none", "nat", "<na>") else s


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


def _client_from_platform_uri() -> Optional[str]:
    """Best-effort client id from MI_AGENT_PLATFORM_URI
    (``blob://{processed}/platform/{client}/latest/…``)."""
    uri = os.environ.get("MI_AGENT_PLATFORM_URI") or ""
    parts = [p for p in uri.replace("blob://", "").split("/") if p]
    if "platform" in parts:
        i = parts.index("platform")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def _platform_client_id(df) -> str:
    explicit = os.environ.get("MI_AGENT_CLIENT_ID")
    if explicit:
        return explicit
    from_uri = _client_from_platform_uri()
    if from_uri:
        return from_uri
    if "client_id" in getattr(df, "columns", []):
        vals = df["client_id"].dropna()
        if not vals.empty:
            return str(vals.iloc[0])
    return "platform"


def _period_from_platform_uri() -> Optional[str]:
    """A reporting period embedded in MI_AGENT_PLATFORM_URI, if any
    (``…/platform/{client}/2026-01-31/…`` or ``…/2026-01/…``). ``/latest/`` has
    none → None. Month periods are normalised to the month-end date."""
    import calendar
    import re
    uri = os.environ.get("MI_AGENT_PLATFORM_URI") or ""
    for seg in uri.replace("blob://", "").split("/"):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", seg):
            return seg
        m = re.match(r"^(\d{4})-(\d{2})$", seg)      # YYYY-MM → month-end
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12:
                return f"{y:04d}-{mo:02d}-{calendar.monthrange(y, mo)[1]:02d}"
    return None


def _scan_any_date_column(sub) -> Optional[str]:
    """Last-resort: the max parseable date in ANY date-like column, so a real
    date in the frame is never reported as null."""
    import pandas as pd
    for col in getattr(sub, "columns", []):
        name = str(col).lower()
        if not ("date" in name or "cut_off" in name or "cutoff" in name
                or name.endswith("_dt")):
            continue
        try:
            rd = pd.to_datetime(sub[col], errors="coerce").dropna()
        except Exception:  # noqa: BLE001
            continue
        if not rd.empty:
            return rd.max().date().isoformat()
    return None


def _platform_reporting_date(sub, run_id: str) -> Optional[str]:
    """Reporting date for a platform (sub)frame, in priority order:
    reporting_date → data_cut_off_date → cut_off_date (via infer_reporting_date),
    then the platform period path, then MI_AGENT_REPORTING_DATE, then any other
    date-like column. Never null when a real date exists in the frame."""
    from_data = snapshots_mod.infer_reporting_date(run_id, sub)
    if from_data:
        return from_data
    from_path = _period_from_platform_uri()
    if from_path:
        return from_path
    env = os.environ.get("MI_AGENT_REPORTING_DATE")
    if env:
        return env
    return _scan_any_date_column(sub)


def _pid_label(sub, pid: str) -> str:
    if "source_portfolio_label" in getattr(sub, "columns", []):
        for v in sub["source_portfolio_label"].dropna():
            cleaned = _clean_provenance_value(v)
            if cleaned:
                return cleaned
    return pid


def _platform_snapshot_index() -> Optional[Dict[str, Any]]:
    """Portfolio/run index derived from the loaded **platform canonical**.

    When ``MI_AGENT_PLATFORM_URI`` is configured the active dataset is the combined
    platform canonical (no on-disk onboarding runs), so the portfolio /
    reporting-date dropdowns are built from the loaded dataframe. Portfolios are
    derived from ``source_portfolio_id`` (so ``direct_001`` is the selectable
    funded portfolio), each with one run at that portfolio's latest reporting
    date. Falls back to a single client entry when the canonical has no
    provenance. Returns ``None`` when the platform canonical is not the active
    source.
    """
    if data_source_kind() != KIND_PLATFORM_CANONICAL:
        return None
    try:
        df = get_dataframe()
    except Exception as exc:  # noqa: BLE001 - discovery must never 500
        logger.warning("platform snapshot index: dataframe load failed: %s", exc)
        return None
    run_id = os.environ.get("MI_AGENT_RUN_ID") or "latest"

    portfolios: List[Dict[str, Any]] = []
    if "source_portfolio_id" in df.columns:
        ids = df["source_portfolio_id"].dropna().astype(str).str.strip()
        distinct = sorted({p for p in ids.unique() if p and p.lower() != "nan"})
        for pid in distinct:
            sub = df[ids == pid]
            portfolios.append({
                "client_id": pid,                    # React selects on client_id
                "label": _pid_label(sub, pid),
                "source_portfolio_id": pid,
                "runs": [{
                    "run_id": run_id,
                    "reporting_date": _platform_reporting_date(sub, run_id),
                    "loan_count": int(len(sub)),
                    "current_outstanding_balance": round(snapshots_mod._balance_sum(sub), 2),
                }],
            })

    if not portfolios:  # no provenance → single client entry (prior behaviour)
        client_id = _platform_client_id(df)
        portfolios = [{
            "client_id": client_id, "label": str(client_id).upper(),
            "runs": [{
                "run_id": run_id,
                "reporting_date": _platform_reporting_date(df, run_id),
                "loan_count": int(len(df)),
                "current_outstanding_balance": round(snapshots_mod._balance_sum(df), 2),
            }],
        }]

    return {"portfolios": portfolios, "source": data_source_label()}


def _blob_platform_index(root: str) -> Optional[Dict[str, Any]]:
    """The dated funded platform-canonical index for a ``blob://`` onboarding
    output root, or None when nothing dated is published under it."""
    try:
        from apps.blob_trigger_app.storage import open_storage
        storage = open_storage()
        return platform_blob_mod.build_index(
            root, storage, label_fn=_pid_label,
            balance_fn=snapshots_mod._balance_sum,
            default_client_id=os.environ.get("MI_AGENT_CLIENT_ID"))
    except Exception as exc:  # noqa: BLE001 - discovery must never 500
        logger.warning("blob platform snapshot index failed for %s: %s", root, exc)
        return None


def _blob_funded_evolution(root: str, cid: str, trid: Optional[str]) -> Dict[str, Any]:
    """Funded evolution over the dated platform canonicals under a ``blob://`` root.

    Uses the SOURCE PORTFOLIO id (e.g. ``direct_001``) — not the selected run — and
    aggregates ALL dated cuts for it (truncated to ``trid`` when that is a date).
    ``total`` / a type lens aggregates across the matching source portfolios. Never
    collapses to the currently-selected run."""
    from apps.blob_trigger_app.storage import open_storage
    from .funded_prep import prepare_funded_mi_dataset
    frames = platform_blob_mod.build_funded_evolution_frames(
        root, open_storage(), cid, trid, prepare_funded_mi_dataset)
    result = evolution_mod.assemble_funded_evolution(
        frames, cid, trid,
        lineage={
            "source": "governed dated platform canonicals (platform_canonical_typed.csv)",
            "metric": "funded book actuals per reporting cut",
            "note": "One period per dated platform canonical for the selected "
                    "source portfolio / lens; no cross-run merge.",
        })
    return result


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


def _resolve_run_dataframe(client_id: str, run_id: str, root: Optional[str]):
    """``(df, prep_report)`` for a specific run, preferring on-disk discovery and
    falling back to the active env-configured dataframe for the active run."""
    # A dated cut under a blob:// platform root: load THAT canonical (scoped to the
    # source portfolio), not the active/latest one — so selecting an earlier month
    # shows that month's data.
    if root and platform_blob_mod.is_blob_root(root):
        try:
            from apps.blob_trigger_app.storage import open_storage
            raw = platform_blob_mod.resolve_run_frame(
                root, open_storage(), client_id, run_id)
            if raw is not None and not raw.empty:
                from .funded_prep import prepare_funded_mi_dataset
                return prepare_funded_mi_dataset(raw)
        except Exception as exc:  # noqa: BLE001 - fall back to active source
            logger.warning("blob platform run resolution failed for %s/%s: %s",
                           client_id, run_id, exc)
    if root and not platform_blob_mod.is_blob_root(root):
        tape = snapshots_mod.resolve_tape_path(root, client_id, run_id)
        if tape is not None:
            return snapshots_mod.load_prepared_run(tape)
    # Fall back to the active data source if it matches the requested run.
    info = data_source_info()
    if info.get("client_id") == client_id and info.get("run_id") == run_id:
        return get_dataframe(), info
    # Platform canonical: the combined dataset IS the run. Serve it for the
    # synthesized portfolio/run from _platform_snapshot_index(); when the requested
    # id is a source_portfolio_id present in the canonical, scope to that book.
    if data_source_kind() == KIND_PLATFORM_CANONICAL:
        df = get_dataframe()
        if client_id and "source_portfolio_id" in df.columns:
            ids = df["source_portfolio_id"].astype(str).str.strip()
            if (ids == client_id).any():
                df = df[ids == client_id]
        return df, info
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


#: A dated published pipeline snapshot under a ``blob://`` root:
#: ``…/pipeline/{client}/{YYYY-MM-DD}/pipeline_snapshot.csv``. The ``latest/``
#: pointer folder is excluded because ``latest`` is not a ``YYYY-MM-DD`` date.
_BLOB_DATED_SNAPSHOT_RE = re.compile(
    r"/(?P<date>\d{4}-\d{2}-\d{2})/pipeline_snapshot\.csv$")


def _blob_dated_snapshots(root: str, storage) -> List[Dict[str, str]]:
    """List the DATED published pipeline snapshots under a ``blob://`` root, using
    the storage abstraction (same helper that downloads MI_AGENT_PIPELINE_URI).

    Includes only ``{YYYY-MM-DD}/pipeline_snapshot.csv`` blobs, EXCLUDES the
    ``latest/`` pointer, and returns ``[{date, uri}]`` sorted chronologically. A
    non-blob root, or any listing error, yields ``[]`` (the caller then falls back
    to unchanged filesystem discovery)."""
    if not str(root).startswith("blob://"):
        return []
    try:
        uris = storage.list(root)
    except Exception as exc:  # noqa: BLE001 - discovery must never 500
        logger.warning("blob pipeline listing failed for %s: %s", root, exc)
        return []
    dated: List[Dict[str, str]] = []
    for uri in uris:
        if "/latest/" in uri:
            continue  # the latest/ pointer is never a dated historical source
        m = _BLOB_DATED_SNAPSHOT_RE.search(uri)
        if m:
            dated.append({"date": m.group("date"), "uri": uri})
    dated.sort(key=lambda d: d["date"])
    return dated


#: Local mirror of the blob dated snapshots, keyed by root and content signature
#: (sorted uri:etag) so we only re-download when a snapshot is added/republished.
_PIPELINE_MIRROR_CACHE: Dict[str, Any] = {"root": None, "sig": None, "local": None}


def _materialise_pipeline_root(root: Optional[str]) -> Optional[str]:
    """Return a LOCAL discovery root for ``root``.

    Filesystem roots are returned unchanged (fixtures behave exactly as before).
    A ``blob://`` root is mirrored to a local scratch tree
    (``{scratch}/pipeline_root/{client}/{date}/pipeline_snapshot.csv``) containing
    ONLY the dated snapshots (``latest/`` excluded), so every downstream consumer —
    ``/mi/pipeline/snapshots``, ``/mi/evolution/pipeline`` and the historical model
    — discovers the SAME set of dated sources through the existing filesystem
    discovery. etag-cached, so repeated requests do not re-download."""
    if not root or not str(root).startswith("blob://"):
        return root
    try:
        from pathlib import Path as _Path
        from apps.blob_trigger_app.storage import open_storage, split_blob_uri
        storage = open_storage()
        dated = _blob_dated_snapshots(root, storage)
        if not dated:
            return root  # nothing dated to mirror; blob discovery yields []
        sig = ";".join(f"{d['uri']}:{storage.etag(d['uri']) or ''}" for d in dated)
        cache = _PIPELINE_MIRROR_CACHE
        if (cache.get("root") == root and cache.get("sig") == sig
                and cache.get("local") and _Path(cache["local"]).exists()):
            return cache["local"]
        scratch = os.environ.get("MI_AGENT_SCRATCH", "/tmp/trakt/mi_platform")
        base = _Path(scratch) / "pipeline_root"
        _container, key = split_blob_uri(root)
        prefix = key.rstrip("/")
        for d in dated:
            # Preserve the {client}/{date}/pipeline_snapshot.csv tail below the root
            # prefix so folder-date + client inference resolve on the local mirror.
            _c, ukey = split_blob_uri(d["uri"])
            rel = ukey[len(prefix):].lstrip("/") if ukey.startswith(prefix) else \
                f"{d['date']}/pipeline_snapshot.csv"
            dest = base / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            storage.download_file(d["uri"], dest)
        cache.update(root=root, sig=sig, local=str(base))
        return str(base)
    except Exception as exc:  # noqa: BLE001 - never break discovery on mirror errors
        logger.warning("pipeline blob mirror failed for %s: %s", root, exc)
        return root


def _pipeline_discovery_root() -> Optional[str]:
    """The pipeline root to run governed discovery/evolution/history against —
    filesystem unchanged, ``blob://`` mirrored locally so all consumers share the
    same dated snapshot set."""
    return _materialise_pipeline_root(
        os.environ.get("MI_AGENT_PIPELINE_ROOT") or _pipeline_root())


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


#: etag-cached local copy of the blob pipeline snapshot (avoid re-download when
#: unchanged; re-download when a new weekly run republishes it).
_PIPELINE_URI_CACHE: Dict[str, Any] = {"etag": None, "path": None}


def _resolve_pipeline_uri_local() -> Optional[str]:
    """Resolve MI_AGENT_PIPELINE_URI (the durable weekly pipeline snapshot pointer,
    CSV, or ``latest/`` dir) to a LOCAL CSV path, etag-cached so a re-published
    weekly extract renders on the next request without a restart. ``None`` when
    unset/absent — filesystem resolution below is then unchanged."""
    uri = os.environ.get("MI_AGENT_PIPELINE_URI")
    if not uri:
        return None
    try:
        import json as _json
        from pathlib import Path as _Path
        from apps.blob_trigger_app.storage import open_storage
        storage = open_storage()
        csv_uri = uri
        if uri.endswith(".json"):
            ptr = _json.loads(storage.read_text(uri))
            csv_uri = ptr.get("blob_name") or ptr.get("source_file")
        elif not uri.endswith(".csv"):
            csv_uri = f"{uri.rstrip('/')}/pipeline_snapshot.csv"
        if not csv_uri or not storage.exists(csv_uri):
            return None
        et = storage.etag(csv_uri)
        cached = _PIPELINE_URI_CACHE
        if (et and et == cached.get("etag") and cached.get("path")
                and _Path(cached["path"]).exists()):
            return cached["path"]
        local = storage._local_path(csv_uri)
        if _Path(str(local)).exists():
            dest = str(local)
        else:
            scratch = os.environ.get("MI_AGENT_SCRATCH", "/tmp/trakt/mi_platform")
            dest = str(storage.download_file(csv_uri, _Path(scratch) / "pipeline_snapshot.csv"))
        _PIPELINE_URI_CACHE.update(etag=et, path=dest)
        return dest
    except Exception as exc:  # noqa: BLE001 — never 500 pipeline resolution
        logger.warning("pipeline blob resolution failed for %s: %s", uri, exc)
        return None


def _latest_pipeline_extract_date(client_id: str) -> Optional[str]:
    """The latest available weekly pipeline extract date for ``client_id`` from
    governed discovery (the max dated snapshot). Used to recover the real as-of
    date when the source was resolved via the ``latest/`` pointer (whose path
    carries no date), so the pipeline is disclosed as of its true extract date."""
    root = _pipeline_discovery_root()
    if not root:
        return None
    try:
        srcs = pipeline_mod.discover_pipeline_sources(root, client_id=client_id)
    except Exception:  # noqa: BLE001 - discovery must never break resolution
        return None
    dates = [s.get("pipeline_as_of_date") or s.get("pipeline_extract_date")
             for s in srcs]
    dates = [d for d in dates if d]
    return max(dates) if dates else None


def _weekly_files_window(client_id: str, as_of: Optional[str]) -> list:
    """The governed weekly-extract window (every unique dated extract up to and
    including ``as_of``) for ``client_id``, from the SAME discovery the evolution
    and history endpoints use — including a ``blob://`` root's dated snapshots.

    Used to attach ``weekly_files`` to a source resolved via the ``latest/``
    blob pointer (whose single CSV carries no prior-week history), so week-on-week
    tile deltas can select and aggregate the real prior extract. Returns ``[]``
    when there is no discovery root or fewer than two dated extracts.
    """
    root = _pipeline_discovery_root()
    if not root:
        return []
    try:
        inv = pipeline_mod.weekly_extract_inventory(root, client_id)
    except Exception:  # noqa: BLE001 - discovery must never break resolution
        return []
    extracts = inv.get("extracts", []) or []
    if as_of:
        extracts = [e for e in extracts
                    if (e.get("pipeline_extract_date") or "") <= as_of]
    return extracts


def _resolve_pipeline_source(client_id: str, run_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The governed pipeline scope for a client/run (blob URI, explicit env, or
    discovery). Returns a scope dict with the separated date concepts (folder /
    extract / as-of), never a single ambiguous reporting date.

    The pipeline scope ALWAYS reflects the LATEST available weekly extract — the
    funded ``run_id`` never truncates it (funded actuals may lag the pipeline).
    """
    # Durable blob pipeline snapshot (production) wins, then an explicit local file.
    explicit = _resolve_pipeline_uri_local() or os.environ.get("MI_AGENT_PIPELINE_SOURCE")
    if explicit:
        from pathlib import Path as _Path
        p = _Path(explicit)
        if p.exists():
            folder_date = pipeline_mod._folder_date(p.parent)
            extract_date = pipeline_mod._extract_date(p)
            # The latest/ pointer carries no date in its path; recover the true
            # extract date from discovery so the pipeline as-of is not lost/None.
            as_of = extract_date or folder_date or _latest_pipeline_extract_date(client_id)
            return {"client_id": client_id, "source_file": str(p),
                    "run_id": run_id or pipeline_mod._run_id_for(folder_date, extract_date, p),
                    "pipeline_source_folder": str(p.parent),
                    "pipeline_source_folder_date": folder_date,
                    "pipeline_extract_date": extract_date or as_of,
                    "pipeline_as_of_date": as_of,
                    "current_pipeline_snapshot_date": as_of,
                    "current_pipeline_source_file": p.name,
                    # The latest/ pointer is a single CSV with no prior-week
                    # history; attach the governed dated-extract window so the
                    # week-on-week tile deltas can select the real prior extract.
                    "weekly_files": _weekly_files_window(client_id, as_of)}
    root = _pipeline_discovery_root()
    if root:
        return pipeline_mod.resolve_pipeline_source(root, client_id, run_id)
    return None


def _pipeline_history(client_id: str) -> Optional[Dict[str, Any]]:
    """The historical completion-rate model from a client's weekly pipeline files.

    Built from the SAME discovered dated sources as ``/mi/pipeline/snapshots`` and
    ``/mi/evolution/pipeline`` — including a ``blob://`` root's dated snapshots
    (the ``MI_AGENT_PIPELINE_URI`` latest pointer is only the current snapshot; it
    does NOT suppress the multi-week history when the root holds several dated
    snapshots). Returns None for a single explicit local source, no discovery root,
    or when fewer than two weekly extracts exist (no multi-week history to model)."""
    if os.environ.get("MI_AGENT_PIPELINE_SOURCE"):
        return None  # single explicit local source → no multi-week history model
    root = _pipeline_discovery_root()
    if not root:
        return None
    try:
        model = pipeline_mod.build_pipeline_history(root, client_id)
    except Exception as exc:  # noqa: BLE001 - history is additive; never 500
        logger.warning("pipeline history build failed for %s: %s", client_id, exc)
        return None
    if int((model or {}).get("uniqueWeeklyExtractsUsed", 0)) < 2:
        return None  # a single dated snapshot is not a multi-week history
    return model


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
    result = pipeline_mod.compute_pipeline_snapshot(
        df, report, semantics, client_id=source.get("client_id", client_id),
        run_id=run_id or source.get("run_id", ""), source=source, prior_week=prior_week)
    # Disclose the funded-vs-pipeline timing (never truncate): funded anchor = the
    # selected run's reporting date; pipeline anchor = the latest weekly extract.
    result["pipelineTiming"] = timing_mod.timing_disclosure(
        _funded_date_from_run(run_id), result.get("pipelineAsOfDate"))
    return result


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
    # Both anchors + non-blocking timing disclosure (funded actuals vs latest
    # pipeline). The forecast bridge composes funded actuals with the LATEST
    # pipeline; when the pipeline extract is later than the funded cut we disclose
    # it rather than hide the pipeline.
    pipeline_as_of = ((source or {}).get("pipeline_as_of_date")
                      or _latest_pipeline_extract_date(client_id))
    envelope["pipelineTiming"] = timing_mod.timing_disclosure(
        funded_reporting_date or _funded_date_from_run(run_id), pipeline_as_of)
    return envelope


def _funded_date_from_run(run_id: Optional[str]) -> Optional[str]:
    """The funded reporting date implied by a selected run id: a ``YYYY-MM-DD``
    run IS the date; an ``mi_YYYY_MM`` run maps to that month-end; otherwise None."""
    import calendar
    if not run_id:
        return None
    s = str(run_id)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    m = re.fullmatch(r"mi_(\d{4})_(\d{2})", s) or re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return f"{y:04d}-{mo:02d}-{calendar.monthrange(y, mo)[1]:02d}"
    return None


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
        if platform_blob_mod.is_blob_root(root):
            return _blob_funded_evolution(root, cid, trid)
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
    cid, _funded_trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    # The pipeline is a continuous weekly operational view — the selected FUNDED
    # reporting date (carried in portfolioId) must NOT truncate it. Only an EXPLICIT
    # pipeline toRunId query param caps it (rare; no UI toggle needed by default).
    pipeline_cut = toRunId or to_run_id
    root = _pipeline_discovery_root()
    if not root:
        return {"dataset": "pipeline", "portfolioId": cid, "toRunId": pipeline_cut,
                "periods": [], "byStage": [], "singlePeriod": True,
                "error": "no pipeline root configured"}
    try:
        result = evolution_mod.pipeline_evolution(root, cid, pipeline_cut)
        # Disclose the funded-vs-pipeline timing on the evolution response too, so
        # the pipeline evolution view can surface the non-blocking banner.
        latest = None
        dates = [p.get("extract_date") for p in result.get("periods", [])]
        dates = [d for d in dates if d]
        if dates:
            latest = max(dates)
        result["pipelineTiming"] = timing_mod.timing_disclosure(
            _funded_date_from_run(_funded_trid), latest)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("pipeline evolution failed: %s", exc)
        return {"dataset": "pipeline", "portfolioId": cid, "toRunId": pipeline_cut,
                "periods": [], "byStage": [], "singlePeriod": True, "error": str(exc)}


@app.get("/mi/evolution/funnel")
def funnel_evolution(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                     toRunId: Optional[str] = None, to_run_id: Optional[str] = None
                     ) -> Dict[str, Any]:
    """Weekly origination funnel trends (KFI / Application / Offer / Completion
    value + count, 5-week average, latest week, delta vs prior week). Never 500s."""
    cid, _funded_trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    # Origination funnel is weekly-pipeline data — the funded reporting date must
    # NOT truncate it; only an explicit pipeline toRunId caps it.
    pipeline_cut = toRunId or to_run_id
    root = _pipeline_discovery_root()
    if not root:
        return {"dataset": "pipeline_funnel", "portfolioId": cid, "toRunId": pipeline_cut,
                "stages": [], "weeks": [], "series": {}, "summary": {},
                "singlePeriod": True, "error": "no pipeline root configured"}
    try:
        return evolution_mod.pipeline_funnel_evolution(root, cid, pipeline_cut)
    except Exception as exc:  # noqa: BLE001
        logger.warning("funnel evolution failed: %s", exc)
        return {"dataset": "pipeline_funnel", "portfolioId": cid, "toRunId": pipeline_cut,
                "stages": [], "weeks": [], "series": {}, "summary": {},
                "singlePeriod": True, "error": str(exc)}


@app.get("/mi/cohorts")
def cohorts(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
            runId: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
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
        return cohorts_mod.cohort_analysis(
            df, client_id=client_id, portfolio_id=pid, reporting_date=reporting_date)
    except Exception as exc:  # noqa: BLE001 - cohort analysis must never 500
        logger.warning("cohort analysis failed for %s: %s", pid, exc)
        return {"dataset": "cohorts", "portfolioId": pid, "available": False,
                "reason": str(exc), "cohorts": [], "metricsAvailable": []}


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
def download_deck(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                  period: Optional[str] = None):
    """Serve an investor PPTX deck (the latest, or a specific reporting period).

    Returns the .pptx bytes as an attachment with a friendly filename. 404 (JSON)
    when the requested deck does not exist, so the UI can disable the action.
    """
    cid, _trid = _evo_ids(portfolioId, client_id, None, None)
    try:
        resolved = decks_mod.resolve_deck_local(cid, period)
    except Exception as exc:  # noqa: BLE001
        logger.warning("deck download failed for %s: %s", cid, exc)
        resolved = None
    if resolved is None:
        which = period or "latest"
        return JSONResponse(
            status_code=404,
            content={"ok": False,
                     "error": f"No investor deck available for {cid} ({which})."})
    path, filename = resolved
    return FileResponse(str(path), media_type=_PPTX_MEDIA_TYPE, filename=filename)


@app.get("/mi/evolution/forecast")
def forecast_evolution(portfolioId: Optional[str] = None, client_id: Optional[str] = None,
                       toRunId: Optional[str] = None, to_run_id: Optional[str] = None
                       ) -> Dict[str, Any]:
    """Forecast bridge over time (funded balance + weighted pipeline per run)."""
    cid, trid = _evo_ids(portfolioId, client_id, toRunId, to_run_id)
    root = _onboarding_output_root()
    proot = _pipeline_discovery_root()
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
    proot = _pipeline_discovery_root()
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
    proot = _pipeline_discovery_root()
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

    # ---- governed-intent routing (compare / evolution / forecast / risk) ----
    # The new analytical intents are served by the internal evolution /
    # temporal-compare / forecast-extrapolation / risk-limit services and shaped
    # into the existing artifact union. Normal point-in-time questions return
    # None here and fall through to the unchanged MI Agent path below.
    try:
        cid, _rid = (portfolio_id.split("/", 1) + [None])[:2] if (portfolio_id and "/" in portfolio_id) \
            else ((portfolio_id or "client_001"), None)
        routed = chat_routing_mod.try_route(
            req.question, portfolio_id=portfolio_id, view=view,
            output_root=_onboarding_output_root(),
            pipeline_root=_pipeline_discovery_root(),
            semantics=load_mi_semantics(semantics_path()),
            history_model=_pipeline_history(cid), as_of=req.asOfDate)
    except Exception as exc:  # noqa: BLE001 - routing must never break the chat
        logger.warning("chat routing failed; using point-in-time path: %s", exc)
        routed = None
    if routed is not None:
        meta = routed.setdefault("metadata", {})
        if isinstance(meta, dict):
            meta["datasetContext"] = view
        return routed

    try:
        df, frame_error = _resolve_query_frame(view, portfolio_id)
    except FileNotFoundError as exc:
        return _error(str(exc))
    if frame_error:
        return _error(frame_error)

    # LLM parser wiring: governed by ENABLE_LLM_MI_AGENT / ANTHROPIC_API_KEY
    # (see mi_agent_config). When unavailable (disabled, no key, no package) the
    # deterministic parser runs alone — and the response says so.
    llm_cfg = get_llm_config()
    workflow = run_mi_agent_query(
        req.question, df, str(semantics_path()),
        parser_mode=("llm" if llm_cfg.available else "deterministic"),
        llm_enabled=llm_cfg.available,
        model=llm_cfg.model,
        max_repair_attempts=llm_cfg.max_repair_attempts,
        catalog_mode=llm_cfg.catalog_mode,
        zero_cost_first=llm_cfg.zero_cost_first,
        extra_filters=req.filters or None,
        source_portfolio_lens=req.sourcePortfolioLens or None)
    result = adapt_workflow_result(workflow, portfolio_id=portfolio_id, as_of=req.asOfDate)
    # Surface which dataset/view answered (funded | pipeline | forecast) and the
    # active source-portfolio lens (total | direct | acquired | cohort).
    meta = result.setdefault("metadata", {}) if isinstance(result, dict) else {}
    if isinstance(meta, dict):
        meta["datasetContext"] = view
        meta["llm"] = {"enabled": llm_cfg.enabled, "available": llm_cfg.available,
                       "model": llm_cfg.model if llm_cfg.available else None,
                       "status": llm_cfg.status}
        if workflow.get("portfolio_lens"):
            meta["portfolioLens"] = workflow["portfolio_lens"]
    # An LLM that was requested but is unusable (missing key / package) is a
    # configuration fault the operator must see, not a silent downgrade.
    if llm_cfg.enabled and not llm_cfg.available and isinstance(result, dict):
        result.setdefault("warnings", []).extend(llm_cfg.warnings)
    return result
