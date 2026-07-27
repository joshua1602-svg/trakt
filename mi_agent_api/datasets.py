"""mi_agent_api/datasets.py — interface-neutral dataset resolution.

The single place that answers "which governed dataset answers this question, and
what is it?". Everything here was previously a private helper inside
``mi_agent_api.app``, which meant the governed capability had to import the
FastAPI module to find its data — an inverted dependency that also forced
``mi_agent_pptx.mi_api`` to re-implement seven of these helpers and to mutate
``os.environ`` to select a run.

This module imports **no web framework**. The dependency direction is now:

    FastAPI adapter ─┐
    Copilot adapter ─┼──► governed capability ──► datasets (this module) ──► storage
    Python caller  ──┘

Two entry points matter to callers:

  * :func:`describe_active_dataset` — typed :class:`DatasetDescriptor` metadata
    (tenant, source base/kind, reporting date, snapshot id, content fingerprint,
    row count) used for the source-approval decision and for answer provenance.
  * :func:`resolve_authorised_frame` — the dataframe for an
    :class:`~trakt_core.tenancy.AuthorisedPortfolio`. It takes the authorisation
    token rather than raw strings, so data cannot be loaded for a portfolio that
    has not been through :func:`~trakt_core.tenancy.authorise_portfolio_access`.

The pre-existing helpers keep their names and behaviour and are re-exported from
``mi_agent_api.app`` for backward compatibility.
"""

from __future__ import annotations

import calendar
import hashlib
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from mi_agent.mi_query_validator import load_mi_semantics

from . import currency as currency_mod
from . import evolution as evolution_mod
from . import pipeline_contract as pipeline_mod
from . import pipeline_history
from . import platform_snapshots_blob as platform_blob_mod
from . import snapshots as snapshots_mod
from . import workspace as workspace_mod
from .data_source import (
    KIND_PLATFORM_CANONICAL,
    KIND_UNAVAILABLE,
    data_source_info,
    data_source_kind,
    data_source_label,
    get_dataframe,
    resolve_data_source,
    semantics_path,
)

logger = logging.getLogger("mi_agent_api.datasets")


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
    """CLIENT / reporting-run index derived from the loaded **platform canonical**.

    This is the TENANT axis, and only the tenant axis. It previously emitted one
    entry per ``source_portfolio_id``, which put ``direct_001`` / ``acquired_001``
    into the top-level Client selector — two controls then described the same
    thing, and a client of ``direct_001`` with a portfolio of ``Total`` was a
    reachable, self-contradictory state.

    The client is deployment configuration (``dependencies.default_tenant_id``),
    never a row value from the tape. Source portfolios are the PORTFOLIO axis and
    are served by ``/mi/portfolio-context``; they must not appear here.

    Runs are the reporting dates the platform canonical actually carries, so the
    reporting-date control stays data-driven — and selecting one changes only the
    date, never the portfolio.
    """
    if data_source_kind() != KIND_PLATFORM_CANONICAL:
        return None
    try:
        df = get_dataframe()
    except Exception as exc:  # noqa: BLE001 - discovery must never 500
        logger.warning("platform snapshot index: dataframe load failed: %s", exc)
        return None

    from .dependencies import default_tenant_id
    try:
        client_id = default_tenant_id()
    except Exception:  # noqa: BLE001 - fall back to the dataset's own hint
        client_id = _platform_client_id(df)

    default_run = os.environ.get("MI_AGENT_RUN_ID") or "latest"
    runs = _platform_runs(df, default_run)
    return {
        "portfolios": [{
            "client_id": client_id,
            "label": str(client_id).upper(),
            "runs": runs,
        }],
        "source": data_source_label(),
    }


def _platform_runs(df, default_run: str) -> List[Dict[str, Any]]:
    """One run per reporting date present in the platform canonical (oldest first).

    A combined tape may carry several cut-off dates (books cut on different
    dates), so the reporting-date control lists what the data actually holds. The
    counts here are the WHOLE client at that date — portfolio scoping is applied
    downstream from the governed portfolio context, never by this index.
    """
    date_col = next((c for c in ("reporting_date", "data_cut_off_date", "cut_off_date")
                     if c in getattr(df, "columns", [])), None)
    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        as_iso = dates.dt.date.astype("string")
        distinct = sorted(as_iso.dropna().unique().tolist())
        if distinct:
            return [{
                "run_id": value,
                "reporting_date": value,
                "loan_count": int((as_iso == value).sum()),
                "current_outstanding_balance": round(
                    snapshots_mod._balance_sum(df[as_iso == value]), 2),
            } for value in distinct]
    return [{
        "run_id": default_run,
        "reporting_date": _platform_reporting_date(df, default_run),
        "loan_count": int(len(df)),
        "current_outstanding_balance": round(snapshots_mod._balance_sum(df), 2),
    }]


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


def _blob_funded_evolution(root: str, cid: str, trid: Optional[str],
                           scope=None) -> Dict[str, Any]:
    """Funded evolution over the dated platform canonicals under a ``blob://`` root.

    Uses the SOURCE PORTFOLIO id (e.g. ``direct_001``) — not the selected run — and
    aggregates ALL dated cuts for it (truncated to ``trid`` when that is a date).
    ``total`` / a type lens aggregates across the matching source portfolios. Never
    collapses to the currently-selected run."""
    from apps.blob_trigger_app.storage import open_storage
    from .funded_prep import prepare_funded_mi_dataset
    frames = platform_blob_mod.build_funded_evolution_frames(
        root, open_storage(), cid, trid, prepare_funded_mi_dataset)
    # One governed scope filter, applied to every period frame.
    frames = evolution_mod._apply_scope_to_frames(frames, scope)
    result = evolution_mod.assemble_funded_evolution(
        frames, cid, trid,
        lineage={
            "source": "governed dated platform canonicals (platform_canonical_typed.csv)",
            "metric": "funded book actuals per reporting cut",
            "note": "One period per dated platform canonical for the selected "
                    "source portfolio / lens; no cross-run merge.",
        })
    return result


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
    # Platform canonical: the combined dataset IS the run for the client.
    #
    # This deliberately does NOT narrow by treating ``client_id`` as a
    # source_portfolio_id. That legacy shortcut was how a Client selection could
    # silently narrow the book to one portfolio while the Portfolio selector
    # still read "Total". Portfolio scope has exactly one owner — the governed
    # portfolio context — and it is applied by the routes, above this resolver.
    #
    # The run is honoured where the tape carries reporting dates, so choosing a
    # reporting date changes the date and nothing else.
    if data_source_kind() == KIND_PLATFORM_CANONICAL:
        return _platform_run_frame(get_dataframe(), run_id), info
    return None, None


def _platform_run_frame(df, run_id: Optional[str]):
    """The platform canonical narrowed to one reporting date, when it is dated.

    ``latest`` (or an unrecognised run) returns the whole tape, which is the
    prior behaviour for a single-cut platform.
    """
    if df is None or not run_id or str(run_id).lower() == "latest":
        return df
    date_col = next((c for c in ("reporting_date", "data_cut_off_date", "cut_off_date")
                     if c in getattr(df, "columns", [])), None)
    if date_col is None:
        return df
    dates = pd.to_datetime(df[date_col], errors="coerce")
    match = dates.dt.date.astype(str) == str(run_id)
    return df[match] if match.any() else df


#: A trailing dated (or ``latest``) folder in a pipeline snapshot pointer.
_PIPELINE_URI_TAIL_RE = re.compile(r"^(?:\d{4}-\d{2}-\d{2}|latest)$", re.IGNORECASE)


def _pipeline_root_from_uri() -> Optional[str]:
    """Derive a pipeline DISCOVERY ROOT from ``MI_AGENT_PIPELINE_URI`` (the weekly
    snapshot pointer) when ``MI_AGENT_PIPELINE_ROOT`` is not set.

    The URI points at a SINGLE snapshot (``…/{date|latest}/pipeline_snapshot.csv``,
    a ``.json`` pointer, or a ``latest/`` dir). Discovery/evolution/funnel need the
    CONTAINING root so they can enumerate ALL dated weekly cuts, not just one — so
    strip the filename and a trailing ``{date}``/``latest`` folder to reach it."""
    uri = os.environ.get("MI_AGENT_PIPELINE_URI")
    if not uri:
        return None
    path = uri.rstrip("/")
    if path.endswith(".csv") or path.endswith(".json"):
        path = path.rsplit("/", 1)[0]
    last = path.rsplit("/", 1)[-1]
    if _PIPELINE_URI_TAIL_RE.match(last):
        path = path.rsplit("/", 1)[0]
    return path or None


def _pipeline_root() -> Optional[str]:
    """Root to discover governed pipeline sources (18a tape / M2L KFI extracts).

    Precedence: explicit ``MI_AGENT_PIPELINE_ROOT`` → a root DERIVED from the
    weekly ``MI_AGENT_PIPELINE_URI`` pointer → ``MI_AGENT_ONBOARDING_OUTPUT_ROOT``
    → the inferred onboarding root. The URI-derived root comes before the
    onboarding root because the onboarding/platform root holds FUNDED cuts, not
    the weekly pipeline extracts."""
    explicit = os.environ.get("MI_AGENT_PIPELINE_ROOT")
    if explicit:
        return explicit
    derived = _pipeline_root_from_uri()
    if derived:
        return derived
    root = os.environ.get("MI_AGENT_ONBOARDING_OUTPUT_ROOT")
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


def _kfi_lag_weeks_from_model(model: Optional[Dict[str, Any]]) -> Optional[int]:
    """Median KFI->completion lag in whole weeks from an already-built history
    model. Returns None when no timing is available."""
    timing = ((model or {}).get("historicalCompletionTimingByStage") or {}).get("KFI") or {}
    median_days = timing.get("medianDays")
    return max(1, round(float(median_days) / 7.0)) if median_days else None


def _kfi_completion_lag_weeks(client_id: str) -> Optional[int]:
    """Median KFI->completion lag, in whole weeks, from the historical model.
    Convenience wrapper that builds the model; never raises."""
    return _kfi_lag_weeks_from_model(_pipeline_history(client_id))


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
        # Honour the selected reporting run: when the portfolio id carries a
        # run_id, load THAT run's funded book (exactly as /mi/snapshot does)
        # instead of the active/latest dataset. Otherwise an earlier-run
        # selection would be answered from the latest snapshot yet labelled with
        # the selected date — a stale, mislabelled answer. Falls back to the
        # active dataset when no run_id is given or the run cannot be resolved.
        if run_id:
            try:
                run_df, _ = _resolve_run_dataframe(
                    client_id, run_id, _onboarding_output_root())
            except Exception as exc:  # noqa: BLE001 - fall back to active source
                logger.warning("funded run resolution failed for %s/%s: %s",
                               client_id, run_id, exc)
                run_df = None
            if run_df is not None and len(run_df):
                return run_df, None
        return get_dataframe(), None  # active/latest funded dataset

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


def _mi_llm_config() -> SimpleNamespace:
    """LLM-parser configuration for the MI Agent query path.

    Returns an object with ``enabled`` (the parser should attempt the LLM),
    ``available`` (it can actually run — a key is present), ``model``, a
    human-readable ``status``, and any ``warnings``. The LLM is the FALLBACK for
    questions the deterministic parser can't resolve (``zero_cost_first`` keeps
    easy questions free — no LLM call). It is enabled by default whenever an
    ``ANTHROPIC_API_KEY`` is configured; with no key the parser stays
    deterministic-only (never crashes). Operators can force it with
    ``MI_AGENT_LLM_PARSER=on|off|auto`` and override the model with
    ``MI_AGENT_LLM_MODEL``.
    """
    mode = os.environ.get("MI_AGENT_LLM_PARSER", "auto").strip().lower()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if mode in ("off", "0", "false", "no"):
        enabled = False
    elif mode in ("on", "1", "true", "yes"):
        enabled = True
    else:  # auto
        enabled = has_key
    model = os.environ.get("MI_AGENT_LLM_MODEL") or None
    available = bool(enabled and has_key)
    warnings: List[str] = []
    if enabled and not has_key:
        status = "unavailable_no_api_key"
        warnings.append("LLM parser requested but ANTHROPIC_API_KEY is not set; "
                        "using the deterministic parser.")
    elif enabled:
        status = "enabled"
    else:
        status = "disabled"
    return SimpleNamespace(enabled=enabled, model=model, available=available,
                           status=status, warnings=warnings)


_CLIENT_CURRENCY_CACHE: Dict[str, str] = {}


def _apply_request_currency(cid: str, portfolio_id: Optional[str]) -> None:
    """Set the request-scoped display currency for a client (tape -> config ->
    GBP), cached per client. Resolved from the client's funded book, which is
    book-level (one currency), so it covers the routed answers too. Never raises."""
    code = _CLIENT_CURRENCY_CACHE.get(cid)
    if code is None:
        code = "GBP"
        try:
            fdf, ferr = _resolve_query_frame("funded", portfolio_id)
            if fdf is not None and not ferr:
                code = currency_mod.resolve_currency_code(fdf)
        except Exception as exc:  # noqa: BLE001 - currency is presentational
            logger.warning("currency resolution failed for %s: %s", cid, exc)
        _CLIENT_CURRENCY_CACHE[cid] = code
    currency_mod.set_currency(code)


# --------------------------------------------------------------------------- #
# Typed dataset metadata + authorised access
#
# The governance layer needs to know *what* the active dataset is (to decide
# whether it may answer) and *which* dataset a specific authorised portfolio
# maps to. Both are expressed below in terms the capability can consume without
# knowing about blobs, environment variables or file paths.
# --------------------------------------------------------------------------- #

#: Columns scanned, in order, for a dataset-level reporting date.
_REPORTING_DATE_COLUMNS = ("reporting_date", "data_cut_off_date", "cut_off_date")


class DatasetDescriptor(SimpleNamespace):
    """Typed metadata about a resolved dataset.

    ``SimpleNamespace`` subclass rather than a frozen dataclass so the existing
    ``data_source_info()`` dictionary can be carried alongside on ``.info``
    without restating its ~15 keys, while the governed fields stay explicit
    attributes.
    """

    def __init__(self, *, source_base: str, source_kind: str,
                 label: Optional[str] = None, available: bool = True,
                 reporting_date: Optional[str] = None,
                 snapshot_id: Optional[str] = None,
                 content_hash: Optional[str] = None,
                 row_count: Optional[int] = None,
                 source_portfolios: Tuple[str, ...] = (),
                 info: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            source_base=source_base, source_kind=source_kind, label=label,
            available=available, reporting_date=reporting_date,
            snapshot_id=snapshot_id, content_hash=content_hash,
            row_count=row_count, source_portfolios=tuple(source_portfolios),
            info=dict(info or {}))


def _dataset_fingerprint(path: Optional[str]) -> Optional[str]:
    """A cheap, stable content fingerprint for the resolved dataset file.

    Size + mtime + name hashed, not a full content digest: the file is a loan
    tape and hashing it on every request would be wasteful. It is sufficient to
    identify "the same published cut" for provenance and cache invalidation.
    """
    if not path:
        return None
    try:
        st = Path(path).stat()
        seed = f"{Path(path).name}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return None
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _frame_reporting_date(df) -> Optional[str]:
    """The dataset-level reporting date, from the first populated date column."""
    if df is None:
        return None
    for col in _REPORTING_DATE_COLUMNS:
        if col in getattr(df, "columns", []):
            vals = df[col].dropna()
            if not vals.empty:
                return str(vals.iloc[0])[:10]
    return None


def _frame_source_portfolios(df) -> Tuple[str, ...]:
    if df is None or "source_portfolio_id" not in getattr(df, "columns", []):
        return ()
    ids = df["source_portfolio_id"].dropna().astype(str).str.strip()
    return tuple(sorted({i for i in ids.unique() if i and i.lower() != "nan"}))


def describe_active_dataset() -> DatasetDescriptor:
    """Describe the dataset this deployment would answer from.

    Never raises: an unresolvable source yields ``available=False`` with the
    ``unavailable`` base, which the source-approval policy turns into a governed
    ``DATA_SOURCE_UNAVAILABLE``.
    """
    try:
        path, base = resolve_data_source()
    except Exception as exc:  # noqa: BLE001 - resolution must not raise upward
        logger.warning("data source resolution failed: %s", exc)
        return DatasetDescriptor(source_base="unavailable",
                                 source_kind=KIND_UNAVAILABLE, available=False)
    if path is None:
        return DatasetDescriptor(source_base="unavailable",
                                 source_kind=KIND_UNAVAILABLE, available=False)

    info: Dict[str, Any] = {}
    df = None
    try:
        info = data_source_info()
        df = get_dataframe()
    except Exception as exc:  # noqa: BLE001 - describe what we can
        logger.warning("active dataset load failed: %s", exc)

    fingerprint = _dataset_fingerprint(str(path))
    reporting_date = _frame_reporting_date(df)
    # The snapshot id identifies the published cut: the dataset name plus its
    # fingerprint. Stable across requests, changes when a new cut is published.
    snapshot_id = None
    if fingerprint:
        snapshot_id = f"{Path(str(path)).stem}@{fingerprint.split(':', 1)[1][:12]}"
    return DatasetDescriptor(
        source_base=base,
        source_kind=info.get("kind") or data_source_kind(),
        label=info.get("label") or Path(str(path)).name,
        available=df is not None,
        reporting_date=reporting_date,
        snapshot_id=snapshot_id,
        content_hash=fingerprint,
        row_count=int(len(df)) if df is not None else None,
        source_portfolios=_frame_source_portfolios(df),
        info=info,
    )


def resolve_authorised_frame(authorised, view: str = "funded"):
    """``(df, error)`` for an :class:`~trakt_core.tenancy.AuthorisedPortfolio`.

    Requiring the authorisation token — rather than a ``portfolio_id`` string —
    is what makes "no data is loaded before authorisation" checkable: there is no
    code path from a raw request field to a dataframe.
    """
    return _resolve_query_frame(view, authorised.portfolio_id)


def dataset_snapshot_for(authorised) -> DatasetDescriptor:
    """The dataset descriptor for an authorised portfolio.

    Currently the deployment serves one governed dataset per tenant, so this is
    the active dataset narrowed by the portfolio's own reporting run. Kept as a
    distinct function so a future per-portfolio store changes only this body.
    """
    descriptor = describe_active_dataset()
    if authorised.run_id:
        descriptor.reporting_date = descriptor.reporting_date or authorised.run_id
    return descriptor
