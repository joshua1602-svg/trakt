"""Blob-backed funded platform snapshot index.

The funded analogue of the pipeline blob-root discovery. In production the
managed pipeline publishes a dated platform canonical per funded reporting cut:

    platform/{client}/{YYYY-MM-DD}/platform_canonical_typed.csv   (dated cuts)
    platform/{client}/latest/platform_canonical_typed.csv         (current pointer)

``MI_AGENT_ONBOARDING_OUTPUT_ROOT`` may be such a ``blob://`` root. The on-disk
``snapshots.discover_snapshots`` walk is filesystem-only (and keyed on the
onboarding 18_ tape layout), so it cannot enumerate these — leaving
``/mi/snapshots`` empty. This module lists the dated platform canonicals via the
storage abstraction and builds the SAME ``{portfolios:[{…, runs:[…]}]}`` index
the loaded-canonical path produces — ONE client entry (the tenant) whose runs
are the dated cuts, each run covering the WHOLE canonical (all source
portfolios combined).

The client is the TENANT axis only. Source portfolios (``direct_001`` /
``acquired_001``…) are the PORTFOLIO axis, served by ``/mi/portfolio-context``
and applied by the routes via the governed scope resolver — mirroring the
loaded-canonical index, which was reworked for exactly this reason: per-source
client entries made ``Client: acquired_001 · Scope: Total`` a reachable,
self-contradictory state, and left no selectable combined book.

Read-through caching keeps repeated dropdown calls cheap; a re-published dated
canonical (etag change) is picked up automatically.
"""

from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from trakt_core import perf as _perf

logger = logging.getLogger(__name__)

_PLATFORM_CANONICAL_NAME = "platform_canonical_typed.csv"
#: A DATED platform canonical under a blob:// platform root. ``latest/`` is
#: excluded because ``latest`` is not a ``YYYY-MM-DD`` date.
_DATED_RE = re.compile(
    r"/(?P<date>\d{4}-\d{2}-\d{2})/" + re.escape(_PLATFORM_CANONICAL_NAME) + r"$")

#: uri -> (etag, DataFrame). A dated canonical is immutable per etag, so this
#: avoids re-downloading on every /mi/snapshots (or per-run /mi/snapshot) call.
#:
#: BOUNDED (Phase 1A item 5). This was previously an unbounded dict of full
#: DataFrames: every dated canonical ever read stayed resident for the life of
#: the worker, so memory grew with each published cut. It is now an LRU sized to
#: hold one full evolution pass (a year of monthly cuts); overriding the ceiling
#: only changes the hit rate, never the answer, because a miss re-reads.
_READ_CACHE_MAX = max(1, int(os.environ.get("MI_AGENT_CANONICAL_CACHE_MAX", "12")))
_READ_CACHE: "OrderedDict[str, Tuple[Optional[str], pd.DataFrame]]" = OrderedDict()

#: (uri, etag, scope) -> (prepared_df, prep_report). Funded preparation over a
#: full canonical is the expensive half of every evolution / risk /
#: concentration request, and the evolution loader re-ran it for EVERY period on
#: EVERY request (the raw frame above was cached, the prepared one was not).
#:
#: The scope is part of the key because preparation is data-dependent: the
#: percent-vs-fraction LTV heuristic in ``funded_prep._to_ratio`` reads the
#: median of the rows present, so preparing a scoped frame is not equivalent to
#: scoping a prepared one. The existing scope-then-prepare order is preserved
#: exactly; only the result is reused.
_PREPARED_CACHE_MAX = max(1, int(os.environ.get("MI_AGENT_PREPARED_CACHE_MAX", "24")))
_PREPARED_CACHE: "OrderedDict[str, Tuple[pd.DataFrame, Any]]" = OrderedDict()


def _lru_put(cache: "OrderedDict", key: str, value: Any, limit: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def reset_caches() -> None:
    """Drop both memos (tests, and the existing manual refresh path)."""
    _READ_CACHE.clear()
    _PREPARED_CACHE.clear()


def is_blob_root(root: Optional[str]) -> bool:
    return bool(root) and str(root).startswith("blob://")


def list_dated_platform_canonicals(root: str, storage) -> List[Dict[str, str]]:
    """``[{date, uri}]`` for every DATED platform canonical under ``root`` (a
    ``blob://`` platform root), EXCLUDING ``latest/``, sorted chronologically. A
    non-blob root or any listing error yields ``[]``."""
    if not is_blob_root(root):
        return []
    try:
        uris = storage.list(root)
    except Exception:  # noqa: BLE001 - discovery must never 500
        return []
    dated: List[Dict[str, str]] = []
    for uri in uris:
        if "/latest/" in uri:
            continue
        m = _DATED_RE.search(uri)
        if m:
            dated.append({"date": m.group("date"), "uri": uri})
    dated.sort(key=lambda d: d["date"])
    return dated


def _read(uri: str, storage) -> Optional[pd.DataFrame]:
    """Read a platform canonical CSV from ``uri`` (etag-cached). Downloads for a
    Blob backend; reads directly for the filesystem backend."""
    try:
        et = storage.etag(uri)
        cached = _READ_CACHE.get(uri)
        if cached is not None and cached[0] == et:
            _READ_CACHE.move_to_end(uri)
            _perf.cache_event("canonical_raw", _perf.CACHE_HIT)
            return cached[1]
        _perf.cache_event("canonical_raw", _perf.CACHE_MISS)
        from pathlib import Path as _Path
        local = storage._local_path(uri)
        if _Path(str(local)).exists():
            path = str(local)
        else:
            scratch = os.environ.get("MI_AGENT_SCRATCH", "/tmp/trakt/mi_platform")
            # Keep the dated folder in the scratch name so two cuts never collide.
            tail = uri.rstrip("/").split("/")[-2:]  # [date, filename]
            dest = _Path(scratch) / "platform_runs" / "_".join(tail)
            dest.parent.mkdir(parents=True, exist_ok=True)
            path = str(storage.download_file(uri, dest))
        with _perf.counted("file.read.platform_canonical"):
            df = pd.read_csv(path, low_memory=False)
        _lru_put(_READ_CACHE, uri, (et, df), _READ_CACHE_MAX)
        return df
    except Exception:  # noqa: BLE001 - a bad canonical must not break discovery
        return None


def build_index(root: str, storage, *, label_fn=None, balance_fn,
                default_client_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Build ``{"portfolios": [...], "source": root}`` from the dated platform
    canonicals under a ``blob://`` root: ONE tenant client whose runs are the
    dated cuts (oldest → newest), each run covering the whole canonical — all
    source portfolios combined. Portfolio scoping happens via the governed
    portfolio context, never here. Returns ``None`` when there is nothing dated
    to enumerate (the caller then falls back to the loaded-canonical index).

    ``label_fn`` is accepted for call-site stability but no longer used — the
    tenant label is the client id, not a row value from the tape."""
    del label_fn  # tenant axis only; no per-source labelling
    dated = list_dated_platform_canonicals(root, storage)
    if not dated:
        return None
    client_id = default_client_id or _served_client_id()
    runs: List[Dict[str, Any]] = []
    for d in dated:
        df = _read(d["uri"], storage)
        if df is None or df.empty:
            continue
        runs.append({"run_id": d["date"], "reporting_date": d["date"],
                     "loan_count": int(len(df)),
                     "current_outstanding_balance": round(balance_fn(df), 2)})
    if not runs:
        return None
    runs.sort(key=lambda r: (r["reporting_date"] or "", r["run_id"]))
    from . import client_identity
    return {"portfolios": [{"client_id": client_id,
                            "label": client_identity.portfolio_label(client_id),
                            "client_name": client_identity.governed_client_name(client_id),
                            "runs": runs}],
            "source": root}


def _served_client_id() -> str:
    """The tenant this deployment serves, for a blob root that names none.

    This was the literal string ``"platform"``, which is why the dashboard
    header read "Platform" on a blob-triggered run: the index took the raw
    ``MI_AGENT_CLIENT_ID`` and, with it unset, fell through to a placeholder,
    while every other surface resolved a real client via
    ``dependencies.default_tenant_id`` (the env var, else the client embedded in
    ``MI_AGENT_PLATFORM_URI``, else the historical default). One resolver now,
    so discovery and the rest of the API agree on who the client is.
    """
    try:
        from .dependencies import default_tenant_id
        resolved = (default_tenant_id() or "").strip()
        if resolved:
            return resolved
    except Exception as exc:  # noqa: BLE001 - discovery must never 500
        logger.warning("blob platform index: tenant unresolved: %s", exc)
    return "platform"


_TOTAL_SCOPES = {"total", "all", "client_001", "platform", ""}


def _scope_frame(df: pd.DataFrame, scope: Optional[str]) -> Optional[pd.DataFrame]:
    """Scope a full dated canonical to a funded lens:

      * an exact ``source_portfolio_id`` (e.g. ``direct_001``) → that cohort;
      * a ``source_portfolio_type`` (e.g. ``direct`` / ``acquired``) → that book;
      * ``total`` / ``all`` (or an unrecognised scope) → the WHOLE frame (all
        source portfolios aggregated).

    Returns ``None`` only when a recognised cohort/type filter matches no rows."""
    if not scope:
        return df
    low = str(scope).strip().lower()
    if "source_portfolio_id" in df.columns:
        ids = df["source_portfolio_id"].astype(str).str.strip()
        if (ids == scope).any():
            return df[ids == scope]
    if "source_portfolio_type" in df.columns:
        types = df["source_portfolio_type"].astype(str).str.strip().str.lower()
        if (types == low).any():
            return df[types == low]
    if low in _TOTAL_SCOPES:
        return df
    # Unrecognised scope: aggregate all books rather than render nothing.
    return df


def build_funded_evolution_frames(root: str, storage, scope: Optional[str],
                                  to_run_id: Optional[str], prepare_fn
                                  ) -> List[Dict[str, Any]]:
    """Ordered ``[{run_id, reporting_date, df, source}]`` for funded evolution over
    the dated platform canonicals under a ``blob://`` root, scoped to ``scope``
    (source_portfolio_id, type, or total) and truncated to ``to_run_id`` when that
    is a date. Each frame's ``df`` is prepared via ``prepare_fn`` (funded prep) so
    it carries the same derived dimensions as the snapshot path."""
    cut = str(to_run_id) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(to_run_id) or "") else None
    frames: List[Dict[str, Any]] = []
    for d in list_dated_platform_canonicals(root, storage):
        if cut and d["date"] > cut:
            continue
        prepared = _prepared_for(d["uri"], storage, scope, prepare_fn)
        if prepared is None:
            continue
        frames.append({"run_id": d["date"], "reporting_date": d["date"],
                       "df": prepared, "source": d["uri"]})
    return frames


def _prepared_for(uri: str, storage, scope: Optional[str], prepare_fn):
    """The PREPARED funded frame for one dated canonical under ``scope``.

    Memoised on ``(uri, etag, scope)``: the etag is the immutable identity of
    the published bytes, so a re-published or corrected cut naturally misses and
    is re-prepared. ``None`` when the cut is unreadable, empty, out of scope or
    fails preparation — each of which is the SAME outcome as before, and none of
    which is cached as a valid result.

    The frame is handed out as a shallow copy so a consumer that assigns a
    column (the concentration forward states do, on their own copies today)
    cannot leak into the next request's series.
    """
    try:
        et = storage.etag(uri)
    except Exception:  # noqa: BLE001 - an unidentifiable cut is simply not cached
        et = None
    key = f"{uri}|{et}|{scope or 'total'}" if et is not None else None

    if key is not None:
        hit = _PREPARED_CACHE.get(key)
        if hit is not None:
            _PREPARED_CACHE.move_to_end(key)
            _perf.cache_event("canonical_prepared", _perf.CACHE_HIT)
            return hit[0].copy(deep=False)
        _perf.cache_event("canonical_prepared", _perf.CACHE_MISS)
    else:
        _perf.cache_event("canonical_prepared", _perf.CACHE_BYPASS)

    raw = _read(uri, storage)
    if raw is None or raw.empty:
        return None
    # Scope BEFORE preparing, exactly as before: preparation reads the median of
    # the rows present to decide whether LTV is a percentage or a fraction, so
    # reordering these two steps could change a published figure.
    scoped = _scope_frame(raw, scope)
    if scoped is None or scoped.empty:
        return None
    try:
        prepared, _rep = prepare_fn(scoped)
    except Exception:  # noqa: BLE001 - a bad cut never breaks the series
        return None
    if key is not None:
        _lru_put(_PREPARED_CACHE, key, (prepared, _rep), _PREPARED_CACHE_MAX)
        _perf.cache_event("canonical_prepared", _perf.CACHE_STORE)
        return prepared.copy(deep=False)
    return prepared


def canonical_etag(root: str, storage, run_id: str) -> Optional[str]:
    """The storage etag of the dated canonical for ``run_id``, or ``None`` when
    the root/run don't address one (or the backend can't answer). Lets callers
    key derived caches on content identity without downloading anything."""
    if not is_blob_root(root) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(run_id)):
        return None
    uri = f"{root.rstrip('/')}/{run_id}/{_PLATFORM_CANONICAL_NAME}"
    try:
        return storage.etag(uri)
    except Exception:  # noqa: BLE001 - caching is additive
        return None


def resolve_run_frame(root: str, storage, source_portfolio_id: Optional[str],
                      run_id: str) -> Optional[pd.DataFrame]:
    """The RAW dated platform canonical for ``run_id`` (a ``YYYY-MM-DD`` date),
    scoped to ``source_portfolio_id`` when the canonical carries provenance.
    ``None`` when the dated canonical does not exist."""
    if not is_blob_root(root) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(run_id)):
        return None
    uri = f"{root.rstrip('/')}/{run_id}/{_PLATFORM_CANONICAL_NAME}"
    try:
        if not storage.exists(uri):
            return None
    except Exception:  # noqa: BLE001
        return None
    df = _read(uri, storage)
    if df is None:
        return None
    if source_portfolio_id and "source_portfolio_id" in df.columns:
        ids = df["source_portfolio_id"].astype(str).str.strip()
        if (ids == source_portfolio_id).any():
            df = df[ids == source_portfolio_id]
    return df
