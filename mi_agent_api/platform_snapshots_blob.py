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
the loaded-canonical path produces — keyed by ``source_portfolio_id`` (so
``direct_001`` is the selectable funded portfolio) — but with one run per dated
cut instead of a single ``latest``.

Read-through caching keeps repeated dropdown calls cheap; a re-published dated
canonical (etag change) is picked up automatically.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_PLATFORM_CANONICAL_NAME = "platform_canonical_typed.csv"
#: A DATED platform canonical under a blob:// platform root. ``latest/`` is
#: excluded because ``latest`` is not a ``YYYY-MM-DD`` date.
_DATED_RE = re.compile(
    r"/(?P<date>\d{4}-\d{2}-\d{2})/" + re.escape(_PLATFORM_CANONICAL_NAME) + r"$")

#: uri -> (etag, DataFrame). A dated canonical is immutable per etag, so this
#: avoids re-downloading on every /mi/snapshots (or per-run /mi/snapshot) call.
_READ_CACHE: Dict[str, Tuple[Optional[str], pd.DataFrame]] = {}


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
            return cached[1]
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
        df = pd.read_csv(path, low_memory=False)
        _READ_CACHE[uri] = (et, df)
        return df
    except Exception:  # noqa: BLE001 - a bad canonical must not break discovery
        return None


def _portfolios_from_frame(df: pd.DataFrame, date: str, *,
                           label_fn, balance_fn) -> List[Dict[str, Any]]:
    """Per-``source_portfolio_id`` run rows for one dated canonical, mirroring the
    loaded-canonical index. Falls back to a single client entry when the canonical
    carries no provenance."""
    run_common = {"run_id": date, "reporting_date": date}
    rows: List[Dict[str, Any]] = []
    if "source_portfolio_id" in df.columns:
        ids = df["source_portfolio_id"].dropna().astype(str).str.strip()
        for pid in sorted({p for p in ids.unique() if p and p.lower() != "nan"}):
            sub = df[ids == pid]
            rows.append({
                "source_portfolio_id": pid,
                "label": label_fn(sub, pid),
                "run": {**run_common, "loan_count": int(len(sub)),
                        "current_outstanding_balance": round(balance_fn(sub), 2)},
            })
    if not rows:
        rows.append({
            "source_portfolio_id": None, "label": None,
            "run": {**run_common, "loan_count": int(len(df)),
                    "current_outstanding_balance": round(balance_fn(df), 2)}})
    return rows


def build_index(root: str, storage, *, label_fn, balance_fn,
                default_client_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Build ``{"portfolios": [...], "source": root}`` from the dated platform
    canonicals under a ``blob://`` root, each portfolio carrying one run per dated
    cut (oldest → newest). Returns ``None`` when there is nothing dated to
    enumerate (the caller then falls back to the loaded-canonical index)."""
    dated = list_dated_platform_canonicals(root, storage)
    if not dated:
        return None
    portfolios: Dict[str, Dict[str, Any]] = {}
    for d in dated:
        df = _read(d["uri"], storage)
        if df is None or df.empty:
            continue
        for row in _portfolios_from_frame(df, d["date"], label_fn=label_fn,
                                          balance_fn=balance_fn):
            pid = row["source_portfolio_id"]
            key = pid if pid is not None else (default_client_id or "platform")
            label = row["label"] or str(key).upper()
            pf = portfolios.setdefault(key, {
                "client_id": key, "label": label,
                **({"source_portfolio_id": pid} if pid is not None else {}),
                "runs": {}})
            pf["runs"][row["run"]["run_id"]] = row["run"]
    if not portfolios:
        return None
    out: List[Dict[str, Any]] = []
    for pf in portfolios.values():
        runs = sorted(pf["runs"].values(),
                      key=lambda r: (r["reporting_date"] or "", r["run_id"]))
        entry = {"client_id": pf["client_id"], "label": pf["label"], "runs": runs}
        if "source_portfolio_id" in pf:
            entry["source_portfolio_id"] = pf["source_portfolio_id"]
        out.append(entry)
    out.sort(key=lambda p: p["client_id"])
    return {"portfolios": out, "source": root}


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
        raw = _read(d["uri"], storage)
        if raw is None or raw.empty:
            continue
        scoped = _scope_frame(raw, scope)
        if scoped is None or scoped.empty:
            continue
        try:
            prepared, _rep = prepare_fn(scoped)
        except Exception:  # noqa: BLE001 - a bad cut never breaks the series
            continue
        frames.append({"run_id": d["date"], "reporting_date": d["date"],
                       "df": prepared, "source": d["uri"]})
    return frames


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
