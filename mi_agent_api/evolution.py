"""mi_agent_api/evolution.py

Funded / pipeline / forecast EVOLUTION (time series) across the governed monthly
funded runs and weekly pipeline extracts already produced by onboarding.

This module REUSES the existing loaders — ``snapshots`` for funded central tapes
and ``pipeline_contract`` for the governed weekly pipeline extracts — and never
re-implements raw onboarding discovery. Each period carries its own reconciliation
(records / balance / coverage) and lineage (run id, reporting date, source file),
matching the point-in-time MI standard.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from analytics_lib.dates import coerce_dates
from trakt_core import perf as _perf
from analytics_lib.numeric import coerce_numeric
from mi_agent.mi_dataset_profile import PERCENT_POINTS, percent_storage_scale

from . import snapshots as snap
from . import pipeline_contract as pipeline_mod

_BALANCE = "current_outstanding_balance"
# Funded breakdown dimensions exposed over time (kept small + governed).
_FUNDED_BREAKDOWN_DIMS = {
    "broker": ("broker_channel",),
    "region": ("geographic_region_obligor",),
    "ltv_bucket": ("ltv_bucket",),
    # The CONSTITUENT BOOK. The prepared frames have always carried provenance —
    # scoping filters on it — but no breakdown ever asked for it, so the one
    # dimension a multi-book funder cares most about was the one the series
    # could not be cut by. Governed display label first, id as the fallback.
    "portfolio": ("source_portfolio_label", "source_portfolio_id"),
}
MISSING_BUCKET = "Unknown / Missing"


# --------------------------------------------------------------------------- #
# Small metric helpers (balance-weighted, missing-aware) — mirror snapshots.
# --------------------------------------------------------------------------- #
def _bal_sum(df: pd.DataFrame, col: str = _BALANCE) -> Optional[float]:
    if col not in df.columns:
        return None
    return float(coerce_numeric(df[col]).sum())


def _weighted_avg(df: pd.DataFrame, value_col: str, weight_col: str = _BALANCE) -> Optional[float]:
    if value_col not in df.columns or weight_col not in df.columns:
        return None
    v = coerce_numeric(df[value_col])
    w = coerce_numeric(df[weight_col])
    mask = v.notna() & w.notna()
    denom = float(w[mask].sum())
    if denom == 0:
        return None
    return round(float((v[mask] * w[mask]).sum() / denom), 4)


def _simple_avg(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    v = coerce_numeric(df[col])
    return round(float(v.mean()), 4) if v.notna().any() else None


def _reconciliation(df: pd.DataFrame, dataset: str, run_id: str,
                    required: List[str]) -> Dict[str, Any]:
    total_n = int(len(df))
    total_bal = _bal_sum(df)
    missing_measure = [c for c in required if c not in df.columns]
    return {
        "dataset": dataset,
        "run_id": run_id,
        "total_records": total_n,
        "total_balance": (round(total_bal, 2) if total_bal is not None else None),
        "records_included": total_n,
        "balance_included": (round(total_bal, 2) if total_bal is not None else None),
        "records_excluded_missing": 0,
        "balance_excluded_missing": 0.0 if total_bal is not None else None,
        "coverage_by_balance_pct": 100.0 if total_bal else None,
        "missing_dimension_fields": [],
        "missing_measure_fields": missing_measure,
        "filters": {},
    }


def _resolve_breakdown_dim(df: pd.DataFrame, key: str) -> Optional[str]:
    """The first candidate column a breakdown dimension actually has data in.

    Candidates, not a fixed column, so a dimension whose canonical name differs
    between tapes still resolves — the same data-aware resolution
    ``funded_bridge`` performs for its attribution dimension.
    """
    for col in _FUNDED_BREAKDOWN_DIMS.get(key, ()):
        if col in getattr(df, "columns", []) and df[col].notna().any():
            return col
    return None


def _breakdown(df: pd.DataFrame, dim_col: str, value_col: str = _BALANCE
               ) -> List[Dict[str, Any]]:
    """``[{key, value}]`` summing ``value_col`` by ``dim_col``; missing keys go to
    an explicit Unknown / Missing bucket so the breakdown reconciles to the total."""
    if dim_col not in df.columns or value_col not in df.columns:
        return []
    keys = df[dim_col].astype(object)
    blank = ~keys.notna() | keys.astype(str).str.strip().isin(["", "nan", "None", "NaT"])
    keys = keys.where(~blank, MISSING_BUCKET).astype(str)
    grp = coerce_numeric(df[value_col]).groupby(keys).sum()
    return [{"key": str(k), "value": round(float(v), 2)} for k, v in grp.items()]


# --------------------------------------------------------------------------- #
# Funded evolution
# --------------------------------------------------------------------------- #
def _runs_up_to(output_root: str | os.PathLike, client_id: str,
                to_run_id: Optional[str]) -> List[Dict[str, Any]]:
    disc = snap.discover_snapshots(output_root)
    pf = next((p for p in disc.get("portfolios", []) if p.get("client_id") == client_id), None)
    runs = list(pf.get("runs", [])) if pf else []
    if to_run_id:
        cut = next((i for i, r in enumerate(runs) if r["run_id"] == to_run_id), None)
        if cut is not None:
            runs = runs[: cut + 1]
    return runs


def assemble_funded_evolution(frames: List[Dict[str, Any]], client_id: str,
                              to_run_id: Optional[str] = None,
                              breakdowns: Optional[List[str]] = None,
                              *, lineage: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
    """Build the funded evolution series from an ordered list of prepared run
    frames — ``[{run_id, reporting_date, df, source}]`` (oldest → newest).

    Shared by the on-disk tape path and the blob platform-canonical path so the
    metric/reconciliation/breakdown shape is IDENTICAL regardless of source."""
    required = [_BALANCE, "current_loan_to_value", "current_interest_rate",
                "youngest_borrower_age"]
    want_breakdowns = breakdowns or ["portfolio", "broker", "region", "ltv_bucket"]

    periods: List[Dict[str, Any]] = []
    run_ids: List[str] = []
    dates: List[Optional[str]] = []
    sources: List[Optional[str]] = []
    bd_series: Dict[str, List[Dict[str, Any]]] = {b: [] for b in want_breakdowns}

    for fr in frames:
        run_id = fr["run_id"]
        df = fr["df"]
        if df is None:
            continue
        rdate = fr.get("reporting_date") or snap.infer_reporting_date(run_id, df)
        source = fr.get("source")
        run_ids.append(run_id)
        dates.append(rdate)
        sources.append(source)
        periods.append({
            "run_id": run_id,
            "reporting_date": rdate,
            "period": (rdate or run_id)[:7],
            "metrics": {
                "funded_balance": _bal_sum(df),
                "loan_count": int(len(df)),
                # AVERAGE BALANCE PER LOAN. Derived from the two figures either
                # side of it, so it cannot disagree with them. Without it,
                # "show average balance over time" resolved to `funded_balance`
                # and returned the TOTAL series — byte-identical to "show funded
                # balance over time", with nothing saying the measure had
                # changed.
                "avg_balance": (_bal_sum(df) / len(df)) if len(df) else None,
                # Fractions, per the contract on _pct_fraction. This route used
                # the raw weighted average, so it emitted whatever the tape
                # stored while the cohort routes emitted fractions — the same
                # metric key carrying two conventions.
                "wa_ltv": _pct_fraction(df, "current_loan_to_value"),
                "wa_interest_rate": _pct_fraction(df, "current_interest_rate"),
                "avg_borrower_age": _simple_avg(df, "youngest_borrower_age"),
            },
            "reconciliation": _reconciliation(df, "funded", run_id, required),
            "source_file": source,
        })
        for b in want_breakdowns:
            dim_col = _resolve_breakdown_dim(df, b)
            if dim_col:
                for row in _breakdown(df, dim_col):
                    bd_series[b].append({"period": (rdate or run_id)[:7], **row})

    return {
        "dataset": "funded",
        "portfolioId": client_id,
        "toRunId": to_run_id,
        "availableRunIds": run_ids,
        "reportingDates": dates,
        "sourceFiles": sources,
        "periods": periods,
        "breakdowns": bd_series,
        "lineage": lineage or {
            "source": "governed monthly central lender tapes (18_central_lender_tape.csv)",
            "metric": "funded book actuals per reporting month",
            "note": "Each period is an independent funded run; no cross-run merge.",
        },
        "singlePeriod": len(periods) <= 1,
    }


@_perf.stage_fn("funded_frames")
def funded_frames(output_root: str | os.PathLike, client_id: str,
                  to_run_id: Optional[str] = None,
                  scope=None) -> List[Dict[str, Any]]:
    """Ordered prepared funded run frames ``[{run_id, reporting_date, df, source}]``
    (oldest → newest), up to ``to_run_id`` inclusive.

    Blob-aware: the on-disk tape walk (``snap.discover_snapshots``) is
    filesystem-only and cannot enumerate a ``blob://`` platform root, so on such
    a root it returns ZERO periods (the cause of the "no reporting periods" / "£0"
    failures). On a blob root, build from the dated platform canonicals (the same
    source that powers ``/mi/evolution/funded``); fall back to the tape walk on any
    error. Shared by funded_evolution and funded_bridge so both see identical
    periods regardless of source."""
    from . import platform_snapshots_blob as _blob
    if _blob.is_blob_root(output_root):
        try:
            from apps.blob_trigger_app.storage import open_storage
            from .funded_prep import prepare_funded_mi_dataset
            return _apply_scope_to_frames(_blob.build_funded_evolution_frames(
                str(output_root), open_storage(), client_id, to_run_id,
                prepare_funded_mi_dataset), scope)
        except Exception:  # noqa: BLE001 - never break the series on a blob error
            pass
    frames: List[Dict[str, Any]] = []
    for run in _runs_up_to(output_root, client_id, to_run_id):
        run_id = run["run_id"]
        tape = snap.resolve_tape_path(output_root, client_id, run_id)
        if tape is None:
            continue
        try:
            df, _rep = snap.load_prepared_run(tape)
        except Exception:  # noqa: BLE001 - a bad tape never breaks the series
            continue
        frames.append({
            "run_id": run_id,
            "reporting_date": run.get("reporting_date") or snap.infer_reporting_date(run_id, df),
            "df": df,
            "source": str(tape),
        })
    return _apply_scope_to_frames(frames, scope)


def _apply_scope_to_frames(frames: List[Dict[str, Any]], scope) -> List[Dict[str, Any]]:
    """Narrow every period frame to the governed portfolio scope.

    Applied at the SERIES level so every downstream evolution / bridge / cohort
    surface sees exactly the portfolios the workspace selected — one filter, not
    one per view. ``scope=None`` (or Total) is a no-op, which keeps the existing
    consolidated behaviour byte-identical."""
    if scope is None or getattr(scope, "is_total", False):
        return frames
    from mi_agent.portfolio_scope import apply_scope
    return [{**fr, "df": apply_scope(fr.get("df"), scope)} for fr in frames]


def funded_evolution(output_root: str | os.PathLike, client_id: str,
                     to_run_id: Optional[str] = None,
                     breakdowns: Optional[List[str]] = None,
                     scope=None) -> Dict[str, Any]:
    """Funded time series across monthly runs up to ``to_run_id`` (inclusive),
    narrowed to the governed portfolio ``scope``."""
    return assemble_funded_evolution(
        funded_frames(output_root, client_id, to_run_id, scope=scope),
        client_id, to_run_id, breakdowns)


# --------------------------------------------------------------------------- #
# Funded balance BRIDGE (attribution waterfall between two periods)
# --------------------------------------------------------------------------- #
def _period_label(fr: Dict[str, Any]) -> str:
    rd = fr.get("reporting_date") or fr.get("run_id")
    return str(rd)[:7] if rd else str(fr.get("run_id"))


def _scope_frame_lens(df, lens_filters, *, evidence_out=None):
    """Narrow a funded frame to a source-portfolio scope.

    A governed scope resolves to a LIST of portfolio ids (a group is the sum of
    its current members), so list values are matched with membership; a single
    value still matches by equality. Case/whitespace-insensitive; a filter on an
    absent column is a no-op.

    ``evidence_out``, when given, receives what this narrowing actually did —
    the fields narrowed on and the row counts either side. A route that narrows
    and publishes nothing is indistinguishable from one that narrowed nothing,
    and that indistinguishability is what let a whole-book movement be reported
    under the Direct label with a receipt no consumer could challenge. Optional,
    so no existing caller changes behaviour by not passing it.
    """
    if not lens_filters or df is None:
        return df
    work = df
    rows_before = len(work)
    applied = []
    for col, val in lens_filters.items():
        if col not in work.columns:
            continue
        applied.append(col)
        norm = work[col].astype(str).str.strip().str.casefold()
        if isinstance(val, (list, tuple, set)):
            wanted = {str(v).strip().casefold() for v in val}
            work = work[norm.isin(wanted)]
        else:
            work = work[norm == str(val).strip().casefold()]
    if evidence_out is not None and applied:
        evidence_out.append({"fields": tuple(applied),
                             "detail": ", ".join(
                                 sorted(str(v) for val in lens_filters.values()
                                        for v in (val if isinstance(val, (list, tuple, set))
                                                  else [val]))),
                             "rows_before": rows_before, "rows_after": len(work)})
    return work


_MISSING_TOKENS = {"", "nan", "none", "nat", "<na>"}


def _group_balance(df, col: str) -> Dict[str, float]:
    """Funded balance summed by a dimension column; blank/NaN → 'Unknown / Missing'."""
    if df is None or col not in df.columns:
        return {}
    s = df[col].astype(str).str.strip()
    s = s.mask(s.str.casefold().isin(_MISSING_TOKENS), "Unknown / Missing")
    bal = coerce_numeric(df[_BALANCE])
    grp = bal.groupby(s).sum()
    return {str(k): float(v) for k, v in grp.items()}


def funded_bridge(output_root: str | os.PathLike, client_id: str,
                  dimension_col, *, start_period: Optional[str] = None,
                  window_periods: Optional[int] = None,
                  to_run_id: Optional[str] = None,
                  lens_filters: Optional[Dict[str, str]] = None,
                  lens_label: str = "Total", top_n: int = 8) -> Dict[str, Any]:
    """Attribution bridge: opening funded balance (start period) → per-category
    change over ``dimension_col`` → closing funded balance (the LATEST period, or
    ``to_run_id``). The per-category deltas sum EXACTLY to (close − open), so the
    waterfall reconciles to the book. ``lens_filters`` scopes the frames for a
    consolidated (None) vs cohort/type view.

    ``dimension_col`` may be a single column or an ordered list of candidate
    columns (e.g. the region family) — the first one actually present in the data
    is used, so attribution works regardless of which column the tape carries."""
    scoped: List[Dict[str, Any]] = []
    for fr in funded_frames(output_root, client_id, to_run_id):
        d = _scope_frame_lens(fr.get("df"), lens_filters)
        if d is not None and len(d):
            scoped.append({**fr, "df": d})
    if len(scoped) < 2:
        return {"available": False, "lens": lens_label,
                "reason": "at least two funded reporting periods are needed for a bridge"}

    # Resolve the dimension column data-aware from the candidate(s): the first
    # candidate ACTUALLY PRESENT in the data. Never a candidate that is absent —
    # a bridge grouped on a column the tape does not carry groups nothing,
    # `_group_balance` returns {} on both sides, and the waterfall reports
    # £0 → £0 (net £0) for a book that moved materially. That valid-looking zero
    # is a wrong answer, so a requested-but-absent dimension FAILS CLOSED here,
    # which is not the same as a book that did not move.
    candidates = [dimension_col] if isinstance(dimension_col, str) else list(dimension_col or [])
    present_cols = set().union(*[set(f["df"].columns) for f in scoped]) if scoped else set()
    col = next((c for c in candidates if c in present_cols), None)
    if not col:
        if candidates:
            return {"available": False, "lens": lens_label,
                    "reason": ("the requested attribution dimension is not "
                               "available in the funded data"),
                    "requestedDimension": [str(c) for c in candidates]}
        return {"available": False, "lens": lens_label,
                "reason": "no attribution dimension is available in the funded data"}

    end = scoped[-1]                       # the latest period is always the close
    start = None
    if start_period:
        sp = str(start_period)[:7]
        start = next((f for f in scoped if _period_label(f) == sp), None)
    if start is None and window_periods:
        # A STATED WINDOW OPENS THE BRIDGE. "last month" names no start period
        # but does say how far back it reaches, and the plan declares that as
        # `window_periods`. Without this the bridge fell to the earliest
        # snapshot below and reported five months of movement for a question
        # about one. Clamped to the history that exists — a window reaching
        # further back than the tape opens at the earliest period, which is the
        # same governed behaviour as before for that case.
        index = max(0, len(scoped) - 1 - int(window_periods))
        start = scoped[index]
    if start is None or _period_label(start) == _period_label(end):
        start = scoped[0]                  # default: earliest available period
    if _period_label(start) == _period_label(end):
        return {"available": False, "lens": lens_label,
                "reason": "the start and latest period resolve to the same period"}

    a = _group_balance(start["df"], col)
    b = _group_balance(end["df"], col)
    cats = set(a) | set(b)
    contribs = [{"category": c, "start": round(a.get(c, 0.0), 2),
                 "end": round(b.get(c, 0.0), 2),
                 "delta": round(b.get(c, 0.0) - a.get(c, 0.0), 2)} for c in cats]
    contribs.sort(key=lambda r: abs(r["delta"]), reverse=True)
    open_total = round(sum(a.values()), 2)
    close_total = round(sum(b.values()), 2)

    # Top-N contributors by absolute movement + an aggregated "Other" so a
    # many-category bridge stays legible AND still reconciles (Other carries the
    # residual delta).
    if top_n and len(contribs) > top_n:
        head, tail = contribs[:top_n], contribs[top_n:]
        head.append({"category": "Other", "isOther": True, "count": len(tail),
                     "start": round(sum(r["start"] for r in tail), 2),
                     "end": round(sum(r["end"] for r in tail), 2),
                     "delta": round(sum(r["delta"] for r in tail), 2)})
        contribs = head

    return {
        "available": True,
        "dimensionCol": col,
        "lens": lens_label,
        "start": {"period": _period_label(start),
                  "reporting_date": start.get("reporting_date"), "total": open_total},
        "end": {"period": _period_label(end),
                "reporting_date": end.get("reporting_date"), "total": close_total},
        "netChange": round(close_total - open_total, 2),
        "contributions": contribs,
    }


# --------------------------------------------------------------------------- #
# ECONOMIC funded-balance bridge
#
# ``funded_bridge`` above answers "which DIMENSIONS moved the total" — regions,
# brokers, LTV bands. This answers the different question a funder asks first:
# *what happened to the loans?* Opening balance, plus the loans that arrived,
# less the loans that left, plus what the loans present throughout did.
#
# NOTHING IS CALCULATED HERE. Two governed engines already own the economics and
# this composes their output:
#
#   mi_agent.period_change.bridge.balance_bridge   the reconciled identity,
#       over a stable loan key, refusing to report at all on duplicate or
#       missing identifiers, mixed currency or a missing balance field;
#   analytics_lib.history.classify_exits           the exit leg split on
#       EVIDENCE into redemption / default / maturity / unexplained.
#
# The two compose exactly: the classified buckets sum to the bridge's exit leg,
# which is asserted here rather than assumed.
#
# One deliberate restraint. ``movement_on_continuing_loans`` is NOT relabelled
# as interest. On a roll-up book it is mostly accretion; on an amortising book
# it is mostly repayment; on either it also absorbs further advances and any
# restatement. Separating them needs per-loan period movement, which the
# canonical model does not carry. It is reported as what it is — the movement on
# the loans present at both dates.
# --------------------------------------------------------------------------- #

#: Presentation-ready labels for the exit buckets, so both surfaces name them
#: identically. The keys are ``analytics_lib.history``'s own constants.
EXIT_LABELS = {
    "redemption": "Redeemed",
    "default_exit": "Exited in default",
    "maturity": "Matured",
    "unknown_exit": "Exited — reason not evidenced",
}


def _snapshot_frame(frame_record):
    from mi_agent.period_change.models import SnapshotFrame
    return SnapshotFrame(snapshot_id=str(frame_record.get("run_id") or ""),
                         reporting_date=(str(frame_record.get("reporting_date"))
                                         if frame_record.get("reporting_date") else None),
                         frame=frame_record.get("df"))


def _exit_frames(opening, closing):
    """The same two frames, addressable by the exit classifier.

    ``analytics_lib.history`` keys loans on ``loan_identifier`` and nothing
    else. A regime-projected book carries the ESMA RREL1 name
    (``unique_identifier``) INSTEAD of the analytics one, so the classifier
    declines on it — and the bridge then shows a total exit bar for a book whose
    exit reasons are sitting right there on the tape.

    This aliases the column and changes NOTHING else: the identifiers are the
    same strings, the classification rules are the classifier's own, and where
    neither name is present both frames are handed back untouched so the
    classifier declines exactly as it does today. No new analytic is performed.
    """
    from analytics_lib.history import LOAN_ID_FIELD

    for frame in (opening, closing):
        if LOAN_ID_FIELD in getattr(frame, "columns", ()):
            return opening, closing        # already addressable; leave it alone
    alias = next((c for c in _LOAN_ID_COLS
                  if c in getattr(opening, "columns", ())
                  and c in getattr(closing, "columns", ())), None)
    if alias is None:
        return opening, closing
    return (opening.rename(columns={alias: LOAN_ID_FIELD}),
            closing.rename(columns={alias: LOAN_ID_FIELD}))


def funded_balance_movement(output_root: str | os.PathLike, client_id: str,
                            to_run_id: Optional[str] = None, *, scope=None,
                            start_period: Optional[str] = None) -> Dict[str, Any]:
    """The economic opening-to-closing bridge for the governed funded book.

    ``start_period`` (``YYYY-MM``) opens the bridge at a named period; omitted,
    it opens at the period immediately before the close, which is the movement a
    reporting pack describes. Returns ``available: False`` with the engine's own
    reason wherever the bridge declines to report — a bridge that cannot
    reconcile must say so rather than present a partial identity.
    """
    frames = funded_frames(output_root, client_id, to_run_id, scope=scope)
    if len(frames) < 2:
        return {"available": False,
                "reason": ("an opening-to-closing bridge needs two governed "
                           f"reporting periods; {len(frames)} available"),
                "periodsAvailable": len(frames)}

    end = frames[-1]
    start = None
    if start_period:
        want = str(start_period)[:7]
        start = next((f for f in frames[:-1] if _period_label(f) == want), None)
    start = start or frames[-2]
    if _period_label(start) == _period_label(end):
        return {"available": False,
                "reason": "the opening and closing periods resolve to the same period"}

    from mi_agent.period_change.bridge import balance_bridge
    bridge = balance_bridge(_snapshot_frame(start), _snapshot_frame(end)).to_dict()
    if not bridge.get("reconciles"):
        return {"available": False,
                "reason": bridge.get("limitation") or
                          f"the balance bridge reported {bridge.get('status')}",
                "status": bridge.get("status"),
                "bridge": bridge}

    # The exit leg, split on evidence. Absent evidence fields the whole exit
    # balance lands in ``unknown_exit``, which is a data-quality finding and is
    # shown as one — never quietly folded into redemptions.
    exits: Dict[str, Any] = {}
    try:
        from analytics_lib.history import classify_exits
        opening_df, closing_df = _exit_frames(start["df"], end["df"])
        exits = classify_exits(opening_df, closing_df,
                               as_of=str(end.get("reporting_date") or "")) or {}
    except Exception as exc:  # noqa: BLE001 - a bridge without the split is still a bridge
        exits = {"classified": False, "reason": f"{type(exc).__name__}: {exc}"}

    exit_total = round(float(bridge.get("exited_loan_opening_balance") or 0.0), 2)
    components = []
    if exits.get("classified"):
        for key, label in EXIT_LABELS.items():
            bucket = exits.get(key) or {}
            balance = round(float(bucket.get("balance") or 0.0), 2)
            if balance:
                components.append({"key": key, "label": label, "balance": balance,
                                   "loanCount": int(bucket.get("loan_count") or 0)})
        classified_total = round(sum(c["balance"] for c in components), 2)
        # The two engines must agree. They are computed independently from the
        # same pair of snapshots, so a disagreement is a real defect, not a
        # rounding artefact — report it rather than draw a bridge that lies.
        exits_reconcile = abs(classified_total - exit_total) <= 0.01
    else:
        classified_total, exits_reconcile = None, None

    opening = round(float(bridge["opening_balance"]), 2)
    closing = round(float(bridge["closing_balance"]), 2)
    return {
        "available": True,
        "openingPeriod": _period_label(start),
        "closingPeriod": _period_label(end),
        "openingDate": start.get("reporting_date"),
        "closingDate": end.get("reporting_date"),
        "identifierField": bridge.get("identifier_field"),
        "openingBalance": opening,
        "newLoanBalance": round(float(bridge["new_loan_closing_balance"]), 2),
        "exitedLoanBalance": exit_total,
        "continuingMovement": round(float(bridge["movement_on_continuing_loans"]), 2),
        "closingBalance": closing,
        "netChange": round(closing - opening, 2),
        "newLoanCount": bridge.get("new_loan_count"),
        "exitedLoanCount": bridge.get("exited_loan_count"),
        "continuingLoanCount": bridge.get("continuing_loan_count"),
        "reconciles": True,
        "residual": bridge.get("residual"),
        "tolerance": bridge.get("rounding_tolerance"),
        "exitComponents": components,
        "exitsClassified": bool(exits.get("classified")),
        "exitsReconcile": exits_reconcile,
        "exitEvidenceFields": list(exits.get("evidence_fields") or ()),
        "exitClassificationReason": exits.get("reason"),
        "lineage": {
            "identity": ("opening + new loans - exited loans + movement on "
                         "continuing loans = closing"),
            "engine": "mi_agent.period_change.bridge.balance_bridge",
            "exits": "analytics_lib.history.classify_exits (evidence-based)",
            "continuingMovement": (
                "The movement on loans present at BOTH dates. It is not split "
                "into interest, repayment or further advance: that separation "
                "needs per-loan period movement, which the canonical model does "
                "not carry."),
        },
    }


# --------------------------------------------------------------------------- #
# Funded cohort PROGRESSION (static-pool seasoning across reporting periods)
# --------------------------------------------------------------------------- #
_VALUATION_COLS = ("indexed_valuation_amount", "current_valuation_amount",
                   "indexed_value", "original_valuation_amount")
_ORIG_DATE = "origination_date"
_VINTAGE = "vintage_year"


def _pct_fraction(df, col: str, scale_from=None) -> Optional[float]:
    """Balance-weighted average of a percent column, normalised to a FRACTION so
    the UI's ×100 formatter renders it correctly (the tape stores LTV as a
    fraction but the interest rate in points).

    ``scale_from`` is the frame the storage unit is inferred from, and should be
    the WHOLE reporting period rather than the subset being averaged. The unit is
    a property of the tape, not of whichever loans a cohort happens to contain,
    and ``percent_storage_scale`` is a heuristic over the values: a deeply
    seasoned roll-up cohort whose fractional LTVs mostly exceed 1.5 was
    classified as points and divided by 100, so a true 175% LTV rendered as
    1.75%. Inferring from the full period makes a subset unable to re-decide the
    unit, which is what produced a 100x discontinuity inside a single series.
    """
    wavg = _weighted_avg(df, col)
    if wavg is None:
        return None
    basis = scale_from if scale_from is not None and col in getattr(
        scale_from, "columns", ()) else df
    if col in getattr(basis, "columns", ()) and \
            percent_storage_scale(basis[col]) == PERCENT_POINTS:
        return round(wavg / 100.0, 6)
    return wavg


def _nneg_metrics(df) -> Dict[str, Any]:
    """NNEG (no-negative-equity-guarantee) exposure/headroom for a lifetime book:
    exposure = Σ max(0, balance − property value); headroom% = 1 − balance/value
    (balance-weighted). Empty when no valuation column is present."""
    val_col = next((c for c in _VALUATION_COLS if c in df.columns), None)
    if val_col is None:
        return {}
    bal = coerce_numeric(df[_BALANCE])
    val = coerce_numeric(df[val_col])
    mask = bal.notna() & val.notna() & (val > 0)
    if not bool(mask.any()):
        return {}
    b, v = bal[mask], val[mask]
    exposure = float((b - v).clip(lower=0).sum())
    vsum = float(v.sum())
    return {
        "nneg_exposure": round(exposure, 2),
        "nneg_headroom": round(float((v - b).sum()), 2),
        "nneg_headroom_pct": (round(1.0 - float(b.sum()) / vsum, 6) if vsum else None),
    }


def _origination_labels(df, grain: str = "Y"):
    """Per-row origination-cohort label at the requested grain (Y / Q / M), from
    ``origination_date`` (else ``vintage_year`` for year grain). None if neither."""
    if _ORIG_DATE in df.columns:
        od = coerce_dates(df[_ORIG_DATE])
        if od.notna().any():
            g = (grain or "Y").upper()
            if g == "Q":
                return (od.dt.year.astype("Int64").astype(str) + "-Q"
                        + od.dt.quarter.astype("Int64").astype(str))
            if g == "M":
                return od.dt.strftime("%Y-%m")
            return od.dt.year.astype("Int64").astype(str)
    if _VINTAGE in df.columns and df[_VINTAGE].notna().any():
        return df[_VINTAGE].astype("Int64").astype(str)
    return None


#: Loan identifier columns, in preference order. Cohort membership is a set of
#: these; without one, a static pool cannot be formed and the service says so
#: rather than falling back to a per-period filter that only looks like one.
#:
#: The order mirrors ``engine.platform_assembler.LOAN_KEY_FIELDS``, which is what
#: actually keys a platform canonical: ``loan_identifier`` first (the core
#: canonical analytics identifier every governed tape carries), then
#: ``unique_identifier`` (the ESMA Annex 2 RREL1 regulatory identifier, present
#: only on a regime-projected book). Reading only the regulatory name meant a
#: funded MI client with no regime projection had no identifier here at all, so
#: every period after formation was emptied and the progression reported a book
#: collapsing to zero. ``tests/test_evolution.py`` pins this against the
#: assembler's contract. The trailing raw-source names are legacy tolerance.
_LOAN_ID_COLS = ("loan_identifier", "unique_identifier",
                 "loan_id", "account_id", "loan_reference")

#: Bumped whenever the membership rule or the emitted contract changes, so every
#: channel can assert it is reading the same methodology.
COHORT_METHODOLOGY_VERSION = "static-pool-2"


def _loan_id_col(df) -> Optional[str]:
    return next((c for c in _LOAN_ID_COLS if c in getattr(df, "columns", ())), None)


def _member_ids(df, col: str) -> set:
    ids = df[col].dropna().astype(str)
    return set(ids.tolist())


def funded_cohort_progression(output_root: str | os.PathLike, client_id: str, *,
                              lens_filters: Optional[Dict[str, str]] = None,
                              lens_label: str = "Total",
                              vintage: Optional[str] = None, grain: str = "Y",
                              to_run_id: Optional[str] = None) -> Dict[str, Any]:
    """Static-pool progression: a cohort FIXED AT FORMATION, tracked forward.

    The membership rule, and the whole point of this service:

      1. the formation cut is the earliest reporting period in which the cohort
         has any loans;
      2. the loan identifiers present at that cut ARE the cohort, for life;
      3. every later period reports only those identifiers;
      4. surviving membership is therefore always a subset of formation
         membership, and the surviving count can hold or fall but never rise;
      5. a loan boarded later carrying an old origination date does NOT enter an
         already-formed pool;
      6. exits are formation identifiers no longer present.

    This replaced a per-period re-filter on origination vintage. That rule
    reselected members every period, so a late-boarded loan of an earlier vintage
    joined its cohort and the "surviving" count rose — which made retention,
    exits and seasoning measures of a moving population. The vintage filter is
    still what SELECTS the cohort at formation; it is no longer what defines
    membership afterwards.

    Balance may exceed 100% of formation, because roll-up interest accrues on
    surviving loans. Loan count may not.

    A corrected upload for a later reporting date replaces that period's
    observation without touching the formation rule; a corrected FORMATION
    snapshot re-forms the pool, because the formation cut is derived from the
    frames rather than stored.
    """
    frames = list(funded_frames(output_root, client_id, to_run_id))
    scoped: List[Dict[str, Any]] = []
    for fr in frames:
        d = _scope_frame_lens(fr.get("df"), lens_filters)
        if d is not None:
            scoped.append({**fr, "df": d})

    id_col = next((_loan_id_col(f["df"]) for f in scoped if _loan_id_col(f["df"])), None)
    vintage_filterable = True

    # -- 1/2. formation: the earliest period holding this cohort -------------
    formation: Optional[Dict[str, Any]] = None
    formation_at = -1                      # INDEX, not identity: `formation`
    formation_ids: set = set()             # is a rebuilt dict, never `fr` itself
    for i, fr in enumerate(scoped):
        d = fr["df"]
        if vintage:
            labels = _origination_labels(d, grain)
            if labels is None:
                vintage_filterable = False
                continue
            d = d[labels.astype(str) == str(vintage)]
        if len(d):
            formation = {**fr, "df": d}
            formation_at = i
            if id_col:
                formation_ids = _member_ids(d, id_col)
            break

    periods: List[Dict[str, Any]] = []
    data_quality: Dict[str, Any] = {}
    if formation is not None and id_col:
        # Duplicate identifiers within the formation cut are a data-quality
        # finding, not something to silently de-duplicate away: the count and the
        # id-set would then disagree, and every retention figure divides by one
        # of them.
        dup = int(len(formation["df"]) - len(formation_ids))
        if dup:
            data_quality["duplicate_formation_ids"] = dup

    for i, fr in enumerate(scoped):
        if formation is None or i < formation_at:
            continue                       # periods BEFORE formation are not the pool
        d = fr["df"]
        if id_col:
            # -- 3. only the frozen identifiers, never a re-filter ------------
            d = d[d[id_col].astype(str).isin(formation_ids)]
        else:
            # Without an identifier this cannot be a static pool. Report the
            # formation cut alone rather than a series that only looks like one.
            d = d if i == formation_at else d.iloc[0:0]

        surviving_ids = _member_ids(d, id_col) if id_col else set()
        metrics: Dict[str, Any] = {
            "funded_balance": _bal_sum(d),
            "loan_count": int(len(d)),
            # The unit comes from the WHOLE period, never from the cohort subset.
            "wa_ltv": _pct_fraction(d, "current_loan_to_value", fr["df"]),
            "wa_interest_rate": _pct_fraction(d, "current_interest_rate", fr["df"]),
            "avg_borrower_age": _simple_avg(d, "youngest_borrower_age"),
        }
        metrics.update(_nneg_metrics(d))
        formation_balance = _bal_sum(formation["df"]) if formation is not None else None
        formation_count = int(len(formation["df"])) if formation is not None else 0
        exits = max(0, len(formation_ids) - len(surviving_ids)) if id_col else None
        periods.append({
            "period": _period_label(fr),
            "reporting_date": fr.get("reporting_date"),
            "source_run": fr.get("run_id"),
            "snapshot": fr.get("source"),
            "loanCount": int(len(d)),
            "survivingLoanCount": int(len(d)),
            "survivingLoanIds": sorted(surviving_ids) if id_col else None,
            "survivingBalance": metrics["funded_balance"],
            "exitsCount": exits,
            "loanRetention": (round(len(surviving_ids) / len(formation_ids) * 100.0, 4)
                              if id_col and formation_ids else None),
            "balanceRetention": (round(metrics["funded_balance"] / formation_balance * 100.0, 4)
                                 if formation_balance else None),
            "seasoningPeriods": len(periods),
            "monthsOnBook": _simple_avg(d, "months_on_book"),
            "metrics": metrics,
        })

    available = any(p["loanCount"] for p in periods)
    reason = None
    if not available:
        reason = ("no loans match this cohort in any reporting period"
                  if vintage_filterable else
                  "origination vintage is not available on the funded tape")
    elif not id_col:
        reason = ("the funded tape carries no loan identifier, so cohort "
                  "membership cannot be fixed at formation")
    metric_keys = ["funded_balance", "loan_count", "wa_ltv", "wa_interest_rate",
                   "avg_borrower_age"]
    if any("nneg_exposure" in p["metrics"] for p in periods):
        metric_keys += ["nneg_exposure", "nneg_headroom", "nneg_headroom_pct"]

    counts = [p["loanCount"] for p in periods if p["loanCount"]]
    if any(b > a for a, b in zip(counts, counts[1:])):
        # Cannot happen with a frozen set; asserted here so a future regression
        # surfaces in the contract rather than in a slide.
        data_quality["membership_not_fixed"] = True

    return {
        "dataset": "cohort_progression",
        "portfolioId": client_id,
        "available": available and bool(id_col),
        "reason": reason,
        "lens": lens_label,
        "vintage": vintage,
        "grain": grain,
        # -- the versioned static-pool contract ------------------------------
        "methodologyVersion": COHORT_METHODOLOGY_VERSION,
        "membershipRule": "fixed_at_formation",
        "cohortId": f"{lens_label}:{vintage}" if vintage else lens_label,
        "loanIdColumn": id_col,
        "formationDate": formation.get("reporting_date") if formation else None,
        "formationPeriod": _period_label(formation) if formation else None,
        "formationLoanIds": sorted(formation_ids) if id_col else None,
        "formationLoanCount": len(formation_ids) if id_col else (
            int(len(formation["df"])) if formation is not None else 0),
        "formationBalance": _bal_sum(formation["df"]) if formation is not None else None,
        "dataQuality": data_quality,
        "metricsAvailable": metric_keys,
        "periods": periods,
        "singlePeriod": len([p for p in periods if p["loanCount"]]) <= 1,
        "lineage": {
            "source": "governed funded reporting periods (static pool)",
            "metric": "cohort funded metrics per reporting period",
            "note": ("Static pool: the loan identifiers present at the formation "
                     "cut, tracked forward. Membership is frozen at formation, so "
                     "the surviving count can hold or fall but never rise."),
        },
    }


# --------------------------------------------------------------------------- #
# Pipeline evolution (governed weekly extracts)
# --------------------------------------------------------------------------- #
@_perf.stage_fn("pipeline_evolution_series")
def pipeline_evolution(pipeline_root: str | os.PathLike, client_id: str,
                       to_run_id: Optional[str] = None, *,
                       historical_model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pipeline time series across the governed UNIQUE weekly extracts.

    When a governed ``historical_model`` is supplied the weighted-expected-funded
    amount is weighted by the SAME empirical stage completion rates used by the
    forecast bridge (falling back to configured rates only where history is thin),
    so the scale-up 'weighted expected pipeline' matches the Forecast tab rather
    than silently using the config-only fallback."""
    inv = pipeline_mod.weekly_extract_inventory(pipeline_root, client_id)
    extracts = inv.get("extracts", [])
    cut_ym = pipeline_mod._year_month(str(to_run_id)) if to_run_id else None

    periods: List[Dict[str, Any]] = []
    by_stage: List[Dict[str, Any]] = []
    sources: List[str] = []
    dates: List[Optional[str]] = []

    for ext in extracts:
        edate = ext.get("pipeline_extract_date")
        if cut_ym and edate and edate[:7] > cut_ym:
            continue
        try:
            df, report = pipeline_mod.load_prepared_pipeline(
                ext, historical_model=historical_model)
        except Exception:  # noqa: BLE001
            continue
        amount = report.get("total_pipeline_amount")
        weighted = report.get("weighted_expected_funded_amount")
        sources.append(ext.get("source_file", ""))
        dates.append(edate)
        periods.append({
            "extract_date": edate,
            "period": (edate or "")[:7],
            "week": edate,
            "metrics": {
                "pipeline_amount": (round(float(amount), 2) if amount is not None else None),
                "pipeline_case_count": int(report.get("row_count", len(df))),
                "weighted_expected_funded_amount": (round(float(weighted), 2)
                                                    if weighted is not None else None),
            },
            "reconciliation": {
                "dataset": "pipeline",
                "extract_date": edate,
                "total_records": int(report.get("row_count", len(df))),
                "total_balance": (round(float(amount), 2) if amount is not None else None),
                "coverage_by_balance_pct": 100.0,
                "missing_measure_fields": [],
                "filters": {},
            },
            "source_file": ext.get("source_file", ""),
        })
        # Pipeline amount AND case count by stage for this extract (multi-line over
        # time, day-level dates). Both metrics are emitted so the UI can chart
        # amount or count, and derive Application/Offer/Completion conversion.
        if "pipeline_stage" in df.columns:
            stage_str = df["pipeline_stage"].astype(str)
            amt = (coerce_numeric(df[_BALANCE]).groupby(stage_str).sum()
                   if _BALANCE in df.columns else None)
            cnt = stage_str.groupby(stage_str).size()
            for stage, n in cnt.items():
                if str(stage).strip() and str(stage) not in ("nan", "None"):
                    val = float(amt.get(stage, 0.0)) if amt is not None else None
                    by_stage.append({
                        "period": (edate or ""), "week": edate, "stage": str(stage),
                        "value": (round(val, 2) if val is not None else None),
                        "count": int(n)})

    return {
        "dataset": "pipeline",
        "portfolioId": client_id,
        "toRunId": to_run_id,
        "availableExtractDates": dates,
        "sourceFiles": sources,
        "sourceFilesScanned": inv.get("sourceFilesScanned"),
        "uniqueWeeklyExtractsUsed": inv.get("uniqueWeeklyExtractsUsed"),
        "periods": periods,
        "byStage": by_stage,
        # Governed trailing five-week average of the SAME weekly series above,
        # using the SAME window and the SAME trailing-mean helper the funnel
        # already publishes per stage. Additive: no existing field changes.
        "fiveWeekAverage": five_week_average(periods),
        "lineage": {
            "source": "governed weekly pipeline extracts (deduplicated)",
            "metric": "origination pipeline amount / weighted expected funded per extract",
            "fiveWeekAverage": _FIVE_WEEK_BASIS,
            "primarySourcePreference": inv.get("primarySourcePreference"),
        },
        "singlePeriod": len(periods) <= 1,
    }


#: How the total-pipeline five-week average is defined. Stated once, quoted by
#: every channel that shows the comparison, so a card and a chart can never
#: describe the same number differently.
_FIVE_WEEK_BASIS = (
    "trailing mean of the last 5 governed weekly extracts INCLUDING the current "
    "week, on the pipeline STOCK level (the same window and convention as the "
    "origination funnel's fiveWeekAvgStock* fields)")

#: The metrics the trailing average is published for. Each is already a governed
#: per-week value on ``periods[].metrics`` — this adds no new measure.
_FIVE_WEEK_METRICS = ("pipeline_amount", "pipeline_case_count",
                      "weighted_expected_funded_amount")


def five_week_average(periods: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Trailing five-week average of the governed weekly pipeline series.

    THE governed total-pipeline five-week comparison, shared by React, the
    investor deck and the Teams Portfolio Update so none of them has to derive
    one. It computes nothing new: it is the existing trailing-mean helper over
    the existing per-week metrics, at the existing 5-week window.

    ``weeksObserved`` is published alongside every value because a "five-week
    average" over two weeks of history is a materially weaker statement, and a
    caller that cannot see the sample size cannot know to say so. ``available``
    is False when there is no history at all — a caller must then omit the
    comparison with a reason rather than compare against a fabricated baseline.
    """
    window = _CONVERSION_WINDOW
    values: Dict[str, Optional[float]] = {}
    observed: Dict[str, int] = {}
    current: Dict[str, Optional[float]] = {}
    for metric in _FIVE_WEEK_METRICS:
        series = [(p.get("metrics") or {}).get(metric) for p in periods]
        series = [(float(v) if v is not None else None) for v in series]
        values[metric] = _trailing_avg(series, window)
        observed[metric] = _window_count(series, window)
        current[metric] = series[-1] if series else None

    return {
        "available": any(v is not None for v in values.values()),
        "window": window,
        "basis": _FIVE_WEEK_BASIS,
        "weeksObserved": max(observed.values()) if observed else 0,
        "weeksObservedByMetric": observed,
        "current": current,
        "average": values,
        # Signed percentage difference of the current week against the trailing
        # mean, per metric. Emitted here rather than at each call site so every
        # channel divides the same way round and rounds identically.
        "differencePct": {
            m: (round((current[m] - values[m]) / abs(values[m]) * 100.0, 1)
                if current.get(m) is not None and values.get(m) else None)
            for m in _FIVE_WEEK_METRICS
        },
    }


# --------------------------------------------------------------------------- #
# Weekly origination funnel trends (KFI / Application / Offer / Completion)
# --------------------------------------------------------------------------- #
_FUNNEL_STAGES = ("KFI", "APPLICATION", "OFFER", "COMPLETED")
_FUNNEL_LABELS = {"KFI": "KFIs", "APPLICATION": "Applications",
                  "OFFER": "Offers", "COMPLETED": "Completions"}

# The recent conversion rate averages weekly flow over a 5-week window. Require
# at least this many observed weeks in that window before the rate is treated as
# reliable — a 1-2 week rate is too volatile to publish or forecast off.
_CONVERSION_WINDOW = 5
_MIN_CONVERSION_WEEKS = 3


def _window_count(values: List[Optional[float]], window: int) -> int:
    """How many non-null values fall in the trailing ``window`` (i.e. how many
    weeks actually contributed to a trailing average)."""
    tail = [v for v in values[-window:] if v is not None]
    return len(tail)


def _trailing_avg(values: List[Optional[float]], window: int = 5) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    use = vals[-window:]
    return round(sum(use) / len(use), 2)


def _trend(values: List[Optional[float]]) -> str:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return "flat"
    delta = vals[-1] - vals[-2]
    return "up" if delta > 0 else ("down" if delta < 0 else "flat")


def weekly_flow(levels: List[Optional[float]]) -> List[Optional[float]]:
    """Convert a per-week STOCK level series into a per-week FLOW series.

    ``flow[t] = level[t] − level[t-1]`` — the new origination that arrived in
    week ``t`` (net of cases that left the stage). The first week has no prior
    extract, so its flow is ``None`` (never fabricated as the level itself). A
    week whose level is missing, or that follows a missing level, is ``None``.
    This is the semantic the origination funnel charts on by default; the raw
    stock level is retained separately for the optional cumulative line.
    """
    out: List[Optional[float]] = []
    for i, v in enumerate(levels):
        prev = levels[i - 1] if i > 0 else None
        if i == 0 or v is None or prev is None:
            out.append(None)
        else:
            out.append(round(float(v) - float(prev), 2))
    return out


def _conversion_pct(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Conversion share (%) of a stage relative to KFI, divide-by-zero safe."""
    if not denominator or numerator is None:
        return None
    return round(numerator / denominator * 100.0, 2)


def _lagged_value(series: List[Optional[float]], lag: int) -> Tuple[Optional[float], Optional[int]]:
    """The value ``lag`` steps before the latest, with the index it came from.

    Used to shift the KFI denominator back by the KFI->completion timeline so a
    growing pipeline is not compared against itself. ``lag`` is clamped into the
    available history; a missing (``None``) value at the target index returns
    ``(None, idx)`` rather than fabricating a neighbour.
    """
    n = len(series)
    if n == 0:
        return None, None
    idx = n - 1 - max(0, int(lag or 0))
    if idx < 0:
        idx = 0
    return series[idx], idx


@_perf.stage_fn("pipeline_funnel_series")
def pipeline_funnel_evolution(pipeline_root: str | os.PathLike, client_id: str,
                              to_run_id: Optional[str] = None,
                              lag_weeks: Optional[int] = None, *,
                              historical_model: Optional[Dict[str, Any]] = None
                              ) -> Dict[str, Any]:
    """Weekly origination funnel: KFI / Application / Offer / Completion per
    governed weekly extract, FLOW-FIRST.

    For each stage we track two things per week:
      * the STOCK level — total balance / case count sitting at the stage on the
        extract date (``series[stage][*].value|count``); and
      * the weekly FLOW — the week-on-week change in that level
        (``flowSeries[stage][*].flowValue|flowCount``), i.e. the new origination
        that arrived that week.

    The summary therefore reports BOTH bases, clearly separated, so the 5-week
    average and the Δ-vs-prior-week reconcile with one another:
      * ``fiveWeekAvgFlow*`` is the trailing mean of the weekly FLOW (NOT the
        average stock level — that historical bug made a ~£280MM stock average
        sit next to a ~£33MM weekly Δ);
      * ``deltaFlow*`` is the latest weekly flow minus the prior weekly flow;
      * ``*Stock*`` fields carry the level for the optional cumulative line.

    Conversion vs KFI is a *forward* conversion rate: the average weekly FLOW
    into a stage over the last 5 weeks divided by the KFI STOCK as it stood
    ``lag_weeks`` earlier — i.e. the KFI book at the time today's completions
    entered the pipeline. Shifting the denominator back by the KFI->completion
    timeline stops a growing pipeline being compared against itself (the old
    metric summed per-week stock and could exceed 100%). ``lag_weeks`` is the
    median KFI->completion lag in weeks (from the historical completion model);
    when unknown the rate is computed unlagged and flagged as such. Reuses the
    governed weekly pipeline extracts (same source as ``pipeline_evolution``).
    """
    inv = pipeline_mod.weekly_extract_inventory(pipeline_root, client_id)
    extracts = inv.get("extracts", [])
    cut_ym = pipeline_mod._year_month(str(to_run_id)) if to_run_id else None

    weeks: List[Optional[str]] = []
    sources: List[str] = []
    # series[stage] = [{week, value, count}] (STOCK level per week)
    series: Dict[str, List[Dict[str, Any]]] = {s: [] for s in _FUNNEL_STAGES}

    for ext in extracts:
        edate = ext.get("pipeline_extract_date")
        if cut_ym and edate and edate[:7] > cut_ym:
            continue
        try:
            # ``historical_model`` is passed purely so this shares the prepared
            # frame with ``pipeline_evolution`` instead of preparing every
            # extract a second time under a different cache key. It cannot
            # change this function's output: the model affects only
            # ``completion_probability``, ``completion_probability_source`` and
            # ``weighted_expected_funded_amount``, and the funnel reads neither
            # — only ``pipeline_stage`` and ``current_outstanding_balance``,
            # which are byte-identical with and without it (asserted by
            # tests/test_funnel_model_invariance.py).
            df, _report = pipeline_mod.load_prepared_pipeline(
                ext, historical_model=historical_model)
        except Exception:  # noqa: BLE001
            continue
        weeks.append(edate)
        sources.append(ext.get("source_file", ""))
        stage_col = df["pipeline_stage"].astype(str) if "pipeline_stage" in df.columns else None
        bal = coerce_numeric(df[_BALANCE]) if _BALANCE in df.columns else None
        for stage in _FUNNEL_STAGES:
            if stage_col is None:
                series[stage].append({"week": edate, "value": None, "count": 0})
                continue
            mask = stage_col.str.upper() == stage
            value = round(float(bal[mask].sum()), 2) if bal is not None else None
            series[stage].append({"week": edate, "value": value, "count": int(mask.sum())})

    # Per-week weekly-flow series derived from the stock levels (bars chart this).
    flow_series: Dict[str, List[Dict[str, Any]]] = {}
    for stage in _FUNNEL_STAGES:
        pts = series[stage]
        vflow = weekly_flow([p["value"] for p in pts])
        cflow = weekly_flow([float(p["count"]) for p in pts])
        flow_series[stage] = [
            {"week": pts[i]["week"],
             "flowValue": vflow[i],
             "flowCount": (int(cflow[i]) if cflow[i] is not None else None)}
            for i in range(len(pts))
        ]

    kfi_counts = [float(p["count"]) for p in series["KFI"]]
    kfi_values = [p["value"] for p in series["KFI"]]

    # KFI denominator, shifted back by the KFI->completion lag so the numerator
    # (recent completions) is measured against the KFI book those completions
    # actually came from — not today's larger book.
    lagged = int(lag_weeks) if lag_weeks not in (None, "") else None
    lag_applied = lagged if lagged is not None else 0
    kfi_denom_count, kfi_denom_idx = _lagged_value(kfi_counts, lag_applied)
    kfi_denom_value, _ = _lagged_value(kfi_values, lag_applied)
    denom_week = weeks[kfi_denom_idx] if kfi_denom_idx is not None and kfi_denom_idx < len(weeks) else None

    summary: Dict[str, Any] = {}
    for stage in _FUNNEL_STAGES:
        pts = series[stage]
        values = [p["value"] for p in pts]
        counts = [float(p["count"]) for p in pts]
        value_flows = [f["flowValue"] for f in flow_series[stage]]
        count_flows = [(float(f["flowCount"]) if f["flowCount"] is not None else None)
                       for f in flow_series[stage]]

        latest_flow_value = value_flows[-1] if value_flows else None
        latest_flow_count = count_flows[-1] if count_flows else None
        prior_flow_value = value_flows[-2] if len(value_flows) >= 2 else None
        prior_flow_count = count_flows[-2] if len(count_flows) >= 2 else None

        avg_flow_value = _trailing_avg(value_flows, _CONVERSION_WINDOW)
        avg_flow_count = _trailing_avg(count_flows, _CONVERSION_WINDOW)

        # Forward conversion vs KFI (never for KFI itself, the denominator):
        # average weekly flow into this stage (last 5 weeks) over the lagged KFI
        # stock. A weekly rate; transparent about the lag and the denominator
        # week so it can't be misread as a same-period share. Flagged
        # insufficient (not to be forecast off) until a few weeks are observed.
        conversion: Optional[Dict[str, Any]] = None
        if stage != "KFI":
            weeks_in_window = _window_count(value_flows, _CONVERSION_WINDOW)
            sufficient = weeks_in_window >= _MIN_CONVERSION_WEEKS
            conversion = {
                "basis": "avg_weekly_flow_over_lagged_kfi_stock",
                "lagWeeks": lagged,
                "lagApplied": bool(lagged),
                "denominatorWeek": denom_week,
                "avgWeeklyFlowCount": avg_flow_count,
                "avgWeeklyFlowValue": avg_flow_value,
                "kfiStockCount": (int(kfi_denom_count)
                                  if kfi_denom_count is not None else None),
                "kfiStockValue": kfi_denom_value,
                "weeklyRateCount": _conversion_pct(avg_flow_count, kfi_denom_count),
                "weeklyRateValue": _conversion_pct(avg_flow_value, kfi_denom_value),
                "weeksInWindow": weeks_in_window,
                "minWeeks": _MIN_CONVERSION_WEEKS,
                "sufficient": sufficient,
            }

        summary[stage] = {
            "label": _FUNNEL_LABELS[stage],
            # Weekly FLOW (default basis for the origination funnel).
            "latestFlowValue": latest_flow_value,
            "latestFlowCount": (int(latest_flow_count)
                                if latest_flow_count is not None else None),
            "priorFlowValue": prior_flow_value,
            "priorFlowCount": (int(prior_flow_count)
                               if prior_flow_count is not None else None),
            "fiveWeekAvgFlowValue": _trailing_avg(value_flows, 5),
            "fiveWeekAvgFlowCount": _trailing_avg(count_flows, 5),
            "deltaFlowValue": (round(latest_flow_value - prior_flow_value, 2)
                               if latest_flow_value is not None
                               and prior_flow_value is not None else None),
            "deltaFlowCount": (int(latest_flow_count - prior_flow_count)
                               if latest_flow_count is not None
                               and prior_flow_count is not None else None),
            # STOCK level (drives the optional cumulative line).
            "latestStockValue": values[-1] if values else None,
            "latestStockCount": pts[-1]["count"] if pts else 0,
            "fiveWeekAvgStockValue": _trailing_avg(values, 5),
            "fiveWeekAvgStockCount": _trailing_avg(counts, 5),
            "trend": _trend(value_flows),
            "weeksObserved": len([v for v in values if v is not None]),
            "conversion": conversion,
        }

    return {
        "dataset": "pipeline_funnel",
        "portfolioId": client_id,
        "toRunId": to_run_id,
        "stages": list(_FUNNEL_STAGES),
        "stageLabels": _FUNNEL_LABELS,
        "weeks": weeks,
        "sourceFiles": sources,
        "uniqueWeeklyExtractsUsed": inv.get("uniqueWeeklyExtractsUsed"),
        "series": series,
        "flowSeries": flow_series,
        "summary": summary,
        "conversionLagWeeks": lagged,
        "lineage": {
            "source": "governed weekly pipeline extracts (deduplicated)",
            "metric": "weekly KFI / Application / Offer / Completion — weekly flow (default) and stock level",
            "fiveWeekAverage": "trailing mean of the last 5 weeks of WEEKLY FLOW (level week-on-week change), not the average stock level",
            "conversion": ("forward conversion rate: average weekly flow into a stage (last 5 weeks) "
                           "over the KFI stock lagWeeks earlier (the KFI->completion timeline); "
                           "unlagged when the lag is unknown"),
        },
        "singlePeriod": len(weeks) <= 1,
    }


# --------------------------------------------------------------------------- #
# Forecast bridge evolution (funded balance + weighted pipeline, per funded run)
# --------------------------------------------------------------------------- #
def forecast_evolution(output_root: str | os.PathLike,
                       pipeline_root: str | os.PathLike, client_id: str,
                       to_run_id: Optional[str] = None, *,
                       historical_model: Optional[Dict[str, Any]] = None,
                       scope=None,
                       include_pipeline: bool = True) -> Dict[str, Any]:
    """Forecast bridge over time: funded balance per run + the latest weighted
    pipeline contribution available at/under that run's month. A governed
    ``historical_model`` weights the pipeline by the same empirical stage rates as
    the point-in-time bridge (one consistent 'weighted expected pipeline').

    ``scope`` narrows the funded side to the selected portfolios.
    ``include_pipeline=False`` is used when the governed capability resolver says
    no portfolio in scope originates — the funded series is still returned, with
    no fabricated pipeline contribution."""
    funded = funded_evolution(output_root, client_id, to_run_id, scope=scope)
    pipe = (pipeline_evolution(pipeline_root, client_id, to_run_id,
                               historical_model=historical_model)
            if include_pipeline else {"periods": []})
    # Index pipeline weighted-expected by year-month (latest extract per month).
    weighted_by_month: Dict[str, float] = {}
    for p in pipe["periods"]:
        ym = (p.get("period") or "")
        w = p["metrics"].get("weighted_expected_funded_amount")
        if ym and w is not None:
            weighted_by_month[ym] = float(w)  # later extract overwrites -> latest wins

    periods: List[Dict[str, Any]] = []
    for fp in funded["periods"]:
        ym = fp.get("period") or ""
        funded_bal = fp["metrics"].get("funded_balance") or 0.0
        wpipe = weighted_by_month.get(ym)
        periods.append({
            "period": ym,
            "run_id": fp.get("run_id"),
            "reporting_date": fp.get("reporting_date"),
            "metrics": {
                "funded_balance": round(float(funded_bal), 2),
                "weighted_expected_pipeline": (round(wpipe, 2) if wpipe is not None else None),
                "forecast_funded_balance": round(float(funded_bal) + float(wpipe or 0.0), 2),
            },
            "reconciliation": fp.get("reconciliation"),
            "source_file": fp.get("source_file"),
        })

    # WAS THE PRIOR FORECAST RIGHT? The forecast a run published becomes the
    # prediction the NEXT run's actual tests. That is a re-indexing of the series
    # already built above — the same numbers, shifted one period — and it is done
    # here rather than in each surface because it was previously derived in the
    # browser (``EvolutionPanel``'s forecastVariance) and therefore existed on
    # exactly one of the two surfaces that should show it.
    #
    # No new economics: ``prior_forecast`` at period N IS
    # ``forecast_funded_balance`` at period N-1, and the variance is the
    # difference between two figures already reconciled above. The first period
    # carries none, because nothing forecast it.
    for index, period in enumerate(periods):
        if index == 0:
            period["metrics"]["prior_forecast"] = None
            period["metrics"]["forecast_variance"] = None
            continue
        prior = periods[index - 1]["metrics"].get("forecast_funded_balance")
        actual = period["metrics"].get("funded_balance")
        period["metrics"]["prior_forecast"] = prior
        period["metrics"]["forecast_variance"] = (
            round(float(actual) - float(prior), 2)
            if prior is not None and actual is not None else None)

    return {
        "dataset": "forecast",
        "portfolioId": client_id,
        "toRunId": to_run_id,
        "periods": periods,
        #: Periods carrying a testable prior forecast. A forecast-vs-actual view
        #: needs at least one; a track record needs more than one.
        "priorForecastPeriods": sum(
            1 for p in periods if p["metrics"].get("prior_forecast") is not None),
        "lineage": {
            "source": "funded central tapes + governed weighted pipeline",
            "formula": "forecast = funded balance + Σ(weighted expected pipeline)",
            "priorForecast": ("this run's ACTUAL funded balance beside the forecast "
                              "the PRIOR run published — the same series, offset by "
                              "one reporting period"),
        },
        "singlePeriod": len(periods) <= 1,
    }
