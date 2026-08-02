#!/usr/bin/env python3
"""Phase 2A — week-on-week movement attribution for the MI hover / drill layer.

Answers one question, for one weekly point on an existing chart: *what changed,
and what contributed to it?* Nothing here defines a metric. The headline is the
SAME number the chart already plots, recomputed from the SAME prepared frames;
this module only decomposes it.

Two decompositions are supported, and they are deliberately the same machinery:

``PIPELINE_WEEKLY_MOVEMENT``
    Change in total pipeline exposure between two governed weekly extracts.
    The measure is ``current_outstanding_balance``, whose sum IS
    ``pipeline_amount`` in :func:`mi_agent_api.evolution.pipeline_evolution`.

``COMPLETIONS_WEEKLY_MOVEMENT``
    Change in the balance sitting at the COMPLETED stage between the same two
    extracts — i.e. the weekly flow the origination funnel plots. This is a
    PIPELINE-STAGE measure ("cases reaching completed stage"), NOT funded
    balance growth, and is labelled as such.

Both are NET movements (``movement_basis = "net"``), because both are the
difference between two stock levels.

Attribution convention
----------------------
One deterministic convention, stated in every payload:

``current_period_dimension_prior_for_removed``
    A case is attributed to the broker / region it carries in the CURRENT
    period. A case that has left (no current row) is attributed to the
    dimension it carried in the COMPARISON period, because that is the only
    value it has. Cases whose dimension was reassigned between the two periods
    are counted and reported, so the reader knows how much of the movement the
    convention affects, rather than the reassignment being invisible.

Components
----------
Mutually exclusive and exhaustive, so they always sum to the headline:

``new``             present now, absent before
``removed``         present before, absent now
``progressed_out``  present in both, moved from an active stage to a terminal
                    one (COMPLETED / WITHDRAWN) — it has left the ACTIVE
                    pipeline without leaving the extract
``increased``       present in both, balance up
``decreased``       present in both, balance down
``unchanged``       present in both, balance identical

They are never silently combined: a caller that cannot distinguish them would
be reading a different metric.

This module performs NO I/O. It is handed two already-prepared frames.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from analytics_lib.numeric import coerce_numeric

#: Detail types this module can build.
DETAIL_PIPELINE = "PIPELINE_WEEKLY_MOVEMENT"
DETAIL_COMPLETIONS = "COMPLETIONS_WEEKLY_MOVEMENT"

#: The governed case key on a prepared pipeline frame.
CASE_KEY = "pipeline_case_identifier"
#: The governed exposure measure. Its sum reconciles to ``pipeline_amount``.
MEASURE = "current_outstanding_balance"
#: The governed stage column.
STAGE = "pipeline_stage"

#: (payload key, frame column) for each contributor dimension.
DIMENSIONS: Tuple[Tuple[str, str], ...] = (
    ("brokers", "broker_channel"),
    ("regions", "geographic_region_obligor"),
)

#: Label for a blank / missing dimension value. Never dropped, never merged.
UNKNOWN = "Unknown"

#: Stages a case can sit at without being in the ACTIVE pipeline.
TERMINAL_STAGES = ("COMPLETED", "WITHDRAWN")

#: The attribution convention, echoed in every payload.
ATTRIBUTION = "current_period_dimension_prior_for_removed"

#: Bumped when any definition in this module changes.
METHODOLOGY_VERSION = "1"

#: Case-id tokens that mean "no identifier".
_NULL_IDS = ("", "nan", "none", "nat", "null")

COMPONENTS = ("new", "removed", "progressed_out", "increased", "decreased",
              "unchanged")


# --------------------------------------------------------------------------- #
# Case-level normalisation
# --------------------------------------------------------------------------- #
def _case_level(df: Optional[pd.DataFrame], dims: Sequence[str]) -> pd.DataFrame:
    """One row per case: measure summed, each dimension and the stage resolved.

    Duplicate case identifiers are a real condition in weekly extracts (the
    preparation layer already raises ``duplicate_case_identifiers``). Summing
    them keeps the case-level total equal to the frame total, so the
    decomposition still reconciles; the duplicate count is reported separately
    rather than hidden.

    Rows with no usable identifier cannot be matched across periods at all, so
    they are excluded here and counted by :func:`_unkeyed`.
    """
    cols = ["_measure", "_stage", *dims]
    if df is None or df.empty or CASE_KEY not in df.columns:
        empty = pd.DataFrame(columns=cols, index=pd.Index([], name="_case"))
        return empty.astype({"_measure": "float64"})

    key = df[CASE_KEY].astype(str).str.strip()
    keep = ~key.str.lower().isin(_NULL_IDS)

    out = pd.DataFrame(index=df.index)
    out["_case"] = key
    out["_measure"] = (coerce_numeric(df[MEASURE]).fillna(0.0)
                       if MEASURE in df.columns else 0.0)
    out["_stage"] = (df[STAGE].astype(str).str.strip().str.upper()
                     if STAGE in df.columns else "")
    for d in dims:
        out[d] = (df[d].astype(str).str.strip() if d in df.columns else "")
    out = out[keep]

    agg = {"_measure": ("_measure", "sum"), "_stage": ("_stage", "last")}
    for d in dims:
        agg[d] = (d, "last")
    grouped = out.groupby("_case", as_index=True, sort=True).agg(**agg)
    return grouped


def _unkeyed(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Rows that carry no usable case identifier, and what they are worth.

    Reported, never folded into a contributor: they cannot be matched across
    periods, so attributing their balance to anything would be a guess.
    """
    if df is None or df.empty or CASE_KEY not in df.columns:
        return {"cases": 0, "amount": 0.0}
    key = df[CASE_KEY].astype(str).str.strip()
    bad = key.str.lower().isin(_NULL_IDS)
    amount = (float(coerce_numeric(df.loc[bad, MEASURE]).fillna(0.0).sum())
              if MEASURE in df.columns else 0.0)
    return {"cases": int(bad.sum()), "amount": round(amount, 2)}


def _duplicates(df: Optional[pd.DataFrame]) -> int:
    if df is None or df.empty or CASE_KEY not in df.columns:
        return 0
    key = df[CASE_KEY].astype(str).str.strip()
    key = key[~key.str.lower().isin(_NULL_IDS)]
    return int(len(key) - key.nunique())


# --------------------------------------------------------------------------- #
# Per-case decomposition
# --------------------------------------------------------------------------- #
def movement_components(current: Optional[pd.DataFrame],
                        prior: Optional[pd.DataFrame],
                        *,
                        dims: Sequence[str],
                        terminal_stages: Sequence[str] = TERMINAL_STAGES,
                        ) -> pd.DataFrame:
    """Per-case ``delta``, ``component`` and resolved dimension values.

    The returned frame is the single source for every number in the payload:
    the headline is its ``delta`` sum, the component summary groups it by
    ``component``, and each contributor list groups it by a dimension column.
    They therefore cannot disagree.
    """
    cur = _case_level(current, dims)
    pri = _case_level(prior, dims)

    joined = cur.join(pri, how="outer", rsuffix="_prior")
    cur_m = joined["_measure"].astype("float64")
    pri_m = joined["_measure_prior"].astype("float64")

    in_cur = cur_m.notna()
    in_pri = pri_m.notna()
    delta = cur_m.fillna(0.0) - pri_m.fillna(0.0)

    # Dimension resolution: current value, falling back to prior for a case
    # that has left. Blank after both -> Unknown, never dropped.
    out = pd.DataFrame({"delta": delta}, index=joined.index)
    for d in dims:
        cur_d = joined[d] if d in joined.columns else pd.Series("", index=joined.index)
        pri_d = (joined[f"{d}_prior"] if f"{d}_prior" in joined.columns
                 else pd.Series("", index=joined.index))
        resolved = cur_d.where(cur_d.notna() & (cur_d != ""), pri_d)
        out[d] = resolved.where(resolved.notna() & (resolved != ""), UNKNOWN)
        # A reassignment is only meaningful for a case present in BOTH periods
        # with a non-blank value on each side.
        both = in_cur & in_pri & cur_d.notna() & pri_d.notna() \
            & (cur_d != "") & (pri_d != "")
        out[f"_{d}_reassigned"] = both & (cur_d != pri_d)

    terminal = tuple(s.upper() for s in terminal_stages)
    cur_s = joined["_stage"] if "_stage" in joined.columns else pd.Series("", index=joined.index)
    pri_s = (joined["_stage_prior"] if "_stage_prior" in joined.columns
             else pd.Series("", index=joined.index))
    was_active = in_pri & ~pri_s.fillna("").isin(terminal)
    now_terminal = in_cur & cur_s.fillna("").isin(terminal)

    # Successive masks, most specific first, so every case lands in exactly one
    # component (the same discipline as the governed probability hierarchy).
    component = pd.Series("unchanged", index=joined.index, dtype=object)
    remaining = pd.Series(True, index=joined.index)

    is_new = in_cur & ~in_pri
    component[is_new] = "new"
    remaining &= ~is_new

    is_removed = in_pri & ~in_cur
    component[is_removed] = "removed"
    remaining &= ~is_removed

    is_out = remaining & was_active & now_terminal
    component[is_out] = "progressed_out"
    remaining &= ~is_out

    up = remaining & (delta > 0)
    component[up] = "increased"
    remaining &= ~up

    down = remaining & (delta < 0)
    component[down] = "decreased"
    remaining &= ~down
    # whatever is left is "unchanged" (in both, no terminal transition, delta 0)

    out["component"] = component
    return out


def component_summary(components: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """``{component: {amount, cases}}`` for every component, including zeros.

    Every component is always present so a consumer never has to distinguish
    "no cases moved out" from "this build does not report movement out".
    """
    summary: Dict[str, Dict[str, Any]] = {}
    if components.empty:
        return {c: {"amount": 0.0, "cases": 0} for c in COMPONENTS}
    grouped = components.groupby("component")["delta"].agg(["sum", "size"])
    for c in COMPONENTS:
        if c in grouped.index:
            summary[c] = {"amount": round(float(grouped.loc[c, "sum"]), 2),
                          "cases": int(grouped.loc[c, "size"])}
        else:
            summary[c] = {"amount": 0.0, "cases": 0}
    return summary


# --------------------------------------------------------------------------- #
# Contributor ranking
# --------------------------------------------------------------------------- #
def rank_contributors(components: pd.DataFrame, dim: str, *,
                      total: float, top_n: int = 3) -> List[Dict[str, Any]]:
    """The ``top_n`` contributors to the movement along ``dim``.

    These explain the CHANGE — they are deltas, not current balances — so a
    contributor's amount may be negative, and the largest contributor is the
    one that moved the number most in either direction.

    Deterministic: ordered by ``|amount|`` descending, ties broken by name
    ascending. ``share_of_change_pct`` is ``None`` when the total movement is
    zero, rather than an infinity or a misleading 0.
    """
    if components.empty or dim not in components.columns:
        return []
    grouped = components.groupby(dim, sort=False).agg(
        amount=("delta", "sum"), case_count=("delta", "size"))
    grouped = grouped.reset_index().rename(columns={dim: "name"})
    grouped["_abs"] = grouped["amount"].abs()
    grouped = grouped.sort_values(["_abs", "name"], ascending=[False, True])
    rows: List[Dict[str, Any]] = []
    for r in grouped.head(top_n).itertuples(index=False):
        amount = round(float(r.amount), 2)
        rows.append({
            "name": str(r.name),
            "amount": amount,
            "share_of_change_pct": (round(amount / total * 100, 1)
                                    if total not in (0, 0.0) else None),
            "case_count": int(r.case_count),
        })
    return rows


def reassignment_counts(components: pd.DataFrame,
                        dims: Sequence[str]) -> Dict[str, int]:
    """How many cases changed each dimension between the two periods.

    The attribution convention has to pick a side; this says how often that
    choice was actually load-bearing.
    """
    out: Dict[str, int] = {}
    for d in dims:
        col = f"_{d}_reassigned"
        out[d] = int(components[col].sum()) if col in components.columns else 0
    return out


# --------------------------------------------------------------------------- #
# Payload assembly
# --------------------------------------------------------------------------- #
def _filter_completed(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Rows sitting at the COMPLETED stage — the funnel's completion measure."""
    if df is None or df.empty or STAGE not in df.columns:
        return df
    mask = df[STAGE].astype(str).str.strip().str.upper() == "COMPLETED"
    return df[mask]


_HEADLINE = {
    DETAIL_PIPELINE: {
        "label": "Pipeline balance",
        "metric_definition": (
            "Total open pipeline exposure (current_outstanding_balance) across "
            "all cases in the governed weekly extract. Same measure and same "
            "prepared frames as the pipeline evolution series."),
    },
    DETAIL_COMPLETIONS: {
        "label": "Completed case value",
        "metric_definition": (
            "Balance of cases sitting at the COMPLETED pipeline stage in the "
            "governed weekly extract. This is a pipeline-stage measure — cases "
            "reaching completed stage — and is NOT funded balance growth."),
    },
}


def build_movement_detail(detail_type: str,
                          current: Optional[pd.DataFrame],
                          prior: Optional[pd.DataFrame],
                          *,
                          as_of_date: Optional[str],
                          comparison_date: Optional[str],
                          portfolio_id: str,
                          scope: Optional[str] = None,
                          run_id: Optional[str] = None,
                          source_file: Optional[str] = None,
                          comparison_source_file: Optional[str] = None,
                          top_n: int = 3,
                          ) -> Dict[str, Any]:
    """The governed movement-detail payload for one weekly point.

    Deliberately small and flat: it is designed to be consumed later by a drill
    panel, a Teams card or investor commentary without any of them needing to
    recompute anything — but it is not a general insight model, and it carries
    no loan-level rows.
    """
    if detail_type not in _HEADLINE:
        raise ValueError(f"unknown detail_type {detail_type!r}")

    if detail_type == DETAIL_COMPLETIONS:
        current, prior = _filter_completed(current), _filter_completed(prior)
        terminal: Sequence[str] = ()      # already terminal; no "moved out" step
    else:
        terminal = TERMINAL_STAGES

    dims = [col for _key, col in DIMENSIONS]
    components = movement_components(current, prior, dims=dims,
                                     terminal_stages=terminal)

    total = round(float(components["delta"].sum()), 2) if not components.empty else 0.0
    cur_cases = int(len(current)) if current is not None else 0
    pri_cases = int(len(prior)) if prior is not None else 0

    cur_value = (float(coerce_numeric(current[MEASURE]).fillna(0.0).sum())
                 if current is not None and not current.empty
                 and MEASURE in current.columns else 0.0)
    pri_value = cur_value - total

    contributors = {
        key: rank_contributors(components, col, total=total, top_n=top_n)
        for key, col in DIMENSIONS
    }

    return {
        "detail_type": detail_type,
        "portfolio_id": portfolio_id,
        "scope": scope or "total",
        "run_id": run_id,
        "as_of_date": as_of_date,
        "comparison_date": comparison_date,
        "available": bool(comparison_date) and not components.empty,
        "headline_metric": {
            "label": _HEADLINE[detail_type]["label"],
            "value": round(cur_value, 2),
            "change": total,
            "change_pct": (round(total / abs(pri_value) * 100, 1)
                           if pri_value else None),
        },
        "counts": {
            "current": cur_cases,
            "comparison": pri_cases,
            "change": cur_cases - pri_cases,
        },
        "contributors": contributors,
        "components": component_summary(components),
        "methodology": {
            "metric_definition": _HEADLINE[detail_type]["metric_definition"],
            "movement_basis": "net",
            "attribution": ATTRIBUTION,
            "version": METHODOLOGY_VERSION,
            "dimension_reassignments": reassignment_counts(components, dims),
            "unmatched_current": _unkeyed(current),
            "unmatched_comparison": _unkeyed(prior),
            "duplicate_case_identifiers": {
                "current": _duplicates(current),
                "comparison": _duplicates(prior),
            },
        },
        "source_dates": {
            "pipeline_as_of": as_of_date,
            "pipeline_comparison": comparison_date,
            "funded_as_of": None,
            "forecast_generated_at": None,
        },
        "sources": {
            "current": source_file,
            "comparison": comparison_source_file,
        },
    }
