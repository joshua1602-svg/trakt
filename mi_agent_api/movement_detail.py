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

The attribution engine performs NO I/O — it is handed two already-prepared
frames, which keeps it testable and reusable. Frame RESOLUTION is a separate,
clearly marked section at the end of this file.
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


# --------------------------------------------------------------------------- #
# Feature flag
# --------------------------------------------------------------------------- #
#: Off unless a deployment explicitly turns it on. Same mechanism as the Phase
#: 1A kill switches — no new feature-flag framework for one phase.
FLAG_ENV = "TRAKT_MI_ENHANCED_HOVERS"
_ON = ("1", "true", "on", "yes", "enabled")


def enhanced_hovers_enabled() -> bool:
    """True only when this deployment has explicitly enabled the hover layer.

    Read per call rather than cached at import so a test (and an operator
    restart-free toggle) can change it without reloading the module.
    """
    import os
    return (os.environ.get(FLAG_ENV) or "").strip().lower() in _ON


# --------------------------------------------------------------------------- #
# Resolution — the only part of this module that touches storage
# --------------------------------------------------------------------------- #
def select_pair(extracts: List[Dict[str, Any]],
                as_of: Optional[str] = None
                ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """``(current, comparison)`` weekly extracts for ``as_of``.

    ``as_of`` picks the point the user hovered; without it the latest extract is
    used. The comparison is the immediately preceding governed extract — the
    same neighbour the chart draws its line between, so the movement shown is
    the movement plotted. Returns ``(current, None)`` for the first extract in
    the series, which has nothing to compare against.
    """
    dated = sorted((e for e in extracts if e.get("pipeline_extract_date")),
                   key=lambda e: e["pipeline_extract_date"])
    if not dated:
        return None, None
    if as_of:
        idx = next((i for i, e in enumerate(dated)
                    if e["pipeline_extract_date"] == as_of), None)
        if idx is None:
            return None, None
    else:
        idx = len(dated) - 1
    return dated[idx], (dated[idx - 1] if idx > 0 else None)


def resolve_movement_detail(root: str, client_id: str, detail_type: str, *,
                            as_of: Optional[str] = None,
                            historical_model: Optional[Dict[str, Any]] = None,
                            scope: Optional[str] = None,
                            top_n: int = 3) -> Dict[str, Any]:
    """Build the detail for one weekly point from the governed weekly extracts.

    ``historical_model`` is passed straight through to
    ``load_prepared_pipeline`` for one reason only: it makes this reuse the
    frames the pipeline evolution and funnel views have ALREADY prepared and
    cached, instead of preparing every extract a second time under a different
    cache key (the defect Phase 1B-1 removed). It cannot change any number here
    — the model affects only the probability / weighted-expectation columns, and
    nothing in this module reads them.
    """
    from . import pipeline_contract as pipeline_mod

    inv = pipeline_mod.weekly_extract_inventory(root, client_id)
    cur_e, pri_e = select_pair(inv.get("extracts", []), as_of)
    if cur_e is None:
        return unavailable(detail_type, client_id, scope=scope, as_of=as_of,
                           reason="No governed weekly pipeline extract matches "
                                  "this point.")
    if pri_e is None:
        return unavailable(
            detail_type, client_id, scope=scope,
            as_of=cur_e.get("pipeline_extract_date"),
            reason="This is the earliest governed weekly extract, so there is "
                   "no prior week to compare it against.")

    cur, _ = pipeline_mod.load_prepared_pipeline(
        cur_e, historical_model=historical_model)
    pri, _ = pipeline_mod.load_prepared_pipeline(
        pri_e, historical_model=historical_model)

    return build_movement_detail(
        detail_type, cur, pri,
        as_of_date=cur_e.get("pipeline_extract_date"),
        comparison_date=pri_e.get("pipeline_extract_date"),
        portfolio_id=client_id, scope=scope,
        run_id=cur_e.get("run_id"),
        source_file=_basename(cur_e.get("source_file")),
        comparison_source_file=_basename(pri_e.get("source_file")),
        top_n=top_n)


def _basename(path: Optional[str]) -> Optional[str]:
    """The file name only — never a storage path, which is deployment detail."""
    if not path:
        return None
    return str(path).replace("\\", "/").rsplit("/", 1)[-1] or None


def unavailable(detail_type: str, portfolio_id: str, *,
                scope: Optional[str] = None, as_of: Optional[str] = None,
                reason: str) -> Dict[str, Any]:
    """A controlled "no detail" envelope — never a partial or invented one."""
    return {
        "detail_type": detail_type,
        "portfolio_id": portfolio_id,
        "scope": scope or "total",
        "as_of_date": as_of,
        "comparison_date": None,
        "available": False,
        "reason": reason,
        "headline_metric": None,
        "counts": None,
        "contributors": {key: [] for key, _col in DIMENSIONS},
        "components": None,
        "methodology": {"movement_basis": "net", "attribution": ATTRIBUTION,
                        "version": METHODOLOGY_VERSION},
        "source_dates": {"pipeline_as_of": as_of, "pipeline_comparison": None,
                         "funded_as_of": None, "forecast_generated_at": None},
    }


# --------------------------------------------------------------------------- #
# Governed pipeline stage transitions
# --------------------------------------------------------------------------- #
"""Gross source -> destination transitions between two governed snapshots.

The decomposition above is NET: it answers "how much did the number move, and
who moved it". It cannot answer a GROSS question. An OFFER stock that falls from
three cases to one is a net -2, and that is equally consistent with two leaving
and none arriving, or with four leaving and two arriving. Those are different
businesses.

This section publishes the missing case-level classification, on the SAME two
prepared frames, joined on the SAME governed key by the SAME ``_case_level``
helper. It defines no metric, owns no existing number, and changes nothing above
it: every case is classified exactly once, and the classification reconciles
back to the two stock levels the pipeline views already plot.

    prior-only              -> departure
    latest-only             -> new_arrival
    both, same stage        -> stayer
    both, different stage   -> stage_transition

Identity is the governed ``pipeline_case_identifier`` and nothing else. An
amount amendment is an attribute of a case, never evidence about which case it
is: a case that goes KFI GBP200k -> APPLICATION GBP220k is ONE case, one
KFI->APPLICATION transition, +GBP20k — never a departure plus an arrival.
"""

#: Detail type for the transition capability.
DETAIL_STAGE_TRANSITION = "PIPELINE_STAGE_TRANSITION"

#: The four mutually exclusive, collectively exhaustive event classes.
EVENT_NEW_ARRIVAL = "new_arrival"
EVENT_STAYER = "stayer"
EVENT_STAGE_TRANSITION = "stage_transition"
EVENT_DEPARTURE = "departure"
EVENT_CLASSES = (EVENT_NEW_ARRIVAL, EVENT_STAYER, EVENT_STAGE_TRANSITION,
                 EVENT_DEPARTURE)

#: A departure the governed data cannot explain. NOT a withdrawal: absence from
#: the latest extract is not evidence of an outcome, and inventing one here
#: would put a number behind a guess.
UNCLASSIFIED_DEPARTURE = "unclassified_departure"

#: How a departure's outcome was established.
EVIDENCE_PRIOR_TERMINAL = "prior_terminal_stage"
EVIDENCE_NONE = "none"

#: Sentinel the prepared frame already uses for a stage it could not recognise
#: (``pipeline_prep.canonical_stage``). Reused, never redefined.
UNKNOWN_STAGE = "UNKNOWN"

#: Monetary tolerance for a reconciliation residual — floating-point only.
AMOUNT_TOLERANCE = 0.01

#: Bumped when any definition in the transition section changes.
TRANSITION_METHODOLOGY_VERSION = "1"

#: Availability reason codes, using the existing governed diagnostic names.
REASON_NO_COMPARISON = "no_prior_snapshot"
REASON_MISSING_IDENTIFIER = "missing_case_identifier"
REASON_DUPLICATE_IDENTIFIERS = "duplicate_case_identifiers"
REASON_NO_CASES = "no_governed_cases"

_COUNT_RECONCILIATION = ("opening_case_count + new_arrivals + transitions_in "
                         "- transitions_out - departures = closing_case_count")
_AMOUNT_RECONCILIATION = (
    "opening_amount + new_arrival_amount + transferred_in_latest_amount "
    "- transferred_out_prior_amount - departure_prior_amount "
    "+ stayer_amount_change = closing_amount")


def _keyed_case_count(df: Optional[pd.DataFrame]) -> int:
    """Rows carrying a usable governed identifier — the matchable population."""
    if df is None or df.empty or CASE_KEY not in df.columns:
        return 0
    key = df[CASE_KEY].astype(str).str.strip()
    return int((~key.str.lower().isin(_NULL_IDS)).sum())


def _identifier_usable(df: Optional[pd.DataFrame]) -> bool:
    """Whether a frame's cases can be matched on the governed key.

    Mirrors the severity the preparation layer already applies: the
    ``missing_case_identifier`` BLOCKER fires only when the column is absent or
    blank for EVERY row. A partially blank column is a reported exclusion (see
    ``unmatched_*`` in the payload), not a refusal.

    A frame with no rows at all is not an identifier defect — an empty snapshot
    beside a populated one is a governed situation (everything departed), and
    classifying it as a missing identifier would refuse a question the data can
    actually answer.
    """
    if df is None or df.empty:
        return True
    return _keyed_case_count(df) > 0


def stage_transition_events(current: Optional[pd.DataFrame],
                            prior: Optional[pd.DataFrame],
                            *,
                            terminal_stages: Sequence[str] = TERMINAL_STAGES,
                            ) -> pd.DataFrame:
    """One row per governed case, classified into exactly one event class.

    The single source for every number in the transition payload — the
    aggregations below only group this frame, so they cannot disagree with it or
    with each other.

    Columns: ``event_class``, ``source_stage``, ``destination_stage``,
    ``prior_amount``, ``latest_amount``, ``amount_change``, ``governed_outcome``,
    ``outcome_evidence``. Indexed by the governed case identifier.

    ``source_stage`` is ``None`` for a new arrival and ``destination_stage`` is
    ``None`` for a departure, deliberately: neither has one, and a synthetic
    stage token here would be indistinguishable downstream from a real governed
    stage. A presentation layer may draw a new arrival from a synthetic source;
    the engine must not pretend it had one.
    """
    lat = _case_level(current, ())
    pri = _case_level(prior, ())

    joined = lat.join(pri, how="outer", rsuffix="_prior")
    lat_m = joined["_measure"].astype("float64")
    pri_m = joined["_measure_prior"].astype("float64")
    in_lat, in_pri = lat_m.notna(), pri_m.notna()

    def _stage(col: str) -> pd.Series:
        s = (joined[col] if col in joined.columns
             else pd.Series(pd.NA, index=joined.index, dtype=object))
        s = s.astype(object).where(s.notna(), "")
        return s.mask(s.astype(str).str.strip() == "", UNKNOWN_STAGE).astype(str)

    lat_s, pri_s = _stage("_stage"), _stage("_stage_prior")

    is_new = in_lat & ~in_pri
    is_gone = in_pri & ~in_lat
    in_both = in_lat & in_pri
    is_stayer = in_both & (lat_s == pri_s)
    is_move = in_both & (lat_s != pri_s)

    event = pd.Series(EVENT_STAYER, index=joined.index, dtype=object)
    event[is_new] = EVENT_NEW_ARRIVAL
    event[is_gone] = EVENT_DEPARTURE
    event[is_move] = EVENT_STAGE_TRANSITION

    out = pd.DataFrame(index=joined.index)
    out.index.name = "_case"
    out["event_class"] = event
    # A new arrival has no prior stage and a departure has no latest stage;
    # both stay None rather than borrowing the other side's value.
    out["source_stage"] = pri_s.where(in_pri, None)
    out["destination_stage"] = lat_s.where(in_lat, None)
    out["prior_amount"] = pri_m
    out["latest_amount"] = lat_m
    out["amount_change"] = (lat_m - pri_m).where(in_both)

    # Departure outcome: the governed terminal stage the case was LAST RECORDED
    # at. Absence from the latest extract is not itself evidence of anything, so
    # a case that vanishes from an active stage stays unclassified.
    terminal = tuple(str(s).strip().upper() for s in terminal_stages)
    resolved = is_gone & pri_s.isin(terminal)
    outcome = pd.Series(None, index=joined.index, dtype=object)
    outcome[is_gone] = UNCLASSIFIED_DEPARTURE
    outcome[resolved] = pri_s[resolved]
    evidence = pd.Series(None, index=joined.index, dtype=object)
    evidence[is_gone] = EVIDENCE_NONE
    evidence[resolved] = EVIDENCE_PRIOR_TERMINAL
    out["governed_outcome"] = outcome
    out["outcome_evidence"] = evidence
    return out


def _amount(value: Any) -> float:
    """A rounded monetary scalar, with NaN read as absence rather than zero."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else round(f, 2)


def _stage_sort_key(stage: Optional[str]) -> Tuple[int, str]:
    """Funnel position from the one governed stage model; unknowns sort last."""
    from .pipeline_prep import canonical_stage_order
    order = canonical_stage_order()
    s = "" if stage is None else str(stage)
    return (order.index(s) if s in order else len(order), s)


def transition_matrix(events: pd.DataFrame) -> List[Dict[str, Any]]:
    """``source_stage -> destination_stage`` totals for cases in both snapshots.

    The key new capability: a GROSS movement, so a stage's arrivals and
    departures are separately visible instead of netting each other off.
    """
    moves = events[events["event_class"] == EVENT_STAGE_TRANSITION]
    if moves.empty:
        return []
    grouped = moves.groupby(["source_stage", "destination_stage"], sort=False).agg(
        case_count=("event_class", "size"),
        prior_amount=("prior_amount", "sum"),
        latest_amount=("latest_amount", "sum"))
    rows = [{
        "source_stage": str(src),
        "destination_stage": str(dst),
        "case_count": int(r.case_count),
        "prior_amount": _amount(r.prior_amount),
        "latest_amount": _amount(r.latest_amount),
        "amount_change": _amount(r.latest_amount - r.prior_amount),
    } for (src, dst), r in zip(grouped.index, grouped.itertuples(index=False))]
    return sorted(rows, key=lambda x: (_stage_sort_key(x["source_stage"]),
                                       _stage_sort_key(x["destination_stage"])))


def new_arrival_summary(events: pd.DataFrame) -> List[Dict[str, Any]]:
    """New arrivals by the stage they entered at."""
    arrivals = events[events["event_class"] == EVENT_NEW_ARRIVAL]
    if arrivals.empty:
        return []
    grouped = arrivals.groupby("destination_stage", sort=False).agg(
        case_count=("event_class", "size"), latest_amount=("latest_amount", "sum"))
    rows = [{"destination_stage": str(stage),
             "case_count": int(r.case_count),
             "latest_amount": _amount(r.latest_amount)}
            for stage, r in zip(grouped.index, grouped.itertuples(index=False))]
    return sorted(rows, key=lambda x: _stage_sort_key(x["destination_stage"]))


def stayer_summary(events: pd.DataFrame) -> List[Dict[str, Any]]:
    """Cases in both snapshots at the same stage, with any amount amendment."""
    stayers = events[events["event_class"] == EVENT_STAYER]
    if stayers.empty:
        return []
    grouped = stayers.groupby("destination_stage", sort=False).agg(
        case_count=("event_class", "size"),
        prior_amount=("prior_amount", "sum"),
        latest_amount=("latest_amount", "sum"))
    rows = [{"stage": str(stage),
             "case_count": int(r.case_count),
             "prior_amount": _amount(r.prior_amount),
             "latest_amount": _amount(r.latest_amount),
             "amount_change": _amount(r.latest_amount - r.prior_amount)}
            for stage, r in zip(grouped.index, grouped.itertuples(index=False))]
    return sorted(rows, key=lambda x: _stage_sort_key(x["stage"]))


def departure_summary(events: pd.DataFrame) -> List[Dict[str, Any]]:
    """Departures by the stage they left from and the outcome the data supports.

    ``governed_outcome`` is a canonical terminal stage only where the prior
    extract recorded one. Everything else stays ``unclassified_departure`` —
    visible, and never resolved into a withdrawal the data does not evidence.
    """
    gone = events[events["event_class"] == EVENT_DEPARTURE]
    if gone.empty:
        return []
    grouped = gone.groupby(["source_stage", "governed_outcome", "outcome_evidence"],
                           sort=False).agg(
        case_count=("event_class", "size"), prior_amount=("prior_amount", "sum"))
    rows = [{"source_stage": str(src),
             "governed_outcome": str(outcome),
             "outcome_evidence": str(evidence),
             "case_count": int(r.case_count),
             "prior_amount": _amount(r.prior_amount)}
            for (src, outcome, evidence), r
            in zip(grouped.index, grouped.itertuples(index=False))]
    return sorted(rows, key=lambda x: (_stage_sort_key(x["source_stage"]),
                                       x["governed_outcome"]))


def event_totals(events: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """``{event_class: {case_count, prior_amount, latest_amount}}``, zeros included.

    Every class is always present, so a consumer never has to distinguish "no
    departures" from "this build does not report departures".
    """
    out: Dict[str, Dict[str, Any]] = {}
    for cls in EVENT_CLASSES:
        rows = events[events["event_class"] == cls] if not events.empty else events
        out[cls] = {
            "case_count": int(len(rows)),
            "prior_amount": _amount(rows["prior_amount"].sum()) if len(rows) else 0.0,
            "latest_amount": _amount(rows["latest_amount"].sum()) if len(rows) else 0.0,
        }
    return out


def stage_reconciliation(events: pd.DataFrame) -> List[Dict[str, Any]]:
    """Per-stage proof that the classification returns the two stock levels.

    COUNTS — exact integer equality:

        opening + new_arrivals + transitions_in
                - transitions_out - departures = closing

    AMOUNTS — the same identity, with each side of a transition carrying the
    amount it actually had in that snapshot, so an amendment on a moving case
    lands with the case at its destination:

        opening_amount + new_arrival_amount + transferred_in_latest_amount
                       - transferred_out_prior_amount - departure_prior_amount
                       + stayer_amount_change = closing_amount

    Both residuals are published rather than asserted away. A non-zero residual
    is a finding about the data or this code, and hiding it would make the
    capability worthless.
    """
    if events.empty:
        return []
    e = events
    moves = e[e["event_class"] == EVENT_STAGE_TRANSITION]
    arrivals = e[e["event_class"] == EVENT_NEW_ARRIVAL]
    stayers = e[e["event_class"] == EVENT_STAYER]
    gone = e[e["event_class"] == EVENT_DEPARTURE]

    # Opening / closing stock read straight off the per-case frame, so they are
    # the same populations the payload's own event lists are built from.
    opened = e[e["prior_amount"].notna()]
    closed = e[e["latest_amount"].notna()]

    def _count(frame: pd.DataFrame, col: str) -> Dict[str, int]:
        if frame.empty:
            return {}
        return {str(k): int(v) for k, v in frame.groupby(col, sort=False).size().items()}

    def _sum(frame: pd.DataFrame, col: str, amount: str) -> Dict[str, float]:
        if frame.empty:
            return {}
        return {str(k): float(v) for k, v
                in frame.groupby(col, sort=False)[amount].sum().items()}

    open_n, close_n = _count(opened, "source_stage"), _count(closed, "destination_stage")
    arr_n, stay_n = _count(arrivals, "destination_stage"), _count(stayers, "destination_stage")
    in_n, out_n = _count(moves, "destination_stage"), _count(moves, "source_stage")
    dep_n = _count(gone, "source_stage")

    open_a = _sum(opened, "source_stage", "prior_amount")
    close_a = _sum(closed, "destination_stage", "latest_amount")
    arr_a = _sum(arrivals, "destination_stage", "latest_amount")
    in_a = _sum(moves, "destination_stage", "latest_amount")
    out_a = _sum(moves, "source_stage", "prior_amount")
    dep_a = _sum(gone, "source_stage", "prior_amount")
    stay_d = _sum(stayers, "destination_stage", "amount_change")

    stages = sorted(set(open_n) | set(close_n) | set(arr_n) | set(in_n)
                    | set(out_n) | set(dep_n) | set(stay_n), key=_stage_sort_key)
    rows: List[Dict[str, Any]] = []
    for s in stages:
        opening, closing = open_n.get(s, 0), close_n.get(s, 0)
        arrived, moved_in = arr_n.get(s, 0), in_n.get(s, 0)
        moved_out, departed = out_n.get(s, 0), dep_n.get(s, 0)
        o_amt, c_amt = open_a.get(s, 0.0), close_a.get(s, 0.0)
        a_amt, i_amt = arr_a.get(s, 0.0), in_a.get(s, 0.0)
        o_out, d_amt = out_a.get(s, 0.0), dep_a.get(s, 0.0)
        change = stay_d.get(s, 0.0)
        rows.append({
            "stage": s,
            "opening_case_count": opening,
            "new_arrivals": arrived,
            "transitions_in": moved_in,
            "transitions_out": moved_out,
            "departures": departed,
            "stayers": stay_n.get(s, 0),
            "closing_case_count": closing,
            "count_reconciliation_residual":
                (opening + arrived + moved_in - moved_out - departed) - closing,
            "opening_amount": _amount(o_amt),
            "new_arrival_amount": _amount(a_amt),
            "transferred_in_latest_amount": _amount(i_amt),
            "transferred_out_prior_amount": _amount(o_out),
            "departure_prior_amount": _amount(d_amt),
            "stayer_amount_change": _amount(change),
            "closing_amount": _amount(c_amt),
            "amount_reconciliation_residual":
                round((o_amt + a_amt + i_amt - o_out - d_amt + change) - c_amt, 2),
        })
    return rows


def global_reconciliation(events: pd.DataFrame) -> Dict[str, Any]:
    """Every governed identifier belongs to exactly one event class.

    ``prior-only + both + latest-only`` must be the union of the two snapshots'
    identifiers, and the four event classes must partition that same union. No
    case may disappear, and none may be counted twice.
    """
    if events.empty:
        return {"prior_population": 0, "latest_population": 0, "union_population": 0,
                "prior_only": 0, "in_both": 0, "latest_only": 0,
                "classified_events": 0, "duplicate_classifications": 0,
                "residual": 0}
    in_pri = events["prior_amount"].notna()
    in_lat = events["latest_amount"].notna()
    counts = events["event_class"].value_counts()
    classified = int(sum(int(counts.get(c, 0)) for c in EVENT_CLASSES))
    union = int(len(events.index))
    return {
        "prior_population": int(in_pri.sum()),
        "latest_population": int(in_lat.sum()),
        "union_population": union,
        "prior_only": int((in_pri & ~in_lat).sum()),
        "in_both": int((in_pri & in_lat).sum()),
        "latest_only": int((in_lat & ~in_pri).sum()),
        "classified_events": classified,
        # A case can only carry one event_class value, so a duplicate would mean
        # a duplicate identifier survived — which the capability refuses above.
        "duplicate_classifications": union - int(events.index.nunique()),
        "residual": classified - union,
    }


def build_stage_transition_detail(current: Optional[pd.DataFrame],
                                  prior: Optional[pd.DataFrame],
                                  *,
                                  as_of_date: Optional[str],
                                  comparison_date: Optional[str],
                                  portfolio_id: str,
                                  scope: Optional[str] = None,
                                  run_id: Optional[str] = None,
                                  source_file: Optional[str] = None,
                                  comparison_source_file: Optional[str] = None,
                                  ) -> Dict[str, Any]:
    """The governed stage-transition payload for one snapshot pair.

    Additive: it publishes a NEW capability and takes ownership of no existing
    metric. Pipeline stock, stage stock, evolution, weighted expectation,
    forecast, funnel and conversion are untouched and still computed where they
    always were — this reads the same prepared frames they do.

    Carries no case identifiers, in line with the movement payload above.
    """
    if not comparison_date or prior is None:
        return stage_transition_unavailable(
            portfolio_id, scope=scope, as_of=as_of_date,
            reason_code=REASON_NO_COMPARISON,
            reason="There is no prior governed pipeline snapshot to compare "
                   "this one against.")

    if not (_identifier_usable(current) and _identifier_usable(prior)):
        return stage_transition_unavailable(
            portfolio_id, scope=scope, as_of=as_of_date,
            comparison_date=comparison_date,
            reason_code=REASON_MISSING_IDENTIFIER,
            reason=f"{CASE_KEY} is absent or blank for every row in at least "
                   "one snapshot, so cases cannot be matched across them.")

    if not (_keyed_case_count(current) or _keyed_case_count(prior)):
        return stage_transition_unavailable(
            portfolio_id, scope=scope, as_of=as_of_date,
            comparison_date=comparison_date,
            reason_code=REASON_NO_CASES,
            reason="Neither governed snapshot carries a matchable pipeline "
                   "case, so there is no population to classify.")

    dupes = {"current": _duplicates(current), "comparison": _duplicates(prior)}
    if dupes["current"] or dupes["comparison"]:
        return stage_transition_unavailable(
            portfolio_id, scope=scope, as_of=as_of_date,
            comparison_date=comparison_date,
            reason_code=REASON_DUPLICATE_IDENTIFIERS,
            reason=f"{dupes['current']} duplicate {CASE_KEY} value(s) in the "
                   f"latest snapshot and {dupes['comparison']} in the prior "
                   "snapshot prevent deterministic case matching.",
            duplicates=dupes)

    events = stage_transition_events(current, prior)
    by_stage = stage_reconciliation(events)
    counts = {"current": _keyed_case_count(current),
              "comparison": _keyed_case_count(prior)}
    counts["change"] = counts["current"] - counts["comparison"]

    return {
        "detail_type": DETAIL_STAGE_TRANSITION,
        "portfolio_id": portfolio_id,
        "scope": scope or "total",
        "run_id": run_id,
        "as_of_date": as_of_date,
        "comparison_date": comparison_date,
        "available": True,
        "reason": None,
        "reason_code": None,
        "identifier": CASE_KEY,
        "measure": MEASURE,
        "stage_field": STAGE,
        "counts": counts,
        "transitions": transition_matrix(events),
        "new_arrivals": new_arrival_summary(events),
        "stayers": stayer_summary(events),
        "departures": departure_summary(events),
        "event_totals": event_totals(events),
        "reconciliation": {
            "by_stage": by_stage,
            "count_reconciliation_residual":
                sum(abs(r["count_reconciliation_residual"]) for r in by_stage),
            "amount_reconciliation_residual":
                round(sum(abs(r["amount_reconciliation_residual"])
                          for r in by_stage), 2),
            "global": global_reconciliation(events),
            "amount_tolerance": AMOUNT_TOLERANCE,
            "count_identity": _COUNT_RECONCILIATION,
            "amount_identity": _AMOUNT_RECONCILIATION,
        },
        "methodology": {
            "capability_definition": (
                "Gross case-level stage transitions between the latest governed "
                "pipeline snapshot and the immediately prior one, matched on "
                f"{CASE_KEY}. Every case is classified exactly once as a new "
                "arrival, a stayer, a stage transition or a departure. This "
                "capability owns no existing pipeline metric."),
            "movement_basis": "gross",
            "identity_basis": CASE_KEY,
            "identity_note": ("An amount amendment never changes case identity: "
                              "a case whose stage and amount both change stays "
                              "one case with one transition and one amount "
                              "change."),
            "stage_vocabulary": "mi_agent_api.pipeline_prep.canonical_stage",
            "terminal_stages": list(TERMINAL_STAGES),
            "departure_outcome_basis": (
                "The governed terminal stage the case was last recorded at. "
                "Absence from the latest snapshot is NOT treated as evidence of "
                f"an outcome; such a case is reported as {UNCLASSIFIED_DEPARTURE}."),
            "version": TRANSITION_METHODOLOGY_VERSION,
            "unmatched_current": _unkeyed(current),
            "unmatched_comparison": _unkeyed(prior),
            "duplicate_case_identifiers": dupes,
        },
        "source_dates": {
            "pipeline_as_of": as_of_date,
            "pipeline_comparison": comparison_date,
            "funded_as_of": None,
            "forecast_generated_at": None,
        },
        "sources": {"current": source_file, "comparison": comparison_source_file},
    }


def stage_transition_unavailable(portfolio_id: str, *,
                                 scope: Optional[str] = None,
                                 as_of: Optional[str] = None,
                                 comparison_date: Optional[str] = None,
                                 reason_code: str,
                                 reason: str,
                                 duplicates: Optional[Dict[str, int]] = None,
                                 ) -> Dict[str, Any]:
    """A controlled "no transitions" envelope — never a partial or invented one.

    Same shape as the available payload so a consumer reads one contract, with
    ``available: False`` and a governed ``reason_code`` rather than empty
    aggregations that would look like "nothing moved".
    """
    return {
        "detail_type": DETAIL_STAGE_TRANSITION,
        "portfolio_id": portfolio_id,
        "scope": scope or "total",
        "run_id": None,
        "as_of_date": as_of,
        "comparison_date": comparison_date,
        "available": False,
        "reason": reason,
        "reason_code": reason_code,
        "identifier": CASE_KEY,
        "measure": MEASURE,
        "stage_field": STAGE,
        "counts": None,
        "transitions": [],
        "new_arrivals": [],
        "stayers": [],
        "departures": [],
        "event_totals": None,
        "reconciliation": None,
        "methodology": {
            "movement_basis": "gross",
            "identity_basis": CASE_KEY,
            "stage_vocabulary": "mi_agent_api.pipeline_prep.canonical_stage",
            "version": TRANSITION_METHODOLOGY_VERSION,
            "duplicate_case_identifiers": duplicates,
        },
        "source_dates": {"pipeline_as_of": as_of,
                         "pipeline_comparison": comparison_date,
                         "funded_as_of": None, "forecast_generated_at": None},
        "sources": {"current": None, "comparison": None},
    }


def resolve_stage_transition_detail(root: str, client_id: str, *,
                                    as_of: Optional[str] = None,
                                    historical_model: Optional[Dict[str, Any]] = None,
                                    scope: Optional[str] = None,
                                    ) -> Dict[str, Any]:
    """Build the transition detail from the governed weekly pipeline extracts.

    Uses the SAME inventory, the SAME ``load_prepared_pipeline`` and the SAME
    ``select_pair`` neighbour rule as the movement detail above, so the
    snapshots compared here are exactly the snapshots production MI prepares and
    the pipeline views already plot. There is no second preparation path and no
    second snapshot-matching engine.
    """
    from . import pipeline_contract as pipeline_mod

    inv = pipeline_mod.weekly_extract_inventory(root, client_id)
    cur_e, pri_e = select_pair(inv.get("extracts", []), as_of)
    if cur_e is None:
        return stage_transition_unavailable(
            client_id, scope=scope, as_of=as_of,
            reason_code=REASON_NO_COMPARISON,
            reason="No governed weekly pipeline extract matches this point.")
    if pri_e is None:
        return stage_transition_unavailable(
            client_id, scope=scope, as_of=cur_e.get("pipeline_extract_date"),
            reason_code=REASON_NO_COMPARISON,
            reason="This is the earliest governed weekly extract, so there is "
                   "no prior snapshot to compare it against.")

    cur, _ = pipeline_mod.load_prepared_pipeline(
        cur_e, historical_model=historical_model)
    pri, _ = pipeline_mod.load_prepared_pipeline(
        pri_e, historical_model=historical_model)

    return build_stage_transition_detail(
        cur, pri,
        as_of_date=cur_e.get("pipeline_extract_date"),
        comparison_date=pri_e.get("pipeline_extract_date"),
        portfolio_id=client_id, scope=scope,
        run_id=cur_e.get("run_id"),
        source_file=_basename(cur_e.get("source_file")),
        comparison_source_file=_basename(pri_e.get("source_file")))
