"""Pipeline run-off (pull-through) model, measured from weekly snapshots.

The market-standard way to forecast a mortgage pipeline — and the method of
ERE's own Pipeline Run-Off Model workbook — is stage by stage:

  * a case sits in a stage (KFI, Application, Offer) until it ADVANCES to the
    next stage or FALLS OUT (withdrawn / lapsed);
  * each stage has a validity window: a case still sitting in the stage past
    it has, in practice, lapsed;
  * the chance a live case completes, and when, depends on how long it has
    already been in its stage.

The workbook takes those as inputs (App->Offer 60%, 4-week app validity,
Offer->Completion 70%, 18-week offer validity, timing curves by week). With
case-level weekly snapshots they can be MEASURED instead, which is what this
module does:

  * the validity window of a stage is the day by which ``window_quantile``
    (95%) of the cases that will ever advance from it have advanced —
    measured with the cases still waiting counted (a competing-risks,
    Aalen-Johansen estimate, censored at the last snapshot and left-truncated
    at the first), so slow cases still open do not shorten it;
  * a case still sitting in a stage past that window has LAPSED: it counts as
    fallen out of the stage's pull-through (owner decision D26), as it
    already carries no forecast weight (D17) — so a stage whose extracts
    never record a withdrawal (KFIs that do not proceed simply stay open) is
    not reported as converting every case that leaves it;
  * weekly advance and fall-out hazards by weeks-in-stage are estimated from
    every case observed in the stage, with cases still open treated as
    censored and cases already in the stage when observation began entering
    the risk set late (left truncation) — so neither biases the rates;
  * a live case's completion probability is the chance it advances from its
    current stage age, times the chance an Offer then completes; its expected
    completion is the probability-weighted time to get there.

Fall-out timing comes from the first weekly snapshot showing the case
withdrawn. A case first seen already withdrawn has no observable fall-out
time and is left out of the hazards (its outcome is still counted in the
pull-through rate).

KFIs are measured (window, pull-through) for disclosure but are not part of
the forward forecast — the workbook treats the KFI stage as reference only.
"""

from __future__ import annotations

import bisect
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

STAGES = ("KFI", "APPLICATION", "OFFER")
FORECAST_STAGES = ("APPLICATION", "OFFER")
_NEXT = {"KFI": "APPLICATION", "APPLICATION": "OFFER", "OFFER": "COMPLETED"}
_ENTRY_FIELD = {"KFI": "kfi_date", "APPLICATION": "application_date",
                "OFFER": "offer_date"}
_ORDER = {"KFI": 0, "APPLICATION": 1, "OFFER": 2, "COMPLETED": 3}

DEFAULTS: Dict[str, Any] = {
    "min_events": 12,
    "window_quantile": 0.95,
    "max_weeks": 156,
    # Used only where a stage has too little history to measure its window.
    # The values are the workbook's validity assumptions.
    "fallback_validity_days": {"KFI": 14, "APPLICATION": 28, "OFFER": 126},
}


def _days(a: Optional[pd.Timestamp], b: Optional[pd.Timestamp]) -> Optional[int]:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    return int((b - a).days)


def _ts(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(ts) else ts


def _quantile(values: List[int], q: float) -> Optional[int]:
    if not values:
        return None
    vs = sorted(values)
    idx = min(len(vs) - 1, max(0, math.ceil(q * len(vs)) - 1))
    return int(vs[idx])


def _advance_window(sample: List[Tuple[int, int, str]], q: float
                    ) -> Optional[int]:
    """The day by which ``q`` of the eventual advances from a stage happen.

    The cumulative incidence of advancing, with falling out as the competing
    outcome (the Aalen-Johansen estimator): cases still open at the last
    snapshot are censored, and cases already in the stage at the first enter
    the risk set late (left truncation). The cases still waiting are counted
    in every risk set they belong to, which a percentile of the completed
    advances alone leaves out — that would shorten the window by exactly the
    slow cases still in progress. ``sample`` holds (entry day, exit day,
    outcome) per case, days counted from the case's entry to the stage.
    """
    events = sorted({x for _, x, o in sample if o in ("advance", "fallout")})
    if not events:
        return None
    entries = sorted(e for e, _, _ in sample)
    exits = sorted(x for _, x, _ in sample)
    advance_at: Dict[int, int] = {}
    leave_at: Dict[int, int] = {}
    for _, x, o in sample:
        if o in ("advance", "fallout"):
            leave_at[x] = leave_at.get(x, 0) + 1
            if o == "advance":
                advance_at[x] = advance_at.get(x, 0) + 1
    survive, incidence = 1.0, 0.0
    curve: List[Tuple[int, float]] = []
    for t in events:
        at_risk = bisect.bisect_right(entries, t) - bisect.bisect_left(exits, t)
        if at_risk <= 0:
            continue
        incidence += survive * advance_at.get(t, 0) / at_risk
        survive *= max(0.0, 1.0 - leave_at[t] / at_risk)
        curve.append((t, incidence))
    total = curve[-1][1] if curve else 0.0
    if total <= 0:
        return None
    for t, cumulative in curve:
        if cumulative >= q * total - 1e-12:
            return int(t)
    return int(curve[-1][0])


def _stage_exit(case: Dict[str, Any], stage: str
                ) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
    """``(outcome, date)`` for a case leaving ``stage``: ``advance``,
    ``fallout`` or ``open`` (still in the stage at the last snapshot)."""
    nxt = _NEXT[stage]
    final = case.get("final_stage")
    if nxt == "COMPLETED":
        adv = _ts(case.get("completed_on"))
    else:
        adv = _ts(case.get(_ENTRY_FIELD[nxt])) or _ts(case.get("first_seen", {}).get(nxt))
    reached_later = (adv is not None
                     or _ORDER.get(str(final), -1) > _ORDER[stage]
                     or any(_ORDER.get(s, -1) > _ORDER[stage]
                            for s in case.get("first_seen", {})))
    if reached_later:
        return ("advance", adv) if adv is not None else (None, None)
    if final == "WITHDRAWN":
        if not case.get("seen_open"):
            return None, None       # withdrawn before we ever saw it open
        return "fallout", _ts(case.get("withdrawn_on"))
    if final == stage:
        return "open", None
    return None, None


def fit_runoff(cases: Iterable[Dict[str, Any]], window_start: Optional[str],
               window_end: Optional[str],
               settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fit the per-stage run-off model.

    ``cases`` are per-case summaries across the weekly snapshots::

        {kfi_date, application_date, offer_date, completed_on, withdrawn_on,
         first_seen: {stage: date}, final_stage, seen_open}
    """
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in (settings or {}).items() if v is not None})
    min_events = int(cfg["min_events"])
    q = float(cfg["window_quantile"])
    max_weeks = int(cfg["max_weeks"])
    obs_start, obs_end = _ts(window_start), _ts(window_end)
    cases = list(cases)

    stages: Dict[str, Any] = {}
    for stage in STAGES:
        # THE SAMPLE: exits observed inside the snapshot window, and cases
        # still open at its end; entry into the risk set is delayed to the
        # first snapshot for cases already in the stage then. One sample
        # serves the window, the hazards and the pull-through counts.
        sample: List[Tuple[int, int, str]] = []     # (entry, exit, outcome)
        # THE WINDOW'S SAMPLE adds every advance whose dates the record
        # carries, however long ago: its time in the stage is known exactly,
        # so it is evidence of how long advancing takes even when it happened
        # before the first snapshot. Falling out and still waiting are only
        # known from the snapshots, so they enter as the sample above does.
        window_sample: List[Tuple[int, int, str]] = []
        for case in cases:
            entry = _ts(case.get(_ENTRY_FIELD[stage])) or _ts(
                case.get("first_seen", {}).get(stage))
            if entry is None:
                continue
            outcome, when = _stage_exit(case, stage)
            if outcome is None:
                continue
            if outcome == "advance":
                dwell = _days(entry, when)
                if dwell is not None and dwell >= 0:
                    window_sample.append((0, dwell, "advance"))
            if outcome == "open":
                exit_day = _days(entry, obs_end)
            else:
                exit_day = _days(entry, when)
                if (exit_day is None or obs_start is None or when is None
                        or when < obs_start):
                    continue
            if exit_day is None or exit_day < 0:
                continue
            late = max(0, _days(entry, obs_start) or 0) if obs_start is not None else 0
            if late > exit_day:
                continue
            sample.append((late, exit_day, outcome))
            if outcome != "advance":
                window_sample.append((late, exit_day, outcome))

        advance_events = sum(1 for _, _, o in window_sample if o == "advance")
        measured_window = _advance_window(window_sample, q)
        window_measured = advance_events >= min_events and measured_window is not None
        window = int(measured_window) if window_measured else None

        at_risk = [0] * (max_weeks + 1)
        adv_ev = [0] * (max_weeks + 1)
        out_ev = [0] * (max_weeks + 1)
        advanced = fell_out = lapsed = censored = 0
        for late, exit_day, outcome in sample:
            k_in, k_out = late // 7, min(exit_day // 7, max_weeks)
            for k in range(k_in, k_out + 1):
                at_risk[k] += 1
            if outcome == "advance":
                adv_ev[k_out] += 1
                advanced += 1
            elif outcome == "fallout":
                out_ev[k_out] += 1
                fell_out += 1
            elif window is not None and exit_day > window:
                # D26: still open past the stage's measured window — lapsed,
                # so fallen out of the pull-through. The hazards keep it as
                # censored: its outcome is not observed, only overdue.
                lapsed += 1
            else:
                censored += 1
        sufficient = advanced >= min_events
        exits = advanced + fell_out + lapsed
        stages[stage] = {
            # D21 (owner decision 2026-09-30): measured, or none — no
            # configured window stands in for history that is not there.
            "windowDays": window,
            "windowBasis": "measured" if window_measured else "insufficient_history",
            "windowQuantile": q,
            "windowEvidence": advance_events,
            "sufficient": bool(sufficient),
            "advanced": advanced,
            "fellOut": fell_out,
            # D26: how many of the fall-outs are lapsed rather than recorded
            # as withdrawn; None while the window is unmeasured, when a case
            # open past it cannot be told from one still within it.
            "lapsed": lapsed if window is not None else None,
            "stillOpen": censored,
            "pullThrough": (round(advanced / exits, 4) if exits else None),
            "hazardAdvance": [round(adv_ev[k] / at_risk[k], 6) if at_risk[k] else 0.0
                              for k in range(max_weeks + 1)],
            "hazardFallout": [round(out_ev[k] / at_risk[k], 6) if at_risk[k] else 0.0
                              for k in range(max_weeks + 1)],
            "casesObserved": max(at_risk) if at_risk else 0,
        }
    offer = stages["OFFER"]
    app = stages["APPLICATION"]
    return {
        "available": any(stages[s]["sufficient"] for s in FORECAST_STAGES),
        "method": "stage run-off: measured validity windows, weekly advance/"
                  "fall-out hazards by weeks in stage, censored and "
                  "left-truncated at the snapshot window",
        "observationWindowStart": window_start,
        "observationWindowEnd": window_end,
        "minEvents": min_events,
        "stages": stages,
        "appToOfferPullThrough": app["pullThrough"],
        "offerToCompletionPullThrough": offer["pullThrough"],
    }


def evidence(model: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A compact, display-ready summary of a fitted run-off model."""
    m = model or {}
    return {
        "available": bool(m.get("available")),
        "method": m.get("method"),
        "appToOfferPullThrough": m.get("appToOfferPullThrough"),
        "offerToCompletionPullThrough": m.get("offerToCompletionPullThrough"),
        "stages": {
            st: {k: sm.get(k) for k in ("windowDays", "windowBasis", "windowQuantile",
                                        "sufficient", "advanced",
                                        "fellOut", "lapsed", "stillOpen",
                                        "pullThrough")}
            for st, sm in (m.get("stages") or {}).items()},
    }


def advance_from(stage_model: Dict[str, Any], dwell_days: float
                 ) -> Tuple[float, Optional[float]]:
    """``(P(advance | still in stage at dwell_days), expected remaining days)``."""
    h_a = stage_model["hazardAdvance"]
    h_o = stage_model["hazardFallout"]
    a = max(0, int(dwell_days // 7))
    if a >= len(h_a):
        return 0.0, None
    survive, p, days = 1.0, 0.0, 0.0
    for k in range(a, len(h_a)):
        step = survive * h_a[k]
        p += step
        days += step * ((k - a + 0.5) * 7.0)
        survive *= max(0.0, 1.0 - h_a[k] - h_o[k])
        if survive <= 1e-9:
            break
    return (round(p, 6), (days / p) if p > 0 else None)


def complete_from(model: Dict[str, Any], stage: str, dwell_days: float,
                  config_offer: Tuple[Optional[float], Optional[int]]
                  ) -> Tuple[Optional[float], Optional[float]]:
    """``(P(complete), expected days to completion)`` for a live case, or
    ``(None, None)`` when its stage has too little history."""
    stages = model.get("stages") or {}
    sm = stages.get(stage) or {}
    if not sm.get("sufficient"):
        return None, None
    p, days = advance_from(sm, dwell_days)
    if stage == "OFFER":
        return p, days
    om = stages.get("OFFER") or {}
    if om.get("sufficient"):
        p_offer, d_offer = advance_from(om, 0)
    else:
        p_offer, d_offer = config_offer
    if p_offer is None:
        return None, None
    total_days = ((days or 0.0) + (d_offer or 0.0)) if (days is not None) else None
    return round(p * p_offer, 6), total_days
