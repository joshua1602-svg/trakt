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

  * the validity window of a stage is the ``window_quantile`` (95th
    percentile) of the days cases that advanced spent in it;
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
        dwell_adv: List[int] = []           # measured window evidence
        at_risk = [0] * (max_weeks + 1)
        adv_ev = [0] * (max_weeks + 1)
        out_ev = [0] * (max_weeks + 1)
        advanced = fell_out = censored = 0
        for case in cases:
            entry = _ts(case.get(_ENTRY_FIELD[stage])) or _ts(
                case.get("first_seen", {}).get(stage))
            if entry is None:
                continue
            outcome, when = _stage_exit(case, stage)
            if outcome is None:
                continue
            if outcome == "advance":
                d = _days(entry, when)
                if d is not None and d >= 0:
                    dwell_adv.append(d)
            # Hazard sample: exits observed inside the snapshot window, and
            # cases still open at its end; entry into the risk set is delayed
            # to the first snapshot for cases already in the stage then.
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
            k_in, k_out = late // 7, min(exit_day // 7, max_weeks)
            for k in range(k_in, k_out + 1):
                at_risk[k] += 1
            if outcome == "advance":
                adv_ev[k_out] += 1
                advanced += 1
            elif outcome == "fallout":
                out_ev[k_out] += 1
                fell_out += 1
            else:
                censored += 1
        measured_window = _quantile(dwell_adv, q)
        window_measured = len(dwell_adv) >= min_events and measured_window is not None
        sufficient = advanced >= min_events
        stages[stage] = {
            # D21 (owner decision 2026-09-30): measured, or none — no
            # configured window stands in for history that is not there.
            "windowDays": int(measured_window) if window_measured else None,
            "windowBasis": "measured" if window_measured else "insufficient_history",
            "windowQuantile": q,
            "windowEvidence": len(dwell_adv),
            "sufficient": bool(sufficient),
            "advanced": advanced,
            "fellOut": fell_out,
            "stillOpen": censored,
            "pullThrough": (round(advanced / (advanced + fell_out), 4)
                            if advanced + fell_out else None),
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
                                        "fellOut", "stillOpen", "pullThrough")}
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
