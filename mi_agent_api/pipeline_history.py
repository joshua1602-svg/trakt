"""Deterministic historical completion-rate model from weekly pipeline snapshots.

Pipeline files are weekly operational extracts. By tracking the SAME case across
consecutive weekly snapshots (by KFI / account / application id) we can observe
empirical stage -> completion transitions and derive a completion rate and timing
per stage — instead of relying solely on the configured stage probabilities.

This is an INITIAL, deterministic empirical model (not an ML model):
  * a case observed at an active stage that is ever seen COMPLETED counts as a
    completion for that stage;
  * a stage's empirical rate is only trusted when it has at least
    ``MIN_OBSERVATIONS`` observed cases — otherwise callers fall back to config;
  * WITHDRAWN / UNKNOWN cases are never counted as completions.

No probabilities are invented: rates come purely from observed transitions.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .pipeline_prep import ACTIVE_STAGES, case_stage_frame

# Minimum observed cases at a stage before its empirical rate is trusted. Short
# windows under-observe early-stage completions, so we keep this conservative.
MIN_OBSERVATIONS = 12

COMPLETED = "COMPLETED"
WITHDRAWN = "WITHDRAWN"


def _read(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path)
        return pd.read_csv(path, low_memory=False)
    except Exception:  # noqa: BLE001 - a bad weekly file must not break the model
        return None


def build_historical_completion_model(
    weekly_entries: List[Dict[str, Any]],
    *,
    min_observations: int = MIN_OBSERVATIONS,
) -> Dict[str, Any]:
    """Build the historical completion model from chronological weekly snapshots.

    ``weekly_entries`` is a list of ``{source_file, pipeline_extract_date}`` (any
    order — sorted here). Returns the model with per-stage empirical rates/timing,
    the observation window, and ``stage_rates`` (only sufficiently-observed stages)
    for the prep layer to consume.
    """
    entries = sorted(weekly_entries or [],
                     key=lambda e: (e.get("pipeline_extract_date") or "", e.get("source_file") or ""))
    # case_id -> {"stages": {stage: earliest_extract_date}, "completed_on": date, "ever": set}
    timelines: Dict[str, Dict[str, Any]] = {}
    snapshots_used = 0
    dates: List[str] = []

    for entry in entries:
        df = _read(Path(entry.get("source_file", "")))
        if df is None or df.empty:
            continue
        extract_date = entry.get("pipeline_extract_date")
        csf = case_stage_frame(df)
        if csf.empty:
            continue
        snapshots_used += 1
        if extract_date:
            dates.append(extract_date)
        for _, row in csf.iterrows():
            cid = str(row["case_id"]).strip()
            if not cid or cid.lower() in ("nan", "none", ""):
                continue
            stage = str(row["stage"])
            t = timelines.setdefault(cid, {"stages": {}, "completed_on": None, "ever": set()})
            t["ever"].add(stage)
            # First snapshot at which the case was seen at this stage.
            if stage not in t["stages"]:
                t["stages"][stage] = extract_date
            if stage == COMPLETED:
                cd = row.get("completion_date")
                done = (cd.date().isoformat() if isinstance(cd, pd.Timestamp) and pd.notna(cd)
                        else extract_date)
                if t["completed_on"] is None or (done or "") < t["completed_on"]:
                    t["completed_on"] = done

    # Per active stage: observed cases, completions, elapsed-days to completion.
    observed: Dict[str, int] = {s: 0 for s in ACTIVE_STAGES}
    completed: Dict[str, int] = {s: 0 for s in ACTIVE_STAGES}
    elapsed: Dict[str, List[int]] = {s: [] for s in ACTIVE_STAGES}

    for cid, t in timelines.items():
        ever_completed = COMPLETED in t["ever"]
        for stage in ACTIVE_STAGES:
            if stage not in t["stages"]:
                continue
            observed[stage] += 1
            if ever_completed:
                completed[stage] += 1
                first = t["stages"][stage]
                done = t["completed_on"]
                if first and done:
                    try:
                        d = (pd.to_datetime(done) - pd.to_datetime(first)).days
                        if d >= 0:
                            elapsed[stage].append(int(d))
                    except Exception:  # noqa: BLE001
                        pass

    rate_by_stage: Dict[str, Any] = {}
    timing_by_stage: Dict[str, Any] = {}
    stage_rates: Dict[str, float] = {}
    for stage in ACTIVE_STAGES:
        obs = observed[stage]
        comp = completed[stage]
        sufficient = obs >= min_observations
        rate = round(comp / obs, 4) if obs else None
        rate_by_stage[stage] = {"rate": rate, "observed": obs, "completed": comp,
                                "sufficient": bool(sufficient and rate is not None)}
        if elapsed[stage]:
            timing_by_stage[stage] = {"medianDays": int(statistics.median(elapsed[stage])),
                                      "observed": len(elapsed[stage])}
        if sufficient and rate is not None:
            stage_rates[stage] = rate

    return {
        "available": bool(stage_rates),
        "minObservations": int(min_observations),
        "snapshotCount": snapshots_used,
        "casesTracked": len(timelines),
        "historicalCompletionRateByStage": rate_by_stage,
        "historicalCompletionTimingByStage": timing_by_stage,
        "historicalCompletionRateWindow": {
            "fromDate": min(dates) if dates else None,
            "toDate": max(dates) if dates else None,
            "snapshotCount": snapshots_used,
        },
        "stage_rates": stage_rates,
    }
