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
from trakt_core import perf as _perf

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


@_perf.stage_fn("historical_completion_model_build")
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
    file_names: List[str] = []
    historical_rows = 0
    stable_identifier: Optional[str] = None

    for entry in entries:
        df = _read(Path(entry.get("source_file", "")))
        if df is None or df.empty:
            continue
        extract_date = entry.get("pipeline_extract_date")
        csf = case_stage_frame(df)
        if csf.empty:
            continue
        snapshots_used += 1
        historical_rows += int(len(csf))
        file_names.append(Path(entry.get("source_file", "")).name)
        if stable_identifier is None:
            stable_identifier = _identifier_used(df)
        if extract_date:
            dates.append(extract_date)
        # Column-wise extraction rather than ``csf.iterrows()``. iterrows builds
        # a fresh Series per row — ~1.5k rows x 26 snapshots was ~39k Series
        # constructions, and it dominated this function's cost. Zipping the
        # underlying arrays walks the SAME rows in the SAME order with the same
        # per-row logic below; only the row-access mechanism changed.
        case_ids = csf["case_id"].to_numpy()
        stages = csf["stage"].to_numpy()
        completion_dates = (csf["completion_date"].to_numpy()
                            if "completion_date" in csf.columns
                            else [None] * len(csf))
        for cid_raw, stage_raw, cd in zip(case_ids, stages, completion_dates):
            cid = str(cid_raw).strip()
            if not cid or cid.lower() in ("nan", "none", ""):
                continue
            stage = str(stage_raw)
            t = timelines.setdefault(cid, {"stages": {}, "completed_on": None, "ever": set()})
            t["ever"].add(stage)
            # First snapshot at which the case was seen at this stage.
            if stage not in t["stages"]:
                t["stages"][stage] = extract_date
            if stage == COMPLETED:
                # ``to_numpy()`` on a datetime64 column yields numpy datetime64,
                # where ``iterrows`` yielded pd.Timestamp. Normalise back to a
                # Timestamp so the isinstance/NaT check below is unchanged.
                cd_ts = pd.Timestamp(cd) if cd is not None and not pd.isna(cd) else None
                done = (cd_ts.date().isoformat()
                        if isinstance(cd_ts, pd.Timestamp) and pd.notna(cd_ts)
                        else extract_date)
                if t["completed_on"] is None or (done or "") < t["completed_on"]:
                    t["completed_on"] = done

    # Per active stage: observed cases, completions, elapsed-days to completion.
    observed: Dict[str, int] = {s: 0 for s in ACTIVE_STAGES}
    completed: Dict[str, int] = {s: 0 for s in ACTIVE_STAGES}
    elapsed: Dict[str, List[int]] = {s: [] for s in ACTIVE_STAGES}

    # Elapsed-day pairs are COLLECTED here and converted in ONE vectorised pass
    # below. Previously each pair called ``pd.to_datetime`` on two scalars, and
    # each such call re-guessed the datetime format and built a one-element
    # Series — profiling showed 8,420 format guesses per model build, which was
    # the single largest cost in this function.
    pending: Dict[str, List[tuple]] = {s: [] for s in ACTIVE_STAGES}
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
                    pending[stage].append((first, done))

    for stage, pairs in pending.items():
        if not pairs:
            continue
        firsts = pd.to_datetime([p[0] for p in pairs], errors="coerce")
        dones = pd.to_datetime([p[1] for p in pairs], errors="coerce")
        # ``errors="coerce"`` yields NaT for anything unparseable, which becomes
        # NaN days and is dropped below — the same outcome as the previous
        # per-pair try/except, which skipped a pair it could not parse.
        days = (dones - firsts).days
        elapsed[stage].extend(int(d) for d in days if pd.notna(d) and d >= 0)

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

    # Cumulative cohort progression: of the ORIGINAL KFI cohort, the % that has
    # reached each milestone (KFI -> Application -> Offer -> Funded) by each week.
    # This is a true cohort-tracked funnel (case timelines), NOT a ratio of
    # point-in-time stage stocks — so the lines are monotonic and show where the
    # pipeline leaks. The latest Funded % is the canonical "cumulative cohort
    # conversion" (% of the KFI cohort funded to date).
    cohort_weeks = sorted(set(dates))
    cohort_progression = _cohort_progression(timelines, cohort_weeks)
    cumulative_cohort_conversion = (
        cohort_progression["series"]["COMPLETED"][-1]
        if cohort_progression and cohort_progression["series"]["COMPLETED"] else None)

    # Evidence aggregates.
    observed_completion_count = sum(1 for t in timelines.values() if COMPLETED in t["ever"])
    excluded_stage_counts: Dict[str, int] = {}
    for term in ("WITHDRAWN", "UNKNOWN"):
        c = sum(1 for t in timelines.values() if term in t["ever"])
        if c:
            excluded_stage_counts[term] = c
    stages_historical = sorted(stage_rates.keys())
    stages_config_fallback = sorted(s for s in ACTIVE_STAGES
                                    if observed[s] > 0 and s not in stage_rates)

    return {
        "available": bool(stage_rates),
        "minObservations": int(min_observations),
        "snapshotCount": snapshots_used,
        "weeklyFilesUsed": snapshots_used,
        "weeklyFileNames": file_names,
        "historicalRowsUsed": historical_rows,
        "casesTracked": len(timelines),
        "trackedCaseCount": len(timelines),
        "observedCompletionCount": observed_completion_count,
        "stableIdentifierUsed": stable_identifier,
        "stagesUsingHistoricalRates": stages_historical,
        "stagesUsingConfigFallback": stages_config_fallback,
        "excludedStageCounts": excluded_stage_counts,
        "historicalCompletionRateByStage": rate_by_stage,
        "historicalCompletionTimingByStage": timing_by_stage,
        "historicalCompletionRateWindow": {
            "fromDate": min(dates) if dates else None,
            "toDate": max(dates) if dates else None,
            "snapshotCount": snapshots_used,
        },
        "observationWindowStart": min(dates) if dates else None,
        "observationWindowEnd": max(dates) if dates else None,
        "stage_rates": stage_rates,
        "cohortProgression": cohort_progression,
        "cumulativeCohortConversion": cumulative_cohort_conversion,
    }


# Origination funnel milestone order (entry -> exit). Funded == COMPLETED.
_COHORT_FUNNEL_ORDER = ("KFI", "APPLICATION", "OFFER", COMPLETED)


def _cohort_progression(timelines: Dict[str, Dict[str, Any]],
                        weeks: List[str]) -> Optional[Dict[str, Any]]:
    """Cumulative % of the KFI cohort reaching each milestone by each week.

    ``reached_date(case, milestone_i)`` is the earliest date the case was seen at
    milestone ``i`` OR any later milestone — so the funnel is monotonic (Funded
    ⊆ Offer ⊆ Application ⊆ KFI) and a missing intermediate snapshot never breaks
    it. The cohort is every case with any funnel-stage observation (the
    origination population). ISO date strings sort chronologically, so ``<=`` on
    them is a valid time comparison.
    """
    order = _COHORT_FUNNEL_ORDER
    cohort: List[str] = []
    reached: Dict[str, Dict[str, Optional[str]]] = {}
    for cid, t in timelines.items():
        stage_dates = dict(t.get("stages") or {})
        done = t.get("completed_on")
        if done and (COMPLETED not in stage_dates or str(done) < str(stage_dates[COMPLETED])):
            stage_dates[COMPLETED] = done  # prefer the precise completion date
        funnel_dates = {s: d for s, d in stage_dates.items() if s in order and d}
        if not funnel_dates:
            continue
        cohort.append(cid)
        rd: Dict[str, Optional[str]] = {}
        for i, milestone in enumerate(order):
            later = [funnel_dates[order[j]] for j in range(i, len(order))
                     if order[j] in funnel_dates]
            rd[milestone] = min(later) if later else None
        reached[cid] = rd

    n = len(cohort)
    if n == 0 or not weeks:
        return None
    series: Dict[str, List[float]] = {m: [] for m in order}
    for w in weeks:
        for milestone in order:
            hit = sum(1 for cid in cohort
                      if reached[cid][milestone] is not None
                      and str(reached[cid][milestone]) <= w)
            series[milestone].append(round(hit / n * 100.0, 2))
    return {"weeks": list(weeks), "stages": list(order), "series": series,
            "cohortSize": n}


def _identifier_used(df: pd.DataFrame) -> Optional[str]:
    """Which stable identifier the model tracks cases by (KFI / account number)."""
    from .pipeline_prep import resolve_source_columns
    mapping, _ = resolve_source_columns(df)
    for fld, label in (("pipeline_case_identifier", "account/case number"),
                       ("application_identifier", "KFI/application reference")):
        col = mapping.get(fld)
        if col:
            return f"{fld} ({col})"
    return None


def historical_model_evidence(model: Optional[Dict[str, Any]],
                              completion_probability_basis: Optional[str] = None
                              ) -> Dict[str, Any]:
    """Flatten the historical model into the API ``historicalModelEvidence`` block
    (the explicit evidence the UI lineage shows). Safe for a missing/empty model."""
    m = model or {}
    window = m.get("historicalCompletionRateWindow", {}) or {}
    return {
        "weeklyFilesUsed": m.get("weeklyFilesUsed", 0),
        "weeklyFileNames": m.get("weeklyFileNames", []),
        "observationWindowStart": m.get("observationWindowStart") or window.get("fromDate"),
        "observationWindowEnd": m.get("observationWindowEnd") or window.get("toDate"),
        "historicalRowsUsed": m.get("historicalRowsUsed", 0),
        "trackedCaseCount": m.get("trackedCaseCount", m.get("casesTracked", 0)),
        "observedCompletionCount": m.get("observedCompletionCount", 0),
        "stableIdentifierUsed": m.get("stableIdentifierUsed"),
        "stagesUsingHistoricalRates": m.get("stagesUsingHistoricalRates", []),
        "stagesUsingConfigFallback": m.get("stagesUsingConfigFallback", []),
        "excludedStageCounts": m.get("excludedStageCounts", {}),
        "completionProbabilityBasis": completion_probability_basis,
        # Dedup provenance: distinguish files scanned from unique extracts used so a
        # weekly file counted in two run folders is never double-counted as evidence.
        "sourceFilesScanned": m.get("sourceFilesScanned", m.get("weeklyFilesUsed", 0)),
        "uniqueWeeklyExtractsUsed": m.get("uniqueWeeklyExtractsUsed", m.get("weeklyFilesUsed", 0)),
        "duplicatesExcluded": m.get("duplicatesExcluded", 0),
        "primarySourcePreference": m.get("primarySourcePreference"),
        "available": bool(m.get("available")),
    }
