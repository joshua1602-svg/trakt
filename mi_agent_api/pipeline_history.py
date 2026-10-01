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

from . import pipeline_runoff as _runoff
from .pipeline_prep import (ACTIVE_STAGES, STAGE_ENTRY_FIELD, case_stage_frame,
                            stage_validity_windows)
from trakt_core import perf as _perf

# Minimum observed cases at a stage before its empirical rate is trusted. Short
# windows under-observe early-stage completions, so we keep this conservative.
MIN_OBSERVATIONS = 12

COMPLETED = "COMPLETED"
WITHDRAWN = "WITHDRAWN"
_ENTRY_FIELDS = ("kfi_date", "application_date", "offer_date")


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
    runoff_settings: Optional[Dict[str, Any]] = None,
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
        amounts = (csf["amount"].to_numpy() if "amount" in csf.columns
                   else [None] * len(csf))
        entry_dates = [csf[f].to_numpy() if f in csf.columns else [None] * len(csf)
                       for f in _ENTRY_FIELDS]
        for cid_raw, stage_raw, cd, amt, *entries in zip(case_ids, stages,
                                                         completion_dates, amounts,
                                                         *entry_dates):
            cid = str(cid_raw).strip()
            if not cid or cid.lower() in ("nan", "none", ""):
                continue
            stage = str(stage_raw)
            t = timelines.setdefault(cid, {"stages": {}, "completed_on": None, "ever": set()})
            t["ever"].add(stage)
            # Run-off inputs: the latest stage dates the file states, the
            # case's latest stage, and the first snapshot it showed withdrawn.
            for fld, value in zip(_ENTRY_FIELDS, entries):
                if value is not None and not pd.isna(value):
                    t[fld] = pd.Timestamp(value)
            t["final_stage"] = stage
            t["last_seen"] = extract_date
            if stage in ACTIVE_STAGES:
                t["seen_open"] = True
            if stage == WITHDRAWN and not t.get("withdrawn_on"):
                t["withdrawn_on"] = extract_date
            # First snapshot at which the case was seen at this stage.
            if stage not in t["stages"]:
                t["stages"][stage] = extract_date
            if stage == COMPLETED:
                # ``to_numpy()`` on a datetime64 column yields numpy datetime64,
                # where ``iterrows`` yielded pd.Timestamp. Normalise back to a
                # Timestamp so the isinstance/NaT check below is unchanged.
                cd_ts = pd.Timestamp(cd) if cd is not None and not pd.isna(cd) else None
                stated = isinstance(cd_ts, pd.Timestamp) and pd.notna(cd_ts)
                done = cd_ts.date().isoformat() if stated else extract_date
                if t["completed_on"] is None or (done or "") < t["completed_on"]:
                    t["completed_on"] = done
                    t["completed_on_stated"] = bool(stated)
                # D22: the amount the case completed at — its amount on the
                # first extract that shows it completed.
                if "completed_amount" not in t:
                    t["completed_amount"] = (float(amt) if amt is not None
                                             and not pd.isna(amt) else None)

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

    timing_by_stage: Dict[str, Any] = {}
    for stage in ACTIVE_STAGES:
        if elapsed[stage]:
            timing_by_stage[stage] = {"medianDays": int(statistics.median(elapsed[stage])),
                                      "observed": len(elapsed[stage])}

    # WHEN THE LIVE PIPELINE IS EXPECTED TO COMPLETE, from the book's own history
    # (owner decision 2026-09-30: "a date answer based on historical time to
    # complete using the client's time series").

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
    runoff = _runoff.fit_runoff(
        ({"kfi_date": t.get("kfi_date"), "application_date": t.get("application_date"),
          "offer_date": t.get("offer_date"), "completed_on": t.get("completed_on"),
          "withdrawn_on": t.get("withdrawn_on"), "first_seen": t["stages"],
          "final_stage": t.get("final_stage"), "seen_open": t.get("seen_open", False)}
         for t in timelines.values()),
        min(dates) if dates else None, max(dates) if dates else None,
        runoff_settings)

    # D27 (owner decision 2026-10-01): THE HISTORICAL COMPLETION RATE of a
    # stage is the chance a case new to it completes, from the run-off model
    # — each step's measured pull-through along the way to completion, cases
    # still in progress counted as still waiting. "Completed so far / ever
    # seen" counted every case still working through the pipeline as one that
    # did not complete, and so read low (4.8% from KFI on production, where
    # the measured steps multiply to about 10%). The count is kept as the
    # evidence. It is the probability the forecast gives a case that has just
    # entered the stage; the forecast weights no KFI (top of funnel).
    rate_by_stage: Dict[str, Any] = {}
    # THE FORECAST'S WEIGHTING IS NOT CHANGED by D27: it weights a live case
    # by the run-off model (`pipeline_prep`, tier 5) and falls back to this
    # count ratio only where the run-off cannot price the case — a thin
    # history, never production's. That fallback keeps its rule exactly.
    stage_rates: Dict[str, float] = {}
    runoff_stages = runoff.get("stages") or {}
    for stage in ACTIVE_STAGES:
        obs, comp = observed[stage], completed[stage]
        if obs and obs >= min_observations:
            stage_rates[stage] = round(comp / obs, 4)
        fitted = runoff_stages.get(stage) or {}
        rate = fitted.get("completionFromEntry")
        enough = bool(rate is not None and fitted.get("completionFromEntrySufficient")
                      and obs >= min_observations)
        weighted = stage in _runoff.FORECAST_STAGES
        rate_by_stage[stage] = {
            "rate": rate, "sufficient": enough,
            "observed": observed[stage], "completedSoFar": completed[stage],
            "completed": completed[stage],
            "basis": "runoff_from_entry",
            "forecastWeighted": weighted,
            "note": (None if weighted else
                     f"The forecast weights no {_stage_words(stage)} case: it "
                     f"is top of funnel.")}
    stages_historical = sorted(stage_rates.keys())
    # Stages with cases but too little history to measure a rate. The key
    # (`stagesUsingConfigFallback`) is kept for its readers; since D21 nothing
    # stands in for the missing rate — the stage's cases are not weighted.
    stages_config_fallback = sorted(s for s in ACTIVE_STAGES
                                    if observed[s] > 0 and s not in stage_rates)

    # WHEN THE LIVE PIPELINE IS EXPECTED TO COMPLETE, from the book's own history
    # (owner decision 2026-09-30: "a date answer based on historical time to
    # complete using the client's time series"), over the live cases the
    # forecast has not lapsed (D17) — so after the run-off, whose measured
    # validity windows are the forecast's.
    expected_by_stage, expected_all = _expected_completion(
        timelines, timing_by_stage, max(dates) if dates else None,
        min_observations, windows=stage_validity_windows(runoff),
        window_basis={s: (m or {}).get("windowBasis")
                      for s, m in (runoff.get("stages") or {}).items()})

    return {
        "available": bool(stage_rates) or bool(runoff.get("available")),
        "runoff": runoff,
        "minObservations": int(min_observations),
        "snapshotCount": snapshots_used,
        "weeklyFilesUsed": snapshots_used,
        "weeklyFileNames": file_names,
        "historicalRowsUsed": historical_rows,
        "casesTracked": len(timelines),
        "trackedCaseCount": len(timelines),
        "observedCompletionCount": observed_completion_count,
        "stableIdentifierUsed": stable_identifier,
        # D22: the completion run-rate by calendar window, from each case's
        # own completion date and amount.
        "completionRunRate": completion_run_rate(timelines, dates),
        "stagesUsingHistoricalRates": stages_historical,
        "stagesUsingConfigFallback": stages_config_fallback,
        "excludedStageCounts": excluded_stage_counts,
        "historicalCompletionRateByStage": rate_by_stage,
        # D21 with D27: a stage whose way to completion the history cannot yet
        # measure (a step no case was ever seen leaving) states no rate, and
        # this is why.
        "historicalCompletionRateWithheld": (
            "the client's history does not yet measure every step to "
            "completion from " + ", ".join(
                _stage_words(st) for st, row in rate_by_stage.items()
                if row["rate"] is None)
            if any(row["rate"] is None for row in rate_by_stage.values())
            else None),
        "historicalCompletionTimingByStage": timing_by_stage,
        "expectedCompletionByStage": expected_by_stage,
        "expectedCompletion": expected_all,
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


def _stage_words(stage: str) -> str:
    """A stage as a reader writes it: KFI, Application, Offer."""
    text = str(stage or "").upper()
    return text if text == "KFI" else text.title()


#: D22 (owner decision 2026-09-30): the completion run-rate is measured on
#: the CALENDAR, not per extract. The pipeline is reported ad hoc (D15), so
#: "the last five extracts" can be three weeks or eight; a run-rate over N
#: weeks is the amount of the cases that COMPLETED in the N x 7 days to the
#: latest extract, by each case's own completion date (the date the extract
#: states, or the first extract that shows it completed where it states none),
#: at the amount it completed at — per week over those N weeks, and per month
#: at 52/12 weeks a month. Published for every whole number of weeks the
#: history covers, from `RUN_RATE_MIN_WEEKS` (fewer is not a rate) to the span
#: of the extracts (an earlier completion could have left the extracts before
#: any of them saw it). `RUN_RATE_DEFAULT_WEEKS` is the forecast's own window.
RUN_RATE_MIN_WEEKS = 3
RUN_RATE_DEFAULT_WEEKS = 5
WEEKS_PER_MONTH = 52 / 12


def completion_run_rate(timelines: Dict[str, Dict[str, Any]],
                        dates: List[str]) -> Dict[str, Any]:
    """The completion run-rate over every whole-week window the history
    covers, ending at the latest extract (D22). A window holding a completion
    whose amount the extracts do not state publishes no amount for it: a total
    that silently leaves a case out is not the rate."""
    known = sorted(d for d in dates if d)
    if not known:
        return {"available": False, "method": "calendar",
                "reason": "no dated pipeline extracts"}
    as_of = pd.Timestamp(known[-1])
    start = pd.Timestamp(known[0])
    max_weeks = int((as_of - start).days // 7)
    completions = [(pd.Timestamp(t["completed_on"]), t.get("completed_amount"),
                    bool(t.get("completed_on_stated")))
                   for t in timelines.values() if t.get("completed_on")]
    windows: List[Dict[str, Any]] = []
    for weeks in range(RUN_RATE_MIN_WEEKS, max_weeks + 1):
        opens = as_of - pd.Timedelta(days=7 * weeks)
        inside = [c for c in completions if opens < c[0] <= as_of]
        unstated = sum(1 for c in inside if c[1] is None)
        amount = (round(sum(float(c[1]) for c in inside), 2)
                  if not unstated else None)
        weekly = round(amount / weeks, 2) if amount is not None else None
        windows.append({
            "weeks": weeks,
            "from": (opens + pd.Timedelta(days=1)).date().isoformat(),
            "to": as_of.date().isoformat(),
            "cases": len(inside),
            "casesWithoutAmount": unstated,
            "casesDatedByExtract": sum(1 for c in inside if not c[2]),
            "amount": amount,
            "weeklyAmount": weekly,
            "monthlyAmount": (round(weekly * WEEKS_PER_MONTH, 2)
                              if weekly is not None else None),
        })
    return {
        "available": bool(windows),
        "method": "calendar",
        "basis": ("the amount of the cases that completed in the window, by "
                  "each case's completion date, per week over the window and "
                  "per month at 52/12 weeks a month"),
        "asOf": as_of.date().isoformat(),
        "historyStart": start.date().isoformat(),
        "minWeeks": RUN_RATE_MIN_WEEKS,
        "maxWeeks": max_weeks,
        "defaultWeeks": RUN_RATE_DEFAULT_WEEKS,
        "byWindow": windows,
        **({} if windows else {
            "reason": (f"the extracts span {max_weeks} whole week(s); a "
                       f"run-rate needs at least {RUN_RATE_MIN_WEEKS}")}),
    }


def run_rate_window(model: Optional[Dict[str, Any]],
                    weeks: int) -> Optional[Dict[str, Any]]:
    """The published window of `weeks` weeks, or None where the history does
    not cover it."""
    rr = (model or {}).get("completionRunRate") or {}
    return next((w for w in rr.get("byWindow") or () if w.get("weeks") == weeks),
                None)


def _expected_completion(timelines: Dict[str, Dict[str, Any]],
                         timing_by_stage: Dict[str, Any], latest: Optional[str],
                         min_observations: int, *,
                         windows: Optional[Dict[str, int]] = None,
                         window_basis: Optional[Dict[str, Any]] = None
                         ) -> "tuple[Dict[str, Any], Dict[str, Any]]":
    """When the LIVE pipeline is expected to complete, from the book's history.

    A live case is one the latest weekly extract shows at an open stage. Its
    expected completion is the date it was first seen at that stage plus the
    stage's median elapsed days to completion — measured on this book's cases
    that DID complete, first-seen to completion, exactly as
    `historicalCompletionTimingByStage` measures it. So the date is
    CONDITIONAL on completing: most KFI cases never do (the completion rate by
    stage says how many), and the answer says so.

    LAPSED CASES ARE NOT DATED (owner decision D17, 2026-09-30). A live case
    that has sat in its stage longer than the stage's validity window — the
    forecast's own rule, `pipeline_prep.stage_validity_windows`, measured from
    the stage's entry date as the forecast measures it — carries no forecast
    weight, and dating it would put the answer in the past: the 13:49 check
    answered 2026-04-01 for a pipeline as at 2026-09-24. It is counted, and
    left out of the date.

    `(by stage, all live cases)`: per stage the live cases, how many are
    lapsed and against which window, the median expected date over the rest,
    the median days it rests on, the completions that measured it, how many
    dated cases are already past it, and the owner's sufficiency flag; over
    all live cases the median expected date of those not lapsed.
    """
    windows = windows or {}
    by_stage: Dict[str, Any] = {}
    every: List[str] = []
    live_total = lapsed_total = 0
    # D21: a stage whose validity window the history cannot yet measure — its
    # live cases cannot be told lapsed or not, so none is dated, and no date
    # over all live cases is stated without them.
    undetermined: List[str] = []
    if not latest:
        return by_stage, {"medianDate": None, "liveCases": 0, "lapsedCases": 0,
                          "datedCases": 0, "asOf": None}
    as_of = pd.Timestamp(latest)
    for stage in ACTIVE_STAGES:
        live = [t for t in timelines.values()
                if t.get("last_seen") == latest and t.get("final_stage") == stage
                and t["stages"].get(stage)]
        if not live:
            continue
        window = windows.get(stage)
        if window is None:
            live_total += len(live)
            undetermined.append(stage)
            by_stage[stage] = {"liveCases": len(live), "lapsedCases": None,
                               "datedCases": 0, "windowDays": None,
                               "windowBasis": ((window_basis or {}).get(stage)
                                               or "insufficient_history"),
                               "sufficient": False, "undetermined": True}
            continue
        entry_field = STAGE_ENTRY_FIELD.get(stage)

        def lapsed(t: Dict[str, Any]) -> bool:
            entry = pd.to_datetime(t.get(entry_field), errors="coerce") \
                if entry_field else pd.NaT
            return (window is not None and pd.notna(entry)
                    and (as_of - entry).days > window)

        current = [t for t in live if not lapsed(t)]
        live_total += len(live)
        lapsed_total += len(live) - len(current)
        # No completion from the stage in the history: no date, and said so.
        row: Dict[str, Any] = {"liveCases": len(live),
                               "lapsedCases": len(live) - len(current),
                               "datedCases": 0, "windowDays": window,
                               "windowBasis": (window_basis or {}).get(stage),
                               "sufficient": False}
        timing = timing_by_stage.get(stage)
        if timing:
            days = int(timing["medianDays"])
            firsts = pd.to_datetime([t["stages"][stage] for t in current],
                                    errors="coerce")
            expected = sorted((f + pd.Timedelta(days=days)).date().isoformat()
                              for f in firsts if pd.notna(f))
            row.update({"medianDays": days,
                        "completionsObserved": int(timing["observed"])})
            if expected:
                row.update({
                    "medianDate": statistics.median_low(expected),
                    "datedCases": len(expected),
                    "pastTypical": sum(1 for d in expected if d < latest),
                    "sufficient": int(timing["observed"]) >= min_observations,
                })
                every.extend(expected)
        by_stage[stage] = row
    return by_stage, {"medianDate": (statistics.median_low(sorted(every))
                                     if every and not undetermined else None),
                      "liveCases": live_total, "lapsedCases": lapsed_total,
                      "datedCases": len(every), "asOf": latest,
                      "undeterminedStages": undetermined}


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
            "cohortSize": n,
            # The funnel as it stands at the latest week — the figure the
            # Pipeline tab's conversion card shows per milestone — published
            # by name so a reader looks it up rather than indexes a series.
            "asOfWeek": weeks[-1],
            "latest": {m: (series[m][-1] if series[m] else None) for m in order}}


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
        # The stage run-off model behind the forecast (additive).
        "runoff": _runoff.evidence(m.get("runoff")),
    }
