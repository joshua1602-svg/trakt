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

Maturity correction
-------------------
The pooled ratio above has a censoring defect: a case first seen at a stage in
the final snapshot enters the denominator having had no realistic opportunity
to complete, so it is counted as a failure to convert. That understates every
stage rate, worst at the stages with the longest configured funding lag, and
worse still on a book whose origination is growing — because growth pushes more
of the denominator into the immature zone.

The corrected estimator therefore admits a case to a stage's population only
once it has been observed for at least the configured
``stage_days_to_fund(stage)``. Both estimators are always computed and
published side by side; ``TRAKT_MATURITY_CORRECTED_STAGE_RATES`` decides which
one populates ``stage_rates`` (and therefore the forecast). It is OFF by
default, and with it off this module's numerical output is unchanged.

When corrected mode is on, the selection chain is matured empirical -> standard
configured probability. It never falls back to the pooled estimator: doing so
would reinstate the very defect the correction exists to remove.
"""

from __future__ import annotations

import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .pipeline_prep import ACTIVE_STAGES, _stage_days_to_fund, case_stage_frame
from trakt_core import perf as _perf

# Minimum observed cases at a stage before its empirical rate is trusted. Short
# windows under-observe early-stage completions, so we keep this conservative.
MIN_OBSERVATIONS = 12

COMPLETED = "COMPLETED"
WITHDRAWN = "WITHDRAWN"

#: Feature flag for the maturity-corrected estimator. OFF unless explicitly
#: enabled, and read per call so an operator (and a test) can toggle it without
#: a reload — the same convention ``insight_engine.weekly_brief_enabled`` uses.
MATURITY_FLAG_ENV = "TRAKT_MATURITY_CORRECTED_STAGE_RATES"
_ON = ("1", "true", "on", "yes", "enabled")

#: Closed vocabulary for ``rateSource``. Standard sources only — a
#: client-segment-specific source is deliberately not reachable here.
RATE_POOLED = "POOLED_EMPIRICAL"
RATE_MATURED = "MATURED_EMPIRICAL"
RATE_CONFIG = "CONFIG_FALLBACK"


def maturity_correction_enabled() -> bool:
    """Whether stage rates come from the maturity-corrected estimator."""
    return (os.environ.get(MATURITY_FLAG_ENV) or "").strip().lower() in _ON


def _matured_stats(entries: List[Tuple[str, bool]]) -> Dict[str, Any]:
    """Matured counts, rate, sample span and earlier/later cohort split.

    ``entries`` is ``(first_observation_date, ever_completed)`` for every case
    that matured at one stage. The cohort split divides the ELIGIBLE matured
    observations chronologically and as evenly as practicable — not by calendar
    halves, which on a book that grew from zero would put almost everything in
    one side and say nothing about stability.

    A rate is only reported for a cohort that actually has observations; a stage
    too thin to split says so through ``maturedCohortSplitAvailable`` rather
    than reporting an invented rate.
    """
    total = len(entries)
    completed_n = sum(1 for _d, done in entries if done)
    out: Dict[str, Any] = {
        "maturedObserved": total,
        "maturedCompleted": completed_n,
        "maturedRate": (round(completed_n / total, 4) if total else None),
        "maturedSampleStart": None,
        "maturedSampleEnd": None,
        "maturedCohortSplitAvailable": False,
        "maturedObservedEarlierCohort": 0,
        "maturedRateEarlierCohort": None,
        "maturedObservedLaterCohort": 0,
        "maturedRateLaterCohort": None,
    }
    if not total:
        return out

    ordered = sorted(entries, key=lambda e: e[0])
    out["maturedSampleStart"] = ordered[0][0]
    out["maturedSampleEnd"] = ordered[-1][0]
    if total < 2:
        return out

    half = total // 2
    earlier, later = ordered[:half], ordered[half:]

    def _rate(rows: List[Tuple[str, bool]]) -> Optional[float]:
        return (round(sum(1 for _d, done in rows if done) / len(rows), 4)
                if rows else None)

    out["maturedCohortSplitAvailable"] = bool(earlier and later)
    out["maturedObservedEarlierCohort"] = len(earlier)
    out["maturedRateEarlierCohort"] = _rate(earlier)
    out["maturedObservedLaterCohort"] = len(later)
    out["maturedRateLaterCohort"] = _rate(later)
    return out


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
    # Rows whose stable case/application identifier could not be resolved. They
    # are already skipped below (a case that cannot be matched across snapshots
    # cannot inform a longitudinal rate); counting them makes the exclusion
    # answerable instead of invisible.
    unusable_identity_rows = 0

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
                unusable_identity_rows += 1
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
    # MATURED population per stage: (first_observation_date, ever_completed) for
    # every case that had a full configured stage-to-funding window in which to
    # convert before the observation window closed. A case that has not had that
    # long has not FAILED to convert — it has not been given the chance, and
    # counting it as a non-conversion is the censoring defect this corrects.
    days_to_fund = _stage_days_to_fund()
    window_end_ts = pd.Timestamp(max(dates)) if dates else None
    matured: Dict[str, List[Tuple[str, bool]]] = {s: [] for s in ACTIVE_STAGES}

    pending: Dict[str, List[tuple]] = {s: [] for s in ACTIVE_STAGES}
    for cid, t in timelines.items():
        ever_completed = COMPLETED in t["ever"]
        for stage in ACTIVE_STAGES:
            if stage not in t["stages"]:
                continue
            observed[stage] += 1
            first = t["stages"][stage]
            if window_end_ts is not None and first:
                try:
                    age_days = (window_end_ts - pd.Timestamp(first)).days
                except Exception:  # noqa: BLE001 - an unparseable date is not matured
                    age_days = None
                if age_days is not None and age_days >= int(
                        days_to_fund.get(stage, 0)):
                    matured[stage].append((str(first), ever_completed))
            if ever_completed:
                completed[stage] += 1
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

    corrected = maturity_correction_enabled()
    rate_by_stage: Dict[str, Any] = {}
    timing_by_stage: Dict[str, Any] = {}
    stage_rates: Dict[str, float] = {}
    for stage in ACTIVE_STAGES:
        obs = observed[stage]
        comp = completed[stage]
        sufficient = obs >= min_observations
        rate = round(comp / obs, 4) if obs else None
        median_days = (int(statistics.median(elapsed[stage]))
                       if elapsed[stage] else None)
        if median_days is not None:
            timing_by_stage[stage] = {"medianDays": median_days,
                                      "observed": len(elapsed[stage])}

        m_stats = _matured_stats(matured[stage])
        m_obs, m_rate = m_stats["maturedObserved"], m_stats["maturedRate"]
        m_sufficient = m_obs >= min_observations

        # The selection chain. Corrected mode is matured -> config; it NEVER
        # falls back to the pooled estimator, because falling back to the
        # estimator with the known censoring defect would defeat the correction.
        if corrected:
            rate_used = m_rate if (m_sufficient and m_rate is not None) else None
            rate_source = RATE_MATURED if rate_used is not None else RATE_CONFIG
        else:
            rate_used = rate if (sufficient and rate is not None) else None
            rate_source = RATE_POOLED if rate_used is not None else RATE_CONFIG
        if rate_used is not None:
            stage_rates[stage] = rate_used

        rate_by_stage[stage] = {
            # -- existing keys, unchanged in meaning and value ---------------- #
            "rate": rate, "observed": obs, "completed": comp,
            "sufficient": bool(sufficient and rate is not None),
            # -- pooled, named explicitly ------------------------------------ #
            "pooledObserved": obs, "pooledCompleted": comp, "pooledRate": rate,
            # -- maturity-corrected estimator -------------------------------- #
            **m_stats,
            "maturedSufficient": bool(m_sufficient and m_rate is not None),
            "maturityWindowDays": int(days_to_fund.get(stage, 0)),
            "configuredStageDaysToFund": int(days_to_fund.get(stage, 0)),
            "empiricalMedianCompletionDays": median_days,
            # -- what was actually used, and why ----------------------------- #
            "rateUsed": rate_used, "rateSource": rate_source,
        }

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
        # -- maturity correction (additive; the flag decides stage_rates) ----- #
        "maturityCorrectionEnabled": corrected,
        "maturityWindowDaysByStage": {s: int(days_to_fund.get(s, 0))
                                      for s in ACTIVE_STAGES},
        "stagesUsingMaturedRates": sorted(
            s for s in ACTIVE_STAGES
            if rate_by_stage.get(s, {}).get("rateSource") == RATE_MATURED),
        "identityExcludedRowCount": unusable_identity_rows,
        "identityExcludedReason": (
            "rows carrying no resolvable pipeline case / application identifier "
            "cannot be matched across weekly extracts and are excluded from "
            "both the pooled and the matured empirical populations"
            if unusable_identity_rows else None),
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


#: The per-stage maturity fields promoted into the evidence block. Kept to the
#: enablement-relevant set — the full rate block stays on the model for anyone
#: who needs it, so the evidence payload does not become a statistics dump.
_MATURITY_EVIDENCE_FIELDS = (
    "pooledObserved", "pooledCompleted", "pooledRate",
    "maturedObserved", "maturedCompleted", "maturedRate",
    "maturedSufficient", "maturityWindowDays", "configuredStageDaysToFund",
    "empiricalMedianCompletionDays", "maturedSampleStart", "maturedSampleEnd",
    "maturedCohortSplitAvailable",
    "maturedObservedEarlierCohort", "maturedRateEarlierCohort",
    "maturedObservedLaterCohort", "maturedRateLaterCohort",
    "rateUsed", "rateSource",
)


def _maturity_by_stage(model: Dict[str, Any]) -> Dict[str, Any]:
    """Per-stage maturity evidence, flattened for the disclosure surface."""
    by_stage = model.get("historicalCompletionRateByStage") or {}
    out: Dict[str, Any] = {}
    for stage, block in by_stage.items():
        if not isinstance(block, dict):
            continue
        out[stage] = {f: block.get(f) for f in _MATURITY_EVIDENCE_FIELDS
                      if f in block}
    return out


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
        # Maturity-correction evidence. Additive: a consumer that does not know
        # about it reads exactly what it read before.
        "maturityCorrectionEnabled": bool(m.get("maturityCorrectionEnabled")),
        "maturityWindowDaysByStage": m.get("maturityWindowDaysByStage", {}),
        "stagesUsingMaturedRates": m.get("stagesUsingMaturedRates", []),
        "identityExcludedRowCount": m.get("identityExcludedRowCount", 0),
        "identityExcludedReason": m.get("identityExcludedReason"),
        "maturityByStage": _maturity_by_stage(m),
        # Dedup provenance: distinguish files scanned from unique extracts used so a
        # weekly file counted in two run folders is never double-counted as evidence.
        "sourceFilesScanned": m.get("sourceFilesScanned", m.get("weeklyFilesUsed", 0)),
        "uniqueWeeklyExtractsUsed": m.get("uniqueWeeklyExtractsUsed", m.get("weeklyFilesUsed", 0)),
        "duplicatesExcluded": m.get("duplicatesExcluded", 0),
        "primarySourcePreference": m.get("primarySourcePreference"),
        "available": bool(m.get("available")),
    }
