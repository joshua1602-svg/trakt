"""mi_agent_api/concentration_tests_api.py

Governed funded-book concentration-test service for the Risk Limits workspace,
MI Query and Copilot — one evaluation, three surfaces.

Source precedence, always disclosed and never silent:

1. **approved_configuration** — the operator-approved, versioned configuration
   minted by the OCC concentration-test workflow
   (``mi_agent.concentration_tests.store``). Evaluated deterministically by
   ``mi_agent.concentration_tests.evaluation`` against the same governed
   funded frames the evolution/bridge services resolve.
2. **legacy_extracted** — when no approved configuration exists, the existing
   Schedule 8 extracted monitor (``mi_agent_api.risk_limits``) is presented in
   the same envelope, explicitly marked as *not operator-approved*. Existing
   behaviour is preserved, never rebranded as approved.
3. **none** — nothing available; explicit empty state.

Never 500s: analytical shortfalls come back as controlled states.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from mi_agent.concentration_tests.evaluation import (
    drillthrough as _drillthrough,
    evaluate_active_tests,
)
from mi_agent.concentration_tests.library import load_library
from mi_agent.concentration_tests.models import (
    ActiveConfiguration,
    PROPOSAL_PENDING_APPROVAL,
    PROPOSAL_PENDING_CONFIRMATION,
    PROPOSAL_UNSUPPORTED,
)
from mi_agent.concentration_tests.store import ConcentrationStore

from . import snapshots as snap

logger = logging.getLogger("mi_agent_api.concentration_tests")

SOURCE_APPROVED = "approved_configuration"
SOURCE_LEGACY = "legacy_extracted"
SOURCE_NONE = "none"

_LEGACY_STATUS = {
    "green": "pass", "amber": "warning", "red": "breach",
    "needs_review": "unavailable", "unavailable": "unavailable",
}


# --------------------------------------------------------------------------- #
# Governed frame resolution (same loaders the evolution services use)
# --------------------------------------------------------------------------- #
def _resolve_frames(output_root, client_id: str, to_run_id: Optional[str],
                    scope=None) -> Tuple[Optional[pd.DataFrame],
                                         Optional[pd.DataFrame],
                                         Optional[str], Optional[str],
                                         Optional[str]]:
    """(df, prior_df, reporting_date, prior_reporting_date, run_id).

    Mirrors ``risk_limits.compute_risk_limits``: on-disk run tapes first, the
    blob-aware evolution frames as fallback, then governed scope narrowing.
    The prior frame is the governed prior snapshot — never an arbitrary file.
    """
    df = prior_df = None
    reporting_date = prior_reporting_date = None
    run_id = to_run_id
    runs: List[Dict[str, Any]] = []
    if output_root:
        try:
            disc = snap.discover_snapshots(output_root)
            pf = next((p for p in disc.get("portfolios", [])
                       if p.get("client_id") == client_id), None)
            runs = list(pf.get("runs", [])) if pf else []
            if to_run_id:
                cut = next((i for i, r in enumerate(runs)
                            if r["run_id"] == to_run_id), None)
                if cut is not None:
                    runs = runs[: cut + 1]
        except Exception:  # noqa: BLE001
            runs = []
    if runs:
        run_id = runs[-1]["run_id"]
        tape = snap.resolve_tape_path(output_root, client_id, run_id)
        if tape is not None:
            try:
                df, _ = snap.load_prepared_run(tape)
                reporting_date = runs[-1].get("reporting_date") or \
                    snap.infer_reporting_date(run_id, df)
            except Exception:  # noqa: BLE001
                df = None
        if len(runs) >= 2:
            ptape = snap.resolve_tape_path(output_root, client_id,
                                           runs[-2]["run_id"])
            if ptape is not None:
                try:
                    prior_df, _ = snap.load_prepared_run(ptape)
                    prior_reporting_date = runs[-2].get("reporting_date") or \
                        snap.infer_reporting_date(runs[-2]["run_id"], prior_df)
                except Exception:  # noqa: BLE001
                    prior_df = None
    if df is None and output_root:
        try:
            from . import evolution as _evolution
            frames = _evolution.funded_frames(output_root, client_id,
                                              to_run_id, scope=scope)
            if frames:
                df = frames[-1].get("df")
                reporting_date = frames[-1].get("reporting_date") or reporting_date
                run_id = frames[-1].get("run_id") or run_id
                if len(frames) >= 2:
                    prior_df = frames[-2].get("df")
                    prior_reporting_date = frames[-2].get("reporting_date")
        except Exception:  # noqa: BLE001
            pass
    if scope is not None:
        from mi_agent.portfolio_scope import apply_scope
        df = apply_scope(df, scope)
        prior_df = apply_scope(prior_df, scope)
    return df, prior_df, reporting_date, prior_reporting_date, run_id


def _resolve_pipeline(client_id: str, to_run_id: Optional[str], scope=None
                      ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """The current governed pipeline frame + forecast methodology metadata.

    Reuses the SAME resolution the forecast views use: `datasets` discovers the
    latest weekly extract and prepares it with the historical completion model
    (empirical stage rates over the retained weekly-snapshot window, config
    fallback below the sufficiency floor). Returns (None, meta-with-reason)
    whenever a forward state cannot be governed — never a guess.
    """
    meta: Dict[str, Any] = {"available": False}
    context_id = getattr(scope, "context_id", None) if scope is not None else None
    if context_id not in (None, "", "total"):
        meta["reason"] = (
            "Forward-looking states are evaluated for the total book only: "
            "pipeline exposure carries no portfolio provenance, so it cannot "
            "be narrowed to this portfolio context. Funded results are "
            "unaffected.")
        return None, meta
    try:
        from . import datasets as datasets_mod
        source = datasets_mod._resolve_pipeline_source(client_id, to_run_id)
        if source is None:
            meta["reason"] = "No governed pipeline source found for this client."
            return None, meta
        model = datasets_mod._pipeline_history(client_id)
        from . import pipeline_contract as pipeline_mod
        from .pipeline_history import historical_model_evidence
        df, report = pipeline_mod.load_prepared_pipeline(
            source, historical_model=model)
        basis = (report or {}).get("completion_probability_basis") or \
            (report or {}).get("completionProbabilityBasis")
        evidence = historical_model_evidence(model, basis)
        m = model or {}
        meta = {
            "available": True,
            "methodology": "completion_trend_model",
            "methodologyVersion": "1",
            "basis": basis,
            "observationWindowStart": evidence.get("observationWindowStart"),
            "observationWindowEnd": evidence.get("observationWindowEnd"),
            "weeklyExtractsUsed": evidence.get("uniqueWeeklyExtractsUsed"),
            "trackedCaseCount": evidence.get("trackedCaseCount"),
            "observedCompletionCount": evidence.get("observedCompletionCount"),
            "minObservations": m.get("minObservations"),
            "stagesUsingHistoricalRates": evidence.get("stagesUsingHistoricalRates"),
            "stagesUsingConfigFallback": evidence.get("stagesUsingConfigFallback"),
            "stageRates": m.get("historicalCompletionRateByStage"),
            "stageTiming": m.get("historicalCompletionTimingByStage"),
            "excludedStageCounts": evidence.get("excludedStageCounts"),
            "currentSnapshot": (source.get("source_file")
                                if isinstance(source, dict) else str(source)),
            "pipelineAsOfDate": (source.get("pipeline_as_of_date")
                                 if isinstance(source, dict) else None),
            "pointInTimeSafe": False,
            "pointInTimeNote": (
                "Stage rates are built from the currently retained weekly "
                "extracts; historical expected-state comparisons are "
                "unavailable until point-in-time snapshots are persisted."),
        }
        return df, meta
    except Exception as exc:  # noqa: BLE001 - forward states fail closed
        logger.warning("pipeline resolution for concentration failed: %s", exc)
        meta["reason"] = f"The governed pipeline could not be prepared: {exc}"
        return None, meta


def _active_configuration(client_id: str
                          ) -> Tuple[Optional[ActiveConfiguration], str]:
    """(configuration, note). Absent or unreachable → (None, why)."""
    try:
        store = ConcentrationStore()
        config = store.current_configuration(client_id)
        if config is None:
            return None, "No approved concentration-test configuration exists."
        return config, ""
    except Exception as exc:  # noqa: BLE001 - fail closed to legacy, disclosed
        logger.warning("concentration store unreachable: %s", exc)
        return None, f"The concentration-test store is unreachable: {exc}"


def _pending_counts(client_id: str) -> Dict[str, int]:
    try:
        store = ConcentrationStore()
        counts: Dict[str, int] = {}
        for p in store.list_proposals(client_id):
            counts[p.status] = counts.get(p.status, 0) + 1
        return counts
    except Exception:  # noqa: BLE001
        return {}


# --------------------------------------------------------------------------- #
# Legacy adaptation
# --------------------------------------------------------------------------- #
def _adapt_legacy(legacy: Dict[str, Any]) -> Dict[str, Any]:
    """Present the Schedule 8 extracted monitor in the new envelope shape,
    marked unapproved. Values are carried through verbatim — the legacy path
    stays the single calculator for legacy limits."""
    tests: List[Dict[str, Any]] = []
    for t in legacy.get("tests", []):
        movement = t.get("movementVsPrior")
        current = t.get("actualValue")
        status = _LEGACY_STATUS.get(t.get("status"), "unavailable")
        tests.append({
            "testId": t.get("limitId"),
            "metricId": None,
            "displayName": t.get("label"),
            "category": t.get("category"),
            "reportingDate": legacy.get("reportingDate"),
            "currentValue": current,
            "priorValue": (round(current - movement, 4)
                           if current is not None and movement is not None
                           else None),
            "priorReportingDate": None,
            "priorAvailable": movement is not None,
            "absoluteChange": movement,
            "percentagePointChange": (movement if (t.get("unit") or "percent")
                                      in ("percent", "%") else None),
            "relativeChange": None,
            "threshold": t.get("limitValue"),
            "operator": t.get("direction"),
            "warningFraction": 0.9,
            "unit": t.get("unit"),
            "utilization": None,
            "headroom": t.get("headroom"),
            "status": status,
            "priorStatus": None,
            "statusTransition": None,
            "deteriorated": bool(movement and movement > 0),
            "breachAmount": None,
            "dataStatus": ("data_missing" if t.get("missingFields") else "ok"),
            "missingFields": t.get("missingFields") or [],
            "numeratorValue": None,
            "denominatorValue": None,
            "denominatorBasis": "current_balance",
            "loansInNumerator": 0,
            "totalLoans": 0,
            "severity": "high",
            "effectiveDate": None,
            "expiryDate": None,
            "notes": (t.get("notes") or "") +
                     (" Extracted limit — not operator-approved."
                      if t.get("status") == "needs_review" else ""),
            "resolvedColumns": {},
            "parameters": {},
            "definition": {"numerator": "", "aggregation": "",
                           "description": t.get("actualBasis") or "",
                           "definitionNotes": ""},
            "provenance": {
                "sourceReference": legacy.get("sourceFile") or
                                   legacy.get("limitsSource") or "",
                "sourceText": t.get("sourceSnippet") or "",
                "locator": t.get("sourceSection") or "",
                "approvedBy": "", "approvedAt": "",
                "approvalComments": "", "proposalId": "",
            },
            "configurationVersion": None,
            "evaluatedAt": None,
            "legacy": True,
        })
    from mi_agent.concentration_tests.evaluation import summarise
    summary = summarise(tests, reporting_date=legacy.get("reportingDate") or "",
                        prior_available=any(t["priorAvailable"] for t in tests))
    return {"tests": tests, "summary": summary}


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def compute_concentration_tests(output_root, client_id: str,
                                to_run_id: Optional[str], *,
                                scope=None) -> Dict[str, Any]:
    """Full concentration-test envelope for a client/run. Never raises."""
    config, config_note = _active_configuration(client_id)
    df, prior_df, reporting_date, prior_reporting_date, run_id = \
        _resolve_frames(output_root, client_id, to_run_id, scope=scope)
    pending = _pending_counts(client_id)
    open_proposals = sum(pending.get(s, 0) for s in
                         (PROPOSAL_PENDING_APPROVAL,
                          PROPOSAL_PENDING_CONFIRMATION))

    base: Dict[str, Any] = {
        "portfolioId": client_id,
        "toRunId": run_id,
        "reportingDate": reporting_date,
        "fundedDataAvailable": df is not None and not df.empty,
        "proposalCounts": pending,
        "openProposals": open_proposals,
        "unsupportedProposals": pending.get(PROPOSAL_UNSUPPORTED, 0),
    }

    if config is not None:
        lib = load_library()
        evaluated = evaluate_active_tests(
            df, prior_df, config, lib,
            reporting_date=reporting_date or "",
            prior_reporting_date=prior_reporting_date or "")
        # Forward-looking states: Expected Forecast (existing completion-trend
        # model) and Full Pipeline (maximum-exposure stress). Additive — the
        # funded fields above are never touched; absence is explicit.
        pipeline_df, forecast_meta = _resolve_pipeline(client_id, to_run_id,
                                                       scope=scope)
        if pipeline_df is not None:
            from mi_agent.concentration_tests import forward as forward_mod
            evaluated = forward_mod.extend_with_forward_states(
                evaluated, config, lib, df, pipeline_df,
                forecast_meta=forecast_meta)
        else:
            evaluated["states"] = {"available": False,
                                   "reason": forecast_meta.get("reason")}
            evaluated["emergingRisks"] = []
            from mi_agent.concentration_tests import forward as forward_mod
            evaluated["emergingRisks"] = forward_mod.identify_emerging_risks(
                evaluated.get("tests", []))
        return {
            **base,
            "available": bool(evaluated["tests"]),
            "source": SOURCE_APPROVED,
            "approvalStatus": "approved",
            "forecast": forecast_meta,
            **evaluated,
            "lineage": {
                "source": SOURCE_APPROVED,
                "configurationVersion": config.version,
                "configurationHash": config.content_hash,
                "activatedBy": config.activated_by,
                "activatedAt": config.activated_at,
                "libraryVersion": config.library_version,
                "dataSource": "governed funded canonical frames",
                "reportingDate": reporting_date,
                "priorReportingDate": prior_reporting_date,
            },
        }

    # Legacy fallback — the existing extracted monitor, explicitly unapproved.
    from . import risk_limits as risk_mod
    try:
        legacy = risk_mod.compute_risk_limits(output_root, client_id,
                                              to_run_id, scope=scope)
    except Exception as exc:  # noqa: BLE001 - never 500
        logger.warning("legacy risk-limits fallback failed: %s", exc)
        legacy = {"tests": [], "available": False, "limitsSource": "error",
                  "error": str(exc)}
    if legacy.get("tests"):
        adapted = _adapt_legacy(legacy)
        return {
            **base,
            "available": True,
            "source": SOURCE_LEGACY,
            "approvalStatus": "unapproved_legacy",
            "configurationVersion": None,
            "priorAvailable": adapted["summary"]["priorAvailable"],
            "priorReportingDate": None,
            "tests": adapted["tests"],
            "summary": adapted["summary"],
            "legacyEnvelope": {
                "limitsSource": legacy.get("limitsSource"),
                "sourceType": legacy.get("sourceType"),
                "extractionStatus": legacy.get("extractionStatus"),
                "isPlaceholder": legacy.get("isPlaceholder"),
            },
            "lineage": {
                "source": SOURCE_LEGACY,
                "note": ("Extracted limits shown for continuity. They are NOT "
                         "operator-approved concentration tests; approve "
                         "proposals through the OCC review workflow to govern "
                         "this view. " + config_note).strip(),
                "legacySource": legacy.get("limitsSource"),
                "sourceDocument": legacy.get("sourceFile"),
                "reportingDate": legacy.get("reportingDate"),
            },
        }

    return {
        **base,
        "available": False,
        "source": SOURCE_NONE,
        "approvalStatus": None,
        "configurationVersion": None,
        "priorAvailable": False,
        "priorReportingDate": None,
        "tests": [],
        "summary": {"overallStatus": "unavailable", "activeTests": 0,
                    "breaches": 0, "warnings": 0, "passes": 0,
                    "unavailable": 0, "pendingEffectiveDate": 0, "expired": 0,
                    "reportingDate": reporting_date, "priorReportingDate": None,
                    "priorAvailable": False, "deteriorations": 0,
                    "closestToLimit": None},
        "lineage": {"source": SOURCE_NONE, "note": config_note},
    }


def compute_drillthrough(output_root, client_id: str, to_run_id: Optional[str],
                         test_id: str, *, scope=None,
                         max_rows: int = 500) -> Dict[str, Any]:
    """Contributing-loan population for one APPROVED active test."""
    config, note = _active_configuration(client_id)
    if config is None:
        return {"available": False,
                "reason": ("Drill-through is available for approved "
                           "concentration tests only. " + note).strip(),
                "rows": [], "columns": []}
    test = next((t for t in config.tests if t.test_id == test_id), None)
    if test is None:
        return {"available": False,
                "reason": "That test is not in the active configuration.",
                "rows": [], "columns": []}
    df, _prior, reporting_date, _pdate, _run = _resolve_frames(
        output_root, client_id, to_run_id, scope=scope)
    lib = load_library()
    out = _drillthrough(df, lib, test, max_rows=max_rows)
    out["reportingDate"] = reporting_date
    out["configurationVersion"] = config.version
    return out


def compute_pipeline_drivers(output_root, client_id: str,
                             to_run_id: Optional[str], test_id: str, *,
                             scope=None) -> Dict[str, Any]:
    """Pipeline cases driving one test's expected movement. Approved
    configurations only; reconciles to the expected numerator by construction."""
    config, note = _active_configuration(client_id)
    if config is None:
        return {"available": False,
                "reason": ("Pipeline drivers are available for approved "
                           "concentration tests only. " + note).strip(),
                "drivers": []}
    df, _prior, reporting_date, _pd_, _run = _resolve_frames(
        output_root, client_id, to_run_id, scope=scope)
    pipeline_df, forecast_meta = _resolve_pipeline(client_id, to_run_id,
                                                   scope=scope)
    if pipeline_df is None:
        return {"available": False,
                "reason": forecast_meta.get("reason", "No governed pipeline."),
                "drivers": []}
    lib = load_library()
    evaluated = evaluate_active_tests(df, None, config, lib,
                                      reporting_date=reporting_date or "")
    row = next((t for t in evaluated["tests"] if t["testId"] == test_id), None)
    if row is None:
        return {"available": False,
                "reason": "That test is not in the active configuration.",
                "drivers": []}
    from mi_agent.concentration_tests import forward as forward_mod
    out = forward_mod.compute_drivers_for_test(config, lib, test_id, df,
                                               pipeline_df, row)
    out["reportingDate"] = reporting_date
    out["forecast"] = {k: forecast_meta.get(k) for k in
                       ("methodology", "basis", "observationWindowStart",
                        "observationWindowEnd", "weeklyExtractsUsed")}
    return out


def compute_history(output_root, client_id: str, to_run_id: Optional[str],
                    *, scope=None, test_id: Optional[str] = None,
                    max_periods: int = 24) -> Dict[str, Any]:
    """Metric history across REAL governed snapshots — never fabricated.

    Evaluates the CURRENT active configuration against each historical frame
    the evolution loader resolves, so the series is comparable period to
    period under today's approved definitions.
    """
    config, note = _active_configuration(client_id)
    if config is None:
        return {"available": False, "reason": note, "series": []}
    if not output_root:
        return {"available": False,
                "reason": "No governed snapshot history is configured.",
                "series": []}
    try:
        from . import evolution as _evolution
        frames = _evolution.funded_frames(output_root, client_id, to_run_id,
                                          scope=scope)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc), "series": []}
    if not frames:
        return {"available": False,
                "reason": "No historical funded snapshots are available.",
                "series": []}
    lib = load_library()
    frames = frames[-max_periods:]
    tests = [t for t in config.tests
             if test_id is None or t.test_id == test_id]
    if test_id is not None and not tests:
        return {"available": False,
                "reason": "That test is not in the active configuration.",
                "series": []}
    series: List[Dict[str, Any]] = []
    for t in tests:
        points = []
        for frame in frames:
            single = ActiveConfiguration(client_id=config.client_id,
                                         version=config.version, tests=[t],
                                         library_version=config.library_version)
            evaluated = evaluate_active_tests(
                frame.get("df"), None, single, lib,
                reporting_date=str(frame.get("reporting_date") or ""))
            row = evaluated["tests"][0]
            points.append({
                "runId": frame.get("run_id"),
                "reportingDate": frame.get("reporting_date"),
                "value": row["currentValue"],
                "status": row["status"],
            })
        series.append({
            "testId": t.test_id,
            "displayName": t.display_name,
            "unit": t.unit,
            "threshold": t.threshold,
            "operator": t.operator,
            "warningFraction": t.warning_fraction,
            "points": points,
        })
    return {"available": True, "series": series,
            "configurationVersion": config.version,
            "periods": [{"runId": f.get("run_id"),
                         "reportingDate": f.get("reporting_date")}
                        for f in frames]}
