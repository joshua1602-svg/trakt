"""Pipeline MI dataset contract + deterministic pipeline snapshot.

The pipeline analogue of ``mi_dataset_contract`` + ``snapshots`` for the funded
book. It is deterministic and never touches the natural-language parser:

  * :func:`discover_pipeline_sources` walks a root for governed pipeline sources
    (the onboarding ``18a_central_pipeline_tape.csv`` and/or raw M2L KFI / pipeline
    extracts) so the pipeline reporting-date selection is data-driven;
  * :func:`load_prepared_pipeline` reads a pipeline source and applies the
    pipeline MI preparation layer (``pipeline_prep.prepare_pipeline_mi_dataset``);
  * :func:`build_pipeline_dataset_contract` builds the per-field contract by
    REUSING the funded ``build_dataset_contract`` and adds the pipeline-specific
    ``fieldCorrelationToFunded`` + ``forecastReadiness`` metadata;
  * :func:`compute_pipeline_snapshot` derives the API ``pipelineSnapshot`` block
    (row count, pipeline amount, weighted expected funded amount, stage breakdown,
    expected-completion breakdown, available metrics/dimensions, data quality).

Pipeline records stay SEPARATE from the funded central lender tape.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analytics_lib.numeric import coerce_numeric

from .mi_dataset_contract import build_dataset_contract
from .pipeline_prep import (
    field_correlation_to_funded,
    forecast_readiness,
    prepare_pipeline_mi_dataset,
)
from . import pipeline_history as _history

_CENTRAL_PIPELINE_TAPE = "18a_central_pipeline_tape.csv"
_PREPARED_PIPELINE_NAME = "20_prepared_pipeline_mi.csv"

# Governed pipeline source filename patterns (xlsx or csv).
_PIPELINE_SOURCE_GLOBS = [
    _CENTRAL_PIPELINE_TAPE,
    "M2L*KFI*Pipeline*.csv", "M2L*KFI*Pipeline*.xlsx",
    "M2L*KFI*.csv", "M2L*KFI*.xlsx",
    "*KFI*Pipeline*.csv", "*KFI*Pipeline*.xlsx",
]

_FULL_DATE_RE = re.compile(r"(\d{4})[_\-.](\d{2})[_\-.](\d{2})")
_MONTH_RE = re.compile(r"(\d{4})[_\-.](\d{2})")
_NON_CLIENT_PARTS = {"output", "outputs", "runs", "onboarding", "central",
                     "pipeline", "mi", "fixtures", "tests", ""}


# --------------------------------------------------------------------------- #
# Date concepts (pipeline is a continuous weekly operational view — these are
# deliberately DISTINCT from the funded reporting cut-off):
#   pipeline_source_folder_date : the source/fixture folder date (e.g. the
#                                 monthly scope ``pipeline/2025-11-01``);
#   pipeline_extract_date       : the date parsed from the selected weekly file
#                                 (e.g. ``...2025_12_01_115711`` -> 2025-12-01);
#   pipeline_as_of_date         : operational as-of for the snapshot (== the
#                                 latest selected weekly extract date);
#   run_id                      : the funded MI run the scope aligns to, derived
#                                 from the FOLDER month (e.g. ``mi_2025_11``).
# --------------------------------------------------------------------------- #
def _read_source(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path)
        return pd.read_csv(path, low_memory=False)
    except Exception:  # noqa: BLE001 - a bad file must not break discovery
        return None


def _date_in(text: str) -> Optional[str]:
    """A full ``YYYY-MM-DD`` date parsed from a single string token, else None."""
    m = _FULL_DATE_RE.search(text or "")
    if m:
        return "-".join(m.groups())
    m = _MONTH_RE.search(text or "")
    return f"{m.group(1)}-{m.group(2)}-01" if m else None


def _extract_date(file_path: Path) -> Optional[str]:
    """The weekly extract date parsed from the FILE name only."""
    return _date_in(file_path.name)


def _folder_date(folder: Path) -> Optional[str]:
    """The source-scope (folder) date parsed from the folder NAME only."""
    return _date_in(folder.name)


def _run_id_for(folder_date: Optional[str], scope_date: Optional[str],
                path: Path) -> Optional[str]:
    """The funded MI run id the scope aligns to.

    Prefers an explicit ``mi_YYYY_MM`` path component (a promoted run dir); else
    derives ``mi_<year>_<month>`` from the FOLDER month (never the weekly extract
    month, which may roll into the next month)."""
    for part in path.parts:
        if re.fullmatch(r"mi_\d{4}_\d{2}", part):
            return part
    basis = folder_date or scope_date
    if basis:
        m = _MONTH_RE.search(basis)
        if m:
            return f"mi_{m.group(1)}_{m.group(2)}"
    return None


def _infer_client(path: Path, root: Path) -> Optional[str]:
    try:
        parts = list(path.relative_to(root).parts)
    except ValueError:
        parts = list(path.parts)
    for p in parts:
        if p.lower().startswith("client_"):
            return p
    for p in parts:
        if p.lower() not in _NON_CLIENT_PARTS and not _MONTH_RE.search(p) \
                and "." not in p:
            return p
    return None


def discover_pipeline_sources(root: str | os.PathLike,
                              client_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover governed weekly pipeline sources under ``root``, grouped by scope.

    Pipeline files are weekly operational extracts: a single source SCOPE (the
    folder, e.g. ``pipeline/2025-11-01``) may contain several weekly files, and
    the LATEST one (by extract date) represents that scope. One entry is returned
    per (client, scope) with the date concepts kept separate — there is no single
    ambiguous ``reporting_date``:

        {client_id, run_id, pipeline_source_folder, pipeline_source_folder_date,
         pipeline_extract_date, pipeline_as_of_date, source_file, row_count,
         weekly_files}

    Ordered oldest -> newest by folder/as-of date.
    """
    root = Path(root)
    if not root.exists():
        return []

    # Collect candidate weekly files (rich M2L/KFI sources preferred; the thin
    # 18a central pipeline tape is only a last-resort fallback when no source
    # extract exists anywhere under the root).
    source_globs = [g for g in _PIPELINE_SOURCE_GLOBS if g != _CENTRAL_PIPELINE_TAPE]
    files = _collect_files(root, source_globs)
    if not files:
        files = _collect_files(root, [_CENTRAL_PIPELINE_TAPE])

    # Group weekly files by (client, source folder) scope.
    scopes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in files:
        cid = _infer_client(path, root) or (client_id or "client_001")
        if client_id and cid != client_id:
            continue
        folder = path.parent
        folder_date = _folder_date(folder)
        extract_date = _extract_date(path)
        key = (cid, str(folder))
        scope = scopes.setdefault(key, {
            "client_id": cid,
            "pipeline_source_folder": str(folder),
            "pipeline_source_folder_date": folder_date,
            "weekly_files": [],
            "_best": None,
        })
        entry = {"source_file": str(path), "pipeline_extract_date": extract_date}
        scope["weekly_files"].append(entry)
        # Latest weekly file by extract date (fallback to filename order).
        best = scope["_best"]
        if best is None or (extract_date or "") >= (best["pipeline_extract_date"] or ""):
            scope["_best"] = entry

    found: List[Dict[str, Any]] = []
    for scope in scopes.values():
        best = scope.pop("_best")
        if best is None:
            continue
        df = _read_source(Path(best["source_file"]))
        if df is None or df.empty:
            continue
        folder_date = scope["pipeline_source_folder_date"]
        extract_date = best["pipeline_extract_date"]
        as_of = extract_date or folder_date
        scope.update({
            "run_id": _run_id_for(folder_date, as_of, Path(best["source_file"])),
            "pipeline_extract_date": extract_date,
            "pipeline_as_of_date": as_of,
            "source_file": best["source_file"],
            "row_count": int(len(df)),
            "weekly_files": sorted(
                scope["weekly_files"], key=lambda e: e["pipeline_extract_date"] or ""),
        })
        found.append(scope)

    found.sort(key=lambda r: (r["pipeline_source_folder_date"] or "",
                              r["pipeline_as_of_date"] or ""))
    return found


# Funded/funder source files that may be (mis)materialised under output/pipeline/
# but are NOT governed pipeline/KFI sources — never counted for the model.
_NON_PIPELINE_NAME_RE = re.compile(
    r"(funder|principal[\s_]*and[\s_]*interest|central[\s_]*lender|loan[\s_]*extract|"
    r"funded[\s_]*tape)", re.IGNORECASE)


def _is_governed_pipeline_file(path: Path) -> bool:
    """A governed pipeline/KFI source (not a funded/funder principal file)."""
    name = path.name
    if _NON_PIPELINE_NAME_RE.search(name):
        return False
    return True


def _collect_files(root: Path, globs: List[str]) -> List[Path]:
    """Governed pipeline source files under ``root``.

    Raw onboarding INPUT copies (under an ``input/`` path) are skipped — only the
    governed ``output/pipeline/`` materialised sources (or a flat fixture pack) are
    discovered, so a promoted run does not surface its own input twice. Funded /
    funder principal files that may sit alongside pipeline files are excluded by
    name so they are never counted as weekly pipeline model evidence.
    """
    seen: set = set()
    out: List[Path] = []
    for pattern in globs:
        for path in sorted(root.glob(f"**/{pattern}")):
            if path in seen or not path.is_file():
                continue
            try:
                rel_parts = {p.lower() for p in path.relative_to(root).parts[:-1]}
            except ValueError:
                rel_parts = {p.lower() for p in path.parts[:-1]}
            if "input" in rel_parts:
                continue  # raw onboarding input — not a governed pipeline source
            if not _is_governed_pipeline_file(path):
                continue  # funded/funder file — not a governed pipeline source
            seen.add(path)
            out.append(path)
    return out


def load_prepared_pipeline(source: str | os.PathLike | Dict[str, Any],
                           as_of_date: Optional[str] = None,
                           historical_model: Optional[Dict[str, Any]] = None
                           ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read a pipeline source and apply the pipeline MI preparation layer.

    ``source`` may be a path or a discovery scope dict (whose ``source_file`` and
    ``pipeline_as_of_date`` are used). The as-of date is the weekly extract date,
    NOT the funded reporting cut-off. ``historical_model`` (from
    :func:`build_pipeline_history`) supplies empirical stage completion rates.
    """
    if isinstance(source, dict):
        as_of_date = as_of_date or source.get("pipeline_as_of_date")
        source = source.get("source_file", "")
    p = Path(source)
    raw = _read_source(p)
    if raw is None:
        raise FileNotFoundError(f"cannot read pipeline source {p}")
    rd = as_of_date or _extract_date(p)
    return prepare_pipeline_mi_dataset(raw, as_of_date=rd, source_file=p.name,
                                       historical_model=historical_model)


def collect_weekly_history(root: str | os.PathLike,
                           client_id: str) -> List[Dict[str, Any]]:
    """All governed weekly pipeline files for a client across every scope, in
    chronological order (for the historical completion-rate model)."""
    entries: List[Dict[str, Any]] = []
    for scope in discover_pipeline_sources(root, client_id=client_id):
        entries.extend(scope.get("weekly_files", []) or [])
    entries.sort(key=lambda e: (e.get("pipeline_extract_date") or "",
                                e.get("source_file") or ""))
    return entries


def build_pipeline_history(root: str | os.PathLike,
                           client_id: str) -> Dict[str, Any]:
    """Build the historical completion model from a client's weekly pipeline files."""
    from .pipeline_history import build_historical_completion_model
    return build_historical_completion_model(collect_weekly_history(root, client_id))


def resolve_pipeline_source(root: str | os.PathLike, client_id: str,
                            run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The pipeline scope for a client/run, matched by the funded run's month.

    ``run_id`` is the funded MI run (``mi_2025_11``); the matching pipeline scope
    is the one whose source-folder month aligns. Matching uses the scope's
    ``run_id`` and ``pipeline_source_folder_date`` (the monthly scope) — NOT the
    weekly extract date, which may roll into the next month.
    """
    sources = discover_pipeline_sources(root, client_id=client_id)
    if not sources:
        return None
    if run_id:
        want_ym = _year_month(str(run_id))
        for s in sources:
            if (s.get("run_id") == run_id
                    or (want_ym and _year_month(s.get("run_id") or "") == want_ym)
                    or (want_ym and _year_month(s.get("pipeline_source_folder_date") or "") == want_ym)):
                return s
    return sources[-1]  # latest scope


def _year_month(text: str) -> Optional[str]:
    m = _MONTH_RE.search(text or "")
    return f"{m.group(1)}-{m.group(2)}" if m else None


# --------------------------------------------------------------------------- #
# Dataset contract (per-field) + correlation/forecast metadata
# --------------------------------------------------------------------------- #
def build_pipeline_dataset_contract(
    df: pd.DataFrame,
    semantics: dict,
    prep_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Per-field pipeline contract (reuses the funded contract builder) plus the
    pipeline correlation + forecast-readiness blocks."""
    base = build_dataset_contract(df, semantics, None)
    base["recordType"] = "pipeline"
    base["fieldCorrelationToFunded"] = field_correlation_to_funded(df)
    base["forecastReadiness"] = forecast_readiness(df)
    if prep_report:
        base["dataQuality"] = prep_report.get("data_quality", [])
        base["stageCounts"] = prep_report.get("stage_counts", {})
    return base


# --------------------------------------------------------------------------- #
# Pipeline snapshot (API block)
# --------------------------------------------------------------------------- #
def _expected_completion_breakdown(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if "expected_completion_month" not in df.columns:
        return []
    rows: List[Dict[str, Any]] = []
    grp = df.groupby(df["expected_completion_month"].astype(str), dropna=False)
    for month, sub in grp:
        if not month or month in ("nan", "NaT", "None"):
            continue
        weighted = (coerce_numeric(sub["weighted_expected_funded_amount"]).sum()
                    if "weighted_expected_funded_amount" in sub.columns else None)
        rows.append({
            "month": month,
            "caseCount": int(len(sub)),
            "expectedFundedAmount": round(float(
                coerce_numeric(sub["expected_funded_amount"]).sum()
                if "expected_funded_amount" in sub.columns else 0.0), 2),
            "weightedExpectedFundedAmount": (round(float(weighted), 2)
                                             if weighted is not None else None),
        })
    rows.sort(key=lambda r: r["month"])
    return rows


def _dimension_breakdown(df: pd.DataFrame, field: str,
                         key_name: str = "key") -> List[Dict[str, Any]]:
    """Backend-side ``[{key, caseCount, pipelineAmount, weightedExpected...}]`` for
    one dimension (so the UI never aggregates). Ordered by amount desc."""
    if field not in df.columns:
        return []
    rows: List[Dict[str, Any]] = []
    for key, sub in df.groupby(df[field].astype(str), dropna=False):
        if not str(key).strip() or str(key) in ("nan", "NaT", "None"):
            continue
        amount = (coerce_numeric(sub["current_outstanding_balance"]).sum()
                  if "current_outstanding_balance" in sub.columns else 0.0)
        weighted = (coerce_numeric(sub["weighted_expected_funded_amount"]).sum()
                    if "weighted_expected_funded_amount" in sub.columns else None)
        rows.append({
            key_name: str(key),
            "caseCount": int(len(sub)),
            "pipelineAmount": round(float(amount), 2),
            "weightedExpectedFundedAmount": (round(float(weighted), 2)
                                             if weighted is not None else None),
        })
    rows.sort(key=lambda r: r["pipelineAmount"], reverse=True)
    return rows


def _stage_breakdown(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = _dimension_breakdown(df, "pipeline_stage", key_name="stage")
    rows.sort(key=lambda r: r["stage"])
    return rows


def cap_breakdown(rows: List[Dict[str, Any]], top_n: int = 10,
                  key_name: str = "key") -> List[Dict[str, Any]]:
    """Cap a long categorical breakdown to ``top_n`` rows: the top ``top_n - 1``
    categories by amount plus an aggregated ``Other`` row. Totals reconcile to the
    uncapped total. No-op when there are ``<= top_n`` categories.
    """
    if len(rows) <= top_n:
        return rows
    head = rows[: top_n - 1]
    tail = rows[top_n - 1:]
    total_amount = sum(r["pipelineAmount"] for r in rows) or 1.0
    other_amount = round(sum(r["pipelineAmount"] for r in tail), 2)
    weighted_vals = [r.get("weightedExpectedFundedAmount") for r in tail]
    other_weighted = (round(sum(v for v in weighted_vals if v is not None), 2)
                      if any(v is not None for v in weighted_vals) else None)
    other = {
        key_name: "Other",
        "caseCount": sum(r["caseCount"] for r in tail),
        "pipelineAmount": other_amount,
        "weightedExpectedFundedAmount": other_weighted,
        "isOther": True,
        "categoriesIncluded": len(tail),
        "sharePct": round(other_amount / total_amount * 100, 1),
    }
    return head + [other]


def compute_pipeline_snapshot(
    df: pd.DataFrame,
    report: Dict[str, Any],
    semantics: dict,
    *,
    client_id: str,
    run_id: str,
    source: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic pipeline snapshot block for one weekly operational cut.

    Dates are kept distinct: ``pipelineAsOfDate`` / ``pipelineExtractDate`` /
    ``pipelineSourceFolderDate`` describe the weekly pipeline extract; they are
    NOT the funded reporting date. There is deliberately no ``reportingDate`` key.
    """
    contract = build_pipeline_dataset_contract(df, semantics, report)
    weighted = report.get("weighted_expected_funded_amount")
    src = source or {}
    as_of = src.get("pipeline_as_of_date") or report.get("pipeline_as_of_date")
    # Long categorical breakdowns are capped to top 10 (+ Other) for the visual;
    # the uncapped detail stays in ``*BreakdownFull`` for the API / agent.
    broker_full = _dimension_breakdown(df, "broker_channel", key_name="key")
    region_full = _dimension_breakdown(df, "geographic_region_obligor", key_name="key")
    return {
        "ok": True,
        "recordType": "pipeline",
        "portfolioId": f"{client_id}/{run_id}",
        "client_id": client_id,
        "runId": run_id,
        "pipelineAsOfDate": as_of,
        "pipelineExtractDate": src.get("pipeline_extract_date") or report.get("pipeline_as_of_date"),
        "pipelineSourceFolderDate": src.get("pipeline_source_folder_date"),
        "pipelineSourceFolder": src.get("pipeline_source_folder"),
        "sourceFile": src.get("source_file"),
        "pipelineRowCount": int(report.get("row_count", len(df))),
        "pipelineAmount": report.get("total_pipeline_amount"),
        "expectedFundedAmount": report.get("expected_funded_amount"),
        "weightedExpectedFundedAmount": weighted,
        "completionProbabilityBasis": report.get("completion_probability_basis"),
        "completionProbabilitySummary": report.get("completion_probability_summary", {}),
        "historicalCompletionModel": report.get("historical_completion_model", {}),
        "historicalModelEvidence": _history.historical_model_evidence(
            report.get("historical_completion_model"),
            report.get("completion_probability_basis")),
        "stageBreakdown": _stage_breakdown(df),
        "expectedCompletionBreakdown": _expected_completion_breakdown(df),
        "brokerBreakdown": cap_breakdown(broker_full, 10),
        "brokerBreakdownFull": broker_full,
        "regionBreakdown": cap_breakdown(region_full, 10),
        "regionBreakdownFull": region_full,
        "availableMetrics": report.get("metrics_available", []),
        "availableDimensions": report.get("dimensions_available", []),
        "missingDimensions": report.get("missing_dimensions", []),
        "dataQuality": report.get("data_quality", []),
        "fieldCorrelationToFunded": report.get("field_correlation_to_funded", {}),
        "forecastReadiness": report.get("forecast_readiness", {}),
        "datasetContract": contract,
    }
