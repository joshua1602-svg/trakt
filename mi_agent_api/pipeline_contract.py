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

_CENTRAL_PIPELINE_TAPE = "18a_central_pipeline_tape.csv"
_PREPARED_PIPELINE_NAME = "20_prepared_pipeline_mi.csv"

# Governed pipeline source filename patterns (xlsx or csv).
_PIPELINE_SOURCE_GLOBS = [
    _CENTRAL_PIPELINE_TAPE,
    "M2L*KFI*Pipeline*.csv", "M2L*KFI*Pipeline*.xlsx",
    "M2L*KFI*.csv", "M2L*KFI*.xlsx",
    "*KFI*Pipeline*.csv", "*KFI*Pipeline*.xlsx",
]

_DATE_DIR_RE = re.compile(r"(\d{4})[_\-.](\d{2})[_\-.](\d{2})")
_MONTH_RE = re.compile(r"(\d{4})[_\-.](\d{2})")
_NON_CLIENT_PARTS = {"output", "outputs", "runs", "onboarding", "central",
                     "pipeline", "mi", "fixtures", "tests", ""}


# --------------------------------------------------------------------------- #
# Source discovery
# --------------------------------------------------------------------------- #
def _read_source(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path)
        return pd.read_csv(path, low_memory=False)
    except Exception:  # noqa: BLE001 - a bad file must not break discovery
        return None


def _infer_reporting_date(path: Path) -> Optional[str]:
    """Reporting date from the path (``.../2025-11-01/...`` or ``...2025_11_01``)."""
    for part in (path.name, *[p for p in path.parts]):
        m = _DATE_DIR_RE.search(part)
        if m:
            return "-".join(m.groups())
    for part in (path.name, *[p for p in path.parts]):
        m = _MONTH_RE.search(part)
        if m:
            return f"{m.group(1)}-{m.group(2)}-01"
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
    """Discover governed pipeline source files under ``root``.

    Returns a list of ``{client_id, reporting_date, run_id, path, row_count}``
    ordered oldest -> newest by reporting date. Unreadable files are skipped.
    """
    root = Path(root)
    if not root.exists():
        return []
    seen: set = set()
    found: List[Dict[str, Any]] = []
    for pattern in _PIPELINE_SOURCE_GLOBS:
        for path in sorted(root.glob(f"**/{pattern}")):
            if path in seen:
                continue
            seen.add(path)
            df = _read_source(path)
            if df is None or df.empty:
                continue
            cid = _infer_client(path, root) or (client_id or "client_001")
            if client_id and cid != client_id:
                continue
            reporting_date = _infer_reporting_date(path)
            run_id = (f"pipeline_{reporting_date[:7].replace('-', '_')}"
                      if reporting_date else path.stem)
            found.append({
                "client_id": cid,
                "reporting_date": reporting_date,
                "run_id": run_id,
                "path": str(path),
                "row_count": int(len(df)),
            })
    found.sort(key=lambda r: (r["reporting_date"] or "", r["path"]))
    return found


def load_prepared_pipeline(path: str | os.PathLike,
                           reporting_date: Optional[str] = None
                           ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read a pipeline source and apply the pipeline MI preparation layer."""
    p = Path(path)
    raw = _read_source(p)
    if raw is None:
        raise FileNotFoundError(f"cannot read pipeline source {p}")
    rd = reporting_date or _infer_reporting_date(p)
    return prepare_pipeline_mi_dataset(raw, reporting_date=rd, source_file=p.name)


def resolve_pipeline_source(root: str | os.PathLike, client_id: str,
                            run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The pipeline source for a client (and run/reporting date if given).

    ``run_id`` may be a pipeline run id (``pipeline_2025_11``), a reporting date
    (``2025-11-01``) or a funded run id (``mi_2025_11``) — matched by year-month
    so a funded run resolves to the pipeline cut for the same period.
    """
    sources = discover_pipeline_sources(root, client_id=client_id)
    if not sources:
        return None
    if run_id:
        want_ym = _year_month(str(run_id))
        for s in sources:
            if (s["run_id"] == run_id or s["reporting_date"] == run_id
                    or (s["reporting_date"] or "").startswith(str(run_id))
                    or (want_ym and _year_month(s["reporting_date"] or "") == want_ym)
                    or (want_ym and _year_month(s["run_id"]) == want_ym)):
                return s
    return sources[-1]  # latest


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


def compute_pipeline_snapshot(
    df: pd.DataFrame,
    report: Dict[str, Any],
    semantics: dict,
    *,
    client_id: str,
    run_id: str,
    reporting_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministic pipeline snapshot block for one reporting cut."""
    contract = build_pipeline_dataset_contract(df, semantics, report)
    weighted = report.get("weighted_expected_funded_amount")
    return {
        "ok": True,
        "recordType": "pipeline",
        "portfolioId": f"{client_id}/{run_id}",
        "client_id": client_id,
        "runId": run_id,
        "reportingDate": reporting_date or report.get("reporting_date"),
        "pipelineRowCount": int(report.get("row_count", len(df))),
        "pipelineAmount": report.get("total_pipeline_amount"),
        "expectedFundedAmount": report.get("expected_funded_amount"),
        "weightedExpectedFundedAmount": weighted,
        "stageBreakdown": _stage_breakdown(df),
        "expectedCompletionBreakdown": _expected_completion_breakdown(df),
        "brokerBreakdown": _dimension_breakdown(df, "broker_channel", key_name="key"),
        "regionBreakdown": _dimension_breakdown(df, "geographic_region_obligor", key_name="key"),
        "availableMetrics": report.get("metrics_available", []),
        "availableDimensions": report.get("dimensions_available", []),
        "missingDimensions": report.get("missing_dimensions", []),
        "dataQuality": report.get("data_quality", []),
        "fieldCorrelationToFunded": report.get("field_correlation_to_funded", {}),
        "forecastReadiness": report.get("forecast_readiness", {}),
        "datasetContract": contract,
    }
