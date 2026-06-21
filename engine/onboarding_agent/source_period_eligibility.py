"""
source_period_eligibility.py
============================

Generic, config-driven **reporting-period eligibility** for source artefacts.

A monthly MI run (``mi_2025_10``) must build its central lender tape from the
sources that belong to *that* reporting period only — not from every loan-like
key discovered in the input directory. Without this gate a cumulative current-book
file (October 33 loans, November 73 loans) or a future-period pipeline/KFI file
pollutes an earlier run's universe.

This module decides, for each ``(file, sheet)``:

* its inferred reporting period (``YYYY-MM``) and cut-off date, from — in priority
  order — an operator/config override, a period column inside the data
  (``Month Run`` / ``cut_off_date`` / ``reporting month`` …), a date embedded in
  the file name (with a configurable delivery offset), a period folder in the
  path, or the profiler's detected reporting date;
* whether it is **period-eligible** for the run (its period equals the run period,
  or it carries the run period among several, or its period is unknown and the
  config permits unknown-period sources);
* whether it is a **universe source** (an eligible funded / current-book role
  whose rows define the lender-tape universe) vs an enrichment-only source.

It is deliberately NOT lender-specific: roles, period columns, the filename
delivery offset and per-file overrides are all configurable. Nothing here mutates
source data — period detection produces comparison metadata only, recorded as an
auditable artefact (``04c_source_period_eligibility.csv`` / ``.json``).

Real-pack mechanism handled: a single current-book extract with a ``Month Run``
column valued ``October`` / ``November`` is **row-filtered** to the run period, so
``mi_2025_10`` yields 33 loans and ``mi_2025_11`` yields the cumulative 73.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from . import run_context as _rc

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "system" / "onboarding_agent.yaml"

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    # Roles whose (eligible) rows define the central lender tape universe.
    "funded_current_book_roles": [
        "current_loan_report", "loan_book", "servicing_report",
        "current_book", "funded_book", "warehouse_agreement",
    ],
    # Roles that must never define the universe (enrichment / forward-looking).
    "pipeline_roles": ["pipeline_report", "kfi", "application_pipeline"],
    # A file dated YYYY-MM-01 may represent the PRIOR month close: set to -1 to
    # shift a filename-derived period back one month (e.g. 2025_11_01 -> October).
    "filename_delivery_offset_months": 0,
    # Source columns read for an explicit per-row / per-file reporting period.
    "period_columns": [
        "month run", "month_run", "cut off date", "cut_off_date", "cutoff date",
        "reporting month", "reporting_month", "reporting period", "reporting_period",
        "report date", "as at date", "as_at_date", "snapshot date",
    ],
    # Treat unknown-period sources as eligible (permissive). Set false to require
    # an inferred period before a source may contribute.
    "allow_unknown_period": True,
    # Per-file explicit overrides: {file_name: "YYYY-MM"}.
    "overrides": {},
}


def load_config(config_path: str | Path = "") -> Dict[str, Any]:
    cfg = dict(_DEFAULTS)
    path = Path(config_path) if config_path else _CONFIG_PATH
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        block = doc.get("source_period_eligibility") or {}
        if isinstance(block, dict):
            for k, v in block.items():
                cfg[k] = v
    except Exception:
        pass
    return cfg


def _norm_col(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s or "").strip().lower()).strip("_")


def _shift_period(period: str, months: int) -> str:
    """Shift a ``YYYY-MM`` period by ``months`` (can be negative)."""
    m = re.fullmatch(r"(\d{4})-(\d{2})", str(period or ""))
    if not m:
        return period
    y, mo = int(m.group(1)), int(m.group(2))
    idx = (y * 12 + (mo - 1)) + int(months or 0)
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


# --------------------------------------------------------------------------- #
# Period parsing
# --------------------------------------------------------------------------- #

def period_of_value(value: Any, run_year: Optional[int] = None) -> str:
    """Parse a single cell to a ``YYYY-MM`` reporting period (deterministic).

    Handles ISO / D-M-Y / ``Month YYYY`` via :mod:`run_context`, and a *bare*
    month name (``October``) or month number using ``run_year`` for the year.
    Returns ``""`` when it cannot resolve a period without guessing.
    """
    iso = _rc.normalize_to_iso(value)
    if iso:
        return iso[:7]
    s = str(value or "").strip().lower()
    if not s:
        return ""
    # Bare month name / abbreviation -> needs the run year.
    for tok in re.split(r"[\s,_\-/]+", s):
        mo = _rc._MONTHS.get(tok)
        if mo and run_year:
            return f"{run_year:04d}-{mo:02d}"
    # Bare month number 1..12 with a run year.
    if run_year and re.fullmatch(r"(0?[1-9]|1[0-2])", s):
        return f"{run_year:04d}-{int(s):02d}"
    return ""


def run_period(run_id: str, input_dir: str | Path = "") -> Tuple[str, str]:
    """``(period 'YYYY-MM', cutoff 'YYYY-MM-DD')`` for the run, from the run id
    (preferred) or an ``input/2025-10`` period folder. Empty when undetermined."""
    for token in (run_id, Path(str(input_dir)).name if input_dir else ""):
        for iso in _rc.dates_from_period_token(token):
            return iso[:7], iso
    return "", ""


def _path_period(file_path: str) -> str:
    """Reporting period from a period-looking folder in the file's path."""
    for seg in Path(str(file_path)).parts:
        for iso in _rc.dates_from_period_token(seg):
            return iso[:7]
    return ""


def _detect_period_column(df, period_columns: List[str]) -> str:
    if df is None:
        return ""
    wanted = {_norm_col(c) for c in period_columns}
    for col in df.columns:
        if _norm_col(col) in wanted:
            return str(col)
    return ""


def _column_periods(df, column: str, run_year: Optional[int]) -> List[str]:
    """Distinct ``YYYY-MM`` periods present in a period column (order-stable)."""
    if df is None or not column or column not in df.columns:
        return []
    out: List[str] = []
    seen = set()
    for v in df[column].tolist():
        p = period_of_value(v, run_year)
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

_COLUMNS = [
    "source_file", "source_sheet", "artefact_role", "inferred_reporting_period",
    "inferred_cutoff_date", "period_column", "run_id", "run_reporting_period",
    "is_period_eligible", "is_universe_source", "eligibility_basis",
    "reason_excluded", "confidence",
]


@dataclass
class SourcePeriodEligibility:
    source_file: str
    source_sheet: str
    artefact_role: str
    inferred_reporting_period: str
    inferred_cutoff_date: str
    period_column: str
    run_id: str
    run_reporting_period: str
    is_period_eligible: bool
    is_universe_source: bool
    eligibility_basis: str
    reason_excluded: str
    confidence: float

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in _COLUMNS}


def _cutoff_for_period(period: str) -> str:
    m = re.fullmatch(r"(\d{4})-(\d{2})", str(period or ""))
    if not m:
        return ""
    return _rc._month_end_iso(int(m.group(1)), int(m.group(2))) or ""


def _infer_source_period(
    file_name: str, file_path: str, role: str, detected_reporting_date: str,
    df, cfg: Dict[str, Any], run_year: Optional[int],
) -> Tuple[str, str, str, str, float, List[str]]:
    """Return ``(period, cutoff, period_column, basis, confidence, present_periods)``.

    ``period`` is the single inferred period (``""`` when several / unknown);
    ``present_periods`` are all periods carried by a period column (for row-level
    filtering of a cumulative file).
    """
    overrides = cfg.get("overrides") or {}
    period_cols = cfg.get("period_columns") or []

    # 1. Operator / config override (highest priority).
    ov = overrides.get(file_name) or overrides.get(Path(str(file_path)).name)
    if ov:
        p = period_of_value(ov, run_year) or str(ov)[:7]
        if re.fullmatch(r"\d{4}-\d{2}", p):
            return p, _cutoff_for_period(p), "", "operator_config", 1.0, [p]

    # 2. Period column inside the data (supports row-level cumulative filtering).
    col = _detect_period_column(df, period_cols)
    if col:
        present = _column_periods(df, col, run_year)
        if len(present) == 1:
            return present[0], _cutoff_for_period(present[0]), col, "month_run_column", 0.95, present
        if len(present) > 1:
            # Cumulative / multi-period file: no single file period, but rows can
            # be filtered to the run period.
            return "", "", col, "month_run_column", 0.9, present
        # column exists but unparseable -> fall through to weaker signals

    # 3. Filename date (with configurable delivery offset).
    fdates = _rc.dates_from_filename(file_name)
    if fdates:
        p = _shift_period(fdates[0][:7], int(cfg.get("filename_delivery_offset_months", 0) or 0))
        return p, _cutoff_for_period(p), "", "filename_date", 0.6, [p]

    # 4. Period folder in the path.
    fp = _path_period(file_path)
    if fp:
        return fp, _cutoff_for_period(fp), "", "period_folder", 0.7, [fp]

    # 5. Profiler-detected reporting date.
    iso = _rc.normalize_to_iso(detected_reporting_date)
    if iso:
        return iso[:7], _cutoff_for_period(iso[:7]), "", "detected_reporting_date", 0.55, [iso[:7]]

    return "", "", "", "unknown", 0.0, []


def compute_eligibility(
    records: List[Dict[str, Any]],
    run_id: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    config_path: str | Path = "",
    input_dir: str | Path = "",
) -> List[SourcePeriodEligibility]:
    """Resolve period eligibility for each source record.

    ``records``: ``[{file_name, file_path, sheet_name, artefact_role,
    detected_reporting_date, df (optional)}]``.
    """
    cfg = config or load_config(config_path)
    run_p, run_cut = run_period(run_id, input_dir)
    run_year = int(run_p[:4]) if re.fullmatch(r"\d{4}-\d{2}", run_p) else None
    allow_unknown = bool(cfg.get("allow_unknown_period", True))
    universe_roles = {_norm_col(r) for r in (cfg.get("funded_current_book_roles") or [])}

    out: List[SourcePeriodEligibility] = []
    for rec in records:
        role = str(rec.get("artefact_role", "") or "")
        period, cutoff, pcol, basis, conf, present = _infer_source_period(
            rec.get("file_name", ""), rec.get("file_path", ""), role,
            rec.get("detected_reporting_date", ""), rec.get("df"), cfg, run_year)

        eligible = True
        reason = ""
        if run_p:
            if present:
                # File carries one or more concrete periods.
                if run_p in present:
                    eligible = True
                else:
                    eligible = False
                    future = [p for p in present if p > run_p]
                    reason = ("future_period" if future and all(p > run_p for p in present)
                              else "period_mismatch")
            elif period:
                eligible = (period == run_p)
                if not eligible:
                    reason = "future_period" if period > run_p else "period_mismatch"
            else:
                # Unknown period.
                eligible = allow_unknown
                reason = "" if allow_unknown else "no_period_detected"
        # When the run period itself is unknown, nothing can be excluded on period.

        is_universe = bool(eligible and _norm_col(role) in universe_roles)
        out.append(SourcePeriodEligibility(
            source_file=rec.get("file_name", ""),
            source_sheet=rec.get("sheet_name", ""),
            artefact_role=role,
            inferred_reporting_period=period,
            inferred_cutoff_date=cutoff,
            period_column=pcol,
            run_id=run_id,
            run_reporting_period=run_p,
            is_period_eligible=eligible,
            is_universe_source=is_universe,
            eligibility_basis=basis,
            reason_excluded=reason,
            confidence=round(float(conf), 4),
        ))
    return out


# --------------------------------------------------------------------------- #
# Artefacts + loading
# --------------------------------------------------------------------------- #

_ARTEFACT_CSV = "04c_source_period_eligibility.csv"
_ARTEFACT_JSON = "04c_source_period_eligibility.json"


def write_artifacts(rows: List[SourcePeriodEligibility], out_dir: str | Path) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dict_rows = [r.as_dict() for r in rows]
    csv_path = out / _ARTEFACT_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in dict_rows:
            w.writerow(r)
    json_path = out / _ARTEFACT_JSON
    json_path.write_text(json.dumps({"rows": dict_rows}, indent=2, default=str), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}


def load_eligibility(project_dir: str | Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Load 04c as ``{(file_name, sheet_name): row_dict}``. Empty when absent."""
    p = Path(project_dir) / _ARTEFACT_JSON
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in (data.get("rows", []) if isinstance(data, dict) else []):
        out[(r.get("source_file", ""), r.get("source_sheet", ""))] = r
    return out


def resolve_and_write(
    inventory: List[Dict[str, Any]],
    run_id: str,
    out_dir: str | Path,
    *,
    config_path: str | Path = "",
    input_dir: str | Path = "",
    enable_conversion: bool = False,
) -> Dict[str, Any]:
    """Load all (file, sheet) tables, resolve period eligibility, write 04c."""
    from . import source_table_loader as stl

    role_by_file = {i.get("file_name", ""): i.get("classification", "") for i in inventory}
    detected_by_file = {i.get("file_name", ""): i.get("detected_reporting_date", "")
                        for i in inventory}
    tables, _, _ = stl.load_source_tables(inventory, enable_conversion=enable_conversion)
    records: List[Dict[str, Any]] = []
    for t in tables:
        records.append({
            "file_name": t.file_name, "file_path": t.file_path,
            "sheet_name": t.sheet_name,
            "artefact_role": role_by_file.get(t.file_name, ""),
            "detected_reporting_date": detected_by_file.get(t.file_name, ""),
            "df": t.df,
        })
    rows = compute_eligibility(records, run_id, config_path=config_path, input_dir=input_dir)
    paths = write_artifacts(rows, out_dir)
    return {"rows": [r.as_dict() for r in rows], "paths": paths}
