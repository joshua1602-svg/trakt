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

Two cadences apply to the ``central_lender_tape`` output domain. **Universe**
(funded / current-book) sources must match the run period exactly (a cumulative
file is row-filtered to the run period). **Enrichment** sources (collateral
valuation, warehouse, cashflow) follow an *as-of* cadence: a source delivered in
or before the run period is the latest-available enrichment for funded loans
still on book, so it stays eligible; only a future-period enrichment source is
excluded. Enrichment never creates funded rows, so an earlier-period source
cannot change the funded count.

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
    # Roles whose (eligible) rows define the central lender tape universe, in
    # precedence order. warehouse_agreement / collateral_report are enrichment by
    # default and only become universe sources when explicitly configured here.
    "funded_current_book_roles": [
        "current_loan_report", "funded_book", "current_book",
        "loan_book", "servicing_report",
    ],
    # Roles that enrich the lender tape (join fields) but never define its rows.
    "enrichment_roles": ["collateral_report", "warehouse_agreement", "cashflow_report"],
    # Roles evaluated under the pipeline snapshot cadence (NOT the funded-book
    # period). These never drive central_lender_tape but stay available for
    # pipeline_mi / forward_exposure on their own delivery cadence.
    "pipeline_roles": ["pipeline_report", "kfi", "application_pipeline"],
    # A file dated YYYY-MM-01 may represent the PRIOR month close: set to -1 to
    # shift a filename-derived period back one month (e.g. 2025_11_01 -> October).
    "filename_delivery_offset_months": 0,
    # Source columns read for an explicit per-row / per-file reporting period.
    "period_columns": [
        "month run", "month_run", "as of month", "as_of_month",
        "cut off date", "cut_off_date", "cutoff date",
        "reporting month", "reporting_month", "reporting period", "reporting_period",
        "report date", "as at date", "as_at_date", "snapshot date",
    ],
    # Treat unknown-period sources as eligible (permissive). Set false to require
    # an inferred period before a source may contribute.
    "allow_unknown_period": True,
    # Per-file explicit overrides: {file_name: "YYYY-MM"}.
    "overrides": {},
    # Output-domain routing: the same source can be ineligible for one output and
    # eligible for another (a pipeline file drives pipeline_mi, never the tape).
    "output_domains": {
        "central_lender_tape": {"cadence": "funded_book_period"},
        "pipeline_mi": {"eligible_roles": ["pipeline_report", "kfi", "application_pipeline"],
                        "cadence": "pipeline_snapshot"},
        "forward_exposure": {"eligible_roles": ["pipeline_report", "kfi"],
                             "cadence": "pipeline_snapshot"},
        "regulatory": {"eligible_roles": [], "cadence": "funded_book_period"},
    },
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


# --------------------------------------------------------------------------- #
# Expected-balance reconciliation (generic, config-driven, opt-in)
# --------------------------------------------------------------------------- #

def _coerce_float(v: Any) -> Optional[float]:
    """Numeric-normalise a value, stripping commas / currency symbols / blanks."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = re.sub(r"[^0-9.\-]", "", str(v).strip())
    if s in ("", "-", ".", "--"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_expected_balance_checks(config_path: str | Path = "") -> Dict[str, Any]:
    """Load the top-level ``expected_balance_checks`` config block (empty when
    absent). Generic: keyed by client id then run id (or period) — never hard-coded."""
    path = Path(config_path) if config_path else _CONFIG_PATH
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        block = doc.get("expected_balance_checks") or {}
        return block if isinstance(block, dict) else {}
    except Exception:
        return {}


def lookup_expected(
    checks: Dict[str, Any], client_id: str, run_id: str, run_period: str = "",
) -> Optional[Dict[str, Any]]:
    """Resolve the expected-balance entry for a run, trying (in order):
    ``checks[client_id][run_id]`` -> ``[client_id][run_period]`` ->
    ``checks[run_id]`` -> ``checks[run_period]``. ``None`` when not configured."""
    if not isinstance(checks, dict):
        return None
    client_block = checks.get(client_id) if client_id else None
    for src in (client_block, checks):
        if isinstance(src, dict):
            for key in (run_id, run_period):
                entry = src.get(key) if key else None
                if isinstance(entry, dict):
                    return entry
    return None


_BLOCKING_SEVERITIES = {"blocking", "blocker", "fail", "error"}


def reconcile_balance(
    expected: Optional[Dict[str, Any]],
    actual_loan_count: int,
    actual_balance: Optional[float],
    balance_populated: int,
    balance_column_present: bool,
) -> Dict[str, Any]:
    """Run-level reconciliation diagnostic (never raises; pure data).

    Returns the 18f ``expected_balance_check`` block. ``balance_check_status`` is
    one of pass / warning / fail / not_configured — ``fail`` only when the entry is
    explicitly configured as a blocking severity. The default severity is warning,
    so an out-of-tolerance run reports a warning rather than failing promotion."""
    out: Dict[str, Any] = {
        "expected_loan_count": None,
        "actual_loan_count": int(actual_loan_count),
        "loan_count_match": None,
        "expected_current_outstanding_balance": None,
        "actual_current_outstanding_balance": (round(actual_balance, 2)
                                               if actual_balance is not None else None),
        "balance_delta": None,
        "balance_delta_pct": None,
        "tolerance_abs": None,
        "tolerance_pct": None,
        "severity": "warning",
        "balance_check_status": "not_configured",
        "diagnostic": "",
    }
    if not expected:
        return out

    exp_count = expected.get("expected_loan_count")
    exp_bal = _coerce_float(expected.get("expected_current_outstanding_balance"))
    tol_abs = _coerce_float(expected.get("tolerance_abs")) or 0.0
    tol_pct = _coerce_float(expected.get("tolerance_pct")) or 0.0
    severity = str(expected.get("severity", "warning") or "warning").strip().lower()
    out.update(expected_loan_count=exp_count,
               expected_current_outstanding_balance=exp_bal,
               tolerance_abs=tol_abs, tolerance_pct=tol_pct, severity=severity)

    diagnostics: List[str] = []
    loan_count_match = (exp_count is None) or (int(actual_loan_count) == int(exp_count))
    out["loan_count_match"] = loan_count_match
    if not loan_count_match:
        diagnostics.append(f"loan_count {actual_loan_count} != expected {exp_count}")

    balance_ok = True
    if not balance_column_present:
        balance_ok = False
        diagnostics.append("current_outstanding_balance column not present in central tape")
    elif balance_populated == 0 or actual_balance is None:
        balance_ok = False
        diagnostics.append("current_outstanding_balance missing or non-numeric for all rows")
    elif exp_bal is not None:
        delta = round(actual_balance - exp_bal, 2)
        out["balance_delta"] = delta
        out["balance_delta_pct"] = round(delta / exp_bal, 6) if exp_bal else None
        tol = max(tol_abs, abs(exp_bal) * tol_pct)
        balance_ok = abs(delta) <= tol
        if not balance_ok:
            diagnostics.append(f"balance delta {delta} exceeds tolerance {round(tol, 2)}")

    passed = loan_count_match and balance_ok
    if passed:
        out["balance_check_status"] = "pass"
    else:
        out["balance_check_status"] = "fail" if severity in _BLOCKING_SEVERITIES else "warning"
    out["diagnostic"] = "; ".join(diagnostics)
    return out


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
    "source_file", "source_sheet", "artefact_role", "output_domain", "run_id",
    "run_reporting_period", "inferred_reporting_period", "inferred_cutoff_date",
    "delivery_date", "cadence_rule", "is_period_eligible", "is_universe_source",
    "eligibility_basis", "reason_excluded", "confidence", "source_period_column",
    "source_period_raw_value", "source_period_canonical_value",
]


@dataclass
class SourcePeriodEligibility:
    source_file: str
    source_sheet: str
    artefact_role: str
    output_domain: str
    run_id: str
    run_reporting_period: str
    inferred_reporting_period: str
    inferred_cutoff_date: str
    delivery_date: str
    cadence_rule: str
    is_period_eligible: bool
    is_universe_source: bool
    eligibility_basis: str
    reason_excluded: str
    confidence: float
    source_period_column: str
    source_period_raw_value: str
    source_period_canonical_value: str

    # Back-compat alias (older callers used ``period_column``).
    @property
    def period_column(self) -> str:
        return self.source_period_column

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in _COLUMNS}


def _cutoff_for_period(period: str) -> str:
    m = re.fullmatch(r"(\d{4})-(\d{2})", str(period or ""))
    if not m:
        return ""
    return _rc._month_end_iso(int(m.group(1)), int(m.group(2))) or ""


def _column_raw_value(df, column: str, run_period_val: str, run_year: Optional[int]) -> str:
    """A representative raw cell from the period column — the run-period one when
    present (so 04c shows e.g. ``October``), else the first non-empty value."""
    if df is None or not column or column not in df.columns:
        return ""
    first = ""
    for v in df[column].tolist():
        s = str(v).strip()
        if not s or s.lower() in ("nan", "nat", "none", "<na>"):
            continue
        if not first:
            first = s
        if run_period_val and period_of_value(v, run_year) == run_period_val:
            return s
    return first


def _infer_source_period(
    file_name: str, file_path: str, role: str, detected_reporting_date: str,
    df, cfg: Dict[str, Any], run_year: Optional[int], run_period_val: str,
) -> Dict[str, Any]:
    """Resolve a source's reporting-period profile (deterministic, never invented).

    Returns ``{period, cutoff, period_column, basis, confidence, present,
    raw_value, delivery_date}`` where ``present`` lists all periods carried by a
    period column (for row-level filtering of a cumulative file).
    """
    overrides = cfg.get("overrides") or {}
    period_cols = cfg.get("period_columns") or []
    # Delivery date — a filename date (verbatim, no offset) or the profiler date —
    # used for the pipeline snapshot cadence.
    fdates = _rc.dates_from_filename(file_name)
    delivery = fdates[0] if fdates else (_rc.normalize_to_iso(detected_reporting_date) or "")

    def _result(period, basis, conf, present, col="", raw=""):
        return {"period": period, "cutoff": _cutoff_for_period(period), "period_column": col,
                "basis": basis, "confidence": conf, "present": present,
                "raw_value": raw, "delivery_date": delivery}

    # 1. Operator / config override (highest priority).
    ov = overrides.get(file_name) or overrides.get(Path(str(file_path)).name)
    if ov:
        p = period_of_value(ov, run_year) or str(ov)[:7]
        if re.fullmatch(r"\d{4}-\d{2}", p):
            return _result(p, "operator_config", 1.0, [p], raw=str(ov))

    # 2. Period column inside the data (supports row-level cumulative filtering).
    col = _detect_period_column(df, period_cols)
    if col:
        present = _column_periods(df, col, run_year)
        raw = _column_raw_value(df, col, run_period_val, run_year)
        if len(present) == 1:
            return _result(present[0], "month_run_column", 0.95, present, col, raw)
        if len(present) > 1:
            # Cumulative / multi-period file: no single file period, but rows can
            # be filtered to the run period.
            return _result("", "month_run_column", 0.9, present, col, raw)
        # column exists but unparseable -> fall through to weaker signals

    # 3. Filename date (with configurable delivery offset).
    if fdates:
        p = _shift_period(fdates[0][:7], int(cfg.get("filename_delivery_offset_months", 0) or 0))
        return _result(p, "filename_date", 0.6, [p], raw=fdates[0])

    # 4. Period folder in the path.
    fp = _path_period(file_path)
    if fp:
        return _result(fp, "period_folder", 0.7, [fp])

    # 5. Profiler-detected reporting date.
    iso = _rc.normalize_to_iso(detected_reporting_date)
    if iso:
        return _result(iso[:7], "detected_reporting_date", 0.55, [iso[:7]], raw=str(detected_reporting_date))

    return _result("", "unknown", 0.0, [])


def _funded_period_match(prof: Dict[str, Any], run_p: str, allow_unknown: bool) -> Tuple[bool, str]:
    """Funded-book cadence: the source period must equal the run period."""
    if not run_p:
        return True, ""
    present, period = prof["present"], prof["period"]
    if present:
        if run_p in present:
            return True, ""
        return False, ("future_period" if all(p > run_p for p in present) else "period_mismatch")
    if period:
        return (period == run_p, "" if period == run_p
                else ("future_period" if period > run_p else "period_mismatch"))
    return (allow_unknown, "" if allow_unknown else "no_period_detected")


def _enrichment_period_match(prof: Dict[str, Any], run_p: str, allow_unknown: bool) -> Tuple[bool, str]:
    """Enrichment cadence (collateral valuation / warehouse / cashflow).

    An enrichment source delivered in OR BEFORE the run period is the
    latest-available enrichment for funded loans still on book, so it stays
    eligible. Enrichment **never defines the lender-tape universe** (it adds no
    rows — see ``central_tape_builder``), so an earlier-period source is safe and
    cannot change the funded count. Only a FUTURE-period source is excluded (never
    enrich an October book with December valuations).

    This is the fix for the November-LTV regression: a collateral / PropertyExtract
    delivered for October is the latest valuation for the loans still funded in
    November, so it must enrich the November funded book rather than being dropped
    as a ``period_mismatch`` under the strict funded-book cadence.
    """
    if not run_p:
        return True, ""
    present, period = prof["present"], prof["period"]
    if present:
        if any(p <= run_p for p in present):
            return True, ""
        return False, "future_period"
    if period:
        return (period <= run_p, "" if period <= run_p else "future_period")
    return (allow_unknown, "" if allow_unknown else "no_period_detected")


def _pipeline_cadence_match(prof: Dict[str, Any], run_p: str, allow_unknown: bool) -> Tuple[bool, str]:
    """Pipeline snapshot cadence: a snapshot delivered AFTER the funded close is
    valid; do not exclude a pipeline file merely for a later delivery date."""
    if not run_p:
        return True, ""
    dp = (prof["delivery_date"][:7] if prof["delivery_date"] else "") or prof["period"]
    if not dp:
        return (allow_unknown, "" if allow_unknown else "no_period_detected")
    if dp >= run_p:
        return True, ""
    return False, "pre_period_pipeline_snapshot"


def compute_eligibility(
    records: List[Dict[str, Any]],
    run_id: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    config_path: str | Path = "",
    input_dir: str | Path = "",
) -> List[SourcePeriodEligibility]:
    """Resolve **output-domain-aware** period eligibility for each source record.

    Emits one row per applicable output domain: every source gets a
    ``central_lender_tape`` row (so exclusions are explicit); pipeline-role
    sources additionally get ``pipeline_mi`` / ``forward_exposure`` rows under the
    pipeline snapshot cadence. ``records``: ``[{file_name, file_path, sheet_name,
    artefact_role, detected_reporting_date, df (optional)}]``.
    """
    cfg = config or load_config(config_path)
    run_p, _run_cut = run_period(run_id, input_dir)
    run_year = int(run_p[:4]) if re.fullmatch(r"\d{4}-\d{2}", run_p) else None
    allow_unknown = bool(cfg.get("allow_unknown_period", True))
    universe_roles = {_norm_col(r) for r in (cfg.get("funded_current_book_roles") or [])}
    enrichment_roles = {_norm_col(r) for r in (cfg.get("enrichment_roles") or [])}
    pipeline_roles = {_norm_col(r) for r in (cfg.get("pipeline_roles") or [])}
    domains_cfg = cfg.get("output_domains") or {}

    out: List[SourcePeriodEligibility] = []
    for rec in records:
        role = str(rec.get("artefact_role", "") or "")
        role_n = _norm_col(role)
        prof = _infer_source_period(
            rec.get("file_name", ""), rec.get("file_path", ""), role,
            rec.get("detected_reporting_date", ""), rec.get("df"), cfg, run_year, run_p)

        for dom, dcfg in domains_cfg.items():
            cadence = str((dcfg or {}).get("cadence", ""))
            elig_roles = {_norm_col(r) for r in ((dcfg or {}).get("eligible_roles") or [])}
            if dom == "central_lender_tape":
                applies = True  # always recorded, so exclusions are explicit
            else:
                applies = role_n in elig_roles
            if not applies:
                continue

            is_universe = False
            if dom == "central_lender_tape":
                if role_n in pipeline_roles:
                    eligible, reason = False, "pipeline_role_excluded_from_lender_tape"
                elif role_n in universe_roles:
                    # Funded / current-book cadence: the period must equal the run
                    # period (a cumulative file is row-filtered downstream). This
                    # is what keeps mi_2025_10 at 33 rows and mi_2025_11 at 73.
                    eligible, reason = _funded_period_match(prof, run_p, allow_unknown)
                    is_universe = bool(eligible)
                elif role_n in enrichment_roles:
                    # Enrichment cadence: on-or-before the run period stays eligible
                    # (latest-available valuation/attribute); only future-period
                    # enrichment is excluded. Enrichment never creates funded rows.
                    eligible, reason = _enrichment_period_match(prof, run_p, allow_unknown)
                else:
                    eligible, reason = False, "role_not_funded_or_enrichment"
            elif cadence == "pipeline_snapshot":
                eligible, reason = _pipeline_cadence_match(prof, run_p, allow_unknown)
            else:
                eligible, reason = _funded_period_match(prof, run_p, allow_unknown)

            out.append(SourcePeriodEligibility(
                source_file=rec.get("file_name", ""),
                source_sheet=rec.get("sheet_name", ""),
                artefact_role=role,
                output_domain=dom,
                run_id=run_id,
                run_reporting_period=run_p,
                inferred_reporting_period=prof["period"],
                inferred_cutoff_date=prof["cutoff"],
                delivery_date=prof["delivery_date"],
                cadence_rule=cadence,
                is_period_eligible=eligible,
                is_universe_source=is_universe,
                eligibility_basis=prof["basis"],
                reason_excluded=reason,
                confidence=round(float(prof["confidence"]), 4),
                source_period_column=prof["period_column"],
                source_period_raw_value=prof["raw_value"],
                source_period_canonical_value=prof["cutoff"],
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


def load_eligibility(
    project_dir: str | Path, output_domain: str = "central_lender_tape",
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Load the 04c rows for one ``output_domain`` as
    ``{(file_name, sheet_name): row_dict}``. Empty when absent."""
    p = Path(project_dir) / _ARTEFACT_JSON
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in (data.get("rows", []) if isinstance(data, dict) else []):
        # Back-compat: rows written before output_domain default to the tape domain.
        if (r.get("output_domain", "central_lender_tape") or "central_lender_tape") != output_domain:
            continue
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
