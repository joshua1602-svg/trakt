"""Deterministic funded-portfolio snapshot + reporting-run discovery.

This module backs the React MI landing page. It is deliberately **deterministic**
and never touches the natural-language parser:

  * :func:`discover_snapshots` walks the local onboarding output root and reports
    the available portfolios and reporting runs (``mi_2025_10`` / ``mi_2025_11``),
    each with its funded loan count and current outstanding balance, so the UI's
    portfolio / reporting-date dropdowns are data-driven (only real runs appear).

  * :func:`compute_funded_snapshot` derives the landing-page KPI tiles (current
    funded balance, loans funded, weighted-average LTV / rate / age / months on
    book, average loan balance) and the month-on-month change versus the prior
    available run (loan-count / balance change, new / exited loans) straight from
    the prepared MI dataset and its dataset contract — not via ``run_mi_agent_query``.

The funded tape is period-scoped, so the snapshot inherently reflects the funded
universe (33 / 73 loans), never pipeline rows.
"""

from __future__ import annotations

import calendar
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analytics_lib.numeric import coerce_numeric

from .funded_prep import prepare_funded_mi_dataset
from .mi_dataset_contract import build_dataset_contract

_CENTRAL_TAPE_NAME = "18_central_lender_tape.csv"

# A reporting-run directory name carries a YEAR_MONTH (mi_2025_10, 2025-11, …).
_RUN_RE = re.compile(r"(\d{4})[_\-.](\d{2})")
# Path components that are pipeline scaffolding, never a client identifier.
_NON_CLIENT_PARTS = {"output", "outputs", "runs", "onboarding", "central", "mi", ""}


# --------------------------------------------------------------------------- #
# Numeric helpers (deterministic, contract-aware)
# --------------------------------------------------------------------------- #
def _num(series: pd.Series) -> pd.Series:
    return coerce_numeric(series)


def _weighted_average(values: pd.Series, weights: pd.Series) -> Optional[float]:
    """Weight-by-balance average over rows where both value and weight are valid.

    Falls back to a simple mean when the weights sum to zero. Returns ``None``
    when there is no usable value at all.
    """
    v = _num(values)
    w = _num(weights)
    mask = v.notna() & w.notna()
    if not mask.any():
        return None
    vv, ww = v[mask], w[mask]
    total = float(ww.sum())
    if total <= 0:
        return float(vv.mean()) if not vv.empty else None
    return float((vv * ww).sum() / total)


def _simple_mean(values: pd.Series) -> Optional[float]:
    v = _num(values).dropna()
    return float(v.mean()) if not v.empty else None


def _balance_sum(df: pd.DataFrame, col: str = "current_outstanding_balance") -> float:
    if col in df.columns:
        return float(_num(df[col]).sum())
    return 0.0


# --------------------------------------------------------------------------- #
# Reporting date + run/client inference
# --------------------------------------------------------------------------- #
def _last_day_of_month(year: int, month: int) -> str:
    last = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-{last:02d}"


def infer_reporting_date(run_id: str, df: Optional[pd.DataFrame] = None) -> Optional[str]:
    """The reporting (cut-off) date for a run: prefer the dataset's own column,
    otherwise parse ``mi_YYYY_MM`` style run ids to the month-end."""
    if df is not None:
        for col in ("reporting_date", "data_cut_off_date", "cut_off_date"):
            if col in df.columns:
                rd = pd.to_datetime(df[col], errors="coerce").dropna()
                if not rd.empty:
                    return rd.max().date().isoformat()
    m = _RUN_RE.search(run_id or "")
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return _last_day_of_month(year, month)
    return None


def _infer_client_and_run(tape: Path, root: Path) -> Tuple[Optional[str], Optional[str]]:
    """Infer ``(client_id, run_id)`` from a central-tape path under ``root``.

    Layout is ``.../<client_id>/<run_id>/[output/]central/18_central_lender_tape.csv``
    (with tolerated variants). The run dir is the nearest ancestor whose name
    carries a YEAR_MONTH; the client id is the nearest meaningful ancestor above it.
    """
    try:
        rel_parts = tape.relative_to(root).parts
    except ValueError:
        rel_parts = tape.parts
    parts = list(rel_parts[:-1])  # drop the filename
    run_idx = None
    for i in range(len(parts) - 1, -1, -1):
        if _RUN_RE.search(parts[i]):
            run_idx = i
            break
    if run_idx is None:
        return None, None
    run_id = parts[run_idx]
    client_id = None
    for j in range(run_idx - 1, -1, -1):
        if parts[j].lower() not in _NON_CLIENT_PARTS:
            client_id = parts[j]
            break
    return client_id, run_id


def resolve_tape_path(output_root: str | os.PathLike, client_id: str, run_id: str) -> Optional[Path]:
    """Find the promoted central lender tape for a specific client / run."""
    root = Path(output_root)
    candidates = [
        root / client_id / run_id / "output" / "central" / _CENTRAL_TAPE_NAME,
        root / client_id / run_id / "central" / _CENTRAL_TAPE_NAME,
        root / "runs" / client_id / "onboarding" / run_id / "central" / _CENTRAL_TAPE_NAME,
        root / run_id / "output" / "central" / _CENTRAL_TAPE_NAME,
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = sorted(root.glob(f"**/{run_id}/**/{_CENTRAL_TAPE_NAME}"))
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _portfolio_label(client_id: str) -> str:
    return str(client_id).upper()


def discover_snapshots(output_root: str | os.PathLike) -> Dict[str, Any]:
    """Discover available portfolios and reporting runs under ``output_root``.

    Returns ``{"portfolios": [{client_id, label, runs: [{run_id, reporting_date,
    loan_count, current_outstanding_balance}]}]}``. Runs are ordered oldest →
    newest by reporting date so the UI can default to the latest and resolve the
    prior run for month-on-month change. Unreadable tapes are skipped, never fatal.
    """
    root = Path(output_root)
    portfolios: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        return {"portfolios": []}

    for tape in sorted(root.glob(f"**/{_CENTRAL_TAPE_NAME}")):
        client_id, run_id = _infer_client_and_run(tape, root)
        if not client_id or not run_id:
            continue
        try:
            df = pd.read_csv(tape, low_memory=False)
        except Exception:  # noqa: BLE001 - a bad tape must not break discovery
            continue
        run = {
            "run_id": run_id,
            "reporting_date": infer_reporting_date(run_id, df),
            "loan_count": int(len(df)),
            "current_outstanding_balance": round(_balance_sum(df), 2),
        }
        pf = portfolios.setdefault(
            client_id, {"client_id": client_id, "label": _portfolio_label(client_id), "runs": {}}
        )
        pf["runs"][run_id] = run

    out: List[Dict[str, Any]] = []
    for pf in portfolios.values():
        runs = sorted(
            pf["runs"].values(),
            key=lambda r: (r["reporting_date"] or "", r["run_id"]),
        )
        out.append({"client_id": pf["client_id"], "label": pf["label"], "runs": runs})
    out.sort(key=lambda p: p["client_id"])
    return {"portfolios": out}


def find_prior_run(snapshots: Dict[str, Any], client_id: str, run_id: str) -> Optional[Dict[str, Any]]:
    """The previous available run for ``client_id`` before ``run_id`` (by date)."""
    for pf in snapshots.get("portfolios", []):
        if pf.get("client_id") != client_id:
            continue
        runs = pf.get("runs", [])
        idx = next((i for i, r in enumerate(runs) if r["run_id"] == run_id), None)
        if idx is None or idx == 0:
            return None
        return runs[idx - 1]
    return None


# --------------------------------------------------------------------------- #
# Snapshot KPI computation
# --------------------------------------------------------------------------- #
def _hint_scale(contract: Dict[str, Any], field: str) -> Optional[str]:
    return (contract.get("display_hints", {}) or {}).get(field, {}).get("scale")


def _to_points(value: Optional[float], scale: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    return value * 100.0 if scale == "percent_fraction" else value


def _fmt_gbp(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "—"
    sign = "+" if (signed and value >= 0) else ("-" if signed and value < 0 else "")
    v = abs(value) if signed else value
    if abs(v) >= 1e9:
        body = f"£{v / 1e9:.2f}BN"
    elif abs(v) >= 1e6:
        body = f"£{v / 1e6:.1f}MM"
    elif abs(v) >= 1e3:
        body = f"£{v / 1e3:.0f}K"
    else:
        body = f"£{v:,.0f}"
    return f"{sign}{body}"


def _fmt_pct_points(points: Optional[float], *, signed: bool = False) -> str:
    if points is None:
        return "—"
    sign = "+" if (signed and points >= 0) else ""
    return f"{sign}{points:.1f}%"


def _fmt_int(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "—"
    iv = int(round(value))
    return f"{iv:+d}" if signed else f"{iv:,d}"


def _fmt_decimal(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.1f}"


def _kpi(kpi_id: str, label: str, value: str, *, fmt: str, raw: Optional[float],
         available: bool = True, delta: Optional[str] = None,
         delta_intent: Optional[str] = None, hint: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": kpi_id,
        "label": label,
        "value": value,
        "format": fmt,
        "raw": raw,
        "available": available,
        "delta": delta,
        "deltaIntent": delta_intent,
        "hint": hint,
    }


def _has_values(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and _num(df[col]).notna().any()


def portfolio_risk_type(df: pd.DataFrame) -> str:
    """``"erm"`` (equity release / lifetime mortgage) vs ``"standard"`` amortising.

    ERM is detected from product/plan text; this deployment defaults to ERM."""
    for col in ("erm_product_type", "product", "product_type", "loan_plan"):
        if col in df.columns:
            v = df[col].astype(str).str.lower()
            if v.str.contains(
                    r"lifetime|equity[ -]?release|\berm\b|drawdown|lump sum|roll[ -]?up",
                    regex=True, na=False).any():
                return "erm"
    return "erm"


def _risk_tile(df: pd.DataFrame) -> Dict[str, Any]:
    """Portfolio-type-aware risk tile (replaces the duplicate loan-movement tile).

    ERM -> a current-NNEG exposure proxy (balance above current valuation); a
    standard book -> arrears balance. When the inputs are missing, a controlled
    "unavailable" tile lists exactly which fields are absent (never fabricated)."""
    if portfolio_risk_type(df) == "erm":
        bal_col, val_col = "current_outstanding_balance", "current_valuation_amount"
        missing = [c for c in (bal_col, val_col) if not _has_values(df, c)]
        if missing:
            return _kpi("nneg_risk", "NNEG exposure (current)", "Unavailable",
                        fmt="text", raw=None, available=False, delta_intent="neutral",
                        hint="missing inputs: " + ", ".join(missing))
        bal, val = _num(df[bal_col]), _num(df[val_col])
        mask = (bal > val) & bal.notna() & val.notna()
        nneg = float((bal[mask] - val[mask]).sum())
        cnt = int(mask.sum())
        return _kpi("nneg_risk", "NNEG exposure (current)", _fmt_gbp(nneg), fmt="gbp",
                    raw=round(nneg, 2),
                    delta_intent="negative" if nneg > 0 else "positive",
                    hint=f"{cnt} loan(s) with balance above current valuation")
    # standard amortising book -> arrears
    if not _has_values(df, "arrears_balance"):
        return _kpi("arrears_risk", "Arrears balance", "Unavailable", fmt="text",
                    raw=None, available=False, delta_intent="neutral",
                    hint="missing inputs: arrears_balance / days_in_arrears")
    tot = float(_num(df["arrears_balance"]).sum())
    return _kpi("arrears_risk", "Arrears balance", _fmt_gbp(tot), fmt="gbp",
                raw=round(tot, 2), delta_intent="negative" if tot > 0 else "neutral")


def _loan_ids(df: pd.DataFrame) -> set:
    if "loan_identifier" not in df.columns:
        return set()
    return set(df["loan_identifier"].astype(str).str.strip())


def compute_funded_snapshot(
    df: pd.DataFrame,
    semantics: dict,
    *,
    client_id: str,
    run_id: str,
    reporting_date: Optional[str] = None,
    prep_report: Optional[Dict[str, Any]] = None,
    prior_df: Optional[pd.DataFrame] = None,
    prior_run_id: Optional[str] = None,
    prior_reporting_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministic funded-book snapshot for one reporting run.

    Returns KPI tiles, the month-on-month change versus ``prior_df`` (if any),
    and any business-facing warnings (missing data / partial result). All numbers
    are derived from the prepared dataset and the dataset contract — never the parser.
    """
    contract = build_dataset_contract(df, semantics, prep_report)
    warnings: List[str] = []
    diagnostics: List[str] = []

    balance = _balance_sum(df)
    loan_count = int(len(df))
    bal_series = df.get("current_outstanding_balance", pd.Series(dtype=float))

    kpis: List[Dict[str, Any]] = []
    kpis.append(_kpi("balance", "Current funded balance", _fmt_gbp(balance),
                     fmt="gbp", raw=round(balance, 2)))
    kpis.append(_kpi("loans", "Loans funded", _fmt_int(loan_count), fmt="number",
                     raw=loan_count))

    # Weighted-average current LTV (weight by balance), contract-aware scaling.
    if _has_values(df, "current_loan_to_value"):
        wavg = _weighted_average(df["current_loan_to_value"], bal_series)
        pts = _to_points(wavg, _hint_scale(contract, "current_loan_to_value"))
        kpis.append(_kpi("wa_current_ltv", "Weighted avg current LTV",
                         _fmt_pct_points(pts), fmt="pct", raw=pts))
    else:
        kpis.append(_kpi("wa_current_ltv", "Weighted avg current LTV", "—",
                         fmt="pct", raw=None, available=False,
                         hint="LTV inputs unavailable for this run"))
        warnings.append("Weighted average current LTV unavailable: LTV inputs missing.")

    # Weighted-average original LTV (optional).
    if _has_values(df, "original_loan_to_value"):
        wavg = _weighted_average(df["original_loan_to_value"], bal_series)
        pts = _to_points(wavg, _hint_scale(contract, "original_loan_to_value"))
        kpis.append(_kpi("wa_original_ltv", "Weighted avg original LTV",
                         _fmt_pct_points(pts), fmt="pct", raw=pts))

    # Average loan balance.
    avg_balance = balance / loan_count if loan_count else None
    kpis.append(_kpi("avg_balance", "Average loan balance", _fmt_gbp(avg_balance),
                     fmt="gbp", raw=round(avg_balance, 2) if avg_balance is not None else None))

    # Weighted-average current interest rate (optional).
    if _has_values(df, "current_interest_rate"):
        wavg = _weighted_average(df["current_interest_rate"], bal_series)
        pts = _to_points(wavg, _hint_scale(contract, "current_interest_rate"))
        kpis.append(_kpi("wa_rate", "Weighted avg interest rate",
                         _fmt_pct_points(pts), fmt="pct", raw=pts))

    # Weighted-average months on book (optional).
    if _has_values(df, "months_on_book"):
        wavg = _weighted_average(df["months_on_book"], bal_series)
        kpis.append(_kpi("wa_months_on_book", "Weighted avg months on book",
                         _fmt_decimal(wavg), fmt="number", raw=wavg))

    # Weighted-average youngest borrower age (optional).
    if _has_values(df, "youngest_borrower_age"):
        wavg = _weighted_average(df["youngest_borrower_age"], bal_series)
        kpis.append(_kpi("wa_age", "Weighted avg youngest age",
                         _fmt_decimal(wavg), fmt="number", raw=wavg))

    # ---- month-on-month change vs the prior available run -------------------
    monthly_change: Optional[Dict[str, Any]] = None
    if prior_df is not None:
        prior_balance = _balance_sum(prior_df)
        prior_count = int(len(prior_df))
        loan_delta = loan_count - prior_count
        bal_delta = balance - prior_balance
        bal_delta_pct = (bal_delta / prior_balance * 100.0) if prior_balance else None

        cur_ids, prior_ids = _loan_ids(df), _loan_ids(prior_df)
        ids_identifiable = bool(cur_ids) and bool(prior_ids)
        new_loans = len(cur_ids - prior_ids) if ids_identifiable else None
        exited_loans = len(prior_ids - cur_ids) if ids_identifiable else None

        monthly_change = {
            "prior_run_id": prior_run_id,
            "prior_reporting_date": prior_reporting_date,
            "loan_count_change": loan_delta,
            "balance_change": round(bal_delta, 2),
            "balance_change_pct": round(bal_delta_pct, 2) if bal_delta_pct is not None else None,
            "new_loans": new_loans,
            "exited_loans": exited_loans,
            "loans_identifiable": ids_identifiable,
        }

        # Attach deltas to the headline tiles.
        kpis[0]["delta"] = _fmt_gbp(bal_delta, signed=True)
        kpis[0]["deltaIntent"] = "positive" if bal_delta >= 0 else "negative"
        kpis[0]["hint"] = (f"{_fmt_pct_points(bal_delta_pct, signed=True)} vs prior run"
                           if bal_delta_pct is not None else "vs prior run")
        kpis[1]["delta"] = _fmt_int(loan_delta, signed=True)
        kpis[1]["deltaIntent"] = "positive" if loan_delta >= 0 else "negative"
        kpis[1]["hint"] = f"vs {prior_reporting_date or prior_run_id}"

        kpis.append(_kpi("mom_loans", "Monthly change · loans", _fmt_int(loan_delta, signed=True),
                         fmt="number", raw=loan_delta,
                         delta_intent="positive" if loan_delta >= 0 else "negative",
                         hint=f"vs {prior_reporting_date or prior_run_id}"))
        kpis.append(_kpi("mom_balance", "Monthly change · balance", _fmt_gbp(bal_delta, signed=True),
                         fmt="gbp", raw=round(bal_delta, 2),
                         delta_intent="positive" if bal_delta >= 0 else "negative",
                         hint=_fmt_pct_points(bal_delta_pct, signed=True) if bal_delta_pct is not None else None))
        if ids_identifiable:
            # The net "Monthly change · loans" tile already conveys loan movement;
            # the old "New loans since prior run" duplicated it. Keep exited/redeemed
            # (genuinely distinct) and surface new-loans in the monthly_change block.
            kpis.append(_kpi("exited_loans", "Exited / redeemed loans", _fmt_int(exited_loans),
                             fmt="number", raw=exited_loans,
                             delta_intent="negative" if (exited_loans or 0) > 0 else "neutral"))
        else:
            diagnostics.append("Loan-level new/exited counts not identifiable "
                               "(no loan_identifier on one of the runs).")
    else:
        diagnostics.append("No prior reporting date available for this portfolio.")

    # Portfolio-type-aware risk tile (replaces the duplicate loan-movement tile):
    # ERM -> NNEG exposure; standard -> arrears; controlled "unavailable" otherwise.
    kpis.append(_risk_tile(df))

    # Surface genuinely-missing core dimensions as business warnings (not noise).
    for miss in (prep_report or {}).get("missing_dimensions", []) or []:
        if isinstance(miss, dict) and miss.get("reason") in (
            "no_values_after_preparation", "derivation_inputs_missing"
        ):
            diagnostics.append(f"{miss['dimension']}: {miss.get('detail', miss['reason'])}")

    return {
        "ok": True,
        "portfolio": {
            "client_id": client_id,
            "label": _portfolio_label(client_id),
            "run_id": run_id,
            "reporting_date": reporting_date,
        },
        "prior": (
            {"run_id": prior_run_id, "reporting_date": prior_reporting_date}
            if prior_df is not None else None
        ),
        "loan_count": loan_count,
        "current_outstanding_balance": round(balance, 2),
        "kpis": kpis,
        "monthly_change": monthly_change,
        "warnings": warnings,
        "diagnostics": diagnostics,
        "datasetContract": contract,
    }


def load_prepared_run(tape_path: str | os.PathLike) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read a central lender tape and apply the funded MI preparation layer."""
    raw = pd.read_csv(Path(tape_path), low_memory=False)
    return prepare_funded_mi_dataset(raw)
