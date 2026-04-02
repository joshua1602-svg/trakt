#!/usr/bin/env python3
"""
loan_ledger_engine.py

Optional v1 asset-side loan ledger capability.

Inputs:
  - canonical static terms CSV (from messy_to_canonical)
  - payments CSV

Outputs:
  - ledger_transactions.csv
  - canonical_snapshot.csv (and --output path)

This module is intentionally narrow and deterministic for bullet-loan products.
"""

from __future__ import annotations

import argparse
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

DATE_FMT = "%Y-%m-%d"
ALLOWED_PAYMENT_TYPES = {"INTEREST", "PRINCIPAL", "PENALTY", "MIXED"}


@dataclass
class LoanEngineConfig:
    loan_engine_enabled: bool = False
    currency: str = "GBP"
    day_count_convention: str = "ACT365"
    interest_payment_frequency: str = "QUARTERLY"
    prepayment_lockout_period_quarters: int = 2
    legal_maturity_months: int = 24
    penalty_interest_rate: float = 0.15
    reporting_date: Optional[pd.Timestamp] = None
    ledger_db: str = "out/loan_ledger.db"
    payments_file: Optional[str] = None


# ---------------------------------------------------------------------------
# IO + validation
# ---------------------------------------------------------------------------

def _read_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_engine_config(config_path: Optional[str], reporting_date_override: Optional[str]) -> LoanEngineConfig:
    root_cfg = _read_yaml(config_path) if config_path else {}
    pipeline_cfg = root_cfg.get("pipeline") if isinstance(root_cfg.get("pipeline"), dict) else {}
    loan_cfg = root_cfg.get("loan_engine") if isinstance(root_cfg.get("loan_engine"), dict) else {}

    def _pick(name: str, default: Any) -> Any:
        if name in loan_cfg:
            return loan_cfg[name]
        return pipeline_cfg.get(name, default)

    reporting_date_raw = reporting_date_override or _pick("reporting_date", None)
    reporting_date = pd.to_datetime(reporting_date_raw).normalize() if reporting_date_raw else None

    return LoanEngineConfig(
        loan_engine_enabled=bool(_pick("loan_engine_enabled", False)),
        currency=str(_pick("currency", "GBP")),
        day_count_convention=str(_pick("day_count_convention", "ACT365")),
        interest_payment_frequency=str(_pick("interest_payment_frequency", "QUARTERLY")),
        prepayment_lockout_period_quarters=int(_pick("prepayment_lockout_period_quarters", 2)),
        legal_maturity_months=int(_pick("legal_maturity_months", 24)),
        penalty_interest_rate=float(_pick("penalty_interest_rate", 0.15)),
        reporting_date=reporting_date,
        ledger_db=str(_pick("ledger_db", "out/loan_ledger.db")),
        payments_file=_pick("payments_file", None),
    )


def load_terms(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_payments(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["loan_identifier", "payment_date", "payment_amount", "payment_type"])
    return pd.read_csv(p)


def validate_inputs(terms: pd.DataFrame, payments: pd.DataFrame, cfg: LoanEngineConfig) -> None:
    required_terms = {
        "loan_identifier",
        "origination_date",
        "original_principal_balance",
        "current_principal_balance",
        "current_interest_rate",
        "scheduled_interest_payment_frequency",
        "current_valuation_amount",
        "current_valuation_date",
        "account_status",
    }
    missing_terms = sorted(required_terms.difference(set(terms.columns)))
    if missing_terms:
        raise ValueError(f"Terms file missing required columns: {missing_terms}")

    required_payments = {"loan_identifier", "payment_date", "payment_amount", "payment_type"}
    missing_payments = sorted(required_payments.difference(set(payments.columns)))
    if missing_payments:
        raise ValueError(f"Payments file missing required columns: {missing_payments}")

    if cfg.day_count_convention.upper() != "ACT365":
        raise ValueError("Only ACT365 is supported in v1")
    if cfg.interest_payment_frequency.upper() != "QUARTERLY":
        raise ValueError("Only QUARTERLY is supported in v1")

    if not payments.empty:
        bad_types = sorted(set(payments["payment_type"].astype(str).str.upper()) - ALLOWED_PAYMENT_TYPES)
        if bad_types:
            raise ValueError(f"Unsupported payment_type values: {bad_types}")


# ---------------------------------------------------------------------------
# Core loan math helpers
# ---------------------------------------------------------------------------

def _rate_to_decimal(raw: Any) -> float:
    if pd.isna(raw):
        return 0.0
    r = float(raw)
    return r / 100.0 if r > 1 else r


def _to_ts(v: Any) -> Optional[pd.Timestamp]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return pd.to_datetime(v).normalize()
    except Exception:
        return None


def _add_months(d: pd.Timestamp, months: int) -> pd.Timestamp:
    return d + pd.DateOffset(months=months)


def build_schedule(origination: pd.Timestamp, maturity: pd.Timestamp, first_interest_payment_date: Optional[pd.Timestamp]) -> List[pd.Timestamp]:
    first = first_interest_payment_date or _add_months(origination, 3)
    dates: List[pd.Timestamp] = []
    d = first
    while d <= maturity:
        dates.append(d)
        d = _add_months(d, 3)
    return dates


def accrue_interest(principal: float, annual_rate: float, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[float, int]:
    days = max((end - start).days, 0)
    return principal * annual_rate * days / 365.0, days


def accrue_penalty(unpaid_interest_balance: float, penalty_rate: float, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[float, int]:
    days = max((end - start).days, 0)
    return unpaid_interest_balance * penalty_rate * days / 365.0, days


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _event(event_type: str, loan_id: str, event_date: pd.Timestamp, amount: float, state: Dict[str, Any], source: str,
           period_start: Optional[pd.Timestamp] = None, period_end: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "loan_identifier": loan_id,
        "event_type": event_type,
        "event_date": event_date.strftime(DATE_FMT),
        "period_start": period_start.strftime(DATE_FMT) if period_start is not None else None,
        "period_end": period_end.strftime(DATE_FMT) if period_end is not None else None,
        "amount": round(float(amount), 6),
        "principal_balance_post": round(float(state["principal_balance"]), 6),
        "accrued_interest_post": round(float(state["accrued_interest_in_period"]), 6),
        "unpaid_interest_balance_post": round(float(state["unpaid_interest_balance"]), 6),
        "penalty_balance_post": round(float(state["penalty_balance"]), 6),
        "current_ltv_post": round(float(state["current_ltv"]), 8) if state["current_ltv"] is not None else None,
        "account_status_post": state["account_status"],
        "source": source,
        "created_at": _now_iso(),
    }


def update_status(
    principal_balance: float,
    original_principal: float,
    unpaid_interest_balance: float,
    scheduled_coupon: float,
    consecutive_missed: int,
    maturity_date: pd.Timestamp,
    reporting_date: pd.Timestamp,
) -> str:
    if principal_balance <= 0:
        return "REDEEMED"
    if reporting_date > maturity_date and principal_balance > 0:
        return "MATURED_UNPAID"
    if consecutive_missed >= 2:
        return "DEFAULT"
    if unpaid_interest_balance >= max(scheduled_coupon, 0.01):
        return "IN_ARREARS"
    if unpaid_interest_balance > 0:
        return "WATCH_LIST"
    if 0 < principal_balance < original_principal:
        return "PARTIALLY_REDEEMED"
    return "PERFORMING"


def match_payment(payment_type: str, amount: float, state: Dict[str, Any], in_lockout: bool) -> Tuple[List[Tuple[str, float]], Optional[str]]:
    """
    Return ordered actions as list[(bucket, amount)] where bucket in
    interest|penalty|principal|further_advance and optional warning message.
    """
    actions: List[Tuple[str, float]] = []
    warning = None
    pt = payment_type.upper()

    if pt == "INTEREST":
        actions.append(("interest", amount))
    elif pt == "PENALTY":
        actions.append(("penalty", amount))
    elif pt == "PRINCIPAL":
        if amount < 0:
            actions.append(("further_advance", abs(amount)))
        elif in_lockout:
            warning = "PREPAYMENT_LOCKOUT_BREACH"
        else:
            actions.append(("principal", amount))
    elif pt == "MIXED":
        rem = amount
        i_amt = min(rem, max(state["unpaid_interest_balance"], 0.0))
        if i_amt > 0:
            actions.append(("interest", i_amt))
            rem -= i_amt
        p_amt = min(rem, max(state["penalty_balance"], 0.0))
        if p_amt > 0:
            actions.append(("penalty", p_amt))
            rem -= p_amt
        if rem > 0:
            if in_lockout:
                warning = "PREPAYMENT_LOCKOUT_BREACH"
            else:
                actions.append(("principal", rem))
    else:
        warning = f"UNSUPPORTED_PAYMENT_TYPE:{payment_type}"

    return actions, warning


def process_events(terms: pd.DataFrame, payments: pd.DataFrame, cfg: LoanEngineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    events: List[Dict[str, Any]] = []
    snapshots: List[Dict[str, Any]] = []

    payments = payments.copy()
    if not payments.empty:
        payments["payment_date"] = pd.to_datetime(payments["payment_date"]).dt.normalize()
        payments["payment_type"] = payments["payment_type"].astype(str).str.upper()
        payments["payment_amount"] = payments["payment_amount"].astype(float)

    optional_cols = [
        "borrower_legal_name",
        "borrower_jurisdiction",
        "first_interest_payment_date",
        "penalty_interest_rate",
        "collateral_geography",
        "guarantee_type",
    ]

    for _, row in terms.iterrows():
        loan_id = str(row["loan_identifier"])
        origination = _to_ts(row.get("origination_date"))
        if origination is None:
            raise ValueError(f"loan_identifier={loan_id}: origination_date is required and must be parseable")

        explicit_maturity = _to_ts(row.get("maturity_date"))
        maturity = explicit_maturity or _add_months(origination, cfg.legal_maturity_months)
        reporting_date = cfg.reporting_date or max(maturity, pd.Timestamp.utcnow().normalize())
        reporting_date = min(reporting_date, maturity) if maturity <= reporting_date else reporting_date

        original_principal = float(row.get("original_principal_balance", 0.0) or 0.0)
        principal_balance = float(row.get("current_principal_balance", original_principal) or 0.0)
        annual_rate = _rate_to_decimal(row.get("current_interest_rate", 0.0))
        penalty_rate = _rate_to_decimal(row.get("penalty_interest_rate", cfg.penalty_interest_rate))
        valuation = float(row.get("current_valuation_amount", 0.0) or 0.0)
        valuation_date = _to_ts(row.get("current_valuation_date")) or reporting_date
        first_interest_date = _to_ts(row.get("first_interest_payment_date"))
        schedule = build_schedule(origination, maturity, first_interest_date)

        loan_payments = payments[payments["loan_identifier"].astype(str) == loan_id].sort_values("payment_date")

        state = {
            "principal_balance": principal_balance,
            "accrued_interest_in_period": 0.0,
            "cumulative_accrued_interest": 0.0,
            "unpaid_interest_balance": 0.0,
            "penalty_balance": 0.0,
            "interest_collections_in_period": 0.0,
            "redemptions_received_in_period": 0.0,
            "further_advance_amount": 0.0,
            "further_advance_date": None,
            "account_status": str(row.get("account_status", "PERFORMING")),
            "current_ltv": (principal_balance / valuation) if valuation > 0 else None,
        }

        events.append(_event("DRAWDOWN", loan_id, origination, original_principal, state, "terms"))

        if valuation > 0:
            events.append(_event("VALUATION_UPDATE", loan_id, valuation_date, valuation, state, "terms"))

        lockout_end = _add_months(origination, 3 * cfg.prepayment_lockout_period_quarters)
        prev_schedule_date = origination
        consecutive_missed = 0
        last_coupon = 0.0

        for schedule_date in schedule:
            if schedule_date > reporting_date:
                break

            prior_unpaid = state["unpaid_interest_balance"]
            interest_amt, _ = accrue_interest(state["principal_balance"], annual_rate, prev_schedule_date, schedule_date)
            penalty_amt, _ = accrue_penalty(prior_unpaid, penalty_rate, prev_schedule_date, schedule_date)

            if interest_amt > 0:
                state["accrued_interest_in_period"] += interest_amt
                state["cumulative_accrued_interest"] += interest_amt
                state["unpaid_interest_balance"] += interest_amt
                last_coupon = interest_amt
                events.append(_event("INTEREST_ACCRUAL", loan_id, schedule_date, interest_amt, state, "engine", prev_schedule_date, schedule_date))

            if penalty_amt > 0:
                state["penalty_balance"] += penalty_amt
                events.append(_event("PENALTY_ACCRUAL", loan_id, schedule_date, penalty_amt, state, "engine", prev_schedule_date, schedule_date))

            period_payments = loan_payments[(loan_payments["payment_date"] > prev_schedule_date) & (loan_payments["payment_date"] <= schedule_date)]
            for _, pmt in period_payments.iterrows():
                ptype = str(pmt["payment_type"]).upper()
                pdate = pmt["payment_date"]
                amount = float(pmt["payment_amount"])
                in_lockout = pdate < lockout_end and ptype in {"PRINCIPAL", "MIXED"}
                actions, warning = match_payment(ptype, amount, state, in_lockout)

                for bucket, bucket_amount in actions:
                    if bucket == "interest":
                        applied = min(bucket_amount, max(state["unpaid_interest_balance"], 0.0))
                        if applied > 0:
                            state["unpaid_interest_balance"] -= applied
                            state["interest_collections_in_period"] += applied
                            events.append(_event("INTEREST_RECEIPT", loan_id, pdate, applied, state, "payments"))
                    elif bucket == "penalty":
                        applied = min(bucket_amount, max(state["penalty_balance"], 0.0))
                        if applied > 0:
                            state["penalty_balance"] -= applied
                            events.append(_event("PENALTY_RECEIPT", loan_id, pdate, applied, state, "payments"))
                    elif bucket == "principal":
                        applied = min(bucket_amount, max(state["principal_balance"], 0.0))
                        if applied > 0:
                            state["principal_balance"] -= applied
                            state["redemptions_received_in_period"] += applied
                            ev_type = "MATURITY_REDEMPTION" if pdate >= maturity else "PREPAYMENT"
                            events.append(_event(ev_type, loan_id, pdate, applied, state, "payments"))
                    elif bucket == "further_advance":
                        state["principal_balance"] += bucket_amount
                        state["further_advance_amount"] += bucket_amount
                        state["further_advance_date"] = pdate.strftime(DATE_FMT)
                        events.append(_event("FURTHER_ADVANCE", loan_id, pdate, bucket_amount, state, "payments"))

                if warning:
                    events.append(_event("STATUS_CHANGE", loan_id, pdate, 0.0, state, warning))

            shortfall = max(state["unpaid_interest_balance"], 0.0)
            if shortfall > 0:
                consecutive_missed += 1
                events.append(_event("INTEREST_SHORTFALL", loan_id, schedule_date, shortfall, state, "engine"))
            else:
                consecutive_missed = 0

            new_status = update_status(
                principal_balance=state["principal_balance"],
                original_principal=original_principal,
                unpaid_interest_balance=state["unpaid_interest_balance"],
                scheduled_coupon=last_coupon,
                consecutive_missed=consecutive_missed,
                maturity_date=maturity,
                reporting_date=reporting_date,
            )
            if new_status != state["account_status"]:
                state["account_status"] = new_status
                events.append(_event("STATUS_CHANGE", loan_id, schedule_date, 0.0, state, "engine"))

            state["current_ltv"] = (state["principal_balance"] / valuation) if valuation > 0 else None
            prev_schedule_date = schedule_date

        final_status = update_status(
            principal_balance=state["principal_balance"],
            original_principal=original_principal,
            unpaid_interest_balance=state["unpaid_interest_balance"],
            scheduled_coupon=last_coupon,
            consecutive_missed=consecutive_missed,
            maturity_date=maturity,
            reporting_date=reporting_date,
        )
        if final_status != state["account_status"]:
            state["account_status"] = final_status
            events.append(_event("STATUS_CHANGE", loan_id, reporting_date, 0.0, state, "engine"))

        snap = {
            "loan_identifier": loan_id,
            "origination_date": origination.strftime(DATE_FMT),
            "maturity_date": maturity.strftime(DATE_FMT),
            "original_principal_balance": round(original_principal, 6),
            "current_principal_balance": round(state["principal_balance"], 6),
            "current_interest_rate": round(annual_rate, 8),
            "scheduled_interest_payment_frequency": str(row.get("scheduled_interest_payment_frequency", cfg.interest_payment_frequency)),
            "current_valuation_amount": round(valuation, 6),
            "current_valuation_date": valuation_date.strftime(DATE_FMT),
            "current_loan_to_value": round(state["current_ltv"], 8) if state["current_ltv"] is not None else None,
            "accrued_interest_in_period": round(state["accrued_interest_in_period"], 6),
            "cumulative_accrued_interest": round(state["cumulative_accrued_interest"], 6),
            "interest_collections_in_period": round(state["interest_collections_in_period"], 6),
            "arrears_balance": round(state["unpaid_interest_balance"], 6),
            "redemptions_received_in_period": round(state["redemptions_received_in_period"], 6),
            "further_advance_amount": round(state["further_advance_amount"], 6),
            "further_advance_date": state["further_advance_date"],
            "account_status": state["account_status"],
            "data_cut_off_date": reporting_date.strftime(DATE_FMT),
            "penalty_interest_balance": round(state["penalty_balance"], 6),
            "penalty_interest_rate": round(penalty_rate, 8),
        }
        for c in optional_cols:
            if c in terms.columns and c not in snap:
                snap[c] = row.get(c)
            elif c in ("penalty_interest_rate",):
                continue
            elif c not in snap:
                snap[c] = row.get(c) if c in terms.columns else None

        snapshots.append(snap)

    ledger_df = pd.DataFrame(events)
    snapshot_df = pd.DataFrame(snapshots)
    return ledger_df, snapshot_df


# ---------------------------------------------------------------------------
# Persistence + exports
# ---------------------------------------------------------------------------

def persist_ledger(ledger_df: pd.DataFrame, sqlite_path: str) -> None:
    db_path = Path(sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "event_id", "loan_identifier", "event_type", "event_date", "period_start", "period_end", "amount",
        "principal_balance_post", "accrued_interest_post", "unpaid_interest_balance_post", "penalty_balance_post",
        "current_ltv_post", "account_status_post", "source", "created_at",
    ]
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_transactions (
                event_id TEXT PRIMARY KEY,
                loan_identifier TEXT,
                event_type TEXT,
                event_date TEXT,
                period_start TEXT,
                period_end TEXT,
                amount REAL,
                principal_balance_post REAL,
                accrued_interest_post REAL,
                unpaid_interest_balance_post REAL,
                penalty_balance_post REAL,
                current_ltv_post REAL,
                account_status_post TEXT,
                source TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute("DELETE FROM ledger_transactions")
        ledger_df[cols].to_sql("ledger_transactions", conn, if_exists="append", index=False)


def export_ledger_csv(ledger_df: pd.DataFrame, output_dir: Path) -> Path:
    out_path = output_dir / "ledger_transactions.csv"
    ledger_df.to_csv(out_path, index=False)
    return out_path


def write_canonical_snapshot(snapshot_df: pd.DataFrame, output_path: Path) -> Path:
    ordered_cols = [
        "loan_identifier",
        "origination_date",
        "maturity_date",
        "original_principal_balance",
        "current_principal_balance",
        "current_interest_rate",
        "scheduled_interest_payment_frequency",
        "current_valuation_amount",
        "current_valuation_date",
        "current_loan_to_value",
        "accrued_interest_in_period",
        "cumulative_accrued_interest",
        "interest_collections_in_period",
        "arrears_balance",
        "redemptions_received_in_period",
        "further_advance_amount",
        "further_advance_date",
        "account_status",
        "data_cut_off_date",
        "borrower_legal_name",
        "borrower_jurisdiction",
        "first_interest_payment_date",
        "penalty_interest_rate",
        "penalty_interest_balance",
        "collateral_geography",
        "guarantee_type",
    ]
    cols = [c for c in ordered_cols if c in snapshot_df.columns]
    snapshot_df[cols].to_csv(output_path, index=False)

    # Required canonical name for downstream/manual consumers.
    canonical_path = output_path.parent / "canonical_snapshot.csv"
    if canonical_path != output_path:
        snapshot_df[cols].to_csv(canonical_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Optional v1 loan ledger engine")
    ap.add_argument("--input", required=True, help="Canonical static terms CSV")
    ap.add_argument("--payments", default=None, help="Payments CSV (optional override)")
    ap.add_argument("--config", default=None, help="Master config path")
    ap.add_argument("--output", required=True, help="Output snapshot CSV path")
    ap.add_argument("--reporting-date", default=None, help="Override reporting date (YYYY-MM-DD)")
    args = ap.parse_args()

    cfg = load_engine_config(args.config, args.reporting_date)
    if not cfg.loan_engine_enabled:
        raise RuntimeError("loan_ledger_engine invoked while loan_engine_enabled is false")

    terms = load_terms(args.input)
    payments_path = args.payments or cfg.payments_file
    if not payments_path:
        raise ValueError("Payments file is required. Set --payments or loan_engine.payments_file in config")
    payments = load_payments(payments_path)
    validate_inputs(terms, payments, cfg)

    ledger_df, snapshot_df = process_events(terms, payments, cfg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    persist_ledger(ledger_df, cfg.ledger_db)
    ledger_csv = export_ledger_csv(ledger_df, out_path.parent)
    snap_csv = write_canonical_snapshot(snapshot_df, out_path)

    print(f"[Loan Engine] ledger -> {ledger_csv}")
    print(f"[Loan Engine] snapshot -> {snap_csv}")


if __name__ == "__main__":
    main()
