"""tests/helpers/phase6c_consolidation.py

Deterministic SYNTHETIC consolidation helper for the Phase 6C proof. It joins
fragmented source artefacts (borrowers / loans / collateral / cashflows /
portfolio map / pipeline) into one canonical MI snapshot frame per reporting
date, with lightweight lineage metadata and structured issues.

This is a proof fixture, NOT a production consolidation engine: the join rules
are hard-wired for the synthetic fixture, there is no schema inference, no
mapping config, and no onboarding orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from mi_agent.states.models import ERROR, INFO, WARNING, make_issue
from snapshot.model import SnapshotHeader

# Phase 6C issue codes.
MISSING_REQUIRED_ARTIFACT = "missing_required_artifact"
MISSING_OPTIONAL_ARTIFACT = "missing_optional_artifact"
MISSING_JOIN_KEY = "missing_join_key"
UNMATCHED_ARTIFACT_ROWS = "unmatched_artifact_rows"
DUPLICATE_JOIN_KEY = "duplicate_join_key"
MISSING_LINEAGE_FOR_FIELD = "missing_lineage_for_field"
MISSING_OPTIONAL_CONSOLIDATED_FIELD = "missing_optional_consolidated_field"
SNAPSHOT_REGISTRATION_FAILED = "snapshot_registration_failed"
MULTI_ARTIFACT_CONSOLIDATION_WARNING = "multi_artifact_consolidation_warning"

REQUIRED_ARTIFACTS = ("borrowers", "loans")
OPTIONAL_ARTIFACTS = ("collateral", "cashflows", "portfolio_map", "pipeline")

# Which artefact each key consolidated field is sourced from (lineage).
LINEAGE: Dict[str, str] = {
    "loan_id": "loans.csv (funded) / pipeline.csv opportunity_id (pipeline)",
    "borrower_id": "loans.csv / pipeline.csv",
    "funded_status": "loans.csv / pipeline.csv",
    "current_outstanding_balance": "loans.csv balance / pipeline.csv expected_balance",
    "current_interest_rate": "loans.csv / pipeline.csv",
    "erm_product_type": "loans.csv / pipeline.csv",
    "amortisation_type": "loans.csv",
    "origination_date": "loans.csv",
    "funding_date": "loans.csv",
    "internal_risk_grade": "loans.csv",
    "ifrs9_stage": "loans.csv",
    "pd_bucket": "loans.csv",
    "current_loan_to_value": "collateral.csv current_ltv",
    "collateral_geography": "collateral.csv property_region",
    "geographic_region_obligor": "borrowers.csv region",
    "youngest_borrower_age": "borrowers.csv borrower_age",
    "borrower_structure": "borrowers.csv",
    "arrears_status": "cashflows.csv",
    "arrears_balance": "cashflows.csv",
    "portfolio_id": "portfolio_map.csv",
    "portfolio_name": "portfolio_map.csv",
    "spv_id": "portfolio_map.csv",
    "acquired_portfolio_id": "portfolio_map.csv",
    "pipeline_stage": "pipeline.csv",
    "forecast_funding_probability": "pipeline.csv",
    "forecast_funded_balance": "derived: pipeline expected_balance x forecast_funding_probability",
    "broker_channel": "pipeline.csv",
    "origination_channel": "pipeline.csv",
    "months_on_book": "derived: funding_date vs reporting_date",
}

# Consolidated fields callers expect to be present (for optional-field checks).
EXPECTED_CONSOLIDATED_FIELDS = tuple(LINEAGE.keys()) + (
    "opportunity_id", "source_record_id", "stable_entity_id")


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #


def load_artifacts(directory: Path | str
                   ) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
    directory = Path(directory)
    arts: Dict[str, pd.DataFrame] = {}
    issues: List[dict] = []
    for name in REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS:
        path = directory / f"{name}.csv"
        if path.exists():
            arts[name] = pd.read_csv(path, dtype={"reporting_date": str})
        elif name in REQUIRED_ARTIFACTS:
            issues.append(make_issue(MISSING_REQUIRED_ARTIFACT, ERROR,
                                     f"required artefact {name}.csv not found",
                                     field=name))
        else:
            issues.append(make_issue(MISSING_OPTIONAL_ARTIFACT, WARNING,
                                     f"optional artefact {name}.csv not found",
                                     field=name))
    return arts, issues


# --------------------------------------------------------------------------- #
# Consolidate
# --------------------------------------------------------------------------- #


def _months_between(start: pd.Series, as_of: str) -> pd.Series:
    s = pd.to_datetime(start, errors="coerce")
    e = pd.to_datetime(as_of, errors="coerce")
    months = (e.year - s.dt.year) * 12 + (e.month - s.dt.month)
    return months.clip(lower=0).astype("Int64")


def _check_duplicate_keys(df: pd.DataFrame, key: str, name: str,
                          issues: List[dict]) -> None:
    if key in df.columns and df[key].duplicated().any():
        dups = df[key][df[key].duplicated()].unique().tolist()
        issues.append(make_issue(DUPLICATE_JOIN_KEY, WARNING,
                                 f"{name}: duplicate join key {key}: {dups}",
                                 field=key, count=len(dups)))


def _check_unmatched(child: pd.DataFrame, key: str, parent_keys: set,
                     name: str, issues: List[dict]) -> None:
    if key not in child.columns:
        return
    orphans = sorted(set(child[key].dropna()) - parent_keys)
    if orphans:
        issues.append(make_issue(
            UNMATCHED_ARTIFACT_ROWS, WARNING,
            f"{name}: {len(orphans)} row(s) reference unknown {key}: {orphans}",
            field=key, count=len(orphans)))


def consolidate(artifacts: Dict[str, pd.DataFrame], *, client_id: str = "smoke"
                ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], List[dict]]:
    """Consolidate artefacts into one canonical MI frame per reporting date."""
    issues: List[dict] = []
    loans = artifacts.get("loans")
    borrowers = artifacts.get("borrowers")
    if loans is None or borrowers is None:
        issues.append(make_issue(MISSING_REQUIRED_ARTIFACT, ERROR,
                                 "loans and borrowers are required"))
        return {}, LINEAGE, issues

    collateral = artifacts.get("collateral", pd.DataFrame())
    cashflows = artifacts.get("cashflows", pd.DataFrame())
    portfolio_map = artifacts.get("portfolio_map", pd.DataFrame())
    pipeline = artifacts.get("pipeline", pd.DataFrame())

    # Required join keys present?
    for art, key, nm in [(loans, "borrower_id", "loans"),
                         (loans, "loan_id", "loans"),
                         (borrowers, "borrower_id", "borrowers")]:
        if key not in art.columns:
            issues.append(make_issue(MISSING_JOIN_KEY, ERROR,
                                     f"{nm} missing join key {key}", field=key))
    if any(i["severity"] == ERROR for i in issues):
        return {}, LINEAGE, issues

    # Integrity checks (lineage / dup / unmatched).
    loan_ids = set(loans["loan_id"])
    borrower_ids = set(borrowers["borrower_id"])
    _check_duplicate_keys(borrowers, "borrower_id", "borrowers", issues)
    _check_duplicate_keys(collateral, "loan_id", "collateral", issues)
    _check_duplicate_keys(portfolio_map, "loan_id", "portfolio_map", issues)
    _check_unmatched(collateral, "loan_id", loan_ids, "collateral", issues)
    _check_unmatched(portfolio_map, "loan_id", loan_ids, "portfolio_map", issues)
    _check_unmatched(cashflows, "loan_id", loan_ids, "cashflows", issues)
    if not pipeline.empty:
        _check_unmatched(pipeline, "borrower_id", borrower_ids, "pipeline", issues)

    bor = borrowers.set_index("borrower_id")
    col = collateral.set_index("loan_id") if not collateral.empty else pd.DataFrame()
    pmap = portfolio_map.set_index("loan_id") if not portfolio_map.empty else pd.DataFrame()

    def _b(bid, field):
        return bor.at[bid, field] if (bid in bor.index and field in bor.columns) else None

    def _c(lid, field):
        return col.at[lid, field] if (not col.empty and lid in col.index
                                      and field in col.columns) else None

    def _p(lid, field):
        return pmap.at[lid, field] if (not pmap.empty and lid in pmap.index
                                       and field in pmap.columns) else None

    rdates = sorted(set(loans["reporting_date"]))
    frames: Dict[str, pd.DataFrame] = {}
    for rd in rdates:
        rows: List[dict] = []
        # Funded loans for this reporting date.
        for _, ln in loans[loans["reporting_date"] == rd].iterrows():
            lid, bid = ln["loan_id"], ln["borrower_id"]
            cf = cashflows[(cashflows.get("loan_id") == lid)
                           & (cashflows.get("reporting_date") == rd)] \
                if not cashflows.empty else pd.DataFrame()
            cf_row = cf.iloc[0] if not cf.empty else {}
            rows.append({
                "loan_id": lid, "opportunity_id": None, "source_record_id": lid,
                "stable_entity_id": lid, "borrower_id": bid,
                "funded_status": ln["funded_status"], "pipeline_stage": None,
                "current_outstanding_balance": float(ln["balance"]),
                "current_interest_rate": float(ln["interest_rate"]),
                "erm_product_type": ln.get("product_type"),
                "amortisation_type": ln.get("amortisation_type"),
                "origination_date": ln.get("origination_date"),
                "funding_date": ln.get("funding_date"),
                "internal_risk_grade": ln.get("internal_risk_grade"),
                "ifrs9_stage": ln.get("ifrs9_stage"),
                "pd_bucket": ln.get("pd_bucket"),
                "current_loan_to_value": _c(lid, "current_ltv"),
                "collateral_geography": _c(lid, "property_region"),
                "geographic_region_obligor": _b(bid, "region"),
                "youngest_borrower_age": _b(bid, "borrower_age"),
                "borrower_structure": _b(bid, "borrower_structure"),
                "arrears_status": cf_row.get("arrears_status") if len(cf_row) else None,
                "arrears_balance": cf_row.get("arrears_balance") if len(cf_row) else None,
                "portfolio_id": _p(lid, "portfolio_id"),
                "portfolio_name": _p(lid, "portfolio_name"),
                "spv_id": _p(lid, "spv_id"),
                "acquired_portfolio_id": _p(lid, "acquired_portfolio_id"),
                "forecast_funding_probability": None,
                "forecast_funded_balance": None,
                "broker_channel": None, "origination_channel": None,
            })
        # Pipeline opportunities for this reporting date (distinct namespace).
        if not pipeline.empty:
            for _, op in pipeline[pipeline["reporting_date"] == rd].iterrows():
                oid, bid = op["opportunity_id"], op.get("borrower_id")
                bal = float(op["expected_balance"])
                prob = (float(op["forecast_funding_probability"])
                        if pd.notna(op.get("forecast_funding_probability")) else None)
                rows.append({
                    "loan_id": oid, "opportunity_id": oid, "source_record_id": oid,
                    "stable_entity_id": None, "borrower_id": bid,
                    "funded_status": "pipeline",
                    "pipeline_stage": op.get("pipeline_stage"),
                    "current_outstanding_balance": bal,
                    "current_interest_rate": (float(op["interest_rate"])
                                              if pd.notna(op.get("interest_rate")) else None),
                    "erm_product_type": op.get("product_type"),
                    "amortisation_type": None,
                    "origination_date": None, "funding_date": None,
                    "internal_risk_grade": None, "ifrs9_stage": None,
                    "pd_bucket": None, "current_loan_to_value": None,
                    "collateral_geography": None,
                    "geographic_region_obligor": _b(bid, "region"),
                    "youngest_borrower_age": _b(bid, "borrower_age"),
                    "borrower_structure": _b(bid, "borrower_structure"),
                    "arrears_status": None, "arrears_balance": None,
                    "portfolio_id": None, "portfolio_name": None, "spv_id": None,
                    "acquired_portfolio_id": None,
                    "forecast_funding_probability": prob,
                    "forecast_funded_balance": (bal * prob if prob is not None else None),
                    "broker_channel": op.get("broker_channel"),
                    "origination_channel": op.get("origination_channel"),
                })
        frame = pd.DataFrame(rows)
        # Derived: months on book (funded rows only).
        if "funding_date" in frame.columns:
            frame["months_on_book"] = _months_between(frame["funding_date"], rd)
        frames[rd] = frame

    # Lineage completeness for the key fields actually produced.
    produced = set().union(*[set(f.columns) for f in frames.values()]) if frames else set()
    for fld in produced:
        if fld not in LINEAGE and fld not in (
                "snapshot_id", "client_id", "reporting_date", "cut_off_date",
                "upload_timestamp"):
            issues.append(make_issue(MISSING_LINEAGE_FOR_FIELD, INFO,
                                     f"no lineage recorded for {fld!r}", field=fld))
    for fld in EXPECTED_CONSOLIDATED_FIELDS:
        if fld not in produced:
            issues.append(make_issue(MISSING_OPTIONAL_CONSOLIDATED_FIELD, WARNING,
                                     f"expected consolidated field {fld!r} absent",
                                     field=fld))
    if any(i["severity"] == WARNING for i in issues):
        issues.append(make_issue(MULTI_ARTIFACT_CONSOLIDATION_WARNING, INFO,
                                 "consolidation completed with warnings"))
    return frames, LINEAGE, issues


# --------------------------------------------------------------------------- #
# Register
# --------------------------------------------------------------------------- #


def register_snapshots(store, frames: Dict[str, pd.DataFrame], *,
                       client_id: str = "smoke", route: str = "mi"
                       ) -> Tuple[List[str], List[dict]]:
    ids: List[str] = []
    issues: List[dict] = []
    for i, rd in enumerate(sorted(frames)):
        header = SnapshotHeader(
            client_id=client_id, route=route, reporting_date=rd,
            cut_off_date=rd, source_file_id=f"sha256:6c-{client_id}-{i}",
            cadence="monthly", upload_timestamp=f"{rd}T09:00:00")
        try:
            res = store.register_snapshot(header, frames[rd])
            ids.append(res.snapshot_id)
        except Exception as exc:  # pragma: no cover - defensive
            issues.append(make_issue(SNAPSHOT_REGISTRATION_FAILED, ERROR,
                                     f"register failed for {rd}: {exc}",
                                     field=rd))
    return ids, issues
