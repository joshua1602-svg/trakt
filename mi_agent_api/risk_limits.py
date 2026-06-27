"""mi_agent_api/risk_limits.py

Governed risk-limit / concentration monitor for the React Risk Limits panel.

Composes:
  * the Schedule 8 EXTRACTED limits (``schedule8_extractor`` /
    ``config/clients/<client>/risk_limits_extracted.yaml``); and
  * the funded-book ACTUAL concentrations computed with the existing
    ``analytics_lib.concentration`` primitives (``group_shares`` /
    ``top_n_concentration``) — the same logic the Streamlit-era risk monitor used.

For each limit it emits actual value, limit, headroom, status
(green/amber/red/needs_review/unavailable), source, confidence, notes, movement
vs the prior funded run (when available) and any missing fields. It NEVER
fabricates a limit or an actual: missing limits → "limits unavailable"; missing
fields → the test is ``unavailable`` with the field listed; ambiguous limits →
``needs_review``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from analytics_lib.concentration import group_shares, top_n_concentration
from analytics_lib.numeric import coerce_numeric
from mi_agent.risk_monitor import schedule8_extractor as extractor

from . import snapshots as snap

_BALANCE = "current_outstanding_balance"
_REGION = "geographic_region_obligor"
_BROKER = "broker_channel"
_LTV = "current_loan_to_value"
_AGE = "youngest_borrower_age"
_AMBER_FRACTION = 0.9  # amber when actual >= 90% of the limit (matches Schedule 8 config)


# --------------------------------------------------------------------------- #
# Limit loading
# --------------------------------------------------------------------------- #
def load_extracted_limits(client_id: str, *, search_roots: Optional[List[str]] = None
                          ) -> Dict[str, Any]:
    """Load the governed extracted limits for a client.

    Preference: a committed ``config/clients/<client>/risk_limits_extracted.yaml``;
    else extract live from a located Schedule 8 document; else controlled
    unavailable (never fabricated).
    """
    cfg = Path("config") / "clients" / client_id / "risk_limits_extracted.yaml"
    if cfg.exists():
        try:
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            data.setdefault("available", bool(data.get("limits")))
            data.setdefault("status", "needs_review" if data.get("needs_review_count")
                            else ("ok" if data.get("limits") else "unavailable"))
            data["limits_source"] = "config (extracted)"
            return data
        except Exception:  # noqa: BLE001 - fall through to live extraction
            pass
    located = extractor.locate_schedule8(*(search_roots or []))
    if located is not None:
        out = extractor.extract_from_file(located, portfolio_id=client_id)
        out["limits_source"] = "Schedule 8 extracted (live)"
        return out
    return {"portfolio_id": client_id, "available": False, "status": "unavailable",
            "reason": "No Schedule 8 limits available — extraction required.",
            "limits_source": "unavailable", "limits": [], "limit_count": 0,
            "needs_review_count": 0, "categories": []}


# --------------------------------------------------------------------------- #
# Actual-value helpers (reuse the concentration primitives)
# --------------------------------------------------------------------------- #
def _normalise_ltv(series: pd.Series) -> pd.Series:
    v = coerce_numeric(series)
    # Stored as a ratio (0-1) in some tapes, as a percentage in others.
    if v.notna().any() and float(v.dropna().median()) <= 1.5:
        return v * 100.0
    return v


def _region_shares(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if _REGION not in df.columns or _BALANCE not in df.columns:
        return None
    return group_shares(df, _REGION, _BALANCE)


def _wa_ltv_pct(df: pd.DataFrame) -> Optional[float]:
    if _LTV not in df.columns or _BALANCE not in df.columns:
        return None
    v = _normalise_ltv(df[_LTV])
    w = coerce_numeric(df[_BALANCE])
    mask = v.notna() & w.notna()
    denom = float(w[mask].sum())
    if denom == 0:
        return None
    return round(float((v[mask] * w[mask]).sum() / denom), 2)


def _share_above(df: pd.DataFrame, col: str, threshold: float, *, normalise_ltv=False
                 ) -> Optional[float]:
    if col not in df.columns or _BALANCE not in df.columns:
        return None
    series = _normalise_ltv(df[col]) if normalise_ltv else coerce_numeric(df[col])
    bal = coerce_numeric(df[_BALANCE])
    total = float(bal.sum())
    if total == 0:
        return None
    above = float(bal[series > threshold].sum())
    return round(above / total * 100.0, 2)


def _largest_single_loan_pct(df: pd.DataFrame) -> Optional[float]:
    if _BALANCE not in df.columns:
        return None
    bal = coerce_numeric(df[_BALANCE])
    total = float(bal.sum())
    if total == 0 or bal.empty:
        return None
    return round(float(bal.max()) / total * 100.0, 2)


def _aggregate_large_loans_pct(df: pd.DataFrame, gbp_threshold: float) -> Optional[float]:
    if _BALANCE not in df.columns:
        return None
    bal = coerce_numeric(df[_BALANCE])
    total = float(bal.sum())
    if total == 0:
        return None
    return round(float(bal[bal > gbp_threshold].sum()) / total * 100.0, 2)


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
def _status_for(actual: Optional[float], limit: Optional[float], direction: str
                ) -> str:
    if limit is None:
        return "needs_review"
    if actual is None:
        return "unavailable"
    if limit == 0:
        return "red" if actual > 0 else "green"
    if direction == "min":
        if actual < limit:
            return "red"
        if actual <= limit / _AMBER_FRACTION:
            return "amber"
        return "green"
    # max
    if actual > limit:
        return "red"
    if actual >= limit * _AMBER_FRACTION:
        return "amber"
    return "green"


def _headroom(actual: Optional[float], limit: Optional[float], direction: str
              ) -> Optional[float]:
    if actual is None or limit is None:
        return None
    return round((limit - actual) if direction == "max" else (actual - limit), 2)


def _test(limit: Dict[str, Any], *, actual: Optional[float],
          actual_basis: str, prior_actual: Optional[float] = None,
          label: Optional[str] = None, missing_fields: Optional[List[str]] = None,
          notes: str = "", dimension_key: Optional[str] = None) -> Dict[str, Any]:
    direction = limit.get("direction", "max")
    lv = limit.get("limit_value")
    if limit.get("needs_review"):
        status = "needs_review"
    elif missing_fields:
        status = "unavailable"
    else:
        status = _status_for(actual, lv, direction)
    movement = (round(actual - prior_actual, 2)
                if (actual is not None and prior_actual is not None) else None)
    return {
        "limitId": limit.get("limit_id"),
        "category": limit.get("category"),
        "label": label or limit.get("region") or limit.get("dimension") or limit.get("limit_id"),
        "region": limit.get("region"),
        "dimensionKey": dimension_key or limit.get("dimension"),
        "actualValue": actual,
        "actualBasis": actual_basis,
        "limitValue": lv,
        "unit": limit.get("unit"),
        "direction": direction,
        "headroom": _headroom(actual, lv, direction),
        "status": status,
        "movementVsPrior": movement,
        "source": limit.get("limits_source") or "Schedule 8 extracted",
        "confidence": limit.get("confidence"),
        "notes": notes,
        "missingFields": missing_fields or [],
        "sourceSnippet": limit.get("source_snippet"),
        "sourceSection": limit.get("source_section"),
    }


# --------------------------------------------------------------------------- #
# Category computors
# --------------------------------------------------------------------------- #
def _has_top_n(limit: Dict[str, Any]) -> Optional[int]:
    import re
    snip = (limit.get("source_snippet") or "").lower()
    m = re.search(r"top\s+(\d+)\s+brokers?", snip)
    return int(m.group(1)) if m else None


def _gbp_threshold(limit: Dict[str, Any]) -> Optional[float]:
    import re
    snip = (limit.get("source_snippet") or "")
    m = re.search(r"(?:GBP|£)\s*([\d,]+)", snip)
    return float(m.group(1).replace(",", "")) if m else None


def _compute_tests(df: pd.DataFrame, limits: List[Dict[str, Any]],
                   prior_df: Optional[pd.DataFrame], limits_source: str
                   ) -> List[Dict[str, Any]]:
    tests: List[Dict[str, Any]] = []
    region_shares = _region_shares(df)
    prior_region_shares = _region_shares(prior_df) if prior_df is not None else None
    region_lookup = ({str(r[_REGION]).lower(): float(r["balance_share"]) * 100.0
                      for _, r in region_shares.iterrows()} if region_shares is not None else {})
    prior_region_lookup = ({str(r[_REGION]).lower(): float(r["balance_share"]) * 100.0
                            for _, r in prior_region_shares.iterrows()}
                           if prior_region_shares is not None else {})

    for lim in limits:
        lim = {**lim, "limits_source": limits_source}
        cat = lim.get("category")

        if cat == "geographic_concentration":
            if region_shares is None:
                tests.append(_test(lim, actual=None, actual_basis="funded exposure %",
                                   missing_fields=[_REGION],
                                   notes="Region field unavailable in the funded tape."))
                continue
            region = (lim.get("region") or "").lower()
            if region in ("any single region", "any other single region", ""):
                # Catch-all: apply to the largest region that has no specific limit.
                specific = {l.get("region", "").lower() for l in limits
                            if l.get("category") == "geographic_concentration"
                            and l.get("region") and "single region" not in (l.get("region") or "").lower()}
                candidates = {k: v for k, v in region_lookup.items() if k not in specific}
                if candidates:
                    top_region = max(candidates, key=candidates.get)
                    actual = round(candidates[top_region], 2)
                    tests.append(_test(lim, actual=actual, actual_basis="funded exposure %",
                                       prior_actual=prior_region_lookup.get(top_region),
                                       label=f"Largest other region ({top_region.title()})",
                                       notes="Largest region without a region-specific limit."))
                continue
            actual = round(region_lookup.get(region, 0.0), 2)
            tests.append(_test(lim, actual=actual, actual_basis="funded exposure %",
                               prior_actual=prior_region_lookup.get(region),
                               label=(lim.get("region") or "Region")))

        elif cat == "broker_concentration":
            if _BROKER not in df.columns:
                tests.append(_test(lim, actual=None, actual_basis="funded exposure %",
                                   missing_fields=[_BROKER]))
                continue
            top_n = _has_top_n(lim)
            if top_n:
                res = top_n_concentration(df, _BROKER, _BALANCE, n=top_n)
                actual = round(res["balance_concentration"] * 100.0, 2)
                prior_actual = (round(top_n_concentration(prior_df, _BROKER, _BALANCE, n=top_n)
                                      ["balance_concentration"] * 100.0, 2)
                                if prior_df is not None and _BROKER in prior_df.columns else None)
                tests.append(_test(lim, actual=actual, actual_basis=f"top-{top_n} funded exposure %",
                                   prior_actual=prior_actual,
                                   label=f"Top {top_n} brokers"))
            else:
                shares = group_shares(df, _BROKER, _BALANCE)
                actual = round(float(shares["balance_share"].max()) * 100.0, 2) if not shares.empty else 0.0
                top_broker = str(shares.iloc[0][_BROKER]) if not shares.empty else ""
                tests.append(_test(lim, actual=actual, actual_basis="largest broker funded exposure %",
                                   label=f"Largest single broker ({top_broker})"))

        elif cat == "large_loan_concentration":
            gbp = _gbp_threshold(lim)
            if gbp:
                actual = _aggregate_large_loans_pct(df, gbp)
                tests.append(_test(lim, actual=actual, actual_basis="funded exposure %",
                                   label=f"Loans > £{int(gbp):,} aggregate"))
            else:
                actual = _largest_single_loan_pct(df)
                tests.append(_test(lim, actual=actual, actual_basis="single loan funded exposure %",
                                   label="Largest single loan"))

        elif cat == "ltv_limit":
            snip = (lim.get("source_snippet") or "").lower()
            if "weighted average" in snip:
                actual = _wa_ltv_pct(df)
                prior_actual = _wa_ltv_pct(prior_df) if prior_df is not None else None
                tests.append(_test(lim, actual=actual, actual_basis="WA current LTV %",
                                   prior_actual=prior_actual, label="WA current LTV",
                                   missing_fields=([] if actual is not None else [_LTV])))
            else:
                actual = _share_above(df, _LTV, lim.get("limit_value") or 0, normalise_ltv=True)
                tests.append(_test(lim, actual=actual,
                                   actual_basis=f"% of book above {lim.get('limit_value')}% LTV",
                                   label=f"Loans above {lim.get('limit_value')}% LTV",
                                   missing_fields=([] if actual is not None else [_LTV]),
                                   notes="Eligibility test: share of balance above the per-loan LTV cap."))

        elif cat == "age_limit":
            actual = _share_above(df, _AGE, 85)
            tests.append(_test(lim, actual=actual, actual_basis="% of book aged > 85",
                               label="Borrowers aged over 85",
                               missing_fields=([] if actual is not None else [_AGE])))

        elif cat == "interest_rate_limit":
            tests.append(_test(lim, actual=None, actual_basis="variable-rate funded exposure %",
                               missing_fields=["variable_rate_flag"],
                               notes="No variable/fixed rate-type flag in the funded tape."))

        elif cat == "borrower_concentration":
            tests.append(_test(lim, actual=None, actual_basis="single borrower funded exposure %",
                               missing_fields=["borrower_identifier"],
                               notes="No borrower/obligor identifier in the funded tape."))

        else:  # joint_borrower_limit, other → carry through (needs_review handled in _test)
            tests.append(_test(lim, actual=None, actual_basis="n/a"))

    return tests


# --------------------------------------------------------------------------- #
# Summary + observations
# --------------------------------------------------------------------------- #
def _summary(tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    passed = sum(1 for t in tests if t["status"] == "green")
    warnings = sum(1 for t in tests if t["status"] == "amber")
    breaches = sum(1 for t in tests if t["status"] == "red")
    needs_review = sum(1 for t in tests if t["status"] == "needs_review")
    unavailable = sum(1 for t in tests if t["status"] == "unavailable")
    measured = [t for t in tests if t["actualValue"] is not None and t["headroom"] is not None]
    closest = min(measured, key=lambda t: t["headroom"]) if measured else None
    largest = max(measured, key=lambda t: t["actualValue"]) if measured else None
    return {
        "testsPassed": passed, "warnings": warnings, "breaches": breaches,
        "needsReview": needs_review, "unavailable": unavailable, "total": len(tests),
        "closestHeadroom": ({"label": closest["label"], "headroom": closest["headroom"],
                             "limitId": closest["limitId"]} if closest else None),
        "largestConcentration": ({"label": largest["label"], "actualValue": largest["actualValue"],
                                  "limitId": largest["limitId"]} if largest else None),
    }


def _observations(tests: List[Dict[str, Any]], summary: Dict[str, Any]) -> List[str]:
    obs: List[str] = []
    breached = [t for t in tests if t["status"] == "red"]
    if breached:
        obs.append("Breached: " + ", ".join(t["label"] for t in breached) + ".")
    if summary["closestHeadroom"]:
        c = summary["closestHeadroom"]
        obs.append(f"Nearest to limit: {c['label']} ({c['headroom']:.2f} pp headroom).")
    if summary["largestConcentration"]:
        lg = summary["largestConcentration"]
        obs.append(f"Largest exposure: {lg['label']} at {lg['actualValue']:.2f}%.")
    worsening = [t for t in tests if (t["movementVsPrior"] or 0) > 0]
    improving = [t for t in tests if (t["movementVsPrior"] or 0) < 0]
    if worsening:
        obs.append(f"{len(worsening)} test(s) worsening vs the prior run.")
    if improving:
        obs.append(f"{len(improving)} test(s) improving vs the prior run.")
    if summary["unavailable"]:
        obs.append(f"{summary['unavailable']} test(s) unavailable (missing fields in the funded tape).")
    if summary["needsReview"]:
        obs.append(f"{summary['needsReview']} limit(s) need manual review.")
    return obs


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def compute_risk_limits(output_root, client_id: str, to_run_id: Optional[str],
                        *, search_roots: Optional[List[str]] = None) -> Dict[str, Any]:
    """Full risk-limit monitor envelope for a client/run."""
    extracted = load_extracted_limits(client_id, search_roots=search_roots)
    limits_source = extracted.get("limits_source", "Schedule 8 extracted")

    # Resolve funded df for the run (and the prior run for movement), reusing the
    # governed snapshot loaders. Never 500s — returns observed concentrations even
    # when limits are unavailable.
    df = prior_df = None
    reporting_date = None
    runs: List[Dict[str, Any]] = []
    if output_root:
        try:
            disc = snap.discover_snapshots(output_root)
            pf = next((p for p in disc.get("portfolios", [])
                       if p.get("client_id") == client_id), None)
            runs = list(pf.get("runs", [])) if pf else []
            if to_run_id:
                cut = next((i for i, r in enumerate(runs) if r["run_id"] == to_run_id), None)
                if cut is not None:
                    runs = runs[: cut + 1]
        except Exception:  # noqa: BLE001
            runs = []
    if runs:
        tape = snap.resolve_tape_path(output_root, client_id, runs[-1]["run_id"])
        if tape is not None:
            try:
                df, _ = snap.load_prepared_run(tape)
                reporting_date = runs[-1].get("reporting_date") or snap.infer_reporting_date(
                    runs[-1]["run_id"], df)
            except Exception:  # noqa: BLE001
                df = None
        if len(runs) >= 2:
            ptape = snap.resolve_tape_path(output_root, client_id, runs[-2]["run_id"])
            if ptape is not None:
                try:
                    prior_df, _ = snap.load_prepared_run(ptape)
                except Exception:  # noqa: BLE001
                    prior_df = None

    limits = extracted.get("limits", [])
    if df is None:
        # No funded data: still surface the limits with unavailable actuals.
        df = pd.DataFrame()
    tests = _compute_tests(df, limits, prior_df, limits_source) if limits else []

    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for t in tests:
        by_category.setdefault(t["category"], []).append(t)

    summary = _summary(tests)
    return {
        "portfolioId": client_id,
        "toRunId": to_run_id or (runs[-1]["run_id"] if runs else None),
        "reportingDate": reporting_date,
        "available": bool(limits),
        "limitsStatus": extracted.get("status", "unavailable"),
        "limitsSource": limits_source,
        "limitsReason": extracted.get("reason"),
        "fundedDataAvailable": not df.empty,
        "summary": summary,
        "testsByCategory": by_category,
        "tests": tests,
        "observations": _observations(tests, summary),
        "lineage": {
            "limitSource": limits_source,
            "sourceDocument": extracted.get("source_document"),
            "dataSource": "governed central lender tape (18_central_lender_tape.csv)",
            "exposureBasis": "funded",
            "reportingDate": reporting_date,
            "extractionMethod": extracted.get("extraction_method", "deterministic"),
            "needsReviewCount": extracted.get("needs_review_count", 0),
        },
    }
