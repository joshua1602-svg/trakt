"""mi_agent_api/temporal_compare.py

Governed cross-period (temporal) comparison built ON TOP of the existing
evolution time-series — never a parallel data path. Resolves a compare plan
("compare October and November funded balance", "how did pipeline amount change
from last week") into:

  * period A / period B (resolved from month names or relative tokens);
  * value A / value B;
  * absolute delta + percentage delta + direction;
  * source periods (the governed source files);
  * per-period reconciliation;
  * a controlled insufficient-data response when a period or metric is missing.

It reuses ``evolution.funded_evolution`` / ``evolution.pipeline_evolution`` so
the comparison reconciles to the same governed runs the Evolution tab shows.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from . import evolution as evolution_mod

# Map a parser ``spec.metric`` (semantic field) + aggregation onto the metric key
# that the evolution period dict actually carries, per dataset.
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_RELATIVE_PRIOR = ("prior", "previous", "prior pipeline", "prior week", "last week",
                   "prior month", "last month", "prior run", "prior period",
                   "previous week", "previous month")


def resolve_metric_key(dataset: str, metric: Optional[str], aggregation: str
                       ) -> Tuple[str, str, str]:
    """``(evolution_metric_key, label, format)`` for a compare spec.

    ``format`` is one of ``gbp`` | ``count`` | ``percent`` | ``decimal`` so the UI
    can format value A / value B / delta correctly.
    """
    ds = (dataset or "funded").lower()
    agg = (aggregation or "").lower()
    if ds == "pipeline":
        if agg in ("count", "count_distinct"):
            return "pipeline_case_count", "Pipeline case count", "count"
        if metric and "weighted" in metric:
            return "weighted_expected_funded_amount", "Weighted expected funded", "gbp"
        return "pipeline_amount", "Pipeline amount", "gbp"
    # funded (default)
    if agg in ("count", "count_distinct"):
        return "loan_count", "Loan count", "count"
    if metric == "current_loan_to_value":
        return "wa_ltv", "WA current LTV", "percent"
    if metric == "current_interest_rate":
        return "wa_interest_rate", "WA interest rate", "percent"
    if metric == "youngest_borrower_age":
        return "avg_borrower_age", "Average borrower age", "decimal"
    return "funded_balance", "Funded balance", "gbp"


def _period_month(period: Dict[str, Any]) -> Optional[int]:
    for key in ("reporting_date", "period", "extract_date", "week"):
        val = period.get(key)
        if not val:
            continue
        m = re.search(r"\d{4}-(\d{2})", str(val))
        if m:
            return int(m.group(1))
    return None


def _match_period(periods: List[Dict[str, Any]], token: str) -> Optional[Dict[str, Any]]:
    """Resolve a period token to a period dict, or None when unavailable."""
    if not periods:
        return None
    tok = (token or "").strip().lower()
    if tok in ("latest", "current", "this month", "current month", "newest"):
        return periods[-1]
    if tok in _RELATIVE_PRIOR:
        return periods[-2] if len(periods) >= 2 else None
    # Explicit YYYY-MM
    m = re.fullmatch(r"(\d{4})-(\d{2})", tok)
    if m:
        return next((p for p in periods if str(p.get("period", "")).startswith(tok)), None)
    # Month name
    if tok in _MONTHS:
        want = _MONTHS[tok]
        hits = [p for p in periods if _period_month(p) == want]
        return hits[-1] if hits else None
    return None


def _direction(delta: Optional[float]) -> str:
    if delta is None:
        return "unknown"
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "flat"


def compare_periods(periods: List[Dict[str, Any]], *, metric_key: str,
                    period_a: str, period_b: str, label: str = "", fmt: str = "decimal"
                    ) -> Dict[str, Any]:
    """Governed A-vs-B comparison over an evolution ``periods`` list."""
    pa = _match_period(periods, period_a)
    pb = _match_period(periods, period_b)
    available_periods = [p.get("period") for p in periods]
    missing = [tok for tok, p in ((period_a, pa), (period_b, pb)) if p is None]
    base = {
        "metric": metric_key, "metricLabel": label, "format": fmt,
        "requestedPeriods": [period_a, period_b],
        "availablePeriods": available_periods,
    }
    if missing:
        return {**base, "available": False, "status": "insufficient_data",
                "reason": f"requested period(s) unavailable: {', '.join(missing)}"}

    va = pa["metrics"].get(metric_key)
    vb = pb["metrics"].get(metric_key)
    if va is None or vb is None:
        return {**base, "available": False, "status": "insufficient_data",
                "periodA": pa.get("period"), "periodB": pb.get("period"),
                "reason": f"metric '{metric_key}' is unavailable in one of the periods"}

    abs_delta = round(float(vb) - float(va), 4)
    pct_delta = round((abs_delta / float(va)) * 100, 2) if va else None
    return {
        **base, "available": True, "status": "ok",
        "periodA": pa.get("period"), "periodB": pb.get("period"),
        "reportingDateA": pa.get("reporting_date") or pa.get("extract_date"),
        "reportingDateB": pb.get("reporting_date") or pb.get("extract_date"),
        "valueA": va, "valueB": vb,
        "absoluteDelta": abs_delta, "percentageDelta": pct_delta,
        "direction": _direction(abs_delta),
        "sourcePeriods": [pa.get("source_file"), pb.get("source_file")],
        "reconciliation": {"periodA": pa.get("reconciliation"),
                           "periodB": pb.get("reconciliation")},
        "lineage": {
            "source": "governed evolution time-series (cross-run comparison)",
            "note": "Comparison reconciles to the same governed runs as the Evolution tab.",
        },
    }


def run_temporal_compare(output_root, pipeline_root, client_id: str,
                         to_run_id: Optional[str], *, dataset: str,
                         metric: Optional[str], aggregation: str,
                         period_a: str, period_b: str) -> Dict[str, Any]:
    """Build the relevant evolution series then compute the governed comparison."""
    metric_key, label, fmt = resolve_metric_key(dataset, metric, aggregation)
    if (dataset or "funded").lower() == "pipeline":
        evo = evolution_mod.pipeline_evolution(pipeline_root, client_id, to_run_id)
    else:
        evo = evolution_mod.funded_evolution(output_root, client_id, to_run_id)
    out = compare_periods(evo.get("periods", []), metric_key=metric_key,
                          period_a=period_a, period_b=period_b, label=label, fmt=fmt)
    out["dataset"] = dataset
    out["portfolioId"] = client_id
    out["toRunId"] = to_run_id
    return out
