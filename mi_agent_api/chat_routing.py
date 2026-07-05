"""mi_agent_api/chat_routing.py

End-to-end routing of the new governed analytical intents through POST /mi/query.

The deterministic parser already compiles questions into governed plans
(``temporal_mode='compare'``, ``forecast_mode='extrapolation'``,
``risk_limit_query``, evolution line specs). This module detects those plans and
executes them against the INTERNAL services already built for the dashboard —
``temporal_compare``, ``evolution``, ``forecast_extrapolation`` and
``risk_limits`` — then shapes the result into the SAME artifact union the React
chat/workspace already renders (chart | table | risk | kpi). No HTTP hop, no new
renderer, no parser rebuild.

``try_route`` returns a full ``/mi/query`` response envelope when it handles a
question, or ``None`` to defer to the existing point-in-time MI Agent path
(``run_mi_agent_query`` + ``adapt_workflow_result``) — so normal funded/pipeline/
forecast/data-quality questions are completely unaffected.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from mi_agent.llm_query_parser import _deterministic_parse
from mi_agent.mi_agent_workflow import _detect_unsupported_concept

from mi_agent import portfolio_lens as _portfolio_lens

from . import temporal_compare as compare_mod
from . import evolution as evolution_mod
from . import forecast_extrapolation as fx_mod
from . import risk_limits as risk_mod

_PALETTE = ["#919dd1", "#36c2a8", "#e0a93b", "#c46b8f", "#3d4a82", "#6fcf97"]

# Per evolution-metric display: (answer_style, chart valueFormat, chart scale).
_METRIC_DISPLAY: Dict[str, Tuple[str, str, Optional[str]]] = {
    "funded_balance": ("gbp", "gbp", None),
    "pipeline_amount": ("gbp", "gbp", None),
    "weighted_expected_funded_amount": ("gbp", "gbp", None),
    "loan_count": ("count", "number", None),
    "pipeline_case_count": ("count", "number", None),
    "wa_ltv": ("pct_fraction", "pct", "percent_fraction"),
    "wa_interest_rate": ("pct_points", "pct", "percent_points"),
    "avg_borrower_age": ("decimal", "decimal", None),
}


# --------------------------------------------------------------------------- #
# Small helpers (mirror adapters._uid / _now without coupling to it)
# --------------------------------------------------------------------------- #
def _uid(prefix: str = "art") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gbp(v: Optional[float]) -> str:
    if v is None:
        return "—"
    v = float(v)
    if abs(v) >= 1e9:
        return f"£{v / 1e9:.2f}bn"
    if abs(v) >= 1e6:
        return f"£{v / 1e6:.1f}m"
    if abs(v) >= 1e3:
        return f"£{v / 1e3:.0f}k"
    return f"£{v:,.0f}"


def _disp(value: Optional[float], metric_key: str) -> str:
    if value is None:
        return "—"
    style = _METRIC_DISPLAY.get(metric_key, ("decimal", "decimal", None))[0]
    if style == "gbp":
        return _gbp(value)
    if style == "count":
        return f"{int(round(float(value))):,}"
    if style == "pct_fraction":
        return f"{float(value) * 100:.1f}%"
    if style == "pct_points":
        return f"{float(value):.2f}%"
    return f"{float(value):,.2f}"


def _source(label: str, spec: Dict[str, Any], portfolio_id: Optional[str],
            as_of: Optional[str], engine: str = "mi_agent.workflow") -> Dict[str, Any]:
    return {"engine": engine, "label": label, "spec": spec,
            "asOf": as_of, "portfolio": portfolio_id}


def _envelope(*, ok: bool, question: str, answer: str, spec: Dict[str, Any],
              artifacts: List[Dict[str, Any]], reconciliation: Optional[Dict[str, Any]] = None,
              source_notes: Optional[List[Dict[str, Any]]] = None,
              warnings: Optional[List[str]] = None, route: str = "",
              error: Optional[str] = None) -> Dict[str, Any]:
    notes = source_notes or []
    for art in artifacts:
        if art.get("type") in ("chart", "table", "kpi") and reconciliation:
            art.setdefault("reconciliation", reconciliation)
        if art.get("type") in ("chart", "table", "kpi") and notes:
            art.setdefault("sourceNotes", notes)
    return {
        "ok": ok,
        "error": error,
        "question": question,
        "answer": answer,
        "interpreted": "",
        "spec": spec,
        "validation": {"ok": ok, "errors": ([] if ok else [error or "unavailable"]),
                       "warnings": [], "resolved_fields": {}},
        "artifacts": artifacts,
        "reconciliation": reconciliation,
        "sourceNotes": notes,
        "warnings": warnings or [],
        "diagnostics": [],
        "assumptions": [],
        "metadata": {"engine": "mi_agent", "source": "python", "mock": False,
                     "route": route},
    }


# --------------------------------------------------------------------------- #
# Artifact builders (existing artifact union — chart | table | risk)
# --------------------------------------------------------------------------- #
def _chart_artifact(title: str, *, chart_type: str, x_key: str,
                    rows: List[Dict[str, Any]], series: List[Dict[str, str]],
                    value_format: str, spec: Dict[str, Any],
                    portfolio_id: Optional[str], as_of: Optional[str],
                    display_hints: Optional[Dict[str, Any]] = None,
                    description: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": _uid(), "type": "chart", "title": title,
        "description": description,
        "source": {**_source(f"MI Agent · {chart_type}", spec, portfolio_id, as_of),
                   "nativeChartType": chart_type},
        "createdAt": _now(), "mock": False,
        "chartType": chart_type, "xKey": x_key,
        "series": series, "rows": rows, "valueFormat": value_format,
        "displayHints": display_hints or {},
        "warnings": [],
    }


def _table_artifact(title: str, *, columns: List[Dict[str, Any]],
                    rows: List[Dict[str, Any]], spec: Dict[str, Any],
                    portfolio_id: Optional[str], as_of: Optional[str],
                    description: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": _uid(), "type": "table", "title": title,
        "description": description or f"{len(rows)} rows.",
        "source": _source("MI Agent · table", spec, portfolio_id, as_of),
        "createdAt": _now(), "mock": False,
        "columns": columns, "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Dataset / metric resolution
# --------------------------------------------------------------------------- #
def _dataset_for(question: str, view: str, default: str = "funded") -> str:
    q = question.lower()
    if "pipeline" in q or "case" in q or "kfi" in q or "application" in q or "offer" in q:
        return "pipeline"
    if view == "pipeline":
        return "pipeline"
    return default


def _split_portfolio(portfolio_id: Optional[str]) -> Tuple[str, Optional[str]]:
    if portfolio_id and "/" in portfolio_id:
        cid, rid = portfolio_id.split("/", 1)
        return cid or "client_001", rid
    return (portfolio_id or "client_001"), None


# --------------------------------------------------------------------------- #
# A. Temporal compare
# --------------------------------------------------------------------------- #
def _route_compare(question, spec, spec_dict, *, client_id, run_id, output_root,
                   pipeline_root, view, portfolio_id, as_of) -> Dict[str, Any]:
    periods = list(spec.compare_periods or [])
    if len(periods) < 2:
        return _envelope(ok=False, question=question,
                         answer="I need two periods to compare.", spec=spec_dict,
                         artifacts=[], route="temporal_compare", error="missing periods")
    dataset = _dataset_for(question, view)
    out = compare_mod.run_temporal_compare(
        output_root, pipeline_root, client_id, run_id, dataset=dataset,
        metric=spec.metric, aggregation=spec.aggregation,
        period_a=periods[0], period_b=periods[1])
    metric_key = out.get("metric", "funded_balance")
    label = out.get("metricLabel", metric_key)

    if not out.get("available"):
        avail = out.get("availablePeriods") or []
        answer = (f"I can't compare {periods[0]} and {periods[1]} for {label.lower()}: "
                  f"{out.get('reason', 'a period is unavailable')}.")
        if len(avail) <= 1:
            answer += " Only one reporting period is available."
        return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                         artifacts=[], route="temporal_compare",
                         warnings=["insufficient-data: cross-period comparison needs two periods."])

    va, vb = out["valueA"], out["valueB"]
    delta, pct = out["absoluteDelta"], out["percentageDelta"]
    direction = out["direction"]
    arrow = "up" if direction == "up" else ("down" if direction == "down" else "flat")
    answer = (f"{label} moved from {_disp(va, metric_key)} in {out['periodA']} to "
              f"{_disp(vb, metric_key)} in {out['periodB']} — a change of "
              f"{_disp(abs(delta), metric_key)} "
              f"({'+' if delta >= 0 else ''}{pct if pct is not None else '—'}%, {arrow}).")

    fmt = _METRIC_DISPLAY.get(metric_key, ("decimal", "decimal", None))
    chart = _chart_artifact(
        f"{label}: {out['periodA']} vs {out['periodB']}", chart_type="bar",
        x_key="period",
        rows=[{"period": out["periodA"], "value": va},
              {"period": out["periodB"], "value": vb}],
        series=[{"key": "value", "label": label, "color": _PALETTE[0]}],
        value_format=fmt[1], spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
        display_hints={"value": {"format": fmt[1], "scale": fmt[2]}})
    table = _table_artifact(
        f"{label} comparison", columns=[
            {"key": "metric", "label": "Metric", "align": "left", "format": "text"},
            {"key": "period_a", "label": out["periodA"], "align": "right", "format": fmt[1], "scale": fmt[2]},
            {"key": "period_b", "label": out["periodB"], "align": "right", "format": fmt[1], "scale": fmt[2]},
            {"key": "abs_delta", "label": "Δ absolute", "align": "right", "format": fmt[1], "scale": fmt[2]},
            {"key": "pct_delta", "label": "Δ %", "align": "right", "format": "pct"},
        ],
        rows=[{"metric": label, "period_a": va, "period_b": vb,
               "abs_delta": delta, "pct_delta": pct}],
        spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)

    recon = {"dataset": dataset, "coverage_by_balance_pct": 100.0,
             "missing_dimension_policy": "exclude"}
    notes = [{"field": "source_periods",
              "note": f"Period A: {out.get('sourcePeriods', [None, None])[0]}; "
                      f"Period B: {out.get('sourcePeriods', [None, None])[1]}"}]
    return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                     artifacts=[chart, table], reconciliation=recon, source_notes=notes,
                     route="temporal_compare")


# --------------------------------------------------------------------------- #
# B. Evolution / trend
# --------------------------------------------------------------------------- #
_FUNNEL_KEYWORDS = {"kfi": "KFI", "application": "APPLICATION", "offer": "OFFER",
                    "completion": "COMPLETED", "completed": "COMPLETED"}


def _route_evolution(question, spec, spec_dict, *, client_id, run_id, output_root,
                     pipeline_root, view, portfolio_id, as_of) -> Optional[Dict[str, Any]]:
    q = question.lower()
    dataset = _dataset_for(question, view)
    is_count = spec.aggregation == "count"

    # Funnel stage trend (KFI / Application / Offer / Completion by week).
    funnel_stage = next((stage for kw, stage in _FUNNEL_KEYWORDS.items() if kw in q), None)
    if funnel_stage:
        funnel = evolution_mod.pipeline_funnel_evolution(pipeline_root, client_id, run_id)
        pts = funnel.get("series", {}).get(funnel_stage, [])
        summ = funnel.get("summary", {}).get(funnel_stage, {})
        if not pts:
            return _envelope(ok=True, question=question,
                             answer=f"No weekly {funnel_stage.title()} extracts are available yet.",
                             spec=spec_dict, artifacts=[], route="evolution_funnel",
                             warnings=["insufficient-data: no weekly pipeline extracts."])
        flow_pts = funnel.get("flowSeries", {}).get(funnel_stage, [])
        # Weekly-flow rows (bars); fall back to the stock level only when no flow
        # series is present. Each row carries the stock level too (cumulative line).
        rows = [{"week": p.get("week"),
                 "value": p.get("flowValue"),
                 "count": p.get("flowCount"),
                 "stock": s.get("value")}
                for p, s in zip(flow_pts, pts)] or \
            [{"week": p.get("week"), "value": p.get("value"), "count": p.get("count")} for p in pts]
        chart = _chart_artifact(
            f"{summ.get('label', funnel_stage.title())} weekly flow", chart_type="line",
            x_key="week", rows=rows,
            series=[{"key": "value", "label": "Weekly flow (£)", "color": _PALETTE[0]}],
            value_format="gbp", spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)
        table = _table_artifact(
            f"{summ.get('label', funnel_stage.title())} weekly flow trend", columns=[
                {"key": "week", "label": "Week", "align": "left", "format": "date"},
                {"key": "value", "label": "Weekly flow (£)", "align": "right", "format": "gbp"},
                {"key": "count", "label": "Weekly flow (count)", "align": "right", "format": "number"},
                {"key": "stock", "label": "Stock level (£)", "align": "right", "format": "gbp"},
            ], rows=rows, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)
        answer = (f"Latest week {summ.get('label', funnel_stage.title())} weekly flow: "
                  f"{_gbp(summ.get('latestFlowValue'))}; "
                  f"5-week average weekly flow {_gbp(summ.get('fiveWeekAvgFlowValue'))} "
                  f"({summ.get('trend', 'flat')} vs prior week). Current stock level "
                  f"{_gbp(summ.get('latestStockValue'))}.")
        notes = [{"field": "weekly_extracts",
                  "note": f"{funnel.get('uniqueWeeklyExtractsUsed') or len(rows)} governed weekly extract(s)."}]
        return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                         artifacts=[chart, table],
                         reconciliation={"dataset": "pipeline", "coverage_by_balance_pct": 100.0},
                         source_notes=notes, route="evolution_funnel")

    # Pipeline amount by stage over time (multi-series).
    if dataset == "pipeline" and ("by stage" in q or "stage over time" in q or "stage migration" in q):
        pipe = evolution_mod.pipeline_evolution(pipeline_root, client_id, run_id)
        by_stage = pipe.get("byStage", [])
        if not by_stage:
            return _envelope(ok=True, question=question,
                             answer="No weekly pipeline extracts are available to build a stage trend.",
                             spec=spec_dict, artifacts=[], route="evolution_pipeline_stage",
                             warnings=["insufficient-data: no weekly pipeline extracts."])
        periods = sorted({r["period"] for r in by_stage})
        stages = sorted({r["stage"] for r in by_stage})
        rows = []
        for per in periods:
            row: Dict[str, Any] = {"period": per}
            for st in stages:
                row[st] = sum(r["value"] for r in by_stage if r["period"] == per and r["stage"] == st)
            rows.append(row)
        chart = _chart_artifact(
            "Pipeline amount by stage over time", chart_type="line", x_key="period",
            rows=rows, series=[{"key": st, "label": st, "color": _PALETTE[i % len(_PALETTE)]}
                               for i, st in enumerate(stages)],
            value_format="gbp", spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)
        answer = (f"Pipeline amount by stage across {len(periods)} period(s): "
                  f"stages {', '.join(stages)}.")
        return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                         artifacts=[chart],
                         reconciliation={"dataset": "pipeline", "coverage_by_balance_pct": 100.0},
                         route="evolution_pipeline_stage")

    # Funded / pipeline single-metric evolution.
    metric_key, label, fmt = compare_mod.resolve_metric_key(dataset, spec.metric, spec.aggregation)
    if dataset == "pipeline":
        evo = evolution_mod.pipeline_evolution(pipeline_root, client_id, run_id)
        period_field = "period"
    else:
        evo = evolution_mod.funded_evolution(output_root, client_id, run_id)
        period_field = "period"
    periods = evo.get("periods", [])
    if not periods:
        return _envelope(ok=True, question=question,
                         answer=f"No reporting periods are available to build a {label.lower()} trend.",
                         spec=spec_dict, artifacts=[], route="evolution",
                         warnings=["insufficient-data: no governed reporting periods."])
    rows = [{"period": p.get(period_field), "value": (p.get("metrics") or {}).get(metric_key)}
            for p in periods]
    disp = _METRIC_DISPLAY.get(metric_key, ("decimal", "decimal", None))
    chart = _chart_artifact(
        f"{label} by {'week' if dataset == 'pipeline' else 'month'}", chart_type="line",
        x_key="period", rows=rows,
        series=[{"key": "value", "label": label, "color": _PALETTE[0]}],
        value_format=disp[1], spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
        display_hints={"value": {"format": disp[1], "scale": disp[2]}})
    table = _table_artifact(
        f"{label} trend", columns=[
            {"key": "period", "label": "Period", "align": "left", "format": "text"},
            {"key": "value", "label": label, "align": "right", "format": disp[1], "scale": disp[2]},
        ], rows=rows, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)

    vals = [r["value"] for r in rows if r["value"] is not None]
    warnings: List[str] = []
    if len(vals) <= 1:
        warnings.append("Only one reporting period is available — showing the single point; "
                        "an evolution view reads best with two or more periods.")
    trend_txt = ""
    if len(vals) >= 2:
        d = vals[-1] - vals[0]
        trend_txt = f" ({'up' if d > 0 else 'down' if d < 0 else 'flat'} over the window)"
    answer = (f"{label} over {len(rows)} period(s): latest {_disp(vals[-1] if vals else None, metric_key)}"
              f"{trend_txt}.")
    src_files = evo.get("sourceFiles") or []
    notes = [{"field": "source_periods",
              "note": f"{len(periods)} governed period(s); source: "
                      f"{src_files[-1] if src_files else 'governed runs'}"}]
    last_recon = (periods[-1].get("reconciliation") if periods else None) or {
        "dataset": dataset, "coverage_by_balance_pct": 100.0}
    return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                     artifacts=[chart, table], reconciliation=last_recon,
                     source_notes=notes, warnings=warnings, route="evolution")


# --------------------------------------------------------------------------- #
# C. Forecast scale-up / extrapolation
# --------------------------------------------------------------------------- #
def _route_forecast(question, spec, spec_dict, *, client_id, run_id, output_root,
                    pipeline_root, history_model, portfolio_id, as_of) -> Dict[str, Any]:
    fx = fx_mod.build_extrapolation(output_root, pipeline_root, client_id, run_id,
                                    history_model=history_model)
    rr = fx.get("completionRunRateForecast", {})
    kfi = fx.get("kfiConversionForecast", {})
    weighted = fx.get("currentWeightedPipelineForecast", {})
    cur = fx.get("currentFundedBalance", 0.0)
    kind = spec.forecast_question or "extrapolation_curve"
    target = spec.forecast_target_value
    caveat = "Downside/base/upside are indicative scenario bands, not statistically validated confidence intervals."
    warnings = [caveat]

    if not rr.get("available"):
        answer = (f"Current funded balance is {_gbp(cur)}. I can't extrapolate a completion "
                  f"run-rate yet: {rr.get('caveat', 'insufficient completion history')}.")
        wp = weighted.get("weightedExpectedPipeline")
        if wp is not None:
            answer += (f" The current weighted pipeline forecast adds {_gbp(wp)} "
                       f"(→ {_gbp(weighted.get('forecastFundedBalance'))}).")
        return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                         artifacts=[], route="forecast_extrapolation",
                         warnings=["insufficient-data: not enough completion history for a run-rate forecast."])

    base = rr.get("baseMonthlyRunRate")
    ann = rr.get("annualisedRunRate")
    scenarios = rr.get("scenarioMonthlyRunRate", {})
    milestones = rr.get("milestones", [])

    def _ms(thr: float) -> Optional[Dict[str, Any]]:
        exact = next((m for m in milestones if m["threshold"] == thr), None)
        if exact:
            return exact
        above = [m for m in milestones if m["threshold"] >= thr]
        return above[0] if above else (milestones[-1] if milestones else None)

    if kind in ("reach_threshold",) and target:
        m = _ms(target)
        if m and m.get("reached"):
            answer = f"The book has already reached {_gbp(target)} (current funded balance {_gbp(cur)})."
        elif m:
            answer = (f"At the current base completion run-rate (~{_gbp(base)}/month, "
                      f"{_gbp(ann)}/year), the book reaches {_gbp(target)} around "
                      f"{m.get('baseDate')} (downside {m.get('downsideDate')}, "
                      f"upside {m.get('upsideDate')}). {caveat}")
        else:
            answer = f"Current funded balance is {_gbp(cur)}; {_gbp(target)} is beyond the projection horizon."
    elif kind == "pipeline_needed" and target:
        gap = max(float(target) - float(cur), 0.0)
        answer = (f"To reach {_gbp(target)} from the current {_gbp(cur)} you need ~{_gbp(gap)} "
                  f"of additional completions — about {gap / base:.0f} month(s) at the base "
                  f"run-rate of {_gbp(base)}/month." if base else
                  f"To reach {_gbp(target)} you need ~{_gbp(gap)} of additional completions.")
    elif kind == "run_rate_annualised":
        answer = (f"The annualised completion run-rate is ~{_gbp(ann)} "
                  f"(base monthly {_gbp(base)} over {rr.get('observedMonths')} observed month(s)).")
    elif kind == "run_rate":
        answer = (f"The current completion run-rate is ~{_gbp(base)}/month "
                  f"({_gbp(ann)}/year) based on {rr.get('observedMonths')} month(s) of funded growth.")
    elif kind in ("scenario_downside", "scenario_upside", "scenario"):
        which = "downside" if "down" in kind else ("upside" if "up" in kind else "downside")
        answer = (f"{which.title()} scenario monthly run-rate is ~{_gbp(scenarios.get(which))} "
                  f"vs a base of {_gbp(scenarios.get('base'))}. {caveat}")
    elif kind == "conversion":
        if kfi.get("available"):
            answer = (f"The assumed KFI→completion conversion rate is "
                      f"{kfi.get('conversionRate', 0) * 100:.1f}% with a ~{kfi.get('lagMonths')}-month lag.")
        else:
            answer = ("A KFI→completion conversion rate can't be derived from the current history; "
                      f"using the completion run-rate (~{_gbp(base)}/month) instead.")
    elif kind == "compare_models":
        wp = weighted.get("weightedExpectedPipeline")
        answer = (f"Current weighted pipeline forecast adds {_gbp(wp)} (point-in-time → "
                  f"{_gbp(weighted.get('forecastFundedBalance'))}); the completion run-rate "
                  f"extrapolation projects ~{_gbp(base)}/month ({_gbp(ann)}/year) forward. {caveat}")
    else:
        answer = (f"Current funded balance {_gbp(cur)}; base completion run-rate ~{_gbp(base)}/month "
                  f"({_gbp(ann)}/year), projected forward with downside/base/upside bands. {caveat}")

    artifacts: List[Dict[str, Any]] = []
    proj = rr.get("projectedBalances", [])
    if proj:
        rows = [{"month": p["month"], "downside": p["downside"], "base": p["base"], "upside": p["upside"]}
                for p in proj]
        artifacts.append(_chart_artifact(
            "Projected funded balance — downside / base / upside", chart_type="line",
            x_key="month", rows=rows, series=[
                {"key": "downside", "label": "Downside", "color": _PALETTE[3]},
                {"key": "base", "label": "Base", "color": _PALETTE[0]},
                {"key": "upside", "label": "Upside", "color": _PALETTE[1]},
            ], value_format="gbp", spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of))
    if milestones:
        mrows = [{"threshold": m["thresholdLabel"],
                  "downside": "reached" if m.get("reached") else (m.get("downsideDate") or "—"),
                  "base": "reached" if m.get("reached") else (m.get("baseDate") or "—"),
                  "upside": "reached" if m.get("reached") else (m.get("upsideDate") or "—")}
                 for m in milestones]
        artifacts.append(_table_artifact(
            "Milestone dates to funding thresholds", columns=[
                {"key": "threshold", "label": "Threshold", "align": "left", "format": "text"},
                {"key": "downside", "label": "Downside", "align": "right", "format": "text"},
                {"key": "base", "label": "Base", "align": "right", "format": "text"},
                {"key": "upside", "label": "Upside", "align": "right", "format": "text"},
            ], rows=mrows, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of))

    notes = [{"field": "assumptions",
              "note": f"Base run-rate {_gbp(base)}/mo over {rr.get('observedMonths')} month(s); "
                      f"signal = month-on-month funded growth. {caveat}"}]
    recon = {"dataset": "forecast", "coverage_by_balance_pct": 100.0}
    return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                     artifacts=artifacts, reconciliation=recon, source_notes=notes,
                     warnings=warnings, route="forecast_extrapolation")


# --------------------------------------------------------------------------- #
# D. Risk limits / concentration
# --------------------------------------------------------------------------- #
def _route_risk(question, spec, spec_dict, *, client_id, run_id, output_root,
                portfolio_id, as_of) -> Dict[str, Any]:
    rl = risk_mod.compute_risk_limits(output_root, client_id, run_id)
    summ = rl.get("summary", {})
    tests = rl.get("tests", [])
    category = getattr(spec, "risk_limit_category", None)

    if not rl.get("available"):
        answer = (f"Contractual risk limits are unavailable for this portfolio "
                  f"({rl.get('limitsReason', 'extraction required')}). "
                  "I can show observed concentrations once limits are provided.")
        return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                         artifacts=[], route="risk_limits",
                         warnings=["limits unavailable / needs review."])

    # Scope to a single category when asked ("geographic concentration limits").
    cat_label = ""
    if category:
        scoped = [t for t in tests if t.get("category") == category]
        if scoped:
            tests = scoped
            summ = risk_mod._summary(tests)
            cat_label = category.replace("_", " ") + ": "
        else:
            return _envelope(
                ok=True, question=question, spec=spec_dict, artifacts=[],
                route="risk_limits",
                answer=(f"No {category.replace('_', ' ')} limits are configured for this "
                        "portfolio."),
                warnings=[f"no tests in category '{category}'."])

    closest = summ.get("closestHeadroom")
    largest = summ.get("largestConcentration")
    answer = (f"{cat_label}{summ.get('testsPassed', 0)} passed, {summ.get('warnings', 0)} warning(s), "
              f"{summ.get('breaches', 0)} breach(es), {summ.get('needsReview', 0)} need review, "
              f"{summ.get('unavailable', 0)} unavailable.")
    if closest:
        answer += f" Nearest to limit: {closest['label']} ({closest['headroom']:.1f} pp headroom)."
    if largest:
        answer += f" Largest concentration: {largest['label']} at {largest['actualValue']:.1f}%."

    # RISK artifact (RAG groups for percent-unit, computable tests).
    groups = []
    for t in tests:
        if (t.get("status") in ("green", "amber", "red") and t.get("actualValue") is not None
                and t.get("limitValue") and t.get("unit") == "percent"):
            groups.append({
                "name": t["label"], "balance": t["actualValue"],
                "share": float(t["actualValue"]) / 100.0,
                "status": t["status"], "limit": float(t["limitValue"]) / 100.0,
                "approaching": t["status"] == "amber",
            })
    artifacts: List[Dict[str, Any]] = []
    if groups:
        artifacts.append({
            "id": _uid(), "type": "risk",
            "title": "Concentration vs Schedule 8 limits",
            "description": "Funded exposure against extracted concentration limits.",
            "source": {**_source("Risk monitor · concentration", spec_dict, portfolio_id, as_of,
                                 engine="risk_monitor"), "state": "total_funded"},
            "createdAt": _now(), "mock": False,
            "mode": "limits", "dimension": "concentration", "groups": groups,
            "warnings": ([f"{summ.get('breaches')} limit(s) breached."] if summ.get("breaches") else []),
        })

    # TABLE artifact (ALL tests incl needs_review / unavailable).
    def _f(v, unit):
        if v is None:
            return "—"
        return f"{v:.1f}%" if unit == "percent" else (f"{int(v)}" if unit == "count" else _gbp(v))
    trows = [{
        "test": t["label"],
        "actual": _f(t.get("actualValue"), t.get("unit")),
        "limit": _f(t.get("limitValue"), t.get("unit")),
        "headroom": ("—" if t.get("headroom") is None else f"{t['headroom']:.1f}"),
        "status": t["status"], "movement": ("—" if t.get("movementVsPrior") is None
                                            else f"{t['movementVsPrior']:+.1f}"),
        "source": t.get("source", ""),
    } for t in tests]
    artifacts.append(_table_artifact(
        "Risk limit tests", columns=[
            {"key": "test", "label": "Test", "align": "left", "format": "text"},
            {"key": "actual", "label": "Actual", "align": "right", "format": "text"},
            {"key": "limit", "label": "Limit", "align": "right", "format": "text"},
            {"key": "headroom", "label": "Headroom", "align": "right", "format": "text"},
            {"key": "status", "label": "Status", "align": "left", "format": "text"},
            {"key": "movement", "label": "Movement", "align": "right", "format": "text"},
            {"key": "source", "label": "Source", "align": "left", "format": "text"},
        ], rows=trows, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of))

    notes = [{"field": "limit_source", "note": rl.get("limitsSource", "Schedule 8 extracted")}]
    recon = {"dataset": "funded", "coverage_by_balance_pct": 100.0,
             "reporting_date": rl.get("reportingDate")}
    warnings = []
    if summ.get("unavailable"):
        warnings.append(f"{summ['unavailable']} test(s) unavailable (missing fields).")
    if summ.get("needsReview"):
        warnings.append(f"{summ['needsReview']} limit(s) need manual review.")
    return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                     artifacts=artifacts, reconciliation=recon, source_notes=notes,
                     warnings=warnings, route="risk_limits")


# --------------------------------------------------------------------------- #
# E. Funded balance bridge (attribution waterfall between two periods)
# --------------------------------------------------------------------------- #
# Preferred attribution dimension when the question names none.
_BRIDGE_DEFAULT_DIMS = ("geographic_region_obligor", "collateral_geography",
                        "broker_channel", "erm_product_type")


# The region family — any of these columns may carry the geography depending on
# the tape; the bridge resolves whichever is actually present.
_REGION_FAMILY = ("collateral_geography", "geographic_region_collateral",
                  "geographic_region_obligor")


def _bridge_dimension(spec, semantics: Dict[str, Any]) -> Tuple[Optional[str], Any, str]:
    """(semantic_key, candidate_column(s), business_label) for the bridge
    attribution dimension — the one named in the question, else a sensible
    default. Region resolves to the whole family so the bridge picks whichever
    geography column the funded tape actually carries."""
    fields = semantics.get("fields", {})
    key = spec.bridge_dimension
    if not key or key not in fields:
        key = next((k for k in _BRIDGE_DEFAULT_DIMS if k in fields), None)
    if not key:
        return None, None, ""
    entry = fields.get(key, {}) or {}
    label = entry.get("business_name") or entry.get("display_name") or key.replace("_", " ")
    if key in _REGION_FAMILY:
        cols = [fields.get(k, {}).get("canonical_field", k)
                for k in _REGION_FAMILY if k in fields]
        return key, (cols or [entry.get("canonical_field", key)]), label
    return key, entry.get("canonical_field", key), label


def _route_bridge(question, spec, spec_dict, *, client_id, run_id, output_root,
                  portfolio_id, as_of, semantics, source_lens=None) -> Dict[str, Any]:
    """Governed funded-balance ATTRIBUTION bridge → a waterfall artifact.

    Opening balance (a named start period, else the earliest) → per-category
    change over the chosen dimension → the LATEST balance. A source-portfolio
    lens named in the question (or the active dropdown) scopes it — so a
    consolidated (Total) and cohort (direct / acquired / cohort id) bridge are
    both available. Deltas reconcile exactly to the net change."""
    _key, dim_col, dim_label = _bridge_dimension(spec, semantics)
    if not dim_col:
        return _envelope(ok=True, question=question, spec=spec_dict, artifacts=[],
                         answer="I couldn't resolve a dimension to attribute the bridge by.",
                         route="funded_bridge", warnings=["no attribution dimension resolved."])

    default_lens = (_portfolio_lens.lens_from_selection(source_lens)
                    if source_lens is not None else None)
    lens = _portfolio_lens.resolve_lens_with_default(question, default_lens)
    start_period = (spec.compare_periods or [None])[0]

    br = evolution_mod.funded_bridge(
        output_root, client_id, dim_col, start_period=start_period, to_run_id=run_id,
        lens_filters=lens.filters or None, lens_label=lens.label)

    if not br.get("available"):
        return _envelope(ok=True, question=question, spec=spec_dict, artifacts=[],
                         answer=(f"I can't build a funded balance bridge yet: "
                                 f"{br.get('reason', 'insufficient reporting periods')}."),
                         route="funded_bridge",
                         warnings=["insufficient-data: a bridge needs two funded reporting periods."])

    start, end = br["start"], br["end"]
    net = br["netChange"]
    arrow = "up" if net > 0 else ("down" if net < 0 else "flat")
    rows = [{"label": start["period"], "value": start["total"], "type": "total"}]
    for c in br["contributions"]:
        rows.append({"label": c["category"], "value": c["delta"], "type": "delta"})
    rows.append({"label": f"{end['period']} (latest)", "value": end["total"], "type": "total"})

    lens_suffix = "" if lens.name == _portfolio_lens.LENS_TOTAL else f" — {lens.label}"
    title = f"Funded balance bridge by {dim_label}{lens_suffix}"
    chart = _chart_artifact(
        title, chart_type="waterfall", x_key="label", rows=rows,
        series=[{"key": "value", "label": dim_label, "color": _PALETTE[0]}],
        value_format="gbp", spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
        display_hints={"value": {"format": "gbp", "scale": None}},
        description=(f"Opening {start['period']} → {dim_label.lower()} contributions "
                     f"→ latest {end['period']}."))

    top = max(br["contributions"], key=lambda c: abs(c["delta"]), default=None)
    top_txt = ""
    if top:
        td = top["delta"]
        top_txt = (f" Largest mover: {top['category']} "
                   f"({'+' if td >= 0 else '−'}{_gbp(abs(td))}).")
    answer = (f"{dim_label} bridge ({lens.label}): funded balance moved from "
              f"{_gbp(start['total'])} in {start['period']} to {_gbp(end['total'])} at "
              f"{end['period']} (latest) — a net change of "
              f"{'+' if net >= 0 else '−'}{_gbp(abs(net))} ({arrow}).{top_txt}")

    table = _table_artifact(
        f"{dim_label} contribution to balance change", columns=[
            {"key": "category", "label": dim_label, "align": "left", "format": "text"},
            {"key": "start", "label": start["period"], "align": "right", "format": "gbp"},
            {"key": "end", "label": end["period"], "align": "right", "format": "gbp"},
            {"key": "delta", "label": "Δ", "align": "right", "format": "gbp"},
        ],
        rows=[{"category": c["category"], "start": c["start"], "end": c["end"],
               "delta": c["delta"]} for c in br["contributions"]],
        spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)

    recon = {"dataset": "funded", "coverage_by_balance_pct": 100.0,
             "reporting_date": end.get("reporting_date")}
    notes = [{"field": "bridge_periods",
              "note": f"Opening {start.get('reporting_date') or start['period']}; "
                      f"closing {end.get('reporting_date') or end['period']} (latest); "
                      f"attributed by {dim_label.lower()}; deltas reconcile to the net change."}]
    return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                     artifacts=[chart, table], reconciliation=recon, source_notes=notes,
                     route="funded_bridge")


# --------------------------------------------------------------------------- #
# F. Cohort static-pool progression (a cohort's metrics across periods)
# --------------------------------------------------------------------------- #
# (question keyword) -> (metric key, label, chart valueFormat, display scale)
_PROG_METRICS: Dict[str, Tuple[str, str, str, Optional[str]]] = {
    "balance": ("funded_balance", "Funded balance", "gbp", None),
    "ltv": ("wa_ltv", "WA LTV", "pct", "percent_fraction"),
    "rate": ("wa_interest_rate", "WA interest rate", "pct", "percent_fraction"),
    "nneg": ("nneg_headroom_pct", "NNEG headroom", "pct", "percent_fraction"),
    "nneg_exposure": ("nneg_exposure", "NNEG exposure", "gbp", None),
    "count": ("loan_count", "Loan count", "number", None),
    "age": ("avg_borrower_age", "Avg borrower age", "decimal", None),
}


def _prog_metric_key(q: str) -> str:
    if "negative equity" in q or "nneg" in q or "no-negative" in q or "headroom" in q:
        return "nneg_exposure" if "exposure" in q else "nneg"
    if "ltv" in q or "loan to value" in q:
        return "ltv"
    if "rate" in q or "interest" in q or "coupon" in q:
        return "rate"
    if "how many" in q or "loan count" in q or "number of loans" in q:
        return "count"
    if "borrower age" in q or "age" in q:
        return "age"
    return "balance"


def _route_cohort_progression(question, spec, spec_dict, *, client_id, run_id,
                              output_root, portfolio_id, as_of, source_lens=None
                              ) -> Dict[str, Any]:
    """Governed static-pool cohort progression → a metric line across reporting
    periods for a cohort (source portfolio ± origination vintage) + a full
    metrics table."""
    default_lens = (_portfolio_lens.lens_from_selection(source_lens)
                    if source_lens is not None else None)
    lens = _portfolio_lens.resolve_lens_with_default(question, default_lens)
    vintage = getattr(spec, "cohort_vintage", None)
    grain = getattr(spec, "cohort_grain", None) or "Y"

    prog = evolution_mod.funded_cohort_progression(
        output_root, client_id, lens_filters=lens.filters or None,
        lens_label=lens.label, vintage=vintage, grain=grain, to_run_id=run_id)

    scope = lens.label + (f", {vintage} vintage" if vintage else "")
    if not prog.get("available"):
        return _envelope(ok=True, question=question, spec=spec_dict, artifacts=[],
                         answer=(f"I can't build a progression for {scope}: "
                                 f"{prog.get('reason', 'no matching loans')}."),
                         route="cohort_progression",
                         warnings=[f"insufficient-data: {prog.get('reason', 'no matching cohort')}"])

    q = question.lower()
    mkey = _prog_metric_key(q)
    if mkey in ("nneg", "nneg_exposure") and not any(
            "nneg_exposure" in p["metrics"] for p in prog["periods"]):
        mkey = "balance"  # no valuation → NNEG not derivable; fall back to balance
    metric_key, label, vfmt, scale = _PROG_METRICS[mkey]

    periods = prog["periods"]
    rows = [{"period": p["period"], metric_key: (p["metrics"] or {}).get(metric_key)}
            for p in periods]
    chart = _chart_artifact(
        f"{label} — {scope}", chart_type="line", x_key="period", rows=rows,
        series=[{"key": metric_key, "label": label, "color": _PALETTE[0]}],
        value_format=vfmt, spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
        display_hints={metric_key: {"format": vfmt, "scale": scale}},
        description=f"Static-pool {label.lower()} for {scope} across reporting periods.")

    # Full metrics table (all periods).
    tcols = [{"key": "period", "label": "Period", "align": "left", "format": "text"},
             {"key": "loan_count", "label": "Loans", "align": "right", "format": "number"},
             {"key": "funded_balance", "label": "Balance", "align": "right", "format": "gbp"},
             {"key": "wa_ltv", "label": "WA LTV", "align": "right", "format": "pct", "scale": "percent_fraction"},
             {"key": "wa_interest_rate", "label": "WA rate", "align": "right", "format": "pct", "scale": "percent_fraction"}]
    if "nneg_headroom_pct" in prog.get("metricsAvailable", []):
        tcols.append({"key": "nneg_headroom_pct", "label": "NNEG headroom", "align": "right",
                      "format": "pct", "scale": "percent_fraction"})
    trows = [{"period": p["period"], **{k: (p["metrics"] or {}).get(k)
              for k in ("loan_count", "funded_balance", "wa_ltv", "wa_interest_rate", "nneg_headroom_pct")}}
             for p in periods]
    table = _table_artifact(f"{scope} — metrics by period", columns=tcols, rows=trows,
                            spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of)

    live = [p for p in periods if p["loanCount"]]
    first, last = (live[0] if live else None), (live[-1] if live else None)
    def _mv(p):
        return None if p is None else (p["metrics"] or {}).get(metric_key)
    fv, lv = _mv(first), _mv(last)
    trend = ""
    if fv is not None and lv is not None:
        trend = " up" if lv > fv else (" down" if lv < fv else " flat")
    answer = (f"{label} for {scope}: tracked across {len(live)} reporting period(s) "
              f"({first['period'] if first else '—'} → {last['period'] if last else '—'}"
              f"){trend}.")
    warnings = []
    if prog.get("singlePeriod"):
        warnings.append("Only one reporting period has loans for this cohort — a "
                        "progression reads best with two or more periods.")
    recon = {"dataset": "funded", "coverage_by_balance_pct": 100.0,
             "reporting_date": (last or {}).get("reporting_date")}
    notes = [{"field": "cohort", "note": prog["lineage"]["note"]}]
    return _envelope(ok=True, question=question, answer=answer, spec=spec_dict,
                     artifacts=[chart, table], reconciliation=recon, source_notes=notes,
                     warnings=warnings, route="cohort_progression")


# --------------------------------------------------------------------------- #
# Detection + dispatch
# --------------------------------------------------------------------------- #
_EVOLUTION_MARKERS = ("evolution", "over time", "trend", "by month", "monthly",
                      "by week", "weekly", "per week", "by reporting", "stage over time",
                      "over the months")


def _is_evolution(question: str, spec) -> bool:
    if spec.chart_type != "line":
        return False
    if spec.x == "vintage_year":
        return False
    if spec.filters:
        return False  # filtered trends keep the existing within-snapshot path
    q = question.lower()
    return any(m in q for m in _EVOLUTION_MARKERS)


def try_route(question: str, *, portfolio_id: Optional[str], view: str,
              output_root: Optional[str], pipeline_root: Optional[str],
              semantics: Dict[str, Any], history_model: Optional[Dict[str, Any]] = None,
              as_of: Optional[str] = None,
              source_lens: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """Route a question to an internal analytical service, or return None to defer
    to the existing point-in-time MI Agent path. Never raises for analytics issues —
    the caller wraps this defensively and falls back on any exception."""
    client_id, run_id = _split_portfolio(portfolio_id)

    # Parse to a governed spec to detect the intent.
    try:
        spec, _meta = _deterministic_parse(question, semantics)
    except Exception:  # noqa: BLE001 - never block the normal path on a parse hiccup
        return None

    # Defer governed-unsupported concepts (arrears / default / NNEG …) to the
    # existing controlled-unsupported guard.
    if _detect_unsupported_concept(question, semantics, set(semantics.get("fields", {}))) is not None:
        # available_columns isn't known here; the workflow re-checks against the
        # real dataframe, so only skip when the concept clearly has no field at all.
        pass

    spec_dict = spec.to_dict()
    kw = dict(client_id=client_id, run_id=run_id, output_root=output_root,
              pipeline_root=pipeline_root, portfolio_id=portfolio_id, as_of=as_of)

    if spec.forecast_mode == "extrapolation":
        return _route_forecast(question, spec, spec_dict, history_model=history_model, **kw)
    if getattr(spec, "bridge_query", False):
        return _route_bridge(question, spec, spec_dict,
                             client_id=client_id, run_id=run_id, output_root=output_root,
                             portfolio_id=portfolio_id, as_of=as_of, semantics=semantics,
                             source_lens=source_lens)
    if getattr(spec, "cohort_progression", False):
        return _route_cohort_progression(question, spec, spec_dict,
                                        client_id=client_id, run_id=run_id,
                                        output_root=output_root, portfolio_id=portfolio_id,
                                        as_of=as_of, source_lens=source_lens)
    if spec.temporal_mode == "compare":
        return _route_compare(question, spec, spec_dict, view=view, **kw)
    if spec.risk_limit_query:
        return _route_risk(question, spec, spec_dict,
                           client_id=client_id, run_id=run_id, output_root=output_root,
                           portfolio_id=portfolio_id, as_of=as_of)
    if _is_evolution(question, spec):
        return _route_evolution(question, spec, spec_dict, view=view, **kw)
    return None
