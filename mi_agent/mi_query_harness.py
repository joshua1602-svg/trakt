#!/usr/bin/env python3
"""mi_agent/mi_query_harness.py

Deterministic, registry-driven golden harness for the MI ``/query`` path.

It generates a large suite of natural-language MI questions *from the semantic
registry itself* (dimensions, their business names/synonyms, and measures) —
not from a hand-written list — runs each through the REAL pipeline
(``parse_with_repair`` → ``validate`` → ``execute`` → ``adapt``) and evaluates
the fail-closed dimension invariant end to end.

The single guarantee under test (see :mod:`mi_agent.mi_query_contract`):

    Every grouping dimension the parser recognises is either
      (a) APPLIED   — present in the executor group columns / result columns and
                      surfaced on the chart axes / table columns, OR
      (b) REJECTED  — recorded with a reason (``rejected_dimensions`` / a
                      controlled-unsupported refusal),
    never SILENTLY DROPPED.  "Show balance by borrower type by region" grouped by
    borrower type only is a *silent drop* and must fail the harness.

No LLM and no network: ``parse_with_repair`` falls back to the deterministic
parser when no key is configured, so the whole harness is reproducible in CI.
An optional live-LLM mode is gated behind an explicit flag by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .mi_query_contract import all_group_dims, canonical_of, business_name_of
from .mi_agent_workflow import run_mi_agent_query
from .llm_query_parser import _deterministic_parse

try:  # adapter lives in the sibling API package; import is optional for pure runs
    from mi_agent_api.adapters import adapt_workflow_result
except Exception:  # pragma: no cover - adapter always present in this repo
    adapt_workflow_result = None  # type: ignore


# --------------------------------------------------------------------------- #
# Fixture — a deterministic funded tape with canonical dimension + measure cols
# --------------------------------------------------------------------------- #
# Canonical dimension columns we materialise, mapped to their category domain.
_DIMENSION_DOMAINS: Dict[str, List[str]] = {
    "broker_channel": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
    "geographic_region_obligor": ["North", "South East", "East", "Wales", "Scotland"],
    "account_status": ["Performing", "Watch", "Arrears"],
    "borrower_type": ["Single", "Joint"],
    "borrower_structure": ["Sole", "Joint"],
    "erm_product_type": ["Lump Sum", "Drawdown"],
    "origination_channel": ["Direct", "Intermediary"],
    "interest_rate_type": ["Fixed", "Floating"],
    "occupancy_type": ["Owner Occupied", "Second Home"],
    "ltv_bucket": ["<40", "40-60", "60-80", "80+"],
    "age_bucket": ["60-69", "70-79", "80+"],
    "ticket_bucket": ["<100k", "100-250k", "250k+"],
    "term_bucket": ["<10y", "10-20y", "20y+"],
}

# Canonical measure columns → generator.
_MEASURE_SPECS: Dict[str, Tuple[float, float]] = {
    "current_outstanding_balance": (40_000, 400_000),
    "original_principal_balance": (40_000, 400_000),
    "current_valuation_amount": (150_000, 900_000),
    "current_loan_to_value": (15, 75),
    "current_interest_rate": (3, 9),
    "youngest_borrower_age": (60, 92),
}


def build_fixture(n: int = 400, seed: int = 20240611) -> pd.DataFrame:
    """A reproducible funded tape carrying the canonical columns the harness
    generates queries over. Deterministic given ``seed``."""
    rng = np.random.default_rng(seed)
    data: Dict[str, Any] = {}
    for col, (lo, hi) in _MEASURE_SPECS.items():
        if col == "youngest_borrower_age":
            data[col] = rng.integers(int(lo), int(hi), n)
        elif col in ("current_loan_to_value", "current_interest_rate"):
            data[col] = rng.uniform(lo, hi, n).round(2)
        else:
            data[col] = rng.uniform(lo, hi, n).round(2)
    for col, domain in _DIMENSION_DOMAINS.items():
        data[col] = rng.choice(domain, n)
    # A vintage source so origination-year / cohort queries have a column.
    data["origination_date"] = rng.choice(
        pd.date_range("2021-01-01", "2025-06-01", freq="MS").astype(str), n)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# Registry-driven discovery of which dimensions/metrics the fixture supports
# --------------------------------------------------------------------------- #
def probe_usable_dimensions(df: pd.DataFrame, semantics: dict) -> List[str]:
    """The registry dimension keys that both PARSE from their business name and
    EXECUTE cleanly against ``df`` (land in the executor group columns). This is
    how the harness stays schema-driven: we never hardcode the working set."""
    fields = semantics.get("fields", {})
    metric_bn = "balance"
    usable: List[str] = []
    for key, meta in fields.items():
        if meta.get("role") != "dimension":
            continue
        canonical = meta.get("canonical_field", key)
        if canonical not in df.columns:
            continue
        bn = meta.get("business_name") or key
        res = run_mi_agent_query(f"{metric_bn} by {bn}", df, semantics)
        qr = res.get("query_result")
        gfk = set((getattr(qr, "metadata", {}) or {}).get("group_field_keys") or []) if qr is not None else set()
        if res.get("ok") and key in gfk:
            usable.append(key)
    return usable


def probe_grouped_filter_limitation(df: pd.DataFrame, semantics: dict,
                                    dim_key: str = "geographic_region_obligor") -> Dict[str, Any]:
    """Evidence the current handling of a GROUPED query that also carries a value
    filter (e.g. 'balance by region where LTV above 50%'). Returns whether the
    filter reached the mask and whether the omission was surfaced to the user —
    so the calibration report documents this from live behaviour, not a guess."""
    from .mi_query_contract import business_name_of
    bn = business_name_of(dim_key, semantics)
    q = f"balance by {bn} where LTV above 50%"
    res = run_mi_agent_query(q, df, semantics)
    spec = res.get("spec") or {}
    filters = (spec.get("filters") if isinstance(spec, dict) else getattr(spec, "filters", None)) or {}
    recon = (res.get("reconciliation") or {})
    warnings = res.get("warnings") or []
    filter_mentions_warning = any("filter" in str(w).lower() for w in warnings)
    return {
        "query": q,
        "filter_parsed": bool(filters),
        "filters_applied": recon.get("filters_applied"),
        "records_after_filters": recon.get("records_after_filters"),
        "total_records": recon.get("total_records"),
        "omission_surfaced": filter_mentions_warning,
        "supported": bool(filters) and recon.get("filters_applied") is True,
    }


def usable_metrics(df: pd.DataFrame, semantics: dict) -> Dict[str, dict]:
    fields = semantics.get("fields", {})
    out: Dict[str, dict] = {}
    for key, meta in fields.items():
        if meta.get("role") not in ("measure", "metric"):
            continue
        if meta.get("canonical_field", key) in df.columns:
            out[key] = meta
    return out


# --------------------------------------------------------------------------- #
# Query generation
# --------------------------------------------------------------------------- #
@dataclass
class GeneratedCase:
    id: str
    query: str
    kind: str                       # single_dim | two_dim | three_dim | filter_group | top_n | ranking | weighted_avg | count | rejection
    expected_metric: Optional[str] = None
    expected_dims: List[str] = field(default_factory=list)
    expected_filters: List[str] = field(default_factory=list)
    expect_rejection: bool = False  # a genuinely unsupported concept → refuse
    min_columns: int = 1
    chart_expected: bool = True
    table_expected: bool = True


def _metric_phrase(meta: dict, key: str) -> str:
    syn = meta.get("synonyms") or []
    return (syn[0] if syn else meta.get("business_name") or key)


def _dim_phrase(semantics: dict, key: str, use_synonym: bool = False) -> str:
    meta = semantics.get("fields", {}).get(key, {})
    if use_synonym:
        syn = [s for s in (meta.get("synonyms") or []) if s]
        if syn:
            return syn[0]
    return meta.get("business_name") or key


# Concepts with no backing field in the funded tape — must be refused, never
# answered with a stand-in metric (the classic silent-substitution failure).
_UNSUPPORTED_CONCEPTS = [
    "How many loans are in arrears?",
    "Show defaulted balance by region.",
    "Show NNEG exposure by LTV bucket.",
    "Show credit score by broker.",
    "Show recoveries in period by region.",
]


def generate_cases(df: pd.DataFrame, semantics: dict, *,
                   usable_dims: Optional[List[str]] = None,
                   max_two_dim_pairs: int = 60,
                   max_three_dim: int = 20) -> List[GeneratedCase]:
    """Generate the full case suite from the registry + fixture. Deterministic:
    dimension/metric ordering is registry order, pairs are enumerated in order."""
    dims = usable_dims if usable_dims is not None else probe_usable_dimensions(df, semantics)
    metrics = usable_metrics(df, semantics)
    balance = "current_outstanding_balance"
    rate = "current_interest_rate"
    ltv = "current_loan_to_value"
    cases: List[GeneratedCase] = []
    cid = 0

    def nxt(prefix: str) -> str:
        nonlocal cid
        cid += 1
        return f"{prefix}_{cid:04d}"

    # 1) single dimension × several metrics × (business name and a synonym)
    metric_keys = [m for m in (balance, "current_valuation_amount", "original_principal_balance")
                   if m in metrics]
    for d in dims:
        for mk in metric_keys:
            mphrase = _metric_phrase(metrics[mk], mk)
            for use_syn in (False, True):
                dphrase = _dim_phrase(semantics, d, use_syn)
                # skip a duplicate when there is no distinct synonym
                if use_syn and dphrase == _dim_phrase(semantics, d, False):
                    continue
                cases.append(GeneratedCase(
                    id=nxt("single"), query=f"{mphrase} by {dphrase}",
                    kind="single_dim", expected_metric=mk, expected_dims=[d],
                    min_columns=2))

    # 2) two dimensions (heatmap / pivot). The reported failure class.
    pairs = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            pairs.append((dims[i], dims[j]))
    for a, b in pairs[:max_two_dim_pairs]:
        cases.append(GeneratedCase(
            id=nxt("two"),
            query=f"balance by {_dim_phrase(semantics, a)} by {_dim_phrase(semantics, b)}",
            kind="two_dim", expected_metric=balance, expected_dims=[a, b],
            min_columns=3))

    # 3) three dimensions — at least two must survive (grid); extras may be
    #    rejected with a reason, but never silently dropped.
    triples = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            for k in range(j + 1, len(dims)):
                triples.append((dims[i], dims[j], dims[k]))
    for a, b, c in triples[:max_three_dim]:
        cases.append(GeneratedCase(
            id=nxt("three"),
            query=(f"balance by {_dim_phrase(semantics, a)} by "
                   f"{_dim_phrase(semantics, b)} by {_dim_phrase(semantics, c)}"),
            kind="three_dim", expected_metric=balance, expected_dims=[a, b, c],
            min_columns=3))

    # 4a) filtered KPI — the parser applies a value filter to a whole-book count.
    _filter_probes = [
        ("current_loan_to_value", "LTV", [30, 50, 60]),
        ("current_outstanding_balance", "balance", [150000, 200000]),
        ("youngest_borrower_age", "borrower age", [70, 75]),
    ]
    for fld, noun, thresholds in _filter_probes:
        if fld not in metrics and fld not in _MEASURE_SPECS:
            continue
        for thr in thresholds:
            unit = "%" if fld == "current_loan_to_value" else ""
            cases.append(GeneratedCase(
                id=nxt("filter"),
                query=f"how many loans have {noun} above {thr}{unit}",
                kind="filtered_kpi", expected_metric=None,
                expected_filters=[fld], chart_expected=False, min_columns=1))

    # 4b) GROUPED query + value filter — the filter is applied to the mask BEFORE
    #     grouping. Both the grouping AND the filter must survive (the filter
    #     invariant asserts filters_applied). Numeric, range and categorical.
    for d in dims:
        cases.append(GeneratedCase(
            id=nxt("gfilter_num"),
            query=f"balance by {_dim_phrase(semantics, d)} where LTV above 50%",
            kind="grouped_filter", expected_metric=balance, expected_dims=[d],
            expected_filters=[ltv], min_columns=2))
    for d in dims[:6]:
        cases.append(GeneratedCase(
            id=nxt("gfilter_range"),
            query=f"balance by {_dim_phrase(semantics, d)} where LTV between 40 and 60",
            kind="grouped_filter", expected_metric=balance, expected_dims=[d],
            expected_filters=[ltv], min_columns=2))
    # Categorical grouped filter (borrower structure) — only for non-borrower dims.
    if "borrower_type" in metrics or "borrower_type" in semantics.get("fields", {}):
        for d in [x for x in dims if x not in ("borrower_type", "borrower_structure")][:6]:
            cases.append(GeneratedCase(
                id=nxt("gfilter_cat"),
                query=f"balance by {_dim_phrase(semantics, d)} for joint borrowers",
                kind="grouped_filter", expected_metric=balance, expected_dims=[d],
                expected_filters=["borrower_type"], min_columns=2))

    # 4c) unsupported filter shape — a filter on a field ABSENT from the dataset
    #     must be refused or surfaced (unavailable), never silently ignored.
    _absent = [f for f in ("internal_risk_score", "arrears_balance",
                           "number_of_days_in_arrears")
               if f in semantics.get("fields", {}) and f not in df.columns]
    for fld in _absent[:1]:
        bn = business_name_of(fld, semantics)
        cases.append(GeneratedCase(
            id=nxt("filter_unsupported"),
            query=f"how many loans have {bn} above 700",
            kind="filter_unsupported", expect_rejection=True,
            chart_expected=False, table_expected=False))

    # 5) top-N
    for d in dims:
        cases.append(GeneratedCase(
            id=nxt("topn"),
            query=f"top 5 {_dim_phrase(semantics, d)} by balance",
            kind="top_n", expected_metric=balance, expected_dims=[d], min_columns=2))

    # 6) ranking — largest / smallest
    for d in dims:
        cases.append(GeneratedCase(
            id=nxt("rank_hi"),
            query=f"which {_dim_phrase(semantics, d)} has the largest balance",
            kind="ranking", expected_metric=balance, expected_dims=[d], min_columns=2))
        cases.append(GeneratedCase(
            id=nxt("rank_lo"),
            query=f"which {_dim_phrase(semantics, d)} has the smallest balance",
            kind="ranking", expected_metric=balance, expected_dims=[d], min_columns=2))

    # 7) weighted average (rate / LTV) by dimension
    if rate in metrics:
        for d in dims:
            cases.append(GeneratedCase(
                id=nxt("wavg"),
                query=f"weighted average interest rate by {_dim_phrase(semantics, d)}",
                kind="weighted_avg", expected_metric=rate, expected_dims=[d],
                min_columns=2))

    # 8) count by dimension
    for d in dims:
        cases.append(GeneratedCase(
            id=nxt("count"),
            query=f"how many loans by {_dim_phrase(semantics, d)}",
            kind="count", expected_dims=[d], min_columns=2))

    # 9) rejection — unsupported concepts must be refused, not substituted
    for q in _UNSUPPORTED_CONCEPTS:
        cases.append(GeneratedCase(
            id=nxt("reject"), query=q, kind="rejection", expect_rejection=True,
            chart_expected=False, table_expected=False))

    return cases


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    case: GeneratedCase
    ok: bool
    failure_class: Optional[str] = None   # parser | executor | renderer | rejection | error
    detail: str = ""
    parsed_dims: List[str] = field(default_factory=list)
    applied_dims: List[str] = field(default_factory=list)
    rejected_dims: List[str] = field(default_factory=list)
    result_columns: List[str] = field(default_factory=list)
    chart_type: Optional[str] = None
    chart_axes: List[str] = field(default_factory=list)
    # Per-invariant outcomes (tracked separately for the calibration report).
    dimension_invariant_ok: Optional[bool] = None
    filter_invariant_ok: Optional[bool] = None
    filters_applied: Optional[bool] = None


def _artifact_columns(artifacts: List[dict]) -> Tuple[Optional[str], List[str], List[str]]:
    """(chart_type, chart_axis_keys, table_column_keys) from adapted artifacts."""
    chart_type = None
    axes: List[str] = []
    table_cols: List[str] = []
    for a in artifacts or []:
        if a.get("type") == "chart":
            chart_type = a.get("chartType")
            for k in (a.get("xKey"), a.get("yKey"), a.get("valueKey")):
                if k and k not in axes:
                    axes.append(k)
            for s in a.get("series") or []:
                if s.get("key") and s["key"] not in axes:
                    axes.append(s["key"])
        elif a.get("type") == "table":
            table_cols = [c.get("key") for c in a.get("columns") or []]
    return chart_type, axes, table_cols


def evaluate_case(case: GeneratedCase, df: pd.DataFrame, semantics: dict) -> CaseResult:
    """Run one case through the real pipeline, apply the invariant checks, and
    tag the outcome with the dimension- and filter-invariant flags (so the
    calibration report can report the two invariants separately)."""
    try:
        res = run_mi_agent_query(case.query, df, semantics)
    except Exception as exc:  # pragma: no cover - defensive
        return CaseResult(case=case, ok=False, failure_class="error",
                          detail=f"pipeline raised: {exc}")
    result = _evaluate_case_impl(case, res, semantics)
    dinv = res.get("dimension_invariant") or {}
    finv = res.get("filter_invariant") or {}
    result.dimension_invariant_ok = dinv.get("ok")
    result.filter_invariant_ok = finv.get("ok")
    result.filters_applied = finv.get("filters_applied")
    return result


def _evaluate_case_impl(case: GeneratedCase, res: Dict[str, Any], semantics: dict) -> CaseResult:
    """Apply the invariant checks to a completed pipeline result."""
    inv = res.get("dimension_invariant") or {}
    applied = list(inv.get("applied") or [])
    rejected = [r.get("dimension") for r in (inv.get("rejected") or [])]
    dropped = [d.get("dimension") for d in (inv.get("dropped") or [])]
    spec = res.get("spec")
    parsed = all_group_dims(spec) if spec is not None else []

    # Rejection cases: must be refused (not silently answered with a stand-in).
    if case.expect_rejection:
        refused = (res.get("controlled_unsupported") is True
                   or res.get("ok") is False
                   or bool(res.get("missing_fields")))
        answered_with_data = res.get("query_result") is not None and res.get("ok")
        ok = refused and not answered_with_data
        return CaseResult(case=case, ok=ok,
                          failure_class=None if ok else "rejection",
                          detail="refused" if ok else "unsupported concept was answered with data",
                          parsed_dims=parsed)

    # THE invariant: no parsed dimension silently dropped.
    if dropped:
        return CaseResult(case=case, ok=False, failure_class="executor",
                          detail=f"silent drop: {dropped}", parsed_dims=parsed,
                          applied_dims=applied, rejected_dims=rejected)

    if not res.get("ok"):
        return CaseResult(case=case, ok=False, failure_class="error",
                          detail=res.get("error") or "query not ok",
                          parsed_dims=parsed, applied_dims=applied, rejected_dims=rejected)

    qr = res.get("query_result")
    result_cols = [str(c) for c in getattr(qr, "data").columns] if qr is not None else []

    # Filter check — an expressed value filter must actually reach the mask
    # (parsed into spec.filters AND reflected in the reconciliation), never
    # silently ignored.
    if case.expected_filters:
        spec_filters = set((spec or {}).get("filters") or {}) if isinstance(spec, dict) \
            else set(getattr(spec, "filters", {}) or {})
        recon = (getattr(qr, "metadata", {}) or {}).get("reconciliation") or {} if qr is not None else {}
        for f in case.expected_filters:
            if f not in spec_filters:
                return CaseResult(case=case, ok=False, failure_class="filter",
                                  detail=f"expected filter {f} not parsed into spec.filters",
                                  parsed_dims=parsed, applied_dims=applied,
                                  rejected_dims=rejected, result_columns=result_cols)
        if recon and recon.get("filters_applied") is False:
            return CaseResult(case=case, ok=False, failure_class="filter",
                              detail="filter parsed but reconciliation reports filters_applied=False",
                              parsed_dims=parsed, applied_dims=applied,
                              rejected_dims=rejected, result_columns=result_cols)

    # Parser check — the query's dimensions were recognised.
    expected_present = [d for d in case.expected_dims if d in parsed]
    if case.kind in ("single_dim", "two_dim", "filtered_kpi", "grouped_filter",
                     "top_n", "ranking", "weighted_avg", "count") and case.expected_dims:
        missing = [d for d in case.expected_dims if d not in parsed]
        if missing:
            return CaseResult(case=case, ok=False, failure_class="parser",
                              detail=f"expected dimension(s) not parsed: {missing}",
                              parsed_dims=parsed, applied_dims=applied,
                              rejected_dims=rejected, result_columns=result_cols)

    # Executor check — every parsed dimension applied or explicitly rejected.
    for d in parsed:
        canonical = canonical_of(d, semantics)
        if d in applied or canonical in result_cols or d in rejected:
            continue
        return CaseResult(case=case, ok=False, failure_class="executor",
                          detail=f"dimension {d} neither applied nor rejected",
                          parsed_dims=parsed, applied_dims=applied,
                          rejected_dims=rejected, result_columns=result_cols)

    # Metric check — the requested measure is in the payload (grouped/ranked).
    if case.expected_metric and case.kind != "count":
        mcanon = canonical_of(case.expected_metric, semantics)
        if not any(c == mcanon or c.startswith(mcanon + "_") for c in result_cols):
            return CaseResult(case=case, ok=False, failure_class="executor",
                              detail=f"metric {case.expected_metric} not in result columns",
                              parsed_dims=parsed, applied_dims=applied,
                              rejected_dims=rejected, result_columns=result_cols)

    # Renderer check — adapt and confirm axes/table cover the applied dimensions
    # (never a dimension silently absent from BOTH chart axes and table columns).
    chart_type = None
    chart_axes: List[str] = []
    if adapt_workflow_result is not None:
        adapted = adapt_workflow_result(res, portfolio_id="client_001", as_of=None)
        chart_type, chart_axes, table_cols = _artifact_columns(adapted.get("artifacts") or [])
        for d in applied:
            canonical = canonical_of(d, semantics)
            on_chart = canonical in chart_axes
            on_table = canonical in (table_cols or [])
            if not (on_chart or on_table):
                return CaseResult(case=case, ok=False, failure_class="renderer",
                                  detail=f"applied dimension {d} absent from chart axes AND table columns",
                                  parsed_dims=parsed, applied_dims=applied,
                                  rejected_dims=rejected, result_columns=result_cols,
                                  chart_type=chart_type, chart_axes=chart_axes)
        # Safe chart selection: 2 applied categorical dims must not collapse to a
        # single-axis bar/line (that is the silent-drop-on-the-chart failure).
        applied_dim_cols = [canonical_of(d, semantics) for d in applied]
        two_plus_on_chart = sum(1 for c in applied_dim_cols if c in chart_axes) >= 2
        if len(applied) >= 2 and chart_type in ("bar", "line") and not two_plus_on_chart:
            return CaseResult(case=case, ok=False, failure_class="renderer",
                              detail=f"{len(applied)} dims collapsed onto a {chart_type} chart",
                              parsed_dims=parsed, applied_dims=applied,
                              rejected_dims=rejected, result_columns=result_cols,
                              chart_type=chart_type, chart_axes=chart_axes)

    return CaseResult(case=case, ok=True, parsed_dims=parsed, applied_dims=applied,
                      rejected_dims=rejected, result_columns=result_cols,
                      chart_type=chart_type, chart_axes=chart_axes)


def run_suite(df: Optional[pd.DataFrame] = None, semantics: Optional[dict] = None,
              **gen_kwargs) -> Tuple[List[CaseResult], Dict[str, Any]]:
    """Generate + evaluate the full suite. Returns (results, summary)."""
    if semantics is None:
        from .mi_query_validator import load_mi_semantics
        from pathlib import Path
        semantics = load_mi_semantics(
            Path(__file__).resolve().parent / "mi_semantics_field_registry.yaml")
    if df is None:
        df = build_fixture()
    usable = probe_usable_dimensions(df, semantics)
    cases = generate_cases(df, semantics, usable_dims=usable, **gen_kwargs)
    results = [evaluate_case(c, df, semantics) for c in cases]
    summary = summarise(results, usable_dims=usable)
    summary["grouped_filter_support"] = probe_grouped_filter_limitation(df, semantics)
    return results, summary


def summarise(results: List[CaseResult], *, usable_dims: Optional[List[str]] = None) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.ok)
    by_class: Dict[str, int] = {}
    by_kind: Dict[str, Dict[str, int]] = {}
    failures: List[CaseResult] = []
    rejections_ok = 0
    for r in results:
        k = r.case.kind
        by_kind.setdefault(k, {"total": 0, "passed": 0})
        by_kind[k]["total"] += 1
        if r.ok:
            by_kind[k]["passed"] += 1
            if r.case.kind == "rejection":
                rejections_ok += 1
        else:
            failures.append(r)
            by_class[r.failure_class or "unknown"] = by_class.get(r.failure_class or "unknown", 0) + 1

    # Track the two fail-closed invariants SEPARATELY (requirement 6). A case
    # counts toward an invariant only where that invariant applies (flag is not
    # None); ``ok is False`` is a breach.
    def _inv_stats(attr: str) -> Dict[str, int]:
        checked = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
        return {"checked": len(checked),
                "held": sum(1 for v in checked if v),
                "breached": sum(1 for v in checked if v is False)}

    filters_exercised = sum(1 for r in results if r.filters_applied)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "failures_by_class": by_class,
        "by_kind": by_kind,
        "rejections_correct": rejections_ok,
        "usable_dimensions": list(usable_dims or []),
        "dimension_invariant": _inv_stats("dimension_invariant_ok"),
        "filter_invariant": _inv_stats("filter_invariant_ok"),
        "filters_exercised": filters_exercised,
    }


if __name__ == "__main__":  # pragma: no cover
    results, summary = run_suite()
    print(summary)
    for r in results:
        if not r.ok:
            print("FAIL", r.case.id, r.case.kind, "|", r.failure_class, "|",
                  r.detail, "|", r.case.query)
