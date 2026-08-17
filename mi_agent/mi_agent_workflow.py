#!/usr/bin/env python3
"""
mi_agent_workflow.py

Non-UI orchestration for the Streamlit MI Agent: question -> spec -> validate ->
execute -> chart. Kept separate from ``streamlit_mi_agent.py`` so it can be unit
tested without a Streamlit runtime.

Control flow (the deterministic stack is the control layer; the LLM only
proposes an MIQuerySpec — it never executes anything or sees raw data):

    parse_with_repair  ->  validate_mi_query  ->  execute_mi_query  ->  create_mi_chart
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .mi_dataset_profile import profile_dataset, validate_query_data
from .parsed_question import ParsedQuestion
from .mi_chart_factory import MIChartError, MIChartResult, create_mi_chart
from .mi_query_executor import (
    MIQueryExecutionError,
    MIQueryResult,
    execute_mi_query,
)
from .mi_query_spec import MIQuerySpec
from .mi_query_validator import load_mi_semantics, recover_chart_spec, validate_mi_query
from . import portfolio_lens as _portfolio_lens
from . import portfolio_scope as _portfolio_registry

# Chart types the chart factory can render (others are table/summary only).
_RENDERABLE = {"bar", "line", "scatter", "bubble", "heatmap", "treemap"}


#: Validation errors that mean "the dataset does not carry the field this
#: question needs" — the case where we can name the missing field instead of
#: reporting a bare validation failure.
_MISSING_FIELD_RE = re.compile(
    r"Canonical column '([^']+)' \(for semantic field '([^']+)'\) not present")
_UNKNOWN_FIELD_RE = re.compile(r"Unknown semantic field: '([^']+)'")


def _validation_refusal(errors: List[str], semantics: dict) -> str:
    """A refusal that says what could not be fulfilled, in the house style.

    The reference behaviour is the governed unavailable-concept response: name
    the concept, name the field, and state plainly that nothing was substituted.
    A bare "the proposed query failed validation" tells the reader nothing they
    can act on, even though the failing field is known here.
    """
    named: List[str] = []
    for err in errors or []:
        match = _MISSING_FIELD_RE.search(err) or _UNKNOWN_FIELD_RE.search(err)
        if not match:
            continue
        key = match.group(2) if match.re is _MISSING_FIELD_RE else match.group(1)
        entry = (semantics.get("fields", {}) if isinstance(semantics, dict) else {}).get(key) or {}
        label = entry.get("business_name") or entry.get("display_name") or key.replace("_", " ")
        canonical = entry.get("canonical_field", key)
        if label not in [n[0] for n in named]:
            named.append((label, canonical))
    if named:
        if len(named) == 1:
            label, canonical = named[0]
            return (f"'{label}' is not available in this dataset. The MI book for "
                    f"this client does not include {canonical}. This field is not "
                    "reported, so the question cannot be answered from the current "
                    "data (no value was fabricated).")
        labels = ", ".join(f"'{l}'" for l, _ in named)
        columns = ", ".join(c for _, c in named)
        return (f"{labels} are not available in this dataset. The MI book for this "
                f"client does not include {columns}. These fields are not reported, "
                "so the question cannot be answered from the current data (no value "
                "was fabricated).")
    detail = "; ".join(errors or []) or "the requested combination is not supported"
    return ("I could not build a governed query for this question: " + detail +
            ". No substitute figure has been returned.")


def _reporting_date_label(df) -> Optional[str]:
    """The frame's own reporting cut-off, formatted for the execution receipt.

    Read from the data (``data_cut_off_date``), never from the question, so the
    receipt states the date the numbers actually came from. Returns None when the
    frame carries no usable cut-off rather than guessing one.
    """
    try:
        for column in ("data_cut_off_date", "reporting_date"):
            if column not in getattr(df, "columns", []):
                continue
            values = df[column].dropna()
            if values.empty:
                continue
            import pandas as _pd
            stamp = _pd.to_datetime(values.iloc[0], errors="coerce")
            if _pd.isna(stamp):
                continue
            return f"{stamp.day} {stamp.strftime('%B %Y')}"
    except Exception:  # noqa: BLE001 - a label must never break a query
        return None
    return None


def _dedupe(items: List[str]) -> List[str]:
    """De-duplicate a list of strings, preserving first-seen order."""
    seen: set = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


EXAMPLE_QUESTIONS = [
    "Show balance by region",
    "Show weighted average LTV by product type",
    "Show LTV by age bucket and region as a heatmap",
    "Show balance by region and broker as a treemap",
    "Show LTV by borrower age sized by balance",
    "Show top 10 brokers by balance",
    "Show redemptions over time",
]


# --------------------------------------------------------------------------- #
# Spec description (for the UI "Interpreted as:" panel)
# --------------------------------------------------------------------------- #

_AGG_LABEL = {
    "sum": "Sum", "avg": "Average", "weighted_avg": "Weighted average",
    "median": "Median", "count": "Count", "count_distinct": "Distinct count",
    "distribution": "Distribution", "loan_level": "Loan-level",
    "balance_sum": "Balance sum",
}


def _bn(semantics: dict, key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    entry = semantics.get("fields", {}).get(key)
    if entry:
        return entry.get("business_name") or entry.get("display_name") or key
    return key


def describe_spec(spec: MIQuerySpec, semantics: dict,
                  parser_mode: str = "deterministic") -> Dict[str, Any]:
    """Human-readable summary of a spec for display."""
    out: Dict[str, Any] = {}
    out["Chart"] = (spec.chart_type.title() if spec.chart_type not in (None, "none")
                    else f"{spec.intent.title()} (no chart)")
    for slot, label in (("metric", "Metric"), ("dimension", "Dimension"),
                        ("x", "X"), ("y", "Y"), ("size", "Size"), ("color", "Colour")):
        val = getattr(spec, slot)
        if val:
            out[label] = _bn(semantics, val)
    if spec.dimensions:
        out["Dimensions"] = ", ".join(_bn(semantics, d) for d in spec.dimensions)
    if spec.hierarchy:
        out["Hierarchy"] = ", ".join(_bn(semantics, d) for d in spec.hierarchy)
    out["Aggregation"] = _AGG_LABEL.get(spec.aggregation, spec.aggregation)
    if spec.top_n is not None:
        out["Top N"] = spec.top_n
    if spec.filters:
        out["Filters"] = {_bn(semantics, k): v for k, v in spec.filters.items()}
    out["Parser"] = parser_mode
    return out


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #


# Governed concept -> canonical field. When a question names one of these
# concepts but the field's column is absent from the data, the agent returns a
# CONTROLLED unsupported response (never a hallucinated/fallback answer). When the
# field IS present (a future client pack), the query is processed normally.
_UNSUPPORTED_CONCEPTS = [
    (r"\bdays?\s+in\s+arrears\b|\bin\s+arrears\b|\barrears\b", "arrears",
     ["arrears_balance", "days_in_arrears"]),
    (r"\bdefault(ed|s)?\b", "default", ["default_amount"]),
    (r"\brecover(y|ies)\b", "recoveries", ["recoveries_in_period"]),
    (r"\b(losses|loss amount|allocated loss)\b", "losses", ["allocated_losses"]),
    (r"\bnneg\b|\bnegative equity\b|\bno[- ]negative[- ]equity\b", "NNEG",
     ["nneg_flag"]),
    (r"\bindexed\s+(ltv|value|valuation|loan to value)\b", "indexed valuation",
     ["indexed_loan_to_value", "indexed_valuation_amount"]),
    (r"\bcredit score\b", "credit score", ["credit_score"]),
    (r"\bprotected equity\b", "protected equity", ["protected_equity_flag"]),
    (r"\b(roll[- ]?up interest|accrued interest|interest roll[- ]?up)\b",
     "accrued interest", ["accrued_interest"]),
]


def _detect_unsupported_concept(question, semantics, available_columns):
    """Return ``{concept, missing_fields, message}`` when the question names a
    governed concept whose field is absent from this dataset, else None."""
    q = (question or "").lower()
    fields = semantics.get("fields", {}) if isinstance(semantics, dict) else {}
    for pattern, concept, candidate_fields in _UNSUPPORTED_CONCEPTS:
        if not re.search(pattern, q):
            continue
        # A field is "present" if its canonical column exists in the data.
        present = []
        for key in candidate_fields:
            canonical = (fields.get(key, {}) or {}).get("canonical_field", key)
            if canonical in available_columns or key in available_columns:
                present.append(key)
        if present:
            return None  # the field IS available -> process normally
        return {
            "concept": concept,
            "missing_fields": candidate_fields,
            "message": (f"'{concept}' is not available in this dataset. The MI book "
                        f"for this client does not include {', '.join(candidate_fields)}. "
                        "This field is not reported, so the question cannot be answered "
                        "from the current data (no value was fabricated)."),
        }
    return None


def _capability_explanation(question: str, frame, history_periods: int = 1
                            ) -> Optional[str]:
    """Why Trakt cannot answer a question about a KNOWN capability.

    Returns None unless the question names a registered capability that is not
    AVAILABLE for this dataset. A capability that IS available is left alone —
    the ordinary query path owns those, and answering here would be a second
    route to the same number.
    """
    try:
        from trakt_core.capability import (
            AVAILABLE,
            describe_portfolio,
            load_registry,
            match_capability,
            resolve_capability,
        )
    except Exception:
        return None

    try:
        registry = load_registry()
        capability = match_capability(question, registry=registry)
        if capability is None:
            return None

        if frame is None:
            return None
        shape = describe_portfolio(frame, history_periods=history_periods)
        outcome = resolve_capability(capability, shape)
    except Exception:
        return None

    if outcome.status == AVAILABLE:
        if not capability.owned_kpi:
            # A field-plus-aggregation capability: the ordinary query path
            # owns it and computes it correctly. Answering here would be a
            # second route to the same number.
            return None
        # An owned KPI that IS available. Naming its methodology is better
        # than "I couldn't map this question", which is what the caller would
        # otherwise fall through to — and far better than letting a generic
        # aggregation produce a number of its own.
        where = capability.calculation_source or "the shared analytics layer"
        return (
            f"{capability.name} is an owned Trakt metric "
            f"({capability.methodology or 'versioned methodology'}) and is "
            f"AVAILABLE for this portfolio. It is not computed by the ad-hoc "
            f"query path: it comes from {where}, via the governed tool that "
            "wraps it. Request it there so the approved methodology — not a "
            "generic aggregation over a similarly named field — produces the "
            "number.")

    # MI Query is a SINGLE-FRAME engine: one dataset in, one answer out. A
    # capability needing consecutive snapshots therefore cannot be executed
    # here whatever the deployment holds, and saying "1 snapshot is available"
    # would misreport a limitation of this path as a gap in the data.
    needs_history = any(c.type == "history_periods" and c.minimum > 1
                        for c in capability.conditions)
    if needs_history and outcome.reason_code == "INSUFFICIENT_HISTORY":
        where = capability.calculation_source or "the shared analytics layer"
        return (
            f"{capability.name} is measured ACROSS governed snapshots and MI "
            "Query answers from a single dataset, so it cannot be computed on "
            f"this path. It is an owned Trakt metric "
            f"({capability.methodology or 'versioned methodology'}) served by "
            f"{where}; request it through the governed history tools, where "
            "the snapshot window is resolved. No value has been computed and "
            "no other measure has been substituted for the one you asked "
            "about.")

    parts = [f"{capability.name} is {outcome.status} for this portfolio."]
    if outcome.explanation:
        parts.append(outcome.explanation)
    if outcome.missing_inputs:
        parts.append("Missing input(s): " + ", ".join(outcome.missing_inputs) + ".")
    parts.append("No value has been computed and no other measure has been "
                 "substituted for the one you asked about.")
    return " ".join(parts)


def run_mi_agent_query(
    question: str,
    data,
    semantics,
    *,
    llm_enabled: bool = False,
    model: Optional[str] = None,
    parser_mode: str = "deterministic",
    max_repair_attempts: int = 1,
    catalog_mode: str = "core",
    zero_cost_first: bool = True,
    provider: str = "anthropic",
    llm_callable=None,
    extra_filters: Optional[Dict[str, Any]] = None,
    source_portfolio_lens: Optional[Any] = None,
    dataset: Optional[str] = None,
    run_id: Optional[str] = None,
    parsed: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one MI question end to end.

    Parameters
    ----------
    question : natural-language MI question
    data     : pandas DataFrame or path to a canonical CSV
    semantics: dict or path to mi_semantics_field_registry.yaml
    llm_enabled : enable the LLM parser path (else deterministic)
    parser_mode : "deterministic" | "llm" (LLM only used when also llm_enabled)
    llm_callable: optional injected callable(prompt)->str for testing
    extra_filters: optional drill-through filters merged into the parsed spec's
        ``filters`` (e.g. ``{"geographic_region_obligor": "South East"}``). They
        are applied before aggregation by the executor, so the result reflects
        the FULL underlying dataset narrowed to the selection — not just the rows
        already on screen. An unknown filter field is rejected as a controlled
        validation failure (never a 500).
    parsed : an already-parsed :class:`~mi_agent.parsed_question.ParsedQuestion`.
        Supplied by the chat path, which parses ONCE above routing and reuses the
        result here, so the spec that was routed on is the spec that executes.
        When ``None`` (Streamlit, harness, direct callers) this parses itself —
        still exactly one parse per call.

    Returns a dict with: ok, error, parser_mode, spec, spec_obj, interpreted,
    validation, parse_metadata, query_result, chart_result, warnings, metadata.
    """
    if isinstance(semantics, (str, Path)):
        semantics = load_mi_semantics(semantics)

    result: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "question": question,
        "parser_mode": "deterministic",
        "spec": None,
        "spec_obj": None,
        "interpreted": None,
        "validation": None,
        "parse_metadata": None,
        "query_result": None,
        "chart_result": None,
        "warnings": [],
        "metadata": {},
    }
    warnings: List[str] = []

    # ---- read data --------------------------------------------------------
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, (str, Path)):
        try:
            df = pd.read_csv(data)
        except Exception as exc:
            result["error"] = f"Could not read data CSV: {exc}"
            return result
    else:
        result["error"] = "data must be a pandas DataFrame or a path to a CSV"
        return result

    available_columns = set(df.columns)

    # ---- controlled-unsupported guard -------------------------------------
    # If the question asks for a governed concept whose field is NOT in this
    # dataset (e.g. arrears / default / NNEG / credit score on an ERE pack that
    # lacks them), return a CONTROLLED unsupported response instead of silently
    # falling back to a different metric (which would be a misleading answer).
    unsupported = _detect_unsupported_concept(question, semantics, available_columns)
    if unsupported:
        result["controlled_unsupported"] = True
        result["missing_fields"] = unsupported["missing_fields"]
        result["error"] = unsupported["message"]
        result["answer"] = unsupported["message"]
        result["warnings"] = [unsupported["message"]]
        return result

    # ---- parse (ONCE per request) -----------------------------------------
    # The chat path parses above routing and passes the result in, so routing
    # and execution share one spec. Every other caller parses here. Either way
    # the question is parsed exactly once.
    effective_llm = bool(llm_enabled) and parser_mode == "llm"
    try:
        if parsed is None:
            parsed = ParsedQuestion.parse(
                question, semantics, available_columns=available_columns,
                llm_enabled=effective_llm, model=model,
                max_attempts=max_repair_attempts, llm_callable=llm_callable,
                provider=provider, catalog_mode=catalog_mode,
                zero_cost_first=zero_cost_first)
        spec, parse_meta = parsed.spec, parsed.meta
    except Exception as exc:
        result["error"] = f"Parser error: {exc}"
        result["parse_metadata"] = {"parser_mode": parser_mode, "ok": False,
                                    "status": f"parser raised: {exc}"}
        return result

    result["parser_mode"] = parse_meta.get("parser_mode", "deterministic")
    # granular detail (deterministic_zero_cost / llm / llm_repaired / validation_failed)
    result["parser_mode_detail"] = parse_meta.get("parser_mode_detail",
                                                  result["parser_mode"])

    # ---- controlled-unmapped guard -----------------------------------------
    # A question the deterministic parser could not map to ANY governed
    # analytic (no metric, no dimension, no filter, no summary intent) must be
    # refused honestly — never silently answered with a whole-book KPI summary,
    # which answers a different question than the one asked. A question that
    # names a portfolio scope ("show the back book", "direct only") IS a
    # meaningful lens-scoped summary, so it stays on the summary path.
    # An UNRESOLVED METRIC is refused unconditionally — including when a
    # portfolio scope is named, and regardless of parser mode. "the unicorn
    # ratio for the acquired book by region" must not become a balance answer
    # just because it mentions a portfolio. The measure the user asked for does
    # not exist here, and no other measure may stand in for it.
    if parse_meta.get("note") == "unresolved_metric":
        # Before refusing, ask the capability registry whether the question
        # names something Trakt KNOWS ABOUT but cannot produce for THIS book.
        # "What is the contractual WAL of this ERM portfolio?" deserves the
        # reason — repayment is contingent on a borrower event — not "that
        # measure does not exist here", which is both wrong and unhelpful.
        # This resolves no value and substitutes no measure; it only makes a
        # refusal explain itself.
        capability_msg = _capability_explanation(question, df)
        msg = capability_msg or (
            f"{spec.explanation} I haven't computed an answer, and I have not "
            "substituted a different measure for the one you asked about. Ask "
            "for a governed measure — e.g. balance, LTV, interest rate, borrower "
            "age or property value — optionally by a dimension."
        )
        result["unmapped_question"] = True
        result["unresolved_metric"] = True
        result["error"] = msg
        result["answer"] = msg
        result["parse_metadata"] = parse_meta
        result["spec_obj"] = spec
        result["spec"] = spec.to_dict()
        result["validation"] = {
            "ok": False,
            "errors": [f"unresolved_metric: {spec.explanation}"],
            "warnings": [], "resolved_fields": {},
        }
        result["warnings"] = [
            "requested measure is not available in this dataset; no substitute "
            "measure was used."]
        return result

    if (result["parser_mode"] == "deterministic"
            and parse_meta.get("note") == "unmapped"
            and not _portfolio_lens.mentions_portfolio(question)):
        # Sprint 2.5E wired the capability explanation only into the
        # `unresolved_metric` branch. Tracing the real path in the close-out
        # pass showed CPR, contractual WAL, YTM and default rate all arrive
        # HERE instead, so the explanation never fired for any of them and the
        # 2.5E report overstated the integration. Both branches consult it now.
        capability_msg = _capability_explanation(question, df)
        msg = capability_msg or (
            "I couldn't map this question to a governed analytic, so I haven't "
            "computed an answer (nothing was guessed). Try a metric by a "
            "dimension — e.g. 'balance by region', 'weighted average LTV by "
            "ticket size', 'ticket size by borrower type' — a cross-period "
            "comparison ('compare October and November'), a scale-up forecast "
            "('run-rate to £100m'), risk limits ('are we within limits?'), or "
            "'portfolio summary'."
        )
        result["unmapped_question"] = True
        result["error"] = msg
        result["answer"] = msg
        result["parse_metadata"] = parse_meta
        result["spec_obj"] = spec
        result["spec"] = spec.to_dict()
        result["validation"] = {
            "ok": False,
            "errors": ["unmapped_question: no governed metric, dimension or "
                       "intent was recognised in the question"],
            "warnings": [], "resolved_fields": {},
        }
        result["warnings"] = ["question not understood: no governed metric, "
                              "dimension or intent was recognised."]
        return result
    # ---- source-portfolio lens (Total / Direct / Acquired / cohort) -------
    # Deterministic: a portfolio scope named in the question ("acquired book",
    # "direct_001", "back book") overrides the selected lens (the UI dropdown
    # default). The lens is realised as a filter on the provenance fields, so it
    # only narrows rows — it never changes any MI calculation. No-op when the
    # dataset carries no provenance (the filter simply matches nothing to drop).
    _portfolio_scope = None
    if "source_portfolio_id" in available_columns or "source_portfolio_type" in available_columns:
        _default_lens = (_portfolio_lens.lens_from_selection(source_portfolio_lens)
                         if source_portfolio_lens is not None else None)
        _lens = _portfolio_lens.resolve_lens_with_default(question, _default_lens)
        # The lens names the scope; the GOVERNED REGISTRY resolves what that
        # scope means. "direct" is every portfolio currently typed direct, so
        # onboarding direct_002 widens the answer with no code change — and the
        # filter is the resolved id list, never a type string, so a group is
        # exactly the sum of its registered members.
        _registry, _portfolio_scope = _portfolio_registry.resolve(
            df, _portfolio_lens.context_id(_lens))
        _portfolio_lens.apply_scope(spec, _lens, _portfolio_scope)
        result["portfolio_lens"] = spec.portfolio_lens
        result["portfolio_scope"] = _portfolio_scope.to_dict()
        if not _portfolio_scope.is_total:
            warnings.append(f"portfolio scope applied: {_portfolio_scope.label} "
                            f"({', '.join(_portfolio_scope.portfolio_ids) or 'no portfolios'})")
        elif _portfolio_scope.fell_back_to_total:
            # The requested portfolio is not in the governed registry, so the
            # answer widened to the whole book. Recording that only in the scope
            # dict is not a disclosure — a stale selection, a renamed portfolio
            # or a typo would silently change the scope of the answer. Say it.
            warnings.append(
                f"Requested portfolio '{_portfolio_scope.requested_context_id}' is "
                "not in the governed portfolio registry, so this answer covers the "
                "TOTAL book "
                f"({', '.join(_portfolio_scope.portfolio_ids) or 'no portfolios'}) "
                "rather than the requested scope.")

    # ---- merge drill-through filters into the parsed spec -----------------
    # Additive: caller-supplied filters (e.g. a UI drill into one region/broker/
    # year/SPV/stage) combine with any the parser inferred. They are validated +
    # applied by the executor before aggregation, against the full dataset.
    if extra_filters:
        merged = dict(spec.filters or {})
        merged.update(extra_filters)
        spec.filters = merged
        warnings.append(
            "drill-through filters applied: "
            + ", ".join(f"{k}={v!r}" for k, v in extra_filters.items()))

    result["spec_obj"] = spec
    result["spec"] = spec.to_dict()
    result["parse_metadata"] = parse_meta
    result["interpreted"] = describe_spec(spec, semantics, result["parser_mode"])

    # Predicates the user asked for that could not be applied (e.g. a joint-borrower
    # filter when no borrower-structure field exists) are surfaced as warnings —
    # never silently dropped. They also flow into the API query-audit panel.
    for note in getattr(spec, "unavailable_filters", None) or []:
        warnings.append(f"Filter not applied (field unavailable): {note}")

    # ---- validate (with recovery) -----------------------------------------
    # The validator is also a RECOVERY/control layer: when a spec fails only
    # because the chart type is wrong for the plan (e.g. a metric-only query
    # proposed as a bar), auto-correct to a safe KPI/table and re-validate rather
    # than returning a validation failure.
    vr = validate_mi_query(spec, semantics, available_columns=available_columns)
    # Recovery must NOT mask a query that explicitly asked to group ("... by X")
    # but whose dimension could not be resolved — that must fail cleanly so the
    # operator knows the breakdown is unavailable. Only metric-only questions
    # (no grouping marker) are eligible for KPI/table auto-correction.
    _grouping_marker = bool(re.search(r"\b(by|per|across|split|breakdown|grouped)\b",
                                      (question or "").lower()))
    if not vr.ok and not _grouping_marker:
        recovered = recover_chart_spec(spec, semantics, available_columns)
        if recovered is not None:
            rvr = validate_mi_query(recovered, semantics, available_columns=available_columns)
            if rvr.ok:
                spec = recovered
                vr = rvr
                result["spec_obj"] = spec
                result["spec"] = spec.to_dict()
                result["interpreted"] = describe_spec(spec, semantics, result["parser_mode"])
                result["recovered"] = True
                warnings.append(
                    "auto-corrected the plan to a KPI/table (the metric-only query "
                    "did not need a chart dimension)")
    result["validation"] = vr.to_dict()
    warnings.extend(vr.warnings)
    if not vr.ok:
        result["interpreted"]["Validation"] = "Failed"
        result["error"] = _validation_refusal(vr.errors, semantics)
        result["answer"] = result["error"]
        result["controlled_refusal"] = True
        result["warnings"] = _dedupe(warnings)
        result["metadata"] = {
            "parse_metadata": parse_meta,
            "parser_mode_detail": parse_meta.get("parser_mode_detail"),
            "repair_attempts": parse_meta.get("repair_attempts"),
            "repair_skipped_reason": parse_meta.get("repair_skipped_reason"),
            "llm": parse_meta.get("llm"),
        }
        return result

    # ---- data-aware validation (single dataset profile is the source of truth)
    # The name/role validator above never inspects values. Before claiming
    # "Validation: Passed", confirm the requested metric has numeric values, the
    # dimension(s) have non-blank values, filter fields carry values, and a
    # loan-level x/y/size has usable rows. A missing/empty LTV therefore fails
    # here with an exact reason instead of rendering an empty chart.
    profile = profile_dataset(df, semantics)
    result["display_hints"] = profile.get("display_hints", {})
    data_errors = validate_query_data(spec, df, semantics, profile)
    if data_errors:
        val = result.get("validation") or {"ok": True, "errors": [], "warnings": [],
                                            "resolved_fields": {}}
        val["ok"] = False
        val.setdefault("errors", []).extend(e["detail"] for e in data_errors)
        val["data_validation_errors"] = data_errors
        result["validation"] = val
        result["interpreted"]["Validation"] = "Failed"
        result["error"] = "The query cannot be answered from the prepared data: " + \
            "; ".join(f"{e['field']}: {e['reason']}" for e in data_errors)
        result["warnings"] = _dedupe(warnings)
        result["metadata"] = {
            "parse_metadata": parse_meta,
            "parser_mode_detail": parse_meta.get("parser_mode_detail"),
            "data_validation_errors": data_errors,
        }
        return result
    result["interpreted"]["Validation"] = "Passed"

    # ---- execute ----------------------------------------------------------
    # Missing grouping values are bucketed under "Unknown / Missing" by default so
    # results reconcile to the funded book; the operator can opt out by asking to
    # exclude missing data.
    q_lower = (question or "").lower()
    missing_policy = ("exclude"
                      if any(p in q_lower for p in ("exclude missing", "excluding missing",
                                                    "without missing", "drop missing",
                                                    "ignore missing"))
                      else "bucket")
    try:
        qres: MIQueryResult = execute_mi_query(
            spec, df, semantics, validate=False,
            missing_dimension_policy=missing_policy,
            dataset=dataset, run_id=run_id,
        )
    except MIQueryExecutionError as exc:
        result["error"] = f"Execution failed: {exc}"
        # Surface as controlled validation output (never a raw 500). For a
        # duplicate-column defect, carry the structured fields the UI/API expose.
        val = result.get("validation") or {"ok": True, "errors": [], "warnings": [],
                                            "resolved_fields": {}}
        val["ok"] = False
        val.setdefault("errors", []).append(str(exc))
        dup = getattr(exc, "duplicate_columns", None)
        if dup is not None:
            val["duplicate_column_names"] = dup
            val["duplicate_query_fields_affected"] = getattr(exc, "affected_fields", [])
        result["validation"] = val
        if isinstance(result.get("interpreted"), dict):
            result["interpreted"]["Validation"] = "Failed"
        result["warnings"] = _dedupe(warnings)
        return result
    result["query_result"] = qres
    warnings.extend(qres.warnings)

    # ---- fail-closed dimension invariant ----------------------------------
    # Every grouping dimension the parser attached MUST be applied in execution
    # or explicitly rejected with a reason. A parsed dimension that is neither is
    # a SILENT DROP (e.g. "balance by borrower type by region" grouping only by
    # borrower type) — refuse rather than answer with a misleading result.
    from .mi_query_contract import check_dimension_invariant
    invariant = check_dimension_invariant(spec, qres, semantics)
    result["dimension_invariant"] = {
        "ok": invariant.ok, "applied": invariant.applied,
        "rejected": invariant.rejected, "dropped": invariant.dropped}
    if not invariant.ok:
        val = result.get("validation") or {"ok": True, "errors": [], "warnings": [],
                                            "resolved_fields": {}}
        val["ok"] = False
        val.setdefault("errors", []).append(invariant.message())
        val["dropped_dimensions"] = invariant.dropped
        result["validation"] = val
        if isinstance(result.get("interpreted"), dict):
            result["interpreted"]["Validation"] = "Failed"
        result["error"] = invariant.message()
        result["warnings"] = _dedupe(warnings + [invariant.message()])
        return result

    # A dimension the parser recognised but execution could not apply (e.g. a
    # third dimension on a two-axis heatmap) is explicitly rejected WITH a reason.
    # Surface that reason to the user as a warning — a rejection is only
    # fail-closed if it is visible, not merely recorded in metadata.
    for rej in (qres.metadata or {}).get("rejected_dimensions") or []:
        name = rej.get("dimension")
        reason = rej.get("reason") or "not applied"
        warnings.append(f"Grouping not applied for '{name}': {reason}")

    # ---- fail-closed FILTER invariant -------------------------------------
    # Every value filter the parser attached MUST be applied to the execution
    # mask or explicitly surfaced (unavailable field / rejected with a reason).
    # A parsed filter that is silently not applied would return UNFILTERED data
    # under a filtered question — refuse rather than mislead.
    from .mi_query_contract import check_filter_invariant
    filter_invariant = check_filter_invariant(spec, qres, semantics)
    result["filter_invariant"] = {
        "ok": filter_invariant.ok,
        "filters_applied": filter_invariant.filters_applied,
        "parsed_filters": filter_invariant.parsed_filters,
        "applied_filters": filter_invariant.applied_filters,
        "rejected_filters": filter_invariant.rejected_filters,
        "unavailable_filters": filter_invariant.unavailable_filters,
        "dropped": filter_invariant.dropped}
    if not filter_invariant.ok:
        val = result.get("validation") or {"ok": True, "errors": [], "warnings": [],
                                            "resolved_fields": {}}
        val["ok"] = False
        val.setdefault("errors", []).append(filter_invariant.message())
        val["dropped_filters"] = filter_invariant.dropped
        result["validation"] = val
        if isinstance(result.get("interpreted"), dict):
            result["interpreted"]["Validation"] = "Failed"
        result["error"] = filter_invariant.message()
        result["warnings"] = _dedupe(warnings + [filter_invariant.message()])
        return result
    # A filter that could not be applied (field unavailable) is surfaced plainly
    # (the KPI branch already warns on spec.unavailable_filters; keep parity for
    # any rejected_filters recorded by the executor).
    for rej in (qres.metadata or {}).get("rejected_filters") or []:
        name = rej.get("filter")
        reason = rej.get("reason") or "not applied"
        warnings.append(f"Filter not applied for '{name}': {reason}")

    # Reconciliation / coverage footer (every artifact). When some balance is
    # excluded (e.g. the operator asked to exclude missing dims), say so plainly.
    recon = (qres.metadata or {}).get("reconciliation")
    if recon:
        result["reconciliation"] = recon
        excl = recon.get("balance_excluded_missing")
        inc = recon.get("balance_included")
        tot = recon.get("total_balance")
        if excl and inc is not None and tot:
            fields = recon.get("missing_dimension_fields") or []
            because = (" because " + " and/or ".join(fields) + " was missing") if fields else ""
            warnings.append(
                f"This result covers £{inc:,.0f} of the £{tot:,.0f} funded book; "
                f"£{excl:,.0f} was excluded{because}.")

    # A grouped / loan-level result with no rows after preparation is not a
    # "passed" query — surface it as a controlled validation failure with an
    # exact reason rather than rendering an empty chart.
    if qres.result_type in ("table", "loan_level") and qres.row_count == 0:
        reason = ("loan_level_no_usable_rows" if qres.result_type == "loan_level"
                  else "no_values_after_preparation")
        val = result.get("validation") or {"ok": True, "errors": [], "warnings": [],
                                            "resolved_fields": {}}
        val["ok"] = False
        val.setdefault("errors", []).append(
            f"{reason}: the query produced no usable rows after preparation")
        val["data_validation_errors"] = [{"field": spec.dimension or spec.x or "",
                                           "reason": reason,
                                           "detail": "no usable rows after preparation"}]
        result["validation"] = val
        result["interpreted"]["Validation"] = "Failed"
        result["error"] = f"The query produced no usable rows ({reason})."
        result["warnings"] = _dedupe(warnings)
        return result

    # ---- P0: execution receipt + semantic-completeness guard ---------------
    # The dimension and filter invariants above compare the SPEC with execution.
    # They cannot see intent the parser dropped before a spec existed, which is
    # how a scoped question came to be answered from the whole book. Re-derive
    # the material facets from the QUESTION and reconcile them against what
    # actually executed, then either refuse, disclose, or stand.
    try:
        from . import execution_receipt as _receipt_mod

        _requested_dims = _receipt_mod.requested_dimension_terms(
            question, semantics, available_columns=available_columns)
        _facets = _receipt_mod.detect_requested_facets(
            question, semantics, frame=df, requested_dimensions=_requested_dims)
        _facets = _receipt_mod.reconcile_facets(
            _facets, spec=spec, query_result=qres, semantics=semantics,
            available_columns=available_columns, route=None,
            # The point-in-time executor has no stress/scenario capability, so a
            # scenario is never applied on this path. Stating that explicitly is
            # what turns "what would a 10% fall do to LTV" into a refusal rather
            # than today's current-LTV answer.
            scenario_applied=False)
        _substitution = _receipt_mod.detect_substitution(
            _facets, spec=spec, query_result=qres, semantics=semantics)
        if not _substitution:
            _substitution = _receipt_mod.detect_measure_substitution(
                question, metric_key=getattr(spec, "metric", None))
        receipt = _receipt_mod.build_receipt(
            spec=spec, query_result=qres, semantics=semantics, facets=_facets,
            parser_confidence=(parse_meta or {}).get("parser_confidence"),
            period=_reporting_date_label(df))
        verdict, message = _receipt_mod.assess(receipt, substitution=_substitution)
        result["execution_receipt"] = receipt.to_dict()
        result["semantic_guard"] = {"verdict": verdict, "message": message,
                                    "facets": [f.to_dict() for f in _facets],
                                    "substitution": _substitution}
        if verdict == _receipt_mod.VERDICT_REFUSE:
            # Fail closed: the calculation that ran answers a materially
            # different question from the one asked. Never present it.
            result["ok"] = False
            result["error"] = message
            result["answer"] = message
            result["controlled_refusal"] = True
            result["semantic_refusal"] = True
            result["warnings"] = _dedupe(warnings + [message])
            result["metadata"] = {
                "parse_metadata": parse_meta,
                "parser_mode_detail": parse_meta.get("parser_mode_detail"),
                "execution_receipt": result["execution_receipt"],
                "semantic_guard": result["semantic_guard"],
                "llm": parse_meta.get("llm"),
            }
            return result
        if verdict == _receipt_mod.VERDICT_PARTIAL and message:
            warnings.append(message)
    except Exception as exc:  # noqa: BLE001 - the guard must never break a good answer
        # A guard fault is itself a safety event: record it loudly rather than
        # letting an unguarded answer through silently.
        warnings.append(f"semantic completeness guard unavailable: {exc}")
        result["semantic_guard"] = {"verdict": "unavailable", "error": str(exc)}

    # ---- chart (only where a chart type is renderable) --------------------
    chart_result: Optional[MIChartResult] = None
    if spec.chart_type in _RENDERABLE:
        try:
            chart_result = create_mi_chart(qres, semantics)
            warnings.extend(chart_result.warnings)
        except MIChartError as exc:
            warnings.append(f"Chart not rendered: {exc}")
    else:
        warnings.append(
            f"No chart rendered (chart_type={spec.chart_type!r}); "
            f"showing the result table only."
        )
    result["chart_result"] = chart_result

    # ---- end-to-end query trace (parser -> executor -> invariant) ---------
    # Makes it immediately clear whether a fault is parser-, executor- or
    # renderer-side. The adapter enriches it with the emitted chart axes.
    # The trace is a diagnostic aid — it must NEVER break an otherwise-good
    # answer. Build it defensively.
    try:
        from .mi_query_contract import build_query_trace
        result["query_trace"] = build_query_trace(
            question=question, spec=spec, parse_meta=parse_meta,
            query_result=qres, semantics=semantics, invariant=invariant,
            filter_invariant=filter_invariant)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"query trace unavailable: {exc}")

    # ---- governed portfolio coverage --------------------------------------
    # Which portfolios in scope actually answered, which could not, and why —
    # resolved from the SAME fields the executor calculated on, so a coverage
    # statement can never disagree with the number above it. Produced here, in
    # the governed backend; React and Copilot format it, never derive it.
    if _portfolio_scope is not None:
        try:
            _fields = _portfolio_registry.requested_fields(spec)
            _resolved = (result.get("validation") or {}).get("resolved_fields") or {}
            _fields = _dedupe([f for f in list(_fields) + [
                v for v in _resolved.values() if isinstance(v, str)] if f])
            _fields = [f for f in _fields if f in available_columns]
            _coverage = _portfolio_registry.coverage_for_frame(
                df, _portfolio_scope, fields=_fields)
            result["portfolio_coverage"] = _coverage.to_dict()
            if not _coverage.is_fully_consolidated and _coverage.disclosure():
                warnings.append(_coverage.disclosure())
        except Exception as exc:  # noqa: BLE001 - disclosure must never break an answer
            warnings.append(f"portfolio coverage unavailable: {exc}")

    result["ok"] = True
    # The chart factory copies the executor's warnings onto its result, so the
    # same warning can arrive via both qres and chart_result — de-duplicate
    # while preserving order.
    result["warnings"] = _dedupe(warnings)
    result["metadata"] = {
        "parse_metadata": parse_meta,
        "parser_mode_detail": parse_meta.get("parser_mode_detail"),
        "repair_attempts": parse_meta.get("repair_attempts"),
        "repair_skipped_reason": parse_meta.get("repair_skipped_reason"),
        "llm": parse_meta.get("llm"),
        "executor_metadata": qres.metadata,
        "result_type": qres.result_type,
        "row_count": qres.row_count,
        "chart_type": spec.chart_type if chart_result else None,
        "reconciliation": (qres.metadata or {}).get("reconciliation"),
    }
    return result


# --------------------------------------------------------------------------- #
# Export helpers (return bytes/strings; the UI wires them to download buttons)
# --------------------------------------------------------------------------- #


def _reconciliation_footer_lines(recon: Dict[str, Any]) -> List[str]:
    """Human-readable reconciliation/coverage footer lines for an export."""
    if not recon:
        return []

    def _gbp(v):
        return "" if v is None else f"£{float(v):,.2f}"

    lines = [
        "# --- Reconciliation / coverage ---",
        f"# Dataset: {recon.get('dataset') or 'funded'}",
        f"# Run: {recon.get('run_id') or ''}",
        f"# Total records in dataset: {recon.get('total_records')}",
        f"# Total balance in dataset: {_gbp(recon.get('total_balance'))}",
        f"# Records included in this result: {recon.get('records_included')}",
        f"# Balance included in this result: {_gbp(recon.get('balance_included'))}",
        f"# Records excluded due to missing dimensions/measures: {recon.get('records_excluded_missing')}",
        f"# Balance excluded due to missing dimensions/measures: {_gbp(recon.get('balance_excluded_missing'))}",
        f"# Coverage by balance: {recon.get('coverage_by_balance_pct')}%",
        f"# Filters applied: {recon.get('filters') or 'none'}",
    ]
    return lines


def result_csv_bytes(query_result: MIQueryResult,
                     include_reconciliation: bool = True) -> bytes:
    """CSV of the result table, with a reconciliation/coverage footer appended so a
    downloaded total can always be tied back to the funded-book snapshot. The data
    rows are unchanged (the table/chart total == the CSV data total)."""
    if query_result is None or query_result.data is None:
        return b""
    csv_text = query_result.data.to_csv(index=False)
    if include_reconciliation:
        recon = (query_result.metadata or {}).get("reconciliation")
        footer = _reconciliation_footer_lines(recon)
        if footer:
            csv_text = csv_text.rstrip("\n") + "\n\n" + "\n".join(footer) + "\n"
    return csv_text.encode("utf-8")


def chart_html_str(chart_result: Optional[MIChartResult]) -> Optional[str]:
    if chart_result is None:
        return None
    return chart_result.to_html(include_plotlyjs="cdn")


def spec_json_str(spec) -> str:
    if isinstance(spec, MIQuerySpec):
        return spec.to_json()
    return json.dumps(spec, indent=2, default=str)


def metadata_json_str(workflow_result: Dict[str, Any]) -> str:
    payload = {
        "ok": workflow_result.get("ok"),
        "parser_mode": workflow_result.get("parser_mode"),
        "interpreted": workflow_result.get("interpreted"),
        "validation": workflow_result.get("validation"),
        "parse_metadata": workflow_result.get("parse_metadata"),
        "warnings": workflow_result.get("warnings"),
        "metadata": workflow_result.get("metadata"),
    }
    return json.dumps(payload, indent=2, default=str)
