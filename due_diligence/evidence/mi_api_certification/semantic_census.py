"""Semantic census — what the deployed interpretation machinery DECIDES about
each question, before any calculation runs.

WHY A CENSUS AND NOT A SCORE. The recovery sprint changes who owns a semantic
decision, not what any engine computes. A score says whether an answer moved;
a census says WHICH DECISION moved — measure, aggregation, dataset, dimension,
predicate, period pair, time axis, winning capability — so every movement can
be classified as one of the permitted kinds or flagged as a STOP.

It runs IN PROCESS on the platform canonical schema (the local demonstration
book supplies columns and category values; no production call is made) and
records the parse, the interpretation contract, the registry's ordered
candidates and every temporal reader's verdict. It executes nothing.

    python -m ...semantic_census take  --bank BANK.json --out census.json
    python -m ...semantic_census diff  BEFORE.json AFTER.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

#: The decisions compared by `diff`. Prose is deliberately absent.
SEMANTIC_FIELDS = (
    "dataset", "metric", "aggregation", "weight_field", "dimensions", "filters",
    "unavailable_filters", "chart_type", "intent", "temporal_mode",
    "compare_periods", "bridge", "forecast", "risk",
    "qi_subject", "qi_subject_provenance", "qi_dimensions", "qi_predicates",
    "qi_window_periods", "qi_comparison_periods", "qi_dataset", "qi_scope",
    "span", "unit", "explicit_periods", "relative_mode", "time_axis",
    "movement_evidence", "intent_families", "intent_requirements",
    "first_candidate", "candidates",
)


def _env() -> None:
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"


def take(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _env()
    from mi_agent import execution_receipt as R
    from mi_agent import llm_query_parser as P
    from mi_agent import period_request as PR
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.parsed_question import ParsedQuestion
    from mi_agent.period_change import recognition as PCR
    from mi_agent_api import chat_routing  # noqa: F401 - populates REGISTRY
    from mi_agent_api import mi_service, portfolio_context, workspace
    from mi_agent_api.data_source import get_dataframe, semantics_path
    from mi_agent_api.recogniser_registry import REGISTRY, RouteRequest
    from mi_workflows.analytical import intent as I
    from question_interpretation import lexical as L
    from question_interpretation.projection import from_parts

    df = get_dataframe()
    semantics = load_mi_semantics(semantics_path())
    columns = set(df.columns)
    values = mi_service._book_values(df, semantics)
    geography = P.bind_geography(mi_service._resolve_geography(None, None, df))
    try:
        registry = portfolio_context.build_registry(df)
    except Exception:  # noqa: BLE001
        registry = None

    rows: List[Dict[str, Any]] = []
    for q in questions:
        text = q["question"]
        parsed = ParsedQuestion.parse(text, semantics, geography=geography,
                                      available_columns=columns,
                                      available_values=values, llm_enabled=False)
        spec = parsed.spec
        dataset = workspace.resolve_dataset(text)
        dim_terms = R.requested_dimension_terms(text, semantics, list(columns))
        facets = R.detect_requested_facets(text, semantics, frame=df,
                                           requested_dimensions=dim_terms)
        qi = from_parts(text, spec=spec, facets=list(facets), dim_terms=dim_terms,
                        semantics=semantics, registry=registry,
                        available_values=values)
        request = RouteRequest(
            question=text, spec=spec, spec_dict=spec.to_dict(), semantics=semantics,
            view=dataset, client_id="census", run_id=None, portfolio_id=None,
            available_values=values, interpretation_provider=lambda qi=qi: qi,
            parse_meta=parsed.meta, semantics_context=parsed.semantics_context)
        try:
            cands = [(rec.name, round(v.confidence, 2), v.reason)
                     for rec, v in REGISTRY.candidates(request)]
        except Exception as exc:  # noqa: BLE001
            cands = [("<error>", 0.0, repr(exc))]
        reading = I.classify(text, spec=spec)
        span = PR.requested_span(text)
        aspect = L.temporal_aspect(text)
        subject = getattr(qi, "subject", None)
        time = getattr(qi, "time", None)
        rows.append({
            "question_id": q.get("question_id"), "case": q.get("canonical_case_id"),
            "variant": q.get("variant_id"), "question": text,
            "dataset": dataset,
            "metric": spec.metric, "aggregation": spec.aggregation,
            "weight_field": getattr(spec, "weight_field", None),
            "dimensions": sorted(set([*(getattr(spec, "dimensions", None) or []),
                                      *([spec.dimension] if getattr(spec, "dimension", None) else [])])),
            "filters": {k: (v if isinstance(v, (str, int, float)) else json.loads(json.dumps(v, default=str)))
                        for k, v in (spec.filters or {}).items()},
            "unavailable_filters": list(getattr(spec, "unavailable_filters", None) or []),
            "chart_type": spec.chart_type, "intent": spec.intent,
            "temporal_mode": getattr(spec, "temporal_mode", None),
            "compare_periods": list(getattr(spec, "compare_periods", None) or []),
            "bridge": bool(getattr(spec, "bridge_query", False)),
            "forecast": getattr(spec, "forecast_mode", None),
            "risk": bool(getattr(spec, "risk_limit_query", False)),
            "parse_note": parsed.note,
            "qi_subject": getattr(subject, "candidate_concept", None) or getattr(subject, "raw_text", None),
            "qi_subject_provenance": getattr(subject, "provenance", None),
            "qi_dimensions": [getattr(d, "candidate_concept", None) or getattr(d, "raw_text", None)
                              for d in (getattr(qi, "dimensions", None) or [])],
            "qi_predicates": [(getattr(p, "field_key", None), getattr(p, "operator", None),
                               str(getattr(p, "value", None)))
                              for p in (getattr(qi, "row_predicates", None) or [])],
            "qi_window_periods": getattr(time, "window_periods", None),
            "qi_comparison_periods": list(getattr(time, "comparison_periods", None) or []),
            "qi_dataset": getattr(getattr(qi, "dataset", None), "dataset", None),
            "qi_scope": getattr(getattr(qi, "source_scope", None), "scope", None),
            "span": (span.label, span.periods, span.governed) if span else None,
            "unit": PR.requested_unit(text),
            "explicit_periods": list(PCR._explicit_periods(text, spec)),
            "relative_mode": PCR._relative_mode(text),
            "time_axis": L.time_axis_request(text),
            "movement_evidence": list(aspect.evidence),
            "intent_families": list(reading.families),
            "intent_requirements": list(reading.requirements),
            "first_candidate": cands[0][0] if cands else "generic",
            "candidates": cands,
        })
    return rows


#: The movements the sprint permits, by the field that moved. Anything not
#: listed is a STOP.
def classify_movement(before: Dict[str, Any], after: Dict[str, Any],
                      moved: List[str]) -> str:
    if not moved:
        return "unchanged"
    if set(moved) <= {"candidates"}:
        return "candidate_order_only"
    if before["first_candidate"] != after["first_candidate"]:
        return "route_moved"
    if "metric" in moved or "aggregation" in moved or "qi_subject" in moved:
        return "measure_moved"
    if any(f in moved for f in ("span", "explicit_periods", "relative_mode",
                                "qi_window_periods", "qi_comparison_periods",
                                "compare_periods")):
        return "period_moved"
    if "unavailable_filters" in moved:
        return "unknown_category_moved"
    if any(f in moved for f in ("filters", "dimensions", "qi_predicates", "qi_dimensions")):
        return "population_or_axis_moved"
    if "dataset" in moved or "qi_dataset" in moved:
        return "dataset_moved"
    return "other:" + ",".join(moved)


def diff(before: List[Dict[str, Any]], after: List[Dict[str, Any]]) -> Dict[str, Any]:
    b = {r["question_id"] or r["question"]: r for r in before}
    a = {r["question_id"] or r["question"]: r for r in after}
    out: List[Dict[str, Any]] = []
    for key in b:
        if key not in a:
            continue
        moved = [f for f in SEMANTIC_FIELDS
                 if json.dumps(b[key].get(f), sort_keys=True, default=str)
                 != json.dumps(a[key].get(f), sort_keys=True, default=str)]
        kind = classify_movement(b[key], a[key], moved)
        if kind != "unchanged":
            out.append({"question_id": key, "case": b[key].get("case"),
                        "question": b[key]["question"], "kind": kind, "moved": moved,
                        "before": {f: b[key].get(f) for f in moved},
                        "after": {f: a[key].get(f) for f in moved}})
    summary: Dict[str, int] = {}
    for m in out:
        summary[m["kind"]] = summary.get(m["kind"], 0) + 1
    return {"compared": len([k for k in b if k in a]), "moved": len(out),
            "by_kind": summary, "movements": out}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("take")
    t.add_argument("--bank", required=True)
    t.add_argument("--out", required=True)
    d = sub.add_parser("diff")
    d.add_argument("before")
    d.add_argument("after")
    d.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.cmd == "take":
        doc = json.loads(Path(args.bank).read_text(encoding="utf-8"))
        questions = doc.get("questions") if isinstance(doc, dict) else doc
        if questions and isinstance(questions[0], str):
            questions = [{"question": q, "question_id": f"Q{i:04d}"}
                         for i, q in enumerate(questions, start=1)]
        rows = take(questions)
        Path(args.out).write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
        print(f"census: {len(rows)} questions -> {args.out}")
        return 0
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    result = diff(before, after)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"compared {result['compared']}, moved {result['moved']}: {result['by_kind']}")
    for m in result["movements"]:
        print(f"  {m['question_id']} {m.get('case') or ''} [{m['kind']}] {m['question'][:80]}")
        for f in m["moved"]:
            if f == "candidates":
                continue
            print(f"      {f}: {m['before'][f]!r} -> {m['after'][f]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
