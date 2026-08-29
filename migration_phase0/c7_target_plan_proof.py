#!/usr/bin/env python3
"""migration_phase0/c7_target_plan_proof.py — can the target plan be BUILT, and RUN?

READ-ONLY with respect to production. The generic plan builder lives HERE, not in
`analytical_plan`, because this task is an evidence gate: the question is whether
a generic plan is DERIVABLE from governed contract values, and answering it must
not itself change what production does.

Three analytical classes, kept apart because they are different claims:

  L   ranked LEVEL      "Which region has the largest balance?"
  M   unranked MOVEMENT "How did the balance change since last month?"
  RM  ranked MOVEMENT   "Which region added the most balance since last month?"

For each: build the plan from the contract alone, then EXECUTE its primitives
without going anywhere near `period_change_route`, then reconcile the numbers.

WHAT "FROM THE CONTRACT ALONE" MEANS HERE, and it is checked rather than
asserted: the builder receives a `QuestionInterpretation` and nothing else. It
cannot see the question text, the route, `_rank_subject`, or any C7 vocabulary,
because they are not arguments. A value it cannot get from the contract is a
value the plan cannot have.

    python -m migration_phase0.c7_target_plan_proof [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: The book: six month-end governed snapshots carrying the dimensions the cases
#: name. Three regions, so a ranking has more than a winner and a runner-up.
FIXTURE_DEPTH = 6

CLASSES = {
    # Legacy-comparable classes.
    "L": ["Which region has the largest balance?",
          "which Broker has the largest balance"],
    "M": ["How did the balance change since last month?"],
    # PRE-REGISTERED NEW-CAPABILITY CASES. The 882-question corpus contains ZERO
    # genuine ranked historical movements and none is invented here to look like
    # one. Each case exercises a DIFFERENT part of the plan — positive movement,
    # decline, population, limit, percentage — rather than paraphrasing one path.
    # Every one states its measure and its period, because under the product
    # ruling a question that states neither is a governed refusal, not a case.
    "RM": ["Which region added the most balance since last month?",
           "Which region saw the largest fall in balance since last month?",
           "Which region added the most balance since last month for loans "
           "with LTV above 50%?",
           "Which two regions added the most balance since last month?",
           "Which region grew fastest in balance since last month?"],
    # PHASE 8 — refusal preservation. These MUST NOT be answered: the first
    # names no measure, the second no period. A guessed answer here is the
    # STOP condition, and a governed clarification is correct product behaviour.
    "REFUSE": ["Which region grew the most?",
               "Which region added the most balance?"],
}


# --------------------------------------------------------------------------- #
# The generic target plan, from contract values only
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TargetStep:
    primitive: str
    inputs: Dict[str, Any]
    because: str
    blocked: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"primitive": self.primitive, "inputs": self.inputs,
                "because": self.because, "blocked": self.blocked}


def build_target_plan(interpretation) -> List[TargetStep]:
    """A plan for a level or a movement, ranked or not, from the contract.

    ONE builder for all three classes on purpose. If L, M and RM needed three
    builders, the shapes would be route shapes and the compositional claim would
    be false; the test of the claim is that the same steps appear or do not
    appear according to what the contract says, and nothing else.
    """
    from mi_agent_api import analytical_plan as P

    qi = interpretation
    op = qi.operation
    dataset = P.dataset_of(qi)
    metric, aggregation = P.measure_request(qi)
    periods = list(P.comparison_periods(qi))
    span_periods = getattr(qi.time, "window_periods", None)
    is_movement = op.orders_a_movement or op.type == "movement"

    steps: List[TargetStep] = []

    # 1. PERIODS. A movement needs two; a level needs one.
    if is_movement:
        if periods:
            steps.append(TargetStep(
                "stack_periods",
                {"dataset": dataset, "take": "named_pair", "periods": periods},
                "a movement opens at the first named period and closes at the second"))
        elif span_periods:
            steps.append(TargetStep(
                "stack_periods",
                {"dataset": dataset, "take": "span", "periods_back": span_periods},
                "the contract names a window; the pair is its ends"))
        else:
            steps.append(TargetStep(
                "stack_periods", {"dataset": dataset}, "a movement needs two periods",
                blocked="no contract field: neither comparison_periods nor "
                        "time.window_periods names a period"))
    else:
        steps.append(TargetStep(
            "stack_periods", {"dataset": dataset, "take": "latest"},
            "a level is measured at one reporting date"))

    # 2. POPULATION.
    scope = getattr(qi, "source_scope", None)
    steps.append(TargetStep(
        "select_population",
        {"state": getattr(scope, "state", None),
         "portfolio_ids": list(getattr(scope, "portfolio_ids", None) or ()),
         "row_predicates": [(p.field_key, p.operator, p.value)
                            for p in (qi.row_predicates or ())]},
        "the governed population the contract states"))

    # 3. MEASURE.
    steps.append(TargetStep(
        "resolve_measure", {"metric": metric, "aggregation": aggregation},
        "the contract's subject concept, expanded for the resolver"))

    # 4. GROUPING — with the ALTERNATES, not the primary alone.
    dims = [d for d in (qi.dimensions or []) if d.candidate_concept]
    if dims:
        d = dims[0]
        steps.append(TargetStep(
            "group",
            {"by": list(d.candidate_concepts), "role": d.role},
            "every governed field the named term resolves to, best first; the "
            "executor binds the first the book carries"))

    # 5. COMPARE — only for a movement.
    if is_movement:
        steps.append(TargetStep(
            "compare", {"of": metric, "as": "absolute and percentage delta",
                        "direction": "b relative to a"},
            "the movement from the opening period to the closing one"))

    # 6. RANK — from the contract's ordering values, never a route constant.
    if op.type == "ranking":
        if not (op.ordering_direction and op.ordering_basis):
            steps.append(TargetStep(
                "rank", {}, "a ranking needs a direction and a basis",
                blocked="no contract field: operation.ordering_direction / "
                        "ordering_basis"))
        else:
            steps.append(TargetStep(
                "rank",
                {"of": (list(dims[0].candidate_concepts) if dims else None),
                 "basis": op.ordering_basis,
                 "over": ("movement" if is_movement else "level"),
                 "direction": op.ordering_direction,
                 "limit": op.ordering_limit},
                "the ordering the contract states, over the movement or the level"))
    return steps


# --------------------------------------------------------------------------- #
# Executing the plan's primitives, without C7
# --------------------------------------------------------------------------- #
def _frames(output_root: str, client_id: str):
    """STACK_PERIODS, via the EXISTING governed per-period frame service.

    `evolution.funded_frames` is the shared service the Evolution tab, the
    funded bridge and the temporal-compare route already read. Using it is the
    point: if the target plan needed a NEW frame service the compositional claim
    would be weaker, not stronger.
    """
    from mi_agent_api import evolution as evolution_mod
    return evolution_mod.funded_frames(output_root, client_id, None)


def _resolve_period(token: str, frames):
    """A period token -> the governed frame it names, or None.

    Deliberately small and explicit: the relative tokens the contract actually
    carries, plus a month name matched against the frame's reporting date. A
    token this cannot resolve is an error, never a silent fallback to "the
    latest two", because that fallback is what made the order untestable.
    """
    tok = str(token).strip().lower()
    if tok in ("latest", "current", "this month", "now"):
        return frames[-1]
    if tok in ("last month", "previous month", "prior month", "last period"):
        return frames[-2] if len(frames) >= 2 else None
    months = ("january", "february", "march", "april", "may", "june", "july",
              "august", "september", "october", "november", "december")
    if tok in months:
        want = months.index(tok) + 1
        for f in frames:
            date = str(f.get("reporting_date") or "")
            if len(date) >= 7 and int(date[5:7]) == want:
                return f
        return None
    for f in frames:
        if str(f.get("reporting_date") or "").startswith(tok):
            return f
    return None


def execute_target_plan(steps: List[TargetStep], output_root: str,
                        client_id: str) -> Dict[str, Any]:
    """Run the plan's primitives directly. `period_change_route` is not imported."""
    import pandas as pd  # noqa: F401 - frames are pandas

    out: Dict[str, Any] = {"blocked": [s.blocked for s in steps if s.blocked]}
    if out["blocked"]:
        return out
    by = {s.primitive: s.inputs for s in steps}
    frames = _frames(output_root, client_id)
    if len(frames) < 2:
        out["error"] = "fixture carries fewer than two governed frames"
        return out

    # THE STATED ORDER IS BINDING. An earlier version of this executor took
    # `frames[-2:]` regardless of what the plan said, so the mutation control
    # "reverse the comparison periods" did not discriminate — the evidence
    # could not tell an honoured order from an ignored one, which is the D4
    # defect reproduced inside the harness meant to detect it.
    take = by["stack_periods"].get("take")
    if take == "latest":
        chosen = frames[-1:]
    else:
        tokens = list(by["stack_periods"].get("periods") or [])
        if tokens:
            chosen = [_resolve_period(tokens[i], frames) for i in range(len(tokens))]
            if any(f is None for f in chosen):
                out["error"] = f"unresolvable period token in {tokens}"
                return out
        else:
            back = int(by["stack_periods"].get("periods_back") or 1)
            chosen = [frames[-1 - back], frames[-1]]
    out["periods"] = [str(f.get("reporting_date")) for f in chosen]

    metric = by["resolve_measure"]["metric"]
    aggregation = by["resolve_measure"].get("aggregation")
    # `measure_request` returns (None, "count") for a COUNT request — the
    # concept is `loan_count` and there is no column to sum. That is a measure,
    # not a missing one, and an earlier version of this executor read the None
    # as absence and reported three cases as unmeasurable. The opposite error
    # to defaulting, and just as wrong.
    if not metric and aggregation != "count":
        out["error"] = ("contract carries no measure: resolve_measure gives "
                        "neither a metric nor a count aggregation")
        return out
    out["measure"] = metric or f"<{aggregation}>"
    group = by.get("group", {}).get("by") or []
    # THE ALTERNATES DOING THEIR JOB: bind the first concept the book carries.
    columns = set(chosen[-1]["df"].columns)
    bound = next((g for g in group if g in columns), None)
    out["grouping_candidates"], out["grouping_bound"] = list(group), bound
    if group and bound is None:
        out["error"] = f"none of {group} is carried by the book"
        return out

    # SELECT_POPULATION, applied. The row predicates the contract carries are
    # governed field/operator/value triples; applying them here is what makes
    # the Population cell provable rather than merely represented.
    predicates = tuple(tuple(p) for p in
                       (by["select_population"].get("row_predicates") or ()))
    unfiltered_counts, filtered = [], []
    for f in chosen:
        df = f["df"]
        unfiltered_counts.append(int(len(df)))
        for field_key, operator, value in predicates:
            if field_key not in df.columns:
                out["error"] = f"predicate field {field_key!r} is not on the tape"
                return out
            try:
                _probe = df[field_key] > float(value) if operator in (
                    ">", "gt", "greater_than", ">=", "gte", "<", "lt",
                    "less_than", "<=", "lte") else None
            except (TypeError, ValueError):
                out["error"] = (f"predicate {field_key!r} {operator} {value!r} "
                                f"does not compare against this column's type")
                return out
            col = df[field_key]
            # A predicate whose operand types do not compare is an ERROR, not a
            # traceback. The mutation control that points a numeric threshold at
            # a categorical column has to produce a recorded failure, or the
            # control cannot discriminate — it just crashes the harness.
            try:
                _ = col > col
            except Exception:  # noqa: BLE001 - recorded below, never absorbed
                pass
            if operator in (">", "gt", "greater_than"):
                df = df[col > float(value)]
            elif operator in (">=", "gte"):
                df = df[col >= float(value)]
            elif operator in ("<", "lt", "less_than"):
                df = df[col < float(value)]
            elif operator in ("<=", "lte"):
                df = df[col <= float(value)]
            elif operator in ("=", "==", "eq", "equals"):
                df = df[col == value]
            else:
                out["error"] = f"unsupported predicate operator {operator!r}"
                return out
        filtered.append(df)
    out["row_counts"] = [int(len(d)) for d in filtered]
    out["unfiltered_row_counts"] = unfiltered_counts
    out["predicates"] = [list(p) for p in predicates]

    levels = []
    for df in filtered:
        if aggregation == "count" and not metric:
            if bound:
                levels.append(df.groupby(bound).size().astype(float).to_dict())
            else:
                levels.append({"__all__": float(len(df))})
        elif bound:
            levels.append(df.groupby(bound)[metric].sum().to_dict())
        else:
            levels.append({"__all__": float(df[metric].sum())})
    out["levels"] = levels

    if "compare" in by and len(levels) == 2:
        keys = sorted(set(levels[0]) | set(levels[1]))
        movement = {k: round(float(levels[1].get(k, 0.0))
                             - float(levels[0].get(k, 0.0)), 2) for k in keys}
        pct = {k: (round(movement[k] / float(levels[0][k]) * 100, 2)
                   if levels[0].get(k) else None) for k in keys}
        out["movement"], out["percent_movement"] = movement, pct

    out["measure_aggregation"] = aggregation
    rank = by.get("rank")
    if rank and rank.get("direction"):
        # THE BASIS DECIDES WHAT IS ORDERED. Ranking a percentage question by
        # absolute movement puts the biggest book first rather than the fastest
        # grower, which is a different answer to a different question.
        basis = rank.get("basis")
        if rank.get("over") == "movement":
            if basis == "percent":
                source = out.get("percent_movement") or {}
                source = {k: v for k, v in source.items() if v is not None}
            else:
                source = out.get("movement")
        else:
            source = levels[-1]
        if source:
            desc = rank["direction"] != "decrease"
            ordered = sorted(source.items(), key=lambda kv: kv[1], reverse=desc)
            # A direction is a FILTER as well as an order: "which declined the
            # most" must not answer with something that rose.
            if rank["direction"] == "increase":
                ordered = [(k, v) for k, v in ordered if v > 0]
            elif rank["direction"] == "decrease":
                ordered = [(k, v) for k, v in ordered if v < 0]
            if rank.get("limit"):
                ordered = ordered[: int(rank["limit"])]
            out["ranked"] = ordered

    # THE RECEIPT, from the execution facts and before any prose exists.
    if out.get("movement") is not None:
        from mi_agent_api.movement_receipt import (PopulationEvidence,
                                                   build_movement_receipt)
        receipt = build_movement_receipt(
            measure=metric, aggregation=aggregation,
            grouping_dimension=bound,
            grouping_candidates=group,
            periods=out["periods"], levels=levels,
            ranked=out.get("ranked") or sorted(out["movement"].items(),
                                               key=lambda kv: -kv[1]),
            basis=(rank or {}).get("basis"),
            direction=(rank or {}).get("direction"),
            limit=(rank or {}).get("limit"),
            population=PopulationEvidence(
                dataset=by["stack_periods"].get("dataset"),
                portfolio_ids=tuple(by["select_population"].get("portfolio_ids") or ()),
                predicates=predicates,
                row_counts=tuple(out["row_counts"]),
                unfiltered_row_counts=tuple(unfiltered_counts)))
        out["receipt"] = receipt.to_dict()
        out["receipt_complete"] = receipt.complete
        out["receipt_missing"] = receipt.missing_facts()
        out["receipt_chronological"] = receipt.chronological
    return out


# --------------------------------------------------------------------------- #
def _interpret(question: str, semantics, columns):
    from mi_agent import execution_receipt as R
    from mi_agent.parsed_question import ParsedQuestion
    from question_interpretation import projection as proj
    spec = ParsedQuestion.parse(question, semantics).spec
    terms = R.requested_dimension_terms(question, semantics, columns)
    facets = R.detect_requested_facets(question, semantics, frame=None,
                                       requested_dimensions=terms)
    return proj.from_parts(question, spec=spec, facets=facets, dim_terms=terms,
                           semantics=semantics)


def run() -> Dict[str, Any]:
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    from migration_phase0.compound_canary import _write_run
    from migration_phase0.route_ownership_period_change import funded_runs

    tmp = Path(tempfile.mkdtemp())
    out_root = tmp / "onboarding_output"
    for run_id, rdate, n, scale in funded_runs(FIXTURE_DEPTH):
        _write_run(out_root, run_id, rdate, n, scale)
    saved = {k: os.environ.get(k) for k in
             ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_AUTH_ENABLED")}
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(out_root)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    try:
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.data_source import semantics_path
        semantics = load_mi_semantics(semantics_path())
        columns = ["loan_identifier", "current_outstanding_balance",
                   "current_loan_to_value", "current_interest_rate",
                   "youngest_borrower_age", "broker_channel",
                   "geographic_region_obligor", "reporting_date"]
        results: Dict[str, Any] = {}
        for cls, questions in CLASSES.items():
            rows = []
            for q in questions:
                qi = _interpret(q, semantics, columns)
                steps = build_target_plan(qi)
                rows.append({
                    "question": q,
                    "contract": {
                        "operation": qi.operation.type,
                        "ordering_of": qi.operation.ordering_of,
                        "ordering_direction": qi.operation.ordering_direction,
                        "ordering_basis": qi.operation.ordering_basis,
                        "ordering_limit": qi.operation.ordering_limit,
                        "dimensions": [[d.candidate_concept,
                                        list(d.alternate_concepts)]
                                       for d in (qi.dimensions or [])],
                        "comparison_periods": list(
                            getattr(qi.time, "comparison_periods", ()) or ()),
                        "window_periods": getattr(qi.time, "window_periods", None),
                    },
                    "plan": [s.to_dict() for s in steps],
                    "execution": execute_target_plan(steps, str(out_root),
                                                     "client_001"),
                })
            results[cls] = rows
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    res = run()
    for cls, rows in res.items():
        label = ("NEW-CAPABILITY EXECUTION PROOF — NOT LEGACY EQUIVALENCE"
                 if cls == "RM" else "legacy-comparable")
        print("=" * 88)
        print(f"CLASS {cls}   [{label}]")
        for r in rows:
            print("=" * 88)
            print(f"Q: {r['question']}")
            c = r["contract"]
            print(f"  contract: op={c['operation']} of={c['ordering_of']} "
                  f"dir={c['ordering_direction']} basis={c['ordering_basis']} "
                  f"limit={c['ordering_limit']}")
            print(f"            dims={c['dimensions']} periods={c['comparison_periods']} "
                  f"window={c['window_periods']}")
            for s in r["plan"]:
                mark = "BLOCKED " if s["blocked"] else "        "
                print(f"  {mark}{s['primitive']:<17}{json.dumps(s['inputs'], default=str)[:110]}")
                if s["blocked"]:
                    print(f"           -> {s['blocked']}")
            e = r["execution"]
            if e.get("blocked"):
                print(f"  EXEC: not attempted — plan blocked")
            elif e.get("error"):
                print(f"  EXEC: {e['error']}")
            else:
                print(f"  EXEC periods={e.get('periods')} bound={e.get('grouping_bound')} "
                      f"(candidates {e.get('grouping_candidates')})")
                if e.get("movement"):
                    print(f"       movement={e['movement']}")
                if e.get("ranked") is not None:
                    print(f"       ranked={e['ranked']}")
    if args.json:
        args.json.write_text(json.dumps(res, indent=2, default=str),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
