"""TYPE-CONFORMANCE SWEEP over the 252-case calibration bank.

Two questions, kept apart:

  CONFORMANCE  does the type the question asks for match the type the answer
               returned? Same classifier as the 30/80 banks.
  EXPOSURE     is the case's correctness independently declared at all, or does
               it rest only on the evaluator finding nothing to complain about?
               A case with expected_status "answer" and no expected_metric has
               no type pinned: any measure would pass.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
sys.path.insert(0, "/home/user/trakt"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from type_sweep import asked_type, metric_type, asked_breakdown, _OK, ANY
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.mi_query_harness import build_fixture
from mi_agent import mi_calibration as CAL

SEM = load_mi_semantics("/home/user/trakt/mi_agent/mi_semantics_field_registry.yaml")
DF = build_fixture()
cases = CAL.load_bank()

conf, exposure, declared_mismatch = [], [], []
for case in cases:
    q = case["question"]
    status = case.get("expected_status")
    exp_metric = case.get("expected_metric")
    exp_art = case.get("expected_artifact_type")
    # --- EXPOSURE: what does this case actually pin? --------------------- #
    if status == "answer" and not exp_metric and not case.get("expected_metrics"):
        if case.get("execution") == "parse_only":
            reason = "parse_only: validated at parse level, never executed"
        elif asked_type(q) == "COUNT":
            reason = ("COUNT question: a count carries no measure, so the schema "
                      "has no field that can pin the ANSWER TYPE")
        else:
            reason = f"no measure pinned (artifact={exp_art})"
        exposure.append((case["id"], q, reason))
    # --- DECLARED vs ASKED: is the declared expectation itself the right
    #     type for the question? A wrong expectation certifies a wrong answer.
    want = asked_type(q)
    if exp_metric:
        got_declared = metric_type(exp_metric, None)
        allowed = _OK.get(want)
        if allowed is not None and got_declared not in allowed and want != ANY:
            declared_mismatch.append((case["id"], q,
                f"question asks {want}; the EXPECTATION declares {got_declared} "
                f"({exp_metric})"))
    # --- CONFORMANCE: what did it actually return? ----------------------- #
    r = CAL.evaluate_case(case, DF, SEM)
    # A parse_only case is validated at the parse level and never executed, so
    # it carries no observed measure to type-check. Counting those as
    # conformance findings measured the sweep, not the product.
    if (status != "answer" or r.status != "answer"
            or case.get("execution") == "parse_only"):
        continue
    metric = r.observed_metric
    # The evaluator records the observed MEASURE, not the aggregation. A COUNT
    # answer carries no measure, so "no measure + a kpi/bar artifact" is how a
    # count presents here.
    agg = "count" if (metric is None and r.observed_artifacts) else None
    got = metric_type(metric, agg)
    allowed = _OK.get(want)
    if allowed is not None and got not in allowed:
        conf.append((case["id"], q, f"asked {want}, returned {got} "
                                    f"(observed_metric={metric} "
                                    f"artifacts={r.observed_artifacts} "
                                    f"dims={r.observed_dims})"))

print(f"cases: {len(cases)}")
print(f"\n--- CONFORMANCE findings (returned type != asked type): {len(conf)}")
for i, q, d in conf:
    print(f"  {i:12s} {q[:56]:58s} {d}")
print(f"\n--- DECLARED-EXPECTATION mismatches (the expectation itself is the wrong type): {len(declared_mismatch)}")
for i, q, d in declared_mismatch:
    print(f"  {i:12s} {q[:56]:58s} {d}")
print(f"\n--- EXPOSURE (status=answer with no expected_metric pinned): {len(exposure)}")
for i, q, d in exposure:
    print(f"  {i:12s} {q[:56]:58s} {d}")
