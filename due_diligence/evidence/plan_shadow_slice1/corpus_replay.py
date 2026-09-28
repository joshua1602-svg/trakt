#!/usr/bin/env python3
"""Gate 5: every recorded plan, replayed through the shadow adapter. No model calls.

WHY A REPLAY AND NOT THE 843-QUESTION CORPUS. The 843-question corpus is not
present at HEAD — it lives only on the quarantined deterministic branch, which
this slice is not authorised to merge or read into the product. What IS present
at HEAD is seven recorded benchmark runs, each holding 135 interpretations with
the model's raw payload, the compiled `GovernedQueryPlan` and its `plan_id`. The
most recent of those, `run8_135_signoff_2b00172.json`, is the signed-off run, so
it is the substitute corpus here. That is a smaller corpus, and the report says so
rather than presenting 135 as though it were 843.

ZERO LIVE CALLS, STRUCTURALLY. Stage 1 recompiles each recorded raw payload
through the real `DeterministicCompiler` and checks the resulting `plan_id`
against the recorded one. If every id matches, the plans replayed here are the
same plans the signed-off run produced, and no interpretation had to be bought
again. The harness asserts afterwards that no Anthropic client was ever imported.

THE CONTROL IS AN INDEPENDENT ORACLE, NOT THE OLD PATH. The recorded runs hold
interpretation outputs only — no legacy answer was ever stored beside them — so
an old-path-versus-new-path figure comparison is not available from recorded
evidence, and manufacturing one would need live calls on the legacy route. The
control used here is instead arithmetic written out longhand in this file over a
deterministic book, importing nothing from the product. That is a stronger
control than the old path for the figures, and a weaker one for disposition; both
are reported separately and neither is dressed up as the other.

    ONE DISCLOSED RULE. The plan carries a governed canonical VALUE
    (`erm_product_type == 'drawdown'`) and the book carries the display form
    (`'Drawdown'`). The production engine resolves the two through the governed
    registry. This control compares strings case-insensitively after trimming,
    written out below, and consults no product alias table. Without that rule the
    control reads 0 rows where the engine correctly reads 93 — a harness defect,
    found here and recorded rather than blamed on the engine.

Run: `python due_diligence/evidence/plan_shadow_slice1/corpus_replay.py`
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd                                                  # noqa: E402

from mi_agent import plan_runtime_adapter as adapter                  # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics            # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth           # noqa: E402

#: The signed-off recorded run. The other six runs have the same shape and can be
#: passed as argv[1] to replay any of them.
DEFAULT_SOURCE = (_REPO_ROOT / "mi_agent" / "interpretation_v2" / "evidence"
                  / "run8_135_signoff_2b00172.json")
EVIDENCE_DIR = Path(__file__).resolve().parent
REGISTRY = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

#: Plan statistic -> this harness's own aggregation, kept separate from the
#: adapter's `_AGGREGATION` so the two cannot agree by sharing a table.
_CONTROL_STATISTICS = ("sum", "count", "average", "weighted_average")

# Replay dispositions, distinct from the adapter's parity classifications.
OUTSIDE_REPLAY_FIXTURE = "OUTSIDE_REPLAY_FIXTURE"
GROUPED_CELL_PARITY = "GROUPED_CELL_PARITY"
GROUPED_CELL_DIFFERENCE = "GROUPED_CELL_DIFFERENCE"
CONTROL_NOT_COMPUTABLE = "CONTROL_NOT_COMPUTABLE"


# --------------------------------------------------------------------------- #
# stage 1 — is the replayed plan the plan that was signed off?
# --------------------------------------------------------------------------- #

def recompile_fidelity(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Recompile each recorded payload and compare `plan_id` with the record.

    Deterministic and free: `parse_candidate_intent` and `DeterministicCompiler`
    are pure, and the model's raw payload is already on disk.
    """
    from mi_agent.interpretation_v2.compiler import CompilerContext, compile_intent
    from mi_agent.interpretation_v2.intent import parse_candidate_intent

    context = CompilerContext()
    counts = Counter()
    divergent: List[Dict[str, str]] = []

    for record in records:
        question_id = str(record.get("question_id") or "")
        recorded_id = record.get("plan_id")
        try:
            intent = parse_candidate_intent(record.get("raw_payload"))
            compiled = compile_intent(intent, context)
        except Exception as exc:                                     # noqa: BLE001
            counts["REPLAY_FAILED"] += 1
            divergent.append({"question_id": question_id,
                              "detail": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        if not compiled.is_plan:
            # A refusal or clarification reproducing as one is still fidelity.
            if record.get("plan"):
                counts["PLAN_LOST_ON_REPLAY"] += 1
                divergent.append({"question_id": question_id,
                                  "detail": f"recorded a plan, replayed "
                                            f"{compiled.outcome}"})
            else:
                counts["NO_PLAN_BOTH_TIMES"] += 1
            continue
        if compiled.plan.plan_id == recorded_id:
            counts["PLAN_ID_IDENTICAL"] += 1
        else:
            counts["PLAN_ID_DIVERGED"] += 1
            divergent.append({"question_id": question_id,
                              "detail": f"{compiled.plan.plan_id} != {recorded_id}"})
    return {"counts": dict(counts), "divergent": divergent}


# --------------------------------------------------------------------------- #
# the independent control — longhand, product-free
# --------------------------------------------------------------------------- #

def _mask(frame: pd.DataFrame,
          predicates: Sequence[Tuple[str, str, Any]]) -> pd.Series:
    """A boolean mask, written out. Strings compare case-insensitively.

    The case rule is the one disclosed in the module docstring and it is the ONLY
    normalisation this control performs: no alias table, no synonym list, no
    bucket derivation.
    """
    mask = pd.Series(True, index=frame.index)
    for field, op, value in predicates:
        column = frame[field]
        # Folded whenever the PLAN's value is a string, not when the column
        # advertises a particular dtype: a book read through pandas' string dtype
        # reports `str` rather than `object`, and an earlier version of this guard
        # tested the dtype and so never folded at all — which made the control
        # read 0 rows where the engine correctly read 93.
        if isinstance(value, str):
            column = column.astype(str).str.strip().str.lower()
            value = value.strip().lower()
        if op in ("eq", "", "equals"):
            mask &= column == value
        elif op == "ne":
            mask &= column != value
        elif op == "gt":
            mask &= column > value
        elif op in ("ge", "gte"):
            mask &= column >= value
        elif op == "lt":
            mask &= column < value
        elif op in ("le", "lte"):
            mask &= column <= value
        elif op == "in":
            wanted = [str(v).strip().lower() if isinstance(v, str) else v
                      for v in list(value)]
            mask &= column.isin(wanted)
        else:
            # Fail closed. A comparator the control cannot reproduce must not
            # silently become "no restriction".
            raise ValueError(f"the control has no rule for comparator {op!r}")
    return mask


def _predicates(plan: Mapping[str, Any]) -> List[Tuple[str, str, Any]]:
    """Every governed predicate the plan authorises, through the ONE owner.

    This used to read the two `filters` slots itself, which made it a fourth
    reader of something `plan_runtime_adapter.plan_predicates` already owns —
    and the fourth reader drifted the moment slice 3 put an explicit
    Direct/Acquired role into `population.scope_predicates`. Two consequences,
    both of them this harness lying rather than the product failing: a scoped
    plan's `source_portfolio_type` was invisible to the fixture-gap check, so a
    book that simply lacks the column was reported as SHADOW_EXECUTION_ERROR
    instead of OUTSIDE_REPLAY_FIXTURE; and `control_scalar` — the INDEPENDENT
    oracle — would have computed a whole-book figure for a scoped plan and
    called the engine's correct scoped answer a divergence.

    An independent oracle must be independent about the CALCULATION, not about
    which predicates the plan states.
    """
    output = (tuple(plan.get("outputs") or ()) or ({},))[0]
    return [(f["canonical_field"], str(f.get("comparator") or "eq"), f.get("value"))
            for f in adapter.plan_predicates(plan, output)]


def _referenced_fields(plan: Mapping[str, Any]) -> set:
    output = (tuple(plan.get("outputs") or ()) or ({},))[0]
    measure = (tuple(output.get("measures") or ()) or ({},))[0]
    found = set()
    for key in ("canonical_field", "weight_field"):
        if measure.get(key):
            found.add(measure[key])
    for dimension in output.get("dimensions") or ():
        found.add(dimension["canonical_field"])
    for field, _, _ in _predicates(plan):
        found.add(field)
    return found


def control_scalar(book: pd.DataFrame,
                   plan: Mapping[str, Any]) -> Optional[float]:
    """The figure an ungrouped plan asks for, computed independently."""
    output = (tuple(plan.get("outputs") or ()) or ({},))[0]
    measure = (tuple(output.get("measures") or ()) or ({},))[0]
    statistic = str(measure.get("statistic") or "")
    rows = book.loc[_mask(book, _predicates(plan))]
    if statistic == "count":
        return float(len(rows))
    column = measure.get("canonical_field")
    if statistic == "sum":
        return float(rows[column].sum())
    if statistic == "average":
        return None if rows.empty else float(rows[column].mean())
    if statistic == "weighted_average":
        weight = measure["weight_field"]
        total_weight = rows[weight].sum()
        if not total_weight:
            return None
        return float((rows[column] * rows[weight]).sum() / total_weight)
    return None


def control_cells(book: pd.DataFrame, plan: Mapping[str, Any],
                  dimensions: Sequence[str]) -> Dict[Tuple[str, ...], float]:
    """`{group key: figure}` for a grouped plan, computed independently."""
    output = (tuple(plan.get("outputs") or ()) or ({},))[0]
    measure = (tuple(output.get("measures") or ()) or ({},))[0]
    statistic = str(measure.get("statistic") or "")
    column = measure.get("canonical_field")
    rows = book.loc[_mask(book, _predicates(plan))]
    cells: Dict[Tuple[str, ...], float] = {}
    for key, chunk in rows.groupby(list(dimensions), observed=True, dropna=False):
        key_tuple = tuple(str(k) for k in (key if isinstance(key, tuple) else (key,)))
        if statistic == "count":
            cells[key_tuple] = float(len(chunk))
        elif statistic == "sum":
            cells[key_tuple] = float(chunk[column].sum())
        elif statistic == "average":
            cells[key_tuple] = float(chunk[column].mean())
        elif statistic == "weighted_average":
            weight = measure["weight_field"]
            total_weight = chunk[weight].sum()
            cells[key_tuple] = (float((chunk[column] * chunk[weight]).sum()
                                      / total_weight) if total_weight
                                else float("nan"))
    return cells


# --------------------------------------------------------------------------- #
# stage 3 — the grouped half, compared cell for cell
# --------------------------------------------------------------------------- #

def compare_cells(book: pd.DataFrame, plan: Mapping[str, Any], semantics: Any,
                  dimensions: Sequence[str]) -> Tuple[str, str]:
    """Execute the adapter's own spec and check every cell against the control.

    `ShadowOutcome` carries no frame — deliberately, because a grouped execution
    has no scalar to serve — so the grouped comparison is done here, against the
    spec the adapter itself produces.
    """
    from mi_agent.mi_query_executor import execute_mi_query
    try:
        spec = adapter.spec_for_plan(plan)
        result = execute_mi_query(spec, book, semantics)
    except Exception as exc:                                         # noqa: BLE001
        return GROUPED_CELL_DIFFERENCE, f"{type(exc).__name__}: {exc}"[:200]

    frame = getattr(result, "data", None)
    if frame is None or getattr(frame, "empty", True):
        return GROUPED_CELL_DIFFERENCE, "the execution produced no rows"

    value_column = f"{spec.metric}_{spec.aggregation}"
    if value_column not in frame.columns:
        candidates = [c for c in frame.columns if c not in dimensions]
        if not candidates:
            return GROUPED_CELL_DIFFERENCE, "no value column in the result"
        value_column = candidates[0]

    produced = {tuple(str(row[d]) for d in dimensions): float(row[value_column])
                for _, row in frame.iterrows()}
    expected = control_cells(book, plan, dimensions)
    if set(produced) != set(expected):
        return (GROUPED_CELL_DIFFERENCE,
                f"groups differ: {len(produced)} produced, {len(expected)} "
                f"expected; only_produced={sorted(set(produced) - set(expected))[:4]}")
    wrong = [k for k in expected if abs(produced[k] - expected[k]) >= 0.01]
    if wrong:
        return (GROUPED_CELL_DIFFERENCE,
                f"{len(wrong)} of {len(expected)} cells differ, first {wrong[0]}: "
                f"produced={produced[wrong[0]]!r} control={expected[wrong[0]]!r}")
    return GROUPED_CELL_PARITY, f"{len(expected)} cells, all within 0.01"


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def replay(source: Path, *, ledger: Path, case_prefix: str = "",
           check_fidelity: bool = True) -> Dict[str, Any]:
    payload = json.loads(source.read_text())
    records = payload.get("results") or []
    semantics = load_mi_semantics(str(REGISTRY))
    book = truth.canonical_book()
    columns = set(book.columns)

    fidelity = (recompile_fidelity(records) if check_fidelity
                else {"counts": {"NOT_CHECKED": len(records)}, "divergent": [],
                      "why": "recompile fidelity is asserted only for the run "
                             "compiled at the current contract; an older run "
                             "diverges because the NORMAL FORM changed, which is "
                             "a recorded change, not a defect"})

    eligible: List[Mapping[str, Any]] = []
    ineligible = Counter()
    ineligible_cases: Dict[str, List[str]] = {}
    for record in records:
        ok, reason, _ = adapter.check_eligibility(record.get("plan"))
        if ok:
            eligible.append(record)
        else:
            ineligible[reason] += 1
            ineligible_cases.setdefault(reason, []).append(
                str(record.get("question_id")))

    os.environ[adapter.SHADOW_ENV_VAR] = adapter.SHADOW_ON
    os.environ[adapter.LEDGER_ENV_VAR] = str(ledger)

    classifications = Counter()
    grouped_verdicts = Counter()
    fixture_gaps = Counter()
    rows: List[Dict[str, Any]] = []
    try:
        for record in eligible:
            plan = record["plan"]
            question_id = case_prefix + str(record.get("question_id"))
            output = plan["outputs"][0]
            dimensions = [d["canonical_field"]
                          for d in (output.get("dimensions") or ())]

            missing = _referenced_fields(plan) - columns
            if missing:
                for field in sorted(missing):
                    fixture_gaps[field] += 1
                classifications[OUTSIDE_REPLAY_FIXTURE] += 1
                rows.append({"case_id": question_id,
                             "classification": OUTSIDE_REPLAY_FIXTURE,
                             "note": f"the replay book does not carry "
                                     f"{sorted(missing)}"})
                continue

            try:
                control_value = (None if dimensions
                                 else control_scalar(book, plan))
            except Exception as exc:                                 # noqa: BLE001
                classifications[CONTROL_NOT_COMPUTABLE] += 1
                rows.append({"case_id": question_id,
                             "classification": CONTROL_NOT_COMPUTABLE,
                             "note": f"{type(exc).__name__}: {exc}"[:200]})
                continue

            control = {"ok": True, "route": "independent_control_oracle"}
            if control_value is not None:
                control["value"] = control_value

            row = adapter.observe(result=control, frame=book,
                                  semantics=semantics, plan=plan,
                                  case_id=question_id)
            if row is None:
                classifications["NO_LEDGER_ROW"] += 1
                rows.append({"case_id": question_id,
                             "classification": "NO_LEDGER_ROW", "note": ""})
                continue
            classifications[row["classification"]] += 1
            entry = {"case_id": question_id,
                     "classification": row["classification"],
                     "note": row["note"],
                     "control_value": row["control_value"],
                     "new_value": row["new_value"],
                     "dimensions": dimensions}
            if dimensions:
                verdict, note = compare_cells(book, plan, semantics, dimensions)
                grouped_verdicts[verdict] += 1
                entry["grouped_verdict"] = verdict
                entry["grouped_note"] = note
            rows.append(entry)
    finally:
        os.environ.pop(adapter.SHADOW_ENV_VAR, None)
        os.environ.pop(adapter.LEDGER_ENV_VAR, None)

    return {
        "source": str(source.relative_to(_REPO_ROOT)),
        "benchmark": payload.get("benchmark"),
        "interpreter_version": payload.get("interpreter_version"),
        "total": len(records),
        "slice1_eligible": len(eligible),
        "eligible_cases": [str(r.get("question_id")) for r in eligible],
        "eligible_plan_ids": sorted({str(r.get("plan_id")) for r in eligible}),
        "not_eligible": sum(ineligible.values()),
        "not_eligible_by_reason": dict(ineligible.most_common()),
        "not_eligible_cases": {k: sorted(v) for k, v in ineligible_cases.items()},
        "plan_replay_fidelity": fidelity,
        "classifications": dict(classifications.most_common()),
        "grouped_verdicts": dict(grouped_verdicts.most_common()),
        "replay_fixture_field_gaps": dict(fixture_gaps.most_common()),
        "cases": rows,
    }


#: Runs replayed. A partial run is excluded: its unattempted questions carry no
#: plan and would count as NOT_A_PLAN, which would misreport the census rather
#: than widen it.
RUN_GLOB = "run?_135_*.json"
EXCLUDE = ("PARTIAL",)


def discover(evidence: Path) -> List[Path]:
    found = [p for p in sorted(evidence.glob(RUN_GLOB))
             if not any(token in p.name for token in EXCLUDE)]
    # The signed-off run leads, because its fidelity is the one that is checked.
    found.sort(key=lambda p: (p.resolve() != DEFAULT_SOURCE.resolve(), p.name))
    return found


def main(argv: Sequence[str]) -> int:
    sources = ([Path(argv[1])] if len(argv) > 1
               else discover(DEFAULT_SOURCE.parent))
    ledger = EVIDENCE_DIR / "shadow_ledger.jsonl"
    if ledger.exists():
        ledger.unlink()

    runs: List[Dict[str, Any]] = []
    aggregate = {"total": 0, "slice1_eligible": 0, "not_eligible": 0}
    reasons, classes, grouped, gaps = Counter(), Counter(), Counter(), Counter()
    distinct_questions, distinct_plans = set(), set()

    for source in sources:
        signed_off = source.resolve() == DEFAULT_SOURCE.resolve()
        report = replay(source, ledger=ledger,
                        case_prefix=f"{source.name.split('_')[0]}:",
                        check_fidelity=signed_off)
        report["signed_off_run"] = signed_off
        runs.append(report)
        aggregate["total"] += report["total"]
        aggregate["slice1_eligible"] += report["slice1_eligible"]
        aggregate["not_eligible"] += report["not_eligible"]
        reasons.update(report["not_eligible_by_reason"])
        classes.update(report["classifications"])
        grouped.update(report["grouped_verdicts"])
        gaps.update(report["replay_fixture_field_gaps"])
        distinct_questions.update(report["eligible_cases"])
        distinct_plans.update(report["eligible_plan_ids"])

    combined = {
        "what_this_is": "the recorded-plan substitute for the 843-question "
                        "corpus, which is not present at HEAD",
        "runs_replayed": [r["source"] for r in runs],
        "aggregate": dict(aggregate,
                          not_eligible_by_reason=dict(reasons.most_common()),
                          classifications=dict(classes.most_common()),
                          grouped_verdicts=dict(grouped.most_common()),
                          replay_fixture_field_gaps=dict(gaps.most_common())),
        "distinct_eligible_questions": len(distinct_questions),
        "distinct_eligible_plan_ids": len(distinct_plans),
        "ledger": str(ledger.relative_to(_REPO_ROOT)),
        "ledger_rows": (sum(1 for line in ledger.read_text().splitlines()
                            if line.strip()) if ledger.exists() else 0),
        "live_opus_calls": 0,
        "anthropic_client_imported": any(
            name == "anthropic" or name.startswith("anthropic.")
            for name in sys.modules),
        "opus_interpreter_module_imported":
            "mi_agent.interpretation_v2.opus_interpreter" in sys.modules,
        "runs": runs,
    }
    out = EVIDENCE_DIR / "corpus_replay.json"
    out.write_text(json.dumps(combined, indent=2, default=str) + "\n")

    def table(title: str, counts: Mapping[str, int]) -> None:
        if counts:
            print(title)
            for key, count in counts.items():
                print(f"    {key:<28} {count}")

    for report in runs:
        mark = "  (signed off)" if report["signed_off_run"] else ""
        print(f"\n=== {report['source']}{mark}")
        print(f"  TOTAL {report['total']}   SLICE1_ELIGIBLE "
              f"{report['slice1_eligible']}   NOT_ELIGIBLE "
              f"{report['not_eligible']}")
        table("  PLAN_REPLAY_FIDELITY", report["plan_replay_fidelity"]["counts"])
        table("  NOT_ELIGIBLE_BY_REASON", report["not_eligible_by_reason"])
        table("  CLASSIFICATIONS", report["classifications"])
        table("  GROUPED_CELL_VERDICTS", report["grouped_verdicts"])

    print("\n=== AGGREGATE OVER EVERY COMPLETE RECORDED RUN")
    print(f"  TOTAL                  {aggregate['total']}")
    print(f"  SLICE1_ELIGIBLE        {aggregate['slice1_eligible']}")
    print(f"  NOT_ELIGIBLE           {aggregate['not_eligible']}")
    table("  NOT_ELIGIBLE_BY_REASON", dict(reasons.most_common()))
    table("  CLASSIFICATIONS", dict(classes.most_common()))
    table("  GROUPED_CELL_VERDICTS", dict(grouped.most_common()))
    table("  REPLAY_FIXTURE_FIELD_GAPS", dict(gaps.most_common()))
    print(f"  DISTINCT_ELIGIBLE_QUESTIONS  "
          f"{combined['distinct_eligible_questions']}")
    print(f"  DISTINCT_ELIGIBLE_PLAN_IDS   "
          f"{combined['distinct_eligible_plan_ids']}")
    print(f"  LEDGER_ROWS            {combined['ledger_rows']}")
    print(f"  LIVE_OPUS_CALLS        {combined['live_opus_calls']} "
          f"(anthropic client imported: "
          f"{combined['anthropic_client_imported']}; interpreter MODULE imported "
          f"by the package __init__: "
          f"{combined['opus_interpreter_module_imported']})")
    print(f"  WRITTEN                {out.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
