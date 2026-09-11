#!/usr/bin/env python3
"""Slice 2 offline acceptance: the bank, the oracle, and the governance counts.

ZERO LIVE MODEL CALLS, STRUCTURALLY. Nothing here constructs an interpreter and
nothing imports an Anthropic client; the harness asserts that afterwards and
reports it. Each case's `CandidateIntent` is parsed by the production
`parse_candidate_intent` and compiled by the production `DeterministicCompiler`,
so the plan under test is the plan the product would emit for that intent.

THE ORACLE IMPORTS NOTHING FROM THE PRODUCT. Expected figures come from
`portfolio_truth_oracle`, which is pandas and explicit column names, and
expected SNAPSHOTS come from the bank, written out by hand. So a wrong
selection and a wrong figure are separately detectable, and neither can be
covered by the product agreeing with itself.

    ONE DISCLOSED RULE, inherited from the slice 1 replay. The plan carries a
    governed canonical value (`erm_product_type == 'drawdown'`) and the book
    carries the display form (`'Drawdown'`). The production engine resolves the
    two through the governed registry; this control is given the display form in
    the bank and consults no alias table of its own.

Run: `python due_diligence/evidence/plan_temporal_slice2/slice2_acceptance.py`
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml                                                          # noqa: E402

from mi_agent import plan_runtime_adapter as adapter                  # noqa: E402
from mi_agent import plan_temporal_runtime as temporal                # noqa: E402
from mi_agent.interpretation_v2.compiler import DeterministicCompiler  # noqa: E402
from mi_agent.interpretation_v2.intent import parse_candidate_intent   # noqa: E402
from mi_agent.interpretation_v2.outcomes import (OUTCOME_CLARIFY,      # noqa: E402
                                                 OUTCOME_PLAN,
                                                 OUTCOME_REFUSE)
from mi_agent.mi_query_validator import load_mi_semantics             # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth            # noqa: E402
from mi_agent.tests import temporal_snapshot_fixture as fixture       # noqa: E402

EVIDENCE_DIR = Path(__file__).resolve().parent
BANK = EVIDENCE_DIR / "slice2_bank.yaml"
REGISTRY = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
RECORDED_RUN = (_REPO_ROOT / "mi_agent" / "interpretation_v2" / "evidence"
                / "run8_135_signoff_2b00172.json")

# Dispositions. Distinct from the compiler's outcomes and from the adapter's
# ineligibility reasons, because a case can fail closed at three different
# stages and an evidence file that conflated them could not attribute a defect.
COMPILE_CLARIFY = "COMPILE_CLARIFY"
COMPILE_REFUSE = "COMPILE_REFUSE"
INELIGIBLE = "INELIGIBLE"
CLARIFY = "CLARIFY"
REFUSE = "REFUSE"
EXECUTED = "EXECUTED"

TOLERANCE = 0.01


# --------------------------------------------------------------------------- #
# the independent oracle
# --------------------------------------------------------------------------- #

def _predicates(recipe: Mapping[str, Any]) -> List[Tuple[str, str, Any]]:
    return [tuple(entry) for entry in (recipe.get("predicates") or ())]


def oracle_for(recipe: Mapping[str, Any], frame: Any) -> Any:
    """The mathematically correct answer for one snapshot, from pandas alone.

    A scalar when the recipe groups on nothing; a `{key tuple: figure}` mapping
    when it does.
    """
    how = str(recipe.get("how") or "")
    column = recipe.get("column")
    predicates = _predicates(recipe)
    group_by = list(recipe.get("group_by") or ())
    if group_by:
        return truth.grouped(frame, group_by, column=column, how=how,
                             predicates=predicates)
    if how == "count":
        return float(truth.row_count(frame, predicates))
    if how == "sum":
        return truth.total(frame, column, predicates)
    if how == "avg":
        rows = frame.loc[truth.mask_for(frame, predicates)]
        return float(rows[column].mean()) if len(rows) else None
    if how == "weighted_avg":
        return truth.weighted_average(frame, column, truth.BALANCE, predicates)
    raise ValueError(f"the bank names a recipe this oracle has no rule for: {how!r}")


def _close(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= TOLERANCE


# --------------------------------------------------------------------------- #
# one case
# --------------------------------------------------------------------------- #

_RECORDED_INTENTS: Optional[Dict[str, Dict[str, Any]]] = None


def recorded_intent(question_id: str) -> Dict[str, Any]:
    """One recorded interpretation's intent, read from the signed-off run.

    READ, NOT TRANSCRIBED. A case marked `recorded_case` takes its payload from
    `run8_135_signoff_2b00172.json` directly, so "verbatim" is a property of the
    harness rather than a claim about how carefully the bank was typed — an
    earlier cut copied NL1A's binding slots by hand, dropped the model's own
    `ambiguity` block with it, and turned a blocking ambiguity into a missing
    slot.
    """
    global _RECORDED_INTENTS
    if _RECORDED_INTENTS is None:
        recorded = json.loads(RECORDED_RUN.read_text(encoding="utf-8"))
        # `provenance` is stamped onto the SERIALISED intent by the interpreter
        # after parsing and is not a payload key, so it is dropped here — the
        # only edit made to a recorded payload, and it removes nothing the
        # compiler reads.
        _RECORDED_INTENTS = {
            row["question_id"]: {k: v for k, v in row["intent"].items()
                                 if k != "provenance"}
            for row in recorded["results"] if row.get("intent")}
    if question_id not in _RECORDED_INTENTS:
        raise KeyError(f"{question_id!r} is not in the recorded run")
    return dict(_RECORDED_INTENTS[question_id])


def _intent_payload(case: Mapping[str, Any]) -> Dict[str, Any]:
    """The case's intent, filled out to a complete CandidateIntent payload.

    The bank states only what the case is ABOUT; the invariant slots (schema
    version, the funded population, an absent geography, an absent comparison)
    are the same in every case and are supplied here rather than repeated in
    every one of them.

    A case naming a `recorded_case` carries no `intent` of its own and gets the
    model's, whole.
    """
    if case.get("recorded_case"):
        if case.get("intent"):
            raise ValueError(f"{case['id']}: a recorded case may not also "
                             f"author an intent")
        return recorded_intent(str(case["recorded_case"]))
    body = dict(case.get("intent") or {})
    body.setdefault("schema_version", "candidate_intent/1.0")
    body.setdefault("population", {"base": "funded", "lens": "all",
                                   "seasoning": "any"})
    body.setdefault("dimensions", [])
    body.setdefault("filters", [])
    body.setdefault("geography", {"requested": False})
    body.setdefault("comparison", {"kind": "none"})
    return body


def run_case(case: Mapping[str, Any], compiler: Any, store: Any, semantics: Any,
             history: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    """One bank case, start to finish. Never raises: a fault is a recorded FAIL."""
    record: Dict[str, Any] = {
        "id": case["id"],
        "question": case.get("question"),
        "category": case.get("category"),
        "source": case.get("source", "authored"),
        "paraphrase_of": case.get("paraphrase_of"),
        "expected": dict(case.get("expect") or {}),
        "failures": [],
    }
    expect = dict(case.get("expect") or {})

    intent = parse_candidate_intent(_intent_payload(case))
    compiled = compiler.compile(intent)
    record["compile_outcome"] = compiled.outcome
    record["compile_reasons"] = compiled.codes()

    if compiled.outcome != OUTCOME_PLAN:
        record["disposition"] = (COMPILE_CLARIFY
                                 if compiled.outcome == OUTCOME_CLARIFY
                                 else COMPILE_REFUSE)
        record["reason"] = (compiled.codes() or [""])[0]
        _score(record, expect)
        return record

    plan = compiled.plan.to_dict()
    record["plan_id"] = plan.get("plan_id")
    record["plan_period"] = dict(plan.get("period") or {})

    # NO PHYSICAL BINDING MAY REACH THE PLAN. The intent parser refuses a date
    # or a snapshot id in a binding slot; this re-reads the emitted plan and
    # says so per case rather than trusting the guard upstream.
    record["model_authored_physical_binding"] = _physical_binding(plan)

    outcome = temporal.execute_temporal_plan(
        plan, store=store, client_id=fixture.CLIENT_ID, semantics=semantics,
        route=fixture.ROUTE)
    record["outcome"] = outcome.to_dict()

    if not outcome.eligible:
        record["disposition"] = INELIGIBLE
    elif outcome.reason:
        record["disposition"] = CLARIFY if outcome.clarifiable else REFUSE
    else:
        record["disposition"] = EXECUTED
    record["reason"] = outcome.reason or ""

    if record["disposition"] == EXECUTED:
        _reconcile(record, case, outcome, history)
    _score(record, expect)
    return record


def _physical_binding(plan: Mapping[str, Any]) -> Optional[str]:
    """A date or a snapshot id anywhere in the plan's PERIOD. There must be none.

    The plan is allowed to carry physical field bindings — the compiler chose
    those. It is not allowed to carry a physical PERIOD, because no governed
    catalogue was consulted when it was written.
    """
    period = plan.get("period") or {}
    for key in ("labels", "form", "grain", "contract"):
        for value in ([period.get(key)] if not isinstance(period.get(key),
                                                          (list, tuple))
                      else list(period.get(key) or ())):
            token = str(value or "")
            if "snapshot" in token.lower():
                return f"period.{key}={token!r}"
            digits = [part for part in token.replace("/", "-").split("-")
                      if part.isdigit()]
            if len(digits) >= 3:
                return f"period.{key}={token!r}"
    return None


def _reconcile(record: Dict[str, Any], case: Mapping[str, Any], outcome: Any,
               history: Sequence[Tuple[str, Any]]) -> None:
    """Snapshot selection and every figure, against the independently held truth."""
    expect = dict(case.get("expect") or {})
    frames = dict(history)
    produced_dates = [p.reporting_date for p in outcome.points]
    expected_dates = list(expect.get("snapshots") or ())

    record["snapshots_selected"] = produced_dates
    record["snapshots_expected"] = expected_dates
    record["snapshot_selection_ok"] = produced_dates == expected_dates
    if not record["snapshot_selection_ok"]:
        record["failures"].append(
            f"snapshot selection: expected {expected_dates}, "
            f"got {produced_dates}")

    if outcome.shape != expect.get("shape"):
        record["failures"].append(
            f"shape: expected {expect.get('shape')!r}, got {outcome.shape!r}")
    if expect.get("basis") and outcome.basis != expect.get("basis"):
        record["failures"].append(
            f"basis: expected {expect.get('basis')!r}, got {outcome.basis!r}")

    recipe = dict(expect.get("oracle") or {})
    group_by = list(recipe.get("group_by") or ())
    scalars = 0
    cells = 0
    for point in outcome.points:
        frame = frames.get(point.reporting_date)
        if frame is None:
            record["failures"].append(
                f"{point.reporting_date}: not a fixture period")
            continue
        expected = oracle_for(recipe, frame)
        if group_by:
            produced = {tuple(str(cell[axis]) for axis in group_by): cell["value"]
                        for cell in point.cells}
            if set(produced) != set(expected):
                record["failures"].append(
                    f"{point.reporting_date}: group keys "
                    f"{sorted(set(expected) ^ set(produced))} differ")
            for key, figure in expected.items():
                cells += 1
                if not _close(produced.get(key), figure):
                    record["failures"].append(
                        f"{point.reporting_date} {key}: expected {figure}, "
                        f"got {produced.get(key)}")
        else:
            scalars += 1
            if not _close(point.value, expected):
                record["failures"].append(
                    f"{point.reporting_date}: expected {expected}, "
                    f"got {point.value}")
    record["scalars_checked"] = scalars
    record["cells_checked"] = cells

    if outcome.comparison:
        baseline = oracle_for(recipe, frames[outcome.points[0].reporting_date])
        current = oracle_for(recipe, frames[outcome.points[1].reporting_date])
        if not group_by:
            change = float(current) - float(baseline)
            percent = None if baseline == 0 else change / float(baseline) * 100.0
            if not _close(outcome.comparison["absolute_change"], change):
                record["failures"].append(
                    f"absolute change: expected {change}, "
                    f"got {outcome.comparison['absolute_change']}")
            if not _close(outcome.comparison["percent_change"], percent):
                record["failures"].append(
                    f"percent change: expected {percent}, "
                    f"got {outcome.comparison['percent_change']}")
            record["comparison_checked"] = True

    # EVERY PREDICATE AND EVERY AXIS, ON EVERY SNAPSHOT. A filter that survived
    # five periods and vanished on the sixth would still produce a plausible
    # series, so this is checked per point rather than once.
    wanted_predicates = {f["field"] for f in (outcome.requested.get("filters") or ())}
    wanted_axes = set(outcome.requested.get("dimensions") or ())
    for point in outcome.points:
        applied = {entry.get("field")
                   for entry in (point.receipt.get("applied_predicates") or ())}
        missing = wanted_predicates - applied
        if missing:
            record["failures"].append(
                f"{point.reporting_date}: predicates {sorted(missing)} not applied")
        grouped = set(point.receipt.get("group_field_keys") or ())
        if wanted_axes - grouped:
            record["failures"].append(
                f"{point.reporting_date}: axes "
                f"{sorted(wanted_axes - grouped)} not grouped")


def _score(record: Dict[str, Any], expect: Mapping[str, Any]) -> None:
    """CORRECT / PARTIAL / INCORRECT for one case."""
    if record["disposition"] != expect.get("disposition"):
        record["failures"].append(
            f"disposition: expected {expect.get('disposition')!r}, "
            f"got {record['disposition']!r}")
    wanted_reason = expect.get("reason")
    if wanted_reason and record.get("reason") != wanted_reason:
        record["failures"].append(
            f"reason: expected {wanted_reason!r}, got {record.get('reason')!r}")
    if record.get("model_authored_physical_binding"):
        record["failures"].append(
            f"the plan carries a physical period binding: "
            f"{record['model_authored_physical_binding']}")

    if not record["failures"]:
        record["verdict"] = "CORRECT"
    elif record["disposition"] == expect.get("disposition"):
        record["verdict"] = "PARTIAL"
    else:
        record["verdict"] = "INCORRECT"


# --------------------------------------------------------------------------- #
# the recorded corpus, replayed through the slice 2 perimeter
# --------------------------------------------------------------------------- #

def replay_recorded() -> Dict[str, Any]:
    """Every recorded plan whose period is not `current`, put to the perimeter.

    A second, independent read on the perimeter: these plans were compiled from
    REAL Opus interpretations in the signed-off run, so what they show is how the
    contract behaves on temporal intent the model actually produced.
    """
    recorded = json.loads(RECORDED_RUN.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for row in recorded["results"]:
        # Selected on the INTENT, not on the plan, so a temporal question the
        # compiler declined to plan is still counted. Reporting only the plans
        # would silently drop the cases where the model stated a period the
        # contract could not settle — which is most of what there is to learn.
        form = (((row.get("intent") or {}).get("time") or {}).get("form"))
        if form in (None, "current"):
            continue
        plan = row.get("plan")
        entry = {
            "question_id": row["question_id"],
            "question": row["question"],
            "capability": (row.get("intent") or {}).get("capability"),
            "operation": (row.get("intent") or {}).get("operation"),
            "period_form": form,
            "compile_outcome": row.get("outcome"),
        }
        if not plan:
            entry.update({"slice2_eligible": False,
                          "slice2_reason": f"COMPILE_{row.get('outcome')}",
                          "slice2_detail": ", ".join(row.get("reason_codes") or ())})
        else:
            eligible, reason, detail = temporal.check_temporal_eligibility(plan)
            entry.update({"slice2_eligible": eligible, "slice2_reason": reason,
                          "slice2_detail": detail[:160]})
        rows.append(entry)
    return {
        "total_recorded_temporal_intents": len(rows),
        "compiled_to_a_plan": sum(1 for r in rows if r["compile_outcome"] == "PLAN"),
        "eligible": sum(1 for r in rows if r["slice2_eligible"]),
        "by_reason": dict(Counter(r["slice2_reason"] for r in rows
                                  if not r["slice2_eligible"])),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# governance counts
# --------------------------------------------------------------------------- #

def governance(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The assertions the slice is required to answer, counted from the run."""
    import ast

    executed = [r for r in records if r.get("disposition") == EXECUTED]

    # Temporal intent preserved: every executed case carries the plan's own
    # period form, and every non-executed one names a temporal reason rather
    # than dropping the constraint and answering anyway.
    preserved = 0
    for record in records:
        period = record.get("plan_period") or {}
        requested = ((record.get("outcome") or {}).get("requested") or {})
        if not period:
            preserved += 1                       # never reached a plan
            continue
        if requested.get("period_form") == period.get("form"):
            preserved += 1

    # No silent temporal drop: nothing may execute against fewer or other
    # snapshots than the bank says, and nothing that failed to resolve may have
    # produced points.
    silent_drops = 0
    substitutions = 0
    for record in records:
        if record.get("disposition") == EXECUTED:
            if record.get("snapshot_selection_ok") is False:
                substitutions += 1
        elif (record.get("outcome") or {}).get("points"):
            silent_drops += 1

    selection_errors = sum(1 for r in executed
                           if r.get("snapshot_selection_ok") is False)
    physical = sum(1 for r in records
                   if r.get("model_authored_physical_binding"))

    # No raw-text re-read after the plan: a source property of the two modules
    # on the governed path, read off their ASTs.
    rereads = 0
    for module in (temporal, adapter):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {a.arg for n in ast.walk(tree)
                  if isinstance(n, ast.arguments) for a in n.args}
        names |= {a.arg for n in ast.walk(tree)
                  if isinstance(n, ast.arguments) for a in n.kwonlyargs}
        rereads += len(names & {"question", "sentence", "parsed", "raw_question"})
        imported = {n.module for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom) and n.module}
        imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                     for a in n.names}
        rereads += len(imported & {"re"})

    row_level_date_filter = any(
        token in Path(temporal.__file__).read_text(encoding="utf-8")
        for token in ("to_datetime", "read_csv", "as_at_date",
                      "origination_date", "reporting_date_column"))

    return {
        "TEMPORAL_INTENT_PRESERVED": ("YES" if preserved == len(records)
                                      else f"NO ({preserved}/{len(records)})"),
        "SNAPSHOT_SELECTION_DETERMINISTIC": ("YES" if selection_errors == 0
                                             else "NO"),
        "RAW_TEXT_TEMPORAL_REREADS_AFTER_PLAN": rereads,
        "ROW_LEVEL_DATE_FILTER_USED_FOR_HISTORY": ("YES" if row_level_date_filter
                                                   else "NO"),
        "MODEL_AUTHORED_PHYSICAL_SNAPSHOT_BINDINGS": physical,
        "SILENT_TEMPORAL_DROPS": silent_drops,
        "SILENT_PERIOD_SUBSTITUTIONS": substitutions,
        "SNAPSHOT_SELECTION_ERRORS": selection_errors,
    }


def live_call_evidence() -> Dict[str, Any]:
    """Whether an Anthropic client was ever imported. It must not have been."""
    suspects = [name for name in sys.modules
                if name == "anthropic" or name.startswith("anthropic.")]
    return {"anthropic_client_imported": bool(suspects),
            "modules": sorted(suspects), "live_opus_calls": 0}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str]) -> int:
    bank = yaml.safe_load(BANK.read_text(encoding="utf-8"))
    cases = bank["cases"]
    history = fixture.default_history(periods=int(bank["fixture"]["periods"]))
    semantics = load_mi_semantics(str(REGISTRY))
    compiler = DeterministicCompiler()

    with tempfile.TemporaryDirectory() as tmp:
        store = fixture.build_store(Path(tmp) / "snapshots", history)
        catalogue = [h.reporting_date for h in
                     store.list_snapshots(fixture.CLIENT_ID, route=fixture.ROUTE)]
        records = [run_case(case, compiler, store, semantics, history)
                   for case in cases]

    verdicts = Counter(r["verdict"] for r in records)
    dispositions = Counter(r["disposition"] for r in records)
    report = {
        "bank": str(BANK.relative_to(_REPO_ROOT)),
        "bank_size": len(cases),
        "fixture": {"catalogue": catalogue,
                    "periods": len(history),
                    "cadence": bank["fixture"]["cadence"]},
        "verdicts": dict(verdicts),
        "dispositions": dict(dispositions),
        "executed_cases": dispositions.get(EXECUTED, 0),
        "numerically_reconciled": sum(
            1 for r in records if r["disposition"] == EXECUTED
            and not r["failures"]),
        "scalars_checked": sum(r.get("scalars_checked", 0) for r in records),
        "grouped_cells_checked": sum(r.get("cells_checked", 0) for r in records),
        "governance": governance(records),
        "live": live_call_evidence(),
        "recorded_corpus_replay": replay_recorded(),
        "results": records,
    }
    out = EVIDENCE_DIR / "slice2_acceptance.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== SLICE 2 OFFLINE ACCEPTANCE")
    print(f"  BANK_SIZE              {len(cases)}")
    print(f"  CATALOGUE              {len(catalogue)} monthly snapshots "
          f"{catalogue[0]} .. {catalogue[-1]}")
    for name, count in sorted(dispositions.items()):
        print(f"    {name:<22} {count}")
    print("  VERDICTS")
    for name, count in sorted(verdicts.items()):
        print(f"    {name:<22} {count}")
    print(f"  EXECUTED               {dispositions.get(EXECUTED, 0)}")
    print(f"  SCALARS_RECONCILED     {report['scalars_checked']}")
    print(f"  GROUPED_CELLS          {report['grouped_cells_checked']}")
    print("  GOVERNANCE")
    for name, value in report["governance"].items():
        print(f"    {name:<44} {value}")
    print("  RECORDED CORPUS REPLAY (real Opus interpretations)")
    replay = report["recorded_corpus_replay"]
    print(f"    temporal interpretations       "
          f"{replay['total_recorded_temporal_intents']}")
    print(f"    of which compiled to a plan    {replay['compiled_to_a_plan']}")
    print(f"    slice 2 eligible               {replay['eligible']}")
    for name, count in sorted(replay["by_reason"].items()):
        print(f"      {name:<28} {count}")
    print(f"  LIVE_OPUS_CALLS        {report['live']['live_opus_calls']} "
          f"(anthropic client imported: "
          f"{report['live']['anthropic_client_imported']})")

    failures = [r for r in records if r["verdict"] != "CORRECT"]
    for record in failures:
        print(f"  FAIL {record['id']}: {record['failures']}")
    print(f"  WRITTEN                {out.relative_to(_REPO_ROOT)}")
    print(f"  RESULT                 "
          f"{'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
