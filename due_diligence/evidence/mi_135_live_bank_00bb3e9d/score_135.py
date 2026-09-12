#!/usr/bin/env python3
"""Score the collected 135-question evidence. Reads records; asks nothing.

WHO SCORES WHAT, AND WHY IT IS NOT THIS FILE'S OPINION.

  interpretation   `interpretation_v2.equivalence.score_intent`, per dimension,
                   against `banks/expected_intents.yaml`. This is the SIGNED-OFF
                   scorer that produced the committed run1..run8 baselines. A
                   rubric invented here would be a second owner, and the
                   before/after comparison would then be between two different
                   questions.
  legacy answers   the product's OWN integrity checks — `filter_invariant`,
                   `dimension_invariant`, `semantic_guard`, `reconciliation` —
                   read off the envelope rather than recomputed. The legacy path
                   already states whether it dropped a filter or a grouping; a
                   second opinion here would be less authoritative, not more.
  served answers   the governed receipt, compared with what the plan requested.

WHAT IT REFUSES TO DECIDE. The bank carries expected SEMANTICS and no expected
NUMBERS — it says so itself: "an INTERPRETATION benchmark: it scores what a
question MEANS, never what the answer is". So `numeric_parity` is `N/A`
everywhere, stated as an absence rather than filled in from the answer the
product happened to give. Nothing here invents a tolerance, and nothing treats
the served figure as its own ground truth.

Anything this file cannot decide from evidence is INCONCLUSIVE, never a pass.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

RAW = _HERE / "raw_records.json"
RESULTS = _HERE / "MI_135_LIVE_BANK_RESULTS.json"

# user-facing outcomes
FULLY_CORRECT = "FULLY_CORRECT"
PARTIALLY_CORRECT = "PARTIALLY_CORRECT"
WRONG = "WRONG"
APPROPRIATE_CLARIFICATION = "APPROPRIATE_CLARIFICATION"
UNNECESSARY_CLARIFICATION = "UNNECESSARY_CLARIFICATION"
PLAN_CONNECTIVITY_CLARIFICATION = "PLAN_OR_CONNECTIVITY_GAP_MASQUERADING_AS_CLARIFY"
HONEST_REFUSAL = "HONEST_REFUSAL"
BAD_REFUSAL = "BAD_REFUSAL"
INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
INCONCLUSIVE = "INCONCLUSIVE"

# migration outcomes
NEW = "NEW"
LEGACY_FALLBACK = "LEGACY_FALLBACK"
NOT_REACHED = "NOT_REACHED"

#: Capabilities with a governed runtime on the build under test. Everything else
#: is refused at the perimeter BY DESIGN, and a fallback for one of these is a
#: migration statement, not a product defect.
MIGRATED_CAPABILITIES = frozenset({"generic_analysis", "pipeline"})


def _expectations() -> Mapping[str, Any]:
    import yaml
    path = _REPO / "mi_agent/interpretation_v2/banks/expected_intents.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["expectations"]


def _score_interpretation(intent_body: Optional[Mapping[str, Any]],
                          expected: Mapping[str, Any]
                          ) -> Tuple[Dict[str, Optional[bool]], str]:
    """Per-dimension truth from the signed-off scorer, or why it could not run."""
    if not intent_body:
        return {}, "no candidate intent was recorded"
    from mi_agent.interpretation_v2.equivalence import score_intent
    from mi_agent.interpretation_v2.intent import parse_candidate_intent
    from mi_agent.interpretation_v2.vocabulary import load_governed_vocabulary
    try:
        intent = parse_candidate_intent(dict(intent_body))
    except Exception as exc:                                         # noqa: BLE001
        return {}, f"the recorded intent did not parse: {type(exc).__name__}"
    return score_intent(intent, expected,
                        vocabulary=load_governed_vocabulary()), ""


def _legacy_integrity(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """The legacy path's own verdict on whether it kept the question's meaning."""
    def ok(key: str) -> Optional[bool]:
        node = envelope.get(key)
        if not isinstance(node, Mapping):
            return None
        value = node.get("ok")
        return bool(value) if value is not None else None
    return {
        "filter_invariant_ok": ok("filter_invariant"),
        "dimension_invariant_ok": ok("dimension_invariant"),
        "semantic_guard_ok": ok("semantic_guard"),
        "reconciliation_ok": ok("reconciliation"),
        "unavailable_filters": list(
            (envelope.get("spec") or {}).get("unavailable_filters") or ()),
    }


def _silent_drops(record: Mapping[str, Any],
                  envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """Semantic loss on a SERVED answer, read from the evidence, never inferred.

    `N/A` where the question asked for nothing of that kind — an absent filter
    cannot be dropped — so a run of `False` never overstates what was checked.
    """
    execution = record.get("execution") or {}
    requested = execution.get("requested_semantics") or {}
    receipt = execution.get("receipt") or {}
    legacy = _legacy_integrity(envelope)

    def asked(value: Any) -> bool:
        return bool(value)

    def drop(asked_it: bool, kept: bool) -> Any:
        return (not kept) if asked_it else "N/A"

    executed_dims = {str(d) for d in (receipt.get("group_field_keys") or ())}
    wanted_dims = {str(d) for d in (requested.get("dimensions") or ()) if d}
    executed_preds = receipt.get("applied_predicates")
    wanted_preds = requested.get("filters") or ()

    return {
        "silent_measure_drop": drop(
            asked(requested.get("measure_concept")),
            bool(receipt.get("measure_concept") or receipt.get("measure_field")
                 or receipt.get("aggregation"))),
        "silent_dimension_drop": drop(bool(wanted_dims),
                                      wanted_dims <= executed_dims),
        "silent_filter_drop": drop(
            bool(wanted_preds),
            executed_preds is not None
            and len(executed_preds) >= len(wanted_preds)),
        "silent_period_drop": drop(
            str(requested.get("period_form") or "current") != "current",
            bool(receipt.get("selected_periods") or receipt.get("grain")
                 or receipt.get("temporal_basis"))),
        "silent_capability_drop": drop(
            asked(requested.get("capability")),
            receipt.get("capability") in (None, requested.get("capability"))),
        "silent_population_drop": drop(
            asked(requested.get("population_base")),
            receipt.get("population_base") in
            (None, requested.get("population_base"))),
        "silent_scope_drop": drop(
            asked(requested.get("source_reference")
                  or requested.get("population_lens")),
            bool(receipt.get("population_base"))),
        "silent_comparison_drop": drop(
            str(requested.get("comparison_kind") or "none") != "none",
            bool(receipt.get("comparison") or receipt.get("result_shape"))),
        "silent_widening": legacy["unavailable_filters"] and True or False,
        "legacy_integrity": legacy,
    }


def _user_outcome(row: Dict[str, Any]) -> Tuple[str, str]:
    """`(outcome, why)` from the evidence. Undecidable reads INCONCLUSIVE."""
    if row["infrastructure_failure"]:
        return INFRASTRUCTURE_FAILURE, "the request did not reach the service"
    if row["record"] is None:
        return INCONCLUSIVE, "no governed evidence record was found"

    outcome = row["interpretation_outcome"]
    if outcome == "INTERPRETER_FAILURE":
        return INFRASTRUCTURE_FAILURE, "the interpreter did not return a reading"
    if outcome == "CLARIFY":
        return INCONCLUSIVE, "clarification — adjudicated separately by class"
    if outcome == "REFUSE":
        # An honest refusal names a governed obstacle. A refusal of something the
        # estate can express is a bad one; that judgement is made per reason code
        # in the review, so this states the evidence and defers.
        return INCONCLUSIVE, "refusal — adjudicated by reason code"

    scored = row["interpretation_scored"]
    checked = [v for v in scored.values() if v is not None]
    if not checked:
        return INCONCLUSIVE, "the fixture states no dimension for this question"

    drops = row["drops"]
    real_drops = [k for k, v in drops.items()
                  if k.startswith("silent_") and v is True]
    if row["migration_outcome"] == NEW and real_drops:
        return WRONG, f"served with semantic loss: {', '.join(sorted(real_drops))}"

    if all(checked):
        if row["migration_outcome"] == NEW:
            return FULLY_CORRECT, "served by the governed path, semantics intact"
        integrity = drops["legacy_integrity"]
        if integrity["filter_invariant_ok"] is False or \
           integrity["dimension_invariant_ok"] is False:
            return WRONG, "the legacy path reports it dropped part of the question"
        if not row["envelope_ok"]:
            return BAD_REFUSAL, "the reading was right and no answer was returned"
        return FULLY_CORRECT, "answered by the legacy path, semantics intact"
    if any(checked):
        return PARTIALLY_CORRECT, (
            "the reading differs from the fixture on: "
            + ", ".join(k for k, v in sorted(scored.items()) if v is False))
    return WRONG, "the reading does not match the fixture on any stated dimension"


def build_rows() -> List[Dict[str, Any]]:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    expectations = _expectations()
    rows: List[Dict[str, Any]] = []

    for entry in raw["records"]:
        record = entry.get("record")
        envelope = entry.get("envelope") or {}
        compiler = (record or {}).get("compiler") or {}
        interpretation = (record or {}).get("interpretation") or {}
        serving = (record or {}).get("serving") or {}
        execution = (record or {}).get("execution") or {}
        plan = compiler.get("plan") or {}
        expected = (expectations.get(entry["canonical_id"]) or {}).get("expected") or {}

        infra = bool(envelope.get("__transport_error__"))
        model_id = ((record or {}).get("model") or {}).get("model_id") or ""

        outcome = str(compiler.get("outcome") or "").upper()
        if record is None:
            interp_outcome = "NO_RECORD"
        elif (record or {}).get("disposition") == "INTERPRETER_FAILURE":
            interp_outcome = "INTERPRETER_FAILURE"
        elif outcome in ("PLAN", "CLARIFY", "REFUSE"):
            interp_outcome = outcome
        else:
            interp_outcome = outcome or "UNKNOWN"

        decision = str(serving.get("decision") or "")
        migration = (NEW if decision == "NEW"
                     else LEGACY_FALLBACK if decision else
                     (NOT_REACHED if record is None else LEGACY_FALLBACK))
        if infra:
            migration = "INFRASTRUCTURE_FAILURE"

        scored, why_not = _score_interpretation(
            interpretation.get("candidate_intent"), expected)

        output = (tuple(plan.get("outputs") or ()) or ({},))[0]
        row: Dict[str, Any] = {
            "question_id": entry["question_id"],
            "canonical_id": entry["canonical_id"],
            "variant": entry["variant"],
            "original_category": entry["original_category"],
            "shape": entry["shape"],
            "origin_bank": entry["origin_bank"],
            "question": entry["question"],
            "model_id": model_id,
            "infrastructure_failure": infra,
            "record": record and True or None,
            "envelope_ok": bool(envelope.get("ok")),
            "interpretation_outcome": interp_outcome,
            "interpretation_scored": scored,
            "interpretation_unscoreable_because": why_not,
            "expected_intent": expected,
            # read off the plan, never reconstructed from the sentence
            "capability": plan.get("capability"),
            "operation": plan.get("operation"),
            "population_base": (plan.get("population") or {}).get("base"),
            "population_lens": (plan.get("population") or {}).get("lens"),
            "source_reference": (plan.get("population") or {}).get("source_reference"),
            "measures": [m.get("concept") for m in (output.get("measures") or ())],
            "dimensions": [d.get("canonical_field")
                           for d in (output.get("dimensions") or ())],
            "filters": [[f.get("canonical_field"), f.get("comparator"), f.get("value")]
                        for f in (tuple(plan.get("filters") or ())
                                  + tuple(output.get("filters") or ()))],
            "period": plan.get("period"),
            "comparison_kind": plan.get("comparison_kind"),
            "plan_id": compiler.get("plan_id"),
            # the clarification evidence, verbatim
            "clarify_reasons": [dict(r) for r in (compiler.get("reasons") or ())],
            "reason_codes": list(compiler.get("reason_codes") or ()),
            "ambiguities": [dict(a) for a in
                            (interpretation.get("ambiguities") or ())],
            "migration_outcome": migration,
            "served_from": serving.get("response_served_from") or decision,
            "fallback_reason": serving.get("reason") or "",
            "eligibility": (record or {}).get("eligibility"),
            "execution_runtime": execution.get("runtime"),
            "receipt": execution.get("receipt"),
            "expected_capability_migrated":
                (expected.get("capability") in MIGRATED_CAPABILITIES
                 if expected.get("capability") else None),
            # the bank states no numbers; this is an absence, not a pass
            "expected_value": None,
            "numeric_parity": "N/A",
        }
        row["drops"] = _silent_drops(record or {}, envelope)
        row["user_outcome"], row["user_outcome_why"] = _user_outcome(row)
        rows.append(row)
    return rows


def main() -> int:
    rows = build_rows()
    RESULTS.write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str)
                       + "\n", encoding="utf-8")
    print(f"scored {len(rows)} questions -> {RESULTS.name}\n")
    for field in ("user_outcome", "migration_outcome", "interpretation_outcome"):
        print(f"{field}:")
        for key, count in Counter(r[field] for r in rows).most_common():
            print(f"   {key:52s} {count:4d}")
        print()
    drops = Counter()
    for r in rows:
        for k, v in r["drops"].items():
            if k.startswith("silent_") and v is True:
                drops[k] += 1
    print("silent semantic losses:", dict(drops) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
