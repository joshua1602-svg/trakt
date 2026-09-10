"""The 135-question INTERPRETATION benchmark.

This measures what a question MEANS and what Trakt would therefore be
authorised to do. It does not measure answers: nothing here executes a plan,
and no figure is compared to anything.

    for each of 135 questions
        Opus  -> CandidateIntent
        compiler -> PLAN | CLARIFY | REFUSE
        score the material semantic dimensions independently
    for each of 45 canonicals
        do the three variants compile to the same authorised work?

Two rules govern the scoring, and both exist because the alternative reports
confidence nobody earned.

FIRST, a dimension the fixture does not state is UNSCOREABLE, never correct.
The per-question verdict is therefore three-way — fully correct, partially
correct, incorrect — plus a separate unscoreable count, and a question whose
fixture states nothing scoreable is not a pass.

SECOND, a governed refusal is not a failure. A question this vocabulary cannot
express should refuse, and a refusal is recorded with its reason code and
reported beside the plans, not counted against them.

    python -m mi_agent.interpretation_v2.benchmark --live --out report.json
    python -m mi_agent.interpretation_v2.benchmark --replay run.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from .compiler import CompilerContext, DeterministicCompiler
from .equivalence import (
    SCORED_DIMENSIONS,
    compare_results,
    observed_dimensions_from_plan,
    score_intent,
)
from .opus_interpreter import (
    CONFIGURED_MODEL,
    AnthropicInterpreterClient,
    InterpretationOutcome,
    OpusInterpreter,
    ReplayClient,
)
from .outcomes import OUTCOME_CLARIFY, OUTCOME_PLAN, OUTCOME_REFUSE, refuse
from .vocabulary import load_governed_vocabulary

_HERE = Path(__file__).resolve().parent
BANK_PATH = _HERE / "banks" / "interpretation_bank_135.yaml"
EXPECTATIONS_PATH = _HERE / "banks" / "expected_intents.yaml"

#: How a question's semantic result is classified. `governed_refusal` and
#: `clarification` are outcomes of the SYSTEM; `unscoreable` is a statement
#: about the FIXTURE, and keeping them apart is what lets the report say which
#: of the two a number is about.
VERDICT_FULLY_CORRECT = "fully_correct"
VERDICT_PARTIALLY_CORRECT = "partially_correct"
VERDICT_INCORRECT = "incorrect"
VERDICT_GOVERNED_REFUSAL = "governed_refusal"
VERDICT_CLARIFICATION = "clarification"
VERDICT_UNSCOREABLE = "unscoreable"
VERDICT_INTERPRETATION_FAILED = "interpretation_failed"

#: Where a non-plan outcome came from. Reported as the error taxonomy.
ERROR_INTERPRETATION = "interpretation_error"
ERROR_VOCABULARY = "vocabulary_binding_error"
ERROR_COMPILER_REJECTION = "compiler_rejection"
ERROR_MISSING_CONTRACT = "missing_deterministic_contract"
ERROR_AMBIGUOUS_TRUTH = "ambiguous_truth"
ERROR_OTHER = "other"

_TAXONOMY: Mapping[str, str] = {
    "MODEL_UNAVAILABLE": ERROR_INTERPRETATION,
    "MODEL_OUTPUT_MALFORMED": ERROR_INTERPRETATION,
    "MODEL_OUTPUT_CONTAINS_CODE": ERROR_INTERPRETATION,
    "MODEL_OUTPUT_CONTAINS_PHYSICAL_BINDING": ERROR_INTERPRETATION,
    "INTENT_SCHEMA_INVALID": ERROR_INTERPRETATION,
    "INTENT_SCHEMA_VERSION_UNSUPPORTED": ERROR_INTERPRETATION,
    "UNREGISTERED_CONCEPT": ERROR_VOCABULARY,
    "CONCEPT_UNAVAILABLE": ERROR_VOCABULARY,
    "CAPABILITY_UNAVAILABLE": ERROR_MISSING_CONTRACT,
    "PERIOD_UNRESOLVED": ERROR_MISSING_CONTRACT,
    "UNSUPPORTED_OPERATION": ERROR_COMPILER_REJECTION,
    "UNSUPPORTED_STATISTIC": ERROR_COMPILER_REJECTION,
    "UNSUPPORTED_COMPOSITION": ERROR_COMPILER_REJECTION,
    "UNSUPPORTED_FILTER": ERROR_COMPILER_REJECTION,
    "WEIGHT_NOT_PERMITTED": ERROR_COMPILER_REJECTION,
    "INVALID_GEOGRAPHY_BASIS": ERROR_COMPILER_REJECTION,
    "CONFLICTING_CLAIMS": ERROR_COMPILER_REJECTION,
    "MISSING_REQUIRED_SLOT": ERROR_COMPILER_REJECTION,
    "AMBIGUOUS_MEASURE": ERROR_AMBIGUOUS_TRUTH,
    "AMBIGUOUS_DIMENSION": ERROR_AMBIGUOUS_TRUTH,
    "AMBIGUOUS_POPULATION": ERROR_AMBIGUOUS_TRUTH,
    "AMBIGUOUS_PERIOD": ERROR_AMBIGUOUS_TRUTH,
    "AMBIGUOUS_GEOGRAPHY": ERROR_AMBIGUOUS_TRUTH,
    "MODEL_FLAGGED_AMBIGUITY": ERROR_AMBIGUOUS_TRUTH,
}


def load_bank(path: Path = BANK_PATH) -> Mapping[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_expectations(path: Path = EXPECTATIONS_PATH) -> Mapping[str, Any]:
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(
        "expectations", {})


def iter_questions(bank: Mapping[str, Any]):
    """(canonical_id, variant, question_id, question_text) for all 135."""
    for canonical in bank["canonicals"]:
        for variant in canonical["variants"]:
            yield (canonical["id"], variant["variant"],
                   f"{canonical['id']}{variant['variant']}", variant["question"])


@dataclass
class QuestionResult:
    canonical_id: str
    variant: str
    question_id: str
    question: str
    model_id: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    intent: Optional[Mapping[str, Any]] = None
    raw_payload: Optional[Mapping[str, Any]] = None
    outcome: str = ""
    reason_codes: Tuple[str, ...] = ()
    reasons: Tuple[Mapping[str, Any], ...] = ()
    plan_id: str = ""
    plan: Optional[Mapping[str, Any]] = None
    scored: Mapping[str, Optional[bool]] = field(default_factory=dict)
    verdict: str = ""
    error_class: str = ""
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        body = asdict(self)
        body["reason_codes"] = list(self.reason_codes)
        body["reasons"] = [dict(r) for r in self.reasons]
        return body


def _verdict(result: QuestionResult, human_review: bool) -> str:
    if result.outcome == OUTCOME_PLAN:
        checked = [v for v in result.scored.values() if v is not None]
        if not checked:
            return VERDICT_UNSCOREABLE
        if all(checked):
            return VERDICT_FULLY_CORRECT
        if any(checked):
            return VERDICT_PARTIALLY_CORRECT
        return VERDICT_INCORRECT
    if result.outcome == OUTCOME_CLARIFY:
        return VERDICT_CLARIFICATION
    if result.outcome == OUTCOME_REFUSE:
        if "MODEL_UNAVAILABLE" in result.reason_codes or \
                "MODEL_OUTPUT_MALFORMED" in result.reason_codes:
            return VERDICT_INTERPRETATION_FAILED
        return VERDICT_GOVERNED_REFUSAL
    return VERDICT_INCORRECT


def _error_class(codes: Sequence[str]) -> str:
    for code in codes:
        mapped = _TAXONOMY.get(code)
        if mapped:
            return mapped
    return ERROR_OTHER if codes else ""


def run_benchmark(*, interpreter: OpusInterpreter,
                  compiler: Optional[DeterministicCompiler] = None,
                  bank: Optional[Mapping[str, Any]] = None,
                  expectations: Optional[Mapping[str, Any]] = None,
                  limit: Optional[int] = None,
                  progress: bool = False) -> Dict[str, Any]:
    bank = bank or load_bank()
    expectations = expectations if expectations is not None else load_expectations()
    compiler = compiler or DeterministicCompiler(CompilerContext())

    results: List[QuestionResult] = []
    by_canonical: Dict[str, List[Tuple[str, Any]]] = {}

    questions = list(iter_questions(bank))
    if limit:
        questions = questions[:limit]

    for index, (canonical_id, variant, qid, text) in enumerate(questions, start=1):
        started = time.monotonic()
        outcome: InterpretationOutcome = interpreter.interpret(text)
        if outcome.ok:
            compiled = compiler.compile(outcome.intent)
        else:
            compiled = refuse(outcome.reason, compiler_version=compiler.version)
        elapsed = int((time.monotonic() - started) * 1000)

        expected = (expectations.get(canonical_id) or {})
        result = QuestionResult(
            canonical_id=canonical_id, variant=variant, question_id=qid,
            question=text, model_id=outcome.model_id,
            usage=dict(outcome.usage or {}),
            intent=outcome.intent.to_dict() if outcome.intent else None,
            raw_payload=dict(outcome.raw_payload) if outcome.raw_payload else None,
            outcome=compiled.outcome,
            reason_codes=tuple(compiled.codes()),
            reasons=tuple(r.to_dict() for r in compiled.reasons),
            latency_ms=elapsed,
        )
        if compiled.is_plan:
            result.plan_id = compiled.plan.plan_id
            result.plan = compiled.plan.to_dict()
            result.scored = score_intent(outcome.intent, expected.get("expected", {}),
                                         plan=compiled.plan)
        result.error_class = _error_class(result.reason_codes)
        result.verdict = _verdict(result, bool(expected.get("human_review")))
        results.append(result)
        by_canonical.setdefault(canonical_id, []).append((qid, compiled))

        if progress:
            print(f"[{index:>3}/{len(questions)}] {qid:<8} {result.outcome:<8} "
                  f"{result.verdict}", flush=True)

    invariance = [compare_results(cid, rows) for cid, rows in by_canonical.items()]
    return _report(bank, expectations, results, invariance, interpreter)


def _report(bank, expectations, results: Sequence[QuestionResult],
            invariance, interpreter) -> Dict[str, Any]:
    verdicts: Dict[str, int] = {}
    errors: Dict[str, int] = {}
    per_dimension: Dict[str, Dict[str, int]] = {
        d: {"correct": 0, "incorrect": 0, "unscoreable": 0} for d in SCORED_DIMENSIONS}
    models: Dict[str, int] = {}
    tokens = {"input": 0, "output": 0}

    for result in results:
        verdicts[result.verdict] = verdicts.get(result.verdict, 0) + 1
        if result.error_class:
            errors[result.error_class] = errors.get(result.error_class, 0) + 1
        if result.model_id:
            models[result.model_id] = models.get(result.model_id, 0) + 1
        tokens["input"] += int(result.usage.get("input_tokens") or 0)
        tokens["output"] += int(result.usage.get("output_tokens") or 0)
        for dimension in SCORED_DIMENSIONS:
            value = result.scored.get(dimension)
            bucket = ("unscoreable" if value is None
                      else "correct" if value else "incorrect")
            per_dimension[dimension][bucket] += 1

    human_review = sorted(cid for cid, entry in expectations.items()
                          if entry.get("human_review"))
    invariant = [r for r in invariance if r.invariant]

    return {
        "benchmark": "interpretation_v2/135",
        "bank": {"canonical_count": bank["canonical_count"],
                 "question_count": bank["question_count"],
                 "sources": bank["sources"]},
        "model": {
            "configured": CONFIGURED_MODEL,
            "returned": models,
            "successful_calls": sum(1 for r in results if r.intent is not None),
            "failed_calls": sum(1 for r in results if r.intent is None),
            "tokens": tokens,
            "rows_of_data_transmitted": 0,
            "portfolio_values_transmitted": 0,
        },
        "interpreter_version": interpreter.version,
        "vocabulary_version": interpreter.vocabulary.version,
        "questions_attempted": len(results),
        "verdicts": verdicts,
        "outcomes": {
            OUTCOME_PLAN: sum(1 for r in results if r.outcome == OUTCOME_PLAN),
            OUTCOME_CLARIFY: sum(1 for r in results if r.outcome == OUTCOME_CLARIFY),
            OUTCOME_REFUSE: sum(1 for r in results if r.outcome == OUTCOME_REFUSE),
        },
        "error_taxonomy": errors,
        "per_dimension": per_dimension,
        "paraphrase_invariance": {
            "canonicals": len(invariance),
            "invariant": len(invariant),
            "divergent": [r.to_dict() for r in invariance if not r.invariant],
        },
        "human_review_required": {"count": len(human_review), "canonicals": human_review},
        "reason_code_frequency": _reason_frequency(results),
        "results": [r.to_dict() for r in results],
    }


def _reason_frequency(results: Sequence[QuestionResult]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for result in results:
        for code in result.reason_codes:
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _replay_payloads(path: Path) -> Dict[str, Mapping[str, Any]]:
    """Recorded raw model payloads keyed by question, from a prior report."""
    report = json.loads(path.read_text(encoding="utf-8"))
    return {row["question"]: row["raw_payload"] for row in report.get("results", ())
            if row.get("raw_payload")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="call the configured Anthropic model")
    parser.add_argument("--replay", type=Path,
                        help="replay recorded payloads from a prior report")
    parser.add_argument("--model", default=CONFIGURED_MODEL)
    parser.add_argument("--out", type=Path, default=Path("interpretation_v2_run.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.replay:
        client = ReplayClient(_replay_payloads(args.replay), model_id="replay")
    elif args.live:
        client = AnthropicInterpreterClient(model=args.model)
        if not client.available:
            print("ANTHROPIC_API_KEY is not set: LIVE_OPUS_INTERPRETATION = "
                  "NOT_PROVEN", file=sys.stderr)
            return 2
    else:
        parser.error("choose --live or --replay")

    interpreter = OpusInterpreter(client, vocabulary=load_governed_vocabulary())
    report = run_benchmark(interpreter=interpreter, limit=args.limit,
                           progress=not args.quiet)
    args.out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"},
                     indent=1, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
