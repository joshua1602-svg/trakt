#!/usr/bin/env python3
"""Freeze the V3 release-certification contract. Run ONCE, before any model call.

ATTEMPT 2. V2 is SPENT and immutable: its manifest, hashes, evidence and results
stay exactly as recorded, and nothing here writes into that directory. V2 failed
for two instrument reasons and neither was the product's — the bearer died 53
questions in, and the scorer could not parse a recorded intent carrying the
derived `time.stated`. Both instruments are now repaired, which is the ONLY
difference between this attempt and that one.

THE QUESTIONS AND THE OVERLAY ARE UNCHANGED. Same 135, same file, same bytes;
the overlay is still empty for the same reason. What changed is the version of
two test instruments, and this manifest records their new hashes so the change is
visible rather than implied.

WHAT IS FROZEN AND WHAT IS REUSED. The 135 natural-language questions are the
SAME 135, byte for byte, in the file the historical run used; so is the
expectation overlay, and so is the scorer. None of the three is copied here and
none is modified — this file records their sha256 so a reader can prove that.

WHY A SEPARATE MANIFEST AT ALL. The historical run's evidence and results are
immutable, and a second run writing into that directory would overwrite them.
This certification writes into its own directory and points the same collector
and the same scorer at it.

THE VERDICT MAPPING IS PART OF THE CONTRACT AND IS FROZEN BEFORE THE RUN.
`score_135.py` is the signed-off scorer and it emits its own outcome vocabulary,
which is nearly but not exactly the eight categories this certification reports.
Two of its outcomes have no direct counterpart:

    UNNECESSARY_CLARIFICATION                     -> BAD_REFUSAL
    PLAN_OR_CONNECTIVITY_GAP_MASQUERADING_AS_CLARIFY -> BAD_REFUSAL

Both are, by that scorer's own definitions, a case where the governed system
COULD have answered and asked a question instead — which is what BAD_REFUSAL
means here. Mapping them anywhere softer would let a connectivity gap be reported
as a legitimate clarification. The mapping is written down now, before any result
exists, so it cannot be chosen to flatter one.

Its `NOT_TESTED_KNOWN_UNMIGRATED` migration outcome is NOT a user verdict and is
not mapped into one; it is reported in the migration dimension only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HISTORICAL = HERE.parent / "mi_135_live_bank_00bb3e9d"
MANIFEST = HERE / "certification_manifest.json"
HASH = HERE / "certification_manifest.sha256"

BANK_ID = "MI_135_RELEASE_CERTIFICATION_V3"
PRODUCT_SHA = "5c436961ebe279bc0820ab006867b9f8a869bde2"

#: The frozen eight. Definitions are the operator's, recorded verbatim so the
#: scorecard cannot drift between the brief and the report.
VERDICTS = {
    "FULLY_CORRECT":
        "correct interpretation; correct population/scope; correct period; "
        "correct requested metric/dimension/filter; correct deterministic "
        "owner; the answer actually communicates the requested result; no "
        "silent semantic loss",
    "PARTIALLY_CORRECT":
        "materially useful and directionally/categorically correct, but an "
        "explicit requested component is missing, qualified or incomplete; no "
        "wrong figure and no silent widening",
    "HONEST_REFUSAL":
        "the system cannot safely answer under current governed capability or "
        "data; the refusal states the genuine limitation; nothing is fabricated "
        "or silently widened",
    "BAD_REFUSAL":
        "capability and data exist and the governed system should have "
        "answered; the refusal is caused by routing, connectivity or semantic "
        "failure",
    "WRONG":
        "wrong number, population, period, measure, filter or dimension; wrong "
        "deterministic owner producing the wrong answer; an unsupported "
        "confident answer; a materially misleading answer",
    "APPROPRIATE_CLARIFICATION":
        "ambiguity genuinely prevents one governed interpretation and the "
        "existing contract requires the clarification",
    "INCONCLUSIVE":
        "the evidence is insufficient to classify the product answer; the "
        "missing evidence must be stated",
    "INFRASTRUCTURE":
        "the request does not reach or complete through the application "
        "because of transport, platform, auth or service infrastructure; no "
        "product conclusion may be inferred from it",
}

#: score_135.py's own vocabulary -> the frozen eight. Fixed before the run.
USER_OUTCOME_MAP = {
    "FULLY_CORRECT": "FULLY_CORRECT",
    "PARTIALLY_CORRECT": "PARTIALLY_CORRECT",
    "WRONG": "WRONG",
    "APPROPRIATE_CLARIFICATION": "APPROPRIATE_CLARIFICATION",
    "UNNECESSARY_CLARIFICATION": "BAD_REFUSAL",
    "PLAN_OR_CONNECTIVITY_GAP_MASQUERADING_AS_CLARIFY": "BAD_REFUSAL",
    "HONEST_REFUSAL": "HONEST_REFUSAL",
    "BAD_REFUSAL": "BAD_REFUSAL",
    "INFRASTRUCTURE_FAILURE": "INFRASTRUCTURE",
    "INCONCLUSIVE": "INCONCLUSIVE",
}

SERVING_PROVENANCE_MAP = {
    "NEW": "GOVERNED_PLAN",
    "LEGACY_FALLBACK": "LEGACY_FALLBACK",
    "NOT_TESTED_KNOWN_UNMIGRATED": "LEGACY_FALLBACK",
    "NOT_REACHED": "NONE",
    "INFRASTRUCTURE_FAILURE": "NONE",
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    questions = HISTORICAL / "questions.json"
    overlay = HISTORICAL / "expected_capability.json"
    scorer = HISTORICAL / "score_135.py"
    collector = HISTORICAL / "collect_135.py"

    body = json.loads(questions.read_text(encoding="utf-8"))
    rows = body["questions"]
    if len(rows) != 135:
        raise SystemExit(f"the bank must be 135 questions; this one is {len(rows)}")
    if len({r["question"] for r in rows}) != 135:
        raise SystemExit("duplicate question text; the evidence poller matches "
                         "on question text and could not tell two apart")

    return {
        "bank_id": BANK_ID,
        "question_count": len(rows),
        "product_sha": PRODUCT_SHA,
        "expected_deployed_sha": PRODUCT_SHA,
        "supersedes": {
            "attempt": "MI_135_RELEASE_CERTIFICATION_V2",
            "outcome": "INFRASTRUCTURE_BLOCKED",
            "why": "the bearer expired at question 53 and the scorer could not "
                   "parse a recorded intent carrying the derived time.stated; "
                   "both are test-instrument defects and neither is a product "
                   "finding. V2's evidence is retained as diagnostic only and is "
                   "not merged with this attempt.",
            "v2_salvaged_53_not_to_be_reused":
                "18 FULLY_CORRECT, 26 PARTIALLY_CORRECT, 9 HONEST_REFUSAL, "
                "0 WRONG over a 53-question PREFIX. Not a certification, not "
                "extrapolated, not merged.",
        },
        "instrument_repairs_since_v2": [
            "score_135._score_interpretation now projects NESTED intent blocks "
            "back through the parser's own per-slot key tuples, not the top "
            "level alone. time.stated remains DERIVED and is NOT admitted to "
            "the model-facing _TIME_KEYS.",
            "collect_135 aborts after N consecutive 401/403 and never retries "
            "an expired bearer.",
        ],
        "what_this_is":
            "The release certification of the existing 135-question MI estate "
            "against the current candidate. The questions are the SAME 135 in "
            "the SAME file the historical run used; no wording is rewritten and "
            "no expectation is changed. Only the product under test and the "
            "output directory differ.",
        "reuses_unmodified": {
            "questions": {"path": str(questions.relative_to(HERE.parents[2])),
                          "sha256": sha256_of(questions),
                          "inner_bank_sha256": body.get("bank_sha256")},
            "expectation_overlay": {"path": str(overlay.relative_to(HERE.parents[2])),
                                    "sha256": sha256_of(overlay)},
            "scoring_contract": {"path": str(scorer.relative_to(HERE.parents[2])),
                                 "sha256": sha256_of(scorer)},
            "collector": {"path": str(collector.relative_to(HERE.parents[2])),
                          "sha256": sha256_of(collector)},
        },
        "expectation_overlay_changes": [],
        "expectation_overlay_note":
            "EMPTY, DELIBERATELY. The brief permits an overlay for already-"
            "authorised target-state migrations. None is applied: every "
            "migration this programme authorised (material_summary, "
            "attribution, metric_delta, the requested-metric output contract) "
            "changes WHICH OWNER serves a question, not what the question "
            "means, and the overlay expresses meaning. Adding entries here "
            "would move the goalposts before the run rather than after it.",
        "verdicts": VERDICTS,
        "user_outcome_map": USER_OUTCOME_MAP,
        "serving_provenance_map": SERVING_PROVENANCE_MAP,
        "dimensions_recorded_separately": [
            "SERVING_PROVENANCE: GOVERNED_PLAN | LEGACY_FALLBACK | REFUSAL | NONE",
            "INTERPRETATION: CORRECT | PARTIAL | WRONG | NOT_REACHED",
            "EXECUTION: CORRECT_OWNER | WRONG_OWNER | NOT_EXECUTED",
            "OUTPUT_CONTRACT: SATISFIED | PARTIAL | FAILED | NOT_APPLICABLE",
        ],
        "numeric_truth":
            "No independent LIVE numeric parity is claimed. The bank is an "
            "interpretation benchmark and carries expected SEMANTICS, not "
            "expected numbers; the scorer reports numeric_parity as N/A "
            "everywhere. Deterministic owner receipts are used to assess owner, "
            "requested field, period and scope. Offline owner parity remains "
            "valid supporting evidence and is not relabelled as live parity.",
        "historical_baseline": {
            "product_sha": "00bb3e9d8175a395b6643772379866d6bc6169eb",
            "results": str((HISTORICAL / "MI_135_LIVE_BANK_RESULTS.json")
                           .relative_to(HERE.parents[2])),
            "immutable": True,
        },
        "run_rules": [
            "135 questions, once each",
            "no retries on a semantic result; a transport/HTTP failure is "
            "classified INFRASTRUCTURE and not re-asked",
            "no rephrasing, substitution, expectation change, prompt tuning or "
            "model switch",
            "no product or configuration fix during the run",
            "evidence persisted and committed before the workflow exits",
        ],
    }


def main() -> int:
    if MANIFEST.exists() and "--force" not in sys.argv:
        print(f"refusing to overwrite {MANIFEST.name}: it is pinned evidence.")
        return 1
    payload = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    MANIFEST.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    HASH.write_text(f"{digest}  {MANIFEST.name}\n", encoding="utf-8")
    print(f"wrote {MANIFEST.name}")
    print(f"BANK_ID                    {BANK_ID}")
    print(f"QUESTION_COUNT             135")
    print(f"CERTIFICATION_MANIFEST_SHA {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
