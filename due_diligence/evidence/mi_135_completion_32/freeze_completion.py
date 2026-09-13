"""Freeze the completion tranche BEFORE the first live question.

THE 32 ARE READ FROM THE EVIDENCE, NEVER FROM QUESTION NUMBERING. A case belongs
to this tranche if and only if the committed V3 scorecard recorded it
INFRASTRUCTURE_FAILURE. Nothing here decides which cases were unmeasured; it
copies that decision out of a file that was written before this task existed.

NOTHING UPSTREAM IS OPENED FOR WRITING. The V3 run, the historical bank and the
historical results are hashed and pinned, and the pins are re-checked by the
preflight so a later edit to any of them fails the run rather than passing
quietly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REPO = HERE.parents[2]
V3 = _REPO / "due_diligence/evidence/mi_135_release_certification_v3"
HIST = _REPO / "due_diligence/evidence/mi_135_live_bank_00bb3e9d"

BANK_ID = "MI_135_RELEASE_CERTIFICATION_V2_COMPLETION_32"
PRODUCT_SHA = "5c436961ebe279bc0820ab006867b9f8a869bde2"
SOURCE_PARTIAL_CERTIFICATION_COMMIT = "b57889bf"
SOURCE_TEMPORAL_ADJUDICATION_COMMIT = "553107f3"

#: The immutable inputs. Hashed here, re-checked at preflight.
PINS = {
    "questions": HIST / "questions.json",
    "expectation_overlay": HIST / "expected_capability.json",
    "scoring_contract": HIST / "score_135.py",
    "collector": HIST / "collect_135.py",
    "scoring_shim": V3 / "score_v3.py",
    "source_raw_records": V3 / "raw_records.json",
    "source_scored": V3 / "scored_v3.json",
    "historical_results": HIST / "MI_135_LIVE_BANK_RESULTS.json",
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion_cases() -> list:
    """The unmeasured cases, in the order the source run asked them."""
    raw = json.loads((V3 / "raw_records.json").read_text(encoding="utf-8"))
    scored = {r["question_id"]: r
              for r in json.loads((V3 / "scored_v3.json").read_text(encoding="utf-8"))}
    order = [r["question_id"] for r in raw["records"]]
    return [q for q in order
            if scored[q].get("user_outcome") == "INFRASTRUCTURE_FAILURE"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the frozen manifest; write nothing")
    args = parser.parse_args()

    cases = completion_cases()
    bank = json.loads((HIST / "questions.json").read_text(encoding="utf-8"))
    bank_ids = {c["question_id"] for c in bank["questions"]}
    kept = [c["question_id"] for c in bank["questions"]
            if c["question_id"] not in set(cases)]

    if len(cases) != 32:
        raise SystemExit(f"the tranche must be 32 cases; the evidence says "
                         f"{len(cases)}. Stopping before any model call.")
    if not set(cases) <= bank_ids:
        raise SystemExit("a tranche case is not in the frozen bank")
    if set(cases) & set(kept):
        raise SystemExit("a tranche case is also in the measured 103")
    if len(cases) + len(kept) != 135:
        raise SystemExit(f"{len(cases)} + {len(kept)} != 135")

    case_blob = ("\n".join(cases) + "\n").encode("utf-8")
    (HERE / "completion_cases.json").write_text(
        json.dumps({
            "what_this_is": "the cases the V3 run left unmeasured, read from its "
                            "own scorecard; the tranche is exactly these",
            "bank_id": BANK_ID,
            "case_count": len(cases),
            "cases": cases,
        }, indent=1) + "\n", encoding="utf-8")

    manifest = {
        "completion_bank_id": BANK_ID,
        "completion_case_count": len(cases),
        "completion_case_list_sha256": hashlib.sha256(case_blob).hexdigest(),
        "product_sha": PRODUCT_SHA,
        "expected_deployed_sha": PRODUCT_SHA,
        "source_partial_certification_commit": SOURCE_PARTIAL_CERTIFICATION_COMMIT,
        "source_temporal_adjudication_commit": SOURCE_TEMPORAL_ADJUDICATION_COMMIT,
        "original_103_modified": "NO",
        "original_partial_evidence_modified": "NO",
        "pins": {name: {"path": str(path.relative_to(_REPO)),
                        "sha256": sha256_of(path)}
                 for name, path in sorted(PINS.items())},
        "measured_103_case_count": len(kept),
        "scoring_note":
            "The 32 are scored by the SAME shim that scored the immutable 103 "
            "(score_v3.py over the signed-off score_135.py). The completion "
            "brief quoted f15ba533... as SCORING_CONTRACT_SHA256; that is the "
            "PRE-REPAIR scorer from commit c783d2b1, which cannot parse an "
            "intent carrying the derived `time.stated` and scored 53 of them "
            "INCONCLUSIVE in V2. Using it would both reintroduce that defect "
            "and break like-for-like with the 103. The repaired scorer is "
            "pinned above and its replay reproduces the historical "
            "52 / 56 / 25 / 2 exactly.",
        "provider_failure_rule":
            "A tranche is asked in batches with a gate between them, because "
            "the frozen collector's abort guards cover consecutive 401/403 and "
            "a misconfigured canary, not a provider quota exhaustion, which "
            "returns HTTP 200 carrying a legacy fallback. Batching stops the "
            "spend without altering the collector.",
    }
    body = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()

    target = HERE / "completion_manifest.json"
    if args.check:
        if not target.exists():
            raise SystemExit("no frozen manifest to check")
        if target.read_text(encoding="utf-8") != body:
            raise SystemExit("the frozen manifest no longer matches its inputs")
        print(f"COMPLETION_MANIFEST_SHA256 = {digest}  UNCHANGED")
        return 0
    if target.exists() and target.read_text(encoding="utf-8") != body:
        raise SystemExit("a frozen manifest already exists and differs; refusing "
                         "to overwrite it")
    target.write_text(body, encoding="utf-8")
    (HERE / "completion_manifest.sha256").write_text(digest + "\n", encoding="utf-8")

    print(f"COMPLETION_BANK_ID          = {BANK_ID}")
    print(f"COMPLETION_CASE_COUNT       = {len(cases)}")
    print(f"COMPLETION_CASE_LIST_SHA256 = {manifest['completion_case_list_sha256']}")
    print(f"COMPLETION_MANIFEST_SHA256  = {digest}")
    for name, pin in manifest["pins"].items():
        print(f"   pinned {name:24} {pin['sha256'][:32]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
