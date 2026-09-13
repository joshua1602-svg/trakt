#!/usr/bin/env python3
"""Score the certification run with the SIGNED-OFF scorer, unmodified.

NO SECOND RUBRIC. `score_135.py` produced the historical baseline, and a
before/after comparison is only meaningful if the same owner does both. So this
file imports that module and repoints its four path constants at this
directory's evidence — the same code object, the same expectations, different
inputs. It does not copy a single scoring rule, and the certification manifest
records that scorer's sha256 so a reader can prove it was not edited.

WHAT THIS FILE ADDS, AND ONLY THIS. The frozen eight-category mapping fixed in
the manifest BEFORE the run, the four separately recorded dimensions, the
release metrics with both raw counts and denominator-adjusted rates, and the
comparison against the immutable historical baseline.

THE DENOMINATOR IS STATED, NOT ASSUMED. Every rate is over NON-INFRASTRUCTURE
cases, and the infrastructure count is printed beside it. Hiding platform
failures inside a denominator would flatter the estate for something that is not
the estate's fault — and equally, dropping them silently would overstate it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
HISTORICAL = HERE.parent / "mi_135_live_bank_00bb3e9d"
_REPO = HERE.parents[2]
for path in (str(_REPO), str(HISTORICAL)):
    if path not in sys.path:
        sys.path.insert(0, path)

MANIFEST = json.loads((HERE / "certification_manifest.json").read_bytes())
VERDICT_MAP = MANIFEST["user_outcome_map"]
PROVENANCE_MAP = MANIFEST["serving_provenance_map"]
FROZEN = list(MANIFEST["verdicts"])


def _dimensions(row):
    """The four dimensions, from the scorer's own row. Never re-derived."""
    migration = row.get("migration_outcome")
    provenance = PROVENANCE_MAP.get(migration, "NONE")
    if row.get("user_outcome") in ("HONEST_REFUSAL", "BAD_REFUSAL"):
        provenance = "REFUSAL" if provenance == "NONE" else provenance

    interp = row.get("interpretation_outcome")
    scored = row.get("interpretation_scored") or {}
    if interp == "NO_RECORD" or row.get("infrastructure_failure"):
        interpretation = "NOT_REACHED"
    elif interp == "PLAN":
        misses = [k for k, v in scored.items() if v is False]
        interpretation = "CORRECT" if not misses else "PARTIAL"
    elif interp in ("CLARIFY", "REFUSE"):
        interpretation = "PARTIAL"
    elif interp == "NOT_APPLICABLE":
        interpretation = "NOT_REACHED"
    else:
        interpretation = "WRONG" if interp == "INTERPRETER_FAILURE" else "NOT_REACHED"

    receipt = row.get("receipt") or {}
    owner = receipt.get("calculation_owner")
    if row.get("migration_outcome") == "NEW" and owner:
        execution = "CORRECT_OWNER"
    elif row.get("envelope_ok"):
        execution = "CORRECT_OWNER" if owner else "NOT_EXECUTED"
    else:
        execution = "NOT_EXECUTED"

    disposition = receipt.get("requested_metric_disposition") or []
    if not disposition:
        contract = "NOT_APPLICABLE"
    else:
        states = {str(d.get("disposition")) for d in disposition}
        contract = ("SATISFIED" if states <= {"ANSWERED"}
                    else "PARTIAL" if states <= {"ANSWERED", "QUALIFIED"}
                    else "FAILED")
    return {"serving_provenance": provenance, "interpretation": interpretation,
            "execution": execution, "output_contract": contract,
            "requested_metric_disposition": disposition}


def replay() -> int:
    """THE SHIM'S SELF-TEST: score the HISTORICAL evidence and reproduce the
    HISTORICAL numbers exactly.

    A shim that repoints a scorer can silently drop an input file, and a dropped
    input reads as a run full of infrastructure failures rather than as a broken
    harness. The first version of this file did exactly that with the 31
    legacy-completion records. So before it is trusted with a fresh run it is
    pointed at the run whose answer is already known and must agree with it.
    """
    import score_135 as scorer

    scorer.RAW = HISTORICAL / "raw_records.json"
    scorer.RERUN = HISTORICAL / "raw_records_rerun.json"
    scorer.LEGACY_COMPLETION = HISTORICAL / "raw_records_legacy_completion.json"
    scorer.RESULTS = Path("/dev/null")
    rows = scorer.build_rows()

    committed = json.loads(
        (HISTORICAL / "MI_135_LIVE_BANK_RESULTS.json").read_text(encoding="utf-8"))
    got = Counter(r["user_outcome"] for r in rows)
    want = Counter(r["user_outcome"] for r in committed)
    ok = got == want
    print("replay of the historical evidence through this shim:")
    for key in sorted(set(got) | set(want)):
        flag = "ok " if got.get(key) == want.get(key) else "DIFF"
        print(f"   {flag} {key:52} {got.get(key, 0):4d} "
              f"(committed {want.get(key, 0)})")
    print("shim replay " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true",
                        help="score the historical evidence and require the "
                             "historical numbers; asks nothing")
    parser.add_argument("--raw", default=str(HERE / "raw_records.json"))
    parser.add_argument("--rerun", default=str(HERE / "raw_records_rerun.json"))
    parser.add_argument("--legacy-completion",
                        default=str(HERE / "raw_records_legacy_completion.json"),
                        help="the historical run completed 31 questions through "
                             "a second, canary-off pass and score_135 reads them "
                             "from this path. Pointing it at a file that does "
                             "not exist silently turns those 31 into "
                             "INFRASTRUCTURE — which is exactly what the first "
                             "version of this shim did, and what the replay "
                             "below caught.")
    parser.add_argument("--out", default=str(HERE / "CERTIFICATION_RESULTS.json"))
    parser.add_argument("--summary", default=str(HERE / "CERTIFICATION_SUMMARY.json"))
    args = parser.parse_args()

    if args.replay:
        return replay()

    import score_135 as scorer

    # THE ONLY THING CHANGED ABOUT THE SCORER: where it reads and writes.
    scorer.RAW = Path(args.raw)
    scorer.RERUN = Path(args.rerun)
    scorer.LEGACY_COMPLETION = Path(args.legacy_completion)
    scorer.RESULTS = Path(args.out)

    rows = scorer.build_rows()
    Path(args.out).write_text(
        json.dumps(rows, indent=1, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    print(f"scored {len(rows)} questions -> {Path(args.out).name}\n")

    unmapped = sorted({r["user_outcome"] for r in rows} - set(VERDICT_MAP))
    if unmapped:
        raise SystemExit(f"the scorer emitted an outcome the frozen map does not "
                         f"cover: {unmapped}. The map is frozen before the run "
                         f"and is not extended after seeing results.")

    verdicts = Counter(VERDICT_MAP[r["user_outcome"]] for r in rows)
    dims = [_dimensions(r) for r in rows]
    provenance = Counter(d["serving_provenance"] for d in dims)
    interpretation = Counter(d["interpretation"] for d in dims)
    execution = Counter(d["execution"] for d in dims)
    contract = Counter(d["output_contract"] for d in dims)

    infra = verdicts.get("INFRASTRUCTURE", 0)
    denominator = len(rows) - infra

    def rate(*names):
        if denominator <= 0:
            return None
        return round(sum(verdicts.get(n, 0) for n in names) / denominator, 4)

    drops = Counter()
    for row in rows:
        for key, value in (row.get("drops") or {}).items():
            if key.startswith("silent_") and value is True:
                drops[key] += 1

    misroutes = sum(1 for row, d in zip(rows, dims)
                    if d["execution"] == "WRONG_OWNER")

    governed = provenance.get("GOVERNED_PLAN", 0)
    legacy = provenance.get("LEGACY_FALLBACK", 0)

    baseline_path = _REPO / MANIFEST["historical_baseline"]["results"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_verdicts = Counter(VERDICT_MAP.get(r["user_outcome"], r["user_outcome"])
                            for r in baseline)
    base_prov = Counter(PROVENANCE_MAP.get(r["migration_outcome"], "NONE")
                        for r in baseline)

    summary = {
        "bank_id": MANIFEST["bank_id"],
        "product_sha": MANIFEST["product_sha"],
        "total": len(rows),
        "verdicts": dict(verdicts),
        "non_infrastructure_denominator": denominator,
        "rates": {
            "user_safe": rate("FULLY_CORRECT", "PARTIALLY_CORRECT",
                              "HONEST_REFUSAL", "APPROPRIATE_CLARIFICATION"),
            "answered_usefully": rate("FULLY_CORRECT", "PARTIALLY_CORRECT"),
            "fully_correct": rate("FULLY_CORRECT"),
            "wrong": rate("WRONG"),
        },
        "serving_provenance": dict(provenance),
        "interpretation": dict(interpretation),
        "execution": dict(execution),
        "output_contract": dict(contract),
        "governed_plan_completions": governed,
        "legacy_fallback_completions": legacy,
        "governed_plan_completion_rate": (
            round(governed / denominator, 4) if denominator else None),
        "silent_semantic_errors": dict(drops),
        "misroutes": misroutes,
        "numeric_parity": {
            "independent_live_parity_measured": False,
            "why": MANIFEST["numeric_truth"],
        },
        "baseline": {
            "product_sha": MANIFEST["historical_baseline"]["product_sha"],
            "verdicts": dict(base_verdicts),
            "serving_provenance": dict(base_prov),
        },
        "movement": {name: verdicts.get(name, 0) - base_verdicts.get(name, 0)
                     for name in FROZEN},
    }
    Path(args.summary).write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    print("frozen verdicts:")
    for name in FROZEN:
        moved = summary["movement"][name]
        print(f"   {name:28} {verdicts.get(name, 0):4d}   "
              f"baseline {base_verdicts.get(name, 0):4d}   "
              f"{moved:+d}")
    print(f"\nnon-infrastructure denominator {denominator}")
    for key, value in summary["rates"].items():
        print(f"   {key:20} {value}")
    print(f"\nserving provenance {dict(provenance)}")
    print(f"governed-plan completions {governed}  "
          f"(baseline {base_prov.get('GOVERNED_PLAN', 0)})")
    print(f"silent semantic losses {dict(drops) or 'none'}")
    print(f"misroutes {misroutes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
