"""Build the composite 135 from two runs, and prove which record came from where.

NEITHER SOURCE IS REWRITTEN. Both are opened read-only and hashed; the composite
is a third file that carries, for every one of the 135 cases, the run it came
from and that run's sha256. Selecting the better-looking of two results for a
case is impossible here rather than merely discouraged: a case is taken from the
completion tranche if and only if the frozen manifest lists it, and from the
source run otherwise, and the two lists are disjoint by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REPO = HERE.parents[2]
V3 = _REPO / "due_diligence/evidence/mi_135_release_certification_v3"

MANIFEST = json.loads((HERE / "completion_manifest.json").read_text(encoding="utf-8"))
CASES = json.loads((HERE / "completion_cases.json").read_text(encoding="utf-8"))["cases"]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def named(path: Path) -> str:
    """Repo-relative where it can be, absolute otherwise (a dry run elsewhere)."""
    try:
        return str(path.relative_to(_REPO))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion", default=str(HERE / "raw_records.json"))
    parser.add_argument("--out", default=str(HERE / "composite_raw_records.json"))
    parser.add_argument("--manifest-out",
                        default=str(HERE / "composite_manifest.json"))
    args = parser.parse_args()

    source_path = V3 / "raw_records.json"
    completion_path = Path(args.completion)

    # THE SOURCE RUN MUST BE THE ONE THAT WAS FROZEN. A composite built on a
    # source that changed after the tranche was frozen would silently mix two
    # different 103s.
    want = MANIFEST["pins"]["source_raw_records"]["sha256"]
    got = sha256_of(source_path)
    if got != want:
        raise SystemExit(f"the source run changed since the freeze: {got}")

    source = json.loads(source_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    tranche = set(CASES)

    kept = [r for r in source["records"] if r["question_id"] not in tranche]
    fresh = {r["question_id"]: r for r in completion["records"]}

    if len(kept) != 135 - len(tranche):
        raise SystemExit(f"the source run yielded {len(kept)} measured cases")
    missing = sorted(tranche - set(fresh))
    if missing:
        raise SystemExit(f"the completion tranche is short: {missing}")
    extra = sorted(set(fresh) - tranche)
    if extra:
        raise SystemExit(f"the completion tranche asked cases it must not: {extra}")

    records, provenance = [], {}
    for record in source["records"]:
        qid = record["question_id"]
        if qid in tranche:
            records.append(fresh[qid])
            provenance[qid] = "COMPLETION_32"
        else:
            records.append(record)
            provenance[qid] = "SOURCE_103"

    ids = [r["question_id"] for r in records]
    duplicates = sorted({q for q in ids if ids.count(q) > 1})
    if duplicates:
        raise SystemExit(f"duplicate case ids in the composite: {duplicates}")
    if len(records) != 135:
        raise SystemExit(f"the composite holds {len(records)} cases, not 135")

    composite = dict(source)
    composite["records"] = records
    composite["what_this_is"] = (
        "composite raw evidence: the 103 the source run measured, plus the 32 "
        "it left unmeasured, re-asked once each; unscored")
    composite["verdict"] = "COMPOSITE"
    composite["composed_from"] = {
        "source_103": {"path": named(source_path),
                       "sha256": got, "cases": 135 - len(tranche)},
        "completion_32": {"path": named(completion_path),
                          "sha256": sha256_of(completion_path),
                          "cases": len(tranche)},
    }
    body = json.dumps(composite, indent=1) + "\n"
    Path(args.out).write_text(body, encoding="utf-8")

    manifest = {
        "composite_bank_id": "MI_135_RELEASE_CERTIFICATION_V2_COMPLETE",
        "composite_case_count": len(records),
        "duplicate_case_ids": len(duplicates),
        "missing_case_ids": len(missing),
        "product_sha": MANIFEST["product_sha"],
        "composed_from": composite["composed_from"],
        "completion_manifest_sha256": hashlib.sha256(
            (HERE / "completion_manifest.json").read_bytes()).hexdigest(),
        "composite_raw_records_sha256": hashlib.sha256(
            body.encode("utf-8")).hexdigest(),
        "case_provenance": provenance,
    }
    Path(args.manifest_out).write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    print(f"COMPOSITE_CASE_COUNT = {len(records)}")
    print(f"DUPLICATES           = {len(duplicates)}")
    print(f"MISSING              = {len(missing)}")
    print(f"  from SOURCE_103    = {sum(1 for v in provenance.values() if v == 'SOURCE_103')}")
    print(f"  from COMPLETION_32 = {sum(1 for v in provenance.values() if v == 'COMPLETION_32')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
