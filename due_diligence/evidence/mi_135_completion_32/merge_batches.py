"""Join the tranche batches into one evidence file, once each and in bank order.

A case may appear in exactly one batch. Two records for one case would mean the
tranche asked it twice, which is the one thing "ONE LIVE ATTEMPT" forbids, so it
is an error rather than a de-duplication.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = json.loads((HERE / "completion_cases.json").read_text(encoding="utf-8"))["cases"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batches", nargs="+")
    parser.add_argument("--out", default=str(HERE / "raw_records.json"))
    parser.add_argument("--partial-ok", action="store_true",
                        help="write what arrived even if the tranche stopped short")
    args = parser.parse_args()

    header, byid = None, {}
    for path in args.batches:
        p = Path(path)
        if not p.exists():
            continue
        body = json.loads(p.read_text(encoding="utf-8"))
        header = header or body
        for record in body.get("records") or []:
            qid = record["question_id"]
            if qid in byid:
                raise SystemExit(f"{qid} was asked in more than one batch")
            byid[qid] = record

    unexpected = sorted(set(byid) - set(CASES))
    if unexpected:
        raise SystemExit(f"a batch asked a case outside the tranche: {unexpected}")
    ordered = [byid[q] for q in CASES if q in byid]
    short = [q for q in CASES if q not in byid]

    out = dict(header or {})
    out["records"] = ordered
    out["what_this_is"] = ("raw live evidence for the completion tranche; "
                           "unscored")
    out["question_count"] = len(ordered)
    out["tranche_case_count"] = len(CASES)
    out["not_asked"] = short
    out["verdict"] = "COLLECTED" if not short else "STOPPED_SHORT"
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"merged {len(ordered)} of {len(CASES)} tranche cases -> {args.out}")
    if short:
        print(f"NOT ASKED ({len(short)}): {', '.join(short)}")
        if not args.partial_ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
