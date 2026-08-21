#!/usr/bin/env python3
"""question_interpretation/answer_diff.py

Standing condition 7: every stage diffs answer TEXT, not only grades.

Byte-identical answer text is strictly stricter than the calibration bank's
grader. The bounded pre-Stage-2 check found four cases whose
`expected_answer_type: mixed` is satisfied by an observed type of `count`, so a
portfolio summary that silently dropped its balance measure would PASS the bank.
It would not pass this. That is why the grading path is left alone.

    python -m question_interpretation.answer_diff --record baseline.json
    python -m question_interpretation.answer_diff --against baseline.json

Records, per question, the exact bytes a user would see — the answer text, the
error text, the ok flag, the artifact shapes and the warnings — hashed and
stored. `--against` re-runs and reports every byte that moved.

Both surfaces, deterministic arm:
    calibration bank    252 questions, real alderbridge tape
    robustness bank      44 questions x 2 books, through /mi/query

Read-only with respect to production. Writes only the file it is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_EVIDENCE = _REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"
for p in (str(_REPO_ROOT), str(_EVIDENCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

_REAL_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
              / "platform" / "alderbridge" / "2026-06-30"
              / "platform_canonical_typed.csv")


def _user_visible(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Everything a user would actually read, and nothing else.

    Deliberately excludes the spec, the receipt and any metadata: Stage 2 is
    expected to ADD metadata (the interpretation object), and a diff that
    flagged that addition would be measuring the wrong thing. What must not
    move is what reaches the reader.
    """
    artifacts = payload.get("artifacts") or []
    return {
        "ok": payload.get("ok"),
        "answer": (payload.get("answer") or "").strip(),
        "error": str(payload.get("error") or "").strip(),
        "controlledRefusal": payload.get("controlledRefusal", False),
        "warnings": [str(w) for w in (payload.get("warnings") or [])],
        "artifacts": [
            {"type": a.get("type"), "title": a.get("title"),
             "rows": len(a.get("rows") or []),
             "columns": [str(c) for c in (a.get("columns") or [])]}
            for a in artifacts if isinstance(a, dict)
        ],
        "reconciliation": payload.get("reconciliation"),
        "executionSummary": payload.get("executionSummary"),
    }


def _digest(record: Dict[str, Any]) -> str:
    blob = json.dumps(record, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Surface 1 — the calibration bank, through the same workflow the bank grades.
# --------------------------------------------------------------------------- #
def _calibration_records() -> List[Dict[str, Any]]:
    from mi_agent import mi_calibration as CAL
    from mi_agent.mi_query_validator import load_mi_semantics

    semantics = load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    df = CAL.default_bank_frame(_REAL_TAPE)
    out: List[Dict[str, Any]] = []
    for case in CAL.load_bank():
        res = CAL._run(case["question"], df, semantics, False)
        adapted = {}
        try:
            from mi_agent_api.adapters import adapt_workflow_result
            adapted = adapt_workflow_result(res, portfolio_id="client_001",
                                            as_of=None) or {}
        except Exception:                                   # noqa: BLE001
            adapted = {}
        payload = dict(res)
        payload["artifacts"] = adapted.get("artifacts") or []
        payload["reconciliation"] = adapted.get("reconciliation")
        record = _user_visible(payload)
        out.append({"surface": "calibration_bank", "id": case["id"],
                    "question": case["question"], "digest": _digest(record),
                    "record": record})
    return out


# --------------------------------------------------------------------------- #
# Surface 2 — the robustness bank, through /mi/query, one book per process.
# --------------------------------------------------------------------------- #
def _robustness_records(book: str) -> List[Dict[str, Any]]:
    import nl_bank
    from question_interpretation.run_robustness_deterministic import _client_for

    client, portfolio = _client_for(book)
    out: List[Dict[str, Any]] = []
    for entry in nl_bank.bank(book):
        payload = client.post("/mi/query", json={
            "question": entry["question"], "portfolioId": portfolio,
            "datasetContext": "funded"}).json()
        record = _user_visible(payload)
        # Q8 runs the SAME four sentence shapes against three governed X/Y
        # pairs — provenance, seasoning and dimension values — so (intent,
        # variation) repeats three times per book. Keying on it alone collapsed
        # 12 Q8 rows to 4 and silently dropped 16 of the 88 robustness answers,
        # which are precisely the seasoning family Stage 4 must leave unmoved.
        # The pair is part of the identity.
        pair = entry.get("pair")
        ident = "%s.%s" % (entry["intent"], entry["variation"])
        if pair:
            ident = "%s[%s]" % (ident, pair if isinstance(pair, str)
                                else "|".join(str(x) for x in pair))
        out.append({"surface": "robustness_%s" % book, "id": ident,
                    "question": entry["question"], "digest": _digest(record),
                    "record": record})
    return out


def _robustness_via_subprocess(book: str) -> List[Dict[str, Any]]:
    """One book per process: the app binds its book at import time.

    EVERY argument that changes what is measured must be forwarded. The first
    version forwarded only --only-book, so a variant run measured the DEFAULT
    on the 88 robustness records and reported it as the variant, while the 252
    calibration records — collected in-process — did move. A split like that
    reads as "the variant only affects the calibration bank", which is a
    conclusion about the product drawn from a defect in the instrument.
    """
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out_path = Path(fh.name)
    argv = [sys.executable, "-m", __spec__.name, "--only-book", book,
            "--record", str(out_path)]
    proc = subprocess.run(argv, cwd=str(_REPO_ROOT), capture_output=True,
                          text=True)
    if proc.returncode != 0:
        raise RuntimeError("book %s failed:\n%s" % (book, proc.stdout + proc.stderr))
    data = json.loads(out_path.read_text(encoding="utf-8"))
    out_path.unlink(missing_ok=True)
    return data["records"]


def collect(only_book: Optional[str] = None) -> List[Dict[str, Any]]:
    logging.disable(logging.WARNING)
    if only_book:
        return _robustness_records(only_book)
    records = _calibration_records()
    for book in ("alderbridge", "kestrelmoor"):
        records.extend(_robustness_via_subprocess(book))
    return records


def _keyed(records: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    """Index by (surface, id), refusing to collapse two records onto one key.

    A differ that silently drops a case is the instrument defect this
    programme has been caught by repeatedly. Keying is checked, not assumed.
    """
    out: Dict[Any, Dict[str, Any]] = {}
    for r in records:
        key = (r["surface"], r["id"])
        if key in out:
            raise ValueError(
                "duplicate answer-diff key %r — two records would collapse "
                "onto one and %d case(s) would vanish from the comparison; "
                "questions: %r and %r"
                % (key, 1, out[key]["question"], r["question"]))
        out[key] = r
    return out


def compare(before: List[Dict[str, Any]],
            after: List[Dict[str, Any]]) -> Dict[str, Any]:
    b = _keyed(before)
    a = _keyed(after)
    shared = sorted(set(b) & set(a))
    moved = [k for k in shared if b[k]["digest"] != a[k]["digest"]]
    return {
        "compared": len(shared),
        "only_before": sorted("%s/%s" % k for k in (set(b) - set(a))),
        "only_after": sorted("%s/%s" % k for k in (set(a) - set(b))),
        "moved": [{"surface": k[0], "id": k[1], "question": b[k]["question"],
                   "before": b[k]["record"], "after": a[k]["record"]}
                  for k in moved],
        "identical": len(shared) - len(moved),
    }


def _describe_move(entry: Dict[str, Any]) -> List[str]:
    out = []
    for field in ("ok", "answer", "error", "controlledRefusal", "warnings",
                  "artifacts", "reconciliation", "executionSummary"):
        bv, av = entry["before"].get(field), entry["after"].get(field)
        if bv != av:
            out.append("      %s:\n        before: %s\n        after:  %s"
                       % (field, str(bv)[:200], str(av)[:200]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", type=Path, help="collect and write a baseline")
    ap.add_argument("--against", type=Path, help="collect and diff against one")
    ap.add_argument("--only-book", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    records = collect(only_book=args.only_book)

    if args.record:
        args.record.write_text(
            json.dumps({"records": records}, indent=2, default=str),
            encoding="utf-8")
        if not args.only_book:
            print("recorded %d answers across %d surfaces -> %s"
                  % (len(records),
                     len({r["surface"] for r in records}), args.record))
        return 0

    if not args.against:
        ap.error("one of --record or --against is required")

    before = json.loads(args.against.read_text(encoding="utf-8"))["records"]
    result = compare(before, records)
    print("=" * 72)
    print("ANSWER TEXT DIFF — byte-identical is the acceptance")
    print("=" * 72)
    print("compared   %4d" % result["compared"])
    print("identical  %4d" % result["identical"])
    print("MOVED      %4d" % len(result["moved"]))
    for k in ("only_before", "only_after"):
        if result[k]:
            print("%s: %s" % (k, result[k][:10]))
    for entry in result["moved"][:25]:
        print("\n  %s/%s\n    %s" % (entry["surface"], entry["id"],
                                     entry["question"][:88]))
        for line in _describe_move(entry):
            print(line)
    print("\n%s" % ("-" * 72))
    print("PASS — no answer text moved" if not result["moved"]
          else "FAIL — answer text moved; Stage acceptance not met")
    return 0 if not result["moved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
