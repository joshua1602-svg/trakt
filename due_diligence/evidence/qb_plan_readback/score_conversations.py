#!/usr/bin/env python3
"""Does a follow-up get its stand-alone twin's answer? (P0 design §34, §39; D25)

`run_production_bank.sh <principal> conversations` plays every conversation of
`config/mi/golden_questions/conversation_bank_20261001.yaml` and its held-out
set `conversation_holdout_20261002.yaml` that the model reads,
each message sent with the continuation the previous one handed back, and
asks each turn's stand-alone twin — the same question asked in full — beside
it. This scores each turn against what the bank expects of it:

    answer    an opening question: answered
    carry     a follow-up: answered AS its twin (same governed plan, or the
    fill      same answer once the "I read this as" line is set aside), and
              says what it read the message as. Declined as its twin is
              declined also passes: a follow-up never answers more broadly.
    fresh     a complete question: nothing carried (no "read as" line), and
              its twin's outcome
    ask_back  asks for the one missing detail (no figure)
    decline   declines in words (no figure)

    PASS       as the bank expects
    MISS       no figure where one was expected (an ask-back or a decline in
               place of an answer), or an answer where an ask-back was
               expected that matches its twin: to fix, not wrong (D25)
    WRONG      a figure that is not the twin's, a figure where the twin
               declined, a carry into a complete question, or a figure where
               a decline was expected

PASS MARK (D25): no WRONG — every follow-up matches its twin or asks back —
and every group E (start fresh) and F (refuse) turn passes. The held-out
conversations (group H, §39.1) are played in the same run and reported on
their own line: the same mark, on wording the reader's rules were not drawn
from.

    python due_diligence/evidence/qb_plan_readback/score_conversations.py \\
        qb_conversations_conversations_<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: The sentences the conversation adds before an answer: what the message was
#: read as, and why an earlier question could not be used. Set aside before
#: an answer is compared with its twin's.
_PREFIXES = (
    re.compile(r"^Following on from your previous question, I read this as “[^”]*”\.\s*"),
    re.compile(r"^With your reply, I read your question as “[^”]*”\.\s*"),
    re.compile(r"^More than [\d.]+ minutes? passed since I asked, so the earlier "
               r"question has lapsed and I have read this on its own\.\s*"),
    re.compile(r"^The earlier question could not be used here, so I have read "
               r"this on its own\.\s*"),
    re.compile(r"^I could not read this together with your earlier question, so "
               r"I have read it on its own\.\s*"),
)
_TAIL = re.compile(r"\s*Reply with it and I will answer your question\.$")


def body(answer: str) -> str:
    text = str(answer or "").strip()
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            new = prefix.sub("", text)
            if new != text:
                text, changed = new, True
    return _TAIL.sub("", text).strip()


def _asked_back(rec: Dict[str, Any]) -> bool:
    return (rec.get("outcome") == "REFUSED"
            and str(rec.get("serving_reason") or "").startswith("CLARIFY"))


def _same(turn: Dict[str, Any], twin: Dict[str, Any]) -> bool:
    if turn.get("plan_id") and turn.get("plan_id") == twin.get("plan_id"):
        return True
    return body(turn.get("answer")) == body(twin.get("answer"))


def _read_as(rec: Dict[str, Any]) -> Optional[str]:
    return (rec.get("conversation") or {}).get("read_as")


def score(turn: Dict[str, Any], twin: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    """(verdict, why) for one turn."""
    expect = turn.get("expect")
    answered = turn.get("outcome") == "ANSWERED"
    if turn.get("outcome") == "ERROR":
        return "WRONG", "the request failed"
    if expect == "answer":
        return ("PASS", "answered") if answered else ("MISS", "not answered")
    if expect == "decline":
        if answered:
            return "WRONG", "answered where a decline was expected"
        return "PASS", ("asked back (no figure)" if _asked_back(turn)
                        else "declined")
    if expect == "ask_back":
        if _asked_back(turn):
            return "PASS", "asked back"
        if not answered:
            return "MISS", "declined where an ask-back was expected"
        if twin is not None and twin.get("outcome") == "ANSWERED" and _same(turn, twin):
            return "MISS", "answered with a default, as its twin"
        return "MISS", "answered where an ask-back was expected"
    if twin is None:
        return "MISS", "no twin recorded"
    twin_answered = twin.get("outcome") == "ANSWERED"
    if expect == "fresh":
        if _read_as(turn):
            return "WRONG", f"carried into a complete question: {_read_as(turn)!r}"
        if answered != twin_answered:
            return ("WRONG" if answered else "MISS"), "not its twin's outcome"
        if answered and not _same(turn, twin):
            return "WRONG", "a different answer from its twin"
        return "PASS", "read on its own, as its twin"
    # carry / fill
    if answered and not twin_answered:
        return "WRONG", "answered where its twin declined"
    if not answered:
        if _asked_back(turn):
            return "MISS", "asked back"
        if not twin_answered:
            return "PASS", "declined, as its twin"
        return "MISS", "declined where its twin answered"
    if not _same(turn, twin):
        return "WRONG", "a different answer from its twin"
    if not _read_as(turn):
        return "MISS", "its twin's answer, but no statement of what carried over"
    return "PASS", "its twin's answer"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path, help="the conversation run's .jsonl")
    args = ap.parse_args(argv)
    records = [json.loads(line) for line in args.run.read_text().splitlines()
               if line.strip()]
    twins = {r["twin_of"]: r for r in records if r.get("twin_of")}
    turns = [r for r in records if r.get("expect")]
    tally: Dict[str, Counter] = defaultdict(Counter)
    lines: List[str] = []
    wrong = []
    for turn in turns:
        verdict, why = score(turn, twins.get(turn["id"]))
        group = str(turn.get("category") or "").rsplit("_", 1)[-1]
        tally[group][verdict] += 1
        read = _read_as(turn)
        lines.append(f"{verdict:<5} {turn['id']:<12} {turn.get('expect'):<8} "
                     f"{turn['question'][:46]!r:<50} {why}"
                     + (f"\n{'':19}read as: {read!r}" if read else ""))
        if verdict == "WRONG":
            wrong.append(turn["id"])
    print("\n".join(lines))
    print("\nBY GROUP  (pass / miss / wrong)")
    for group in sorted(tally):
        c = tally[group]
        print(f"  {group}  {c['PASS']:>3} / {c['MISS']:>3} / {c['WRONG']:>3}")
    must = [t for t in turns
            if str(t.get("category") or "").rsplit("_", 1)[-1] in ("E", "F")]
    must_failed = [t["id"] for t in must
                   if score(t, twins.get(t["id"]))[0] != "PASS"]
    held_out = tally.get("H")
    if held_out:
        print(f"\nHELD-OUT (H, §39.1): {held_out['PASS']} pass / "
              f"{held_out['MISS']} miss / {held_out['WRONG']} wrong — wording "
              f"the reader's rules were not drawn from")
    print(f"\nD25: wrong answers {len(wrong)}"
          + (f" ({', '.join(wrong)})" if wrong else "")
          + f"; start-fresh and refuse turns {len(must) - len(must_failed)}/{len(must)}"
          + (f" (not passed: {', '.join(must_failed)})" if must_failed else ""))
    print("PASS MARK MET" if not wrong and not must_failed else "PASS MARK NOT MET")
    return 0


if __name__ == "__main__":
    sys.exit(main())
