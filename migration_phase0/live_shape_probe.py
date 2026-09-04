#!/usr/bin/env python3
"""live_shape_probe — the defect questions, offline, on a LIVE-SHAPED book.

WHAT THIS IS NOT. It is not `replay_probe`, and it does not replace it.
`replay_probe` replays the telemetry corpus — what real users actually typed —
against the DEPLOYED service, and nothing here can substitute for that. This is
what remains available when the deployment cannot be reached: the same shipped
path (`POST /mi/query` on the production FastAPI app), a seeded local book, and
a SMALL, NAMED question list. A question list someone types is a question list
someone paraphrases; treat the verdicts here as evidence about the mechanisms
named below and about nothing else.

WHY A SECOND REGION COLUMN. `scripts/run_mi_query_stage_movement_banks.py`
builds the reproducible five-snapshot funded tape this reuses — with ONE region
column, `geographic_region_obligor`. The live tape carries the same region
spellings in `collateral_geography` too, and it was that second claimant that
made every region FILTER resolve to nothing while every region GROUPING kept
working. A rig with one region column cannot exercise it, so this adds the
second column with identical content, which is the live shape.

The three banks that rig runs are the NON-REGRESSION measurement and stay
authoritative: 215 governed questions, compared question by question. This adds
the questions those banks do not hold — the banks carry region groupings and no
region-VALUE filter at all.

    python -m migration_phase0.live_shape_probe --out before.json
    python -m migration_phase0.live_shape_probe --out after.json --compare before.json

Verdicts are `replay_probe`'s, so the two read the same way. Exits non-zero on
a REGRESSION and on nothing else.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

PORTFOLIO = "client_001/mi_2026_06"
AS_OF = "2026-06-30"

FIXED = "FIXED"
REGRESSED = "REGRESSED"
UNCHANGED_OK = "UNCHANGED_OK"
STILL_FAILING = "STILL_FAILING"

#: Every question, with the mechanism it is here to measure. The controls are
#: not decoration: a region filter that starts answering while "Compare us with
#: the market." also starts answering is a worse outcome than either failure.
QUESTIONS: Tuple[Tuple[str, str], ...] = (
    # A value carried by two ALIASED region fields — the filter that resolved
    # to nothing while the grouping worked.
    ("region_filter", "What is the total balance in Wales?"),
    ("region_filter", "What is the average LTV in Scotland?"),
    ("region_filter", "How many loans are in the South East?"),
    # A RESTRICTION ON THE AXIS BEING GROUPED. Both shapes refused: the filter
    # was dropped because its field was the grouping dimension, and the
    # coverage gate then declined to publish a breakdown that had lost the
    # narrowing. Kept as two entries because they are two different arguments —
    # the single-value one is degenerate (one row, and the reader could ask for
    # the figure directly), the multi-value one is an ordinary question with no
    # other phrasing available. The second is why this was fixed.
    ("axis_restricted_to_one_value",
     "Show balance by region for loans in Wales."),
    ("axis_restricted_to_several_values",
     "Show balance by region for loans in Wales and Scotland."),
    # THE SAME COORDINATION, ONE AXIS OVER. This answered before — with
    # `collateral_geography = Wales` and Scotland silently lost. The gate caught
    # it, so nothing wrong was published, but only the gate stood between a
    # dropped region and a confident figure over the wrong population.
    ("several_values_on_another_axis",
     "Show balance by broker for loans in Wales and Scotland."),
    # One analytic, several phrasings. The first answered before the routing
    # precedence was fixed; the rest were claimed by `temporal_compare`.
    ("stage_movement", "How many loans moved into Offer in the last reporting period?"),
    ("stage_movement", "How many loans moved into Offer stage in the last month?"),
    ("stage_movement", "How many loans moved into Offer in the last week?"),
    ("stage_movement", "How many cases moved from KFI into Application in the last month?"),
    ("stage_movement", "How many cases left KFI in the last week?"),
    # Answers today. Any of these moving is the finding.
    ("control", "Show balance by region."),
    ("control", "Show loan count by region."),
    ("control", "What is the total balance?"),
    ("control", "Which region has the largest balance?"),
    ("control", "Balance by region for loans with LTV above 50%."),
    ("control", "How many lump sum loans do we have?"),
    # TWO GROUPING AXES. Measured because the line above could be misread as
    # "region and another dimension together do not work": they do, as a
    # heatmap, in either order and with "and" as well as a second "by".
    ("control", "Balance by region by broker"),
    ("control", "Show me balance by broker and region"),
    # A LIMITS QUESTION THE RECOGNISER NEVER CLAIMED. `_RISK_LIMIT_RE` does not
    # match this sentence; the analytical intent boundary claims it. It was
    # refused for a category the parser invented out of the words after
    # "geographic", and the answer must now be SCOPED to the category it names
    # rather than covering every limit test.
    ("risk_limit_category", "What is the largest geographic concentration versus limit?"),
    # Refuses today, and must keep refusing. The last three are the estate's
    # must-refuse three; "Atlantis" is a place the book has no exposure to, and
    # the stage NAMED without being put in motion is the boundary of the
    # routing fix.
    ("must_refuse", "What is the average LTV in Atlantis?"),
    ("must_refuse", "Compare KFI balance this month vs last month"),
    ("must_refuse", "Show me the trend."),
    ("must_refuse", "What changed?"),
    ("must_refuse", "Compare us with the market."),
)


def _rig():
    """The committed bank rig's own fixture builder, with the second region
    column added. Imported by path because `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "_banks_rig", _REPO / "scripts" / "run_mi_query_stage_movement_banks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    original = mod.write_funded_tape

    def write_funded_tape(root, run_id, reporting_date, n):
        original(root, run_id, reporting_date, n)
        import pandas as pd

        path = (root / "client_001" / run_id / "output" / "central"
                / "18_central_lender_tape.csv")
        frame = pd.read_csv(path)
        frame["collateral_geography"] = frame["geographic_region_obligor"]
        frame.to_csv(path, index=False)

    mod.write_funded_tape = write_funded_tape
    return mod


def run() -> List[Dict[str, Any]]:
    client = _rig().build_client()
    rows: List[Dict[str, Any]] = []
    for group, question in QUESTIONS:
        body = client.post("/mi/query", json={
            "question": question, "portfolioId": PORTFOLIO,
            "asOfDate": AS_OF}).json()
        meta = body.get("metadata") or {}
        rows.append({
            "group": group, "question": question, "ok": bool(body.get("ok")),
            "route": meta.get("route"),
            "code": ((body.get("governance") or {}).get("error") or {}).get("code"),
            # Diagnostics only, the standing rule: the opening of the sentence,
            # never the whole answer. A refusal's reason IS the finding; an
            # answer's figures are the client's.
            "said": (body.get("answer") or body.get("error") or "")[:110],
        })
    return rows


def _verdict(before: Dict[str, Any], now: Dict[str, Any]) -> str:
    was, is_ = bool(before.get("ok")), bool(now.get("ok"))
    if was and is_:
        return UNCHANGED_OK
    if was and not is_:
        return REGRESSED
    if is_ and not was:
        return FIXED
    return STILL_FAILING


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", type=Path,
                    help="a prior --out, compared question by question")
    args = ap.parse_args(argv)

    rows = run()
    for row in rows:
        print("%-34s %-4s %-24s %s"
              % (row["group"], "OK" if row["ok"] else "no",
                 (row["route"] or "-")[:24], row["question"]))
    args.out.write_text(json.dumps(rows, indent=2))
    print("\n%d answered of %d — wrote %s"
          % (sum(1 for r in rows if r["ok"]), len(rows), args.out))

    if not args.compare:
        return 0
    prior = {r["question"]: r for r in json.loads(args.compare.read_text())}
    unknown = [r["question"] for r in rows if r["question"] not in prior]
    counts: Dict[str, int] = {}
    moved: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []
    for row in rows:
        before = prior.get(row["question"])
        if before is None:
            continue
        verdict = _verdict(before, row)
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict in (FIXED, REGRESSED):
            moved.append((verdict, row["question"], before, row))
    print("\nPER-QUESTION VERDICTS vs %s" % args.compare)
    for name in (REGRESSED, FIXED, UNCHANGED_OK, STILL_FAILING):
        if counts.get(name):
            print("  %-14s %d" % (name, counts[name]))
    if unknown:
        # NOT scored either way: a question the prior run never asked has no
        # before to have moved from, and counting it would flatter the run.
        print("  %d question(s) absent from the prior run, not scored: %s"
              % (len(unknown), ", ".join(unknown[:3])))
    for verdict, question, before, now in moved:
        print("\n  %s  %s" % (verdict, question))
        print("     before: %s" % (before.get("said") or "")[:100])
        print("     after : %s" % (now.get("said") or "")[:100])
    if counts.get(REGRESSED):
        print("\nREGRESSIONS: %d question(s) answered before and do not now."
              % counts[REGRESSED])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
