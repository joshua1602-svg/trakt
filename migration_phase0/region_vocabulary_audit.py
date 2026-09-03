#!/usr/bin/env python3
"""region_vocabulary_audit — how much of this book's geography is governed?

WHY THIS EXISTS. "Is the region vocabulary clean?" was a guess until now, and
guesses are how the 2026-09-03 session spent four rounds diagnosing the wrong
thing. This turns it into a number: of the distinct region values the tape
carries, how many resolve to a governed canonical, how many resolve only
through an approved synonym, and how many resolve to nothing at all.

IT CHANGES NOTHING. It reads, cleans and resolves, exactly as
`funded_prep._apply_region_taxonomy` will at runtime, using THE SAME functions
so the audit cannot disagree with the thing it is auditing. It writes no
column, updates no config and touches no snapshot.

WHAT IT IS FOR, in order:

  1. SIZE THE JOB before anyone edits a taxonomy. A book that resolves 95% has
     a different problem from one that resolves 20%.
  2. NAME THE GAPS. Every unresolved key is listed, with the count of distinct
     raw spellings that reached it, so the biggest gaps are approved first.
  3. SAY WHEN IT IS DONE. Re-run after approving synonyms; "unresolved: 0" is
     the finish line, and it is checkable rather than asserted.
  4. PROPOSE, NEVER DECIDE. `--suggest` offers a canonical for each unresolved
     key by token overlap alone. A proposal is printed for a human to approve
     or replace and is NEVER written to the taxonomy by this script, which is
     the same rule `region_taxonomy` already applies to its LLM proposals:
     onboarding only, reviewable, never live.

WHAT LEAVES YOUR ENVIRONMENT. Nothing, unless you send it. The output holds
region NAMES — it has to, since approving a synonym means reading the name that
needs one — and distinct-value counts. It holds no balance, no loan count, no
row count and no borrower data: `--counts-only` drops the names too, leaving
just the resolution profile, if you want a figure you can paste anywhere.

USAGE
    # against a CSV you can point at
    python3 region_vocabulary_audit.py --csv /path/to/canonical.csv

    # against the dataset the running MI API has resolved (inside the app)
    python3 region_vocabulary_audit.py --from-app

    # split by book, and propose canonicals for the gaps
    python3 region_vocabulary_audit.py --csv tape.csv \\
        --by source_portfolio_id --suggest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine import region_taxonomy as RT  # noqa: E402


def _distinct_values(frame, column: str) -> Counter:
    """Distinct raw values in one column, with how many rows carry each.

    Row counts are used ONLY to order the gaps by size and are not printed
    unless `--row-weights` is passed: which region needs a synonym most is a
    question about the vocabulary, and how many loans sit behind it is a
    question about the book.
    """
    counts: Counter = Counter()
    for value in frame[column].dropna().astype(str):
        text = value.strip()
        if text and text.lower() not in ("nan", "none", "null", "<na>"):
            counts[text] += 1
    return counts


def audit_column(taxonomy, values: Counter) -> Dict[str, Any]:
    """Resolve every distinct raw value; report the profile and the gaps.

    THE SAME `clean` AND `resolve_detail` the runtime uses. An audit with its
    own copy of the rule would pass while the runtime failed, which is the
    failure mode this whole exercise exists to avoid.
    """
    by_method: Counter = Counter()
    unresolved: Dict[str, Dict[str, Any]] = {}
    resolved_to: Dict[str, set] = {}
    keys_by_spelling: Dict[str, set] = {}

    for raw, rows in values.items():
        key = RT.clean(raw)
        canonical, method = taxonomy.resolve_detail(raw)
        by_method[method] += 1
        keys_by_spelling.setdefault(key, set()).add(raw)
        if canonical:
            resolved_to.setdefault(canonical, set()).add(raw)
        else:
            entry = unresolved.setdefault(
                key, {"key": key, "spellings": set(), "rows": 0})
            entry["spellings"].add(raw)
            entry["rows"] += rows

    # Two raw spellings that clean to ONE key are the ampersand defect's
    # signature. Reported separately from "unresolved", because a region that
    # resolves can still be split if the cleaner lets two keys through.
    split = {k: sorted(v) for k, v in keys_by_spelling.items() if len(v) > 1}
    return {
        "distinct_raw_values": sum(1 for _ in values),
        "by_method": dict(by_method),
        "resolved_canonicals": len(resolved_to),
        "unresolved_keys": sorted(
            ({**e, "spellings": sorted(e["spellings"])} for e in unresolved.values()),
            key=lambda e: (-e["rows"], e["key"])),
        "spellings_sharing_one_key": split,
        "resolution_pct": (
            round(100.0 * (sum(1 for _ in values) - len(unresolved))
                  / max(1, sum(1 for _ in values)), 1)),
    }


def suggest(taxonomy, key: str) -> Optional[str]:
    """A PROPOSAL for an unresolved key, by token overlap alone.

    Deterministic and deliberately dim: it offers the canonical sharing the
    most words, and nothing else. It knows no geography and must not be
    believed — "Humberside" overlaps "Yorkshire and The Humber" on no token at
    all, and the right answer for it is a decision about the client's book, not
    a rule. Printed for approval; never written anywhere.
    """
    words = set(key.split()) - {"and", "the", "of", "region", "england"}
    best, best_score = None, 0
    for canonical in taxonomy.values:
        other = set(RT.clean(canonical).split()) - {"and", "the", "of"}
        score = len(words & other)
        if score > best_score:
            best, best_score = canonical, score
    return best


def _load_csv(path: str):
    import pandas as pd
    return pd.read_csv(path, low_memory=False)


def _load_from_app():
    """The frame the running MI API has already resolved.

    Imported lazily and only on request: this is the convenient path inside the
    app container and is meaningless outside it.
    """
    from mi_agent_api import data_source
    return data_source.active_frame()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="path to the canonical tape")
    src.add_argument("--from-app", action="store_true",
                     help="use the dataset the running MI API has resolved")
    ap.add_argument("--client", default=None,
                    help="client id, for per-client taxonomy selection")
    ap.add_argument("--by", default=None,
                    help="column to split the audit by, e.g. source_portfolio_id")
    ap.add_argument("--columns", default=None,
                    help="comma-separated region columns (default: the "
                         "taxonomy's own SOURCE_FIELDS)")
    ap.add_argument("--suggest", action="store_true",
                    help="propose a canonical for each unresolved key")
    ap.add_argument("--row-weights", action="store_true",
                    help="show how many rows sit behind each unresolved key")
    ap.add_argument("--counts-only", action="store_true",
                    help="omit region names entirely; profile figures only")
    ap.add_argument("--out", default="region_audit.json")
    args = ap.parse_args(argv)

    taxonomy = RT.resolve_taxonomy(args.client)
    if taxonomy is None:
        print("NOT A MEASUREMENT — no region taxonomy resolved for client "
              f"{args.client!r}. Harmonisation would be a no-op at runtime, "
              "and that is the finding: configure a taxonomy first.",
              file=sys.stderr)
        return 3

    frame = _load_from_app() if args.from_app else _load_csv(args.csv)
    wanted = ([c.strip() for c in args.columns.split(",")] if args.columns
              else list(RT.SOURCE_FIELDS))
    present = [c for c in wanted if c in getattr(frame, "columns", [])]
    if not present:
        print("NOT A MEASUREMENT — none of the region columns "
              f"{wanted} is in this dataset, so `region_taxonomy.apply` would "
              "no-op at runtime whatever the taxonomy says.", file=sys.stderr)
        return 3

    print("taxonomy %s (%d canonical values, %d approved synonyms)"
          % (taxonomy.name, len(taxonomy.values_by_key), len(taxonomy.synonyms)))
    print("region columns present: %s" % ", ".join(present))

    groups: List[Tuple[str, Any]] = [("(whole dataset)", frame)]
    if args.by and args.by in getattr(frame, "columns", []):
        groups = [(str(name), part) for name, part in frame.groupby(args.by)]
    elif args.by:
        print("  (--by %s is not a column here; auditing the whole dataset)"
              % args.by)

    report: Dict[str, Any] = {"taxonomy": taxonomy.name, "columns": present,
                              "groups": {}}
    worst = 100.0
    for label, part in groups:
        report["groups"][label] = {}
        for column in present:
            values = _distinct_values(part, column)
            if not values:
                continue
            result = audit_column(taxonomy, values)
            worst = min(worst, result["resolution_pct"])
            print("\n=== %s · %s ===" % (label, column))
            print("  %d distinct value(s), %.1f%% resolve   %s"
                  % (result["distinct_raw_values"], result["resolution_pct"],
                     json.dumps(result["by_method"])))
            for key, spellings in sorted(result["spellings_sharing_one_key"].items()):
                if not args.counts_only:
                    print("  SPLIT SPELLINGS -> one key %r: %s"
                          % (key, ", ".join(repr(s) for s in spellings)))
                else:
                    print("  SPLIT SPELLINGS -> one key (%d spellings)"
                          % len(spellings))
            if result["unresolved_keys"]:
                print("  UNRESOLVED (%d):" % len(result["unresolved_keys"]))
                for entry in result["unresolved_keys"]:
                    if args.counts_only:
                        continue
                    line = "    %-34s" % entry["key"]
                    if args.row_weights:
                        line += "  rows=%d" % entry["rows"]
                    if args.suggest:
                        line += "  # PROPOSAL: %s" % (
                            suggest(taxonomy, entry["key"]) or "(no overlap)")
                    print(line)
            if args.counts_only:
                result.pop("unresolved_keys", None)
                result.pop("spellings_sharing_one_key", None)
            report["groups"][label][column] = result

    if args.suggest and not args.counts_only:
        print("\n=== paste into config/mi/region_taxonomy.yaml under `synonyms:` "
              "AFTER approving each line ===")
        for label, columns in report["groups"].items():
            for column, result in columns.items():
                for entry in result.get("unresolved_keys", []):
                    print("      %s: %s   # PROPOSAL — approve or replace"
                          % (entry["key"],
                             suggest(taxonomy, entry["key"]) or "TODO"))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=list)
    print("\nwrote %s" % args.out)
    # Non-zero while any value is ungoverned, so this can gate a pipeline:
    # "unresolved: 0" is the finish line and it is checked, not asserted.
    return 0 if worst >= 100.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
