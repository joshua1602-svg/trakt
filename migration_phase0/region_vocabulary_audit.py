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

#: A marker only the tree this script needs would carry. Searched for rather
#: than assumed, because this file is COPIED to a box and run from wherever it
#: landed: `__file__/../..` is the repo root in the repository and "/" in
#: /home, where the first live run of it died.
_MARKER = Path("engine") / "region_taxonomy.py"

#: Where a deployed copy of the tree actually lives, checked after the obvious
#: candidates so a repository checkout always wins.
#:
#: `/home/site/wwwroot` is the App Service SHARE and holds `startup.sh`; it is
#: NOT where the application runs from. Oryx extracts the artefact to a
#: per-deployment directory under /tmp and starts it there — the first live run
#: of this audit failed against wwwroot for exactly that reason, while the
#: operator's own shell prompt was sitting in `/tmp/8df09fb1c38da69`.
_DEPLOY_ROOTS = ("/home/site/wwwroot", "/app", "/opt/app")

#: The Oryx extraction pattern. Directories matching it are OFFERED in the
#: failure message and never selected automatically: a box carries older
#: extractions beside the live one, and a developer box carries checkouts and
#: worktrees, so picking one would be the audit quietly choosing whose copy of
#: `clean` to believe. That is the single thing this script must not do.
_DEPLOY_GLOBS = ("/tmp/*",)


def _find_tree() -> Optional[Path]:
    """The directory holding `engine/region_taxonomy.py`, or None.

    Order: whatever the interpreter can ALREADY import (an environment that has
    the tree on its path is the most authoritative answer available), then an
    explicit TRAKT_ROOT, this file's own ancestors, the working directory and
    its ancestors, the known deployment roots, and finally the Oryx extraction
    directories, newest first.
    """
    try:                          # already importable — nothing to search for
        import engine as _engine
        found = Path(_engine.__file__).resolve().parents[1]
        if (found / _MARKER).is_file():
            return found
    except Exception:             # noqa: BLE001 - absence is the normal case
        pass

    candidates: List[Path] = []
    explicit = os.environ.get("TRAKT_ROOT")
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve()
    candidates.extend(here.parents)
    candidates.append(Path.cwd().resolve())
    candidates.extend(Path.cwd().resolve().parents)
    candidates.extend(Path(p) for p in _DEPLOY_ROOTS)
    for candidate in candidates:
        try:
            if (candidate / _MARKER).is_file():
                return candidate
        except OSError:          # an unreadable path is not a candidate
            continue

    return None


def _offer_candidates() -> List[Path]:
    """Trees the operator might have meant, for the failure message only.

    Never returned to the caller as a choice. On a redeployed App Service box
    /tmp holds older extractions beside the live one; on a developer box it
    holds checkouts and worktrees. Selecting one would be this script deciding
    whose `clean` to audit with, which is the failure it exists to refuse.
    """
    import glob
    found: List[Path] = []
    for pattern in _DEPLOY_GLOBS:
        for match in glob.glob(str(Path(pattern) / _MARKER)):
            found.append(Path(match).resolve().parents[1])
    return sorted(set(found))


_TREE = _find_tree()
if _TREE is not None and str(_TREE) not in sys.path:
    sys.path.insert(0, str(_TREE))

try:
    from engine import region_taxonomy as RT  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - the message IS the behaviour
    sys.stderr.write(
        "NOT A MEASUREMENT — could not find `engine/region_taxonomy.py`.\n"
        "This audit must resolve through the SAME code the runtime uses, so it\n"
        "will not fall back to a copy of the rule. Point it at the tree:\n"
        "    TRAKT_ROOT=<app directory> python3 %s ...\n"
        "(on App Service the app runs from the Oryx extraction directory under\n"
        "/tmp, NOT from /home/site/wwwroot, which holds only startup.sh.)\n"
        % Path(__file__).name)
    _offered = _offer_candidates()
    if _offered:
        sys.stderr.write(
            "\nTrees found, NOT chosen for you — pick the one the app is "
            "running from:\n"
            + "".join("    TRAKT_ROOT=%s\n" % p for p in _offered))
    raise SystemExit(3)


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

    `get_dataframe()` is the accessor — the one the API's own routes call.
    Imported lazily and only on request: this is the convenient path inside the
    app container and is meaningless outside it.

    The name is checked rather than assumed. An earlier draft called
    `active_frame()`, which does not exist, and the operator found out by
    running it against a live box.
    """
    from mi_agent_api import data_source

    accessor = getattr(data_source, "get_dataframe", None)
    if accessor is None:  # pragma: no cover - the message IS the behaviour
        raise SystemExit(
            "NOT A MEASUREMENT — `mi_agent_api.data_source` on this box has no "
            "`get_dataframe`. Use --csv and point at the canonical tape.")
    frame = accessor()
    label = getattr(data_source, "data_source_label", lambda: "?")()
    print("dataset: %s" % label)
    return frame


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
        # ONE LINE PER KEY. A key appears once per book and once per region
        # column, so the first run emitted ten lines for four keys — a YAML
        # block with duplicate keys, where the last silently wins. Rows are
        # summed across the appearances so the biggest gap is still first.
        gaps: Dict[str, int] = {}
        for columns in report["groups"].values():
            for result in columns.values():
                for entry in result.get("unresolved_keys", []):
                    gaps[entry["key"]] = gaps.get(entry["key"], 0) + entry["rows"]
        for key in sorted(gaps, key=lambda k: (-gaps[k], k)):
            print("      %s: %s   # PROPOSAL — approve or replace"
                  % (key, suggest(taxonomy, key) or "TODO"))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=list)
    print("\nwrote %s" % args.out)
    # Non-zero while any value is ungoverned, so this can gate a pipeline:
    # "unresolved: 0" is the finish line and it is checked, not asserted.
    return 0 if worst >= 100.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
