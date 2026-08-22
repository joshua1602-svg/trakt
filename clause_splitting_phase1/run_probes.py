#!/usr/bin/env python3
"""clause_splitting_phase1/run_probes.py

Runs the adversarial probe set (§4.0a) and the time-series probe set (§4.0c)
against one or more trees, and tabulates the result.

    python -m clause_splitting_phase1.run_probes --tree rc
    python -m clause_splitting_phase1.run_probes --tree rc --tree branch --json out.json

Read-only. Parses questions; executes nothing; writes nothing unless --json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from clause_splitting_phase1.scoring import ProbeResult, score, tabulate  # noqa: E402
from clause_splitting_phase1.trees import get_tree  # noqa: E402

SETS = {
    "adversarial": _HERE / "probes" / "adversarial_probes.yaml",
    "time_series": _HERE / "probes" / "time_series_probes.yaml",
}


def load_set(name: str) -> Dict[str, Any]:
    return yaml.safe_load(SETS[name].read_text(encoding="utf-8"))


def run_set(set_name: str, tree_name: str) -> List[ProbeResult]:
    doc = load_set(set_name)
    tree, has_unresolvable = get_tree(tree_name)
    out = []
    for probe in doc["probes"]:
        spans = tree(probe["question"])
        out.append(score(probe, spans, tree_name,
                         has_unresolvable_channel=has_unresolvable))
    return out


def _print_set(set_name: str, per_tree: Dict[str, List[ProbeResult]], verbose: bool) -> None:
    doc = load_set(set_name)
    print("\n" + "=" * 78)
    print("%s  (%s, %d probes)" % (doc["instrument"], set_name, len(doc["probes"])))
    print("=" * 78)
    summary = tabulate(per_tree)
    hdr = "%-26s" % "tree"
    for k in ("total", "passed", "failed", "substantive", "structural"):
        hdr += "%12s" % k
    print(hdr)
    for tree, s in summary.items():
        row = "%-26s" % tree
        for k in ("total", "passed", "failed", "failed_substantive",
                  "failed_structural_only"):
            row += "%12d" % s[k]
        print(row + "   pass rate %.1f%%" % (100 * s["pass_rate"]))

    shapes = sorted({r.shape or "unclassified"
                     for rs in per_tree.values() for r in rs})
    if len(shapes) > 1:
        print("\nby shape:")
        print("%-24s" % "shape" + "".join("%22s" % t for t in per_tree))
        for shape in shapes:
            row = "%-24s" % shape
            for tree, rs in per_tree.items():
                sel = [r for r in rs if (r.shape or "unclassified") == shape]
                row += "%22s" % ("%d/%d" % (sum(1 for r in sel if r.ok), len(sel)))
            print(row)

    if verbose:
        for tree, rs in per_tree.items():
            print("\n--- %s: failing probes ---" % tree)
            for r in rs:
                if r.ok:
                    continue
                tag = " [structural]" if r.structural_only else ""
                rfwr = " [right-for-wrong-reason]" if r.right_for_wrong_reason else ""
                print("  %s%s%s\n    %s" % (r.probe_id, tag, rfwr, r.question))
                for m in r.mismatches:
                    print("      - %s" % m)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", action="append", default=None,
                    help="tree to score (repeatable). Default: rc")
    ap.add_argument("--set", action="append", choices=sorted(SETS), default=None,
                    help="probe set to run (repeatable). Default: both")
    ap.add_argument("--json", type=Path, default=None, help="write full results here")
    ap.add_argument("-v", "--verbose", action="store_true", help="list failing probes")
    args = ap.parse_args(argv)

    trees = args.tree or ["rc"]
    sets = args.set or sorted(SETS)

    blob: Dict[str, Any] = {}
    for set_name in sets:
        per_tree = {t: run_set(set_name, t) for t in trees}
        _print_set(set_name, per_tree, args.verbose)
        blob[set_name] = {
            "summary": tabulate(per_tree),
            "results": {t: [vars(r) for r in rs] for t, rs in per_tree.items()},
        }

    if args.json:
        args.json.write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
