#!/usr/bin/env python3
"""clause_splitting_phase1/run_blast_radius.py

Vocabulary blast-radius test — §4.0(b).

Adds the five declared synonyms to the registry IN MEMORY, runs both banks on
each tree unperturbed and perturbed, and counts regressions on each side.

    python -m clause_splitting_phase1.run_blast_radius --tree rc

A regression is a case that PASSED unperturbed and FAILS perturbed. Newly
passing cases are counted too and reported separately: a vocabulary addition
that fixes as much as it breaks is still a tree whose behaviour moves when a
client adds a word, which is the property under test.

Nothing on disk is written. The registry file is read once and deep-copied
before any synonym is added, so the tagged release candidate stays as tagged.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from clause_splitting_phase1.trees import load_semantics  # noqa: E402

_SPEC = _HERE / "probes" / "vocabulary_blast_radius.yaml"
_REAL_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
              / "platform" / "alderbridge" / "2026-06-30"
              / "platform_canonical_typed.csv")


def load_spec() -> Dict[str, Any]:
    return yaml.safe_load(_SPEC.read_text(encoding="utf-8"))


def overrides_from(spec: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for s in spec["synonyms"]:
        out[s["field"]].append(s["term"])
    return dict(out)


# --------------------------------------------------------------------------- #
# Bank runners. Each returns {case_id: passed}.
# --------------------------------------------------------------------------- #
def _run_calibration_bank(semantics: dict, df) -> Dict[str, bool]:
    from mi_agent.mi_calibration import run_bank
    results, _ = run_bank(df=df, semantics=semantics)
    # A declared known_gap is an expected failure; it is neither a pass nor a
    # regression, so it is excluded from both sides of the comparison.
    return {r.id: r.ok for r in results if not r.known_gap}


def _run_generated_harness(semantics: dict, df) -> Dict[str, bool]:
    from mi_agent.mi_query_harness import run_suite
    results, _ = run_suite(df=df, semantics=semantics)
    return {r.case.id: r.ok for r in results}


BANKS = {
    "calibration_bank": _run_calibration_bank,
    "generated_harness": _run_generated_harness,
}


# --------------------------------------------------------------------------- #
# Disclosed extension, beyond the two banks §4.0(b) specifies.
#
# The two banks are fixed question sets written in the registry's ORIGINAL
# vocabulary. A new synonym can only move them by colliding with wording those
# questions already use. That makes the specified test a measure of collateral
# damage, and it is silent when no collision happens to exist in the corpus —
# which is not the same as the tree being safe.
#
# The probe sets are a third corpus, authored before any of this was run, so
# scoring them under perturbation adds sensitivity without adding hindsight.
# It is reported SEPARATELY from the specified two-bank result, never folded
# into it.
# --------------------------------------------------------------------------- #
def _run_probe_corpus(set_name: str):
    def _runner(semantics: dict, df) -> Dict[str, bool]:
        from clause_splitting_phase1.run_probes import load_set
        from clause_splitting_phase1.scoring import score
        from clause_splitting_phase1.trees import get_tree
        tree, has_unres = get_tree("rc", semantics)
        doc = load_set(set_name)
        return {p["id"]: score(p, tree(p["question"]), "rc",
                               has_unresolvable_channel=has_unres).ok
                for p in doc["probes"]}
    return _runner


PROBE_CORPORA = {
    "probes_adversarial": _run_probe_corpus("adversarial"),
    "probes_time_series": _run_probe_corpus("time_series"),
}


def compare(before: Dict[str, bool], after: Dict[str, bool]) -> Dict[str, Any]:
    shared = sorted(set(before) & set(after))
    regressions = [c for c in shared if before[c] and not after[c]]
    recoveries = [c for c in shared if not before[c] and after[c]]
    return {
        "cases_before": len(before),
        "cases_after": len(after),
        "cases_compared": len(shared),
        "cases_only_before": sorted(set(before) - set(after)),
        "cases_only_after": sorted(set(after) - set(before)),
        "passed_before": sum(1 for c in shared if before[c]),
        "passed_after": sum(1 for c in shared if after[c]),
        "regressions": regressions,
        "recoveries": recoveries,
        "moved": len(regressions) + len(recoveries),
    }


def run_tree(tree_name: str, spec: Dict[str, Any],
             include_probes: bool = False) -> Dict[str, Any]:
    if tree_name != "rc":
        raise NotImplementedError(
            "bank execution for tree %r is not wired in Phase 1; the branch "
            "tree has no consumers yet, by design (§4.3)." % tree_name)
    # The calibration bank grades against a REAL funded tape and deliberately
    # refuses to fall back to build_fixture: that fallback is what let the bank
    # report 251/252 while scoring 125/252 on a real book. The generated
    # harness is a different instrument and keeps its own synthetic fixture.
    from mi_agent.mi_calibration import default_bank_frame
    from mi_agent.mi_query_harness import build_fixture
    real_book = default_bank_frame(_REAL_TAPE)
    synthetic = build_fixture()
    base_sem = load_semantics()
    pert_sem = load_semantics(overrides_from(spec))

    runners = dict(BANKS)
    if include_probes:
        runners.update(PROBE_CORPORA)

    frames = defaultdict(lambda: synthetic)
    frames["calibration_bank"] = real_book

    out: Dict[str, Any] = {}
    for bank_id, runner in runners.items():
        frame = frames[bank_id]
        before = runner(base_sem, frame)
        after = runner(pert_sem, frame)
        out[bank_id] = compare(before, after)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", action="append", default=None, help="default: rc")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--include-probes", action="store_true",
                    help="also score the probe corpora under perturbation "
                         "(disclosed extension, reported separately)")
    args = ap.parse_args(argv)

    spec = load_spec()
    trees = args.tree or ["rc"]

    print("=" * 78)
    print("vocabulary blast-radius — %d synonyms added in memory" % len(spec["synonyms"]))
    print("=" * 78)
    for s in spec["synonyms"]:
        print("  %-16s -> %-30s (%s, collision risk %s)"
              % (s["term"], s["field"], s["role"], s["collision_risk"]))

    blob: Dict[str, Any] = {}
    for tree in trees:
        blob[tree] = run_tree(tree, spec, include_probes=args.include_probes)
        print("\ntree: %s" % tree)
        print("%-22s%12s%12s%14s%12s" % ("bank", "compared", "pass before",
                                         "pass after", "regressions"))
        for bank_id, c in blob[tree].items():
            print("%-22s%12d%12d%14d%12d"
                  % (bank_id, c["cases_compared"], c["passed_before"],
                     c["passed_after"], len(c["regressions"])))
            if c["regressions"]:
                print("    regressed: %s" % ", ".join(c["regressions"][:20]))
            if c["recoveries"]:
                print("    newly passing: %s" % ", ".join(c["recoveries"][:20]))
            if c["cases_only_before"] or c["cases_only_after"]:
                print("    case set moved: -%d +%d"
                      % (len(c["cases_only_before"]), len(c["cases_only_after"])))

    if args.json:
        args.json.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
