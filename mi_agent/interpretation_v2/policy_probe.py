"""A small cohort probe, for deciding whether a policy change is worth a full run.

A complete benchmark costs about 1.1M input tokens. A policy change can fail in
two opposite directions, and both are visible on a fraction of the bank:

    A. it does not reduce unrequested elaboration  -> the change did nothing
    B. it converts enrichment into CLARIFY         -> the change made it timid

B is the dangerous one, because on a headline it can look like discipline. So
the cohort deliberately carries four groups and reports them separately:

    policy        the canonicals that diverged by volunteering content
    preservation  the canonicals that dropped something explicit
    concise       terse questions that were fully correct and must stay PLAN
    ambiguity     questions that SHOULD clarify and must keep doing so

TWO METRICS, BECAUSE ONE IS BLIND
---------------------------------
The fixture-based defect metric compares each reading against the
human-reviewable expectations authored before any run. It is the right metric
for PRESERVATION — a fixture that states a filter can prove the filter was
dropped — but it is structurally blind to ELABORATION on exactly the canonicals
that exhibit it: none of the seven policy canonicals states its measures,
because each was marked human_review precisely BECAUSE the question does not
settle them.

So `elaboration_spread` carries that half. It needs no truth reference: for one
canonical, a concept present in some variant and absent from another is content
one reader volunteered and another did not. It cannot say which reader was
right. It says how much the readings disagreed in scope, which is what the
policy is meant to reduce.

    python -m mi_agent.interpretation_v2.policy_probe --live --out probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .benchmark import load_bank, load_expectations, run_benchmark
from .compiler import CompilerContext, DeterministicCompiler
from .opus_interpreter import (
    CONFIGURED_MODEL,
    AnthropicInterpreterClient,
    OpusInterpreter,
    ReplayClient,
)
from .outcomes import OUTCOME_CLARIFY, OUTCOME_PLAN, OUTCOME_REFUSE
from .vocabulary import load_governed_vocabulary

#: The cohort groups, by canonical id.
#:
#: The first four came from Run 6 evidence: the seven INTERPRETATION_POLICY
#: canonicals, the two MODEL_MISS canonicals, a control of terse questions that
#: were fully correct, and a control of questions that legitimately clarify.
#:
#: Run 7 forced two more. `completeness` is the class the minimum-sufficient
#: policy broke: a question whose operation REQUIRES a measure the wording never
#: names. Run 7 emitted movement with an empty measure list, the compiler raised
#: MISSING_REQUIRED_SLOT, and three readings that had been plans became
#: clarifications. Q18/Q19/Q20 are that family in the frozen bank, and they are
#: the probe's primary subject now. `gains` holds what Phase 2A actually improved
#: and nothing else already covered, so a recalibration cannot quietly buy the
#: fix back by giving up what the fix bought. Phase 2A improved Q16B, Q20A and
#: SM08A; the first two are already in `preservation` and `completeness` (their
#: gains ARE those groups' subject), so only SM08 needs a home here. Groups do
#: not overlap, so `outcomes_by_group` sums to the cohort.
#:
#: One membership note, recorded rather than edited. NL7 sits in `policy` because
#: Run 6 classified it there, but Run 7 showed its measure set is genuinely open
#: — "riskier" names no governed measure, and three readers picked three
#: different baskets. On the recalibrated border it is a REQUIRED-AND-OPEN
#: element, so NL7 clarifying is the correct outcome, not a regression. The group
#: is left alone so these numbers stay comparable with the Phase 2A probe.
COHORT: Mapping[str, Sequence[str]] = {
    "policy": ("NL6", "NL7", "Q08", "Q10", "Q24", "Q25", "SM09"),
    "completeness": ("Q18", "Q19", "Q20"),
    "preservation": ("Q16", "Q21"),
    "concise": ("Q01", "Q02", "Q11", "SM01"),
    "ambiguity": ("BB01", "NL8"),
    "gains": ("SM08",),
}

DEFECTS = ("ADDED_UNREQUESTED_MEASURE", "ADDED_UNREQUESTED_DIMENSION",
           "DROPPED_EXPLICIT_MEASURE", "DROPPED_EXPLICIT_DIMENSION",
           "DROPPED_EXPLICIT_FILTER", "DROPPED_EXPLICIT_TIME",
           "DROPPED_EXPLICIT_COMPARISON")


def cohort_bank(groups: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """The frozen bank, filtered to the cohort. Questions are never altered."""
    wanted = set()
    for name in (groups or COHORT):
        wanted.update(COHORT[name])
    bank = dict(load_bank())
    canonicals = [c for c in bank["canonicals"] if c["id"] in wanted]
    bank["canonicals"] = canonicals
    bank["canonical_count"] = len(canonicals)
    bank["question_count"] = sum(len(c["variants"]) for c in canonicals)
    return bank


def _sets(payload: Optional[Mapping[str, Any]]) -> Dict[str, set]:
    """Measure, dimension, filter-concept, time-form and comparison as sets."""
    if not payload:
        return {"measures": set(), "dimensions": set(), "filters": set(),
                "time": set(), "comparison": set()}
    measures = {m.get("concept") for m in (payload.get("measures") or ())}
    dimensions = set(payload.get("dimensions") or ())
    filters = {f.get("concept") for f in (payload.get("filters") or ())}
    for output in payload.get("outputs") or ():
        measures |= {m.get("concept") for m in (output.get("measures") or ())}
        dimensions |= set(output.get("dimensions") or ())
        filters |= {f.get("concept") for f in (output.get("filters") or ())}
    if (payload.get("geography") or {}).get("group_by"):
        dimensions.add("__geography__")
    return {
        "measures": {m for m in measures if m},
        "dimensions": {d for d in dimensions if d},
        "filters": {f for f in filters if f},
        "time": {(payload.get("time") or {}).get("form", "current")},
        "comparison": {(payload.get("comparison") or {}).get("kind", "none")},
    }


def _expected_sets(expected: Mapping[str, Any], vocabulary) -> Dict[str, Any]:
    """The fixture's stated expectation, resolved through the governed index."""
    def canon(terms):
        out = set()
        for term in terms or ():
            concept = vocabulary.resolve(str(term).strip().lower())
            out.add(concept.concept_id if concept else str(term).strip().lower())
        return out

    stated: Dict[str, Any] = {}
    if expected.get("measures") is not None:
        stated["measures"] = canon(expected["measures"])
    if expected.get("dimensions") is not None:
        stated["dimensions"] = canon(expected["dimensions"])
    if expected.get("filters") is not None:
        stated["filters"] = canon(f["concept"] for f in expected["filters"])
    if expected.get("temporal") is not None:
        stated["time"] = {expected["temporal"]}
    if expected.get("comparison") is not None:
        stated["comparison"] = {expected["comparison"]}
    return stated


def measure_defects(run: Mapping[str, Any]) -> Dict[str, Any]:
    """Per-question defect counts against the pre-authored fixtures."""
    vocabulary = load_governed_vocabulary()
    expectations = load_expectations()
    counts = {d: 0 for d in DEFECTS}
    detail: List[Dict[str, Any]] = []
    unmeasurable = 0

    for row in run["results"]:
        expected = (expectations.get(row["canonical_id"]) or {}).get("expected") or {}
        stated = _expected_sets(expected, vocabulary)
        if not stated:
            unmeasurable += 1
            continue
        observed = _sets(row.get("raw_payload"))
        found: List[str] = []

        if "measures" in stated:
            if observed["measures"] - stated["measures"]:
                found.append("ADDED_UNREQUESTED_MEASURE")
            if stated["measures"] - observed["measures"] and row["outcome"] == OUTCOME_PLAN:
                found.append("DROPPED_EXPLICIT_MEASURE")
        if "dimensions" in stated:
            extra = observed["dimensions"] - stated["dimensions"] - {"__geography__"}
            if extra:
                found.append("ADDED_UNREQUESTED_DIMENSION")
            if stated["dimensions"] - observed["dimensions"] and row["outcome"] == OUTCOME_PLAN:
                found.append("DROPPED_EXPLICIT_DIMENSION")
        if "filters" in stated:
            if stated["filters"] - observed["filters"] and row["outcome"] == OUTCOME_PLAN:
                found.append("DROPPED_EXPLICIT_FILTER")
        if "time" in stated:
            if observed["time"] != stated["time"] and row["outcome"] == OUTCOME_PLAN:
                found.append("DROPPED_EXPLICIT_TIME")
        if "comparison" in stated:
            if observed["comparison"] != stated["comparison"] and row["outcome"] == OUTCOME_PLAN:
                found.append("DROPPED_EXPLICIT_COMPARISON")

        for defect in found:
            counts[defect] += 1
        if found:
            detail.append({"question_id": row["question_id"],
                           "outcome": row["outcome"], "defects": found})

    return {"counts": counts, "unmeasurable_questions": unmeasurable,
            "detail": detail}


def elaboration_spread(run: Mapping[str, Any]) -> Dict[str, Any]:
    """How much the three readers DISAGREED about what to include.

    The fixture-based defect metric cannot see the defect this phase targets:
    none of the seven INTERPRETATION_POLICY canonicals states its measures,
    because each was marked human_review or under-determined precisely BECAUSE
    the question does not settle them. Measuring "added beyond the fixture"
    there would measure nothing.

    So this metric needs no truth reference. For one canonical, a concept that
    appears in SOME variant but not ALL is content one reader volunteered and
    another did not — which is exactly what unrequested elaboration looks like
    from the outside. It cannot say WHICH reader was right, and it does not try;
    it says how much the readings disagreed in scope, and the policy should
    reduce it.
    """
    by_canonical: Dict[str, List[Mapping[str, Any]]] = {}
    for row in run["results"]:
        by_canonical.setdefault(row["canonical_id"], []).append(row)

    per_canonical: Dict[str, Dict[str, int]] = {}
    totals = {"measure_spread": 0, "dimension_spread": 0,
              "canonicals_with_spread": 0}
    for canonical_id, rows in sorted(by_canonical.items()):
        sets = [_sets(r.get("raw_payload")) for r in rows
                if r.get("raw_payload")]
        if len(sets) < 2:
            continue
        measures = [s["measures"] for s in sets]
        dimensions = [s["dimensions"] for s in sets]
        m_spread = len(set.union(*measures) - set.intersection(*measures))
        d_spread = len(set.union(*dimensions) - set.intersection(*dimensions))
        per_canonical[canonical_id] = {"measure_spread": m_spread,
                                       "dimension_spread": d_spread}
        totals["measure_spread"] += m_spread
        totals["dimension_spread"] += d_spread
        if m_spread or d_spread:
            totals["canonicals_with_spread"] += 1
    return {"totals": totals, "per_canonical": per_canonical}


def summarise(run: Mapping[str, Any]) -> Dict[str, Any]:
    by_group: Dict[str, Dict[str, int]] = {}
    group_of = {cid: name for name, ids in COHORT.items() for cid in ids}
    for row in run["results"]:
        group = group_of.get(row["canonical_id"], "other")
        bucket = by_group.setdefault(group, {})
        bucket[row["outcome"]] = bucket.get(row["outcome"], 0) + 1
    return {
        "questions": run["questions_attempted"],
        "model": run["model"]["returned"],
        "successful_calls": run["model"]["successful_calls"],
        "failed_calls": run["model"]["failed_calls"],
        "verdicts": run["verdicts"],
        "outcomes": run["outcomes"],
        "outcomes_by_group": by_group,
        "defects": measure_defects(run),
        "elaboration_spread": elaboration_spread(run),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--score-only", type=Path,
                        help="score an existing run file against the cohort")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--groups", nargs="*", choices=sorted(COHORT))
    args = parser.parse_args(argv)

    if args.score_only:
        run = json.loads(args.score_only.read_text(encoding="utf-8"))
        wanted = set()
        for name in (args.groups or COHORT):
            wanted.update(COHORT[name])
        run = dict(run, results=[r for r in run["results"]
                                 if r["canonical_id"] in wanted])
        run["questions_attempted"] = len(run["results"])
        run["verdicts"] = {}
        run["outcomes"] = {}
        for row in run["results"]:
            run["verdicts"][row["verdict"]] = run["verdicts"].get(row["verdict"], 0) + 1
            run["outcomes"][row["outcome"]] = run["outcomes"].get(row["outcome"], 0) + 1
        report = summarise(run)
    else:
        if args.replay:
            prior = json.loads(args.replay.read_text(encoding="utf-8"))
            client = ReplayClient({r["question"]: r["raw_payload"]
                                   for r in prior["results"] if r.get("raw_payload")})
        elif args.live:
            client = AnthropicInterpreterClient(model=CONFIGURED_MODEL)
            if not client.available:
                print("ANTHROPIC_API_KEY is not set")
                return 2
        else:
            parser.error("choose --live, --replay or --score-only")
        interpreter = OpusInterpreter(client, vocabulary=load_governed_vocabulary())
        run = run_benchmark(interpreter=interpreter,
                            compiler=DeterministicCompiler(CompilerContext()),
                            bank=cohort_bank(args.groups), progress=True)
        report = summarise(run)
        if args.out:
            args.out.write_text(json.dumps({"summary": report, "run": run},
                                           indent=1, default=str),
                                encoding="utf-8")

    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
