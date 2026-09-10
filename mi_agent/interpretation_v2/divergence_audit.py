"""Why do three paraphrases of one question disagree? Classified from evidence.

This is the Phase 1 / 1.5 instrument. It takes a completed benchmark run and, for
every non-invariant canonical, says what the FIRST material semantic difference
was and which of seven causes it belongs to. It changes nothing and decides
nothing; it exists so that a remedy is proposed against named cases rather than
against a hunch.

THE DISTINCTION THAT MATTERS
----------------------------
A raw payload diff overstates disagreement badly. Two intents that read
``{"base": "funded"}`` and ``{"base": "funded", "lens": "all", "seasoning":
"any"}`` are the SAME intent — the compiler fills those defaults — and a naive
comparison reports them as divergent. So the audit compares a NORMAL FORM:
defaults filled, top-level measures folded into the one output list, the legacy
``period_pair`` comparison dropped, and period labels treated as wording rather
than meaning for every relative form.

What survives that normalisation is real. What does not survive it was never a
disagreement about meaning, only about which of two permitted encodings the
model happened to pick — and that is the definition of contract redundancy.

THE SEVEN CAUSES
----------------
``REPRESENTATIONAL``
    Same meaning, alternative CandidateIntent encoding. Fixable by removing the
    redundancy from the contract. Named cases only.
``INTERPRETATION_POLICY``
    The model understood the core request and then added or omitted
    analytically-related content around it — a second limit measure, an extra
    grouping, a companion figure. Not a misreading and not a schema defect: the
    system has never stated how much an answer should volunteer.
``MODEL_MISS``
    The question states something explicitly and a variant missed or misread it.
``GENUINE_AMBIGUITY``
    The wording reasonably admits more than one governed interpretation, and the
    honest outcome may be to clarify.
``MISSING_CONTRACT_SLOT``
    The meaning is understood but CandidateIntent cannot represent it.
``COMPILER``
    The same CandidateIntent compiled differently. Would be a determinism bug.
``OTHER``
    Anything the six above do not cover, including a frozen bank whose three
    variants genuinely ask different questions.

    python -m mi_agent.interpretation_v2.divergence_audit \\
        evidence/run4_135_with_metadata_access.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: The Phase 1.5 taxonomy. It splits the old `under_determined_question` bucket
#: into the two things that were hiding inside it, which is the distinction that
#: decides where the next phase belongs:
#:
#:   GENUINE_AMBIGUITY      the wording admits more than one governed reading, so
#:                          the honest answer is to clarify. A question problem.
#:   INTERPRETATION_POLICY  the model understood the core request and then added
#:                          or dropped analytically-related content around it. A
#:                          policy problem — the system has no stated rule about
#:                          how much to volunteer.
#:
#: Those call for completely different remedies, and the old label could not tell
#: them apart.
REPRESENTATIONAL = "REPRESENTATIONAL"
INTERPRETATION_POLICY = "INTERPRETATION_POLICY"
MODEL_MISS = "MODEL_MISS"
GENUINE_AMBIGUITY = "GENUINE_AMBIGUITY"
MISSING_CONTRACT_SLOT = "MISSING_CONTRACT_SLOT"
COMPILER = "COMPILER"
OTHER = "OTHER"
UNCERTAIN = "UNCERTAIN"

CAUSES = (REPRESENTATIONAL, INTERPRETATION_POLICY, MODEL_MISS, GENUINE_AMBIGUITY,
          MISSING_CONTRACT_SLOT, COMPILER, OTHER, UNCERTAIN)

#: For an INTERPRETATION_POLICY case, which direction the divergence runs.
#: Derived mechanically from the payloads against a reference set, so the label
#: is not a second opinion on top of the adjudication.
POLICY_ADDED_MEASURE = "added unrequested measure"
POLICY_OMITTED_MEASURE = "omitted required measure"
POLICY_ADDED_DIMENSION = "added unrequested dimension"
POLICY_OMITTED_DIMENSION = "omitted required dimension"
POLICY_ADDED_CAPABILITY = "added unrequested capability/analysis"
POLICY_OTHER = "other"

#: Slot comparison order: coarsest first, so the "first material difference" is
#: the one that explains the others rather than a downstream consequence.
SLOT_ORDER: Tuple[str, ...] = (
    "capability", "operation", "population", "measure_set", "dimension_set",
    "output_nesting", "filters", "geography", "time", "comparison", "target",
)


def normal_form(payload: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The candidate normal form this audit tests, applied to a raw payload.

    Deliberately a pure function over the payload: it infers nothing from the
    question, reads no English, and binds nothing. It only removes the schema's
    own permitted variation.
    """
    if not payload:
        return None

    population = dict(payload.get("population") or {})
    population.setdefault("base", "funded")
    population.setdefault("lens", "all")
    population.setdefault("seasoning", "any")

    comparison = dict(payload.get("comparison") or {})
    kind = comparison.get("kind", "none")
    # Legacy: a period pair is stated by the operation and the time form.
    if kind == "period_pair":
        kind = "none"
    comparison_n = ((kind, comparison.get("left"), comparison.get("right"))
                    if kind != "none" else ("none", None, None))

    time = dict(payload.get("time") or {})
    form = time.get("form", "current")
    # Period labels are the question's own WORDS. They carry meaning only where
    # they are the sole statement of which period ("April to June").
    labels = (tuple(sorted(time.get("labels") or ()))
              if form == "explicit_period" else ())
    periods_back = time.get("periods_back")
    # "last month" is one period back whether or not the model said so.
    if form in ("relative_pair", "previous_reporting_period") and periods_back is None:
        periods_back = 1
    time_n = (form, time.get("grain"), periods_back, labels)

    outputs = list(payload.get("outputs") or ())
    top_dimensions = tuple(sorted(payload.get("dimensions") or ()))
    if not outputs:
        outputs = [{"measures": payload.get("measures") or [],
                    "dimensions": list(top_dimensions), "filters": []}]

    measure_set = frozenset()
    dimension_set = frozenset(top_dimensions)
    folded: List[Tuple[Any, ...]] = []
    for output in outputs:
        measures = tuple(sorted((m.get("concept"), m.get("statistic"))
                                for m in (output.get("measures") or ())))
        dimensions = tuple(sorted(output.get("dimensions") or top_dimensions))
        filters = tuple(sorted((f.get("concept"), f.get("comparator"),
                                str(f.get("value")))
                               for f in (output.get("filters") or ())))
        measure_set |= {m[0] for m in measures}
        dimension_set |= set(dimensions)
        folded.append((measures, dimensions, filters))

    return {
        "capability": payload.get("capability"),
        "operation": payload.get("operation"),
        "population": (population["base"], population["lens"],
                       population["seasoning"]),
        "measure_set": tuple(sorted(measure_set)),
        "dimension_set": tuple(sorted(dimension_set)),
        "output_nesting": len(outputs),
        "filters": tuple(sorted((f.get("concept"), f.get("comparator"),
                                 str(f.get("value")))
                                for f in (payload.get("filters") or ()))),
        "geography": tuple(sorted((payload.get("geography") or {}).items())),
        "time": time_n,
        "comparison": comparison_n,
        "target": (tuple(sorted((payload.get("target") or {}).items()))
                   if payload.get("target") else None),
        "_folded": tuple(sorted(folded, key=repr)),
    }


def policy_detail(forms: Sequence[Optional[Dict[str, Any]]],
                  expected: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Which way an INTERPRETATION_POLICY divergence runs, derived mechanically.

    The reference is the human-reviewable fixture where it states measures or
    dimensions, because that is independent truth authored before any run. Where
    it does not, the reference is what all three variants AGREED on — the core
    request none of them disputed — so "added" means beyond the agreed core and
    "omitted" means short of it.
    """
    live = [f for f in forms if f]
    if len(live) < 2:
        return []
    detail: List[str] = []
    expected = expected or {}

    def reference(slot: str, fixture_key: str) -> set:
        if expected.get(fixture_key) is not None:
            return {str(v).strip().lower() for v in expected[fixture_key]}
        return set.intersection(*[set(f[slot]) for f in live])

    for slot, fixture_key, added, omitted in (
            ("measure_set", "measures", POLICY_ADDED_MEASURE, POLICY_OMITTED_MEASURE),
            ("dimension_set", "dimensions", POLICY_ADDED_DIMENSION,
             POLICY_OMITTED_DIMENSION)):
        ref = reference(slot, fixture_key)
        seen = [set(f[slot]) for f in live]
        if len({frozenset(s) for s in seen}) == 1:
            continue
        if any(s - ref for s in seen):
            detail.append(added)
        if any(ref - s for s in seen):
            detail.append(omitted)

    if len({f["capability"] for f in live}) > 1:
        detail.append(POLICY_ADDED_CAPABILITY)
    if not detail:
        detail.append(POLICY_OTHER)
    return detail


def _differing_slots(forms: Sequence[Optional[Dict[str, Any]]]) -> List[str]:
    out = []
    for slot in SLOT_ORDER:
        seen = {repr(f[slot]) if f else None for f in forms}
        if len(seen) > 1:
            out.append(slot)
    return out


def audit(run: Mapping[str, Any],
          adjudication: Optional[Mapping[str, str]] = None,
          expectations: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Classify every non-invariant canonical in a completed run."""
    by_question = {r["question_id"]: r for r in run["results"]}
    divergent = {d["canonical_id"]: d["divergent_slots"]
                 for d in run["paraphrase_invariance"]["divergent"]}
    # An entry may be a bare cause or {cause, reason} — the reviewable form.
    raw_adjudication = dict(adjudication or {})
    reasons = {k: (v.get("reason", "") if isinstance(v, Mapping) else "")
               for k, v in raw_adjudication.items()}
    adjudication = {k: (v if isinstance(v, str) else (v or {}).get("cause", UNCERTAIN))
                    for k, v in raw_adjudication.items()}

    cases: List[Dict[str, Any]] = []
    for canonical_id, plan_slots in sorted(divergent.items()):
        variants = [f"{canonical_id}{v}" for v in "ABC"]
        rows = [by_question[q] for q in variants]
        forms = [normal_form(r.get("raw_payload")) for r in rows]
        remaining = _differing_slots(forms)
        cases.append({
            "canonical_id": canonical_id,
            "variants": [{"question_id": r["question_id"],
                          "question": r["question"],
                          "outcome": r["outcome"],
                          "reason_codes": list(r["reason_codes"]),
                          "normal_form": {k: v for k, v in (f or {}).items()
                                          if not k.startswith("_")}}
                         for r, f in zip(rows, forms)],
            "plan_level_divergent_slots": list(plan_slots),
            "first_material_difference": remaining[0] if remaining else None,
            "all_material_differences": remaining,
            "resolved_by_normal_form": not remaining,
            "cause": adjudication.get(canonical_id, UNCERTAIN),
            "adjudication_reason": reasons.get(canonical_id, ""),
            "policy_detail": policy_detail(
                forms, ((expectations or {}).get(canonical_id) or {}).get("expected")),
        })

    by_cause: Dict[str, List[str]] = {c: [] for c in CAUSES}
    for case in cases:
        by_cause[case["cause"]].append(case["canonical_id"])
    resolved = [c["canonical_id"] for c in cases if c["resolved_by_normal_form"]]

    return {
        "source_run": run.get("benchmark"),
        "non_invariant_canonicals": len(cases),
        "invariant_before": run["paraphrase_invariance"]["invariant"],
        "canonicals_total": run["paraphrase_invariance"]["canonicals"],
        "resolved_by_normal_form_alone": resolved,
        "counts_by_cause": {c: len(ids) for c, ids in by_cause.items() if ids},
        "canonicals_by_cause": {c: ids for c, ids in by_cause.items() if ids},
        "cases": cases,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--adjudication", type=Path,
                        help="JSON or YAML mapping canonical_id -> cause "
                             "(or {cause, reason})")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    run = json.loads(args.run.read_text(encoding="utf-8"))
    adjudication = None
    if args.adjudication:
        text = args.adjudication.read_text(encoding="utf-8")
        if args.adjudication.suffix in (".yaml", ".yml"):
            import yaml
            adjudication = (yaml.safe_load(text) or {}).get("adjudication", {})
        else:
            adjudication = json.loads(text)
    from .benchmark import load_expectations
    report = audit(run, adjudication, load_expectations())
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, default=str),
                            encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
