"""Why do three paraphrases of one question disagree? Classified from evidence.

This is the Phase 1 instrument. It takes a completed benchmark run and, for every
non-invariant canonical, says what the FIRST material semantic difference was and
which of six causes it belongs to. It changes nothing and decides nothing; it
exists so that a schema change is proposed against named cases rather than
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

THE SIX CAUSES
--------------
``contract_redundancy``
    The schema permits two encodings of one meaning and the model chose
    differently. Fixable by removing the redundancy. Named cases only.
``under_determined_question``
    The question does not settle it. "Summarise the main concentrations" does
    not say whether that is one measure or three, and no schema change makes
    three readers agree.
``model_miss``
    The question states something and one variant dropped it. An interpretation
    error under an identical contract.
``wording_difference``
    The three frozen variants genuinely ask different things. Not a defect.
``missing_contract_slot``
    The intent had nowhere to put something the question states.
``compiler_difference``
    Identical intents compiled differently. Would be a determinism bug.

    python -m mi_agent.interpretation_v2.divergence_audit \\
        evidence/run4_135_with_metadata_access.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

CONTRACT_REDUNDANCY = "contract_redundancy"
UNDER_DETERMINED = "under_determined_question"
MODEL_MISS = "model_miss"
WORDING_DIFFERENCE = "wording_difference"
MISSING_CONTRACT_SLOT = "missing_contract_slot"
COMPILER_DIFFERENCE = "compiler_difference"
UNCERTAIN = "uncertain"

CAUSES = (CONTRACT_REDUNDANCY, UNDER_DETERMINED, MODEL_MISS, WORDING_DIFFERENCE,
          MISSING_CONTRACT_SLOT, COMPILER_DIFFERENCE, UNCERTAIN)

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


def _differing_slots(forms: Sequence[Optional[Dict[str, Any]]]) -> List[str]:
    out = []
    for slot in SLOT_ORDER:
        seen = {repr(f[slot]) if f else None for f in forms}
        if len(seen) > 1:
            out.append(slot)
    return out


def audit(run: Mapping[str, Any],
          adjudication: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
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
    report = audit(run, adjudication)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, default=str),
                            encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
