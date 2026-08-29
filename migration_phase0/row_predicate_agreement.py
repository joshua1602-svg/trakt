#!/usr/bin/env python3
"""migration_phase0/row_predicate_agreement.py — does the contract describe what runs?

READ-ONLY. Step 2 of the C6 filter binding sequence.

`projection._row_predicates` carries the resolved `field + operator + value` onto
the interpretation. The question this answers is whether that description is
FAITHFUL to what the shipped executor actually does.

A tautology to avoid: the projection builds its claims through
`population.material_predicates(spec.filters)`, so comparing them against
`material_predicates(spec.filters)` proves nothing — it is the same call twice.
This compares against EXECUTION instead, on a real governed frame:

    shipped   : mi_query_executor._apply_filters(frame, spec, semantics)
    contract  : population.apply_population(frame, contract_predicates, semantics)

and requires the two to select the SAME ROWS. Row-set identity tests field,
operator and value together, and can fail on any of them — percent scaling, a
comparator alias, categorical casing, a canonical-column mismatch.

Two results, and they are NOT the same claim:

  DESCRIPTION  the claim's field/operator/value equals the spec entry the
               executor was handed. This is what step 2 was asked to prove.
  EXECUTION    `apply_population` on those claims selects the same rows
               `_apply_filters` selects. This is the step 3 PRECONDITION, and
               it is measured here early precisely because step 3 assumes it.

A third section measures REACHABILITY: which corpus questions actually reach
`apply_population` in production today. A divergence only matters if something
reaches it, and step 3 would route the whole filtered corpus through it.

Fails loudly if governed semantics do not load.

    python -m migration_phase0.row_predicate_agreement [--json out.json]
                                                       [--no-reachability]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")


def _corpus() -> List[str]:
    out, seen = [], set()
    for f in CORPORA:
        p = _REPO / f
        if not p.exists():
            continue
        for row in json.loads(p.read_text(encoding="utf-8"))["rows"]:
            q = row.get("question") or ""
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def _family(field: str) -> str:
    f = (field or "").lower()
    for needle, name in (("loan_to_value", "LTV"), ("age", "borrower age"),
                         ("outstanding", "balance"), ("interest", "interest rate"),
                         ("geograph", "geography"), ("months_on_book", "months on book"),
                         ("borrower_type", "borrower type"),
                         ("number_of_borrowers", "borrower type"),
                         ("pipeline_stage", "pipeline stage")):
        if needle in f:
            return name
    return f"other:{field}"


def _cause(claims, frame, semantics, unavailable) -> str:
    """Why did the two paths select different rows? Named, not guessed.

    `_apply_filters` normalises the VALUE before comparing (a percent threshold
    against a column stored as a fraction is divided by 100) and RAISES for an
    absent column. `apply_population` does neither: it delegates only the
    comparator to `_apply_numeric_op`, and it records an absent column as
    unavailable while leaving the frame alone.
    """
    from mi_agent.mi_dataset_profile import PERCENT_FRACTION, percent_storage_scale
    fields = (semantics or {}).get("fields") or {}
    if unavailable:
        return "absent_column (executor raises; apply_population widens)"
    for c in claims:
        entry = fields.get(c.field_key) or {}
        col = entry.get("canonical_field") or c.field_key
        if entry.get("format") != "percent" or col not in frame.columns:
            continue
        if percent_storage_scale(frame[col]) != PERCENT_FRACTION:
            continue
        vals = c.value if isinstance(c.value, (list, tuple)) else [c.value]
        if any(isinstance(v, (int, float)) and abs(v) > 1.5 for v in vals):
            return "percent_scale (executor rescales points->fraction)"
    return "unclassified"


def _reachability(questions, semantics) -> Dict[str, Any]:
    """Which questions actually reach `apply_population` in production today?

    A spy on the module attribute, so this observes the shipped call graph
    rather than re-deriving which routes narrow which way.
    """
    from demo_platform import config as cfg
    from mi_agent import population as P
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    seen: Dict[str, List[Any]] = {}
    current = {"q": None}
    original = P.apply_population

    def spy(frame, predicates, sem=None):
        preds = list(predicates or [])
        out = original(frame, preds, sem)
        seen.setdefault(current["q"], []).append(
            [(p.field, p.op, p.value) for p in preds])
        return out

    P.apply_population = spy
    try:
        ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
        for q in questions:
            current["q"] = q
            try:
                execute_governed_mi_query(MiQueryRequest(question=q), ctx)
            except Exception:  # noqa: BLE001 - a refusal is a valid outcome here
                pass
    finally:
        P.apply_population = original
    return seen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--no-reachability", action="store_true")
    args = ap.parse_args(argv)

    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)

    from migration_phase0.assurance_semantics import load_assurance_semantics
    from mi_agent.parsed_question import ParsedQuestion
    from mi_agent.mi_query_executor import _apply_filters
    from mi_agent.population import Predicate, apply_population
    from mi_agent_api import evolution as evolution_mod
    from question_interpretation import projection as proj
    from mi_agent import execution_receipt as R

    semantics = load_assurance_semantics()
    root = os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"]
    frames = evolution_mod.funded_frames(root, cfg.CLIENT_ID, None)
    if not frames:
        raise SystemExit("ASSURANCE INVALID - no governed funded frames to test against")
    frame = frames[-1]["df"]
    if frame is None or not len(frame):
        raise SystemExit("ASSURANCE INVALID - the governed frame is empty")

    print("=" * 92)
    print("ROW PREDICATE AGREEMENT — contract description vs shipped execution")
    print("=" * 92)
    print(f"frame: {len(frame):,} rows, {len(frame.columns)} columns")

    rows: List[Dict[str, Any]] = []
    fam_ok, fam_bad = Counter(), Counter()
    questions = predicates = 0
    agree_rows = disagree_rows = 0
    field_ok = op_ok = value_ok = 0
    unresolved = 0

    for q in _corpus():
        # THE PRODUCTION SHAPE, deliberately. `projection.project` is the
        # read-only Stage 1 harness: it calls `_deterministic_parse` directly
        # and so never sees `resolve_seasoning_role`, which runs inside
        # `parse_with_repair` and MUTATES `spec.filters` (that is where "new
        # lending" becomes `months_on_book <= 1`). Production assembles through
        # `from_parts` on the spec `ParsedQuestion.parse` returns, which carries
        # the seasoning predicate. Measuring `project()` here would have scored
        # the contract against filters the executor never received, and would
        # have understated the predicate count by exactly the seasoning cases.
        spec = ParsedQuestion.parse(q, semantics).spec
        spec_filters = dict(getattr(spec, "filters", None) or {})
        if not spec_filters:
            continue
        cols = list(frame.columns)
        dim_terms = R.requested_dimension_terms(q, semantics, cols)
        facets = R.detect_requested_facets(q, semantics, frame=frame,
                                           requested_dimensions=dim_terms)
        qi = proj.from_parts(q, spec=spec, facets=facets, dim_terms=dim_terms,
                             semantics=semantics)
        claims = list(getattr(qi, "row_predicates", None) or [])
        questions += 1
        predicates += len(claims)

        # SHIPPED: the executor resolves and applies from the spec.
        applied_fields: List[str] = []
        try:
            shipped = _apply_filters(frame.copy(), spec, semantics, [], applied_fields)
            shipped_idx = set(shipped.index)
            shipped_ok = True
        except Exception as exc:  # noqa: BLE001 - a controlled validation failure
            shipped_idx, shipped_ok = set(), False
            unresolved += 1

        # CONTRACT: the same rows, described only by the claims.
        contract_preds = [Predicate(c.field_key, c.operator, c.value) for c in claims]
        narrowed, evidence = apply_population(frame, contract_preds, semantics)
        contract_idx = set(narrowed.index) if narrowed is not None else set()

        same = shipped_ok and shipped_idx == contract_idx
        if shipped_ok:
            if same:
                agree_rows += 1
            else:
                disagree_rows += 1

        # Per-predicate field/op/value against the spec entry the executor used.
        for c in claims:
            raw = spec_filters.get(c.field_key)
            f_ok = c.field_key in spec_filters
            if isinstance(raw, dict):
                o_ok = str(raw.get("op")) == str(c.operator)
                v_ok = raw.get("value") == c.value
            else:
                o_ok = c.operator in ("eq", "in")
                v_ok = (raw == c.value) or (isinstance(raw, (list, tuple, set))
                                            and list(raw) == c.value)
            field_ok += int(f_ok); op_ok += int(o_ok); value_ok += int(v_ok)
            (fam_ok if (f_ok and o_ok and v_ok and same) else fam_bad)[
                _family(c.field_key)] += 1

        rows.append({
            "question": q, "spec_filters": spec_filters,
            "claims": [{"field": c.field_key, "op": c.operator, "value": c.value}
                       for c in claims],
            "shipped_rows": len(shipped_idx) if shipped_ok else None,
            "contract_rows": len(contract_idx),
            "row_sets_identical": same,
            "executor_applied": list(applied_fields),
            "unavailable": list(evidence.unavailable),
            "cause": None if same else _cause(claims, frame, semantics,
                                              list(evidence.unavailable)),
        })

    print(f"\nquestions carrying filters : {questions}")
    print(f"predicates on the contract : {predicates}")
    print(f"\nFIELD  agreement : {field_ok}/{predicates}")
    print(f"OP     agreement : {op_ok}/{predicates}")
    print(f"VALUE  agreement : {value_ok}/{predicates}")
    print(f"\nROW-SET identity (contract selects exactly what the executor selects):")
    print(f"   identical  : {agree_rows}/{questions}")
    print(f"   DIFFERENT  : {disagree_rows}")
    print(f"   executor raised (controlled validation failure): {unresolved}")

    print(f"\n{'family':<20}{'agree':>7}{'disagree':>10}")
    for fam in sorted(set(fam_ok) | set(fam_bad)):
        print(f"{fam:<20}{fam_ok[fam]:>7}{fam_bad[fam]:>10}")

    bad = [r for r in rows if not r["row_sets_identical"]]
    if bad:
        causes = Counter(r["cause"] for r in bad)
        print(f"\nDISAGREEMENT CAUSES ({len(bad)} questions):")
        for cause, n in causes.most_common():
            print(f"   {n:>4}  {cause}")
        print("\n   examples:")
        for r in bad[:6]:
            print(f"      shipped={r['shipped_rows']} contract={r['contract_rows']} "
                  f"{r['claims']}  | {r['question'][:46]}")

    if not args.no_reachability:
        reached = _reachability([r["question"] for r in rows], semantics)
        print(f"\nREACHABILITY — questions reaching apply_population today: "
              f"{len(reached)}/{len(rows)}")
        exposed = [q for q in reached
                   if not next(r["row_sets_identical"] for r in rows
                               if r["question"] == q)]
        for q, calls in list(reached.items())[:8]:
            print(f"   {calls[0]}  | {q[:52]}")
        print(f"\n   of those, DISAGREEING (a live wrong answer): {len(exposed)}")
        print("   VERDICT: " + (
            "the divergence is LATENT — nothing that reaches "
            "apply_population today disagrees. It becomes LIVE the moment "
            "apply_population becomes the plan-level population primitive."
            if not exposed else
            "the divergence is LIVE on the questions listed above."))

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
