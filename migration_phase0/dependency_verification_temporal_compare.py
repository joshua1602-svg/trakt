#!/usr/bin/env python3
"""migration_phase0/dependency_verification_temporal_compare.py

READ-ONLY. Conversion 5 §4 — prove, BEFORE any production change, that
``temporal_compare`` requires exactly the generic work the C5 cost-regime
re-baseline enumerated, and no more.

For every question the route actually owns, at every workspace tab, this puts
the PRODUCTION reading and the CONTRACT reading of the same fact side by side:

    dataset   production what `_route_compare` actually computes.
              contract   `interpretation.dataset.dataset`.

              READ THIS BEFORE BELIEVING A ZERO. Since the dataset ownership
              remediation these two share ONE owner,
              `workspace.resolve_dataset`, so this is a WIRING check — does the
              contract carry the owner's answer to the route? — and no longer
              an AGREEMENT check between two rules. A zero here means the
              handoff is intact, not that two independent readings happen to
              coincide; the second reading no longer exists.

              Before the remediation it compared `chat_routing._dataset_for`
              against the contract and found 3 disagreements in 26 readings,
              plus 7 more that were cured by wiring the resolved view in.
    measure   production `(spec.metric, spec.aggregation)` -> resolve_metric_key
              contract   `subject.candidate_concept` -> the same resolver
    periods   production `spec.compare_periods`
              contract   `time.comparison_periods` -- the STRUCTURAL pair, and
              `time.comparison_period` for the wording beside it

A disagreement here is not a detail. It is the difference between a bounded
accessor and a semantic-owner change, and the whole C5 budget rests on which
of the two this is.

    python -m migration_phase0.dependency_verification_temporal_compare
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from migration_phase0.route_ownership_temporal_compare import CASES, DATASETS  # noqa: E402

#: Only the cases executed routing proved the route owns.
OWNED: Tuple[Tuple[str, str], ...] = tuple(
    (c, q) for c, q, other in CASES if other is None)


def _env() -> Tuple[str, Dict[str, Any]]:
    import logging
    import warnings
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    # THE governed semantics, via the one assurance loader, which delegates to
    # production. This used to probe three plausible loader names on
    # `mi_service`, none of which exist, and fall through to {} - so every
    # measurement below ran against an EMPTY registry and still printed clean
    # numbers. `load_assurance_semantics` raises instead of degrading.
    from migration_phase0.assurance_semantics import load_assurance_semantics
    sem = load_assurance_semantics()
    return cfg.CLIENT_ID, sem


def _contract_measure(qi, dataset: str, resolve) -> Tuple[Any, ...]:
    """The measure key the CONTRACT can reach, with no second owner consulted.

    `subject.candidate_concept` collapses the parser's (metric, aggregation)
    pair onto one governed concept: it carries the metric when the parser named
    one, and the literal concept `loan_count` when the parser named none but
    counted. Reconstructing the resolver's two inputs from that one value is
    the whole question this line answers.
    """
    concept = getattr(qi.subject, "candidate_concept", None)
    if concept == "loan_count":
        return resolve(dataset, None, "count")
    return resolve(dataset, concept, "sum")


def main() -> int:
    client_id, semantics = _env()
    from mi_agent.parsed_question import ParsedQuestion
    from mi_agent_api import workspace as ws
    from mi_agent_api.temporal_compare import resolve_metric_key
    from question_interpretation import projection as proj
    from mi_agent import execution_receipt as R

    print("=" * 78)
    print("temporal_compare — C5 §4 DEPENDENCY VERIFICATION")
    print(f"owned cases: {len(OWNED)}  x tabs {DATASETS} "
          f"= {len(OWNED) * len(DATASETS)} readings")
    print("=" * 78)

    rows: List[Dict[str, Any]] = []
    ds_disagree_today = ds_disagree_wired = 0
    measure_disagree = 0
    periods_structural = 0

    for case, question in OWNED:
        spec = ParsedQuestion.parse(question, semantics).spec
        dim_terms = R.requested_dimension_terms(question, semantics, None)
        facets = list(R.detect_requested_facets(question, semantics, frame=None,
                                                requested_dimensions=dim_terms))
        for tab in DATASETS:
            # EXACTLY what mi_service hands the router, and exactly what
            # `_route_compare` computes -- the one owner, tab-independent.
            view = ws.resolve_dataset(question)
            prod_ds = ws.resolve_dataset(question)
            qi_today = proj.from_parts(question, spec=spec, facets=facets,
                                       dim_terms=dim_terms, semantics=semantics,
                                       registry=None, caller_scope=None,
                                       caller_dataset=None)
            qi_wired = proj.from_parts(question, spec=spec, facets=facets,
                                       dim_terms=dim_terms, semantics=semantics,
                                       registry=None, caller_scope=None,
                                       caller_dataset=view)
            c_today = qi_today.dataset.dataset
            c_wired = qi_wired.dataset.dataset

            prod_measure = resolve_metric_key(prod_ds, getattr(spec, "metric", None),
                                              getattr(spec, "aggregation", ""))
            # The measure is compared AT THE SAME DATASET, so a dataset
            # disagreement cannot masquerade as a measure disagreement.
            ctr_measure = _contract_measure(qi_wired, prod_ds, resolve_metric_key)

            cp = qi_wired.time.comparison_period
            prod_periods = list(getattr(spec, "compare_periods", None) or [])
            # The pair as the CONTRACT states it, compared to the parser's own
            # list. A structural reading that does not match the production
            # value would be worse than none: it would look closed.
            contract_periods = list(
                getattr(qi_wired.time, "comparison_periods", ()) or ())
            structural = contract_periods if (
                contract_periods and contract_periods == prod_periods) else []

            d_today = prod_ds != c_today
            d_wired = prod_ds != c_wired
            d_measure = prod_measure != ctr_measure
            ds_disagree_today += d_today
            ds_disagree_wired += d_wired
            measure_disagree += d_measure
            periods_structural += bool(structural)

            flag = "!!" if (d_wired or d_measure) else ("~ " if d_today else "  ")
            print(f"{flag} {case:<3} tab={tab:<8} "
                  f"dataset prod={prod_ds:<9} contract(today)={str(c_today):<9} "
                  f"contract(view wired)={str(c_wired):<9}")
            print(f"      measure prod={prod_measure[0]:<28} contract={str(ctr_measure[0]):<28}")
            print(f"      periods prod={prod_periods!s:<32} "
                  f"contract.raw_text={getattr(cp, 'raw_text', None)!r} "
                  f"structural={structural or 'NONE'}")
            rows.append({
                "case": case, "question": question, "tab": tab,
                "resolvedView": view,
                "datasetProduction": prod_ds,
                "datasetContractToday": c_today,
                "datasetContractViewWired": c_wired,
                "datasetDisagreesToday": d_today,
                "datasetDisagreesWired": d_wired,
                "measureProduction": list(prod_measure),
                "measureContract": list(ctr_measure),
                "measureDisagrees": d_measure,
                "periodsProduction": prod_periods,
                "periodsContractRawText": getattr(cp, "raw_text", None),
                "periodsContractStructural": structural,
            })

    out = _REPO / "migration_phase0" / "DEPENDENCY_TEMPORAL_COMPARE.json"
    out.write_text(json.dumps({"rows": rows}, indent=2, default=str))

    n = len(rows)
    print("\n" + "=" * 78)
    print(f"readings                                   : {n}")
    print(f"dataset disagreements, contract AS BUILT    : {ds_disagree_today}")
    print(f"dataset disagreements, WITH the view wired  : {ds_disagree_wired}")
    print(f"measure disagreements (at the same dataset) : {measure_disagree}")
    # THE DENOMINATOR IS ASSERTED. Every owned case is a two-period comparison
    # -- that is what makes it a `temporal_compare` question -- so every reading
    # must carry the pair. A number lower than n is a gap; a number that merely
    # rose is not evidence of anything.
    expected_structural = n
    print(f"readings whose periods are STRUCTURAL       : {periods_structural} "
          f"of an EXPECTED {expected_structural}")
    if periods_structural != expected_structural:
        print("!! the comparison-period pair is NOT carried structurally on "
              "every owned reading — the C5 dependency model still has a gap")
    print(f"written                                    : {out.relative_to(_REPO)}")
    print("=" * 78)
    gaps = (ds_disagree_wired or measure_disagree
            or periods_structural != expected_structural)
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())
