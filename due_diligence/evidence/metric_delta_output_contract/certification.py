#!/usr/bin/env python3
"""The requested-metric output contract, as a certification gate.

WHY THIS EXISTS. The confirmation canary scored M03 a PASS. Its route was
`governed_plan_metric_delta`, its owner was `run_period_change_analysis`, its
mode was `requested_metric`, and `current_interest_rate` appeared in both
`requested_fields` and `selected_measures`. Every gate that bank pinned held —
and the served answer did not contain the −3.21 pp the receipt held. A
certification that reads only the route, the owner and the receipt cannot tell a
served answer from a served silence.

WHAT IS CHECKED, AND FROM WHERE. Three fields and one disposition, each taken
from structured evidence rather than from prose:

    REQUESTED_FIELD   receipt.requested_fields
    EXECUTED_FIELD    receipt.selected_measures
    SERVED_METRIC_FIELD  the "Metric movements" artefact's own canonical_field
    DISPOSITION       receipt.requested_metric_disposition

WHY THE ANSWER TEXT IS READ AT ALL. Because the defect lived there and nowhere
else: in M03 the receipt was right, the metric table was right, and the prose
was wrong. So the answer is checked — but the tokens it must contain are DERIVED
FROM THE RECEIPT, not written here. The display name comes from the served
table and the movement figure from the owner's own value, so this asserts
"the answer states what was computed" without pinning a sentence. No governed
wording is fixed and rewording the narrative cannot fail this.

IT SCORES; IT DOES NOT EXECUTE. Nothing here calls a model, an endpoint or an
owner. It is given a served envelope and its evidence record and returns a
verdict, so it can be exercised offline against fixtures — which `--self-test`
does, in both directions of the M03 fault.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

ANSWERED = "ANSWERED"
QUALIFIED = "QUALIFIED"
GOVERNED_REFUSAL = "GOVERNED_REFUSAL"
DISPOSITIONS = (ANSWERED, QUALIFIED, GOVERNED_REFUSAL)

#: A disposition that must put a figure in front of the reader. A governed
#: refusal must not, and must say which metric it is refusing about.
STATES_A_FIGURE = (ANSWERED, QUALIFIED)


def _artifact_rows(envelope: Mapping[str, Any], title: str) -> List[Dict[str, Any]]:
    for artifact in envelope.get("artifacts") or ():
        if str(artifact.get("title") or "").strip().lower() == title.lower():
            return list(artifact.get("rows") or ())
    return []


def _movement_tokens(row: Mapping[str, Any]) -> List[str]:
    """The rendered movement, TAKEN FROM THE SERVED TABLE ITSELF.

    THIS USED TO IMPORT THE PRODUCT'S FORMATTER, AND THAT WAS WRONG TWICE OVER.
    `mi_agent_api.period_change_route` drags in the whole `mi_agent` package —
    pandas, yaml, plotly — and this harness is stdlib-only on purpose, so no
    dependency install stands between a dispatch and the gate. It passed on a
    workstation that happened to have them and died on the runner. Reimplementing
    the formatter here would have been worse: a second renderer that can disagree
    with the first is exactly what this gate exists to catch.

    The served envelope already carries the answer. `_metric_rows` builds the
    "Metric movements" table with `_format_movement`, the same function and the
    same inputs the answer clause uses, in the same response. So the token comes
    from the product, rendered by the product, in the very request being scored —
    with no import and no second formatter anywhere.
    """
    rendered = str(row.get("movement") or "").strip()
    # "—" is the table's own placeholder for "no movement was computed".
    return [rendered] if rendered and rendered != "\u2014" else []


def score_requested_metric(case: Mapping[str, Any], envelope: Mapping[str, Any],
                           receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """One requested-metric case. `(fields, disposition, failures)`."""
    failures: List[str] = []

    requested = list(receipt.get("requested_fields") or ())
    executed = list(receipt.get("selected_measures") or ())
    served_rows = _artifact_rows(envelope, "Metric movements")
    served = [str(row.get("canonical_field")) for row in served_rows]

    expected = list(case.get("requested_fields") or ())
    if expected and requested != expected:
        failures.append(f"requested_fields {requested}, pinned {expected}")
    if not requested:
        failures.append("the receipt records no requested field")
    missing = [f for f in requested if f not in executed]
    if missing:
        failures.append(f"the owner did not analyse {missing}; excluded "
                        f"{receipt.get('excluded_candidates')}")
    not_served = [f for f in requested if f not in served]
    if not_served:
        failures.append(f"{not_served} was executed but does not appear in the "
                        f"served metric table {served}")

    dispositions = {str(row.get("canonical_field")): str(row.get("disposition"))
                    for row in (receipt.get("requested_metric_disposition") or ())}
    if not dispositions:
        failures.append("the receipt records no requested_metric_disposition; a "
                        "requested-metric answer that cannot say how it came out "
                        "cannot be certified")
    for field, disposition in dispositions.items():
        if disposition not in DISPOSITIONS:
            failures.append(f"{field}: disposition {disposition!r} is not one of "
                            f"{list(DISPOSITIONS)}")

    pinned = case.get("expected_disposition")
    if pinned and sorted(dispositions.values()) != sorted(
            pinned if isinstance(pinned, list) else [pinned]):
        failures.append(f"disposition {sorted(dispositions.values())}, pinned "
                        f"{pinned}")

    # THE GATE M03 WOULD HAVE FAILED. The reader must be told about the metric
    # they named, and told the figure where one exists.
    answer = str(envelope.get("answer") or "")
    by_field = {str(row.get("canonical_field")): row for row in served_rows}
    for field, disposition in dispositions.items():
        row = by_field.get(field) or {}
        name = str(row.get("metric") or field)
        if name and name not in answer:
            failures.append(f"the answer never names {name!r}, the metric the "
                            f"reader asked about")
        if disposition not in STATES_A_FIGURE:
            continue
        movement = next((m for m in (receipt.get("metric_movements") or ())
                         if str(m.get("field")) == field), {})
        tokens = _movement_tokens(row)
        if tokens and not any(token in answer for token in tokens):
            failures.append(
                f"{field}: the owner computed {movement.get('movement_value')!r} "
                f"and the answer states no such figure — this is the M03 defect, "
                f"a route that executed correctly and answered a different "
                f"question")

    return {
        "requested_field": requested,
        "executed_field": executed,
        "served_metric_field": served,
        "requested_metric_disposition": dispositions,
        "failures": failures,
        "pass": not failures,
    }


# --------------------------------------------------------------------------- #
# self-test — both directions of the M03 fault, offline and free
# --------------------------------------------------------------------------- #
def _fixture(*, answer: str, disposition: str = QUALIFIED,
             movement: Optional[float] = -3.2147396826825796):
    receipt = {
        "requested_fields": ["current_interest_rate"],
        "selected_measures": ["current_interest_rate"],
        "excluded_candidates": [],
        "metric_movements": [{"field": "current_interest_rate",
                              "movement_value": movement,
                              "movement_unit": "percentage_point",
                              "status": "partially_available"}],
        "requested_metric_disposition": [
            {"canonical_field": "current_interest_rate",
             "disposition": disposition, "status": "partially_available"}],
    }
    envelope = {
        "answer": answer,
        "artifacts": [{"title": "Metric movements",
                       "rows": [{"canonical_field": "current_interest_rate",
                                 "metric": "Current Interest Rate",
                                 # As `_metric_rows` renders it. Not typed by
                                 # hand anywhere the product can be asked.
                                 "movement": ("\u2014" if movement is None
                                              else "\u22123.21 pp")}]}],
    }
    return receipt, envelope


def self_test() -> int:
    case = {"requested_fields": ["current_interest_rate"],
            "expected_disposition": QUALIFIED}
    m03 = ("Between 30 November 2025 and 30 June 2026, 0 of 1 governed metrics "
           "could be compared across both snapshots. The balance bridge "
           "reconciles: opening £8.9m +£150.2m new lending, −£0 exits, +£0 on "
           "continuing loans, closing £159.1m.")
    # The figure is written with the estate's own minus sign, because that is
    # what `_format_movement` produces and this gate compares against it rather
    # than against a hand-typed rendering.
    fixed = ("Current Interest Rate moved \u22123.21 pp between 30 November 2025 and "
             "30 June 2026, from 9.53% to 6.32% (weighted average weighted by "
             "current_outstanding_balance). This is a partially available "
             "comparison. The balance bridge reconciles: opening £8.9m.")
    named_only = ("Current Interest Rate could not be compared. The balance "
                  "bridge reconciles: opening £8.9m.")

    cases = [
        ("the served M03 answer FAILS — bridge only, no figure", m03, case, False),
        ("an answer that states the movement PASSES", fixed, case, True),
        ("naming the metric without its figure still FAILS", named_only, case,
         False),
        ("a governed refusal that names the metric PASSES",
         "Current Interest Rate could not be compared between the two dates.",
         {"requested_fields": ["current_interest_rate"],
          "expected_disposition": GOVERNED_REFUSAL}, True),
    ]
    failed = 0
    for label, answer, spec, expected in cases:
        disposition = spec.get("expected_disposition")
        receipt, envelope = _fixture(
            answer=answer, disposition=disposition,
            movement=(None if disposition == GOVERNED_REFUSAL
                      else -3.2147396826825796))
        verdict = score_requested_metric(spec, envelope, receipt)
        ok = verdict["pass"] is expected
        failed += not ok
        print(f"{'ok ' if ok else 'FAIL'} {label:58} "
              f"pass={verdict['pass']} {'; '.join(verdict['failures'])[:90]}")
    print("certification self-test " + ("PASSED" if not failed else "FAILED"))
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    parser.error("this module is a scorer; --self-test is its only entry point")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
