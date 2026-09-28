#!/usr/bin/env python3
"""Slice 1B local acceptance: eight cases, no live model call, one question each.

WHAT IT ADJUDICATES. `bank.json` was written and hashed BEFORE this ran. It
carries, per case, the expected eligibility, the expected plan identity, the
expected bound semantics, and — for the six eligible cases — the figure or the
grid an INDEPENDENT oracle computes. The oracle is
`mi_agent.tests.portfolio_truth_oracle`, which imports nothing from the product:
a boolean mask, a groupby and a sum, written longhand. Asking the product what
the answer should be would be the product marking its own homework.

WHAT IT COMPARES, for each of the six eligible cases:

    plan semantics      the plan the frozen interpreter + compiler produce is
                        the plan the bank registered — same plan_id, same
                        statistic, same measure, same axes, same predicates
    shadow vs serving   slice 1A's shadow and slice 1B's serving path agree on
                        the figure, the grid and the receipt. Slice 1B must
                        change only WHETHER a result ships, never what it is
    oracle              the served figure, and every served cell, equals the
                        independently computed one
    provenance          the envelope that came back is the NEW deterministic
                        result and not the legacy one

THE FRAME. The oracle's canonical book, plus the two band columns the deployed
frame carries and it does not (`ticket_bucket`, `interest_rate_bucket`). Both the
product and the oracle read the SAME column, so the banding rule is irrelevant to
every claim here — this is about filtering, grouping and aggregation.

NO LIVE MODEL CALLS. The interpreter is a `ReplayClient` over the frozen run-8
sign-off payloads. A question not in that file cannot be asked.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REPO_ROOT = HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd                                                   # noqa: E402

from mi_agent import plan_runtime_adapter as adapter                  # noqa: E402
from mi_agent import plan_serving_canary as canary                    # noqa: E402
from mi_agent import plan_shadow_evidence as evidence                 # noqa: E402
from mi_agent import plan_shadow_wiring as wiring                     # noqa: E402
from mi_agent.interpretation_v2.opus_interpreter import (              # noqa: E402
    OpusInterpreter, ReplayClient)
from mi_agent.mi_query_validator import load_mi_semantics             # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth            # noqa: E402

FROZEN = (_REPO_ROOT / "mi_agent" / "interpretation_v2" / "evidence"
          / "run8_135_signoff_2b00172.json")
BANK = HERE / "bank.json"

CANARY_PRINCIPAL = "11111111-2222-3333-4444-555555555555"
OTHER_PRINCIPAL = "99999999-8888-7777-6666-555555555555"
TOLERANCE = 0.01


class Principal:
    def __init__(self, actor_id):
        self.actor_id = actor_id


def bank_frame():
    """The oracle's book, plus the two bands the deployed frame also carries."""
    book = truth.canonical_book()
    book["ticket_bucket"] = pd.cut(
        book[truth.BALANCE],
        bins=[-1, 100_000, 200_000, 350_000, float("inf")],
        labels=["<100k", "100-200k", "200-350k", "350k+"]).astype(str)
    book["interest_rate_bucket"] = pd.cut(
        book[truth.RATE], bins=[-1, 4, 5, 6, float("inf")],
        labels=["<4%", "4-5%", "5-6%", "6%+"]).astype(str)
    return book


def grid_hash(cells):
    body = json.dumps(sorted([list(key) + [round(value, 6)]
                              for key, value in cells.items()]))
    return hashlib.sha256(body.encode()).hexdigest()


def served_cells(record):
    """The grid the serving path recorded, as the oracle's `{key tuple: value}`."""
    out = {}
    for cell in (record["execution"].get("grouped_cells") or ()):
        key = tuple(str(cell[axis]) for axis in
                    record["execution"]["requested_semantics"]["dimensions"])
        out[key] = float(cell["value"])
    return out


def artefact_cells(envelope, dimensions):
    """The grid the SERVED ENVELOPE actually carries, read off its artifacts.

    The record is evidence; this is what the caller received. Proving the two
    agree is what makes "the response returned came from NEW" a fact about the
    response rather than about the record.
    """
    for artefact in (envelope.get("artifacts") or ()):
        rows = artefact.get("rows") or artefact.get("data") or ()
        if not rows or not isinstance(rows[0], dict):
            continue
        if not all(axis in rows[0] for axis in dimensions):
            continue
        value_keys = [k for k in rows[0]
                      if k not in dimensions
                      and isinstance(rows[0][k], (int, float))
                      and not isinstance(rows[0][k], bool)]
        if not value_keys:
            continue
        return {tuple(str(row[axis]) for axis in dimensions):
                float(row[value_keys[0]]) for row in rows}
    return {}


def scalar_of_envelope(envelope, record):
    """The PLAN'S figure in the served envelope, from the KPI naming its field.

    Not the first KPI: a KPI artefact carries the loan count alongside the
    measure, so "the first one" read 130 loans for a question about £37.9MM of
    balance. The field name is taken from the BOUND SPEC the record holds —
    `loan_count` for a count, `<measure>_<aggregation>` otherwise — which is the
    same name `plan_runtime_adapter._scalar_of` reads.
    """
    spec = record["execution"].get("bound_spec") or {}
    aggregation = str(spec.get("aggregation") or "")
    wanted = ("loan_count" if aggregation == "count"
              else f"{spec.get('metric')}_{aggregation}")
    for artefact in (envelope.get("artifacts") or ()):
        for kpi in (artefact.get("kpis") or ()):
            raw = kpi.get("rawValue")
            if (kpi.get("field") == wanted
                    and isinstance(raw, (int, float))
                    and not isinstance(raw, bool)):
                return float(raw)
    return None


def same_grid(left, right):
    """Two grids, compared at the oracle's tolerance rather than bit-for-bit.

    The record holds a full float and the envelope holds its JSON rendering —
    280813.04000000004 and 280813.04 are the same figure, and an `==` here called
    them a divergence.
    """
    if set(left) != set(right):
        return False
    return all(abs(left[key] - right[key]) <= TOLERANCE for key in left)


def main() -> int:
    bank_text = BANK.read_text()
    bank = json.loads(bank_text)
    bank_sha = hashlib.sha256(bank_text.encode()).hexdigest()

    payloads = {r["question"]: r["raw_payload"]
                for r in json.loads(FROZEN.read_text())["results"]
                if r.get("raw_payload")}
    built = []

    def factory():
        built.append(1)
        return OpusInterpreter(ReplayClient(payloads, model_id="claude-opus-5"))

    semantics = load_mi_semantics(
        str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
    book = bank_frame()

    tmp = tempfile.mkdtemp()
    os.environ[evidence.SINK_ENV_VAR] = os.path.join(tmp, "evidence.jsonl")
    os.environ[canary.SERVE_ENV_VAR] = "canary"
    os.environ[canary.PRINCIPALS_ENV_VAR] = CANARY_PRINCIPAL
    os.environ[adapter.SHADOW_ENV_VAR] = "off"
    wiring.set_interpreter_factory(factory)

    report = {"bank_sha256": bank_sha, "bank_version": bank["bank_version"],
              "live_model_calls": 0, "cases": [],
              "isolation": {}, "totals": {}}
    defects = []

    # ISOLATION, proved before anything is served: the same configuration, a
    # different individual, nothing served and no plan built.
    before = len(built)
    report["isolation"] = {
        "canary_principal_handled": canary.handles(Principal(CANARY_PRINCIPAL)),
        "other_principal_handled": canary.handles(Principal(OTHER_PRINCIPAL)),
        "no_identity_handled": canary.handles(None),
        "interpreters_built_by_the_membership_test": len(built) - before,
    }
    if (not report["isolation"]["canary_principal_handled"]
            or report["isolation"]["other_principal_handled"]
            or report["isolation"]["no_identity_handled"]
            or report["isolation"]["interpreters_built_by_the_membership_test"]):
        defects.append("isolation: the membership test did not behave")

    for case in bank["cases"]:
        question = case["question"]
        legacy = {"ok": True, "value": -1.0, "answer": "the legacy answer",
                  "artifacts": [], "warnings": []}
        before_legacy = json.dumps(legacy, sort_keys=True)

        # THE SHADOW (slice 1A) on the same question and the same frame, so the
        # two paths can be compared rather than assumed equal.
        os.environ[adapter.SHADOW_ENV_VAR] = "shadow"
        os.environ[wiring.CANARY_ENV_VAR] = "acme"
        wiring.set_dispatch(wiring._dispatch_inline)
        shadow = wiring.observe_request(
            question=question, client_id="acme", run_id="2026-03",
            result=legacy, frame=book, semantics=semantics, view="funded",
            portfolio_id="acme/2026-03")
        wiring.set_dispatch(None)
        os.environ[adapter.SHADOW_ENV_VAR] = "off"
        os.environ.pop(wiring.CANARY_ENV_VAR, None)

        # THE SERVING PATH (slice 1B).
        envelope = canary.serve(
            question=question, context=Principal(CANARY_PRINCIPAL),
            client_id="acme", run_id="2026-03", legacy_result=legacy,
            frame=book, semantics=semantics, view="funded",
            portfolio_id="acme/2026-03", render_portfolio_id="acme/2026-03",
            as_of=None)
        rows = [json.loads(line) for line in
                Path(os.environ[evidence.SINK_ENV_VAR]).read_text().splitlines()
                if line.strip()]
        record = rows[-1]

        row = {"case_id": case["case_id"], "question": question,
               "expected_eligible": case["expected_eligible"],
               "plan_id": record["compiler"].get("plan_id"),
               "plan_id_matches_bank":
                   record["compiler"].get("plan_id") == case["expected_plan_id"],
               "eligible": bool((record.get("eligibility") or {}).get("eligible")),
               "ineligibility_reason":
                   (record.get("eligibility") or {}).get("reason") or "",
               "served": "NEW" if envelope is not None else "LEGACY_FALLBACK",
               "record_says_served":
                   record["serving"]["response_served_from"],
               "legacy_envelope_unmutated":
                   json.dumps(legacy, sort_keys=True) == before_legacy,
               "model_id": record["model"].get("model_id"),
               "semantics_match_bank": None, "shadow_agrees": None,
               "oracle_value": case["oracle_value"],
               "served_value": None, "oracle_cells": case["oracle_cells"],
               "served_cells": None, "grid_sha256_matches_oracle": None,
               "envelope_grid_matches_record": None, "notes": []}

        def defect(text):
            row["notes"].append(text)
            defects.append(f"{case['case_id']}: {text}")

        if not row["plan_id_matches_bank"]:
            defect("the plan is not the plan the bank registered")
        if not row["legacy_envelope_unmutated"]:
            defect("the legacy envelope was mutated")
        if row["model_id"] != "claude-opus-5":
            defect(f"model substitution: {row['model_id']!r}")
        if row["eligible"] != case["expected_eligible"]:
            defect(f"eligibility mismatch: {row['eligible']} != "
                   f"{case['expected_eligible']}")
        if row["served"] != row["record_says_served"]:
            defect("the record disagrees with what was served")

        if not case["expected_eligible"]:
            # A CONTROL. It must be refused for the registered reason, and the
            # legacy envelope must be what serves.
            if row["ineligibility_reason"] != case["expected_ineligibility_reason"]:
                defect(f"wrong ineligibility reason: "
                       f"{row['ineligibility_reason']!r}")
            if envelope is not None:
                defect("an ineligible control was SERVED")
            report["cases"].append(row)
            continue

        requested = record["execution"].get("requested_semantics") or {}
        row["semantics_match_bank"] = (
            requested.get("statistic") == case["expected_statistic"]
            and requested.get("measure_field") == case["expected_measure_field"]
            and list(requested.get("dimensions") or ()) == case["expected_dimensions"]
            and [[f["field"], f["comparator"], f["value"]]
                 for f in (requested.get("filters") or ())]
            == case["expected_filters"])
        if not row["semantics_match_bank"]:
            defect("the bound semantics are not the registered ones")

        if envelope is None:
            defect(f"an eligible case was NOT served: "
                   f"{record['serving'].get('reason')}")
            report["cases"].append(row)
            continue

        # SHADOW vs SERVING. Slice 1B changes whether a result ships, not what
        # it is, so the two paths must agree on the figure and the receipt.
        shadow_value = (shadow or {}).get("execution", {}).get("value")
        row["shadow_agrees"] = (
            shadow_value == record["execution"]["value"]
            and ((shadow or {}).get("eligibility") or {}).get("eligible") is True)
        if not row["shadow_agrees"]:
            defect(f"slice 1A and slice 1B disagree: shadow={shadow_value!r} "
                   f"serving={record['execution']['value']!r}")

        if case["expected_dimensions"]:
            cells = served_cells(record)
            oracle = truth.grouped(book, case["expected_dimensions"],
                                   column=truth.BALANCE, how="sum")
            row["served_cells"] = len(cells)
            row["grid_sha256_matches_oracle"] = (
                grid_hash(cells) == case["oracle_grid_sha256"]
                == grid_hash(oracle))
            if not row["grid_sha256_matches_oracle"]:
                missing = set(oracle) - set(cells)
                wrong = {k: (oracle[k], cells[k]) for k in set(oracle) & set(cells)
                         if abs(oracle[k] - cells[k]) > TOLERANCE}
                defect(f"grid differs from the oracle: {len(missing)} missing "
                       f"cell(s), {len(wrong)} wrong value(s)")
            from_envelope = artefact_cells(envelope, case["expected_dimensions"])
            row["envelope_grid_matches_record"] = (same_grid(from_envelope, cells)
                                                   if from_envelope else None)
            if not from_envelope:
                defect("the served envelope carries no grid for a grouped plan")
            elif not same_grid(from_envelope, cells):
                defect("the served envelope's grid is not the recorded grid")
        else:
            row["served_value"] = record["execution"]["value"]
            if abs(float(row["served_value"]) - float(case["oracle_value"])) \
                    > TOLERANCE:
                defect(f"figure differs from the oracle: "
                       f"served={row['served_value']} "
                       f"oracle={case['oracle_value']}")
            from_envelope = scalar_of_envelope(envelope, record)
            row["envelope_value"] = from_envelope
            if from_envelope is None or abs(from_envelope
                                            - float(case["oracle_value"])) > TOLERANCE:
                defect(f"the served ENVELOPE does not carry the deterministic "
                       f"figure (it carries {from_envelope!r})")
            if abs(float(row["served_value"]) - float(legacy["value"])) < TOLERANCE:
                defect("the served figure is the legacy control's")

        report["cases"].append(row)

    wiring.set_interpreter_factory(None)
    eligible = [c for c in report["cases"] if c["expected_eligible"]]
    report["live_model_calls"] = 0
    report["totals"] = {
        "cases": len(report["cases"]),
        "eligible": len(eligible),
        "served_new": sum(1 for c in report["cases"] if c["served"] == "NEW"),
        "ineligible": sum(1 for c in report["cases"]
                          if not c["expected_eligible"]),
        "legacy_fallback": sum(1 for c in report["cases"]
                               if c["served"] == "LEGACY_FALLBACK"),
        "numerical_diffs": sum(1 for c in report["cases"]
                               if c["grid_sha256_matches_oracle"] is False
                               or (c["served_value"] is not None
                                   and c["oracle_value"] is not None
                                   and abs(float(c["served_value"])
                                           - float(c["oracle_value"])) > TOLERANCE)),
        "semantic_drops": sum(1 for c in report["cases"]
                              if c["semantics_match_bank"] is False
                              or not c["plan_id_matches_bank"]),
        "shadow_disagreements": sum(1 for c in report["cases"]
                                    if c["shadow_agrees"] is False),
        "defects": len(defects),
    }
    report["defects"] = defects
    report["verdict"] = "PASS" if not defects else "FAIL"

    out = HERE / "slice1b_local_acceptance.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")

    print(f"bank               {bank_sha}")
    print(f"live model calls   {report['live_model_calls']}")
    print(f"isolation          canary={report['isolation']['canary_principal_handled']} "
          f"other={report['isolation']['other_principal_handled']} "
          f"anonymous={report['isolation']['no_identity_handled']}")
    for row in report["cases"]:
        print(f"  {row['case_id']}  {row['served']:16s} "
              f"{'ELIGIBLE' if row['eligible'] else row['ineligibility_reason']:24s} "
              f"{row['plan_id']}")
    print(f"totals             {json.dumps(report['totals'])}")
    for text in defects:
        print(f"  DEFECT  {text}")
    print(f"verdict            {report['verdict']}")
    print(f"written            {out.name}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
