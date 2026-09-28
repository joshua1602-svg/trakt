#!/usr/bin/env python3
"""The production catalogue binding, proved through `serve`. No model calls.

WHAT IS UNDER TEST. Not the temporal runtime — that is settled — but the
BINDING: that `GovernedFundedSnapshotStore` reads production's own catalogue
contract, that the store reaches the temporal runtime through the real serving
seam, and that the portfolio boundary and the fail-closed rules survive the
trip.

THE CATALOGUE HERE IS PRODUCTION-SHAPED, NOT PRODUCTION. It is the exact
contract `datasets.snapshot_index` returns — `{portfolios: [{client_id, label,
runs: [{run_id, reporting_date, loan_count, ...}]}]}` — built over the
independent truth oracle's history, and it is injected through the adapter's own
`index_provider` / `frame_loader` seams. So what is exercised is the adapter's
reading of the contract; what is NOT exercised is a real blob or a real
onboarding root, which this environment does not have and which this task was
not authorised to stand up.

TWO CLIENTS ARE IN THE CATALOGUE ON PURPOSE. A binding that scopes correctly
against a single-client index has proved nothing about scoping.

Run: `python due_diligence/evidence/plan_temporal_slice2/temporal_production_binding.py`
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import plan_serving_canary as canary                     # noqa: E402
from mi_agent import plan_shadow_evidence as evidence_mod              # noqa: E402
from mi_agent import plan_shadow_wiring as wiring                      # noqa: E402
from mi_agent import plan_temporal_runtime as temporal                 # noqa: E402
from mi_agent.interpretation_v2.opus_interpreter import (               # noqa: E402
    OpusInterpreter, ReplayClient)
from mi_agent.mi_query_validator import load_mi_semantics              # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth             # noqa: E402
from mi_agent_api import governed_snapshot_store as store_mod          # noqa: E402
from snapshot.model import SnapshotNotFoundError                       # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "temporal_production_binding.json"
REGISTRY = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
RETEST = HERE / "temporal_live_retest_result.json"
ORIGINAL = HERE / "temporal_live_run_result.json"
RUN8 = (_REPO_ROOT / "mi_agent" / "interpretation_v2" / "evidence"
        / "run8_135_signoff_2b00172.json")

CANARY_PRINCIPAL = "00000000-1111-2222-3333-444444444444"
THIS_CLIENT = "ERE"
OTHER_CLIENT = "OTHER_TENANT"
CATALOGUE = ["2025-11-30", "2025-12-31", "2026-01-31", "2026-02-28",
             "2026-03-31", "2026-04-30", "2026-05-31", "2026-06-30"]


# --------------------------------------------------------------------------- #
# a production-SHAPED catalogue over independent data
# --------------------------------------------------------------------------- #

def build_catalogue(history: Sequence[Tuple[str, Any]], *,
                    other_client: bool = True) -> Tuple[Dict[str, Any], Dict]:
    """`(index, frames)` in the exact contract `datasets.snapshot_index` returns."""
    runs = [{"run_id": f"run_{date.replace('-', '')}",
             "reporting_date": date,
             "loan_count": int(len(frame)),
             "current_outstanding_balance": round(
                 truth.total(frame, truth.BALANCE), 2)}
            for date, frame in history]
    frames = {(THIS_CLIENT, run["run_id"]): frame
              for run, (_, frame) in zip(runs, history)}

    portfolios = [{"client_id": THIS_CLIENT, "label": "ERE", "runs": runs}]
    if other_client:
        # ANOTHER TENANT, with months this client does not have. If scoping were
        # wrong, a "since March" would find March 2025 here.
        other_runs = [{"run_id": "run_20250331", "reporting_date": "2025-03-31",
                       "loan_count": 1, "current_outstanding_balance": 1.0},
                      {"run_id": "run_20250430", "reporting_date": "2025-04-30",
                       "loan_count": 1, "current_outstanding_balance": 1.0}]
        portfolios.append({"client_id": OTHER_CLIENT, "label": "Other",
                           "runs": other_runs})
        for run in other_runs:
            frames[(OTHER_CLIENT, run["run_id"])] = history[0][1]

    return {"portfolios": portfolios, "source": "production-shaped"}, frames


def make_store(index: Mapping[str, Any], frames: Mapping) -> Any:
    """The real adapter over the shaped catalogue. The loader's own contract."""
    def loader(client_id: str, run_id: str, root: Optional[str]):
        frame = frames.get((client_id, run_id))
        return (frame, {"client_id": client_id, "run_id": run_id})
    return store_mod.GovernedFundedSnapshotStore(
        index_provider=lambda: dict(index), frame_loader=loader,
        output_root=None, client_id=THIS_CLIENT)


# --------------------------------------------------------------------------- #
# driving the real serving path
# --------------------------------------------------------------------------- #

def serve_case(question: str, *, store: Any, semantics: Any, book: Any,
               client_id: str = THIS_CLIENT) -> Tuple[Optional[Dict], Dict]:
    captured: Dict[str, Any] = {}
    original = evidence_mod.write
    try:
        evidence_mod.write = lambda record: captured.update(record)
        payload = canary.serve(
            question=question, context=SimpleNamespace(actor_id=CANARY_PRINCIPAL),
            client_id=client_id, run_id=None,
            legacy_result={"ok": True, "value": 0.0}, frame=book,
            semantics=semantics, view="funded", portfolio_id=client_id,
            as_of=None, snapshot_store=store, snapshot_client_id=client_id,
            snapshot_route=store_mod.FUNDED_ROUTE)
    finally:
        evidence_mod.write = original
    return payload, captured


def oracle(recipe: Mapping[str, Any], frame: Any) -> Any:
    predicates = list(recipe.get("predicates") or ())
    group_by = list(recipe.get("group_by") or ())
    if group_by:
        return truth.grouped(frame, group_by, column=recipe.get("column"),
                             how=recipe["how"], predicates=predicates)
    if recipe["how"] == "count":
        return float(truth.row_count(frame, predicates))
    return truth.total(frame, recipe["column"], predicates)


def payload_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    for artifact in (payload.get("artifacts") or ()):
        rows = artifact.get("rows") or artifact.get("data")
        if rows and isinstance(rows[0], Mapping) \
                and temporal.REPORTING_DATE in rows[0]:
            return [dict(r) for r in rows]
    return []


def load_payloads() -> Tuple[Dict[str, Dict], Dict[str, str]]:
    payloads: Dict[str, Dict] = {}
    questions: Dict[str, str] = {}
    for path in (ORIGINAL, RETEST):
        body = json.loads(path.read_text(encoding="utf-8"))
        for row in body["results"]:
            if row.get("raw_payload"):
                payloads[row["question"]] = row["raw_payload"]
                questions[row["id"]] = row["question"]
    # One slice 1 control, from the frozen run-8 sign-off.
    run8 = json.loads(RUN8.read_text(encoding="utf-8"))
    for row in run8["results"]:
        if row["question_id"] == "Q01A" and row.get("raw_payload"):
            payloads[row["question"]] = row["raw_payload"]
            questions["SLICE1"] = row["question"]
    return payloads, questions


# --------------------------------------------------------------------------- #

def main() -> int:
    payloads, questions = load_payloads()
    semantics = load_mi_semantics(str(REGISTRY))
    history = truth.canonical_history()
    frames = dict(history)
    index, run_frames = build_catalogue(history)

    wiring.set_interpreter_factory(lambda: OpusInterpreter(ReplayClient(payloads)))
    os.environ[canary.SERVE_ENV_VAR] = canary.SERVE_CANARY
    os.environ[canary.PRINCIPALS_ENV_VAR] = CANARY_PRINCIPAL

    records: List[Dict[str, Any]] = []
    values = cells = 0

    def record(name: str, why: str, problems: List[str], **extra) -> None:
        records.append({"control": name, "why": why, "problems": problems,
                        "verdict": "PASS" if not problems else "FAIL", **extra})

    try:
        store = make_store(index, run_frames)
        book = history[-1][1]

        # 1 — a slice 1 request is untouched by any of this.
        payload, captured = serve_case(questions["SLICE1"], store=store,
                                       semantics=semantics, book=book)
        problems: List[str] = []
        serving = captured.get("serving") or {}
        if serving.get("decision") != canary.SERVED_NEW:
            problems.append(f"slice 1 served from {serving.get('decision')!r}")
        if (captured.get("eligibility") or {}).get("perimeter") == "slice2_temporal":
            problems.append("a current-period plan reached the temporal runtime")
        expected = float(truth.row_count(
            book, [("youngest_borrower_age", "gt", 55),
                   ("current_loan_to_value", "gt", 50)]))
        got = (captured.get("execution") or {}).get("value")
        if got is None or abs(float(got) - expected) > 0.01:
            problems.append(f"slice 1 figure {got} != {expected}")
        record("CURRENT_SLICE1_CONTROL", "a slice 1 request is unchanged",
               problems, value=got, expected=expected)

        # 2-4 — temporal series, filtered, grouped.
        for name, case, why, recipe, dims, filters in (
                ("TEMPORAL_SERIES_CONTROL", "T01",
                 "funded balance series", {"how": "sum", "column": truth.BALANCE},
                 [], []),
                ("FILTERED_TEMPORAL_CONTROL", "T06",
                 "filtered temporal series",
                 {"how": "count",
                  "predicates": [("erm_product_type", "eq", "Drawdown")]},
                 [], ["erm_product_type"]),
                ("GROUPED_TEMPORAL_CONTROL", "T07",
                 "temporal grouped result",
                 {"how": "count", "group_by": ["ltv_bucket"]},
                 ["ltv_bucket"], [])):
            payload, captured = serve_case(questions[case], store=store,
                                           semantics=semantics, book=book)
            problems = []
            block = ((captured.get("execution") or {}).get("temporal") or {})
            selected = [p["reporting_date"] for p in (block.get("points") or ())]
            want = CATALOGUE[-6:] if case == "T01" else list(CATALOGUE)
            if payload is None:
                problems.append(f"not served: "
                                f"{(captured.get('serving') or {}).get('reason')}")
            if selected != want:
                problems.append(f"snapshots {selected} != {want}")
            governed = ((payload or {}).get("metadata") or {}).get("governedPlan") or {}
            for entry in ((governed.get("executed") or {}).get("snapshots") or ()):
                applied = {p.get("field")
                           for p in (entry.get("applied_predicates") or ())}
                for wanted in filters:
                    if wanted not in applied:
                        problems.append(
                            f"{entry['reporting_date']}: filter {wanted} absent")
                if sorted(entry.get("group_field_keys") or ()) != sorted(dims):
                    problems.append(
                        f"{entry['reporting_date']}: grouped on "
                        f"{entry.get('group_field_keys')}")
            bound = (captured.get("execution") or {}).get("bound_spec") or {}
            column = ("loan_count" if bound.get("aggregation") == "count"
                      else f"{bound.get('metric')}_{bound.get('aggregation')}")
            rows = payload_rows(payload or {})
            for date in want:
                produced_rows = [r for r in rows
                                 if str(r.get(temporal.REPORTING_DATE)) == date]
                exp = oracle(recipe, frames[date])
                if dims:
                    produced = {tuple(str(r[a]) for a in dims): r.get(column)
                                for r in produced_rows}
                    if set(produced) != set(exp):
                        problems.append(f"{date}: group keys differ")
                    for key, figure in exp.items():
                        cells += 1
                        if produced.get(key) is None \
                                or abs(float(produced[key]) - figure) > 0.01:
                            problems.append(
                                f"{date} {key}: {produced.get(key)} != {figure}")
                else:
                    values += 1
                    if len(produced_rows) != 1:
                        problems.append(f"{date}: {len(produced_rows)} rows")
                    elif abs(float(produced_rows[0][column]) - exp) > 0.01:
                        problems.append(
                            f"{date}: {produced_rows[0][column]} != {exp}")
            record(name, why, problems, snapshots=len(selected), rows=len(rows))

        # 5 — an unavailable historical period fails closed.
        payload, captured = serve_case(questions["T13"], store=store,
                                       semantics=semantics, book=book)
        problems = []
        reason = (captured.get("serving") or {}).get("reason") or ""
        block = ((captured.get("execution") or {}).get("temporal") or {})
        if payload is not None:
            problems.append("an unavailable period was served")
        if not reason.startswith("TEMPORAL_NOT_RESOLVED"):
            problems.append(f"reason {reason!r}")
        if block.get("points"):
            problems.append("a shortened series was produced")
        record("UNAVAILABLE_PERIOD_CONTROL",
               "24 periods asked of an 8-period catalogue", problems,
               reason=reason)

        # 6 — another client's runs are unreachable, structurally.
        problems = []
        mine = store.list_snapshots(THIS_CLIENT, route=store_mod.FUNDED_ROUTE)
        # Asked of the BOUND store, this must come back empty; the unbound view
        # is built separately only to learn what would have been reachable.
        theirs = store.list_snapshots(OTHER_CLIENT, route=store_mod.FUNDED_ROUTE)
        if theirs:
            problems.append("the bound store listed another client's runs")
        unbound = store_mod.GovernedFundedSnapshotStore(
            index_provider=lambda: dict(index),
            frame_loader=lambda *a: (None, None), output_root=None)
        theirs = unbound.list_snapshots(OTHER_CLIENT,
                                        route=store_mod.FUNDED_ROUTE)
        if {h.reporting_date for h in mine} & {h.reporting_date for h in theirs}:
            problems.append("the two clients share a reporting date")
        if any(h.client_id != THIS_CLIENT for h in mine):
            problems.append("a foreign client id leaked into this scope")
        other_id = theirs[0].snapshot_id if theirs else "OTHER_TENANT/run_20250331"
        try:
            store.get_snapshot(other_id)                 # must not resolve here
            problems.append("another client's snapshot resolved")
        except SnapshotNotFoundError:
            pass
        # And the pipeline route is not this catalogue at all.
        if store.list_snapshots(THIS_CLIENT, route="pipeline"):
            problems.append("a pipeline route returned funded runs")
        # An id the catalogue does not list is not loadable — the only approval
        # semantics the estate keeps today.
        try:
            store.load_loans(f"{THIS_CLIENT}/run_19990101")
            problems.append("an unlisted run was loaded")
        except SnapshotNotFoundError:
            pass
        record("CROSS_PORTFOLIO_CONTROL",
               "another client / route / unlisted run is unreachable", problems,
               this_client=len(mine), other_client=len(theirs))

        # 7 — no catalogue: slice 1 works, slice 2 fabricates nothing.
        empty = store_mod.GovernedFundedSnapshotStore(
            index_provider=lambda: {"portfolios": [], "source": "unavailable"},
            frame_loader=lambda *a: (None, None), output_root=None,
            client_id=THIS_CLIENT)
        problems = []
        payload, captured = serve_case(questions["SLICE1"], store=empty,
                                       semantics=semantics, book=book)
        if payload is None:
            problems.append("slice 1 stopped working without a catalogue")
        payload, captured = serve_case(questions["T01"], store=empty,
                                       semantics=semantics, book=book)
        block = ((captured.get("execution") or {}).get("temporal") or {})
        if payload is not None:
            problems.append("a temporal question was served with no catalogue")
        if block.get("points"):
            problems.append("history was fabricated from an empty catalogue")
        record("NO_CATALOGUE_CONTROL",
               "slice 1 survives; slice 2 fabricates nothing", problems,
               reason=(captured.get("serving") or {}).get("reason"))

        # 8 — and with no store at all, which is production today.
        problems = []
        payload, captured = serve_case(questions["T01"], store=None,
                                       semantics=semantics, book=book)
        reason = (captured.get("serving") or {}).get("reason") or ""
        if payload is not None:
            problems.append("a temporal question was served with no store")
        if reason != canary.TEMPORAL_STORE_UNAVAILABLE:
            problems.append(f"reason {reason!r}")
        record("NO_STORE_CONTROL", "the production state today", problems,
               reason=reason)
    finally:
        wiring.set_interpreter_factory(None)
        os.environ.pop(canary.SERVE_ENV_VAR, None)
        os.environ.pop(canary.PRINCIPALS_ENV_VAR, None)

    failures = [r for r in records if r["verdict"] != "PASS"]
    report = {
        "catalogue_contract": "datasets.snapshot_index "
                              "({portfolios:[{client_id,runs:[{run_id,reporting_date}]}]})",
        "catalogue_is": "production-SHAPED over the independent oracle history",
        "clients_in_catalogue": [THIS_CLIENT, OTHER_CLIENT],
        "model_calls": 0,
        "series_values_reconciled": values,
        "grouped_cells_reconciled": cells,
        "verdicts": dict(Counter(r["verdict"] for r in records)),
        "results": records,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== SLICE 2 PRODUCTION SNAPSHOT-STORE BINDING")
    print(f"  catalogue              {report['catalogue_contract']}")
    print(f"  clients                {THIS_CLIENT} + {OTHER_CLIENT}")
    for r in records:
        print(f"    {r['verdict']} {r['control']:<30} {r['why']}")
        for problem in r["problems"]:
            print(f"          {problem}")
    print(f"  SERIES_VALUES          {values}")
    print(f"  GROUPED_CELLS          {cells}")
    print(f"  MODEL_CALLS            0")
    print(f"  WRITTEN                {OUT.relative_to(_REPO_ROOT)}")
    print(f"  RESULT                 {'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
