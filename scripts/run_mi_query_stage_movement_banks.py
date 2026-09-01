#!/usr/bin/env python3
"""scripts/run_mi_query_stage_movement_banks.py — the three banks, separately.

Runs every question through the REAL shipped path — ``POST /mi/query`` on the
production FastAPI app — and reports THREE independent scores, never one
flattering aggregate:

    A. the authoritative 166-question MI Query bank   NON-REGRESSION
    B. the stage-movement bank                        THE NEW CAPABILITY
    C. the near-neighbour bank                        SEMANTIC ISOLATION

Each answers a different question, so combining them would hide the only two
outcomes that can fail this work: a question the agent used to answer correctly
and now does not, and a question another capability owns that stage movement
took.

    python scripts/run_mi_query_stage_movement_banks.py --out out/banks
    python scripts/run_mi_query_stage_movement_banks.py --out out/after \\
        --compare out/before

WHY THE FIXTURE IS BUILT HERE. The 640-loan book the shipping 166-question
record was measured on lived in an onboarding output that is not in this
repository, so an absolute re-score of that bank is not reproducible from the
tree. What IS reproducible — and is the acceptance criterion that matters — is
PER-QUESTION CONTINUITY: the same deterministic funded tape and the same
governed pipeline fixture, before and after, with every answer, route and
verdict compared question by question. ``--compare`` reports any question whose
answer, route or verdict moved at all.

Deterministic and offline: the funded tape is seeded, the pipeline extracts are
the committed fixture, and the free-form LLM parser arm is off.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

BANKS = _REPO / "tests" / "fixtures" / "mi_query_stage_movement"
STAGE_BANK = BANKS / "STAGE_MOVEMENT_BANK.yaml"
NEIGHBOUR_BANK = BANKS / "NEAR_NEIGHBOUR_BANK.yaml"
#: The governed two-snapshot pipeline pack the stage-transition capability is
#: already asserted against. Every stage-movement expectation is arithmetic on it.
PIPELINE_FIXTURE = _REPO / "tests" / "fixtures" / "pipeline_transition_2w"

PORTFOLIO = "client_001/mi_2026_06"
AS_OF = "2026-06-30"
STAGE_ROUTE = "pipeline_stage_movement"

#: The five month-end snapshots the shipping record was measured across, at the
#: same row counts, so the funded book has the depth its temporal routes need.
MONTHS: Tuple[Tuple[str, str, int], ...] = (
    ("mi_2026_02", "2026-02-28", 520), ("mi_2026_03", "2026-03-31", 545),
    ("mi_2026_04", "2026-04-30", 570), ("mi_2026_05", "2026-05-31", 600),
    ("mi_2026_06", "2026-06-30", 640))


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #
def write_funded_tape(root: Path, run_id: str, reporting_date: str, n: int) -> None:
    """One governed month-end funded tape, seeded on its own run id."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(sum(ord(c) for c in run_id))
    out = root / "client_001" / run_id / "output" / "central"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "loan_identifier": [f"L{i:05d}" for i in range(n)],
        "current_outstanding_balance": rng.uniform(60_000, 480_000, n).round(2),
        "original_principal_balance": rng.uniform(50_000, 400_000, n).round(2),
        "current_valuation_amount": rng.uniform(180_000, 1_200_000, n).round(2),
        "current_loan_to_value": rng.uniform(15, 75, n).round(1),
        "current_interest_rate": rng.uniform(3, 9, n).round(2),
        "youngest_borrower_age": rng.integers(60, 92, n),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma", "Delta"], n),
        "geographic_region_obligor": rng.choice(
            ["North", "South East", "Midlands", "Wales", "Scotland"], n),
        "erm_product_type": rng.choice(["Lump Sum", "Drawdown"], n),
        "origination_channel": rng.choice(["Direct", "Intermediary"], n),
        "interest_rate_type": rng.choice(["Fixed", "Floating"], n),
        "occupancy_type": rng.choice(["Owner Occupied", "Second Home"], n),
        "borrower_type": rng.choice(["Single", "Joint"], n),
        "account_status": rng.choice(["Performing", "Watch"], n),
        "source_portfolio": rng.choice(["direct_001", "acquired_001"], n,
                                       p=[0.7, 0.3]),
        "origination_date": rng.choice(
            pd.date_range("2021-01-01", "2025-06-01", freq="MS").astype(str), n),
        "reporting_date": [reporting_date] * n,
    }).to_csv(out / "18_central_lender_tape.csv", index=False)


def build_client():
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    root = Path(tempfile.mkdtemp(prefix="mi_banks_")) / "onboarding_output"
    for run_id, date, rows in MONTHS:
        write_funded_tape(root, run_id, date, rows)
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(root)
    os.environ["MI_AGENT_PIPELINE_ROOT"] = str(PIPELINE_FIXTURE)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    os.environ.setdefault("MI_AGENT_LLM_PARSER", "off")

    from mi_agent_api import datasets as _datasets

    cfg = _datasets._mi_llm_config()
    if cfg.enabled or cfg.available:
        raise SystemExit("RUN INVALID - the free-form LLM parser arm is live; "
                         "set MI_AGENT_LLM_PARSER=off")
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    return TestClient(app)


# --------------------------------------------------------------------------- #
# Banks
# --------------------------------------------------------------------------- #
def authoritative_166() -> List[Tuple[str, str, str]]:
    """The shipped acceptance bank: 75 formulations + the frozen CFO 91."""
    import yaml

    out: List[Tuple[str, str, str]] = []
    bank = yaml.safe_load(
        (_REPO / "migration_phase0" / "MI_FINAL_ACCEPTANCE_75.yaml").read_text())
    for case in bank["cases"]:
        for f in case["formulations"]:
            out.append(("BANK75", f["id"], f["q"]))
    ready = json.loads(
        (_REPO / "migration_phase0" / "MI_FINAL_LIVE_DATA_READINESS.json").read_text())
    for i, row in enumerate(ready["cfo_91"]["results"]):
        out.append(("CFO91", "CFO%02d" % (i + 1), row["question"]))
    return out


def yaml_bank(path: Path, tag: str) -> List[Tuple[str, str, str]]:
    import yaml

    doc = yaml.safe_load(path.read_text())
    return [(tag, f["id"], f["q"])
            for case in doc["cases"] for f in case["formulations"]]


def stage_bank_specs() -> Dict[str, Dict[str, Any]]:
    import yaml

    doc = yaml.safe_load(STAGE_BANK.read_text())
    return {f["id"]: case for case in doc["cases"] for f in case["formulations"]}


# --------------------------------------------------------------------------- #
# Execution and grading
# --------------------------------------------------------------------------- #
def run_bank(client, questions) -> List[Dict[str, Any]]:
    rows = []
    for bank, qid, question in questions:
        env = client.post("/mi/query", json={
            "question": question, "portfolioId": PORTFOLIO,
            "asOfDate": AS_OF}).json()
        meta = env.get("metadata") or {}
        rows.append({
            "bank": bank, "id": qid, "question": question,
            "ok": bool(env.get("ok")), "route": meta.get("route"),
            "answer": (env.get("answer") or "").strip(),
            "error": (env.get("error") or "").strip(),
            "artifact_rows": max(
                [len(a.get("rows") or []) for a in (env.get("artifacts") or [])]
                or [0]),
        })
        print(".", end="", flush=True)
    print()
    return rows


def grade_166(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """CFO91 against its FROZEN expectations; BANK75 by delivery.

    The CFO bank's `expect` / `must` / `must_not` are assertions about the
    ANSWER's shape and vocabulary and transfer to any governed book. The 75
    bank's oracle is a set of figures computed from a tape that is not in this
    repository, so those rows are recorded as DELIVERED / DECLINED and their
    non-regression is established by per-question comparison rather than by a
    verdict this rig cannot honestly compute.
    """
    import yaml

    from migration_phase0 import pack_grader

    specs = {q["q"]: q for q in yaml.safe_load(
        (_REPO / "migration_phase0" / "CFO_ACCEPTANCE_BANK.yaml").read_text()
    )["questions"]}
    out = []
    for r in rows:
        spec = specs.get(r["question"])
        if r["bank"] == "CFO91" and spec:
            verdict = pack_grader.grade_cfo(
                {"ok": r["ok"], "answer": r["answer"], "error": r["error"],
                 "artefacts": [{"rows": r["artifact_rows"]}]}, spec)
            out.append({**r, "grade": verdict["grade"], "why": verdict["why"]})
        else:
            out.append({**r, "grade": "DELIVERED" if r["ok"] else "DECLINED",
                        "why": "oracle is tape-specific; compared per question"})
    return out


def grade_stage(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """CORRECT / WRONG / HONEST_DECLINE against the bank's frozen assertions.

    A delivered answer that did NOT come from the governed stage-transition
    route is WRONG whatever it says: something else answered a transition
    question, which is the substitution this bank exists to detect.
    """
    specs = stage_bank_specs()
    out = []
    for r in rows:
        case = specs[r["id"]]
        low = (r["answer"] or "").lower()
        if not r["ok"]:
            out.append({**r, "grade": "HONEST_DECLINE",
                        "why": (r["error"] or "")[:200]})
        elif r["route"] != STAGE_ROUTE:
            out.append({**r, "grade": "WRONG",
                        "why": "answered by %r, not the governed stage-transition "
                               "capability" % r["route"]})
        elif [m for m in (case.get("must") or []) if str(m).lower() not in low]:
            out.append({**r, "grade": "WRONG", "why": "a required figure is absent"})
        elif [m for m in (case.get("must_not") or []) if str(m).lower() in low]:
            out.append({**r, "grade": "WRONG", "why": "a forbidden figure is present"})
        else:
            out.append({**r, "grade": "CORRECT", "why": ""})
    return out


def compare(before: List[Dict[str, Any]], after: List[Dict[str, Any]]
            ) -> List[Dict[str, Any]]:
    """Every question whose answer, route or verdict moved at all."""
    prior = {r["id"]: r for r in before}
    moved = []
    for r in after:
        was = prior.get(r["id"])
        if was is None:
            continue
        if (was.get("grade") != r.get("grade") or was.get("route") != r.get("route")
                or was.get("answer") != r.get("answer")
                or was.get("ok") != r.get("ok")):
            moved.append({"id": r["id"], "question": r["question"],
                          "grade": [was.get("grade"), r.get("grade")],
                          "route": [was.get("route"), r.get("route")],
                          "ok": [was.get("ok"), r.get("ok")],
                          "answer_before": (was.get("answer") or "")[:200],
                          "answer_after": (r.get("answer") or "")[:200]})
    return moved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True,
                    help="directory for the three graded captures")
    ap.add_argument("--compare", type=Path,
                    help="an earlier --out directory to diff against")
    ap.add_argument("--only", choices=("166", "stage", "neighbours"),
                    help="run one bank instead of all three")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    client = build_client()
    failures = 0

    if args.only in (None, "166"):
        print("A. authoritative 166-question MI Query bank")
        graded = grade_166(run_bank(client, authoritative_166()))
        (args.out / "bank_166.json").write_text(json.dumps(graded, indent=1))
        cfo = Counter(g["grade"] for g in graded if g["bank"] == "CFO91")
        b75 = Counter(g["grade"] for g in graded if g["bank"] == "BANK75")
        print("   CFO91 :", dict(cfo))
        print("   BANK75:", dict(b75))

    if args.only in (None, "stage"):
        print("B. stage-movement bank")
        graded = grade_stage(run_bank(client, yaml_bank(STAGE_BANK, "STAGE")))
        (args.out / "bank_stage.json").write_text(json.dumps(graded, indent=1))
        counts = Counter(g["grade"] for g in graded)
        print("   ", dict(counts),
              " correct: %.1f%%" % (100.0 * counts["CORRECT"] / len(graded)))
        if counts["WRONG"]:
            failures += counts["WRONG"]
            for g in graded:
                if g["grade"] == "WRONG":
                    print("    WRONG", g["id"], g["question"], "|", g["why"])

    if args.only in (None, "neighbours"):
        print("C. near-neighbour bank")
        rows = run_bank(client, yaml_bank(NEIGHBOUR_BANK, "NEIGHBOUR"))
        for r in rows:
            r["grade"] = "STAGE_MOVEMENT" if r["route"] == STAGE_ROUTE else "OWN_ROUTE"
        (args.out / "bank_neighbours.json").write_text(json.dumps(rows, indent=1))
        stolen = [r for r in rows if r["grade"] == "STAGE_MOVEMENT"]
        print("    %d of %d kept their own owner" % (len(rows) - len(stolen), len(rows)))
        for r in stolen:
            failures += 1
            print("    HIJACKED", r["id"], r["question"])

    if args.compare:
        print("\nPER-QUESTION CONTINUITY vs %s" % args.compare)
        for name in ("bank_166", "bank_stage", "bank_neighbours"):
            old, new = args.compare / f"{name}.json", args.out / f"{name}.json"
            if not (old.exists() and new.exists()):
                continue
            moved = compare(json.loads(old.read_text()), json.loads(new.read_text()))
            print("   %-16s %d question(s) moved" % (name, len(moved)))
            for m in moved:
                print("      %s | %s" % (m["id"], m["question"][:70]))
                print("         grade %s -> %s · route %s -> %s"
                      % (*m["grade"], *m["route"]))
            (args.out / f"{name}_moved.json").write_text(json.dumps(moved, indent=1))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
