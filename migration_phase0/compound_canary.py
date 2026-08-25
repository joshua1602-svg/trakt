#!/usr/bin/env python3
"""migration_phase0/compound_canary.py — the adversarial compound-question canary.

READ-ONLY. Executes every case in ``question_interpretation/compound_canary_bank.yaml``
through the LIVE ``/mi/query`` path against two governed funded snapshots, and
grades each DECLARED element as

    HONOURED   the element is visible in machine-readable evidence
    DISCLOSED  not honoured, and the answer says so
    DROPPED    not honoured, and nothing says so          <- the defect class

The bank pins INVARIANTS, not answers. This instrument reports invariant
breaches, and separately reports MOVEMENT against the freeze observations, so a
conversion that fixes a known defect is distinguishable from one that breaks a
canary. See the bank's header for the distinction.

    python -m migration_phase0.compound_canary [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

BANK = _REPO / "question_interpretation" / "compound_canary_bank.yaml"

#: Two governed month-end funded runs carrying the dimensions the bank names.
#: Deliberately the SAME shape route_ownership_evolution builds, so a canary and
#: an owned-surface census disagree about behaviour and never about the book.
FUNDED_RUNS = (("mi_2026_04", "2026-04-30", 60, 1.0),
               ("mi_2026_05", "2026-05-31", 70, 1.15))

PORTFOLIO = "client_001/mi_2026_05"
AS_OF = "2026-05-31"

HONOURED, DISCLOSED, UNEVIDENCED, DROPPED = (
    "HONOURED", "DISCLOSED", "UNEVIDENCED", "DROPPED")
#: Short codes for the table. Never one letter — DISCLOSED and DROPPED share it,
#: and an earlier run of this instrument printed both as "D", which made a
#: correct grading unreadable and nearly hid three findings.
CODE = {HONOURED: "ok", DISCLOSED: "disc", UNEVIDENCED: "un", DROPPED: "DROP"}

#: Elements whose ONLY machine-readable channel is `metadata.rankedMovement`,
#: which exactly one route publishes. Off that route they grade UNEVIDENCED
#: rather than DROPPED — see the bank's note on why that distinction exists.
_RANKED_CHANNEL = ("RANK", "DIMENSION", "DIRECTION", "BASIS", "TOP_N",
                   "MEASURE", "SPAN")

#: The route's own controlled non-substitution wording. Matched literally: this
#: is the estate's disclosure vocabulary, not a sentiment heuristic.
_DISCLOSURE_PHRASES = (
    "i have not", "i haven't", "i could not", "no substitute was used",
    "not comparable", "did not increase", "did not decrease",
)


def _write_run(root: Path, run_id: str, reporting_date: str, n: int,
               scale: float) -> None:
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(sum(ord(c) for c in run_id))
    d = root / "client_001" / run_id / "output" / "central"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "loan_identifier": [f"{run_id}_{i}" for i in range(n)],
        "current_outstanding_balance": (rng.uniform(120_000, 280_000, n) * scale).round(2),
        "current_loan_to_value": rng.uniform(20, 55, n).round(1),
        "current_interest_rate": rng.uniform(3, 8, n).round(2),
        "youngest_borrower_age": rng.integers(62, 88, n),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma", "Delta"], n),
        "geographic_region_obligor": rng.choice(["London", "South East", "Scotland"], n),
        "reporting_date": [reporting_date] * n,
    }).to_csv(d / "18_central_lender_tape.csv", index=False)


def load_bank() -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(BANK.read_text(encoding="utf-8"))


def cases(bank: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for family in bank["families"]:
        for case in family["cases"]:
            out.append({**case, "family": family["id"],
                        "family_name": family["name"],
                        "period_order_checked": bool(
                            family.get("period_order_checked"))})
    return out


# --------------------------------------------------------------------------- #
# Element grading
#
# The order below is the whole design. A withheld answer substitutes nothing, so
# it is graded first and never counts as a drop. A route-independent
# contradiction is graded next, because it is the only signal that survives
# leaving the one route with an evidence channel. Absence of a channel is graded
# LAST, and as UNEVIDENCED rather than DROPPED.
# --------------------------------------------------------------------------- #
def _says_it_did_not(resp: Dict[str, Any]) -> bool:
    """True when the answer uses the estate's controlled non-substitution wording.

    An unrelated warning is NOT disclosure. An earlier version of this function
    treated any warning as disclosing any element, which graded a delivered
    narrative and a flat refusal identically and suppressed a true I4 breach.
    """
    text = str(resp.get("answer") or "").lower()
    return any(p in text for p in _DISCLOSURE_PHRASES)


def _delivers_rows(resp: Dict[str, Any]) -> bool:
    return any(a.get("rows") for a in (resp.get("artifacts") or []))


def _movement_grade(resp: Dict[str, Any], rm: Dict[str, Any]) -> str:
    """Whether a CHANGE was answered, readable on any route.

    The one element that does not depend on `rankedMovement`, and therefore the
    one that catches a question about movement being answered with a level by a
    different route entirely.
    """
    if rm.get("applied") and rm.get("openingPeriod") and rm.get("closingPeriod"):
        return HONOURED
    text = str(resp.get("answer") or "").lower()
    if _delivers_rows(resp):
        if "as at" in text and "between" not in text:
            return DROPPED           # a level, delivered, where a change was asked
        if "between" in text or " to " in text or "moved from" in text:
            return HONOURED
    return DISCLOSED if _says_it_did_not(resp) else DROPPED


def _grade_element(element: str, resp: Dict[str, Any]) -> str:
    meta = resp.get("metadata") or {}
    rm = meta.get("rankedMovement") or {}
    applied = bool(rm.get("applied"))

    if element == "MOVEMENT":
        return _movement_grade(resp, rm)

    # A refusal withholds the answer rather than substituting for it.
    if not resp.get("ok"):
        return DISCLOSED

    if element == "SCOPE":
        if meta.get("lensApplied"):
            return HONOURED
        return DISCLOSED if _says_it_did_not(resp) else DROPPED

    if element not in _RANKED_CHANNEL:
        raise CanaryMeasurementError(f"no grading rule for element {element!r}")

    field = {"RANK": "applied", "DIMENSION": "canonicalField",
             "DIRECTION": "direction", "BASIS": "basis", "TOP_N": "topN",
             "MEASURE": "canonicalField", "SPAN": "openingPeriod"}[element]
    if applied and rm.get(field):
        return HONOURED
    if _says_it_did_not(resp):
        return DISCLOSED
    if not rm:
        # No channel at all on this route: unverifiable, not proven absent.
        return UNEVIDENCED
    # The channel exists and does not carry the element. For TOP_N on an applied
    # ranking this is the strict and correct reading: the count was read and
    # discarded.
    return DROPPED


_PERIOD_RE = re.compile(r"\b(\d{4}-\d{2})\b")


def _period_order_breached(resp: Dict[str, Any]) -> Optional[str]:
    """I7: the pair a change is reported over must run earlier -> later."""
    found = _PERIOD_RE.findall(str(resp.get("answer") or ""))
    if len(found) < 2:
        return None
    if found[0] > found[1]:
        return f"reported {found[0]} -> {found[1]}, which runs backwards"
    return None


def run() -> Dict[str, Any]:
    import logging
    import warnings
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    tmp = Path(tempfile.mkdtemp())
    out_root = tmp / "onboarding_output"
    for run_id, rdate, n, scale in FUNDED_RUNS:
        _write_run(out_root, run_id, rdate, n, scale)

    # MUTATED AND RESTORED — the discipline route_ownership_evolution learned the
    # hard way: an assurance instrument that leaves the governed roots repointed
    # corrupts every test that runs after it in the same process.
    saved = {k: os.environ.get(k) for k in
             ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_AUTH_ENABLED")}
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(out_root)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    try:
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app

        client = TestClient(app)
        bank = load_bank()
        rows: List[Dict[str, Any]] = []
        for case in cases(bank):
            try:
                resp = client.post("/mi/query", json={
                    "question": case["question"], "portfolioId": PORTFOLIO,
                    "asOfDate": AS_OF}).json()
            except Exception as exc:  # noqa: BLE001 - re-raised, never absorbed
                raise CanaryMeasurementError(
                    f"canary {case['id']} could not be executed: {exc!r}") from exc
            meta = resp.get("metadata") or {}
            rm = meta.get("rankedMovement") or {}
            grades = {e: _grade_element(e, resp) for e in case["declares"]}
            rows.append({
                "id": case["id"], "family": case["family"],
                "family_name": case["family_name"], "question": case["question"],
                "declares": list(case["declares"]), "grades": grades,
                "route": meta.get("route"), "ok": bool(resp.get("ok")),
                "delivers_rows": _delivers_rows(resp),
                "ranked_applied": bool(rm.get("applied")),
                "ranked_field": rm.get("canonicalField"),
                "ranked_basis": rm.get("basis"),
                "ranked_direction": rm.get("direction"),
                "ranked_top_n": rm.get("topN"),
                "period_order_breach": (_period_order_breached(resp)
                                        if case["period_order_checked"] else None),
                "answer": str(resp.get("answer") or "")[:400],
            })
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    expected = len(cases(load_bank()))
    if len(rows) != expected or not rows:
        raise CanaryMeasurementError(
            f"CANARY INVALID - {len(rows)} reading(s) for {expected} case(s)")
    return {"rows": rows, "bank": load_bank()}


# --------------------------------------------------------------------------- #
# Invariant evaluation
# --------------------------------------------------------------------------- #
def evaluate(result: Dict[str, Any]) -> Dict[str, Any]:
    rows = {r["id"]: r for r in result["rows"]}
    bank = result["bank"]
    breaches: List[Dict[str, Any]] = []

    # I1 / I2 — a declared element graded DROPPED. UNEVIDENCED is counted
    # separately below: it is an evidence gap, not a proven substitution.
    for row in result["rows"]:
        if row.get("period_order_breach"):
            breaches.append({
                "invariant": "I7_period_order_is_chronological",
                "case": row["id"], "element": "SPAN",
                "route": str(row["route"]),
                "detail": row["period_order_breach"]})
        for element, grade in row["grades"].items():
            if grade == DROPPED:
                breaches.append({
                    "invariant": "I1_honour_or_disclose", "case": row["id"],
                    "element": element, "route": row["route"],
                    "detail": f"{element} declared, not honoured, not disclosed"})

    # I4 — paraphrases must reach the same honouring outcome.
    for family in bank["families"]:
        for group in family.get("paraphrase_sets") or []:
            outcomes = {cid: tuple(sorted(rows[cid]["grades"].items()))
                        for cid in group if cid in rows}
            if len(set(outcomes.values())) > 1:
                breaches.append({
                    "invariant": "I4_phrasing_is_not_meaning",
                    "case": " / ".join(group), "element": "-",
                    "route": " / ".join(str(rows[c]["route"]) for c in group
                                        if c in rows),
                    "detail": "same declared elements, different honouring: "
                              + "; ".join(f"{c}={dict(o)}" for c, o in outcomes.items())})

    # I3 — an element honoured at the base must not be un-honoured by adding one.
    for lat in bank.get("lattices") or []:
        base = rows.get(lat["base"])
        if base is None:
            continue
        honoured = {e for e, g in base["grades"].items() if g == HONOURED}
        for step in lat["steps"]:
            row = rows.get(step["case"])
            if row is None:
                continue
            for element in honoured:
                grade = row["grades"].get(element)
                if grade == DROPPED:
                    breaches.append({
                        "invariant": "I3_composition_preserves_parts",
                        "case": f"{lat['base']} -> {step['case']}",
                        "element": element, "route": str(row["route"]),
                        "detail": f"{element} honoured at the base, dropped after "
                                  f"adding {step['adds']}"})

    # I6 — a family answered entirely by refusals proves nothing.
    unexercised = []
    for family in bank["families"]:
        ids = [c["id"] for c in family["cases"]]
        if not any(rows[i]["delivers_rows"] or rows[i]["ranked_applied"]
                   for i in ids if i in rows):
            unexercised.append(family["id"])

    unevidenced = [
        {"case": r["id"], "route": str(r["route"]),
         "elements": sorted(e for e, g in r["grades"].items() if g == UNEVIDENCED)}
        for r in result["rows"]
        if any(g == UNEVIDENCED for g in r["grades"].values())]

    return {"breaches": breaches, "unexercised_families": unexercised,
            "unevidenced": unevidenced}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    result = run()
    verdict = evaluate(result)
    rows = result["rows"]

    print("=" * 88)
    print(f"ADVERSARIAL COMPOUND-QUESTION CANARY — {len(rows)} cases, "
          f"{len(result['bank']['families'])} families")
    print("=" * 88)
    print(f"\n{'case':<7}{'route':<26}{'declared -> graded'}")
    print("-" * 88)
    for row in rows:
        graded = " ".join(f"{e}:{CODE[g]}" for e, g in row["grades"].items())
        print(f"{row['id']:<7}{str(row['route']):<26}{graded}")

    print(f"\n{'':-<88}")
    print(f"INVARIANT BREACHES: {len(verdict['breaches'])}")
    for b in verdict["breaches"]:
        print(f"  [{b['invariant']:<34}] {b['case']:<22} {b['element']:<10} "
              f"{b['detail'][:90]}")
    if verdict["unevidenced"]:
        n = sum(len(u["elements"]) for u in verdict["unevidenced"])
        print(f"\nUNEVIDENCED: {n} declared element(s) across "
              f"{len(verdict['unevidenced'])} case(s) — the answering route "
              f"publishes no channel for them, so honouring is UNVERIFIABLE "
              f"either way. Not counted as a breach; counted as an evidence gap.")
        for u in verdict["unevidenced"]:
            print(f"  {u['case']:<7} route={u['route']:<24} {u['elements']}")

    if verdict["unexercised_families"]:
        print(f"\nUNEXERCISED FAMILIES (I6, evidence not counted): "
              f"{verdict['unexercised_families']}")
    else:
        print("\nEvery family exercised at least one delivering case (I6 satisfied).")

    if args.json:
        args.json.write_text(json.dumps({**result, **verdict}, indent=2,
                                        default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
