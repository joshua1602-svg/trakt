#!/usr/bin/env python3
"""migration_phase0/completeness_calibration.py — does the check separate?

READ-ONLY, deterministic, no model call. Runs
`question_interpretation.completeness` over four standing banks and reports
whether it fires where a concept is demonstrably lost and stays silent where
nothing is.

    python -m migration_phase0.completeness_calibration [--json out.json]

The pre-registered classes come from the 75-question acceptance run at
`MI_FINAL_LIVE_DATA_READINESS.json` and the failure diagnosis over it:

    type (a)  concept no governed field carries          0
    type (b)  concept binds to the WRONG governed field  2   Q21B Q21C
    type (c)  phrasing nothing recognises                21
    type (d)  guard refuses something already computed   2   Q01B Q20B

The check exists to catch type (c). It is not expected to fire on type (d) —
nothing was lost there — and that silence is asserted, not tolerated.

EXITS NON-ZERO if the measured separation moves from the pre-registered
figures. A calibration instrument that cannot fail is not calibration.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class CalibrationError(RuntimeError):
    """The calibration could not be measured. Never absorbed into a pass."""


#: The type-(c) failures on the 75-question bank. Twenty-one at the diagnosis;
#: twenty now, because Q19C was one of them and has been FIXED — see REGRADED.
TYPE_C: Tuple[str, ...] = (
    "Q01C", "Q02B", "Q03A", "Q03C", "Q04A", "Q05B", "Q05C", "Q07B", "Q10A",
    "Q10C", "Q12C", "Q15B", "Q15C", "Q16B", "Q17B", "Q17C", "Q22B",
    "Q22C", "Q23B", "Q24B")
TYPE_B: Tuple[str, ...] = ("Q21B", "Q21C")
TYPE_D: Tuple[str, ...] = ("Q01B", "Q20B")

#: Grades that MOVED since `MI_FINAL_LIVE_DATA_READINESS.json` was frozen,
#: with the reason. The frozen file is not retro-edited: it is the record of
#: what was measured then, and a record edited to agree with a later run is not
#: evidence. The movement is declared here instead, where it is visible.
REGRADED: Dict[str, Tuple[str, str]] = {
    "Q19C": ("EXACT",
             "was WRONG / SILENT at £22.6m — the whole book's movement "
             "reported for the Direct book, which moved £12.4m. The route "
             "derived its lens from the contract, `lens_from_contract` "
             "returned `{source_portfolio_type: direct}`, and "
             "`_apply_lens_filter` reads `source_portfolio_id` and returned "
             "the frame unchanged. Now answers £12.4m (£105.0m → £117.4m over "
             "441 of 640 loans), which matches the pre-registered independent "
             "truth of 12,366,371.4 exactly."),
}

#: PRE-REGISTERED. Each is a question the check gets wrong, named, with the
#: reason. A new name appearing here is a regression; a name leaving it is an
#: improvement that must be explained before the figure is moved.
#: NONE. The five that stood here were one cause and it was a live defect:
#: routes that narrowed to a book or read a dataset and published no record of
#: having done so. Q19B (period_movement), S29 (concentration_analysis) now
#: publish `metadata.scopeApplied`; CFO69/CFO70/S32 were already publishing
#: `reconciliation.dataset`, and this check was reading the DECISION
#: (`metadata.datasetContext`) instead of it. All five are silent, and the one
#: whose receipt could not be told from a wrong answer's is fixed.
EXPECTED_FALSE_POSITIVES: Dict[str, str] = {}

EXPECTED_MISSES: Dict[str, str] = {
    "Q15B": "THE CHECK'S RECALL IS THE OWNERS' RECALL. `portfolio_lens` does "
            "not read `Direct-book` outside a selector position — `For "
            "Direct-book` (Q05B) and `of Direct-book balance` (Q17B) carry a "
            "selector mark and `Break Direct-book balance` does not — and no "
            "other owner claims it either. A concept NO OWNER RESOLVES cannot "
            "be seen lost by any deterministic detector. This is the stated "
            "limit of the deterministic arm, not a tuning shortfall, and it is "
            "the argument for a proposal step: a reader that proposes concepts "
            "does not need the grammar to reach them.",
}

#: PRE-REGISTERED separation, measured at the head this file was written on.
EXPECTED = {
    "type_c_fires": 19, "type_c_total": 20,
    "exact_fires": 0, "exact_total": 31,
    "cfo_exact_fires": 0, "cfo_exact_total": 73,
    "deliver_fires": 0, "deliver_total": 53,
}

PORTFOLIO = "client_001/mi_2026_06"
AS_OF = "2026-06-30"


def _fixture_env() -> str:
    env = os.environ.get("MI_COMPLETENESS_FIXTURE", "/tmp/cfo_env")
    if not Path(env, "onboarding_output").is_dir():
        raise CalibrationError(
            "CALIBRATION INVALID - fixture root %r has no onboarding_output. "
            "Set MI_COMPLETENESS_FIXTURE to the acceptance fixture." % env)
    return env


def _client_and_book():
    env = _fixture_env()
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = "%s/onboarding_output" % env
    os.environ["TRAKT_PORTFOLIO_REGISTRY"] = "%s/portfolio_registry.yaml" % env
    os.environ.setdefault("MI_AGENT_PIPELINE_ROOT",
                          str(_REPO / "tests" / "fixtures" / "pipeline_history_5w"))
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"

    from fastapi.testclient import TestClient
    from mi_agent_api import mi_service as S
    from mi_agent_api import workspace as W
    from mi_agent_api.app import app
    from mi_agent_api.dependencies import build_dependencies
    from migration_phase0.assurance_semantics import load_assurance_semantics

    semantics = load_assurance_semantics()
    frame, err = S._resolve_frame(build_dependencies().datasets, W.DEFAULT_VIEW,
                                  PORTFOLIO)
    if frame is None:
        raise CalibrationError("CALIBRATION INVALID - frame did not load: %s" % err)
    values = S._book_values(frame, semantics)
    if not values:
        raise CalibrationError(
            "CALIBRATION INVALID - the book's value catalogue loaded EMPTY; "
            "every value reading would silently resolve to nothing")
    return TestClient(app), frame, semantics, values


def _questions() -> List[Tuple[str, str, str, str]]:
    """``[(bank, id, question, expectation)]`` over the four standing banks."""
    import yaml
    out: List[Tuple[str, str, str, str]] = []

    bank = yaml.safe_load((_REPO / "migration_phase0" /
                           "MI_FINAL_ACCEPTANCE_75.yaml").read_text())
    grades = {r["id"]: r["grade"] for r in json.loads(
        (_REPO / "migration_phase0" /
         "MI_FINAL_LIVE_DATA_READINESS.json").read_text())["rows"]}
    for case in bank["cases"]:
        for f in case["formulations"]:
            qid = f["id"]
            grade = REGRADED[qid][0] if qid in REGRADED else grades.get(qid, "?")
            out.append(("BANK75", qid, f["q"], grade))

    cfo = json.loads((_REPO / "migration_phase0" /
                      "MI_FINAL_LIVE_DATA_READINESS.json").read_text())["cfo_91"]
    for i, r in enumerate(cfo["results"]):
        out.append(("CFO91", "CFO%02d" % (i + 1), r["question"], r.get("grade", "?")))

    for path, tag in (("SIMPLE_COMPOSITION_BANK.yaml", "SIMPLE"),
                      ("CFO_GENERALISATION_SUPPLEMENT.yaml", "GEN")):
        doc = yaml.safe_load((_REPO / "migration_phase0" / path).read_text())
        for i, e in enumerate(doc["questions"]):
            out.append((tag, e.get("id") or "%s%02d" % (tag, i + 1), e["q"],
                        e.get("expect", "?")))
    return out


def run() -> Dict[str, Any]:
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    from question_interpretation import completeness as C

    client, frame, semantics, values = _client_and_book()
    columns = set(frame.columns)
    rows: List[Dict[str, Any]] = []

    for bank, qid, question, expectation in _questions():
        env = client.post("/mi/query", json={"question": question,
                                             "portfolioId": PORTFOLIO,
                                             "asOfDate": AS_OF}).json()
        stated = C.stated_concepts(question, semantics, available_values=values,
                                   available_columns=columns, frame=frame)
        lost = C.unresolved_concepts(stated, C.from_envelope(env))
        rows.append({"bank": bank, "id": qid, "question": question,
                     "expectation": expectation, "ok": env.get("ok"),
                     "route": (env.get("metadata") or {}).get("route"),
                     "stated": [c.as_dict() for c in stated],
                     "lost": [c.as_dict() for c in lost],
                     "fires": bool(lost)})

    by_id = {r["id"]: r for r in rows}
    b75 = [r for r in rows if r["bank"] == "BANK75"]
    exact = [r for r in b75 if r["expectation"] == "EXACT"]
    cfo_exact = [r for r in rows if r["bank"] == "CFO91"
                 and r["expectation"] == "EXACT"]
    deliver = [r for r in rows if r["bank"] in ("SIMPLE", "GEN")
               and r["expectation"] == "DELIVER"]

    measured = {
        "type_c_fires": sum(by_id[i]["fires"] for i in TYPE_C),
        "type_c_total": len(TYPE_C),
        "exact_fires": sum(r["fires"] for r in exact),
        "exact_total": len(exact),
        "cfo_exact_fires": sum(r["fires"] for r in cfo_exact),
        "cfo_exact_total": len(cfo_exact),
        "deliver_fires": sum(r["fires"] for r in deliver),
        "deliver_total": len(deliver),
    }
    false_positives = sorted(
        r["id"] for r in (exact + cfo_exact + deliver) if r["fires"])
    misses = sorted(i for i in TYPE_C if not by_id[i]["fires"])

    return {
        "regraded_since_the_diagnosis": {k: {"grade": v[0], "why": v[1]}
                                         for k, v in REGRADED.items()},
        "what_this_check_does_not_see": {
            "mis_binding": "IT COMPARES PRESENCE, NOT CORRECTNESS. Q21C bound "
                           "the function word `among` as a categorical value "
                           "and every concept the sentence states is still in "
                           "the contract, so this check is silent — by design. "
                           "A concept carried into the WRONG governed field is "
                           "invisible here and always will be; disagreement "
                           "reporting between a proposal and the registry is "
                           "what covers that case, not this.",
            "concepts_no_owner_resolves": EXPECTED_MISSES["Q15B"],
        },
        "measured": measured,
        "pre_registered": EXPECTED,
        "matches_pre_registration": measured == EXPECTED,
        "false_positives": false_positives,
        "expected_false_positives": sorted(EXPECTED_FALSE_POSITIVES),
        "misses": misses,
        "expected_misses": sorted(EXPECTED_MISSES),
        "type_b": {i: {"fires": by_id[i]["fires"], "lost": by_id[i]["lost"]}
                   for i in TYPE_B},
        "type_d": {i: {"fires": by_id[i]["fires"], "lost": by_id[i]["lost"]}
                   for i in TYPE_D},
        "rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="out")
    args = ap.parse_args(argv)

    result = run()
    m, e = result["measured"], result["pre_registered"]
    print("completeness calibration")
    print("  type-(c) lost concepts   FIRES %d/%d" % (m["type_c_fires"], m["type_c_total"]))
    print("  75-bank EXACT            fires %d/%d" % (m["exact_fires"], m["exact_total"]))
    print("  CFO-91 EXACT             fires %d/%d" % (m["cfo_exact_fires"], m["cfo_exact_total"]))
    print("  composition DELIVER      fires %d/%d" % (m["deliver_fires"], m["deliver_total"]))
    print("  false positives: %s" % (", ".join(result["false_positives"]) or "none"))
    print("  misses:          %s" % (", ".join(result["misses"]) or "none"))
    for label, key in (("type-(b)", "type_b"), ("type-(d)", "type_d")):
        for qid, v in sorted(result[key].items()):
            print("  %-9s %-6s %s" % (label, qid, "FIRES" if v["fires"] else "silent"))

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print("  wrote %s" % args.out)

    ok = True
    if m != e:
        ok = False
        print("\nCALIBRATION MOVED — pre-registered %s, measured %s" % (e, m))
    if result["false_positives"] != result["expected_false_positives"]:
        ok = False
        print("\nFALSE POSITIVE SET MOVED — expected %s, measured %s"
              % (result["expected_false_positives"], result["false_positives"]))
    if result["misses"] != result["expected_misses"]:
        ok = False
        print("\nMISS SET MOVED — expected %s, measured %s"
              % (result["expected_misses"], result["misses"]))
    for qid, v in result["type_d"].items():
        if v["fires"]:
            ok = False
            print("\nTYPE-(d) FIRED — %s lost nothing and the check says it did: %s"
                  % (qid, v["lost"]))
    print("\n%s" % ("CALIBRATION HOLDS" if ok else "CALIBRATION FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
