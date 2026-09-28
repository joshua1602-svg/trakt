#!/usr/bin/env python3
"""The live change_form interpretation gate. Eighteen interpretations, one each.

    NATURAL LANGUAGE  ->  OPUS  ->  CandidateIntent

WHAT IT MAY NOT DO. It changes no product code, calls no MI API, needs no
bearer, touches no App Service and executes nothing against a book. The boundary
ends at the structured intent, and at a compile of that intent for the record.
A call that fails is RECORDED as a failure; it is not retried, and the question
is not asked a second way to obtain a friendlier sample.

THE MANIFEST IS PINNED. Its sha256 is verified against the hash committed before
any call was made. This file never writes either — `preregister.py` is the only
thing that does, and it is not invoked from here.

WHAT IS SCORED. `change_form` is the primary gate. Then, as separate counts
because they fail for different reasons: whether a broad "what changed" invented
a specific measure it was not given, whether an explicitly named measure
survived, whether a Direct/Acquired scope survived, whether anything was
clarified that did not need clarifying, and whether a filter or dimension was
invented. `capability` and `operation` are RECORDED but not gated: Sprint A made
the compiler authoritative over execution ownership on purpose.

A MEASURE IS SCORED AFTER VOCABULARY RESOLUTION. `balance`, `outstanding_balance`
and `current_outstanding` are the same governed concept; counting them as misses
would measure typing, not reading.

    --dry-run   replays authored stand-in payloads through the SAME adjudication
                and the SAME compile check, so the harness is proved before a
                penny is spent. It makes no model call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

HERE = Path(__file__).resolve().parent
_REPO = HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

MANIFEST = HERE / "change_form_v2_bank_manifest.json"
HASH = HERE / "change_form_v2_bank_manifest.sha256"
RESULT = HERE / "change_form_v2_result.json"

#: Time forms that do NOT preserve a two-state comparison. Recorded as a
#: temporal failure rather than passed over.
_SINGLE_STATE = frozenset({"current", "series", "range", "forward_looking"})


# --------------------------------------------------------------------------- #
# the pinned bank
# --------------------------------------------------------------------------- #

def load_manifest() -> Mapping[str, Any]:
    raw = MANIFEST.read_text(encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    expected = HASH.read_text(encoding="utf-8").split()[0].strip()
    if digest != expected:
        raise SystemExit(
            f"BANK HASH MISMATCH\n  manifest {digest}\n  pinned   {expected}\n"
            f"The bank changed after it was pinned. Refusing to run.")
    return json.loads(raw)


# --------------------------------------------------------------------------- #
# reading a payload
# --------------------------------------------------------------------------- #

def _canonical(concept: Optional[str]) -> Optional[str]:
    """A model-authored concept name, resolved to its governed concept_id."""
    if not concept:
        return None
    try:
        from mi_agent.interpretation_v2.vocabulary import load_governed_vocabulary
        found = load_governed_vocabulary().resolve(str(concept))
        return found.concept_id if found else str(concept)
    except Exception:                                            # noqa: BLE001
        return str(concept)


def _measures(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for m in (payload.get("measures") or ()):
        if isinstance(m, Mapping):
            out.append({"concept": m.get("concept"),
                        "canonical": _canonical(m.get("concept")),
                        "statistic": m.get("statistic"),
                        "weight": m.get("weight")})
        else:
            out.append({"concept": str(m), "canonical": _canonical(str(m)),
                        "statistic": None, "weight": None})
    return out


def _read(payload: Mapping[str, Any]) -> Dict[str, Any]:
    pop = payload.get("population") or {}
    time = payload.get("time") or {}
    comparison = payload.get("comparison") or {}
    ambiguity = payload.get("ambiguity") or ()
    return {
        "change_form": payload.get("change_form"),
        "capability": payload.get("capability"),
        "operation": payload.get("operation"),
        "measures": _measures(payload),
        "base": pop.get("base") if isinstance(pop, Mapping) else None,
        "lens": pop.get("lens") if isinstance(pop, Mapping) else None,
        "seasoning": pop.get("seasoning") if isinstance(pop, Mapping) else None,
        "source_reference": (pop.get("source_reference")
                             if isinstance(pop, Mapping) else None),
        "time_form": time.get("form") if isinstance(time, Mapping) else None,
        "time_grain": time.get("grain") if isinstance(time, Mapping) else None,
        "periods_back": (time.get("periods_back")
                         if isinstance(time, Mapping) else None),
        "comparison_kind": (comparison.get("kind")
                            if isinstance(comparison, Mapping) else None),
        "dimensions": list(payload.get("dimensions") or ()),
        "filters": [
            {"concept": f.get("concept"), "comparator": f.get("comparator"),
             "value": f.get("value")} if isinstance(f, Mapping) else {"raw": f}
            for f in (payload.get("filters") or ())],
        "blocking_ambiguities": [
            {"slot": a.get("slot"), "note": a.get("note")}
            for a in ambiguity
            if isinstance(a, Mapping) and a.get("blocking")],
        "disclosed_ambiguities": [
            {"slot": a.get("slot"), "note": a.get("note")}
            for a in ambiguity
            if isinstance(a, Mapping) and not a.get("blocking")],
    }


# --------------------------------------------------------------------------- #
# adjudication
# --------------------------------------------------------------------------- #

def adjudicate(case: Mapping[str, Any], payload: Optional[Mapping[str, Any]],
               error: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": case["id"], "question": case["question"],
        "expected_change_form": case["change_form"],
        "expected_measure": (case["measure"] or {}).get("family")
                            if case.get("measure") else None,
        "expected_scope": case.get("scope"),
        "interpreted": payload is not None, "error": error,
        "problems": [], "failure_classes": [],
    }
    if payload is None:
        row.update({"verdict": "NO_INTERPRETATION"})
        row["failure_classes"].append("F")
        row["problems"].append(f"no interpretation: {error}"[:200])
        return row

    read = _read(payload)
    row.update(read)

    # -- A. the primary gate: change_form ---------------------------------- #
    row["change_form_ok"] = read["change_form"] == case["change_form"]
    if not row["change_form_ok"]:
        row["failure_classes"].append("A")
        row["problems"].append(
            f"change_form expected {case['change_form']!r}, "
            f"got {read['change_form']!r}")

    # -- B. the measure ---------------------------------------------------- #
    canon = [m["canonical"] for m in read["measures"] if m["canonical"]]
    expect = case.get("measure")
    if expect is None:
        # material_summary: a specific measure must NOT be invented.
        row["measure_invented"] = bool(canon)
        row["measure_ok"] = not canon
        if canon:
            row["failure_classes"].append("B")
            row["problems"].append(
                f"a broad change question was given specific measure(s) "
                f"{canon}: the composition owns its candidate set")
    else:
        row["measure_invented"] = False
        hit = [c for c in canon if c in expect["concepts"]]
        row["measure_ok"] = bool(hit)
        row["measure_matched"] = hit
        if not hit:
            row["failure_classes"].append("B")
            row["problems"].append(
                f"measure {expect['family']!r} expected one of "
                f"{expect['concepts']}, got {canon or 'NOTHING'}")
        # Recorded, reported, NOT gated.
        stats = [m["statistic"] for m in read["measures"] if m["statistic"]]
        row["statistic_as_expected"] = (
            None if not expect["statistics"]
            else any(s in expect["statistics"] for s in stats))

    # -- C. the temporal relationship -------------------------------------- #
    form = read["time_form"]
    row["temporal_ok"] = form in case["temporal_forms"]
    row["period_result"] = (
        "PAIR_PRESERVED" if row["temporal_ok"]
        else ("SINGLE_STATE" if form in _SINGLE_STATE else f"UNEXPECTED:{form}"))
    # RECORDED, NOT GATED. The deterministic owners do not honour a `current`
    # anchor for material_summary or attribution; that is an independent
    # temporal-contract defect, reported separately and not repaired in this
    # sprint. Gating it here would make a temporal failure read as a change_form
    # failure, which is the conflation this bank exists to avoid.
    if not row["temporal_ok"]:
        row["temporal_note"] = (
            f"anchored at {form!r} rather than one of {case['temporal_forms']}; "
            f"recorded, not counted against this gate")

    # -- D. Direct / Acquired scope ---------------------------------------- #
    want = case.get("scope")
    if want:
        row["scope_ok"] = read["lens"] == want
        if not row["scope_ok"]:
            row["failure_classes"].append("D")
            row["problems"].append(
                f"scope {want!r} stated in the question, lens is "
                f"{read['lens']!r}")
    else:
        # A scope the question did NOT state must not appear.
        row["scope_ok"] = read["lens"] in (None, "all")
        if not row["scope_ok"]:
            row["failure_classes"].append("D")
            row["problems"].append(
                f"lens {read['lens']!r} was not stated in the question")

    # -- E. inventions and unnecessary clarification ----------------------- #
    row["invented_filters"] = read["filters"]
    row["invented_dimensions"] = read["dimensions"]
    if read["filters"]:
        row["failure_classes"].append("E")
        row["problems"].append(f"invented filter(s): {read['filters']}")
    if read["dimensions"]:
        row["failure_classes"].append("E")
        row["problems"].append(
            f"invented dimension(s): {read['dimensions']}")
    row["unnecessary_clarification"] = bool(read["blocking_ambiguities"])
    if read["blocking_ambiguities"]:
        row["failure_classes"].append("E")
        row["problems"].append(
            f"blocking ambiguity on an answerable question: "
            f"{read['blocking_ambiguities']}")

    # -- silent semantic drop ---------------------------------------------- #
    # Something the question STATES is absent, and nothing was flagged. Silent
    # is the operative word: a drop disclosed by a blocking ambiguity is a
    # clarification, counted above, not a silent drop.
    silent = []
    if expect is not None and not row["measure_ok"] \
            and not read["blocking_ambiguities"]:
        silent.append("measure")
    if want and read["lens"] != want and not read["blocking_ambiguities"]:
        silent.append("scope")
    row["silent_semantic_drops"] = silent

    row["verdict"] = "PASS" if not row["problems"] else "FAIL"
    return row


# --------------------------------------------------------------------------- #
# the deterministic compile, for the record only
# --------------------------------------------------------------------------- #

def compile_check(case: Mapping[str, Any],
                  payload: Mapping[str, Any]) -> Dict[str, Any]:
    """What the governed layer DOES with the reading. Recorded, not gated.

    This gate measures interpretation. The compile is carried because a reading
    that cannot compile is worth knowing about, and because the form's resolved
    (capability, operation, mode) is the evidence that `change_form` reached the
    deterministic layer as the authority Sprint A made it.
    """
    from mi_agent.interpretation_v2.compiler import (CompilerContext,
                                                     DeterministicCompiler)
    from mi_agent.interpretation_v2.intent import parse_candidate_intent

    row: Dict[str, Any] = {"id": case["id"]}
    try:
        intent = parse_candidate_intent(dict(payload))
    except Exception as exc:                                     # noqa: BLE001
        row.update({"verdict": "UNPARSEABLE",
                    "problem": f"{type(exc).__name__}: {exc}"[:200]})
        return row
    result = DeterministicCompiler(CompilerContext()).compile(intent)
    plan = getattr(result, "plan", None)
    row["outcome"] = str(result.outcome)
    row["codes"] = [r.code for r in (result.reasons or ())]
    row["compiled"] = plan is not None
    row["verdict"] = "COMPILED" if plan is not None else "NOT_COMPILED"
    if plan is not None:
        row["plan_capability"] = plan.capability
        row["plan_operation"] = plan.operation
        row["plan_base"] = plan.population.base
        row["plan_scope"] = sorted((p.canonical_field, p.value)
                                   for p in plan.population.scope_predicates)
    return row


# --------------------------------------------------------------------------- #
# stand-ins: the dry run. Authored readings, NOT model output.
# --------------------------------------------------------------------------- #

def _standin(change_form: str, *, measures=(), lens="all",
             form="relative_pair", operation="movement",
             capability="period_movement") -> Dict[str, Any]:
    return {
        "schema_version": "candidate_intent/1.0", "capability": capability,
        "operation": operation, "change_form": change_form,
        "population": {"base": "funded", "lens": lens, "seasoning": "any"},
        "measures": [{"concept": c} for c in measures],
        "dimensions": [], "filters": [],
        "time": {"form": form, "periods_back": 1},
        "comparison": {"kind": "none"}, "ambiguity": [], "evidence": [],
    }


STAND_INS = {
    "V01": _standin("material_summary", operation="summary"),
    "V03": _standin("material_summary", operation="summary", lens="acquired"),
    "V06": _standin("metric_delta", measures=["loan"]),
    "V07": _standin("metric_delta", measures=["arrears_balance"]),
    "V09": _standin("metric_delta", measures=["indexed_loan_to_value"]),
    "V11": _standin("attribution", measures=["balance"], operation="bridge",
                    capability="funded_bridge"),
    "V18": _standin("level_comparison", measures=["outstanding_balance"],
                    lens="direct", operation="compare",
                    capability="generic_analysis"),
}


# --------------------------------------------------------------------------- #

def _totals(rows: List[Mapping[str, Any]],
            cases: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in rows if r.get("interpreted")]
    by_form: Dict[str, List[int]] = {}
    for r in rows:
        want = r["expected_change_form"]
        got, tot = by_form.setdefault(want, [0, 0])
        by_form[want] = [got + (1 if r.get("change_form_ok") else 0), tot + 1]

    measure_cases = [r for r in rows if cases[r["id"]].get("measure")]
    broad_cases = [r for r in rows if not cases[r["id"]].get("measure")]
    scope_cases = [r for r in rows if cases[r["id"]].get("scope")]
    return {
        "attempted": len(rows), "interpreted": len(scored),
        "change_form_total": f"{sum(1 for r in rows if r.get('change_form_ok'))} / {len(rows)}",
        "by_change_form": {k: f"{v[0]} / {v[1]}" for k, v in sorted(by_form.items())},
        "explicit_measure_preserved":
            f"{sum(1 for r in measure_cases if r.get('measure_ok'))} / {len(measure_cases)}",
        "material_summary_specific_measure_invented":
            f"{sum(1 for r in broad_cases if r.get('measure_invented'))} / {len(broad_cases)}",
        "scope_preserved":
            f"{sum(1 for r in scope_cases if r.get('scope_ok'))} / {len(scope_cases)}",
        "temporal_pair_preserved":
            f"{sum(1 for r in rows if r.get('temporal_ok'))} / {len(rows)}",
        "unnecessary_clarifications":
            sum(1 for r in rows if r.get("unnecessary_clarification")),
        "invented_filters": sum(len(r.get("invented_filters") or ()) for r in rows),
        "invented_dimensions":
            sum(len(r.get("invented_dimensions") or ()) for r in rows),
        "silent_semantic_drops":
            sum(len(r.get("silent_semantic_drops") or ()) for r in rows),
        "passed": sum(1 for r in rows if r.get("verdict") == "PASS"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="replay authored stand-ins; no model call")
    parser.add_argument("--json-out", default=str(RESULT))
    parser.add_argument("--only", nargs="*", help="limit to these case ids")
    args = parser.parse_args(argv)

    manifest = load_manifest()
    cases = {c["id"]: c for c in manifest["cases"]}
    selected = [c for c in manifest["cases"]
                if not args.only or c["id"] in set(args.only)]
    budget = int(manifest["authorised_live_calls"])

    interpreter = None
    served_default = ""
    if not args.dry_run:
        from mi_agent.interpretation_v2.opus_interpreter import (
            AnthropicInterpreterClient, OpusInterpreter)
        from mi_agent.interpretation_v2.vocabulary import load_governed_vocabulary
        client = AnthropicInterpreterClient(model=manifest["model"])
        if not client.available:
            print("PROVIDER_FAILURE: no ANTHROPIC_API_KEY in the environment",
                  file=sys.stderr)
            print("LIVE_MODEL_CALLS_SUCCESSFUL = 0", file=sys.stderr)
            return 2
        interpreter = OpusInterpreter(client,
                                      vocabulary=load_governed_vocabulary())

    report: Dict[str, Any] = {
        "bank_id": manifest["bank_id"], "model": manifest["model"],
        "baseline_sha": manifest["baseline_sha"],
        "bank_sha256": HASH.read_text(encoding="utf-8").split()[0].strip(),
        "mode": "dry_run" if args.dry_run else "live",
        "live_calls": 0, "provider_failures": 0,
        "cases": [], "compile_checks": [],
    }

    for case in selected:
        payload, error, served = None, "", served_default
        if args.dry_run:
            stand_in = STAND_INS.get(case["id"])
            if stand_in is None:
                continue
            payload = dict(stand_in)
        else:
            if report["live_calls"] >= budget:
                error = "the authorised call budget is spent"
            else:
                report["live_calls"] += 1
                try:
                    outcome = interpreter.interpret(case["question"])
                    served = getattr(outcome, "model_id", "") or ""
                    payload = (dict(outcome.raw_payload)
                               if getattr(outcome, "raw_payload", None) else None)
                    if payload is None:
                        error = f"no payload: {getattr(outcome, 'reason', '')}"
                except Exception as exc:                         # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"[:300]

        # A provider-side refusal stops the run. A favourable sample obtained by
        # continuing past a quota wall is not a measurement.
        if error and any(w in error.lower() for w in
                         ("credit", "quota", "rate_limit", "rate limit",
                          "authentication", "unauthor", "permission",
                          "billing", "insufficient", "401", "403", "429")):
            report["provider_failures"] += 1
            row = adjudicate(case, None, error)
            row["served_model"] = served
            report["cases"].append(row)
            print(f"  PROVIDER_FAILURE {case['id']}: {error}")
            print(f"\nPROVIDER_FAILURE after "
                  f"{report['live_calls'] - 1} successful call(s)")
            break

        row = adjudicate(case, payload, error)
        row["served_model"] = served
        # v1 scored every slot but kept no verbatim payload, so its byte-exact
        # model output was lost. Carried here: the reading is evidence, and a
        # reading you cannot re-read is weaker evidence.
        row["raw_payload"] = payload
        report["cases"].append(row)
        print(f"  {row['verdict']:16s} {case['id']}  "
              f"form={row.get('change_form')!r} "
              f"(want {case['change_form']!r})  "
              f"lens={row.get('lens')!r} time={row.get('time_form')!r} "
              f"meas={[m['canonical'] for m in row.get('measures') or []]}")
        for problem in row["problems"]:
            print(f"        {problem}")

        if payload is not None:
            checked = compile_check(case, payload)
            report["compile_checks"].append(checked)
            print(f"        compile {checked['verdict']}"
                  f" {checked.get('plan_capability','')}"
                  f"/{checked.get('plan_operation','')}"
                  f" {checked.get('codes') or ''}")

    report["totals"] = _totals(report["cases"], cases)
    print("\nTOTALS " + json.dumps(report["totals"], indent=1))
    Path(args.json_out).write_text(json.dumps(report, indent=1, default=str),
                                   encoding="utf-8")
    print(f"written {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
