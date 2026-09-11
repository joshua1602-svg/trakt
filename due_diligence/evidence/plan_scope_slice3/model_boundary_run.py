#!/usr/bin/env python3
"""Slice 3 model boundary: can Opus state governed portfolio scope?

Twelve fresh interpretations, one per case, no retry. The manifest is verified
against the sha256 committed before any call was made; this harness never writes
to it.

WHAT IT MAY NOT DO. It changes no product code, calls no MI API, needs no
bearer, and executes nothing against a book. The boundary ends at the compiled
plan's scope predicates. A call that fails is RECORDED as a failure; it is not
retried and the question is not asked a second way.

THE MODEL IS SCORED ON WHAT THE GOVERNED LAYER COULD DO WITH ITS OUTPUT. Not on
whether it typed what the manifest typed: `operation`, `time.form` and the
statistic are deliberately unpinned. What is scored is whether the scope is the
right KIND, whether a named portfolio arrives as the reader's phrase rather than
an id the model invented, whether the rest of the question survived alongside it,
and whether the two attribution questions kept an attribution capability.

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

MANIFEST = HERE / "model_bank_manifest.json"
HASH = HERE / "model_bank_manifest.sha256"
RESULT = HERE / "model_boundary_result.json"

TOTAL, ROLE, NAMED, ATTRIBUTION = "total", "role", "named", "attribution"
ROLE_FIELD = "source_portfolio_type"
SOURCE_FIELD = "source_portfolio_id"

#: Capabilities that carry attribution/movement meaning. A question asking how
#: much of a change came from somewhere must land on one of these; landing on
#: `generic_analysis` is the silent capability loss the controls are for.
ATTRIBUTION_CAPABILITIES = frozenset({"period_movement", "funded_bridge",
                                      "pipeline_stage_movement"})
ATTRIBUTION_OPERATIONS = frozenset({"movement", "bridge", "transition",
                                    "arrivals", "departures", "reconciliation"})


def verify_manifest() -> Dict[str, Any]:
    body = MANIFEST.read_text(encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    recorded = HASH.read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        raise SystemExit(f"the manifest changed since it was hashed: "
                         f"{digest} != {recorded}")
    return json.loads(body)


def registry_from(spec: Mapping[str, Any]) -> Any:
    """The governed registry, built by the product's own factory."""
    from trakt_core.portfolio import build_registry
    records = [{"source_portfolio_id": p["source_portfolio_id"],
                "source_portfolio_type": p["source_portfolio_type"]}
               for p in spec["portfolios"]]
    metadata = {p["source_portfolio_id"]: {
        "source_portfolio_label": p.get("source_portfolio_label"),
        "aliases": list(p.get("aliases") or ())} for p in spec["portfolios"]}
    return build_registry(records, metadata=metadata,
                          client_id=spec.get("client_id"))


# --------------------------------------------------------------------------- #
# reading one intent
# --------------------------------------------------------------------------- #

def population_of(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return dict((payload.get("population") or {}))


def stated_filters(payload: Mapping[str, Any]) -> List[str]:
    return [str(f.get("concept") or "") for f in (payload.get("filters") or ())]


def stated_dimensions(payload: Mapping[str, Any]) -> List[str]:
    return [str(d) for d in (payload.get("dimensions") or ())]


def authored_physical_binding(payload: Mapping[str, Any]) -> Optional[str]:
    """Did the MODEL author a physical source binding?

    Naming the identity COLUMN in a filter is the violation: that is the model
    deciding which rows to read rather than naming a book and leaving the
    binding to the registry. A `source_reference` is not a violation even when
    the reader happened to say the id — that is what the reader said.
    """
    for flt in (payload.get("filters") or ()):
        concept = str(flt.get("concept") or "").strip().lower()
        if concept in (SOURCE_FIELD, "source_portfolio", "portfolio_id"):
            return f"filters[].concept={concept}"
    for key in ("source_portfolio_id", "portfolio_id", "dataset", "run_id"):
        if payload.get(key) or population_of(payload).get(key):
            return f"payload carries {key!r}"
    return None


def temporal_stated(payload: Mapping[str, Any]) -> bool:
    time = payload.get("time") or {}
    form = str(time.get("form") or "").strip().lower()
    operation = str(payload.get("operation") or "").strip().lower()
    comparison = str((payload.get("comparison") or {}).get("kind")
                     or "").strip().lower()
    return (form not in ("", "current")
            or operation in ("series", "compare", "movement", "bridge")
            or comparison not in ("", "none"))


def adjudicate(case: Mapping[str, Any], payload: Optional[Mapping[str, Any]],
               error: str = "") -> Dict[str, Any]:
    """The ten measures, each recorded independently."""
    out: Dict[str, Any] = {"id": case["id"], "question": case["question"],
                           "why": case["why"], "expected_scope": case["scope"],
                           "problems": [], "error": error}
    if payload is None:
        out.update({"interpreted": False, "verdict": "FAIL"})
        out["problems"].append(f"no interpretation: {error}"[:200])
        return out

    population = population_of(payload)
    lens = str(population.get("lens") or "all").strip().lower()
    reference = population.get("source_reference")
    reference = str(reference).strip() if reference else None
    capability = str(payload.get("capability") or "").strip().lower()
    operation = str(payload.get("operation") or "").strip().lower()
    physical = authored_physical_binding(payload)

    out.update({
        "interpreted": True, "capability": capability, "operation": operation,
        "lens": lens, "source_reference": reference,
        "filters": stated_filters(payload),
        "dimensions": stated_dimensions(payload),
        "temporal_stated": temporal_stated(payload),
        "PHYSICAL_SOURCE_BINDING_AUTHORED": bool(physical),
    })
    problems = out["problems"]
    if physical:
        problems.append(f"authored a physical source binding: {physical}")

    expected = case["scope"]
    # -- SCOPE_INTENT_PRESENT / SCOPE_TYPE_CORRECT ------------------------- #
    if expected == TOTAL:
        out["SCOPE_INTENT_PRESENT"] = True          # the default IS the scope
        out["SCOPE_TYPE_CORRECT"] = lens in ("", "all") and not reference
        if lens not in ("", "all"):
            problems.append(f"total narrowed to lens={lens!r}")
        if reference:
            problems.append(f"total narrowed to source {reference!r}")
    elif expected == ROLE:
        out["SCOPE_INTENT_PRESENT"] = lens in ("direct", "acquired") or bool(reference)
        out["SCOPE_TYPE_CORRECT"] = lens == case["role"] and not reference
        if lens != case["role"]:
            problems.append(f"role {case['role']!r} read as lens={lens!r}")
        if reference:
            problems.append(f"a ROLE was read as the named portfolio "
                            f"{reference!r}")
    elif expected == NAMED:
        out["SCOPE_INTENT_PRESENT"] = bool(reference) or lens != "all"
        out["SCOPE_TYPE_CORRECT"] = bool(reference)
        if not reference:
            problems.append("the named portfolio was not carried in "
                            "source_reference"
                            + (f"; it became lens={lens!r}" if lens != "all"
                               else " and was dropped"))
    elif expected == ATTRIBUTION:
        out["SCOPE_INTENT_PRESENT"] = True
        out["SCOPE_TYPE_CORRECT"] = True            # scope is not the control
    out["ROLE_CORRECT"] = (lens == (case.get("role") or "all")
                           if expected in (TOTAL, ROLE) else None)
    out["SOURCE_REFERENCE_CORRECT"] = (bool(reference) if expected == NAMED
                                       else (not reference))

    # -- what must survive alongside the scope ------------------------------ #
    out["MEASURE_PRESERVED"] = bool(payload.get("measures"))
    if not out["MEASURE_PRESERVED"]:
        problems.append("no measure survived")
    wanted_filters = [f for f in (case.get("filters") or ())]
    got = {f.lower() for f in out["filters"]}
    missing = [f for f in wanted_filters if f.lower() not in got]
    out["FILTERS_PRESERVED"] = not missing
    if missing:
        problems.append(f"ordinary filter(s) dropped: {missing}")
    wanted_dims = [d for d in (case.get("dimensions") or ())]
    got_d = {d.lower() for d in out["dimensions"]}
    missing_d = [d for d in wanted_dims if d.lower() not in got_d]
    out["DIMENSIONS_PRESERVED"] = not missing_d
    if missing_d:
        problems.append(f"dimension(s) dropped: {missing_d}")
    out["TEMPORAL_INTENT_PRESERVED"] = (out["temporal_stated"]
                                        if case.get("temporal") else None)
    if case.get("temporal") and not out["temporal_stated"]:
        problems.append("the temporal constraint was dropped")

    # -- CAPABILITY_PRESERVED ---------------------------------------------- #
    forbidden = {c.lower() for c in (case.get("capability_not") or ())}
    if forbidden:
        kept = (capability in ATTRIBUTION_CAPABILITIES
                or operation in ATTRIBUTION_OPERATIONS)
        out["CAPABILITY_PRESERVED"] = kept
        if not kept:
            problems.append(
                f"attribution intent lost: capability={capability!r} "
                f"operation={operation!r} carries no movement/attribution "
                f"meaning")
    else:
        out["CAPABILITY_PRESERVED"] = True

    out["verdict"] = "PASS" if not problems else "FAIL"
    return out


# --------------------------------------------------------------------------- #
# the deterministic compile check (M01-M10)
# --------------------------------------------------------------------------- #

def compile_check(case: Mapping[str, Any], payload: Mapping[str, Any],
                  registry: Any) -> Dict[str, Any]:
    """Compile the RECORDED intent against the governed registry."""
    from mi_agent.interpretation_v2.compiler import (CompilerContext,
                                                     DeterministicCompiler)
    from mi_agent.interpretation_v2.intent import parse_candidate_intent

    out: Dict[str, Any] = {"id": case["id"], "problems": []}
    try:
        intent = parse_candidate_intent(payload)
    except Exception as exc:                                         # noqa: BLE001
        out.update({"compiled": False,
                    "detail": f"{type(exc).__name__}: {exc}"[:160]})
        out["problems"].append("the recorded intent does not parse")
        out["verdict"] = "FAIL"
        return out

    result = DeterministicCompiler(
        CompilerContext(source_registry=registry)).compile(intent)
    out["outcome"] = result.outcome
    out["codes"] = list(result.codes())
    if not result.is_plan:
        # A refusal or clarification is not a binding error; it is recorded and
        # judged against what the case expected.
        out.update({"compiled": False, "scope": []})
        out["verdict"] = "REFUSED"
        return out

    scope = sorted((p.canonical_field, p.value)
                   for p in result.plan.population.scope_predicates)
    out.update({"compiled": True, "scope": scope,
                "plan_id": result.plan.plan_id})
    expected = case["scope"]
    if expected == TOTAL and scope:
        out["problems"].append(f"total produced a scope predicate: {scope}")
    if expected == ROLE and scope != [(ROLE_FIELD, case["role"])]:
        out["problems"].append(f"expected {ROLE_FIELD}={case['role']!r}, "
                               f"got {scope}")
    if expected == NAMED:
        wanted = [(SOURCE_FIELD, case["resolves_to"])]
        if scope != wanted:
            out["problems"].append(f"expected {wanted}, got {scope}")
    out["verdict"] = "PASS" if not out["problems"] else "FAIL"
    return out


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

STAND_INS = {
    "S3-M01": {"population": {"base": "funded", "lens": "all"},
               "capability": "generic_analysis", "operation": "point_in_time",
               "measures": [{"concept": "current_outstanding_balance",
                             "statistic": "sum"}], "time": {"form": "current"}},
    "S3-M03": {"population": {"base": "funded", "lens": "direct"},
               "capability": "generic_analysis", "operation": "point_in_time",
               "measures": [{"concept": "current_outstanding_balance",
                             "statistic": "sum"}], "time": {"form": "current"}},
    "S3-M07": {"population": {"base": "funded", "lens": "all",
                              "source_reference": "ALP Acquired Back Book"},
               "capability": "generic_analysis", "operation": "point_in_time",
               "measures": [{"concept": "current_outstanding_balance",
                             "statistic": "sum"}], "time": {"form": "current"}},
    "S3-M11": {"population": {"base": "funded", "lens": "acquired"},
               "capability": "period_movement", "operation": "movement",
               "measures": [{"concept": "current_outstanding_balance",
                             "statistic": "sum"}],
               "time": {"form": "relative_pair"}},
}


def _base(payload: Mapping[str, Any]) -> Dict[str, Any]:
    body = {"schema_version": "candidate_intent/1.0", "dimensions": [],
            "filters": [], "geography": {"requested": False},
            "comparison": {"kind": "none"}}
    body.update(payload)
    body.setdefault("population", {}).setdefault("seasoning", "any")
    return body


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="replay stand-ins through the same adjudication; "
                             "makes no model call")
    parser.add_argument("--json-out", default=str(RESULT))
    args = parser.parse_args(argv)

    manifest = verify_manifest()
    registry = registry_from(manifest["registry"])
    cases = manifest["cases"]
    budget = int(manifest["authorised_live_calls"])

    interpreter = None
    if not args.dry_run:
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            print("::error::no ANTHROPIC_API_KEY in the environment")
            return 2
        from mi_agent.interpretation_v2.opus_interpreter import (
            AnthropicInterpreterClient, OpusInterpreter)
        interpreter = OpusInterpreter(
            AnthropicInterpreterClient(model=manifest["model"]))

    report: Dict[str, Any] = {
        "bank_id": manifest["bank_id"],
        "manifest_sha256": HASH.read_text(encoding="utf-8").split()[0],
        "model": manifest["model"], "dry_run": bool(args.dry_run),
        "authorised_live_calls": budget, "live_calls": 0,
        "cases": [], "compile_checks": [],
    }

    for case in cases:
        payload, error, served, retrievals = None, "", "", 0
        if args.dry_run:
            stand_in = STAND_INS.get(case["id"])
            if stand_in is None:
                continue
            payload = _base(stand_in)
        else:
            if report["live_calls"] >= budget:
                error = "the authorised call budget is spent"
            else:
                report["live_calls"] += 1
                try:
                    outcome = interpreter.interpret(case["question"])
                    # WHICH MODEL ACTUALLY ANSWERED. A silent fallback would
                    # make the acceptance claim false, and the only way to know
                    # is to read back what the API says it served.
                    served = getattr(outcome, "model_id", "") or ""
                    retrievals = len(getattr(outcome, "metadata_calls", ()) or ())
                    payload = (dict(outcome.raw_payload)
                               if getattr(outcome, "raw_payload", None) else None)
                    if payload is None:
                        error = f"no payload: {getattr(outcome, 'reason', '')}"
                except Exception as exc:                             # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"[:200]

        adjudicated = adjudicate(case, payload, error)
        adjudicated["raw_payload"] = payload
        adjudicated["served_model"] = served
        adjudicated["metadata_retrievals"] = retrievals
        report["cases"].append(adjudicated)
        print(f"  {adjudicated['verdict']} {case['id']}  "
              f"lens={adjudicated.get('lens')!r} "
              f"src={adjudicated.get('source_reference')!r} "
              f"cap={adjudicated.get('capability')!r}/"
              f"{adjudicated.get('operation')!r}")
        for problem in adjudicated["problems"]:
            print(f"        {problem}")

        if payload is not None and case["scope"] != ATTRIBUTION:
            checked = compile_check(case, payload, registry)
            report["compile_checks"].append(checked)
            print(f"        compile {checked['verdict']}: "
                  f"{checked.get('scope')}")
            for problem in checked["problems"]:
                print(f"          {problem}")

    scored = [c for c in report["cases"] if c.get("interpreted")]
    report["totals"] = {
        "cases": len(report["cases"]),
        "interpreted": len(scored),
        "passed": sum(1 for c in report["cases"] if c["verdict"] == "PASS"),
        "physical_bindings": sum(
            1 for c in scored if c.get("PHYSICAL_SOURCE_BINDING_AUTHORED")),
        "compile_pass": sum(1 for c in report["compile_checks"]
                            if c["verdict"] == "PASS"),
        "compile_checked": len(report["compile_checks"]),
    }
    report["served_models"] = sorted(
        {c.get("served_model") for c in report["cases"] if c.get("served_model")})
    if not args.dry_run and report["served_models"] != [manifest["model"]]:
        # Not fatal — the run happened and its findings stand — but the report
        # must never imply a model answered that did not.
        print(f"::warning::served model(s) {report['served_models']} are not "
              f"exactly [{manifest['model']!r}]")
    Path(args.json_out).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  TOTALS {json.dumps(report['totals'])}")
    print(f"  written {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
