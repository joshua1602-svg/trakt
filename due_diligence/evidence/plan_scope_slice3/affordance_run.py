#!/usr/bin/env python3
"""Slice 3 affordance: one fresh measurement of the repaired contract.

Ten interpretations, one call per question, no retries. The manifest is verified
against the sha256 committed before any call was made; this harness never writes
to it, and `affordance_preregister.py` — the only thing that does — is never
invoked from the runner or the workflow.

WHAT IT MAY NOT DO. It changes no product code, calls no MI API, needs no
bearer, and executes nothing against a book. The boundary ends at the compiled
plan's scope predicates.

HOW A FAILURE IS COUNTED. Three outcomes reach a reader as a wrong answer and
are counted separately from every other kind of miss:

    SILENT_SCOPE_DROP       scope the question stated is absent, and nothing
                            was flagged — the answer is over a wider book
    SILENT_SCOPE_WIDENING   scope was replaced by something broader (a named
                            book became its role, a role became the total)
    SILENT_SCOPE_NARROWING  an axis the question did not state was added — the
                            answer is over a narrower book

"Silent" is the operative word: an axis left empty WITH a blocking ambiguity is
recorded as CLARIFIED and counted apart. A safe clarification is not a silent
failure. Nor is it a pass: an axis the contract now describes should resolve.

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
from typing import Any, Dict, List, Mapping, Optional

HERE = Path(__file__).resolve().parent
_REPO = HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

MANIFEST = HERE / "affordance_bank_manifest.json"
HASH = HERE / "affordance_bank_manifest.sha256"
RESULT = HERE / "affordance_result.json"

LIFECYCLE, ROLE, NAMED, ATTRIBUTION = "lifecycle", "role", "named", "attribution"
ROLE_FIELD = "source_portfolio_type"
SOURCE_FIELD = "source_portfolio_id"
SEASONING_FIELD = "seasoning_segment"

ATTRIBUTION_CAPABILITIES = frozenset({"period_movement", "funded_bridge",
                                      "pipeline_stage_movement"})
ATTRIBUTION_OPERATIONS = frozenset({"movement", "bridge", "transition",
                                    "arrivals", "departures", "reconciliation"})

#: Anything that looks like a physical handle rather than a reader's phrase. A
#: `source_reference` matching one of these is the model authoring an id.
_PHYSICAL_MARKERS = ("_id", "run_", "blob", "://", ".csv", ".parquet", "/")


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

    # Labels travel on the provenance records; aliases are the GOVERNED overlay
    # and are keyed by id, exactly as `build_registry` documents. Built the same
    # way as the boundary run's registry so the two runs are comparable.
    records = [{"source_portfolio_id": p["source_portfolio_id"],
                "source_portfolio_type": p["source_portfolio_type"]}
               for p in spec["portfolios"]]
    metadata = {p["source_portfolio_id"]: {
        "source_portfolio_label": p.get("source_portfolio_label"),
        "aliases": list(p.get("aliases") or ())} for p in spec["portfolios"]}
    return build_registry(records, metadata=metadata,
                          client_id=spec.get("client_id"))


# --------------------------------------------------------------------------- #
# reading one payload
# --------------------------------------------------------------------------- #

def population_of(payload: Mapping[str, Any]) -> Dict[str, Any]:
    pop = payload.get("population")
    return dict(pop) if isinstance(pop, Mapping) else {}


def blocking_slots(payload: Mapping[str, Any]) -> List[str]:
    return [str(a.get("slot") or "") for a in (payload.get("ambiguity") or ())
            if isinstance(a, Mapping) and a.get("blocking")]


def stated_terms(payload: Mapping[str, Any], key: str) -> List[str]:
    out: List[str] = []
    for item in (payload.get(key) or ()):
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, Mapping) and item.get("concept"):
            out.append(str(item["concept"]))
    for output in (payload.get("outputs") or ()):
        if isinstance(output, Mapping):
            out.extend(stated_terms(output, key))
    return out


def temporal_stated(payload: Mapping[str, Any]) -> bool:
    time = payload.get("time")
    if not isinstance(time, Mapping):
        return False
    return bool(time.get("form") and time.get("form") != "current")


def authored_physical_reference(reference: Optional[str]) -> Optional[str]:
    if not reference:
        return None
    low = reference.strip().lower()
    if any(marker in low for marker in _PHYSICAL_MARKERS):
        return reference
    return None


# --------------------------------------------------------------------------- #
# adjudication
# --------------------------------------------------------------------------- #

def adjudicate(case: Mapping[str, Any], payload: Optional[Mapping[str, Any]],
               error: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": case["id"], "question": case["question"], "axis": case["axis"],
        "why": case["why"], "error": error, "interpreted": payload is not None,
        "problems": [], "clarified_slots": [],
        "SILENT_SCOPE_DROP": False, "SILENT_SCOPE_WIDENING": False,
        "SILENT_SCOPE_NARROWING": False, "TEMPORAL_DROP": False,
        "SEASONING_INFERRED": False, "PHYSICAL_SOURCE_BINDING_AUTHORED": False,
        "ATTRIBUTION_CAPABILITY_DROP": False, "AXIS_CORRECT": False,
        "NAMED_SOURCE_ATOMIC": None, "NAMED_SOURCE_REFERENCE_PRESERVED": None,
    }
    if payload is None:
        out["problems"].append(error or "no interpretation was returned")
        out["verdict"] = "NO_INTERPRETATION"
        return out

    pop = population_of(payload)
    base = str(pop.get("base") or "")
    lens = str(pop.get("lens") or "")
    seasoning = str(pop.get("seasoning") or "")
    reference = pop.get("source_reference") or None
    blocking = blocking_slots(payload)
    capability = str(payload.get("capability") or "")
    operation = str(payload.get("operation") or "")
    problems = out["problems"]

    out.update({"base": base or None, "lens": lens or None,
                "seasoning": seasoning or None, "source_reference": reference,
                "capability": capability, "operation": operation,
                "blocking": blocking})
    out["clarified_slots"] = blocking

    # -- the model must never author a handle ------------------------------- #
    physical = authored_physical_reference(reference)
    if physical:
        out["PHYSICAL_SOURCE_BINDING_AUTHORED"] = True
        problems.append(f"authored a physical source handle: {physical!r}")

    # -- seasoning must not be inferred from lifecycle or from a name ------- #
    if case.get("seasoning_forbidden") and seasoning not in ("", "any"):
        out["SEASONING_INFERRED"] = True
        out["SILENT_SCOPE_NARROWING"] = True
        problems.append(
            f"seasoning={seasoning!r} was inferred from wording that names a "
            f"population or a book; it narrows the answer")

    # -- LIFECYCLE ----------------------------------------------------------- #
    wanted_base = case.get("base")
    if wanted_base:
        # An empty base resolves to 'funded' by governed default, so an unstated
        # base is correct precisely when funded is what was wanted.
        settled = base or "funded"
        out["BASE_CORRECT"] = settled == wanted_base
        if settled != wanted_base:
            if "population.base" in blocking:
                problems.append(f"population left open (asked for {wanted_base!r})")
            else:
                problems.append(f"population {wanted_base!r} read as {settled!r}")
                if wanted_base == "pipeline":
                    out["SILENT_SCOPE_WIDENING"] = True
    else:
        out["BASE_CORRECT"] = None

    # -- ROLE ---------------------------------------------------------------- #
    wanted_role = case.get("role")
    settled_lens = lens or "all"
    if wanted_role:
        out["ROLE_CORRECT"] = settled_lens == wanted_role
        if settled_lens != wanted_role:
            if any(s.startswith("population.lens") for s in blocking):
                problems.append(f"role left open (asked for {wanted_role!r})")
            else:
                problems.append(f"role {wanted_role!r} read as lens={settled_lens!r}")
                out["SILENT_SCOPE_DROP" if settled_lens == "all"
                    else "SILENT_SCOPE_NARROWING"] = True
    else:
        out["ROLE_CORRECT"] = None
        if case["axis"] in (LIFECYCLE, NAMED) and settled_lens != "all":
            out["SILENT_SCOPE_NARROWING"] = True
            problems.append(f"a role was added that the question did not "
                            f"state: lens={settled_lens!r}")

    # -- NAMED IDENTITY ------------------------------------------------------ #
    if case["axis"] == NAMED:
        out["NAMED_SOURCE_REFERENCE_PRESERVED"] = bool(reference)
        if not reference:
            if any(s.startswith("population.source") or s == "filters"
                   for s in blocking):
                problems.append("the named book was left open rather than "
                                "carried in source_reference")
            else:
                out["SILENT_SCOPE_DROP" if settled_lens == "all"
                    else "SILENT_SCOPE_WIDENING"] = True
                problems.append(
                    "the named book was not carried in source_reference"
                    + (f"; it became lens={settled_lens!r}" if settled_lens != "all"
                       else " and was dropped"))
        # ATOMIC: the name is the whole scope. No role and no seasoning may be
        # read out of the words inside it.
        if case.get("atomic"):
            extras = []
            if settled_lens != "all":
                extras.append(f"lens={settled_lens!r}")
            if seasoning not in ("", "any"):
                extras.append(f"seasoning={seasoning!r}")
            out["NAMED_SOURCE_ATOMIC"] = not extras
            if extras:
                problems.append(
                    f"the governed name was decomposed into {', '.join(extras)}; "
                    f"those words are part of the book's name")
    else:
        if reference:
            out["SILENT_SCOPE_NARROWING"] = True
            problems.append(f"a book was named that the question did not name: "
                            f"{reference!r}")

    # -- what must survive alongside the scope ------------------------------ #
    out["MEASURE_PRESERVED"] = bool(payload.get("measures"))
    if not out["MEASURE_PRESERVED"] and case["axis"] != LIFECYCLE:
        problems.append("no measure survived")
    if case.get("temporal"):
        out["TEMPORAL_DROP"] = not temporal_stated(payload)
        if out["TEMPORAL_DROP"]:
            problems.append("the temporal constraint was dropped")

    if case.get("capability_not"):
        kept = (capability in ATTRIBUTION_CAPABILITIES
                or operation in ATTRIBUTION_OPERATIONS)
        out["ATTRIBUTION_CAPABILITY_DROP"] = not kept
        if not kept:
            problems.append(
                f"attribution intent lost: capability={capability!r} "
                f"operation={operation!r} carries no movement meaning")

    out["AXIS_CORRECT"] = not problems
    if problems:
        out["verdict"] = "CLARIFIED" if (blocking and not any(
            out[k] for k in ("SILENT_SCOPE_DROP", "SILENT_SCOPE_WIDENING",
                             "SILENT_SCOPE_NARROWING", "TEMPORAL_DROP",
                             "ATTRIBUTION_CAPABILITY_DROP",
                             "PHYSICAL_SOURCE_BINDING_AUTHORED"))) else "FAIL"
    else:
        out["verdict"] = "PASS"
    return out


# --------------------------------------------------------------------------- #
# the deterministic compile check
# --------------------------------------------------------------------------- #

def compile_check(case: Mapping[str, Any], payload: Mapping[str, Any],
                  registry: Any) -> Dict[str, Any]:
    from mi_agent.interpretation_v2.compiler import (CompilerContext,
                                                     DeterministicCompiler)
    from mi_agent.interpretation_v2.intent import parse_candidate_intent

    row: Dict[str, Any] = {"id": case["id"], "problems": []}
    try:
        intent = parse_candidate_intent(dict(payload))
    except Exception as exc:                                     # noqa: BLE001
        row.update({"verdict": "UNPARSEABLE",
                    "problems": [f"{type(exc).__name__}: {exc}"[:160]]})
        return row

    compiler = DeterministicCompiler(CompilerContext(source_registry=registry))
    result = compiler.compile(intent)
    plan = getattr(result, "plan", None)
    row["outcome"] = str(result.outcome)
    row["codes"] = [r.code for r in (result.reasons or ())]
    row["compiled"] = plan is not None
    if plan is None:
        row["verdict"] = "REFUSED"
        row["scope"] = []
        return row

    scope = sorted((p.canonical_field, p.value)
                   for p in plan.population.scope_predicates)
    row["scope"] = scope
    row["base"] = plan.population.base

    expected = []
    if case.get("role"):
        expected.append((ROLE_FIELD, case["role"]))
    if case.get("resolves_to"):
        expected.append((SOURCE_FIELD, case["resolves_to"]))
    expected.sort()
    if scope != expected:
        row["problems"].append(f"expected {expected}, got {scope}")
    if case.get("base") and plan.population.base != case["base"]:
        row["problems"].append(
            f"population {case['base']!r} compiled as {plan.population.base!r}")
    row["verdict"] = "PASS" if not row["problems"] else "FAIL"
    return row


# --------------------------------------------------------------------------- #
# stand-ins, so the harness is proved before it is paid for
# --------------------------------------------------------------------------- #

def _base(payload: Mapping[str, Any]) -> Dict[str, Any]:
    body = {"schema_version": "candidate_intent/1.0", "dimensions": [],
            "filters": [], "geography": {"requested": False},
            "comparison": {"kind": "none"},
            "capability": "generic_analysis", "operation": "point_in_time",
            "measures": [{"concept": "current_outstanding_balance"}],
            "time": {"form": "current"}}
    body.update(payload)
    return body


STAND_INS: Mapping[str, Mapping[str, Any]] = {
    # the lifecycle phrase, correctly on the lifecycle axis
    "M1": {"population": {"base": "funded"}},
    # its mirror
    "M2": {"population": {"base": "pipeline"}},
    # two axes in one phrase, and no third
    "M3": {"population": {"base": "funded", "lens": "acquired"}},
    # a governed name kept whole
    "M7": {"population": {"base": "funded",
                          "source_reference": "ALP Acquired Back Book"}},
    # the failure the atomic rule exists to catch
    "M8": {"population": {"base": "funded", "lens": "acquired",
                          "seasoning": "back_book",
                          "source_reference": "ALP back book"}},
    # attribution survives three readable axes
    "M10": {"population": {"base": "funded", "lens": "acquired"},
            "capability": "period_movement", "operation": "movement",
            "time": {"form": "relative_pair", "labels": ["this month"],
                     "periods_back": 1}},
}


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
                    # THE REGISTRY IS PASSED IN, which is the whole point: the
                    # boundary run asked these questions of a model that could
                    # not see which books exist.
                    outcome = interpreter.interpret(
                        case["question"], source_registry=registry)
                    served = getattr(outcome, "model_id", "") or ""
                    retrievals = len(getattr(outcome, "metadata_calls", ()) or ())
                    payload = (dict(outcome.raw_payload)
                               if getattr(outcome, "raw_payload", None) else None)
                    if payload is None:
                        error = f"no payload: {getattr(outcome, 'reason', '')}"
                except Exception as exc:                         # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"[:200]

        row = adjudicate(case, payload, error)
        row["raw_payload"] = payload
        row["served_model"] = served
        row["metadata_retrievals"] = retrievals
        report["cases"].append(row)
        print(f"  {row['verdict']:16s} {case['id']:4s} base={row.get('base')!r} "
              f"lens={row.get('lens')!r} seasoning={row.get('seasoning')!r} "
              f"src={row.get('source_reference')!r} "
              f"cap={row.get('capability')!r}/{row.get('operation')!r}")
        for problem in row["problems"]:
            print(f"        {problem}")

        if payload is not None and case["axis"] != ATTRIBUTION:
            checked = compile_check(case, payload, registry)
            report["compile_checks"].append(checked)
            print(f"        compile {checked['verdict']}: {checked.get('scope')}")
            for problem in checked["problems"]:
                print(f"          {problem}")

    rows = report["cases"]
    def _count(key):
        return sum(1 for r in rows if r.get(key))

    def _axis(axis, key="AXIS_CORRECT"):
        got = [r for r in rows if r["axis"] == axis]
        return f"{sum(1 for r in got if r.get(key))}/{len(got)}"

    named = [r for r in rows if r["axis"] == NAMED]
    report["adjudication"] = {
        "BACK_BOOK_TO_FUNDED": _axis(LIFECYCLE),
        "FRONT_BOOK_TO_PIPELINE": "/".join([
            str(sum(1 for r in rows if r["id"] == "M2" and r.get("BASE_CORRECT"))),
            "1"]),
        "DIRECT_ROLE_PRESERVED": "/".join([
            str(sum(1 for r in rows if r["id"] in ("M4", "M6")
                    and r.get("ROLE_CORRECT"))), "2"]),
        "ACQUIRED_ROLE_PRESERVED": "/".join([
            str(sum(1 for r in rows if r["id"] in ("M3", "M5")
                    and r.get("ROLE_CORRECT"))), "2"]),
        "NAMED_SOURCE_ATOMIC": "/".join([
            str(sum(1 for r in named if r.get("NAMED_SOURCE_ATOMIC"))),
            str(len(named))]),
        "NAMED_SOURCE_REFERENCE_PRESERVED": "/".join([
            str(sum(1 for r in named
                    if r.get("NAMED_SOURCE_REFERENCE_PRESERVED"))),
            str(len(named))]),
        "SEASONING_INFERRED_FROM_BACK_BOOK": _count("SEASONING_INFERRED"),
        "PHYSICAL_SOURCE_BINDINGS_AUTHORED_BY_MODEL":
            _count("PHYSICAL_SOURCE_BINDING_AUTHORED"),
        "ATTRIBUTION_CAPABILITY_DROPS": _count("ATTRIBUTION_CAPABILITY_DROP"),
        "SILENT_SCOPE_DROPS": _count("SILENT_SCOPE_DROP"),
        "SILENT_SCOPE_WIDENINGS": _count("SILENT_SCOPE_WIDENING"),
        "SILENT_SCOPE_NARROWINGS": _count("SILENT_SCOPE_NARROWING"),
        "TEMPORAL_DROPS": _count("TEMPORAL_DROP"),
    }
    report["totals"] = {
        "cases": len(rows),
        "interpreted": sum(1 for r in rows if r.get("interpreted")),
        "passed": sum(1 for r in rows if r["verdict"] == "PASS"),
        "clarified": sum(1 for r in rows if r["verdict"] == "CLARIFIED"),
        "failed": sum(1 for r in rows if r["verdict"] == "FAIL"),
        "compile_pass": sum(1 for c in report["compile_checks"]
                            if c["verdict"] == "PASS"),
        "compile_checked": len(report["compile_checks"]),
    }
    report["served_models"] = sorted(
        {r.get("served_model") for r in rows if r.get("served_model")})
    if not args.dry_run and report["served_models"] != [manifest["model"]]:
        print(f"::warning::served model(s) {report['served_models']} are not "
              f"exactly [{manifest['model']!r}]")

    Path(args.json_out).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  TOTALS {json.dumps(report['totals'])}")
    print(f"  ADJUDICATION {json.dumps(report['adjudication'])}")
    print(f"  written {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
