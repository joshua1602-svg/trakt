#!/usr/bin/env python3
"""Phase 5B: a FRESH live claude-opus-5 interpretation of temporal questions.

One question only: when a real model reads "how has funded balance changed over
the last six months?", does what it emits carry temporal intent the deterministic
layer can ACT on — or only words that clarify?

    raw question
      -> OpusInterpreter.interpret            (live; the frozen interpreter)
      -> CandidateIntent                       (the model's only output)
      -> DeterministicCompiler.compile         (frozen)
      -> GovernedQueryPlan
      -> plan_temporal_runtime.check_temporal_eligibility
      -> plan_temporal_runtime.resolve_temporal
      -> SnapshotSelector -> resolved snapshot headers
      STOP.

WHAT THIS HARNESS MAY NOT DO. It changes no product code. It calls no MI API and
needs no bearer. It never executes a measure against a book — the boundary under
test ends at snapshot selection, and running the executor as well would spend
budget proving something the offline acceptance already proved. It reads the
pre-registered manifest and verifies its sha256 against the file committed before
any call was made; it never writes to it. It makes at most
`authorised_live_calls` model calls — exactly one per case, no probe, no
reworded second try, no diagnostic call. A call that fails is RECORDED as a
failure; it is not retried and the question is not asked a second way.

THE MODEL IS SCORED ON WHAT THE GOVERNED LAYER COULD DO WITH ITS OUTPUT, not on
whether it typed what the manifest typed. Which span form it chose, which
operation, which words — none of that is pinned. Eligibility, the resolved
snapshot set, and the surviving measure/filters/dimensions are.

THE CATALOGUE IS A FIXTURE, AND IT IS THE ONLY DATA HERE. Eight monthly periods
built by `portfolio_truth_oracle.canonical_history`, registered through the real
`LocalFsSnapshotStore`. No client book is read. Relative windows resolve against
the catalogue's latest period, so the expected answer does not drift with the
wall clock.

    --dry-run   replays authored stand-in payloads through the SAME pipeline,
                so the harness is proved before a penny is spent. The stand-ins
                are this repository's guesses at what a model might emit; they
                prove the harness, and they prove nothing about Opus.

Run: `ANTHROPIC_API_KEY=... python due_diligence/evidence/plan_temporal_slice2/temporal_live_run.py`
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import plan_temporal_runtime as temporal                 # noqa: E402
from mi_agent.interpretation_v2.compiler import DeterministicCompiler   # noqa: E402
from mi_agent.interpretation_v2.opus_interpreter import (               # noqa: E402
    AnthropicInterpreterClient, OpusInterpreter, ReplayClient)
from mi_agent.interpretation_v2.outcomes import refuse                  # noqa: E402
from mi_agent.tests import temporal_snapshot_fixture as fixture         # noqa: E402

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "temporal_bank_manifest.json"
MANIFEST_HASH = HERE / "temporal_bank_manifest.sha256"
DRY_PAYLOADS = HERE / "temporal_dry_run_payloads.json"
#: The same fifteen questions with four payloads deliberately broken, so the
#: harness is shown to CATCH a failure rather than only to run. A harness whose
#: only demonstrated outcome is PASS has demonstrated nothing.
DEGRADED_PAYLOADS = HERE / "temporal_dry_run_payloads_degraded.json"
OUT = HERE / "temporal_live_run_result.json"
RETEST_OUT = HERE / "temporal_live_retest_result.json"
DRY_OUT = HERE / "temporal_dry_run_result.json"
DEGRADED_OUT = HERE / "temporal_dry_run_degraded_result.json"

# Verdicts. The first three are the boundary holding; the rest name exactly how
# it did not, because "the live run failed" is not a finding anyone can act on.
BOUNDARY_HELD = "TEMPORAL_BOUNDARY_HELD"
JUSTIFIED_CLARIFY = "JUSTIFIED_CLARIFY"
JUSTIFIED_REFUSE = "JUSTIFIED_REFUSE"
JUSTIFIED_INELIGIBLE = "JUSTIFIED_INELIGIBLE"
TEMPORAL_INTENT_ABSENT = "TEMPORAL_INTENT_ABSENT"
PERIOD_UNRESOLVABLE = "PERIOD_UNRESOLVABLE"
WRONG_SNAPSHOTS = "WRONG_SNAPSHOTS"
SEMANTIC_DEVIATION = "SEMANTIC_DEVIATION"
UNEXPECTED_INELIGIBLE = "UNEXPECTED_INELIGIBLE"
UNEXPECTED_NON_PLAN = "UNEXPECTED_NON_PLAN"
INTERPRETER_FAILED = "INTERPRETER_FAILED"

_HELD = (BOUNDARY_HELD, JUSTIFIED_CLARIFY, JUSTIFIED_REFUSE,
         JUSTIFIED_INELIGIBLE)


# --------------------------------------------------------------------------- #
# reading a live plan
# --------------------------------------------------------------------------- #

def plan_semantics(plan: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The governed semantics a temporal plan must preserve. Read off the plan."""
    if not plan:
        return {}
    output = (tuple(plan.get("outputs") or ()) or ({},))[0]
    measure = (tuple(output.get("measures") or ()) or ({},))[0]
    period = plan.get("period") or {}
    return {
        "capability": plan.get("capability"),
        "operation": plan.get("operation"),
        "measure_concept": measure.get("concept"),
        "measure_field": measure.get("canonical_field"),
        "statistic": measure.get("statistic"),
        "weight_field": measure.get("weight_field"),
        "dimensions": sorted(d.get("canonical_field")
                             for d in (output.get("dimensions") or ())),
        "filters": sorted([f.get("canonical_field"),
                           str(f.get("comparator") or "eq"),
                           _scalar(f.get("value"))]
                          for f in (tuple(plan.get("filters") or ())
                                    + tuple(output.get("filters") or ()))),
        "population_base": (plan.get("population") or {}).get("base"),
        "population_lens": (plan.get("population") or {}).get("lens"),
        "period_form": period.get("form"),
        "period_labels": list(period.get("labels") or ()),
        "period_grain": period.get("grain"),
        "period_periods_back": period.get("periods_back"),
        "comparison_kind": plan.get("comparison_kind"),
    }


def _scalar(value: Any) -> Any:
    """A filter value flattened for comparison. A list keeps its order."""
    if isinstance(value, (list, tuple)):
        return [str(v).lower() for v in value]
    return str(value).lower() if isinstance(value, str) else value


def physical_binding(plan: Mapping[str, Any]) -> Optional[str]:
    """A date or snapshot id anywhere in the plan's PERIOD. There must be none.

    The same check the offline acceptance makes. It is repeated against LIVE
    output because "the intent parser refuses physical bindings" is a claim about
    a guard, and this is the first run in which a real model is on the other side
    of it.
    """
    period = plan.get("period") or {}
    for key in ("labels", "form", "grain", "contract"):
        raw = period.get(key)
        values = list(raw or ()) if isinstance(raw, (list, tuple)) else [raw]
        for value in values:
            token = str(value or "")
            if "snapshot" in token.lower():
                return f"period.{key}={token!r}"
            digits = [p for p in token.replace("/", "-").split("-")
                      if p.isdigit()]
            if len(digits) >= 3:
                return f"period.{key}={token!r}"
    return None


# --------------------------------------------------------------------------- #
# scoring one case
# --------------------------------------------------------------------------- #

def compare_semantics(case: Mapping[str, Any],
                      live: Mapping[str, Any]) -> List[str]:
    """Facets the manifest pinned that the live plan did not carry."""
    problems: List[str] = []
    wanted_measure = case.get("expected_measure_concept")
    if wanted_measure and live.get("measure_concept") != wanted_measure:
        problems.append(f"measure_concept: {wanted_measure!r} -> "
                        f"{live.get('measure_concept')!r}")
    allowed = list(case.get("expected_statistics") or ())
    if allowed and live.get("statistic") not in allowed:
        problems.append(f"statistic: one of {allowed} -> "
                        f"{live.get('statistic')!r}")
    wanted_dims = sorted(case.get("expected_dimensions") or ())
    if wanted_dims != sorted(live.get("dimensions") or ()):
        problems.append(f"dimensions: {wanted_dims} -> {live.get('dimensions')}")
    wanted_filters = sorted([f[0], f[1], _scalar(f[2])]
                            for f in (case.get("expected_filters") or ()))
    if wanted_filters != sorted(live.get("filters") or ()):
        problems.append(f"filters: {wanted_filters} -> {live.get('filters')}")
    wanted_base = case.get("expected_population_base")
    if wanted_base and live.get("population_base") != wanted_base:
        problems.append(f"population_base: {wanted_base!r} -> "
                        f"{live.get('population_base')!r}")
    return problems


def score(case: Mapping[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """The verdict for one case, from what the governed layer could do with it."""
    expected_disposition = case["expected_interpretation_disposition"]
    outcome = record.get("compile_outcome")

    if record.get("interpreter_error"):
        record["verdict"] = INTERPRETER_FAILED
        return record

    if outcome != "PLAN":
        if expected_disposition in (outcome, "ANY"):
            record["verdict"] = (JUSTIFIED_REFUSE if outcome == "REFUSE"
                                 else JUSTIFIED_CLARIFY)
        else:
            record["verdict"] = UNEXPECTED_NON_PLAN
        return record

    if expected_disposition not in ("PLAN", "ANY"):
        # A plan where the compiler was expected to stop. Only a fail-closed
        # perimeter can rescue this, so fall through and let eligibility decide.
        record["note_disposition"] = (f"expected {expected_disposition}, "
                                      f"the compiler produced a plan")

    # THE TIME CONSTRAINT IS CHECKED BEFORE ELIGIBILITY, and that ordering is
    # load-bearing. The slice 2 perimeter excludes `current` by design, so a
    # model that answers a temporal question with a current-period plan is
    # refused as PERIOD_NOT_TEMPORAL — correctly, but under a name that reads as
    # "this plan is slice 1's" rather than as "the model dropped the time
    # constraint". Those are the same event and only one of them is a finding
    # about the model. Checked first, it is reported as the finding.
    # Found by the degraded self-test, which is what that test is for.
    live = record.get("live_semantics") or {}
    if case.get("temporal_required") and live.get("period_form") in (None, "current"):
        record["verdict"] = TEMPORAL_INTENT_ABSENT
        record["problems"] = [f"period.form={live.get('period_form')!r} on a "
                              f"temporal question — the time constraint did not "
                              f"survive the model boundary"]
        return record

    eligible = record.get("eligible")
    expected_eligibility = case["expected_eligibility"]
    if not eligible:
        if expected_eligibility in ("INELIGIBLE", "NOT_REACHED"):
            record["verdict"] = JUSTIFIED_INELIGIBLE
        else:
            record["verdict"] = UNEXPECTED_INELIGIBLE
            record["problems"] = [
                f"expected {expected_eligibility}, the perimeter refused it: "
                f"{record.get('ineligible_reason')} — "
                f"{record.get('ineligible_detail')}"]
        return record
    if expected_eligibility in ("INELIGIBLE", "NOT_REACHED"):
        record["verdict"] = UNEXPECTED_INELIGIBLE
        record["problems"] = [f"expected {expected_eligibility}, the perimeter "
                              f"admitted it"]
        return record

    expected_snapshots = case.get("expected_snapshots")
    if not record.get("resolved"):
        # The plan carries temporal intent the resolver cannot settle. This is a
        # fail-closed outcome, and whether it is the RIGHT one depends on what
        # was pre-registered.
        if expected_snapshots == []:
            record["verdict"] = JUSTIFIED_CLARIFY
        else:
            record["verdict"] = PERIOD_UNRESOLVABLE
            record["problems"] = [f"{record.get('resolution_reason')}: "
                                  f"{record.get('resolution_detail')}"]
        return record

    if expected_snapshots == []:
        record["verdict"] = WRONG_SNAPSHOTS
        record["problems"] = [f"expected the resolver to clarify; it selected "
                              f"{record.get('snapshots')}"]
        return record

    problems = list(compare_semantics(case, live))
    if record.get("snapshots") != list(expected_snapshots or ()):
        record["problems"] = problems + [
            f"snapshots: {list(expected_snapshots or ())} -> "
            f"{record.get('snapshots')}"]
        record["verdict"] = WRONG_SNAPSHOTS
        return record
    if problems:
        record["problems"] = problems
        record["verdict"] = SEMANTIC_DEVIATION
        return record
    record["verdict"] = BOUNDARY_HELD
    return record


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def verify_manifest() -> Dict[str, Any]:
    body = MANIFEST.read_text(encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    recorded = MANIFEST_HASH.read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        raise SystemExit(f"the manifest has changed since it was hashed: "
                         f"{digest} != {recorded}")
    return json.loads(body)


def run_case(case: Mapping[str, Any], interpreter: Any, compiler: Any,
             store: Any) -> Dict[str, Any]:
    """One question, start to snapshot selection. Never raises."""
    record: Dict[str, Any] = {
        "id": case["id"], "question": case["question"],
        "category": case["category"], "problems": [],
    }
    started = time.time()
    outcome = interpreter.interpret(case["question"])
    record["latency_ms"] = int((time.time() - started) * 1000)
    record["model_id"] = outcome.model_id
    record["usage"] = dict(outcome.usage or {})
    record["raw_payload"] = outcome.raw_payload

    if not outcome.ok:
        record["interpreter_error"] = (outcome.reason.to_dict()
                                       if outcome.reason is not None else "unknown")
        record["compile_outcome"] = None
        return score(case, record)

    record["candidate_intent"] = (outcome.intent.to_dict()
                                  if outcome.intent is not None else None)
    compiled = compiler.compile(outcome.intent)
    record["compile_outcome"] = compiled.outcome
    record["compile_reasons"] = compiled.codes()

    if not compiled.is_plan:
        return score(case, record)

    plan = compiled.plan.to_dict()
    record["plan_id"] = plan.get("plan_id")
    record["live_semantics"] = plan_semantics(plan)
    record["physical_binding"] = physical_binding(plan)

    eligible, reason, detail = temporal.check_temporal_eligibility(plan)
    record["eligible"] = eligible
    record["ineligible_reason"] = reason
    record["ineligible_detail"] = detail[:220]
    if not eligible:
        return score(case, record)

    resolution = temporal.resolve_temporal(plan, store,
                                           client_id=fixture.CLIENT_ID,
                                           route=fixture.ROUTE)
    record["resolved"] = resolution.ok
    record["resolution_reason"] = resolution.reason
    record["resolution_detail"] = resolution.detail[:220]
    record["shape"] = resolution.shape
    record["basis"] = resolution.basis
    record["selector_mode"] = (resolution.selector.mode
                               if resolution.selector is not None else None)
    record["snapshots"] = list(resolution.reporting_dates)
    return score(case, record)


def main(argv: Sequence[str]) -> int:
    manifest = verify_manifest()
    cases = manifest["cases"]
    budget = int(manifest["authorised_live_calls"])
    degraded = "--degraded" in argv
    dry_run = degraded or "--dry-run" in argv

    # `--only T01,T05` asks a NAMED SUBSET and nothing else. It exists so a
    # correction can be re-measured on the cases it was meant to correct,
    # without re-buying the cases that already passed — re-asking those would
    # cost money to learn nothing and would quietly re-roll a result already
    # recorded. The budget falls to the size of the subset.
    only: Optional[List[str]] = None
    for index, token in enumerate(argv):
        if token == "--only" and index + 1 < len(argv):
            only = [part.strip() for part in argv[index + 1].split(",")
                    if part.strip()]
    if only is not None:
        unknown = sorted(set(only) - {c["id"] for c in cases})
        if unknown:
            raise SystemExit(f"--only names cases the manifest does not carry: "
                             f"{unknown}")
        cases = [c for c in cases if c["id"] in only]
        budget = len(cases)

    if dry_run:
        source = DEGRADED_PAYLOADS if degraded else DRY_PAYLOADS
        payloads = json.loads(source.read_text(encoding="utf-8"))
        client = ReplayClient({c["question"]: payloads[c["id"]] for c in cases})
    else:
        client = AnthropicInterpreterClient(model=manifest["model"])
        if not client.available:
            print("=== SLICE 2 TEMPORAL LIVE RUN — NOT RUN")
            print("  no ANTHROPIC_API_KEY in the environment, so no live call "
                  "was made and no result file was written.")
            print("  The harness is proved on frozen payloads with --dry-run; "
                  "that proves the harness, not the model.")
            return 2

    interpreter = OpusInterpreter(client)
    compiler = DeterministicCompiler()

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        store = fixture.build_store(Path(tmp) / "snapshots",
                                    fixture.default_history())
        catalogue = [h.reporting_date for h in
                     store.list_snapshots(fixture.CLIENT_ID, route=fixture.ROUTE)]
        if catalogue != list(manifest["fixture_catalogue"]):
            raise SystemExit(f"the fixture catalogue does not match the "
                             f"pre-registration: {catalogue}")
        records = [run_case(case, interpreter, compiler, store)
                   for case in cases[:budget]]

    from collections import Counter
    verdicts = Counter(r["verdict"] for r in records)
    held = sum(verdicts.get(v, 0) for v in _HELD)
    usage = {key: sum(int((r.get("usage") or {}).get(key) or 0) for r in records)
             for key in ("input_tokens", "output_tokens",
                         "cache_read_input_tokens", "cache_creation_input_tokens")}
    calls = sum(1 for r in records if r.get("model_id"))

    report = {
        "bank_id": manifest["bank_id"],
        "manifest_sha256": MANIFEST_HASH.read_text(encoding="utf-8").split()[0],
        "live": not dry_run,
        "model": manifest["model"],
        "authorised_live_calls": budget,
        "live_calls_made": 0 if dry_run else calls,
        "fixture_catalogue": catalogue,
        "verdicts": dict(verdicts),
        "boundary_held": held,
        "of": len(records),
        "temporal_intent_absent": verdicts.get(TEMPORAL_INTENT_ABSENT, 0),
        "period_unresolvable": verdicts.get(PERIOD_UNRESOLVABLE, 0),
        "wrong_snapshots": verdicts.get(WRONG_SNAPSHOTS, 0),
        "physical_bindings": sum(1 for r in records if r.get("physical_binding")),
        "token_usage": usage,
        "results": records,
    }
    report["payload_source"] = ("degraded stand-ins" if degraded else
                                "plausible stand-ins" if dry_run else "live model")
    report["subset"] = only
    # A SUBSET NEVER WRITES OVER THE FULL RUN'S FILE. A `--dry-run --only`
    # rehearsal of four cases would otherwise land on top of the committed
    # fifteen-case dry run and silently shrink the evidence — which it did once,
    # here, before this line existed.
    out = (DEGRADED_OUT if degraded else DRY_OUT) if dry_run else OUT
    if only:
        out = out.with_name(f"{out.stem}_subset.json") if dry_run else RETEST_OUT
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"=== SLICE 2 TEMPORAL "
          f"{'DEGRADED SELF-TEST' if degraded else 'DRY' if dry_run else 'LIVE'} "
          f"RUN")
    print(f"  cases                  {len(records)}")
    print(f"  live calls             {report['live_calls_made']} "
          f"(authorised {budget})")
    print(f"  BOUNDARY HELD          {held}/{len(records)}")
    for name, count in sorted(verdicts.items()):
        print(f"    {name:<28} {count}")
    print(f"  temporal intent absent {report['temporal_intent_absent']}")
    print(f"  period unresolvable    {report['period_unresolvable']}")
    print(f"  wrong snapshots        {report['wrong_snapshots']}")
    print(f"  physical bindings      {report['physical_bindings']}")
    if not dry_run:
        print(f"  tokens                 {usage}")
    for record in records:
        if record["verdict"] not in _HELD:
            print(f"  {record['verdict']} {record['id']}: "
                  f"{record.get('problems') or record.get('interpreter_error')}")
    print(f"  WRITTEN                {out.relative_to(_REPO_ROOT)}")
    if degraded:
        # Inverted on purpose: the degraded set PASSES when the harness caught
        # every injected fault and failed nothing else.
        caught = len(records) - held
        print(f"  FAULTS CAUGHT          {caught} (4 injected)")
        return 0 if caught == 4 else 1
    return 0 if held == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
