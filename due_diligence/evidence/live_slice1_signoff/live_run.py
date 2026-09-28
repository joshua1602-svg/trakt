#!/usr/bin/env python3
"""Phases 2-6: a FRESH live claude-opus-5 interpretation, end to end.

One question only: can a live model call travel through the frozen
interpretation_v2 layer, compile to a governed plan, be judged by the slice 1
eligibility gate, drive the existing deterministic executor through
`plan_runtime_adapter`, and land on the independently correct figure — without a
legacy parser reading the question again and without the shadow being able to
change what a user would have been served.

WHAT THIS HARNESS MAY NOT DO. It reads the pre-registered manifest and verifies
its sha256 against the file committed before any call was made; it never writes
to it. It changes no product code. It makes at most `authorised_live_calls`
successful model calls, retrying only a transport failure of one of those same
calls, and it never makes a probe, a reworded attempt, or a diagnostic call. If
something fails, the failure is recorded — nothing is retuned.

HOW THE LEGACY-BYPASS PROOF IS MADE (phase 5). A static import-closure proof is
NOT available and this harness does not pretend otherwise: `mi_agent/__init__.py`
imports `llm_query_parser` and `mi_agent_workflow` eagerly, so importing anything
under `mi_agent` puts them in `sys.modules` whether they are used or not. That is
a pre-existing property of the package, not something slice 1 introduced. So the
proof here is a RUNTIME one, made two ways at once:

    * `sys.setprofile` records every function call whose file is one of the nine
      legacy semantic modules, for the whole per-case pipeline including the live
      model call. It sees only the calling thread — disclosed, because the SDK
      could in principle call out on another.
    * named legacy entry points are wrapped in counting shims for the entire run,
      which catch a call from any thread.

Run: `ANTHROPIC_API_KEY=... python due_diligence/evidence/live_slice1_signoff/live_run.py`
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import plan_runtime_adapter as adapter                  # noqa: E402
from mi_agent.interpretation_v2.compiler import (CompilerContext,     # noqa: E402
                                                 DeterministicCompiler)
from mi_agent.interpretation_v2.opus_interpreter import (              # noqa: E402
    AnthropicInterpreterClient, OpusInterpreter)
from mi_agent.interpretation_v2.outcomes import refuse                # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics             # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth            # noqa: E402

sys.path.insert(0, str(_REPO_ROOT / "due_diligence" / "evidence"
                      / "plan_shadow_slice1"))
import corpus_replay as control                                       # noqa: E402

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "live_bank_manifest.json"
MANIFEST_HASH = HERE / "live_bank_manifest.sha256"
OUT = HERE / "live_run_result.json"
LEDGER = HERE / "live_shadow_ledger.jsonl"
REGISTRY = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

REQUIRED_MODEL = "claude-opus-5"

#: Whole modules that must not be entered on the new path.
LEGACY_FILES = frozenset({
    "parsed_question.py", "llm_query_parser.py", "recogniser_registry.py",
    "chat_routing.py", "portfolio_lens.py", "semantic_resolver.py",
    "mi_agent_workflow.py", "execution_receipt.py", "mi_query_contract.py",
})

#: Named entry points, wrapped for the whole run. `(module path, attribute path)`.
LEGACY_ENTRY_POINTS = (
    ("mi_agent.parsed_question", "ParsedQuestion.parse"),
    ("mi_agent.llm_query_parser", "parse_user_question"),
    ("mi_agent.llm_query_parser", "parse_with_repair"),
    ("mi_agent.llm_query_parser", "parse_llm_response_to_spec"),
    ("mi_agent.llm_query_parser", "build_prompt"),
    ("mi_agent.llm_query_parser", "find_field"),
    ("mi_agent_api.recogniser_registry", "RecogniserRegistry.candidates"),
    ("mi_agent_api.recogniser_registry", "RecogniserRegistry.ordered"),
    ("mi_agent_api.chat_routing", "_is_portfolio_summary"),
    ("mi_agent_api.chat_routing", "_is_pipeline_summary"),
    ("mi_agent_api.chat_routing", "_names_something_else"),
    ("mi_agent.portfolio_lens", "lens_phrase_spans"),
    ("mi_agent.portfolio_lens", "names_selected_scope"),
    ("mi_agent.portfolio_lens", "names_total_scope"),
    ("mi_agent.portfolio_lens", "names_a_book_noun"),
    ("mi_agent.portfolio_lens", "mask_scope_phrases"),
    ("mi_agent.execution_receipt", "requested_dimension_terms"),
    ("mi_agent.execution_receipt", "dimension_role"),
    ("mi_agent.mi_agent_workflow", "run_mi_agent_query"),
)

# Interpretation outcome classes.
EXACT_PLAN_MATCH = "EXACT_PLAN_MATCH"
SEMANTICALLY_EQUIVALENT = "SEMANTICALLY_EQUIVALENT"
JUSTIFIED_CLARIFY = "JUSTIFIED_CLARIFY"
JUSTIFIED_REFUSE = "JUSTIFIED_REFUSE"
INTERPRETATION_DEVIATION = "INTERPRETATION_DEVIATION"

#: Facets whose loss is a SILENT EXPLICIT SEMANTIC DROP rather than a change.
LOSABLE = ("filters", "dimensions", "measure_field", "statistic",
           "comparison_kind", "period_form", "population_lens",
           "geography_requested")


# --------------------------------------------------------------------------- #
# instrumentation
# --------------------------------------------------------------------------- #

class LegacyWatch:
    """Counts any call into the legacy semantic surface, two ways."""

    def __init__(self) -> None:
        self.profile_hits: Dict[str, Dict[str, int]] = {}
        self.shim_hits: Dict[str, int] = {}
        self._installed: List[Tuple[Any, str, Any]] = []

    # -- named entry points, active for the whole run ------------------------
    def install_shims(self) -> List[str]:
        import importlib
        missing: List[str] = []
        for module_path, attribute in LEGACY_ENTRY_POINTS:
            label = f"{module_path}.{attribute}"
            try:
                module = importlib.import_module(module_path)
            except Exception:                                        # noqa: BLE001
                missing.append(label)
                continue
            owner: Any = module
            parts = attribute.split(".")
            for part in parts[:-1]:
                owner = getattr(owner, part, None)
                if owner is None:
                    break
            name = parts[-1]
            original = getattr(owner, name, None) if owner is not None else None
            if original is None:
                missing.append(label)
                continue
            self.shim_hits[label] = 0
            self._installed.append((owner, name, original))
            self._wrap(owner, name, original, label)
        return missing

    def _wrap(self, owner: Any, name: str, original: Any, label: str) -> None:
        watch = self
        is_class_method = isinstance(
            owner.__dict__.get(name) if hasattr(owner, "__dict__") else None,
            classmethod)
        raw = original.__func__ if is_class_method else original

        def shim(*args, **kwargs):
            watch.shim_hits[label] = watch.shim_hits.get(label, 0) + 1
            return raw(*args, **kwargs)

        setattr(owner, name, classmethod(shim) if is_class_method else shim)

    def remove_shims(self) -> None:
        for owner, name, original in reversed(self._installed):
            setattr(owner, name, original)
        self._installed = []

    # -- the profiler, active per case --------------------------------------
    def _hook(self, frame, event, _arg):
        if event != "call":
            return
        base = os.path.basename(frame.f_code.co_filename)
        if base in LEGACY_FILES:
            bucket = self.profile_hits.setdefault(base, {})
            name = frame.f_code.co_name
            bucket[name] = bucket.get(name, 0) + 1

    def __enter__(self) -> "LegacyWatch":
        sys.setprofile(self._hook)
        return self

    def __exit__(self, *_) -> None:
        sys.setprofile(None)

    @property
    def total_profile_calls(self) -> int:
        return sum(sum(v.values()) for v in self.profile_hits.values())

    @property
    def total_shim_calls(self) -> int:
        return sum(self.shim_hits.values())


# --------------------------------------------------------------------------- #
# offline comparison against the pre-registration
# --------------------------------------------------------------------------- #

def live_semantics(plan: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The same projection `preregister._semantics` makes, over the live plan."""
    sys.path.insert(0, str(HERE))
    import preregister
    return preregister._semantics(plan)


def _comparable(value: Any) -> Any:
    """Tuples and lists compare equal, recursively.

    The manifest is JSON, which has no tuple, so a pre-registered filter triple
    reads back as a list while the live projection produces a tuple. An earlier
    version of this comparison tested them directly and called nine identical
    replays INTERPRETATION_DEVIATION on nothing but that. Found in the dry run,
    before any live call was made.
    """
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _comparable(item) for key, item in value.items()}
    return value


def compare_semantics(expected: Mapping[str, Any],
                      live: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """`(drops, changes)`. A drop is a stated facet that vanished."""
    drops, changes = [], []
    for facet in sorted(set(expected) | set(live)):
        want, got = expected.get(facet), live.get(facet)
        if _comparable(want) == _comparable(got):
            continue
        vanished = bool(want) and not got
        if facet in LOSABLE and vanished:
            drops.append(f"{facet}: {want!r} -> {got!r}")
        else:
            changes.append(f"{facet}: {want!r} -> {got!r}")
    return drops, changes


def classify_interpretation(case: Mapping[str, Any], outcome_name: str,
                            live_plan: Optional[Mapping[str, Any]],
                            live_plan_id: str) -> Dict[str, Any]:
    expected_disposition = case["expected_interpretation_disposition"]
    expected = case["expected_governed_semantics"]
    live = live_semantics(live_plan)
    drops, changes = ([], [])

    if outcome_name == "PLAN" and expected_disposition == "PLAN":
        drops, changes = compare_semantics(expected, live)
        if not drops and not changes:
            verdict = (EXACT_PLAN_MATCH
                       if live_plan_id == case["expected_frozen_plan_id"]
                       else SEMANTICALLY_EQUIVALENT)
        else:
            verdict = INTERPRETATION_DEVIATION
    elif outcome_name == "CLARIFY" and expected_disposition == "CLARIFY":
        verdict = JUSTIFIED_CLARIFY
    elif outcome_name == "REFUSE" and expected_disposition == "REFUSE":
        verdict = JUSTIFIED_REFUSE
    else:
        verdict = INTERPRETATION_DEVIATION
        if expected_disposition == "PLAN" and outcome_name in ("CLARIFY", "REFUSE"):
            changes = [f"disposition: PLAN -> {outcome_name} "
                       f"(no plan, so no facet was silently dropped)"]
        else:
            changes = [f"disposition: {expected_disposition} -> {outcome_name}"]
            if outcome_name == "PLAN":
                _, changes_extra = compare_semantics(expected, live)
                changes += changes_extra

    return {"verdict": verdict, "live_semantics": live,
            "silent_semantic_drops": drops, "semantic_changes": changes}


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def verify_manifest() -> Dict[str, Any]:
    body = MANIFEST.read_text()
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    recorded = MANIFEST_HASH.read_text().split()[0]
    if digest != recorded:
        raise SystemExit(f"the manifest has changed since it was hashed: "
                         f"{digest} != {recorded}")
    return json.loads(body)


def execute_case(case: Mapping[str, Any], live_plan: Mapping[str, Any],
                 book: Any, semantics: Any) -> Dict[str, Any]:
    """Phase 4. Independent truth only; the legacy answer is never the oracle."""
    dimensions = live_semantics(live_plan)["dimensions"]
    missing = sorted(control._referenced_fields(live_plan) - set(book.columns))
    if missing:
        return {"disposition": "UNEXECUTABLE_FIXTURE_GAP",
                "detail": f"the replay frame does not carry {missing}"}

    out = adapter.execute_shadow_governed_plan(live_plan, book, semantics)
    record: Dict[str, Any] = {
        "shadow_error": out.error,
        "receipt": dict(out.receipt),
        "spec": {"metric": getattr(out.spec, "metric", None),
                 "aggregation": getattr(out.spec, "aggregation", None),
                 "dimensions": list(getattr(out.spec, "dimensions", None) or ()),
                 "filters": dict(getattr(out.spec, "filters", None) or {})},
    }
    if out.error:
        record["disposition"] = "SHADOW_EXECUTION_ERROR"
        return record

    # The spec must still carry every facet the plan stated.
    expected_filters = {f[0] for f in live_semantics(live_plan)["filters"]}
    record["filter_drops"] = sorted(expected_filters - set(record["spec"]["filters"]))
    record["dimension_drops"] = sorted(set(dimensions)
                                       - set(record["spec"]["dimensions"]))
    record["measure_drop"] = bool(
        live_semantics(live_plan)["measure_field"]
        and record["spec"]["aggregation"] != "count"
        and record["spec"]["metric"] != live_semantics(live_plan)["measure_field"])

    if dimensions:
        verdict, note = control.compare_cells(book, live_plan, semantics, dimensions)
        expected_cells = case.get("independent_expected_cells") or {}
        record.update({
            "disposition": "EXECUTED_GROUPED",
            "grouped_verdict": verdict, "grouped_note": note,
            "cells_expected_in_manifest": len(expected_cells),
            "independently_correct": verdict == control.GROUPED_CELL_PARITY,
        })
        return record

    expected_value = case.get("independent_expected_value")
    produced = out.value
    correct = (expected_value is not None and produced is not None
               and abs(produced - expected_value) < 0.01)
    record.update({"disposition": "EXECUTED_SCALAR",
                   "expected_value": expected_value, "produced_value": produced,
                   "independently_correct": bool(correct)})
    return record


def shadow_isolation(live_plan: Optional[Mapping[str, Any]], book: Any,
                     semantics: Any, case_id: str) -> Dict[str, Any]:
    """Phase 6. The control envelope must come back untouched, always.

    THE ENVELOPE HERE IS A STUB, and its `value` is deliberately a number no
    question would produce. Phase 6 asks one thing — can anything the shadow does
    reach what a user would have been served — so the only assertions that matter
    are that the envelope comes back byte-identical and that nothing was raised.
    The ledger classification this produces is therefore NOT a parity result and
    must not be read as one; phase 4 owns parity, against the independent control.
    """
    legacy = {"ok": True, "value": 123456.78,
              "route": "phase6_stub_control_not_a_parity_baseline",
              "answer": "the legacy answer, unchanged"}
    before = json.dumps(legacy, sort_keys=True)
    os.environ[adapter.SHADOW_ENV_VAR] = adapter.SHADOW_ON
    os.environ[adapter.LEDGER_ENV_VAR] = str(LEDGER)
    escaped = ""
    row = None
    try:
        row = adapter.observe(result=legacy, frame=book, semantics=semantics,
                              plan=live_plan, case_id=case_id,
                              view="live_signoff", portfolio_id="oracle_book")
    except BaseException as exc:                                     # noqa: BLE001
        escaped = f"{type(exc).__name__}: {exc}"
    finally:
        os.environ.pop(adapter.SHADOW_ENV_VAR, None)
        os.environ.pop(adapter.LEDGER_ENV_VAR, None)
    return {
        "control_unchanged": json.dumps(legacy, sort_keys=True) == before,
        "exception_escaped": escaped,
        "ledger_classification": (row or {}).get("classification", ""),
        "ledger_classification_is_not_a_parity_result": True,
        "shadow_returned_a_row": row is not None,
    }


def main(argv) -> int:
    manifest = verify_manifest()
    budget = int(manifest["authorised_live_calls"])
    cases = manifest["cases"]
    if len(cases) > budget:
        raise SystemExit("the bank exceeds the authorised call budget")

    # A DRY RUN IS NOT A LIVE RUN and is never reported as one. It replays the
    # frozen payloads through the identical pipeline so the harness's own
    # mechanics are proved before any budget is spent — the alternative is
    # discovering a harness defect with paid calls, which has already happened
    # once in this engagement and must not happen again.
    dry_run = "--dry-run" in argv
    if dry_run:
        from mi_agent.interpretation_v2.opus_interpreter import ReplayClient
        frozen = json.loads(
            (_REPO_ROOT / manifest["frozen_source"]).read_text())
        client = ReplayClient({r["question"]: r["raw_payload"]
                               for r in frozen["results"] if r.get("raw_payload")},
                              model_id="REPLAY-NOT-A-LIVE-CALL")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("no ANTHROPIC_API_KEY; this phase needs live calls")
        client = AnthropicInterpreterClient(model=REQUIRED_MODEL, api_key=api_key)
    interpreter = OpusInterpreter(client)
    compiler = DeterministicCompiler(CompilerContext())
    semantics = load_mi_semantics(str(REGISTRY))
    book = truth.canonical_book()
    if LEDGER.exists():
        LEDGER.unlink()

    watch = LegacyWatch()
    shims_missing = watch.install_shims()

    calls = {"attempted": 0, "successful": 0, "transport_failures": 0,
             "retries": 0}
    models_seen: Dict[str, int] = {}
    results: List[Dict[str, Any]] = []
    #: The live plans, kept in memory so the forced-failure check reuses one
    #: rather than buying another interpretation.
    live_plans: Dict[str, Optional[Mapping[str, Any]]] = {}

    try:
        for case in cases:
            question = case["question"]
            attempt = 0
            while True:
                attempt += 1
                calls["attempted"] += 1
                started = time.monotonic()
                with watch:
                    outcome = interpreter.interpret(question)
                    compiled = (compiler.compile(outcome.intent) if outcome.ok
                                else refuse(outcome.reason,
                                            compiler_version=compiler.version))
                elapsed = int((time.monotonic() - started) * 1000)
                # ONLY a transport failure is retried. MODEL_UNAVAILABLE is the
                # one code the client raises for an API/network failure;
                # MODEL_OUTPUT_MALFORMED is the model answering badly, and
                # retrying that would be an unauthorised second attempt at the
                # same question rather than a repair of a dropped connection.
                transport = (outcome.reason is not None
                             and outcome.reason.code == "MODEL_UNAVAILABLE")
                if transport and attempt <= 3:
                    calls["transport_failures"] += 1
                    calls["retries"] += 1
                    print(f"  {case['case_id']}  transport failure "
                          f"({outcome.reason.detail[:80]}), retry {attempt}",
                          flush=True)
                    continue
                break

            calls["successful"] += 1
            model_id = outcome.model_id or ""
            models_seen[model_id] = models_seen.get(model_id, 0) + 1

            live_plan = compiled.plan.to_dict() if compiled.is_plan else None
            live_plan_id = compiled.plan.plan_id if compiled.is_plan else ""
            live_plans[case["case_id"]] = live_plan
            interpretation = classify_interpretation(
                case, compiled.outcome, live_plan, live_plan_id)

            eligible, reason, detail = adapter.check_eligibility(live_plan)
            eligibility = {
                "live_eligible": eligible, "live_reason": reason,
                "live_detail": detail[:200],
                "expected_eligible": case["expected_slice1_eligible"],
                "expected_reason": case["expected_ineligible_reason"],
                "matches": (eligible == case["expected_slice1_eligible"]
                            and reason == case["expected_ineligible_reason"]),
            }

            execution = ({"disposition": "NOT_EXECUTED_INELIGIBLE"} if not eligible
                         else execute_case(case, live_plan, book, semantics))

            with watch:
                isolation = shadow_isolation(live_plan, book, semantics,
                                             case["case_id"])

            results.append({
                "case_id": case["case_id"],
                "frozen_question_id": case["frozen_question_id"],
                "question": question,
                "model_id": model_id,
                "usage": dict(outcome.usage or {}),
                "latency_ms": elapsed,
                "live_outcome": compiled.outcome,
                "live_reason_codes": list(compiled.codes()),
                "live_plan_id": live_plan_id,
                "expected_frozen_plan_id": case["expected_frozen_plan_id"],
                "interpretation": interpretation,
                "eligibility": eligibility,
                "execution": execution,
                "shadow_isolation": isolation,
            })
            print(f"  {case['case_id']}  {compiled.outcome:<8} "
                  f"{interpretation['verdict']:<24} "
                  f"elig={'Y' if eligible else reason:<22} "
                  f"{execution.get('disposition', '')}", flush=True)

        # One forced failure with a LIVE plan: an executor blow-up must not escape.
        # No extra model call — the plan is one already interpreted above, handed
        # a frame the executor cannot possibly use.
        forced: Dict[str, Any] = {"ran": False}
        eligible_plan = next((live_plans[r["case_id"]] for r in results
                              if r["eligibility"]["live_eligible"]), None)
        if eligible_plan is not None:
            forced = shadow_isolation(eligible_plan, object(), semantics,
                                      "FORCED_EXECUTOR_FAILURE")
            forced["ran"] = True
    finally:
        watch.remove_shims()

    report = {
        "phase": "Slice 1 live Opus integration sign-off — phases 2-6",
        "manifest": str(MANIFEST.relative_to(_REPO_ROOT)),
        "manifest_sha256": MANIFEST_HASH.read_text().split()[0],
        "manifest_verified_unchanged": True,
        "required_model": REQUIRED_MODEL,
        "models_returned": models_seen,
        "model_substitutions": sum(
            count for model, count in models_seen.items()
            if REQUIRED_MODEL not in model),
        "calls": calls,
        "legacy_watch": {
            "profile_hits": watch.profile_hits,
            "profile_total_calls": watch.total_profile_calls,
            "shim_hits": {k: v for k, v in watch.shim_hits.items() if v},
            "shim_total_calls": watch.total_shim_calls,
            "entry_points_wrapped": len(watch.shim_hits),
            "entry_points_not_found": shims_missing,
            "disclosure": "sys.setprofile sees only the calling thread; the named "
                          "entry-point shims catch a call from any thread",
        },
        "forced_executor_failure_with_a_live_plan": forced,
        "cases": results,
    }
    report["live"] = not dry_run
    out = OUT if not dry_run else OUT.with_name("dry_run_result.json")
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\nWRITTEN {out.relative_to(_REPO_ROOT)}"
          f"{'  (DRY RUN — no live call was made)' if dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
