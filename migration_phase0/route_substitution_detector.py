#!/usr/bin/env python3
"""migration_phase0/route_substitution_detector.py — did another route answer?

READ-ONLY on production. Injects a controlled fault INSIDE a claimed route's
execution path — after the point where falling through to another candidate
becomes prohibited — and reports, from execution, the four facts that decide
whether a substitution happened:

    claimed route · failed route · final response route · claim boundary crossed

THE INVARIANT IS DELIBERATELY NARROW. A route legitimately declining and another
answering is NOT a substitution: that is the routing model working. The check
fires only when the claim boundary was crossed and the final answer came from
somewhere else:

    FAIL  if  claimed_route != final_route  AND  claimed execution failed

Structural evidence, not prose comparison. The prose of a substituted answer is
usually plausible — that is exactly why the defect survived: a `temporal_compare`
refusal about ranking reads like a considered answer to a ranking question.

    python -m migration_phase0.route_substitution_detector [--json out.json]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class InjectedExecutionFault(RuntimeError):
    """Deliberate. Never raised by production code."""


class DetectorError(RuntimeError):
    """The detector could not measure. Never absorbed into a clean result."""


@contextlib.contextmanager
def _watching_handlers():
    """Record which handlers are ENTERED. Restores the registry unconditionally."""
    from mi_agent_api.recogniser_registry import REGISTRY

    entered: List[str] = []
    originals = {rec.name: rec.handle for rec in REGISTRY.ordered()}
    for rec in REGISTRY.ordered():
        def handle(request, _name=rec.name, _fn=rec.handle):
            entered.append(_name)
            return _fn(request)
        object.__setattr__(rec, "handle", handle)
    try:
        yield entered
    finally:
        for rec in REGISTRY.ordered():
            if rec.name in originals:
                object.__setattr__(rec, "handle", originals[rec.name])


@contextlib.contextmanager
def _fault_after(module: Any, attribute: str, fired: Dict[str, int]):
    """Run the real callable, THEN raise.

    Raising INSTEAD of the analysis would not prove the boundary was crossed —
    it would prove only that something failed early. Executing first and then
    failing places the fault unambiguously after the claim.
    """
    original = getattr(module, attribute)

    def faulting(*args, **kwargs):
        original(*args, **kwargs)
        fired["n"] = fired.get("n", 0) + 1
        raise InjectedExecutionFault(f"injected after {attribute} ran")

    setattr(module, attribute, faulting)
    try:
        yield
    finally:
        setattr(module, attribute, original)


def _targets() -> List[Dict[str, Any]]:
    """Each claimed route, with a fault site inside its own execution."""
    from mi_agent_api import chat_routing
    from mi_agent_api import period_change_route as pcr

    return [
        {"route": "period_change_analysis",
         "question": ("Which two geographic region obligors added the most "
                      "balance since last month?"),
         "module": pcr, "attribute": "movement_receipt_for"},
        {"route": "period_movement",
         "question": "What has changed since last month?",
         "module": chat_routing._plan, "attribute": "period_movement"},
    ]


def run() -> Dict[str, Any]:
    import logging
    warnings.simplefilter("ignore")
    logging.disable(logging.ERROR)

    from migration_phase0.compound_canary import _write_run
    from migration_phase0.route_ownership_period_change import funded_runs

    tmp = Path(tempfile.mkdtemp())
    root = tmp / "onboarding_output"
    for run_id, reporting_date, rows, scale in funded_runs(2):
        _write_run(root, run_id, reporting_date, rows, scale)
    saved = {k: os.environ.get(k) for k in
             ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_AUTH_ENABLED")}
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(root)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    try:
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        client = TestClient(app)

        def ask(question):
            return client.post("/mi/query", json={
                "question": question, "portfolioId": "client_001/mi_2026_06",
                "asOfDate": "2026-06-30"}).json()

        rows: List[Dict[str, Any]] = []
        for target in _targets():
            baseline = ask(target["question"])
            claimed = (baseline.get("metadata") or {}).get("route")
            if claimed != target["route"] or not baseline.get("ok"):
                raise DetectorError(
                    f"DETECTOR INVALID — {target['route']} did not deliver its "
                    f"baseline (route={claimed!r} ok={baseline.get('ok')!r}); a "
                    f"fault control over a refusal proves nothing")
            fired: Dict[str, int] = {}
            with _fault_after(target["module"], target["attribute"], fired):
                with _watching_handlers() as entered:
                    faulted = ask(target["question"])
            meta = faulted.get("metadata") or {}
            rows.append({
                "claimed_route": claimed,
                "question": target["question"],
                "fault_site": f"{target['module'].__name__}.{target['attribute']}",
                "fault_executed": int(fired.get("n", 0)),
                "handlers_entered": list(entered),
                "alternate_executions": [n for n in entered if n != claimed],
                "final_route": meta.get("route"),
                # THE BOUNDARY IS DERIVED FROM THIS RUN, NOT FROM THE ANSWER.
                #
                # The first cut of this detector read `metadata.claimBoundary
                # Crossed` — a flag only the FIX publishes. Against the defective
                # tree it was absent, so the invariant read False and the
                # detector printed "FAILS CLOSED · SUBSTITUTIONS 0" over a run in
                # which `temporal_compare` had visibly answered after
                # `period_change_analysis` failed, and exited 0. A control whose
                # signal only exists once the defect is fixed cannot detect the
                # defect.
                #
                # These two facts are collected by the detector itself and are
                # true in either tree: the fault site inside the claimed route
                # executed, and that route's handler was entered.
                "claim_boundary_crossed": bool(fired.get("n", 0)
                                               and claimed in entered),
                # Kept as PUBLISHED evidence, deliberately not as the signal.
                "published_boundary_flag": bool(meta.get("claimBoundaryCrossed")),
                "execution_failure": bool(meta.get("executionFailure")),
                "ok": faulted.get("ok"),
            })
        # A restore check: the registry and the fault sites must be back.
        restored = ask(_targets()[0]["question"])
        if not restored.get("ok"):
            raise DetectorError("DETECTOR INVALID — the baseline did not restore "
                                "after fault injection")
        if not rows:
            raise DetectorError("DETECTOR INVALID — no target was measured")
        return {"rows": rows}
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def substituted(row: Dict[str, Any]) -> bool:
    """The invariant. Narrow by design — see the module docstring."""
    return bool(row["claim_boundary_crossed"]
                and row["claimed_route"] != row["final_route"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    result = run()
    print("=" * 88)
    print("ROUTE SUBSTITUTION DETECTOR — controlled post-claim execution faults")
    print("=" * 88)
    breaches = 0
    for row in result["rows"]:
        bad = substituted(row)
        breaches += bad
        print(f"\n[{'SUBSTITUTED' if bad else 'FAILS CLOSED'}] "
              f"{row['claimed_route']}")
        print(f"    question           : {row['question'][:66]}")
        print(f"    fault site         : {row['fault_site']}")
        print(f"    fault executed     : {row['fault_executed']}")
        print(f"    handlers entered   : {row['handlers_entered']}")
        print(f"    alternate executed : {len(row['alternate_executions'])} "
              f"{row['alternate_executions']}")
        print(f"    final route        : {row['final_route']}")
        print(f"    boundary crossed   : {row['claim_boundary_crossed']} "
              f"(measured; published flag={row['published_boundary_flag']})")
        print(f"    execution failure  : {row['execution_failure']}")
    print(f"\nSUBSTITUTIONS: {breaches} of {len(result['rows'])}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str),
                             encoding="utf-8")
    return 1 if breaches else 0


if __name__ == "__main__":
    raise SystemExit(main())
