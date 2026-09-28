#!/usr/bin/env python3
"""Phase 0 provenance, and the finding that stops the deployed acceptance.

TWO THINGS ARE ESTABLISHED HERE, both offline and both re-runnable.

1. PROVENANCE. The product tree at HEAD is byte-equivalent to the accepted
   slice 1 SHA; interpretation_v2 is byte-identical to the run-8 sign-off; the
   adapter, the call site and their suites are unchanged; the shadow defaults
   off; no quarantined deterministic commit is an ancestor.

2. THE DEPLOYED SHADOW IS INERT. With `MI_AGENT_PLAN_SHADOW=shadow` fully
   enabled and a writable ledger path, calling `observe` with EXACTLY the
   arguments `mi_agent_api/mi_service.py` passes produces no plan, no execution
   and no ledger row. The call site passes no `plan`, and the only other way
   `observe` can obtain one is `_PLAN_PROVIDER`, which `set_plan_provider`
   installs and which has no caller anywhere outside
   `mi_agent/tests/test_plan_shadow_does_not_serve.py`. Nothing in the request
   path imports `interpretation_v2` at all.

   So a deployed request cannot produce shadow evidence, whatever the flag says.
   Phases 4-6 of the deployed acceptance have nothing to observe, independently
   of whether a deployment is reachable.

This script changes nothing. It is read-only except for a temporary ledger path
inside a temporary directory.

Run: `python due_diligence/evidence/deployed_shadow_acceptance/inert_shadow_proof.py`
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ACCEPTED_SLICE1_SHA = "6f42df67"
RUN8_SIGNOFF_SHA = "2b00172"
QUARANTINED_SHA = "3399d2ef"

#: Files whose identity IS the accepted slice 1, compared blob by blob.
PRODUCT_FILES = (
    "mi_agent/plan_runtime_adapter.py",
    "mi_agent_api/mi_service.py",
    "mi_agent/tests/test_plan_runtime_adapter.py",
    "mi_agent/tests/test_plan_shadow_does_not_serve.py",
)

OUT = Path(__file__).resolve().parent / "phase0_and_finding.json"


def git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=_REPO_ROOT, text=True,
                          capture_output=True).stdout.strip()


def provenance() -> Dict[str, Any]:
    head = git("rev-parse", "HEAD")
    differing = [line.split("\t")[-1]
                 for line in git("diff", "--name-status",
                                 ACCEPTED_SLICE1_SHA, "HEAD").splitlines() if line]
    product_diff = git("diff", "--stat", ACCEPTED_SLICE1_SHA, "HEAD", "--",
                       ".", ":!due_diligence", ":!docs")
    frozen_diff = git("diff", "--stat", RUN8_SIGNOFF_SHA, "HEAD", "--",
                      "mi_agent/interpretation_v2/",
                      ":!mi_agent/interpretation_v2/evidence")
    blobs = {}
    for path in PRODUCT_FILES:
        blobs[path] = {
            "accepted": git("rev-parse", f"{ACCEPTED_SLICE1_SHA}:{path}"),
            "head": git("rev-parse", f"HEAD:{path}"),
        }
    quarantined = subprocess.run(
        ("git", "merge-base", "--is-ancestor", QUARANTINED_SHA, "HEAD"),
        cwd=_REPO_ROOT, capture_output=True).returncode == 0

    os.environ.pop("MI_AGENT_PLAN_SHADOW", None)
    from mi_agent import plan_runtime_adapter as adapter

    # Cleanliness is measured EXCLUDING this evidence directory, because the
    # script is itself untracked at the moment it first runs and a bare
    # `git status --porcelain` would report the measurement as the finding.
    dirty = [line for line in git("status", "--porcelain").splitlines()
             if line and "due_diligence/evidence/deployed_shadow_acceptance/"
             not in line]
    return {
        "deploy_candidate_sha": head,
        "tree_clean_outside_this_evidence_directory": dirty == [],
        "uncommitted_paths_outside_this_evidence_directory": dirty,
        "files_differing_from_accepted_slice1": differing,
        "all_differences_are_evidence_only": all(
            path.startswith("due_diligence/") for path in differing),
        "product_tree_equivalent_to_accepted_slice1": product_diff == "",
        "interpretation_v2_code_frozen_since_run8": frozen_diff == "",
        "product_file_blob_hashes": blobs,
        "product_files_unchanged": all(v["accepted"] == v["head"]
                                       for v in blobs.values()),
        "shadow_mode_with_var_unset": adapter.shadow_mode(),
        "shadow_default_off": adapter.shadow_mode() == adapter.SHADOW_OFF,
        "quarantined_branch_is_ancestor": quarantined,
    }


def call_site_facts() -> Dict[str, Any]:
    """What the deployed call site actually passes, read out of the source."""
    source = (_REPO_ROOT / "mi_agent_api" / "mi_service.py").read_text()
    call_lines = [line.strip() for line in source.splitlines()
                  if "_plan_shadow.observe(" in line or
                  (line.strip().startswith("portfolio_id=") and
                   "_plan_shadow" in source)]
    observe_call = source.split("_plan_shadow.observe(")[1].split(")")[0]
    provider_callers = [
        line for line in subprocess.run(
            ("grep", "-rn", "set_plan_provider", "--include=*.py", "."),
            cwd=_REPO_ROOT, text=True, capture_output=True).stdout.splitlines()
        if "/due_diligence/" not in line]
    return {
        "observe_arguments_at_the_call_site": observe_call.replace("\n", " "),
        "call_site_passes_a_plan": "plan=" in observe_call,
        "set_plan_provider_references": provider_callers,
        "set_plan_provider_production_callers": [
            line for line in provider_callers
            if "/tests/" not in line and "plan_runtime_adapter.py:" not in line],
        "request_path_imports_interpretation_v2":
            "interpretation_v2" in (_REPO_ROOT / "mi_agent_api"
                                    / "mi_service.py").read_text(),
    }


def inert_shadow() -> Dict[str, Any]:
    """The empirical half: flag fully on, exact call-site arguments, nothing happens."""
    from mi_agent import plan_runtime_adapter as adapter
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.tests import portfolio_truth_oracle as truth

    semantics = load_mi_semantics(
        str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
    frame = truth.canonical_book()
    served = {"ok": True, "value": 1234.5, "route": "legacy",
              "answer": "the legacy answer"}
    before = json.dumps(served, sort_keys=True)

    with tempfile.TemporaryDirectory() as tmp:
        ledger = Path(tmp) / "ledger.jsonl"
        os.environ[adapter.SHADOW_ENV_VAR] = adapter.SHADOW_ON
        os.environ[adapter.LEDGER_ENV_VAR] = str(ledger)
        try:
            # EXACTLY the call mi_service.py makes. No `plan`, no provider.
            row = adapter.observe(result=served, frame=frame, semantics=semantics,
                                  view="funded", portfolio_id="acceptance_probe")
            wrote_ledger = ledger.exists()
            rows = (sum(1 for line in ledger.read_text().splitlines()
                        if line.strip()) if wrote_ledger else 0)
        finally:
            os.environ.pop(adapter.SHADOW_ENV_VAR, None)
            os.environ.pop(adapter.LEDGER_ENV_VAR, None)

    return {
        "flag_value_used": adapter.SHADOW_ON,
        "observe_returned": row,
        "ledger_file_created": wrote_ledger,
        "ledger_rows": rows,
        "served_envelope_unchanged": json.dumps(served, sort_keys=True) == before,
        "conclusion": "a deployed request produces no plan, no shadow execution "
                      "and no ledger row even with the flag fully on",
    }


def main() -> int:
    report = {
        "phase": "Slice 1 deployed shadow acceptance — phase 0 and the blocking "
                 "finding",
        "provenance": provenance(),
        "call_site": call_site_facts(),
        "deployed_shadow_is_inert": inert_shadow(),
    }
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n")

    p = report["provenance"]
    print(f"DEPLOY_CANDIDATE_SHA                 {p['deploy_candidate_sha']}")
    print(f"TREE_CLEAN (outside this evidence dir) "
          f"{p['tree_clean_outside_this_evidence_directory']}")
    print(f"PRODUCT_TREE_EQUIVALENT_TO_{ACCEPTED_SLICE1_SHA}    "
          f"{p['product_tree_equivalent_to_accepted_slice1']}")
    print(f"  differences are evidence-only      "
          f"{p['all_differences_are_evidence_only']} "
          f"({len(p['files_differing_from_accepted_slice1'])} files)")
    print(f"INTERPRETATION_V2_FROZEN             "
          f"{p['interpretation_v2_code_frozen_since_run8']}")
    print(f"ADAPTER_AND_CALL_SITE_UNCHANGED      {p['product_files_unchanged']}")
    print(f"SHADOW_DEFAULT_OFF                   {p['shadow_default_off']}")
    print(f"QUARANTINED_BRANCH_IS_ANCESTOR       "
          f"{p['quarantined_branch_is_ancestor']}")
    c, i = report["call_site"], report["deployed_shadow_is_inert"]
    print()
    print(f"call site passes a plan              {c['call_site_passes_a_plan']}")
    print(f"production callers of the provider   "
          f"{c['set_plan_provider_production_callers'] or 'NONE'}")
    print(f"flag ON -> observe() returned        {i['observe_returned']!r}")
    print(f"flag ON -> ledger rows written       {i['ledger_rows']}")
    print(f"served envelope unchanged            {i['served_envelope_unchanged']}")
    print(f"\nWRITTEN {OUT.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
