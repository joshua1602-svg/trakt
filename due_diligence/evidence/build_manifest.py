"""Emit the evidence MANIFEST for the analytical intent / forecast hardening packs.

Two provenance rules this builder enforces, both learned from a real error:

  1. HARNESS HASH AND RUN-FILE HASH MOVE TOGETHER. A run file records which
     harness revision produced it. A manifest that hashes the current harness
     beside run files an earlier revision produced is a provenance error, not a
     work-in-progress state. Where they legitimately diverge — because the
     harness has since improved and re-measurement is deliberately deferred —
     the divergence is RECORDED with its reason, never left implicit.

  2. THE COMPARATOR IS PINNED TOO. A before/after claim whose "before" is not
     hashed is unverifiable by a third party. The baseline run files, the
     harness that produced them and the bank are all in the manifest.
"""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

REPO = Path("/home/user/trakt")
EV = REPO / "due_diligence/evidence"
V1 = EV / "analytical_intent_v1"
FCH = EV / "forecast_composition_hardening"
S = Path("/tmp/claude-0/-home-user-trakt/8fa44461-4d82-5a00-b9a3-1df72087ab19/scratchpad")

#: The tree the V1 run files were produced at. Established from run-file mtimes
#: (22:23:35-22:49:40) against the commit log: last commit before the window was
#: 044d13b (22:15:31); production code last changed by 9125e77 (22:14:54); every
#: later commit touched only due_diligence/ (git diff --name-only 044d13b..HEAD).
V1_MEASURED_AT_TREE = "044d13b"
V1_PRODUCTION_CODE_AT = "9125e77"
#: sha256 of the harness revision that actually produced BOTH the baseline and
#: the V1 run files. Recovered from `git show 5f5d697:...nl_harness.py`.
HARNESS_THAT_PRODUCED_RUNS = (
    "de059ef3fd07357092d8109747d0c46405f4329413cd17e4e9a01ec9477e8a9c")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


entries: list[dict] = []


def add(group: str, path: Path, role: str, base: Path, **extra):
    if not path.exists():
        entries.append({"group": group, "path": path.name, "role": role,
                        "status": "MISSING"}); return
    try:
        rel = str(path.relative_to(base))
    except ValueError:
        rel = path.name
    entries.append({"group": group, "path": rel, "role": role,
                    "bytes": path.stat().st_size, "sha256": sha(path), **extra})


# --- 1. fixture data: input to BOTH the agent and every TRUTH recomputation --
for date in ("2026-04-30", "2026-05-31", "2026-06-30"):
    add("fixture:funded:alderbridge",
        REPO / f"demo_platform/workspace/store/processed/platform/alderbridge/{date}/platform_canonical_typed.csv",
        f"funded loan tape @ {date}", REPO)
    add("fixture:funded:kestrelmoor",
        S / f"kestrelmoor_store/processed/platform/kestrelmoor/{date}/platform_canonical_typed.csv",
        f"funded loan tape @ {date}", S)
for book in ("alderbridge", "kestrelmoor"):
    root = S / "pipeline_root" / book / "pipeline"
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for csv in sorted(d.glob("*.csv")):
            add(f"fixture:pipeline:{book}", csv, f"weekly pipeline extract @ {d.name}", S)

# --- 2. production code + config the answers and traces depend on ------------
for rel, role in (
    ("mi_workflows/analytical/intent.py", "intent classifier"),
    ("mi_workflows/analytical/planner.py", "plan construction"),
    ("mi_workflows/analytical/executors.py", "capability adapters"),
    ("mi_workflows/analytical/narrative.py", "answer composition"),
    ("mi_workflows/analytical/registry.py", "capability declarations"),
    ("mi_workflows/analytical/contract.py", "finding contract"),
    ("mi_agent/seasoning.py", "governed lending windows"),
    ("mi_agent_api/pipeline_prep.py", "pipeline preparation + stage probabilities"),
    ("mi_agent_api/pipeline_history.py", "empirical completion model"),
    ("mi_agent_api/forecast_bridge.py", "funded + pipeline forecast bridge"),
    ("analytics/pipeline_expected_funding.py", "parallel expected-funding implementation"),
    ("config/mi/buckets.yaml", "seasoning + lending-window thresholds"),
    ("config/client/pipeline_expected_funding.yaml", "stage probabilities, lags, stage inclusion"),
    ("config/business_semantics_registry.yaml", "aggregation / weighting / directionality"),
    ("mi_agent/mi_semantics_field_registry.yaml", "field roles and formats"),
):
    add("code+config", REPO / rel, role, REPO)

# --- 3. the FROZEN semantic expectations (authored before any code change) ---
add("expectation", FCH / "frozen_expectations.yaml",
    "semantic expectations for all 44 variations — frozen, never edited in-sprint", EV)

# --- 4. measurement machinery -------------------------------------------------
for name, role in (
    ("nl_bank.py", "the 44-variation bank"),
    ("nl_score.py", "FROZEN scorer — scores baseline and V1 alike"),
    ("nl_reconcile.py", "numeric TRUTH recomputation"),
    ("v1_final.py", "distribution + baseline comparison + gate"),
    ("v1_trace.py", "the ten traces"),
    ("v1_score_run.py", "per-arm scoring"),
    ("truth_pipeline.py", "TRUTH for the pipeline conversion forecast"),
    ("classify_substance.py", "mechanical substance partition"),
    ("check_semantics.py", "semantic expectation comparison"),
):
    add("measurement", V1 / name, role, EV)
add("measurement", V1 / "nl_harness.py",
    "measurement harness AS IT NOW STANDS — records analyticalIntent and the "
    "built plan at run time. This revision has NOT yet produced a run file; "
    "see manifestNotes.harnessDivergence.", EV)

# --- 5. run files, each naming the harness that produced it ------------------
for name in ("v1_nl_alderbridge_production.json", "v1_nl_alderbridge_forced_llm.json",
             "v1_nl_kestrelmoor_production.json", "v1_nl_kestrelmoor_forced_llm.json"):
    add("run-file:v1", S / name, "V1 measurement output (uncompressed sha256)", S,
        producedByHarnessSha256=HARNESS_THAT_PRODUCED_RUNS,
        producedAtTree=V1_MEASURED_AT_TREE)

# --- 6. THE PINNED COMPARATOR ------------------------------------------------
for name in ("nl_alderbridge_production.json", "nl_alderbridge_forced_llm.json",
             "nl_kestrelmoor_production.json", "nl_kestrelmoor_forced_llm.json"):
    add("run-file:baseline", S / name,
        "BASELINE measurement output at 104c89d (uncompressed sha256)", S,
        producedByHarnessSha256=HARNESS_THAT_PRODUCED_RUNS,
        producedAtTree="104c89d")
add("baseline", V1 / "baseline/nl_harness_that_produced_runs.py",
    "the harness revision that produced BOTH the baseline and the V1 run files", EV)

manifest = {
    "packs": ["analytical_intent_v1", "forecast_composition_hardening"],
    "reportHead": git("rev-parse", "--short", "HEAD"),
    "generatedFromTree": git("rev-parse", "HEAD"),
    "v1RunFilesProducedAtTree": V1_MEASURED_AT_TREE,
    "v1ProductionCodeAt": V1_PRODUCTION_CODE_AT,
    "manifestNotes": {
        "standingRule": (
            "Harness hash and run-file hash move together. A manifest that "
            "hashes a harness revision beside run files produced by a different "
            "revision is a provenance error. Where they diverge the divergence "
            "is recorded with its reason, as below."),
        "harnessDivergence": (
            "The four V1 run files and the four baseline run files were ALL "
            "produced by harness sha256 " + HARNESS_THAT_PRODUCED_RUNS[:16] + "… "
            "(6,285 bytes), pinned at evidence/analytical_intent_v1/baseline/"
            "nl_harness_that_produced_runs.py. The harness under "
            "evidence/analytical_intent_v1/nl_harness.py has since been changed "
            "to record metadata.analyticalIntent and the built plan at run time. "
            "That revision has produced NO run file yet: re-measurement is "
            "deliberately deferred so a single re-run covers both the harness "
            "change and any answer change from this sprint. At that point both "
            "hashes move together and this note is removed."),
        "arithmeticVsSemantic": (
            "Reconciled-findings counts prove arithmetic fidelity against the "
            "population the agent EXECUTED. They do not prove that population "
            "was the semantically correct one. The frozen expectation file is "
            "the separate control for that."),
        "v1ProvenanceCorrection": (
            "Earlier V1 documentation cited 8c9d04e as the commit the runs were "
            "produced at. The correct tree is " + V1_MEASURED_AT_TREE + " with "
            "production code at " + V1_PRODUCTION_CODE_AT + "; every later commit "
            "touched only due_diligence/."),
    },
    "artefacts": entries,
}
out = EV / "MANIFEST.json"
out.write_text(json.dumps(manifest, indent=2) + "\n")
missing = [e for e in entries if e.get("status") == "MISSING"]
print(f"wrote {out.relative_to(REPO)} — {len(entries)} artefacts, {len(missing)} missing")
for e in missing:
    print("   MISSING:", e["path"])
import collections
for g, n in sorted(collections.Counter(e["group"] for e in entries).items()):
    print(f"   {g:32s} {n}")
