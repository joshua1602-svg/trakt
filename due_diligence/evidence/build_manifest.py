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

# --- 7. FORECAST & COMPOSITION HARDENING -------------------------------------
for name, role in (
    ("three_axis.py", "SEMANTIC / ARITHMETIC / PRESENTATION scorer against the "
                      "frozen expectations; arithmetic delegated to nl_reconcile.py"),
    ("ab_summary.py", "like-for-like A/B over two code revisions, frozen scorer"),
    ("censoring.py", "TRANCHE C investigation — censoring and maturity of the "
                     "empirical stage rates. No serving path imports it."),
    ("classify_substance_guarded.py",
     "substance partition, identical rule to analytical_intent_v1/"
     "classify_substance.py, guarded so it can be imported"),
    ("tb_prep.py", "prep-layer report, post-Tranche-B"),
    ("tb_prep_before.py",
     "prep-layer report with the PRE-Tranche-B exclusion tier reproduced "
     "in-process; no production file edited"),
    ("neutrality2.py", "A5 numerical-neutrality proof for Tranche A"),
    ("neutrality_probe.py", "A5 support"),
):
    add("measurement:fch", FCH / name, role, EV)

add("measurement:fch", FCH / "nl_harness_det.py",
    "DETERMINISTIC bank harness — a separate instrument, LLM parser off, so the "
    "parse is a pure function of the question and the parser mix is identical "
    "across revisions by construction. Produced the det_pre / det_post runs.", EV)
HARNESS_DET = sha(FCH / "nl_harness_det.py") if (FCH / "nl_harness_det.py").exists() else None
HARNESS_CURRENT = sha(V1 / "nl_harness.py") if (V1 / "nl_harness.py").exists() else None

#: The like-for-like A/B. Stored gzipped; the sha256 is of the UNCOMPRESSED
#: bytes, matching the V1 convention.
import gzip
for tree, label in (("det_pre", "pre-sprint 49e00b5"), ("det_post", "post-Tranche-B")):
    for name in ("nl_alderbridge_production.json", "nl_alderbridge_forced_llm.json",
                 "nl_kestrelmoor_production.json", "nl_kestrelmoor_forced_llm.json"):
        gz = FCH / "tranche_b" / tree / (name + ".gz")
        if not gz.exists():
            entries.append({"group": f"run-file:{tree}", "path": name,
                            "role": "MISSING", "status": "MISSING"})
            continue
        raw = gzip.decompress(gz.read_bytes())
        entries.append({
            "group": f"run-file:{tree}",
            "path": str(gz.relative_to(EV)),
            "role": f"752-run bank, deterministic parse, {label} (uncompressed sha256)",
            "bytes": gz.stat().st_size,
            "uncompressedBytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "gzSha256": sha(gz),
            "producedByHarnessSha256": HARNESS_DET,
            "producedAtTree": "49e00b5" if tree == "det_pre" else "6f61365"})

#: The LLM arm, run on an exhausted API key. Published because it is real
#: measurement; NOT comparable to the V1 baseline, because its parser mix differs.
for tree, label in (("tb_pre", "pre-sprint 49e00b5"), ("tb", "post-Tranche-B")):
    for name in ("nl_alderbridge_production.json", "nl_alderbridge_forced_llm.json",
                 "nl_kestrelmoor_production.json", "nl_kestrelmoor_forced_llm.json"):
        gz = FCH / "tranche_b" / "llm_arm_degraded" / tree / (name + ".gz")
        if not gz.exists():
            continue
        raw = gzip.decompress(gz.read_bytes())
        entries.append({
            "group": f"run-file:llm-degraded:{tree}",
            "path": str(gz.relative_to(EV)),
            "role": (f"752-run bank, LLM arm on an EXHAUSTED key, {label} "
                     "(uncompressed sha256). See report §7.1 — not comparable "
                     "to the V1 baseline."),
            "bytes": gz.stat().st_size,
            "uncompressedBytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "gzSha256": sha(gz),
            "producedByHarnessSha256": HARNESS_CURRENT,
            "producedAtTree": "49e00b5" if tree == "tb_pre" else "6f61365"})

for rel, role in (
    ("tranche_b/ab_deterministic.txt", "the like-for-like A/B result"),
    ("tranche_b/ab_llm_arm_degraded.txt", "the degraded LLM arm, with every differing run"),
    ("tranche_b/three_axis_det_pre.txt", "three-axis score, pre-sprint"),
    ("tranche_b/three_axis_det_post.txt", "three-axis score, post-Tranche-B"),
    ("tranche_b/reconcile_det_pre.txt", "independent reconciliation, pre-sprint"),
    ("tranche_b/reconcile_det_post.txt", "independent reconciliation, post-Tranche-B"),
    ("tranche_b/truth_pipeline_output.txt", "independent pipeline recomputation output"),
    ("tranche_b/prep_before.txt", "prep-layer report, pre-Tranche-B tier"),
    ("tranche_b/prep_after.txt", "prep-layer report, post-Tranche-B"),
    ("tranche_b/simple_mi_bank_30.json", "30-question simple-MI bank, post-Tranche-B"),
    ("tranche_b/wide_bank_80.json", "80-question wide bank, post-Tranche-B"),
    ("tranche_b/cfo9_alderbridge.json", "nine canonical CFO questions, Alderbridge"),
    ("tranche_b/cfo9_kestrelmoor.json", "nine canonical CFO questions, Kestrelmoor"),
    ("tranche_b/q9_findings_alderbridge.json", "Q9 findings, Alderbridge"),
    ("tranche_b/q9_findings_kestrelmoor.json", "Q9 findings, Kestrelmoor"),
    ("tranche_a_neutrality.txt", "Tranche A numerical-neutrality proof output"),
    ("tranche_c_censoring.txt", "Tranche C censoring / maturity output"),
):
    add("output:fch", (FCH / rel).resolve(), role, EV.resolve())
add("output:fch", REPO / "due_diligence/MI_AGENT_FORECAST_COMPOSITION_HARDENING.md",
    "the report", REPO)

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
            "RESOLVED IN PART. The four V1 run files and the four baseline run "
            "files were ALL produced by harness sha256 "
            + HARNESS_THAT_PRODUCED_RUNS[:16] + "… (6,285 bytes), pinned at "
            "evidence/analytical_intent_v1/baseline/"
            "nl_harness_that_produced_runs.py. The revision under "
            "evidence/analytical_intent_v1/nl_harness.py, which records "
            "metadata.analyticalIntent at run time, HAS now produced run files: "
            "the run-file:llm-degraded:* groups name it. Those runs are NOT a "
            "replacement for the V1 measurement — the API key was exhausted "
            "mid-sprint and their parser mix differs from the baseline's (145 "
            "LLM parses of 752, against 645), so they are published as evidence "
            "and excluded from any before/after claim. See report section 7.1. "
            "A full-credit remeasurement remains outstanding."),
        "deterministicComparator": (
            "The before/after claims in the Forecast & Composition Hardening "
            "report rest on run-file:det_pre and run-file:det_post, produced by "
            "nl_harness_det.py with the LLM parser off on BOTH sides. That makes "
            "the parse a pure function of the question text and the parser mix "
            "identical by construction (752 of 752 deterministic each side), so "
            "a difference between the two is attributable to the code. It does "
            "not exercise the LLM parse path."),
        "packagedScriptsDiffer": (
            "forecast_composition_hardening/classify_substance_guarded.py has "
            "the same substance() rule as analytical_intent_v1/"
            "classify_substance.py; it differs only in being import-safe and in "
            "carrying an extra run set. nl_score.py is NOT duplicated into this "
            "pack — the frozen control exists once and is imported."),
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
