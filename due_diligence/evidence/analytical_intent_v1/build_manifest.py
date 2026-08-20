"""Emit MANIFEST.json — every artefact the traces depend on, hashed.

Covers the three classes the pack previously left open:
  * the FIXTURE CSVs, which are the input to both the agent and the TRUTH column
  * the intent classifier and the plan/composition config the traces depend on
  * the measurement harness, scorer, reconciler and tracer
"""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

REPO = Path("/home/user/trakt")
S = Path(__file__).resolve().parent.parent
#: The tree state the committed run files were produced at. Established from the
#: run-file mtimes (22:23:35-22:49:40) against the commit log: the last commit
#: before the measurement began was 044d13b (22:15:31); production code was last
#: touched by 9125e77 (22:14:54). Every commit after 044d13b changed only
#: due_diligence/, verified by `git diff --name-only 044d13b..HEAD`.
MEASURED_AT = "044d13b"
PRODUCTION_CODE_AT = "9125e77"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def head() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


entries = []


def add(group: str, path: Path, role: str, base: Path | None = None):
    if not path.exists():
        entries.append({"group": group, "path": str(path), "role": role,
                        "status": "MISSING"}); return
    entries.append({"group": group,
                    "path": str(path.relative_to(base or REPO)) if str(path).startswith(str(base or REPO)) else path.name,
                    "role": role, "bytes": path.stat().st_size, "sha256": sha(path)})


# 1. Fixture data — the input to BOTH the agent and the TRUTH recomputation.
for date in ("2026-04-30", "2026-05-31", "2026-06-30"):
    add("fixture:funded:alderbridge",
        REPO / f"demo_platform/workspace/store/processed/platform/alderbridge/{date}/platform_canonical_typed.csv",
        f"funded loan tape @ {date}")
    add("fixture:funded:kestrelmoor",
        S / f"kestrelmoor_store/processed/platform/kestrelmoor/{date}/platform_canonical_typed.csv",
        f"funded loan tape @ {date}", base=S)
for book in ("alderbridge", "kestrelmoor"):
    root = S / "pipeline_root" / book / "pipeline"
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for csv in sorted(d.glob("*.csv")):
            add(f"fixture:pipeline:{book}", csv, f"weekly pipeline extract @ {d.name}", base=S)

# 2. The code and config the traces depend on.
for rel, role in (
    ("mi_workflows/analytical/intent.py", "the intent classifier — step 3 of every trace"),
    ("mi_workflows/analytical/planner.py", "plan construction"),
    ("mi_workflows/analytical/executors.py", "capability adapters"),
    ("mi_workflows/analytical/registry.py", "capability declarations"),
    ("mi_agent/seasoning.py", "the governed lending windows"),
    ("config/mi/buckets.yaml", "seasoning + lending-window thresholds"),
    ("config/client/pipeline_expected_funding.yaml", "stage probabilities, lags, stage inclusion"),
    ("config/business_semantics_registry.yaml", "aggregation / weighting / directionality"),
    ("mi_agent/mi_semantics_field_registry.yaml", "field roles and formats"),
):
    add("code+config", REPO / rel, role)

# 3. Measurement artefacts.
for name, role in (
    ("nl_bank.py", "the 44-variation bank"),
    ("nl_harness.py", "the measurement harness"),
    ("nl_score.py", "FROZEN scorer — scores baseline and V1 alike"),
    ("nl_reconcile.py", "numeric TRUTH recomputation"),
    ("v1_final.py", "distribution + baseline comparison + gate"),
    ("v1_trace.py", "the ten traces"),
    ("v1_score_run.py", "per-arm scoring"),
):
    add("measurement", S / name, role, base=S)
for name in ("v1_nl_alderbridge_production.json", "v1_nl_alderbridge_forced_llm.json",
             "v1_nl_kestrelmoor_production.json", "v1_nl_kestrelmoor_forced_llm.json"):
    add("run-file", S / name, "752-run measurement output (uncompressed sha256)", base=S)

# 4. Audit additions.
for name, role in (("truth_pipeline.py", "TRUTH for the pipeline conversion forecast"),
                   ("classify_substance.py", "mechanical substance partition")):
    add("audit", S / "audit" / name, role, base=S)

manifest = {
    "pack": "analytical_intent_v1",
    "reportHead": head(),
    "runFilesProducedAtTree": MEASURED_AT,
    "productionCodeLastChangedAt": PRODUCTION_CODE_AT,
    "note": ("Every commit after " + MEASURED_AT + " changed only due_diligence/. "
             "The run files therefore describe the production code at "
             + PRODUCTION_CODE_AT + ", which is identical to that at reportHead."),
    "artefacts": entries,
}
out = REPO / "due_diligence/evidence/analytical_intent_v1/MANIFEST.json"
out.write_text(json.dumps(manifest, indent=2) + "\n")
missing = [e for e in entries if e.get("status") == "MISSING"]
print(f"wrote {out.relative_to(REPO)} — {len(entries)} artefacts, {len(missing)} missing")
for e in missing:
    print("   MISSING:", e["path"])
import collections
for g, n in collections.Counter(e["group"] for e in entries).most_common():
    print(f"   {g:32s} {n}")
