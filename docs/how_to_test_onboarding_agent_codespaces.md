# How to test the Trakt Onboarding Agent in GitHub Codespaces

A simple, step-by-step guide for an operator or founder who wants to check that
the Onboarding Agent works. You do **not** need to be a developer. Just copy each
command, paste it into the Codespaces terminal, and press Enter.

---

## What you are testing

The **Onboarding Agent** takes a client's onboarding pack (a folder of files) and:

1. reads the files and works out what data is inside;
2. maps the client's columns to Trakt's standard ("canonical") data model;
3. asks questions where it cannot decide on its own (gaps);
4. lets you answer those questions;
5. builds a **central lender tape** (one clean row per funded loan);
6. builds a **central pipeline tape** (one row per application in the pipeline);
7. records **lineage** (where every value came from — so it is auditable);
8. saves **client memory** (so next time it does not re-ask the same questions);
9. writes an **Azure-ready trigger** file that the rest of Trakt can pick up.

This test runs the whole thing on safe, made-up data and checks the results.

## What the test uses

The test uses synthetic (fake) files in:

```
synthetic_onboarding_pack_domain_based/scenario_a_combined/
```

These simulate a real client pack containing:

- a **master loan / collateral tape** (loans + property in one file),
- a **cashflow report**,
- a **pipeline report** (applications, some not yet funded),
- a **warehouse funding agreement** (a document, not a spreadsheet).

No real client data is involved.

---

## Step 1 — Open Codespaces

1. Open the repository in **GitHub Codespaces** (green **Code** button → **Codespaces** → **Create codespace**).
2. Wait for the terminal at the bottom to finish loading.
3. Make sure you are in the repo root:

```bash
pwd
ls
```

You should see folders like:

```
engine/
config/
tests/
synthetic_onboarding_pack_domain_based/
```

If you see those, you are in the right place.

## Step 2 — Install / check dependencies

Install the Python libraries the project uses:

```bash
pip install -r requirements.txt
```

If you later see a "module not found" message for the workbench or charts, install
these as well (harmless if already installed):

```bash
pip install streamlit rapidfuzz plotly
```

That is all the setup you need.

## Step 3 — Run the synthetic onboarding demo

```bash
python -m engine.onboarding_agent.demo_onboarding_v1 \
  --output-dir onboarding_output/demo_onboarding_v1 \
  --client-id demo_client \
  --run-id demo_run_001
```

**What this does (plain English):** it runs the full onboarding story end to end —
reads the sample client files, creates questions, applies a set of demo answers,
saves client memory, builds the consolidated tapes, and writes the pipeline
trigger.

**What you should see** is a summary like this:

```
Demo completed.
Input files: 4
Domains detected: borrower, cashflow, collateral, loan, pipeline
Central lender tape rows: 8
Central pipeline rows: 4
Blocking gaps before answers: 8
Blocking gaps after answers: 0
Unresolved mapping gaps before memory: 10
Unresolved mapping gaps after memory: 7
Client memory entries saved: 4
Pipeline trigger written: .../output/manifests/23_pipeline_trigger.json
Readiness: ready_for_pipeline
Ready for MI: yes
Ready for regulatory projection: yes
```

The key things to notice: **blocking gaps went from 8 to 0** (the answers closed
them), and **readiness is `ready_for_pipeline`**.

## Step 4 — Check that output files were created

```bash
find onboarding_output/demo_onboarding_v1 -maxdepth 5 -type f | sort
```

The most important files are:

| File | What it is |
| --- | --- |
| `output/central/18_central_lender_tape.csv` | Consolidated funded-loan tape |
| `output/central/18a_central_pipeline_tape.csv` | Consolidated pipeline / application tape |
| `output/lineage/18b_central_tape_lineage.csv` | Where each value came from |
| `output/gaps/18c_central_tape_gaps.csv` | Any unresolved issues / questions |
| `output/manifests/21_pipeline_handoff_readiness.json` | Readiness result |
| `output/manifests/23_pipeline_trigger.json` | Dry-run trigger for the downstream pipeline |

## Step 5 — View the central lender tape

```bash
python - <<'PY'
import pandas as pd
p = "onboarding_output/demo_onboarding_v1/output/central/18_central_lender_tape.csv"
df = pd.read_csv(p)
print("shape:", df.shape)
print(df.head(10).to_string(index=False))
PY
```

You should see **one row per funded / live loan** (loans L0001–L0008).

## Step 6 — View the central pipeline tape

```bash
python - <<'PY'
import pandas as pd
p = "onboarding_output/demo_onboarding_v1/output/central/18a_central_pipeline_tape.csv"
df = pd.read_csv(p)
print("shape:", df.shape)
print(df.head(10).to_string(index=False))
PY
```

You should see **application / pipeline rows**. Some are linked to a funded loan
(the `linked_to_central_lender_tape` column is `True`); some are application-only
(`False`).

## Step 7 — Check lineage

```bash
python - <<'PY'
import pandas as pd
p = "onboarding_output/demo_onboarding_v1/output/lineage/18b_central_tape_lineage.csv"
df = pd.read_csv(p)
print("shape:", df.shape)
print("source files:", sorted(df["source_file"].dropna().unique()))
print(df.head(10).to_string(index=False))
PY
```

Lineage proves **where each value came from** (which file and column). This is
what makes the consolidated tape auditable. You should see the synthetic source
files (e.g. `master_loan_collateral_tape.csv`) listed, and `cashflow_report.csv`
appears as a cross-check in the `validation_sources` column.

## Step 8 — Check gaps

```bash
python - <<'PY'
import pandas as pd
p = "onboarding_output/demo_onboarding_v1/output/gaps/18c_central_tape_gaps.csv"
df = pd.read_csv(p)
print("shape:", df.shape)
print(df.to_string(index=False))
PY
```

Gaps are unresolved issues. In the demo, the major gaps are closed by the demo
answers, so you should see only a small number of low-severity items (for example
a single "missing required field" note), not blocking problems.

## Step 9 — Check readiness and pipeline trigger

```bash
python - <<'PY'
import json
from pathlib import Path
for p in [
    "onboarding_output/demo_onboarding_v1/output/manifests/21_pipeline_handoff_readiness.json",
    "onboarding_output/demo_onboarding_v1/output/manifests/23_pipeline_trigger.json",
]:
    print("\n---", p, "---")
    data = json.loads(Path(p).read_text())
    print(json.dumps(data, indent=2)[:3000])
PY
```

**Readiness** tells you whether the run is ready for MI, regulatory projection,
warehouse analysis, etc. The **trigger JSON** is the file a future Azure workflow
would use to start the next stage of the Trakt pipeline. In this test nothing is
uploaded anywhere — the trigger is written locally only (a "dry run").

## Step 10 — Check client memory

```bash
find onboarding_output/demo_client/client_memory -maxdepth 3 -type f -print -exec sed -n '1,120p' {} \;
```

Client memory stores **approved decisions** so future runs for the same client do
not ask the same questions again. It is **client-specific** — it never becomes a
global rule that affects other clients.

## Step 11 — Run the workbench (visual review tool)

```bash
python -m streamlit run engine/onboarding_agent/streamlit_onboarding_workbench.py
```

- Codespaces will pop up a notice offering to **open the forwarded port** — click it
  (or open the **Ports** tab and open the Streamlit port, usually `8501`).
- In the workbench **sidebar**, enter:

```
project_dir = onboarding_output/demo_onboarding_v1
client_id   = demo_client
run_id      = demo_run_001
mode        = regulatory_mi
```

Then click **Load**. Have a look through the tabs:

- **Overview** — status (Ready / Needs review / Blocked), counts.
- **Domains** — which data domains are covered.
- **Mappings** — how each column was mapped, and anything needing review.
- **Gaps** — the questions, grouped by severity.
- **Conflicts** — where two sources disagree.
- **Source precedence** — which source "wins" a conflict.
- **Enums** — odd category values (e.g. `employment_status = manual`).
- **Client memory** — saved decisions.
- **Readiness & artefacts** — the final readiness and file list.

Press `Ctrl+C` in the terminal to stop the workbench when you are done.

## Step 12 — Run the targeted tests

```bash
pytest tests/test_onboarding_demo_v1.py \
  tests/test_onboarding_workbench_smoke.py \
  tests/test_onboarding_mapping_memory.py \
  -q --tb=short
```

These check that the demo, the workbench support functions, and client memory are
all working. You should see something like `22 passed`.

## Step 13 — Clean up (optional)

```bash
rm -rf onboarding_output/demo_onboarding_v1 onboarding_output/demo_client
```

This removes only the generated demo outputs. It does **not** delete any source
code or the synthetic input files.

---

## What success looks like

- [ ] The demo command completes with a summary.
- [ ] The central lender tape exists and has rows.
- [ ] The pipeline tape exists and has rows.
- [ ] The lineage file exists and references the synthetic source files.
- [ ] The gaps file exists (and blocking gaps are closed).
- [ ] The readiness JSON exists.
- [ ] The pipeline trigger JSON exists.
- [ ] The client memory files exist.
- [ ] The workbench opens and loads the run.
- [ ] The tests pass.

## Common issues and simple fixes

1. **"ModuleNotFoundError" when running a command**
   ```bash
   export PYTHONPATH=.
   ```
   Then run the command again.

2. **"streamlit: command not found" / Streamlit missing**
   ```bash
   pip install streamlit
   ```

3. **A chart or fuzzy-match library is missing**
   ```bash
   pip install rapidfuzz plotly
   ```

4. **"file not found" when viewing an output file**
   Make sure you ran the demo command in **Step 3** first — the output files are
   only created after the demo runs.

5. **The Codespaces browser window for the workbench does not open**
   Open the **Ports** tab at the bottom of Codespaces and click the globe icon
   next to the Streamlit port to open it in a browser.

---

## Optional — one-shot verification script

Run this any time after the demo to confirm everything is in place:

```bash
python - <<'PY'
import json
import pandas as pd
from pathlib import Path
root = Path("onboarding_output/demo_onboarding_v1")
checks = [
    root / "output/central/18_central_lender_tape.csv",
    root / "output/central/18a_central_pipeline_tape.csv",
    root / "output/lineage/18b_central_tape_lineage.csv",
    root / "output/gaps/18c_central_tape_gaps.csv",
    root / "output/manifests/21_pipeline_handoff_readiness.json",
    root / "output/manifests/23_pipeline_trigger.json",
]
print("Checking demo outputs...")
for p in checks:
    print(("FOUND " if p.exists() else "MISSING ") + str(p))
if (root / "output/central/18_central_lender_tape.csv").exists():
    df = pd.read_csv(root / "output/central/18_central_lender_tape.csv")
    print("Central lender tape rows:", len(df))
if (root / "output/central/18a_central_pipeline_tape.csv").exists():
    df = pd.read_csv(root / "output/central/18a_central_pipeline_tape.csv")
    print("Central pipeline tape rows:", len(df))
p = root / "output/manifests/23_pipeline_trigger.json"
if p.exists():
    data = json.loads(p.read_text())
    print("Trigger status:", data.get("status"))
    print("Ready for MI:", data.get("ready_for_mi_agent"))
PY
```

A healthy run prints `FOUND` for every file, non-zero row counts, and
`Trigger status: ready_for_pipeline`.

---

> **Note:** This is a test of the v1 Onboarding Agent on synthetic data. It does
> not upload anything to Azure and does not run the downstream Gates 1–5. Monthly
> drift detection and the live Azure wrapper are intentionally not part of this
> version. For the deeper background, see `docs/onboarding_v1_demo.md`.
