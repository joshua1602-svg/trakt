# Re-running the production bank

## What this run is for

1. **The forecast definitions (vocabulary 2.3.0).** The 2026-09-28 run showed
   the model putting different questions into two forecast measures because it
   was shown a name and no definition. Those two shapes are HELD on the governed
   path. This run shows whether, with definitions, only the questions that mean
   those figures still land there — which is what lifting the hold needs.
2. **Everything built since.** The first live measurement of the forecast
   runtime (milestones, the forecast funded balance), the pipeline at named
   months (D7), and stage movement.

The run asks the live model once per question, as the last one did, so the
Anthropic credit must cover ~135 interpretations.

## Steps

**1. Deploy the branch.** GitHub → Actions → *Deploy trakt-mi-api (App
Service)* → *Run workflow* → branch `claude/wizardly-faraday-o712ib` → *Run*.
Wait for it to go green.

**2. Confirm what is deployed.** Open `https://app.traktinfra.io/api/` — the
`build.commit` it reports must be the branch head (the commit that added this
file or later).

**3. Switch the canary on.** Azure portal → App Service `trakt-mi-api` →
*Environment variables*:

| setting | value |
|---|---|
| `MI_AGENT_PLAN_SERVE` | `canary` |
| `MI_AGENT_PLAN_SERVE_PRINCIPALS` | your Entra object id (as for the last run) |
| `MI_AGENT_PLAN_SHADOW_EVIDENCE` | `/home/LogFiles/mi-plan-shadow/evidence.jsonl` |

*Apply*; the app restarts. Only the listed principal gets governed answers;
everyone else is served exactly as before.

**4. Run the bank.** Azure portal → `trakt-mi-api` → *SSH* → *Go*, then:

    for d in $(ls -td /tmp/*/); do [ -f "$d/mi_agent_api/app.py" ] && cd "$d" && break; done
    bash mi_agent_api/run_production_bank.sh <your Entra object id>

`mi_agent_api/run_production_bank.sh` asks the same 135 questions as
`qb_full.txt` (the six categories that select exactly `qb_questions.json`; the
runner's default adds the limits categories, 160 questions). It refuses, before
asking anything, when the deployed vocabulary is older than 2.6.0, when the canary is
not on for that principal, or when the evidence sink is unset. The run is under
`nohup`: Ctrl-C or closing the tab stops the watching, not the run. The log,
`/home/qb_plan_rerun_<stamp>.log`, starts with the START TIME (UTC) and is what
to hand back; the `.jsonl` beside it holds the full answers.

**Spot check first (catalogue batch 1 and D12).** Asks only the questions the
new pipeline concepts are for — 16 model interpretations, not 135:

    bash mi_agent_api/run_production_bank.sh <your Entra object id> pipeline_003,pipeline_007,pipeline_008,pipeline_009,pipeline_010,pipeline_012,pipeline_013,pipeline_016,pipeline_018,pipeline_019,pipeline_020,pipeline_021,pipeline_strat_001,pipeline_strat_002,pipeline_strat_003,pipeline_evolution_008

Each answer should be a figure the Pipeline tab shows for the same extract
(weighted pipeline, expected completion month, overdue / next month, broker,
product, LTV band, reporting region).

**5. Switch the canary off.** `MI_AGENT_PLAN_SERVE` = `off`.

**6. Hand back the output and the start time.** The readback is then:

- set `since` in `READBACK_REQUEST.json` to the start time and commit it — that
  runs `qb-plan-readback.yml` once (read only: no question, no model, nothing
  written to the service) and publishes the recorded plans as an artifact;
- `python due_diligence/evidence/qb_plan_readback/compare_runs.py --rerun
  qb_plan_readback.json` compares every question's reading with the baseline,
  under today's compiler and perimeter.

## What decides the hold

`compare_runs.py` prints, for each held shape, the bank numbers that land in it.
On the baseline:

    forecast_projection/forecast_funded_balance   [57, 76, 77, 87, 90, 91, 92, 94, 95, 114, 117, 127]
    point_in_time/forecast_completion_rate        [98, 112, 113, 121, 125, 126, 134]

Of those, the questions that MEAN the figure are [87] (the expected funded
balance) and [112], [113] (the run-rate). The rest are refused for another
reason already (a pipeline base, an axis, a geography, a grain) or were
misreads: [76], [77], [94], [114], [117], [127] and [98], [121]. A shape can be
released when the misreads have left it. The release is a code change pinned by
`tests/interpretation_v2/test_production_bank_perimeter.py`, with the re-run's
intents added beside the baseline's.

## Run log

    2026-09-28  baseline run (vocabulary 2.2.0), main build; canary left on
    2026-09-29  06:58:59 UTC re-run on ce5d1276 (vocabulary 2.3.0), 135/135;
                read back by run 36547210324; readings in
                qb_rerun_intents_20260929.json
    2026-09-29  09:47 UTC d74ae632 deployed (live pipeline D8, weighted-average
                valuation D10); MI_AGENT_PLAN_SERVE set to off — operator
                confirmed, build.commit d74ae632 confirmed by the operator
