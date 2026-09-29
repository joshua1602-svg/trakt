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
    bash mi_agent_api/run_production_bank.sh <your Entra object id> all

`all` is required for the whole bank: without a second argument the script
refuses and asks nothing, so a command whose question list was lost to a line
break cannot spend the full run. Paste a spot check as ONE line, and check the
preflight says `questions selected: N (spot check)` before leaving it.

`mi_agent_api/run_production_bank.sh` asks the same 135 questions as
`qb_full.txt` (the six categories that select exactly `qb_questions.json`; the
runner's default adds the limits categories, 160 questions). It refuses, before
asking anything, when the deployed vocabulary is older than 2.7.0, when the canary is
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

**Spot check for catalogue batch 2 (the forecast semantic model, vocabulary
2.7.0).** The forecast questions the model named missing concepts for:

    bash mi_agent_api/run_production_bank.sh <your Entra object id> forecast_002,forecast_004,forecast_005,forecast_006,forecast_008,forecast_010,forecast_011,forecast_012,forecast_013,forecast_018,forecast_019,forecast_020,forecast_scale_007,forecast_scale_008,forecast_scale_009,forecast_scale_010,forecast_scale_011,forecast_scale_012,forecast_scale_017,forecast_scale_021,forecast_runoff_002

Each answer should be a figure the Forecast tab shows for the same book (its
parts, loan count, exclusions by reason, by region / LTV band, the curve and
its bands, the annualised run-rate, the milestone table). The readings of
forecast_003, forecast_014, forecast_scale_006/_007/_008/_011 decide whether
the D6 holds can be released.

**Combined spot check (a5604967: batch 1, D12 region, batch 2, funded breadth,
answer wording) — run 2026-09-29 15:14:42 UTC, 45/51 served (design §19).**
Every question those changes are for, 51 model interpretations:

    bash mi_agent_api/run_production_bank.sh <your Entra object id> pipeline_003,pipeline_007,pipeline_008,pipeline_009,pipeline_010,pipeline_012,pipeline_013,pipeline_016,pipeline_018,pipeline_019,pipeline_020,pipeline_021,pipeline_strat_001,pipeline_strat_002,pipeline_strat_003,pipeline_evolution_008,forecast_002,forecast_004,forecast_005,forecast_006,forecast_008,forecast_010,forecast_011,forecast_012,forecast_013,forecast_018,forecast_019,forecast_020,forecast_scale_007,forecast_scale_008,forecast_scale_009,forecast_scale_010,forecast_scale_011,forecast_scale_012,forecast_scale_017,forecast_scale_021,forecast_runoff_002,funded_kpi_014,funded_kpi_015,funded_kpi_016,funded_kpi_020,funded_breakdown_1d_001,funded_breakdown_1d_002,funded_breakdown_1d_003,funded_breakdown_1d_004,funded_breakdown_1d_005,funded_breakdown_1d_006,funded_breakdown_1d_007,funded_breakdown_1d_008,funded_breakdown_1d_023,funded_breakdown_1d_025

What good looks like: each answer is a figure the dashboard shows for the same
book; a breakdown names its measure, grouping and leading groups; every answer
states its as-at date. The funded region questions answer only where the
production book carries the reporting region; otherwise the evidence records
FIELD_NOT_IN_BOOK and the old path answers.

**Follow-up spot check (vocabulary 2.9.0: the exclusions' population, `lapsed`,
the answer standard, region basis) — the one to run next.** The six the
combined check did not serve for a fixable reason, one question per changed
wording, and the three hold readings the combined check did not ask — 13
model interpretations:

    bash mi_agent_api/run_production_bank.sh <your Entra object id> forecast_018,forecast_019,forecast_020,forecast_runoff_002,funded_kpi_001,funded_breakdown_1d_001,pipeline_007,pipeline_008,forecast_006,forecast_scale_021,forecast_003,forecast_014,forecast_scale_006

What good looks like: the three exclusion questions and `lapsed` are served as
the Forecast tab's 'excluded from weighting' figures; money reads £87.1m on
every path; a pipeline breakdown leads with three groups and says how many
there are; every region answer says "by the property's location".

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
    2026-09-29  12:28 UTC 14effc33 deployed (run 36567963153): monthly series
                D7, scale D9 (production stage not set), catalogue batch 1,
                pipeline region D12
    2026-09-29  14:53 UTC 438932ac deployed (run 36585701951; a5604967:
                batch 2 forecast semantic model, funded breadth, answer
                wording; vocabulary 2.8.0)
    2026-09-29  15:10:48 UTC full bank started on 438932ac by mistake (the
                spot check's id list was split onto its own line); stopped by
                the operator after funded_kpi_001..005 (kill, ps confirmed).
                The script now refuses without `all` or an id list.
    2026-09-29  15:14:42 UTC combined spot check on 438932ac (vocabulary
                2.8.0), 51 selected (spot check); log
                /home/qb_plan_rerun_20260929T151442Z.log — read back from
                this START TIME
    2026-09-29  15:14:42 run read back by run 36594305684 (51 records):
                45/51 served NEW (the morning's run: 1/51). Not served:
                forecast_018/019/020 (POPULATION_NOT_FORECAST — read
                correctly, runtime executed forecast only), forecast_runoff_002
                (clarify: 'lapsed' undefined), funded_breakdown_1d_003 and
                _025 (FIELD_NOT_IN_BOOK — correct). No question landed in a
                held shape. Changes: design §19; vocabulary 2.9.0
    2026-09-29  after the 15:14:42 spot check: MI_AGENT_PLAN_SERVE set to
                off — operator confirmed. 19:30 UTC 6df5e712 deployed (run 36619090442)
                (vocabulary 2.9.0: exclusions' population, `lapsed`, the
                answer standard, region basis)
