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

**The full bank, scored against the buckets (vocabulary 2.10.0, design §20) —
the one to run next.** The proof stage's baseline: every question, once.

    bash mi_agent_api/run_production_bank.sh <your Entra object id> all

Read it back as below, then score it against the pass mark (D13):

    python due_diligence/evidence/qb_plan_readback/score_against_buckets.py --rerun qb_plan_readback.json

What good looks like: every MUST ANSWER question served by the new path except
the two that wait on configuration (time to scale, D9: the portfolio's stage)
and the run-rate [112] until the hold is released on this run's evidence; the
conversion rate [49], the pull-through [134] and the assumed KFI-to-completion
rate [121] read into the stage-movement model's own concepts; the pipeline's
change [82], [83] served with both extracts named.

**Follow-up spot check (vocabulary 2.9.0: the exclusions' population, `lapsed`,
the answer standard, region basis) — run 2026-09-29 19:37:49 UTC, 8/13.** The six the
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
    2026-09-29  19:37:49 UTC follow-up spot check on 6df5e712 (vocabulary
                2.9.0), 13 questions; read back by runs 36622570069 and
                36622780280 (the reader now prints why each unserved
                question fell). 8/13 served NEW: the answer standard
                (money, leaders, region basis) confirmed live on funded,
                pipeline and forecast answers; forecast_003 and
                forecast_019 served. Not served: forecast_018 (reason
                not_in [completed, withdrawn] — 'active' pipeline; one
                equality only), forecast_020 (read as pipeline_stage =
                WITHDRAWN; the pipeline runtime takes no stage filter),
                forecast_runoff_002 (clarify: no pipeline concept found for
                'lapsed'), forecast_014 (clarify: 'completion basis' has no
                concept — batch 2b), forecast_scale_006 (held shape
                point_in_time/forecast_completion_rate; reads correctly).
                Hold: [98] has left the shape, [113] reads its own concept;
                [121] and [134] (the conversion-rate misreads) not yet
                re-asked — the hold stays until they are
    2026-09-29  D13 (owner): the question buckets are the pass mark;
                conversion/pull-through and pipeline change must answer.
                Built (design §20): one semantic engine; conversion as the
                stage-movement model; the pipeline's change between two
                dated extracts. Vocabulary 2.10.0. Next: the full bank,
                scored with score_against_buckets.py
    2026-09-29  21:11 UTC b1ef994f deployed (run 36631547906, success; vocabulary
                2.10.0: the semantic engine, conversion, pipeline change)
    2026-09-29  21:21:00 UTC full bank on b1ef994f (vocabulary 2.10.0),
                135 selected with `all`; log
                /home/qb_plan_rerun_20260929T212100Z.log — read back from
                this START TIME and scored against the buckets (D13)

    2026-09-29  the 21:21:00 full bank, read back by runs 36642581333 and
                36642689706 (the reader now prints the interpreter's own
                failure reason). 64/135 served NEW. From [68] pipeline_023
                on, EVERY question (68) failed at the language step with the
                provider's "Your credit balance is too low" (MODEL_UNAVAILABLE)
                — the account ran out of credits mid-run; not a code
                failure. Of the 67 questions asked before that: must answer
                48/48 NEW, nice to have 15/15 NEW, fine to decline 1/4 NEW
                (003 obligor region and 025 occupancy declined FIELD_NOT_IN_
                BOOK, 022 age bucket on the pipeline declined DIMENSION_NOT_
                SUPPORTED). One WRONG answer: [59] "How much pipeline is
                current month?" read 'current month' as the reporting period
                and answered with the whole live pipeline (£969.8m); it is
                this month's completions. Fixed in vocabulary 2.11.0 (the
                expected-completion timing definition). [44] "1 groups" in
                the receipt line: fixed. Not answered: the 68 questions after
                the credits ran out — re-ask exactly those, by id, once
                2.11.0 is deployed (plus [59])
    2026-09-30  MI_AGENT_PLAN_SERVE set to off — operator confirmed. 06:42 UTC
                622351a2 deploying (run 36679674222; vocabulary 2.12.0:
                'current month' pipeline timing (2.11.0), D14 the smallest
                loan has a balance, §21 one model call per question). Next:
                the full bank with `all` — the measurement of the one-call
                interpreter (readback projects model_calls and model_ms)
    2026-09-30  06:48 UTC 622351a2 deployed (run 36679674222, success; the
                first attempt succeeded, no retry). Vocabulary 2.12.0
    2026-09-30  07:10:30 UTC full bank on 622351a2 (vocabulary 2.12.0, one
                model call per question), 135 selected with `all`; log
                /home/qb_plan_rerun_20260930T071030Z.log. 110/135 served NEW
                (last night 64, credits exhausted from [68]; yesterday
                morning 51). After the run MI_AGENT_PLAN_SERVE set to off —
                operator confirmed. Read back from this START TIME
    2026-09-30  the 07:10:30 run read back by run 36689644793: must answer
                83/88, nice to have 21/31, fine to decline 6/16 on the
                governed path; one model call on all 135 (5.5s median model
                time; cost per question 13.1k -> 7.4k). Wrong: [87] expected
                funded balance (weighted pipeline answered), [102] 'active'
                dropped. Fixed in vocabulary 2.13.0 with [82] 'prior' = the
                previous extract; the run-rate hold stays ([125, 126] dropped
                their window). Design §22
    2026-09-30  owner decisions: ERE established, scale £250MM (OCC must
                activate portfolio.stage); governed first; [135] a date from
                the book's history. Built (design §23): governed attempt
                before the legacy parse; expected_completion_date; vocabulary
                2.14.0
    2026-09-30  10:07 UTC b43cd904 deploying (run 36700450236; vocabulary
                2.14.0: 2.13.0's definition fixes, [82] prior = previous
                extract, governed first, scale £250MM, expected completion
                date). ERE's portfolio.stage must be activated in OCC for
                [122, 128]
    2026-09-30  10:10 UTC b43cd904 deployed (run 36700450236, success, no
                retry). Vocabulary 2.14.0
    2026-09-30  owner correction: ERE is a pre-securitisation SPV (scale
                £100MM), not established; £250MM is the established stage's
                threshold. Authored stage reverted; OCC must activate
                portfolio.stage: pre_securitisation_spv for [122, 128]
    2026-09-30  the live OCC (trakt-ops-api, deployed from main 2026-09-28)
                has no Securitisation stage field: it was added on this
                branch (4f0fc542). PR #510 moves ONLY that field to main
                (branch claude/occ-securitisation-stage); merging deploys
                trakt-ops-api. Then: Amend ERE -> New SPV before
                securitisation -> Approve -> Activate
    2026-09-30  10:26:01 UTC 11-question check on b43cd904 (vocabulary
                2.14.0, governed first). [87] expected funded balance now
                £94.1m (fixed); [82] latest vs prior served NEW (+£2.9m,
                346s: the cold weekly history); [102] declined (fixed);
                [125, 126] declined; [122, 128] scale not configured (OCC
                stage pending, PR #510); [135] refused at compile — read back
                from this START TIME. Governed answers still ~27-32s: the
                legacy path was not the cost
    2026-09-30  the 10:26:01 check read back (7f3064ca prints what the model
                read and why the compiler refused): [135] was read RIGHT —
                pipeline_stage_movement, expected_completion_date by
                origin_stage — but labelled point_in_time, and the compiler
                refuses a grouped single figure ("a grouping makes output
                'primary' a breakdown"). Normalisation rule 7 (normal form
                1.1): a single figure whose every output groups IS a
                breakdown. The by-stage answer now names the axis 'current
                stage' and carries the caveat. The pipeline history is
                fetched only by a plan that reads it; per-input timing on
                every bank question. Design §24. OCC suite on main with and
                without PR #510: the same 34 failures, none new (PR comment)
    2026-09-30  owner direction: no local / tactical fixes to pass the bank;
                fixes must hold for other wordings. Audit (design §25): three
                bank questions quoted verbatim in the model's view and six
                restated — all reworded as meanings (vocabulary 2.15.0), and
                guarded. Rule 7 proven over 75 catalogue breakdowns; it
                supersedes the fail-closed refusal (the owner now refuses a
                split it does not publish). 105 held-out variants
                (holdout_variants_20260930.yaml) + score_variants.py: run them
                beside the bank on the next deploy; findings = a variant not
                answering as its bank question does
    2026-09-30  12:15 UTC 927fe882 deployed (run 36713297553, success, no
                retry). Vocabulary 2.15.0 (no bank wording in the model's
                view), normal form 1.1 (a grouped figure is a breakdown),
                pipeline history on demand, per-stage timing in the bank log.
                Next: the bank beside the held-out variants (start with
                --categories holdout_recent), scored by score_variants.py
    2026-09-30  12:30 UTC 2d6df14d deployed (run 36714882653, success, no
                retry): the bank runner asks the held-out variants
                (run_production_bank.sh <principal> variants-recent: the 18
                questions changed for since 2026-09-29, each followed by its
                variants, 42 in all; `variants` for all 105)
    2026-09-30  13:49:43 UTC held-out variants check on 2d6df14d (42 asked,
                all recorded — the uploaded log was mid-run): 17 of 21
                variants answered as their bank question. Owner decisions
                D15 (the pipeline's "previous" = the snapshot before the
                latest, whatever the gap), D16 (expected to complete in a
                month = face value, weighted alongside), D17 (the expected
                completion date leaves lapsed cases out). Built, vocabulary
                2.16.0 (design §26). Storage revalidation is one listing
    2026-09-30  15:15 UTC 2e4ab2f2 deployed (run 36734994989, success, no
                retry). Vocabulary 2.16.0: D15 (the pipeline's "previous" is
                the snapshot before the latest), D16 (a month's expected
                completions at face value, weighted alongside), D17 (lapsed
                cases not dated); storage revalidation is one listing; the
                second held-out set (holdout_variants_20260930b.yaml). Next:
                variants-recent, then the scale questions once OCC activates
                ERE's stage (PR #510)
    2026-09-30  15:53:05 UTC second held-out set on 2e4ab2f2 (42 asked, all
                recorded; readback run 36744000164). 18 of 21 variants
                answered as their bank question (first set: 17). Typical
                question 7-10s (was ~20s): the model ~6s, storage ~0.3s. D15
                and D16 served as decided. Findings: [58] variant "is any
                overdue, and how much" read as two measures, refused by the
                pipeline runtime (one measure), and the LEGACY path answered
                the whole pipeline — a wrong answer; [82] variant "pipeline
                movement since the prior extract" read as stage movement
                (cases_moved, amount_moved; movement) — the stage owner's
                reconciliation, not recognised; the [135] "by stage" variant
                asked a different question (authoring error, not a finding).
                Scale still SCALE_NOT_CONFIGURED: the MI service asks OCC for
                client_001's activated configuration, OCC holds it as ERE —
                no identity crosswalk. [135] now 2026-11-02 over 137 dated
                cases; 4,698 of 4,835 live cases are lapsed by the forecast's
                own windows
    2026-09-30  Owner decisions D18 ("Do not use the old system") and D19
                ("There should only be one single activated client in trakt").
                Built (design §27): for a served principal the answer is the
                governed answer or the governed DECLINE (plan_decline: what
                was understood and why not, no figure; recorded DECLINED) and
                the legacy path is never asked — the [58] variant is now
                declined instead of answered with the whole pipeline. The
                served tenant (client_001) reads the single activated OCC
                client's configuration (zero or several: nothing), so scale
                has its stage. Not deployed yet
    2026-09-30  Built (design §28): several figures of one population in one
                answer (plan_composition: each figure through its own
                runtime's gates, all or none, same data, one answer, one
                table per axis) — the [58] variant "is any overdue, and how
                much" is now answered with both figures; stage movement owns
                "what moved" over its figures (vocabulary 2.17.0) — the [82]
                variant "pipeline movement since the prior extract" is
                answered from the movement owner's totals. Not deployed yet
    2026-09-30  18:34 UTC ac0caf0c deployed (run 36759287297, success, no
                retry). Vocabulary 2.17.0: D18 (a governed decline is the
                answer; legacy never asked for a served principal), D19 (the
                served tenant reads the single activated OCC client's
                configuration), several figures in one answer (§28.1), what
                moved in the whole pipeline (§28.2). Suites: no new failure
                against 2e4ab2f2's. Next: variants-recent (the runner refuses
                a vocabulary below 2.17.0), then the full bank with a fresh
                held-out set for D13
    2026-09-30  18:42:51 UTC variants-recent on ac0caf0c (42 asked, all
                recorded; files uploaded): 31 answered, 11 declined, none by
                the legacy path. Scored: 16 SAME, 5 DECLINED both sides (by
                design), LOST [82] "what's changed since the last snapshot"
                (period read only as a stated pair), DIFFERENT [82] "week on
                week, how has the pipeline moved" (stage flows £1.18bn vs the
                live pipeline's +£2.9m) and [135]-by-stage (authoring error).
                "Is any overdue, and how much" answered with both figures.
                Scale still SCALE_NOT_CONFIGURED although OCC holds ERE v2 with
                portfolio.stage = pre_securitisation_spv — D19 did not resolve
                in production; read-only check requested. Built (design §29):
                one reading of the latest pair (plan_reading), the stage flow
                figures' definitions (vocabulary 2.18.0). Not deployed yet
    2026-09-30  Scale diagnosed by the owner's read-only check (SSH): served
                tenant ERE, OCC readable, ERE the one activated client (v2,
                pre_securitisation_spv), scale £100MM when asked for ERE. The
                request asked for client_001 — the deployment's book label for
                a question naming no portfolio. D19 now reads every
                identifier of the deployment's one book as that client
                (dependencies.served_client_ids; design §29.3). Not deployed
    2026-09-30  19:22 UTC 2e3278ff deployed (run 36764818908, success, no
                retry). Vocabulary 2.18.0: one reading of the latest pair
                (§29.1), the stage flow figures' definitions (§29.2), D19
                reading every identifier of the deployment's one book as that
                client (§29.3). Next: variants-recent (the runner refuses a
                vocabulary below 2.18.0) — scale should answer at £100MM
    2026-09-30  19:28:05 UTC variants-recent on 2e3278ff (42 asked, all
                recorded; files uploaded): 35 answered, 7 declined, none by
                the legacy path. Scale answered by all four (£100.0m, around
                2027-07, £12.9m to go) — D19 live. "Week on week, how has the
                pipeline moved" answered with the live pipeline's change.
                Scored 19 SAME, 3 DECLINED both sides, the by-stage authoring
                error, LOST [82] "what's changed since the last snapshot" (a
                summary naming amount and case count, refused). Built (§30):
                the pipeline summary accepts the headline figures it states.
                Not deployed yet
    2026-09-30  19:52 UTC 8463fd03 deployed (run 36768321984, success, no
                retry): a "what changed" summary may name the headline figures
                it states (§30). Suites: no new failure. Next: the full bank
                (`all`, the D13 measurement), then a third held-out set
    2026-09-30  19:59:06 UTC full bank on 8463fd03 (135 asked, all recorded;
                files uploaded): 104 answered, 31 declined, none by the legacy
                path. Must-answer 80/88. The last eight questions [128]-[135]
                met the model provider's credit limit (INTERPRETER_FAILURE) —
                six of the eight must-answer misses; unmeasured, to be asked
                again alone. Real: [112] current run rate (hold), [117] base
                forecast (read as "now"), and "expected completions by month"
                led with the weighted amount (D16). Built (design §31): the
                run-rate hold released on this run's readings
                (qb_recorded_intents_20260930_1959.json), a projection read as
                "now" is its owner's horizon, D16 in each month (vocabulary
                2.19.0). Not deployed yet
    2026-09-30  Owner decisions D20-D22 (design §32). Built: D20 "over
                time" with no span is every reporting date held (compiler
                default, pipeline every extract, funded every run); D22 the
                completion run-rate on the calendar from each case's own
                completion date, published for every whole-week window the
                history covers — the forecast's five-week figure no longer
                counts each extract-to-extract change as a week, and a named
                window (8-week, 12-week) is served (vocabulary 2.20.0; the
                runner refuses a vocabulary below 2.20.0). D21 (measured or
                decline) recorded; built separately. Not deployed yet
    2026-09-30  D21 built (design §33): stage completion rates AND validity
                windows are measured from the client's history or not used —
                no configured value stands in; every weighted figure (and the
                forecast on it) depending on an unmeasured stage is withheld
                with the reason; the agent declines it (RATE_NOT_MEASURED /
                FIGURE_WITHHELD). Dashboard shows n/a (ships from main). Before
                deploying: read-only check that production measures every stage
    2026-09-30  Owner's read-only check before D21 (production, deployed
                8463fd03): every stage's rate measured (APPLICATION, KFI,
                OFFER), none without enough history; every validity window
                measured (KFI 9d / 1,422 events, APPLICATION 120d / 887, OFFER
                80d / 580); no case weighted at a configured rate (81 live
                cases by run-off, £12.0m). D21 changes no production figure.
                Suites for 617a9469: no new failure. Deploy triggered
                22:52 UTC (run 36788018871)
    2026-09-30  22:54 UTC 617a9469 deployed (run 36788018871, success, no
                retry): D20 "over time", D21 measured or declined, D22 the
                run-rate on the calendar (vocabulary 2.20.0). Next: the 15
                questions by id (the 8 cut off by the credit limit, the three
                19:59 fixes, [74][75] over time, [125][126] named windows)
    2026-09-30  22:58:11 UTC 15 questions by id on 617a9469 (files
                uploaded): 15 answered, all by the governed path. The 8 cut
                off by the credit limit all answered; [112] current run-rate
                and [117] base forecast answered; "expected completions by
                month" leads with face value, weighted alongside (D16); [74]
                [75] by stage over time across 90 extracts (D20); [125] 8-week
                £4.1m/month, [126] 12-week £4.5m/month (D22). The calendar
                run-rate is £3.0m/month over the 5 weeks to 2026-09-24 (24
                cases, £3.4m) — the old per-extract figure was £1.2m — so
                scale moves from ~2027-07 to ~2027-01 and the base curve
                reaches £140.5m by 2028-02. Must-answer: 88/88 answered across
                the 19:59 run and this one. First (cold) question 318s: 90
                extract preparations and the case history; the rest 7-17s.
                Found: "lapsed" is stated two ways ([133] the weighted stages,
                £538k / 4 cases; [135] every stage, 4,698 cases, mostly KFIs
                past a measured 9-day window)
    2026-10-01  10:32 UTC fe3fb063 deployed (run 36849588712): D23 "how
                much / how many" (vocabulary 2.21.0) and the runner's `twins`
                selection (§34)
    2026-10-01  16:26:14 UTC twins on fe3fb063 (26 asked, files uploaded):
                15 answered, 11 declined, all by the governed path. Answered
                right (13): simple-average LTV by region / age band; the
                funded balance's change since last month (+£5.9m) and its
                drivers; the pipeline's largest LTV band and region; the
                aged-75-80 and 40-50%-LTV slices (loan count, WA rate 9.7%,
                £41.5m, by age band); latest-vs-previous by stage; balance by
                property region at NUTS3. ANSWERED WRONG (2): "application to
                completion pull-through" was answered with the stage
                pull-through from Application (65.4%, Application to OFFER)
                and "KFI to completion" with KFI to Application (100.0%,
                advanced 1,388, fell out 0) — "X to completion" is the
                historical completion rate from X; the two coincide only at
                Offer. The KFI 100% also shows no case is ever recorded as
                falling out at KFI: a KFI that does not proceed stays open.
                Declined rightly (5): drivers by region (not built), forecast
                loan count by region (not published), ITL3 both bases (the
                book does not carry ITL3), the Offer-stage filter (F04, as
                designed). Declined, gaps (6): ranking on the funded book
                ([A03] [D03], OPERATION_NOT_GENERIC) and on the pipeline
                ([A10], the capability lists no rank); a filter to one value
                of a published figure ([A13] the upside milestone,
                FILTERS_NOT_SUPPORTED); a filter value the book does not
                record ([A04] lifetime mortgage: no rows; [A06] Active: empty
                population) withheld as unreliable rather than said. Slow:
                the latest-vs-previous comparison prepared all 90 extracts on
                a cold process (143s). Seen: the "NUTS3" collateral field
                holds 11 coarse regions (South East, East Anglia), not NUTS3
                areas
    2026-10-01  16:38:26 UTC full bank on fe3fb063 (135 asked, files
                uploaded): 116 answered, 19 declined, all by the governed
                path. Must-answer 87/88 in one run: the miss is "What is the
                base forecast?" (forecast_scale_011), read this time as the
                forecast funded balance narrowed to the base scenario and
                declined (FILTERS_NOT_SUPPORTED) — the 22:58 run read it as
                the scale-up projection and answered. Nice-to-have 24/31.
                Fine-to-decline: 11 declined, 5 answered, each correctly
                (overdue cases 0, cohort conversion by milestone, completion
                rate by stage, withdrawn £127.6m, lapsed £538k). Found:
                "pipeline by stage for October and November" answered with
                the stage names only, the figures in the table (sentence now
                states each stage's figure at each date); the 8-week run-rate
                and "forecast completion basis" read "for the pipeline" and
                declined POPULATION_NOT_MEASURED; "balance by origination
                channel" returns the 232 broker names — the book's
                origination channel carries the broker (a data check for the
                client); property type is one value (RBLD) for all 568 loans
    2026-10-01  Built, not deployed: the five fixes for the twins and 135
                runs and D26 (design §35; vocabulary 2.22.0 — the runner
                refuses an older deploy). Stage rates ("X to completion" is
                the completion rate; a case open past its stage's measured
                window has lapsed and fallen out), filter values checked
                against the book's own values, ranking (highest / lowest /
                top N), one value of a published breakdown (a stage, a
                scenario), and a two-snapshot comparison preparing only those
                two extracts. cv_F04 revised: "Just the Offers" now carries
                (its twin answers); the refusal it tested moved to a closed
                stage (27 new twins). Before deploying: the read-only stage
                window check (D26). After deploying, re-run:
                  twins (27)
                  pipeline_010,pipeline_011,forecast_runoff_003,
                  forecast_scale_011,forecast_scale_015,forecast_scale_019,
                  forecast_018,forecast_019,forecast_020,
                  pipeline_evolution_011
    2026-10-01  D26 read-only stage window check on fe3fb063 (owner, SSH;
                history 2025-09-08 to 2026-09-24, 6,223 cases; totals only).
                KFI: window 9 days by both methods. Of 1,422 KFIs that became
                applications, 0.1% took over 14 days and none over 30. 4,800
                KFIs are still open, 4,298 of them over 90 days old — the
                extracts never close a KFI that does not proceed. KFI to
                Application pull-through 100% -> 22.6% (1,388 advanced; 4,745
                lapsed). Insensitive to the window: 22.7% at 14 days, 23.0% at
                30, 24.4% at 90. APPLICATION: window 120 days both ways,
                65.4% unchanged, none lapsed (all 17 open under 31 days).
                OFFER: window 80 -> 82 days, 71.5% -> 71.1% (5 lapsed).
                KFIs are not in the forward forecast, so the forecast moves
                only by the Offer window's two days. For the client: are
                KFIs re-issued for the same borrower? If so, 22.6% is per
                KFI, not per borrower
    2026-10-01  18:31 UTC 2e9e1cc4 deployed (run 36907157740, success, no
                retry). Vocabulary 2.22.0: the five fixes and D26 (design
                §35). Next: the 27 twins and the ten bank questions listed
                above
