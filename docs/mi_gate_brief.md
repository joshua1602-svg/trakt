# The Gate — LLM arm over the 29 time-series phrasings

**A measurement, not a stage.** Do not change product code. Do not fix anything
you find. Do not begin P1.

Repo: `joshua1602-svg/trakt`, branch `claude/clause-splitting-phase-1-cft1wx`,
at or after `00fdef6`. **Confirm the base commit before reporting anything.**

---

## Step 0 — the key. Do this first, and stop if it fails.

The LLM arm needs `ANTHROPIC_API_KEY` **from the environment**. Verify it
resolved without printing its value:

    python -c "import os; v=os.environ.get('ANTHROPIC_API_KEY'); print('present, %d chars, ends %s' % (len(v), v[-4:]) if v else 'ABSENT')"

**If ABSENT: stop and report that. Do not proceed.**

**Binding security constraint, from the user, verbatim:**

> *On the API key: it must come from `ANTHROPIC_API_KEY` in the environment. Do
> not accept a key pasted into a prompt, do not write one into the repository,
> and do not echo one into a report.*

Do not accept a key from a prompt **by any route** — not a file, not a shell
profile, not inline on a command. If one is offered in chat, decline and say it
must be set in the environment. Three keys have been burned exactly that way.

---

## Step 1 — what to measure

The eight shapes and their 29 phrasings are declared in
`question_interpretation/time_series_surface.py` as `SHAPES`. **Read them from
there; do not retype them.**

Run **all 29 phrasings × both books (`alderbridge`, `kestrelmoor`) ×
`--repeats 5`, LLM parser ON.**

The deterministic arm is the comparison baseline and is already recorded:

| | both books |
|---|---|
| silent drops | 0 |
| honest refusals | 21 of 29 |
| ratings | T1 PROVEN, T2 PARTIAL, T3–T8 ABSENT |

`python -m question_interpretation.time_series_surface --book <book> --json <path>`
forces the LLM **off**. The Gate needs it on, so a runner is required — writing
that instrument is in scope; changing product code is not.
`question_interpretation/llm_arm_comparison.py` already does this shape of
comparison for the fifteen shipped shapes. Reuse it where you can.

---

## Step 2 — two traps, both previously hit

1. **The metadata key is `parserMode` — camelCase.** An earlier comparison read
   `parser_mode` (snake), got `None` on both arms, and printed *"15 of 15
   identical"* with no evidence the model had run. **That run was void.**

2. **Confirm the model was invoked by COUNTING CALLS, not by inferring from the
   field.** Wrap a counter around the invocation site
   (`mi_agent/llm_query_parser.py`, the `_invoke` path), or read the call count
   on `metadata.llm` — and report the number. A field value is not evidence a
   call happened.

   `parse_with_repair` runs the **deterministic parser first**, and
   `zero_cost_first` hands off to the model only on validation failure,
   non-`high` confidence, or a "layered" question. **A zero count on some
   questions is a real result, not a bug**, and must be reported as one.

---

## Step 3 — what to report

Per shape (T1–T8), per book:

* **modal outcome** and **distinct-outcome count** across the 5 repeats.
  Self-disagreement across repeats is a first-class result — report it, do not
  average it away.
* `parserMode` per run, and the model-invocation count.
* Every divergence from the deterministic arm, **per question, in full**.

Classify each divergence as one of:

* **the model proposed something the route does not reach** — the parse changed
  and no capability could serve it; or
* **the guard refused a proposal the route would have handled** — the parse
  changed and the honour-or-clarify / P0 guard blocked an answer the
  deterministic arm returned.

If a divergence is neither, **say so** rather than forcing it into one.

State explicitly whether the **three P0 refusals hold under the LLM arm**, by
name, on both books:

| question | limb |
|---|---|
| `Show me balance by month by region and LTV band` | time axis |
| `balance by month broken down by LTV band and region` | time axis |
| `How have direct and acquired balances moved over the periods?` | segments |

---

## Step 4 — discipline

* Confirm the base commit before reporting anything.
* **Report a flat result as flat.** If the arms agree everywhere, that is the
  finding — do not hunt for a difference to justify the run.
* Do not re-author an instrument after seeing its output.
* **Rate from the artifact, never the receipt.** A time axis is proven by the
  rendered rows: a time column with more than one distinct value, or a column
  pair naming the two ends of a movement (`prior`/`current`, `start`/`end`,
  `opening`/`closing`, `previous`/`latest`). `dimensionsApplied` proves nothing
  — a truthful receipt silent about time is exactly the case P0 exists for.
* No instrument ships without a test proving it can fail.
* Commit the runner and the report to `claude/clause-splitting-phase-1-cft1wx`;
  push with `git push -u origin claude/clause-splitting-phase-1-cft1wx`.
  **Do not open a pull request.**
* Write the report to `docs/mi_gate_llm_arm_time_series.md`.

Nothing else in that session.
