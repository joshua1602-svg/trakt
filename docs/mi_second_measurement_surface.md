# The second measurement surface — deterministic arm, established

Standing condition 1 requires **both surfaces at every stage**. Stage 1 recorded
that the calibration bank runs here but the 44-variation robustness bank's
recorded figures could not be reproduced. This closes as much of that gap as is
closeable without an API key, and states precisely what remains open.

| | |
|---|---|
| Base | `4e051f3`; `28ece25` ancestor of HEAD ✓ |
| Production code changed | **none** — `git diff 4e051f3..HEAD -- mi_agent mi_agent_api due_diligence` is empty |
| Frozen artefacts | **imported, never modified** — `nl_bank.bank` supplies the questions, `nl_score.grade` supplies the verdict |
| Reproduce | `python -m question_interpretation.run_robustness_deterministic --all-books` |

## What was established

The 44-variation bank, both books, through the **same endpoint** the recorded
harness uses (`POST /mi/query`), with the **same frozen scorer**, and one
documented difference: the LLM parser is off.

| Outcome | alderbridge | kestrelmoor |
|---|---:|---:|
| `CORRECT` | 32 | 32 |
| `SAFE_REFUSAL` | 10 | 10 |
| `CORRECT_WITH_DISCLOSED_LIMITATION` | 2 | 2 |
| **unsafe outcomes** | **0** | **0** |

**Same verdict on both books: 44 / 44.**

That last line is a useful corroboration rather than a coincidence. The recorded
LLM-arm evidence reports *"44 of 44 variations produced the same verdict on both
books (100%)"*. The deterministic arm reproduces that book-invariance
independently, which is evidence the harness is pointed at the right thing.

The run is **reproducible run-for-run**: executed twice, every variation's
outcome and route identical.

### The Stage 4 acceptance family, measured by name

Stage 4's acceptance is "the seasoning families are unmoved", measured
explicitly and by name rather than inside an aggregate. The baseline:

| Intent | Variations | Baseline |
|---|---:|---|
| `Q1` — origination profile change | 4 | all `CORRECT` |
| `Q7` — vintage risk comparison | 4 | all `CORRECT` |
| `Q8` — population movement comparison | 12 | all `CORRECT` |
| **total** | **20** | **20 / 20 `CORRECT`, on both books** |

These 20 are the population that moved in `32c263a`. Any Stage 4 attempt is
measured against this line, per book, by name.

## Standing condition 4 — this instrument can fail

`question_interpretation/tests/test_robustness_instrument.py`, 6 tests, proving
the instrument detects the regression it exists to catch rather than merely
reporting a number:

* the frozen scorer grades a healthy `Q1` run `CORRECT`;
* the frozen scorer **downgrades the `32c263a` shape** — a seasoning population
  read as unapplied, so a question the route *can* express refuses — to
  `SAFE_REFUSAL`, not `CORRECT`;
* the summary reports the seasoning family **by name**, and **moves visibly**
  when that family regresses (16 → 4 `CORRECT`);
* a refusal with no stated reason grades `SILENT_SEMANTIC_ERROR`, not
  `SAFE_REFUSAL`, so "unsafe outcomes remain zero" cannot be satisfied by a
  silent failure.

The scorer is imported, not copied. If `nl_score.grade` ever stops downgrading
that shape, these tests fail rather than quietly passing.

## What is still open, stated as open

**This is a strictly weaker surface than the recorded one, and it is not a
substitute for it.**

| | Recorded arm | This arm |
|---|---|---|
| Parser | LLM on, production cost controls | **LLM off** |
| Runs | 752 — 44 × 2 books × 3–5 repeats × 2 arms | 88 — 44 × 2 books × 1 |
| Repeat variance | measured | **not measurable** — deterministic by construction |
| Reproducible here | **no** — `nl_harness.py` asserts `ANTHROPIC_API_KEY` | yes |

The consequences, plainly:

1. **The 91.0% correct/disclosed figure and the 160-run regression are LLM-arm
   measurements.** They cannot be reproduced or re-measured here, and the
   77.3% correct/disclosed above (34 of 44) is **not comparable to 91.0%** —
   different arm, different denominator. Neither number should be quoted
   against the other.
2. **Repeat variance is untestable.** The recorded evidence's central claim —
   that 98% of groups produced an identical grade across every repeat — is a
   property of the LLM arm. A deterministic arm cannot vary, so it cannot
   confirm or refute it.
3. **A regression that only manifests through the LLM parser will not be caught
   here.** The 160-run regression was an LLM-arm result. Whether it would also
   have shown deterministically is unknown and untested.

So standing condition 1 is **partially** honoured from Stage 2 onward: two real
surfaces, both runnable, both with failure proofs — and one of them measuring a
narrower thing than the condition was written for.

**This needs a decision before Stage 3**, which is the first stage that changes
behaviour. Either an API key is provided so the LLM arm can run, or the
condition is explicitly relaxed to the deterministic arm with the third
consequence above accepted in writing. It should not be left to be discovered
at the first conversion.

## Not done

Stage 2 is not started. No production code has been modified.
