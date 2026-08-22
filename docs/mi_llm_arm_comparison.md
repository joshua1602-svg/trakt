# The fifteen shipped shapes, deterministic arm vs LLM arm

Requested as one comparison at the end. **Partially delivered: the LLM arm could
not be run in this environment, and no LLM result is reported or estimated.**

---

## 1. What could not be done, stated first

`ANTHROPIC_API_KEY` is not set here. `llm_query_parser._llm_call` constructs
`anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))`, so every
call would fail. **No LLM outcome is reported for any of the fifteen questions,
and none is inferred from the deterministic result.**

The harness is written, committed and runs the moment a key is present:

```
python -m question_interpretation.llm_arm_comparison --repeats 5
```

It reports per question: the deterministic outcome, the modal LLM outcome, how
many distinct outcomes appeared across the repeats, and — where they differ —
which of the two attributions the brief asked for. With no key it prints the gate
table and says the arm was not run.

## 2. A correction to the premise, and it does not make the concern smaller

The brief said the model "proposes and the deterministic layer checks". Measured,
**it is the other way round.** `parse_with_repair` runs the deterministic parser
FIRST, and `zero_cost_first` hands off to the model only when the deterministic
spec fails validation, or its confidence is not `high`, or the question is
"layered". The model is the fallback, not the proposer.

That correction matters for *how* a divergence would arise. It does not soften
the finding below.

## 3. What this environment CAN establish — and it is the load-bearing half

The gate is pure deterministic logic, so it runs with no key:

```
id    question                                           owned   why not
A1    Please provide a portfolio summary                 False   confidence=medium
A2    Give me a summary of the portfolio                 False   confidence=medium
A3    Can you summarise the book for me?                 False   confidence=medium
A4    What are the headline numbers for the portfolio?   False   confidence=medium
A5    Tell me the basics about this book                 False   unmapped
B1    What is the LTV for loan tickets above £150k?      False   confidence=medium
B2    What is the average LTV on loans over £150k?       False   confidence=medium
B3    Show me the LTV for loans with a balance above £…  TRUE
B4    For tickets larger than £150k, what is the LTV?    False   confidence=medium
B5    What LTV are we running on loans bigger than £150k False   confidence=medium
C1    Show me balance by LTV by ticket size              False   layered
C2    Balance broken down by LTV band and ticket size    False   layered
C3    What is the balance by ticket size and LTV bucket? False   layered
C4    Give me a breakdown of balance across LTV and tic… False   layered
C5    Show balance split by LTV bucket, then by ticket s… False   layered

deterministically OWNED — the model is never called : 1 of 15
HANDED to the model when a key is present           : 14 of 15
```

### The finding

**Fourteen of the fifteen shapes fixed this week are handed to the model in
production.** Only B3 is owned outright.

Three distinct reasons, and each is worth its own line:

* **C1–C5 — all five, because they are "layered".** `_is_layered_question`
  returns True for any `" and "` with eight or more characters on each side.
  **Every two-dimension question contains "and".** So the entire shape that item
  2 was about goes to the model by construction.
* **B1, B2, B4, B5 — confidence `medium`.** These are precisely items 1 and 3's
  fixes: the comparator vocabulary and the threshold subject.
* **A1–A4 — confidence `medium`; A5 — `unmapped` before item 4, and it is the
  route recogniser rather than the spec that item 4 changed.**

So the honest position on this week's work: **every result quoted from the
deterministic arm describes the fallback path for fourteen of these fifteen
questions.** Whether a user sees any of it depends on what the model proposes and
on whether validation falls back to the deterministic spec — which is exactly
what the un-run half of this comparison would have measured.

## 4. What the un-run half would report

Per question, `deterministic outcome | LLM outcome | attribution`, where the
attribution is one of:

* **the model proposed something the fix does not reach** — the LLM arm answers,
  but over a different measure, population or grouping than the deterministic arm
  now produces;
* **the guard refused a proposal the fix would have handled** — the
  deterministic arm answers and the LLM arm is blocked;
* **the LLM arm disagreed with itself** — more than one distinct outcome across
  the repeats, reported as instability rather than as a result.

**Repeats: 5 per question, 75 LLM runs**, stated because the arm's
self-disagreement is larger than any individual change. The report prints the
modal outcome with its share (e.g. `3/5`) and names every unstable question, so a
single run is never quoted as the arm's behaviour.

## 5. What this does not license

It does not license the claim that the fixes fail for users — that is unmeasured.
Nor the claim that they hold: `parse_with_repair` validates the model's spec and
can fall back to the deterministic one, so some fraction of the fourteen may
behave identically. **Both readings are open, and the gate result is the reason
the question is worth asking rather than the answer to it.**
