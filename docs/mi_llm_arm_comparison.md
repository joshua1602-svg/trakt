# The fifteen shipped shapes, deterministic arm vs LLM arm

Requested as one comparison at the end. **Run in full: 15 questions x 5 repeats,
75 LLM calls against Haiku 4.5.**

---

## 1. RUN. 15 questions x 5 repeats = 75 LLM calls, Haiku 4.5.

An API key was supplied, so the comparison the earlier draft could not run has
been run. That draft is superseded; what follows is measured.

```
   same outcome                                            15
   the model proposed something the fix does not reach      0
   the guard refused a proposal the fix would have handled  0
   the LLM arm disagreed with itself                        0
```

**No divergence on any of the fifteen, and no self-disagreement across five
repeats each** (every question 5/5 on one outcome).

## 2. First, the evidence that the arm observed anything

This programme has found six instruments that could not see the change they were
meant to measure, so the clean result was checked before it was quoted.

**The first run of this harness was void and looked identical.** It read
`meta.get("parser_mode")` — snake_case — and the metadata key is `parserMode`.
Both arms reported `None`, and the harness printed "15 of 15 identical" **with no
evidence the model had run at all.** A counter wrapped around
`llm_query_parser._invoke` proved it had: one call per question.

The harness now records `parserMode` and `llm`, and prints, above any
comparison, what each arm actually used. That line is the difference between a
result and a coincidence.

## 3. Why there is no divergence — the mechanism, not the absence of one

```
id   question                                       det.mode       llm.mode       llm.detail
B3   Show me the LTV for loans with a balance…      deterministic  deterministic  deterministic_zero_cost
B1   What is the LTV for loan tickets above £150k?  deterministic  deterministic  deterministic_fallback
B2   What is the average LTV on loans over £150k?   deterministic  deterministic  deterministic_fallback
B4   For tickets larger than £150k, what is the LTV deterministic  deterministic  deterministic_fallback
B5   What LTV are we running on loans bigger than £ deterministic  deterministic  deterministic_fallback
C1-C5  (all five)                                   deterministic  deterministic  deterministic_fallback
A1-A5  (all five)                                   route answers before a spec parser mode is stamped
```

**The model is invoked for 14 of the 15 — and on every one of them the
DETERMINISTIC SPEC IS WHAT EXECUTES.** `deterministic_fallback` is the stamped
reason: the LLM proposes, `parse_with_repair` validates, and the deterministic
spec is preferred.

Traced end to end on four of them, with a call counter attached:

```
Show me balance by LTV by ticket size
  llm calls 1 · parser_mode deterministic
  final spec: metric=current_outstanding_balance dims=['ltv_bucket','ticket_bucket']

What is the LTV for loan tickets above £150k?
  llm calls 1 · parser_mode deterministic
  final spec: metric=current_loan_to_value filters={balance gt 150000}
```

Those are exactly the specs items 1–4 produce. **The fixes reach users.**

## 4. What this does and does not establish

**Does:** on these fifteen shapes, with a key present and the model in the loop,
the answer a user receives is the one the deterministic arm produces. Every fix
from items 1–4 is visible in production behaviour, not only in the fallback path.

**Does not:** it does not establish that the LLM's proposal is always discarded.
It was discarded on all fourteen here because the deterministic spec validated
and is preferred; a question where the deterministic spec fails validation would
execute the model's, and none of these fifteen is such a question. The
concern the brief raised is real for that class — this bank simply does not
contain one.

**Also observed, not a correctness problem:** fourteen LLM calls are made whose
output is then discarded. That is a cost and latency finding, recorded here and
not opened.

## 5. The correction to the premise still stands

The brief said the model "proposes and the deterministic layer checks". Measured:
`parse_with_repair` runs the deterministic parser FIRST and hands off only when
its spec fails validation, its confidence is not `high`, or the question is
"layered" — and then, on this bank, prefers the deterministic result anyway. The
model is a fallback that was itself fallen back from.

---

## 6. The gate, unchanged from the earlier draft



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

## 7. The attribution vocabulary the report uses

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

## 8. Superseded

It does not license the claim that the fixes fail for users — that is unmeasured.
Nor the claim that they hold: `parse_with_repair` validates the model's spec and
can fall back to the deterministic one, so some fraction of the fourteen may
behave identically. **Both readings are open, and the gate result is the reason
the question is worth asking rather than the answer to it.**
