# Standing findings — MI Query Agent interpretation programme

Findings that outlived the stage that produced them. Each is here because it
changed how the next piece of work was scoped, and each is stated so that it can
be checked rather than believed.

---

## F1 · A vocabulary gap does not produce silence. It produces the nearest expressible thing.

**Measured, Stage 3.** The concept vocabulary offered to the model was
constrained hard, and both constraints held exactly as designed:

- **zero** raw governed field keys proposable;
- **zero** fields this book does not carry proposable.

And the vocabulary still produced a three-way narrowing error.

There is no concept kind for a numeric threshold, so *"borrowers over 75"*
cannot be said in registered concepts at all. The model did not fall silent. It
reached for the nearest expressible things: the FIELD as a `measure` — declined
32 times — and the BUCKET VALUES as category values, `75-80`, `80-85`, `85+`,
the first of which filled an empty slot. *"Balance for borrowers over 75"*
became *"balance for `age_bucket` == 75-80"*, on a question that is EXACT today.
8 of 157 correct-today questions would have taken that fill.

**The rule.** A constraint that cannot express a concept does not stop the model
reaching for it. When designing a constrained vocabulary, the question is not
"can it say anything wrong?" — it is **"can it say everything the sentence
says?"** Where it cannot, enumerate what the nearest expressible substitutes are
and measure whether they are reached, because the substitution is silent by
construction: every proposal is in-vocabulary, every binding is registry-owned,
and every guard reports green.

This generalises past this programme. It is the same shape as a validator that
accepts only known enum members and a producer that has no member for the state
it is in.

---

## F2 · An API key in the environment switches on the shipped free-form parser arm, so any "deterministic before" captured that way is void.

**Measured, Stage 3.** `mi_agent_api.datasets._mi_llm_config` runs `auto` by
default and sets `enabled = has_key`. With `ANTHROPIC_API_KEY` present, every
`/mi/query` call is parsed by the free-form LLM arm — the one that emits a whole
`MIQuerySpec` — which is the arrangement the concept-proposal split exists to
replace.

The first full Stage 3 run was captured that way and had to be discarded. It was
caught only because two must-refuse questions answered while the merge had
filled nothing on either, and it took a stashed working tree to establish that
the merge was not the cause.

**The guard, and it stays.** Any harness measuring a deterministic baseline
while holding a key for its own model calls must refuse to run unless
`_mi_llm_config()` reports `enabled=False available=False`. `MI_AGENT_LLM_PARSER=off`
forces that while leaving the key available for direct calls. The check belongs
at the top of the harness, as a hard exit, not in its documentation:
`migration_phase0/must_refuse_both_arms.py` and the Stage 3 harness both carry
it.

---

## F3 · An instrument that cannot be measured must report NOT MEASURED, never clean.

**Stage 3.** `question_interpretation.mi_recognition_diagnosis` refused to run —
`TRAKT_RUNTIME_MODE` resolved to `production`, so `trakt_core.policy` would
refuse both books as synthetic fixtures and every shape would rate ABSENT. A
clean-looking zero would have been indistinguishable from a passing surface.

Reported as **not measured**. That stays the convention: the absence of a
finding is evidence about the instrument before it is evidence about the
product.

---

## OPEN · A live wrong answer on the shipped path, independent of this split

**Not caused by this programme, and not fixed by it.** Logged here so it is not
absorbed into the interpretation work.

With an `ANTHROPIC_API_KEY` present and `MI_AGENT_LLM_PARSER` unset:

| question | outcome | repeats |
|---|---|---|
| `What changed?` | refused | 5 / 5 |
| `Show me the trend.` | **ANSWERED** | 5 / 5 |
| `Compare us with the market.` | **ANSWERED** | 5 / 5 |

*"Compare us with the market."* is answered with `640 loans · Current
Outstanding Balance: £172.1MM`. **There is no market data in this platform.** A
whole-book figure is returned to a question asking for a comparison against
something the estate has never held, and nothing in the answer says so.

All three refuse on the deterministic arm and the frozen CFO bank has expected
`REFUSE` for all three since it was written. The mechanism is that the free-form
arm supplies the missing element itself, so no governed default is ever
recorded and the guards that fire on a recorded default never see one — the same
mechanism the Opus acceptance run walked through.

**Owner: separate work.** Recorded every run by
`migration_phase0/must_refuse_both_arms.py`, which exits non-zero only if the
DETERMINISTIC arm stops refusing and prints the LLM arm in full regardless.
Failing the instrument on a known-open finding would only teach the estate to
stop running it.

---

## OPEN · The receipt stamps a threshold applied/lost the wrong way round

**Found while measuring the threshold kind. Not caused by it, not fixed by it.**

*"How much outstanding balance do we have where borrower age exceeds 75 and LTV
is over 40%?"* (Q02B) publishes:

```
served facets : ('threshold', 'LTV over 75', 'applied')
                ('threshold', 'LTV over 40', 'lost')
spec filters  : ('current_loan_to_value',)
```

Both labels are wrong — `execution_receipt._detect_thresholds` does not resolve
the field, so the borrower-age threshold is labelled `LTV` — and the two
statuses are **inverted against the contract**: the facet stamped *applied*
names a predicate the spec does not carry, and the facet stamped *lost* names
one it does.

The consequence for anything reading the receipt is that a threshold the
contract holds reads as lost and one it does not hold reads as satisfied. It is
why Q02B was classified as a threshold loss and is not one.

**Owner: separate work.** Recorded in
`migration_phase0/MI_THRESHOLD_KIND_RESULT.json` under
`prediction_B.why_the_third_did_not_land`.

---

## OPEN · A registry gap the estate reports to the reader as a data gap

**Found in Stage 4. Three questions, one registry entry.**

*"Show a table of balance by LTV bucket and interest-rate bucket."* (Q13A, and
its two siblings Q13B and Q13C) is answered with:

> 'interest rate bucket' is not available in this dataset. This book does not
> report it, so the question cannot be answered from the current data (no value
> was fabricated).

**The book reports it.** `interest_rate_bucket` is a fully populated column on
the acceptance tape — 640 of 640 rows, five bands: `4-5%`, `5-6%`, `6-7%`,
`7-8%`, `>=8%`. What is missing is the **registry declaration**: the field is
not in `semantics["fields"]`, so `requested_dimension_terms` resolves nothing
for any spelling of the term, no owner claims it, and the concept vocabulary
cannot offer it either.

The refusal is honest about having fabricated nothing and **false about the
data**. A reader is told the book does not hold something it holds.

This was classified as a capability gap in the frozen 75-bank grades. It is
not. Every other bucket axis on this tape — `ltv_bucket`, `age_bucket`,
`ticket_bucket` — is declared and works.

**Owner: separate work, and the cheapest remedy measured in Stage 4** — three
questions for one registry entry and no code, against the threshold kind's
three questions for a new concept kind.

---

## CLOSED · The registry gap the estate reported as a data gap

Fixed in `9968025`. `interest_rate_bucket` declared from the same template as
the other derived buckets; Q13A/B/C `FALSE_REFUSAL -> CORRECT` on both arms at
30 cells against the frozen truth; `data_claim_audit`'s
`FALSE_about_the_book` class fell 3 -> 0 and its pre-registered set is now
**empty**, so any refusal that starts lying about a client's book fails the
audit on its first run.

---

## F1, SECOND INSTANCE · A guard list is not a guard

The standing finding says *a vocabulary gap does not produce silence, it
produces the nearest expressible thing*. `portfolio_lens._unknown_named_book`
was the second instance, and it is worth recording because the failure was
louder than the first.

The function reads a run of capitalised tokens before "Book"/"Portfolio" as a
proper name. Its only guard was `_GENERIC_BOOK_WORDS`, whose last block is
commented *"question scaffolding that can be sentence-initial or capitalised"*
and holds `show`, `give`, `summarise`, `provide`, `list`, `report`. It did not
hold `break`, `split`, `plot` or `which`. So

    "Break Direct portfolio balance down across LTV, ticket size and borrower age."

refused with *"'Break Direct portfolio' is not a governed portfolio for this
book"* — the reader's own verb quoted back as the name of a book they never
named. Not silence: the nearest expressible thing.

**The fix was a property, not a longer list.** A sentence-initial token is
capitalised by orthographic convention whatever it is, so its capital is no
evidence of a name; discount it. Measured over 1,446 corpus questions: three
fragments removed, none raised, every genuine unknown book name still caught.
A list is a closed set; sentence position is a property of every sentence.

**The corollary, recorded because it is the more useful half.** The list was
not two words short. It was short by however many words nobody had thought of,
and there was no way to know which — `which` is not even a verb. Whenever a
guard is a hand-maintained list of the members of an open class, the question
to ask is what property the members share, not which ones are missing.

**Still open in the same function:** a token carrying a trailing separator.
"Direct-book" yields `Direct-`, and `direct-` matches no entry in any word
list, so the run is judged a name whatever the list holds. Same shape — a token
the guard cannot match rather than a word nobody listed. Coupled to the
unshipped qualifier/noun separator fix; four questions still refuse with
`'Direct- book'`.

---

## F6 · A configuration path resolved against the process working directory

`trakt_core.capability.load_registry` resolves
`config/system/mi_capability_registry.yaml` **relative to the process cwd**, and
`mi_agent_workflow._capability_explanation` wraps the load in a bare
`except Exception: return None`.

Run from anywhere but the repository root, the `FileNotFoundError` is swallowed
and two capability-aware refusals silently degrade:

```
from the repo root   "Cure Rate is measured ACROSS governed snapshots and MI Query
                      answers from a single dataset… request it through the governed
                      history tools, where the snapshot window is resolved."
from anywhere else   "I couldn't map this question to a governed analytic."
```

No wrong number either way — both refuse. What is lost is the governed
methodology and the route to the number, replaced by an admission of ignorance
about a question the estate can name precisely. **A deployment's answer quality
should not depend on which directory it was started from**, and an absorbed
`FileNotFoundError` is how it came to.

Found while checking review-pack fidelity: two CFO questions differed between
two runs of the same code on the same tape, and the only difference was cwd.

**Owner: separate work.** Every measurement in this programme is now run from
the repository root.

---

## OPEN · Coordinated axis lists are read as measures

Two questions, one shape, and the second was hidden behind a false refusal until
`1062e43`'s successor removed it.

    Q17C  "Break Direct portfolio balance down across LTV, ticket size and
           borrower age."
    Q22C  "Which of the Direct and Acquired books drove more of the
           month-on-month balance increase?"

Q17C names three axes and the deterministic parser carries one. `ticket size`
becomes `ticket_bucket`; `borrower age` is carried in the wrong ROLE, as a
measure ("Average Borrower Age"); `LTV` vanishes. Truth is 143 cells, the answer
is 5 rows, and nothing in the receipt says an axis was lost. Its sibling Q17A —
which spells each axis with the word "bucket" — is correct today, so the defect
is the bare wording in a coordinated list, not the axes.

Q22C is the same family on the other side: the scope mask consumes "Acquired
books" and leaves a bare "Direct" from the elided coordination, which then
raises a lost narrowing.

The Stage 1 completeness check fires on Q17C today, deterministically, naming
`measure 'borrower age'` as stated-and-unresolved. It does not see bare `LTV` —
no owner resolves it as a dimension there, which is the recorded limit that the
check's recall is the owners' recall.

**Owner: separate work, and it is the strongest case yet for wiring the
completeness check.** Measured in
`migration_phase0/MI_FRAGMENT_AND_COLLISION_RESULT.md` §4.

---

## OPEN · The comparison recogniser does not accept "How do X and Y differ?"

    "Compare the Direct and Acquired books."             -> portfolio_risk_comparison
    "How do the Direct and Acquired portfolios differ?"  -> no route

The second falls to the generic path. On the deterministic arm it answers with a
plain 640-loan count — an answer to a different question, escaping a WRONG grade
only because there is no computable truth for it. On the merge arm the merge
fills `source_portfolio_type`, the generic path cannot honour it, and the
silently-dropped-dimension guard refuses. Every component behaved correctly; the
gap is the recogniser's phrasing coverage.

Recorded here because it was previously mislabelled — by me — as a merge-arm
duplicate dimension. It is not: `spec.dimensions=['source_portfolio_type']` with
`spec.dimension='source_portfolio_type'` is the documented single-axis
convention, and the "duplicate" was an artefact of the review pack's own capture
concatenating the two spec fields.
