# Slice 3 model boundary — measured, not tuned

Twelve fresh `claude-opus-5` interpretations against a bank hashed and committed
before the first call. Nothing was repaired in response to what they showed.

---

## 1. Run identity

| | |
|---|---|
| `LIVE_OPUS_CALLS` | **12** (`authorised_live_calls: 12`, `live_calls: 12`) |
| Model served | `claude-opus-5` — read back from the API on every case, not assumed |
| Bank | `slice3_model_boundary_v1`, sha256 `a5ed680a77cab4095a3158ba91f78e20bceeaec4f826d96c4674dbcc835963cd` |
| Bank committed at | `34b36c14`, **before** the first call; the dispatch pinned the same hash |
| Run | `actions/runs/34600054571`, questions asked 12:40:19–12:44:15Z |
| Retries | **none** — no question was asked twice or asked a second way |
| `PRODUCT_FILES_CHANGED` | **0** |

`interpretation_v2`, the compiler, the vocabulary and the prompts are byte-identical
to `f244e211`. No deployment, no `/mi/query`, no serving canary.

The harness was proved first: the push path ran the offline adjudication over
authored stand-ins (4/4 cases, 3/3 compile checks) before a penny was spent.

---

## 2. Scoreboard

```
cases 12   interpreted 12   case verdict PASS 10 / FAIL 2
compile checked 10 (M01–M10)   PASS 6   FAIL 1   REFUSED 3
PHYSICAL_SOURCE_BINDING_AUTHORED   0 / 12
```

| measure | true | false | n/a | failing |
|---|---|---|---|---|
| `PHYSICAL_SOURCE_BINDING_AUTHORED` *(inverted — false is correct)* | 0 | 12 | 0 | — |
| `SCOPE_INTENT_PRESENT` | 11 | 1 | 0 | M08 |
| `SCOPE_TYPE_CORRECT` | 10 | 2 | 0 | M08, M09 |
| `ROLE_CORRECT` | 6 | 0 | 6 | — |
| `SOURCE_REFERENCE_CORRECT` | 10 | 2 | 0 | M08, M09 |
| `MEASURE_PRESERVED` | 12 | 0 | 0 | — |
| `FILTERS_PRESERVED` | 12 | 0 | 0 | — |
| `DIMENSIONS_PRESERVED` | 12 | 0 | 0 | — |
| `TEMPORAL_INTENT_PRESERVED` | 2 | 0 | 10 | — |
| `CAPABILITY_PRESERVED` | 12 | 0 | 0 | — |

**The headline case count overstates the result.** Two cases scored PASS that do
not work at the governed boundary (M07, M10). §6 says why, and the measures are
at fault, not the scoring. Read the axis verdicts below, not `passed: 10`.

---

## 3. Axis verdicts

### TOTAL (M01, M02) — **HOLDS**
`"What is funded balance?"` → `population.base=funded`, no lens. `"…across the
whole book"` → `lens=all`. Both compile to **zero** scope predicates. The default
is not narrowed by accident and is not narrowed by the word "total".

### ROLE — Direct / Acquired (M03–M06) — **HOLDS, 4/4**
Every one produced exactly one predicate, `source_portfolio_type = direct|acquired`,
and everything else survived alongside it:

| | question | compiled scope | also kept |
|---|---|---|---|
| M03 | direct book | `source_portfolio_type=direct` | — |
| M04 | acquired portfolio | `source_portfolio_type=acquired` | — |
| M05 | loan count **by LTV bucket**, acquired | `source_portfolio_type=acquired` | `dimensions=[ltv_bucket]`, `operation=breakdown` |
| M06 | acquired **drawdown** loans, **LTV > 50%** | `source_portfolio_type=acquired` | `erm_product_type eq drawdown`, `current_loan_to_value gt 50` |

The role axis is genuinely at the model boundary today, and it composes with
dimensions and with ordinary filters.

### ATTRIBUTION (M11, M12) — **HOLDS, 2/2**
Both kept `capability=period_movement` / `operation=movement` **and**
`lens=acquired`. Adding portfolio scope did not cost the attribution capability —
the specific silent capability loss this control exists to catch did not occur.

### NAMED SOURCE PORTFOLIO (M07–M10) — **DOES NOT HOLD, 0/4 usable**
Not one of the four produced a plan that would serve the named portfolio.

---

## 4. Why the named axis fails — the model is never told it exists

This is the finding. It is not a model-capability failure.

**(a) The field carries no meaning.** `source_reference` reaches the model as

```json
"source_reference": {"type": ["string", "null"], "maxLength": 120}
```

with **no `description`** — while its siblings carry explicit enums
(`lens: [acquired, all, direct]`, `seasoning: [any, back_book, front_book]`).

**(b) The prompt never mentions it.** The system prompt is 14,024 characters.
Occurrences of `source_reference`: **0**. Of "source portfolio": **0**.

**(c) The metadata service actively instructs against it.**
`get_allowed_values` for `source_portfolio_id`, `source_portfolio_label`,
`originator_name` and `portfolio_cohort` all return:

> `has_governed_values: false` — *"Trakt governs no value list for this concept.
> Do NOT assert a filter value against it — record it in `ambiguity` with
> `blocking=true` instead."*

So on M08/M09/M10 **the model did exactly what the governed layer told it to do.**
Its own words, from M09:

> *"'ALP' appears to name a specific book/portfolio …, but Trakt governs no value
> list for `source_portfolio_id`, `source_portfolio_label`, `originator_name` or
> `seller_name`, so the value cannot be bound. Please confirm which governed
> segment 'ALP' refers to."*

It searched hard before giving up: the named cases spent **7, 12, 12 and 13**
metadata retrievals against **3–6** for the role cases. It looked for a registry,
found none, and obeyed the instruction it was given.

**(d) The alias table guarantees the collision.** `seasoning_segment` declares
`back_book` among its aliases, and its governed definition reads:

> *"Front book vs back book … **Independent of provenance (direct / acquired)**."*

A portfolio whose proper name is *"ALP Acquired Back Book"* therefore contains two
tokens that are declared aliases of two **orthogonal** governed dimensions. With
no source-portfolio registry to match the whole phrase against, decomposing the
name into those axes is the only reading available to the model.

---

## 5. The four named cases, exactly

**M07 — `"What is funded balance for ALP Acquired Back Book?"` → would serve, and would serve WRONG.**
The model carried the full phrase in `source_reference` **and** decomposed the same
name into two more axes. It compiled to **three** predicates:

```
seasoning_segment    = back_book      ← from the portfolio's NAME
source_portfolio_id  = alp_acquired   ← correct, and sufficient on its own
source_portfolio_type= acquired       ← from the NAME; redundant here
```

All three ambiguities it raised were `blocking: false`, so **nothing refuses this
plan**. Any loan in `alp_acquired` that is front-book by months-on-book is silently
excluded. This is the most serious finding in the run: a silent narrowing that
reaches the reader as an answer.

**M08 — `"…for the ALP back book?"` (a declared alias) → scope dropped, then refused.**
`source_reference` empty, `lens=all`; "back book" read as `seasoning=back_book`;
'ALP' raised as `blocking: true`. The compiler returned **CLARIFY**. Had the phrase
been carried, it would have resolved — `"ALP back book"` → `alp_acquired`. (Note the
resolver is case-insensitive but literal: `"the ALP back book"`, with the article,
returns `SOURCE_UNKNOWN`.)

**M09 — `"Show ALP Acquired Back Book funded balance each month."` → named widened to role, then refused.**
`lens=acquired` + `seasoning=back_book`, no `source_reference` — a silent widening
from one portfolio to the whole acquired back book, caught only because 'ALP' was
raised as `blocking: true` → **CLARIFY**. The series survived (`form=series`,
`grain=monthly`): the named source cost the scope, not the temporal intent.

**M10 — `"…this period versus the previous period?"` → reference truncated, refused.**
`source_reference='ALP'` — the fragment, not the reader's phrase. The deterministic
resolver returned `SOURCE_UNKNOWN` → **CONCEPT_UNAVAILABLE**, plus the blocking
ambiguity → **REFUSE**. The comparison survived (`form=relative_pair`).

**The system fails safe in 3 of 4.** Only M07 would have been served, and it would
have been served narrow.

---

## 6. What my own measures got wrong

Recorded, not repaired.

1. `SCOPE_TYPE_CORRECT` for a NAMED case is `bool(source_reference)`. That passes
   **M10**, whose reference (`'ALP'`) cannot resolve, and passes **M07**, whose
   reference is carried alongside two contradicting axes. A reference that does not
   resolve is not scope intent captured.
2. **No measure counts extra governed axes asserted from a portfolio's name.** The
   M07 defect was caught only by the compile check, and only because the compile
   check compares the predicate set exactly.
3. `PHYSICAL_SOURCE_BINDING_AUTHORED = 0/12` is a real result but a weaker one than
   it looks: the metadata service explicitly forbids asserting a value against
   `source_portfolio_id`. The control held partly because the model was told to
   hold it.

---

## 7. Failure classification

The brief's lettered categories did not survive context compaction, so these are
named descriptively for mapping rather than guessed at.

| # | category | cases | attributable to |
|---|---|---|---|
| 1 | **Affordance gap** — no description, no prompt mention, no value list, explicit instruction not to bind | M07, M08, M09, M10 | the governed layer, not the model |
| 2 | **Name decomposition** — a portfolio's proper name read as independent governed axes (`seasoning_segment`, `lens`) | M07, M09, M10 | the alias table (`back_book` is a declared `seasoning_segment` alias) |
| 3 | **Reference truncation** — reader's phrase reduced to an unresolvable fragment | M10 | model |
| 4 | **Scope dropped** — named portfolio absent from every field | M08 | model |

Nothing here is a failure of the model to understand portfolio language. Given a
governed source-portfolio registry to match against, categories 1 and 2 dissolve;
3 and 4 are the only residual model behaviour, and both currently fail safe.

---

## 8. Bottom line

```
TOTAL        axis at the model boundary   YES  (2/2)
ROLE         axis at the model boundary   YES  (4/4, composes with filters + dimensions)
ATTRIBUTION  survives portfolio scope     YES  (2/2)
NAMED SOURCE axis at the model boundary   NO   (0/4 usable)
SILENT WRONG ANSWER REACHING THE READER   1    (M07)
```

Not repaired. Stopping here.
