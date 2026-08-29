# The `direct` collision — scope, before implementing

Base `3629b2f`, tree clean. **Nothing is shipped by this report.** Every number
below comes from a simulation applied by monkeypatch in a scratch process, run
against a control that reproduces the recorded review pack.

The headline first, because it contradicts the pre-registration I was asked to
make:

> **Neither ruling recovers a single question.** Three answers move on the
> 166-question pack; all three still refuse. The five questions I pre-registered
> — Q17B, Q22C, Q15B, Q17C, Q07B — recover **none**, and the measurement says
> why: they are **four different defects**, and the `direct` collision is not the
> largest of them. The fragment problem does **not** close. It survives, it is
> independent, and the collision fix makes it *worse* by moving two more
> questions into it.

---

## 1. Where the collision actually lives

### The tape

One book, two fields, both carrying the token `direct`:

| field | `direct` rows | other values | `mi_tier` | `source_criteria` |
|---|---:|---|---|---|
| `origination_channel` | 146 | `broker` (494) | extended | `curated` |
| `source_portfolio_type` | **441** | `acquired` (199) | **core** | **`segmentation_key`** |

A 3× population difference behind one word. `acquired` is carried by
`source_portfolio_type` **only** — so ruling 1 is already the behaviour for
"acquired", and the ruling changes nothing there.

**`direct` is the only value collision on this book.** Every governed value the
tape carries was enumerated and cross-tabulated against every governed field;
exactly one value resolves to more than one field, and it is this one. That
matters for the shape of the fix: this is not a class needing a general
arbitration mechanism, it is one collision that two *existing* declarations
already separate.

### Four owners read the word. They already disagree.

| # | owner | what it does with a bare `direct` today |
|---|---|---|
| 1 | `mi_agent/categorical_spans.py:72` `value_field` | **refuses** — two fields claim it, so `len(hits) != 1` returns `None` |
| 2 | `mi_agent/portfolio_lens.py:765` `resolve_lens` | resolves the **provenance** book for every *qualified* form ("the direct book", "direct loans", "the Direct portfolio") |
| 3 | `mi_agent/execution_receipt.py:466` `dimension_values` | **silently picks `origination_channel`** |
| 4 | the parser's categorical filter binding | binds nothing, for any `direct` phrasing |

Owner 3 is the live defect. It builds `{value: field}` with
`out.setdefault(token, key)` — **first field wins by registry iteration order**.
`origination_channel` precedes `source_portfolio_type` in the registry, so
`dimension_values['direct'] == 'origination_channel'`, chosen by file order and
disclosed to nobody. Its consumer `_detect_lost_narrowing` is what prints

> `Direct (Origination Channel) — this narrowing was not applied…`

on Q04A, Q05B and Q17B — a refusal naming the wrong field, under the ruling.

So owners 1 and 3 **already contradict each other**: one refuses the value as
ambiguous, the other resolves it silently to the losing field. That contradiction
is the defect, and it exists independently of which sense the ruling picks.

### Can the precedence be expressed in one place?

**Yes — and it needs no new registry key, no new binder, and no list.**

There are two distinct precedences here, and only one of them is missing:

**(a) Span precedence — already one place, already correct.**
`portfolio_lens.scope_phrase_spans` / `mask_scope_phrases` is the single
declaration that the scope owner claims "the direct book" before any other
reader sees "direct". `_detect_lost_narrowing` masks with it, and so do the
filter and dimension parsers. This is why "What is the balance in the direct
book?" answers 441 correctly today. Nothing needs adding.

**(b) Value precedence — missing, and derivable from what is already declared.**
For a bare `direct` with no book noun, no owner declares which field wins. But
the registry *already* separates the two, twice over:

```
origination_channel     mi_tier: extended   source_criteria: [curated]
source_portfolio_type   mi_tier: core       source_criteria: [segmentation_key]
```

The rule that expresses ruling 1 without naming `direct` anywhere:

> **When a value resolves to more than one governed field, the field the
> registry declares a `segmentation_key` wins. Between fields of equal
> standing, the value stays ambiguous and is refused as it is today.**

That is one rule, read from declarations that exist, in one place. It is the
opposite of standing finding F1's failure mode: a mechanism, not a vocabulary
list. Adding `direct` to a hard-coded map would be the list.

**Which reader must change: one, not two.** Measured below — putting the rule in
owner 1 (`value_field`) as well as owner 3 costs something; putting it in owner 3
alone costs nothing. Owner 1 should keep deferring to the scope owner, because
`source_portfolio_type` is a scope-owned field and narrowing it is owner 2's job.
Details in §3.

**This is not the multi-owner pattern.** One declaration, one reader that acts on
it, and a second reader that deliberately defers. Owners 1 and 3 stay
independent maps for the same reason `COMPARATOR_PHRASES` is shared but its
consumers are not: they share the vocabulary, not the decision.

---

## 2. Blast radius: zero

**No question answers correctly today because `direct` binds to channel.** There
is no trade to show you.

**On the 166-question review pack**, three questions touch `origination_channel`
at all, and all three name the channel *literally*:

| id | question | how it binds | moved by either ruling? |
|---|---|---|---|
| CFO11 | Show balance by origination channel. | `spec_dimensions: [origination_channel]` | no |
| CFO64 | Show origination channel concentration. | `spec_dimensions: [origination_channel]` | no |
| CFO55 | Which broker channel added the most balance since last month? | receipt only | no |

CFO10 and CFO49 ("broker channel") bind `broker_channel`, a different field
entirely. **No question binds `origination_channel = direct`, and none can:** the
tape carries only the bare tokens `broker` and `direct` for that field, and
`value_field('direct channel')` and `value_field('direct-to-consumer')` both
return `None` today. The channel sense of `direct` is currently unreachable by
any phrasing.

**Beyond the pack**, 1,445 distinct questions across 64 recorded corpora were
scanned for provenance/channel vocabulary; 124 name it, 98 of those are outside
the 166. All 98 were re-run under every ruling, in both fixtures — including the
`Gamma Direct` collision fixture that motivated `categorical_spans`:

| fixture | answer today | stopped answering | started answering | population changed |
|---|---:|---:|---:|---:|
| standard | 68 / 98 | **0** | 0 | **0** |
| Gamma Direct collision | 69 / 98 | **0** | 0 | **0** |

Nothing degrades, in either fixture, under any of the three simulations.

---

## 3. What each ruling actually does — measured

Three simulations, run over the 166 pack on **both** arms (the merge arm replayed
from the recorded proposals with the model entry point replaced by a tripwire —
**zero model calls**, the key is exhausted), plus a control with no patch.

*Control fidelity: 164/166 identical to the recorded pack on both arms. The two
that differ (CFO84 "cure rate by vintage", CFO86 "roll rates by bucket") differ
in refusal **wording** only, grade unchanged in both, neither names `direct`, and
both are identical across all four simulations — so they cancel in every diff.*

| simulation | answers moved | grades changed | improved | degraded |
|---|---:|---:|---:|---:|
| ruling 1 (bare `direct` → provenance) | 3 | **0** | 0 | 0 |
| hyphen (qualifier–noun separator) | 3 | **0** | 0 | 0 |
| both | 3 | **0** | 0 | 0 |

Identical on the off arm and the merge arm.

### Ruling 1 alone changes a refusal's wording, nothing else

Q04A, Q05B, Q17B:

```
before   Direct (Origination Channel) — this narrowing was not applied…
after    Direct (Source Portfolio Type) — this narrowing was not applied…
```

Same refusal, same population, same grade. **That is still worth having** — it is
a receipt that stops naming the wrong field — but it is a correctness gain in
disclosure, not in reach, and it should be described that way rather than counted
as a recovery.

### The hyphen is the mechanism behind the "Direct-book" family

`_qualified_span_re` and `_SCOPE_PHRASE_RE` both join qualifier to noun with
`\s+`. `_SCOPE_QUALIFIERS` already contains `direct` and `acquired`, so
"direct book" resolves and **"Direct-book" does not** — one character class,
in one shared helper, used by both the scope and lens resolvers.

Simulated, Q04A moves from `scope=total, pop=297` to `scope=direct, pop=206`,
with the `Direct` narrowing **applied** and the facet gone. That is the collision
closing, on the one question where nothing else is in the way.

### …and the hyphen fix then walks two questions into a different defect

Q05B and Q17B do **not** improve. They change failure:

```
Q17B  before  scope=total   facet: lost_narrowing 'Direct' (Origination Channel)
      after   scope=direct  filters {source_portfolio_id: [direct_001]}   ← correct!
              …and still refuses: 'Direct- book' is not a governed portfolio
                                   for this book
```

The lens now resolves correctly **and a second reader refuses anyway**. Which is
the answer to the question you asked.

---

## 4. The fragment defect survives — and it is not this defect

You asked whether "Break Direct- book" swallowing the verb closes with the
collision fix or survives, because they are plausibly two defects. **They are two
defects, and the fragment one is the larger.** Three independent proofs:

**1. It fires with no hyphen.** Q17C — "Break Direct **portfolio** balance down
across LTV, ticket size and borrower age." — already resolves `scope=direct`
correctly today, and still refuses with `'Break Direct portfolio' is not a
governed portfolio`. No hyphen, correct lens, same refusal.

**2. It fires on `acquired`.** "Break **Acquired** portfolio balance down by
region." → fragment `'Break Acquired portfolio'`. `acquired` has no collision at
all. The defect has nothing to do with which field owns `direct`.

**3. It is a capitalised-verb gap, proved by a one-character control:**

| question | result |
|---|---|
| `Break Direct portfolio balance down…` | **FRAGMENT** `'Break Direct portfolio'` |
| `break Direct portfolio balance down…` | clean, `lens=direct` |
| `Please break Direct portfolio balance down…` | clean, `lens=direct` |
| `Show Direct portfolio balance by region.` | clean, `lens=direct` |
| `Split Direct book balance by region.` | **FRAGMENT** `'Split Direct book'` |

### Located exactly

`mi_agent/portfolio_lens.py:540` `_unknown_named_book`, which runs at line 827 —
**before** the qualified-noun gate that owns the provenance vocabulary. It scans
backwards from a book noun for a run of capitalised tokens and treats the run as
a proper name. A sentence-initial imperative verb is capitalised, so it is
indistinguishable from a proper name by capitalisation alone — which is the one
signal this function has, and its own docstring says so.

The guard against that is `_GENERIC_BOOK_WORDS` at line 453, whose final block is
commented *"question scaffolding that can be sentence-initial or capitalised"*:

```
"summarise", "summarize", "summary", "show", "give", "tell", "what",
"how", "please", "provide", "list", "report", "describe", "explain",
```

**`break` and `split` are not on it.** That is why `Show Direct portfolio` is
clean and `Break Direct portfolio` is not. This is standing finding F1 in its
other form: a hand-maintained vocabulary list standing in for a mechanism, and
the gap does not produce silence — it produces a refusal quoting a verb back at
the reader as if it were the name of a book.

**Adding two words to that list is the wrong fix**, and I am not proposing it.
The mechanism the function actually wants is *"a capitalised run is a proper name
only if it is not sentence-initial, or the sentence-initial token is not a
verb"* — the position is the signal it is missing, and position is free.

**Cost of the fragment defect today: Q15B and Q17C. After the hyphen fix: plus
Q05B and Q17B.** It costs more than the collision does.

---

## 5. The pre-registration, and where it was wrong

I pre-registered Q17B, Q22C, Q15B, Q17C and Q07B as "five questions, three
phrasings of the same collision". Measured, they are **four defects**:

| id | recovers? | actual cause |
|---|---|---|
| **Q17B** | no | hyphen **and** fragment. The ruling closes the first; the second still refuses. |
| **Q15B** | no | hyphen **and** fragment. Fragment already present at baseline. |
| **Q17C** | no | **fragment only.** No hyphen; lens already resolves `direct` correctly today. |
| **Q22C** | no | **elided coordination.** "the Direct and Acquired books" — the scope mask consumes "Acquired books" and leaves a bare "Direct", which then raises `lost_narrowing`. Neither ruling touches it; `_COMPARISON_MARKERS` does not contain "drove more". |
| **Q07B** | no | **merge-arm duplicate dimension** — `spec_dimensions: [source_portfolio_type, source_portfolio_type]` → *"parsed dimension(s) neither applied nor rejected"*. Not the collision at all; the off arm answers this question. |

The two questions the collision *does* touch that I did **not** pre-register are
**Q04A** and **Q05B**. Q04A is the only one where the ruling gets all the way
through to a correct narrowing.

I would rather record that the prediction was wrong and why than restate it.

---

## 6. Two findings the measurement produced that were not asked for

### F4 — a redundant predicate is what ruling 1 costs if it is put in the wrong owner

Simulating ruling 1 in owner 1 (`value_field`) as well as owner 3 changed four
questions outside the pack — all four still correct, same population:

```
before   Total Balance · Source Portfolio in direct_001 · 441 loans
after    Total Balance · Source Portfolio Type = direct ·
                         Source Portfolio in direct_001 · 441 loans
```

Two filters expressing one claim, one from the categorical binder and one from
the scope owner. Harmless on this book — `source_portfolio_type = direct` and
`source_portfolio_id = direct_001` select the same 441 rows — and **not harmless
in general**: a book with `direct_001` and `direct_002`, or a caller-selected
`acquired_001` cohort against a sentence saying "direct loans", turns a redundant
predicate into an empty intersection nobody declared.

It also *removed* a `lost_narrowing 'Direct' applied` facet and a *"interpretation
confidence: low"* caveat on two questions — a receipt improvement arriving as a
side effect of a change that should not have touched the filter set at all.

**Recorded as: the ruling belongs in the receipt reader, not the filter binder.
A scope-owned field is narrowed by the scope owner.**

### F5 — the qualified channel sense the ruling implies does not exist yet

Ruling 1 says the channel sense "requires a qualifier — 'direct channel',
'direct-to-consumer'". Today **neither of those resolves to anything**:
`value_field('direct channel')` → `None`, `value_field('direct-to-consumer')` →
`None`. The tape carries only the bare token. So ruling 1 costs the channel sense
nothing, because the channel sense is already unreachable — but it also does not
*create* the qualified route the ruling describes. Making "direct channel" bind
`origination_channel = direct` is a value-synonym entry, separate work, and I
have not assumed it is wanted.

---

## 7. What I would build, and in what order

Costed by measured recovery per unit of change. Nothing here is built.

| # | change | where | measured effect |
|---|---|---|---|
| 1 | the segmentation-key precedence rule | `execution_receipt.dimension_values` — one rule, read from `source_criteria`, no new registry key, no new binder | 3 refusals stop naming the wrong field. **0 recoveries.** |
| 2 | hyphen in the qualifier–noun separator | `portfolio_lens._qualified_span_re` + `_SCOPE_PHRASE_RE` — one character class in one shared helper | Q04A's narrowing applies. **0 recoveries** (Q04A still loses London); Q05B and Q17B move into defect 3. |
| 3 | **the fragment defect** — position, not a longer word list | `portfolio_lens._unknown_named_book` | on today's estate: Q15B, Q17C. **With 1+2: Q05B, Q17B as well.** |
| 4 | elided coordination in the scope mask | `portfolio_lens` + `_COMPARISON_MARKERS` | Q22C, and probably Q22B |
| 5 | merge-arm duplicate dimension | `concept_merge_arm._apply_to_spec` | Q07B on the merge arm |

**1 and 2 are cheap, correct and safe — zero blast radius, measured on 264
questions across two fixtures — but on their own they recover nothing.** The
recovery you pre-registered lives in 3, 4 and 5, and 3 is the one that has to
land with 2 or the hyphen fix is a net wash.

I have not built any of them. Awaiting your ruling on order, then the
interest-rate-bucket fix, which is already scoped and approved.

---

### How to reproduce

Simulation harness (scratch, not in the repo):
`dpatch.py` (the three simulated rulings), `dsim_run.py` / `dsim_merge.py`
(pack replay, model tripwire), `dblast.py` (the 98 outside-pack questions),
`dcoll.py` / `dfrag2.py` (owner probes).

Environment: `MI_AGENT_LLM_PARSER=off` throughout — standing finding F2. The
merge arm ran with `enabled()` opened directly rather than by putting a
placeholder credential in the environment, and with `llm_query_parser._call_llm`
replaced by a `SystemExit` tripwire, so a live call would have killed the run
rather than quietly answering. **Successful model responses this task: 0.**
