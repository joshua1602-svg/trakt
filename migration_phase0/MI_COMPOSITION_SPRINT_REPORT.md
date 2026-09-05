# Composition sprint — deliverables

```
base    4aed376   (pre-sprint)
head    410e647
diff    8 files, +1460 / -51
```

---

## A. Architecture before / after

| Concept | Owners before | Owner after |
|---|---|---|
| **a count was requested** | `_COUNT_INTENT_RE`+`_wants_count`; `_COUNT_MEASURE_RE`; `_SHARE_COUNT_RE`+`_counts_a_row_noun`; `is_count_q` (inline); `named_measure_concepts` (receipt) — **5, three adjacency-bound, one blind to counts entirely** | `lexical.COUNT_REQUEST_RE` / `counts_rows()` / `count_request_spans()` |
| **population** | `_parse_filters`+`_borrower_structure_filter` (single-output); `_parse_filters`+`_parse_categorical_filter` (multi) — **2, neither a superset** | `_resolve_population()` |
| **`amount` → balance** | a terminal parse branch; `_DEFAULTED_MEASURE_RE`; absent from the receipt vocabulary — **3** | `lexical.DEFAULTED_MEASURE_RE` |
| **role of a span** | `_measure_hits` had 3 guards; `_detect_metric` had 0 | `_span_holds_another_role()` |
| **was a threshold applied** | inferred from row-count reduction | the executor's `applied_filter_fields` |
| **a bound's unit** | hard-coded for `£` only | `lexical.bound_unit()` + declared field vocabulary |
| **reference to a prior result** | did not exist | `lexical.refers_to_prior_result()` |
| **clause-local scope** | did not exist | `_clause_local_narrowing()` (detects; refuses) |
| **conversational scope** | does not exist | **not built** |

## B. Files and functions changed

**`question_interpretation/lexical.py`** (+194) — now the single owner of the
question's grammar vocabulary. Added `ROW_NOUNS`, `COUNT_REQUEST_RE`,
`counts_rows`, `count_request_spans`, `row_noun_alternation`, `POSTFIX_GE/LE`,
`postfix_operator_alternation`, `BOUND_UNITS`, `bound_unit`,
`DEFAULTED_MEASURE_RE`, `names_defaulted_measure`, `refers_to_prior_result`;
`_FILTER_AFTER_RE` extended to the postfix form.
*Invariant owned:* what the words of a question mean, before anyone acts on them.

**`mi_agent/llm_query_parser.py`** (+375/−…) —
`_resolve_population` (one population owner, both paths);
`_span_holds_another_role` (role ownership, both measure resolvers);
`_unit_owner` / `_field_names_unit` (a unit-bearing bound binds the field
measured in that unit); `_clause_local_narrowing` (a bound about one output);
`_postfix_pattern`; the grouped-bar branch gained the `_counts_a_row_noun` arm
and `metric_defaulted`; the `amount` terminal branch reads the population owner;
`is_count_q` deleted.

**`mi_agent/execution_receipt.py`** (+58) — the threshold facet reads
`applied_filter_fields` instead of inferring from row count;
`named_measure_concepts` reads the count and `amount` owners.

## C. Duplicate owners removed or delegated

Deleted outright: `is_count_q` (inline fifth reading of "count").
Delegated to a single owner: `_COUNT_INTENT_RE`, `_COUNT_MEASURE_RE`,
`_SHARE_COUNT_RE`, `_DEFAULTED_MEASURE_RE`, the postfix comparator patterns,
the three role guards inside `_measure_hits`, the currency-only unit rule, and
the second population resolver in `_measure_set_recognizer`.

**Net: eleven independent readings reduced to seven owners, with nothing new
layered on top of an old layer.**

## D. Test evidence

Four new characterisation files (+685), each falsified against the unfixed tree
before its fix, each stating an invariant over several fields rather than the
sentences that exposed it.

**Wide MI regression** — `mi_agent/tests`, `mi_agent_api/tests`,
`question_interpretation`, identical file sets, baseline in a separate worktree
at `4aed376`:

```
base   38 failure nodes
head   37 failure nodes
NEW    (none)
GONE   test_checked_in_registry_matches_generator   ← worktree artefact:
       the registry records an absolute source path, so it fails from
       /tmp/claude-0/base and passes from /home/user/trakt. Not a fix.
```

Pre-existing failures (37) are unchanged in identity and reason; spot-checked
`test_complex_query_executes_all_filters`, which fails byte-identically on base.

## E. Bank prediction

**I do not have the M101–M128 question texts** — this brief gave counts, themes
and five worked examples, not the bank. So the prediction is by CLASS, and any
row whose text I have not seen is outside it.

| Class | Prediction | Why, architecturally |
|---|---|---|
| count + measure, either clause order | **fixed** | count discovery is modifier-tolerant and single-owned; the pair now reaches the measure set, which already executes |
| composed request naming a borrower-type population | **fixed** | both paths resolve population through one owner; this class was answering over the whole book while scoring VERIFIED |
| pipeline `amount` inside a composed request | **fixed** | one measure vocabulary; the word is visible to the measure set |
| pipeline rate threshold with `amount` | **fixed** | the terminal branch carries the population |
| count over a bucketed population | **fixed** | role ownership; the axis is no longer the measure |
| clause-local narrowing (`of that balance …`) | **refuses, honestly** — was silently narrowing every output | detected, not executable |
| pipeline product | **remains ATOMIC_BLOCKED** | `product_type` vs `erm_product_type`; Phase 2 alias, untouched here |
| NNEG grouped, pipeline forecast | **remains refused** | no governed grouped primitive / not exposed; §15 |
| conversational sequences | **not addressed** | no transport, no scope model |

## F. Live validation

The banks are unchanged. On the Kudu box:

```bash
# 0. refresh the probe — migration_phase0 is NOT in the deploy package,
#    so deploying does not update /home/replay_probe.py
scp migration_phase0/replay_probe.py <kudu>:/home/replay_probe.py

export MI_BEARER='<Authorization header value, without "Bearer ">'

# 1. atomic perimeter (100)
python3 /home/replay_probe.py --from-log /home/perimeter_100.json \
    --base https://app.traktinfra.io/api --portfolio ERE/2026-06-30 \
    --out /home/replay_perimeter.json

# 2. multi-output bank (28)
python3 /home/replay_probe.py --from-log /home/multi_28.json \
    --base https://app.traktinfra.io/api --portfolio ERE/2026-06-30 \
    --out /home/replay_multi.json

# 3. regression bank (115)
python3 /home/replay_probe.py --from-log /home/bank_115.json \
    --base https://app.traktinfra.io/api --portfolio ERE/2026-06-30 \
    --out /home/replay_115.json
```

`--limit N` for a smoke subset; `--retries` defaults to 1 and retries only
NOT_MEASURED, so a refusal is never re-asked into looking like an answer.

## G. Verdict

**Atomic semantic safety — GO.** Zero new failures across the MI estate on
identical file sets. §14A, §14B and §14C are closed through general invariants;
P049 no longer carries the borrower-age predicate; the age and borrower-type
vocabularies are untouched and asserted. One correction to the record: the
threshold guard was refusing correct answers whenever a bound matched every row,
which was a live false-refusal class nobody had counted.

**Same-turn composition — CONDITIONAL GO.** Same-population composition is
sound and reuses the existing executor: requested outputs are single-owned, the
population survives composition, and an output that cannot execute fails closed
(demonstrated: 3 requested / 2 executable refuses and names the missing one).
The condition is clause-local composition, which is **detected and refused, not
executed**. That converts a silent wrong answer into an honest refusal — a real
improvement, and not the §3 deliverable.

**Conversational progression — NO-GO.** Not started. `QueryRequest` carries no
conversation or session identifier, so this needs a transport decision before it
needs code. The shared owner it depends on — `refers_to_prior_result` — exists
and is already used by same-turn clause scope, so §13's one-population-model
requirement is satisfiable when the rest is built.

## Open decisions

1. **Transport for §10–13.** (a) `conversationId` + server-side scope store, or
   (b) the client echoes back the prior turn's governed scope. Nothing built so
   far depends on the choice.
2. **Where the next effort goes** — `QueryPlan` execution for §3/§8, or §10–13.
   Both in one sprint would leave half of each, which is the layered debt §22C
   warns against.
