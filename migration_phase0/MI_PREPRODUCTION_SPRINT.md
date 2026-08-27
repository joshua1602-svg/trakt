# Final pre-production sprint — result

Start SHA `359d287b4986123885247b9be48626c103ae147a` (clean tree, 0 modified files).
Final SHA: see `git log` — this document is committed with it.

Three objectives were set. **Two completed. The third stopped on its own
stated stop condition.**

| # | Objective | Outcome |
|---|---|---|
| 1 | Close the remaining two deterministic CR6 defects | **Done** — Q22B and Q22C recovered, every figure matched against independent truth |
| 2 | Replace the unsafe LLM-as-`MIQuerySpec` behaviour | **Done** — withdrawn from every serving surface; the constrained replacement was already in the estate and is verified against 2A–2F |
| 3 | Use the constrained layer to recover the 24 CR4 questions | **STOPPED** — no successful Opus response is obtainable in this environment; see §3 |

---

## 1. CR6 — Q22B/C

### The divergence

Both questions route to `analytical_composition`, build the plan
`[period_movement, period_movement]`, and **measure both books correctly**:

| population | prior | current | change |
|---|---|---|---|
| Direct (441 loans, 424 prior) | £105.0m | £117.4m | **+£12.4m** |
| Acquired (199 loans, 176 prior) | £44.5m | £54.7m | **+£10.2m** |

Then they refused, each naming a narrowing "not applied" that had been applied.
Q22C: `lost_narrowing "Direct"` → LOST. Q22B: `grouping_dimension
"direct or acquired"` → LOST.

The first divergence is not in the guard. It is in the **contract**, one stage
earlier — the same *stated-but-not-carried* shape this programme has been
tracking:

* a population declares what it narrowed to in `predicate`;
* every predicate `mi_agent.population` produces is **field-named**, so
  `narrowedTo` — the one channel the receipt reads to answer *"was this answer
  scoped to X?"* — is built by splitting those apart;
* **one narrowing is not field-named.** A portfolio lens describes itself as
  `"portfolio lens = Direct (direct_001)"`, which names a lens. Parsing it
  yields the field `"portfolio lens"`, which is not a field, so the builder
  skipped it — `if piece.startswith("portfolio lens"): continue`;
* a lens narrowing that HAD run therefore left no trace any consumer could
  read. `narrowedTo: []` for a plan that narrowed twice.

### The fix

`PopulationRef.narrowed_on` — the machine-readable twin for a narrowing whose
predicate text names no field. The lens branch fills it with the
source-portfolio type field and the lens name, which is what that filter scoped
the rows to however the registry enumerates it into ids. Nothing is re-derived,
no arithmetic changes, and the reader-facing predicate is byte-for-byte
unchanged.

**No guard changed, and no guard was bypassed.** Both refusals lift through the
readers that were already there:

* `_analytical_narrowed_to` matches the value → Q22C;
* `_two_or_more_populations` sees two values of one field → Q22B, through the
  axis-or-filter threshold it already owned and could not previously reach.

Strictness is unchanged and asserted: a plan narrowing to **one** population is
still a filter and not a breakdown; a facet naming a population the plan did
**not** narrow to is still LOST; a predicate that selected **no rows** is still
not a narrowing.

### A second producer, found by the fix failing

`executors.period_movement` composed the lens predicate text itself, under a
comment asserting it matched `populations.apply`. The assertion held until the
two had to say more than text: the fix went into `apply`, the capability that
runs these two questions used the other, and both answers stayed refused with
the fix in the tree. Both now go through one primitive,
`populations.lens_narrowing`, and a test asserts there is exactly one.

*This is the fail-open pattern's sibling and belongs with it: when a comment
asserts that two components agree, check that one of them calls the other.*

### Proof

| surface | result |
|---|---|
| Q22B/C vs independent truth | all 6 figures + the larger-population ordering **exact** |
| Q22A (successful sibling, `funded_bridge`) | +£22.6m total = 12.4 + 10.2 ✓ |
| 75 bank + CFO 91 (166) | **only Q22B/C moved**; 164 byte-identical |
| six registered pipeline answers | 6/6 unchanged |
| frozen 278-module regression | 85 failing names, **exact** |
| new test `tests/test_p1_lens_narrowing_declared.py` | 9 pass; **8 fail without the change** |

Mechanically both moved `FALSE_REFUSAL → NO_COMPUTABLE_TRUTH`: this truth shape
(a nested per-population dict) is one the pack grader has no figure-checker for,
and it graded the successful sibling Q22A the same way before this sprint. Per
F3 that is the honest label — *not measured*, never *clean* — and the answers
were therefore checked field by field against `independent_truth` instead. The
grader was not changed.

---

## 2. The model boundary

### The audited live path (before)

```
mi_service._run_mi_query
  -> llm_cfg = datasets._mi_llm_config()          # auto == a key is consent (F2)
  -> ParsedQuestion.parse(..., llm_enabled=llm_cfg.enabled)
  -> llm_query_parser.parse_with_repair
  -> _invoke  ->  parse_llm_response_to_spec(text)   # a WHOLE MIQuerySpec
  -> downstream: routes, guards, planner, executor, receipt
```

`run_mi_agent_query` re-parses only when `parsed is None`, and `mi_service`
always supplies it — so the API reached the arm through **exactly one gate**.
A second serving surface reached it independently: the Streamlit workbench's
parser radio, under help text promising that "the validator and executor remain
the control layer". That is precisely what a model-written contract defeats:
every guard reads the contract, so a contract the model wrote is checked
against itself.

### Closed

* `datasets._mi_llm_config` now answers `enabled=False` for **every**
  environment. What the operator asked for is reported back as `requested`,
  with status `withdrawn_unsafe_boundary` — withdrawn is not the same as
  ignored.
* the workbench radio no longer offers the arm.
* the offline A/B harnesses are untouched: they call `parse_with_repair`
  directly with `llm_enabled=True`, which is a **measurement** of the rejected
  architecture, not a request path. `must_refuse_both_arms` now measures two
  identical arms and records the status that says why.

`tests/test_p2_model_contract_boundary.py` pins it: five environments including
"a key alone" (F2) and an explicit `MI_AGENT_LLM_PARSER=on`, plus a source-level
assertion that no serving surface selects the mode and that `mi_service` takes
`llm_enabled` from that one gate and nothing else.

### Proof of no blast

| check | result |
|---|---|
| 166 (75 bank + CFO 91) vs the Phase 1 commit | **166/166 identical** |
| the same 166 with `ANTHROPIC_API_KEY` set **and** `MI_AGENT_LLM_PARSER=on` | **166/166 identical** |
| must-refuse controls, key present and arm requested | 3/3 still refuse |
| six pipeline answers | 6/6 unchanged |
| frozen 278-module regression | 85 names, **exact** |

The second row is the point: at `359d287` that environment switched the
free-form arm on. **F2 is closed.**

### The replacement, against 2A–2F

It already existed — `mi_agent_api/concept_merge_arm` over
`question_interpretation/{concept_proposal,claim_merge}` — and runs the
required direction:

```
question -> semantic proposal -> deterministic binding/merge
         -> the same governed contract -> the same guards
```

| | requirement | evidence |
|---|---|---|
| **2A** | proposals are semantic claims with a source span | `ProposedConcept(kind, term, covers, comparator, value)`; `covers` is the words of the question the concept is for |
| **2B** | no slot for a canonical field | asserted by test over the dataclass fields. `term` must be **copied exactly** from a book-scoped vocabulary of natural-language terms; `bind()` resolves it through the registry — one governed owner → bind, several → `AMBIGUOUS`, none → `UNREGISTERED`. *"NEVER A NEAREST MATCH."* The vocabulary is scoped to the book's own columns on purpose, so `erm_sub_product_type` — declared in the registry, absent from the tape — is not proposable |
| **2C** | monotonic; add, never erase | `merge()`: an occupied slot yields `AGREED` or a `DECLINED_*` finding — *"a filled slot is never overwritten; the disagreement is reported and neither side is picked"*. `_apply_to_spec` writes only `filled_by_model` |
| **2D** | provenance preserved | fills carry `PROV_MODEL_INFERRED` and publish under `metadata.conceptMerge`; `PROV_DEFAULT` counts as **filled**, which is what keeps "Show me the trend." refusing |
| **2E** | guards remain downstream | the merged spec is the same `MIQuerySpec` the deterministic path produces; audit above shows no remaining bypass |
| **2F** | data boundary | the prompt is built from the vocabulary and the question. No rows, balances, aggregates, answers or results |

Its own flag is `MI_AGENT_CONCEPT_MERGE`, off by default, and **a key alone does
not turn it on either** — asserted by an existing test.

---

## 3. Phase 3 — STOPPED on the brief's stop condition

> *"Stop if the SDK/model configuration means successful model responses cannot
> be proven."* · *"Do not count `_invoke` attempts as successful model usage."*

Measured today (2026-08-27):

* `ANTHROPIC_API_KEY` is **absent** from this container's environment; no key
  file exists anywhere on disk;
* the `anthropic` SDK **is** installed (1.1.0), so this is not a packaging gap;
* every API key supplied in this window was tried. All returned:

  ```
  400 invalid_request_error — You have reached your specified API usage limits.
  You will regain access on 2026-09-01 at 00:00 UTC.
  ```

  request ids include `req_011CeTy5sHpCPRNbaRc29t6g`,
  `req_011CeTy5tMZDxqgxA6Q3oXp4` and `req_011CeTz94ibpoA1MZZxjPhHh`.

**The cause is a spend ceiling, not a configuration.** This was separated
rather than assumed. On the same key, in the same process:

| call | result |
|---|---|
| `models.list` | **SUCCESS** — returns `claude-opus-5`, `claude-sonnet-5`, `claude-fable-5` |
| `messages.create` · `claude-opus-5` | 400 usage limit |
| `messages.create` · `claude-opus-4-1` | 400 usage limit |
| `messages.create` · `claude-haiku-4-5` | 400 usage limit |
| `messages.create` · `claude-3-5-haiku` | 400 usage limit |

So the key authenticates, the SDK and network path work end to end, and the
account is entitled to the model this sprint targets. Inference alone is
refused, across every model including the cheapest — there is no model choice
or endpoint that routes around it.

**Successful API completions: 0. Valid semantic proposals: 0. Accepted
deterministic bindings: 0. Returned model identifier: none — no response was
returned to read one from.** Access returns five days from now.

Shadow mode cannot run, so Phase 4 was not entered and no CR4 case was moved.
Replaying previously recorded proposals was rejected as a substitute: it cannot
satisfy *"confirm the configured Opus model from the runtime response"*, and
presenting it as shadow evidence would be the stop condition absorbed into a
new baseline.

### CR4 — 24 cases, unchanged

Start and final dispositions are identical for all 24. 15 refuse, 8 answer
WRONG (F1 — a vocabulary gap produces the nearest expressible thing, not
silence), 1 (Q23A) grades CORRECT mechanically.

**Recovered correctly: 0. Still safely refusing: 15. WRONG: 8 — all
pre-existing at `359d287`, none introduced by this sprint.**

### CR5 — 6 protected refusals

CFO60, CFO61, CFO71, Q25A, Q25B, Q25C — all six unchanged. None began
answering.

### `funded_bridge` — ruling

**Do not narrow.** The brief conditions any narrowing on the Opus semantic
layer being *active*; it is not, so the replacement path for "What is the
weighted expected pipeline contribution?" and "Show funded vs pipeline
contribution." remains unreachable under the failing wording, and the condition
for narrowing is unmet. `funded_bridge` is unchanged.

---

## 4. Movement, enumerated

Every answer that moved across the 166 (75 bank + CFO 91), start to final:

| id | before | after |
|---|---|---|
| Q22B | REFUSED (`FALSE_REFUSAL`) | ANSWERED — both movements, matched to truth |
| Q22C | REFUSED (`FALSE_REFUSAL`) | ANSWERED — both movements, matched to truth |

**Nothing else moved.** 164/166 byte-identical, deterministic arm. There is no
Opus arm to compare: it could not be run.

Grade totals: `CORRECT 118 · TRUE_REFUSAL 15 · WRONG 7 · FALSE_REFUSAL 24 → 22 ·
NO_COMPUTABLE_TRUTH 2 → 4`. No failure was moved between categories to improve
a headline: the two that moved are answers that did not exist before, and their
figures were checked individually.

The 1,446-question surface was not re-run: nothing in either change touches
recognition or routing, the 278-module frozen manifest (which spans that
surface's owners) is exact, and no model arm exists to sweep. Recorded as not
measured rather than as a pass.

---

## 5. Residual blockers to production freeze

1. **The sprint's primary objective is unmet.** 24 CR4 questions still have no
   natural-language route to capabilities that sibling evidence proves exist.
   Blocked on API access, not on design.
2. **The constrained layer has never made a live call.** Its rules are verified
   by construction and by unit test; its behaviour against real model output is
   unmeasured. It stays off.
3. **8 WRONG answers** in the CR4 set, pre-existing — recognition failures
   producing the nearest expressible thing (F1) rather than a refusal. These
   are the most serious residual defects in the estate.
4. **7 WRONG** across the wider 166, pre-existing and separately tracked.
5. `funded_bridge` still shadows two questions whose successor calculation
   exists but is unreachable.
6. **F6 is open** — a config path resolved against the process cwd with the
   error swallowed. It has already cost one measurement in this programme.
7. The separator evasion and the five-reader consolidation remain open.

---

## Recommendation

# DO NOT FREEZE

Not because anything regressed — nothing did. The deterministic substrate is
harder to misuse than it was at `359d287`: a narrowing that runs is now legible
to the receipt, and the model can no longer write the governed contract on any
serving surface. Both were verified with zero blast against every protected
surface.

But the sprint's own hard gates cannot be evaluated. Items 10–15 of the
required report — successful invocation evidence, returned model ID, shadow
results, CR4 recovery counts — have no measurement behind them, and the
governing definition of success ("more reach + same truth + zero blast")
achieved *same truth* and *zero blast* with **no additional reach at all**.
Freezing now would freeze 24 known-unreachable capabilities and 8 known-wrong
answers, with the layer intended to address them never once exercised.

The block is an exhausted API spend allowance, not a design failure and not a
configuration one — `models.list` succeeds on the same key that every
`messages.create` is refused for. Access returns 2026-09-01. Phases 3 and 4
should run then, against these two commits, with no further code changes
needed to start them.
