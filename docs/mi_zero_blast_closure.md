# MI Agent — Final Zero-Blast Commercial Closure

Baseline: `7cadf04`. Every production change below is one of the six defects
pre-registered for this sprint. Nothing else was touched.

The governing rule was **no blast**: a fix that moves a question outside its
pre-registered target surface is a defect in the fix, not a bonus. One change
was measured to blast and was reverted rather than narrowed until it passed.

---

## 1. Filtered historical comparison keeps its population, or refuses

**Defect.** "Which region added the most balance since last month for lump sum
loans?" returned the whole-book ranking, byte-identical to the same question
without the filter, with `ok=true`.

**First divergence — three of them, each closed where it happened.**

1. `_compare_recognizer` called `_parse_filters` without the value catalogue —
   the only one of five call sites passing neither `available_columns` nor
   `available_values`. Its numeric half worked and its categorical half did
   not: two halves of one filter pass reading different inputs.
2. The guard that stops a period phrase being read as a predicate tested the
   filter's **clause** span. A clause is as wide as the sentence the splitter
   did not split, so on a one-clause question the test asks "does this question
   mention a period at all?" — true of every comparison question by
   construction. It now tests the span of the **value**, which is what the
   guard's own docstring says it means to catch. "October To November" is
   spelled out of the period phrase and still drops.
3. The route never applied the population. It now does, in `build_snapshots` —
   the one place every compared snapshot is built, so a predicate cannot reach
   one date and miss the other.

**No new semantic owner.** The population comes from the existing chain:
`RowPredicateClaim` → `SELECT_POPULATION` → `Predicate` →
`governed_predicate_mask`, executed by the existing fail-closed
`population.apply_population`. The route reads no filter meaning of its own.

**Receipt.** `movement_receipt_for` published `predicates=()` unconditionally,
on the stated ground that this route narrows by scope and not by row predicate.
True before, false now — so it carries the predicates execution ran and the row
counts before and after.

**Proof (from the snapshot CSVs, not the agent).**

| question | opening | closing | predicate | result |
|---|---|---|---|---|
| lump sum, ranked by region | 345 rows, £87.3m | 396 rows, £105.4m | `erm_product_type = lump_sum` | Scotland £10.1m → £19.1m (+£9.0m, +89.8%) |
| drawdown, ranked by region | 255 rows, £62.1m | 244 rows, £66.7m | `erm_product_type = drawdown` | Wales £5.6m → £10.2m (+£4.6m, +81.7%) |
| LTV > 50, ranked by region | 159 rows, £40.6m | 144 rows, £37.1m | `current_loan_to_value > 50` | Scotland £4.5m → £6.6m (+£2.1m, +47.9%) |
| control (no filter) | 600 rows, £149.5m | 640 rows, £172.1m | none | Scotland £16.5m → £28.9m (+£12.4m, +75.0%) |

Every ranked group, both endpoints and both percentages match the tape.
Filtered answers identical to the unfiltered control: **0 of 6**.

A second bug was found inside the first: the route was handed
`semantics_context` (registry metadata, documented "empty today") where
`governed_predicate_mask` needs the MI semantics. Given the wrong dict it
compared a stored LTV **ratio** against 50 and narrowed every snapshot to zero
rows.

## 2. No implicit metric, no implicit comparison period

**Period.** The route's no-implicit-period rule already existed, gated on
`rank_intent.requested`, so the narrative half of the same route kept inventing
the same default. Every answer this route gives is a comparison between two
dates; which two is never optional here. The gate is removed, the rule
unchanged. "What has changed since last month?" names its window and is
untouched.

**Metric.** A series must plot something, so the parser substitutes the governed
balance when no measure is named. The defect was that it left no trace: "show me
the trend" and "show me the balance trend" produced identical specs. The spec
now records the substitution (`metric_defaulted`), the contract carries it as
`SubjectClaim.provenance` — reusing `SCOPE_PROVENANCES` rather than inventing a
second vocabulary — and the route refuses the **bare** case from contract fields
alone.

Bare means the question supplies nothing the measure could be determined from.
An explicit dataset, a named analytic or a grouping dimension all count, so
"show pipeline evolution by stage" and "show regional concentration evolution
over time" both default their metric and both still answer.

## 3. Pipeline evolution wording

`is_line` required `metric is not None` before a newly-carried axis could make a
line. That proxy asked "did the reader name a measure?" and answered it with the
parser's own default already applied, so it could not tell a **defaulted**
measure from one **determined** by the governed dataset the question names. It
is replaced by the rule above; both questions the proxy was measured against
still refuse, and now say which metric is missing.

## 4. Broker and product concentration — correct governance refusals, retained

The registry decides, and it was read rather than argued with:

| field | `portfolio_comparability` | whole-book concentration |
|---|---|---|
| `broker_channel` | `requires_scale_alignment` | not governed |
| `erm_product_type` | `requires_scale_alignment` | not governed |
| `origination_channel` | `comparable` | answers |
| `geographic_region_obligor` | `comparable` | answers |

Combining categories two originators spell differently would compare unlike
things. The per-book forms answer; the whole-book forms do not. **No capability
was added, and none should be.**

One real defect was found underneath: "Show broker concentration." resolves
`broker` to `broker_channel`, loses it to the qualifier rule (`broker` is also a
value of `origination_channel`), and reached the generic-concentration branch
with no dimension — where it was handed the **region** default and measured
geographic concentration for a question about brokers. Only the receipt's
substitution guard stopped that reaching the reader. A default is legitimate
where the reader named no axis; it is not a replacement for one we discarded.

## 5. Outstanding-offer value — genuine ambiguity, refusal retained

No field declares "value" as a synonym, and the governed pipeline extract
carries a `Loan Amount`, an `Estimated Value` and a `Property Value`. Both
readings are governed and present, so neither is chosen.

## 6. Risk limits no longer depend on the working directory

`Path("config")` was relative. From the repository root the limit questions are
answered from the 15 committed limits; from anywhere else they were refused with
"extraction required" — a false statement about the book, made silently and in
the safe-looking direction. Resolved against
`Path(__file__).resolve().parents[1] / "config"`, the mechanism
`business_semantics` and `portfolio_metadata` already use. Proven from three
working directories: byte-identical answers, **cwd-dependent results 0**.

## 7. Two bounded rendering defects

* The KPI headline stripped the aggregation suffix and threw it away, so
  `youngest_borrower_age_avg` read "Youngest Borrower Age: 74" beside a
  provenance line saying "Average Borrower Age" and a raw value of 74.33 — the
  portfolio average headlined as the youngest borrower in the book. The label
  now carries the aggregation, from `execution_receipt._MEASURE_AGG_WORDS`, and
  the field's governed business name. 16 answers changed label; **0 changed a
  number.**
* A single-row result is a single figure whatever chart was chosen for it.
  "What is the balance of offer stage cases?" said "Here is the bar for your
  query, covering 1 group" with the money only in the KPI artifact. The scalar
  branch is reached after the ranked lead, so no ranked answer changes. A
  per-group share no longer leads a grouped answer — a share of the only group
  is 100% by construction and read like a finding.

---

## The one thing that blasted, and was reverted

**Unknown attributive qualifier** ("how many platinum loans do we have?" → 640,
whole book, no mention that the qualifier was dropped). It is NOT about
"platinum": the prepositional form of the same question refuses correctly, and
"offshore" behaves identically. The defect is the **shape**.

Reporting the attributive residue generically — the words before the head noun,
minus request framing, aggregation words and analytical framing — was
implemented, and measured to break three questions that work today:

* "How many direct loans do we have?" (residue `direct`, a governed portfolio
  scope) → refused
* "What is the balance of offer stage cases?" (residue `balance offer stage`) →
  refused
* "Show balance by loan type." (residue `balance by`) → refused

Closing it safely needs the residue checked against the scope owner, the
dimension owner and the grouping markers — a recognition change wider than this
sprint authorises, whose failure mode is refusing questions that work. **The
change was reverted rather than narrowed until it passed.** The probe moved to
`migration_phase0/ROBUSTNESS_RESIDUALS.yaml`, where it is expected to FAIL and
cannot contribute a passing grade to a readiness run. "platinum" is not in the
registry and no synonym for it was added.

---

## Oracle corrections

These are corrections to the **measuring instruments**, reported separately from
the production changes so a reader can tell one from the other.

**`direct` was not an objective probe.** The collision sweep asked one literal
sentence under two incompatible hidden expectations — 146 loans via
`origination_channel`, 441 via `source_portfolio_type` — and scored the same
answer CORRECT under one and WRONG ADDITIONAL CLAIM under the other. The two
"wrong additional claims" this sweep has always reported were neither.
Ambiguous spellings are now computed from the book's own catalogue (a value two
governed fields both carry — exactly one exists: `direct`), each reading is
probed in words that mean only it, and the bare literal is kept once as an
ambiguity probe recording which reading policy resolves it to. Production
precedence is unchanged and is documented policy: bare `direct` resolves to
portfolio scope.

**Four invariants no oracle may bless**, applied after each bank's own grading
and able only to downgrade:

1. an explicit governed value delivered with no execution ledger proving it
   applied;
2. a filtered answer byte-identical to its unfiltered twin;
3. DISCLOSED without a disclosure — the old rule tested whether the answer
   *named* a period and a measure and called that a stated assumption;
4. `EITHER` read as "any `ok=true` answer is acceptable".

Each was a real miss. (1) and (2) caught the whole-book ranking the supplement
graded CORRECT for a lump-sum question, because the old check compared only the
leading group name — Scotland either way. (3) is proven pre/post against the
recorded baseline answers: "Show me the trend." and "What changed?" graded
DISCLOSED under the old rule and WRONG under the new one.

**"Capability exists for this question"** was asserted for every unregistered
refusal without consulting the thing that decides whether one exists. The oracle
now reads the registry's `portfolio_comparability` declaration.

**No frozen expectation was rewritten to make current behaviour look better.**
The frozen CFO bank already expected REFUSE for both ambiguous questions; this
sprint brought production into line with it, not the other way round.

---

## Verdict

**NOT READY FOR REAL-CLIENT-DATA ACCEPTANCE** — one specific blocker, everything
else closed.

All thirteen gates in §19 are met and every pre-registered defect is closed. The
verdict does not turn on the gates. It turns on the sprint's own closing
invariant:

> The MI Agent cannot silently answer an easier question than the one the client
> actually asked.

It measurably can. An attributive qualifier that no governed field claims is
dropped without a word, and the whole book is returned as the answer. Measured
in the vocabulary a real client would use:

| question | answer |
|---|---|
| How many platinum loans do we have? | 640 loans · £172.1MM |
| How many retirement interest only loans do we have? | 640 loans · £172.1MM |
| How many enhanced loans do we have? | 640 loans · £172.1MM |
| How many tier 1 loans do we have? | 640 loans · £172.1MM |
| How many prime loans do we have? | 640 loans · £172.1MM |
| What is the balance of legacy loans? | £172.1MM · 640 loans |
| …for Atlantis loans? (movement route) | the whole-book ranking |

Seven for seven, `ok=true`, with no mention that the qualifier was ignored. This
is the most common shape a first real-data session will produce, because a real
tape's vocabulary is largely ungoverned until it is mapped. The synthetic banks
do not show it because they ask in the vocabulary the book already carries.

It is a **wrong answer delivered with confidence** — first in the stated priority
order, ahead of misleading answers and embarrassing refusals.

**A green verdict was available and was not taken.** §10 authorised documenting
this as a residual; §19's gates are all met; and the probe now sits in a bank
that contributes no passing grade. Declaring READY on that basis would be a
verdict produced by moving a failure out of the denominator, which is the one
thing this programme's rules forbid above all others.

**What clears it.** One bounded piece of work: make the attributive form agree
with the prepositional form, which already refuses correctly — `"what is the
balance for platinum loans?"` refuses today. The two shapes of one question must
not disagree. It needs the residue checked against the scope owner, the
dimension owner and the grouping markers, and it must be measured against the
three questions that the first attempt broke:

* `How many direct loans do we have?` (441, the Direct book)
* `What is the balance of offer stage cases?`
* `Show balance by loan type.`
