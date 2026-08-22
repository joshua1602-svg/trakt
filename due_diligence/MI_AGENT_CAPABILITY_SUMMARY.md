# MI Query Agent — Capability Summary

Written for a commercial reader. Every rating names the measurement behind it.
Where a rating rests on cases this work constructed rather than on questions the
test corpora already contained, that is said in the line itself.

Measured on the Alderbridge book: **11,035 loans, £1,964,886,258.21 funded, 76
columns, three governed reporting periods.** A second synthetic book
(Kestrelmoor) is used wherever a result must be shown to be a property of the
product and not of one tape.

---

## 1. How a question becomes an answer

A person types an ordinary sentence. Four things happen to it.

**One — the sentence is read.** A deterministic reader matches the words against
a governed field registry: the list of things this portfolio can actually be
asked about, with each field's name, its synonyms, its unit, and which
statistics are permitted for it. This reader produces a *proposed calculation* —
a measure, any groupings, any filters, a period.

**Two — the language model is consulted, but only sometimes.** If the
deterministic reading validates cleanly and is unambiguous, the model is never
called and the question costs nothing. If the reading fails, or is uncertain, or
the sentence is layered ("X and Y", comparisons, conditionals), the model is
asked for its own reading. The model's proposal is then validated against the
same registry, and where the deterministic reading is sound it is preferred.
**The model is a fallback, not the front door.**

**Three — the calculation runs against the book**, and nothing else. There is no
free-text path to the data: every figure comes from a governed field, computed
by a governed statistic, over a stated population.

**Four — the answer is checked before it is shown.** A guard compares what the
sentence asked for against what the calculation actually did. If the question
asked for a narrowing, a grouping, a comparison or a projection that did not
reach the calculation, the guard blocks the answer and says so. It does not
quietly return a broader figure with a footnote.

**The governed registry is the spine.** It is what makes "the front book" mean a
specific population rather than a guess, what makes a percentage a
weighted average rather than a mean, and what lets the product say "this book
does not carry that field" instead of substituting a near-miss.

**The guards sit at the end**, between the calculation and the reader. Their
rule is *honour or clarify*: an answer is shown only if everything the reader
asked for was applied. Disclosure is not a substitute for honouring.

---

## 2. Capability by query type

### Point-in-time facts — **Works**
*"What is the total balance?", "How many loans?", "What is the average LTV?"*

**Evidence:** the calibration bank — 259 questions against the real Alderbridge
book — passes 259/259 with zero hard failures and zero known gaps. This is the
largest corpus-based measurement in the pack and it is not constructed.

**Works:** whole-book totals, counts, averages, weighted averages, minima and
maxima across the 76 governed fields.
**Does not:** any field the book does not carry (see refusals).

### Slice, dice and chart, including two dimensions — **Works**
*"Balance by region", "balance by LTV band and ticket size"*

**Evidence:** five phrasings of a two-dimension cross-tab, all correct — 50
populated cells across LTV band × ticket size, rendered as a heatmap and a
table, **reconciling to £1,964,886,258.21, the exact book total.** Two loans
with no LTV band are surfaced as "Unknown / Missing" rather than dropped.

**This rating rests on constructed cases.** The corpora contained no
two-dimension cross-tab of this shape; the five questions were written for this
measurement. What the corpora show is that the change broke nothing else.

**Works:** one or two dimensions, chart and table, with the totals reconciling.
Three or more dimensions produce a table rather than a chart rather than
silently dropping one.
**Does not:** combine a grouping with a time axis (see time series).

### Filtered and threshold questions — **Works**
*"LTV for loans over £150k", "borrowers aged 70 or above"*

**Evidence:** thirty comparator phrasings probed against both the component that
applies a filter and the component that discloses it. They now agree on 29 of 30
(previously 16 of 30). Five phrasings — *bigger than, larger than, higher than,
smaller than, lower than* — previously returned the **whole book** with no
disclosure; they now return the 5,857 loans the sentence asks for.

**The correction is real and measured on this book: 45.40% weighted LTV over
5,857 loans, where 43.15% over 11,035 was returned before.**

**This rating rests on constructed cases.** Two of 676 corpus questions contain a
newly-covered comparator, and both were written by this work.

**Works:** thresholds in either direction, negated forms ("no more than"),
ranges, and postfix forms ("above 50% LTV").
**Does not:** bind correctly when no currency marker is present and a different
measure is named earlier in the sentence — recorded, not fixed.

### Comparisons between populations — **Works on measured phrasings**
*"How has direct lending moved against acquired?"*

**Evidence:** the robustness bank's comparison family scores 12 of 12 correct on
**both books**, and **8 labelled claims are verified figure-by-figure against the
book** — each population's loan count and closing balance checked against a
recomputation. A control that trebles every figure fails all 8, so the check
discriminates.

**Works:** provenance splits, seasoning splits, and named dimension values,
across two periods.
**Does not:** one probe phrasing — *"How does the front book compare with the
back book?"* — is refused, because the word "compare" is read as a measure. The
same comparison asked as *"Compare direct and acquired balances"* answers
correctly. **This is a defect, found while writing this summary, and it is not
fixed.**

### Time series — **Works, narrowly. Never measured by a standing surface.**
*"Balance by month", "balance over time"*

**Evidence:** a direct probe run for this summary — not a corpus measurement, and
stated as such. *"Show me balance by month"* and *"balance over time"* answer
over the book's three governed periods. *"Funded balance by quarter"* is refused.
*"How has the balance moved over the last six months?"* is refused with the
reason: the window spans six reporting periods and the book carries three.

**Works:** a single measure over the governed periods, optionally scoped to one
population.
**Does not:** a time series broken down by a dimension. *"Balance by month by
region"* answers over time and **discloses that the region breakdown was not
applied.** That is honest, and it is a gap.

### Forward-looking and forecast — **Partially works, and no figure is verified**
*"What is the run rate?", "When will we reach £2bn?", "What is the forecast balance?"*

**Evidence:** the robustness bank's four forward-looking families. Run-rate and
threshold-date questions answer; forecast-balance questions refuse on this book
because it carries no governed pipeline source.

**No forward figure in this product has been verified against the book, and none
can be by the methods used here** — there is no recomputation for "when will we
reach £2bn". Six of the nine robustness families are unverifiable by
construction, and that is stated rather than worked around.

**Works:** run rate from funded growth; a milestone date by extrapolation.
**Does not:** any forecast requiring a pipeline, on a book without one.

### Concentration and limits — **Works, partially configured**
*"Are we within our concentration limits?", "Which limits are closest?"*

**Evidence:** a direct probe for this summary, plus the robustness bank's two
limits families (one 4/4 correct, one 3/4 with one avoidable refusal). The probe
returns: **8 tests passed, 0 warnings, 1 breach, 0 needing review, 3
unavailable**, and names the nearest test to its limit.

**Works:** evaluation against a configured limit schedule, with pass/warn/breach
status and the nearest test named.
**Does not:** three of the twelve tests cannot be evaluated on this book for want
of configuration or data. Broker concentration is refused outright — this book
has no broker field. **No limit figure has been verified against a recomputation.**

### Portfolio summary — **Works**
*"Portfolio summary", "book overview", "how is the book doing?"*

**Evidence:** eleven phrasings, all returning **one answer from one route** —
headline KPIs, regional exposure chart, and a provenance table. Before this work
the same eleven produced three different outcomes: the full summary, a
two-figure card, or a refusal.

**Works:** the book's overall position, however worded, including phrasings never
written down as examples.
**Does not:** a question naming a measure the registry lacks stays a refusal
rather than becoming a summary — deliberately.

---

## 3. What it refuses, and why that is the product

The agent is built to **decline rather than approximate**. Four kinds of refusal,
each with a stated reason:

**The book does not carry it.**
> *"'Broker' is not available in this dataset. The MI book for this client does
> not include broker_id."*

**The data does not span it.**
> *"You asked about the last 6 months, which spans 6 reporting period(s). This
> book carries 3 governed periods."*

**Part of the request could not be applied.**
> *"I understood that you asked for region, but that could not be applied to the
> calculation."* — and the figure is withheld, not footnoted.

**The question is genuinely ambiguous.**
> *"I could not tell how you meant ticket. Did you want the book split by it, or
> narrowed to one value of it? I have not answered over the whole book in the
> meantime."*

The final clause is the design. A system that answers over the whole book while
mentioning the gap in small print produces confident wrong numbers. This one
withholds.

**The cost of that stance, measured:** on a fifteen-question benchmark of
ordinary management questions, the agent originally produced **five refusals it
should not have** — questions it had already computed correctly and then
declined. All five are now closed. The benchmark stands at **fifteen correct,
zero wrong answers, zero refusals.**

---

## 4. What is verified, and what is not

Three qualifiers travel with the headline figures. They are not caveats added
late; each was earned by finding that a measurement meant less than it appeared.

**The 91% figure measures shape, not correctness.** The robustness bank's
historical 91.0% recorded that answers *arrived and were plausibly shaped* — that
a route claimed the question and the planned capabilities matched. It did not
check a single number. Proven by taking answers the grader called correct and
**multiplying every figure by three**: three of three still graded correct, with
loan counts trebled and dates mangled. Every figure derived from that bank
carries this qualifier.

**"32 correct" means 32 answers correctly shaped, of which 8 are figure-verified.**
Under the extended grader the 44-question bank reads 32 correct, 6 avoidable
refusals, 4 honest refusals, 2 disclosed-limitation. Of those, **8 labelled
claims are checked against the book**. Two families remain unverified; **24 of
the 44 are unverifiable by construction** — forward figures, run rates, limit
headroom.

**A tolerance against a large value set is not a check.** A first attempt to
verify the bank collected 322 book values and asked whether each answer quoted
*any* matching figure: 20 of 20 passed — and 20 of 20 still passed **with every
figure trebled**. One grader read no figures; the other read every figure; both
called a wrong answer correct, because matching against a haystack is not
checking a claim. That attempt was discarded.

**Constructed coverage.** Several ratings above rest on cases written for the
measurement rather than on corpus questions. Where a corpus result is clean, that
means *the change did not reach the corpus* — not that the corpus proves the
change correct. The corpora are enumerated from the same artefacts the product
reads, so they are blind in the same direction the product is.

**Do the fixes reach real users?** Measured directly, with the language model in
the loop: fifteen questions, five repeats each, 75 model calls. **All fifteen
returned the same answer with the model enabled as without**, and no
self-disagreement. The model is invoked for fourteen of the fifteen, and on every
one the deterministic calculation is what executes.

The limit of that claim, stated plainly: it holds for the fifteen measured
questions, all of which have a sound deterministic reading. **A question whose
deterministic reading fails validation would execute the model's proposal
instead, and no question of that kind has been measured.**

---

## 5. Known gaps

1. **Concentration limits are partially configured.** Three of twelve tests
   report unavailable on this book. Broker concentration cannot be evaluated at
   all — the field is absent.
2. **No segmented time series.** A measure over time can be scoped to one
   population but cannot be broken down by a dimension. The product discloses
   this rather than silently returning the ungrouped series.
3. **NNEG is absent from the data.** Zero of the book's 76 columns carry a
   no-negative-equity-guarantee field, so no NNEG question can be answered.
4. **Forecasting is gated on a pipeline source.** Forecast-balance questions
   refuse on books without one. Run-rate and milestone questions are unaffected.
5. **Receipt and disclosure backlog** — each recorded, none a wrong number:
   - a bare "70+" threshold is applied correctly but not disclosed;
   - the analytical path publishes no population figure, so the automated
     population check cannot run against it;
   - a threshold with no currency marker can bind to the wrong column when a
     different measure is named earlier in the sentence;
   - *"How does the front book compare with the back book?"* is refused because
     "compare" is read as a measure.

---

## Appendix — code references

Deliberately kept out of the body.

| Concept in the body | Where it lives |
|---|---|
| the deterministic reader | `mi_agent/llm_query_parser.py` |
| the model fallback and its gate | `parse_with_repair`, `zero_cost_first` |
| the governed field registry | `mi_agent/mi_semantics_field_registry.yaml` |
| the guards | `mi_agent/execution_receipt.py` (`reconcile_facets`) |
| routing to specialist capabilities | `mi_agent_api/chat_routing.py` |
| the service entry point | `mi_agent_api/mi_service.execute_governed_mi_query` |
| calibration bank (259) | `mi_agent/mi_calibration.py` |
| robustness bank (44 × 2 books) | `question_interpretation/run_robustness_deterministic.py` |
| routed surface (32) | `question_interpretation/routed_surface.py` |
| answer differ (729 × 5 surfaces) | `question_interpretation/answer_diff.py` |
| the fifteen shapes | `question_interpretation/shipped_shapes.py` |
| deterministic vs model comparison | `question_interpretation/llm_arm_comparison.py` |
