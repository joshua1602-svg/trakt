# D2 — one owner for "axis or filter": pre-registered prediction

Written before implementing. Nothing below the rule in §6 was measured after
seeing a result; everything in §2 and §3 is BASELINE measurement of the code as
it stands at `761e22f`, taken before any change was designed.

Base: HEAD `761e22f`; merge-base `4e051f3`; `4e051f3` and `28ece25` both
ancestors; clean tree.

---

## 1. The decision, and its readers

**D2 — whether a named dimension is an axis or a filter.** The census
(`9ee0e5b`) classed it **disagreeing**, 3 owners, 37 of 693 questions diverging,
and framed the fix as *"Stage 4's split not reaching the routed path — finishing
work already begun."*

Four readers, not three. The census missed the fourth because it is a default
rather than a decision:

| # | reader | where | what it reads | what it decides |
|---|---|---|---|---|
| 1 | `_deterministic_parse` | parse | the sentence | writes `spec.filters` / `spec.dimensions` |
| 2 | `_split_named_dimension_roles` | `reconcile_facets` — **point-in-time only** | reader 1's slots | filter, axis, or unresolved |
| 3 | `reconcile_routed_facets` grouping branch | routed only | the ANSWER's axes | stamps a status, and by doing so ratifies a role |
| 4 | `requested_dimension_terms` | both | the question text | **raises every named dimension as `grouping_dimension`** — an axis by default, deciding nothing |

Reader 4 is the one that matters on the routed path: it asserts "axis" for every
dimension the question names, and reader 3 then stamps that assertion.

Unconsulted by all four: `question_interpretation.lexical.grouping_cut` and
`lexical.is_filter_subject` — **the declared lexical owners of the sentence's own
role markers.** `grouping_cut` already knows where "by X" opens a grouping
clause; nothing in the role decision asks it.

---

## 2. Baseline, measured

### 2.1 The 37 are all one kind of divergence

Comparing, per corpus question, the grouping field keys asserted after reader 2
against those asserted by reader 4 alone:

```
questions where the two paths assert a different role: 37
by surface: ere_mi_calibration_250 5 · ere_mi_questions 20
            nl_robustness_alderbridge 6 · nl_robustness_kestrelmoor 6
by target kind: {'unresolved_role': 37}
```

**Every one is `grouping_dimension` → `unresolved_role`. None is a population
reclassification.** So D2 is not "Stage 4's population split not reaching the
routed path". It is the CLARIFY default not reaching it. The census's framing is
wrong in its subject, and I am recording that before acting on it rather than
after.

### 2.2 Where the 31 unique questions actually go

Run through `execute_governed_mi_query`, deterministic arm, Alderbridge:

```
route: (point-in-time) 16 · evolution 4 · geo_exposure 2 · risk_limits 3
       analytical_composition 6
```

* **16 never route.** Reader 2 governs them and reader 4's assertion is
  discarded — the paths do not both run, so there is no divergence to see. All
  16 return `ok=False` with **no receipt facets at all**: they refuse upstream of
  the guard, at measure or capability resolution (*"'profitability' is not a
  governed measure"*, *"'high ltv' does not state a threshold"*). Reader 2's
  clarification is never reached on any of them.
* **15 route.** Reader 2 never runs. 11 answer `ok` with the dimension stamped
  APPLIED; 4 refuse with it LOST (`evolution`, which drops the breakdown — B9's
  territory, not D2's).

**Consequence: copying reader 2 to the routed path would move 10 currently
correct answers to CLARIFY and fix nothing.** That is the `32c263a` shape — a
role default falling to the side that declines to answer — and it is the reason
this document exists rather than a patch.

### 2.3 The live defect next door is not the role

Post-hoc classification of every routed `grouping_dimension: applied` on the
343-question corpus, against the tier ladder in `reconcile_routed_facets`:

```
tier 3b (an axis exists, no name matches it): 13
no axes at all:                                1
```

Tier 3b is the residue the code names in its own comment: *"THAT a breakdown
happened is proven, WHICH one is not."* Of the 13:

* **9 are correct.** `geo_exposure` publishes `area`/`code` and really is
  broken down by region; `concentration_analysis` publishes `category` and
  really is broken down by the requested dimension. Neither ever matches by
  name, so all their grouping claims land here.
* **4 are false.** `risk_limits` publishes `actual · headroom · limit · source ·
  status · test` — a limit-test table with **no dimension axis whatever** — and
  the receipt certifies a breakdown regardless:

| id | question | receipt claims | answer is broken down by |
|---|---|---|---|
| `risk_limits_005` | What is the largest geographic concentration? | `collateral_geography: applied` | nothing |
| `risk_limits_006` | Show geographic concentration against limits. | `collateral_geography: applied` | nothing |
| `risk_limits_010` | Are any regional limits breached? | `collateral_geography: applied` | nothing |
| `risk_limits_013` | Show concentration limit status. | `account_status: applied` | nothing |

`risk_limits_013` is the plainest: the word *status* in *"limit status"* resolved
to `account_status`, nothing gave it a role, and the receipt vouches for an
account-status breakdown of an answer reading *"8 passed, 0 warning(s), 1
breach(es)"*.

**These four are D7's defect (B12 — whether a requested grouping was actually
applied), not D2's.** The role decision did not make the false claim; the
evidence rule did. Separating them is deliberate: D7 is next in the work order,
and folding its fix in here would make one commit answer two questions.

---

## 3. What this means for D2's scope — stated plainly

**D2 as specified has no live consequence on either corpus.** On the 16
point-in-time questions the receipt is never reached; on the 15 routed ones the
right answer today is the one reader 3 gives. There is no user-visible movement
available to a correct D2 fix on the corpora we have.

That is not a reason to skip it. Two things follow instead, and both are in
scope:

1. **The consolidation still lands**, because the disagreement is real and
   currently masked only by routing — *a defect prevented by routing, not an
   absent one*, which is the accidental safety this programme has now declined
   to rely on three times.
2. **The corpora do not exercise it, so the coverage is constructed.** A clean
   surface is evidence about coverage before it is evidence about the product.
   This commit ships cases that reach the role decision on the ROUTED path,
   because no case in 693 does.

---

## 4. The class, and the illustration

**The class:** *every reader that needs to know whether a named dimension is an
axis or a filter consults one owner, and the owner consults every source that
can settle it — including the sentence's own role markers, whose lexical owner
already exists and is read by nobody.*

**The illustration:** the 37 questions in §2.1, and the four false
certifications in §2.3. The 37 are the measured extent of the disagreement, not
its definition; the four are named to be handed to D7, not fixed here.

The distinction matters because of the standing rule earned in `7c46f81`: **a
scoping query built around one reader cannot bound a change that replaces the
arrangement of readers.** The 37 came from a reader-2-shaped query. The class
covers questions that query cannot see — any question where readers 1 and 4
would disagree on a path where reader 2 does not run. Movement outside the 37 is
therefore **expected and not a stop condition**; movement outside the CLASS is.

---

## 5. Every place the owner's answer arrives — and what was already deriving it

Required before the consolidation lands.

| # | site | derives it today? | after |
|---|---|---|---|
| 1 | `_split_named_dimension_roles` (point-in-time) | **yes** — reader 2 | becomes the owner's only caller on this path; derives nothing |
| 2 | `_guard_routed_answer` (routed) | **yes, by default** — reader 4 raises every named dimension as an axis | consumes the owner's answer |
| 3 | `reconcile_routed_facets` grouping branch | **yes, implicitly** — stamping a grouping ratifies the role | keeps stamping STATUS; the ROLE is settled before it runs |
| 4 | `population_facets(spec, semantics)` in `_guard_routed_answer` | **yes**, for filters — raises populations from `routed["spec"].filters` | **collision hazard, see below** |
| 5 | `reconcile_population` | no — stamps status only | unchanged |
| 6 | `seasoning.resolve_population_predicate` | **yes**, for the seasoning vocabulary — and it is the OWNER of that (`7c46f81`) | the D2 owner must run downstream of it and must not re-decide any phrase it took |
| 7 | `lexical.grouping_cut` / `is_filter_subject` | owns the sentence marker; **read by nobody** | becomes the owner's source 3; unchanged itself |
| 8 | `answer_type.subject_side` | consumes `grouping_cut` | unchanged — the owner reads that decision, it does not alter it |

### The collision at site 4, named in advance

If the owner emits a `KIND_POPULATION` facet on the routed path, it meets
`population_facets(spec)`, which raises populations from the ROUTE's spec. Both
label as `f"the population {predicate.describe()}"`, so the existing
`(kind, field_key, label)` dedupe catches them **only when the two specs agree**
— and they need not, because the owner would read `parsed.spec` while the ledger
reads `routed["spec"]`.

Worse, a survivor is pulled out by
`_population = [f for f in facets if f.kind == KIND_POPULATION]`, skips
`reconcile_routed_facets` entirely, is never handed to `reconcile_population`,
keeps its LOST default and **refuses the answer**. That is `e35a01b`'s failure
mode — a reclassification into a kind with no receiver — arriving on the routed
path, and it is the same duplicate-raise that was live for ten minutes in
`7c46f81`.

Both are checked by construction in this commit, not by inspection.

---

## 6. The rule

One owner. Sources consulted in this order; the first that answers wins:

1. **reader 1's filter slot** — a candidate key in `spec.filters` → FILTER.
2. **reader 1's axis slot** — a candidate key in `spec.dimension(s)` → AXIS.
3. **the sentence** — the term's occurrence lies at or after
   `lexical.grouping_cut` → AXIS; `lexical.is_filter_subject` over the term's
   span → FILTER. *(New. This is the source nobody consulted.)*
4. **the book** — no candidate key is a column → AXIS, so the existing
   `KIND_GROUPING` branch stamps UNAVAILABLE and the reader is told the thing
   they can act on. Unchanged from reader 2's `_unexpressible` rule.
5. **otherwise** → UNRESOLVED.

Reader 3 is **not** a source. Execution evidence settles STATUS, never ROLE:
letting the answer's axes decide what the question meant is how the false
APPLIED at §2.3 would be laundered into a role, and it is D7's to fix.

On the routed path the owner's UNRESOLVED answer leaves the facet **as it is
today** — a grouping, stamped by reader 3. It does not clarify. Deciding what a
routed answer owes an unresolved role requires evidence D7 is about to repair,
and taking it here would move ten correct answers to CLARIFY (§2.2).

**So on the routed path this commit changes exactly one thing: a named dimension
reader 1 positively slotted as a FILTER stops being asserted as a breakdown.**

---

## 7. Pre-registered prediction

### 7.1 What moves

**Nothing, on any of the three surfaces, on either book.**

Stated as the falsifiable claim it is:

* answer text: **343 of 343 identical**
* robustness: both books unchanged, `32/10/2`
* routed surface: **13 of 13**
* lexical decisions: **693 of 693** — the owner READS `grouping_cut`; it must not
  move it
* seasoning families, by name: **Q1 4, Q7 4, Q8 12, all CORRECT**, both books

The prediction is "nothing moves" because §2 measured why: on the 16
point-in-time questions the receipt is unreachable, and on the 15 routed ones
source 3 or the existing grouping default gives the same answer reader 3 already
ratifies. Source 3 newly settles 13 questions carrying "by X" — every one of
them to AXIS, which is what reader 4 already defaulted them to.

### 7.2 What must move — the constructed coverage

Because 693 questions do not reach it, four cases are constructed and added to
the routed surface. Each fails on `761e22f` or is not exercised there:

1. a routed question naming a dimension reader 1 slots as a **filter** — the
   receipt must carry `row_population`, not `grouping_dimension`;
2. the same, where `parsed.spec` and `routed["spec"]` carry **different**
   filters — the collision at §5 must not produce two facets;
3. the same, where the resulting population reaches
   `reconcile_routed_facets` — it must be stamped, not left LOST;
4. a routed question naming a dimension with **no** role from any source — the
   receipt must be unchanged from today, proving §6's last paragraph rather than
   asserting it.

### 7.3 What must not move

1. **No answer text changes.** Any change stops the work.
2. **No verdict changes on any surface.**
3. **No lexical decision changes.** 693 of 693.
4. **The seasoning families stay at their by-name counts**, both books.
5. **The stamping matrix stays at 0 live holes.**
6. **`RECLASSIFICATION_TARGETS` stays closed** — if the routed path can now
   produce `KIND_POPULATION` by reclassification, its receiver must be proven on
   that path, and `test_reclassification_targets.py` must be the thing that says
   so.

### 7.4 Stop conditions

Stop and report; do not absorb.

* any answer moving on any surface;
* any verdict moving on any surface;
* any lexical decision moving;
* a live hole appearing in the stamping matrix;
* a duplicate facet on any receipt — the `7c46f81` ten-minute defect recurring;
* any facet reaching a reconciler with no branch to receive it.

### 7.5 Acceptance

* one owner; readers 2, 3 and 4 each demonstrably consume rather than derive,
  and §5's table is confirmed line by line;
* the sentence's role markers are read by the role decision for the first time;
* four constructed cases exercise the routed role decision that 693 corpus
  questions do not;
* all three surfaces, deterministic arm, both books; seasoning by name;
* the four false certifications at §2.3 are handed to D7 **named and
  unfixed**, and D7 is where they are fixed.

---

## 8. Corrections recorded here, not afterwards

1. **The census's D2 framing is wrong in its subject.** It reads *"Stage 4's fix
   not reaching the routed path"*, implying the population split. All 37
   divergences are the CLARIFY default, and zero are populations. The count is
   right; the diagnosis attached to it was not.
2. **The census listed 3 owners; there are 4.** `requested_dimension_terms`
   decides nothing and therefore defaults to "axis" for every named dimension —
   which on the routed path is the operative decision. A reader that defaults is
   a reader.
3. **The census said "Reachable: yes" and cited a routed receipt asserting a
   breakdown the parser did not put on an axis.** That receipt is reachable and
   is wrong four times over — but the wrong claim comes from the tier-3b
   evidence rule, not the role decision. D2 is reachable in its READERS and
   currently unreachable in its ANSWERS.
