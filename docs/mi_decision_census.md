# Multi-owner decision census

Base confirmed before anything below was measured: `git log` HEAD = `7c46f81`;
`git merge-base HEAD origin/claude/mi-analytical-capability-layer-vlkjfw` =
`4e051f3`; `4e051f3` and `28ece25` both ancestors of HEAD; working tree clean.

No fixes, no designs, nothing implemented.

---

## 1. The headline number

**14 decisions enumerated.**

| class | count |
|---|---|
| single | 5 |
| derived | 2 |
| agree-by-construction | 0 |
| **agree-by-maintenance (debt)** | **3** |
| **disagreeing (defect)** | **4** |

**The count is 3 debt, 4 defect.** Of the four defects, **three already have
identifiers** (B12, B14, and the routed/point-in-time role split) and **one is
new** (D10).

Read it as a small number, with the coverage caveat in §5 doing real work: the
decision list is authored, so this is a lower bound and the census's own biggest
blind spot.

---

## 2. The table

| id | decision | owners | class | id |
|---|---|---|---|---|
| D1 | where the subject ends and the first condition begins | 1 | derived | closed by Stage 3 |
| D2 | whether a named dimension is an axis or a filter | 3 | **disagreeing** | — |
| D3 | whether a phrase names a governed population | 1 | single | closed by `7c46f81` |
| D4 | what time grain the question requested | 1 | derived | — |
| D5 | what reporting window the question requested | 1 | single | — |
| D6 | whether a field is available in the book being asked about | 4 | **disagreeing** | **B14** |
| D7 | whether a requested grouping was actually applied | 2 | **disagreeing** | **B12** |
| D8 | whether a requested population was actually applied | 3 | **agree-by-maintenance** | — |
| D9 | what measure the question named, and whether it resolved | 3 | **agree-by-maintenance** | — |
| D10 | whether the question compares two things or reports one | 3 | **disagreeing** | **new** |
| D11 | which governed capability should answer | 1 | single | — |
| D12 | what grain the ANSWER is published at | 1 | single | — |
| D13 | whether the question asks for a series over time | **0** | single | B9 |
| D14 | what geographic scope the question named | 2 | **agree-by-maintenance** | — |

### The four defects

**D2 — axis or filter, across the two paths.** Owners: the parser (writes the
spec), `requested_dimension_terms` (reads the question text, consults neither
the spec nor the columns), and Stage 4's `_split_named_dimension_roles` (reads
the spec). The split reconciles the first two — **but it runs inside
`reconcile_facets`, so only on the point-in-time path.** Measured: the two paths
assert a different role on **37 of 693** questions. Reachable: yes. What a user
sees: a routed receipt asserting a breakdown the parser did not put on an axis —
which is the mechanism the false APPLIED rode in on.

**D6 — is the field available? (B14).** Four owners, all consulting *whichever
frame was loaded*, and the frame is chosen from the anticipated route before the
question is answered. Reachable: yes. What a user sees: *"'Seasoning Segment' is
not available in this dataset"* — true of the frame, false of the book.

**D7 — was the grouping applied? (B12).** Two owners with different evidence:
group keys and result columns on the point-in-time path, three tiers over the
artifact frame on the routed one. Reachable: yes, by construction —
*"geographic exposure by ltv bucket"* is certified by tier three. What a user
sees: a receipt certifying a breakdown by LTV bucket over an answer broken down
by ITL3 area.

**D10 — is this a comparison? (new).** Owners: `lending_windows_named` (two
windows named means compare — it gates whether a population is selected at all),
the cohort detector, and `plan.is_composite` (which gates whether the analytical
route claims the question). Measured over the 27 seasoning questions where both
have an opinion: **6 disagree**, all in one direction — the seasoning reader
calls it a comparison and the cohort detector does not. Reachable: yes.
What a user sees: nothing wrong today, because the disagreement resolves the
safe way on these questions — but the two owners gate *different things*, and
one gates whether the analytical route runs at all.

### The three debts

**D8 — was the population applied?** Three owners, three evidence sources
(`populationApplied`, the analytical plan's declared predicates, the executor's
`applied_filter_fields`), each scoped to a path the others do not run on, so
they cannot contradict one another today. Nothing prevents it: a route both
declaring `populationApplied` and publishing `applied_filter_fields` would be
adjudicated twice. **32c263a was this decision's literal arm disagreeing with
its governed arm**, before the governed arm was put first.

**D9 — what measure was named?** Three owners: the executor's registry
resolution, the detector's unresolved-slot reader, and route-fixed measures.
Different evidence, different times, each scoped so the others do not decide.

**D14 — what geographic scope was named?** Two owners: the detector matching the
frame's geo VALUES, and the parser matching the registry. A place name the frame
carries but the registry does not resolve — or the reverse — diverges.

### D13, the limiting case

**Zero owners.** Nothing decides whether a question asks for a series, so a
series question answered by a point-in-time capability is not detected. One
corpus question of 90. Recorded because an absent owner is where this census's
framing runs out, not because it is a disagreement.

---

## 3. Three corrections the census made to itself

Recorded because the number would otherwise have been wrong by a factor of
three, and because each is the same mistake: **probing two owners of DIFFERENT
decisions and calling the difference a disagreement.**

1. **D2, first draft: 76 disagreements.** It compared *what the parser put on an
   axis* against *what the detector recorded as named*. 62 of the 76 were the
   detector recording a dimension the parser had **dropped** — which is the whole
   reason the detector exists, so a dropped dimension can be disclosed rather
   than vanish. Reframed to ask both owners the same question — what role does
   the receipt end up asserting — the answer is 37, and it is a genuine
   path-dependent divergence.
2. **D3: 16 disagreements.** `_governed_population_predicates` answers *"which
   governed window does this facet's WORDING name"*, for comparison against a
   plan's declared population. That is D8's evidence, not an owner of D3. Probe
   removed; D3 is single-owner as of `7c46f81`.
3. **D9: 267 disagreements.** It compared *"did the parser set a metric"* against
   *"did the detector find an unresolved slot"*. Almost every one was a count
   question, which legitimately has neither. Probe removed rather than reported.

The brief asked for a small number reported as small and for distinct decisions
not to be merged. The pressure ran the other way: the instrument's natural
failure mode was to inflate.

---

## 4. What is new

| finding | status |
|---|---|
| D2's path-dependence | the mechanism behind the false APPLIED, but not previously stated as one decision with three owners |
| D10 | **new**, no identifier |
| D8, D9, D14 as debt | new as *classified debt*; none previously recorded |

---

## 5. Coverage — what this census could not reach

Per the standing rule, this qualifies the number above.

* **Decisions taken inside a route's execution.** `chat_routing` is thousands of
  lines of route bodies; a decision made and consumed entirely inside one is
  invisible here unless it surfaces in the spec, the facets or the artifacts.
  Both route-level defects found so far (B12, B14) surfaced only because
  something downstream disagreed.
* **The LLM parser arm.** Every probe is deterministic.
* **Any book but alderbridge.** A decision diverging only on different columns is
  invisible.
* **Owners I did not think to look for.** The decision list is **authored** —
  there is no mechanical way to enumerate "decisions" — so **this is a lower
  bound and its own biggest blind spot.** The earlier inventory missed the
  seasoning decision exactly this way, having counted 86 functions and 11 entry
  points without asking what decision each was making.
* **Divergence needing two owners on different inputs.** Owners are probed on the
  same question, so a defect caused by one owner seeing a different frame —
  B14's shape — is found only because it was already known.

---

## 6. What the number informs

Stated as the brief frames it, and not as a recommendation, which is not mine to
make:

**Four defects and three debts across fourteen decisions is a small count.** The
three closed decisions (D1, D3, and D4's consolidation) each went from several
owners to one without a rebuild, and each took one commit against a
pre-registered prediction. D2, D6, D7 and D10 are the same shape and there is no
evidence here that they need a different treatment.

The caveat that could change the reading is §5's fourth bullet: the list is
authored. If the true number is materially larger, this census cannot show it —
it can only show that fourteen decisions were examined and four diverge.
