# Phrasing reachability — are the absences capability gaps or routing gaps?

> **UPDATE — superseded in two places.**
> **(1)** The third routing gap below (the P0 refusal) is FIXED — see
> `docs/mi_p0_segment_pair_refusal_fix.md`. On the declared 29 the strict count
> is now **2**, not 3, and T8 is PROVEN.
> **(2)** The phrasing bank has been widened from 29 to 61 before P1 scoping, as
> the caveat at the foot of this document recommended. The wider measurement
> found **more** work, not less: 14 of 61 deliver, 8 strict routing gaps, 39
> capability gaps, and T3–T6 stand at **0 of 30**. See
> `docs/mi_phrasing_bank_widened.md`.
>
> The reasoning below — and the strict-vs-loose distinction, now a standing
> definition in `docs/mi_sibling_rule.md` — is unchanged and still governs.

**The question this answers:** of the 29 time-series phrasings, how many that
fail have a **working sibling** — the same request, reachable under a different
wording? That number decides whether P1 is building a capability or fixing a
route, and the two are not the same project.

Measured on both books, both arms, all four runs identical.
Instrument: `question_interpretation/mi_capability_recontent.py`.

---

## The headline

```
  phrasings that DELIVER                                       8 of 29
  phrasings that do NOT deliver                               21

    ROUTING gaps    — the SAME request works in other words     3
    CAPABILITY gaps — no wording reaches this request          18
```

**Three of 21, or 14%.** The absences are overwhelmingly capability gaps.
**P1's scope survives** — but not untouched, because the three are not randomly
distributed and one of them is a P0 refusal.

---

## Why the strict number is the honest one

A looser reading gives **7 of 21**, and that number is wrong to plan from.

The eight shapes group phrasings by SHAPE, not by REQUEST. `Which region has
grown fastest?` and `Which LTV band moved most between periods?` are both shape
7, but they are not the same question in different words — one asks about
regions, the other about LTV bands. Counting the LTV phrasing as *reachable*
because the region phrasing works would claim a routing fix reaches something it
cannot.

So a sibling counts only when it shares the shape **and the target** — the thing
the answer must be broken down or filtered by. That is the difference between
7 and 3, and it is the difference between "half the absences are routing" and
"a seventh of them are".

| grouping | routing gaps | reads as |
|---|---|---|
| same shape | 7 of 21 | a third of the absences are routing — **overstated** |
| same shape **and same target** | **3 of 21** | the absences are real capability gaps, with three exceptions |

---

## The three routing gaps, in full

All three are **shape 7 and shape 8** — the two shapes this reissue moved from
ABSENT to PARTIAL. That is not a coincidence: a shape with no working phrasing
cannot have a routing gap by definition, so routing gaps only ever appear where
something already works.

### 1. `Which region grew the most over the last three months?` — refuses

Reached by: **`Which region has grown fastest?`** — which returns rank +
`category` (12 UK regions) + `start_value`/`end_value`.

Same request, same target (`region`), same shape. The working phrasing carries no
time window; the failing one names *"the last three months"*.

### 2. `Rank regions by balance growth over time` — refuses

Reached by the same working phrasing. Same request, same target.

### 3. `How have direct and acquired balances moved over the periods?` — refuses

Reached by: **`Compare balance over time for direct and acquired`** — which
returns `population` carrying `Direct`/`Acquired` with `prior`/`current`, and in
prose: *"Across 2026-04-30 → 2026-06-30, Direct, 7,126 loans: £1.36bn → £1.39bn
(+£21.5m). … Acquired, 3,909 loans: £568.3m → £579.4m (+£11.1m)."*

**This one is a named P0 refusal**, and it is the fourth instance of one phrasing
masking a gap another exposes — the first in the time-series family.

---

## What this one costs, specifically

The P0 refusal is *honest about itself*:

> *"I understood that you asked for Direct and Acquired tracked separately, but
> that could not be applied to the calculation."*

That statement is true of the route it took (`cohort_progression`) and false of
the product. Another wording reaches `analytical_composition`, which serves the
request completely. **The guard is not over-refusing a thing that does not
exist — it is refusing a thing that exists one route away.**

That is the brief's second divergence category — *the guard refused a proposal
the route would have handled* — occurring **between phrasings rather than between
arms**. The Gate could not surface it, because the Gate compares the LLM arm
against the deterministic arm and both arms refuse this phrasing identically.
Only comparing phrasings against each other exposes it.

---

## The 18 capability gaps

| shape | target | phrasings | reachable by any wording? |
|---|---|---|---|
| T2 | threshold filter | 3 | **no** — every numeric/threshold filter refuses |
| T3 | region | 3 | **no** |
| T3 | LTV band | 1 | **no** |
| T4 | region | 2 | **no** |
| T4 | LTV band | 1 | **no** |
| T5 | region | 3 | **no** |
| T6 | region | 3 | **no** |
| T6 | LTV band | 1 | **no** |
| T7 | LTV band | 1 | **no** — the region phrasing works, the LTV band one does not |

Two clean statements fall out:

* **The per-period breakdown family (T3–T6) is genuinely absent.** 14 phrasings,
  every one refusing, no wording reaching any of them. No routing fix touches
  this; it is a build.
* **Threshold filters over time are genuinely absent** (T2, 3 phrasings). Only a
  seasoning-population scope works.

And one uncomfortable one:

* **T7's LTV-band ranking fails while its region ranking succeeds.** The ranked
  movement capability exists and is wired to one dimension. That is a
  *dimension-coverage* gap inside a working capability — cheaper than a build,
  more than a route.

---

## What this means for P1

**The scope does not need rewriting, but it needs three amendments.**

1. **The core of P1 stands.** 18 of 21 absences are real, and 14 of them are the
   single coherent per-period-breakdown family. Nothing here suggests that work
   is unnecessary.

2. **Three phrasings should come out of the build and into a routing fix.** Two
   regional-ranking phrasings and one cohort-comparison phrasing already work
   under other words. Building them would be building something that ships.

3. **The P0 refusal on `How have direct and acquired balances moved over the
   periods?` should be treated as a routing defect, not as evidence of an
   absence.** It currently tells a user the product cannot do something it can.
   Of the three, this is the one with a live client-facing cost, because the
   refusal is confident and wrong about the product.

**One caveat on the number.** 3 of 21 is measured against the 29 phrasings the
surface declares — four or fewer wordings per shape. It is a floor on routing
reachability, not a ceiling: a wording nobody has tried may reach a request
counted here as a capability gap. The 18 are "no *declared* wording reaches
this", and widening the phrasing bank is the cheapest way to find out whether
that holds — worth doing before committing P1's estimate, since every phrasing
that turns out reachable moves work out of the build.
