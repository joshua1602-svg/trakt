# The widened phrasing bank — reachability re-measured before scoping P1

**61 phrasings, both books, both arms identical.** The declared 29 plus 32 new
ones written as a lender would type them.

The declared bank is four-or-fewer phrasings per shape, drawn from the same
vocabulary the system was built against. Reachability measured on it is a
**floor**. This widens it before P1's estimate is committed, because every
wording that turns out reachable moves work out of the build.

The 32 were written before they were run, and **none was adjusted after seeing
its result** — a bank tuned against its own outcome measures nothing.

---

## The headline: natural phrasing does WORSE, not better

```
                          delivers        rate
  declared 29               9 of 29        31%
  widened 32                5 of 32        16%
  combined 61              14 of 61        23%
```

**The widened bank did not move work out of the build. It found more of it.**

| | declared 29 | combined 61 |
|---|---|---|
| strict ROUTING gaps | 2 | **8** |
| strict CAPABILITY gaps | 18 | **39** |

Routing gaps roughly held their share (10% of non-delivering, against 15%
before). The absences are still overwhelmingly real.

---

## Shape ratings move DOWN under lender phrasing

| shape | declared 29 | **combined 61** | |
|---|---|---|---|
| T1 | PROVEN | **PARTIAL** | ⚠ 1 of 4 lender phrasings works |
| T2 | PARTIAL | PARTIAL | |
| T3 | ABSENT | ABSENT | 0 of 8 |
| T4 | ABSENT | ABSENT | 0 of 7 |
| T5 | ABSENT | ABSENT | 0 of 7 |
| T6 | ABSENT | ABSENT | 0 of 8 |
| T7 | PARTIAL | PARTIAL | |
| T8 | **PROVEN** | PARTIAL | PROVEN on all 3 declared, after the P0 fix |

### The finding that matters: T1 is fragile

T1 — *metric × time*, the simplest shape on the surface and PROVEN on all four
declared phrasings — answers **one of four** lender phrasings:

```
OK   give me the balance trend                             evolution
✗    what has the book done over the last few periods      route=None
✗    how is the loan book tracking month to month          route=None
✗    outstanding balances by period                        route=None
```

These are ordinary requests for the most basic capability the surface has, and
three of them reach no route at all.

**This is not P1's build.** The capability exists and is proven. It is a
recognition gap sitting on top of a working capability — cheaper than a build and
invisible to any measurement that only uses the declared vocabulary. It is
precisely what widening the bank was for.

---

## What this confirms for P1

**The per-period breakdown family is confirmed absent, and more strongly than
before.** T3–T6 now stand at **0 of 30 phrasings** delivering, across two
vocabularies:

| shape | declared | widened | total |
|---|---|---|---|
| T3 | 0 of 4 | 0 of 4 | **0 of 8** |
| T4 | 0 of 3 | 0 of 4 | **0 of 7** |
| T5 | 0 of 3 | 0 of 4 | **0 of 7** |
| T6 | 0 of 4 | 0 of 4 | **0 of 8** |

No wording in either bank reaches any of them. **P1's core scope is not reduced
by this exercise; it is confirmed.**

Several widened phrasings do reach a route and still fail — `how are the regions
trending` reaches `evolution`, `how has each region moved over the periods`
reaches `period_change_analysis`, `movement by LTV band period on period` reaches
`funded_bridge` — each returning something that does not carry the requested
breakdown, and refusing honestly. The routes exist; the per-period segmented
answer does not.

---

## The eight strict routing gaps

Three phrasing families, none of them in T3–T6:

| shape | target | failing phrasings | reached by |
|---|---|---|---|
| T1 | none | 3 | `Show me balance by month` |
| T2 | seasoning | 1 | `How has balance moved over time for the front book?` |
| T7 | region | 2 | `Which region has grown fastest?` |
| T8 | seasoning / source_split | 2 | the working T8 phrasings |

**Every one is a recognition problem on a capability that already ships.** None
is build work. They belong in a routing/NLU workstream, not in P1.

---

## Two things this does not establish

**It is still a floor.** 61 phrasings is wider than 29; it is not the language.
A wording nobody has typed may still reach a request counted here as a capability
gap. The claim remains *"no wording in either bank reaches this"*.

**The widened bank is not a benchmark of user success.** It says nothing about
how often a real user picks a working phrasing. A 23% delivery rate across 61
phrasings and a 23% chance a lender gets an answer are different statements, and
only the first is measured.

---

## Reproducing

```
TRAKT_RUNTIME_MODE=development \
  python -m question_interpretation.mi_capability_recontent \
    --book alderbridge --bank combined
```

`--bank declared | widened | combined`. The bank is declared in
`question_interpretation/mi_phrasing_bank.py`; reachability is grouped by the
standing sibling rule (`docs/mi_sibling_rule.md`) — same shape **and** same
target.
