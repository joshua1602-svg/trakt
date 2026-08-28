# Final residual-defect sprint — freeze decision

Start `c8089f9`. End: this commit. Three commits, all bounded.

| commit | residual |
|---|---|
| `c75c556` | Phase 1 — a field the reader has already placed is not also an axis |
| `4652d6a` | Phase 3 — Q10B independent truth, plus two instrument defects it exposed |
| this | the report |

**Production files changed:** `question_interpretation/claim_merge.py` (+35).
Measurement/oracle: `migration_phase0/pack_grader.py` (+16),
`MI_FINAL_LIVE_DATA_READINESS.json` (one line), plus tests and two documents.

A fourth correction was built, measured, and **reverted** — see Phase 4.

---

## Phase 1 — the "three cases" were not three

Reproduction over 6 runs each reframed the set before any code was written:

| case | answered | disposition |
|---|---|---|
| total balance for **North** loans | 5/6 | genuine defect — **corrected** |
| total balance for **drawdown** loans | 6/6 | same defect, rarer — **corrected** |
| How do the **Direct and Acquired** portfolios differ? | 0/6 | **R1 correct refusal** |
| **pipeline by stage for broker Alpha** | 0/6 | **R1 correct refusal** (Phase 2's fourth) |

### Common first divergence, for the two that were real

```
"What is the total balance for North loans?"
  DETERMINISTIC  filters={geographic_region_obligor: "North"}   dims=[]
  PROPOSAL       dimension · "region" · span "North loans"
  BINDING        -> geographic_region_obligor
  MERGED         dimensions += [geographic_region_obligor]      (slot was empty)
  DIVERGENCE     the field the reader NARROWED on is also written as an AXIS
  REFUSAL        "parsed dimension(s) neither applied nor rejected"
```

### The governed operation was sufficient

Yes. The deterministic contract already places the field as a row predicate
with `explicit_user` provenance. Offering the same governed field as an axis is
a **second placement of a concept already placed**, and that is readable from
the contract with no wording.

**One direction only, and the asymmetry is measured.** The mirror — a field
held as an AXIS with the model supplying the value — is the recovery this arm
exists for: "Balance by region for London loans." parses as an axis with the
scope LOST, the deterministic path refuses saying so, and the model restoring
`London` answers it (£22.4m, 83 loans). Declining that direction too would take
back seven correct answers to buy two. Both pinned by test.

### Phase 2 — the fourth movement: R1, correct refusal

`pipeline by stage for broker Alpha`. The governed pipeline extract holds
exactly **one** broker, `Broker Synthetic`; "Alpha Network" exists only in the
funded book. The deterministic answer **silently dropped the named broker and
returned the whole pipeline** as a 5-group heatmap. The Opus refusal — *"No
loans in this book match that filter (Broker) … I have not returned a
whole-book figure in its place"* — is correct. **No production fix.**

`How do the Direct and Acquired portfolios differ?` is the same class: the
deterministic answer is a whole-book count (graded WRONG). **R1.**

---

## Phase 3 — Q10B: a real oracle, and two broken instruments

Truth computed from the governed pipeline extract with pandas, neither
implementation as oracle, documented before classification
(`MI_Q10B_TRUTH.md`):

* a governed size band exists — `ticket_bucket`, derived from balance;
* **"size" is not in the registered dimension vocabulary**, which is why the
  deterministic parser kept `stage` and dropped `size`;
* stage × size = **8 non-empty groups**; stage alone = 5.

| | output | verdict |
|---|---|---|
| deterministic | 5 groups, by Pipeline Stage | **WRONG / SILENT** |
| Opus arm | 8 groups, by Ticket Size and Pipeline Stage | **CORRECT**, 6/6 runs |

Two instrument defects surfaced on the way:

1. with `cells: 8` recorded, the grader **still passed** the five-group answer,
   because the check accepted the number appearing anywhere in the prose and
   the answer says "8 loans". The artefact is now the evidence wherever one was
   rendered. No other `cells` truth changed grade;
2. that fix changed nothing until it emerged that **every grade this programme
   has published came from a scratch copy of the grader**, byte-identical apart
   from its docstring. The copy is now an import of the reviewable file. Third
   instance of the two-producers pattern.

Q10B deterministic moves CORRECT → WRONG **with a byte-identical answer**: a
reclassification by a fixed instrument, not a regression. No production code
was changed for Q10B; closing the gap needs "size" in the vocabulary, which is
grammar and out of scope.

---

## Phase 4 — neither can be neutralised. This decides the sprint.

### Q19A — built, measured, REVERTED

`cohort_progression` tracks a **closed February cohort** (375 → 258 loans) and
reports

> *"Funded balance for Direct: tracked across 5 reporting period(s)
> (2026-02 → 2026-06) **down**."*

for a book that grew in **every** period — £81.7m → £117.4m, +£35.6m
(independently computed from the five governed snapshots). Both numbers are
honest; the sentence answers a question nobody asked and contradicts the one
they did. `period_movement` owns it and gets it right.

I applied the existing requirement-coverage invariant at the route seam.
**It worked**: Q19A became CORRECT, `period_change_analysis`, delta
12,366,371.40 matched, and it was the *only* movement on the 166.

**The frozen manifest then caught it at 86 names.** The gate had taken a
legitimate cohort question on another fixture — `how has funded balance evolved
for the direct book`. The intent owner reads the two **identically**: same
families (`MOVEMENT_TREND`), same operations (`DELTA`), same requirements
(`period_comparison`). Separating them needs the wording, which is forbidden.

**Reverted. Frozen manifest back to exactly 85.** Q19A: **WRONG 6/6.**

### Q04C — no governed signal exists

Population and data are correct: the artefact holds exactly the right 24 loans
summing to exactly £7,201,378.77, the independent truth, disclosed as
*"Loan-level Balance · London · Borrower Age > 75 · Source Portfolio in
direct_001"*. The fault is the output shape — `aggregation=loan_level` where
the reader asked for a total.

Its correct sibling Q04B has an **identical contract** apart from that one
field. `statistic_named` returns `None` for both: **"total" is not a governed
statistic**, and `loan_level` is a deliberately protected analytic mode. The
only difference is the word "Show". Correcting it, or detecting it well enough
to refuse, needs grammar. **WRONG 6/6.**

---

## Repeated-run table (6 independent Opus invocations each, final build)

| control | result |
|---|---|
| Q23A · Q23C · CFO74 · CFO63 · CFO65 | **CORRECT 6/6** each |
| Q01C · Q02B · Q03A · Q03C · Q05C · Q17C | **CORRECT 6/6** each |
| **Q16B** | **CORRECT 5/6, WRONG 1/6** |
| Q04C | **WRONG 6/6** |
| Q19A | **WRONG 6/6** |
| Q10B | CORRECT 6/6 |
| Q22B · Q22C · Q10A | answered 6/6 |
| Q25A · Q25B · Q25C | refused 6/6 |
| total balance for North / drawdown loans | answered 6/6 (corrected) |
| Direct-and-Acquired · broker Alpha | refused 6/6 (correct refusals) |
| "How many acquired loans do we have?" | answered 6/6 |
| "Show the loans included in Unknown / Missing age." | **refused 3/6, answered 3/6** |
| must-refuse ×3 | **TRUE_REFUSAL 6/6** each |

**Q16B is a gate breach.** In run 4 the model simply did not propose
`drawdown`; nothing was applied, and the answer reverted to the whole book —
42 groups instead of 39, with the scope silently absent. That is the
deterministic wrong answer, not a new one, but it means **the recovery is
stochastic**: this arm's seven proven recoveries hold only when the model
proposes the concept, and one run in six it did not.

**"Show the loans included in Unknown / Missing age."** answers 3/6 with
*"Average Borrower Age: 74 · 640 loans · entire funded portfolio"* — a
whole-book substitution for a question about one bucket. Pre-existing
deterministic defect; the arm exposes it half the time.

---

## Final acceptance

### 24 CR4

**RECOVERED 7 · SAFE REFUSAL 14 · WRONG 2 · REGRESSED 0.**

### 75 and CFO 91

| arm | grades |
|---|---|
| deterministic | CORRECT 117 · FALSE_REFUSAL 22 · WRONG 8 · NO_ORACLE 4 · TRUE_REFUSAL 15 |
| Opus | CORRECT 125 · FALSE_REFUSAL 20 · **WRONG 2** · NO_ORACLE 4 · TRUE_REFUSAL 15 |

Deterministic **answers 166/166 byte-identical** to `c8089f9`; the single grade
change is Q10B, by the fixed instrument. Opus vs deterministic: 12 movements,
**8 recoveries, 0 regressions**, CFO 91 gained no wrong answer.

### 1,446 sweep — fully measured, from question 1

1,446/1,446 calls, all `claude-opus-5`. 5 malformed replies degraded safely.
**0 unbindable, 0 ambiguous**, 234 conflicts all fail-closed.

**30 movements (2.1%)**: 13 refused→answered, 8 answered-with-changed-text,
6 refusal-reason changed, **3 answered→refused**.

The three: `How do the Direct and Acquired portfolios differ?` (R1),
`How many acquired loans do we have?` and `Show the loans included in Unknown /
Missing age.` — both stochastic instances of the dimension-not-consumed class,
measured at 6/6 answered and 3/6 answered respectively. **Correction 3's two
targets no longer appear.** No movement is attributable to the corrections:
they only ever suppress a fill, so the applied set can only shrink.

### Frozen regression

**85 failing/erroring names, exact.**

---

## Every remaining known wrong answer

| id | arm | frequency | why it stands |
|---|---|---|---|
| **Q04C** | both | 6/6 | "total" is not a governed statistic; only the word "Show" separates it from its correct sibling |
| **Q19A** | both | 6/6 | the intent owner cannot distinguish a two-period delta from a window progression; the gate that fixed it broke a legitimate cohort question |
| **Q16B** | Opus | 1/6 | recovery depends on the model proposing `drawdown`; when it does not, the deterministic wrong answer stands |
| Q10B | deterministic | always | "size" is not in the dimension vocabulary (Opus arm is correct) |
| Q03A · Q05C · Q07B · Q17C | deterministic | always | recovered by the Opus arm |
| "Unknown / Missing age" | both | 3/6 | whole-book substitution for a bucket question |

---

# DO NOT FREEZE

The sprint's four objectives: **1 and 3 succeeded, 2 dispositioned, 4 failed.**

What improved is real and holds: the axis/filter defect is corrected in the one
direction that is resolvable from the contract, the two correct refusals are
explained rather than patched, Q10B has an oracle computed from the data, and
two instrument defects — one of which meant every published grade came from an
unreviewed copy — are closed. Deterministic answers did not move at all; the
frozen manifest is exact.

It fails the sprint's own bar, on the brief's own terms:

> *known wrong → controlled refusal is acceptable. known wrong → remain wrong is not.*
> *If they cannot be neutralised generically and safely, STOP and report DO NOT FREEZE.*

Q04C and Q19A remain wrong at 6/6, and neither can be corrected or made to fail
closed without adding grammar. I built the Q19A correction, proved it worked,
and reverted it when the frozen manifest showed it taking a legitimate question
— that revert is the honest outcome, not a failure of effort.

**And one thing the sprint discovered that outranks both**: Q16B shows the
Opus recoveries are stochastic. A "proven recovery" that holds 5 runs in 6 is
not a production guarantee, and nothing in the current design makes the model's
proposal reliable. That is a property of the architecture, not of these two
questions, and it should be understood before any freeze — not after.

Recommended next, each separately scoped: a governed aggregate-vs-listing
distinction (unblocks Q04C), a period-count signal from the intent owner
(unblocks Q19A), and a stability policy for the semantic arm — at minimum,
measuring per-question proposal stability and refusing where it is not stable.
