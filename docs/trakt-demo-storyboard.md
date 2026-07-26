# Trakt — Outbound Demo Film
## Remotion storyboard, brand tokens and build spec

**Audience:** principals and COOs at niche non-bank lenders — equity release, specialist residential, bridging, forward-flow buyers. Firms of 4–20 people with no data team, who buy back books and run month-end by hand.

**Objective:** a cold outbound asset. It has one job — make the recipient reply and ask to see Trakt run on their own tape.

**Runtime:** 90 seconds · 1920×1080 · 30fps · 2,700 frames
**Data:** the existing synthetic Alderbridge Lending Platform set. No new figures invented.

**Watch context:** this will be opened in an email or on LinkedIn, on a laptop, **with the sound off**. On-screen copy must carry the argument unaided. Narration is a bonus track, not the load-bearing layer. Burn in captions.

---

## 1. Brand tokens

Single source of truth. Create `src/theme.ts` and export these. **No scene file may declare a colour, font size, weight, radius or duration inline.** Every value below is referenced by token name or it doesn't ship.

### 1.1 Colour

| Token | Hex | Use |
|---|---|---|
| `ink` | `#060B1F` | Ground. Every scene. Never a gradient. |
| `hull` | `#0F1D4D` | Panels and surfaces. One elevation only. |
| `rule` | `#24356E` | Hairlines, dividers, inactive states, grid. |
| `paper` | `#E8ECF7` | Primary type. |
| `mute` | `#8B99C4` | Secondary type, labels, stamps. |
| `signal` | `#4DE0C4` | **The accent.** Reserved exclusively for the value being proven in that scene. |
| `flag` | `#F2A93B` | Exceptions and referred-for-review. Appears exactly twice in the film. |

The current cut's core failure is that everything glows equally. **`signal` is a scarcity resource** — at most one element on screen carries it at any moment. If two things are cyan, neither is important.

Kill: all box shadows, all glow filters, all gradient fills on panels. Depth comes from a 1px `rule` border and nothing else.

### 1.2 Typography

Three roles, three faces, loaded via `@remotion/google-fonts` so renders are deterministic.

| Role | Face | Use |
|---|---|---|
| Display | **Archivo** 700, tracking −2% | Claims. The line the viewer must remember. |
| Body | **Inter** 400/500 | Supporting sentences, captions, narration burn-in. |
| Data | **IBM Plex Mono** 500, `tabular-nums` | Anything Trakt computed or any source identifier. |

**The mono rule is semantic, not decorative.** If a figure came out of the pipeline — a balance, a count, an LTV, a field name, a portfolio code, a timestamp — it is set in mono. If it's Trakt talking about itself, it's Archivo or Inter. A viewer never has to be told which numbers are governed; the typeface says so. This is the film's signature device and it must be applied without exception.

`tabular-nums` is mandatory on every animated counter or the digits will jitter through the count-up.

**Scale** (px at 1080p):

| Token | Size / line | Face |
|---|---|---|
| `display` | 128 / 0.95 | Archivo |
| `headline` | 64 / 1.05 | Archivo |
| `stat` | 88 / 1.0 | Plex Mono |
| `body` | 26 / 1.45 | Inter |
| `label` | 14 / 1.2, tracking +10%, uppercase | Inter 500 |
| `stamp` | 15 / 1.2 | Plex Mono |

Six sizes. Nothing else exists. The current cut has roughly a dozen, which is why no frame has a focal point.

### 1.3 The signature: provenance rules

Every computed figure sits above a 1px `rule` hairline with a `stamp`-sized mono line beneath it:

```
£1.96bn
────────────────────────────
ALP_ORIGINATION + ALP_ACQUIRED · 2026-06-30
```

This is the visual argument for governance and it recurs in every scene. It's also the thing that makes the film look like a lending product rather than a SaaS landing page.

### 1.4 Chrome

- Trakt lockup: top-left, 40px from each edge, **one fixed size for the whole film**. It never scales, moves or animates after the first scene.
- `SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER`: bottom-left, `label` token, `mute`. Fixed position, always present, never animated.
- **Remove all per-scene eyebrow labels** (`STAGE 1 · ONBOARD ONCE`, `THE REPORTING PROBLEM`, etc.). They narrate the structure of a product tour to someone who hasn't agreed to take one.
- Captions: bottom third, `body` token, `paper` on a 70%-opacity `ink` plate, max two lines.

### 1.5 Motion

| Token | Value |
|---|---|
| `quick` | 12 frames |
| `base` | 20 frames |
| `slow` | 30 frames |
| `spring` | `{ damping: 200, mass: 0.6 }` — critically damped, no overshoot |
| `enter` | opacity 0→1 + translateY 12px→0 |
| `stagger` | 3 frames between siblings |

One entrance, one spring, one stagger, film-wide. Nothing scales in. Nothing bounces. The restraint is the point — this is a governance product being sold to a regulated firm.

---

## 2. Composition

```
<Composition id="TraktDemo" width={1920} height={1080} fps={30} durationInFrames={2700} />
```

```tsx
<AbsoluteFill style={{ backgroundColor: theme.color.ink }}>
  <Series>
    <Series.Sequence durationInFrames={420}><S1Cost /></Series.Sequence>
    <Series.Sequence durationInFrames={720}><S2Onboard /></Series.Sequence>
    <Series.Sequence durationInFrames={600}><S3Dataset /></Series.Sequence>
    <Series.Sequence durationInFrames={600}><S4Omnichannel /></Series.Sequence>
    <Series.Sequence durationInFrames={360}><S5Close /></Series.Sequence>
  </Series>
  <Chrome />      {/* logo + disclaimer, outside Series, never remounts */}
  <Captions />
</AbsoluteFill>
```

`Chrome` sits outside the `Series` deliberately — that structurally guarantees the branding cannot drift between scenes.

Scene transitions: hard cut on `ink`. No crossfades, no wipes.

---

## 3. Scenes

### S1 · The cost — frames 0–420 (0:00–0:14)

> **Narration:** "You bought a back book. It arrived as a servicer extract that looks nothing like your origination system. Every month-end, someone rebuilds the mapping by hand."

**On screen**
- 0–90: a faint `rule` ledger grid fades up. Two column stacks assemble in mono — left `ALP_ORIGINATION` (Loan Reference, Completion Date, Original Advance…), right `ALP_ACQUIRED` (Account Number, Origination, Initial Drawdown…). Show **six rows each, not eighteen**. The point is that they don't match, and six proves it as well as eighteen.
- 90–150: the two stacks drift apart. A `flag` hairline tries and fails to connect them.
- 150–300: display line, centred, over the grid:

  **Every portfolio you buy is another reconciliation you own.**

- 300–420: hold, then the grid dims to 20%.

**Data used:** source header names only, from both extracts.

**Why it's changed:** the current opener presents 37 column chips and asks the viewer to notice an absence. Six columns and a failed connection states it in two seconds.

---

### S2 · Onboarded in under 48 hours — frames 420–1140 (0:14–0:38)

> **Narration:** "Trakt onboards a portfolio once. Agents profile the source, map it to a canonical model and apply your lending rules. Thirty-six fields mapped, twenty-two decisions specific to your book — and three referred to a human, because the platform knows what it doesn't know. Approved once, applied unchanged every period after."

**On screen**
- 420–480: `stat` counter, dead centre, mono, counting up — **00:00 → 47:12** as an elapsed clock. Provenance rule beneath: `TAPE RECEIVED → GOVERNED OUTPUT`.
- 480–660: clock shrinks to a corner stamp. Mapping lines draw left-to-right, staggered, source header → canonical field. Use four real pairs from the synthetic contract:
  - `Loan Reference` → `loan_identifier`
  - `Completion Date` → `origination_date`
  - `Account Number` → `loan_identifier`
  - `Reporting Cut-Off` → `data_cut_off_date`
- 660–780: three counters resolve in a row — `36 fields mapped` · `22 client-specific decisions` · `3 referred for review`. **The third is `flag` and the other two are `mute`.** This is the first of the film's two `flag` moments.
- 780–900: hold on the referred-for-review item. This is the single most important beat in the film — it is the proof that Trakt exercises judgement and knows its own limits. An ETL tool cannot do this, and every prospect has been sold an ETL tool before.
- 900–1080: display line:

  **Onboarded in under 48 hours. Approved once, applied every period.**

- 1080–1140: hold.

**Data used:** 36 / 22 / 3 from the approved onboarding contract; the four mapping pairs.

**Open item:** the 48-hour figure needs to be one you'll defend on the call it generates. If the anchor client's real elapsed time supports it, make it the hero. If not, `days, not months` is weaker but safe. Don't ship a bracketed number.

---

### S3 · One dataset, every output — frames 1140–1740 (0:38–0:58)

> **Narration:** "From there, one governed dataset. One point nine six billion across eleven thousand and thirty-five loans, two source portfolios, every row provenance-stamped. Your regulatory submission, your investor pack and your management information are the same numbers — because they're the same dataset."

**On screen**
- 1140–1260: two `stat` figures converge to centre — `£1.39bn` and `£579.4m` — and resolve into **`£1.96bn`** in `signal`. Provenance rule beneath: `11,035 LOANS · 2 SOURCE PORTFOLIOS · 30 JUNE 2026`. This is the scene's one `signal` element.
- 1260–1440: six artifact cards fan outward from the figure, staggered. Identical card component, mono titles:
  - `annex2_submission.xml` — Regulatory
  - `investor_pack.pptx` — 18 slides · 2026-06
  - `canonical_tape.csv` — 11,035 rows · 42 fields
  - `validation_report.json` — 75 exceptions · 0.68%
  - `concentration_monitor.json` — 12 limits tested
  - `audit_manifest.json` — 6 files content-hashed · SHA-256
- 1440–1560: a single `signal` check resolves across all six — **`33/33 reconciliation checks passed`**.
- 1560–1740: display line:

  **One dataset. Every output reconciles.**

**Why it's changed:** regulatory reporting is your hardest claim to replicate and it appears nowhere in the current cut. Here it leads the fan-out. Order the cards regulatory-first, left to right.

---

### S4 · Three ways in — frames 1740–2340 (0:58–1:18)

> **Narration:** "Consume it however you already work. Run it as a managed service and never log in. Ask it from Copilot, in the tools you have. Or open the workspace when you need to drill in. Same engine, same answer, every time."

**On screen**
- 1740–1830: the frame divides into three equal panels with `rule` hairlines. `label` tokens: `MANAGED SERVICE` · `MICROSOFT 365 COPILOT` · `MI AGENT WORKSPACE`.
- 1830–2040: each panel fills with a reduced, near-abstract rendering — a scheduled artifact drop; a Copilot thread calling `askTraktMi`; a workspace with the regional bar chart. **Reduce each to three elements maximum.** These are recognisable silhouettes, not readable screens.
- 2040–2160: **the payload.** `£1.96bn` counts up simultaneously in all three panels, landing on the same frame, in `signal`. Then `+£18.1m` beneath each, also simultaneous. Nothing else moves.
- 2160–2280: display line:

  **Three ways in. One governed answer.**

- 2280–2340: hold.

**Data used:** £1.96bn, +£18.1m, `askTraktMi`, the regional split for the workspace chart.

**Why it's changed:** the current cut makes this point in a text panel ("both call the same deterministic MI engine"). It's the strongest differentiator in the product and it deserves to be shown, not asserted. Simultaneity is the whole argument — if the three counters land on different frames the beat is dead.

---

### S5 · Close — frames 2340–2700 (1:18–1:30)

> **Narration:** "Trakt. A data operating system for specialist lenders."

**On screen**
- 2340–2460: everything clears to `ink`. Trakt lockup animates once from top-left to centre — the only time chrome moves in the entire film.
- 2460–2580: beneath it, `body`:

  **A data operating system for specialist lenders.**
  Onboarding · Orchestration · Regulatory reporting · Management information · Governed AI access

- 2580–2700: the ask, in `signal`:

  **See it run on your own tape.**
  `[contact]` · `[link]`

**Note:** the current cut ends with a positioning statement and no ask. For an outbound asset that's the whole ballgame. "See it run on your own tape" is the strongest CTA available to you — it's concrete, it's low-commitment, and it's the thing that actually converts this audience.

---

## 4. Component contract

Build these five and compose every scene from them. Consistency is enforced by construction, not by discipline.

| Component | Props | Rule |
|---|---|---|
| `<Stat>` | `value, stamp, accent?` | Mono, `tabular-nums`, always renders its provenance rule. |
| `<Claim>` | `children` | Archivo `display`. One per scene, maximum. |
| `<ArtifactCard>` | `filename, meta[]` | Fixed dimensions. Mono filename, `mute` meta. |
| `<Counter>` | `from, to, format, frames` | Wraps `interpolate` + `spring`. Never hand-roll a count-up. |
| `<Chrome>` | — | Logo + disclaimer. Rendered once, outside `Series`. |

---

## 5. Pre-render checklist

- [ ] No hex value, font size or duration appears outside `theme.ts`
- [ ] `signal` appears on exactly one element per frame
- [ ] `flag` appears in exactly two places in the film (S2 referred-for-review, and one exception reference)
- [ ] Every computed figure is mono with `tabular-nums` and carries a provenance rule
- [ ] No box shadows, glows or gradients survive
- [ ] Six type sizes in use, no more
- [ ] Logo is pixel-identical in every frame — scrub the render and confirm
- [ ] Every claim is legible and complete **with the sound off**
- [ ] Captions burned in
- [ ] Regulatory reporting appears before management information in S3
- [ ] The 48-hour number is one you can defend
- [ ] Disclaimer is present in every single frame

---

## 6. Delivery

- **Master:** 1920×1080 H.264, ~8 Mbps, `.mp4`
- **LinkedIn/email variant:** 1080×1080 crop — build as a second `Composition` sharing the same scene components with a `square` layout flag, not as a re-edit
- **Still frames:** export frames 1020, 1500 and 2100 as `.png` for use in the outbound email body itself

---

## 7. One thing to reconsider

Archivo/Inter/Plex Mono is a deliberate move away from the rounded geometric sans in the current wireframes — geometric rounded reads consumer-fintech, and you're selling governance to a regulated lender. If Trakt already has a committed brand face, keep it in the display role and retain Plex Mono for the data role regardless. The mono-for-computed-values rule is the part that's doing the work, and it survives any typeface decision.
