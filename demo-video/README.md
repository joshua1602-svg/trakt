# Trakt outbound demo film

**SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.**

A 90-second Remotion film built to the enclosed storyboard: a cold outbound asset
aimed at principals and COOs at niche non-bank lenders — equity release, specialist
residential, bridging, forward-flow buyers. Firms of 4–20 people with no data team,
who buy back books and run month-end by hand.

It has one job: make the recipient reply and ask to see Trakt run on their own tape.

It will be opened in an email or on LinkedIn, on a laptop, **with the sound off**. The
on-screen copy carries the argument unaided; narration is a bonus track that does not
ship with the file. Captions are burned in.

Every figure on screen is produced by running the real Trakt pipeline over generated
source data for a fictional lender ("Alderbridge Lending Platform"). Nothing is drawn
from, derived from or approximated from any live client, and the render is blocked if
the reconciliation gate or the data-safety scan fails.

---

## Quick start

```bash
# 1. Build the demonstration (≈4 minutes). Generates the source extracts, runs the
#    pipeline per portfolio per period, assembles the platform view, produces the
#    governed artefacts, exports the fixtures, and runs both gates.
python -m demo_platform.run_demo --all

# 2. Install the film project.
cd demo-video && npm install

# 3. Render.
npm run render          # 1920x1080 master, plus the script and caption files
npm run render:square   # 1080x1080 LinkedIn / email variant
npm run render:all      # both
npm run stills          # the three email frames plus the review sweep
npm run stills:square   # the same frames in the square crop
npm run preview         # Remotion Studio, for interactive editing
npm run check           # fonts + typecheck + theme lint + tests + safety gate
npm run signal          # the accent audit, from the rendered pixels (slow)
```

`npm run check` runs ahead of every render and every stills pass. It is the
storyboard's pre-render checklist, mechanised. `npm run signal` is the accent audit —
it renders a frame sweep and counts full-strength accented elements from the pixels, so
"one `signal` element per frame" is measured rather than trusted. It is slow, so it is
not part of `check`; run it after any change to what carries the accent.

Output lands in `demo-video/out/`:

| File | What it is |
|---|---|
| `trakt-demo-1080p.mp4` | 1920×1080, 30 fps, H.264 — the master |
| `trakt-demo-square.mp4` | 1080×1080 — the LinkedIn and email variant |
| `stills/trakt-demo-frame-{1020,1500,2100}.png` | the three frames for the outbound email body |
| `stills/review/*.png` | seventeen frames for a human to check before publishing — one at each scene midpoint, plus the beats worth their own look |
| `voiceover-script.md` | narration script with per-scene timings and pace |
| `captions.srt` / `captions.vtt` | subtitles matching the burned-in captions exactly |
| `scene-timings.csv` | scene-level timing sheet (seconds and frames) |
| `music-markers.md` | timing markers for an optional licensed track |

---

## Structure

90 seconds, 2,700 frames, 30 fps. Five scenes, hard cuts on `ink`.

| # | Scene | Frames | Time | What it proves |
|---|---|---|---|---|
| 1 | The cost | 0–420 | 0:00–0:14 | Two source schemas that share no column names, and what month-end costs |
| 2 | Onboarded in under 48 hours | 420–1140 | 0:14–0:38 | The approved contract, and what the platform refuses to guess |
| 3 | One dataset, every output | 1140–1740 | 0:38–0:58 | Six governed outputs off one dataset, regulatory first |
| 4 | Three ways in | 1740–2340 | 0:58–1:18 | The same answer in three channels, on the same frame |
| 5 | Close | 2340–2700 | 1:18–1:30 | The positioning, and the ask |

`src/timeline.ts` is the single source for scene lengths, captions and narration. The
composition, the burned-in captions, the subtitle files and the voice-over script are
all generated from it, so they cannot drift apart.

### Where the numbers come from

`src/data/fixtures.ts` imports the JSON the demonstration run wrote into
`public/fixtures/` and **selects** from it. It computes nothing. If a figure is not in
a fixture it does not appear on screen — a scene that asks for a number the fixture
does not carry throws at bundle time rather than rendering a placeholder.

| On screen | Fixture | Produced by |
|---|---|---|
| `£1.96bn`, `11,035 loans`, `+£18.1m`, the regional split | `demo_metrics.json` | `POST /mi/query` → `mi_agent_workflow` |
| `39` / `34` / `2`, the four mapping pairs, the referred item | `demo_manifest.json` | `engine/gate_1_alignment/semantic_alignment.py` |
| `41:20` | *not a fixture* | `src/claims.ts` — a **stated** claim, see below |
| the six artifact cards | `artefact_catalogue.json` | `demo_platform/artefacts.py` |
| `33/33 checks passed` | `assertion_report.json` | `demo_platform/assertions.py` |
| the disclaimer | every fixture | `demo_platform/config.py` |

---

## Brand tokens

`src/theme.ts` is the single source of truth. No scene or component file declares a
colour, font size, weight, radius or duration inline, and `scripts/lint-theme.mjs`
fails the render if one appears.

Two rules carry more weight than the rest.

**`signal` (`#4DE0C4`) is a scarcity resource**, and it has exactly two strengths and no
third. Full strength marks the value being proven — at most one element per frame. Soft
(`theme.signalSoftOpacity`, 40%) is for confirmations and check states only, such as
"XSD validated" on the regulatory card. There is no second green; a test asserts that.

Full strength appears on: S3's consolidated balance, S3's `33/33 checks passed`, S3's
final claim, S4's three figures and S5's ask. Where a scene passes the accent from one
element to another — S3 hands it from the balance to the reconciliation check, then to
the claim — `<Stat>` takes a fractional `accent` and they cross-fade, so they are never
both cyan. S4's three figures are the film's one stated exception: they are the same
value proven in three places, which is the argument of the scene.

`flag` (amber) stays locked to S2's referred-for-review beat and S1's failed connection.
Nowhere else, and the lint asserts the count.

**The mono role is semantic.** If a figure came out of the pipeline — a balance, a
count, an LTV, a field name, a portfolio code, a filename, a timestamp — it is set in
IBM Plex Mono. If it is Trakt talking about itself, it is Archivo (claims) or Inter
(supporting copy and captions). The viewer never has to be told which numbers are
governed; the typeface says so.

Six type sizes exist. The lint asserts that count and that every one of them is used.

### Typefaces

Archivo 700, Inter 400, IBM Plex Mono 500 — all three SIL Open Font License 1.1.

The storyboard asks for them via `@remotion/google-fonts` "so renders are
deterministic", but that package resolves its `@font-face` sources to
`fonts.gstatic.com` and fetches them at frame time, which would make a render depend
on the network. So the URLs still come from `@remotion/google-fonts` — they are
provably the Google Fonts originals — and `scripts/vendor-fonts.mjs` downloads the
files once into `public/fonts/`, recording each source URL and SHA-256 in
`public/fonts/manifest.json`. `npm run fonts:verify` fails if a file drifts from the
manifest, and it runs before every render.

---

## Component contract

Six components. Every scene is composed from these, so consistency is enforced by
construction rather than by discipline.

| Component | Rule |
|---|---|
| `<Figure>` | The only way to put a computed value on screen. Data face, `tabular-nums`. A `<Counter>` outside one fails the lint |
| `<Stat>` | A `<Figure>` that **always** renders its provenance rule — there is no way to put a governed figure on screen without saying where it came from |
| `<Claim>` | Archivo display. One per scene, maximum |
| `<ArtifactCard>` | Fixed dimensions. The **human label** is the title, in the body face; the filename drops into the mono meta line with the figures |
| `<Counter>` | Wraps `interpolate` + `spring`. The only count-up in the film |
| `<Chrome>` | Lockup and disclaimer, rendered once **outside** `<Series>` |

`Chrome` sitting outside the `Series` is what structurally guarantees the branding
cannot drift between scenes, and that the disclaimer is present in every single frame
rather than in every frame someone remembered to add it to.

The square variant is the same component tree with a `square` layout flag, not a
re-edit. `src/theme.ts` carries both geometries; components read them through
`useGeometry()`.

---

## Decisions taken against the storyboard

Places where the storyboard asked for something the data does not support, or
contradicted itself. Each is resolved in favour of what is defensible, and each is
called out, because the alternative was to quietly ship a number nobody can stand
behind on the call the film generates.

### 1. Measured figures and stated claims are kept apart

The film asserts two kinds of number and they must never be confused.

**Measured** figures come from `public/fixtures/*.json`, which the demonstration run
wrote. They are read through `src/data/fixtures.ts`, set in IBM Plex Mono, and always
carry a provenance rule naming where they came from.

**Stated** claims are commercial claims about the product and its market. They live in
`src/claims.ts` — one file, so anyone auditing the film can see the complete list of
things it asserts without a fixture behind it. Currently: the onboarding window, the
month-end cost line, S1's opening line and S4's three use lines.

The onboarding clock is a stated claim. It counts **`41:20`** in `HH:MM` against a
"under 48 hours" claim, deliberately short of the threshold — a figure with visible
headroom survives the first question about it, and one that lands exactly on its own
limit does not. `ONBOARDING_HOURS` and `ONBOARDING_CLAIM_HOURS` are both in
`src/claims.ts` and a test asserts the gap between them.

A number in `claims.ts` is the business's to defend on the call the film generates. A
number in a fixture is the pipeline's. Nothing should move between the two.

### 2. The counter values (S2)

The storyboard specifies `36 fields mapped · 22 client-specific decisions · 3 referred
for review`. The measured Gate 1 result for the acquired back book is **39 / 34 / 2**,
and those are what the film shows. Calculated values are not adjusted to match a
script: the source data is engineered and the outputs are then reported as measured.
These figures moved because the source schemas were extended to carry the fields the
ESMA Annex 2 submission needs.

The referred-for-review beat survives intact and is the film's most important frame.
The acquired book's `Further Advance This Period` column is genuinely unresolvable by
Gate 1, and the panel shows the reviewer's own note explaining why it is deliberately
not mapped to a canonical balance field. The second referred header is the synthetic
watermark, which is honest — it is a header a reviewer would triage.

### 3. `signal` in S4

The storyboard's general rule is one accented element per frame. Its S4 instruction is
explicitly three at once: *"`£1.96bn` counts up simultaneously in all three panels,
landing on the same frame, in `signal`."* The specific instruction wins — they are the
same figure proven in three places, which is the whole argument of the scene, and the
three counters share one `at` value so they cannot drift apart.

### Also

- **Card metadata follows the measured artefacts** where they differ from the
  storyboard's draft: the canonical tape is `platform_canonical_typed.csv` with 61
  fields, not `canonical_tape.csv` with 42, because that is the file the assembler
  writes and the column count it has.
- **`[contact] · [link]`** is not shipped as a bracketed placeholder. Put the real
  address and URL in `CONTACT` in `src/scenes/S5Close.tsx` and they appear beneath the
  ask; leave it empty and the ask stands alone.
- **Captions do not sit over a display claim.** A claim is already burned-in copy at
  the display size; repeating it in body text underneath competes with it. The close
  has no captions at all for the same reason — it is entirely display copy, and neither
  does S1's opening line, which is doing a caption's job already.
- **No Microsoft app icon.** S4's Copilot panel carries the product name as a wordmark
  and no mark. Microsoft's trademark guidelines state their "logos, app and product
  icons, illustrations, photographs, videos, and designs can never be used without an
  express license", which this project does not hold. The `label` token is set in the
  body face, so the panel label already *is* the wordmark the fallback calls for; the
  text rules are met (Microsoft precedes the product name, unaltered, no affiliation
  implied, Trakt's lockup more prominent). If a licence is obtained, drop the asset into
  `public/brand/` — `S4Omnichannel.tsx` says where.
- **Scene beat boundaries shift by a few frames** from the storyboard's timings where a
  beat was shorter than the time needed to read its caption. The film is watched with
  the sound off, so a beat that outruns its caption is a beat the viewer does not get.
  Scene lengths and the 2,700-frame total are exactly as specified.
- **The delivery bitrate.** The master is encoded with an 8 Mbps target and a 10 Mbps
  cap, as specified, and lands at ≈2.2 Mbps. That is the encoder declining to spend the
  budget, not the budget failing to arrive: flat `ink`, hairlines and static type with
  hard cuts give libx264 nothing to spend bits on, and it reaches its quantiser floor
  well below the target. Every render prints the achieved rate next to the target so
  the gap is visible rather than assumed.

---

## Pre-render checklist

The storyboard's twelve items. Ten are mechanised; two need eyes.

| # | Item | Enforced by |
|---|---|---|
| 1 | No hex, font size or duration outside `theme.ts` | `scripts/lint-theme.mjs` |
| 2 | `signal` on exactly one element per frame | `scripts/audit-signal.mjs` — renders a frame sweep, counts full-strength accented elements from the pixels |
| 3 | `flag` in exactly two places | `scripts/lint-theme.mjs` (asserts two scenes) |
| 4 | Every computed figure mono, `tabular-nums`, with a provenance rule | `<Figure>`/`<Stat>` are the only hosts for a `<Counter>`, `<Stat>` cannot render without a stamp, and `scripts/lint-theme.mjs` fails on a counter outside them |
| 5 | No box shadows, glows or gradients | `scripts/lint-theme.mjs` |
| 6 | Six type sizes, no more | `scripts/lint-theme.mjs` + `npm test` |
| 7 | Logo pixel-identical in every frame | `<Chrome>` outside `<Series>`; one fixed size in `theme.lockup` |
| 8 | Every claim legible with the sound off | **human** — `npm run stills` |
| 9 | Captions burned in | `<Captions>` in `Film.tsx`; `npm test` asserts hold times against a reading rate |
| 10 | Regulatory before management information in S3 | `npm test` |
| 11 | The elapsed-time number is defensible | `npm test` fails if any scene copy asserts "48 hours" |
| 12 | Disclaimer in every single frame | `<Chrome>` outside `<Series>` |

---

## Constraints the render honours

- **No network at render time.** Fixtures are imported into the bundle at build time;
  the typefaces are vendored under `public/fonts`; the browser is resolved from an
  existing local Chrome or Chromium install (`scripts/render.mjs → findBrowser`).
- **No live client environment, no Azure, no authentication.** The demonstration runs
  against a local filesystem blob backend with the MI agent's auth and LLM paths off.
- **No prohibited content.** `npm run safety` re-runs the Python data-safety scan and
  the reconciliation gate, then re-scans the bundled fixtures. It runs before every
  render and a finding fails the build.
- **UK English and pounds sterling** throughout, at the precision the product's own
  answers use (`src/format.ts`).

---

## Limitations

- **The film ships silent.** No music is embedded: commercial tracks are copyrighted
  and this repository carries no licensed audio. `out/music-markers.md` gives the
  markers to cut a licensed track to, and nothing in the picture depends on a cue.
- **The narration is a script, not a track.** `out/voiceover-script.md` is written to
  sit inside each scene at ~150 words per minute; recording it is a separate job.
- **The Copilot and workspace panels in S4 are silhouettes, not screens.** They are
  reduced to three elements each, by design — recognisable shapes, not readable UI. The
  action names and the regional split in them are real.
- **One production configuration gap is worked around, not fixed.**
  `config/system/enum_mapping.yaml` maps `collateral_type` onto property-type codes
  (`R1`/`R2`/`C1`/`C2`) that the auth.099.001.04 `CollTp` enumeration does not accept,
  so the ESMA submission cannot be produced from the production mapping as it stands.
  `demo_platform/artefacts.py` writes a demo-scoped copy with identity entries for the
  codes the XSD itself enumerates and passes it through the projector's own
  `--enum-mapping` argument. Production configuration is untouched. The underlying gap
  is real and worth fixing in its own change.
