# Trakt product demonstration film

**SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.**

A 3½-minute Remotion film showing how Trakt operates as an AI-native managed
service for specialist lenders, built end to end from a fictional UK lender
("Alderbridge Lending Platform") whose data is generated, processed and measured
by the real Trakt pipeline.

Nothing in this film is drawn from, derived from, or approximated from any live
client. Every figure on screen is produced by running the production pipeline
over generated source data, and the render is blocked if the reconciliation gate
or the data-safety scan fails.

---

## Quick start

```bash
# 1. Build the demonstration (≈2 minutes). Generates the source data, runs the
#    pipeline, assembles the platform view, produces the artefacts, exports the
#    fixtures, and runs both gates.
python -m demo_platform.run_demo --all

# 2. Install the video project.
cd demo-video && npm install

# 3. Render.
npm run render          # 1080p master + the script and caption files
npm run render:all      # master + web cut + teaser
npm run stills          # one still per scene, for visual review
npm run preview         # Remotion Studio, for interactive editing
npm test                # storyboard + fixture consistency tests
npm run safety          # the pre-render data-safety gate on its own
```

Output lands in `demo-video/out/`:

| File | What it is |
|---|---|
| `trakt-product-demo-1080p.mp4` | 1920×1080, 30 fps, H.264, CRF 18 — the presentation master |
| `trakt-product-demo-web.mp4` | the same cut at CRF 26, for a website or an email |
| `trakt-product-demo-teaser.mp4` | a short selection cut (the problem, the month-on-month answer, the close) |
| `stills/Still-N-*.png` | one still per scene for visual review before publishing |
| `voiceover-script.md` | the full narration script with per-scene timings and pace |
| `captions.srt` / `captions.vtt` | subtitles matching the on-screen captions exactly |
| `scene-timings.csv` | scene-level timing sheet (seconds and frames) |
| `music-markers.md` | timing markers for an optional licensed music track |

---

## How the numbers get here

The film reads **one** fixture set. It computes nothing.

```
demo_platform/generator.py
   models each loan's own economics and writes two DIFFERENT raw source schemas,
   three month-ends each                                    ── 6 CSV extracts
        │
demo_platform/onboarding.py
   the real file profiler + the real Gate 1 mapper with each portfolio's
   approved alias contract                                  ── onboarding contracts
        │
engine/orchestrator/trakt_run.py --mode mi   (× 2 portfolios × 3 periods)
   Gate 1 alignment → canonical transform → Gate 2 → Gate 2.5 → Gate 3 → Gate 3b
        │
engine/platform_assembler.py   (× 3 periods)
   latest accepted canonical per source_portfolio_id        ── platform canonical
        │
   published to processed/platform/{client}/{date}/  and  …/latest/
        │
mi_agent_api  (filesystem-backed blob:// store — no Azure, no auth, no LLM)
   /mi/snapshots · /mi/snapshot · /mi/geo/exposure · /mi/evolution/funded
   /mi/risk-limits · POST /mi/query · /v1/copilot/mi/query
        │
demo_platform/metrics.py
   captures the governed metrics AND the exact envelopes both surfaces return
        │
demo-video/public/fixtures/*.json   ←── the film reads only this
```

Because the React MI Agent answer and the Copilot answer are captured from the
**same** `/mi/query` handler, they cannot disagree. The assertion gate proves it
by extracting every monetary and percentage figure from both answers and
comparing them.

---

## What is real, and what is a reconstruction

Being precise about this matters more than the film looking impressive.

**Real — executed by production code during the build:**

- the two source schemas and the mapping decisions (Gate 1 `HeaderMapper`, with
  each portfolio's approved alias overlay);
- source profiling (`engine/onboarding_agent/file_profiler.py`);
- the whole gate sequence, including the validation exceptions shown
  (`engine/gate_2_transform`, `engine/gate_3_validation`);
- the platform assembly and its manifest (`engine/platform_assembler.py`);
- every metric, every answer, both surfaces' envelopes (`mi_agent_api`);
- the investor deck — 18 slides built by `mi_agent_pptx` from the same `/mi/*`
  computations the dashboard uses;
- the concentration risk monitor, extracted live from a synthetic Schedule 8 by
  the production Schedule 8 extractor.

**Reconstruction — the UI, not the data:**

- Scenes 4 and 5 rebuild the `frontend/mi-agent-ui` three-region layout from the
  product's own design tokens, rather than screen-recording the live app. The
  live app fetches asynchronously and animates on mount, neither of which is
  frame-deterministic. **Every value shown is the value the production endpoint
  returned.**
- Scene 6 is a neutral representation of the Microsoft 365 Copilot chat surface.
  No Microsoft logo, icon, font or other brand asset is embedded, and Trakt is
  not implied to own or supply Copilot. The three actions named on screen —
  `askTraktMi`, `getLatestInvestorDeck`, `getLatestCanonicalTape` — are exactly
  the three in `deploy/copilot-agent/ai-plugin.json`.

**Excluded, and why:**

- **ESMA Annex 2 regulatory output.** The projector is run during the build; it
  stops at the enum-review gate (`purpose` carries unmapped values in strict
  Annex 2 mode), which is the correct production behaviour — a regime projection
  is not allowed to proceed on unreviewed enums. The artefact catalogue records it
  as unavailable with that reason and Scene 7 omits the card. The demonstration
  does not claim a regulatory delivery it did not perform.
- **Month-on-month deltas on `/mi/snapshot`.** That endpoint resolves its prior
  run through the on-disk onboarding-tape walk, which does not enumerate a
  `blob://` platform root, so it returns no prior period here. Scene 5 takes its
  deltas from the governed movement service instead. Production behaviour was
  left alone rather than changed to suit the film.
- **The Tier 7 LLM field mapper.** Requires human confirmation before any mapping
  is applied, and the build runs offline and deterministically. Headers the
  deterministic tiers cannot resolve are shown as referred for review and closed
  by the client's alias contract — the same review-first path, without the LLM.

---

## Data safety

Two gates, and the render fails closed on either.

**`demo_platform/safety.py`** scans everything the film can read — the exported
fixtures, the artefact catalogue, the onboarding contracts, all six generated
source extracts, the Remotion source and the bundled fixtures — for:

- prohibited client strings. The list is **read live out of the production client
  configs** (`config/client/*.yaml`), so whatever the live config calls the
  client is automatically forbidden here. It does not depend on anyone
  remembering to add a name to a list;
- email addresses, GUIDs and tenant identifiers, Azure storage endpoints and web
  hosts, storage connection strings and account keys, SAS/signed URLs, bearer
  tokens, JWTs, private keys, live artefact download tokens, and production
  branch references.

**`demo-video/scripts/safety-scan.mjs`** runs before every render and refuses to
proceed unless the fixtures are present and marked synthetic, the Python safety
report passed, the reconciliation gate passed, and a fresh re-scan of everything
that gets bundled finds nothing.

Every frame carries a discreet footer, the closing scene states the disclaimer in
full, and every generated source row carries a `Synthetic Data Notice` column.

---

## Editing the film

**Timing and copy** live in `src/timeline.ts` — one table controlling scene order,
duration, captions and narration. The composition length, the subtitle files, the
voice-over script and the music markers are all derived from it, so they cannot
drift apart. `npm test` enforces the run-time band, the per-scene duration bands,
caption legibility (reading pace), narration pace, and UK English.

**Visual identity** lives in `src/design/tokens.ts`, mirrored from
`frontend/mi-agent-ui/src/lib/theme.ts` and `src/index.css`. If the product's
palette changes, change it there and mirror it here.

**Data** lives in `src/data/fixtures.ts`, which is the only module that reads the
JSON. Scenes select from it; they never compute.

**Motion** lives in `src/design/motion.ts`. Every helper is a pure function of the
frame number — no time, no randomness, no measured layout — so a rendered frame is
identical on every machine and every run.

---

## Rendering notes

- **No network access at render time.** The fixtures are imported at bundle time,
  and the browser is resolved from an existing local Chrome/Chromium install.
  `scripts/render.mjs` looks in the standard locations plus the Playwright layout
  (`/opt/pw-browsers/chromium*`); set `REMOTION_BROWSER_EXECUTABLE` to override.
  If none is found, Remotion falls back to downloading its own Chrome Headless
  Shell, which does need network access.
- **Fonts.** The product ships Inter with a system fallback stack. The render
  environment has no network access, so the stack resolves to whatever the
  headless browser has; the type scale is chosen to read cleanly either way. If
  you want Inter specifically, install it on the render host or add it as a local
  `@remotion/fonts` asset.
- **Determinism.** Concurrency is pinned to 1 and the GL renderer to `swangle`, so
  peak memory is predictable and no frame is composed by a differently-warmed
  browser tab.
- **Before publishing**, look at `out/stills/`. Check for overflow, text clipping,
  contrast, inconsistent figures, accidental production data and visual
  artefacts. The tests cover the numbers; only a human can sign off the picture.

---

## Regenerating everything from a clean checkout

```bash
pip install -r requirements.txt
python -m demo_platform.run_demo --all
python -m pytest tests/test_demo_platform_*.py -q
cd demo-video && npm install && npm test && npm run render:all && npm run stills
```

Individual stages, all idempotent:

```bash
python -m demo_platform.run_demo --generate      # source extracts
python -m demo_platform.run_demo --onboard       # onboarding contracts
python -m demo_platform.run_demo --orchestrate   # pipeline + assembler + publish
python -m demo_platform.run_demo --artefacts     # deck, validation, risk, audit
python -m demo_platform.run_demo --metrics       # export the film's fixtures
python -m demo_platform.run_demo --assert        # reconciliation gate
python -m demo_platform.run_demo --safety        # data-safety scan
```

`TRAKT_DEMO_ROOT` relocates the whole generated workspace, which is how the
reproducibility test builds twice in isolation and compares content hashes.
