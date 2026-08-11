# Content map — every claim on the page, and what backs it

The purpose of this document is to stop marketing running ahead of the product.
Every visible claim is classified:

| Class | Meaning |
|---|---|
| **Evidenced** | Implemented in this repository. The cited file is the proof. |
| **Demonstration** | Describes what the public demo does, here and now. |
| **Illustrative** | A depiction of product workflow whose figures are not engine output. Labelled as such on the page itself. |
| **Deployment** | True of a configured client deployment, not a platform guarantee. Worded to say so. |
| **Roadmap** | Not shipped. Labelled as roadmap on the page itself, never mixed in with live capability. |
| **Positioning** | A statement of what Trakt is for. Not a capability claim. |
| **Excluded** | Considered and deliberately **not** said, because the repository does not support it. |

**Pass 3 repositioned the page.** The previous page presented a governed
reporting dataset; the product had moved beyond it. The page now presents one
governed portfolio layer powering monitoring, forecasting, covenant controls,
reporting and portfolio Q&A. Two Pass-2 statements in this document were found
to be **factually stale against the repository** and are corrected below (see
§7 and the Excluded list): the "no timer trigger or cron anywhere" claim
(`function_app.py:83` is an Azure timer trigger draining the Teams notification
outbox) and the "no Teams bot, no adaptive cards, no proactive messaging" claim
(`mi_agent_api/teams_bot.py`, `trakt_notifications/cards.py` and the
`trakt_notifications/` outbox/delivery pipeline are all three).

The demonstration still runs on three governed books across two reporting
periods (`synthetic_demo/output/multibook/…`, built by
`synthetic_demo/run_multibook_pipeline.sh`). Book identities mirror
`demo_platform/config.py`. Every demo figure is as at 30 June 2026.

Anyone editing page copy should add a row here, or delete one.

**Pass 4 was a clarity refactor, not a repositioning.** Body copy was cut
roughly 45%; each concept is now explained once; and the page carries two
deliberate, labelled, user-started demos — the query demo (section 2) and
the controls demo (section 4). Neither autoplays and neither loops silently:
`QueryDemo.tsx` gates the interactive demo behind a start affordance, and
`DemoPlayer.tsx` gives the controls film poster → play → pause/replay →
"Watch again". The reporting band and the onboarding content were removed
from the homepage — reporting claims survive in the platform output chips,
and onboarding detail now belongs to a product page, not this one.

**Pass 6 simplified.** The page is now eight narrative sections: value
proposition · portfolio query demo · platform · risk & controls (the controls
demo) · portfolio intelligence (distribution) · delivery model (five static
tiles) · governance · contact. The horizontal accordion was replaced with a
static tile row, body copy was halved again, the operating-model section was
absorbed into the platform diagram, and both demo sections now share one
shape: eyebrow, headline, one line, then the demo at full container width.

**Pass 5 (grid, motion and caveat copy).** One 12-column grid across the
two-column sections; scroll reveals (240ms ease-out, 60ms stagger, once,
gated on JS + `prefers-reduced-motion: no-preference` so no-JS and
reduced-motion visitors always see final state). The synthetic-data
disclosure model changed by explicit direction: **exactly one disclosure —
the amber "Synthetic data" pill on the demo's portfolio header** (mirrored on
the poster pre-start). The intro sentence, answer-footer tags, and
hero/lens-card mentions were removed; the as-at date appears once, in the
portfolio header. Internal identifiers (`ALP_Platform_202606`) no longer
reach the page — the deploy gate keeps its identity via the meta tag.

---

## A. Navigation

| Copy | Class | Evidence |
|---|---|---|
| Demo · Platform · Risk & Controls · Intelligence · Governance · Book a demo | Positioning | — |

---

## 1. Value proposition

| Copy | Class | Evidence |
|---|---|---|
| "One governed view of your lending portfolios." | Positioning | The operating-layer claim, plural deliberately: multi-book is the differentiator. Leads on the governed layer rather than on reporting or on AI — both are outputs. |
| "Connect loan data, documents and funding requirements once. Trakt turns them into live portfolio monitoring, forecasting, controls and reporting." | Evidenced | Layer: ingest→canonical→validate→output pipeline (`engine/orchestrator/trakt_run.py`, gates in `README.md`). Monitoring: `mi_agent/risk_monitor/`, `frontend/mi-agent-ui/`. Forecasting: `mi_agent_api/pipeline_prep.py`, `forecast_bridge.py`, `evolution.py`. Covenant controls: `mi_agent/concentration_tests/`, `config/risk/concentration_test_library.yaml`. Reporting: `mi_agent_pptx/`, `engine/gate_5_delivery/`. Funding requirements → controls: `mi_agent/risk_monitor/schedule8_extractor.py` + `config/clients/client_001/risk_limits_extracted.yaml`. "Live" means continuously monitored against the current book — never "real-time"; the pipeline is batch and snapshot-based. |
| Proof point: "One governed portfolio model" | Evidenced | `trakt_core/portfolio.py` registry and scope resolution; the canonical model in `config/system/fields_registry.yaml`. |
| Proof point: "Deterministic, traceable calculations" | Evidenced | Deterministic parser + executor (`mi_agent/llm_query_parser.py` deterministic path, `mi_agent/mi_query_executor.py`); channel parity (`mi_agent_api/tests/test_channel_parity.py`); lineage (`engine/gate_2_transform/lineage_tracker.py`). |
| Proof point: "Reporting, controls and AI from the same data" | Evidenced | One analytical implementation behind every output (`mi_agent_api/mi_service.py`); decks render from the same MI payloads (`mi_agent_pptx/mi_api.py`); controls evaluate the same governed frame (`mi_agent/concentration_tests/`). |
| **Moved out of the hero:** "Every figure is reconciled by construction rather than by comparison." | Evidenced | Now leads §7 Governance — too abstract for a first screen, exactly right as the proof beneath the trust claims. Same evidence: `mi_agent_api/mi_service.py`, `test_channel_parity.py`. |
| Interface preview showing three books, a platform total and a sponsor total | Evidenced | Rendered from `data/demo-pack.json`. Unchanged from Pass 2. |
| Preview footer "Deterministic · As at … · Synthetic portfolio" | Demonstration | A provenance label travelling with the figures. |

---

## 2. Portfolio query demo — "Ask the portfolio. Get a governed answer."

Demo 1 of two. The interactive demo moved here from the intelligence section:
the page shows the product before explaining it. User-started — nothing runs
until the visitor presses "Watch query demo".

| Copy | Class | Evidence |
|---|---|---|
| "Ask portfolio questions in natural language and get answers from the same governed calculations, wherever your team works." | Evidenced | `mi_agent/mi_query_executor.py` + `mi_agent/interpreter/`; channel parity `test_channel_parity.py`; surfaces evidenced in §6. |
| The scripted opening: starting the demo asks "Show the funded balance by book." | Demonstration | `QueryDemo.tsx` mounts `CopilotDemo` with `initialQuestion` (the `balance_by_book` intent); one-shot, consumes one session question, never replays on Reset. The visitor sees query → governed answer from frame zero. |
| The compressed answer: "Both totals are correct — they answer different questions, so Trakt returns both rather than choosing one." + the table | Evidenced | `_balance_by_book` in `scripts/build_demo_pack.py`; the two-totals reasoning is unchanged, the prose is one sentence. The SPV1 story ("originated by the sponsor, securitised and sold; servicing, risk retention and investor reporting retained") moved to a tooltip on the SPV1 row (`row.tooltip`, rendered by `Artifacts.tsx` with `title` + `aria-label`, keyboard-focusable). The former "interpreted" line and the dangling coverage note were removed. |
| Answer footers carry the client name only | Demonstration | `PORTFOLIO_SCOPE` in `api/demo/query/route.ts` is `demoPack.client.name` — no internal portfolio identifier. |
| "Trakt declines what it cannot derive." — **its own page section** (§2b), with the two example prompts as buttons | Evidenced | Extracted from the demo card (`RefusalSection.tsx`): inside the card it sat behind a play control and was visible only as text in a still. Pressing a prompt dispatches to `QueryDemo`, which starts the demo and asks it — so the visitor watches Trakt decline rather than being told it does. Stated once on the page; the poster replica no longer repeats it. Refusal paths unchanged (`src/lib/intents.ts`). |
| The SPV1 row note is a real toggle, not a `title` tooltip | Demonstration | `RowNote` in `Artifacts.tsx`: a button with `aria-expanded` revealing the note inline. Native tooltips never open on touch, which made the explanation unreachable at phone width — asserted now in the mobile Playwright run. |
| The poster before start | Demonstration | A non-interactive replica of the demo surface carrying the real synthetic scope (client, books, exposure count, as-at date) from `data/demo-pack.json`. |
| "Same question. Same calculation. Same answer." | Evidenced | `mi_agent_api/mi_service.py` single implementation; `test_channel_parity.py`. |
| "The portfolios are wholly synthetic, and the page accepts no uploads." | Demonstration | True by construction — no upload endpoint exists. **Single instance on the page**, moved here with the demo it describes. |
| Everything inside the interactive demo (answers, refusals, report previews, session counter) | Evidenced / Demonstration | Unchanged — see the Pass-2 appendix, which remains accurate for the demo surface. |

---

## 3. Platform — "Build the portfolio once. Use it everywhere."

The diagram carries the explanation — Pass 4 removed the surrounding
paragraphs; the "lenses on the same truth" idea lives once, in §5.

| Copy | Class | Evidence |
|---|---|---|
| Data and documents → one governed portfolio layer → every output | Evidenced | The gate sequence in `README.md` and `engine/orchestrator/trakt_run.py`; document intake via `engine/onboarding_agent/file_classifier.py` and `document_extractor.py` (text/markdown; PDF/DOCX is a stated placeholder, so the page never claims a file format). |
| Output chips: Portfolio MI · Forecasting · Risk & covenant controls · Investor reporting · Regulatory reporting · AI & Copilot | Evidenced | Respectively: `analytics/`, `frontend/mi-agent-ui/`; `mi_agent_api/pipeline_prep.py` + `evolution.py` + `forecast_bridge.py`; `mi_agent/concentration_tests/` + `mi_agent/risk_monitor/`; `configs/pptx/investor_pack.yaml` + `mi_agent_api/decks.py`; `config/regime/` + `engine/gate_5_delivery/`; `deploy/copilot-agent/` + `mi_agent_api/copilot_actions.py`. These chips also carry the reporting claims of the retired reporting band — management, investor and regulatory outputs from one layer, with regime names still off the homepage. |

---

## 3. Controls and forward risk

New in Pass 3, and the page's strongest differentiator: the Pass-2 page
compressed all of this into the words "limit monitoring".

| Copy | Class | Evidence |
|---|---|---|
| "Structure covenant and concentration requirements once. Trakt monitors them against the funded book, forecast and pipeline — showing today's position and emerging breaches." | Evidenced | `mi_agent/risk_monitor/schedule8_extractor.py` — deterministic extraction of structured limits (category, value, direction, unit, source snippet, confidence, `needs_review`) from a concentration-limit schedule; committed output `config/clients/client_001/risk_limits_extracted.yaml` (15 limits, 1 flagged for review); tested by `tests/concentration_tests/test_schedule_8.py`. Review/approval before activation: operator-approved `ActiveConfiguration` (`mi_agent/concentration_tests/store.py`, `tests/concentration_tests/test_governance.py`) and the OCC approval workflow (`operations_control/engine.py`). **Deliberate wording:** "structures … for review", never "AI reads your contracts" — extraction is deterministic, text-based and human-reviewed; PDF/DOCX parsing is a placeholder. |
| "Every active control is evaluated three ways — against the funded book, against the expected forecast, and against the full pipeline" | Evidenced | `mi_agent/concentration_tests/forward.py` evaluates each approved test in three explicitly-labelled states: `funded`, `expected_forecast` (pipeline weighted by governed completion probability), `full_pipeline` (labelled a stress, never a prediction). Surfaced in `frontend/mi-agent-ui/src/components/risk/ConcentrationDetailPanel.tsx`, `mi_agent_pptx/concentration.py` and the Teams insight cards. |
| "…with the projected breach horizon when a limit is approaching." | Evidenced | `expected_breach_horizon` and `pipeline_drivers` in `mi_agent/concentration_tests/forward.py`; `identify_emerging_risks`. |
| "Know what is breached today — and what the portfolio is moving toward." | Positioning | The commercial statement of the three-state evaluation above. Carries the green accent as the page's key forward-risk claim. |
| The requirement → reviewed → active lifecycle | Evidenced | The extractor → review → approved-configuration → evaluation chain above. **Pass 4 removed the static path-chips row**: the lifecycle is carried inside the demo itself (its review/activation scene), so it is not stated twice. Activation remains visibly a human decision in the film. |
| Demo heading: "See a portfolio requirement become a live control." · caption "From documented requirement to live monitoring. Figures illustrative." | Positioning / Illustrative | The demo component's own label ("Risk & controls demo") and caption; the illustrative provenance stays in DOM text as well as in the film's burned-in stamp. |
| **Demo behaviour: user-started, never autoplaying, never looping** | Demonstration | `DemoPlayer.tsx`: poster + "Watch controls demo" overlay (`~18 sec`), play/pause/restart controls, a progress bar, and a "Watch again" overlay on completion. `preload="none"` — the asset costs nothing until requested. Reduced-motion visitors see no motion they did not ask for; the static `ControlPreview` renders only if no source is playable. |
| Control preview: Geographic concentration ≤ 30% — Funded 24.1% Pass · Expected forecast 28.7% Warning · Including full pipeline 31.4% Projected breach · horizon Nov 2026 · Single obligor 6.2% Pass | Illustrative | The **workflow** depicted is live (rows above); the **figures** are not engine output and the card is labelled "Illustrative" on the page. This is the single sanctioned exception to "every number on the page comes from the engine" — see the Excluded list. Inside this product depiction the product's RAG semantics apply (mint pass / amber warning / rose projected breach) — documented in `app/globals.css`. |

### The demo loop (`/controls-demo.webm` + `/controls-demo.mp4`)

An 18-second muted autoplay loop rendered by
`demo-video/src/landing/ControlsDemo.tsx` (registered as `LandingControlsDemo`,
rendered by `demo-video/scripts/render.mjs --preset=controls
--preset=controls-webm`; poster by `still-controls.mjs`). VP9 WebM serves
Chromium/Firefox; H.264 MP4 serves Safari/iOS. Embedded by `src/components/site/ControlsDemoLoop.tsx`:
the server sends only the poster, the video mounts when the section approaches
the viewport, playback pauses off-screen, and `prefers-reduced-motion` (or a
failed load) renders the static `ControlPreview` instead — real DOM text, the
same end state.

| Depicted state | Class | Evidence |
|---|---|---|
| Clauses identified in a "Portfolio covenant schedule — extract" (Schedule 4 — Concentration requirements) and structured into controls | Evidenced (workflow), Illustrative (document) | `mi_agent/risk_monitor/schedule8_extractor.py` structures concentration limits from schedule text with source snippets, confidence and `needs_review`; the on-screen document is synthetic, regime- and asset-agnostic, and drawn for the film. **The source is deliberately titled a covenant/concentration schedule, not a full facility agreement — the extractor parses schedule text, and the demo does not imply arbitrary-contract parsing.** The stamp "Illustrative · synthetic data" is burned into every frame. |
| "6 controls identified · 1 flagged for review" | Evidenced (shape) | Mirrors the shipped extractor output shape — `config/clients/client_001/risk_limits_extracted.yaml` carries 15 limits with 1 `needs_review`. The counts on screen are illustrative. |
| Closing line: "Know where the portfolio stands today — and what it's moving toward." | Positioning | Set in the page's heading tone (`ink-100`), not the accent — the final frame reads as product UI, and the risk colours stay inside the statuses and bars. The section's own accent line above the loop is unchanged. |
| Identified → Reviewed → Activated, with "Human review before activation — nothing goes live unapproved" | Evidenced | Operator-approved `ActiveConfiguration` (`mi_agent/concentration_tests/store.py`, `tests/concentration_tests/test_governance.py`); OCC approval workflow (`operations_control/engine.py`). Nothing in the loop activates itself. |
| Funded book / expected forecast / full pipeline evaluation with projected breach horizon | Evidenced (capability), Illustrative (figures) | `mi_agent/concentration_tests/forward.py` — the three labelled states and `expected_breach_horizon`. The 24.1 / 28.7 / 31.4 / Nov 2026 figures are illustrative, stated in the caption ("Figures illustrative") and the burned-in stamp. |
| **Not depicted / not claimed** | — | No autonomous activation, no legal interpretation, no guaranteed extraction of every covenant type, no real-time refresh, no PDF/DOCX parsing (the document is shown as an extract, not a file format), no client data. |

---

## 4. Onboarding — removed from the homepage

Pass 4 demoted the standalone onboarding section to a `<details>` disclosure;
a follow-up removed the disclosure too, so the homepage now makes **no
onboarding claims at all** — the e2e suite asserts "How onboarding works" is
absent and no `#onboarding` section exists. Governance flows directly into
the closing CTA.

The evidence for a future onboarding product page is unchanged and strong:
`engine/onboarding_agent/` (LLM-assisted mapping under a deterministic-first
policy with a review queue), `agents/onboarding_agent.py`, and
`operations_control/` (case management, approval before `activate()`,
publication, recovery, deployed API + UI, 17 test modules). The exclusions
recorded in earlier passes still bind any future onboarding copy: no
"instant"/"automatic" onboarding, no per-portfolio speed escalation, and no
conversational-onboarding claim while the OCC Agent
(`operations_control/occ_agent/`) remains not production-enabled
(`docs/occ_agent/01_operating_process_implementation.md` §11).

---

## 5. Operating model — removed from the homepage

The section is gone. Its claim — "no separate datasets to reconcile" — now
sits in the platform diagram's step 2, where the governed layer is actually
described; the section had been making the same claim 400px further down,
under a headline ("Every relevant lens") that promised lenses it no longer
showed once the scope card was deleted.

The underlying evidence is unchanged and still supports the step 2 line:
`trakt_core/portfolio.py` (`PortfolioRegistry`, `resolve_scope()`,
`ScopeCoverage`), scope rendering in
`frontend/mi-agent-ui/src/components/PortfolioContextSelector.tsx`, and
aggregation in `mi_agent_api/datasets.py`. The book names the section used to
list are still visible by name in the query demo's table.

---

## 6. Portfolio intelligence (the example)

The only demo surface on the page, and the only place the synthetic-portfolio
disclaimer appears.

| Copy | Class | Evidence |
|---|---|---|
| "Ask portfolio questions in Trakt, Microsoft Teams or Microsoft 365 Copilot. Every answer runs against the same governed portfolio calculations and evidence." | Evidenced | `mi_agent/mi_query_executor.py` + `mi_agent/interpreter/`; `deploy/copilot-agent/` (declarative agent, Teams app manifest) + `mi_agent_api/copilot_actions.py`; `mi_agent_api/teams_bot.py` (Bot Framework endpoint, JWKS-validated, fail-closed). **This section is distribution only** — the query demo itself lives in §2, so nothing is demonstrated twice. |
| Suggested question: "Show the current reporting validation exceptions." | Evidenced | Regime-agnostic label for the `annex_exceptions` intent (`scripts/build_demo_pack.py`): the homepage surface stays regime-neutral while the answer remains the engine's genuine Annex reconciliation, and typed regime-specific questions still resolve via the intent's phrase list (`tests/intents.test.ts`). |
| "Approved risk findings can also be delivered proactively into Teams." | Evidenced | `trakt_notifications/` (19 modules: `cards.py`, `teams_client.py`, `outbox.py`, `delivery.py`, `trigger.py`, `recipients.py`); approval writes intent, a worker delivers, dedup/supersession by deterministic batch id; timer-driven outbox drain (`function_app.py:83`); tests under `tests/notifications/` incl. `test_end_to_end.py`; `docs/teams_proactive_notifications.md`. **Corrects Pass 2**, which asserted no bot, no cards, no proactive messaging. |
| Delivery strip: "Available today — Trakt workspace · Microsoft Teams · Microsoft 365 Copilot", each chip carrying a small glyph | Evidenced | Workspace: `frontend/mi-agent-ui/src/components/`. Teams/Copilot: as above. The former delivery-model section, reduced to its substance; managed-service substance (recurring production with no user interaction) is carried by the platform output chips (`apps/blob_trigger_app`, `mi_agent_pptx/cli.py`). **Icon note:** the glyphs are neutral strokes in the page's own icon style (window / people / chat-with-spark) — the repository holds no Microsoft brand assets and the page deliberately does not imitate Microsoft logos; the text labels carry the product names. |
| *(The interactive demo, its disclaimer and its suggested questions now live in §2 — this section carries distribution only.)* | — | — |

---

## 7. Governance and platform

| Copy | Class | Evidence |
|---|---|---|
Pass 4 compressed this section to four one-line cards plus the relocated
reconciliation proof — the long paragraph forms repeated claims already made
elsewhere. The evidence underneath each card is unchanged.

| Copy | Class | Evidence |
|---|---|---|
| Lead line: "Every figure is reconciled by construction rather than by comparison." | Evidenced | Relocated from the hero. `mi_agent_api/mi_service.py` single implementation; `mi_agent_api/tests/test_channel_parity.py`. |
| Card — Deterministic: "Same calculation, every channel." | Evidenced | `mi_agent_api/mi_service.py`, `test_channel_parity.py`; deterministic parser/executor (`mi_agent/mi_query_executor.py`). LLM use remains confined to interpretation and mapping suggestion (`mi_agent/llm_query_parser.py`, `engine/onboarding_agent/llm_*`); calculation is deterministic (`mi_workflows/engine.py`: "no I/O, no LLM"). |
| Card — Traceable: "Every published figure ties back to source." | Evidenced | `engine/gate_2_transform/lineage_tracker.py`, `exception_db.py` (hash-chained), `export_audit_pack.py`, `tests/test_repin_deterministic.py`. |
| Card — Controlled: "Configuration is reviewed before activation." | Evidenced | `operations_control/engine.py` approval/publication; `apps/blob_trigger_app/approvals.py`; operator-approved `ActiveConfiguration` for risk limits; `tests/test_approval_policy.py`. |
| Card — Isolated: "Client environments and authorisation are separated behind Microsoft Entra ID." | Deployment + Evidenced | Entra: `mi_agent_api/copilot_auth.py` (defaults to `entra`, 503 unconfigured), `mi_agent_api/auth.py`. Core enforcement: `trakt_core/tenancy.py` — tenant from `ExecutionContext` only, never the request; `TENANT_MISMATCH` / `PORTFOLIO_NOT_AUTHORISED` before any read; `tests/test_governance_context_and_tenancy.py`, `tests/operations_control/test_tenancy.py`. **Wording note:** "controlled separation between organisations on a common platform" — never "multi-tenant SaaS". `config/tenancy.yaml` does not exist; production deployments are single-tenant per client by design. |
| "Built for specialist lending portfolios on a common canonical model with asset-specific configuration — designed so new lending asset classes are added through configuration and verified through the same pipeline, not by rebuilding the platform." | Evidenced (architecture), deliberately not a coverage claim | `docs/asset_class_hardening_framework.md` + `simulation/` (40 files): equity release, bridge and asset/equipment finance generated and driven through the real Gate 1 → MI → risk pathway, seeded determinism enforced in CI (`.github/workflows/hardening-smoke.yml`); key finding: no new canonical fields required (`config/system/fields_registry.yaml` `portfolio_type` + `--extra-aliases-dir` overlays). **The page claims the architecture, never the classes**: no non-equity-release production client exists and regulatory delivery remains two regimes. |
| "Designed to extend from user-directed workflows toward increasingly agentic operation, within the same governed control framework." | Positioning (future direction, worded as design intent) | Replaces the former explicit ROADMAP block — a roadmap list read as product documentation, not enterprise marketing. The underlying direction is real but unshipped: `trakt_core/context.py` reserved channels `CHANNEL_ENTERPRISE_AGENT` / `CHANNEL_AGENT_TO_AGENT`; `docs/governed_capability_architecture.md` documents the adapter shape; only a test fixture exists. "Designed to extend … toward" claims nothing live, and the e2e suite asserts the section contains neither "roadmap" nor "autonomous". |

---

## 6b. Delivery model — five static tiles

Five modes, one line each, in a static grid (5 columns desktop / 2 tablet /
1 mobile). **Pass 6 replaced the horizontal accordion** added in Pass 5: the
page is a vertical scroll and carries no expand/collapse interaction, so the
whole section is legible in one pass. Copy is trimmed from the ledger-
evidenced delivery claims; the live/roadmap split keeps its accent meaning
(mint shipped, grey not). No client JavaScript — `DeliveryModes` in
`Content.tsx` is a server component.

| Tile | Class | Evidence |
|---|---|---|
| Managed service — "Recurring reporting and governance artefacts, produced with no user interaction." (Available today) | Evidenced | `apps/blob_trigger_app`, `mi_agent_pptx/cli.py`, `export_audit_pack.py`. |
| Trakt Agent — "The full analytical environment: dashboards, charting and drill-through." (Available today) | Evidenced | `frontend/mi-agent-ui/src/components/`. |
| Copilot — "Portfolio questions inside the tools your teams already use." (Available today) | Evidenced | `deploy/copilot-agent/`, `mi_agent_api/copilot_actions.py`, `mi_agent_api/teams_bot.py`. |
| Enterprise agent — "Trakt running inside a client's own agent estate." (Roadmap, grey) | Roadmap | `trakt_core/context.py` `CHANNEL_ENTERPRISE_AGENT` reserved; test fixture only. Labelled Roadmap on the page, never mixed with live capability. |
| Agent-to-agent — "Upstream and downstream systems consulting the layer directly." (Roadmap, grey) | Roadmap | `CHANNEL_AGENT_TO_AGENT` reserved; outbound Teams delivery exists (`trakt_notifications/`), agent callback does not. |

---

## 8. Reporting band — removed in Pass 4

The band restated what the platform section's output chips already claim
(management, investor and regulatory outputs from one layer), so it was
removed to cut repetition. The underlying evidence is unchanged and now
attaches to the §3 chips: `mi_agent_pptx/` + `configs/pptx/investor_pack.yaml`
+ `mi_agent_api/decks.py`; regime projection and delivery in
`engine/gate_4_projection/`, `gate_4b_delivery/`, `gate_5_delivery/`
validated against committed XSDs; recurring production via
`apps/blob_trigger_app` and CLI/API invocation. **Still true and still
enforced:** no regime names on the homepage (ESMA annexes anchor an asset
class and jurisdiction; they belong on product pages), and the page says
*recurring*, never "scheduled", because client scheduling remains a
deployment arrangement.

---

## 9. Contact and footer

| Copy | Class | Evidence |
|---|---|---|
| "See your portfolio through one governed view." | Positioning | The closing restatement of the hero proposition; an offer to demonstrate, not a capability claim. |
| "We will demonstrate Trakt against your own portfolios, funding requirements and Microsoft 365 environment." | Positioning | Microsoft 365 integration is evidenced (`deploy/copilot-agent/`); funding-requirement interpretation is evidenced (`schedule8_extractor.py`). |
| Lead form fields and the no-marketing-list commitment | Deployment | Unchanged from Pass 2 — `src/lib/lead-validation.ts`, `src/lib/leads.ts`; see README § "Who owns incoming leads". |
| Footer: audience line and copyright | Positioning | The audience the brief specifies. |

---

## The green accent

`mint-400` / `#36c2a8`, taken verbatim from
`frontend/mi-agent-ui/src/index.css:21` — the MI Agent's UI green. In the
product it marks a proven state; it does the same job here and no other.

| Where | Why |
|---|---|
| The three trust proof points (tick marks; borders from `sm` upward) | Proven properties of the engine. |
| "Every figure is reconciled by construction rather than by comparison." | The page's strongest verification claim. |
| "Know what is breached today — and what the portfolio is moving toward." | The page's key forward-risk claim, backed by the three-state evaluation. |
| "Available today" and the three delivery chips it heads | A shipped/not-shipped signal. Roadmap stays muted grey. |
| "Trakt declines what it cannot derive." | A governance guarantee. |
| "Deterministic" in the hero preview footer | A confirmatory provenance signal. |
| "Pass" states inside the control preview | Product RAG semantics inside an interface depiction — see the amber note. |

The amber token remains "synthetic, or a boundary Trakt will not cross", with
one documented exception: inside interface previews that depict product UI,
the product's own RAG semantics apply (amber warning, rose projected breach).
The control preview is provenance-labelled "Illustrative" so neither meaning
leaks into page prose. See `app/globals.css`.

Contrast: 8.5:1 on `navy-950`, 8.0:1 on `navy-900` — WCAG AA/AAA. The chart
fill green `#2E7D5B` is still never used as type (3.8:1, fails AA).

---

## Excluded — considered and deliberately not said

| Not claimed | Why |
|---|---|
| SOC 2, ISO 27001, or any certification | Trakt does not hold them. |
| "GDPR compliant" | An organisational assessment, not a code property. |
| A hosting geography | Region is a deployment choice. |
| Penetration tested / independently assessed | No evidence of a completed test. |
| Regulatory approval, authorisation or endorsement | Trakt produces reporting *artefacts*; it holds no permission. |
| "Bank-grade", "military-grade", "zero-risk", "fully secure" | Unfalsifiable. Absent. |
| "AI-powered", "AI reads your contracts", "chat with your CSV", "upload your loan tape" | The AI that ships is confined to interpretation and mapping suggestion under review; document extraction is deterministic and text-based (PDF/DOCX is a placeholder). |
| "Instant onboarding", "automatic onboarding", per-portfolio speed escalation | The claim the repository supports is a governed, repeatable process — §4. |
| Conversational onboarding | Built (`operations_control/occ_agent/`) but not production-enabled; eleven documented preconditions stand. |
| "Multi-tenant SaaS" | Isolation is code-enforced, but deployments are single-tenant per client and `config/tenancy.yaml` does not exist. |
| Any named asset class as supported today (beyond the demo's synthetic scope), or "every lending asset class" | The hardening framework proves the architecture, not production coverage. |
| Regime/annex names on the homepage | They anchor an asset class and jurisdiction; product pages carry them. |
| Enterprise-agent or agent-to-agent availability | Reserved channels; labelled Roadmap in §7. |
| Uptime or SLA figures | None agreed. |
| Named clients, logos, testimonials, customer counts | None available. |
| "Real-time" | The pipeline is batch and snapshot-based. |
| Any figure not produced by the engine | Every number on the page comes from `data/demo-pack.json`, **except** the control preview in §3, which is labelled "Illustrative" on the page and classified Illustrative here. |

**Retired from this list (Pass 2 → Pass 3):** "no timer trigger or cron
anywhere in the repository" (false — `function_app.py:83`); "no Teams bot, no
adaptive cards, no proactive messaging" (false — `mi_agent_api/teams_bot.py`,
`trakt_notifications/`); "exactly three Copilot actions" (the plugin manifest
has since changed; the page no longer counts actions).

---

## Future capabilities — do not claim yet

| Capability | Status |
|---|---|
| Conversational onboarding (OCC Agent) | Built and tested; not production-enabled (`docs/occ_agent/01_operating_process_implementation.md` §11). |
| PDF/DOCX document interpretation | Explicit placeholder in `engine/onboarding_agent/document_extractor.py`. |
| Additional asset classes as production offerings | Architecture proven via `simulation/` + CI; no non-ERE production client; regulatory delivery remains Annex 2/12. |
| Scenario modelling / stress testing as a product surface | `mi_agent_api/scenario.py` is a small what-if; `analytics/scenario_engine.py` is not wired to the API. |
| LLM-generated commentary | Narratives are template-driven (`insight_generators.py`). |
| Public API access for clients | No public developer API is offered. |
| Multi-tenant operation of a single deployment | `config/tenancy.example.yaml` only; single-tenant is the current production shape. |

---

## Pass 2 appendix — the demo books, period-on-period and Annex exception reasoning

The following Pass-2 material remains accurate for the demonstration surface
and is retained verbatim as the reference for demo figures.

### The books

| Book | `source_portfolio_id` | Provenance | Balance sheet | Exposures | Balance |
|---|---|---|---|---:|---:|
| ALP Origination Book | `alp_origination` | direct | warehoused, destined for SPV2 | 47 | £15,432,544 |
| ALP Acquired Back Book | `alp_acquired` | acquired | warehoused, destined for SPV2 | 37 | £11,974,544 |
| SPV1 Sponsored Securitisation | `spv1_sponsored` | direct | **sold and derecognised**; servicing, risk retention and investor reporting retained | 34 | £9,862,973 |
| **Platform total** (warehoused) | — | — | on balance sheet | **84** | **£27,407,089** |
| **Sponsor total** (incl. SPV1) | — | — | everything the sponsor reports on | **118** | **£37,270,061** |

Sum of books reconciles to the sponsor total to the penny at both reporting
dates. Two totals are carried deliberately: a book the sponsor originated,
securitised and sold is off balance sheet and still carries reporting
obligations.

| Claim | Class | Evidence |
|---|---|---|
| Book identity is a first-class attribute, not a query filter | Evidenced | Stamped at Gate 2 (`engine/provenance.py`); resolved by `trakt_core/portfolio.py`; `synthetic_demo/assemble_multibook.py` refuses rows that cannot name their book. |
| "Show the funded balance by book" | Evidenced | `_balance_by_book` in `scripts/build_demo_pack.py`. |
| Period-on-period movement (31 May → 30 June 2026) | Evidenced | `_period_movement`; prior snapshot SHA-256 asserted before comparison. |
| Annex 2 exception reasoning (three seeded exceptions, reconciled dispositions) | Evidenced | `engine/gate_4b_delivery/annex2_exception_reconciler.py`; the seeded `BLOCKS_DELIVERY` / `DEFAULTED_AT_DELIVERY` / `OUT_OF_REGIME_SCOPE` cases as documented in Pass 2. |
| Session question counter, refusal explanations, report previews | Evidenced / Demonstration | Unchanged — `CopilotDemo.tsx`, `src/lib/intents.ts`, `mi_agent_api/` modules as recorded in Pass 2. |
