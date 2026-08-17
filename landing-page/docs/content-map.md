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

**Pass 10 — one claim, stated once, demonstrated four times.** The page was
making its central argument in four vocabularies and demonstrating it in none.
"Same question. Same calculation. Same answer." moves from a 13px caption under
the query demo to the hero, at the page's claim weight, where it frames every
demo below it. The Delivery Model headline stops restating the layer and names
the span of its tiles instead. The agent-to-agent section drops its roadmap
framing: delegation is evidenced, so it is stated in the present tense, the two
agents are named, and the topology diagram is **deleted** — with a sentence
stating the claim and `A2ADemo` performing it, a picture of the same exchange
was the idea a third time.

The proof is the repeated figure. £37,270,061 appears on the hero's Copilot
preview, in the query demo's answer, and must appear in the agent-to-agent demo. Nothing on the page points this out and nothing should:
the claim is asserted once in the hero and then simply keeps being true.

**Pass 9 closeout — three fixes before merge.** The governance grid drops to a
single column below `lg`: five cards in two columns rendered 2+2+1, leaving
"Agent-addressable" alone at half width, which is the same rendering-fault look
the five-column row was introduced to remove. Verified at 1760 / 1440 / 1024 /
834 / 390 and now guarded — the orphan has appeared twice, so the e2e suite
measures the rendered rows at every breakpoint rather than trusting the class
list. The refusal block is tightened onto the demo it belongs to: the demo
figure's own `max-w-4xl`, a soft rule at half the previous gap, type unchanged.
And the agent-to-agent topology is squared up — see §6c.

**Pass 9 also adopted a colour system** — one colour, one job. Recorded below
under "The colour system", and enforced by two e2e guards. Read that section
before changing any accent on the page.

**Pass 9 — section order and the Boundaries fold.** The standalone Boundaries
section is gone; the refusal claim now sits inside the query-demo section,
beneath the frame, above a rule and at heading scale — a sub-claim of the demo
rather than a destination, and no longer in adjacent territory to the
agent-to-agent headline ("Agents don't calculate the portfolio. Trakt does.").
It is **not** small print: the e2e suite measures its computed font size and
fails below 18px. Delivery Model moves above Risk & Controls, so the reader
learns how they would consume Trakt before being shown a capability demo. The
order is now: hero · portfolio query demo (with the refusal claim) · platform ·
delivery model · risk & controls · agent-to-agent · governance · contact.
Governance stays second to last deliberately — reassurance placed early answers
a question the reader has not formed yet. The two demos stay apart, each beside
the claim it proves; batched, three posters read as one repeated thing.

**Pass 8 — strapline, merge, nav, CTA.** The strapline "Agentic portfolio
intelligence. Deterministic by design." is adopted in three places and three
only: the hero eyebrow, the site metadata, and the footer descriptor's opening
phrase. The Portfolio Intelligence section is deleted into the Delivery Model
(§6, §6b). The nav is rebuilt against the sections that exist, its breakpoint
moved `md` → `lg`, and its CTA rewritten to name what the closing CTA offers
(§A). Section order is unchanged. The page is now **nine** narrative sections:
value proposition · portfolio query demo · boundaries · platform · risk &
controls · delivery model · agent-to-agent · governance · contact.

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

Rebuilt in Pass 8 against the sections the page actually has. Two lists, one
source: `LINKS` in `Nav.tsx` carries every navigable section in page order and
flags which appear on the desktop bar.

| Copy | Class | Evidence |
|---|---|---|
| **Mobile menu (6, page order):** Portfolio query · Platform · Capability · Risk & controls · Agent-to-agent · Governance | Positioning | Boundaries was removed in Pass 9 with the section itself. **Delivery became Capability in Pass 12**, when the Delivery Model section was replaced by the capability matrix in the same slot — the entry count and page order are unchanged. |
| **Desktop bar (5):** Platform · Capability · Risk & controls · Agent-to-agent · Governance | Positioning | "Capability" is two characters wider than "Delivery"; the bar was re-measured at 1024 and still does not overflow (the capture probe reads `navOverflow: 0` at 1760/1440/1024). Originally measured: 525px against a 589px budget at 1024, the narrowest width the bar now appears at. Portfolio query is dropped because it opens the page and the hero's primary button already points at it; Boundaries is no longer a section at all. Agent-to-agent is on the bar deliberately — it is the section a technical reader arrives looking for. |
| **Breakpoint moved `md` → `lg`** | — | At 768 the bar's contents measured 701px inside a 704px row. Between 768 and 1023 the page now uses the menu button. |
| **Nav CTA: "Demo on your portfolio"** (was "Book a demo") | Positioning | The page carries demos of its own, so inviting a visitor to book one implied those did not count. The distinction the CTA sells is that it runs against *their* portfolio rather than the synthetic one. **The hero secondary carries the identical label and the identical anchor**, so the page offers this one destination under one name; the e2e suite asserts the two controls agree on both wording and target, because they had already drifted into three names ("Book a demo", "Book a portfolio walkthrough", "Book a tailored demonstration"). The closing CTA keeps "Book a tailored demonstration" deliberately — it sits above the form itself, where it has its own context and competes with nothing. |
| *("Demo" as a nav label is retired: the page carries more than one, so a label naming the format rather than the section told the reader nothing.)* | — | — |

---

## 1. Value proposition

| Copy | Class | Evidence |
|---|---|---|
| **Eyebrow — the strapline: "Agentic portfolio intelligence. Deterministic by design."** | Positioning | Adopted in Pass 8, replacing "Trakt for specialist lending". Two halves doing two jobs: the category the product competes in, and the property that separates it from everything else in that category. "Deterministic" is not a marketing word here — it is the page's most-evidenced claim (`mi_workflows/engine.py` "no I/O, no LLM"; `mi_agent_api/tests/test_channel_parity.py`). **Stated on the page exactly once.** It is a strapline, not a section heading, and must not be repeated as body copy. Type treatment is the eyebrow's existing style, measured at 555px against a 656px column at 1440 and 770px at 834; at 390 the column is 350px and the second sentence is a `block` below `sm` so the wrap lands on the full stop by construction. |
| "One governed view of your lending portfolios." | Positioning | The operating-layer claim, plural deliberately: multi-book is the differentiator. Leads on the governed layer rather than on reporting or on AI — both are outputs. |
| "Connect loan data, documents and funding requirements once. Trakt turns them into live portfolio monitoring, forecasting, controls and reporting **for specialist lenders**." | Evidenced | Layer: ingest→canonical→validate→output pipeline (`engine/orchestrator/trakt_run.py`, gates in `README.md`). Monitoring: `mi_agent/risk_monitor/`, `frontend/mi-agent-ui/`. Forecasting: `mi_agent_api/pipeline_prep.py`, `forecast_bridge.py`, `evolution.py`. Covenant controls: `mi_agent/concentration_tests/`, `config/risk/concentration_test_library.yaml`. Reporting: `mi_agent_pptx/`, `engine/gate_5_delivery/`. Funding requirements → controls: `mi_agent/risk_monitor/schedule8_extractor.py` + `config/clients/client_001/risk_limits_extracted.yaml`. "Live" means continuously monitored against the current book — never "real-time"; the pipeline is batch and snapshot-based. **"for specialist lenders" was added in Pass 8**: the strapline took the eyebrow's slot, and without this the page named no audience above the fold — the headline says "lending portfolios", not *specialist* lending, and `<title>`/description are not visible. Audience scope is unchanged and unextended (see the footer descriptor). |
| **"Same question. Same calculation. Same answer."** — beneath the sub-copy, above the proof chips | Evidenced | `mi_agent_api/mi_service.py` is the single analytical implementation behind every channel; `mi_agent_api/tests/test_channel_parity.py` asserts the channels agree. **Promoted here in Pass 10** from a 13px caption under the query demo, where it was the page's central claim doing footnote duty and read as commentary on one demo. In the hero it frames all four surfaces below it, and it sits beside the Copilot preview — the first place the reader sees £37,270,061. Treatment is `text-lg font-medium text-ink-100`, the page's established claim grammar (risk & controls, governance, agent-to-agent), not a fourth style. **Stated once on the page.** |
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
| "Trakt declines what it cannot derive." — **inside this section, beneath the demo frame**, with the two example prompts as buttons | Evidenced | `RefusalSection.tsx`. It has moved twice, and both moves were corrections. It began inside the demo card, where it sat behind a play control and was visible only as text in a still. Pass 8 made it its own section (§2b), which over-promoted a sub-claim of the demo into a destination and put it in adjacent territory to the agent-to-agent headline. **Pass 9 folded it back in**, above a rule and at `h3` scale — the conclusion the demo has just earned, not a caption. Pressing a prompt dispatches `ASK_EVENT` to `QueryDemo`, which starts the demo and asks it, so the visitor watches Trakt decline rather than being told it does. Stated once on the page; the poster replica does not repeat it. Refusal paths unchanged (`src/lib/intents.ts`). **Guarded:** the e2e suite asserts `#refusal` no longer exists, that the claim appears exactly once, that a prompt still drives the demo to a refusal, and that the computed font size is ≥18px — so a later pass cannot quietly return it to small print. |
| The SPV1 row note is a real toggle, not a `title` tooltip | Demonstration | `RowNote` in `Artifacts.tsx`: a button with `aria-expanded` revealing the note inline. Native tooltips never open on touch, which made the explanation unreachable at phone width — asserted now in the mobile Playwright run. |
| The poster before start | Demonstration | A non-interactive replica of the demo surface carrying the real synthetic scope (client, books, exposure count, as-at date) from `data/demo-pack.json`. |
| *("Same question. Same calculation. Same answer." has moved to the hero — see §1. It was the page's central claim rendered at 13px under one demo.)* | — | — |
| **Total rows in an answer table render as totals** — semibold, `ink-100` | Demonstration | `isTotalRow` in `Artifacts.tsx`. Not tidiness: the sponsor total is £37,270,061, the same figure the hero's Copilot preview already showed, and the hero's claim is that every surface returns the same answer. Rendered as one plain 14px cell among five identical rows, the repetition was invisible unless the reader was already looking for it. Keyed off the first column's own label rather than a flag on the data, because the demo pack is engine-generated and hash-checked and should not carry styling hints. In the committed pack this matches four rows across two artifacts — the platform and sponsor totals in the balance and movement tables — and nothing else. |
| "The portfolios are wholly synthetic, and the page accepts no uploads." | Demonstration | True by construction — no upload endpoint exists. **Single instance on the page**, moved here with the demo it describes. |
| Everything inside the interactive demo (answers, refusals, report previews, session counter) | Evidenced / Demonstration | Unchanged — see the Pass-2 appendix, which remains accurate for the demo surface. |

---

## 3. Platform — "Build the portfolio once. Use it everywhere."

The diagram carries the explanation — Pass 4 removed the surrounding
paragraphs; the "lenses on the same truth" idea lives once, in §5.

| Copy | Class | Evidence |
|---|---|---|
| Data and documents → one governed portfolio layer → every output | Evidenced | The gate sequence in `README.md` and `engine/orchestrator/trakt_run.py`; document intake via `engine/onboarding_agent/file_classifier.py` and `document_extractor.py` (text/markdown; PDF/DOCX is a stated placeholder, so the page never claims a file format). |
| Step 3 body: **"Portfolio, control, reporting, investigation and delivery."** | — | **Replaced the six output chips in Pass 12.** The chips — *Portfolio MI · Forecasting · Risk & covenant controls · Investor reporting · Regulatory reporting · AI & Copilot* — were an ungrouped, weaker version of §3b's matrix, sitting one section above it, and one of them made a claim the page must not: an unqualified **"Forecasting"** reads as funded-book forecasting, and Trakt forecasts the pipeline. The replacement is a bare enumeration of §3b's five group labels — no verb, no claim — so the diagram indexes the matrix instead of competing with it. **The box was kept rather than reduced to a pointer:** the arrow motif between the boxes is drawn for three, and two boxes with one arrow reads as an unfinished diagram. Naming the five groups as chips was the other candidate and was rejected — it would print the same five words twice within 200px. The chips\' underlying evidence has not disappeared; it now attaches per item in §3b. |

---

## 3b. Capability — "One portfolio. Every workflow." (`Capability.tsx`)

**New in Pass 12.** The Platform section says "Build the portfolio once. Use it
everywhere." and nothing on the page said what *everywhere* meant. This does,
as a matrix rather than prose: seventeen items in five groups is a shape a
reader finds their own job in, not an argument to follow.

**Placement** is immediately after §3 and before Risk & Controls, in the slot
the Delivery Model section used to hold. A matrix of capabilities read before
the layer that produces them is a feature list.

**No item descriptions, no paragraph.** A line under each item would make this
a fourth explanatory section. The items are nouns a lender already knows, and
the depth lives in the sections around it — §2 for MI Query, §3 for the
controls, §6c for the agents. **Overview here, depth there**, which is the
relationship the INVESTIGATE column and §6c are in deliberately: the matrix
names the two agents, §6c gives them bodies and a recorded run.

**No status markers, from Pass 13.** Portfolio Acquisition DD and Enterprise
agent A2A carried a ROADMAP label for one pass — grey caps plus an `sr-only`
"(roadmap)" — and both were removed by decision. The matrix now reads as one
flat capability set, and the e2e suite asserts that: no label, no `sr-only`, and
every item resolving to a single colour, because a muted item with no label
would be the worst of both.

**What that costs, stated once and not re-argued.** Two items' repository
evidence does not match an unqualified shipped claim, and the qualification now
lives only here:

  * **Portfolio Acquisition DD** — no implementation found. `due_diligence/`
    holds Trakt's own internal review documents and an Annex 2 mapping-impact
    script; there is no agent, session or tool that assesses a portfolio being
    acquired.
  * **Enterprise agent A2A** — the mechanism is demonstrated (`trakt_a2a/`,
    `enterprise_agent/client.py`, `tests/test_a2a_delegation.py`, one recorded
    run) but the card advertises a single skill and this is a proof rather than
    a channel a client uses.

The page is the client's to state; the ledger's job is to record what the
repository shows, and it does.

**No colour is spent here.** Group labels are `peri-400`, the eyebrow role;
items are `ink-300`; roadmap items `ink-500`. No mint, because the matrix
states availability in words and does not need to state it twice.

### The seventeen items

| Group | Item | Class | Grounding |
|---|---|---|---|
| Portfolio | Pipeline | Evidenced | `analytics/pipeline_expected_funding.py`, `pipeline_forward_risk.py`, `pipeline_reconciliation.py`, `pipeline_snapshot_selector.py`, `pipeline_persistence.py`, `pipeline_prep.py`. **The word is "Pipeline", never "Forecasting":** Trakt forecasts the pipeline, not the funded book, and the generic word claims the second. |
| Portfolio | Total AUM | Evidenced | `demo_platform/metrics.py` ("Sponsor AUM"), `mi_agent_pptx/deck.py`. The demo pack's two governed totals — platform and sponsor — are the same measure on the page. |
| Portfolio | Portfolio / asset / SPV views | Evidenced | `trakt_core/resource.py` and `trakt_core/entitlement.py` (the scoping model); `mi_workflows/portfolio_risk_comparison.py`, whose vocabulary carries "portfolio", "book", "spv", "warehouse pool". |
| Control | Concentration limits | Evidenced | `mi_workflows/concentration_analysis.py`, `operations_control/concentration.py`, `mi_agent_pptx/concentration.py`. Demonstrated on the page by the controls film. |
| Control | Proactive risk watchlist | Evidenced | `mi_agent_pptx/watchlist.py`; `trakt_notifications/risk_review.py`, which produces a risk review for every approved update *including the empty ones* — "a silent week is indistinguishable from a broken pipeline". The delivery half is `trakt_notifications/` (outbox, dedup, timer drain). |
| Control | Governance & lineage | Evidenced | `engine/provenance.py`, `engine/validation_agent/`, `operations_control/audit_chain.py`, `trakt_tools/handlers/provenance.py`. Also the claim §7 makes in full. |
| Report | MI Analytics | Evidenced | `mi_agent_pptx/` — 20 modules: deck builder, chart and metric resolvers, cohorts, movement, insights, preflight — and `analytics/` for the measures beneath them. **Renamed from "Management MI" in Pass 13:** the old label named an audience, and what the group lists is what the thing produces. |
| Report | Warehouse reporting | Evidenced | `config/organisations.example.yaml` and `config/entitlements.example.yaml` model `organisation_type: warehouse_funder` as a party that signs in *from its own Microsoft directory* and sees its funded population only; enforced by `trakt_core/entitlement.py`. Reporting **to** a warehouse funder, scoped by entitlement — not a report template. |
| Report | Securitisation reporting | Evidenced | `operations_control/annex2/` — governed wrappers over the proven Annex 2 route: `engine/gate_4_projection/regime_projector.py`, `engine/gate_4b_delivery/annex2_delivery_normalizer.py`, `engine/gate_5_delivery/xml_builder_annex2.py`, validated against the ESMA XSD. **Never shortened to "Securitisation":** it has to stay distinguishable from "Securitisation Readiness" two groups along, which is a different thing done by a different component. |
| Investigate | MI Query | Evidenced | `mi_agent/mi_query_executor.py` + `mi_agent/interpreter/`. Demonstrated on the page by §2. |
| Investigate | Securitisation Readiness | Evidenced | `readiness_agent/` (agent, session) over `trakt_tools/registry.py` and its handlers. Demonstrated on the page by §6c's recorded run. |
| Investigate | **Portfolio Acquisition DD** | **Unmarked — see the status note above** | **No implementation.** `due_diligence/` holds Trakt's own internal review documents plus `build_annex2_impact_report.py`, which is Annex 2 mapping impact — not an assessment of a portfolio being acquired. No agent, no session, no tools. |
| Delivery | Managed service | Evidenced | `operations_control/` (67 modules — the OCC: intervention, approval, population, preflight, stages), `apps/blob_trigger_app`, `mi_agent_pptx/cli.py`, `export_audit_pack.py`. **First in the column deliberately:** it is the primary commercial delivery mode, and deleting the Delivery Model section would otherwise have removed it from the page entirely. |
| Delivery | Trakt | Evidenced | `frontend/mi-agent-ui/src/components/`. |
| Delivery | Microsoft Copilot | Evidenced | `deploy/copilot-agent/`, `mi_agent_api/copilot_actions.py`. |
| Delivery | Microsoft Teams | Evidenced | `mi_agent_api/teams_bot.py` (asking); `trakt_notifications/` (arriving unasked — the claim now stated in §3). |
| Delivery | **Enterprise agent A2A** | **Unmarked — see the status note above** | The mechanism is demonstrated — `trakt_a2a/` (server, card, identity, tasks), `enterprise_agent/client.py`, `tests/test_a2a_delegation.py`, and one recorded run — but it is a proof, not a channel a client uses today. **Availability and demonstration are different claims.** The ROADMAP marker that stated the difference was removed in Pass 13. |

### Responsive behaviour

Five cards is the number that produced the same defect twice in the governance
grid — one card alone on its row, reading as a rendering fault rather than a
wrap. Measured at every breakpoint the grid changes at:

| Width | Layout | Card widths |
|---|---|---|
| 1760 | 5 across | 307 × 5 |
| 1440 | 5 across | 256 × 5 |
| 1024 | 3 + 2 | 299 × 3, 299 × 2 |
| 834 | 2 + 2 + 1 full width | 377 × 2, 377 × 2, **770** |
| 390 | single column | 350 |

The 834 row is why DELIVERY carries `sm:col-span-2 lg:col-span-1`: at two
columns the fifth card would otherwise sit alone at half width beside an empty
cell. It is the card with five items, so it earns the width rather than merely
filling it. **The row-occupancy guard was rewritten in the same pass** — it
used to flag any tail row of one, which would have failed this deliberate
full-width card; it now measures the tail's *width* against a full row and only
fails when the card is genuinely narrow. It runs over both five-card grids.

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
| **Demo behaviour: user-started, never autoplaying, never looping** | Demonstration | `DemoPlayer.tsx`: poster + "Watch controls demo" overlay (`~18 sec`), play/pause/restart controls, a progress bar, and "Watch again" on completion. `preload="none"` — the asset costs nothing until requested. Reduced-motion visitors see no motion they did not ask for; the static `ControlPreview` renders only if no source is playable. **Pass 11 moved the transport out of the frame.** It had been pinned across the bottom of the picture for the whole run, and the A2A film draws to the frame edge, so its closing lines were covered from the moment play was pressed; the completion state was worse, putting a scrim and a centred button over the last frame — which, for a demo that ends on its assessment, is the frame most worth reading. Auto-hiding was rejected: on a phone there is no hover, so the controls would return only by tapping the thing you are trying to watch. Cost is ~40px of layout below the frame while playing. **Guarded by measurement** — the progress bar's box must not overlap the video's. |
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
| **"Illustrative · synthetic data", burned into every frame — RETAINED DELIBERATELY** | Illustrative | Do not remove this in a later pass; it was proposed for removal as duplicative caveat noise and deliberately kept. It is **not** the same claim as the query demo's amber "Synthetic data" pill. The pill says the *portfolio* is synthetic while the figures are genuine engine output; this stamp says the *figures themselves* (24.1 / 28.7 / 31.4 / Nov 2026) are invented for the film. On a page whose claims are deterministic, traceable and reconciled by construction, presenting invented numbers as system output would be a real honesty cost. The `ControlPreview` fallback badge was redundant with it and is gone; this one stays. |
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

## 6. Portfolio intelligence — removed from the homepage

**Deleted in Pass 8.** The section listed the same thing as the delivery model
two sections below it, in a second format: its chips read *Trakt workspace ·
Microsoft Teams · Microsoft 365 Copilot* while the tiles read *Managed service ·
Trakt Agent · Copilot · Agent access*. "Trakt workspace" and "Trakt Agent" were
one surface under two names, and Copilot appeared in both. Same duplication the
Operating Model section carried before it was folded into the platform diagram.

Gone with it: the eyebrow, the headline "Portfolio intelligence where your team
already works.", the three channel chips and their neutral glyphs, and the
"Available today" strip label — the tiles carry availability themselves. The
`DeliveryStrip` component, `DELIVERY_SURFACES` and `ChannelIcon` are deleted
from `Content.tsx`; the e2e suite asserts `#intelligence` no longer exists and
that "Trakt workspace" appears nowhere on the page.

**One claim carried forward**, verbatim, as the delivery section's body line —
see §6b. It was the only line the tiles did not already cover.

The underlying evidence is unchanged and attaches to the surviving copy:
`mi_agent/mi_query_executor.py` + `mi_agent/interpreter/`; `deploy/copilot-agent/`
+ `mi_agent_api/copilot_actions.py`; `mi_agent_api/teams_bot.py`;
`frontend/mi-agent-ui/src/components/`. **Icon note, retained for whoever
reinstates channel glyphs elsewhere:** the repository holds no Microsoft brand
assets, the page never imitated Microsoft logos, and the text labels carried the
product names.

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

## 6b. Delivery model — DELETED in Pass 12

The section is gone: `#delivery`, `DeliveryModes`, `DELIVERY_MODES`, the four
tiles, the 01–04 numerals and the headline. Its four tiles said what §3b's
DELIVERY column now says in five items, two sections below the matrix that
repeated them — the same duplication removed from Operating Model, Portfolio
Intelligence and Boundaries before it.

Where each part went:

| Part | Disposition |
|---|---|
| The four tiles | §3b's DELIVERY column, as five items — **Managed service** added at the head, because deleting this section would otherwise have removed the primary commercial delivery mode from the page. |
| Body line **"Approved risk findings are pushed to Teams."** | To §3, Risk & Controls, as a second line under the claim. It is a statement about what happens to a *breach*, not about a delivery channel, and it now sits beside the demo that produces the finding. Evidence unchanged: `trakt_notifications/` — 19 modules, approval writes intent, a worker delivers, dedup and supersession by deterministic batch id, timer-driven drain at `function_app.py:83`, `tests/notifications/test_end_to_end.py`, `docs/teams_proactive_notifications.md`. **Still asserted to appear exactly once on the page.** |
| Headline **"From a managed service to your own agents."** | **Not rehomed, deliberately.** The ladder it named is carried by the DELIVERY items themselves — managed service at one end, A2A at the other — and a positioning headline with no section under it is furniture. Recorded as a real loss rather than a wash: the page no longer states the progression in words. |
| The 01–04 numerals | Gone with the headline whose argument they illustrated. They lived for one pass. |
| The nav entry | `Delivery → #delivery` became `Capability → #capability`. Six entries, unchanged page order. |

**The tile copy is not carried forward** and does not need to be: it had been
rewritten one pass earlier into second-person capability sentences, and §3b
states the same channels as bare nouns.

---

## 6c. Agent-to-agent (`AgentSection.tsx`)

New in Pass 7 as a roadmap section. **Pass 10 removed the roadmap framing**:
delegation is proven, so the claim is present tense, the "Roadmap" label is
gone and the "is designed to" hedge with it. Sits after the delivery model and
before governance, so the reader meets the delivery surfaces first and the
governance answer immediately after.

| Copy | Class | Evidence |
|---|---|---|
| Eyebrow: "Agent-to-agent" · heading "Agents don't calculate the portfolio. Trakt does." | Positioning | The commercial statement of the architectural boundary the page already argues: calculation is deterministic and lives in the engine (`mi_workflows/engine.py` — "no I/O, no LLM"), not in a language model. |
| **"An agent that knows nothing about Trakt can discover it and delegate an objective."** | Evidenced | The section's claim, and the strongest on the page. **Discovery and delegation are the claim** — not that an agent called a known endpoint. Stated once, under the headline, in the page's claim grammar (`text-lg font-medium text-ink-100`). It leads with the part that is hard to believe — no prior knowledge — because that is what makes it a claim rather than a description of an API call. |
| "Your agents already have somewhere to work. Give them somewhere trustworthy to get credit intelligence." | Positioning | No capability claimed. Supporting line, `ink-400`, below the claim. |
| **The delegation film — the section's one visual** | Demonstration | `demo-video/src/landing/A2ADemo.tsx`, composition `LandingA2ADemo` (1200×960, 50s), rendered by `scripts/render.mjs --preset=a2a --preset=a2a-webm`; poster from the dedicated `LandingA2APoster` composition via `still-a2a.mjs`. Embedded through `DemoPlayer` exactly as the controls film is: poster, play control, pause and replay, `preload="none"`, and `A2APreview` as the text equivalent when no source is playable. **It replaces an animated CSS component** that shipped in the same PR and was superseded within it — that component autoplayed, had no poster and no controls, which the page's own rule forbids. The film's disclosure is burned into every frame as "Illustrative · synthetic data", so the amber pill returns to being stated exactly once on the page, in the query demo. |
**The one-line noun-phrase rule for tile bodies is retired.** It governed the
Delivery Model tiles and was carried across to these two by habit. The
second-person capability voice below was asked for and is the standard for tile
bodies from Pass 11 on; the rule should not be cited against copy that was
requested. What remains is a *length* budget, not a grammar: both bodies were trimmed to
18 words in Pass 12 — 28 and 29 words read long beside a delegation line and a
supporting line in the same block. The grammar slip in the first tile
("Points it" → "Point it") was fixed in the same pass.

| Tile — **Securitisation Readiness Agent**: "Point it at a warehouse: eligibility, coverage, concentration — and what fails a deal, before the arranger says so." | Evidenced | The agent the recorded run exercises, so its evidence is the run itself. **"before the arranger says so"** is the line's whole point: the agent is not summarising a book, it is anticipating the counterparty who will reject it. **Trimmed 28 → 18 words in Pass 12.** The clause that went — "it works the whole book" — was already implied by naming the three checks. |
| Tile — **Portfolio Acquisition Intelligence Agent**: "Reads the portfolio you are buying, finds what the seller's summary leaves out — and what it cannot resolve." | Evidenced | **"What the seller's summary leaves out"** is the diligence buyer's actual fear, stated plainly; **"what it cannot resolve"** is the part almost no tool will admit to. **Trimmed 29 → 18 words in Pass 12, and evidence-tracing is what was dropped** — deliberately. It is the page's ambient claim, made by the hero, by §7 and by the A2A demo's own description, whereas the seller's-summary line is unique to this tile. **The full name is kept** although it is the longer of the two — "Portfolio" is what separates this from corporate M&A diligence, which is worth more than symmetry between two labels. |

| Tile — **Portfolio Surveillance Agent** (first): "Reports the whole position in words: balance, movements, the trends behind them — and what is starting to turn." | Deployment | **New in Pass 13, renamed from "MI Query Agent" and moved to the head of the row in Pass 14.** All four things the copy names exist: balance from the governed totals; movements from `mi_agent_pptx/movement.py` ("*which* regions, channels, LTV bands and ticket bands moved, and by how much, is the analysis"); trends from `mi_agent_pptx/insights.py`; emerging risk from `identify_emerging_risks` in the forward-risk module, surfaced through `mi_agent_api/concentration_tests_api.py` and `trakt_notifications/risk_review.py`. **"In words" is a deterministic claim, not an LLM one** — `insights.py` states it outright: "There is **no LLM anywhere** — not in generation, not in wording, not in selection. Every sentence is a template over a figure that a governed compute function already produced." That is the section headline applied to sentences instead of sums, and it is why this tile can lead. **First in the row** because it is what an agent asks for before it asks for anything else: what is the position, and what changed. Classified **Deployment** rather than Evidenced for the same reason as its neighbours: `trakt_a2a/card.py` advertises exactly one skill today, `securitisation_readiness_assessment`, so exposing this over A2A is a card entry and a handler binding on an existing boundary — not new analytics. Named "Agent" like its neighbours because what an agent delegates to is an agent, not an endpoint. |

**Pass 14 — the third tile is renamed and leads.** "MI Query Agent" named the
mechanism; "Portfolio Surveillance Agent" names what it produces, and the copy
was rewritten to say what that is. It moved to the head of the row because the
position and what changed is the first thing anything asks for. The e2e guard
asserts rendered *order* now, not just presence, in reading order — so it holds
at one, two and three columns.

**Pass 13 — a third tile, and the grid changes with it.** Three cards orphan in
a two-column layout exactly as five do, so the last tile takes
`sm:col-span-2 lg:col-span-1` and the grid opens to three columns at `lg` — the
resolution the capability matrix already uses. The row-occupancy guard now runs
over three grids: governance, the matrix, and these tiles.

**Pass 11 — the agents lead the section; the film follows.** They were small
named lines *beneath* a fifty-second video: the products of the section, ranked
below a recording of one of them, in a position most readers never reach. They
are now outlined tiles above the demo, with a line glyph each, and the reader
meets the claim before the evidence for it. Asserted by geometry in the e2e
suite — each tile's box must sit above the video's — because a grid change
could reorder them visually while the markup still read correctly.

**Their outline was mint for one pass and is now `border-line`.** Mint means
available, and §3b differentiates these two agents on evidence: Securitisation
Readiness ships, Portfolio Acquisition DD is marked ROADMAP, and the A2A
channel itself is marked ROADMAP. A mint outline on both tiles would have this
section contradicting the matrix two screens above it. The
`data-state-colour="agent-availability"` marker went with the tint — there is
no system state left in this row to declare.

This is not a reopening of the delegation claim, which stands: **the mechanism
is demonstrated and the channel is not yet shipped**, and those are different
statements. §6c makes the first. §3b's DELIVERY column makes the second.

**The glyphs are peri, not mint**, because a glyph marks what a thing *is*, not
what state it is in. Two inline SVGs in `AgentSection.tsx` — a tranched stack
and a portfolio under examination — rather than an icon dependency for two
marks.
| **The topology diagram is deleted** | — | Removed in Pass 10, not hidden or commented out. It was scaffolding drawn while the delegation claim could not be made in words; with both a sentence stating the claim and a demo performing it, a diagram of the same exchange was the idea a third time — the duplication removed from Operating Model, Portfolio Intelligence and Boundaries before it. **Its removal also deleted the `Fragment` usage that broke the deploy** when the A2A branch and the topology restructure were merged: fewer moving parts in the file that two streams of work both touch. |

### Constraint on the Securitisation Readiness render

**Its result must land on £37,270,061 — the sponsor total — not a new figure.**

The hero asserts "Same question. Same calculation. Same answer." and the page
then demonstrates it by showing the same number on surface after surface: the
hero's Copilot preview, the query demo's answer table (now emphasised as a
total so the repetition is visible), the controls demo's portfolio, and finally
an agent's response. A fourth surface returning a *different* number would
quietly refute the page's central claim, and no caption could repair it.

This binds the render, not the copy. Nothing on the page points the repetition
out and nothing should — the claim is asserted once and then keeps being true.

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
| Footer: "**Agentic portfolio intelligence** for specialist lenders, non-bank lenders, private-credit managers, servicing businesses and securitisation participants." + copyright | Positioning | The audience the brief specifies. Pass 8 aligned the opening phrase with the strapline so the site runs one product description rather than two; the audience list is the part the strapline does not carry, which is why it stays. |

### Site metadata (`src/app/layout.tsx`)

| Copy | Class | Evidence |
|---|---|---|
| `<title>` — "Trakt \| Agentic portfolio intelligence. Deterministic by design." | Positioning | The strapline verbatim, as the title, because this is what a search result and a link preview surface. Was "Trakt \| Governed Portfolio Intelligence". |
| Meta / OpenGraph / Twitter description — the strapline, the existing sentence ("Trakt connects portfolio data … for specialist lenders."), then **"Enterprise agents discover Trakt's agents and delegate an objective, and get an evidence-backed result."** | Positioning + Evidenced | One `STRAPLINE` constant feeds the title and opens the description, so the two cannot drift. The delegation clause was added in Pass 10 to the **description only** — title, OpenGraph title and the strapline are unchanged, because delegation is a proof of the "agentic" category the strapline already claims rather than a different position. |
| OpenGraph image (`opengraph-image.tsx`) — **untouched** | — | It renders the headline ("One governed view."), not the eyebrow, so the strapline change does not reach it. Its `alt` export deliberately still reads "Trakt \| Governed Portfolio Intelligence"; change it only alongside the image itself. |
| `trakt:pack` meta tag | — | Unchanged. Deploy-gate provenance, not copy. |

---

## The colour system — a standing rule

Adopted in Pass 9 after an audit found **green doing six unrelated jobs**:
availability on the delivery tiles, pass state in the control preview, emphasis
on two claim lines, hero tick marks, a provenance label, and the border on
refusal prompts and demo suggestion chips. A colour with six jobs signals
nothing. Each family below now has exactly one role; adding a second is a
regression, not a refinement. Full rationale and contrast figures live in the
token block at the top of `src/app/globals.css`.

| Colour | Sole role | Where it may appear |
|---|---|---|
| **Green** `mint-400` | **System state**: pass, available, healthy | The "Available today" label and tile border (`DeliveryModes`) · a passing evaluation in `ControlPreview` · the lead form's success panel. Nowhere else. |
| **Amber** `amber-400` | **Disclosure**, and warning state | The "Synthetic data" badge · a book marked sold and derecognised · the declined-answer treatment · a warning evaluation in the control preview |
| **Red** `rose-400` | **Something is wrong** | A breached or projected-breach limit · a rejected form input · a failed request |
| **Blue** `peri-400` | **Meaning: section eyebrows only** | `SectionHeading`'s eyebrow, and the hero eyebrow carrying the strapline |
| **Blue** `peri-300` / `peri-500` | **Structure and interaction** | Buttons, hover borders, progress bar, platform arrows, proof-point ticks, the Trakt node outline in the topology |
| **White** `ink-100` | **Every claim and emphasis line** | The risk & controls line and the governance line, at `text-lg font-medium` |

Three points that the wording has to keep, because each one is how a system
like this drifts:

1. **Amber is DISCLOSURE, not status.** "This book is off balance sheet" is a
   disclosure — the reader is being told something about the nature of what
   they are looking at. Written as "status", a later pass will colour arrears
   or maturity amber and the family is gone.
2. **The periwinkle rank matters.** `peri-400` is meaning; `peri-300` and
   `peri-500` are structure. Never reach for `peri-500` as an accent on type:
   it measures 5.67:1 on navy-950, which is fine behind a border and fails as
   body copy.
3. **A claim is not a state.** Where a claim needs weight it takes type — size,
   weight, spacing — never an accent.

**Guards** (`e2e/landing.spec.ts`), the same class as the transparency sweep:

- *"green marks system state and nothing else"* — walks every element on the
  rendered page and fails on any mint outside a container that declares itself
  `data-state-colour`. The allow-list lives in the components, not the test.
  Run twice: on load, and again with the query demo running, because the
  suggestion chips and answer cards only exist after it starts.
- *"peri-500 is a border and structural colour, never type"* — fails if
  `peri-500` is the computed text colour of anything that is not inside an
  `aria-hidden` subtree. The platform arrows are aria-hidden glyphs, so the
  page's own markup separates a structural mark from type.

**Both guards resolve colours by painting, not by string matching.** Tailwind
emits opacity modifiers as `oklab(… / .35)` and Chromium reports that verbatim,
so the first version of the green guard — which looked for `rgb(54, 194, 168`
— passed happily while a green border still sat on the refusal prompts. Filling
the same colour over black and over white recovers the source channels and the
alpha whatever colour space the value arrived in. **Both guards were verified
by reintroducing the drift and watching them fail**; a guard that has never
failed is not known to work.

**No re-render needed.** The controls film and its poster use mint for a
passing evaluation, amber for a warning and rose for a projected breach — the
state rule exactly — and the burned-in "Illustrative · synthetic data" stamp is
amber, the disclosure rule exactly. The video was audited against the system
and conflicts with none of it.

---

## The green accent — superseded by the colour system above

Kept as a record of what changed, and of the four usages that were **removed**
in Pass 9, so nobody restores them believing they were an oversight:

| Removed | Why it went |
|---|---|
| Tick marks and borders on the three hero proof points | Decoration, not a state. Ticks are now `peri-300`, borders `border-line`. |
| "Every figure is reconciled by construction rather than by comparison." | A claim, not a state. Now `ink-100` at `text-lg font-medium`. |
| "Know what is breached today — and what the portfolio is moving toward." | Same. Weight through type, not colour. |
| "Deterministic" in the hero preview footer | A provenance label, not a state. Recoloured to `ink-400` in Pass 9 and **deleted outright in Pass 11**: with no figure to qualify and no rule beside it, a lone word under the sponsor total read as a stray label rather than a claim about the numbers above it. The claim is made in the hero copy and demonstrated by the demo. |
| Borders on the refusal prompts and the demo suggestion chips | Affordance styling. Worse than arbitrary: green on a *refusal* control read as a success state. Now `border-line` with a `peri-500` hover. |

**Retained**, because each is genuinely a system state: the "Pass" row in the
control preview, and the lead form's success panel. Each sits inside a
container declaring `data-state-colour`, which is what the e2e guard reads.

Two green usages were added and removed inside two passes, which is worth
recording rather than tidying away. The delivery tile borders went in Pass 11
when the "Available today" labels were dropped and the outline took over
carrying availability; they went out in Pass 12 with the section itself. The
agent tile borders went in and out on the same schedule, and for a better
reason — §3b now states availability in words, and one of the two agents is
marked ROADMAP there, so mint on both tiles was asserting something the matrix
denies. **Green now marks state in exactly two places on the page**, which is
the fewest it has ever marked.

Contrast: 8.5:1 on `navy-950`, 8.0:1 on `navy-900` — WCAG AA/AAA. The chart
fill green `#2E7D5B` is still never used as type (3.8:1, fails AA).

---

## Open items — deferred deliberately, not forgotten

Work that is known, agreed and **not** to be done as a standalone pass. Each
line names the batch it belongs to, so it ships with related work rather than
as an isolated change.

| Item | Batch it belongs to |
|---|---|
| **The session-limit CTA still reads "Book a portfolio walkthrough."** Two instances — `CopilotDemo.tsx` and `ReportPreview.tsx` — both pointing at `#book-a-demo`, shown only when a visitor exhausts the demo's question allowance. Pass 8 unified the nav and hero controls on "Demo on your portfolio" and left these alone; they are now the only place the retired label survives. Aligning them is a two-line change, but it lands inside the demo, so it should be verified by actually reaching the limit rather than by reading the diff. | The next pass that touches the query demo. |
| **`LandingControlsPoster` vertical rhythm.** Centring the play plate meant shifting the monitoring card up and the closing keyline down, which left roughly 170px of dead space below the keyline. This is the same void problem already fixed on the query poster (which takes its natural content height rather than a fixed ratio) and should be tightened for consistency. Contained entirely in `demo-video/src/landing/ControlsPoster.tsx` — the card scale and the keyline's `at` offset. | The Remotion re-render for the **Securitisation Readiness Agent** demo. Do not render the poster twice. |
| **The A2A poster's vertical rhythm**, the same defect as the controls poster above: the delegation poster's content sits in the upper two-thirds of a 1200×960 frame, leaving visible dead space beneath it once the play plate is centred. `demo-video/src/landing/A2APoster.tsx`. | The next Remotion render. Both posters together, once. |
| **The hero's third proof chip.** "Agent-addressable, with evidence behind every answer" was proposed and not signed off; the chip is unchanged. | The next hero pass. |
| **"Forecasting" survives in three places outside the matrix** — the hero sub-copy ("live portfolio monitoring, forecasting, controls and reporting"), the metadata description in `layout.tsx`, and the OpenGraph image's chip row. §3b forbids the generic word as a *capability item* because Trakt forecasts the pipeline, not the funded book; these three predate that rule and were left alone rather than rewritten inside a pass that had not asked for hero or metadata changes. The e2e guard is scoped to the matrix and the platform diagram accordingly. | The next hero or metadata pass. Rewriting the OG image means re-rendering it. |

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
| ~~Enterprise-agent or agent-to-agent availability~~ | **No longer excluded.** Withheld while the channels were fixture-only; claimed from Pass 10 on the strength of the recorded delegation run, and unlabelled from Pass 11 — see §6b tile 04 and §6c. |
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
