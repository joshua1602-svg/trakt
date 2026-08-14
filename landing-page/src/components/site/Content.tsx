import { DemoPlayer } from "@/components/site/DemoPlayer";
import { Reveal } from "@/components/site/Reveal";
import { Card, SectionHeading, cx } from "@/components/ui";

/**
 * The static marketing sections: platform architecture, risk & controls
 * (hosting the controls demo), portfolio lenses, distribution, governance
 * and the compact onboarding disclosure.
 *
 * Every claim here is mapped to repository evidence in
 * `landing-page/docs/content-map.md`; wording that describes a capability the
 * repository does not yet support is not present. Copy is deliberately terse —
 * each concept is explained once on the page, and the product visuals carry
 * the proof.
 */

/* -------------------------------------------------------------------------- */
/* Platform — build the portfolio once, use it everywhere                     */
/* -------------------------------------------------------------------------- */

/**
 * Step 2's body is the distinctive claim, not the duplicate one. "Mapped,
 * validated and traceable" restated the Governance cards and was cut; this
 * line came from the deleted Operating Model section, which made the same
 * claim 400px further down. The diagram is where the layer is described, so
 * this is where the claim belongs.
 */
const FLOW = [
  {
    title: "Data and documents",
    copy: "Loan tapes, servicing data, valuations, facility documents.",
  },
  {
    title: "One governed portfolio layer",
    copy: "No separate datasets to reconcile.",
  },
] as const;

/**
 * Step 3's six chips are gone: *Portfolio MI · Forecasting · Risk & covenant
 * controls · Investor reporting · Regulatory reporting · AI & Copilot*. They
 * were an ungrouped, weaker version of the capability matrix that now sits
 * directly beneath this section — and one of them, an unqualified
 * "Forecasting", claimed something the page must not: Trakt forecasts the
 * pipeline, not the funded book.
 *
 * The box keeps its place rather than becoming a bare pointer. The arrow motif
 * between the boxes (`→` at lg, `↓` below) is drawn for three; two boxes and
 * one arrow reads as an unfinished diagram. What it carries instead is a bare
 * enumeration of the five groups — no verb, no claim, just the index — so the
 * diagram names the shape and the matrix below expands it. Naming the five
 * groups as chips was the obvious alternative and was rejected: it would put
 * the identical five words twice within 200px, which is the duplication this
 * page has removed three times already.
 */

export function Architecture() {
  return (
    <>
      <Reveal>
        <SectionHeading
          id="platform"
          eyebrow="The platform"
          title="Build the portfolio once. Use it everywhere."
        />
      </Reveal>

      {/* The diagram carries the explanation — no paragraph above or below. */}
      <ol className="mt-9 grid gap-4 lg:grid-cols-3">
        {FLOW.map((step, index) => (
          <li key={step.title} className="relative">
            <Reveal delay={index * 60} className="h-full">
              <Card className="h-full">
                <h3 className="text-[15px] font-semibold text-ink-100">{step.title}</h3>
                {step.copy ? (
                  <p className="mt-2 text-sm leading-relaxed text-ink-400">{step.copy}</p>
                ) : null}
              </Card>
            </Reveal>
            <span
              aria-hidden="true"
              className="absolute -bottom-3 left-1/2 -translate-x-1/2 text-peri-500 lg:-right-3 lg:bottom-auto lg:left-auto lg:top-1/2 lg:-translate-y-1/2 lg:translate-x-0"
            >
              <span className="lg:hidden">↓</span>
              <span className="hidden lg:inline">→</span>
            </span>
          </li>
        ))}
        <li>
          <Reveal delay={120} className="h-full">
            <Card className="h-full">
              <h3 className="text-[15px] font-semibold text-ink-100">Every output</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-400">
                Portfolio, control, reporting, investigation and delivery.
              </p>
            </Card>
          </Reveal>
        </li>
      </ol>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* Risk and controls — hosts Demo 2                                           */
/* -------------------------------------------------------------------------- */

/**
 * Structurally identical to the query-demo section: eyebrow, headline, one
 * line, then the demo at full container width. Two demos presented in two
 * different shapes reads as unfinished, so the shape is shared.
 *
 * The demo's own eyebrow and heading are gone — "See a portfolio requirement
 * become a live control" restated the section headline directly above it.
 */
export function ForwardControls() {
  return (
    <>
      <Reveal>
        <SectionHeading
          id="controls"
          eyebrow="Risk & controls"
          title="Turn portfolio requirements into live controls."
        />
        <p className="mt-4 max-w-[72ch] text-lg font-medium leading-relaxed text-ink-100">
          Know what is breached today — and what the portfolio is moving toward.
        </p>
        {/* Carried here from the deleted delivery section. It is the one
            behaviour on the page that is not user-initiated — every other
            surface describes somewhere a person goes to ask, and this is the
            case where Trakt arrives unasked — and it belongs beside the demo
            that produces the finding rather than in a list of channels. The
            claim is the Teams outbox (`trakt_notifications/`), not Copilot.
            Asserted to appear exactly once on the page. */}
        <p className="mt-3 max-w-[72ch] text-sm leading-relaxed text-ink-400">
          Approved risk findings are pushed to Teams.
        </p>
      </Reveal>

      <Reveal delay={60}>
        <div className="mt-8">
          <DemoPlayer
            overlayLabel="Watch controls demo"
            durationLabel="~18 sec"
            poster="/controls-demo-poster.png"
            webmSrc="/controls-demo.webm"
            mp4Src="/controls-demo.mp4"
            description="Demonstration: clauses in a portfolio covenant schedule extract are identified and structured into proposed controls, reviewed and activated by a person, then monitored against the funded book, the expected forecast and the full pipeline, ending on a projected breach horizon."
            caption="From documented requirement to live monitoring."
            fallback={<ControlPreview />}
            plateId="controls"
          />
        </div>
      </Reveal>
    </>
  );
}

/**
 * The static control card: the demo's end state in real DOM text. Renders
 * when the video cannot (its only remaining role).
 *
 * The figures are illustrative and labelled as such. Inside this product
 * depiction the product's RAG semantics apply: mint passing, amber warning,
 * rose projected breach. See the token note in `app/globals.css`.
 */
const CONTROL_STATES = [
  { label: "Funded book", value: 24.1, status: "Pass", tone: "mint" },
  { label: "Expected forecast", value: 28.7, status: "Warning", tone: "amber" },
  { label: "Including full pipeline", value: 31.4, status: "Projected breach", tone: "rose" },
] as const;

const CONTROL_SCALE = 36;
const CONTROL_LIMIT = 30;

const TONE_TEXT = {
  mint: "text-mint-400",
  amber: "text-amber-400",
  rose: "text-rose-400",
} as const;

const TONE_FILL = {
  mint: "bg-mint-400/70",
  amber: "bg-amber-400/70",
  rose: "bg-rose-400/70",
} as const;

export function ControlPreview() {
  return (
    <div
      // A system-state context: the three evaluation rows are pass / warning /
      // projected breach, which is exactly what mint, amber and rose mean.
      // The e2e colour guard reads this attribute rather than a hard-coded
      // selector list.
      data-state-colour="control-preview"
      className="rounded-2xl border border-line bg-navy-900/80 p-4 shadow-[0_24px_60px_-30px_rgba(0,0,0,0.9)] sm:p-5"
      role="img"
      aria-label="Preview of a concentration control evaluated against the funded book, the expected forecast and the full pipeline"
    >
      <p className="border-b border-line pb-3 text-xs font-medium text-ink-300">
        Concentration controls
      </p>

      <div className="pt-4">
        <div className="flex items-baseline justify-between gap-3">
          <h3 className="text-[13px] font-semibold text-ink-100">
            Geographic concentration — any single region
          </h3>
          <p className="shrink-0 text-[11px] text-ink-400">
            Limit ≤ {CONTROL_LIMIT}% of balance
          </p>
        </div>

        <dl className="mt-3 space-y-2.5">
          {CONTROL_STATES.map((state) => (
            <div key={state.label}>
              <div className="flex items-baseline justify-between gap-3">
                <dt className="text-[11px] text-ink-300">{state.label}</dt>
                <dd
                  className={cx(
                    "shrink-0 text-[12px] font-semibold tabular-nums",
                    TONE_TEXT[state.tone],
                  )}
                >
                  {state.value}% · {state.status}
                </dd>
              </div>
              <div className="relative mt-1 h-1.5 overflow-hidden rounded-full bg-navy-850">
                <div
                  className={cx("animate-bar h-full rounded-full", TONE_FILL[state.tone])}
                  style={{ width: `${(state.value / CONTROL_SCALE) * 100}%` }}
                />
                <div
                  aria-hidden="true"
                  className="absolute inset-y-0 w-px bg-ink-400"
                  style={{ left: `${(CONTROL_LIMIT / CONTROL_SCALE) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </dl>

        <p className="mt-4 border-t border-line-soft pt-2.5 text-[11px] text-ink-300">
          Projected breach horizon: <span className="font-semibold text-ink-100">Nov 2026</span>
        </p>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Delivery model — deleted in Pass 12                                        */
/* -------------------------------------------------------------------------- */

/**
 * `DeliveryModes` and `DELIVERY_MODES` are gone, and with them the whole
 * `#delivery` section. Its four tiles said what the capability matrix's
 * DELIVERY column now says in five items — Managed service · Trakt · Microsoft
 * Copilot · Microsoft Teams · Enterprise agent A2A — at a section's cost, two
 * sections above the matrix that repeated them.
 *
 * Three things it carried, and where each went:
 *
 *   • "Approved risk findings are pushed to Teams." — to Risk & Controls
 *     above. It is a claim about what happens to a breach, not about a
 *     delivery channel, and it now sits beside the demo that produces the
 *     finding. Still asserted to appear exactly once on the page.
 *   • The headline "From a managed service to your own agents." — NOT
 *     rehomed. The ladder it named is carried by the DELIVERY items
 *     themselves, managed service at one end and A2A at the other. A
 *     positioning headline with no section under it is furniture.
 *   • The 01–04 numerals — gone with the headline whose argument they
 *     illustrated.
 *
 * "Managed service" is deliberately FIRST in the matrix column. Without it the
 * page would lose its primary commercial delivery mode entirely.
 */
/* -------------------------------------------------------------------------- */
/* Governance                                                                 */
/* -------------------------------------------------------------------------- */

/** Four claims, one line each — governance shown, not re-explained. */
const GOVERNANCE_PROPERTIES = [
  { title: "Deterministic", copy: "Same calculation, every channel." },
  { title: "Traceable", copy: "Every published figure ties back to source." },
  { title: "Controlled", copy: "Configuration is reviewed before activation." },
  {
    title: "Isolated",
    copy: "Client environments and authorisation are separated behind Microsoft Entra ID.",
  },
  {
    title: "Agent-addressable",
    copy: "Agents authenticate as themselves, inside tenant and portfolio boundaries, with explicit permissions.",
  },
] as const;

export function Governance() {
  return (
    <>
      <Reveal>
        <SectionHeading
          id="governance"
          eyebrow="Governance"
          title="Deterministic underneath. Governed throughout."
        />
        <p className="mt-4 max-w-[72ch] text-lg font-medium leading-relaxed text-ink-100">
          Every figure is reconciled by construction rather than by comparison.
        </p>
      </Reveal>

      {/* Five properties, and five is an awkward number in a grid. Five
          columns at xl fits them on one row. Three columns at lg leaves a
          half-full tail row, which reads as a wrap. Two columns left the fifth
          card alone at half width with an empty cell beside it — the same
          rendering-fault look the five-column row was introduced to remove —
          so below lg the cards stack in a single column instead. */}
      <ul className="mt-9 grid gap-4 lg:grid-cols-3 xl:grid-cols-5">
        {GOVERNANCE_PROPERTIES.map((property, index) => (
          <li key={property.title}>
            <Reveal delay={index * 60} className="h-full">
              <Card className="flex h-full flex-col">
                <h3 className="text-[15px] font-semibold text-ink-100">{property.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-400">{property.copy}</p>
              </Card>
            </Reveal>
          </li>
        ))}
      </ul>

      {/* One line, answering the extensibility objection. The agentic
          direction is cut: the Delivery Model tiles show it, in grey. */}
      <Reveal delay={120}>
        <p className="mt-8 max-w-[72ch] text-sm leading-relaxed text-ink-500">
          New asset classes are added through configuration, not a rebuild.
        </p>
      </Reveal>
    </>
  );
}
