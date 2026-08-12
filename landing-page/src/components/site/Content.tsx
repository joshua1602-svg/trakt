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

/** Six peer outputs of one layer — chips, not cards. */
const OUTPUTS = [
  "Portfolio MI",
  "Forecasting",
  "Risk & covenant controls",
  "Investor reporting",
  "Regulatory reporting",
  "AI & Copilot",
] as const;

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
              <ul className="mt-3 flex flex-wrap gap-1.5">
                {OUTPUTS.map((output) => (
                  <li
                    key={output}
                    className="rounded-full border border-line bg-navy-850 px-2.5 py-1 text-[11px] font-medium text-ink-300"
                  >
                    {output}
                  </li>
                ))}
              </ul>
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
        <p className="mt-4 max-w-[72ch] text-[15px] font-medium leading-relaxed text-mint-400">
          Know what is breached today — and what the portfolio is moving toward.
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
/* Distribution — the channel strip in the intelligence section              */
/* -------------------------------------------------------------------------- */

/**
 * Three live surfaces reading one layer, each with a small glyph. The glyphs
 * are neutral strokes in the page's own icon style — the repository carries
 * no Microsoft brand assets, and imitation logos would be worse than none —
 * so the text labels carry the product names.
 */
const DELIVERY_SURFACES = [
  {
    name: "Trakt workspace",
    icon: "M3 5h18v14H3zM3 9h18M9 9v10",
  },
  {
    name: "Microsoft Teams",
    icon: "M9 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM16.5 9a2.5 2.5 0 1 0 0-5M2.5 20v-1.5A4.5 4.5 0 0 1 7 14h4a4.5 4.5 0 0 1 4.5 4.5V20M17 13.5h.5a4 4 0 0 1 4 4V19",
  },
  {
    name: "Microsoft 365 Copilot",
    icon: "M4 5h16v11H9l-5 4zM15.5 8l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z",
  },
] as const;

function ChannelIcon({ path }: { path: string }) {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className="shrink-0 text-peri-300"
    >
      <path d={path} />
    </svg>
  );
}

export function DeliveryStrip() {
  return (
    <div className="mt-6 flex flex-wrap items-center gap-x-2.5 gap-y-2 text-[12px] text-ink-200">
      <span className="text-[11px] font-semibold uppercase tracking-wider text-mint-400">
        Available today
      </span>
      {DELIVERY_SURFACES.map((surface) => (
        <span
          key={surface.name}
          className="inline-flex items-center gap-1.5 rounded-full border border-mint-400/30 bg-navy-850 px-3 py-1.5"
        >
          <ChannelIcon path={surface.icon} />
          {surface.name}
        </span>
      ))}
      {/* The section's single body line. Push, not pull — that is the claim,
          and "approved" is what keeps it governed. */}
      <span className="basis-full text-[12px] text-ink-300 sm:basis-auto">
        Approved risk findings are pushed to Teams.
      </span>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Delivery model — five static tiles                                         */
/* -------------------------------------------------------------------------- */

/**
 * Five delivery modes, stated once each. A static row: no expand, no
 * collapse, no keyboard handling — the whole section is legible in one pass,
 * which is the point. Mint marks what ships; grey marks what does not.
 */
const DELIVERY_MODES = [
  {
    name: "Managed service",
    // Longer than its neighbours on purpose: it is the only line that
    // separates a managed service from automated software.
    copy: "Reporting run by Trakt, not your team.",
    available: true,
  },
  {
    name: "Trakt Agent",
    copy: "Dashboards, charting and drill-through.",
    available: true,
  },
  {
    name: "Copilot",
    copy: "Portfolio questions inside Teams and Microsoft 365.",
    available: true,
  },
  {
    // One roadmap tile, not two: the agent-to-agent section below shows both
    // patterns — client-owned agents and cross-institution exchange — in a
    // single topology, so splitting them here made the distinction twice and
    // less well.
    name: "Agent access",
    copy: "Client and counterparty agents querying Trakt directly.",
    available: false,
  },
] as const;

export function DeliveryModes() {
  return (
    <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {DELIVERY_MODES.map((mode, index) => (
        <li key={mode.name}>
          <Reveal delay={index * 60} className="h-full">
            <Card
              className={cx(
                "flex h-full flex-col",
                mode.available ? "border-mint-400/25" : "border-line-soft bg-navy-900/40",
              )}
            >
              <p
                className={cx(
                  "text-[11px] font-semibold uppercase tracking-wider",
                  mode.available ? "text-mint-400" : "text-ink-500",
                )}
              >
                {mode.available ? "Available today" : "Roadmap"}
              </p>
              <h3
                className={cx(
                  "mt-2 text-[15px] font-semibold",
                  mode.available ? "text-ink-100" : "text-ink-300",
                )}
              >
                {mode.name}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-400">{mode.copy}</p>
            </Card>
          </Reveal>
        </li>
      ))}
    </ul>
  );
}

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
        <p className="mt-4 max-w-[72ch] text-[15px] font-medium leading-relaxed text-mint-400">
          Every figure is reconciled by construction rather than by comparison.
        </p>
      </Reveal>

      {/* Five properties, so five columns once there is room for them. A
          four-column grid left the fifth card alone on its own row with three
          empty columns beside it — which reads as a rendering fault, not a
          layout. Below xl the tail row is half-full rather than a lone card. */}
      <ul className="mt-9 grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
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
