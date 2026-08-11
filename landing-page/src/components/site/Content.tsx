import { DemoPlayer } from "@/components/site/DemoPlayer";
import { Card, SectionHeading, cx } from "@/components/ui";
import type { DemoScopeInfo } from "@/types/demo";

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

const FLOW = [
  {
    title: "Data and documents",
    copy: "Loan tapes, servicing data, valuations and facility documents.",
  },
  {
    title: "One governed portfolio layer",
    copy: "Mapped, validated and traceable.",
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
      <SectionHeading
        id="platform"
        eyebrow="The platform"
        title="Build the portfolio once. Use it everywhere."
      />

      {/* The diagram carries the explanation — no paragraph above or below. */}
      <ol className="mt-9 grid gap-4 lg:grid-cols-3">
        {FLOW.map((step, index) => (
          <li key={step.title} className="relative">
            <Card className="h-full">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-peri-400">
                Step {index + 1}
              </p>
              <h3 className="mt-2 text-[15px] font-semibold text-ink-100">{step.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-400">{step.copy}</p>
            </Card>
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
          <Card className="h-full">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-peri-400">
              Step 3
            </p>
            <h3 className="mt-2 text-[15px] font-semibold text-ink-100">Every output</h3>
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
        </li>
      </ol>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* Risk and controls — hosts Demo 2                                           */
/* -------------------------------------------------------------------------- */

export function ForwardControls() {
  return (
    <div className="grid items-center gap-10 lg:grid-cols-[1fr_1.05fr] lg:gap-14">
      <div>
        <SectionHeading
          id="controls"
          eyebrow="Risk & controls"
          title="Turn portfolio requirements into live controls."
          intro="Structure covenant and concentration requirements once. Trakt monitors them against the funded book, forecast and pipeline — showing today's position and emerging breaches."
        />
        <p className="mt-5 max-w-xl text-[15px] font-medium leading-relaxed text-mint-400">
          Know what is breached today — and what the portfolio is moving toward.
        </p>
      </div>

      <div>
        <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-peri-400">
          Risk &amp; controls demo
        </p>
        <h3 className="mb-4 text-lg font-semibold tracking-tight text-ink-100">
          See a portfolio requirement become a live control.
        </h3>
        <DemoPlayer
          overlayLabel="Watch controls demo"
          durationLabel="~18 sec"
          poster="/controls-demo-poster.png"
          webmSrc="/controls-demo.webm"
          mp4Src="/controls-demo.mp4"
          description="Demonstration: clauses in a portfolio covenant schedule extract are identified and structured into proposed controls, reviewed and activated by a person, then monitored against the funded book, the expected forecast and the full pipeline, ending on a projected breach horizon."
          caption="From documented requirement to live monitoring. Figures illustrative."
          fallback={<ControlPreview />}
        />
      </div>
    </div>
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
      <p className="flex items-center justify-between border-b border-line pb-3 text-xs font-medium text-ink-300">
        Concentration controls
        <span className="rounded-full border border-amber-400/35 bg-amber-400/10 px-2 py-0.5 text-[10px] font-medium text-amber-400">
          Illustrative
        </span>
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
                  className={cx("h-full rounded-full", TONE_FILL[state.tone])}
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
/* Portfolio operating model                                                  */
/* -------------------------------------------------------------------------- */

export function Lenses({ scope }: { scope: DemoScopeInfo }) {
  return (
    <div className="grid items-center gap-10 lg:grid-cols-[1.05fr_1fr] lg:gap-14">
      <SectionHeading
        id="lenses"
        eyebrow="Operating model"
        title="One portfolio truth. Every relevant lens."
        intro="Origination books, acquired portfolios and funding vehicles sit in one governed model — individually reportable and consolidated, without separate datasets to reconcile."
      />

      <LensPreview scope={scope} />
    </div>
  );
}

/**
 * A static recreation of the portfolio scope selector, using this platform's
 * real figures from the demo pack — the same governed books the query demo
 * answers from.
 */
function LensPreview({ scope }: { scope: DemoScopeInfo }) {
  return (
    <div
      className="rounded-2xl border border-line bg-navy-900/80 p-4 shadow-[0_24px_60px_-30px_rgba(0,0,0,0.9)] sm:p-5"
      role="img"
      aria-label="Preview of the portfolio scope selector: consolidated view and individual governed books"
    >
      <p className="border-b border-line pb-3 text-xs font-medium text-ink-300">
        Portfolio scope
      </p>

      <ul className="space-y-1.5 pt-4">
        <li className="flex items-baseline justify-between gap-3 rounded-lg border border-peri-500/60 bg-navy-850 px-3 py-2">
          <p className="text-[12px] font-semibold text-ink-100">Consolidated platform</p>
          <p className="shrink-0 text-[13px] font-semibold tabular-nums text-ink-100">
            {scope.totalBalanceDisplay}
          </p>
        </li>
        {scope.books.map((book) => (
          <li
            key={book.id}
            className="flex items-baseline justify-between gap-3 rounded-lg border border-line-soft bg-navy-900/60 px-3 py-2"
          >
            <p className="min-w-0 text-[12px] text-ink-300">
              {book.label}
              {book.balanceSheetStatus === "sold" ? (
                <span className="ml-1.5 text-[10px] text-amber-400">sold</span>
              ) : null}
            </p>
            <p className="shrink-0 text-[13px] font-semibold tabular-nums text-ink-200">
              {book.balanceDisplay}
            </p>
          </li>
        ))}
      </ul>

      <p className="mt-3 flex flex-wrap gap-x-2 border-t border-line-soft pt-2 text-[10px] text-ink-500">
        <span>As at {scope.asOfDisplay}</span>
        <span>· Synthetic portfolio</span>
      </p>
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
      <span className="basis-full text-[12px] text-ink-400 sm:basis-auto">
        Approved risk findings can also be delivered proactively into Teams.
      </span>
    </div>
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
] as const;

export function Governance() {
  return (
    <>
      <SectionHeading
        id="governance"
        eyebrow="Governance"
        title="Deterministic underneath. Governed throughout."
      />
      <p className="mt-4 max-w-xl text-[15px] font-medium leading-relaxed text-mint-400">
        Every figure is reconciled by construction rather than by comparison.
      </p>

      <ul className="mt-9 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {GOVERNANCE_PROPERTIES.map((property) => (
          <Card as="li" key={property.title} className="flex flex-col">
            <h3 className="text-[15px] font-semibold text-ink-100">{property.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink-400">{property.copy}</p>
          </Card>
        ))}
      </ul>

      <p className="mt-8 max-w-3xl text-sm leading-relaxed text-ink-500">
        Built on a common canonical model with asset-specific configuration — new
        lending asset classes are added through configuration, not by rebuilding the
        platform. Designed to extend from user-directed workflows toward increasingly
        agentic operation, within the same governed control framework.
      </p>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* Onboarding — a compact disclosure, out of the main narrative               */
/* -------------------------------------------------------------------------- */

const ONBOARDING_STEPS = [
  { title: "Source data", copy: "Tapes and documents, as they arrive." },
  { title: "Assisted mapping", copy: "Mappings and configuration proposed." },
  { title: "Team review", copy: "Approved before anything activates." },
  { title: "Live portfolio", copy: "Monitored and reportable with every other book." },
] as const;

export function OnboardingDisclosure() {
  return (
    <details className="group rounded-2xl border border-line bg-navy-900/60 px-5 py-4 sm:px-6">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-[15px] font-semibold text-ink-100 [&::-webkit-details-marker]:hidden">
        How onboarding works
        <span
          aria-hidden="true"
          className="text-peri-400 transition-transform group-open:rotate-90"
        >
          ›
        </span>
      </summary>
      <ol className="mt-5 grid gap-4 border-t border-line-soft pt-5 sm:grid-cols-2 lg:grid-cols-4">
        {ONBOARDING_STEPS.map((step, index) => (
          <li key={step.title} className="flex gap-3">
            <span
              aria-hidden="true"
              className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-line bg-navy-850 text-[11px] font-semibold tabular-nums text-peri-300"
            >
              {index + 1}
            </span>
            <div>
              <h3 className="text-sm font-semibold text-ink-100">{step.title}</h3>
              <p className="mt-1 text-[13px] leading-relaxed text-ink-400">{step.copy}</p>
            </div>
          </li>
        ))}
      </ol>
    </details>
  );
}
