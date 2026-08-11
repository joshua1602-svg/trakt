import { ControlsDemoLoop } from "@/components/site/ControlsDemoLoop";
import { Card, SectionHeading, cx } from "@/components/ui";
import type { DemoScopeInfo } from "@/types/demo";

/**
 * The static marketing sections: platform architecture, controls and forward
 * risk, governed onboarding, portfolio lenses, governance and the reporting
 * band.
 *
 * Every claim here is mapped to repository evidence in
 * `landing-page/docs/content-map.md`; wording that describes a capability the
 * repository does not yet support is not present, and anything that is a
 * deployment choice rather than a shipped capability says so. The two interface
 * previews (`ControlPreview`, `LensPreview`) are labelled where their figures
 * are illustrative rather than engine-produced.
 *
 * These are server components — nothing here is interactive.
 */

/* -------------------------------------------------------------------------- */
/* Platform — build the portfolio once, use it everywhere                     */
/* -------------------------------------------------------------------------- */

const FLOW = [
  {
    title: "Data and documents",
    copy: "Loan tapes, servicing extracts, valuations and facility documents, in whatever shape they arrive.",
  },
  {
    title: "One governed portfolio layer",
    copy: "Mapped to a canonical model, typed, enriched and validated — with lineage and evidence retained.",
  },
] as const;

/**
 * The outputs are chips, not cards: six peer outputs of one layer. Giving any
 * of them a card would quietly reintroduce the reporting-first identity this
 * page is moving away from.
 */
const OUTPUTS = [
  "Portfolio MI",
  "Forecasting",
  "Risk & covenant controls",
  "Investor reporting",
  "Regulatory reporting",
  "AI & Copilot interaction",
] as const;

export function Architecture() {
  return (
    <>
      <SectionHeading
        id="platform"
        eyebrow="The platform"
        title="Build the portfolio once. Use it everywhere."
        intro="Everything Trakt produces is generated from one governed portfolio layer, not maintained beside it — so management, investor, risk and regulatory views are lenses on the same truth, never separate versions of it."
      />

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
/* Controls and forward risk                                                  */
/* -------------------------------------------------------------------------- */

/**
 * The activation path stays visible in the preview: extraction is assisted,
 * activation is a human decision. The page never implies a control creates
 * itself.
 */
const CONTROL_PATH = [
  "Documented requirement",
  "Structured control",
  "Reviewed",
  "Active",
] as const;

export function ForwardControls() {
  return (
    <div className="grid items-center gap-10 lg:grid-cols-[1.05fr_1fr] lg:gap-14">
      <div>
        <SectionHeading
          id="controls"
          eyebrow="Risk & controls"
          title="Turn portfolio requirements into live controls."
          intro="Trakt structures concentration and covenant requirements from facility documentation into controls your team reviews and activates. Every active control is then evaluated three ways — against the funded book, against the expected forecast, and against the full pipeline — with the projected breach horizon when a limit is approaching."
        />
        <p className="mt-5 max-w-xl text-[15px] font-medium leading-relaxed text-mint-400">
          Know what is breached today — and what the portfolio is moving toward.
        </p>
        <ol className="mt-6 flex flex-wrap items-center gap-x-2 gap-y-1.5 text-[12px] text-ink-300">
          {CONTROL_PATH.map((step, index) => (
            <li key={step} className="flex items-center gap-2">
              {index > 0 ? (
                <span aria-hidden="true" className="text-peri-500">
                  →
                </span>
              ) : null}
              <span className="rounded-full border border-line bg-navy-850 px-2.5 py-1">
                {step}
              </span>
            </li>
          ))}
        </ol>
      </div>

      {/* The 18-second workflow loop; reduced-motion and video-failure
          visitors get the static ControlPreview instead. */}
      <ControlsDemoLoop />
    </div>
  );
}

/**
 * A static recreation of a concentration control as the product monitors it:
 * one limit, three evaluation bases, and the projected breach horizon.
 *
 * The figures are illustrative and labelled as such — unlike the hero preview,
 * they are not produced by the engine from the demo pack. Inside this product
 * depiction the product's own RAG semantics apply: mint marks a passing state,
 * amber a warning, rose a projected breach. See the token note in
 * `app/globals.css`.
 */
const CONTROL_STATES = [
  { label: "Funded book", value: 24.1, status: "Pass", tone: "mint" },
  { label: "Expected forecast", value: 28.7, status: "Warning", tone: "amber" },
  { label: "Including full pipeline", value: 31.4, status: "Projected breach", tone: "rose" },
] as const;

/** Bar scale ceiling: keeps the 30% limit marker inside the track. */
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

/** Exported: the reduced-motion and no-video fallback for the demo loop. */
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

        <div className="mt-3 flex items-baseline justify-between gap-3 rounded-lg border border-line-soft bg-navy-950/70 px-3 py-2">
          <p className="text-[11px] text-ink-300">Single obligor — largest exposure</p>
          <p className="shrink-0 text-[12px] font-semibold tabular-nums text-mint-400">
            6.2% · Pass
          </p>
        </div>

        <p className="mt-3 flex flex-wrap gap-x-2 border-t border-line-soft pt-2 text-[10px] text-ink-500">
          <span>Evaluated against funded book, forecast and pipeline</span>
        </p>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Governed onboarding                                                        */
/* -------------------------------------------------------------------------- */

/** A genuine sequence, so the numbers stay. */
const ONBOARDING_STEPS = [
  {
    title: "Source data and documents",
    copy: "Loan tapes, servicing extracts and documentation, in whatever shape they arrive.",
  },
  {
    title: "Assisted interpretation",
    copy: "The onboarding agent proposes mappings and configuration — deterministic first, with every suggestion queued for review.",
  },
  {
    title: "Governed configuration",
    copy: "Your team reviews and approves before anything is activated. Nothing reaches the governed layer unapproved.",
  },
  {
    title: "Live portfolio",
    copy: "The book joins the governed layer — monitored, reportable and answerable alongside every other portfolio.",
  },
] as const;

export function Onboarding() {
  return (
    <>
      <SectionHeading
        id="onboarding"
        eyebrow="Implementation"
        title="From source files to a live portfolio — under governance."
        intro="Trakt's onboarding agent interprets source tapes and documentation and proposes mappings and configuration; your team reviews before anything activates — a repeatable process as additional portfolios are added."
      />
      <ol className="mt-9 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {ONBOARDING_STEPS.map((step, index) => (
          <Card as="li" key={step.title} className="flex flex-col">
            <span
              aria-hidden="true"
              className="mb-3 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-line bg-navy-850 text-xs font-semibold tabular-nums text-peri-300"
            >
              {index + 1}
            </span>
            <h3 className="text-[15px] font-semibold text-ink-100">{step.title}</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-ink-400">{step.copy}</p>
          </Card>
        ))}
      </ol>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* Portfolio lenses                                                           */
/* -------------------------------------------------------------------------- */

export function Lenses({ scope }: { scope: DemoScopeInfo }) {
  return (
    <div className="grid items-center gap-10 lg:grid-cols-[1.05fr_1fr] lg:gap-14">
      <div>
        <SectionHeading
          id="lenses"
          eyebrow="Operating model"
          title="One portfolio truth. Every relevant lens."
          intro="Any number of books — direct originations, acquired back books, sponsored securitisations — held in one governed model, each reportable on its own and in aggregate."
        />
        <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-ink-300">
          Individual books, acquired portfolios, funding vehicles and the consolidated
          platform are governed views over the same underlying data, with capability
          and coverage disclosed per scope. Every stakeholder sees the lens relevant
          to them — without a separate dataset to reconcile.
        </p>
      </div>

      <LensPreview scope={scope} />
    </div>
  );
}

/**
 * A static recreation of the portfolio scope selector, using this platform's
 * real figures from the demo pack — the same governed books the hero preview
 * and the interactive example answer from, so the page carries one platform in
 * one vocabulary.
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
        <span>Coverage disclosed per lens</span>
        <span>· As at {scope.asOfDisplay}</span>
        <span>· Synthetic portfolio</span>
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Governance and platform                                                    */
/* -------------------------------------------------------------------------- */

const GOVERNANCE_PROPERTIES = [
  {
    title: "Deterministic calculation",
    copy: "Same question, same number, every channel — one analytical implementation behind the workspace, Copilot and reports, parity-tested.",
  },
  {
    title: "Reviewed configuration",
    copy: "Mappings, controls and configuration are approved by people before activation, and changes are governed.",
  },
  {
    title: "Traceable outputs",
    copy: "Lineage from source header to published figure, with validation evidence and reproducible runs.",
  },
  {
    title: "Client separation",
    copy: "Client environments are isolated behind Microsoft Entra ID, with tenant authorisation enforced in the platform core — controlled separation between organisations on a common platform.",
  },
] as const;

export function Governance() {
  return (
    <>
      <SectionHeading
        id="governance"
        eyebrow="Governance"
        title="Deterministic underneath. Governed throughout."
        intro="AI in Trakt interprets, navigates and accelerates — it never writes your numbers. Controlled data, deterministic calculations and human approvals sit underneath every answer the platform gives."
      />

      <ul className="mt-9 grid gap-4 sm:grid-cols-2">
        {GOVERNANCE_PROPERTIES.map((property) => (
          <Card as="li" key={property.title} className="flex flex-col">
            <h3 className="text-[15px] font-semibold text-ink-100">{property.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink-400">{property.copy}</p>
          </Card>
        ))}
      </ul>

      <p className="mt-8 max-w-3xl text-[15px] leading-relaxed text-ink-300">
        Built for specialist lending portfolios on a common canonical model with
        asset-specific configuration — designed so new lending asset classes are added
        through configuration and verified through the same pipeline, not by
        rebuilding the platform.
      </p>

      {/* Quiet direction, not a roadmap block: a future posture stated as
          design intent, never as live capability. */}
      <p className="mt-4 max-w-3xl text-sm leading-relaxed text-ink-500">
        Designed to extend from user-directed workflows toward increasingly agentic
        operation, within the same governed control framework.
      </p>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* Reporting band                                                             */
/* -------------------------------------------------------------------------- */

/**
 * Deliberately a band, not a marquee: reporting is an output of the governed
 * layer, no longer the page's identity. Three former capability tiles live in
 * these four chips.
 */
const REPORTING_OUTPUTS = [
  "Management reporting",
  "Investor & funding-partner packs",
  "Regulatory submissions",
  "Bespoke analysis",
] as const;

export function ReportingBand() {
  return (
    <Card>
      <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr] lg:items-center">
        <div>
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.16em] text-peri-400">
            Reporting
          </p>
          <h2
            id="reporting-heading"
            className="text-balance text-xl font-semibold tracking-tight text-ink-100 sm:text-2xl"
          >
            The portfolio truth that runs the business also reports it.
          </h2>
          <p className="mt-3 text-[15px] leading-relaxed text-ink-300">
            Recurring packs and submissions are generated from the same governed
            layer — field-validated, submission-ready and traceable to source.
          </p>
        </div>
        <ul className="flex flex-wrap gap-2">
          {REPORTING_OUTPUTS.map((output) => (
            <li
              key={output}
              className="rounded-full border border-line bg-navy-850 px-3 py-1.5 text-[12px] font-medium text-ink-200"
            >
              {output}
            </li>
          ))}
        </ul>
      </div>
    </Card>
  );
}

/* -------------------------------------------------------------------------- */
/* Delivery strip — folded into the intelligence section                      */
/* -------------------------------------------------------------------------- */

/**
 * The former delivery-model section, reduced to its substance: three live
 * surfaces reading one governed layer, each with a small glyph so the
 * channels read at a glance. The glyphs are neutral strokes in the page's
 * own icon style — the repository carries no Microsoft brand assets, and
 * imitation logos would be worse than none — so the text labels carry the
 * product names. Roadmap channels live in the governance section.
 */
const DELIVERY_SURFACES = [
  {
    name: "Trakt workspace",
    // A window with panels: the analytical workspace.
    icon: "M3 5h18v14H3zM3 9h18M9 9v10",
  },
  {
    name: "Microsoft Teams",
    // Two people: the collaboration surface.
    icon: "M9 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM16.5 9a2.5 2.5 0 1 0 0-5M2.5 20v-1.5A4.5 4.5 0 0 1 7 14h4a4.5 4.5 0 0 1 4.5 4.5V20M17 13.5h.5a4 4 0 0 1 4 4V19",
  },
  {
    name: "Microsoft 365 Copilot",
    // A chat bubble with a spark: the assistant surface.
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
