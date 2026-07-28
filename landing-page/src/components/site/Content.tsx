import { Card, SectionHeading, cx } from "@/components/ui";

/**
 * The static marketing sections: capabilities, delivery model, connected
 * operating model and portfolio lifecycle.
 *
 * Every claim here is mapped to repository evidence in
 * `landing-page/docs/content-map.md`; wording that describes a capability the
 * repository does not yet support is not present, and anything that is a
 * deployment choice rather than a shipped capability says so.
 *
 * These are server components — nothing here is interactive.
 */

/* -------------------------------------------------------------------------- */
/* Capabilities                                                               */
/* -------------------------------------------------------------------------- */

interface Capability {
  id: string;
  name: string;
  copy: string;
  icon: string;
}

/**
 * The ingestion claim is not one capability among several — it is the reason the
 * others are possible — so it sits above the grid rather than inside it.
 *
 * It previously led the grid as a green-bordered tile. That fails on a phone:
 * only one card is visible at a time, so comparative emphasis reads as an
 * inconsistency rather than a hierarchy. Above the heading it cannot be scrolled
 * past, and every tile below can then be identical.
 */
const INGESTION_CLAIM =
  "Any number of books — direct originations, acquired back books, sponsored " +
  "securitisations — held in one governed model, each reportable on its own and " +
  "in aggregate.";

const CAPABILITIES: Capability[] = [
  {
    id: "analytics",
    name: "Portfolio analytics and monitoring",
    icon: "M4 19V5M4 19h16M8 15v-4M13 15V8M18 15v-6",
    copy: "Dashboards, cohorts, concentrations, drill-through and limit monitoring, funded book and pipeline.",
  },
  {
    id: "management-reporting",
    name: "Management reporting",
    icon: "M6 3h9l4 4v14H6zM15 3v4h4",
    copy: "Recurring management information, and one consistent answer across finance, risk and operations.",
  },
  {
    id: "investor-reporting",
    name: "Investor reporting",
    icon: "M4 18l5-6 4 3 6-8M20 7h-4M20 7v4",
    copy: "Investor and funding-partner packs, with stratifications and commentary.",
  },
  {
    id: "regulatory",
    name: "Regulatory reporting",
    icon: "M12 3l8 4v5c0 5-3.4 8.3-8 9-4.6-.7-8-4-8-9V7z",
    copy: "ESMA Annex 2 and Annex 12 output, field-validated and built into submission-ready XML.",
  },
  {
    id: "governance",
    name: "Governance and audit",
    icon: "M5 5h14v14H5zM9 9h6M9 13h6M9 17h3",
    copy: "Lineage from source header to published figure, with validation evidence and reproducible runs.",
  },
];

function CapabilityIcon({ path }: { path: string }) {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d={path} />
    </svg>
  );
}

export function CapabilityStack() {
  return (
    <>
      <p className="max-w-3xl text-lg font-medium leading-relaxed text-ink-200">
        {INGESTION_CLAIM}
      </p>

      <div className="mt-9">
        <SectionHeading id="capabilities" eyebrow="The platform" title="Capabilities on one governed layer" />
      </div>

      {/* Every tile identical: same border, same icon treatment, no accent. */}
      <ul className="mt-9 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {CAPABILITIES.map((capability) => (
          <Card
            as="li"
            key={capability.id}
            className="flex flex-col transition-colors hover:border-peri-500/50"
          >
            <span className="mb-3.5 flex h-9 w-9 items-center justify-center rounded-lg border border-line bg-navy-850 text-peri-300">
              <CapabilityIcon path={capability.icon} />
            </span>
            <h3 className="text-[15px] font-semibold text-ink-100">{capability.name}</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink-400">{capability.copy}</p>
          </Card>
        ))}
      </ul>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* Delivery model                                                             */
/* -------------------------------------------------------------------------- */

/**
 * Two groups, never flattened.
 *
 * What ships today and what is a deployment choice are different claims, and
 * the page's credibility rests on the distinction. Every live mode below is
 * evidenced in the repository; both roadmap items are reserved channels
 * (`trakt_core/context.py`) with a documented adapter shape and no shipped
 * route, so they are labelled as roadmap and carry no accent.
 */
const LIVE_MODES = [
  {
    name: "Managed service",
    copy: "Recurring reporting, regulatory output and governance artefacts, produced with no user interaction.",
  },
  {
    name: "Trakt Agent workspace",
    copy: "The full analytical environment: dashboards, charting, drill-through and portfolio investigation.",
  },
  {
    name: "Microsoft 365 Copilot and Teams",
    copy: "Portfolio questions and artefact requests inside the tools your teams already use.",
  },
] as const;

const ROADMAP_MODES = [
  {
    name: "Enterprise agent deployment",
    copy: "Trakt running inside a client's own agent estate.",
  },
  {
    name: "Agent-to-agent integration",
    copy: "Upstream and downstream systems consulting the governed layer directly.",
  },
] as const;

export function DeliveryModel() {
  return (
    <>
      <SectionHeading
        id="delivery"
        eyebrow="Delivery model"
        title="Every mode reads the same governed layer"
      />

      <div className="mt-9">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-mint-400">
          Available today
        </p>
        <ul className="mt-3 grid gap-4 lg:grid-cols-3">
          {LIVE_MODES.map((mode) => (
            <Card as="li" key={mode.name} className="flex flex-col border-mint-400/40">
              <h3 className="text-[15px] font-semibold text-mint-400">{mode.name}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-400">{mode.copy}</p>
            </Card>
          ))}
        </ul>
      </div>

      <div className="mt-8">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-500">
          Roadmap
        </p>
        <ul className="mt-3 grid gap-4 lg:grid-cols-2">
          {ROADMAP_MODES.map((mode) => (
            <Card
              as="li"
              key={mode.name}
              className="flex flex-col border-line-soft bg-navy-900/40"
            >
              <h3 className="text-[15px] font-semibold text-ink-300">{mode.name}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-500">{mode.copy}</p>
            </Card>
          ))}
        </ul>
      </div>

      <p className="mt-8 text-[15px] leading-relaxed text-ink-300">
        The answer is calculated once and distributed; the channel never changes it.
        Deployments are isolated per client behind Microsoft Entra ID, with retention
        and document delivery set at onboarding.
      </p>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* How it works                                                               */
/* -------------------------------------------------------------------------- */

const FLOW = [
  {
    title: "Source data and documents",
    copy: "Loan tapes, servicing extracts, valuations and facility documents, in whatever shape they arrive.",
  },
  {
    title: "Trakt governed data layer",
    copy: "Mapped to a canonical model, typed, enriched and validated — with the evidence retained.",
  },
  {
    title: "Analytics and business rules",
    copy: "Deterministic metrics, stratifications, cohorts, concentration measures and regulatory rules.",
  },
] as const;

export function OperatingModel() {
  return (
    <>
      <SectionHeading
        id="how-it-works"
        eyebrow="How it works"
        title="One calculation, governed centrally"
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
            {index < FLOW.length - 1 ? (
              <span
                aria-hidden="true"
                className="absolute -bottom-3 left-1/2 -translate-x-1/2 text-peri-500 lg:-right-3 lg:bottom-auto lg:left-auto lg:top-1/2 lg:-translate-y-1/2 lg:translate-x-0"
              >
                <span className="lg:hidden">↓</span>
                <span className="hidden lg:inline">→</span>
              </span>
            ) : null}
          </li>
        ))}
      </ol>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* Where it applies                                                           */
/* -------------------------------------------------------------------------- */

/** A genuine sequence, so the numbers stay. */
const LIFECYCLE = [
  {
    title: "Onboard",
    copy: "Launch or acquire a book, standardised into the governed model without building another reporting stack.",
  },
  {
    title: "Run",
    copy: "Monitor performance, answer management questions and automate recurring information.",
  },
  {
    title: "Finance",
    copy: "Warehouse, investor and lender reporting, through to securitisation and refinancing, from controlled portfolio data.",
  },
  {
    title: "Exit",
    copy: "Prepare validated datasets, stratifications and portfolio reporting for sell-side mandates.",
  },
] as const;

export function Lifecycle() {
  return (
    <>
      <SectionHeading
        id="lifecycle"
        eyebrow="Where it applies"
        title="Built for the portfolio lifecycle"
      />
      <ol className="mt-9 grid gap-4 sm:grid-cols-2">
        {LIFECYCLE.map((stage, index) => (
          <Card as="li" key={stage.title} className={cx("flex gap-4")}>
            <span
              aria-hidden="true"
              className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-line bg-navy-850 text-xs font-semibold tabular-nums text-peri-300"
            >
              {index + 1}
            </span>
            <div>
              <h3 className="text-[15px] font-semibold text-ink-100">{stage.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-ink-400">{stage.copy}</p>
            </div>
          </Card>
        ))}
      </ol>
    </>
  );
}
