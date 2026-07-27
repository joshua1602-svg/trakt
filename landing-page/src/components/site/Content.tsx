"use client";

import { Card, SectionHeading, cx } from "@/components/ui";
import { track } from "@/lib/analytics";

/**
 * The static marketing sections: omnichannel, capability stack, connected
 * operating model and portfolio lifecycle.
 *
 * Every claim here is mapped to repository evidence in
 * `landing-page/docs/content-map.md`; wording that describes a capability the
 * repository does not yet support is not present.
 */

/* -------------------------------------------------------------------------- */
/* F. Omnichannel                                                             */
/* -------------------------------------------------------------------------- */

const CHANNELS = [
  {
    id: "copilot",
    name: "Microsoft 365 Copilot and Teams",
    // Trakt ships a declarative agent, packaged as a Teams app and surfaced
    // through Microsoft 365 Copilot. There is no standalone native Teams bot,
    // so the copy claims the surfaces, not a second product.
    copy: "Access Trakt's declarative agent through supported Microsoft 365 Copilot and Teams surfaces — ask portfolio questions, request reports and retrieve current management information without leaving the tools you already use.",
  },
  {
    id: "workspace",
    name: "Trakt Workspace",
    copy: "Use the full analytical workspace for dashboards, drill-through, monitoring and portfolio investigation.",
  },
  {
    id: "automated",
    name: "Automated Delivery",
    copy: "Generate scheduled reports, alerts, data outputs and governance artefacts without manual intervention.",
  },
] as const;

export function Omnichannel() {
  return (
    <>
      <SectionHeading
        id="channels"
        eyebrow="Access"
        title="Intelligence delivered where your teams already work"
        intro="The Copilot demonstration above is one way into Trakt, not the whole product. The same governed answer reaches every channel."
      />
      <ul className="mt-9 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {CHANNELS.map((channel) => (
          <Card as="li" key={channel.id} className="flex flex-col">
            <h3 className="text-[15px] font-semibold text-ink-100">{channel.name}</h3>
            <p className="mt-2.5 text-sm leading-relaxed text-ink-400">{channel.copy}</p>
          </Card>
        ))}
      </ul>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* G. Capability stack                                                        */
/* -------------------------------------------------------------------------- */

interface Capability {
  id: string;
  name: string;
  copy: string;
  items: string[];
  icon: string;
}

const CAPABILITIES: Capability[] = [
  {
    id: "integration",
    name: "Portfolio Integration",
    icon: "M4 7h16M4 12h16M4 17h10",
    copy: "Onboard new portfolios, acquisitions and servicing datasets into a consistent, traceable data model.",
    items: [
      "Data ingestion",
      "Mapping",
      "Standardisation",
      "Deduplication",
      "Portfolio migration",
      "Acquisition integration",
    ],
  },
  {
    id: "analytics",
    name: "Portfolio Analytics",
    icon: "M4 19V5M4 19h16M8 15v-4M13 15V8M18 15v-6",
    copy: "Monitor funded portfolios, pipelines, cohorts, concentrations, credit performance and operating trends.",
    items: [
      "Dashboards",
      "Natural-language analysis",
      "Drill-through",
      "Cohort analysis",
      "Trend analysis",
      "Concentration analysis",
    ],
  },
  {
    id: "management-reporting",
    name: "Management Reporting",
    icon: "M6 3h9l4 4v14H6zM15 3v4h4",
    copy: "Automate recurring management information and provide consistent answers across finance, risk, operations and leadership.",
    items: [
      "KPI packs",
      "Board reporting",
      "Variance analysis",
      "Management commentary",
      "Scheduled MI",
    ],
  },
  {
    id: "investor-reporting",
    name: "Investor Reporting",
    icon: "M4 18l5-6 4 3 6-8M20 7h-4M20 7v4",
    copy: "Produce repeatable investor and funding-partner reporting from the same governed portfolio data.",
    items: [
      "Investor packs",
      "Portfolio stratifications",
      "Performance commentary",
      "Scheduled delivery",
      "Funding-partner reporting",
    ],
  },
  {
    id: "regulatory",
    name: "Regulatory Reporting",
    icon: "M12 3l8 4v5c0 5-3.4 8.3-8 9-4.6-.7-8-4-8-9V7z",
    copy: "Transform validated portfolio data into prescribed regulatory and securitisation reporting structures.",
    items: [
      "ESMA Annex 2 reporting",
      "ESMA Annex 12 reporting",
      "Field validation",
      "Rule application",
      "Submission-ready XML",
    ],
  },
  {
    id: "governance",
    name: "Governance and Audit",
    icon: "M5 5h14v14H5zM9 9h6M9 13h6M9 17h3",
    copy: "Maintain transparent lineage from source data through transformation, calculation and final output.",
    items: [
      "Provenance",
      "Validation evidence",
      "Exception logs",
      "Versioning",
      "Approval records",
      "Reproducible outputs",
    ],
  },
  {
    id: "monitoring",
    name: "Portfolio Monitoring",
    icon: "M3 12h4l3 7 4-14 3 7h4",
    copy: "Identify data-quality issues, limit breaches, performance deterioration and emerging portfolio risks.",
    items: [
      "Risk alerts",
      "Covenant monitoring",
      "Concentration limits",
      "Exception management",
      "Data-quality monitoring",
    ],
  },
  {
    id: "omnichannel",
    name: "Omnichannel Intelligence",
    icon: "M12 3v18M3 12h18M6.5 6.5l11 11M17.5 6.5l-11 11",
    copy: "Provide governed portfolio answers through the Microsoft 365 Copilot and Teams surfaces, the Trakt workspace, APIs and scheduled reporting.",
    items: [
      "Conversational access",
      "Workflow actions",
      "Role-based delivery",
      "Reusable intelligence",
      "Scheduled outputs",
    ],
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
      <SectionHeading
        id="capabilities"
        eyebrow="What you buy"
        title="A governed operating layer for specialist lending"
        intro="Eight capability areas, connected through one data layer — not eight products bolted together."
      />
      <ul className="mt-9 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
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
            <details
              className="mt-3.5"
              onToggle={(event) => {
                if ((event.currentTarget as HTMLDetailsElement).open) {
                  track("capability_interaction", { capabilityId: capability.id });
                }
              }}
            >
              <summary className="cursor-pointer text-xs font-medium text-peri-300 hover:text-peri-200">
                Illustrative capabilities
              </summary>
              <ul className="mt-2.5 space-y-1">
                {capability.items.map((item) => (
                  <li key={item} className="flex gap-2 text-xs text-ink-400">
                    <span aria-hidden="true" className="text-peri-500">
                      ·
                    </span>
                    {item}
                  </li>
                ))}
              </ul>
            </details>
          </Card>
        ))}
      </ul>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* H. Connected operating model                                               */
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

const OUTPUTS = [
  "Copilot",
  "Workspace",
  "Reports",
  "Regulatory outputs",
  "Alerts",
] as const;

export function OperatingModel() {
  return (
    <>
      <SectionHeading
        id="how-it-works"
        eyebrow="How it works"
        title="One calculation, governed centrally, delivered everywhere"
        intro="Trakt calculates the answer once, governs it centrally and distributes it through every required channel."
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

      <div className="mt-8 rounded-xl border border-peri-500/35 bg-peri-400/[0.06] p-5 sm:p-6">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-peri-300">
          Delivered as
        </p>
        <ul className="mt-3 flex flex-wrap gap-2">
          {OUTPUTS.map((output) => (
            <li
              key={output}
              className="rounded-full border border-line bg-navy-900/70 px-3.5 py-1.5 text-sm text-ink-200"
            >
              {output}
            </li>
          ))}
        </ul>
        <p className="mt-4 text-sm leading-relaxed text-ink-300">
          Because every channel reads the same governed layer, the figure in a board
          pack, an investor report, a regulatory submission and a Copilot answer is
          the same figure — reconciled by construction rather than by comparison.
        </p>
      </div>
    </>
  );
}

/* -------------------------------------------------------------------------- */
/* I. Portfolio lifecycle                                                     */
/* -------------------------------------------------------------------------- */

const LIFECYCLE = [
  {
    title: "Launch or acquire a portfolio",
    copy: "Standardise incoming data, identify quality issues and establish the initial reporting environment.",
  },
  {
    title: "Run the portfolio",
    copy: "Monitor performance, answer management questions and automate recurring information.",
  },
  {
    title: "Finance the portfolio",
    copy: "Support warehouse, investor and lender reporting from controlled portfolio data.",
  },
  {
    title: "Securitise or refinance",
    copy: "Prepare validated datasets, regulatory outputs, portfolio stratifications and transaction reporting.",
  },
  {
    title: "Integrate additional portfolios",
    copy: "Bring new books into the existing operating model without building another reporting stack.",
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
      <ol className="mt-9 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {LIFECYCLE.map((stage, index) => (
          <Card as="li" key={stage.title} className="flex gap-4">
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

/* -------------------------------------------------------------------------- */
/* J. Governance and trust                                                    */
/* -------------------------------------------------------------------------- */

const GOVERNANCE = [
  {
    title: "Synthetic public demonstration",
    copy: "This page runs on a wholly synthetic portfolio. It accepts no uploads, exposes no client environment and returns no exposure-level record.",
  },
  {
    title: "Deterministic calculation logic",
    copy: "Portfolio figures are produced by a deterministic engine against a governed dataset. The same question returns the same number, every time and in every channel.",
  },
  {
    title: "Traceable source-to-output lineage",
    copy: "Each reporting cut retains its header mapping, transformation record, validation exceptions and delivery preflight, so any published figure can be traced back to source.",
  },
  {
    title: "Role-based access in production",
    copy: "Client deployments sit behind Microsoft Entra ID; Copilot actions require a validated bearer token and fail closed when they are not configured.",
  },
  {
    title: "Client-isolated environments",
    copy: "Trakt runs one deployment per client, with that client's data sources and storage — the public demonstration reaches none of them.",
  },
  {
    title: "Configurable data retention",
    copy: "Reporting history, artefact retention and output storage are configured per client environment rather than fixed by the platform.",
  },
  {
    title: "Controlled document and report delivery",
    copy: "Generated documents are delivered through short-lived, server-redeemed links. Storage credentials and storage paths never leave the server.",
  },
] as const;

export function Governance() {
  return (
    <>
      <SectionHeading
        id="governance"
        eyebrow="Why you can rely on it"
        title="Governance you can point at"
        intro="What follows is what Trakt does today. Where something is a deployment choice rather than a platform guarantee, it says so."
      />
      <ul className="mt-9 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {GOVERNANCE.map((item) => (
          <Card as="li" key={item.title}>
            <h3 className="text-[15px] font-semibold text-ink-100">{item.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink-400">{item.copy}</p>
          </Card>
        ))}
      </ul>
      <p
        className={cx(
          "mt-6 rounded-lg border border-line bg-navy-900/60 px-4 py-3",
          "text-xs leading-relaxed text-ink-400",
        )}
      >
        Deployment, hosting, retention and access-control arrangements are configured
        for each client environment, and are agreed as part of onboarding rather than
        being fixed by the platform.
      </p>
    </>
  );
}
