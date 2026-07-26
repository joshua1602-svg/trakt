import { CopilotDemo } from "@/components/demo/CopilotDemo";
import {
  CapabilityStack,
  Governance,
  Lifecycle,
  OperatingModel,
  Omnichannel,
} from "@/components/site/Content";
import { DemoVideo } from "@/components/site/DemoVideo";
import { Footer } from "@/components/site/Footer";
import { Hero } from "@/components/site/Hero";
import { LeadForm } from "@/components/site/LeadForm";
import { Nav } from "@/components/site/Nav";
import { Card, Section, SectionHeading, buttonStyles } from "@/components/ui";
import { buildMeta } from "@/lib/demo-pack";

/**
 * The landing page.
 *
 * A server component: the demo metadata (synthetic scope, allow-listed
 * questions, report actions, session limits) is read from the demo pack on the
 * server and passed down, so the page paints complete on first byte with no
 * client-side fetch and no layout shift. Interaction beyond that goes through
 * `/api/demo/*`.
 */
export default function Page() {
  const meta = buildMeta();

  return (
    <>
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <Nav />

      <main id="main">
        <div id="top" className="pt-28 pb-16 sm:pt-32 sm:pb-20">
          <Section id="product">
            <Hero scope={meta.scope} />
          </Section>
        </div>

        {/* C — Product overview video */}
        <Section id="overview" className="pb-16 sm:pb-20">
          <SectionHeading
            id="overview"
            eyebrow="Overview"
            title="Two minutes on what Trakt does"
            intro="A short walkthrough of the governed data layer, the analytics it powers and the channels it reaches."
          />
          <div className="mt-8">
            <DemoVideo />
          </div>
        </Section>

        {/* D — Interactive synthetic demonstration */}
        <Section id="live-demo" className="pb-6">
          <SectionHeading
            id="live-demo"
            eyebrow="Live demonstration"
            title="Ask a portfolio question. Get a governed answer."
            intro="This is the Trakt Copilot experience, running against a synthetic equity release portfolio. Answers are computed by the same deterministic engine that serves the Trakt workspace and Microsoft 365 Copilot."
          />
          <div className="mt-8">
            <CopilotDemo meta={meta} />
          </div>
        </Section>

        {/* E — Demo scope explanation */}
        <Section id="demo-scope" className="pb-16 sm:pb-20">
          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            <Card>
              <h3 className="text-sm font-semibold text-ink-100">What this shows</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-400">
                This demonstration shows Trakt&apos;s conversational interface. In
                production, the same governed data layer powers portfolio analytics,
                reporting, monitoring and operational workflows across the
                organisation.
              </p>
            </Card>
            <Card>
              <h3 className="text-sm font-semibold text-ink-100">What the data is</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-400">
                The demonstration uses a wholly synthetic portfolio. No client or
                consumer information is displayed. The page accepts no uploads, returns
                no exposure-level records and reaches no client environment.
              </p>
            </Card>
          </div>
        </Section>

        {/* F — Omnichannel */}
        <Section id="channels" className="pb-16 sm:pb-20">
          <Omnichannel />
        </Section>

        {/* G — Capability stack */}
        <Section id="capabilities" className="pb-16 sm:pb-20">
          <CapabilityStack />
        </Section>

        {/* H — Connected operating model */}
        <Section id="how-it-works" className="pb-16 sm:pb-20">
          <OperatingModel />
        </Section>

        {/* I — Portfolio lifecycle */}
        <Section id="lifecycle" className="pb-16 sm:pb-20">
          <Lifecycle />
        </Section>

        {/* J — Governance */}
        <Section id="governance" className="pb-16 sm:pb-20">
          <Governance />
        </Section>

        {/* K — Final CTA */}
        <Section id="book-a-demo" className="pb-20 sm:pb-24">
          <div className="rounded-2xl border border-line bg-navy-900/70 p-6 sm:p-9">
            <div className="grid gap-9 lg:grid-cols-[1fr_1.05fr] lg:gap-12">
              <div>
                <h2
                  id="book-a-demo-heading"
                  className="text-balance text-2xl font-semibold tracking-tight text-ink-100 sm:text-3xl"
                >
                  See Trakt applied to your operating model
                </h2>
                <p className="mt-4 text-[15px] leading-relaxed text-ink-300">
                  We will demonstrate how Trakt can integrate with your existing
                  portfolio data, reporting requirements and Microsoft 365 environment.
                </p>
                <a href="#overview" className={`${buttonStyles.secondary} mt-6`}>
                  Watch the overview
                </a>
              </div>
              <div>
                <LeadForm />
              </div>
            </div>
          </div>
        </Section>
      </main>

      <Footer />
    </>
  );
}
