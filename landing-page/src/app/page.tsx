import { CopilotDemo } from "@/components/demo/CopilotDemo";
import { AttributionCapture } from "@/components/site/AttributionCapture";
import {
  Architecture,
  DeliveryStrip,
  ForwardControls,
  Governance,
  Lenses,
  Onboarding,
  ReportingBand,
} from "@/components/site/Content";
import { Footer } from "@/components/site/Footer";
import { Hero } from "@/components/site/Hero";
import { LeadForm } from "@/components/site/LeadForm";
import { Nav } from "@/components/site/Nav";
import { Section, SectionHeading } from "@/components/ui";
import { buildMeta } from "@/lib/demo-pack";

/**
 * The landing page.
 *
 * A server component: the demo metadata (synthetic scope, allow-listed
 * questions, report actions, session limits) is read from the demo pack on the
 * server and passed down, so the page paints complete on first byte with no
 * client-side fetch and no layout shift. Interaction beyond that goes through
 * `/api/demo/*`.
 *
 * Nine sections, in the order a sceptical institutional reader asks their
 * questions: what is it, how does it fit together, what does it control, how
 * do I get onto it, how does it scale across books, show me, why can I rely on
 * it, what does it produce, talk to me.
 */
export default function Page() {
  const meta = buildMeta();

  return (
    <>
      <AttributionCapture />
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <Nav />

      <main id="main">
        {/* 1 — Value proposition */}
        <div id="top" className="pt-28 pb-16 sm:pt-32 sm:pb-20">
          <Section id="product">
            <Hero scope={meta.scope} />
          </Section>
        </div>

        {/* 2 — The platform: data + documents → governed layer → outputs */}
        <Section id="platform" className="pb-16 sm:pb-20">
          <Architecture />
        </Section>

        {/* 3 — Controls and forward risk */}
        <Section id="controls" className="pb-16 sm:pb-20">
          <ForwardControls />
        </Section>

        {/* 4 — Governed onboarding */}
        <Section id="onboarding" className="pb-16 sm:pb-20">
          <Onboarding />
        </Section>

        {/* 5 — Multi-portfolio lenses */}
        <Section id="lenses" className="pb-16 sm:pb-20">
          <Lenses scope={meta.scope} />
        </Section>

        {/* 6 — Portfolio intelligence. The only demo surface on the page, and
            the only place the synthetic-portfolio disclaimer appears. The
            inner #example anchor is the hero CTA's landing point. */}
        <Section id="intelligence" className="pb-16 sm:pb-20">
          <SectionHeading
            id="intelligence"
            eyebrow="Portfolio intelligence"
            title="Portfolio intelligence where your team already works."
            intro="Ask portfolio questions in natural language — in the Trakt workspace, in Microsoft Teams, or through Microsoft 365 Copilot. Every answer comes from the deterministic engine, so the same question returns the same number in every channel — and Trakt declines what it cannot derive. Try it against three governed books below. The portfolios are wholly synthetic, and the page accepts no uploads."
          />
          <DeliveryStrip />
          <div id="example" className="mt-8 scroll-mt-24">
            <CopilotDemo meta={meta} />
          </div>
        </Section>

        {/* 7 — Governance and platform properties */}
        <Section id="governance" className="pb-16 sm:pb-20">
          <Governance />
        </Section>

        {/* 8 — Reporting as an output, not the identity */}
        <Section id="reporting" className="pb-16 sm:pb-20">
          <ReportingBand />
        </Section>

        {/* 9 — Contact */}
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
                  We will demonstrate Trakt against your own portfolios, funding
                  requirements and Microsoft 365 environment.
                </p>
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
