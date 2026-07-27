import { CopilotDemo } from "@/components/demo/CopilotDemo";
import { AttributionCapture } from "@/components/site/AttributionCapture";
import {
  CapabilityStack,
  DeliveryModel,
  Lifecycle,
  OperatingModel,
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
 * Seven sections, in the order a sceptical institutional reader asks their
 * questions: what is it, what does it do, how do I get it, how does it work,
 * when do I use it, show me, talk to me.
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

        {/* 2 — Capabilities */}
        <Section id="capabilities" className="pb-16 sm:pb-20">
          <CapabilityStack />
        </Section>

        {/* 3 — Delivery model */}
        <Section id="delivery" className="pb-16 sm:pb-20">
          <DeliveryModel />
        </Section>

        {/* 4 — How it works */}
        <Section id="how-it-works" className="pb-16 sm:pb-20">
          <OperatingModel />
        </Section>

        {/* 5 — Where it applies */}
        <Section id="lifecycle" className="pb-16 sm:pb-20">
          <Lifecycle />
        </Section>

        {/* 6 — Example. The only demo surface on the page, and the only place
            the synthetic-portfolio disclaimer appears. */}
        <Section id="example" className="pb-16 sm:pb-20">
          <SectionHeading
            id="example"
            eyebrow="Example"
            title="Ask a portfolio question. Get a governed answer."
            intro="Every answer comes from the deterministic engine that serves the workspace and Microsoft 365 Copilot. The portfolio is wholly synthetic, and the page accepts no uploads."
          />
          <div className="mt-8">
            <CopilotDemo meta={meta} />
          </div>
        </Section>

        {/* 7 — Contact */}
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
                  We will demonstrate Trakt against your own portfolio data, reporting
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
