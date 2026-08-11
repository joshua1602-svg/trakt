import { AttributionCapture } from "@/components/site/AttributionCapture";
import {
  Architecture,
  DeliveryModes,
  DeliveryStrip,
  ForwardControls,
  Governance,
} from "@/components/site/Content";
import { Footer } from "@/components/site/Footer";
import { Hero } from "@/components/site/Hero";
import { LeadForm } from "@/components/site/LeadForm";
import { Nav } from "@/components/site/Nav";
import { QueryDemo } from "@/components/site/QueryDemo";
import { RefusalSection } from "@/components/site/RefusalSection";
import { Reveal } from "@/components/site/Reveal";
import { Section, SectionHeading, buttonStyles } from "@/components/ui";
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
 * The narrative, in the order a first-time visitor asks their questions:
 * what is it → show me (query demo) → how does it work → what does it control
 * (controls demo) → how does it scale across books → where do users reach it →
 * why trust it → what next. Two demos, both user-started, never autoplaying:
 * first the visitor interrogates the portfolio, then they watch Trakt monitor
 * that same portfolio.
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
        {/* 1 — What is Trakt? */}
        <div id="top" className="pt-28 pb-16 sm:pt-32 sm:pb-20">
          <Section id="product">
            <Hero scope={meta.scope} />
          </Section>
        </div>

        {/* 2 — Demo 1: portfolio query. The only demo surface for querying,
            and the only place the synthetic-portfolio disclaimer appears. */}
        <Section id="query-demo" className="pb-16 sm:pb-20">
          <Reveal>
            <SectionHeading
              id="query-demo"
              eyebrow="Portfolio query demo"
              title="Ask the portfolio. Get a governed answer."
              intro="Ask portfolio questions in natural language and get answers from the same governed calculations, wherever your team works."
            />
          </Reveal>
          <Reveal delay={60}>
            <div id="example" className="mt-8 scroll-mt-24">
              <QueryDemo meta={meta} />
            </div>
          </Reveal>
          <Reveal delay={120}>
            <p className="mt-4 text-[13px] font-medium text-ink-300">
              Same question. Same calculation. Same answer.
            </p>
          </Reveal>
        </Section>

        {/* 2b — The refusal claim, out of the demo card and into its own
            section: behind a play control it was only ever a still. */}
        <Section id="refusal" className="pb-16 sm:pb-20">
          <Reveal>
            <RefusalSection meta={meta} />
          </Reveal>
        </Section>

        {/* 3 — How does the platform work? */}
        <Section id="platform" className="pb-16 sm:pb-20">
          <Architecture />
        </Section>

        {/* 4 — Risk & controls, with Demo 2. */}
        <Section id="controls" className="pb-16 sm:pb-20">
          <ForwardControls />
        </Section>

        {/* The Operating Model section is gone: its claim ("no separate
            datasets to reconcile") now sits in the platform diagram's step 2,
            where the governed layer is actually described. */}

        {/* 5 — Where users reach it. Distribution only — the query demo has
            already appeared in section 2. */}
        <Section id="intelligence" className="pb-16 sm:pb-20">
          <Reveal>
            {/* The deleted intro restated the query-demo section's own intro
                and named the same three channels the chips below name. */}
            <SectionHeading
              id="intelligence"
              eyebrow="Portfolio intelligence"
              title="Portfolio intelligence where your team already works."
            />
            <DeliveryStrip />
          </Reveal>
        </Section>

        {/* 6b — The delivery model: five modes, stated once each, live and
            roadmap kept visually distinct. Static — nothing to open. */}
        <Section id="delivery" className="pb-16 sm:pb-20">
          <Reveal>
            <SectionHeading
              id="delivery"
              eyebrow="Delivery model"
              title="Every mode reads the same governed layer."
            />
          </Reveal>
          <div className="mt-9">
            <DeliveryModes />
          </div>
        </Section>

        {/* 7 — Why trust the outputs? */}
        <Section id="governance" className="pb-16 sm:pb-20">
          <Governance />
        </Section>

        {/* 8 — What next? */}
        <Section id="book-a-demo" className="pb-20 sm:pb-24">
          <div className="rounded-2xl border border-line bg-navy-900/70 p-6 sm:p-9">
            <div className="grid gap-9 lg:grid-cols-12 lg:gap-x-8">
              <div className="lg:col-span-6">
                <h2
                  id="book-a-demo-heading"
                  className="text-balance text-2xl font-semibold tracking-tight text-ink-100 sm:text-3xl"
                >
                  See your portfolio through one governed view.
                </h2>
                <p className="mt-4 max-w-[72ch] text-[15px] leading-relaxed text-ink-300">
                  We will demonstrate Trakt against your own portfolios, funding
                  requirements and Microsoft 365 environment.
                </p>
                <a href="#query-demo" className={`${buttonStyles.secondary} mt-6`}>
                  Explore the live demo
                </a>
              </div>
              <div className="lg:col-span-6">
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
