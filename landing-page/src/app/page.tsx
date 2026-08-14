import { AgentSection } from "@/components/site/AgentSection";
import { AttributionCapture } from "@/components/site/AttributionCapture";
import { Capability } from "@/components/site/Capability";
import { Architecture, ForwardControls, Governance } from "@/components/site/Content";
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
 * what is it → show me, and what will it refuse (query demo) → how does it
 * work → what does it actually cover → what does it control (controls demo) →
 * where is it going → why trust it → what next.
 *
 * The capability matrix sits above risk & controls deliberately: a capability
 * demonstrated before the reader knows the shape of what Trakt does — and
 * which channel they would reach it through — is a capability with nowhere to
 * land. It replaced the Delivery Model section in that slot. The two demos
 * stay apart, each beside the claim it proves; both are user-started and
 * neither autoplays.
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
          {/* "Same question. Same calculation. Same answer." has moved to the
              hero. As a caption here it was the page's central claim doing
              footnote duty, and it read as commentary on this one demo rather
              than on the four surfaces the page shows. The demo keeps its own
              figcaption. */}
          {/* The refusal claim, folded back in. It is a sub-claim of this
              demo, not a destination, and as its own section it sat in
              adjacent territory to the agent-to-agent headline. Beneath the
              frame it keeps heading-scale type: same function, one less
              section, no loss of weight. */}
          <Reveal delay={120}>
            <RefusalSection meta={meta} />
          </Reveal>
        </Section>

        {/* 3 — How does the platform work? */}
        <Section id="platform" className="pb-16 sm:pb-20">
          <Architecture />
        </Section>

        {/* 3b — What "everywhere" means. The platform section makes the claim;
            this is the only place on the page that answers it. Directly after
            it by design: a matrix of sixteen capabilities read before the
            layer that produces them is a feature list. */}
        <Section id="capability" className="pb-16 sm:pb-20">
          <Capability />
        </Section>

        {/* The Operating Model section is gone: its claim ("no separate
            datasets to reconcile") now sits in the platform diagram's step 2,
            where the governed layer is actually described. */}

        {/* The Portfolio Intelligence section is gone: its channel chips named
            the same surfaces as the delivery tiles below — "Trakt workspace"
            and "Trakt Agent" were one thing under two names — in a second
            format, two sections apart. Its one uncovered claim, proactive
            Teams delivery, is now the delivery section's body line. */}

        {/* The Delivery Model section is gone. Its four tiles said what the
            capability matrix's DELIVERY column says in five items, two
            sections further down; its body line moved to Risk & Controls,
            beside the demo that produces the finding. See Content.tsx. */}

        {/* 4 — Risk & controls, with Demo 2. Below the delivery model by
            design: a reader meeting a controls demo before they know how they
            would consume Trakt at all is being shown a capability with no
            channel to reach it through. Platform → delivery → capability and
            its proof. The demos stay apart, each next to the claim it
            proves — three posters in a row read as one repeated thing. */}
        <Section id="controls" className="pb-16 sm:pb-20">
          <ForwardControls />
        </Section>

        {/* 6c — Agent-to-agent. Two named agents and the delegation claim; the
            topology diagram is deleted, not hidden. The tiles are the visual,
            and the Securitisation Readiness demo takes that slot when it
            exists. No demo placeholder: an empty reserved frame reads as
            broken however it is labelled. */}
        <Section id="agents" className="pb-16 sm:pb-20">
          <AgentSection />
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
