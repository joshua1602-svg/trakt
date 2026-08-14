import { A2APreview } from "@/components/site/A2APreview";
import { DemoPlayer } from "@/components/site/DemoPlayer";
import { Reveal } from "@/components/site/Reveal";
import { Card, SectionHeading } from "@/components/ui";

/**
 * Agent-to-agent.
 *
 * The section's claim is delegation: an agent that has never heard of Trakt
 * can find a Trakt agent and hand it an objective. Stated once, under the
 * headline, in the page's claim grammar (`text-lg font-medium text-ink-100`)
 * — the same treatment the hero, risk & controls and governance lines use.
 * No roadmap label and no "is designed to": the mechanism is proven, so the
 * claim is present tense.
 *
 * The film is the section's one visual, played the way the controls demo is
 * played: poster, a play control, pause and replay, `preload="none"`, and a
 * text equivalent for anyone who cannot or would rather not watch. It replaces
 * an animated CSS component that autoplayed with no poster and no controls —
 * page furniture wearing a demo's clothes, and against the page's own rule
 * that every demo is user-started.
 *
 * The topology diagram that used to sit above it is deleted — it was
 * scaffolding drawn while the delegation claim could not be made in words, and
 * a diagram of the exchange beside both the sentence describing it and a demo
 * performing it was the same idea three times. Its removal also takes the
 * `Fragment` usage with it, which is what the deploy tripped over when two
 * branches merged.
 *
 * The two agents sit ABOVE the demo, as outlined tiles rather than the small
 * named lines they were. Underneath a fifty-second film they were furniture
 * nobody reached — the products of the section, ranked below a recording of
 * one of them. Above it they are the claim, and the film is the evidence for
 * the claim, which is the right order for both.
 *
 * Their outline is `border-line`, not mint. Mint means available, and the
 * capability matrix above differentiates these two agents on evidence:
 * Securitisation Readiness ships, Portfolio Acquisition DD is marked roadmap.
 * A mint outline on both would have this section contradicting the matrix two
 * screens up. Their glyphs are peri, because a glyph marks what a thing is,
 * not what state it is in.
 */

const AGENTS = [
  {
    name: "Securitisation Readiness Agent",
    copy: "Point it at a warehouse and it works the whole book — eligibility, coverage, concentration — and tells you what would fail a deal before the arranger does.",
    Glyph: TrancheGlyph,
  },
  {
    // The full name is kept although it is the longer of the two: "Portfolio"
    // is what separates this from corporate M&A diligence, which is worth
    // more than symmetry between two labels.
    name: "Portfolio Acquisition Intelligence Agent",
    copy: "Reads a portfolio you are buying, finds what the seller's summary leaves out, and hands back every risk with the evidence behind it — and the questions still open.",
    Glyph: DiligenceGlyph,
  },
] as const;

export function AgentSection() {
  return (
    <>
      <Reveal>
        <SectionHeading
          id="agents"
          eyebrow="Agent-to-agent"
          title="Agents don't calculate the portfolio. Trakt does."
        />
        <p className="mt-4 max-w-[72ch] text-lg font-medium leading-relaxed text-ink-100">
          An agent that knows nothing about Trakt can discover it and delegate an
          objective.
        </p>
        <p className="mt-3 max-w-[72ch] text-sm leading-relaxed text-ink-400">
          Your agents already have somewhere to work. Give them somewhere trustworthy
          to get credit intelligence.
        </p>
      </Reveal>

      <ul className="mt-8 grid gap-4 sm:grid-cols-2">
        {AGENTS.map((agent, index) => (
          <li key={agent.name}>
            <Reveal delay={index * 60} className="h-full">
              <Card className="flex h-full flex-col">
                <agent.Glyph />
                <h3 className="mt-3 text-base font-semibold text-ink-100">{agent.name}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-300">{agent.copy}</p>
              </Card>
            </Reveal>
          </li>
        ))}
      </ul>

      <Reveal delay={120}>
        <div className="mt-8">
          <DemoPlayer
            overlayLabel="Watch the delegation demo"
            durationLabel="~50 sec"
            poster="/a2a-demo-poster.png"
            webmSrc="/a2a-demo.webm"
            mp4Src="/a2a-demo.mp4"
            description="Demonstration: an enterprise agent that knows nothing of Trakt's internals discovers the Trakt Securitisation Readiness Agent over A2A, delegates the objective “assess this portfolio for securitisation readiness”, and receives a structured assessment — Trakt choosing what to investigate across 30 governed queries, and every finding traced to the calculation behind it."
            caption="One recorded run against a synthetic portfolio. Figures illustrative."
            fallback={<A2APreview />}
            plateId="a2a"
          />
        </div>
      </Reveal>
    </>
  );
}

/**
 * Two line glyphs, drawn here rather than imported: the page carries no icon
 * dependency and does not need one for two marks. Each says what its agent
 * operates on — a tranched stack, and a portfolio under examination.
 */

function TrancheGlyph() {
  return (
    <svg
      width="26"
      height="26"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="text-peri-300"
    >
      <rect x="3" y="4" width="18" height="4.5" rx="1.2" />
      <rect x="3" y="10.5" width="18" height="4.5" rx="1.2" />
      <rect x="3" y="17" width="11" height="4.5" rx="1.2" />
    </svg>
  );
}

function DiligenceGlyph() {
  return (
    <svg
      width="26"
      height="26"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="text-peri-300"
    >
      <path d="M4 20V9M9 20V4M14 20v-7" />
      <circle cx="18" cy="12" r="3.4" />
      <path d="M20.5 14.5L23 17" />
    </svg>
  );
}
