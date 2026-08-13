import { A2ADemo } from "@/components/site/A2ADemo";
import { Reveal } from "@/components/site/Reveal";
import { SectionHeading } from "@/components/ui";

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
 * `A2ADemo` is the section's one visual. The topology diagram that used to sit
 * above it is deleted — it was scaffolding drawn while the delegation claim
 * could not be made in words, and a diagram of the exchange beside both the
 * sentence describing it and a demo performing it was the same idea three
 * times. Its removal also takes the `Fragment` usage with it, which is what
 * the deploy tripped over when two branches merged.
 *
 * The two agents are named lines beneath the demo rather than tiles: with a
 * demo present, a tile row would be a second visual, and the section budget
 * is one.
 */

const AGENTS = [
  {
    name: "Securitisation Readiness Agent",
    copy: "Warehoused loans against eligibility, coverage and concentration.",
  },
  {
    // The full name is kept although it is the longer of the two: "Portfolio"
    // is what separates this from corporate M&A diligence, which is worth
    // more than symmetry between two labels.
    name: "Portfolio Acquisition Intelligence Agent",
    copy: "Portfolio risks traced to evidence, with the open questions.",
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

      <Reveal delay={60}>
        <A2ADemo />
      </Reveal>

      <Reveal delay={120}>
        <dl className="mt-8 grid gap-x-8 gap-y-4 sm:grid-cols-2">
          {AGENTS.map((agent) => (
            <div key={agent.name}>
              <dt className="text-[15px] font-semibold text-ink-100">{agent.name}</dt>
              <dd className="mt-1 text-sm leading-relaxed text-ink-400">{agent.copy}</dd>
            </div>
          ))}
        </dl>
      </Reveal>
    </>
  );
}
