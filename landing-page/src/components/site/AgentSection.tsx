import { A2ADemo } from "@/components/site/A2ADemo";
import { Reveal } from "@/components/site/Reveal";
import { SectionHeading, cx } from "@/components/ui";

/**
 * Agent-to-agent — a roadmap section.
 *
 * Marked roadmap in the same grey as the Agent access delivery tile: both
 * channels are reserved in `trakt_core/context.py` with a test fixture as
 * their only implementation, so nothing here may read as live. "Trakt is
 * designed to" is deliberate roadmap phrasing and is not to be tightened
 * into a present-tense claim.
 *
 * The demo slot is deliberately empty. The Securitisation Readiness Agent
 * footage does not exist, and a mocked still — a fake terminal, invented
 * agent messages, figures nobody computed — would be the one thing on this
 * page that fabricates a capability. An empty reserved frame is the
 * deliverable until there is something real to put in it.
 */

const NODES = [
  { label: "External agent", connector: "A2A" },
  { label: "Client enterprise agent", connector: "Governed tool calls" },
  { label: "Trakt", connector: null },
] as const;

export function AgentSection() {
  return (
    <>
      <Reveal>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <SectionHeading
            id="agents"
            eyebrow="Agent-to-agent"
            title="Agents don't calculate the portfolio. Trakt does."
          />
        </div>
        <p className="mt-4 max-w-[72ch] text-[15px] leading-relaxed text-ink-200">
          Your agents already have somewhere to work. Give them somewhere trustworthy
          to get credit intelligence.
        </p>
        <p className="mt-3 max-w-[72ch] text-sm leading-relaxed text-ink-400">
          Trakt is designed to make governed portfolio intelligence available to
          enterprise agents — with controlled access, deterministic calculations and
          evidence behind every result.
        </p>
        <p className="mt-4 text-[11px] font-semibold uppercase tracking-wider text-ink-500">
          Roadmap
        </p>
      </Reveal>

      <Reveal delay={60}>
        <Topology />
      </Reveal>

      <Reveal delay={90}>
        <A2ADemo />
      </Reveal>
    </>
  );
}

/**
 * Three nodes on one line, protocol names as connector labels only. No
 * multi-tier stack: the platform section owns that shape. No vendor mark is
 * used as an actor — Copilot is a surface Trakt is reached through, named in
 * prose elsewhere, not a party in this exchange.
 */
function Topology() {
  return (
    <ol
      className="mt-9 flex flex-col items-stretch gap-2 md:flex-row md:items-center"
      aria-label="Agent-to-agent topology"
    >
      {NODES.map((node, index) => (
        <li
          key={node.label}
          className={cx("flex flex-col gap-2 md:flex-row md:items-center", index === 2 && "md:flex-1")}
        >
          <div
            className={cx(
              "rounded-xl border px-5 py-4 text-center md:min-w-[13rem]",
              index === 2
                ? "border-peri-500/60 bg-navy-850 text-ink-100"
                : "border-line-soft bg-navy-900/50 text-ink-300",
            )}
          >
            <p className="text-sm font-semibold">{node.label}</p>
          </div>
          {node.connector ? (
            <div
              aria-hidden="true"
              className="flex items-center justify-center gap-2 px-2 text-[11px] font-medium uppercase tracking-wider text-ink-500 md:flex-col md:gap-1"
            >
              <span className="md:hidden">↓</span>
              <span>{node.connector}</span>
              <span className="hidden md:inline">→</span>
            </div>
          ) : null}
        </li>
      ))}
    </ol>
  );
}
