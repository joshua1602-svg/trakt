import { cx } from "@/components/ui";

/**
 * The agent-to-agent demonstration — 38 seconds, continuous, looping.
 *
 * This fills the slot `AgentSection` deliberately left empty, and it may only
 * be filled now because the footage finally exists. Sprint 4 ran five delegated
 * assessments end to end; every number, state name, verdict and count below is
 * taken from those recorded runs.
 *
 * One run, not an average
 * -----------------------
 * Every figure comes from **run 1** of the five, and from that run alone. Runs
 * 2–5 reached the same verdict by slightly different routes (30–33 governed
 * calls, 6–11 findings), so averaging them would have produced a portfolio that
 * no recorded run actually saw. Quoting one run keeps the numbers mutually
 * consistent: 30 governed calls, 31% London, 12% above 80% LTV, six findings,
 * eight diligence items — all of it the same assessment.
 *
 * Where the demo differs from a first sketch, the recorded run won
 * ----------------------------------------------------------------
 * The concentration figure is **31%**, not a rounder 28%: 31% is what
 * `concentration` returned on the synthetic book, and it is the number that
 * makes the three-rulebook comparison true (PASS ≤35, BREACH ≤27, FLAG >25).
 * The counts are the run's real ones — six material findings, eight
 * further-diligence items, four declared gaps — not a tidier three and two, and
 * not the ten-and-ten an earlier draft carried, which matched no run at all. A
 * demo that rounds its evidence is advertising something other than the
 * product.
 *
 * Why the keyframes are generated
 * -------------------------------
 * CSS keyframe percentages cannot be driven by a custom property. The first
 * attempt tried, and every narration beat stayed on screen for 34 of the 38
 * seconds, because one shared keyframe faded in at 6% and out at 94% of the
 * whole loop regardless of each beat's own window — nine captions stacked on
 * top of one another. So each element gets its own keyframe, generated from the
 * same `at`/`until` seconds the storyboard is written in. One master loop, many
 * windows, and no JavaScript at runtime: nothing to drift, nothing to reset on
 * a tab switch, and a seam that closes because the first and last frames match.
 *
 * Reduced motion is not an afterthought: the global `prefers-reduced-motion`
 * rule collapses every animation to its final frame, so the final beat is the
 * one that reads correctly standing still.
 *
 * On the caller's name
 * --------------------
 * The caller is "Enterprise Agent", unbranded. Naming a real lender on a public
 * page asserts a client relationship, and this repository contains no such
 * deployment — the synthetic environment is ERE. The badge says synthetic once,
 * where it cannot be missed.
 */

const LOOP_S = 38;
const FADE_S = 0.4;

type Beat = {
  at: number;
  until: number;
  label: string;
  lines: readonly string[];
  tone?: "neutral" | "breach" | "verdict";
  extra?: "rulebooks" | "triad";
};

/** Every string below is taken from run 1's own record — the objective sent,
 *  the skill tags published on the Agent Card, the task lifecycle, the call
 *  count, the findings and the counts. */
const BEATS: readonly Beat[] = [
  { at: 0.3, until: 4.2, label: "Objective",
    lines: ["“Assess this portfolio for securitisation readiness.”"] },
  // The card's own published tags, verbatim — the caller matches on these, and
  // showing anything more fluent would misrepresent what discovery reads.
  { at: 4.4, until: 8.0, label: "Specialist discovered",
    lines: ["Securitisation · Credit · Portfolio · Risk · Structured finance · Due diligence"] },
  { at: 8.2, until: 12.0, label: "Delegated",
    lines: ["Identity ✓   Portfolio entitlement ✓   submitted → working"] },
  { at: 12.2, until: 20.6, label: "Investigating",
    lines: ["30 governed queries, chosen by the specialist"] },
  { at: 20.8, until: 24.4, label: "Concentration found", tone: "breach",
    lines: ["London — 31% of balance"], extra: "rulebooks" },
  { at: 24.6, until: 27.0, label: "Investigating related exposure",
    lines: ["12% above 80% LTV — the same loans carry the stale valuations"] },
  { at: 27.2, until: 31.6, label: "Fact · Rule · Judgement", tone: "verdict",
    lines: [], extra: "triad" },
  { at: 31.8, until: 35.6, label: "Assessment received ✓", tone: "verdict",
    lines: ["Material remediation required",
            "6 material findings · 8 diligence items · 4 declared gaps"] },
  { at: 35.8, until: 37.9, label: "Agent-ready",
    lines: ["Specialist lending intelligence."] },
];

/** The capabilities the specialist reached, in the order the run reached them. */
const INVESTIGATION = [
  "Portfolio composition", "Regulatory readiness", "Concentrations",
  "LTV & collateral", "Vintage performance", "Arrears & defaults",
  "Prepayments", "WAL / YTM",
] as const;

const INVESTIGATION_START = 12.4;
const INVESTIGATION_STEP = 0.85;

/** One fact, three authorities, three different answers — from the run's own
 *  `rule` field. */
const RULEBOOKS = [
  { authority: "Warehouse facility", limit: "≤35%", outcome: "PASS" },
  { authority: "Securitisation criteria", limit: "≤27%", outcome: "BREACH" },
  { authority: "Trakt screening", limit: ">25%", outcome: "FLAG" },
] as const;

type Packet = { at: number; label: string; wire: "a2a" | "mcp"; back: boolean };

/** Marks crossing the connections. Labelled in business words — never protocol
 *  detail — and timed so the network is busy whenever the story says it is. */
const PACKETS: readonly Packet[] = [
  { at: 8.3, label: "Task", wire: "a2a", back: false },
  { at: 10.4, label: "Accepted", wire: "a2a", back: true },
  { at: 12.5, label: "Query", wire: "mcp", back: false },
  { at: 14.1, label: "Fact", wire: "mcp", back: true },
  { at: 15.7, label: "Query", wire: "mcp", back: false },
  { at: 17.3, label: "Evidence", wire: "mcp", back: true },
  { at: 18.9, label: "Query", wire: "mcp", back: false },
  { at: 20.5, label: "Evidence", wire: "mcp", back: true },
  { at: 24.9, label: "Drill-down", wire: "mcp", back: false },
  { at: 26.3, label: "Evidence", wire: "mcp", back: true },
  { at: 31.9, label: "Result", wire: "a2a", back: true },
  { at: 34.0, label: "Received", wire: "a2a", back: false },
];

const pc = (seconds: number) =>
  `${(Math.min(Math.max(seconds, 0), LOOP_S) / LOOP_S * 100).toFixed(3)}%`;

/** A fade-in / hold / fade-out window inside the shared loop. */
function windowFrames(name: string, at: number, until: number,
                      enter: string, exit: string, hold = "none"): string {
  const inAt = at + FADE_S;
  const outAt = Math.max(inAt + 0.1, until - FADE_S);
  return (
    `@keyframes ${name}{` +
    `0%,${pc(at)}{opacity:0;transform:${enter}}` +
    `${pc(inAt)}{opacity:1;transform:${hold}}` +
    `${pc(outAt)}{opacity:1;transform:${hold}}` +
    `${pc(until)},100%{opacity:0;transform:${exit}}}`
  );
}

/** Every keyframe the demo needs, emitted once into the document. */
function timelineCss(): string {
  const rules: string[] = [];

  BEATS.forEach((beat, i) => {
    rules.push(windowFrames(`a2ab${i}`, beat.at, beat.until,
                            "translateY(8px)", "translateY(-8px)"));
  });

  INVESTIGATION.forEach((_, i) => {
    const at = INVESTIGATION_START + i * INVESTIGATION_STEP;
    rules.push(
      `@keyframes a2ac${i}{` +
      `0%,${pc(at)}{opacity:.28;border-color:var(--color-line-soft)}` +
      `${pc(at + 0.4)}{opacity:1;border-color:var(--color-peri-500)}` +
      `${pc(20.6)}{opacity:1;border-color:var(--color-peri-500)}` +
      `${pc(21.6)},100%{opacity:.4;border-color:var(--color-line-soft)}}`);
  });

  PACKETS.forEach((packet, i) => {
    const from = packet.wire === "a2a"
      ? `translateX(${packet.back ? "2.6rem" : "-2.6rem"})`
      : `translateY(${packet.back ? "1.5rem" : "-1.5rem"})`;
    const to = packet.wire === "a2a"
      ? `translateX(${packet.back ? "-2.6rem" : "2.6rem"})`
      : `translateY(${packet.back ? "-1.5rem" : "1.5rem"})`;
    rules.push(windowFrames(`a2ap${i}`, packet.at, packet.at + 1.7,
                            from, to, "none"));
  });

  return rules.join("");
}

const LOOP = {
  animationDuration: `${LOOP_S}s`,
  animationIterationCount: "infinite" as const,
  animationTimingFunction: "ease-in-out" as const,
  animationFillMode: "both" as const,
};

export function A2ADemo() {
  return (
    <figure
      className="mt-10 overflow-hidden rounded-2xl border border-line-soft bg-navy-900/60"
      aria-label={
        "Animated demonstration: an enterprise agent discovers the Trakt " +
        "Securitisation Readiness Agent, delegates an objective, and receives " +
        "an evidence-backed assessment. A text summary follows the figure."
      }
    >
      {/* Generated once, server-rendered, no runtime cost. */}
      <style dangerouslySetInnerHTML={{ __html: timelineCss() }} />
      <Header />
      <div className="px-4 pb-5 pt-4 sm:px-6">
        <Topology />
        <Narration />
      </div>
      <Caption />
    </figure>
  );
}

function Header() {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line-soft px-4 py-3 sm:px-6">
      <div className="flex items-center gap-2">
        <span className="a2a-wire h-1.5 w-1.5 rounded-full bg-peri-400" />
        <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-400">
          Agent-to-agent delegation
        </p>
      </div>
      {/* Said once, where it cannot be missed. */}
      <span className="rounded-full border border-line-soft bg-navy-850 px-2.5 py-1 text-[10px] font-medium uppercase tracking-wider text-ink-400">
        Synthetic demonstration
      </span>
    </div>
  );
}

/**
 * The persistent network — visible for the whole loop, so the viewer sees a
 * system with things moving through it rather than a sequence of slides.
 *
 * Desktop places the two agents side by side with the intelligence beneath
 * Trakt. Mobile stacks all three: a shrunk horizontal topology stops being
 * readable long before it stops fitting.
 */
function Topology() {
  return (
    <div className="mx-auto max-w-[46rem]">
      <div className="grid grid-cols-1 items-center gap-2 md:grid-cols-[1fr_7rem_1fr]">
        <Node title="Enterprise Agent" subtitle="Holds the objective"
              detail="No knowledge of Trakt's internals" />
        <Wire kind="a2a" label="A2A" />
        <Node title="Trakt" subtitle="Securitisation Readiness Agent"
              detail="Decides what to investigate" primary />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[1fr_7rem_1fr]">
        <div className="hidden md:block" />
        <div className="hidden md:block" />
        <Wire kind="mcp" label="MCP" />
      </div>
      {/* Full width: the governed layer is the floor of the whole system, not
          an appendage of the right-hand column — and a half-width node left a
          dead quadrant under the caller. */}
      <Node title="Governed Portfolio Intelligence"
            subtitle="Deterministic calculation · evidence · provenance" muted />

      <Capabilities />
    </div>
  );
}

function Node({ title, subtitle, detail, primary, muted }: {
  title: string; subtitle?: string; detail?: string;
  primary?: boolean; muted?: boolean;
}) {
  return (
    <div className={cx(
      "rounded-xl border px-4 py-3 text-center",
      primary ? "border-peri-500/60 bg-navy-850"
        : muted ? "border-line-soft bg-navy-900/60" : "border-line bg-navy-900/50",
    )}>
      <p className={cx("text-[13px] font-semibold sm:text-sm",
                       primary ? "text-ink-100" : "text-ink-200")}>{title}</p>
      {subtitle ? <p className="mt-0.5 text-[11px] leading-snug text-ink-400">{subtitle}</p> : null}
      {detail ? <p className="mt-0.5 text-[10px] leading-snug text-ink-500">{detail}</p> : null}
    </div>
  );
}

/**
 * A connection carrying a faint current, with packets crossing it.
 *
 * The protocol name sits on the wire and the packets travel *beside* it — the
 * first version put both in the same place and the labels collided, which read
 * as a rendering fault rather than as traffic.
 */
function Wire({ kind, label }: { kind: "a2a" | "mcp"; label: string }) {
  const horizontal = kind === "a2a";
  return (
    <div aria-hidden="true"
         className={cx("relative flex items-center justify-center",
                       horizontal ? "h-10 md:h-full" : "h-11")}>
      <span className={cx(
        "a2a-wire absolute bg-peri-500/40",
        horizontal
          ? "left-1/2 top-0 bottom-0 w-px md:left-0 md:right-0 md:top-1/2 md:bottom-auto md:h-px md:w-auto"
          : "left-1/2 top-0 bottom-0 w-px",
      )} />
      <span className="relative z-10 rounded-full border border-line-soft bg-navy-950 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-ink-500">
        {label}
      </span>
      {PACKETS.map((packet, i) => packet.wire !== kind ? null : (
        <span
          key={`${packet.label}-${i}`}
          className={cx(
            "absolute whitespace-nowrap rounded-full border border-peri-500/50 bg-navy-850 px-1.5 py-0.5 text-[8px] font-medium uppercase tracking-wider text-peri-200",
            // Offset clear of the wire label so the two never overlap.
            // Beside the wire, never on it: the label and the packet in the
            // same place read as a rendering fault rather than as traffic.
            horizontal
              ? "top-0 md:top-1"
              : packet.back
                ? "left-1/2 -translate-x-[7.5rem]"
                : "left-1/2 translate-x-[1.5rem]",
          )}
          style={{ ...LOOP, animationName: `a2ap${i}` }}
        >
          {packet.label}
        </span>
      ))}
    </div>
  );
}

/** The investigation, lighting up as the specialist reaches each area. */
function Capabilities() {
  return (
    <ul aria-hidden="true" className="mt-3 grid grid-cols-2 gap-1.5 sm:grid-cols-4">
      {INVESTIGATION.map((label, i) => (
        <li key={label}
            className="rounded-lg border border-line-soft bg-navy-900/60 px-2 py-1.5 text-center text-[10px] leading-tight text-ink-300"
            style={{ ...LOOP, animationName: `a2ac${i}` }}>
          {label}
        </li>
      ))}
    </ul>
  );
}

/**
 * The narrative band. One beat visible at a time, in a fixed-height box so
 * nothing below it moves — a demo that reflows on every step reads as slides,
 * which is what this must not be.
 */
function Narration() {
  return (
    // Tall enough on mobile for the longest beat — the Fact/Rule/Judgement
    // triad, whose judgement wraps at 390px. Sized for the worst case rather
    // than the average, because a band that clips its own evidence is worse
    // than one carrying some empty space on the shorter beats.
    <div className="relative mx-auto mt-4 h-[6.75rem] max-w-[46rem] sm:h-[5rem]">
      {BEATS.map((beat, i) => (
        <div key={beat.label}
             className="absolute inset-0 flex flex-col items-center justify-start text-center"
             style={{ ...LOOP, animationName: `a2ab${i}` }}>
          <p className={cx(
            "text-[10px] font-semibold uppercase tracking-wider",
            beat.tone === "breach" ? "text-amber-300"
              : beat.tone === "verdict" ? "text-peri-200" : "text-ink-500",
          )}>{beat.label}</p>
          {beat.lines.map((line) => (
            <p key={line} className="mt-1 text-[13px] leading-snug text-ink-100 sm:text-[15px]">
              {line}
            </p>
          ))}
          {beat.extra === "rulebooks" ? <Rulebooks /> : null}
          {beat.extra === "triad" ? <Triad /> : null}
        </div>
      ))}
    </div>
  );
}

function Rulebooks() {
  return (
    <ul className="mt-1.5 flex flex-wrap items-center justify-center gap-1">
      {RULEBOOKS.map((rule) => (
        <li key={rule.authority}
            className={cx(
              "rounded-full border px-2 py-0.5 text-[9px] font-medium sm:text-[10px]",
              rule.outcome === "BREACH"
                ? "border-amber-400/50 bg-amber-400/10 text-amber-200"
                : "border-line bg-navy-850 text-ink-300",
            )}>
          {rule.authority} {rule.limit} · {rule.outcome}
        </li>
      ))}
    </ul>
  );
}

/** The distinction the product turns on, kept visually separate. */
function Triad() {
  // Run 1's own three fields, shortened but not softened. The judgement is the
  // run's: a 4-point breach of one rulebook that another rulebook clears.
  const rows = [
    { label: "Fact", value: "London 31% of balance" },
    { label: "Rule", value: "Securitisation criteria ≤27%" },
    { label: "Judgement", value: "Breaches by 4pp — warehouse limit still passes" },
  ];
  return (
    <dl className="mt-1 grid grid-cols-[4.5rem_1fr] gap-x-3 text-left">
      {rows.map((row) => (
        <div key={row.label} className="contents">
          <dt className="text-[9px] font-semibold uppercase tracking-wider leading-5 text-ink-500 sm:text-[10px]">
            {row.label}
          </dt>
          <dd className="text-[12px] leading-5 text-ink-100 sm:text-[13px]">{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * The text equivalent — for screen readers, for reduced-motion visitors, and
 * for anyone who would rather read than watch. It is also the honest claim,
 * stated in words a viewer who looks away mid-loop cannot misread.
 */
function Caption() {
  return (
    <figcaption className="border-t border-line-soft px-4 py-3 text-[12px] leading-relaxed text-ink-400 sm:px-6">
      An enterprise agent discovers the Trakt Securitisation Readiness Agent,
      delegates the objective “assess this portfolio for securitisation
      readiness”, and receives a structured, evidence-backed assessment. Trakt
      chooses what to investigate. Every figure is from one recorded run against
      a synthetic portfolio — 30 governed queries, six material findings, eight
      diligence items and four declared gaps — including the 31% regional
      concentration that passes a warehouse limit, breaches a securitisation
      criterion and flags an internal threshold, and the high-LTV cohort whose
      loans carry the stale valuations.
    </figcaption>
  );
}
