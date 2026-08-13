/**
 * Landing-page agent-to-agent demo — 45 seconds for the Agent-to-agent
 * section: a client enterprise agent delegates a business objective, Trakt
 * decides how to investigate it, and an evidence-backed conclusion returns.
 *
 * Everything on screen is from ONE recorded run
 * ---------------------------------------------
 * Sprint 4 ran five delegated assessments end to end. This is run 1, and only
 * run 1: thirty governed calls in their recorded order, with their recorded
 * arguments, statuses, error codes and per-call durations, and the findings the
 * specialist actually produced from them. Averaging the five was rejected — it
 * would describe a portfolio no recorded run assessed.
 *
 * Written for the person who would buy it
 * ---------------------------------------
 * The trace says what each call ASKED, not which function it called. Thirty
 * rows of `stratify` and `cohort_comparison` proved the system was real to an
 * engineer and told a lending analyst nothing; "Split by account status" and
 * "Where the arrears sit" tell them what the specialist wanted to know. The
 * tool's own name and arguments survive on the result card beside the trace,
 * which is where the technical register belongs — one glance for credibility,
 * not thirty rows of it.
 *
 * The same reasoning governs the counters. "Governed compute · 3,405 ms" is
 * three translations away from anything an analyst wants: the unit, the word
 * and the concept. What matters is that the figures were calculated rather
 * than estimated, and that the calculation is a rounding error against the
 * thinking. So: CALCULATION TIME 3.4 sec against AGENT REASONING 2 min 17 sec.
 *
 * Refusals are amber, not rose
 * ----------------------------
 * Two calls were rejected with TOOL_INPUT_INVALID and the specialist adjusted.
 * The first cut painted those rows the same rose as the securitisation breach,
 * which said "two things went wrong" to anyone reading colour — and left a
 * viewer asking why one `cohort_comparison` is red and the next is green. They
 * are amber now, tagged `refused` and `adjusted` in the row itself, because
 * the governed layer declining a request it cannot honour is a boundary held,
 * not a fault. Rose is left to mean exactly one thing: the breach.
 *
 * What the run does and does not license
 * --------------------------------------
 * The transcript carries per-call `elapsed_ms` and order, but no wall-clock
 * timestamp per call, so this composition is sequence-driven and never claims a
 * schedule. The call durations are real; the gaps between calls are not, and
 * are compressed. Calculation time is the real running sum. Agent reasoning is
 * recorded in aggregate only and never per call, so it appears once, at the
 * end, as a run total rather than as a counter that climbs.
 *
 * Two figures come from the artifact rather than the transcript: the 31% and
 * the three rulebook verdicts. The governed digests are deliberately lean —
 * `concentration` returns `groups_count`, not the percentage — so the number
 * and its three judgements are quoted from the specialist's own finding, which
 * cites the call that produced them.
 *
 * Storyboard (30 fps, 1350 frames):
 *
 *   0.0 –  3.3 s  Two agents, centred. A line grows from the client to Trakt
 *                 and turns green on connection.
 *   3.3 –  5.2 s  The pair lifts to the top; the governed layer opens beneath.
 *   5.2 – 16.0 s  Orientation and the two long calls; `evaluate_rule_packs`
 *                 holds for its real 1,126 ms.
 *  16.0 – 20.6 s  Two requests are refused and the specialist adjusts.
 *  21.4 – 24.5 s  One call isolates: London, 31% of balance.
 *  24.5 – 29.6 s  Three governing documents answer it differently — pass,
 *                 flag, breach — resolving one at a time.
 *  30.0 – 33.0 s  The trace resumes: seven more checks after the breach.
 *  33.0 – 36.8 s  The specialist goes further of its own accord.
 *  36.8 – 39.4 s  The last five checks complete; the two times stand together.
 *  39.4 – 45.0 s  A line grows back to the client and the verdict holds.
 *
 * No portfolio total appears. The assessed book is not the one the hero and
 * query demo carry, and showing either figure would misattribute one to the
 * other; every figure here is a percentage or a count instead.
 */

import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

import T from "./theme";

export const A2A_DEMO = {
  id: "LandingA2ADemo",
  fps: 30,
  width: 1200,
  height: 960,
  duration: 1350,
} as const;

/* ------------------------------------------------------------------ */
/* The recorded run                                                    */
/* ------------------------------------------------------------------ */

type Call = {
  seq: number;
  /** The governed tool, as recorded. Shown on the result card, not the row. */
  tool: string;
  /** What that call asked, in the words of someone reading a loan book. */
  asks: string;
  /** Measured, from the transcript. Drives how long a row holds working. */
  ms: number;
  /** The governed layer rejected the arguments (TOOL_INPUT_INVALID). */
  refused?: true;
  /** The specialist's next attempt after a refusal. */
  adjusted?: true;
};

/**
 * All thirty governed calls, in the order the specialist made them.
 *
 * Every `asks` line is a reading of that call's own recorded arguments — the
 * `dimension`, `filters`, `metric` and `months` it was given — not a
 * decoration. Where two rows read alike, the arguments differed: 13 splits by
 * `account_status` where 9 asked for `days_past_due`, and 24 and 25 are the
 * same two tools again with filters attached.
 */
const CALLS: readonly Call[] = [
  { seq: 1, tool: "portfolio_capabilities", asks: "What can be measured", ms: 71.32 },
  { seq: 2, tool: "portfolio_summary", asks: "Portfolio composition", ms: 15.99 },
  { seq: 3, tool: "readiness_framework", asks: "Which tests apply", ms: 57.44 },
  { seq: 4, tool: "data_completeness", asks: "Data completeness", ms: 3.16 },
  { seq: 5, tool: "evaluate_rule_packs", asks: "Securitisation criteria", ms: 1125.67 },
  { seq: 6, tool: "evaluate_covenants", asks: "Facility covenants", ms: 36.14 },
  { seq: 7, tool: "regulatory_readiness", asks: "ESMA reporting readiness", ms: 4.35 },
  { seq: 8, tool: "valuation_age_profile", asks: "Age of the valuations", ms: 182.05 },
  { seq: 9, tool: "stratify", asks: "Split by arrears bucket", ms: 0.8, refused: true },
  { seq: 10, tool: "portfolio_history", asks: "Twelve months of history", ms: 1134.45 },
  { seq: 11, tool: "cohort_comparison", asks: "Compare high-LTV loans", ms: 0.66, refused: true },
  { seq: 12, tool: "cohort_comparison", asks: "Loans above 80% LTV", ms: 204.29, adjusted: true },
  { seq: 13, tool: "stratify", asks: "Split by account status", ms: 11.81, adjusted: true },
  { seq: 14, tool: "transition_analysis", asks: "How loans changed status", ms: 11.97 },
  { seq: 15, tool: "rank_loans", asks: "Highest-LTV loans", ms: 3.92 },
  { seq: 16, tool: "concentration", asks: "Concentration by region", ms: 6.98 },
  { seq: 17, tool: "readiness_metrics", asks: "Largest loan · top-20 share", ms: 5.68 },
  { seq: 18, tool: "stratify", asks: "Fixed versus variable rate", ms: 4.6 },
  { seq: 19, tool: "stratify", asks: "LTV by region", ms: 7.12 },
  { seq: 20, tool: "default_analysis", asks: "Default rate", ms: 18.47 },
  { seq: 21, tool: "prepayment_analysis", asks: "Prepayment rate", ms: 3.59 },
  { seq: 22, tool: "list_validation_exceptions", asks: "Data validation exceptions", ms: 0.2 },
  { seq: 23, tool: "contractual_analytics", asks: "Weighted-average life", ms: 65.23 },
  { seq: 24, tool: "concentration", asks: "Where the arrears sit", ms: 7.63 },
  { seq: 25, tool: "stratify", asks: "Where the high-LTV sits", ms: 5.71 },
  { seq: 26, tool: "period_change", asks: "Change since last period", ms: 383.02 },
  { seq: 27, tool: "get_loans", asks: "Loan-level detail", ms: 2.54 },
  { seq: 28, tool: "explain_values", asks: "Where these values came from", ms: 24.39 },
  { seq: 29, tool: "rank_loans", asks: "Largest exposures", ms: 5.04 },
  { seq: 30, tool: "loss_analysis", asks: "Loss severity", ms: 1.12 },
];

/** The run's reasoning time, recorded in aggregate only — never per call. */
const REASONING_S = 137;

/**
 * Time in units a person reads without converting.
 *
 * The transcript is in milliseconds and the first cut printed milliseconds,
 * which is the engineer's unit. The point of the pair of counters is a
 * comparison made without arithmetic: a few seconds of calculation inside a
 * couple of minutes of thinking.
 */
const inSeconds = (ms: number): string => `${(ms / 1000).toFixed(1)} sec`;

const inMinutes = (s: number): string =>
  s < 60 ? `${s} sec` : `${Math.floor(s / 60)} min ${s % 60} sec`;

/**
 * Calculation time after each call, as the real running sum.
 *
 * Not a share of the total by call count: two calls — `evaluate_rule_packs` at
 * 1,126 ms and `portfolio_history` at 1,134 ms — spend two thirds of the whole
 * budget between them, so a linear counter would read 1.8 sec at call 16 where
 * the run actually recorded 2.9. The staircase is the interesting shape and it
 * is free to show correctly.
 */
const CUMULATIVE_MS: ReadonlyMap<number, number> = new Map(
  CALLS.map((call, i) => [
    call.seq,
    Math.round(CALLS.slice(0, i + 1).reduce((sum, c) => sum + c.ms, 0)),
  ]),
);

/**
 * What a call returned, where the run recorded something worth showing.
 *
 * `value` says what came back in plain terms; `technical` keeps the recorded
 * artefact — the error code, the method identifier — underneath it, so the
 * plain reading never replaces the evidence for it.
 */
type Detail = {
  seq: number;
  /** The recorded arguments, in the tool's own vocabulary. */
  argument?: string;
  value: string;
  technical?: string;
  tone: "mint" | "amber" | "rose" | "neutral";
};

const DETAILS: readonly Detail[] = [
  { seq: 1, value: "27 of 28 measures available", tone: "neutral" },
  { seq: 5, value: "Internal flags kept separate from breaches", tone: "neutral" },
  {
    seq: 7,
    argument: "regime: ESMA_Annex2",
    value: "14 of 18 required fields missing",
    tone: "amber",
  },
  { seq: 8, value: "12% of the book on valuations over 5 years old", tone: "amber" },
  {
    seq: 9,
    argument: "dimension: days_past_due",
    value: "Refused — no such split exists",
    technical: "TOOL_INPUT_INVALID",
    tone: "amber",
  },
  {
    seq: 11,
    argument: "measures: [3]",
    value: "Refused — those measures aren't offered",
    technical: "TOOL_INPUT_INVALID",
    tone: "amber",
  },
  { seq: 12, argument: "measures removed", value: "Asked again · answered", tone: "mint" },
  {
    seq: 13,
    argument: "dimension: account_status",
    value: "Asked a different way · answered",
    tone: "mint",
  },
  { seq: 21, value: "6.6% a year repaid early", technical: "OBSERVED_CPR@v2", tone: "neutral" },
  {
    seq: 24,
    argument: "filter: account_status = Arrears",
    value: "All in one region",
    tone: "amber",
  },
  { seq: 25, argument: "filter: current_LTV = 92%", value: "All in one region", tone: "amber" },
  { seq: 26, value: "Needs two snapshots · only one exists", tone: "neutral" },
];

/** One number, three governing documents. From the finding's `rule` field. */
export const VERDICTS = [
  { authority: "Warehouse facility", limit: "≤ 35%", outcome: "Pass", tone: "mint" },
  { authority: "Trakt screening", limit: "> 25%", outcome: "Flag", tone: "amber" },
  { authority: "Securitisation criteria", limit: "≤ 27%", outcome: "Breach", tone: "rose" },
] as const;

/* ------------------------------------------------------------------ */
/* Timeline                                                            */
/* ------------------------------------------------------------------ */

const F = {
  /** The request line starts growing towards Trakt. */
  reach: 18,
  connected: 78,
  lift: 100,
  lifted: 148,
  panel: 156,
  working: 120,
  mcp: 150,
  concIsolate: 642,
  concValue: 678,
  verdict: [735, 790, 845] as const,
  resume: 900,
  drill: 990,
  drillValue: 1032,
  wrap: 1104,
  complete: 1164,
  ret: 1182,
  received: 1236,
  end: 1254,
} as const;

/** Where each call lights, by segment. Recorded order; compressed time. */
const SEGMENTS: readonly { seqs: readonly number[]; from: number; to: number }[] = [
  { seqs: [1, 2, 3, 4], from: 255, to: 390 },
  { seqs: [5, 6, 7], from: 390, to: 480 },
  /* 23 frames a call — the most room any stretch here gets, because a refusal
     and the adjustment after it are the beat that most needs reading. */
  { seqs: [8, 9, 10, 11, 12, 13], from: 480, to: 618 },
  { seqs: [14, 15], from: 618, to: 636 },
  { seqs: [16], from: 642, to: 654 },
  { seqs: [17, 18, 19, 20, 21, 22, 23], from: 900, to: 978 },
  { seqs: [24, 25], from: 1002, to: 1050 },
  { seqs: [26, 27, 28, 29, 30], from: 1104, to: 1164 },
];

const FIRES: ReadonlyMap<number, number> = new Map(
  SEGMENTS.flatMap(({ seqs, from, to }) =>
    seqs.map((seq, i): [number, number] => [
      seq,
      Math.round(from + ((to - from) * i) / seqs.length),
    ]),
  ),
);

const firesAt = (seq: number): number => FIRES.get(seq) ?? 0;

const executed = (frame: number): number =>
  CALLS.filter((c) => frame >= firesAt(c.seq)).length;

/* ------------------------------------------------------------------ */
/* Motion                                                             */
/* ------------------------------------------------------------------ */

/** The single entrance: opacity 0→1 with a 12px rise, no overshoot. */
const Rise: React.FC<{
  at?: number;
  style?: React.CSSProperties;
  children: React.ReactNode;
}> = ({ at = 0, style, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = spring({
    frame: frame - at,
    fps,
    config: T.motion.spring,
    durationInFrames: T.motion.enter,
  });
  return (
    <div
      style={{ opacity: p, transform: `translateY(${(1 - p) * T.motion.rise}px)`, ...style }}
    >
      {children}
    </div>
  );
};

const ramp = (frame: number, from: number, to: number): number =>
  interpolate(frame, [from, to], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

/** Cross-fade a panel in and out; both ends clamp, so nothing flickers. */
const Between: React.FC<{
  from: number;
  to: number;
  fade?: number;
  children: React.ReactNode;
}> = ({ from, to, fade = 10, children }) => {
  const frame = useCurrentFrame();
  if (frame < from - fade || frame > to + fade) return null;
  const opacity = Math.min(ramp(frame, from, from + fade), 1 - ramp(frame, to, to + fade));
  return <AbsoluteFill style={{ opacity }}>{children}</AbsoluteFill>;
};

/* ------------------------------------------------------------------ */
/* Primitives, mirroring the landing page's card language              */
/* ------------------------------------------------------------------ */

export const TONE_INK = {
  neutral: T.color.ink300,
  peri: T.color.peri,
  mint: T.color.mint,
  amber: T.color.amber,
  rose: T.color.rose,
} as const;

export const TONE_EDGE = {
  neutral: T.color.line,
  peri: "rgba(145,157,209,0.55)",
  mint: "rgba(54,194,168,0.45)",
  amber: "rgba(224,169,59,0.4)",
  rose: "rgba(224,96,122,0.45)",
} as const;

type Tone = keyof typeof TONE_INK;

const Chip: React.FC<{ tone?: Tone; children: React.ReactNode }> = ({
  tone = "neutral",
  children,
}) => (
  <span
    style={{
      display: "inline-block",
      border: `1.5px solid ${TONE_EDGE[tone]}`,
      borderRadius: T.radius.chip,
      padding: "5px 14px",
      fontSize: T.size.chip,
      fontWeight: 500,
      color: TONE_INK[tone],
      background: T.color.raised,
      whiteSpace: "nowrap",
    }}
  >
    {children}
  </span>
);

const Panel: React.FC<{ style?: React.CSSProperties; children: React.ReactNode }> = ({
  style,
  children,
}) => (
  <div
    style={{
      background: T.color.inset,
      border: `1.5px solid ${T.color.line}`,
      borderRadius: T.radius.panel,
      ...style,
    }}
  >
    {children}
  </div>
);

const Label: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({
  children,
  style,
}) => (
  <p
    style={{
      margin: 0,
      fontSize: T.size.label,
      letterSpacing: "0.08em",
      textTransform: "uppercase",
      color: T.color.ink500,
      fontWeight: 600,
      ...style,
    }}
  >
    {children}
  </p>
);

/* ------------------------------------------------------------------ */
/* The topology                                                        */
/* ------------------------------------------------------------------ */

const Node: React.FC<{ title: string; lit: boolean; width: number }> = ({
  title,
  lit,
  width,
}) => (
  <div
    style={{
      width,
      height: 84,
      padding: "0 24px",
      display: "flex",
      flexDirection: "column",
      justifyContent: "center",
      borderRadius: T.radius.panel,
      border: `1.5px solid ${lit ? "rgba(145,157,209,0.55)" : T.color.line}`,
      background: lit ? T.color.raised : T.color.inset,
      textAlign: "center",
    }}
  >
    <p
      style={{
        margin: 0,
        fontSize: T.size.node,
        fontWeight: 600,
        color: lit ? T.color.ink100 : T.color.ink300,
      }}
    >
      {title}
    </p>
  </div>
);

/**
 * A line that grows from one agent to the other, with a head on the leading
 * edge, turning green when it arrives.
 *
 * This replaces labelled chips sliding across the frame. A packet drifting
 * over open space is a cartoon of a network; a connection being made is what
 * actually happens, and it can carry its own label without the label having to
 * travel. It also fixes a collision that had no good answer: the objective is
 * wider than the gap between the two tiles, so anything carrying it across
 * that gap ended up over an agent's name.
 */
const Reaching: React.FC<{
  from: number;
  to: number;
  back?: boolean;
  top: number;
}> = ({ from, to, back = false, top }) => {
  const frame = useCurrentFrame();
  if (frame < from) return null;
  const p = ramp(frame, from, to);
  const arrived = frame >= to;
  const colour = arrived ? T.color.mint : T.color.periDeep;
  const head = {
    position: "absolute" as const,
    top: top - 4,
    width: 0,
    height: 0,
    borderTop: `5px solid transparent`,
    borderBottom: `5px solid transparent`,
  };
  return (
    <>
      <div
        style={{
          position: "absolute",
          top,
          [back ? "right" : "left"]: 0,
          width: `${p * 100}%`,
          height: 1.5,
          background: colour,
        }}
      />
      <span
        style={
          back
            ? { ...head, right: `calc(${p * 100}% - 2px)`, borderRight: `8px solid ${colour}` }
            : { ...head, left: `calc(${p * 100}% - 6px)`, borderLeft: `8px solid ${colour}` }
        }
      />
    </>
  );
};

/**
 * The connection between the two agents: what was asked going one way, what
 * came back going the other, each on its own line so neither has to move.
 */
const Connection: React.FC = () => {
  const frame = useCurrentFrame();
  const returning = frame >= F.ret;
  return (
    <div style={{ position: "relative", flex: 1, height: 84 }}>
      <Reaching from={F.reach} to={F.connected} top={34} />
      {returning ? <Reaching from={F.ret} to={F.ret + 54} back top={62} /> : null}

      <span
        style={{
          position: "absolute",
          top: 26,
          left: "50%",
          transform: "translateX(-50%)",
          padding: "3px 12px",
          borderRadius: T.radius.chip,
          border: `1.5px solid ${T.color.line}`,
          background: T.color.surface,
          fontSize: T.size.seq,
          letterSpacing: "0.1em",
          fontWeight: 600,
          color: frame >= F.connected ? T.color.mint : T.color.ink500,
        }}
      >
        A2A
      </span>

      {/* Labels sit still above and below their own line. */}
      <div style={{ position: "absolute", top: 4, left: 0, right: 0, textAlign: "center" }}>
        <span style={{ fontSize: T.size.chip, color: T.color.ink300 }}>
          Assess for securitisation readiness
        </span>
      </div>
      {returning ? (
        <Rise at={F.ret + 40}>
          <div style={{ position: "absolute", top: 68, left: 0, right: 0, textAlign: "center" }}>
            <span style={{ fontSize: T.size.chip, color: T.color.mint }}>
              Assessment · evidence attached
            </span>
          </div>
        </Rise>
      ) : null}
    </div>
  );
};

/** submitted → working → completed, as the run recorded it. */
const TaskState: React.FC = () => {
  const frame = useCurrentFrame();
  if (frame < F.working) return null;
  const done = frame >= F.received;
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "flex-end",
        gap: 10,
        marginTop: 12,
        alignItems: "center",
      }}
    >
      <span style={{ fontSize: T.size.seq, color: T.color.ink500 }}>submitted</span>
      <span style={{ fontSize: T.size.seq, color: T.color.ink500 }}>→</span>
      <span
        style={{
          fontSize: T.size.seq,
          color: done ? T.color.ink500 : T.color.peri,
          fontWeight: 600,
        }}
      >
        working
      </span>
      {done ? (
        <>
          <span style={{ fontSize: T.size.seq, color: T.color.ink500 }}>→</span>
          <span style={{ fontSize: T.size.seq, color: T.color.mint, fontWeight: 600 }}>
            completed
          </span>
        </>
      ) : null}
    </div>
  );
};

const Topology: React.FC = () => {
  const frame = useCurrentFrame();
  const traktLit = frame >= F.connected;
  return (
    <div style={{ padding: "20px 32px 0" }}>
      <div style={{ display: "flex", alignItems: "flex-start" }}>
        <Node title="Client enterprise agent" lit width={392} />
        <Connection />
        <Node title="Trakt Securitisation Readiness Agent" lit={traktLit} width={392} />
      </div>
      <TaskState />
    </div>
  );
};

/** The descent into the governed layer. Drawn, not faded — it is a path. */
const McpDescent: React.FC = () => {
  const frame = useCurrentFrame();
  const p = ramp(frame, F.mcp, F.mcp + 16);
  return (
    <div style={{ position: "relative", height: 46, marginTop: 4 }}>
      {/* Under the Trakt node's centre: 32px padding + 744 + 196 of 1200. */}
      <div
        style={{
          position: "absolute",
          left: "81%",
          top: 0,
          width: 1.5,
          height: p * 46,
          background: T.color.periDeep,
        }}
      />
      <span
        style={{
          position: "absolute",
          left: "81%",
          top: 12,
          transform: "translateX(-50%)",
          padding: "3px 12px",
          borderRadius: T.radius.chip,
          border: `1.5px solid ${T.color.line}`,
          background: T.color.surface,
          fontSize: T.size.seq,
          letterSpacing: "0.1em",
          fontWeight: 600,
          color: T.color.peri,
          opacity: p,
        }}
      >
        MCP
      </span>
    </div>
  );
};

/* ------------------------------------------------------------------ */
/* The governed layer                                                  */
/* ------------------------------------------------------------------ */

/**
 * One check in the trace.
 *
 * A slot exists from the first frame; what it CONTAINS does not. The number is
 * there because thirty is the shape of the thing and the panel should be full
 * from the start, but the question only appears when the specialist actually
 * asks it — the whole point being that nobody, including the caller, knew what
 * those thirty checks would be.
 */
const TraceRow: React.FC<{ call: Call }> = ({ call }) => {
  const frame = useCurrentFrame();
  const at = firesAt(call.seq);
  if (frame < at) {
    return (
      <div style={{ height: 34, display: "flex", alignItems: "center", gap: 10, opacity: 0.4 }}>
        <Seq seq={call.seq} />
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: T.radius.chip,
            border: `1.5px solid ${T.color.line}`,
            flexShrink: 0,
          }}
        />
        <span style={{ width: 132, height: 1.5, background: T.color.lineSoft }} />
      </div>
    );
  }
  /* The row holds "working" for the call's own measured duration, scaled so a
     1,126 ms call visibly outlasts a 0.2 ms one without stalling the edit. */
  const holdFrames = Math.min(18, Math.max(3, Math.round(call.ms / 70) + 3));
  const settled = frame >= at + holdFrames;
  const tone: Tone = !settled
    ? "peri"
    : call.refused
      ? "amber"
      : call.adjusted
        ? "mint"
        : "mint";
  const tag = settled ? (call.refused ? "refused" : call.adjusted ? "adjusted" : null) : null;
  return (
    <div
      style={{
        height: 34,
        display: "flex",
        alignItems: "center",
        gap: 10,
        opacity: interpolate(frame, [at, at + 6], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        }),
      }}
    >
      <Seq seq={call.seq} />
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: T.radius.chip,
          background: TONE_INK[tone],
          flexShrink: 0,
        }}
      />
      {/* One line, always. A label that wraps is a row 34px taller than its
          neighbour, which breaks the two-column grid for every row below it —
          "Compare the high-LTV loans · refused" did exactly that. */}
      <span
        style={{
          fontSize: T.size.trace,
          whiteSpace: "nowrap",
          color: call.refused && settled ? T.color.amber : T.color.ink200,
        }}
      >
        {call.asks}
      </span>
      {tag ? (
        <span
          style={{
            fontSize: T.size.seq,
            whiteSpace: "nowrap",
            color: call.refused ? T.color.amber : T.color.mint,
            opacity: 0.85,
          }}
        >
          · {tag}
        </span>
      ) : null}
    </div>
  );
};

const Seq: React.FC<{ seq: number }> = ({ seq }) => (
  <span
    style={{
      width: 24,
      textAlign: "right",
      fontSize: T.size.seq,
      color: T.color.ink500,
      fontVariantNumeric: "tabular-nums",
      flexShrink: 0,
    }}
  >
    {seq}
  </span>
);

/**
 * The trace, two columns, filling row by row.
 *
 * Row-major, not column-major: filling the first column before touching the
 * second left half the panel empty for the first twenty seconds. Alternating
 * puts checks 1 and 2 side by side, so the block grows downward from the start
 * and the frame is used the whole way through.
 */
const TraceGrid: React.FC = () => (
  <div style={{ display: "flex", gap: 24, flex: 1 }}>
    {[0, 1].map((column) => (
      <div key={column} style={{ flex: 1 }}>
        {CALLS.filter((_, i) => i % 2 === column).map((call) => (
          <TraceRow key={call.seq} call={call} />
        ))}
      </div>
    ))}
  </div>
);

/**
 * The most recent check that returned something worth reading.
 *
 * One card that changes rather than twelve that accumulate — and the one place
 * the technical register belongs: the tool's real name, its real arguments and
 * its real error code, beside a plain reading of what came back.
 */
const DetailCard: React.FC = () => {
  const frame = useCurrentFrame();
  const current = DETAILS.filter((d) => frame >= firesAt(d.seq) + 4).slice(-1)[0];
  if (!current) return null;
  const call = CALLS.find((c) => c.seq === current.seq);
  return (
    <Panel style={{ width: 344, padding: "22px 24px", alignSelf: "center" }}>
      <Label>Result</Label>
      <p style={{ margin: "12px 0 0", fontSize: T.size.trace, color: T.color.ink200 }}>
        {call?.tool}
      </p>
      {current.argument ? (
        <p style={{ margin: "5px 0 0", fontSize: T.size.seq, color: T.color.ink500 }}>
          {current.argument}
        </p>
      ) : null}
      <p
        style={{
          margin: "14px 0 0",
          fontSize: T.size.value,
          fontWeight: 600,
          lineHeight: 1.25,
          color: TONE_INK[current.tone],
        }}
      >
        {current.value}
      </p>
      {current.technical ? (
        <p style={{ margin: "10px 0 0", fontSize: T.size.seq, color: T.color.ink500 }}>
          {current.technical}
        </p>
      ) : null}
    </Panel>
  );
};

/**
 * The two figures an analyst would actually ask about.
 *
 * Not "governed compute" and "model time": the first is a unit and a word they
 * would have to translate, and the second sounds like an infrastructure bill.
 * What the pair establishes is that the numbers were calculated rather than
 * guessed, and that the calculation is a rounding error beside the thinking.
 * Reasoning time arrives once, at the end, because the run records it in
 * aggregate and never per call — a counter climbing beside the checks would be
 * an invention.
 */
const Meters: React.FC = () => {
  const frame = useCurrentFrame();
  const last = CALLS.filter((c) => frame >= firesAt(c.seq)).slice(-1)[0];
  const calculation = last ? CUMULATIVE_MS.get(last.seq) ?? 0 : 0;
  return (
    <div style={{ display: "flex", gap: 34, alignItems: "flex-start" }}>
      <Meter label="Checks run" value={`${executed(frame)}`} />
      <Meter label="Calculation time" value={inSeconds(calculation)} />
      {frame >= F.wrap ? (
        <Rise at={F.complete}>
          <Meter label="Agent reasoning" value={inMinutes(REASONING_S)} tone="peri" />
        </Rise>
      ) : null}
    </div>
  );
};

const Meter: React.FC<{ label: string; value: string; tone?: Tone }> = ({
  label,
  value,
  tone = "neutral",
}) => (
  <div style={{ textAlign: "right" }}>
    <Label>{label}</Label>
    <p
      style={{
        margin: "6px 0 0",
        fontSize: T.size.meter,
        fontWeight: 600,
        color: tone === "peri" ? T.color.peri : T.color.ink100,
        fontVariantNumeric: "tabular-nums",
      }}
    >
      {value}
    </p>
  </div>
);

/* ------------------------------------------------------------------ */
/* The climax — one number, three governing documents                  */
/* ------------------------------------------------------------------ */

const VerdictRow: React.FC<{ verdict: (typeof VERDICTS)[number]; at: number }> = ({
  verdict,
  at,
}) => {
  const frame = useCurrentFrame();
  const shown = frame >= at;
  return (
    <Rise at={at}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 20,
          padding: "16px 22px",
          borderRadius: T.radius.row,
          border: `1.5px solid ${shown ? TONE_EDGE[verdict.tone] : T.color.lineSoft}`,
          background: T.color.raised,
        }}
      >
        <p style={{ margin: 0, fontSize: T.size.row, color: T.color.ink200 }}>
          {verdict.authority}
        </p>
        <div style={{ display: "flex", alignItems: "baseline", gap: 20 }}>
          <span
            style={{
              fontSize: T.size.row,
              color: T.color.ink400,
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {verdict.limit}
          </span>
          <span
            style={{
              fontSize: T.size.value,
              fontWeight: 600,
              color: TONE_INK[verdict.tone],
              minWidth: 96,
              textAlign: "right",
            }}
          >
            {verdict.outcome}
          </span>
        </div>
      </div>
    </Rise>
  );
};

/**
 * The concentration card — the demo's whole argument in one panel.
 *
 * Exported because the poster is drawn from it. `at` rebases the internal
 * timing so a still can render it settled.
 */
export const ConcentrationCard: React.FC<{ at?: number }> = ({ at }) => {
  const frame = useCurrentFrame();
  const base = at ?? F.concValue;
  const verdictAt = at === undefined ? F.verdict : ([at + 12, at + 24, at + 36] as const);
  const digits = ramp(frame, base, base + 20);
  return (
    <Panel style={{ width: 900, padding: "30px 40px 34px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <Label>Concentration · geographic region</Label>
        <span style={{ fontSize: T.size.chip, color: T.color.ink400 }}>check 16</span>
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 22,
          marginTop: 14,
          paddingBottom: 22,
          borderBottom: `1.5px solid ${T.color.lineSoft}`,
        }}
      >
        <span
          style={{
            fontSize: T.size.figure,
            fontWeight: 600,
            color: T.color.ink100,
            fontVariantNumeric: "tabular-nums",
            opacity: digits,
          }}
        >
          31%
        </span>
        <span style={{ fontSize: T.size.row, color: T.color.ink300, opacity: digits }}>
          London · of balance · 124 loans
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 22 }}>
        {VERDICTS.map((verdict, i) => (
          <VerdictRow key={verdict.authority} verdict={verdict} at={verdictAt[i] ?? 0} />
        ))}
      </div>
    </Panel>
  );
};

/** What the specialist did next, unprompted: the same region, twice over. */
const DrillCard: React.FC = () => (
  <Panel style={{ width: 900, padding: "30px 40px 34px" }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
      <Label>Followed up · nobody asked for this</Label>
      <span style={{ fontSize: T.size.chip, color: T.color.ink400 }}>checks 24 · 25</span>
    </div>
    <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 22 }}>
      {[
        { q: "Where the arrears sit", v: "70.8% vs 6.8%", t: "rose" },
        { q: "Where the high-LTV sits", v: "The same region", t: "amber" },
      ].map((row, i) => (
        <Rise key={row.q} at={F.drillValue + i * 30}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 20,
              padding: "16px 22px",
              borderRadius: T.radius.row,
              border: `1.5px solid ${TONE_EDGE[row.t as Tone]}`,
              background: T.color.raised,
            }}
          >
            <p style={{ margin: 0, fontSize: T.size.row, color: T.color.ink200 }}>{row.q}</p>
            <span
              style={{
                fontSize: T.size.value,
                fontWeight: 600,
                color: TONE_INK[row.t as Tone],
                whiteSpace: "nowrap",
              }}
            >
              {row.v}
            </span>
          </div>
        </Rise>
      ))}
    </div>
  </Panel>
);

/** The artifact the client agent received. */
const AssessmentCard: React.FC = () => (
  // 960, not 900: at 900 the four chips wrapped and left "Evidence attached"
  // stranded on a line of its own.
  <Panel style={{ width: 960, padding: "34px 40px 38px" }}>
    <Label>Assessment received</Label>
    <p
      style={{
        margin: "16px 0 0",
        fontSize: T.size.keyline,
        fontWeight: 600,
        color: T.color.rose,
      }}
    >
      Material remediation required
    </p>
    <div style={{ display: "flex", gap: 14, marginTop: 26, flexWrap: "wrap" }}>
      <Chip tone="neutral">6 material findings</Chip>
      <Chip tone="neutral">8 for further diligence</Chip>
      <Chip tone="amber">4 it could not assess</Chip>
      <Chip tone="mint">Evidence attached</Chip>
    </div>
  </Panel>
);

/* ------------------------------------------------------------------ */
/* Chrome + assembly                                                   */
/* ------------------------------------------------------------------ */

export const Chrome: React.FC = () => (
  <div
    style={{
      position: "absolute",
      top: 0,
      left: 0,
      right: 0,
      height: 68,
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "0 32px",
      borderBottom: `1.5px solid ${T.color.line}`,
    }}
  >
    <p style={{ margin: 0, fontSize: T.size.chrome, fontWeight: 600, color: T.color.ink300 }}>
      Trakt · Agent-to-agent
    </p>
    <p style={{ margin: 0, fontSize: T.size.chip, fontWeight: 500, color: T.color.amber }}>
      Illustrative · synthetic data
    </p>
  </div>
);

const Centred: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div
    style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}
  >
    {children}
  </div>
);

const TraceView: React.FC = () => (
  <div style={{ display: "flex", gap: 24, height: "100%" }}>
    <TraceGrid />
    <DetailCard />
  </div>
);

/** The governed panel: the trace, or whichever card has taken it over. */
const GovernedLayer: React.FC = () => (
  <Rise at={F.panel} style={{ padding: "0 32px" }}>
    <Panel style={{ height: 640, padding: "24px 28px", display: "flex", flexDirection: "column" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          paddingBottom: 18,
          borderBottom: `1.5px solid ${T.color.lineSoft}`,
        }}
      >
        <Label style={{ paddingTop: 6 }}>Governed portfolio intelligence</Label>
        <Meters />
      </div>
      <div style={{ position: "relative", flex: 1, marginTop: 18 }}>
        {/* The trace is the panel's resting state; the cards take it over and
            hand it back. It stays up through the return, because the completed
            counters are the point of that moment. */}
        <Between from={F.panel} to={F.concIsolate}>
          <TraceView />
        </Between>
        <Between from={F.resume} to={F.drill - 12}>
          <TraceView />
        </Between>
        <Between from={F.wrap} to={F.end - 12}>
          <TraceView />
        </Between>
        <Between from={F.concIsolate} to={F.resume - 12}>
          <Centred>
            <ConcentrationCard />
          </Centred>
        </Between>
        <Between from={F.drill} to={F.wrap - 12}>
          <Centred>
            <DrillCard />
          </Centred>
        </Between>
        <Between from={F.end} to={A2A_DEMO.duration}>
          <Centred>
            <AssessmentCard />
          </Centred>
        </Between>
      </div>
    </Panel>
  </Rise>
);

/**
 * The opening.
 *
 * Two agents, centred and alone, while the connection is made — then the pair
 * lifts to the top and the governed layer opens beneath it. Starting with the
 * finished three-tier architecture asked the viewer to read a diagram before
 * anything had happened; starting with two parties and a line being drawn asks
 * them to watch one thing.
 */
const LIFT_PX = 336;

const A2ADemo: React.FC = () => {
  const frame = useCurrentFrame();
  const lift = 1 - ramp(frame, F.lift, F.lifted);
  return (
    <AbsoluteFill style={{ background: T.color.surface, fontFamily: T.family }}>
      <style dangerouslySetInnerHTML={{ __html: T.fontFaceCss() }} />
      <Chrome />
      <AbsoluteFill style={{ top: 68 }}>
        <div style={{ transform: `translateY(${lift * LIFT_PX}px)` }}>
          <Topology />
        </div>
        <McpDescent />
        <GovernedLayer />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

export default A2ADemo;
