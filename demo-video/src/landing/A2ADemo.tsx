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
  duration: 1470,
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
 *
 * They are all noun phrases, and that is a rule rather than a habit. A list
 * mixing "Highest-LTV loans" with "Compare high-LTV loans" reads as two
 * different kinds of thing — one a subject, one an instruction — and the eye
 * catches on it every time. Thirty labels in one grammatical shape scan as a
 * single list of what was examined.
 */
const CALLS: readonly Call[] = [
  { seq: 1, tool: "portfolio_capabilities", asks: "Available measures", ms: 71.32 },
  { seq: 2, tool: "portfolio_summary", asks: "Portfolio composition", ms: 15.99 },
  { seq: 3, tool: "readiness_framework", asks: "Applicable readiness tests", ms: 57.44 },
  { seq: 4, tool: "data_completeness", asks: "Data completeness", ms: 3.16 },
  { seq: 5, tool: "evaluate_rule_packs", asks: "Securitisation criteria", ms: 1125.67 },
  { seq: 6, tool: "evaluate_covenants", asks: "Facility covenants", ms: 36.14 },
  { seq: 7, tool: "regulatory_readiness", asks: "ESMA reporting readiness", ms: 4.35 },
  { seq: 8, tool: "valuation_age_profile", asks: "Valuation age", ms: 182.05 },
  { seq: 9, tool: "stratify", asks: "Arrears bands", ms: 0.8, refused: true },
  { seq: 10, tool: "portfolio_history", asks: "Twelve-month history", ms: 1134.45 },
  { seq: 11, tool: "cohort_comparison", asks: "High-LTV cohort", ms: 0.66, refused: true },
  { seq: 12, tool: "cohort_comparison", asks: "Loans above 80% LTV", ms: 204.29, adjusted: true },
  { seq: 13, tool: "stratify", asks: "Account status split", ms: 11.81, adjusted: true },
  { seq: 14, tool: "transition_analysis", asks: "Status transitions", ms: 11.97 },
  { seq: 15, tool: "rank_loans", asks: "Highest-LTV loans", ms: 3.92 },
  { seq: 16, tool: "concentration", asks: "Regional concentration", ms: 6.98 },
  { seq: 17, tool: "readiness_metrics", asks: "Largest loan · top-20 share", ms: 5.68 },
  { seq: 18, tool: "stratify", asks: "Rate type split", ms: 4.6 },
  { seq: 19, tool: "stratify", asks: "LTV by region", ms: 7.12 },
  { seq: 20, tool: "default_analysis", asks: "Default rate", ms: 18.47 },
  { seq: 21, tool: "prepayment_analysis", asks: "Prepayment rate", ms: 3.59 },
  { seq: 22, tool: "list_validation_exceptions", asks: "Validation exceptions", ms: 0.2 },
  { seq: 23, tool: "contractual_analytics", asks: "Weighted-average life", ms: 65.23 },
  { seq: 24, tool: "concentration", asks: "Arrears by region", ms: 7.63 },
  { seq: 25, tool: "stratify", asks: "High-LTV by region", ms: 5.71 },
  { seq: 26, tool: "period_change", asks: "Change since last period", ms: 383.02 },
  { seq: 27, tool: "get_loans", asks: "Loan-level detail", ms: 2.54 },
  { seq: 28, tool: "explain_values", asks: "Value provenance", ms: 24.39 },
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
  { seq: 26, value: "Needs two snapshots · only one exists", tone: "amber" },
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
  mcp: 150,
  /** All thirty done; the reasoning total joins the calculation total. */
  complete: 939,
  ret: 963,
  report: 1026,
  /**
   * The six findings, in the artifact's own order — but the third opens where
   * it sits. Listing all six and then reaching back up to expand one made the
   * reader lose their place; a finding that unfolds in sequence reads as part
   * of the list rather than as an annotation on it.
   */
  finding: [1050, 1074, 1098, 1284, 1308, 1332] as const,
  fact: 1128,
  rule: 1152,
  /** One number, three governing documents, resolving one at a time. */
  verdict: [1176, 1200, 1224] as const,
  judgement: 1254,
  diligence: 1362,
  gaps: 1386,
} as const;

/** Where each call lights, by segment. Recorded order; compressed time. */
const SEGMENTS: readonly { seqs: readonly number[]; from: number; to: number }[] = [
  { seqs: [1, 2, 3, 4], from: 255, to: 375 },
  { seqs: [5, 6, 7], from: 375, to: 465 },
  /* 23 frames a check — the most room any stretch here gets, because a refusal
     and the adjustment after it are the beat that most needs reading. */
  { seqs: [8, 9, 10, 11, 12, 13], from: 465, to: 603 },
  { seqs: [14, 15], from: 603, to: 639 },
  { seqs: [16], from: 639, to: 663 },
  { seqs: [17, 18, 19, 20, 21, 22, 23], from: 663, to: 789 },
  { seqs: [24, 25], from: 789, to: 837 },
  { seqs: [26, 27, 28, 29, 30], from: 837, to: 927 },
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
      {/* Offset by the head's own 8px, so the TIP lands on the tile's edge
          rather than the head's box. At -2/-6 the arrow finished six pixels
          inside the enterprise agent instead of touching it. */}
      <span
        style={
          back
            ? { ...head, right: `calc(${p * 100}% - 8px)`, borderRight: `8px solid ${colour}` }
            : { ...head, left: `calc(${p * 100}% - 8px)`, borderLeft: `8px solid ${colour}` }
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
  /* Dead centre of the 84px tiles beside it, less half the rule's own weight.
     The line used to sit at 34 to leave room for a second line below it for
     the reply; against a tile whose text is centred, eleven pixels high reads
     as a mistake. There is one line now, and the arrow reverses along it. */
  const centre = 41;
  return (
    <div style={{ position: "relative", flex: 1, height: 84 }}>
      {returning ? (
        <Reaching from={F.ret} to={F.ret + 54} back top={centre} />
      ) : (
        <Reaching from={F.reach} to={F.connected} top={centre} />
      )}

      <span
        style={{
          position: "absolute",
          top: centre - 15,
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

      {/* The label sits still above the line and changes with the direction. */}
      <div style={{ position: "absolute", top: 6, left: 0, right: 0, textAlign: "center" }}>
        <span
          style={{
            fontSize: T.size.chip,
            color: returning ? T.color.mint : T.color.ink300,
          }}
        >
          {returning ? "Assessment · evidence attached" : "Assess for securitisation readiness"}
        </span>
      </div>
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
 * The one colour scale on this surface.
 *
 * A row read mint while the result card beside it read amber — "ESMA
 * reporting readiness" ticked green with "14 of 18 required fields missing"
 * next to it. Two scales, one screen, opposite answers. The row now takes its
 * tone from what the check actually returned, so the dot, the label and the
 * card can never disagree: amber where the run recorded something needing
 * attention or refused the request, mint where it answered cleanly.
 */
const concernOf = (seq: number): Tone => {
  const call = CALLS.find((c) => c.seq === seq);
  if (call?.refused) return "amber";
  return DETAILS.find((d) => d.seq === seq)?.tone === "amber" ? "amber" : "mint";
};

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
  const tone: Tone = !settled ? "peri" : concernOf(call.seq);
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
          color: settled && concernOf(call.seq) === "amber" ? T.color.amber : T.color.ink200,
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
    <Panel
      style={{
        width: 344,
        padding: "22px 24px",
        /* Stretched, not centred on its own height. A card floating in the
           middle of a tall column with air above and below it looks like a
           tooltip that lost its anchor. */
        alignSelf: "stretch",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <Label>Result</Label>
      {/* Named exactly as the row that produced it. The card used to lead with
          the tool — `regulatory_readiness` beside a row reading "ESMA
          reporting readiness" — so the two never visibly belonged together. */}
      <p style={{ margin: "14px 0 0", fontSize: T.size.trace, color: T.color.ink300 }}>
        Check {current.seq} · {call?.asks}
      </p>
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
      {/* The technical record, kept and demoted: this is the evidence for the
          plain line above it, not a competing headline. */}
      <div
        style={{
          marginTop: "auto",
          paddingTop: 18,
          borderTop: `1.5px solid ${T.color.lineSoft}`,
        }}
      >
        <p style={{ margin: 0, fontSize: T.size.seq, color: T.color.ink400 }}>{call?.tool}</p>
        {current.argument ? (
          <p style={{ margin: "4px 0 0", fontSize: T.size.seq, color: T.color.ink500 }}>
            {current.argument}
          </p>
        ) : null}
        {current.technical ? (
          <p style={{ margin: "4px 0 0", fontSize: T.size.seq, color: T.color.ink500 }}>
            {current.technical}
          </p>
        ) : null}
      </div>
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
      {frame >= F.complete ? (
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
/* The assessment                                                      */
/* ------------------------------------------------------------------ */

/**
 * The six findings the specialist returned, in the artifact's own order.
 *
 * Titles are the run's, trimmed to one line. Severities are the run's too:
 * three high, three medium. The concentration finding is the one opened out,
 * not because it is the gravest — it is a medium — but because it is the one
 * where three governing documents read the same number three different ways,
 * which is the thing this whole surface exists to show.
 */
const FINDINGS = [
  { severity: "high", title: "Severe performance deterioration across all arrears bands" },
  { severity: "high", title: "High-LTV cohort on stale valuations, severe arrears" },
  { severity: "medium", title: "Geographic concentration, arrears in the same region", open: true },
  { severity: "high", title: "Regulatory disclosure blocked by missing identifiers" },
  { severity: "medium", title: "Borrower concentration cannot be assessed" },
  { severity: "medium", title: "Loans defaulting without a visible delinquency stage" },
] as const;

const SEVERITY_TONE = { high: "rose", medium: "amber" } as const;

const SeverityChip: React.FC<{ severity: "high" | "medium" }> = ({ severity }) => (
  <span
    style={{
      display: "inline-block",
      width: 74,
      textAlign: "center",
      padding: "3px 0",
      borderRadius: T.radius.chip,
      border: `1.5px solid ${TONE_EDGE[SEVERITY_TONE[severity]]}`,
      color: TONE_INK[SEVERITY_TONE[severity]],
      background: T.color.raised,
      fontSize: T.size.seq,
      fontWeight: 600,
      flexShrink: 0,
    }}
  >
    {severity}
  </span>
);

/** One verdict, as a chip in the opened finding's rule line. */
const VerdictChip: React.FC<{ verdict: (typeof VERDICTS)[number]; at: number }> = ({
  verdict,
  at,
}) => {
  const frame = useCurrentFrame();
  const shown = frame >= at;
  return (
    <Rise at={at}>
      <span
        style={{
          display: "inline-block",
          padding: "7px 16px",
          borderRadius: T.radius.chip,
          border: `1.5px solid ${shown ? TONE_EDGE[verdict.tone] : T.color.lineSoft}`,
          background: T.color.raised,
          fontSize: T.size.trace,
          whiteSpace: "nowrap",
          color: T.color.ink300,
        }}
      >
        {verdict.authority} {verdict.limit}{" "}
        <span style={{ color: TONE_INK[verdict.tone], fontWeight: 600 }}>{verdict.outcome}</span>
      </span>
    </Rise>
  );
};

/** Fact, rule and judgement — the three fields every finding carries. */
const OpenedFinding: React.FC = () => (
  <div
    style={{
      paddingLeft: 88,
      /* Padded top AND bottom: without the lower gap the judgement line sat
         against the next finding's title and read as part of it. */
      padding: "12px 0 18px 88px",
      display: "flex",
      flexDirection: "column",
      gap: 12,
    }}
  >
    <Rise at={F.fact}>
      <div style={{ display: "flex", gap: 16, alignItems: "baseline" }}>
        <FieldLabel>Fact</FieldLabel>
        <span style={{ fontSize: T.size.trace, color: T.color.ink100 }}>
          London · 31% of balance · 124 loans
        </span>
      </div>
    </Rise>
    <div style={{ display: "flex", gap: 16, alignItems: "baseline" }}>
      <Rise at={F.rule}>
        <FieldLabel>Rule</FieldLabel>
      </Rise>
      <div style={{ display: "flex", gap: 10 }}>
        {VERDICTS.map((verdict, i) => (
          <VerdictChip key={verdict.authority} verdict={verdict} at={F.verdict[i] ?? 0} />
        ))}
      </div>
    </div>
    <Rise at={F.judgement}>
      <div style={{ display: "flex", gap: 16, alignItems: "baseline" }}>
        <FieldLabel>Judgement</FieldLabel>
        <span style={{ fontSize: T.size.trace, color: T.color.ink100 }}>
          Breaches securitisation by 4 points · warehouse passes
        </span>
      </div>
    </Rise>
  </div>
);

const FieldLabel: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span
    style={{
      width: 88,
      flexShrink: 0,
      fontSize: T.size.seq,
      letterSpacing: "0.08em",
      textTransform: "uppercase",
      color: T.color.ink500,
      fontWeight: 600,
    }}
  >
    {children}
  </span>
);

const FindingRow: React.FC<{
  finding: (typeof FINDINGS)[number];
  at: number;
}> = ({ finding, at }) => (
  <Rise at={at}>
    <div style={{ display: "flex", alignItems: "center", gap: 14, height: 40 }}>
      <SeverityChip severity={finding.severity} />
      <span style={{ fontSize: T.size.trace, color: T.color.ink100, whiteSpace: "nowrap" }}>
        {finding.title}
      </span>
      {/* Every finding carries the same three fields. Saying so in the row
          means the one opened below is an example, not an exception. */}
      <span
        style={{
          marginLeft: "auto",
          fontSize: T.size.seq,
          color: T.color.ink500,
          whiteSpace: "nowrap",
        }}
      >
        fact · rule · judgement
      </span>
    </div>
  </Rise>
);

/**
 * The rest of the artifact, in its own words.
 *
 * The footer used to state three counts — six findings, eight diligence
 * items, four it could not assess — under a list that showed only the six.
 * Two of the three numbers described things the reader had no sight of, which
 * makes a summary into an assertion. These are the run's own diligence items
 * and declared gaps, trimmed to their subject and visibly cut off, so every
 * number on the page refers to something on the page.
 */
/* Three each, trimmed to the subject: at full length the row ran past the
   panel and cut off its own "+5 more", which is the one word that makes it
   a truncation rather than a claim to be complete. */
const DILIGENCE = [
  "Refresh Scotland valuations",
  "Direct-to-default transitions",
  "Borrower identifiers",
] as const;

const GAPS = [
  "Borrower concentration",
  "Seasoning / vintage",
  "Period-on-period change",
] as const;

const Section: React.FC<{
  label: string;
  count: number;
  items: readonly string[];
  at: number;
}> = ({ label, count, items, at }) => (
  <Rise at={at}>
    <div style={{ display: "flex", alignItems: "baseline", gap: 20, height: 38 }}>
      <span
        style={{
          width: 200,
          flexShrink: 0,
          fontSize: T.size.seq,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: T.color.ink500,
          fontWeight: 600,
        }}
      >
        {label} · {count}
      </span>
      <span style={{ fontSize: T.size.trace, color: T.color.ink300, whiteSpace: "nowrap" }}>
        {items.join("  ·  ")}
      </span>
      <span style={{ fontSize: T.size.trace, color: T.color.ink500, whiteSpace: "nowrap" }}>
        · +{count - items.length} more
      </span>
    </div>
  </Rise>
);

/**
 * The assessment, as the client agent received it.
 *
 * This replaces three separate screens — one per interesting call, then a
 * verdict card carrying nothing but a large red line. Those showed the
 * plumbing and then shouted the conclusion; neither told a lender what they
 * were being handed. What arrives over A2A is a structured report: six
 * findings with severities, each carrying fact, rule and judgement, plus the
 * work the specialist could not do. Showing it truncated is honest and is also
 * the point — the artifact is bigger than a frame.
 */
const Assessment: React.FC = () => {
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Label>Securitisation readiness assessment</Label>
        <Rise at={F.report + 12}>
          <Chip tone="rose">Material remediation required</Chip>
        </Rise>
      </div>

      <div style={{ marginTop: 22 }}>
        <span
          style={{
            fontSize: T.size.seq,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: T.color.ink500,
            fontWeight: 600,
          }}
        >
          Material findings · 6
        </span>
      </div>

      <div style={{ marginTop: 10, display: "flex", flexDirection: "column" }}>
        {FINDINGS.map((finding, i) => (
          <React.Fragment key={finding.title}>
            <FindingRow finding={finding} at={F.finding[i] ?? 0} />
            {"open" in finding && finding.open ? <OpenedFinding /> : null}
          </React.Fragment>
        ))}
      </div>

      <div
        style={{
          marginTop: "auto",
          paddingTop: 18,
          borderTop: `1.5px solid ${T.color.lineSoft}`,
        }}
      >
        <Section label="Further diligence" count={8} items={DILIGENCE} at={F.diligence} />
        <Section label="Could not assess" count={4} items={GAPS} at={F.gaps} />
      </div>
    </div>
  );
};

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


const TraceView: React.FC = () => (
  <div style={{ display: "flex", gap: 24, height: "100%" }}>
    <TraceGrid />
    <DetailCard />
  </div>
);

/** The governed panel: the trace, or whichever card has taken it over. */
const GovernedLayer: React.FC = () => (
  <Rise at={F.panel} style={{ padding: "0 32px" }}>
    <Panel style={{ height: 700, padding: "24px 28px", display: "flex", flexDirection: "column" }}>
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
        {/* The trace runs unbroken from the first check to the last. It used
            to be interrupted twice by cards built around single calls, which
            put the interesting moments in the wrong place: mid-investigation,
            before the specialist had finished, and framed as "look at this
            call" rather than "here is what came back". */}
        <Between from={F.panel} to={F.report - 24}>
          <TraceView />
        </Between>
        <Between from={F.report} to={A2A_DEMO.duration}>
          <Assessment />
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
/* Dropping the task-state line shortened the topology, so the centred start
   sits lower and the panel has 60px more to fill. */
const LIFT_PX = 374;

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
