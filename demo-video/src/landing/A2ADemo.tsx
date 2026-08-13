/**
 * Landing-page agent-to-agent demo — 42 seconds for the Agent-to-agent
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
 * What the run does and does not license
 * --------------------------------------
 * The transcript carries per-call `elapsed_ms` and order, but no wall-clock
 * timestamp per call, so this composition is sequence-driven and never claims a
 * schedule. The call durations are real; the gaps between calls are not, and
 * are compressed. The counters state the two totals that ARE recorded, side by
 * side, because a single "3,405 ms" would read as "the assessment took three
 * seconds" when the model spent 137 seconds thinking. Deterministic work in
 * milliseconds against model time in minutes is the architecture, so both are
 * on screen.
 *
 * Two figures come from the artifact rather than the transcript: the 31% and
 * the three rulebook verdicts. The governed digests are deliberately lean —
 * `concentration` returns `groups_count`, not the percentage — so the number
 * and its three judgements are quoted from the specialist's own finding, which
 * cites the call that produced them.
 *
 * Storyboard (30 fps, 1260 frames):
 *
 *   0.0 –  3.5 s  The objective crosses A2A. Trakt lights on arrival.
 *   3.5 –  5.5 s  Accepted; submitted → working; the governed layer opens.
 *   5.5 – 13.0 s  Orientation and the two long calls. `evaluate_rule_packs`
 *                 holds for its real 1,126 ms.
 *  13.0 – 16.0 s  Two calls fail with TOOL_INPUT_INVALID and the agent
 *                 corrects its own arguments. Recorded, and the one beat a
 *                 screenshot destroys entirely.
 *  16.0 – 21.0 s  `concentration` isolates: London, 31% of balance.
 *  21.0 – 26.5 s  Three governing documents answer one number differently —
 *                 pass, flag, breach — resolving one at a time.
 *  26.5 – 30.5 s  The trace resumes: seven more calls after the breach.
 *  30.5 – 35.0 s  The agent goes further of its own accord — two filtered
 *                 calls that only make sense as follow-up.
 *  35.0 – 38.0 s  The last five calls complete. Thirty, 3,405 ms governed.
 *  38.0 – 42.0 s  The artifact returns across A2A and the verdict holds.
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
  duration: 1260,
} as const;

/* ------------------------------------------------------------------ */
/* The recorded run                                                    */
/* ------------------------------------------------------------------ */

type Call = {
  seq: number;
  tool: string;
  /** Measured, from the transcript. Drives how long a row holds working. */
  ms: number;
  error?: boolean;
};

/** All thirty governed calls, in the order the specialist made them. */
const CALLS: readonly Call[] = [
  { seq: 1, tool: "portfolio_capabilities", ms: 71.32 },
  { seq: 2, tool: "portfolio_summary", ms: 15.99 },
  { seq: 3, tool: "readiness_framework", ms: 57.44 },
  { seq: 4, tool: "data_completeness", ms: 3.16 },
  { seq: 5, tool: "evaluate_rule_packs", ms: 1125.67 },
  { seq: 6, tool: "evaluate_covenants", ms: 36.14 },
  { seq: 7, tool: "regulatory_readiness", ms: 4.35 },
  { seq: 8, tool: "valuation_age_profile", ms: 182.05 },
  { seq: 9, tool: "stratify", ms: 0.8, error: true },
  { seq: 10, tool: "portfolio_history", ms: 1134.45 },
  { seq: 11, tool: "cohort_comparison", ms: 0.66, error: true },
  { seq: 12, tool: "cohort_comparison", ms: 204.29 },
  { seq: 13, tool: "stratify", ms: 11.81 },
  { seq: 14, tool: "transition_analysis", ms: 11.97 },
  { seq: 15, tool: "rank_loans", ms: 3.92 },
  { seq: 16, tool: "concentration", ms: 6.98 },
  { seq: 17, tool: "readiness_metrics", ms: 5.68 },
  { seq: 18, tool: "stratify", ms: 4.6 },
  { seq: 19, tool: "stratify", ms: 7.12 },
  { seq: 20, tool: "default_analysis", ms: 18.47 },
  { seq: 21, tool: "prepayment_analysis", ms: 3.59 },
  { seq: 22, tool: "list_validation_exceptions", ms: 0.2 },
  { seq: 23, tool: "contractual_analytics", ms: 65.23 },
  { seq: 24, tool: "concentration", ms: 7.63 },
  { seq: 25, tool: "stratify", ms: 5.71 },
  { seq: 26, tool: "period_change", ms: 383.02 },
  { seq: 27, tool: "get_loans", ms: 2.54 },
  { seq: 28, tool: "explain_values", ms: 24.39 },
  { seq: 29, tool: "rank_loans", ms: 5.04 },
  { seq: 30, tool: "loss_analysis", ms: 1.12 },
];

/** The run's model time, recorded in aggregate only — never per call. */
const MODEL_S = 137;

/**
 * Governed compute after each call, as the real running sum.
 *
 * Not a share of the total by call count: two calls — `evaluate_rule_packs` at
 * 1,126 ms and `portfolio_history` at 1,134 ms — spend two thirds of the whole
 * budget between them, so a linear counter would read 1,816 ms at call 16
 * where the run actually recorded 2,871. The staircase is the interesting
 * shape and it is free to show correctly.
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
 * Most of these are lifted straight from the call's own `result_digest`. The
 * two marked `fromFinding` are not in any digest — the governed layer returns
 * structure and counts, not payloads — and are quoted from the specialist's
 * finding, which names the call it measured them with.
 */
type Detail = {
  seq: number;
  /** Shown under the tool name, in the query's own vocabulary. */
  argument?: string;
  value: string;
  tone: "mint" | "amber" | "rose" | "neutral";
};

const DETAILS: readonly Detail[] = [
  { seq: 1, value: "27 available · 1 model-required", tone: "neutral" },
  { seq: 5, value: "screening flags counted separately", tone: "neutral" },
  { seq: 7, argument: "regime: ESMA_Annex2", value: "14 blocking gaps of 18", tone: "amber" },
  { seq: 8, value: "12.0% of balance on stale valuations", tone: "amber" },
  { seq: 9, argument: "dimension: days_past_due", value: "TOOL_INPUT_INVALID", tone: "rose" },
  { seq: 11, argument: "measures: [3]", value: "TOOL_INPUT_INVALID", tone: "rose" },
  { seq: 12, argument: "measures removed", value: "corrected · returned", tone: "mint" },
  { seq: 13, argument: "dimension: account_status", value: "corrected · returned", tone: "mint" },
  { seq: 21, value: "CPR 6.55% · OBSERVED_CPR@v2", tone: "neutral" },
  { seq: 24, argument: "filter: account_status = Arrears", value: "one region", tone: "amber" },
  { seq: 25, argument: "filter: current_LTV = 92%", value: "one region", tone: "amber" },
  { seq: 26, value: "unavailable · one snapshot", tone: "neutral" },
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
  task: 15,
  taskLand: 51,
  accepted: 72,
  working: 96,
  mcp: 114,
  panel: 132,
  concIsolate: 552,
  concValue: 588,
  verdict: [645, 700, 755] as const,
  resume: 810,
  drill: 900,
  drillValue: 942,
  wrap: 1014,
  complete: 1074,
  ret: 1092,
  received: 1134,
  end: 1152,
} as const;

/**
 * When each call lights, by segment of the storyboard.
 *
 * Written as segments rather than thirty literals so the shape stays legible:
 * every call keeps its recorded position, and only the compression varies. The
 * failure-and-correction segment is given the most room per call of any
 * stretch here — it is the beat that most needs to be read rather than
 * glimpsed.
 */
const SEGMENTS: readonly { seqs: readonly number[]; from: number; to: number }[] = [
  { seqs: [1, 2, 3, 4], from: 165, to: 300 },
  { seqs: [5, 6, 7], from: 300, to: 390 },
  /* 23 frames a call — the most room any stretch here gets. */
  { seqs: [8, 9, 10, 11, 12, 13], from: 390, to: 528 },
  { seqs: [14, 15], from: 528, to: 546 },
  { seqs: [16], from: 552, to: 564 },
  { seqs: [17, 18, 19, 20, 21, 22, 23], from: 810, to: 888 },
  { seqs: [24, 25], from: 912, to: 960 },
  { seqs: [26, 27, 28, 29, 30], from: 1014, to: 1074 },
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

/** Calls executed by this frame — the counter, and which rows are lit. */
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
/* The topology — persistent for the whole run                        */
/* ------------------------------------------------------------------ */

const Node: React.FC<{
  title: string;
  subtitle: string;
  lit: boolean;
  width: number;
}> = ({ title, subtitle, lit, width }) => (
  <div
    style={{
      width,
      /* Fixed height, contents centred: the Trakt node's name runs to two
         lines and the client's does not, and letting the boxes size to their
         text left the A2A wire meeting them at different heights. */
      height: 96,
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
    <p style={{ margin: "4px 0 0", fontSize: T.size.chip, color: T.color.ink400 }}>{subtitle}</p>
  </div>
);

/**
 * A label crossing the A2A wire.
 *
 * The payload is the point: an objective going one way, an artifact coming
 * back. It crosses in 0.6s because that is the shape of the measurement — the
 * protocol accounts for tens of milliseconds of a run that takes minutes, so a
 * slow, weighty hop would misdescribe it.
 */
const Crossing: React.FC<{
  at: number;
  text: string;
  tone?: Tone;
  back?: boolean;
}> = ({ at, text, tone = "peri", back = false }) => {
  const frame = useCurrentFrame();
  const p = ramp(frame, at, at + 18);
  if (frame < at - 6 || frame > at + 42) return null;
  const travel = 168;
  const x = (back ? 1 - p : p) * travel * 2 - travel;
  const opacity = Math.min(ramp(frame, at, at + 5), 1 - ramp(frame, at + 30, at + 42));
  return (
    <div
      style={{
        position: "absolute",
        left: "50%",
        top: 0,
        transform: `translateX(${x - 100}px)`,
        opacity,
        width: 200,
        display: "flex",
        justifyContent: "center",
      }}
    >
      <Chip tone={tone}>{text}</Chip>
    </div>
  );
};

const Wire: React.FC<{ label: string; lit: boolean }> = ({ label, lit }) => (
  <div
    style={{
      position: "relative",
      flex: 1,
      height: 96,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    }}
  >
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        top: 60,
        height: 1.5,
        background: lit ? T.color.periDeep : T.color.line,
      }}
    />
    <span
      style={{
        position: "absolute",
        top: 46,
        padding: "3px 12px",
        borderRadius: T.radius.chip,
        border: `1.5px solid ${T.color.line}`,
        background: T.color.surface,
        fontSize: T.size.seq,
        letterSpacing: "0.1em",
        fontWeight: 600,
        color: lit ? T.color.peri : T.color.ink500,
      }}
    >
      {label}
    </span>
  </div>
);

const Topology: React.FC = () => {
  const frame = useCurrentFrame();
  const traktLit = frame >= F.taskLand;
  const clientLit = frame < F.taskLand || frame >= F.received;
  return (
    <div style={{ padding: "26px 32px 0" }}>
      <div style={{ display: "flex", alignItems: "flex-start" }}>
        <Node
          title="Client enterprise agent"
          subtitle="Holds the objective"
          lit={clientLit}
          width={392}
        />
        <Wire label="A2A" lit={frame >= F.task} />
        <Node
          title="Trakt Securitisation Readiness Agent"
          subtitle="Decides what to investigate"
          lit={traktLit}
          width={392}
        />
      </div>
      {/* The crossings ride the wire's band, not the nodes. */}
      <div style={{ position: "relative", height: 0 }}>
        <div style={{ position: "absolute", left: 0, right: 0, top: -84 }}>
          <Crossing at={F.task} text="Assess for securitisation readiness" />
          <Crossing at={F.accepted} text="Accepted" tone="mint" back />
          <Crossing at={F.ret} text="Assessment · evidence attached" tone="peri" back />
        </div>
      </div>
      <TaskState />
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
        style={{ fontSize: T.size.seq, color: done ? T.color.ink500 : T.color.peri, fontWeight: 600 }}
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

/** The descent into the governed layer. Drawn, not faded, so it reads as a path. */
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
 * One call in the trace.
 *
 * A slot exists from the first frame; what it CONTAINS does not. The number is
 * there because thirty is the shape of the thing and the panel should be full
 * from the start, but the tool name only appears when the specialist actually
 * chooses it — the whole point being that nobody, including the caller, knew
 * what those thirty calls would be. An empty lower half and a pre-filled plan
 * are both wrong; a numbered slot resolving into a name is neither.
 */
const TraceRow: React.FC<{ call: Call }> = ({ call }) => {
  const frame = useCurrentFrame();
  const at = firesAt(call.seq);
  if (frame < at) {
    return (
      <div style={{ height: 34, display: "flex", alignItems: "center", gap: 10, opacity: 0.4 }}>
        <span
          style={{
            width: 24,
            textAlign: "right",
            fontSize: T.size.seq,
            color: T.color.ink500,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {call.seq}
        </span>
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
  const tone: Tone = !settled ? "peri" : call.error ? "rose" : "mint";
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
      <span
        style={{
          width: 24,
          textAlign: "right",
          fontSize: T.size.seq,
          color: T.color.ink500,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {call.seq}
      </span>
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: T.radius.chip,
          background: TONE_INK[tone],
          flexShrink: 0,
        }}
      />
      <span
        style={{
          fontSize: T.size.trace,
          color: call.error && settled ? T.color.rose : T.color.ink200,
        }}
      >
        {call.tool}
      </span>
    </div>
  );
};

/**
 * The trace, two columns, filling row by row.
 *
 * Row-major, not column-major: filling the first column before touching the
 * second left half the panel empty for the first twenty seconds, which is the
 * dead space this rebuild exists to remove. Alternating puts calls 1 and 2
 * side by side, so the block grows downward from the start and the frame is
 * used the whole way through.
 */
const TraceGrid: React.FC = () => (
  <div style={{ display: "flex", gap: 28, flex: 1 }}>
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
 * The most recent call that returned something worth reading.
 *
 * One card that changes rather than twelve that accumulate: the trace already
 * carries the volume, and this carries the meaning.
 */
const DetailCard: React.FC = () => {
  const frame = useCurrentFrame();
  const current = DETAILS.filter((d) => frame >= firesAt(d.seq) + 4).slice(-1)[0];
  if (!current) return null;
  const call = CALLS.find((c) => c.seq === current.seq);
  // Centred in its column: flush to the top it left a third of the panel empty
  // beneath it, and the trace beside it is what should be growing.
  return (
    <Panel style={{ width: 356, padding: "22px 24px", alignSelf: "center" }}>
      <Label>Returned</Label>
      <p
        style={{
          margin: "12px 0 0",
          fontSize: T.size.trace,
          color: T.color.ink200,
        }}
      >
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
    </Panel>
  );
};

/**
 * The counters — what the deterministic layer actually spent.
 *
 * Governed compute is unambiguous by name and correct by arithmetic: a bare
 * counter reaching 3,405 ms reads as "the assessment took three seconds", and
 * the assessment took a little over two minutes.
 *
 * Model time is the other half of that, and it arrives only at the end, once,
 * as a run total. The run records it in aggregate and never per call, so a
 * model-time counter climbing alongside the calls would be an invention — and
 * the contrast lands harder as a single reveal anyway: three and a half
 * seconds of deterministic work inside a hundred and thirty-seven of thinking.
 */
const Meters: React.FC = () => {
  const frame = useCurrentFrame();
  const done = executed(frame);
  const last = CALLS.filter((c) => frame >= firesAt(c.seq)).slice(-1)[0];
  const governed = last ? CUMULATIVE_MS.get(last.seq) ?? 0 : 0;
  return (
    <div style={{ display: "flex", gap: 34, alignItems: "flex-start" }}>
      <Meter label="Governed calls" value={`${done} / ${CALLS.length}`} />
      <Meter label="Governed compute" value={`${governed.toLocaleString("en-GB")} ms`} />
      {frame >= F.wrap ? (
        <Rise at={F.complete}>
          <Meter label="Model time" value={`${MODEL_S} s`} tone="peri" />
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

const VerdictRow: React.FC<{
  verdict: (typeof VERDICTS)[number];
  at: number;
}> = ({ verdict, at }) => {
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
 * Exported because the poster is drawn from it: a still of this, fully
 * resolved, is the one frame that reads as working software standing still.
 * `at` rebases the internal timing so the poster can render it settled.
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
        <span style={{ fontSize: T.size.chip, color: T.color.ink400 }}>
          seq 16 · concentration
        </span>
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

/** What the agent did next, unprompted: two filtered calls on the same region. */
const DrillCard: React.FC = () => (
  <Panel style={{ width: 900, padding: "30px 40px 34px" }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
      <Label>Follow-up · chosen by the specialist</Label>
      <span style={{ fontSize: T.size.chip, color: T.color.ink400 }}>seq 24 · 25</span>
    </div>
    <div style={{ display: "flex", flexDirection: "column", gap: 12, marginTop: 22 }}>
      {[
        { q: "concentration · filter: account_status = Arrears", v: "70.8% vs 6.8%", t: "rose" },
        { q: "stratify · filter: current_LTV = 92%", v: "one region · all of it", t: "amber" },
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
            <p style={{ margin: 0, fontSize: T.size.trace, color: T.color.ink300 }}>{row.q}</p>
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
  <Panel style={{ width: 900, padding: "34px 40px 38px" }}>
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
      <Chip tone="neutral">8 diligence items</Chip>
      <Chip tone="amber">4 declared gaps</Chip>
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
    style={{
      height: "100%",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    }}
  >
    {children}
  </div>
);

/** The trace and its detail card — shown whenever no card has taken over. */
const TraceView: React.FC = () => (
  <div style={{ display: "flex", gap: 28, height: "100%" }}>
    <TraceGrid />
    <DetailCard />
  </div>
);

/** The governed panel: trace and meters, or whichever card has taken over. */
const GovernedLayer: React.FC = () => {
  return (
    <Rise at={F.panel} style={{ padding: "0 32px" }}>
      <Panel style={{ height: 654, padding: "24px 28px", display: "flex", flexDirection: "column" }}>
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
              hand it back. It stays up through the return crossing, because the
              completed counters are the point of that moment. */}
          <Between from={F.panel} to={F.concIsolate}>
            <TraceView />
          </Between>
          <Between from={F.resume} to={F.drill - 12}>
            <TraceView />
          </Between>
          <Between from={F.wrap} to={F.end - 12}>
            <TraceView />
          </Between>
          {/* Cards centre in the panel body. Top-aligning them left a band of
              dead panel under every one, which is the fault this rebuild is
              correcting, not a thing to reproduce at a smaller scale. */}
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
};

const A2ADemo: React.FC = () => (
  <AbsoluteFill style={{ background: T.color.surface, fontFamily: T.family }}>
    <style dangerouslySetInnerHTML={{ __html: T.fontFaceCss() }} />
    <Chrome />
    <AbsoluteFill style={{ top: 68 }}>
      <Topology />
      <McpDescent />
      <GovernedLayer />
    </AbsoluteFill>
  </AbsoluteFill>
);

export default A2ADemo;
