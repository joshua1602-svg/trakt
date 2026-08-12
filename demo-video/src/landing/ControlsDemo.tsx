/**
 * Landing-page controls demo — an 18-second loop for the Risk & Controls
 * section: document → structured requirement → reviewed control → live
 * monitoring → forward risk.
 *
 * Storyboard (30 fps, 540 frames):
 *
 *   0.0 –  4.0 s   A portfolio covenant schedule extract opens; three clauses
 *                  highlight. No typing effects — the document simply appears.
 *   4.0 –  8.0 s   The clauses become structured controls beside the document
 *                  ("Proposed"), with the wider count stated honestly.
 *   8.0 – 11.0 s   Governed review: identified → reviewed → activated. A
 *                  human step, visible — nothing activates itself.
 *  11.0 – 17.0 s   The activated control monitored three ways: funded book,
 *                  expected forecast, full pipeline — then the projected
 *                  breach horizon.
 *  16.5 – 18.0 s   The key line holds: know where the portfolio stands today —
 *                  and what it's moving toward. Loops.
 *
 * Every state depicted corresponds to shipped workflow (see
 * landing-page/docs/content-map.md §3); the figures are illustrative and the
 * frame carries a permanent "Illustrative · synthetic data" stamp.
 */

import React from "react";
import {
  AbsoluteFill,
  Sequence,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

import T from "./theme";

export const CONTROLS_DEMO = {
  id: "LandingControlsDemo",
  fps: 30,
  width: 1200,
  height: 960,
  duration: 540,
} as const;

/* ------------------------------------------------------------------ */
/* Motion helpers                                                     */
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
      style={{
        opacity: p,
        transform: `translateY(${(1 - p) * T.motion.rise}px)`,
        ...style,
      }}
    >
      {children}
    </div>
  );
};

/** Left-to-right reveal for the clause highlights. */
const useReveal = (at: number, frames = 12): number => {
  const frame = useCurrentFrame();
  return interpolate(frame, [at, at + frames], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
};

/* ------------------------------------------------------------------ */
/* Small primitives, mirroring the landing page's card language        */
/* ------------------------------------------------------------------ */

const Chip: React.FC<{
  tone?: "neutral" | "peri" | "mint" | "amber";
  children: React.ReactNode;
}> = ({ tone = "neutral", children }) => {
  const borders = {
    neutral: T.color.line,
    peri: "rgba(145,157,209,0.55)",
    mint: "rgba(54,194,168,0.45)",
    amber: "rgba(224,169,59,0.4)",
  } as const;
  const colors = {
    neutral: T.color.ink300,
    peri: T.color.peri,
    mint: T.color.mint,
    amber: T.color.amber,
  } as const;
  return (
    <span
      style={{
        display: "inline-block",
        border: `1.5px solid ${borders[tone]}`,
        borderRadius: T.radius.chip,
        padding: "6px 16px",
        fontSize: T.size.chip,
        fontWeight: 500,
        color: colors[tone],
        background: T.color.raised,
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </span>
  );
};

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

/* ------------------------------------------------------------------ */
/* Scene 1 — the document                                             */
/* ------------------------------------------------------------------ */

const CLAUSES = [
  "(a) geographic exposure to any single region shall not exceed 30% of the aggregate portfolio balance;",
  "(b) exposure secured on higher-risk collateral shall not exceed 10% of the aggregate portfolio balance;",
  "(c) aggregate exposure to any single obligor shall not exceed 10% of the aggregate portfolio balance.",
] as const;

const HIGHLIGHT_AT = [58, 76, 94] as const;

const Clause: React.FC<{ text: string; highlightAt: number }> = ({ text, highlightAt }) => {
  const reveal = useReveal(highlightAt);
  return (
    <div style={{ position: "relative", padding: "10px 14px", marginLeft: 18 }}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          width: `${reveal * 100}%`,
          background: "rgba(145,157,209,0.16)",
          borderLeft: `3px solid ${reveal > 0 ? T.color.periDeep : "transparent"}`,
          borderRadius: 6,
        }}
      />
      <p
        style={{
          position: "relative",
          margin: 0,
          fontSize: T.size.docBody,
          lineHeight: 1.55,
          color: T.color.ink200,
        }}
      >
        {text}
      </p>
    </div>
  );
};

const DocumentPane: React.FC<{ compact?: boolean }> = ({ compact = false }) => (
  <Panel style={{ padding: compact ? "26px 28px" : "34px 40px", width: compact ? 464 : 780 }}>
    <p
      style={{
        margin: 0,
        fontSize: T.size.label,
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        color: T.color.ink500,
        fontWeight: 600,
      }}
    >
      Portfolio covenant schedule — extract
    </p>
    <p
      style={{
        margin: "8px 0 0",
        fontSize: T.size.docTitle,
        fontWeight: 600,
        color: T.color.ink100,
      }}
    >
      Schedule 4 — Concentration requirements
    </p>
    <div
      style={{
        marginTop: 20,
        paddingTop: 20,
        borderTop: `1.5px solid ${T.color.lineSoft}`,
        display: "flex",
        flexDirection: "column",
        gap: 6,
      }}
    >
      <p
        style={{
          margin: "0 0 4px",
          fontSize: T.size.docBody,
          lineHeight: 1.55,
          color: T.color.ink400,
        }}
      >
        The Borrower shall procure that, at each Determination Date:
      </p>
      {CLAUSES.map((text, i) => (
        <Clause key={text} text={text} highlightAt={compact ? 0 : HIGHLIGHT_AT[i] ?? 0} />
      ))}
      <p
        style={{
          margin: "6px 0 0",
          fontSize: T.size.docBody,
          lineHeight: 1.55,
          color: T.color.ink500,
        }}
      >
        Breach of any Portfolio Covenant shall constitute a Review Event under
        clause 9.2 …
      </p>
    </div>
  </Panel>
);

const SceneDocument: React.FC = () => (
  <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
    <Rise at={6}>
      <DocumentPane />
    </Rise>
    <Rise at={104} style={{ marginTop: 28 }}>
      <Chip tone="peri">3 requirements identified in this extract</Chip>
    </Rise>
  </AbsoluteFill>
);

/* ------------------------------------------------------------------ */
/* Scenes 2–3 — extraction, review and activation                     */
/* ------------------------------------------------------------------ */

const CONTROLS = [
  { name: "Geographic concentration", threshold: "≤ 30%" },
  { name: "Higher-risk collateral", threshold: "≤ 10%" },
  { name: "Single obligor", threshold: "≤ 10%" },
] as const;

/** Local frames within the extraction sequence (which starts at 120). */
const ROW_AT = [22, 37, 52] as const;
const REVIEW_AT = 145; // 8.8 s absolute
const TICK_AT = [152, 158, 164] as const;
const ACTIVE_AT = 180; // 10.0 s absolute

const Tick: React.FC<{ at: number }> = ({ at }) => {
  const frame = useCurrentFrame();
  const on = frame >= at;
  return (
    <span
      style={{
        display: "inline-flex",
        width: 26,
        height: 26,
        borderRadius: 999,
        border: `1.5px solid ${on ? "rgba(54,194,168,0.55)" : T.color.line}`,
        color: on ? T.color.mint : T.color.ink500,
        alignItems: "center",
        justifyContent: "center",
        fontSize: T.size.tick,
        opacity: on ? 1 : 0.55,
        flexShrink: 0,
      }}
    >
      {on ? "✓" : ""}
    </span>
  );
};

const ControlRow: React.FC<{
  name: string;
  threshold: string;
  at: number;
  tickAt: number;
}> = ({ name, threshold, at, tickAt }) => {
  const frame = useCurrentFrame();
  const active = frame >= ACTIVE_AT;
  return (
    <Rise at={at}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 16,
          background: T.color.raised,
          border: `1.5px solid ${T.color.lineSoft}`,
          borderRadius: T.radius.row,
          padding: "16px 20px",
        }}
      >
        <Tick at={tickAt} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <p style={{ margin: 0, fontSize: T.size.row, fontWeight: 600, color: T.color.ink100 }}>
            {name}
          </p>
          <p style={{ margin: "3px 0 0", fontSize: T.size.chip, color: T.color.ink400 }}>
            Basis: portfolio balance
          </p>
        </div>
        <span
          style={{
            fontSize: T.size.row,
            fontWeight: 600,
            color: T.color.ink200,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {threshold}
        </span>
        <Chip tone={active ? "mint" : "peri"}>{active ? "Active" : "Proposed"}</Chip>
      </div>
    </Rise>
  );
};

const StatusPill: React.FC = () => {
  const frame = useCurrentFrame();
  const stage = frame >= ACTIVE_AT ? 2 : frame >= REVIEW_AT ? 1 : 0;
  const labels = ["6 controls identified", "Reviewed", "Activated"] as const;
  const tones = ["neutral", "peri", "mint"] as const;
  return <Chip tone={tones[stage]}>{labels[stage]}</Chip>;
};

/** The panel header follows the workflow state, so label and pill agree. */
const PanelLabel: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <p
      style={{
        margin: 0,
        fontSize: T.size.label,
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        color: T.color.ink500,
        fontWeight: 600,
      }}
    >
      {frame >= ACTIVE_AT ? "Active controls" : "Proposed controls"}
    </p>
  );
};

const SceneExtraction: React.FC = () => {
  const frame = useCurrentFrame();
  const docDim = interpolate(frame, [0, 14], [1, 0.55], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <AbsoluteFill style={{ flexDirection: "row", alignItems: "center", padding: "0 48px", gap: 32 }}>
      <div style={{ opacity: docDim, transform: "scale(0.92)", transformOrigin: "left center" }}>
        <DocumentPane compact />
      </div>
      <div style={{ fontSize: T.size.arrow, color: T.color.periDeep, flexShrink: 0 }}>→</div>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 14 }}>
        <Rise at={14}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 16,
            }}
          >
            <PanelLabel />
            <StatusPill />
          </div>
        </Rise>
        {CONTROLS.map((control, i) => (
          <ControlRow
            key={control.name}
            name={control.name}
            threshold={control.threshold}
            at={ROW_AT[i] ?? 0}
            tickAt={TICK_AT[i] ?? 0}
          />
        ))}
        <Rise at={70}>
          <p style={{ margin: "4px 0 0", fontSize: T.size.chip, color: T.color.ink400 }}>
            6 controls identified ·{" "}
            <span style={{ color: T.color.amber }}>1 flagged for review</span>
          </p>
        </Rise>
        <Rise at={REVIEW_AT + 4}>
          <p style={{ margin: 0, fontSize: T.size.chip, color: T.color.ink300 }}>
            Human review before activation — nothing goes live unapproved.
          </p>
        </Rise>
      </div>
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------------ */
/* Scene 4 — current + forward monitoring                             */
/* ------------------------------------------------------------------ */

const SCALE_MAX = 36;
const LIMIT = 30;

const STATES = [
  { label: "Funded book", value: 24.1, status: "Pass", color: T.color.mint, at: 14 },
  { label: "Expected forecast", value: 28.7, status: "Warning", color: T.color.amber, at: 52 },
  {
    label: "Including full pipeline",
    value: 31.4,
    status: "Projected breach",
    color: T.color.rose,
    at: 90,
  },
] as const;

const MonitorRow: React.FC<(typeof STATES)[number]> = ({ label, value, status, color, at }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const p = spring({
    frame: frame - at,
    fps,
    config: T.motion.spring,
    durationInFrames: 26,
  });
  return (
    <Rise at={at}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
        <p style={{ margin: 0, fontSize: T.size.row, color: T.color.ink300 }}>{label}</p>
        <p
          style={{
            margin: 0,
            fontSize: T.size.value,
            fontWeight: 600,
            color,
            fontVariantNumeric: "tabular-nums",
            opacity: interpolate(frame, [at + 16, at + 26], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          {value}% · {status}
        </p>
      </div>
      <div
        style={{
          position: "relative",
          marginTop: 10,
          height: 10,
          borderRadius: 999,
          background: T.color.raised,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${(p * value * 100) / SCALE_MAX}%`,
            maxWidth: "100%",
            borderRadius: 999,
            background: color,
            opacity: 0.75,
          }}
        />
        <div
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: `${(LIMIT / SCALE_MAX) * 100}%`,
            width: 2,
            background: T.color.ink400,
          }}
        />
      </div>
    </Rise>
  );
};

/** The monitoring card, shared by the film's closing scene and the poster. */
export const MonitoringCard: React.FC = () => (
  <Panel style={{ width: 820, padding: "36px 44px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <p
          style={{
            margin: 0,
            fontSize: T.size.label,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: T.color.ink500,
            fontWeight: 600,
          }}
        >
          Concentration controls
        </p>
        <Chip tone="mint">Active</Chip>
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginTop: 18,
          paddingBottom: 18,
          borderBottom: `1.5px solid ${T.color.lineSoft}`,
        }}
      >
        <p style={{ margin: 0, fontSize: T.size.docTitle, fontWeight: 600, color: T.color.ink100 }}>
          Geographic concentration — any single region
        </p>
        <p
          style={{
            margin: 0,
            fontSize: T.size.chip,
            color: T.color.ink400,
            whiteSpace: "nowrap",
            flexShrink: 0,
            paddingLeft: 24,
          }}
        >
          Limit ≤ 30% of balance
        </p>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 26, marginTop: 26 }}>
        {STATES.map((state) => (
          <MonitorRow key={state.label} {...state} />
        ))}
      </div>
      <Rise at={135} style={{ marginTop: 28 }}>
        <p
          style={{
            margin: 0,
            paddingTop: 20,
            borderTop: `1.5px solid ${T.color.lineSoft}`,
            fontSize: T.size.row,
            color: T.color.ink300,
          }}
        >
          Projected breach horizon:{" "}
          <span style={{ color: T.color.ink100, fontWeight: 600 }}>Nov 2026</span>
        </p>
      </Rise>
  </Panel>
);

/* The close is set in the page's heading tone, not an accent: it should read
   as part of the product UI, with the risk colours staying inside the
   statuses and bars above. */
export const MonitoringKeyline: React.FC<{ at?: number }> = ({ at = 165 }) => (
  <Rise at={at} style={{ marginTop: 30, width: 820 }}>
      <p
        style={{
          margin: 0,
          fontSize: T.size.keyline,
          lineHeight: 1.3,
          fontWeight: 600,
          color: T.color.ink100,
          textAlign: "center",
        }}
      >
        Know where the portfolio stands today —
        <br />
        and what it&rsquo;s moving toward.
    </p>
  </Rise>
);

const SceneMonitoring: React.FC = () => (
  <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
    <MonitoringCard />
    <MonitoringKeyline />
  </AbsoluteFill>
);

/* ------------------------------------------------------------------ */
/* Chrome + assembly                                                  */
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
      Trakt · Portfolio controls
    </p>
    <p style={{ margin: 0, fontSize: T.size.chip, fontWeight: 500, color: T.color.amber }}>
      Illustrative · synthetic data
    </p>
  </div>
);

/** Fade a scene in over its first 8 frames; scenes cut out hard. */
const SceneFade: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 8], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return <AbsoluteFill style={{ opacity }}>{children}</AbsoluteFill>;
};

const ControlsDemo: React.FC = () => (
  <AbsoluteFill style={{ background: T.color.surface, fontFamily: T.family }}>
    <style dangerouslySetInnerHTML={{ __html: T.fontFaceCss() }} />
    <Chrome />
    <AbsoluteFill style={{ top: 68 }}>
      <Sequence from={0} durationInFrames={120}>
        <SceneFade>
          <SceneDocument />
        </SceneFade>
      </Sequence>
      <Sequence from={120} durationInFrames={210}>
        <SceneFade>
          <SceneExtraction />
        </SceneFade>
      </Sequence>
      <Sequence from={330} durationInFrames={210}>
        <SceneFade>
          <SceneMonitoring />
        </SceneFade>
      </Sequence>
    </AbsoluteFill>
  </AbsoluteFill>
);

export default ControlsDemo;
