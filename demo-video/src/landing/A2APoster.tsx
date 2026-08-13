/**
 * The agent-to-agent demo's poster still.
 *
 * A separate composition rather than a frame of the loop, for the same reason
 * the controls poster is: the landing page centres its play control, and the
 * frame this poster is drawn from puts the concentration card exactly there.
 * A centred plate would cover the three verdicts, and one number answered
 * three different ways by three governing documents IS the demo's argument.
 *
 * It is also a re-arrangement rather than a scaled-down copy. The loop stacks
 * the verdicts vertically, which is right when they resolve one at a time; a
 * still has no "one at a time", and a column tall enough to read ran straight
 * through the plate's band. Scaling it small enough to clear the band left the
 * 31% too quiet to carry the frame. So the poster lays the same three verdicts
 * out ACROSS the band above the plate, where all three read at once, and puts
 * the run's counters at the foot — the same figures the loop ends on.
 *
 * The clear band is not estimated. `landing-page/e2e/landing.spec.ts` asserts
 * the controls plate against a band running from 41% to 73% of the frame
 * height, so this poster is drawn to the same figures: at 1200x960 every mark
 * ends above y=394 and nothing resumes until y=701, both measured from the top
 * of the frame including the chrome. A first pass sized this by eye and put
 * the three verdicts squarely under the plate.
 */

import React from "react";
import { AbsoluteFill } from "remotion";

import { Chrome, TONE_EDGE, TONE_INK, VERDICTS } from "./A2ADemo";
import T from "./theme";

export const A2A_POSTER = {
  id: "LandingA2APoster",
  fps: 30,
  width: 1200,
  height: 960,
  duration: 120,
  stillFrame: 60,
} as const;

/** The run's totals, as the loop ends on them. */
const COUNTERS = [
  { label: "Checks run", value: "30" },
  { label: "Calculation time", value: "3.4 sec" },
  { label: "Agent reasoning", value: "2 min 17 sec" },
] as const;

const Node: React.FC<{ title: string }> = ({ title }) => (
  <div
    style={{
      width: 372,
      height: 80,
      padding: "0 22px",
      display: "flex",
      flexDirection: "column",
      justifyContent: "center",
      borderRadius: T.radius.panel,
      border: "1.5px solid rgba(145,157,209,0.55)",
      background: T.color.raised,
      textAlign: "center",
    }}
  >
    <p style={{ margin: 0, fontSize: T.size.node, fontWeight: 600, color: T.color.ink100 }}>
      {title}
    </p>
  </div>
);

/** The topology, resolved — nothing in flight, both ends lit. */
const RestingTopology: React.FC = () => (
  <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
    <Node title="Client enterprise agent" />
    <div
      style={{
        width: 132,
        position: "relative",
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
          height: 1.5,
          background: T.color.periDeep,
        }}
      />
      <span
        style={{
          position: "relative",
          padding: "3px 12px",
          borderRadius: T.radius.chip,
          border: `1.5px solid ${T.color.line}`,
          background: T.color.surface,
          fontSize: T.size.seq,
          letterSpacing: "0.1em",
          fontWeight: 600,
          color: T.color.peri,
        }}
      >
        A2A
      </span>
    </div>
    <Node title="Trakt Securitisation Readiness Agent" />
  </div>
);

/** One governing document's answer, sized to be read alongside the other two. */
const VerdictBadge: React.FC<{ verdict: (typeof VERDICTS)[number] }> = ({ verdict }) => (
  <div
    style={{
      flex: 1,
      padding: "12px 18px",
      borderRadius: T.radius.row,
      border: `1.5px solid ${TONE_EDGE[verdict.tone]}`,
      background: T.color.raised,
    }}
  >
    <p style={{ margin: 0, fontSize: T.size.trace, color: T.color.ink300 }}>{verdict.authority}</p>
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        justifyContent: "space-between",
        marginTop: 6,
      }}
    >
      <span
        style={{
          fontSize: T.size.row,
          color: T.color.ink400,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {verdict.limit}
      </span>
      <span style={{ fontSize: T.size.value, fontWeight: 600, color: TONE_INK[verdict.tone] }}>
        {verdict.outcome}
      </span>
    </div>
  </div>
);

const ConcentrationBand: React.FC = () => (
  <div
    style={{
      width: 1064,
      padding: "16px 30px 4px",
      background: T.color.inset,
      border: `1.5px solid ${T.color.line}`,
      borderRadius: T.radius.panel,
    }}
  >
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
        Securitisation readiness assessment
      </p>
      <span
        style={{
          padding: "5px 14px",
          borderRadius: T.radius.chip,
          border: `1.5px solid ${TONE_EDGE.rose}`,
          background: T.color.raised,
          fontSize: T.size.chip,
          fontWeight: 500,
          color: TONE_INK.rose,
          whiteSpace: "nowrap",
        }}
      >
        Material remediation required
      </span>
    </div>
    <div style={{ display: "flex", alignItems: "baseline", gap: 22, marginTop: 2 }}>
      <span
        style={{
          fontSize: T.size.figure,
          fontWeight: 600,
          color: T.color.ink100,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        31%
      </span>
      <span style={{ fontSize: T.size.row, color: T.color.ink300 }}>
        London · of balance · 124 loans
      </span>
    </div>
    <div style={{ display: "flex", gap: 14, marginTop: 8 }}>
      {VERDICTS.map((verdict) => (
        <VerdictBadge key={verdict.authority} verdict={verdict} />
      ))}
    </div>
  </div>
);

const Counters: React.FC = () => (
  <div style={{ display: "flex", justifyContent: "center", gap: 84 }}>
    {COUNTERS.map((counter) => (
      <div key={counter.label} style={{ textAlign: "center" }}>
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
          {counter.label}
        </p>
        <p
          style={{
            margin: "8px 0 0",
            fontSize: T.size.keyline,
            fontWeight: 600,
            color: T.color.ink100,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {counter.value}
        </p>
      </div>
    ))}
  </div>
);

const A2APoster: React.FC = () => (
  <AbsoluteFill style={{ background: T.color.surface, fontFamily: T.family }}>
    <style dangerouslySetInnerHTML={{ __html: T.fontFaceCss() }} />
    <Chrome />
    <AbsoluteFill style={{ top: 68 }}>
      {/* Positioned from the top: bottom-anchoring let the band overflow. */}
      <div style={{ position: "absolute", top: 0, left: 0, right: 0 }}>
        <RestingTopology />
      </div>
      <div
        style={{
          position: "absolute",
          top: 100,
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "center",
        }}
      >
        <ConcentrationBand />
      </div>
      {/* Below the plate's band, not inside it. */}
      <div style={{ position: "absolute", top: 640, left: 0, right: 0 }}>
        <Counters />
      </div>
    </AbsoluteFill>
  </AbsoluteFill>
);

export default A2APoster;
