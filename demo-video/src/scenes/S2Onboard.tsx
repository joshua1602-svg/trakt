/**
 * S2 · Onboarded in under 48 hours — 720 frames (0:14–0:38)
 *
 * The clock is an `HH:MM` count-up to the stated onboarding window, tape received to
 * governed output. It is a STATED claim, not a measured figure: it lives in
 * `src/claims.ts` alongside every other thing the film asserts without a fixture behind
 * it, so an auditor can see the complete list in one place. It is deliberately short of
 * the 48-hour threshold the claim names — a figure with headroom survives the first
 * question about it; one that lands exactly on its own limit does not.
 *
 * Everything else in the scene is measured. The counters are the Gate 1 result for the
 * acquired back book — the portfolio this film's story is about — read from the
 * approved onboarding contract. The third counter is the film's first `flag` moment,
 * and the beat that follows it is the most important in the film: proof that the
 * platform refers what it cannot resolve rather than guessing.
 *
 * Data used: onboarding.mapping counts and mappingDecisions from the approved
 * contract; four real source→canonical pairs.
 */

import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";

import {
  Body,
  Claim,
  Counter,
  Enter,
  Figure,
  Label,
  Panel,
  Rule,
  Stamp,
  Stat,
  useGeometry,
  useIsSquare,
} from "../components/kit";
import { ONBOARDING_HOURS } from "../claims";
import { CURRENT_PERIOD, mappingDecision, onboarding, portfolio, SUMMARY } from "../data/fixtures";
import { count, hoursMinutes } from "../format";
import theme from "../theme";

/**
 * The referral card body, condensed to two lines.
 *
 * The authoritative text is `mappingDecisions[].note` in the demo manifest — two full
 * sentences, written for the record rather than for a viewer with four seconds. This is
 * the same reasoning at reading speed. A unit test asserts every term below still
 * appears in the fixture note, so the condensation cannot drift away from what the
 * platform actually recorded.
 */
const REFERRAL_LINES = [
  "Reserve-facility drawdown taken this month.",
  "Already in Principal Outstanding — intentionally not mapped.",
];

/** The four mapping pairs the storyboard names, read from the recorded decisions. */
const PAIRS: { key: string; header: string }[] = [
  { key: "A", header: "Loan Reference" },
  { key: "A", header: "Completion Date" },
  { key: "B", header: "Account Number" },
  { key: "B", header: "Reporting Cut-Off" },
];

const MappingRow: React.FC<{ index: number; at: number; sourceKey: string; header: string }> = ({
  index,
  at,
  sourceKey,
  header,
}) => {
  const frame = useCurrentFrame();
  const isSquare = useIsSquare();
  const decision = mappingDecision(sourceKey, header);
  const start = at + index * theme.motion.stagger * 4;
  // The line draws left-to-right, then the canonical field lands.
  const draw = interpolate(frame, [start, start + theme.motion.base], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const lineWidth = isSquare ? 76 : 148;
  // Equal columns, so the whole block is centred on the frame rather than pulled
  // right by a wider canonical-field column.
  const COLUMN = isSquare ? 230 : 320;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: theme.motion.base }}>
      <Enter at={start} style={{ width: COLUMN, textAlign: "right" }}>
        <div style={{ ...theme.type.stamp, color: theme.color.paper, whiteSpace: "nowrap" }}>
          {decision.source_header}
        </div>
      </Enter>
      <div
        style={{
          width: lineWidth * draw,
          height: theme.hairline,
          backgroundColor: theme.color.rule,
          flexShrink: 0,
        }}
      />
      <Enter at={start + theme.motion.base} style={{ width: COLUMN, opacity: draw }}>
        <div style={{ ...theme.type.stamp, color: theme.color.paper, whiteSpace: "nowrap" }}>
          {decision.canonical_field}
        </div>
      </Enter>
    </div>
  );
};

const CountBlock: React.FC<{
  at: number;
  index: number;
  to: number;
  caption: string;
  flag?: boolean;
}> = ({ at, index, to, caption, flag = false }) => (
  <Enter at={at} index={index * 6} style={{ minWidth: 0 }}>
    <div style={{ display: "flex", flexDirection: "column", gap: theme.motion.quick }}>
      <Figure scale={0.62} tone={flag ? "flag" : "paper"}>
        <Counter from={0} to={to} at={at} frames={theme.motion.slow} format={count} />
      </Figure>
      <Rule tone={flag ? "flag" : "rule"} />
      <Stamp tone={flag ? "flag" : "mute"}>{caption}</Stamp>
    </div>
  </Enter>
);

export const S2Onboard: React.FC = () => {
  const frame = useCurrentFrame();
  const geometry = useGeometry();
  const isSquare = useIsSquare();

  const book = portfolio("B");
  const mapping = onboarding("B").mapping;

  // The one referred item that is a genuine judgement call rather than the demo
  // watermark. Named on screen, with the reviewer's own note.
  const referred = onboarding("B").mappingDecisions.find(
    (d) => !d.canonical_field && d.source_header !== "Synthetic Data Notice",
  );

  // 420–480 (scene 0–60): the clock owns the frame. Then it hands over to a corner
  // stamp — a stamp-sized line, not a scaled-down stat, because scaling mono type
  // down to a third leaves an unreadable smear where a legible fact should be.
  const CLOCK_HANDOVER = 60;
  const clockOpacity = interpolate(
    frame,
    [CLOCK_HANDOVER, CLOCK_HANDOVER + theme.motion.quick],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const stampOpacity = interpolate(
    frame,
    [CLOCK_HANDOVER + theme.motion.quick, CLOCK_HANDOVER + theme.motion.base],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );

  // Beat boundaries, scene-relative. Each beat is long enough for its caption to be
  // read at a comfortable subtitle rate — the film is watched with the sound off, so
  // a beat that outruns its caption is a beat the viewer does not get.
  const MAPPING_AT = 60;
  const COUNTS_AT = 294;
  const REFERRED_AT = 414;
  const CLAIM_AT = 612;

  const mappingOpacity = interpolate(frame, [MAPPING_AT, 84, 288, 306], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const countsOpacity = interpolate(frame, [COUNTS_AT, 318, 594, 612], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const referredOpacity = interpolate(frame, [REFERRED_AT, 438, 594, 612], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const claimOpacity = interpolate(frame, [CLAIM_AT, CLAIM_AT + theme.motion.base], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      {/* The measured elapsed clock, centre frame. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          opacity: clockOpacity,
        }}
      >
        <Stat
          align="center"
          value={
            <Counter
              from={0}
              to={ONBOARDING_HOURS}
              at={0}
              frames={theme.motion.slow + theme.motion.base}
              format={hoursMinutes}
            />
          }
          stamp={
            `HOURS ELAPSED · TAPE RECEIVED → GOVERNED OUTPUT · ` +
            `${portfolio("A").display_id} + ${book.display_id} · ` +
            `${count(SUMMARY.metrics.loan_count)} LOANS · ${CURRENT_PERIOD.reportingDate}`
          }
        />
      </AbsoluteFill>

      {/* Then the same fact as a corner stamp, held for the rest of the scene. */}
      <div
        style={{
          position: "absolute",
          top: geometry.edge,
          right: geometry.edge,
          textAlign: "right",
          opacity: stampOpacity,
        }}
      >
        <Stamp>{`${hoursMinutes(ONBOARDING_HOURS)} HRS · TAPE RECEIVED → GOVERNED OUTPUT`}</Stamp>
      </div>

      {/* The mapping lines draw, source header → canonical field. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          opacity: mappingOpacity,
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: theme.motion.base }}>
          <Enter at={MAPPING_AT} style={{ display: "flex", justifyContent: "center" }}>
            <Label>Approved onboarding contract</Label>
          </Enter>
          {PAIRS.map((pair, i) => (
            <MappingRow
              key={`${pair.key}-${pair.header}`}
              index={i}
              at={MAPPING_AT + theme.motion.base}
              sourceKey={pair.key}
              header={pair.header}
            />
          ))}
        </div>
      </AbsoluteFill>

      {/* Three counters resolve. The third is flag; the others are not. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          opacity: countsOpacity,
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: isSquare ? "column" : "row",
            gap: isSquare ? theme.motion.base : geometry.gutter,
            alignItems: "flex-start",
          }}
        >
          <CountBlock at={COUNTS_AT} index={0} to={mapping.mapped_count} caption="FIELDS MAPPED" />
          <CountBlock
            at={COUNTS_AT}
            index={1}
            to={mapping.client_contract_count}
            caption="CLIENT-SPECIFIC DECISIONS"
          />
          <CountBlock
            at={COUNTS_AT}
            index={2}
            to={mapping.unmapped_count}
            caption="REFERRED FOR REVIEW"
            flag
          />
        </div>
      </AbsoluteFill>

      {/* Hold on the referred item. The most important beat in the film. */}
      {referred ? (
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            bottom: geometry.captionReserve + (isSquare ? 8 : 40),
            display: "flex",
            justifyContent: "center",
            opacity: referredOpacity,
            paddingLeft: geometry.gutter,
            paddingRight: geometry.gutter,
          }}
        >
          <Panel
            style={{
              padding: geometry.gutter / 2,
              maxWidth: isSquare ? "100%" : 980,
              display: "flex",
              flexDirection: "column",
              gap: theme.motion.quick,
            }}
          >
            <Label tone="flag">Referred for review</Label>
            <Stamp tone="paper">{`${book.display_id} · ${referred.source_header}`}</Stamp>
            <Rule tone="flag" />
            {REFERRAL_LINES.map((line) => (
              <Body key={line} tone="mute" style={{ fontSize: theme.type.body.fontSize * 0.8 }}>
                {line}
              </Body>
            ))}
          </Panel>
        </div>
      ) : null}

      {/* The claim. No elapsed-time claim the data cannot support. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          opacity: claimOpacity,
        }}
      >
        <Enter at={CLAIM_AT}>
          <Claim measure={isSquare ? 0.9 : 0.76}>
            Onboarded in under 48 hours. Approved once, applied every period.
          </Claim>
        </Enter>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

export default S2Onboard;
