/**
 * S1 · The cost — 420 frames (0:00–0:14)
 *
 * Opens on the buyer, not on artefacts: "You bought a back book. It didn't come with
 * your data model." The viewer has to recognise themselves before any column name is
 * worth showing them.
 *
 * Then the two source header stacks assemble, drift apart, and a `flag` hairline tries
 * and fails to connect them. Six rows each, not eighteen: the point is that they don't
 * match, and six proves it as well as eighteen does.
 *
 * The scene closes on the display claim and, beneath it, the one line in the whole film
 * about what Trakt replaces rather than what it does. Every other claim is capability;
 * this one is cost, and for a twelve-person lender it is the reason to keep watching.
 *
 * The two added beats are absorbed inside the scene's existing 420 frames — 60 come out
 * of the closing hold — so total runtime is unchanged.
 *
 * Data used: source header names only, read from both extracts' declared schemas.
 * This is the film's second and last `flag` moment.
 */

import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";

import { MONTH_END_COST, OPENING_LINE } from "../claims";
import { Body, Claim, Enter, Label, Rule, useGeometry, useIsSquare } from "../components/kit";
import { schemaFor } from "../data/fixtures";
import theme from "../theme";

const ROWS = 6;

/** The first six business headers of a schema, in source-file order. */
const headers = (key: string): string[] =>
  schemaFor(key)
    .columns.map((c) => c.header)
    .slice(0, ROWS);

const Stack: React.FC<{
  portfolioKey: string;
  title: string;
  at: number;
  drift: number;
}> = ({ portfolioKey, title, at, drift }) => (
  <div
    style={{
      display: "flex",
      flexDirection: "column",
      gap: theme.motion.quick,
      transform: `translateX(${drift}px)`,
    }}
  >
    <Enter at={at}>
      <Label>{title}</Label>
    </Enter>
    <Enter at={at + theme.motion.stagger}>
      <Rule />
    </Enter>
    {headers(portfolioKey).map((header, i) => (
      <Enter key={header} at={at + theme.motion.base} index={i}>
        <div
          style={{
            ...theme.type.stamp,
            color: theme.color.paper,
            paddingTop: theme.motion.quick / 2,
            paddingBottom: theme.motion.quick / 2,
            whiteSpace: "nowrap",
          }}
        >
          {header}
        </div>
      </Enter>
    ))}
  </div>
);

/** A faint ruled ledger grid — the hand-built spreadsheet the film is replacing. */
const LedgerGrid: React.FC<{ opacity: number }> = ({ opacity }) => {
  const { width, height } = useGeometry();
  const step = 60;
  const columns = Math.ceil(width / step);
  const rows = Math.ceil(height / step);
  return (
    <AbsoluteFill style={{ opacity }}>
      {Array.from({ length: columns }, (_, i) => (
        <div
          key={`v${i}`}
          style={{
            position: "absolute",
            left: i * step,
            top: 0,
            bottom: 0,
            width: theme.hairline,
            backgroundColor: theme.color.rule,
          }}
        />
      ))}
      {Array.from({ length: rows }, (_, i) => (
        <div
          key={`h${i}`}
          style={{
            position: "absolute",
            top: i * step,
            left: 0,
            right: 0,
            height: theme.hairline,
            backgroundColor: theme.color.rule,
          }}
        />
      ))}
    </AbsoluteFill>
  );
};

export const S1Cost: React.FC = () => {
  const frame = useCurrentFrame();
  const geometry = useGeometry();
  const isSquare = useIsSquare();

  // Beat boundaries, scene-relative.
  const OPENER_OUT = 66;
  const STACKS_AT = 78;
  const CLAIM_AT = 240;
  /** The claim holds alone for `slow` before the cost line arrives beneath it. */
  const COST_AT = CLAIM_AT + theme.motion.slow;

  // The opening line owns the frame, then clears as the grid comes up behind it.
  const openerOpacity = interpolate(
    frame,
    [0, theme.motion.base, OPENER_OUT, OPENER_OUT + theme.motion.quick],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const gridOpacity = interpolate(frame, [54, 120, 234, 282], [0, 0.5, 0.5, 0.2], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // The two stacks drift apart, then hold while the connector fails.
  const drift = interpolate(frame, [114, 168], [0, isSquare ? 26 : 54], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // The connector that tries and fails: two `flag` stubs reach out from the stacks,
  // stop short of each other, and stay short. A single centred segment reads as a join;
  // two stubs with a gap between them reads as the failure it is.
  const reach = interpolate(frame, [120, 156, 180], [0, 1, 0.62], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const stubMax = isSquare ? 34 : 74;
  // The stacks clear the frame entirely before the claim lands. Dimming them instead
  // leaves mono headers reading through the display type.
  const stacksOpacity = interpolate(frame, [216, 240], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <LedgerGrid opacity={gridOpacity} />

      {/* The buyer, before any artefact is named. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          opacity: openerOpacity,
        }}
      >
        <Body
          style={{
            textAlign: "center",
            width: geometry.width * (isSquare ? 0.84 : 0.56),
          }}
        >
          {OPENING_LINE}
        </Body>
      </AbsoluteFill>

      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          opacity: stacksOpacity,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: isSquare ? theme.motion.base : geometry.gutter / 2,
          }}
        >
          <Stack portfolioKey="A" title="Origination system" at={STACKS_AT} drift={-drift} />
          {/* The film's second and last `flag` moment. */}
          <div
            style={{
              alignSelf: "center",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              width: isSquare ? geometry.gutter * 2 : geometry.gutter * 4,
            }}
          >
            <div
              style={{
                width: stubMax * reach,
                height: theme.accentLine,
                backgroundColor: theme.color.flag,
              }}
            />
            <div
              style={{
                width: stubMax * reach,
                height: theme.accentLine,
                backgroundColor: theme.color.flag,
              }}
            />
          </div>
          <Stack
            portfolioKey="B"
            title="Servicer extract"
            at={STACKS_AT + theme.motion.quick}
            drift={drift}
          />
        </div>
      </AbsoluteFill>

      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: theme.motion.slow,
          paddingBottom: geometry.captionReserve,
        }}
      >
        <Enter at={CLAIM_AT}>
          <Claim measure={isSquare ? 0.9 : 0.84}>
            Every portfolio you buy is another reconciliation you own.
          </Claim>
        </Enter>
        {/* The film's only cost line. Held back so the claim lands on its own first. */}
        <Enter at={COST_AT}>
          <Body tone="mute" style={{ textAlign: "center" }}>
            {MONTH_END_COST}
          </Body>
        </Enter>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

export default S1Cost;
