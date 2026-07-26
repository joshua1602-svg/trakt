/**
 * S1 · The cost — 420 frames (0:00–0:14)
 *
 * Two source header stacks assemble, drift apart, and a `flag` hairline tries and
 * fails to connect them. Six rows each, not eighteen: the point is that they don't
 * match, and six proves it as well as eighteen does.
 *
 * Data used: source header names only, read from both extracts' declared schemas.
 * This is the film's second and last `flag` moment.
 */

import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";

import { Claim, Enter, Label, Rule, useGeometry, useIsSquare } from "../components/kit";
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

  // 0–90 grid fades up; 300–420 it dims back to 20% behind the claim.
  const gridOpacity = interpolate(frame, [0, 90, 288, 336], [0, 0.5, 0.5, 0.2], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // 90–150 the two stacks drift apart, then hold while the connector fails.
  const drift = interpolate(frame, [90, 150], [0, isSquare ? 26 : 54], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // The connector that tries and fails: two `flag` stubs reach out from the two
  // stacks, stop short of each other, and stay short. A single centred segment reads
  // as a join; two stubs with a gap between them reads as the failure it is.
  const reach = interpolate(frame, [96, 132, 156], [0, 1, 0.62], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const stubMax = isSquare ? 34 : 74;
  // The stacks clear the frame entirely before the claim lands. Dimming them
  // instead leaves mono headers reading through the display type.
  const stacksOpacity = interpolate(frame, [264, 288], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <LedgerGrid opacity={gridOpacity} />

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
          <Stack portfolioKey="A" title="Origination system" at={0} drift={-drift} />
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
          <Stack portfolioKey="B" title="Servicer extract" at={theme.motion.quick} drift={drift} />
        </div>
      </AbsoluteFill>

      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
        }}
      >
        <Enter at={288}>
          <Claim maxWidth={isSquare ? "90%" : "84%"}>
            Every portfolio you buy is another reconciliation you own.
          </Claim>
        </Enter>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

export default S1Cost;
