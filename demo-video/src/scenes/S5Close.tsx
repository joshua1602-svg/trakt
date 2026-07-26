/**
 * S5 · Close — 360 frames (1:18–1:30)
 *
 * Everything clears to `ink`. The lockup animates once from its fixed top-left
 * position to centre — the only time chrome moves in the entire film. `Chrome`
 * suppresses its own lockup for this scene's duration so there is never a second
 * one on screen, and the disclaimer stays exactly where it has been for 78 seconds.
 *
 * Then the capability line, then the ask. For a cold outbound asset the ask is the
 * whole ballgame: "See it run on your own tape" is concrete and low-commitment,
 * which is what converts this audience.
 */

import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

import { Body, Enter, Headline, Lockup, Stamp, useGeometry, useIsSquare } from "../components/kit";
import theme from "../theme";

/** Frames, scene-relative, at which the lockup finishes its one move. */
export const LOCKUP_TRAVEL_END = 60;

/**
 * Rendered width of the lockup at its one fixed size: the mark, the gap, and
 * "trakt" set in Archivo bold at the lockup size. Used to centre it, because a
 * `left: 50%` with a transform would fight the travel animation.
 */
const LOCKUP_WIDTH = theme.lockup.markSize + theme.lockup.gap + 78;

/**
 * The outbound destination, shown beneath the ask.
 *
 * Deliberately EMPTY. The storyboard's close reads "`[contact]` · `[link]`", and its
 * own checklist forbids shipping a bracketed number or placeholder — the same
 * applies to a placeholder address. Put the real address and URL here before the
 * film goes out and they appear automatically; leave them empty and the ask stands
 * on its own, which is honest, if less useful.
 */
const CONTACT: string[] = [];

const CAPABILITIES = [
  "Onboarding",
  "Orchestration",
  "Regulatory reporting",
  "Management information",
  "Governed AI access",
];

export const S5Close: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const geometry = useGeometry();
  const isSquare = useIsSquare();

  // The lockup's single move: from the chrome position to centre.
  const travel = spring({
    frame,
    fps,
    config: theme.motion.spring,
    durationInFrames: LOCKUP_TRAVEL_END,
  });
  const centreY = geometry.height / 2 - (isSquare ? 168 : 190);
  const top = interpolate(travel, [0, 1], [geometry.edge, centreY]);
  const left = interpolate(travel, [0, 1], [
    geometry.edge,
    geometry.width / 2 - LOCKUP_WIDTH / 2,
  ]);

  const lineOpacity = interpolate(frame, [120, 144], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const askOpacity = interpolate(frame, [240, 264], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <div style={{ position: "absolute", top, left }}>
        <Lockup />
      </div>

      <div
        style={{
          position: "absolute",
          left: geometry.gutter,
          right: geometry.gutter,
          top: centreY + (isSquare ? 78 : 92),
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: theme.motion.base,
          opacity: lineOpacity,
        }}
      >
        <Enter at={120}>
          <Headline maxWidth={isSquare ? "94%" : "70%"}>
            A data operating system for specialist lenders.
          </Headline>
        </Enter>
        <Enter at={120 + theme.motion.quick}>
          <Body
            tone="mute"
            style={{
              textAlign: "center",
              fontSize: theme.type.body.fontSize * (isSquare ? 0.72 : 1),
            }}
          >
            {CAPABILITIES.join(" · ")}
          </Body>
        </Enter>
      </div>

      {/* The ask. The film's only accented claim. */}
      <div
        style={{
          position: "absolute",
          left: geometry.gutter,
          right: geometry.gutter,
          bottom: geometry.captionReserve + (isSquare ? 34 : 56),
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: theme.motion.quick,
          opacity: askOpacity,
        }}
      >
        <Enter at={240}>
          <div
            style={{
              ...theme.type.headline,
              fontSize: theme.type.headline.fontSize * geometry.displayScale,
              color: theme.color.signal,
              textAlign: "center",
            }}
          >
            See it run on your own tape.
          </div>
        </Enter>
        {CONTACT.length ? (
          <Enter at={240 + theme.motion.quick}>
            <Stamp>{CONTACT.join(" · ")}</Stamp>
          </Enter>
        ) : null}
      </div>
    </AbsoluteFill>
  );
};

export default S5Close;
