/**
 * The scene frame: the page background, the persistent brand chrome, the caption
 * band and the discreet synthetic-data footer that every frame carries.
 *
 * Scenes render their content inside `<Frame>` and never draw chrome themselves,
 * so the film's furniture is identical from the first frame to the last.
 */

import React from "react";
import { AbsoluteFill, useCurrentFrame } from "remotion";
import {
  C,
  FONT,
  LAYOUT,
  PAGE_BACKGROUND,
  SYNTHETIC_FOOTER,
  TYPE,
} from "../design/tokens";
import { fadeInOut, s } from "../design/motion";
import type { Caption } from "../timeline";
import { TraktWordmark } from "./primitives";

/** Reserved height for the caption band, so content never collides with it. */
export const CAPTION_BAND_HEIGHT = 148;

export const Frame: React.FC<{
  children: React.ReactNode;
  captions?: Caption[];
  /** Scene label shown in the top-right chrome. */
  chapter?: string;
  /** Hide the brand chrome (used by the closing lock-up). */
  bare?: boolean;
}> = ({ children, captions = [], chapter, bare = false }) => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill
      style={{
        background: PAGE_BACKGROUND,
        fontFamily: FONT.sans,
        color: C.ink100,
        WebkitFontSmoothing: "antialiased",
        fontFeatureSettings: '"tnum"',
      }}
    >
      {/* Content region — inset above the caption band. */}
      <AbsoluteFill
        style={{
          padding: `${bare ? 0 : 92}px ${LAYOUT.margin}px ${CAPTION_BAND_HEIGHT}px`,
          display: "flex",
          flexDirection: "column",
        }}
      >
        {children}
      </AbsoluteFill>

      {!bare ? (
        <>
          {/* Brand chrome, top-left. */}
          <div style={{ position: "absolute", top: 44, left: LAYOUT.margin }}>
            <TraktWordmark />
          </div>
          {/* Chapter label, top-right. */}
          {chapter ? (
            <div
              style={{
                position: "absolute",
                top: 52,
                right: LAYOUT.margin,
                fontSize: TYPE.micro,
                letterSpacing: "0.16em",
                textTransform: "uppercase",
                color: C.ink500,
                fontWeight: 600,
              }}
            >
              {chapter}
            </div>
          ) : null}
        </>
      ) : null}

      {/* Caption band. */}
      <div
        style={{
          position: "absolute",
          left: LAYOUT.margin,
          right: LAYOUT.margin,
          bottom: 62,
          minHeight: 62,
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-end",
        }}
      >
        {captions.map((caption, i) => {
          const opacity = fadeInOut(frame, s(caption.at), s(caption.hold));
          if (opacity <= 0.001) return null;
          return (
            <div key={`${caption.text}-${i}`} style={{ opacity, position: "absolute", left: 0, right: 0, bottom: 0 }}>
              <div
                style={{
                  fontSize: TYPE.subheading,
                  lineHeight: 1.42,
                  fontWeight: 400,
                  color: C.ink100,
                  letterSpacing: "-0.01em",
                  maxWidth: 1400,
                  textShadow: "0 2px 18px rgba(0,0,0,0.55)",
                }}
              >
                {caption.text}
              </div>
              {caption.sub ? (
                <div
                  style={{
                    marginTop: 9,
                    fontSize: TYPE.bodySmall,
                    lineHeight: 1.45,
                    color: C.ink400,
                    maxWidth: 1200,
                  }}
                >
                  {caption.sub}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>

      {/* Discreet synthetic-data footer — present on every frame. */}
      <div
        style={{
          position: "absolute",
          bottom: 22,
          left: LAYOUT.margin,
          fontSize: 12,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "rgba(140,149,176,0.55)",
          fontWeight: 500,
        }}
      >
        {SYNTHETIC_FOOTER}
      </div>
    </AbsoluteFill>
  );
};
