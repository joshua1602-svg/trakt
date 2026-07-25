/**
 * Scene 8 — Closing.
 *
 * The headline, the supporting line, the five capabilities as a restrained rule,
 * and the brand lock-up. No pricing, no deployment claims, no client references.
 */

import React from "react";
import { useCurrentFrame } from "remotion";
import { Frame } from "../components/Frame";
import { TraktMark } from "../components/primitives";
import { C, LAYOUT, SYNTHETIC_FOOTER, TYPE } from "../design/tokens";
import { enter, fade, s } from "../design/motion";
import type { Caption } from "../timeline";

const CAPABILITIES = [
  "Onboarding",
  "Orchestration",
  "Management information",
  "Investor reporting",
  "Controlled AI interaction",
];

export const Scene8Closing: React.FC<{ captions: Caption[] }> = ({ captions }) => {
  const frame = useCurrentFrame();
  return (
    <Frame captions={captions} bare>
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          padding: `0 ${LAYOUT.margin}px`,
          gap: 34,
        }}
      >
        {/* Headline. */}
        <div
          style={{
            fontSize: TYPE.display,
            fontWeight: 600,
            lineHeight: 1.14,
            letterSpacing: "-0.028em",
            color: C.ink100,
            maxWidth: 1290,
            ...enter(frame, s(0.3), s(0.9), 12),
          }}
        >
          From recurring portfolio data
          <br />
          to governed portfolio intelligence.
        </div>

        {/* Supporting line. */}
        <div
          style={{
            fontSize: TYPE.subheading,
            lineHeight: 1.5,
            color: C.ink300,
            maxWidth: 1080,
            opacity: fade(frame, s(1.5), s(0.9)),
          }}
        >
          Trakt combines onboarding, orchestration, management information, investor
          reporting and controlled AI interaction in one managed operating layer.
        </div>

        {/* The five capabilities. */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 20,
            marginTop: 4,
            flexWrap: "wrap",
          }}
        >
          {CAPABILITIES.map((capability, i) => (
            <React.Fragment key={capability}>
              {i > 0 ? (
                <span
                  style={{
                    color: C.navy500,
                    fontSize: TYPE.bodySmall,
                    opacity: fade(frame, s(2.5) + i * s(0.13), s(0.4)),
                  }}
                >
                  ·
                </span>
              ) : null}
              <span
                style={{
                  fontSize: TYPE.bodySmall,
                  letterSpacing: "0.05em",
                  textTransform: "uppercase",
                  color: C.ink400,
                  fontWeight: 600,
                  opacity: fade(frame, s(2.4) + i * s(0.13), s(0.45)),
                }}
              >
                {capability}
              </span>
            </React.Fragment>
          ))}
        </div>

        {/* Rule. */}
        <div
          style={{
            height: 1,
            background: `linear-gradient(to right, ${C.navy500}, transparent)`,
            width: `${fade(frame, s(3.6), s(1.3)) * 100}%`,
            maxWidth: 1290,
          }}
        />

        {/* Brand lock-up. */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 22,
            ...enter(frame, s(4.2), s(1.0), 10),
          }}
        >
          <TraktMark size={70} />
          <div>
            <div
              style={{
                fontSize: TYPE.title,
                fontWeight: 700,
                letterSpacing: "-0.03em",
                color: C.ink100,
                lineHeight: 1.05,
              }}
            >
              Trakt
            </div>
            <div
              style={{
                marginTop: 8,
                fontSize: TYPE.body,
                color: C.ink400,
                letterSpacing: "-0.005em",
              }}
            >
              The data operating layer for specialist lenders.
            </div>
          </div>
        </div>
      </div>

      {/* Final disclaimer, stated plainly at the end of the film. */}
      <div
        style={{
          position: "absolute",
          bottom: 62,
          left: LAYOUT.margin,
          right: LAYOUT.margin,
          fontSize: TYPE.micro,
          color: C.ink500,
          lineHeight: 1.6,
          maxWidth: 1180,
          opacity: fade(frame, s(6.0), s(1.0)),
        }}
      >
        {SYNTHETIC_FOOTER}. Alderbridge Lending Platform is a fictional lender created
        for this demonstration; every loan, balance, valuation and reporting period
        shown is synthetic and produced by the Trakt pipeline from generated source
        data. No client data of any kind appears in this film.
      </div>
    </Frame>
  );
};
