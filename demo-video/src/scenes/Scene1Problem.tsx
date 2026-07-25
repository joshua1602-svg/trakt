/**
 * Scene 1 — The reporting problem.
 *
 * Two extracts arrive each month in different shapes, and four downstream
 * obligations must all reconcile to one truth. The column names shown are the
 * REAL headers of the two generated source schemas, so the difference the scene
 * claims is the difference the pipeline actually resolves.
 */

import React from "react";
import { useCurrentFrame } from "remotion";
import { Frame } from "../components/Frame";
import { Card, Chevron, Eyebrow, Pill } from "../components/primitives";
import { C, LAYOUT, TYPE } from "../design/tokens";
import { enter, fade, s } from "../design/motion";
import { PERIODS, portfolio, schemaFor } from "../data/fixtures";
import { periodLabel } from "../design/format";
import type { Caption } from "../timeline";

const OBLIGATIONS = [
  "Management information",
  "Investor reporting",
  "Risk monitoring",
  "Regulatory outputs",
];

const ExtractCard: React.FC<{
  portfolioKey: string;
  delay: number;
  frame: number;
}> = ({ portfolioKey, delay, frame }) => {
  const p = portfolio(portfolioKey);
  const schema = schemaFor(portfolioKey);
  const headers = schema.headers.filter((h) => h !== "Synthetic Data Notice");
  return (
    <Card
      style={{ flex: 1, minWidth: 0, ...enter(frame, delay, s(0.7)) }}
      padding={24}
    >
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
        <Eyebrow>{p.display_id}</Eyebrow>
        <span style={{ fontSize: TYPE.micro, color: C.ink500 }}>
          {periodLabel(PERIODS[PERIODS.length - 1].period)}
        </span>
      </div>
      <div
        style={{
          marginTop: 8,
          fontSize: TYPE.body,
          fontWeight: 600,
          color: C.ink100,
          letterSpacing: "-0.01em",
        }}
      >
        {p.description}
      </div>
      <div style={{ marginTop: 4, fontSize: TYPE.micro, color: C.ink500 }}>
        {schema.system_of_record}
      </div>
      <div
        style={{
          marginTop: 16,
          display: "flex",
          flexWrap: "wrap",
          gap: 7,
        }}
      >
        {headers.map((h, i) => {
          const o = fade(frame, delay + s(0.35) + i * s(0.055), s(0.3));
          return (
            <span
              key={h}
              style={{
                opacity: o,
                padding: "6px 11px",
                borderRadius: 5,
                background: "rgba(255,255,255,0.035)",
                border: `1px solid ${C.lineSoft}`,
                fontSize: TYPE.label,
                color: C.ink300,
                whiteSpace: "nowrap",
              }}
            >
              {h}
            </span>
          );
        })}
      </div>
      <div
        style={{
          marginTop: 16,
          paddingTop: 14,
          borderTop: `1px solid ${C.lineSoft}`,
          fontSize: TYPE.micro,
          color: C.ink500,
          lineHeight: 1.55,
        }}
      >
        {headers.length} columns · the lender's own vocabulary · no shared column
        names with the other extract
      </div>
    </Card>
  );
};

export const Scene1Problem: React.FC<{ captions: Caption[] }> = ({ captions }) => {
  const frame = useCurrentFrame();

  return (
    <Frame captions={captions} chapter="The reporting problem">
      <div style={{ display: "flex", flexDirection: "column", gap: 26, flex: 1, minHeight: 0 }}>
        {/* Recurring cadence strip. */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            ...enter(frame, s(0.2), s(0.6)),
          }}
        >
          <Eyebrow>Every reporting period</Eyebrow>
          <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
            {PERIODS.map((p, i) => (
              <React.Fragment key={p.period}>
                {i > 0 ? <Chevron size={13} /> : null}
                <Pill
                  tone={p.role === "current" ? "accent" : "neutral"}
                  style={{ opacity: fade(frame, s(0.4) + i * s(0.14), s(0.3)) }}
                >
                  {periodLabel(p.period)}
                </Pill>
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Two incoming extracts, in genuinely different shapes. */}
        <div style={{ display: "flex", gap: LAYOUT.gutter, alignItems: "stretch", flex: 1, minHeight: 0 }}>
          <ExtractCard portfolioKey="A" delay={s(0.7)} frame={frame} />
          <ExtractCard portfolioKey="B" delay={s(1.1)} frame={frame} />
        </div>

        {/* The downstream obligations that must all agree. */}
        <div
          style={{
            marginTop: "auto",
            display: "flex",
            alignItems: "center",
            gap: 18,
            opacity: fade(frame, s(6.4), s(0.8)),
          }}
        >
          <Eyebrow style={{ whiteSpace: "nowrap" }}>Must all reconcile</Eyebrow>
          <div style={{ display: "flex", gap: 12, flex: 1 }}>
            {OBLIGATIONS.map((o, i) => (
              <div
                key={o}
                style={{
                  flex: 1,
                  padding: "15px 17px",
                  borderRadius: LAYOUT.radiusSmall,
                  border: `1px solid ${C.line}`,
                  background: "rgba(255,255,255,0.02)",
                  fontSize: TYPE.bodySmall,
                  color: C.ink300,
                  ...enter(frame, s(6.6) + i * s(0.12), s(0.5)),
                }}
              >
                {o}
              </div>
            ))}
          </div>
        </div>
      </div>
    </Frame>
  );
};
