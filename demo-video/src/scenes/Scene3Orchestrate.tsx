/**
 * Scene 3 — Orchestrate continuously.
 *
 * A new month-end extract arrives for each portfolio and moves through the real
 * gate sequence, then the Assembler consolidates both books into the platform
 * view. Every figure — row counts, balances, exception counts, the consolidated
 * total — is read from the run and assembler manifests the pipeline produced.
 */

import React from "react";
import { useCurrentFrame } from "remotion";
import { Frame } from "../components/Frame";
import { Card, Chevron, Eyebrow, Pill, Tick, TraktMark } from "../components/primitives";
import { C, LAYOUT, THEME, TYPE } from "../design/tokens";
import { enter, fade, grow, s } from "../design/motion";
import { useVideoConfig } from "remotion";
import {
  ARTEFACTS,
  CURRENT_PERIOD,
  MOVEMENT,
  SUMMARY,
  contribution,
  lens,
  portfolio,
} from "../data/fixtures";
import { count, longDate, money } from "../design/format";
import type { Caption } from "../timeline";

/** The gate sequence the MI pipeline actually runs (engine/orchestrator/trakt_run.py). */
const GATES = [
  { id: "recognise", label: "Source recognised", detail: "portfolio + reporting date" },
  { id: "contract", label: "Approved contract applied", detail: "no manual mapping" },
  { id: "transform", label: "Canonical transform", detail: "typing, derivation, geography" },
  { id: "gate2", label: "Canonical validation", detail: "Gate 2 · schema conformance" },
  { id: "gate3", label: "Business rules", detail: "Gate 3 · cross-field rules" },
  { id: "output", label: "Portfolio canonical", detail: "provenance-stamped output" },
];

const PortfolioRun: React.FC<{
  portfolioKey: string;
  frame: number;
  delay: number;
}> = ({ portfolioKey, frame, delay }) => {
  const p = portfolio(portfolioKey);
  const l = lens(portfolioKey);
  const current = l.summary.metrics;
  const perGate = s(1.05);
  return (
    <Card
      style={{
        flex: 1,
        minWidth: 0,
        display: "flex",
        flexDirection: "column",
        ...enter(frame, delay, s(0.6)),
      }}
      padding={20}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <Eyebrow>{p.display_id}</Eyebrow>
          <div style={{ marginTop: 4, fontSize: TYPE.bodySmall, color: C.ink400 }}>
            {p.description}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div
            style={{
              fontSize: TYPE.subheading,
              fontWeight: 600,
              color: C.ink100,
              fontVariantNumeric: "tabular-nums",
              opacity: fade(frame, delay + perGate * GATES.length, s(0.5)),
            }}
          >
            {money(current.funded_balance)}
          </div>
          <div
            style={{
              fontSize: TYPE.micro,
              color: C.ink500,
              opacity: fade(frame, delay + perGate * GATES.length, s(0.5)),
            }}
          >
            {count(current.loan_count)} loans
          </div>
        </div>
      </div>

      <div
        style={{
          marginTop: 16,
          display: "flex",
          flexDirection: "column",
          gap: 8,
          flex: 1,
          justifyContent: "space-between",
        }}
      >
        {GATES.map((g, i) => {
          const at = delay + s(0.5) + i * perGate;
          const done = fade(frame, at, s(0.4));
          return (
            <div
              key={g.id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 11,
                padding: "11px 13px",
                borderRadius: 6,
                border: `1px solid ${done > 0.5 ? "rgba(46,125,91,0.32)" : C.lineSoft}`,
                background:
                  done > 0.5 ? "rgba(46,125,91,0.075)" : "rgba(255,255,255,0.018)",
              }}
            >
              <div style={{ width: 14, flexShrink: 0, opacity: done }}>
                <Tick size={13} />
              </div>
              <span
                style={{
                  fontSize: TYPE.label,
                  fontWeight: done > 0.5 ? 600 : 400,
                  color: done > 0.5 ? C.ink100 : C.ink500,
                  whiteSpace: "nowrap",
                }}
              >
                {g.label}
              </span>
              <span
                style={{
                  marginLeft: "auto",
                  fontSize: TYPE.micro,
                  color: C.ink500,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {g.detail}
              </span>
            </div>
          );
        })}
      </div>
    </Card>
  );
};

export const Scene3Orchestrate: React.FC<{ captions: Caption[] }> = ({ captions }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const validation = ARTEFACTS.validationReport as {
    rowsValidated?: number;
    canonicalValidation?: string;
    businessRuleExceptions?: number;
    exceptionRatePct?: number | null;
  };
  const a = contribution("A");
  const b = contribution("B");
  const assemblerAt = s(15.0);
  const assemblerProgress = grow(frame, fps, assemblerAt);

  return (
    <Frame captions={captions} chapter="Stage 2 · Orchestrate continuously">
      <div style={{ display: "flex", flexDirection: "column", gap: 18, flex: 1, minHeight: 0 }}>
        {/* New extracts arriving. */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            ...enter(frame, s(0.2), s(0.5)),
          }}
        >
          <Pill tone="accent">New month-end extract received</Pill>
          <span style={{ fontSize: TYPE.bodySmall, color: C.ink400 }}>
            {longDate(CURRENT_PERIOD.reportingDate)}
          </span>
        </div>

        <div style={{ display: "flex", gap: LAYOUT.gutter }}>
          <PortfolioRun portfolioKey="A" frame={frame} delay={s(0.7)} />
          <PortfolioRun portfolioKey="B" frame={frame} delay={s(1.1)} />
        </div>

        {/* Assembler. */}
        <Card
          style={{ opacity: fade(frame, assemblerAt - s(0.4), s(0.7)) }}
          accent={C.peri400}
          padding={20}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 26 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
              <TraktMark size={28} />
              <div>
                <Eyebrow>Assembler</Eyebrow>
                <div style={{ marginTop: 3, fontSize: TYPE.bodySmall, color: C.ink400 }}>
                  Latest accepted canonical per portfolio
                </div>
              </div>
            </div>

            {/* Two books flowing into one platform view. */}
            <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 14, minWidth: 0 }}>
              {[a, b].map((c, i) => (
                <div
                  key={c.id}
                  style={{
                    flex: 1,
                    padding: "11px 14px",
                    borderRadius: 6,
                    border: `1px solid ${C.lineSoft}`,
                    background: "rgba(255,255,255,0.025)",
                    opacity: fade(frame, assemblerAt + i * s(0.2), s(0.4)),
                    minWidth: 0,
                  }}
                >
                  <div
                    style={{
                      fontSize: TYPE.micro,
                      color: C.ink500,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {c.label}
                  </div>
                  <div
                    style={{
                      marginTop: 3,
                      fontSize: TYPE.body,
                      fontWeight: 600,
                      color: C.ink100,
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {money(c.current)}
                  </div>
                </div>
              ))}
              <div style={{ opacity: assemblerProgress, flexShrink: 0 }}>
                <Chevron size={20} colour={C.peri300} />
              </div>
              <div
                style={{
                  flex: 1.25,
                  padding: "11px 16px",
                  borderRadius: 6,
                  border: `1px solid ${C.navy500}`,
                  background: "rgba(145,157,209,0.10)",
                  opacity: assemblerProgress,
                  minWidth: 0,
                }}
              >
                <div style={{ fontSize: TYPE.micro, color: C.peri300, whiteSpace: "nowrap" }}>
                  Consolidated platform canonical
                </div>
                <div
                  style={{
                    marginTop: 3,
                    display: "flex",
                    alignItems: "baseline",
                    gap: 11,
                  }}
                >
                  <span
                    style={{
                      fontSize: TYPE.heading,
                      fontWeight: 600,
                      color: C.ink100,
                      letterSpacing: "-0.02em",
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {money(MOVEMENT.current.funded_balance)}
                  </span>
                  <span style={{ fontSize: TYPE.bodySmall, color: C.ink300 }}>
                    {count(MOVEMENT.current.loan_count)} loans
                  </span>
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Run status — the manifest the pipeline wrote. */}
        <div
          style={{
            marginTop: "auto",
            display: "flex",
            gap: 12,
            opacity: fade(frame, s(21.5), s(0.8)),
          }}
        >
          {[
            {
              label: "Run status",
              value: "Completed",
              tone: "positive" as const,
            },
            {
              label: "Rows validated",
              value: count(validation.rowsValidated ?? null),
              tone: "neutral" as const,
            },
            {
              label: "Canonical validation",
              value: validation.canonicalValidation ?? "—",
              tone: "positive" as const,
            },
            {
              label: "Business-rule exceptions",
              value: `${count(validation.businessRuleExceptions ?? null)} rows`,
              tone: "warning" as const,
            },
            {
              label: "Composite key",
              value: "Unique",
              tone: "positive" as const,
            },
            {
              label: "Source portfolios",
              value: `${SUMMARY.cohorts.length} consolidated`,
              tone: "accent" as const,
            },
          ].map((tile, i) => (
            <div
              key={tile.label}
              style={{
                flex: 1,
                padding: "13px 15px",
                borderRadius: LAYOUT.radiusSmall,
                border: `1px solid ${C.line}`,
                background: "rgba(255,255,255,0.02)",
                ...enter(frame, s(21.7) + i * s(0.1), s(0.45)),
                minWidth: 0,
              }}
            >
              <div
                style={{
                  fontSize: TYPE.micro,
                  color: C.ink500,
                  letterSpacing: "0.05em",
                  textTransform: "uppercase",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {tile.label}
              </div>
              <div
                style={{
                  marginTop: 6,
                  fontSize: TYPE.bodySmall,
                  fontWeight: 600,
                  color:
                    tile.tone === "positive"
                      ? THEME.positive
                      : tile.tone === "warning"
                        ? THEME.rag.amber
                        : tile.tone === "accent"
                          ? C.peri300
                          : C.ink100,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {tile.value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Frame>
  );
};
