/**
 * Scene 7 — Outputs.
 *
 * The governed outputs the recurring run actually produced, as elegant document
 * cards. Each card's figures come from the artefact catalogue: row counts, slide
 * counts, file sizes, exception counts, risk-limit test results.
 *
 * An artefact the run could NOT produce is not shown. The ESMA Annex 2 projection
 * is the case in point: it requires a confirmed enum-review pass that this
 * demonstration does not run, so the catalogue records it as unavailable and this
 * scene omits it rather than implying a capability that did not execute.
 */

import React from "react";
import { useCurrentFrame } from "remotion";
import { Frame } from "../components/Frame";
import { DocGlyph, Eyebrow, Pill, Tick } from "../components/primitives";
import { C, LAYOUT, THEME, TYPE } from "../design/tokens";
import { enter, fade, s } from "../design/motion";
import { ARTEFACTS, ASSERTIONS, CURRENT_PERIOD, MOVEMENT, SUMMARY } from "../data/fixtures";
import { count, fileSize, longDate, money, percent } from "../design/format";
import type { Caption } from "../timeline";

interface CardSpec {
  title: string;
  format: string;
  lines: string[];
  tone: "positive" | "accent" | "warning" | "neutral";
}

/** Build the card list from the artefact catalogue — availability decides inclusion. */
const buildCards = (): CardSpec[] => {
  const cards: CardSpec[] = [];

  const tape = ARTEFACTS.canonicalTape;
  if (tape?.available) {
    cards.push({
      title: String(tape.title ?? "Consolidated canonical tape"),
      format: String(tape.format ?? "CSV"),
      lines: [
        `${count(Number(tape.rows ?? 0))} loans · ${count(Number(tape.columns ?? 0))} canonical fields`,
        `${money(MOVEMENT.current.funded_balance)} across ${SUMMARY.cohorts.length} source portfolios`,
        `Composite key: source portfolio + loan identifier`,
      ],
      tone: "accent",
    });
  }

  const validation = ARTEFACTS.validationReport;
  if (validation?.available) {
    const rate = validation.exceptionRatePct as number | null;
    cards.push({
      title: String(validation.title ?? "Validation report"),
      format: String(validation.format ?? "CSV + JSON"),
      lines: [
        `${count(Number(validation.rowsValidated ?? 0))} rows validated`,
        `Canonical validation: ${String(validation.canonicalValidation ?? "—")}`,
        `${count(Number(validation.businessRuleExceptions ?? 0))} business-rule exceptions` +
          (rate != null ? ` (${percent(rate, 2)} of rows)` : ""),
      ],
      tone: "warning",
    });
  }

  cards.push({
    title: "MI dashboard",
    format: "React · Trakt MI Agent",
    lines: [
      `Consolidated and per-portfolio views`,
      `${count(MOVEMENT.current.loan_count)} loans at ${longDate(CURRENT_PERIOD.reportingDate)}`,
      `Deterministic natural-language MI`,
    ],
    tone: "accent",
  });

  const deck = ARTEFACTS.investorDeck;
  if (deck?.available) {
    cards.push({
      title: String(deck.title ?? "Investor pack"),
      format: String(deck.format ?? "PowerPoint"),
      lines: [
        `${count(Number(deck.slides ?? 0))} slides · ${fileSize(Number(deck.sizeBytes ?? 0))}`,
        `Reporting period ${String(deck.reportingPeriod ?? "")}`,
        `Every figure equals the dashboard`,
      ],
      tone: "positive",
    });
  }

  const risk = ARTEFACTS.riskMonitor;
  if (risk?.available) {
    const tested = Number(risk.limitCount ?? 0);
    const passed = Number(risk.testsPassed ?? 0);
    const breaches = Number(risk.breaches ?? 0);
    const largest = risk.largestConcentration as
      | { label?: string; actualValue?: number }
      | undefined;
    cards.push({
      title: String(risk.title ?? "Concentration risk monitor"),
      format: String(risk.format ?? "JSON"),
      lines: [
        `${count(tested)} concentration limits tested`,
        `${count(passed)} within limit · ${count(breaches)} requiring attention`,
        largest?.label
          ? `Largest: ${largest.label} at ${percent(largest.actualValue ?? null)}`
          : `Source: ${String(risk.limitsSource ?? "governing document")}`,
      ],
      tone: breaches > 0 ? "warning" : "positive",
    });
  }

  const audit = ARTEFACTS.auditManifest;
  if (audit?.available) {
    const files = (audit.sourceFiles ?? []) as unknown[];
    const outputs = (audit.assembledOutputs ?? []) as unknown[];
    cards.push({
      title: String(audit.title ?? "Audit manifest"),
      format: String(audit.format ?? "JSON"),
      lines: [
        `${count(files.length)} source files, content-hashed`,
        `${count(outputs.length)} assembled outputs with SHA-256`,
        `Six provenance fields on every loan`,
      ],
      tone: "positive",
    });
  }

  return cards;
};

const ArtefactCard: React.FC<{ spec: CardSpec; frame: number; delay: number }> = ({
  spec,
  frame,
  delay,
}) => {
  const accent =
    spec.tone === "positive"
      ? THEME.positive
      : spec.tone === "warning"
        ? THEME.rag.amber
        : C.peri400;
  return (
    <div
      style={{
        background: C.surfaceDashboard,
        border: `1px solid ${C.line}`,
        borderRadius: LAYOUT.radius,
        padding: "20px 22px",
        display: "flex",
        flexDirection: "column",
        gap: 14,
        justifyContent: "flex-start",
        position: "relative",
        overflow: "hidden",
        minWidth: 0,
        ...enter(frame, delay, s(0.65), 12),
      }}
    >
      <div
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          bottom: 0,
          width: 3,
          background: accent,
        }}
      />
      <div style={{ display: "flex", alignItems: "flex-start", gap: 13 }}>
        <DocGlyph size={25} colour={accent} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontSize: TYPE.body,
              fontWeight: 600,
              color: C.ink100,
              letterSpacing: "-0.01em",
            }}
          >
            {spec.title}
          </div>
          <div style={{ marginTop: 3, fontSize: TYPE.micro, color: C.ink500 }}>
            {spec.format}
          </div>
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
        {spec.lines.map((line) => (
          <div
            key={line}
            style={{
              display: "flex",
              gap: 9,
              alignItems: "flex-start",
              fontSize: TYPE.bodySmall,
              color: C.ink300,
              lineHeight: 1.42,
            }}
          >
            <span style={{ marginTop: 4, flexShrink: 0 }}>
              <Tick size={11} colour={accent} />
            </span>
            <span>{line}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export const Scene7Outputs: React.FC<{ captions: Caption[] }> = ({ captions }) => {
  const frame = useCurrentFrame();
  const cards = buildCards();

  return (
    <Frame captions={captions} chapter="Governed outputs">
      <div style={{ display: "flex", flexDirection: "column", gap: 22, flex: 1, minHeight: 0 }}>
        {/* One dataset at the head of the fan-out. */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 18,
            padding: "16px 22px",
            borderRadius: LAYOUT.radius,
            border: `1px solid ${C.navy500}`,
            background: "rgba(145,157,209,0.075)",
            ...enter(frame, s(0.2), s(0.6)),
          }}
        >
          <div>
            <Eyebrow color={C.peri300}>Single governed dataset</Eyebrow>
            <div
              style={{
                marginTop: 5,
                fontSize: TYPE.subheading,
                fontWeight: 600,
                color: C.ink100,
                letterSpacing: "-0.015em",
              }}
            >
              Consolidated platform canonical · {longDate(CURRENT_PERIOD.reportingDate)}
            </div>
          </div>
          <div style={{ flex: 1 }} />
          <Pill tone="positive">
            <Tick size={11} />
            {ASSERTIONS.checksPassed}/{ASSERTIONS.checksRun} reconciliation checks passed
          </Pill>
        </div>

        {/* The outputs. */}
        <div
          style={{
            flex: 1,
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            // Content-sized rows: a card that is mostly empty space reads as a
            // layout accident, so the grid takes only the height it needs and the
            // provenance line below absorbs the remainder.
            gridAutoRows: "min-content",
            alignContent: "start",
            gap: 18,
            minHeight: 0,
          }}
        >
          {cards.map((spec, i) => (
            <ArtefactCard key={spec.title} spec={spec} frame={frame} delay={s(0.9) + i * s(0.24)} />
          ))}
        </div>

        <div
          style={{
            marginTop: "auto",
            display: "flex",
            alignItems: "center",
            gap: 14,
            opacity: fade(frame, s(13.5), s(0.8)),
          }}
        >
          <Eyebrow>Provenance</Eyebrow>
          <span style={{ fontSize: TYPE.bodySmall, color: C.ink400 }}>
            Every output carries the source portfolio, the reporting date and the
            content hash of the file it was derived from.
          </span>
        </div>
      </div>
    </Frame>
  );
};
