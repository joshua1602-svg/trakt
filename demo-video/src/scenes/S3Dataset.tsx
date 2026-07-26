/**
 * S3 · One dataset, every output — 600 frames (0:38–0:58)
 *
 * Two portfolio balances converge and resolve into the consolidated figure — the
 * scene's ONE `signal` element. Six artifact cards then fan out from it, ordered
 * regulatory-first, because the ESMA Annex 2 submission is the hardest claim in the
 * product to replicate. A single reconciliation check resolves across all six.
 *
 * Every card's metadata is read from the artefact catalogue the run wrote. A card
 * whose artefact reported itself unavailable is not rendered: the film shows what
 * the pipeline produced and nothing else.
 */

import React from "react";
import { AbsoluteFill, interpolate, interpolateColors, useCurrentFrame } from "remotion";

import {
  ArtifactCard,
  Claim,
  Counter,
  Enter,
  Figure,
  Label,
  Stat,
  useGeometry,
  useIsSquare,
} from "../components/kit";
import { ARTEFACTS, ASSERTIONS, CURRENT_PERIOD, PORTFOLIOS, SUMMARY, lens } from "../data/fixtures";
import { count, longDate, money, percent, periodLabel } from "../format";
import theme from "../theme";

/**
 * The six governed outputs, regulatory first in reading order.
 *
 * Each card leads with what the artefact IS, in plain English; the filename and the
 * figures sit beneath it in the data face. Every value is read from the artefact
 * catalogue the run wrote — a card whose artefact reported itself unavailable is not
 * rendered, so the film cannot show an output the pipeline did not produce.
 */
const cardSpecs = (): { title: string; meta: string[]; confirm?: string }[] => {
  const regulatory = ARTEFACTS.regulatoryOutput;
  const deck = ARTEFACTS.investorDeck;
  const tape = ARTEFACTS.canonicalTape;
  const validation = ARTEFACTS.validationReport;
  const risk = ARTEFACTS.riskMonitor;
  const audit = ARTEFACTS.auditManifest;

  const specs: {
    available: boolean;
    title: string;
    meta: string[];
    confirm?: string;
  }[] = [
    {
      available: Boolean(regulatory?.available),
      title: "Regulatory submission",
      meta: [
        `${regulatory?.fileName} · ESMA Annex 2 · ` +
          `${count(Number(regulatory?.exposureRecords))} exposures`,
      ],
      confirm: regulatory?.xsdValidated ? "XSD validated" : undefined,
    },
    {
      available: Boolean(deck?.available),
      title: "Investor pack",
      meta: [
        `${deck?.fileName} · ${count(Number(deck?.slides))} slides · ` +
          `${periodLabel(CURRENT_PERIOD.period)}`,
      ],
    },
    {
      available: Boolean(tape?.available),
      title: "Loan-level dataset",
      meta: [
        `${tape?.fileName} · ${count(Number(tape?.rows))} rows · ` +
          `${count(Number(tape?.columns))} fields`,
      ],
    },
    {
      available: Boolean(validation?.available),
      title: "Validation report",
      meta: [
        `validation_report.json · ` +
          `${count(Number(validation?.businessRuleExceptions))} exceptions · ` +
          `${percent(Number(validation?.exceptionRatePct), 2)}`,
      ],
    },
    {
      available: Boolean(risk?.available),
      title: "Concentration monitor",
      meta: [
        `concentration_monitor.json · ${count(Number(risk?.limitCount))} limits tested`,
      ],
    },
    {
      available: Boolean(audit?.available),
      title: "Audit trail",
      meta: [
        `audit_manifest.json · ` +
          `${count((audit?.sourceFiles as unknown[] | undefined)?.length ?? 0)} files ` +
          `hashed · SHA-256`,
      ],
    },
  ];
  return specs
    .filter((spec) => spec.available)
    .map(({ title, meta, confirm }) => ({ title, meta, confirm }));
};

export const S3Dataset: React.FC = () => {
  const frame = useCurrentFrame();
  const geometry = useGeometry();
  const isSquare = useIsSquare();

  const a = lens("A").summary.metrics.funded_balance ?? 0;
  const b = lens("B").summary.metrics.funded_balance ?? 0;
  const total = SUMMARY.metrics.funded_balance ?? 0;
  const cards = cardSpecs();
  const codes = PORTFOLIOS.map((p) => p.display_id).join(" + ");

  // 1140–1260: the two figures converge and resolve into the consolidated one.
  const converge = interpolate(frame, [0, 90], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const partsOpacity = interpolate(frame, [72, 100], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const totalOpacity = interpolate(frame, [84, 108], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // The consolidated figure rises out of centre to make room for the fan-out.
  const totalLift = interpolate(frame, [108, 138], [0, isSquare ? -178 : -196], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const spread = isSquare ? 190 : 430;

  const cardsOpacity = interpolate(frame, [120, 150, 408, 432], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const checkOpacity = interpolate(frame, [300, 324, 402, 420], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // `signal` is a scarcity resource: at most one element per frame carries it. The
  // consolidated figure holds the accent until the reconciliation check takes it,
  // and the hand-over is a cross-fade of the same token so they never overlap.
  const totalAccent = interpolate(frame, [288, 312], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const checkAccent = interpolate(frame, [288, 312], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const statOpacity = interpolate(frame, [408, 432], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const claimOpacity = interpolate(frame, [432, 456], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      {/* The two source figures, converging. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          opacity: partsOpacity,
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start" }}>
          <div style={{ transform: `translateX(${-spread * converge}px)` }}>
            <Stat
              align="center"
              scale={isSquare ? 0.5 : 0.62}
              value={money(a)}
              stamp={PORTFOLIOS[0].display_id}
            />
          </div>
          <div style={{ transform: `translateX(${spread * converge}px)` }}>
            <Stat
              align="center"
              scale={isSquare ? 0.5 : 0.62}
              value={money(b)}
              stamp={PORTFOLIOS[1].display_id}
            />
          </div>
        </div>
      </AbsoluteFill>

      {/* The consolidated figure. The scene's one signal element. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          opacity: totalOpacity * statOpacity,
        }}
      >
        <div style={{ transform: `translateY(${totalLift}px)` }}>
          <Stat
            accent={totalAccent}
            align="center"
            value={money(total)}
            stamp={
              `${count(SUMMARY.metrics.loan_count)} LOANS · ${codes} · ` +
              longDate(CURRENT_PERIOD.reportingDate).toUpperCase()
            }
          />
        </div>
      </AbsoluteFill>

      {/* Six artifact cards, regulatory first. */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: isSquare ? geometry.height * 0.3 : geometry.height * 0.35,
          display: "flex",
          justifyContent: "center",
          opacity: cardsOpacity,
        }}
      >
        {/* Three per row, two rows. The width is pinned so the sixth card cannot
            wrap onto a third row and collide with the reconciliation check. */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, max-content)",
            gap: isSquare ? theme.motion.quick : theme.motion.base,
            alignItems: "stretch",
          }}
        >
          {cards.map((card, i) => (
            <Enter key={card.title} at={126} index={i * 2}>
              <ArtifactCard title={card.title} meta={card.meta} confirm={card.confirm} />
            </Enter>
          ))}
        </div>
      </div>

      {/* One check resolving across all six. */}
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          bottom: geometry.captionReserve - (isSquare ? 22 : 12),
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: theme.motion.quick,
          opacity: checkOpacity,
        }}
      >
        <Label>Reconciliation</Label>
        <Figure
          scale={isSquare ? 0.36 : 0.44}
          color={interpolateColors(checkAccent, [0, 1], [theme.color.paper, theme.color.signal])}
        >
          <Counter
            from={0}
            to={ASSERTIONS.checksPassed}
            at={306}
            format={(v) => `${count(v)}/${count(ASSERTIONS.checksRun)} checks passed`}
          />
        </Figure>
      </div>

      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          opacity: claimOpacity,
        }}
      >
        <Enter at={432}>
          <Claim accent measure={isSquare ? 0.88 : 0.62}>
            One dataset. Every output reconciles.
          </Claim>
        </Enter>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

export default S3Dataset;
