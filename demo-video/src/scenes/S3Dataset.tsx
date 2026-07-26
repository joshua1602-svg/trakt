/**
 * S3 · Three portfolios, one sponsor view, every output — 780 frames (0:30–0:56)
 *
 * Two scopes coexist in this scene and confusing them is the main risk in the film:
 *
 *   SPONSOR  — £2.81bn / 15,215 loans. The consolidation beat ONLY. It is the pitch: a
 *              number the sponsor cannot produce from any one system today, because
 *              their own ledger has derecognised the deal they sold, the servicer sees
 *              only its own population, and the investor reports are per-deal.
 *   PLATFORM — £1.96bn / 11,035 loans. Everything after the cut, without exception:
 *              the fan-out, every MI figure, the movement, the regional split.
 *
 * The scene is built so the two cannot be mixed up by accident — the sponsor figure is
 * read from `SPONSOR_SCOPE` and the rest from `PLATFORM_SCOPE`, and a unit test asserts
 * the sponsor figure appears in exactly one scene file.
 *
 * Beats: consolidation (300) · platform figure, fan-out and reconciliation (480).
 */

import React from "react";
import { AbsoluteFill, interpolate, interpolateColors, useCurrentFrame } from "remotion";

import {
  ArtifactCard,
  Claim,
  Counter,
  Enter,
  Figure,
  Headline,
  Label,
  Rule,
  Stamp,
  Stat,
  useGeometry,
  useIsSquare,
} from "../components/kit";
import {
  ARTEFACTS,
  ASSERTIONS,
  CURRENT_PERIOD,
  PLATFORM_SCOPE,
  SPONSOR_SCOPE,
  portfolioLanes,
} from "../data/fixtures";
import { count, longDate, money, percent, periodLabel } from "../format";
import theme from "../theme";

/** Where the consolidation beat ends and the platform beat begins. */
const PLATFORM_AT = 300;

/**
 * The six governed outputs, regulatory first in reading order.
 *
 * Each card leads with what the artefact IS, in plain English, and carries a SCOPE
 * prefix — which makes the per-portfolio point without a beat of its own, and is
 * domain-correct: Annex 2 is deal-level reporting, so it belongs to SPV1, while the
 * consolidated dataset and its checks are the platform's. The audit trail is ALL,
 * because it covers every portfolio governed through the gates.
 */
const cardSpecs = (): { title: string; scope: string; meta: string; confirm?: string }[] => {
  const regulatory = ARTEFACTS.regulatoryOutput;
  const deck = ARTEFACTS.investorDeck;
  const tape = ARTEFACTS.canonicalTape;
  const validation = ARTEFACTS.validationReport;
  const risk = ARTEFACTS.riskMonitor;
  const audit = ARTEFACTS.auditManifest;

  const specs = [
    {
      available: Boolean(regulatory?.available),
      title: "Regulatory submission",
      scope: "SPV1",
      meta: `${regulatory?.fileName} · ESMA Annex 2`,
      confirm: regulatory?.xsdValidated ? "XSD validated" : undefined,
    },
    {
      available: Boolean(deck?.available),
      title: "Investor pack",
      scope: "SPV1",
      meta:
        `${deck?.fileName} · ${count(Number(deck?.slides))} slides · ` +
        `${periodLabel(CURRENT_PERIOD.period)}`,
    },
    {
      available: Boolean(tape?.available),
      title: "Consolidated dataset",
      scope: "PLATFORM",
      meta:
        `${tape?.fileName} · ${count(Number(tape?.rows))} rows · ` +
        `${count(Number(tape?.columns))} fields`,
    },
    {
      available: Boolean(validation?.available),
      title: "Validation report",
      scope: "PLATFORM",
      meta:
        `validation_report.json · ` +
        `${count(Number(validation?.businessRuleExceptions))} exceptions · ` +
        `${percent(Number(validation?.exceptionRatePct), 2)}`,
    },
    {
      available: Boolean(risk?.available),
      title: "Concentration monitor",
      scope: "PLATFORM",
      meta: `concentration_monitor.json · ${count(Number(risk?.limitCount))} limits tested`,
    },
    {
      available: Boolean(audit?.available),
      title: "Audit trail",
      scope: "ALL",
      meta: "audit_manifest.json · content-hashed · SHA-256",
    },
  ];
  return specs
    .filter((spec) => spec.available)
    .map(({ title, scope, meta, confirm }) => ({ title, scope, meta, confirm }));
};

/** One portfolio lane in the consolidation beat. */
const Lane: React.FC<{
  at: number;
  index: number;
  displayId: string;
  status: string;
  statusDetail: string;
  balance: number;
  loans: number;
}> = ({ at, index, displayId, status, statusDetail, balance, loans }) => {
  const geometry = useGeometry();
  const isSquare = useIsSquare();
  const frame = useCurrentFrame();
  const start = at + index * theme.motion.stagger * 6;
  // The lanes enter from the left, which is what makes them read as inbound.
  const slide = interpolate(frame, [start, start + theme.motion.base], [-40, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  return (
    <Enter at={start} style={{ width: "100%" }}>
      <div
        style={{
          transform: `translateX(${slide}px)`,
          display: "flex",
          flexDirection: "column",
          gap: theme.motion.quick / 2,
          width: "100%",
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: theme.motion.base }}>
          <Stamp tone="paper" style={{ width: isSquare ? 230 : 330, flexShrink: 0 }}>
            {displayId}
          </Stamp>
          <Stamp style={{ width: isSquare ? 210 : 300, flexShrink: 0 }}>{status}</Stamp>
          <Figure scale={isSquare ? 0.22 : 0.3} style={{ width: isSquare ? 150 : 210 }}>
            {money(balance)}
          </Figure>
          <Stamp>{`${count(loans)} loans`}</Stamp>
        </div>
        <Stamp>{statusDetail}</Stamp>
        <Rule width={geometry.width * (isSquare ? 0.84 : 0.72)} />
      </div>
    </Enter>
  );
};

export const S3Dataset: React.FC = () => {
  const frame = useCurrentFrame();
  const geometry = useGeometry();
  const isSquare = useIsSquare();

  const lanes = portfolioLanes();
  const cards = cardSpecs();

  // --- Beat 1 · consolidation (0–300) ------------------------------------
  const lanesOpacity = interpolate(frame, [0, theme.motion.base, 186, 210], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const sponsorOpacity = interpolate(frame, [198, 222, 288, PLATFORM_AT], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const sponsorClaimOpacity = interpolate(frame, [240, 264, 288, PLATFORM_AT], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // --- Beat 2 · the platform figure and its outputs (300–780) -------------
  const rel = frame - PLATFORM_AT;
  const converge = interpolate(rel, [0, 90], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const partsOpacity = interpolate(rel, [0, 18, 72, 100], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const totalOpacity = interpolate(rel, [84, 108], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const totalLift = interpolate(rel, [108, 138], [0, isSquare ? -178 : -196], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const spread = isSquare ? 190 : 430;
  const cardsOpacity = interpolate(rel, [120, 150, 372, 396], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const checkOpacity = interpolate(rel, [264, 288, 366, 384], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // `signal` hand-over: the platform figure holds the accent until the reconciliation
  // check takes it, and the check clears before the claim takes it. Never two at once.
  const totalAccent = interpolate(rel, [252, 276], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const checkAccent = interpolate(rel, [252, 276], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const statOpacity = interpolate(rel, [372, 396], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const claimOpacity = interpolate(rel, [396, 420], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const platformTotal = PLATFORM_SCOPE.fundedBalance;
  const codes = PLATFORM_SCOPE.portfolios.join(" + ");

  return (
    <AbsoluteFill>
      {/* Beat 1 · three portfolios. One sold, two held. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          paddingLeft: geometry.gutter,
          paddingRight: geometry.gutter,
          opacity: lanesOpacity,
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            // Two `slow` between lanes, not one. Three lanes at the tight gap occupy a
            // third of the frame and read as a caption-sized fragment floating in it;
            // opening them up makes the beat look like the subject of the shot.
            gap: theme.motion.slow * 2,
            width: geometry.width * (isSquare ? 0.84 : 0.72),
          }}
        >
          {lanes.map((lane, i) => (
            <Lane
              key={lane.displayId}
              at={0}
              index={i}
              displayId={lane.displayId}
              status={lane.status}
              statusDetail={lane.statusDetail}
              balance={lane.fundedBalance}
              loans={lane.loanCount}
            />
          ))}
        </div>
      </AbsoluteFill>

      {/* The sponsor view. The ONLY place this scope appears in the film. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve + (isSquare ? 140 : 180),
          opacity: sponsorOpacity,
        }}
      >
        <Stat
          accent
          align="center"
          value={money(SPONSOR_SCOPE.fundedBalance)}
          stamp={
            `${count(SPONSOR_SCOPE.loanCount)} LOANS · ` +
            `${count(SPONSOR_SCOPE.portfolios.length)} PORTFOLIOS · ` +
            longDate(CURRENT_PERIOD.reportingDate).toUpperCase()
          }
        />
      </AbsoluteFill>

      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          bottom: geometry.captionReserve + (isSquare ? 20 : 60),
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: theme.motion.base,
          opacity: sponsorClaimOpacity,
        }}
      >
        {/* HEADLINE, not display. The display size takes three lines at this measure and
            collides with the figure's own provenance stamp; more importantly the £2.81bn
            is the thing being proven here, so the line under it has to be the second
            voice. */}
        <Headline measure={isSquare ? 0.94 : 0.62}>
          Sold, held, or on its way to the next deal. One view.
        </Headline>
        <Enter at={264}>
          <Stamp>No single system gives you this number today.</Stamp>
        </Enter>
      </div>

      {/* Beat 2 · back to PLATFORM scope for everything that follows. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          opacity: partsOpacity,
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start" }}>
          {lanes
            .filter((lane) => PLATFORM_SCOPE.portfolios.includes(lane.displayId))
            .map((lane, i) => (
              <div
                key={lane.displayId}
                style={{ transform: `translateX(${(i === 0 ? -spread : spread) * converge}px)` }}
              >
                <Stat
                  align="center"
                  scale={isSquare ? 0.5 : 0.62}
                  value={money(lane.fundedBalance)}
                  stamp={lane.displayId}
                />
              </div>
            ))}
        </div>
      </AbsoluteFill>

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
            value={money(platformTotal)}
            stamp={
              `${count(PLATFORM_SCOPE.loanCount)} LOANS · ${codes} · ` +
              longDate(CURRENT_PERIOD.reportingDate).toUpperCase()
            }
          />
        </div>
      </AbsoluteFill>

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
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, max-content)",
            gap: isSquare ? theme.motion.quick : theme.motion.base,
            alignItems: "stretch",
          }}
        >
          {cards.map((card, i) => (
            <Enter key={card.title} at={PLATFORM_AT + 126} index={i * 2}>
              <ArtifactCard
                title={card.title}
                scope={card.scope}
                meta={[card.meta]}
                confirm={card.confirm}
              />
            </Enter>
          ))}
        </div>
      </div>

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
            at={PLATFORM_AT + 270}
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
        <Enter at={PLATFORM_AT + 396}>
          <Claim accent measure={isSquare ? 0.88 : 0.62}>
            One dataset. Every output reconciles.
          </Claim>
        </Enter>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

export default S3Dataset;
