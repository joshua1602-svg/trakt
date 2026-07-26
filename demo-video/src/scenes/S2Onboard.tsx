/**
 * S2 · Disparate cuts in, one governed portfolio out — 480 frames (0:14–0:30)
 *
 * There is NO mapping animation in this scene, and there is none anywhere else in the
 * film. Field-header mapping is the weakest thing the product does and the easiest to
 * dismiss — a prospect will assume a language model already does it, and they are not
 * entirely wrong. It used to occupy the film's first "here is what Trakt does" moment,
 * which meant it defined the viewer's model of the product no matter how few seconds it
 * ran for. Position mattered more than duration, so it is gone rather than shortened.
 *
 * The scene never raises the objection either. A line rebutting it would keep mapping as
 * the topic.
 *
 * What replaces it is what actually arrives: five artefact types, from five owners, on
 * three reporting cycles. Nobody looking at a trustee's quarterly PDF beside a servicer's
 * monthly CSV thinks "column matching". Showing it is stronger than arguing it, so the
 * beat is uncaptioned and unexplained.
 *
 * Mapping does appear — once, as six words in the receipt strip, alongside the other
 * things the run produced. That is the correct weight for it.
 *
 * Beats: the clock (120) · what arrives (150) · the receipt (150) · what comes out (60).
 */

import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";

import { ONBOARDING_HOURS } from "../claims";
import {
  Body,
  Counter,
  Enter,
  Headline,
  Label,
  Panel,
  Rule,
  Stamp,
  Stat,
  useGeometry,
  useIsSquare,
} from "../components/kit";
import {
  ARRIVALS,
  ASSERTIONS,
  ARTEFACTS,
  CURRENT_PERIOD,
  PORTFOLIOS,
  onboarding,
  portfolio,
} from "../data/fixtures";
import { count, hoursMinutes } from "../format";
import theme from "../theme";

/**
 * The referral card body, condensed to two lines.
 *
 * The authoritative text is `mappingDecisions[].note` in the demo manifest — two full
 * sentences, written for the record rather than for a viewer with four seconds. This is
 * the same reasoning at reading speed. A unit test asserts every term below still
 * appears in the fixture note, so the condensation cannot drift away from what the
 * platform actually recorded.
 *
 * The note itself ends "intentionally not mapped to a canonical balance field", which is
 * exactly right for a record and wrong for this card: the film puts the word "mapped" on
 * screen once, in the receipt strip, and a second use here would double its weight for
 * no gain. "Not counted twice" is the same fact in the language a COO uses.
 */
const REFERRAL_LINES = [
  "Reserve-facility drawdown taken this month.",
  "Already inside Principal Outstanding — not counted twice.",
];

/**
 * One arriving cut: what it is, then format, owner and frequency.
 *
 * The title is set in the BODY face, not the data face. Two reasons, and the second is
 * the one that matters: an artefact's name is not a computed value, so mono would be
 * wrong under the film's own rule; and five rows of 15px monospace is a table a viewer
 * skips. Whether these five rows are read is the whole beat — if they are not, the scene
 * has an unsupported claim over an unreadable list. So the title carries the weight and
 * the three metadata columns stay in the data face beside it, exactly as an artifact
 * card is arranged.
 */
const ArrivalRow: React.FC<{
  at: number;
  index: number;
  title: string;
  format: string;
  owner: string;
  frequency: string;
}> = ({ at, index, title, format, owner, frequency }) => {
  const isSquare = useIsSquare();
  return (
    <Enter at={at} index={index * 2}>
      <div style={{ display: "flex", alignItems: "baseline", gap: theme.motion.base }}>
        <Body
          style={{
            width: isSquare ? 300 : 400,
            flexShrink: 0,
            fontSize: theme.type.body.fontSize * (isSquare ? 0.72 : 0.86),
          }}
        >
          {title}
        </Body>
        <Stamp style={{ width: isSquare ? 60 : 70, flexShrink: 0 }}>{format}</Stamp>
        <Stamp style={{ width: isSquare ? 220 : 260, flexShrink: 0 }}>{owner}</Stamp>
        <Stamp>{frequency}</Stamp>
      </div>
    </Enter>
  );
};

/** One level of the granularity claim. */
const GranularityRow: React.FC<{ at: number; index: number; level: string; detail: string }> = ({
  at,
  index,
  level,
  detail,
}) => {
  const isSquare = useIsSquare();
  return (
    <Enter at={at} index={index * 3}>
      <div style={{ display: "flex", alignItems: "baseline", gap: theme.motion.base }}>
        <Label style={{ width: isSquare ? 190 : 240, flexShrink: 0 }}>{level}</Label>
        <Stamp tone="paper">{detail}</Stamp>
      </div>
    </Enter>
  );
};

export const S2Onboard: React.FC = () => {
  const frame = useCurrentFrame();
  const geometry = useGeometry();
  const isSquare = useIsSquare();

  const book = portfolio("B");
  const mapping = onboarding("B").mapping;
  const tape = ARTEFACTS.canonicalTape;
  const referred = onboarding("B").mappingDecisions.find(
    (d) => !d.canonical_field && d.source_header !== "Synthetic Data Notice",
  );

  // Beat boundaries, scene-relative.
  const CLOCK_HANDOVER = 108;
  const ARRIVALS_AT = 120;
  const ARRIVALS_CLAIM_AT = 210;
  const RECEIPT_AT = 270;
  const REFERRAL_AT = 330;
  const GRANULARITY_AT = 420;

  const clockOpacity = interpolate(
    frame,
    [CLOCK_HANDOVER, CLOCK_HANDOVER + theme.motion.quick],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const stampOpacity = interpolate(
    frame,
    [CLOCK_HANDOVER + theme.motion.quick, CLOCK_HANDOVER + theme.motion.base],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const arrivalsOpacity = interpolate(
    frame,
    [ARRIVALS_AT, ARRIVALS_AT + theme.motion.base, 252, 270],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const arrivalsClaimOpacity = interpolate(
    frame,
    [ARRIVALS_CLAIM_AT, ARRIVALS_CLAIM_AT + theme.motion.base, 252, 270],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const receiptOpacity = interpolate(
    frame,
    [RECEIPT_AT, RECEIPT_AT + theme.motion.base, 402, 420],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const referralOpacity = interpolate(
    frame,
    [REFERRAL_AT, REFERRAL_AT + theme.motion.base, 402, 420],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const granularityOpacity = interpolate(
    frame,
    [GRANULARITY_AT, GRANULARITY_AT + theme.motion.base],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );

  return (
    <AbsoluteFill>
      {/* Beat 1 · the clock. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          opacity: clockOpacity,
        }}
      >
        <Stat
          align="center"
          value={
            <Counter
              from={0}
              to={ONBOARDING_HOURS}
              at={0}
              frames={theme.motion.slow + theme.motion.base}
              format={hoursMinutes}
            />
          }
          stamp={
            `HOURS ELAPSED · DATA RECEIVED → GOVERNED OUTPUT · ` +
            `${count(PORTFOLIOS.length)} PORTFOLIOS · ` +
            `${CURRENT_PERIOD.reportingDate}`
          }
        />
      </AbsoluteFill>

      <div
        style={{
          position: "absolute",
          top: geometry.edge,
          right: geometry.edge,
          textAlign: "right",
          opacity: stampOpacity,
        }}
      >
        <Stamp>{`${hoursMinutes(ONBOARDING_HOURS)} HRS · DATA RECEIVED → GOVERNED OUTPUT`}</Stamp>
      </div>

      {/* Beat 2 · what arrives. Five formats, five owners, three frequencies. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve + (isSquare ? 100 : 110),
          opacity: arrivalsOpacity,
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: theme.motion.quick }}>
          {ARRIVALS.map((arrival, i) => (
            <ArrivalRow
              key={arrival.title}
              at={ARRIVALS_AT}
              index={i}
              title={arrival.title}
              format={arrival.format}
              owner={arrival.owner}
              frequency={arrival.frequency}
            />
          ))}
        </div>
      </AbsoluteFill>

      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          bottom: geometry.captionReserve + (isSquare ? 10 : 40),
          display: "flex",
          justifyContent: "center",
          opacity: arrivalsClaimOpacity,
        }}
      >
        {/* HEADLINE, not display. The display size is for a line that owns the whole
            frame; this one sits under five rows of evidence and has to be the smaller
            voice of the two, or the slogan buries the argument it is drawing. */}
        <Headline measure={isSquare ? 0.94 : 0.66}>
          Disparate portfolio cuts in. One governed portfolio out.
        </Headline>
      </div>

      {/* Beat 3 · the receipt, with the referred item beneath it.

          One centred column holds both, and the card stays in the layout from the first
          frame of the beat at opacity 0. That is deliberate: if the card were positioned
          separately the strip would sit alone in the middle of an empty frame for two
          seconds and then acquire a card 200px below it. Reserving the space means the
          composition is settled before anything is legible, and the card resolves into a
          gap the viewer has already accepted.

          The strip is a 3x2 grid, not a wrapped row. Six mono items on one line across
          1,350px reads as a ticker; three columns with the counts left-aligned reads as
          what it is, which is a receipt. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: theme.motion.slow * 2,
          paddingBottom: geometry.captionReserve,
          paddingLeft: geometry.gutter,
          paddingRight: geometry.gutter,
        }}
      >
        <div
          style={{
            display: "grid",
            // Same width and the same three columns as the card below, so the receipt and
            // the referral read as one block rather than two things that happen to be
            // stacked. `1fr` rather than `max-content` for the same reason: even columns.
            width: isSquare ? "100%" : 940,
            gridTemplateColumns: isSquare ? "repeat(2, 1fr)" : "repeat(3, 1fr)",
            columnGap: theme.motion.base,
            rowGap: theme.motion.base,
            justifyItems: "start",
            opacity: receiptOpacity,
          }}
        >
          {[
            { text: `${count(mapping.mapped_count)} fields mapped`, flag: false },
            { text: `${count(mapping.client_contract_count)} client-specific decisions`, flag: false },
            { text: `${count(mapping.unmapped_count)} referred for review`, flag: true },
            { text: `${count(Number(tape?.columns))} canonical fields`, flag: false },
            { text: `${count(Number(tape?.rows))} rows typed and validated`, flag: false },
            { text: `${count(ASSERTIONS.checksRun)} reconciliation checks`, flag: false },
          ].map((item, i) => (
            <Enter key={item.text} at={RECEIPT_AT} index={i * 2}>
              <Stamp tone={item.flag ? "flag" : "mute"}>{item.text}</Stamp>
            </Enter>
          ))}
        </div>

        {referred ? (
          <Panel
            style={{
              padding: geometry.gutter / 2,
              width: isSquare ? "100%" : 940,
              display: "flex",
              flexDirection: "column",
              gap: theme.motion.quick,
              opacity: referralOpacity,
            }}
          >
            <Label tone="flag">Referred for review</Label>
            <Stamp tone="paper">{`${book.display_id} · ${referred.source_header}`}</Stamp>
            <Rule tone="flag" />
            {REFERRAL_LINES.map((line) => (
              <Body key={line} tone="mute" style={{ fontSize: theme.type.body.fontSize * 0.8 }}>
                {line}
              </Body>
            ))}
          </Panel>
        ) : null}
      </AbsoluteFill>

      {/* Beat 4 · what comes out. Three levels of granularity, then the claim. */}
      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: theme.motion.slow,
          paddingBottom: geometry.captionReserve,
          opacity: granularityOpacity,
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: theme.motion.base }}>
          <GranularityRow
            at={GRANULARITY_AT}
            index={0}
            level="Loan level"
            detail={
              `${count(Number(tape?.rows))} rows · ${count(Number(tape?.columns))} canonical ` +
              "fields · six provenance fields each"
            }
          />
          <GranularityRow
            at={GRANULARITY_AT}
            index={1}
            level="Portfolio level"
            detail={`${count(PORTFOLIOS.length)} portfolios · separately governed`}
          />
          <GranularityRow
            at={GRANULARITY_AT}
            index={2}
            level="Sponsor level"
            detail="one aggregate view"
          />
        </div>
        <Enter at={GRANULARITY_AT + theme.motion.base}>
          <Body style={{ textAlign: "center" }}>
            Granular enough for audit. Aggregated enough for the board.
          </Body>
        </Enter>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

export default S2Onboard;
