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
 * What replaces it is a FUNNEL. Five artefact types enter across the top of the frame,
 * hold, then converge inward and downward into a single governed band. Five rows stacked
 * in a list read as a list; five tiles collapsing into one band read as a transformation,
 * which is the actual claim. Nobody looking at a warehouse agreement beside a monthly
 * cash-flow workbook thinks "column matching". Showing it is stronger than arguing it, so
 * the beat is uncaptioned and unexplained.
 *
 * Mapping does appear — once, as six words in the receipt strip, alongside the other
 * things the run produced. That is the correct weight for it.
 *
 * Beats: the clock (114) · the funnel (180) · the receipt (132) · what comes out (54).
 */

import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";

import { ARRIVING_ARTEFACTS, FUNNEL_CLAIM, ONBOARDING_HOURS } from "../claims";
import {
  Body,
  Claim,
  Counter,
  Enter,
  Label,
  Panel,
  Rule,
  Stamp,
  Stat,
  useGeometry,
  useIsSquare,
} from "../components/kit";
import {
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
 * One arriving artefact, as a tile that will later collapse into the band.
 *
 * The title is set in the BODY face, not the data face: an artefact's name is not a
 * computed value, so mono would be wrong under the film's own rule, and a tile whose
 * title a viewer skips is a tile that proves nothing. Format and frequency stay in the
 * data face beneath it.
 *
 * `converge` runs 0 -> 1 across the collapse. The tile slides toward the centre of the
 * row and down toward the band, shrinking and fading as it goes, so five things
 * visibly become one thing.
 */
const ArtefactTile: React.FC<{
  at: number;
  index: number;
  /** Signed offset from the centre of the run, in tile-plus-gap units. */
  offset: number;
  pitch: number;
  converge: number;
  /** Square stacks the five tiles into a column: five 34px titles across 1080px is not a
   *  row, it is five overflowing boxes. The collapse then runs vertically. */
  vertical: boolean;
  title: string;
  format: string;
  frequency: string;
}> = ({ at, index, offset, pitch, converge, vertical, title, format, frequency }) => {
  const slide = -offset * pitch * converge;
  return (
    <Enter at={at} index={index * 3} style={{ flex: 1, minWidth: 0 }}>
      <div
        style={{
          transform:
            (vertical
              ? `translate(0px, ${slide + 60 * converge}px) `
              : `translate(${slide}px, ${140 * converge}px) `) +
            `scale(${1 - 0.4 * converge})`,
          opacity: 1 - converge,
        }}
      >
        <Panel
          style={{
            height: vertical ? undefined : TILE_HEIGHT,
            padding: theme.motion.base,
            display: "flex",
            flexDirection: vertical ? "row" : "column",
            alignItems: vertical ? "baseline" : undefined,
            justifyContent: "space-between",
            gap: theme.motion.base,
          }}
        >
          <Body>{title}</Body>
          <Stamp>{`${format} · ${frequency}`}</Stamp>
        </Panel>
      </div>
    </Enter>
  );
};

/** Tile height in the wide row. Fixed, so five unequal titles stay one row. */
const TILE_HEIGHT = 150;

/** Rendered height of a square tile plus its gap — the distance it travels per step. */
const TILE_PITCH_SQUARE = 100;

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
  const CLOCK_HANDOVER = 102;
  // --- Beat 2 · the funnel (114-294) --------------------------------------
  const TILES_AT = 114;
  /** The collapse. Tiles have been up ~40 frames by here, which is long enough to read
   *  five titles and short enough that the frame has not gone static. */
  const CONVERGE_AT = 198;
  const BAND_AT = 222;
  const FUNNEL_CLAIM_AT = 252;
  // --- Beats 3 and 4 -------------------------------------------------------
  const RECEIPT_AT = 294;
  const REFERRAL_AT = 342;
  const GRANULARITY_AT = 426;

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
  /** 0 -> 1 across the collapse. Drives every tile's transform, so they move together. */
  const converge = interpolate(frame, [CONVERGE_AT, CONVERGE_AT + theme.motion.slow], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const tilesOpacity = interpolate(frame, [TILES_AT, TILES_AT + theme.motion.base], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const bandOpacity = interpolate(
    frame,
    [BAND_AT, BAND_AT + theme.motion.base, RECEIPT_AT - 18, RECEIPT_AT],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const funnelClaimOpacity = interpolate(
    frame,
    [FUNNEL_CLAIM_AT, FUNNEL_CLAIM_AT + theme.motion.base, RECEIPT_AT - 18, RECEIPT_AT],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const receiptOpacity = interpolate(
    frame,
    [RECEIPT_AT, RECEIPT_AT + theme.motion.base, GRANULARITY_AT - 18, GRANULARITY_AT],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
  );
  const referralOpacity = interpolate(
    frame,
    [REFERRAL_AT, REFERRAL_AT + theme.motion.base, GRANULARITY_AT - 18, GRANULARITY_AT],
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

      {/* Beat 2 · the funnel. Five artefacts in, one governed dataset out.

          The tiles are laid out as a normal flex row and then TRANSFORMED inward, rather
          than being absolutely positioned and animated to a target. The row keeps them
          equal-width and evenly pitched for free, and the transform is what makes the
          collapse read — five things arriving at one place, not five things fading. */}
      <div
        style={{
          position: "absolute",
          top: isSquare ? geometry.edge + theme.motion.slow * 2 : 130,
          left: geometry.gutter,
          right: geometry.gutter,
          display: "flex",
          flexDirection: isSquare ? "column" : "row",
          gap: theme.motion.quick,
          opacity: tilesOpacity,
        }}
      >
        {ARRIVING_ARTEFACTS.map((artefact, i) => (
          <ArtefactTile
            key={artefact.title}
            at={TILES_AT}
            index={i}
            offset={i - (ARRIVING_ARTEFACTS.length - 1) / 2}
            pitch={
              isSquare
                ? TILE_PITCH_SQUARE
                : (geometry.width - geometry.gutter * 2 + theme.motion.quick) /
                  ARRIVING_ARTEFACTS.length
            }
            vertical={isSquare}
            converge={converge}
            title={artefact.title}
            format={artefact.format}
            frequency={artefact.frequency}
          />
        ))}
      </div>

      {/* What the funnel resolves into. */}
      <div
        style={{
          position: "absolute",
          top: isSquare ? geometry.height * 0.36 : 330,
          left: geometry.gutter,
          right: geometry.gutter,
          opacity: bandOpacity,
        }}
      >
        <Panel
          style={{
            padding: geometry.gutter / 2,
            display: "flex",
            flexDirection: "column",
            gap: theme.motion.quick,
          }}
        >
          <Label tone="signal" style={{ opacity: theme.signalSoftOpacity }}>
            Governed portfolio dataset
          </Label>
          <Rule />
          <Stamp tone="paper" style={{ overflowWrap: "anywhere" }}>
            {`${count(Number(tape?.rows))} rows · ${count(Number(tape?.columns))} canonical ` +
              "fields · six provenance fields on every row"}
          </Stamp>
        </Panel>
      </div>

      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: isSquare ? geometry.height * 0.5 : 500,
          bottom: geometry.captionReserve,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          opacity: funnelClaimOpacity,
        }}
      >
        <Claim measure={isSquare ? 0.92 : 0.78}>{FUNNEL_CLAIM}</Claim>
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
