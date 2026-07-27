/**
 * S4 · Three ways in — 660 frames (0:56–1:18)
 *
 * The strongest twenty seconds in the film, and the only place the product is shown
 * rather than described. Three panels, each holding REAL content — a scheduled artifact
 * drop, a Copilot thread with a question and an answer, and a workspace query with a
 * chart. The panels used to hold three monospace stubs, which read as three near-empty
 * boxes and threw away the whole beat.
 *
 * Then the payload: the consolidated balance counts up SIMULTANEOUSLY in a result band
 * beneath all three panels, on one baseline, landing on the same frame. Simultaneity is
 * the argument; if the three counters land on different frames the beat is dead. They
 * share one `at` and one `<Counter>` element, so they cannot drift.
 *
 * On the Microsoft 365 Copilot panel — the mark, and why this is a wordmark.
 *
 * Microsoft Legal's trademark page states that "our logos, app and product icons,
 * illustrations, photographs, videos, and designs can never be used without an express
 * license". This project holds no such licence, and no licensed asset exists in
 * `public/brand/` to render. The app icon is therefore out, and the compliant identifier
 * is the product NAME as a wordmark, which is what the panel carries: set in the body
 * face at full contrast, at the head of the panel, as the most prominent text in it.
 *
 * The same page permits truthfully naming a Microsoft product in text, and the panel
 * meets the conditions it sets: "Microsoft" precedes the product name, the name is
 * unaltered and unabbreviated, it is used adjectivally ahead of the panel's content, no
 * endorsement or affiliation is implied, Trakt's own lockup is the most prominent brand
 * on screen, and the trademark attribution sits in the result band below.
 *
 * If a licence is obtained: drop the official asset into `public/brand/`, render it with
 * <Img src={staticFile(...)} /> beside `MICROSOFT_CHANNEL`, and honour the minimum size
 * and clear space the licence specifies.
 *
 * On `signal`: the film's general rule is one accented element per frame. This scene is
 * the stated exception — the three figures are one value proven in three places, which
 * is the point of the scene, so they carry it together.
 */

import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

import { CHANNEL_USE, COPILOT_ASK, MICROSOFT_CHANNEL, MS_TRADEMARK_NOTICE } from "../claims";
import {
  Body,
  Claim,
  Counter,
  Enter,
  Figure,
  Inline,
  Panel,
  Rule,
  Stamp,
  useGeometry,
  useIsSquare,
} from "../components/kit";
import { COPILOT, CURRENT_PERIOD, MOVEMENT, SUMMARY } from "../data/fixtures";
import { money, noBreakSigns, percent, periodLabel, shortDate, signedMoney } from "../format";
import theme from "../theme";

// --------------------------------------------------------------------------- //
// Beat boundaries
// --------------------------------------------------------------------------- //
/** Panel content enters here, staggered across the three. */
const CONTENT_AT = 84;
/** The workspace bars grow from here. The only motion in the frame besides the count. */
const BARS_AT = 150;
/** The payload. ONE `at`, shared by all three panels. */
const PAYLOAD_AT = 312;
const MOVEMENT_AT = 360;
const CLAIM_AT = 510;

// --------------------------------------------------------------------------- //
// Panel 1 · Managed service
// --------------------------------------------------------------------------- //
/**
 * What lands in the folder, and when.
 *
 * Three filenames and a delivery stamp. The filenames are the artefacts the run actually
 * produced — `scripts/lint-theme.mjs` cannot check that, but `timeline.test.ts` asserts
 * each one against `artefact_catalogue.json`.
 */
const ManagedPanel: React.FC<{ at: number }> = ({ at }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: theme.motion.quick }}>
    {["annex2_submission.xml", "investor_pack.pptx", "platform_canonical_typed.csv"].map(
      (name, i) => (
        <Enter key={name} at={at} index={i * 2}>
          <Stamp tone="paper">{name}</Stamp>
        </Enter>
      ),
    )}
    <Enter at={at} index={8} style={{ paddingTop: theme.motion.quick }}>
      <Rule />
    </Enter>
    <Enter at={at} index={10}>
      <Stamp>DELIVERED 07:00 · FIRST BUSINESS DAY</Stamp>
    </Enter>
  </div>
);

// --------------------------------------------------------------------------- //
// Panel 2 · Microsoft 365 Copilot
// --------------------------------------------------------------------------- //
/**
 * A chat thread: the question right-aligned, the answer on a light plate.
 *
 * The answer is condensed from `copilot.answers[1]` in the metrics fixture — the recorded
 * turn for this question, written for the record rather than for a viewer with six
 * seconds. A unit test asserts every figure and phrase below still appears in that
 * recorded answer, so the condensation cannot drift from what the agent actually said.
 *
 * The two figures are wrapped in <Inline>, because the mono role is semantic and holds
 * inside a sentence.
 */
const CopilotPanel: React.FC<{ at: number; isSquare: boolean }> = ({ at, isSquare }) => {
  const movement = MOVEMENT.delta.funded_balance ?? 0;
  const ltv = SUMMARY.metrics.wa_ltv_points ?? 0;
  const region = MOVEMENT.primaryRegion?.region ?? "";
  const scale = isSquare ? 0.72 : 0.82;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: theme.motion.quick }}>
      {/* The user turn. Right-aligned, no plate — it is the human, not the product. */}
      <Enter at={at} style={{ display: "flex", justifyContent: "flex-end" }}>
        <Body
          style={{
            fontSize: theme.type.body.fontSize * scale,
            textAlign: "right",
            maxWidth: "88%",
          }}
        >
          {COPILOT_ASK}
        </Body>
      </Enter>

      {/* The tool call. A check state, so `signal` at the soft strength — never full. */}
      <Enter at={at} index={4}>
        <Stamp tone="signal" style={{ opacity: theme.signalSoftOpacity }}>
          {`Called ${COPILOT.actions[0]} ✓`}
        </Stamp>
      </Enter>

      {/* The answer, on a light plate. */}
      <Enter at={at} index={7}>
        <div
          style={{
            backgroundColor: theme.color.paper,
            borderRadius: theme.radius.plate,
            padding: theme.motion.base,
          }}
        >
          <Body tone="ink" style={{ fontSize: theme.type.body.fontSize * scale }}>
            {`Funded balances up `}
            <Inline>{money(movement)}</Inline>
            {`, driven by completions in the ${region}. WA LTV stable at `}
            <Inline>{percent(ltv, 1)}</Inline>
            {`.`}
          </Body>
        </div>
      </Enter>

      <Enter at={at} index={10}>
        {/* "· deterministic MI engine" wrapped to a second line and the result band
            already says it three inches below. Cut. */}
        <Stamp>Grounded in Trakt</Stamp>
      </Enter>
    </div>
  );
};

// --------------------------------------------------------------------------- //
// Panel 3 · MI Agent workspace
// --------------------------------------------------------------------------- //
/**
 * A query, an answer and four bars.
 *
 * The chart is the reason this panel exists: it is the capability the Copilot channel
 * deliberately does not have. Region name and balance share a line and the bar runs the
 * full panel width beneath them, which is the only arrangement that leaves the bar long
 * enough for its growth to read at all.
 */
const WorkspacePanel: React.FC<{ at: number; isSquare: boolean }> = ({ at, isSquare }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const regions = SUMMARY.topRegions.slice(0, 4);
  const largest = Math.max(...regions.map((r) => r.balance), 1);
  const scale = isSquare ? 0.72 : 0.82;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: theme.motion.quick }}>
      <Enter at={at}>
        <Body style={{ fontSize: theme.type.body.fontSize * scale }}>
          Show funded balance by region.
        </Body>
      </Enter>
      <Enter at={at} index={4}>
        {/* One region, not three: the chart immediately beneath carries all four, and
            repeating them in prose is the line that overflows the panel. */}
        <Stamp tone="paper">
          {`${regions[0].region} leads at ${money(regions[0].balance)}`}
        </Stamp>
      </Enter>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: theme.motion.quick,
          paddingTop: theme.motion.quick / 2,
        }}
      >
        {regions.map((region, i) => {
          const start = BARS_AT + i * theme.motion.stagger * 3;
          const grow = spring({
            frame: frame - start,
            fps,
            config: theme.motion.spring,
            durationInFrames: theme.motion.slow,
          });
          return (
            <Enter key={region.region} at={at} index={7 + i * 2}>
              <div
                style={{ display: "flex", flexDirection: "column", gap: theme.motion.quick / 2 }}
              >
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <Stamp tone="paper">{region.region}</Stamp>
                  <Stamp>{money(region.balance)}</Stamp>
                </div>
                {/* A full-width TRACK with the bar inside it. Without the track the bar
                    sits alone under the label and reads as an underline of the text
                    rather than as a quantity, which is the whole point of the panel. */}
                <div
                  style={{
                    width: "100%",
                    height: theme.motion.quick,
                    backgroundColor: theme.color.rule,
                  }}
                >
                  <div
                    style={{
                      // `mute`, not `rule`: a hairline-coloured bar on a hull panel is
                      // invisible, and an invisible bar is not a chart.
                      backgroundColor: theme.color.mute,
                      height: "100%",
                      width: `${(region.balance / largest) * 100 * grow}%`,
                    }}
                  />
                </div>
              </div>
            </Enter>
          );
        })}
      </div>
    </div>
  );
};

const CHANNELS = [
  { label: "Managed service", Content: ManagedPanel },
  { label: MICROSOFT_CHANNEL, Content: CopilotPanel },
  { label: "MI Agent workspace", Content: WorkspacePanel },
] as const;

// --------------------------------------------------------------------------- //
export const S4Omnichannel: React.FC = () => {
  const frame = useCurrentFrame();
  const geometry = useGeometry();
  const isSquare = useIsSquare();

  const total = SUMMARY.metrics.funded_balance ?? 0;
  const movement = MOVEMENT.delta.funded_balance ?? 0;

  /**
   * WIDE geometry. The three panels are a fixed-height row and the result band is
   * positioned from the bottom of it, so the three figures sit on one baseline whatever
   * length the panel content runs to.
   *
   * SQUARE stacks the three panels, where a shared baseline is impossible by
   * construction, so there the figure lives inside each panel and the band is one scope
   * stamp. The square crop also drops the panel content entirely: at 24px mono a
   * filename is 400px wide and the panels are 330px, and the fix for an overflow is to
   * cut, never to shrink. See the README for the full list.
   */
  const PAD = theme.motion.base + theme.motion.quick / 2;
  const PANEL_TOP = isSquare ? geometry.edge + theme.motion.slow : 96;
  const PANEL_HEIGHT = 540;
  const BAND_TOP = PANEL_TOP + PANEL_HEIGHT + theme.motion.slow;
  const FIGURE_SCALE = isSquare ? 0.46 : 0.62;

  const panelsOpacity = interpolate(frame, [0, 24, 486, 510], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const payloadOpacity = interpolate(frame, [PAYLOAD_AT, PAYLOAD_AT + 18], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const movementOpacity = interpolate(frame, [MOVEMENT_AT, MOVEMENT_AT + 18], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const claimOpacity = interpolate(frame, [CLAIM_AT, CLAIM_AT + 24], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const movementStamp = (
    <div style={{ opacity: movementOpacity }}>
      <Stamp>
        {noBreakSigns(`${signedMoney(movement)} vs ${periodLabel(MOVEMENT.priorPeriod)}`)}
      </Stamp>
    </div>
  );

  /**
   * The payload, rendered once per column. One `at`, one format, one spring.
   *
   * The movement rides with it in the WIDE layout, where the three columns are side by
   * side and the repetition is the argument. The square crop drops it entirely — see the
   * band below.
   */
  const payload = (
    <>
      <Figure scale={FIGURE_SCALE} tone="signal">
        <Counter from={0} to={total} at={PAYLOAD_AT} frames={theme.motion.slow} format={money} />
      </Figure>
      {isSquare ? null : (
        <div style={{ paddingTop: theme.motion.quick / 2 }}>{movementStamp}</div>
      )}
    </>
  );

  return (
    <AbsoluteFill>
      <div
        style={{
          position: "absolute",
          top: PANEL_TOP,
          left: geometry.gutter,
          right: geometry.gutter,
          display: "flex",
          flexDirection: isSquare ? "column" : "row",
          alignItems: "stretch",
          gap: isSquare ? theme.motion.quick : theme.motion.base,
          opacity: panelsOpacity,
        }}
      >
        {CHANNELS.map(({ label, Content }, i) => (
          <Panel
            key={label}
            style={{
              flex: 1,
              height: isSquare ? undefined : PANEL_HEIGHT,
              padding: PAD,
              display: "flex",
              flexDirection: "column",
              gap: theme.motion.quick,
              minWidth: 0,
            }}
          >
            {/* The channel name. On panel 2 this IS the Microsoft wordmark — body face,
                full contrast, unaltered, and the most prominent text in the panel. */}
            <Body>{label}</Body>
            <Rule />
            {/* The differentiation: what using it actually feels like. */}
            <Body
              tone="mute"
              style={{ fontSize: theme.type.body.fontSize * (isSquare ? 0.72 : 0.82) }}
            >
              {CHANNEL_USE[label]}
            </Body>
            {isSquare ? (
              <div style={{ opacity: payloadOpacity, paddingTop: theme.motion.quick }}>
                {payload}
              </div>
            ) : (
              <div style={{ paddingTop: theme.motion.quick }}>
                <Content at={CONTENT_AT + i * theme.motion.stagger * 2} isSquare={isSquare} />
              </div>
            )}
          </Panel>
        ))}
      </div>

      {/* The result band. Three columns on the SAME grid as the panels above, so each
          figure sits under its own channel and the three share one baseline. */}
      <div
        style={{
          position: "absolute",
          top: isSquare ? undefined : BAND_TOP,
          bottom: isSquare ? geometry.captionReserve : undefined,
          left: geometry.gutter,
          right: geometry.gutter,
          display: "flex",
          flexDirection: "column",
          gap: theme.motion.base,
          opacity: panelsOpacity,
        }}
      >
        {/* CUT in square: the movement. Three stacked panels plus a two-line trademark
            notice leave the band about 50px of room, and between the month's movement and
            the three use lines the use lines are what a scrolling viewer needs. The
            balance, the scope and the attribution all survive. */}
        {isSquare ? null : (
          <div style={{ display: "flex", gap: theme.motion.base, opacity: payloadOpacity }}>
            {CHANNELS.map(({ label }) => (
              <div key={label} style={{ flex: 1, minWidth: 0, paddingLeft: PAD }}>
                {payload}
              </div>
            ))}
          </div>
        )}
        <Rule />
        <div style={{ display: "flex", flexDirection: "column", gap: theme.motion.quick / 2 }}>
          {/* Which scope the viewer is looking at. S3 shows a sponsor-level figure for
              ten seconds; without this a viewer could reasonably carry it into a scene
              where every figure is the platform canonical. */}
          <Stamp>
            {`PLATFORM CANONICAL · ${shortDate(CURRENT_PERIOD.reportingDate).toUpperCase()} · ` +
              "DETERMINISTIC MI ENGINE"}
          </Stamp>
          {/* Trademark attribution for the wordmark in panel 2. It sits here rather than
              in the chrome because the chrome is pixel-identical across the whole film
              and this notice belongs to the one scene that makes the reference. */}
          <Stamp>{MS_TRADEMARK_NOTICE}</Stamp>
        </div>
      </div>

      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          opacity: claimOpacity,
        }}
      >
        <Enter at={CLAIM_AT}>
          <Claim measure={isSquare ? 0.86 : 0.58}>Three ways in. One governed answer.</Claim>
        </Enter>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

export default S4Omnichannel;
