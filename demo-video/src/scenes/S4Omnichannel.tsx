/**
 * S4 · Three ways in — 660 frames (0:56–1:18)
 *
 * Three panels, each reduced to three elements: a scheduled artifact drop, a Copilot
 * thread calling `askTraktMi`, and the workspace with the regional bars. Recognisable
 * silhouettes, not readable screens.
 *
 * Each panel carries a plain-English use line beneath its label. Without it two of the
 * three panels are just lists of monospace strings and the three read as
 * interchangeable — the use line is the differentiation, so it sits above the mono
 * content, in the body face, at full contrast.
 *
 * Then the payload: the consolidated balance counts up SIMULTANEOUSLY in all three
 * panels, landing on the same frame, followed by the movement beneath each — also
 * simultaneous. Simultaneity is the whole argument; if the three counters land on
 * different frames the beat is dead. They share one `at`, so they cannot drift.
 *
 * The panels are sized and positioned so the three figures sit on a shared horizontal
 * eye-line at the vertical centre of the frame. A fixed-height top block above the
 * figure guarantees the eye-line holds whatever length the content above it runs to.
 *
 * On `signal`: the film's general rule is one accented element per frame. This scene is
 * the stated exception — the three figures are one value proven in three places, which
 * is the point of the scene, so they carry it together.
 *
 * Data used: the consolidated balance, the month's movement, the Copilot action names
 * and the regional split — all from the fixtures.
 */

import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";

import { CHANNEL_USE } from "../claims";
import {
  Body,
  Claim,
  Counter,
  Enter,
  Figure,
  Label,
  Panel,
  Rule,
  Stamp,
  useGeometry,
  useIsSquare,
} from "../components/kit";
import { COPILOT, CURRENT_PERIOD, MOVEMENT, SUMMARY } from "../data/fixtures";
import { money, noBreakSigns, periodLabel, shortDate, signedMoney } from "../format";
import theme from "../theme";

/** A row in a silhouette: a leader hairline and one mono string. */
const SilhouetteRow: React.FC<{
  at: number;
  index: number;
  children: React.ReactNode;
  tone?: "paper" | "mute";
}> = ({ at, index, children, tone = "mute" }) => (
  <Enter at={at} index={index}>
    <div style={{ display: "flex", alignItems: "center", gap: theme.motion.quick }}>
      <div
        style={{
          width: theme.motion.quick,
          height: theme.hairline,
          backgroundColor: theme.color.rule,
          flexShrink: 0,
        }}
      />
      <Stamp tone={tone}>{children}</Stamp>
    </div>
  </Enter>
);

/** A scheduled artifact drop. */
const ManagedSilhouette: React.FC<{ at: number }> = ({ at }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: theme.motion.quick }}>
    {["annex2_submission.xml", "investor_pack.pptx", "platform_canonical_typed.csv"].map(
      (name, i) => (
        <SilhouetteRow key={name} at={at} index={i}>
          {name}
        </SilhouetteRow>
      ),
    )}
  </div>
);

/** A Copilot thread: the plugin's three actions, the one it calls first at full contrast. */
const CopilotSilhouette: React.FC<{ at: number }> = ({ at }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: theme.motion.quick }}>
    {COPILOT.actions.slice(0, 3).map((action, i) => (
      <SilhouetteRow key={action} at={at} index={i} tone={i === 0 ? "paper" : "mute"}>
        {`${action}()`}
      </SilhouetteRow>
    ))}
  </div>
);

/** The workspace: three regional bars, from the real regional split. */
const WorkspaceSilhouette: React.FC<{ at: number }> = ({ at }) => {
  const regions = SUMMARY.topRegions.slice(0, 3);
  const largest = Math.max(...regions.map((r) => r.balance), 1);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: theme.motion.quick }}>
      {regions.map((region, i) => (
        <Enter key={region.region} at={at} index={i}>
          <div style={{ display: "flex", alignItems: "center", gap: theme.motion.quick }}>
            <Stamp style={{ width: "42%", flexShrink: 0 }}>{region.region}</Stamp>
            <div
              style={{
                width: `${(region.balance / largest) * 58}%`,
                height: theme.motion.quick / 2,
                // `mute`, not `rule`: a hairline-coloured bar on a hull panel is
                // invisible, and an invisible bar is not a silhouette of anything.
                backgroundColor: theme.color.mute,
              }}
            />
          </div>
        </Enter>
      ))}
    </div>
  );
};

/**
 * On the Microsoft 365 Copilot panel.
 *
 * The panel carries the product NAME as a wordmark and no app icon. Microsoft's
 * trademark guidelines state that their "logos, app and product icons, illustrations,
 * photographs, videos, and designs can never be used without an express license", which
 * this project does not hold; the instruction's own fallback is a wordmark rather than an
 * approximation of the mark.
 *
 * The `label` token IS that wordmark — it is set in the body face (Inter 500), which is
 * what the fallback asks for — so no second element is needed, and adding one would just
 * print "Microsoft 365" twice. The guidelines' text rules are met by the label itself:
 * "Microsoft" precedes the product name, the name is unaltered and used adjectivally
 * ahead of the panel's content, no affiliation is implied, and Trakt's own lockup is the
 * most prominent brand on screen.
 *
 * If a licence is obtained, drop the official asset into `public/brand/` and render it
 * beside the label with <Img src={staticFile(...)} /> at the sizing the licence specifies.
 */

const CHANNELS = [
  { label: "Managed service", Silhouette: ManagedSilhouette },
  { label: "Microsoft 365 Copilot", Silhouette: CopilotSilhouette },
  { label: "MI Agent workspace", Silhouette: WorkspaceSilhouette },
] as const;

export const S4Omnichannel: React.FC = () => {
  const frame = useCurrentFrame();
  const geometry = useGeometry();
  const isSquare = useIsSquare();

  const total = SUMMARY.metrics.funded_balance ?? 0;
  const movement = MOVEMENT.delta.funded_balance ?? 0;

  // Panel geometry. The top block is a FIXED height, so the figure beneath it starts at
  // the same offset in all three panels and the eye-line holds.
  const PAD = isSquare ? theme.motion.base : theme.motion.base + theme.motion.quick / 2;
  /**
   * WIDE only. A fixed top block is what holds the three figures on one eye-line, and it
   * is only safe when its height is known to exceed the content. In the square crop the
   * three panels are stacked, so there is no eye-line to hold and a fixed height just
   * clips: there the panels size to their content instead.
   */
  const TOP_BLOCK = 190;
  const FIGURE_SCALE = isSquare ? 0.4 : 0.46;
  /** Rendered height of the figure itself: the `stat` token has lineHeight 1. */
  const FIGURE_HEIGHT = theme.type.stat.fontSize * FIGURE_SCALE * geometry.statScale;
  const FIGURE_BLOCK = FIGURE_HEIGHT + theme.motion.quick / 2 + theme.type.stamp.fontSize * 1.2;
  const PANEL_HEIGHT = PAD * 2 + TOP_BLOCK + FIGURE_BLOCK;
  /**
   * Wide: position the row so the FIGURE — not the figure block — sits on the frame's
   * vertical centre. The three figures are the beat; they have to be level and central,
   * and centring the block instead leaves them a movement-stamp's height high.
   *
   * Square stacks the three panels, where a shared horizontal eye-line is impossible by
   * construction, so there the row is simply positioned from the top.
   */
  const rowTop = isSquare
    ? geometry.edge + theme.motion.slow * 2
    : geometry.height / 2 - (PAD + TOP_BLOCK + FIGURE_HEIGHT / 2);

  const panelsOpacity = interpolate(frame, [0, 24, 486, 510], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const silhouetteOpacity = interpolate(frame, [90, 114, 300, 318], [0, 1, 1, 0.4], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // The payload. ONE `at`, shared by all three panels.
  const PAYLOAD_AT = 312;
  const MOVEMENT_AT = 360;
  const payloadOpacity = interpolate(frame, [PAYLOAD_AT, PAYLOAD_AT + 18], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const movementOpacity = interpolate(frame, [MOVEMENT_AT, MOVEMENT_AT + 18], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const claimOpacity = interpolate(frame, [510, 534], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <div
        style={{
          position: "absolute",
          top: rowTop,
          left: geometry.gutter,
          right: geometry.gutter,
          display: "flex",
          flexDirection: isSquare ? "column" : "row",
          alignItems: "stretch",
          gap: isSquare ? theme.motion.quick : theme.motion.base,
          opacity: panelsOpacity,
        }}
      >
        {CHANNELS.map(({ label, Silhouette }, i) => (
          <Panel
            key={label}
            style={{
              flex: 1,
              height: isSquare ? undefined : PANEL_HEIGHT,
              padding: PAD,
              display: "flex",
              flexDirection: "column",
              minWidth: 0,
            }}
          >
            <div
              style={{
                height: isSquare ? undefined : TOP_BLOCK,
                display: "flex",
                flexDirection: "column",
                gap: theme.motion.quick,
                paddingBottom: isSquare ? theme.motion.quick : undefined,
              }}
            >
              <Label>{label}</Label>
              <Rule />
              {/* The differentiation: what using it actually feels like. */}
              <Body style={{ fontSize: theme.type.body.fontSize * (isSquare ? 0.72 : 0.86) }}>
                {CHANNEL_USE[label]}
              </Body>
              <div style={{ opacity: silhouetteOpacity }}>
                <Silhouette at={90 + i * theme.motion.stagger} />
              </div>
            </div>

            <div style={{ opacity: payloadOpacity }}>
              <Figure scale={FIGURE_SCALE} tone="signal">
                <Counter
                  from={0}
                  to={total}
                  at={PAYLOAD_AT}
                  frames={theme.motion.slow}
                  format={money}
                />
              </Figure>
              <div style={{ opacity: movementOpacity, paddingTop: theme.motion.quick / 2 }}>
                <Stamp>
                  {noBreakSigns(
                    `${signedMoney(movement)} vs ${periodLabel(MOVEMENT.priorPeriod)}`,
                  )}
                </Stamp>
              </div>
            </div>
          </Panel>
        ))}
      </div>

      {/* One provenance rule for all three, and it names the SCOPE.
          S3 shows a sponsor-level figure for ten seconds; without a scope stamp here a
          viewer could reasonably carry that number into this scene, where every figure
          is the platform canonical. "PLATFORM CANONICAL" closes that off in three words.
          Wide only — under a stack of three square panels there is no room for a fourth
          block, and each panel's own movement stamp already names the period. */}
      {isSquare ? null : (
        <div
          style={{
            position: "absolute",
            left: geometry.gutter,
            right: geometry.gutter,
            top: rowTop + PANEL_HEIGHT + theme.motion.slow,
            display: "flex",
            flexDirection: "column",
            gap: theme.motion.quick / 2,
            opacity: payloadOpacity * panelsOpacity,
          }}
        >
          <Rule />
          <Stamp>
            {`PLATFORM CANONICAL · ${shortDate(CURRENT_PERIOD.reportingDate).toUpperCase()} · ` +
              "DETERMINISTIC MI ENGINE"}
          </Stamp>
        </div>
      )}

      <AbsoluteFill
        style={{
          alignItems: "center",
          justifyContent: "center",
          paddingBottom: geometry.captionReserve,
          opacity: claimOpacity,
        }}
      >
        <Enter at={510}>
          <Claim measure={isSquare ? 0.86 : 0.58}>Three ways in. One governed answer.</Claim>
        </Enter>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

export default S4Omnichannel;
