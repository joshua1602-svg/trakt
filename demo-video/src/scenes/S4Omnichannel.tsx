/**
 * S4 · Three ways in — 600 frames (0:58–1:18)
 *
 * Three panels, each reduced to three elements: a scheduled artifact drop, a Copilot
 * thread calling `askTraktMi`, and the workspace with the regional bars. These are
 * recognisable silhouettes, not readable screens.
 *
 * Then the payload: the consolidated balance counts up SIMULTANEOUSLY in all three
 * panels, landing on the same frame, followed by the movement beneath each — also
 * simultaneous. Simultaneity is the whole argument. If the three counters land on
 * different frames the beat is dead.
 *
 * On `signal`: the storyboard's general rule is one accented element per frame, and
 * its S4 instruction is explicitly three at once. They are the same figure proven in
 * three places, which is the scene's entire point, so the specific instruction wins.
 *
 * Data used: the consolidated balance, the month's movement, the Copilot action
 * names and the regional split — all from the fixtures.
 */

import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";

import {
  Claim,
  Counter,
  Enter,
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

/** A scheduled artifact drop: three rows, no chrome. */
const ManagedSilhouette: React.FC<{ at: number }> = ({ at }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: theme.motion.quick }}>
    {["annex2_submission.xml", "investor_pack.pptx", "platform_canonical_typed.csv"].map(
      (name, i) => (
        <Enter key={name} at={at} index={i}>
          <div style={{ display: "flex", alignItems: "center", gap: theme.motion.quick }}>
            <div
              style={{
                width: theme.motion.quick,
                height: theme.hairline,
                backgroundColor: theme.color.rule,
              }}
            />
            <Stamp>{name}</Stamp>
          </div>
        </Enter>
      ),
    )}
  </div>
);

/** A Copilot thread: the action call and two of its arguments. */
const CopilotSilhouette: React.FC<{ at: number }> = ({ at }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: theme.motion.quick }}>
    {COPILOT.actions.slice(0, 3).map((action, i) => (
      <Enter key={action} at={at} index={i}>
        <div style={{ display: "flex", alignItems: "center", gap: theme.motion.quick }}>
          <div
            style={{
              width: theme.motion.quick,
              height: theme.hairline,
              backgroundColor: theme.color.rule,
            }}
          />
          <Stamp tone={i === 0 ? "paper" : "mute"}>{`${action}()`}</Stamp>
        </div>
      </Enter>
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
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <Stamp>{region.region}</Stamp>
            <div
              style={{
                width: `${(region.balance / largest) * 100}%`,
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

  // 1740–1830: the frame divides into three. 1830–2040: the silhouettes fill.
  const panelsOpacity = interpolate(frame, [0, 24, 408, 432], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const silhouetteOpacity = interpolate(frame, [90, 114, 288, 306], [0, 1, 1, 0.4], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  // 2040–2160: the payload. One `at`, shared by all three, so they cannot drift.
  const PAYLOAD_AT = 300;
  const MOVEMENT_AT = 342;
  const payloadOpacity = interpolate(frame, [PAYLOAD_AT, PAYLOAD_AT + 18], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const movementOpacity = interpolate(frame, [MOVEMENT_AT, MOVEMENT_AT + 18], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const claimOpacity = interpolate(frame, [432, 456], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <div
        style={{
          position: "absolute",
          inset: 0,
          paddingTop: geometry.edge + (isSquare ? 96 : 190),
          paddingBottom: geometry.captionReserve + (isSquare ? 30 : 48),
          paddingLeft: geometry.gutter,
          paddingRight: geometry.gutter,
          display: "flex",
          flexDirection: isSquare ? "column" : "row",
          gap: isSquare ? theme.motion.quick : theme.motion.base,
          opacity: panelsOpacity,
        }}
      >
        {CHANNELS.map(({ label, Silhouette }, i) => (
          <Panel
            key={label}
            style={{
              flex: 1,
              padding: isSquare ? theme.motion.base : geometry.gutter / 2,
              display: "flex",
              flexDirection: "column",
              gap: theme.motion.quick,
              minWidth: 0,
            }}
          >
            <Label>{label}</Label>
            <Rule />
            <div style={{ opacity: silhouetteOpacity, flex: 1 }}>
              <Silhouette at={90 + i * theme.motion.stagger} />
            </div>

            {/* The payload. Same `at` in every panel — they land together. */}
            <div style={{ opacity: payloadOpacity }}>
              <div
                style={{
                  ...theme.type.stat,
                  fontSize:
                    theme.type.stat.fontSize * (isSquare ? 0.3 : 0.46) * geometry.statScale,
                  color: theme.color.signal,
                }}
              >
                <Counter
                  from={0}
                  to={total}
                  at={PAYLOAD_AT}
                  frames={theme.motion.slow}
                  format={money}
                />
              </div>
              <div style={{ opacity: movementOpacity, paddingTop: 6 }}>
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

      {/* One provenance rule for all three: the same dataset, the same date. */}
      <div
        style={{
          position: "absolute",
          left: geometry.gutter,
          right: geometry.gutter,
          bottom: geometry.captionReserve + (isSquare ? 2 : 10),
          display: "flex",
          flexDirection: "column",
          gap: 6,
          opacity: payloadOpacity * panelsOpacity,
        }}
      >
        <Rule />
        <Stamp>
          {`SAME DATASET · ${shortDate(CURRENT_PERIOD.reportingDate).toUpperCase()} · ` +
            "DETERMINISTIC MI ENGINE"}
        </Stamp>
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
          <Claim maxWidth={isSquare ? "86%" : "58%"}>Three ways in. One governed answer.</Claim>
        </Enter>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

export default S4Omnichannel;
