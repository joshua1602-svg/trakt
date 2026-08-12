/**
 * The controls demo's poster still.
 *
 * A separate composition rather than a frame of the film. The play control on
 * the landing page is centred in the frame, and in the film's closing frame
 * the concentration card occupies the centre — so a centred plate would cover
 * "Funded book · 24.1% · Pass", and the three rows going pass → warning →
 * projected breach are the demo's whole argument.
 *
 * The poster therefore carries the same card and the same closing line, with
 * the card lifted into the upper third and the line dropped to the foot,
 * leaving the middle band clear for the plate. Same content, same figures,
 * same "Illustrative · synthetic data" stamp — only the arrangement differs.
 */

import React from "react";
import { AbsoluteFill } from "remotion";

import { Chrome, MonitoringCard, MonitoringKeyline } from "./ControlsDemo";
import T from "./theme";

export const CONTROLS_POSTER = {
  id: "LandingControlsPoster",
  fps: 30,
  width: 1200,
  height: 960,
  duration: 240,
  /** Late enough that every spring in the card has settled. */
  stillFrame: 200,
} as const;

/** Scale that keeps the card clear of the centred plate's band. */
const CARD_SCALE = 0.55;

const ControlsPoster: React.FC = () => (
  <AbsoluteFill style={{ background: T.color.surface, fontFamily: T.family }}>
    <style dangerouslySetInnerHTML={{ __html: T.fontFaceCss() }} />
    <Chrome />
    <AbsoluteFill style={{ top: 68 }}>
      <div
        style={{
          position: "absolute",
          top: 20,
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "center",
        }}
      >
        <div style={{ transform: `scale(${CARD_SCALE})`, transformOrigin: "top center" }}>
          <MonitoringCard />
        </div>
      </div>
      {/* Positioned from the top: bottom-anchoring let the two-line close
          overflow the frame. */}
      <div
        style={{
          position: "absolute",
          top: 600,
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "center",
        }}
      >
        <MonitoringKeyline at={0} />
      </div>
    </AbsoluteFill>
  </AbsoluteFill>
);

export default ControlsPoster;
