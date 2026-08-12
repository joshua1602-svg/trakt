/**
 * Compositions.
 *
 *   TraktDemo        1920x1080 master — the deliverable
 *   TraktDemoSquare  1080x1080 LinkedIn and email variant, the SAME scene
 *                    components with a `square` layout flag
 *
 * The still frames the outbound email body uses (see `STILL_FRAMES` in the timeline)
 * are rendered out of `TraktDemo` itself by `scripts/stills.mjs`, so a still can
 * never show something the film does not.
 *
 * The durations come from src/timeline.ts, so the composition cannot fall out of step
 * with the storyboard.
 */

import React from "react";
import { Composition } from "remotion";

import Film from "./Film";
import ControlsDemo, { CONTROLS_DEMO } from "./landing/ControlsDemo";
import ControlsPoster, { CONTROLS_POSTER } from "./landing/ControlsPoster";
import { FPS, totalFrames } from "./timeline";
import theme from "./theme";

const DURATION = totalFrames();

export const RemotionRoot: React.FC = () => (
  <>
    <Composition
      id="TraktDemo"
      component={Film}
      durationInFrames={DURATION}
      fps={FPS}
      width={theme.layout.wide.width}
      height={theme.layout.wide.height}
      defaultProps={{ layout: "wide" as const }}
    />
    <Composition
      id="TraktDemoSquare"
      component={Film}
      durationInFrames={DURATION}
      fps={FPS}
      width={theme.layout.square.width}
      height={theme.layout.square.height}
      defaultProps={{ layout: "square" as const }}
    />
    {/* The landing page's Risk & Controls loop — its own theme (the landing
        page's tokens, not the film's), registered here to reuse the render
        pipeline. See src/landing/ControlsDemo.tsx. */}
    {/* The poster still — the card lifted so a centred play control clears
        the three monitored rows. See src/landing/ControlsPoster.tsx. */}
    <Composition
      id={CONTROLS_POSTER.id}
      component={ControlsPoster}
      durationInFrames={CONTROLS_POSTER.duration}
      fps={CONTROLS_POSTER.fps}
      width={CONTROLS_POSTER.width}
      height={CONTROLS_POSTER.height}
    />
    <Composition
      id={CONTROLS_DEMO.id}
      component={ControlsDemo}
      durationInFrames={CONTROLS_DEMO.duration}
      fps={CONTROLS_DEMO.fps}
      width={CONTROLS_DEMO.width}
      height={CONTROLS_DEMO.height}
    />
  </>
);

export default RemotionRoot;
