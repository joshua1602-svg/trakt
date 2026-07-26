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
  </>
);

export default RemotionRoot;
