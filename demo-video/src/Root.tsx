/**
 * Remotion composition registry.
 *
 * Three compositions, all from the same scene components:
 *
 *   TraktProductDemo   1920×1080 · the film
 *   TraktProductTeaser 1920×1080 · a short selection cut
 *   SceneStills        1920×1080 · a still-frame harness for visual review
 *
 * The web deliverable is the same composition rendered at a lower bitrate and
 * scale, not a separate composition — a second composition could drift.
 */

import React from "react";
import { Composition, Sequence, Still } from "remotion";
import { Film, Teaser, framesForScenes } from "./Film";
import { SCENES, TEASER_SCENE_IDS, totalFrames } from "./timeline";
import { FPS } from "./design/motion";
import { LAYOUT } from "./design/tokens";
import { SCENE_COMPONENTS } from "./Film";

export const RemotionRoot: React.FC = () => (
  <>
    <Composition
      id="TraktProductDemo"
      component={Film}
      durationInFrames={totalFrames()}
      fps={FPS}
      width={LAYOUT.width}
      height={LAYOUT.height}
    />
    <Composition
      id="TraktProductTeaser"
      component={Teaser}
      durationInFrames={framesForScenes(TEASER_SCENE_IDS)}
      fps={FPS}
      width={LAYOUT.width}
      height={LAYOUT.height}
      defaultProps={{ sceneIds: TEASER_SCENE_IDS }}
    />
    {/*
      One Still per scene, held at the frame where that scene is fully composed.
      These are what `npm run stills` renders for visual review — checking for
      overflow, clipping, contrast and inconsistent figures before publishing.
    */}
    {SCENES.map((scene) => {
      const Component = SCENE_COMPONENTS[scene.id];
      const reviewFrame = Math.round(FPS * scene.seconds * 0.82);
      return (
        <Still
          key={`still-${scene.id}`}
          id={`Still-${scene.number}-${scene.id}`}
          component={() => (
            <StillHarness sceneId={scene.id} frame={reviewFrame}>
              <Component captions={scene.captions} />
            </StillHarness>
          )}
          width={LAYOUT.width}
          height={LAYOUT.height}
        />
      );
    })}
  </>
);

/**
 * Holds a scene at a fixed frame so a Still renders the settled composition
 * rather than frame 0, where every entrance animation is still at zero opacity.
 *
 * A Still always evaluates at frame 0; wrapping the scene in a Sequence with a
 * negative `from` shifts the scene's internal clock forward to the review frame,
 * which is the supported way to hold a composition at a chosen moment.
 */
const StillHarness: React.FC<{
  sceneId: string;
  frame: number;
  children: React.ReactNode;
}> = ({ frame, children }) => (
  <Sequence from={-frame} durationInFrames={frame + 2} layout="none">
    {children}
  </Sequence>
);
