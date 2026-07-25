/**
 * The film: eight scenes, cross-faded, driven entirely by `timeline.ts`.
 *
 * Scene order, length and copy live in the timeline table, so adjusting the pace
 * is a one-file change and the captions, the voice-over script and the subtitle
 * file all follow automatically.
 */

import React from "react";
import { AbsoluteFill, Sequence, useCurrentFrame } from "remotion";
import { SCENES, TRANSITION_SECONDS, sceneFrames, sceneStarts, type SceneSpec } from "./timeline";
import { s } from "./design/motion";
import { PAGE_BACKGROUND } from "./design/tokens";
import { Scene1Problem } from "./scenes/Scene1Problem";
import { Scene2Onboard } from "./scenes/Scene2Onboard";
import { Scene3Orchestrate } from "./scenes/Scene3Orchestrate";
import { Scene4MiSummary } from "./scenes/Scene4MiSummary";
import { Scene5MiMovement } from "./scenes/Scene5MiMovement";
import { Scene6Copilot } from "./scenes/Scene6Copilot";
import { Scene7Outputs } from "./scenes/Scene7Outputs";
import { Scene8Closing } from "./scenes/Scene8Closing";

type SceneComponent = React.FC<{ captions: SceneSpec["captions"] }>;

/** Scene id → component. Kept explicit so a missing scene is a type error. */
export const SCENE_COMPONENTS: Record<string, SceneComponent> = {
  problem: Scene1Problem,
  onboard: Scene2Onboard,
  orchestrate: Scene3Orchestrate,
  "mi-summary": Scene4MiSummary,
  "mi-movement": Scene5MiMovement,
  copilot: Scene6Copilot,
  outputs: Scene7Outputs,
  closing: Scene8Closing,
};

/**
 * A cross-fade wrapper. The outgoing scene fades out over the incoming scene's
 * first frames, which reads as a considered dissolve rather than a cut, and never
 * leaves a blank frame between scenes.
 */
const CrossFade: React.FC<{
  children: React.ReactNode;
  durationInFrames: number;
  isFirst: boolean;
  isLast: boolean;
}> = ({ children, durationInFrames, isFirst, isLast }) => {
  const frame = useCurrentFrame();
  const ramp = s(TRANSITION_SECONDS);
  const fadeIn = isFirst ? 1 : Math.min(1, Math.max(0, frame / ramp));
  const fadeOut = isLast
    ? 1
    : Math.min(1, Math.max(0, (durationInFrames - frame) / ramp));
  return (
    <AbsoluteFill style={{ opacity: Math.min(fadeIn, fadeOut) }}>{children}</AbsoluteFill>
  );
};

export const Film: React.FC = () => {
  const starts = sceneStarts();
  return (
    <AbsoluteFill style={{ background: PAGE_BACKGROUND }}>
      {SCENES.map((scene, i) => {
        const Component = SCENE_COMPONENTS[scene.id];
        if (!Component) {
          throw new Error(`No component registered for scene "${scene.id}"`);
        }
        const duration = sceneFrames(scene);
        return (
          <Sequence
            key={scene.id}
            from={starts[i]}
            durationInFrames={duration}
            name={`${scene.number}. ${scene.title}`}
            layout="none"
          >
            <CrossFade
              durationInFrames={duration}
              isFirst={i === 0}
              isLast={i === SCENES.length - 1}
            >
              <Component captions={scene.captions} />
            </CrossFade>
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

/**
 * The teaser cut: the same scenes, unmodified, re-sequenced back to back. Nothing
 * is re-authored for the teaser — it is a selection, so it cannot contradict the
 * full film.
 */
export const Teaser: React.FC<{ sceneIds: readonly string[] }> = ({ sceneIds }) => {
  const scenes = sceneIds
    .map((id) => SCENES.find((sc) => sc.id === id))
    .filter((sc): sc is SceneSpec => Boolean(sc));
  const ramp = s(TRANSITION_SECONDS);
  let cursor = 0;
  return (
    <AbsoluteFill style={{ background: PAGE_BACKGROUND }}>
      {scenes.map((scene, i) => {
        const Component = SCENE_COMPONENTS[scene.id];
        const duration = sceneFrames(scene);
        const from = cursor;
        cursor += duration - (i < scenes.length - 1 ? ramp : 0);
        return (
          <Sequence
            key={scene.id}
            from={from}
            durationInFrames={duration}
            name={`${scene.number}. ${scene.title}`}
            layout="none"
          >
            <CrossFade
              durationInFrames={duration}
              isFirst={i === 0}
              isLast={i === scenes.length - 1}
            >
              <Component captions={scene.captions} />
            </CrossFade>
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

/** Total frames for a given selection of scene ids (used by the teaser composition). */
export const framesForScenes = (sceneIds: readonly string[]): number => {
  const ramp = s(TRANSITION_SECONDS);
  const scenes = sceneIds
    .map((id) => SCENES.find((sc) => sc.id === id))
    .filter((sc): sc is SceneSpec => Boolean(sc));
  return scenes.reduce(
    (total, scene, i) => total + sceneFrames(scene) - (i < scenes.length - 1 ? ramp : 0),
    0,
  );
};
