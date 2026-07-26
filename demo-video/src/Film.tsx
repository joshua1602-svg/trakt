/**
 * The film.
 *
 * `Chrome` and `Captions` sit OUTSIDE the `<Series>` deliberately: that structurally
 * guarantees the lockup and the disclaimer cannot drift between scenes, and it means
 * the disclaimer is present in every single frame rather than in every frame someone
 * remembered to add it to.
 *
 * Scene transitions are hard cuts on `ink`. No crossfades, no wipes.
 *
 * The same component tree renders both deliveries. `layout` is the only difference
 * between the 1920x1080 master and the 1080x1080 square variant — the square is the
 * same film with a tighter safe area, not a re-edit.
 */

import React from "react";
import { AbsoluteFill, Series, useCurrentFrame } from "remotion";

import { Chrome, LayoutProvider, useGeometry } from "./components/kit";
import { SYNTHETIC_NOTICE } from "./data/fixtures";
import S1Cost from "./scenes/S1Cost";
import S2Onboard from "./scenes/S2Onboard";
import S3Dataset from "./scenes/S3Dataset";
import S4Omnichannel from "./scenes/S4Omnichannel";
import S5Close from "./scenes/S5Close";
import theme, { type Layout } from "./theme";
import { SCENES, captions, startOf } from "./timeline";

const SCENE_COMPONENTS: Record<string, React.FC> = {
  cost: S1Cost,
  onboard: S2Onboard,
  dataset: S3Dataset,
  omnichannel: S4Omnichannel,
  close: S5Close,
};

/**
 * Burned-in captions: bottom third, `body` token, `paper` on a 70%-opacity `ink`
 * plate, max two lines. The film is watched with the sound off, so these carry the
 * argument unaided; the narration is the bonus track.
 */
const Captions: React.FC = () => {
  const frame = useCurrentFrame();
  const geometry = useGeometry();
  const active = captions().find((c) => frame >= c.from && frame < c.from + c.hold);
  if (!active) return null;
  const fade = Math.min(
    1,
    (frame - active.from) / theme.motion.quick,
    (active.from + active.hold - frame) / theme.motion.quick,
  );
  return (
    <div
      style={{
        position: "absolute",
        left: geometry.gutter,
        right: geometry.gutter,
        bottom: geometry.edge + 52,
        display: "flex",
        justifyContent: "center",
        opacity: Math.max(0, fade),
      }}
    >
      <div
        style={{
          backgroundColor: theme.color.ink,
          opacity: theme.captionPlateOpacity,
          position: "absolute",
          inset: "-16px -20px",
          borderRadius: theme.radius.plate,
        }}
      />
      <div
        style={{
          ...theme.type.body,
          fontSize: theme.type.body.fontSize * (geometry.width < theme.layout.wide.width ? 0.82 : 1),
          color: theme.color.paper,
          textAlign: "center",
          maxWidth: "88%",
          position: "relative",
        }}
      >
        {active.text}
      </div>
    </div>
  );
};

/** The one place the vendored `@font-face` rules are injected. */
const Fonts: React.FC = () => <style dangerouslySetInnerHTML={{ __html: theme.fontFaceCss() }} />;

export const Film: React.FC<{ layout?: Layout }> = ({ layout = "wide" }) => {
  const frame = useCurrentFrame();
  // The close is the only scene that owns the lockup, because it moves it.
  const inClose = frame >= startOf("close");
  return (
    <LayoutProvider layout={layout}>
      <AbsoluteFill style={{ backgroundColor: theme.color.ink }}>
        <Fonts />
        <Series>
          {SCENES.map((scene) => {
            const Scene = SCENE_COMPONENTS[scene.id];
            return (
              <Series.Sequence key={scene.id} durationInFrames={scene.frames}>
                <Scene />
              </Series.Sequence>
            );
          })}
        </Series>
        <Chrome notice={SYNTHETIC_NOTICE} hideLockup={inClose} />
        <Captions />
      </AbsoluteFill>
    </LayoutProvider>
  );
};

export default Film;
