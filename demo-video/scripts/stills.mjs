/**
 * Export the outbound-email still frames.
 *
 * The storyboard names frames 1020, 1500 and 2100 for use in the email body itself.
 * They are rendered out of `TraktDemo` at those exact frames — not from separate
 * Still compositions — so a still can never show something the film does not.
 *
 * A second pass writes one still per scene into `out/stills/review/`. Those are for
 * a human to look at before publishing: overflow, clipping, contrast, a figure that
 * disagrees with another surface, anything that reads as production data.
 *
 * Output: `out/stills/`.
 */

import { mkdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { bundle } from "@remotion/bundler";
import { renderStill, selectComposition } from "@remotion/renderer";
import { findBrowser } from "./render.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
const OUT = join(ROOT, "out", "stills");
const REVIEW = join(OUT, "review");

/** Kept in step with STILL_FRAMES in src/timeline.ts by a unit test. */
const EMAIL_FRAMES = [1020, 1500, 2100];

/** Mid-scene frames, for the visual review pass. */
const REVIEW_FRAMES = [
  { frame: 210, name: "s1-cost" },
  { frame: 470, name: "s2-onboard-clock" },
  { frame: 700, name: "s2-onboard-mapping" },
  { frame: 900, name: "s2-onboard-referred" },
  { frame: 1250, name: "s3-dataset-total" },
  { frame: 1500, name: "s3-dataset-cards" },
  { frame: 1900, name: "s4-omnichannel-panels" },
  { frame: 2100, name: "s4-omnichannel-payload" },
  { frame: 2500, name: "s5-close" },
  { frame: 2640, name: "s5-close-ask" },
];

const main = async () => {
  // `--square` reviews the 1080x1080 variant instead. Same frames, same tree — the
  // point is to catch copy that fits at 1920 and collides at 1080.
  const square = process.argv.includes("--square");
  const outDir = square ? join(OUT, "square") : OUT;
  const reviewDir = square ? join(REVIEW, "square") : REVIEW;
  mkdirSync(outDir, { recursive: true });
  mkdirSync(reviewDir, { recursive: true });
  const browserExecutable = await findBrowser();

  console.log("[stills] bundling…");
  const serveUrl = await bundle({
    entryPoint: join(ROOT, "src", "index.ts"),
    publicDir: join(ROOT, "public"),
  });

  const composition = await selectComposition({
    serveUrl,
    id: square ? "TraktDemoSquare" : "TraktDemo",
    browserExecutable,
  });

  const shoot = async (frame, output, tag) => {
    await renderStill({
      composition,
      serveUrl,
      output,
      frame,
      imageFormat: "png",
      browserExecutable,
      ...(process.env.REMOTION_GL ? { chromiumOptions: { gl: process.env.REMOTION_GL } } : {}),
      timeoutInMilliseconds: 120000,
      onBrowserLog: (log) => {
        if (log.type === "error") console.error(`[stills][browser] ${log.text}`);
      },
    });
    console.log(`[stills] ${tag}  frame ${frame}  ${Math.round(statSync(output).size / 1024)} KB`);
  };

  for (const frame of EMAIL_FRAMES) {
    await shoot(frame, join(outDir, `trakt-demo-frame-${frame}.png`), "email  ");
  }
  for (const { frame, name } of REVIEW_FRAMES) {
    await shoot(frame, join(reviewDir, `${name}.png`), "review ");
  }

  console.log(
    `\n[stills] ${EMAIL_FRAMES.length} email stills in ${outDir}` +
      `, ${REVIEW_FRAMES.length} review stills in ${reviewDir}`,
  );
};

main().catch((error) => {
  console.error("\n[stills] FAILED");
  console.error(error);
  process.exit(1);
});
