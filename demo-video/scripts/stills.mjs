/**
 * Export the outbound-email still frames.
 *
 * The storyboard names three frames for use in the email body itself.
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
const EMAIL_FRAMES = [700, 1420, 2100];

/**
 * Review frames.
 *
 * One at the MIDPOINT of each scene, plus the midpoint of every BEAT — the scenes are
 * cut 420 / 480 / 780 / 660 / 360 and the beats inside S2 and S3 change the whole frame,
 * so a per-scene sample would miss most of the film. These are the frames to scrub when
 * checking legibility, overflow and contrast.
 */
const REVIEW_FRAMES = [
  { frame: 30, name: "s1-01-opening-line" },
  { frame: 150, name: "s1-02-failed-connection" },
  { frame: 210, name: "s1-03-midpoint" },
  { frame: 300, name: "s1-04-cost-anchor" },
  // S2 starts at 420. Beats: clock 0-120, arrivals 120-270, receipt 270-420, out 420-480.
  // S2 starts at 420. Beats: clock 0-114, funnel 114-294, receipt 294-426, out 426-480.
  { frame: 470, name: "s2-01-clock" },
  { frame: 590, name: "s2-02-tiles" },
  { frame: 640, name: "s2-03-converging" },
  { frame: 700, name: "s2-04-band-and-claim" },
  { frame: 800, name: "s2-05-receipt-and-referral" },
  { frame: 870, name: "s2-06-granularity" },
  // S3 starts at 900. Beats: lanes 0-210, sponsor 198-300, platform 300-780.
  { frame: 1000, name: "s3-01-lanes" },
  { frame: 1140, name: "s3-02-sponsor" },
  { frame: 1180, name: "s3-03-sponsor-claim" },
  { frame: 1240, name: "s3-04-platform-parts" },
  { frame: 1330, name: "s3-05-platform-total" },
  { frame: 1420, name: "s3-06-cards" },
  { frame: 1540, name: "s3-07-reconciliation" },
  { frame: 1650, name: "s3-08-claim" },
  // S4 starts at 1680. Content 1764, bars 1830, payload 1992, claim 2190.
  { frame: 1800, name: "s4-01-panels" },
  { frame: 1900, name: "s4-02-content" },
  { frame: 2100, name: "s4-03-payload" },
  { frame: 2250, name: "s4-04-claim" },
  { frame: 2520, name: "s5-01-midpoint" },
  { frame: 2640, name: "s5-02-ask" },
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
