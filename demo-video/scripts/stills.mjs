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

/**
 * Review frames.
 *
 * One at the MIDPOINT of each scene — 210 / 780 / 1440 / 2040 / 2520 — which is the set
 * to scrub when checking the frame as a whole, plus the specific beats worth their own
 * look: the failed connection, the referral hold, the card fan-out and the simultaneous
 * payload.
 */
const REVIEW_FRAMES = [
  { frame: 30, name: "s1-01-opening-line" },
  { frame: 150, name: "s1-02-failed-connection" },
  { frame: 210, name: "s1-03-midpoint" },
  { frame: 300, name: "s1-04-cost-anchor" },
  { frame: 450, name: "s2-01-clock" },
  { frame: 600, name: "s2-02-mapping" },
  { frame: 780, name: "s2-03-midpoint" },
  { frame: 900, name: "s2-04-referral" },
  { frame: 1080, name: "s2-05-claim" },
  { frame: 1260, name: "s3-01-total" },
  { frame: 1440, name: "s3-02-midpoint" },
  { frame: 1620, name: "s3-03-claim" },
  { frame: 1900, name: "s4-01-panels" },
  { frame: 2040, name: "s4-02-midpoint" },
  { frame: 2100, name: "s4-03-payload" },
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
