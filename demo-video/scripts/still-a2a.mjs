/**
 * Render the agent-to-agent demo's poster still, and a review strip.
 *
 *   node scripts/still-a2a.mjs            poster only
 *   node scripts/still-a2a.mjs --review   poster + one still per beat
 *
 * The poster comes from the dedicated `LandingA2APoster` composition, not a
 * frame of the loop: the page centres its play control and the frame the
 * poster is drawn from puts the concentration card exactly there.
 *
 * The review strip samples every beat of the storyboard — the frames to scrub
 * for overflow, clipping, contrast and a figure that disagrees with another
 * surface. They come out of the loop itself, so a review still can never show
 * something the loop does not.
 */

import { mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { bundle } from "@remotion/bundler";
import { renderStill, selectComposition } from "@remotion/renderer";

import { findBrowser } from "./render.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
const OUT = join(ROOT, "out");
const REVIEW = join(OUT, "stills", "a2a");

const POSTER_FRAME = 60;

/** One per beat, plus both sides of every transition that swaps the panel. */
const REVIEW_FRAMES = [
  { frame: 30, name: "01-objective-crossing" },
  { frame: 84, name: "02-accepted" },
  { frame: 150, name: "03-layer-opens" },
  { frame: 250, name: "04-orientation" },
  { frame: 330, name: "05-rule-packs" },
  { frame: 380, name: "06-regulatory-gaps" },
  { frame: 420, name: "07-first-invalid" },
  { frame: 470, name: "08-second-invalid" },
  { frame: 520, name: "09-corrected" },
  { frame: 560, name: "10-concentration-fires" },
  { frame: 610, name: "11-thirty-one-percent" },
  { frame: 670, name: "12-verdict-pass" },
  { frame: 725, name: "13-verdict-flag" },
  { frame: 785, name: "14-verdict-breach" },
  { frame: 850, name: "15-trace-resumes" },
  { frame: 975, name: "16-follow-up" },
  { frame: 1000, name: "17-same-region" },
  { frame: 1060, name: "18-completing" },
  { frame: 1110, name: "19-artifact-returns" },
  { frame: 1200, name: "20-verdict-held" },
];

const main = async () => {
  const review = process.argv.includes("--review");
  mkdirSync(OUT, { recursive: true });
  const browserExecutable = await findBrowser();
  const browser = browserExecutable ? { browserExecutable } : {};
  console.log("[still] bundling…");
  const serveUrl = await bundle({ entryPoint: join(ROOT, "src/index.ts") });

  const poster = await selectComposition({ serveUrl, id: "LandingA2APoster", ...browser });
  const output = join(OUT, "a2a-demo-poster.png");
  await renderStill({ composition: poster, serveUrl, output, frame: POSTER_FRAME, ...browser });
  console.log(`[still] wrote ${output}`);

  if (!review) return;
  mkdirSync(REVIEW, { recursive: true });
  const demo = await selectComposition({ serveUrl, id: "LandingA2ADemo", ...browser });
  for (const { frame, name } of REVIEW_FRAMES) {
    await renderStill({
      composition: demo,
      serveUrl,
      output: join(REVIEW, `${name}.png`),
      frame,
      ...browser,
    });
    console.log(`[still] ${name} (frame ${frame})`);
  }
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
