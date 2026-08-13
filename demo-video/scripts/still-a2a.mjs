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
  { frame: 40, name: "01-two-agents-reaching" },
  { frame: 90, name: "02-connected" },
  { frame: 130, name: "03-lifting" },
  { frame: 200, name: "04-layer-opens" },
  { frame: 340, name: "05-orientation" },
  { frame: 430, name: "06-rule-packs" },
  { frame: 520, name: "07-valuations" },
  { frame: 545, name: "08-first-refusal" },
  { frame: 580, name: "09-second-refusal" },
  { frame: 612, name: "10-adjusted" },
  { frame: 660, name: "11-concentration-isolates" },
  { frame: 710, name: "12-thirty-one-percent" },
  { frame: 760, name: "13-verdict-pass" },
  { frame: 815, name: "14-verdict-flag" },
  { frame: 875, name: "15-verdict-breach" },
  { frame: 950, name: "16-trace-resumes" },
  { frame: 1065, name: "17-follow-up" },
  { frame: 1170, name: "18-both-times" },
  { frame: 1215, name: "19-returning" },
  { frame: 1300, name: "20-verdict-held" },
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
