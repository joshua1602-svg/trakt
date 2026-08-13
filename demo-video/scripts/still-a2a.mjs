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
  { frame: 330, name: "05-orientation" },
  { frame: 500, name: "06-valuation-age" },
  { frame: 530, name: "07-first-refusal" },
  { frame: 570, name: "08-second-refusal" },
  { frame: 600, name: "09-adjusted" },
  { frame: 700, name: "10-mid-trace" },
  { frame: 830, name: "11-follow-up-checks" },
  { frame: 950, name: "12-both-times" },
  { frame: 1000, name: "13-returning" },
  { frame: 1110, name: "14-first-three-findings" },
  { frame: 1160, name: "15-fact-and-rule" },
  { frame: 1240, name: "16-verdicts-resolving" },
  { frame: 1270, name: "17-judgement" },
  { frame: 1345, name: "18-remaining-findings" },
  { frame: 1400, name: "19-diligence-and-gaps" },
  { frame: 1465, name: "20-report-held" },
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
