/**
 * Render the agent-to-agent demo's poster still, and a review strip.
 *
 *   node scripts/still-a2a.mjs            poster only
 *   node scripts/still-a2a.mjs --review   poster + one still per beat
 *
 * The poster IS a frame of the film — frame 1210, the moment the three
 * verdicts have resolved against one number. It used to come from a separate
 * `LandingA2APoster` composition, drawn because the page centred its play
 * control over the concentration card. That bought a poster the film never
 * shows: press play and the picture changes completely. The player now anchors
 * the A2A plate low in the frame instead, which clears everything frame 1210
 * draws (all of it above 65% of the height), so the still can be the real
 * thing and cannot drift from it again.
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

/**
 * The three verdicts have resolved and the run's counters are final: the one
 * frame that carries the whole argument. Content ends at 64% of the frame
 * height, and `landing-page/e2e/landing.spec.ts` asserts the play plate stays
 * below that.
 */
const POSTER_FRAME = 1210;

/** One per beat, plus both sides of every transition that swaps the panel. */
const REVIEW_FRAMES = [
  { frame: 40, name: "01-two-agents-reaching" },
  { frame: 90, name: "02-connected" },
  { frame: 130, name: "03-lifting" },
  { frame: 200, name: "04-layer-opens" },
  { frame: 330, name: "05-orientation" },
  { frame: 500, name: "06-valuation-age" },
  { frame: 545, name: "07-first-refusal" },
  { frame: 585, name: "08-second-refusal" },
  { frame: 615, name: "09-adjusted" },
  { frame: 720, name: "10-mid-trace" },
  { frame: 830, name: "11-follow-up-checks" },
  { frame: 930, name: "12-both-times" },
  { frame: 985, name: "13-returning" },
  { frame: 1090, name: "14-first-three-findings" },
  { frame: 1135, name: "15-fact-and-rule" },
  { frame: 1210, name: "16-verdicts-resolved" },
  { frame: 1300, name: "17-remaining-findings" },
  { frame: 1360, name: "18-diligence-and-gaps" },
  { frame: 1400, name: "19-report-held" },
  { frame: 1470, name: "20-close" },
];

const main = async () => {
  const review = process.argv.includes("--review");
  mkdirSync(OUT, { recursive: true });
  const browserExecutable = await findBrowser();
  const browser = browserExecutable ? { browserExecutable } : {};
  console.log("[still] bundling…");
  const serveUrl = await bundle({ entryPoint: join(ROOT, "src/index.ts") });

  const demo = await selectComposition({ serveUrl, id: "LandingA2ADemo", ...browser });
  const output = join(OUT, "a2a-demo-poster.png");
  await renderStill({ composition: demo, serveUrl, output, frame: POSTER_FRAME, ...browser });
  console.log(`[still] wrote ${output}`);

  if (!review) return;
  mkdirSync(REVIEW, { recursive: true });
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
