/**
 * Render the landing controls demo's poster still.
 *
 *   node scripts/still-controls.mjs
 *
 * The poster is the loop's resolved end state (monitoring + breach horizon +
 * key line), which is also what reduced-motion visitors effectively see — so
 * the still can never show something the loop does not.
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

/** Late enough that every element has settled, before the loop restarts. */
const POSTER_FRAME = 530;

const main = async () => {
  mkdirSync(OUT, { recursive: true });
  const browserExecutable = await findBrowser();
  console.log("[still] bundling…");
  const serveUrl = await bundle({ entryPoint: join(ROOT, "src/index.ts") });
  const composition = await selectComposition({
    serveUrl,
    id: "LandingControlsDemo",
    ...(browserExecutable ? { browserExecutable } : {}),
  });
  const output = join(OUT, "controls-demo-poster.png");
  await renderStill({
    composition,
    serveUrl,
    output,
    frame: POSTER_FRAME,
    ...(browserExecutable ? { browserExecutable } : {}),
  });
  console.log(`[still] wrote ${output}`);
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
