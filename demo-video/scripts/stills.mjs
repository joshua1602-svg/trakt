/**
 * Render one still per scene for visual review.
 *
 * These are what a human should look at before publishing: overflow, text
 * clipping, contrast, inconsistent figures, accidental production data, visual
 * artefacts. Output lands in `out/stills/`.
 */

import { mkdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { bundle } from "@remotion/bundler";
import { getCompositions, renderStill } from "@remotion/renderer";
import { findBrowser } from "./render.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
const OUT = join(ROOT, "out", "stills");

const main = async () => {
  mkdirSync(OUT, { recursive: true });
  const browserExecutable = await findBrowser();

  console.log("[stills] bundling…");
  const serveUrl = await bundle({
    entryPoint: join(ROOT, "src", "index.ts"),
    publicDir: join(ROOT, "public"),
  });

  const compositions = await getCompositions(serveUrl, { browserExecutable });
  const stills = compositions.filter((c) => c.id.startsWith("Still-"));
  if (!stills.length) {
    throw new Error("No Still- compositions found in the registry.");
  }

  for (const composition of stills) {
    const output = join(OUT, `${composition.id}.png`);
    await renderStill({
      composition,
      serveUrl,
      output,
      imageFormat: "png",
      browserExecutable,
      chromiumOptions: { gl: "swangle" },
      timeoutInMilliseconds: 120000,
      onBrowserLog: (log) => {
        if (log.type === "error") console.error(`[stills][browser] ${log.text}`);
      },
    });
    const size = statSync(output).size;
    console.log(`[stills] ${composition.id}  ${Math.round(size / 1024)} KB`);
  }

  console.log(`\n[stills] ${stills.length} stills in ${OUT}`);
};

main().catch((error) => {
  console.error("\n[stills] FAILED");
  console.error(error);
  process.exit(1);
});
