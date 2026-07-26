/**
 * Audit the accent, frame by frame, from the rendered pixels.
 *
 * `signal` is a scarcity resource: at most ONE element per frame may carry it at full
 * strength. That is a claim about the picture, not about the source, so this checks the
 * picture — it renders a sampled sweep of the film, counts connected regions of
 * full-strength `signal` in each frame, and reports any frame carrying more than one.
 *
 *   node scripts/audit-signal.mjs            every 10th frame
 *   node scripts/audit-signal.mjs --step=5   denser sweep
 *
 * S4's payload is the film's stated exception — three panels showing one and the same
 * figure, which is the argument of the scene — so its frame range is reported separately
 * rather than counted as a violation.
 */

import { mkdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { bundle } from "@remotion/bundler";
import { renderStill, selectComposition } from "@remotion/renderer";
import { findBrowser } from "./render.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
const WORK = join(ROOT, "out", ".signal-audit");

/**
 * Frames where three panels legitimately show the same accented figure.
 *
 * S4 starts at 1680 (420 + 480 + 780). Its payload fades in at scene frame 312 and the
 * panels fade out over 486–510, so the exception runs 1992–2190 absolute.
 */
const S4_PAYLOAD = { from: 1992, to: 2190, why: "S4 payload — one figure, three panels" };

const readToken = (name) => {
  const theme = readFileSync(join(ROOT, "src", "theme.ts"), "utf8");
  const match = theme.match(new RegExp(`${name}: "(#[0-9A-Fa-f]{6})"`));
  if (!match) throw new Error(`no ${name} token in theme.ts`);
  const hex = match[1];
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
};

/**
 * Dilate a boolean mask by a rectangle, separably: a horizontal sliding-window max, then
 * a vertical one. This is a morphological close, and its whole job is to merge the glyphs
 * of one text line into a single region while leaving genuinely separate elements apart.
 *
 * The radii matter. Mono glyphs at the `stat` size sit ~20px apart, and words in a
 * display claim ~30px, so 34px horizontally merges a line. Two distinct accented elements
 * in this film are always more than 100px apart vertically, so 16px vertically keeps them
 * separate. Without this the counter reports one region PER LETTER.
 */
const dilate = (mask, width, height, rx, ry) => {
  const rows = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    const base = y * width;
    let lastHit = -Infinity;
    // Forward pass marks anything within rx to the right of a hit...
    for (let x = 0; x < width; x += 1) {
      if (mask[base + x]) lastHit = x;
      if (x - lastHit <= rx) rows[base + x] = 1;
    }
    lastHit = Infinity;
    // ...and the backward pass, anything within rx to the left.
    for (let x = width - 1; x >= 0; x -= 1) {
      if (mask[base + x]) lastHit = x;
      if (lastHit - x <= rx) rows[base + x] = 1;
    }
  }
  const out = new Uint8Array(mask.length);
  for (let x = 0; x < width; x += 1) {
    let lastHit = -Infinity;
    for (let y = 0; y < height; y += 1) {
      if (rows[y * width + x]) lastHit = y;
      if (y - lastHit <= ry) out[y * width + x] = 1;
    }
    lastHit = Infinity;
    for (let y = height - 1; y >= 0; y -= 1) {
      if (rows[y * width + x]) lastHit = y;
      if (lastHit - y <= ry) out[y * width + x] = 1;
    }
  }
  return out;
};

/**
 * Count accented ELEMENTS: flood fill the closed mask, but measure each region by how
 * much original ink it contains, so a region that is mostly dilation is discarded.
 */
const countRegions = (mask, width, height, minInk) => {
  const closed = dilate(mask, width, height, 34, 16);
  const seen = new Uint8Array(closed.length);
  const queue = new Int32Array(closed.length);
  let regions = 0;
  for (let start = 0; start < closed.length; start += 1) {
    if (!closed[start] || seen[start]) continue;
    let head = 0;
    let tail = 0;
    queue[tail += 1] = start;
    seen[start] = 1;
    let ink = 0;
    while (head < tail) {
      const at = queue[head += 1];
      if (mask[at]) ink += 1;
      const x = at % width;
      const y = (at - x) / width;
      const neighbours = [
        x > 0 ? at - 1 : -1,
        x < width - 1 ? at + 1 : -1,
        y > 0 ? at - width : -1,
        y < height - 1 ? at + width : -1,
      ];
      for (const next of neighbours) {
        if (next >= 0 && closed[next] && !seen[next]) {
          seen[next] = 1;
          queue[tail += 1] = next;
        }
      }
    }
    if (ink >= minInk) regions += 1;
  }
  return regions;
};

const main = async () => {
  const stepArg = process.argv.find((a) => a.startsWith("--step="));
  const step = stepArg ? Number(stepArg.slice("--step=".length)) : 10;
  const [sr, sg, sb] = readToken("signal");
  const softOpacity = Number(
    readFileSync(join(ROOT, "src", "theme.ts"), "utf8").match(
      /signalSoftOpacity = ([0-9.]+)/,
    )?.[1] ?? 0.4,
  );

  rmSync(WORK, { recursive: true, force: true });
  mkdirSync(WORK, { recursive: true });
  const browserExecutable = await findBrowser();
  const serveUrl = await bundle({
    entryPoint: join(ROOT, "src", "index.ts"),
    publicDir: join(ROOT, "public"),
  });
  const composition = await selectComposition({
    serveUrl,
    id: "TraktDemo",
    browserExecutable,
  });

  const { PNG } = await import("pngjs");
  const violations = [];
  const exceptions = [];
  let scanned = 0;

  for (let frame = 0; frame < composition.durationInFrames; frame += step) {
    const output = join(WORK, `f${frame}.png`);
    await renderStill({
      composition,
      serveUrl,
      output,
      frame,
      imageFormat: "png",
      browserExecutable,
      timeoutInMilliseconds: 120000,
    });
    const png = PNG.sync.read(readFileSync(output));
    const { width, height, data } = png;
    const mask = new Uint8Array(width * height);
    for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      // Full-strength signal only. A soft (40%) element composited over `ink` or `hull`
      // lands far from the token, so a tight tolerance excludes it by construction.
      if (
        Math.abs(r - sr) <= 22 &&
        Math.abs(g - sg) <= 22 &&
        Math.abs(b - sb) <= 22
      ) {
        mask[p] = 1;
      }
    }
    // The lockup mark is `signal` and is present in every frame by design: it is chrome,
    // not a proven value. Exclude the chrome inset.
    for (let y = 0; y < 110; y += 1) {
      for (let x = 0; x < 260; x += 1) mask[y * width + x] = 0;
    }
    const regions = countRegions(mask, width, height, 600);
    scanned += 1;
    if (regions > 1) {
      const inS4 = frame >= S4_PAYLOAD.from && frame <= S4_PAYLOAD.to;
      (inS4 ? exceptions : violations).push({ frame, regions });
    }
    rmSync(output, { force: true });
  }
  rmSync(WORK, { recursive: true, force: true });

  console.log(
    `[signal] scanned ${scanned} frames (every ${step}), soft strength ${softOpacity}`,
  );
  if (exceptions.length) {
    console.log(
      `[signal] ${exceptions.length} frame(s) in the stated exception ` +
        `(${S4_PAYLOAD.why}): ${exceptions.map((e) => `${e.frame}×${e.regions}`).join(", ")}`,
    );
  }
  if (violations.length) {
    console.error("\n[signal] FAILED — more than one full-strength element on:");
    for (const v of violations) console.error(`  frame ${v.frame}: ${v.regions} regions`);
    process.exit(1);
  }
  console.log("[signal] PASS — at most one full-strength signal element per frame");
};

await main();
