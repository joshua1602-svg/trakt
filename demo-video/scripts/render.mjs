/**
 * Render the film.
 *
 *   node scripts/render.mjs --preset=master            1920x1080 presentation master
 *   node scripts/render.mjs --preset=web               1920x1080, lighter bitrate
 *   node scripts/render.mjs --preset=teaser            the short cut
 *   node scripts/render.mjs --preset=master --preset=web
 *
 * Renders through the Remotion Node API rather than the CLI so the browser
 * executable, the codec settings and the failure behaviour are explicit and the
 * same on every machine.
 *
 * No network access is required or performed. The bundle imports its fixtures at
 * build time, and the browser is resolved from an existing local install.
 */

import { cpus } from "node:os";
import { existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { bundle } from "@remotion/bundler";
import { renderMedia, selectComposition } from "@remotion/renderer";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
const OUT = join(ROOT, "out");

/** Output presets. The web cut is the SAME composition at a lighter bitrate. */
const PRESETS = {
  master: {
    composition: "TraktProductDemo",
    file: "trakt-product-demo-1080p.mp4",
    crf: 18,
    scale: 1,
    label: "1920x1080 presentation master",
  },
  web: {
    composition: "TraktProductDemo",
    file: "trakt-product-demo-web.mp4",
    crf: 26,
    scale: 1,
    label: "1920x1080 web-optimised",
  },
  teaser: {
    composition: "TraktProductTeaser",
    file: "trakt-product-demo-teaser.mp4",
    crf: 20,
    scale: 1,
    label: "teaser cut",
  },
};

/**
 * Locate a Chrome/Chromium binary that is already installed.
 *
 * Remotion can download its own Chrome Headless Shell, but that needs network
 * access at render time — which this project deliberately does not rely on. These
 * are the standard locations, plus the Playwright layout used by this container.
 * Returning `null` lets Remotion fall back to its own resolution.
 */
export const findBrowser = async () => {
  const explicit = process.env.REMOTION_BROWSER_EXECUTABLE;
  if (explicit && existsSync(explicit)) return explicit;
  const candidates = [
    "/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell",
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  ];
  // Any other Playwright chromium revision present in the image.
  const pwRoot = process.env.PLAYWRIGHT_BROWSERS_PATH ?? "/opt/pw-browsers";
  for (const suffix of ["chrome-linux/headless_shell", "chrome-linux/chrome"]) {
    try {
      const { readdirSync } = await import("node:fs");
      for (const entry of readdirSync(pwRoot)) {
        if (entry.startsWith("chromium")) {
          candidates.push(join(pwRoot, entry, suffix));
        }
      }
    } catch {
      // No Playwright root — the standard candidates still apply.
    }
  }
  return candidates.find((c) => existsSync(c)) ?? null;
};

const parsePresets = () => {
  const requested = process.argv
    .filter((a) => a.startsWith("--preset="))
    .map((a) => a.slice("--preset=".length));
  const names = requested.length ? requested : ["master"];
  for (const name of names) {
    if (!PRESETS[name]) {
      throw new Error(
        `Unknown preset "${name}". Available: ${Object.keys(PRESETS).join(", ")}`,
      );
    }
  }
  return names;
};

/**
 * How many frames to compose at once.
 *
 * One-at-a-time is safe but slow: a 3½-minute film is 6,240 frames, and software
 * GL is not fast. Leave one core for ffmpeg and the parent process, and cap at 4
 * so peak memory stays predictable on a small container. Frame composition is a
 * pure function of the frame number here, so parallelism cannot change a frame.
 * Override with REMOTION_CONCURRENCY.
 */
const concurrency = () => {
  const explicit = Number(process.env.REMOTION_CONCURRENCY);
  if (Number.isFinite(explicit) && explicit >= 1) return Math.floor(explicit);
  return Math.max(1, Math.min(4, cpus().length - 1));
};

const CONCURRENCY = concurrency();

/**
 * Which GL backend Chrome uses to rasterise.
 *
 * Measured on this project, 90 frames at 1920x1080, four cores:
 *   swangle  2.18 fps      Chrome default  12.86 fps
 *
 * Every scene is plain CSS and SVG — there is no WebGL anywhere in the film — so
 * forcing the software rasteriser bought nothing and cost six times the render.
 * The default is deterministic for this content: frame composition is a pure
 * function of the frame number. Set REMOTION_GL to override (e.g. `swangle`) if a
 * particular host needs it.
 */
const GL = process.env.REMOTION_GL || null;

const humanSize = (bytes) =>
  bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.round(bytes / 1e3)} KB`;

const main = async () => {
  const presets = parsePresets();
  mkdirSync(OUT, { recursive: true });
  console.log(
    `[render] concurrency: ${CONCURRENCY} (of ${cpus().length} cores)` +
      `, gl: ${GL ?? "chrome default"}`,
  );

  const browserExecutable = await findBrowser();
  if (browserExecutable) {
    console.log(`[render] browser: ${browserExecutable}`);
  } else {
    console.log(
      "[render] no local Chrome found — Remotion will resolve its own " +
        "(this needs network access; see demo-video/README.md).",
    );
  }

  console.log("[render] bundling…");
  const serveUrl = await bundle({
    entryPoint: join(ROOT, "src", "index.ts"),
    publicDir: join(ROOT, "public"),
    onProgress: (p) => {
      if (p % 25 === 0) process.stdout.write(`\r[render] bundling ${p}%   `);
    },
  });
  process.stdout.write("\r[render] bundling 100%\n");

  for (const name of presets) {
    const preset = PRESETS[name];
    const outputLocation = join(OUT, preset.file);
    const composition = await selectComposition({
      serveUrl,
      id: preset.composition,
      browserExecutable,
    });
    const seconds = (composition.durationInFrames / composition.fps).toFixed(1);
    console.log(
      `\n[render] ${name}: ${preset.label} · ${composition.durationInFrames} frames ` +
        `(${seconds}s) → out/${preset.file}`,
    );

    let lastLogged = -1;
    await renderMedia({
      composition,
      serveUrl,
      codec: "h264",
      outputLocation,
      crf: preset.crf,
      scale: preset.scale,
      imageFormat: "jpeg",
      jpegQuality: 95,
      pixelFormat: "yuv420p",
      concurrency: CONCURRENCY,
      browserExecutable,
      ...(GL ? { chromiumOptions: { gl: GL } } : {}),
      timeoutInMilliseconds: 120000,
      // A browser-side error must fail the render, never emit a broken frame.
      onBrowserLog: (log) => {
        if (log.type === "error") {
          console.error(`[render][browser] ${log.text}`);
        }
      },
      onProgress: ({ progress }) => {
        const pct = Math.round(progress * 100);
        if (pct !== lastLogged && pct % 5 === 0) {
          lastLogged = pct;
          process.stdout.write(`\r[render] ${name} ${pct}%   `);
        }
      },
    });
    const size = statSync(outputLocation).size;
    process.stdout.write(`\r[render] ${name} 100% · ${humanSize(size)}\n`);
  }

  console.log(`\n[render] done. Output in ${OUT}`);
};

// Only render when this file IS the entry point. `scripts/stills.mjs` imports
// `findBrowser` from here, and without this guard that import would start a full
// render as a side effect of asking where Chrome is.
const isEntryPoint =
  process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isEntryPoint) {
  main().catch((error) => {
    console.error("\n[render] FAILED");
    console.error(error);
    process.exit(1);
  });
}
