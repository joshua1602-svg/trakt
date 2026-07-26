/**
 * Vendor the film's three typefaces into `public/fonts/`.
 *
 * The storyboard asks for the faces to be loaded through `@remotion/google-fonts`
 * "so renders are deterministic". That package resolves its `@font-face` sources
 * to `https://fonts.gstatic.com/...` and fetches them at FRAME time, which makes a
 * render depend on the network — something this project is explicitly not allowed
 * to do. So the URLs still come from `@remotion/google-fonts` (they are provably
 * the Google Fonts originals), but the files are downloaded ONCE, here, and the
 * render reads them from disk through `staticFile()`.
 *
 * Run with network access:
 *
 *     node scripts/vendor-fonts.mjs
 *
 * It writes the woff2 files plus `public/fonts/manifest.json`, which records the
 * source URL and SHA-256 of each file. The manifest is committed, so a later run
 * can be checked against it — and `npm run fonts:verify` fails if a file drifts.
 *
 * All three faces are SIL Open Font License 1.1.
 */

import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const PROJECT = join(HERE, "..");
const FONT_DIR = join(PROJECT, "public", "fonts");
const MANIFEST = join(FONT_DIR, "manifest.json");

/** module name in @remotion/google-fonts, weight, subset, output file. */
const WANTED = [
  { module: "Archivo", weight: "700", subset: "latin", file: "archivo-latin.woff2", licence: "OFL-1.1" },
  { module: "Inter", weight: "400", subset: "latin", file: "inter-latin.woff2", licence: "OFL-1.1" },
  { module: "IBMPlexMono", weight: "500", subset: "latin", file: "ibm-plex-mono-500-latin.woff2", licence: "OFL-1.1" },
];

const sha256 = (buf) => `sha256:${createHash("sha256").update(buf).digest("hex")}`;

const sourceUrl = async ({ module: name, weight, subset }) => {
  const mod = await import(`@remotion/google-fonts/${name}`);
  const info = mod.getInfo();
  const url = info.fonts?.normal?.[weight]?.[subset];
  if (!url) {
    throw new Error(`${name}: no normal/${weight}/${subset} source in @remotion/google-fonts`);
  }
  return { url, family: info.fontFamily };
};

const main = async () => {
  const verify = process.argv.includes("--verify");
  mkdirSync(FONT_DIR, { recursive: true });
  const entries = [];

  for (const want of WANTED) {
    const { url, family } = await sourceUrl(want);
    const target = join(FONT_DIR, want.file);

    if (verify) {
      if (!existsSync(target)) throw new Error(`missing vendored font ${want.file}`);
      entries.push({ ...want, family, url, sha256: sha256(readFileSync(target)) });
      continue;
    }

    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    writeFileSync(target, buf);
    entries.push({ ...want, family, url, sha256: sha256(buf), bytes: buf.length });
    console.log(`  ${want.file}  ${(buf.length / 1024).toFixed(1)} KB  ${family} ${want.weight}`);
  }

  if (verify) {
    const recorded = JSON.parse(readFileSync(MANIFEST, "utf8"));
    const drift = entries.filter((e) => {
      const was = recorded.fonts.find((f) => f.file === e.file);
      return !was || was.sha256 !== e.sha256;
    });
    if (drift.length) {
      console.error("Vendored fonts do not match the manifest:", drift.map((d) => d.file));
      process.exit(1);
    }
    console.log(`  ${entries.length} vendored fonts match the manifest`);
    return;
  }

  writeFileSync(
    MANIFEST,
    `${JSON.stringify(
      {
        note:
          "Vendored from Google Fonts at build time so the render never touches " +
          "the network. Regenerate with `node scripts/vendor-fonts.mjs`; check " +
          "with `--verify`.",
        licence: "All three faces are SIL Open Font License 1.1.",
        fonts: entries,
      },
      null,
      2,
    )}\n`,
  );
  console.log(`  wrote ${MANIFEST}`);
};

await main();
