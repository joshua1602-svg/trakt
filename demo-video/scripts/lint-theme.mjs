/**
 * Enforce the storyboard's pre-render checklist mechanically.
 *
 * The checklist says "no hex value, font size or duration appears outside
 * theme.ts" and "no box shadows, glows or gradients survive". Those are only real
 * if something fails the build when they reappear, so this does:
 *
 *   1. no hex colour literal outside src/theme.ts
 *   2. no `fontSize:` numeric literal outside src/theme.ts — sizes must be a token,
 *      optionally scaled by a token
 *   3. no box-shadow, filter: blur/drop-shadow, textShadow or gradient anywhere
 *   4. exactly six type sizes exist, and every one of them is used
 *   5. `theme.color.flag` is referenced in exactly two scenes
 *
 * Run as part of `npm run render`, before a frame is drawn.
 */

import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
const SRC = join(ROOT, "src");
const THEME = join(SRC, "theme.ts");

const walk = (dir) =>
  readdirSync(dir).flatMap((entry) => {
    const full = join(dir, entry);
    return statSync(full).isDirectory() ? walk(full) : [full];
  });

const sourceFiles = walk(SRC).filter(
  (f) => /\.tsx?$/.test(f) && !f.endsWith(".test.ts") && f !== THEME,
);

const findings = [];
const flagFiles = new Set();

const RULES = [
  {
    id: "hex-colour",
    // A hex colour literal. theme.ts is the only place these may appear.
    pattern: /#[0-9a-fA-F]{3,8}\b/g,
    message: "hex colour literal — use a theme.color token",
  },
  {
    id: "font-size-literal",
    // `fontSize: theme.type.body.fontSize * 0.8` is fine: the size still originates
    // in theme.ts. A bare number or a string literal is not.
    pattern: /fontSize:\s*(?!theme\.)(?:\d|"|')/g,
    message: "literal font size — use a theme.type token",
  },
  {
    id: "shadow",
    pattern: /box-?[Ss]hadow|text-?[Ss]hadow|drop-?shadow|filter:\s*blur/g,
    message: "shadow or blur — depth comes from a 1px rule and nothing else",
  },
  {
    id: "gradient",
    pattern: /linear-gradient|radial-gradient|conic-gradient/g,
    message: "gradient — panels are one flat colour",
  },
];

for (const file of sourceFiles) {
  const text = readFileSync(file, "utf8");
  const rel = relative(ROOT, file);
  if (/theme\.color\.flag|tone="flag"|tone={"flag"}/.test(text)) flagFiles.add(rel);
  for (const rule of RULES) {
    for (const match of text.matchAll(rule.pattern)) {
      const line = text.slice(0, match.index).split("\n").length;
      findings.push(`${rel}:${line}  ${rule.message}  (${match[0].trim()})`);
    }
  }
}

// --------------------------------------------------------------------------- //
// The mono rule: a computed value is never set in a display or body face.
// --------------------------------------------------------------------------- //
// `<Counter>` only ever renders something the pipeline produced, so it is the reliable
// marker for a computed value. It must sit inside `<Figure>` or `<Stat>` — the two
// components that guarantee the data face and `tabular-nums`. A scene that wraps a
// counter in its own styled div can silently put a governed figure in Archivo, which is
// exactly the violation the rule exists to prevent.
const MONO_HOSTS = ["<Figure", "<Stat", "<Counter"];
for (const file of sourceFiles) {
  if (!file.includes(`${sep}scenes${sep}`)) continue;
  const text = readFileSync(file, "utf8");
  const rel = relative(ROOT, file);
  for (const match of text.matchAll(/<Counter\b/g)) {
    // Walk backwards to the nearest opening component tag and check what it is.
    const before = text.slice(0, match.index);
    const host = MONO_HOSTS.map((tag) => ({ tag, at: before.lastIndexOf(tag) }))
      .filter((h) => h.at >= 0)
      .sort((a, b) => b.at - a.at)[0];
    const line = before.split("\n").length;
    if (!host || host.tag === "<Counter") {
      findings.push(
        `${rel}:${line}  a <Counter> outside <Figure>/<Stat> — a computed value must be ` +
          "mono with tabular-nums",
      );
      continue;
    }
    // And nothing between the host and the counter may re-declare a face.
    const between = text.slice(host.at, match.index);
    if (/fontFamily|theme\.type\.(display|headline|body|label)\b/.test(between)) {
      findings.push(
        `${rel}:${line}  a display or body face is applied around a <Counter> — ` +
          "computed values are IBM Plex Mono",
      );
    }
  }
}

// Six type sizes, all of them used.
const themeText = readFileSync(THEME, "utf8");
const typeStart = themeText.indexOf("export const type = {");
const typeBlock = themeText.slice(typeStart, themeText.indexOf("\n} as const;", typeStart));
const sizes = [...typeBlock.matchAll(/^ {2}(\w+): \{$/gm)].map((m) => m[1]);
if (sizes.length !== 6) {
  findings.push(`src/theme.ts  ${sizes.length} type sizes declared, expected exactly 6`);
}
const allSource = sourceFiles.map((f) => readFileSync(f, "utf8")).join("\n");
for (const size of sizes) {
  if (!allSource.includes(`theme.type.${size}`)) {
    findings.push(`src/theme.ts  type size '${size}' is declared but never used`);
  }
}

// `flag` appears in exactly two places in the film.
const flagScenes = [...flagFiles].filter((f) => f.includes("src/scenes/"));
if (flagScenes.length !== 2) {
  findings.push(
    `flag is used in ${flagScenes.length} scene(s) [${flagScenes.join(", ")}] — ` +
      "the storyboard allows exactly two",
  );
}

if (findings.length) {
  console.error("[lint-theme] FAILED\n");
  for (const finding of findings) console.error(`  ${finding}`);
  console.error(`\n[lint-theme] ${findings.length} finding(s)`);
  process.exit(1);
}

const counters = sourceFiles
  .filter((f) => f.includes(`${sep}scenes${sep}`))
  .reduce((n, f) => n + [...readFileSync(f, "utf8").matchAll(/<Counter\b/g)].length, 0);

console.log(
  `[lint-theme] PASS — ${sourceFiles.length} files, ${sizes.length} type sizes, ` +
    `flag in ${flagScenes.length} scenes, ${counters} counters all in the data face`,
);
