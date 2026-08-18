/**
 * Type-scale lint.
 *
 * The page had twelve distinct font sizes before the scale existed, in
 * twenty-one size-and-weight combinations, and none of them were chosen —
 * they accumulated one component at a time. Five tokens replaced them
 * (`app/globals.css`), and this is what stops the thirteenth from appearing:
 * a raw size class anywhere under `src/` fails the build.
 *
 * It refuses arbitrary values (`text-[15px]`) and Tailwind's own size scale
 * (`text-sm`, `text-2xl`) alike. The second is the one worth naming: nothing
 * about `text-sm` looks wrong in a diff, which is exactly how a page ends up
 * with 14px doing four unrelated jobs.
 *
 * `globals.css` is exempt because it is where the tokens are declared and
 * where the old sizes are written down as the record of what went.
 *
 * Deliberately not checked here: font WEIGHT. The scale owns size only —
 * `font-medium` and `font-semibold` stay component decisions, because a scale
 * that owned both would need twenty tokens to say what five plus two
 * utilities already say.
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = join(ROOT, "src");

/** The five tokens. Anything else that sets a font size is a regression. */
const ALLOWED = new Set(["display", "headline", "subhead", "body", "small"]);

/** Exempt: the file that declares the tokens and records what they replaced. */
const EXEMPT = new Set(["src/app/globals.css"]);

/**
 * `text-*` is overloaded in Tailwind — `text-ink-100` is a colour, not a size.
 * Matching every `text-…` and filtering by shape would let a new colour name
 * fail the lint, so this matches the two things that actually set a size:
 * an arbitrary value in px/rem/em, and Tailwind's named size scale.
 */
const ARBITRARY = /\btext-\[\s*[\d.]+(px|rem|em)\s*\]/g;
const NAMED = /\btext-(xs|sm|base|lg|xl|[2-9]xl)\b/g;
const TOKEN = /\btext-([a-z]+)\b/g;

const files = [];
const walk = (dir) => {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) walk(full);
    else if ([".ts", ".tsx", ".css"].includes(extname(full))) files.push(full);
  }
};
walk(SRC);

const offences = [];
for (const file of files) {
  const rel = relative(ROOT, file);
  if (EXEMPT.has(rel)) continue;
  const source = readFileSync(file, "utf8");
  source.split("\n").forEach((line, i) => {
    for (const pattern of [ARBITRARY, NAMED]) {
      pattern.lastIndex = 0;
      for (const match of line.matchAll(pattern)) {
        offences.push({ rel, line: i + 1, found: match[0] });
      }
    }
  });
}

// A typo'd token (`text-headine`) would slip past both patterns above and
// silently render at the inherited size, which is worse than a raw class
// because nothing looks wrong. Catch anything token-shaped that is not one.
const COLOUR_FAMILIES = new Set(["navy", "peri", "mint", "amber", "rose", "ink", "line"]);
for (const file of files) {
  const rel = relative(ROOT, file);
  if (EXEMPT.has(rel)) continue;
  const source = readFileSync(file, "utf8");
  source.split("\n").forEach((line, i) => {
    TOKEN.lastIndex = 0;
    for (const match of line.matchAll(TOKEN)) {
      const word = match[1];
      if (ALLOWED.has(word) || COLOUR_FAMILIES.has(word)) continue;
      if (/^(xs|sm|base|lg|xl|[2-9]xl)$/.test(word)) continue; // already reported
      if (["balance", "left", "right", "center", "wrap", "nowrap", "ellipsis"].includes(word)) {
        continue; // text-balance and friends are not sizes
      }
      offences.push({ rel, line: i + 1, found: match[0], note: "not a scale token" });
    }
  });
}

if (offences.length) {
  console.error(`[lint-type] FAIL — ${offences.length} raw font size(s):\n`);
  for (const o of offences) {
    console.error(`  ${o.rel}:${o.line}  ${o.found}${o.note ? `  (${o.note})` : ""}`);
  }
  console.error(`\n  Use one of: ${[...ALLOWED].map((t) => `text-${t}`).join(", ")}`);
  process.exit(1);
}

console.log(
  `[lint-type] PASS — ${files.length} files, ${ALLOWED.size} type tokens, no raw sizes`,
);
