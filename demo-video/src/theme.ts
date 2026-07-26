/**
 * The film's brand tokens. Single source of truth.
 *
 * No scene or component file may declare a colour, font size, weight, radius or
 * duration inline. Every value on screen is referenced by token name from here, and
 * `scripts/lint-theme.mjs` fails the render if a hex code, a `px` font size, a box
 * shadow or a gradient appears anywhere under `src/scenes` or `src/components`.
 *
 * Two rules in this file carry more weight than the rest:
 *
 *   `signal` is a scarcity resource. At most ONE element on screen carries it in
 *   any given frame. If two things are cyan, neither is important.
 *
 *   The mono role is SEMANTIC. If a figure came out of the pipeline — a balance, a
 *   count, an LTV, a field name, a portfolio code, a filename, a timestamp — it is
 *   set in mono. If it is Trakt talking about itself, it is Archivo or Inter. The
 *   viewer never has to be told which numbers are governed; the typeface says so.
 */

import { staticFile } from "remotion";

// --------------------------------------------------------------------------- //
// Colour
// --------------------------------------------------------------------------- //
export const color = {
  /** Ground. Every scene. Never a gradient. */
  ink: "#060B1F",
  /** Panels and surfaces. One elevation only. */
  hull: "#0F1D4D",
  /** Hairlines, dividers, inactive states, grid. */
  rule: "#24356E",
  /** Primary type. */
  paper: "#E8ECF7",
  /** Secondary type, labels, stamps. */
  mute: "#8B99C4",
  /** The accent. Reserved for the value being proven in that scene. */
  signal: "#4DE0C4",
  /** Exceptions and referred-for-review. Appears exactly twice in the film. */
  flag: "#F2A93B",
} as const;

// --------------------------------------------------------------------------- //
// Typography
// --------------------------------------------------------------------------- //
/** Vendored under public/fonts by scripts/vendor-fonts.mjs — no network at render. */
export const fontFaces = [
  {
    family: "Archivo",
    file: "fonts/archivo-latin.woff2",
    /** Variable file: declare the full axis so 700 is a real weight, not synthetic. */
    weight: "100 900",
  },
  { family: "Inter", file: "fonts/inter-latin.woff2", weight: "100 900" },
  { family: "IBM Plex Mono", file: "fonts/ibm-plex-mono-500-latin.woff2", weight: "500" },
] as const;

/** The `@font-face` block the film injects once, at the top of the composition. */
export const fontFaceCss = (): string =>
  fontFaces
    .map(
      (f) =>
        `@font-face{font-family:"${f.family}";font-style:normal;` +
        `font-weight:${f.weight};font-display:block;` +
        `src:url("${staticFile(f.file)}") format("woff2");}`,
    )
    .join("");

export const family = {
  /** Claims. The line the viewer must remember. */
  display: '"Archivo", system-ui, sans-serif',
  /** Supporting sentences, captions, narration burn-in. */
  body: '"Inter", system-ui, sans-serif',
  /** Anything Trakt computed, or any source identifier. */
  data: '"IBM Plex Mono", ui-monospace, monospace',
} as const;

export const weight = {
  regular: 400,
  medium: 500,
  bold: 700,
} as const;

/**
 * Six type sizes. Nothing else exists.
 *
 * Each entry is a complete, ready-to-spread CSSProperties object, so a scene can
 * never assemble a seventh size out of parts.
 */
export const type = {
  display: {
    fontFamily: family.display,
    fontWeight: weight.bold,
    fontSize: 128,
    lineHeight: 0.95,
    letterSpacing: "-0.02em",
  },
  headline: {
    fontFamily: family.display,
    fontWeight: weight.bold,
    fontSize: 64,
    lineHeight: 1.05,
    letterSpacing: "-0.02em",
  },
  stat: {
    fontFamily: family.data,
    fontWeight: weight.medium,
    fontSize: 88,
    lineHeight: 1,
    fontVariantNumeric: "tabular-nums",
    fontFeatureSettings: '"tnum" 1',
  },
  body: {
    fontFamily: family.body,
    fontWeight: weight.regular,
    fontSize: 26,
    lineHeight: 1.45,
  },
  label: {
    fontFamily: family.body,
    fontWeight: weight.medium,
    fontSize: 14,
    lineHeight: 1.2,
    letterSpacing: "0.1em",
    textTransform: "uppercase" as const,
  },
  stamp: {
    fontFamily: family.data,
    fontWeight: weight.medium,
    fontSize: 15,
    lineHeight: 1.2,
    fontVariantNumeric: "tabular-nums",
    fontFeatureSettings: '"tnum" 1',
  },
} as const;

// --------------------------------------------------------------------------- //
// Motion
// --------------------------------------------------------------------------- //
/**
 * One entrance, one spring, one stagger, film-wide. Nothing scales in. Nothing
 * bounces. The restraint is the point: this is a governance product being sold to
 * a regulated firm.
 */
export const motion = {
  quick: 12,
  base: 20,
  slow: 30,
  /** Critically damped — no overshoot. */
  spring: { damping: 200, mass: 0.6 },
  /** The single entrance: opacity 0 -> 1 with a 12px rise. */
  enterRise: 12,
  /** Frames between siblings in a staggered group. */
  stagger: 3,
} as const;

// --------------------------------------------------------------------------- //
// Layout
// --------------------------------------------------------------------------- //
/**
 * Two layouts share every scene component. `wide` is the 1920x1080 master;
 * `square` is the 1080x1080 LinkedIn and email variant, which is the same film with
 * a tighter safe area and a smaller display size — not a re-edit.
 */
export type Layout = "wide" | "square";

export const layout = {
  wide: {
    width: 1920,
    height: 1080,
    /** Chrome inset, and the scene safe area. */
    edge: 40,
    gutter: 72,
    /** Display claims shrink in the square crop so two lines still fit. */
    displayScale: 1,
    statScale: 1,
    /** Bottom room the captions plate occupies. */
    captionReserve: 200,
  },
  square: {
    width: 1080,
    height: 1080,
    edge: 32,
    gutter: 48,
    displayScale: 0.62,
    statScale: 0.72,
    captionReserve: 210,
  },
} as const;

/**
 * The Trakt lockup. One fixed size for the whole film — it never scales, moves or
 * animates after the close, which is the only time chrome moves at all.
 *
 * Deliberately NOT one of the six type sizes: it is a brand asset with its own
 * optical size, not a role in the type scale, and treating it as a seventh size
 * would be the start of the drift the storyboard is trying to stop.
 */
export const lockup = {
  wordSize: 30,
  markSize: 26,
  markStroke: 1.5,
  gap: 14,
} as const;

export const radius = { card: 4, plate: 6 } as const;

/** The only line weight in the film. Depth comes from this and nothing else. */
export const hairline = 1;

/**
 * The one exception: S1's failed connector. A 1px amber line on `ink` antialiases
 * toward grey and stops reading as `flag` at all, which defeats the point of the beat.
 * Two pixels is the minimum at which the accent survives, and it is used nowhere else.
 */
export const accentLine = 2;

/** Caption plate opacity over `ink`, per the storyboard. */
export const captionPlateOpacity = 0.7;

export const theme = {
  color,
  lockup,
  accentLine,
  family,
  weight,
  type,
  motion,
  layout,
  radius,
  hairline,
  captionPlateOpacity,
  fontFaces,
  fontFaceCss,
} as const;

export default theme;
