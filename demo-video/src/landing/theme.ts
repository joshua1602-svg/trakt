/**
 * Landing-page tokens for the controls demo composition.
 *
 * This is NOT the film theme. The controls demo plays inline inside the
 * landing page's Risk & Controls section, so it must read as that page —
 * the values below are carried verbatim from
 * `landing-page/src/app/globals.css` (which itself carries the product's
 * scale from `frontend/mi-agent-ui/src/index.css`). The film's theme lint
 * covers `src/scenes` and `src/components`; this directory declares its own
 * single source of tokens for the same reason the film does.
 *
 * Accent discipline matches the page: mint marks a proven/passing state,
 * amber marks review/warning and the illustrative provenance stamp, rose
 * marks a projected breach, periwinkle is structural emphasis. Inside this
 * product depiction the product's RAG semantics apply — see the token note
 * in `landing-page/src/app/globals.css`.
 */

import { staticFile } from "remotion";

export const color = {
  /** Page ground behind the card. */
  ground: "#0c1024",
  /** The card surface (navy-900 composited over the ground). */
  surface: "#10152b",
  /** Raised chips and inner panels. */
  raised: "#161c38",
  /** Inset document / monitor panels. */
  inset: "#0d1226",
  line: "#232b48",
  lineSoft: "#1c2440",
  ink100: "#eef1f8",
  ink200: "#d5dbe9",
  ink300: "#b8c0d6",
  ink400: "#8c95b0",
  ink500: "#6b7493",
  peri: "#919dd1",
  periDeep: "#7d8ac6",
  mint: "#36c2a8",
  amber: "#e0a93b",
  rose: "#e0607a",
} as const;

/**
 * Inter, vendored for the film, doubles as the closest deterministic stand-in
 * for the landing page's system-ui stack — the render must not depend on
 * whatever fontconfig the container ships.
 */
export const fontFaceCss = (): string =>
  `@font-face{font-family:"Inter";font-style:normal;font-weight:100 900;` +
  `font-display:block;src:url("${staticFile("fonts/inter-latin.woff2")}") format("woff2");}`;

export const family = '"Inter", system-ui, sans-serif';

/** Designed at 1200x960; the page displays it at roughly 40% scale. */
export const size = {
  chrome: 20,
  docTitle: 24,
  docBody: 23,
  label: 18,
  row: 24,
  chip: 18,
  value: 26,
  keyline: 34,
  arrow: 30,
  tick: 15,
  /* Added for the agent-to-agent composition. It shows a governed call trace —
     thirty rows of tool names — which needs a step below `chip` to fit two
     columns without truncating `list_validation_exceptions`, and a display
     size for the one number the whole sequence drives to. */
  seq: 14,
  trace: 17,
  node: 21,
  meter: 22,
  figure: 62,
} as const;

export const motion = {
  /** Frames of the single entrance (opacity + 12px rise, no overshoot). */
  enter: 18,
  rise: 12,
  spring: { damping: 200, mass: 0.6 },
} as const;

export const radius = { card: 16, panel: 12, row: 10, chip: 999 } as const;

const theme = { color, family, size, motion, radius, fontFaceCss } as const;
export default theme;
