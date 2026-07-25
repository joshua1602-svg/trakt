/**
 * Trakt visual identity, mirrored from the product.
 *
 * Sourced verbatim from:
 *   frontend/mi-agent-ui/src/lib/theme.ts     (brand + categorical palette)
 *   frontend/mi-agent-ui/src/index.css        (@theme design tokens)
 *   analytics/charts_plotly.py                (the Python chart theme)
 *
 * Nothing here is invented. If the product's palette changes, change it there
 * and mirror it here — a film that does not match the product is worse than no
 * film.
 */

/** Brand + chart palette (frontend/mi-agent-ui/src/lib/theme.ts). */
export const THEME = {
  navy: "#232D55", // PRIMARY
  peri: "#919DD1", // SECONDARY
  accent: "#BFBFBF",
  positive: "#2E7D5B",
  negative: "#B23A48",
  neutral: "#8893A8",
  categorical: [
    "#232D55",
    "#3d4a82",
    "#5a67a8",
    "#919dd1",
    "#36c2a8",
    "#e0a93b",
    "#c46b8f",
  ],
  rag: {
    green: "#2E7D5B",
    amber: "#E0A93B",
    red: "#B23A48",
    belowMinimum: "#5A6275",
  },
} as const;

/** Surface + ink tokens (frontend/mi-agent-ui/src/index.css @theme). */
export const C = {
  navy950: "#0c1024",
  navy900: "#11162e",
  navy850: "#161c38",
  navy800: "#1b2240",
  navy700: "#232d55",
  navy600: "#2f3a68",
  navy500: "#3d4a82",

  peri400: "#919dd1",
  peri300: "#aab4dd",
  peri200: "#c6cdea",

  mint400: "#36c2a8",
  amber400: "#e0a93b",
  rose400: "#e0607a",

  ink100: "#eef1f8",
  ink300: "#b8c0d6",
  ink400: "#8c95b0",
  ink500: "#6b7493",

  line: "#232b48",
  lineSoft: "#1c2440",

  surfaceDashboard: "#12152b",
  surfaceDashboardLine: "#283154",
  surfaceArtifact: "#23262f",
  surfaceArtifactLine: "#383c47",
  surfaceChat: "#131a33",
  surfaceChatLine: "#243059",
} as const;

/**
 * Typography. The product ships Inter with a system fallback stack; the render
 * environment has no network access, so the stack resolves to whatever the
 * headless browser has. The scale below is chosen to read cleanly in either.
 */
export const FONT = {
  sans:
    'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, ' +
    'Helvetica, Arial, "Liberation Sans", sans-serif',
  mono: 'ui-monospace, "SF Mono", "JetBrains Mono", "DejaVu Sans Mono", Menlo, monospace',
} as const;

/** Type scale, in px at 1920x1080. */
export const TYPE = {
  hero: 88,
  display: 62,
  title: 44,
  heading: 32,
  subheading: 25,
  body: 21,
  bodySmall: 18,
  label: 15,
  micro: 13,
} as const;

/** Layout constants for the 1920x1080 frame. */
export const LAYOUT = {
  width: 1920,
  height: 1080,
  /** Outer safe margin — nothing meaningful sits outside this. */
  margin: 128,
  gutter: 32,
  radius: 14,
  radiusSmall: 8,
} as const;

/** The page background used by every scene, mirroring the product's body style. */
export const PAGE_BACKGROUND =
  `radial-gradient(1200px 600px at 80% -10%, rgba(145,157,209,0.10), transparent 60%), ${C.navy950}`;

/** A restrained panel border, used consistently for every card. */
export const PANEL_BORDER = `1px solid ${C.line}`;

/** The discreet footer disclaimer carried on every frame. */
export const SYNTHETIC_FOOTER =
  "Synthetic demonstration data — not a real customer";
