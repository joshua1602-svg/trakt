/**
 * Brand + chart palette — single source of truth, mirroring the Python theme
 * (mi_agent_pptx/pptx_theme.py, which generates the investor deck). NOT
 * analytics/charts_plotly.py — that file is legacy, imported only by the
 * standalone analytics/streamlit_app_erm.py prototype, unreachable from the
 * live product.
 *
 * SLATE & CYAN. `cyan` is the one brand/data accent — the same hue as the
 * UI's interface-state accent (index.css --color-cyan-400) — used at full
 * brightness for a single dominant series (a bar chart, a line, a scatter)
 * because it only has to stand out from the surface, not from seven
 * neighbours. `categorical` is a SEPARATE, validated 8-hue set for charts
 * with several simultaneous series (treemap, cohort vintages): the
 * dataviz skill's reference palette, order and hexes unchanged, with slot 1
 * re-stepped to cyan-600 (a darker step than the UI accent — cyan-400 is
 * provably too light for the dark-mode categorical band, per
 * `node scripts/validate_palette.js`) and re-validated against this app's
 * actual dashboard surface (#232830): all six checks pass except one WARN
 * (slot 6 green sits under 3:1 contrast — every categorical chart here
 * already ships a legend, which is the required relief channel).
 */
export const THEME = {
  navy: "#1c2027", // structural dark (mirrors --color-navy-800)
  cyan: "#22d3ee", // PRIMARY accent — single-series bars/lines/scatter
  accent: "#8893A8",
  positive: "#2E7D5B",
  negative: "#B23A48",
  neutral: "#8893A8",
  // Validated categorical set (dataviz skill reference palette, slot 1
  // re-stepped to cyan-600 for the dark categorical band; see comment above).
  categorical: [
    "#0891b2", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
  ],
  // RAG colours
  rag: {
    green: "#2E7D5B",
    amber: "#E0A93B",
    red: "#B23A48",
    below_minimum: "#5A6275",
  },
} as const;

/** Movement-type colours for migration matrices. */
export const MOVEMENT_COLORS: Record<string, string> = {
  improved: THEME.positive,
  deteriorated: THEME.negative,
  unchanged: THEME.navy,
  new: THEME.cyan,
  exited: THEME.accent,
  changed: THEME.neutral,
};
