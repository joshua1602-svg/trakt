/**
 * Brand + chart palette — single source of truth, mirroring the Python theme
 * (analytics/charts_plotly.py and mi_agent/mi_chart_factory.DEFAULT_THEME).
 */
export const THEME = {
  navy: "#232D55", // PRIMARY
  peri: "#919DD1", // SECONDARY
  accent: "#BFBFBF",
  positive: "#2E7D5B",
  negative: "#B23A48",
  neutral: "#8893A8",
  // Categorical series palette (navy → periwinkle ramp + supporting hues)
  categorical: ["#232D55", "#3d4a82", "#5a67a8", "#919dd1", "#36c2a8", "#e0a93b", "#c46b8f"],
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
  unchanged: "#3d4a82",
  new: THEME.peri,
  exited: THEME.accent,
  changed: THEME.neutral,
};
