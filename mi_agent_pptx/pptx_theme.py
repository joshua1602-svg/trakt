"""mi_agent_pptx.pptx_theme — brand theme for the investor PPTX pack.

Single source of truth for colours, fonts and chart styling, mirroring the
MI Agent **React** dashboard theme
(``frontend/mi-agent-ui/src/lib/theme.ts`` and ``src/index.css``) — NOT the
legacy Streamlit / light-theme PPTX generator.

The deck is a dark, enterprise, institutional-grade pack: near-black navy
surfaces, periwinkle secondary, Inter typography, tabular figures. Matplotlib
charts are rendered onto the *same* navy surface so there are no white pasted
boxes on the coloured slide background.

Nothing here performs I/O; it is a pure styling module so it can be imported by
both the chart renderer and the pptx assembler without side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert ``#rrggbb`` (or ``rrggbb``) to an ``(r, g, b)`` tuple."""
    s = hex_str.lstrip("#")
    if len(s) != 6:
        raise ValueError(f"expected a 6-digit hex colour, got {hex_str!r}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


@dataclass(frozen=True)
class PptxTheme:
    """Immutable brand theme mirroring the MI Agent React dashboard."""

    name: str = "trakt_mi_agent_dark"

    # --- brand palette (shared across React / Plotly / this deck) -----------
    navy: str = "#232D55"          # PRIMARY
    peri: str = "#919DD1"          # SECONDARY (periwinkle)
    accent: str = "#BFBFBF"

    # --- dark surfaces (from index.css design tokens) -----------------------
    bg_page: str = "#0c1024"       # --color-navy-950 (page background)
    bg_panel: str = "#12152b"      # --surface-dashboard (chart / card panel)
    bg_panel_alt: str = "#161d3a"  # --surface-artifact (alt panel)
    line: str = "#232b48"          # --color-line
    line_soft: str = "#1c2440"     # --color-line-soft (grid)

    # --- ink / text ---------------------------------------------------------
    ink_100: str = "#eef1f8"       # primary text
    ink_300: str = "#b8c0d6"       # secondary text
    ink_400: str = "#8c95b0"       # muted text
    ink_500: str = "#6b7493"       # faint text / footers

    # --- semantic accents ---------------------------------------------------
    positive: str = "#2E7D5B"
    negative: str = "#B23A48"
    neutral: str = "#8893A8"
    mint: str = "#36c2a8"
    amber: str = "#e0a93b"
    rose: str = "#e0607a"

    # --- categorical series palette (navy -> periwinkle ramp + hues) --------
    categorical: List[str] = field(default_factory=lambda: [
        "#232D55", "#3d4a82", "#5a67a8", "#919dd1",
        "#36c2a8", "#e0a93b", "#c46b8f",
    ])

    # --- RAG (risk monitor) -------------------------------------------------
    rag: Dict[str, str] = field(default_factory=lambda: {
        "green": "#2E7D5B",
        "amber": "#E0A93B",
        "red": "#B23A48",
        "below_minimum": "#5A6275",
        # aliases used across risk artefacts
        "within limit": "#2E7D5B",
        "approaching": "#E0A93B",
        "breach": "#B23A48",
        "needs_review": "#5A6275",
        "unavailable": "#5A6275",
    })

    # --- typography ---------------------------------------------------------
    # Inter is the React UI font; it may not be installed on every host, so the
    # chart renderer falls back gracefully to the sans stack. python-pptx just
    # names the font and PowerPoint substitutes if absent.
    font_sans: str = "Inter"
    font_fallbacks: Tuple[str, ...] = (
        "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif",
    )
    font_mono: str = "DejaVu Sans Mono"

    # --- sequential scale (navy -> periwinkle) for heatmaps -----------------
    sequential: List[str] = field(default_factory=lambda: [
        "#11162e", "#3d4a82", "#919dd1",
    ])

    # ---------------------------------------------------------------- helpers
    def rag_color(self, status: str) -> str:
        """Resolve a RAG/status string to a hex colour, defaulting to neutral."""
        if not status:
            return self.rag["below_minimum"]
        return self.rag.get(str(status).strip().lower(), self.neutral)

    def categorical_color(self, index: int) -> str:
        """Cycle the categorical palette for series index *index*."""
        pal = self.categorical
        return pal[index % len(pal)]

    def rgb(self, hex_str: str) -> Tuple[int, int, int]:
        return hex_to_rgb(hex_str)


# Default shared instance.
THEME = PptxTheme()
