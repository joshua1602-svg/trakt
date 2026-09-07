"""mi_agent_pptx.pptx_theme — brand theme for the investor PPTX pack.

Single source of truth for colours, fonts and chart styling, mirroring the
MI Agent **React** dashboard theme
(``frontend/mi-agent-ui/src/lib/theme.ts`` and ``src/index.css``) — NOT the
legacy Streamlit / light-theme PPTX generator.

SLATE & CYAN. The deck is a dark, enterprise, institutional-grade pack:
near-black slate surfaces, one cyan accent, Inter typography, tabular
figures. Matplotlib charts are rendered onto the *same* slate surface so
there are no white pasted boxes on the coloured slide background.

The ``peri`` field name is kept even though its value is now the cyan
accent (``#22d3ee``, not periwinkle) — it is read at ~50 call sites across
this package (chart_resolver.py, deck.py, pptx_builder.py, render.py), and
renaming it would touch every one of them for zero visual change. Only the
hex values move when the brand repaints; the field names are the stable
contract.

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
    navy: str = "#1c2027"          # PRIMARY (structural dark; was #232D55)
    peri: str = "#22d3ee"          # SECONDARY / accent (cyan; was periwinkle)
    accent: str = "#8893A8"

    # --- dark surfaces (from index.css design tokens) -----------------------
    bg_page: str = "#171a1f"       # --color-app-ground (page background)
    bg_panel: str = "#232830"      # --surface-dashboard (chart / card panel)
    bg_panel_alt: str = "#2c323c"  # --surface-artifact (alt panel)
    line: str = "#262a31"          # --color-line
    line_soft: str = "#1a1c20"     # --color-line-soft (grid)

    # --- ink / text ---------------------------------------------------------
    ink_100: str = "#eef1f2"       # primary text
    ink_300: str = "#9da4ab"       # secondary text
    ink_400: str = "#767d87"       # muted text
    ink_500: str = "#656b74"       # faint text / footers

    # --- semantic accents ---------------------------------------------------
    positive: str = "#2E7D5B"
    negative: str = "#B23A48"
    neutral: str = "#8893A8"
    mint: str = "#36c2a8"
    amber: str = "#e0a93b"
    rose: str = "#e0607a"

    # --- categorical series palette --------------------------------------
    # Mirrors frontend/mi-agent-ui/src/lib/theme.ts THEME.categorical exactly
    # — the dataviz skill's validated 8-hue set, slot 1 re-stepped to cyan-600
    # for the dark categorical band. See that file's comment for the
    # validation detail; keep the two lists identical.
    categorical: List[str] = field(default_factory=lambda: [
        "#0891b2", "#d95926", "#199e70", "#c98500",
        "#d55181", "#008300", "#9085e9", "#e66767",
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

    # --- sequential scale (slate -> cyan) for heatmaps ----------------------
    sequential: List[str] = field(default_factory=lambda: [
        "#101318", "#1c2027", "#22d3ee",
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
