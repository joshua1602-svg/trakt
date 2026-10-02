#!/usr/bin/env python3
"""make_icons.py — draw the Teams app icons from the product's own palette.

Writes ``color.png`` (192x192, full colour) and ``outline.png`` (32x32, white
glyph on a transparent background — what Teams requires of the outline icon).

The mark is the "T and dot" already used as the website favicon
(``landing-page/src/app/icon.svg``), recoloured to the dashboard / investor-deck
theme (``mi_agent_pptx/pptx_theme.py``): slate ``#1c2027`` ground, cyan
``#22d3ee`` stroke, near-white dot. There is no separate logo asset.

Run only when the icons need to change; the PNGs are committed and
``package_agent.py`` ships them. Needs Pillow, which the build itself does not.

    python deploy/copilot-agent/make_icons.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent

SLATE = (0x1C, 0x20, 0x27, 255)   # pptx_theme.navy
CYAN = (0x22, 0xD3, 0xEE, 255)    # pptx_theme.peri / THEME.cyan
INK = (0xEE, 0xF1, 0xF2, 255)     # pptx_theme.ink_100
WHITE = (255, 255, 255, 255)

SS = 8  # supersampling factor, for clean edges at small sizes


def _mark(draw: ImageDraw.ImageDraw, *, size: int, box: float, stroke_rgba,
          dot_rgba, stroke_w: float) -> None:
    """Draw the T and dot centred in a ``box`` fraction of the canvas."""
    u = size * box / 16.0                    # one design unit
    ox = size * (1 - box) / 2
    oy = size * (1 - box) / 2 + size * box * 0.02

    def p(x: float, y: float) -> tuple[float, float]:
        return ox + x * u, oy + y * u

    w = stroke_w * u
    # crossbar
    x0, y0 = p(1.0, 2.2)
    x1, y1 = p(15.0, 2.2)
    draw.rounded_rectangle([x0 - w / 2, y0 - w / 2, x1 + w / 2, y1 + w / 2],
                           radius=w / 2, fill=stroke_rgba)
    # stem
    sx, sy0 = p(8.0, 2.2)
    _, sy1 = p(8.0, 10.6)
    draw.rounded_rectangle([sx - w / 2, sy0 - w / 2, sx + w / 2, sy1 + w / 2],
                           radius=w / 2, fill=stroke_rgba)
    # dot
    dx, dy = p(8.0, 14.4)
    r = 1.15 * u
    draw.ellipse([dx - r, dy - r, dx + r, dy + r], fill=dot_rgba)


def _render(size: int, *, background, box: float, stroke_rgba, dot_rgba,
            stroke_w: float) -> Image.Image:
    big = size * SS
    img = Image.new("RGBA", (big, big), background or (0, 0, 0, 0))
    _mark(ImageDraw.Draw(img), size=big, box=box, stroke_rgba=stroke_rgba,
          dot_rgba=dot_rgba, stroke_w=stroke_w)
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    color = _render(192, background=SLATE, box=0.56, stroke_rgba=CYAN,
                    dot_rgba=INK, stroke_w=2.0)
    color.save(HERE / "color.png", optimize=True)
    # Teams: outline icon is a single-colour glyph on transparency, in 32x32.
    outline = _render(32, background=None, box=0.72, stroke_rgba=WHITE,
                      dot_rgba=WHITE, stroke_w=2.4)
    outline.save(HERE / "outline.png", optimize=True)
    print("wrote color.png (192x192) and outline.png (32x32)")


if __name__ == "__main__":
    main()
