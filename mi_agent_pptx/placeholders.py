"""mi_agent_pptx.placeholders — branded placeholders & appendix coverage notes.

When an artifact, field or chart the deck expects is unavailable, the deck must
degrade to an *institutional-grade branded placeholder* (not a crash and not a
blank slide) and record a coverage note for the methodology/appendix. This
module owns both:

* :func:`render_placeholder_png` — a dark, on-brand placeholder image that
  aligns to the slide background (no white pasted box), with a title and a
  concise "unavailable" message.
* :class:`AppendixNotes` — an accumulator for coverage/limitation notes that the
  builder renders on the Methodology / Notes slide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless, deterministic rendering
import matplotlib.pyplot as plt  # noqa: E402

from .pptx_theme import PptxTheme, THEME  # noqa: E402


def render_placeholder_png(
    out_path: str | Path,
    title: str,
    message: str = "Data artifact unavailable for this run",
    theme: PptxTheme = THEME,
    *,
    subtitle: str = "trakt MI Agent",
    width_in: float = 9.6,
    height_in: float = 4.6,
    dpi: int = 200,
) -> Path:
    """Render a branded 'unavailable' placeholder PNG and return its path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor(theme.bg_panel)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(theme.bg_panel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Dashed brand frame to read as an intentional placeholder, not an error.
    frame = plt.Rectangle(
        (0.03, 0.08), 0.94, 0.84, fill=False, linewidth=1.4,
        edgecolor=theme.line, linestyle=(0, (6, 4)),
    )
    ax.add_patch(frame)

    ax.text(0.5, 0.60, title, ha="center", va="center",
            color=theme.ink_100, fontsize=17, fontweight="bold")
    ax.text(0.5, 0.44, message, ha="center", va="center",
            color=theme.ink_400, fontsize=11)
    ax.text(0.5, 0.16, subtitle,
            ha="center", va="center", color=theme.ink_500, fontsize=8.5,
            style="italic")

    fig.savefig(out_path, facecolor=theme.bg_panel, dpi=dpi)
    plt.close(fig)
    return out_path


@dataclass
class AppendixNotes:
    """Accumulator for coverage / limitation notes shown in the appendix."""

    notes: List[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    def extend(self, messages: List[str]) -> None:
        for m in messages:
            self.add(m)

    def as_list(self) -> List[str]:
        return list(self.notes)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.notes)
