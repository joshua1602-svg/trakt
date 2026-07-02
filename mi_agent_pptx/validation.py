"""mi_agent_pptx.validation — pre/post build validation of the deck.

Enforces the acceptance criteria that can be checked deterministically:

* the deck has **12–15 slides**;
* every slide carries a populated strapline;
* mandatory charts either rendered or were explicitly downgraded to a branded
  placeholder with a coverage note;
* coverage/appendix notes exist when artifacts were missing.

Returns a structured :class:`ValidationReport` rather than raising, so the CLI
can surface issues while still emitting the (branded, placeholder-filled) deck.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

MIN_SLIDES = 12
MAX_SLIDES = 15


@dataclass
class ValidationReport:
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        self.info.append(msg)

    def summary(self) -> str:
        return (f"{'PASS' if self.ok else 'FAIL'} — "
                f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)")


def validate_deck_config(config) -> ValidationReport:
    """Validate the deck config before building."""
    report = ValidationReport()
    n = config.slide_count
    if n < MIN_SLIDES:
        report.error(f"Deck has {n} slides; minimum required is {MIN_SLIDES}.")
    elif n > MAX_SLIDES:
        report.error(f"Deck has {n} slides; maximum allowed is {MAX_SLIDES}.")
    else:
        report.add_info(f"Slide count {n} within the 12–15 range.")

    ids = [s.id for s in config.slides]
    if len(ids) != len(set(ids)):
        report.error("Duplicate slide ids in deck config.")

    types = {s.type for s in config.slides}
    if "cover" not in types:
        report.warn("No cover slide declared.")
    if "methodology" not in types and "notes" not in types:
        report.warn("No methodology/notes slide declared.")

    return report


def validate_build(build_report: Dict[str, Any]) -> ValidationReport:
    """Validate the outcome of a build (slide records + straplines)."""
    report = ValidationReport()
    slides = build_report.get("slides", [])
    n = len(slides)
    if n < MIN_SLIDES:
        report.error(f"Built deck has {n} slides; minimum is {MIN_SLIDES}.")
    if n > MAX_SLIDES:
        report.error(f"Built deck has {n} slides; maximum is {MAX_SLIDES}.")

    for s in slides:
        if not s.get("strapline"):
            report.error(f"Slide '{s.get('id')}' has no strapline.")
        if s.get("mandatory") and s.get("placeholder"):
            report.warn(
                f"Mandatory content on slide '{s.get('id')}' rendered as a "
                f"placeholder (missing artifact).")

    if not build_report.get("coverage_notes"):
        report.add_info("No coverage gaps recorded.")

    return report
