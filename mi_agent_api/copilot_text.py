"""mi_agent_api/copilot_text.py — channel-safe text normalisation for Copilot.

Microsoft 365 Copilot renders the adapter's strings in a surface whose font /
sanitiser does not cover the full Unicode range the MI Agent emits. Characters
such as the em dash, the arrow used in provenance notes ("parser → executor"),
the Greek deltas and sigmas used in metric labels, and any stray control /
zero-width / private-use codepoint come back as black squares or replacement
glyphs after values and sentences.

This module is a PRESENTATION-ONLY normalisation applied by the Copilot adapter
and by nothing else. It is deliberately NOT applied to the React envelope, to the
shared MI service, or to any stored artifact:

  * it never touches numbers — only the characters used to decorate them;
  * it never re-parses, re-formats or re-computes a value;
  * it never removes information — every replacement is an ASCII equivalent of
    the same meaning ("→" becomes "->", "≥" becomes ">=").

Normalisation form
------------------
NFKC (compatibility composition) is applied AFTER the explicit replacement map,
so the characters we care about are mapped deterministically and NFKC only folds
the compatibility residue (fullwidth forms, ligatures, non-standard spaces).

Preserved by contract: ``£`` and other currency signs, ``%``, decimal and
thousands punctuation, ISO and long-form dates, ordinary Markdown bullets and
emphasis, and newlines / tabs.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

#: Decorative / symbolic characters replaced with a plain equivalent of the SAME
#: meaning. Applied before NFKC so the mapping is explicit and testable.
SYMBOL_REPLACEMENTS: Dict[str, str] = {
    # dashes and minus signs
    "—": "-",      # — em dash
    "–": "-",      # – en dash
    "−": "-",      # − minus sign
    "‐": "-",      # ‐ hyphen
    "‑": "-",      # ‑ non-breaking hyphen
    "⁃": "-",      # ⁃ hyphen bullet
    # separators
    "·": "-",      # · middle dot (used as a label separator)
    "•": "-",      # • bullet -> Markdown-safe list marker
    "…": "...",    # … ellipsis
    # arrows / relations used in provenance and trace notes
    "→": "->",     # →
    "←": "<-",     # ←
    "⇒": "=>",     # ⇒
    "≤": "<=",     # ≤
    "≥": ">=",     # ≥
    "≈": "~",      # ≈
    "≠": "!=",     # ≠
    "∈": " in ",   # ∈
    "⊆": " subset of ",  # ⊆
    # arithmetic
    "×": "x",      # ×
    "÷": "/",      # ÷
    "±": "+/-",    # ±
    # Greek letters used as metric decorations
    "Δ": "Delta",  # Δ
    "Σ": "Sum",    # Σ
    "μ": "mu",     # μ
    # status marks
    "✓": "[ok]",   # ✓
    "✔": "[ok]",   # ✔
    "✗": "[x]",    # ✗
    "✘": "[x]",    # ✘
    "⚠": "[!]",    # ⚠
    # smart quotes
    "“": '"', "”": '"', "„": '"',
    "‘": "'", "’": "'", "‚": "'",
    "′": "'", "″": '"',
    # explicit replacement character (already-broken upstream text)
    "�": "",
}

#: Whitespace-like codepoints folded to an ordinary space (a non-breaking space
#: between a value and its unit is a frequent black-square source).
_SPACE_LIKE = "          " \
              "    　"

#: Zero-width / invisible formatting codepoints — removed outright.
_ZERO_WIDTH = "​‌‍⁠﻿­᠎"

#: Malformed internal citation remnants some upstream renderers leave behind:
#: 【4:0†source】-style brackets, 【…】 fragments, and [[cite:…]] / [cite:…] markers.
_CITATION_PATTERNS = (
    re.compile(r"【[^】]{0,80}】"),        # 【…】
    re.compile(r"【[^】]{0,80}$"),             # unterminated 【…
    re.compile(r"\[\[\s*cite[^\]]{0,120}\]\]", re.I),  # [[cite:…]]
    re.compile(r"\[\s*cite\s*:[^\]]{0,120}\]", re.I),  # [cite:…]
    re.compile(r"†source", re.I),                 # †source remnant
)

#: Private-use and unassigned-plane ranges (Copilot renders these as squares).
_PRIVATE_USE = re.compile(
    "[-\U000f0000-\U000ffffd\U00100000-\U0010fffd]")
#: Variation selectors (including emoji presentation VS16).
_VARIATION_SELECTORS = re.compile("[︀-️\U000e0100-\U000e01ef]")
#: C0/C1 control characters except the whitespace we permit (\t, \n, \r).
_CONTROLS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def normalise_text(value: str) -> str:
    """Return *value* rendered safely in the Copilot surface.

    Idempotent: normalising already-normalised text is a no-op. Never raises —
    a non-string is returned unchanged by :func:`normalise_payload`.
    """
    if not value:
        return value
    text = value
    # 1. Strip malformed citation remnants before anything else re-shapes them.
    for pattern in _CITATION_PATTERNS:
        text = pattern.sub("", text)
    # 2. Explicit symbol map (meaning-preserving ASCII equivalents).
    for src, dst in SYMBOL_REPLACEMENTS.items():
        if src in text:
            text = text.replace(src, dst)
    # 3. Documented Unicode normalisation form: NFKC.
    text = unicodedata.normalize("NFKC", text)
    # 4. Remove invisible / unrenderable codepoints.
    text = _VARIATION_SELECTORS.sub("", text)
    text = _PRIVATE_USE.sub("", text)
    text = text.translate({ord(c): None for c in _ZERO_WIDTH})
    text = _CONTROLS.sub("", text)
    # 5. Fold exotic spaces to an ordinary space (NFKC leaves NBSP alone).
    text = text.translate({ord(c): " " for c in _SPACE_LIKE})
    # 6. Tidy the whitespace the replacements may have doubled up, per line so
    #    Markdown structure (blank lines, list items) is preserved.
    lines = [re.sub(r"[ \t]{2,}", " ", ln).rstrip() for ln in text.split("\n")]
    return "\n".join(lines).strip("\n") if len(lines) > 1 else lines[0].strip()


def normalise_payload(value: Any) -> Any:
    """Recursively normalise every string in a JSON-shaped payload.

    Dict KEYS are left untouched (they are contract identifiers, not prose), as
    are all non-string scalars — numeric values are never modified.
    """
    if isinstance(value, str):
        return normalise_text(value)
    if isinstance(value, dict):
        return {k: normalise_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        out: List[Any] = [normalise_payload(v) for v in value]
        return out
    return value
