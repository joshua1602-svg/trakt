"""mi_agent_pptx.strat_order — the dashboard's bar ordering, for the deck.

A line-for-line port of ``frontend/mi-agent-ui/src/lib/stratOrder.ts``
(``cleanBucketLabel`` / ``bucketBound`` / ``sortStratBars``), so a bar list
reads in the same order on a slide as on the screen it mirrors.

Stratifications arrive from the engine ranked by balance. For an ordinal
dimension — LTV band, age band, rate band, vintage year, ticket size — that
ranking is unreadable: the deck drew LTV as 20-30%, 40-50%, 50-60%, 70-80%,
80-90%, 60-70%, 30-40%, while the dashboard drew 20-30%, 30-40%, 40-50% from
the same payload. Buckets are therefore sorted by the numeric bound parsed from
their label when the dimension is ordinal, alphabetically otherwise, with
"Unknown"-style buckets last. Values are never altered — order and label
tidy-ups only.

Keep this file and ``stratOrder.ts`` in step; ``tests/test_deck_dashboard_design.py``
holds both to the same cases.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

#: Buckets that mean "no data" — always sorted last, never parsed.
_UNKNOWN_RE = re.compile(r"^(unknown|not supplied|not available|none|n/?a|other|-)$", re.I)
_YEAR_FLOAT_RE = re.compile(r"^(\d{1,4})\.0$")
_NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")


def clean_bucket_label(label: Any) -> str:
    """Tidy a bucket label for display: ``"2008.0"`` → ``"2008"``."""
    s = str(label if label is not None else "").strip()
    m = _YEAR_FLOAT_RE.match(s)
    return m.group(1) if m else s


def bucket_bound(label: Any) -> Optional[float]:
    """The ordinal sort bound parsed from a bucket label, or ``None``.

    ``"40-50%"`` → 40 · ``"<20%"`` → 19.5 · ``"85+"`` → 85 · ``"2008.0"`` →
    2008 · ``"£100K-£150K"`` → 100.
    """
    s = clean_bucket_label(label)
    if not s or _UNKNOWN_RE.match(s):
        return None
    below = bool(re.match(r"^\s*[<≤]", s))
    m = _NUMBER_RE.search(re.sub(r"[£$€,\s]", "", s))
    if not m:
        return None
    n = float(m.group(0))
    return n - 0.5 if below else n


def order_bars(bars: Sequence[Dict[str, Any]], *, label_key: str = "label"
               ) -> List[Dict[str, Any]]:
    """Order bars for display exactly as the dashboard does.

    Ordinal when at least half the real (non-unknown) labels parse to a bound;
    then ascending by bound, otherwise alphabetical. Unknown-style buckets sink
    to the end either way. Labels are tidied; the input is not mutated.
    """
    entries = []
    for bar in bars or ():
        label = clean_bucket_label(bar.get(label_key))
        unknown = (not label) or bool(_UNKNOWN_RE.match(label))
        entries.append({"bar": {**bar, label_key: label}, "label": label,
                        "unknown": unknown,
                        "bound": None if unknown else bucket_bound(label)})
    real = [e for e in entries if not e["unknown"]]
    numeric = sum(1 for e in real if e["bound"] is not None)
    ordinal = bool(real) and numeric * 2 >= len(real)

    def key(e):
        bound = (e["bound"] if e["bound"] is not None else float("inf")) if ordinal else 0.0
        return (e["unknown"], bound, _natural(e["label"]))

    return [e["bar"] for e in sorted(entries, key=key)]


def _natural(label: str):
    """``localeCompare(..., {numeric: true, sensitivity: "base"})``, roughly:
    case-insensitive, with runs of digits compared as numbers."""
    return [(0, float(part)) if part.isdigit() else (1, part.lower())
            for part in re.split(r"(\d+)", label) if part]
