"""apps.blob_trigger_app.file_roles — logical source-file role classification.

Monthly reporting packs for the SAME source portfolio arrive with cosmetically
different file names period to period, e.g.::

    LoanExtract One - OMNI_test.csv        vs  LoanExtract One OMNI_test.csv
    PropertyExtract - Omni_test.csv        vs  PG_PropertyExtract Internal OMNI_test.csv
    Funder Principal And Interest_test.csv

These are NOT schema drift — they are the same logical roles (loan extract,
property/collateral extract, funder P&I extract) with filename/layout aliasing.
Fingerprinting on exact file names therefore mis-routes an equivalent pack as a
new/changed source and forces a needless re-approval every period.

This module maps a file name to a stable *logical role* deterministically, so a
pack fingerprint can be keyed on ``{role: columns}`` rather than
``{filename: columns}``. Classification is:

    1. operator-approved registry aliases (``SourceRecord.file_role_aliases``,
       ``role -> [name patterns]``) — promoted with the mapping, so once approved
       future equivalent packs run deterministically; then
    2. built-in default role keywords (loan / property / funder P&I / cashflow /
       collateral); then
    3. a stable fallback (the digit-stripped normalised name) so a genuinely new
       file still produces a deterministic key and a real new schema still flips
       the fingerprint.

Pure string logic — no I/O, no registry/Azure dependency — so it can run inside
``fingerprint_pack`` before any registry lookup.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Built-in logical roles + the keyword groups that identify them. A group matches
# when ALL of its keywords appear in the normalised name (as a substring of the
# spaced or the compacted form); a role matches if ANY of its groups match.
# Ordered: more specific / compound roles first so e.g. a "loan collateral tape"
# is not swallowed by a bare "collateral" rule.
DEFAULT_ROLE_RULES: List[Tuple[str, List[List[str]]]] = [
    ("funder_pi_extract", [["funder", "principal", "interest"],
                           ["principal", "and", "interest"],
                           ["funder", "p&i"], ["funder", "pi"]]),
    ("property_extract",  [["propertyextract"], ["property", "extract"],
                           ["valuation", "extract"], ["collateral", "extract"]]),
    ("loan_extract",      [["loanextract"], ["loan", "extract"],
                           ["loan", "tape"], ["loanbook"], ["loan", "report"]]),
    ("cashflow_extract",  [["cashflow"], ["cash", "flow"]]),
    ("collateral_extract", [["collateral"]]),
]

# Tokens that carry no logical meaning and would otherwise destabilise the role
# (period stamps handled separately as pure-digit tokens).
_NOISE_TOKENS = {"test", "final", "copy", "v1", "v2", "draft", "internal", "omni"}


def _normalise(name: str) -> Tuple[str, str]:
    """Return ``(spaced, compact)`` normalised forms of a file name.

    Lower-cased, extension dropped, separators collapsed to single spaces, and
    pure-digit period tokens (``2025``, ``11``, ``30``, ``202511``) + known noise
    tokens removed. ``spaced`` keeps word boundaries; ``compact`` removes them so
    ``loan extract`` and ``loanextract`` both match a keyword.
    """
    stem = Path(str(name)).stem.lower()
    tokens = [t for t in re.split(r"[^a-z0-9&]+", stem) if t]
    kept = [t for t in tokens if not t.isdigit() and t not in _NOISE_TOKENS]
    spaced = " ".join(kept)
    compact = "".join(kept)
    return spaced, compact


def _group_matches(group: List[str], spaced: str, compact: str) -> bool:
    return all((kw in spaced) or (kw in compact) for kw in group)


def classify_file(name: str,
                  aliases: Optional[Dict[str, List[str]]] = None) -> str:
    """Classify a file name to a logical role.

    ``aliases`` (``role -> [name patterns]``) are operator-approved overrides and
    win over the built-in rules. Falls back to the digit-stripped normalised name
    so unknown files stay deterministic across periods.
    """
    spaced, compact = _normalise(name)

    # 1) operator-approved registry aliases win.
    for role, patterns in (aliases or {}).items():
        for pat in (patterns or []):
            p_spaced, p_compact = _normalise(pat)
            if not p_compact:
                continue
            if (p_spaced and p_spaced in spaced) or (p_compact in compact):
                return role

    # 2) built-in default role keywords.
    for role, groups in DEFAULT_ROLE_RULES:
        if any(_group_matches(g, spaced, compact) for g in groups):
            return role

    # 3) stable fallback — the normalised name (never the raw filename, so period
    # stamps do not destabilise it). A genuinely different file still flips the key.
    return compact or "unknown"


def pack_role_map(paths, aliases: Optional[Dict[str, List[str]]] = None
                  ) -> Dict[str, str]:
    """Map each pack file path to its logical role key (diagnostic view).

    When two files resolve to the SAME role, they are disambiguated
    deterministically by role + zero-based index (``role``, ``role#1`` …) in
    filename order, so no file is silently merged away.
    """
    by_role: Dict[str, List[str]] = {}
    for path in sorted(str(p) for p in paths):
        role = classify_file(Path(path).name, aliases)
        by_role.setdefault(role, []).append(path)
    out: Dict[str, str] = {}
    for role, members in by_role.items():
        for i, path in enumerate(members):
            out[path] = role if i == 0 else f"{role}#{i}"
    return out
