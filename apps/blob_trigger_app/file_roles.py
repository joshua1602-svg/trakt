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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Minimum header-overlap (Jaccard) for an incoming file's columns to be accepted
#: as a match for an approved role's column signature. Below this, header evidence
#: is treated as absent (→ filename fallback, and drift when approved schemas exist).
HEADER_MATCH_THRESHOLD = 0.6

# Confidence attached to non-header bases (header uses the real Jaccard score).
_ALIAS_CONFIDENCE = 0.5
_KEYWORD_CONFIDENCE = 0.4

# role_basis vocabulary (recorded in per-file diagnostics).
BASIS_HEADER = "header_signature"
BASIS_ALIAS = "registry_alias"
BASIS_KEYWORD = "filename_keyword"
BASIS_FALLBACK = "fallback_unknown"

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


# --------------------------------------------------------------------------- #
# Header-first classification (production rule)
# --------------------------------------------------------------------------- #

def normalise_column(col: str) -> str:
    """Normalise a single header for signature comparison (case/separator/space
    insensitive): lower-cased, non-alphanumerics removed."""
    return re.sub(r"[^a-z0-9]+", "", str(col).lower())


def normalise_columns(cols: Sequence[str]) -> List[str]:
    return [normalise_column(c) for c in cols]


def header_match_score(columns: Sequence[str],
                       signature: Sequence[str]) -> Tuple[float, int, int]:
    """Score an incoming header set against an approved role signature.

    Returns ``(confidence, matched_count, unmatched_count)`` where confidence is
    the Jaccard overlap of the normalised column SETS (order-independent — a real
    reorder is caught by the order-sensitive fingerprint, not by role matching)
    and ``unmatched_count`` is the size of the symmetric difference (headers
    present on only one side).
    """
    inc = set(normalise_columns(columns))
    sig = set(normalise_columns(signature))
    if not inc or not sig:
        return 0.0, 0, len(inc ^ sig)
    matched = inc & sig
    union = inc | sig
    return len(matched) / len(union), len(matched), len(union) - len(matched)


@dataclass
class FileRoleAssignment:
    filename: str
    columns: List[str]
    assigned_role: str
    role_basis: str
    confidence: float
    matched_columns_count: int
    unmatched_columns_count: int

    def diagnostic(self) -> Dict[str, object]:
        return {
            "filename": self.filename,
            "assigned_role": self.assigned_role,
            "role_basis": self.role_basis,
            "confidence": round(self.confidence, 4),
            "matched_columns_count": self.matched_columns_count,
            "unmatched_columns_count": self.unmatched_columns_count,
        }


@dataclass
class PackClassification:
    assignments: List[FileRoleAssignment] = field(default_factory=list)
    ambiguous_role_conflict: bool = False
    conflicting_roles: List[str] = field(default_factory=list)
    drift_suspected: bool = False
    drift_files: List[str] = field(default_factory=list)

    def diagnostics(self) -> List[Dict[str, object]]:
        return [a.diagnostic() for a in self.assignments]

    def role_columns(self) -> Dict[str, List[str]]:
        """``{role_key: columns}`` for fingerprinting. Colliding roles are kept
        (disambiguated by index) so nothing is merged away; the caller fails
        closed on ``ambiguous_role_conflict`` before trusting the fingerprint."""
        by_role: Dict[str, List[List[str]]] = {}
        for a in self.assignments:
            by_role.setdefault(a.assigned_role, []).append(a.columns)
        out: Dict[str, List[str]] = {}
        for role, collists in by_role.items():
            for i, cols in enumerate(sorted(collists, key=lambda c: tuple(c))):
                out[role if i == 0 else f"{role}#{i}"] = cols
        return out


def _best_header_role(columns: Sequence[str],
                      role_schemas: Dict[str, Sequence[str]],
                      threshold: float) -> Tuple[Optional[str], float, int, int]:
    scored = []
    for role, sig in role_schemas.items():
        conf, matched, unmatched = header_match_score(columns, sig)
        scored.append((role, conf, matched, unmatched))
    if not scored:
        return None, 0.0, 0, len(list(columns))
    scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
    role, conf, matched, unmatched = scored[0]
    if conf >= threshold:
        return role, conf, matched, unmatched
    # No signature clears the bar — report the closest for diagnostics.
    return None, conf, matched, unmatched


def classify_pack(files: Sequence[Tuple[str, Sequence[str]]], *,
                  role_schemas: Optional[Dict[str, Sequence[str]]] = None,
                  aliases: Optional[Dict[str, List[str]]] = None,
                  threshold: float = HEADER_MATCH_THRESHOLD) -> PackClassification:
    """Classify each pack file to a logical role, HEADER-FIRST.

    Priority per file:
      1. **header signature** — normalised headers vs approved role schemas
         (``role_schemas``); a match ≥ ``threshold`` assigns the role regardless
         of filename;
      2. **registry alias** — approved filename patterns (``aliases``);
      3. **filename keyword** — built-in role keywords;
      4. **fallback** — the stable normalised name.

    When approved ``role_schemas`` exist, a file that does NOT clear the header
    threshold is flagged ``drift_suspected`` even if a filename hint assigns a
    role — filename evidence never silently promotes a header-mismatched file to a
    known source. Two files resolving to the same role set ``ambiguous_role_conflict``.
    """
    have_schemas = bool(role_schemas)
    result = PackClassification()

    for name, columns in files:
        cols = list(columns)
        role = basis = None
        conf = 0.0
        matched, unmatched = 0, len(cols)

        # 1) header signature.
        if have_schemas:
            role, conf, matched, unmatched = _best_header_role(cols, role_schemas, threshold)
            if role is not None:
                basis = BASIS_HEADER

        # 2) registry filename aliases (fallback hint).
        if role is None and aliases:
            r = _match_from_default_or_aliases(name, aliases, use_defaults=False)
            if r:
                role, basis, conf = r, BASIS_ALIAS, _ALIAS_CONFIDENCE

        # 3) filename keyword (fallback hint).
        if role is None:
            r = _match_from_default_or_aliases(name, None, use_defaults=True)
            if r:
                role, basis, conf = r, BASIS_KEYWORD, _KEYWORD_CONFIDENCE

        # 4) stable unknown fallback.
        if role is None:
            role, basis, conf = (_normalise(name)[1] or "unknown"), BASIS_FALLBACK, 0.0

        # Approved header schemas exist but this file was placed by filename/fallback
        # evidence → not a clean known-source match; flag as drift, don't silently pass.
        if have_schemas and basis != BASIS_HEADER:
            result.drift_suspected = True
            result.drift_files.append(name)

        result.assignments.append(FileRoleAssignment(
            filename=name, columns=cols, assigned_role=role, role_basis=basis,
            confidence=conf, matched_columns_count=matched,
            unmatched_columns_count=unmatched))

    # Cross-file role collision → fail closed as ambiguous.
    seen: Dict[str, int] = {}
    for a in result.assignments:
        seen[a.assigned_role] = seen.get(a.assigned_role, 0) + 1
    conflicts = sorted(r for r, n in seen.items() if n > 1)
    if conflicts:
        result.ambiguous_role_conflict = True
        result.conflicting_roles = conflicts
    return result


def _match_from_default_or_aliases(name: str,
                                   aliases: Optional[Dict[str, List[str]]],
                                   *, use_defaults: bool) -> Optional[str]:
    """Return a role from filename evidence only: approved aliases (use_defaults
    False) or built-in keyword rules (use_defaults True). ``None`` when no rule
    fires (so the caller can fall through to the next tier)."""
    spaced, compact = _normalise(name)
    if aliases:
        for role, patterns in aliases.items():
            for pat in (patterns or []):
                p_spaced, p_compact = _normalise(pat)
                if not p_compact:
                    continue
                if (p_spaced and p_spaced in spaced) or (p_compact in compact):
                    return role
    if use_defaults:
        for role, groups in DEFAULT_ROLE_RULES:
            if any(_group_matches(g, spaced, compact) for g in groups):
                return role
    return None
