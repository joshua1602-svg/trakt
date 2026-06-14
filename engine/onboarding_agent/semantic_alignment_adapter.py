"""
semantic_alignment_adapter.py
=============================

PART 3/4 — a clean, deterministic adapter that exposes the **existing Gate 1
semantic alignment engine** (``engine.gate_1_alignment.semantic_alignment``) to
the Onboarding Agent as a first-class mapping stage.

Why this exists
---------------
The Onboarding Agent already reuses Gate 1's :class:`HeaderMapper` inside
``mapping_proposer``. This adapter makes that reuse *explicit and auditable*: it
runs the full deterministic tier chain in one place and returns onboarding
:class:`MappingCandidate` objects, so the parity audit and the mapping trace can
both point at exactly the same code path.

Deterministic tier chain (from ``HeaderMapper.map_one``):
    1. exact            (case-insensitive canonical name)
    2. normalized       (token-normalised canonical name)
    3. alias            (alias libraries, aliases_*.yaml)
    4. token_set        (token-set Jaccard >= 0.85)        ── semantic tier
    5. fuzz_token_set   (RapidFuzz token_set_ratio >= 88)  ── semantic tier
    6. fuzz_ratio_norm  (RapidFuzz ratio on normalised >= 92) ── semantic tier

The last three tiers are the *semantic* tiers — fuzzy, similarity-based matches
that go beyond an exact/alias hit. ``semantic_alignment_used`` flags when one of
them decided the mapping.

This is **deterministic**. It is NOT the LLM. It writes no files; it only
returns candidate data to the Onboarding Agent / parity tool. It is also
**field-scope-aware**: a target excluded by the mode's field scope is reported as
out of scope (so the caller can divert it) and never silently promoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.gate_1_alignment.semantic_alignment import (
    HeaderMapper,
    load_aliases_from_dir,
    load_field_registry,
    normalise_name,
    select_registry_fields,
)
from .file_profiler import redact_value
from .onboarding_models import MappingCandidate

# The fuzzy/similarity tiers that count as "semantic alignment" (beyond a
# deterministic exact/normalized/alias hit).
SEMANTIC_TIERS = ("token_set", "fuzz_token_set", "fuzz_ratio_norm")
# Confidence at/above which a mapping is auto-accepted (mirrors mapping_proposer).
REVIEW_THRESHOLD = 0.92


@dataclass
class SemanticAlignmentResult:
    """One header's result from the deterministic semantic alignment engine."""

    source_column: str = ""
    normalized_column: str = ""
    candidate: str = ""
    method: str = "unmapped"
    confidence: float = 0.0
    semantic_alignment_used: bool = False
    field_scope_status: str = "in_scope"   # in_scope | out_of_scope | unmapped
    requires_review: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_column": self.source_column,
            "normalized_column": self.normalized_column,
            "candidate": self.candidate,
            "method": self.method,
            "confidence": round(self.confidence, 4),
            "semantic_alignment_used": self.semantic_alignment_used,
            "field_scope_status": self.field_scope_status,
            "requires_review": self.requires_review,
        }


def is_semantic_tier(method: str) -> bool:
    """True when ``method`` is one of the fuzzy semantic alignment tiers."""
    return method in SEMANTIC_TIERS


def align_header(
    mapper: HeaderMapper, header: str, field_scope: Any = None
) -> SemanticAlignmentResult:
    """Run the full deterministic tier chain for one header.

    Returns a :class:`SemanticAlignmentResult` carrying the selected candidate,
    the tier/method that decided it, whether a *semantic* (fuzzy) tier was used,
    and the field-scope status (so an out-of-scope target is never silently
    promoted to an active mapping).
    """
    canon, method, conf = mapper.map_one(header)
    res = SemanticAlignmentResult(
        source_column=str(header),
        normalized_column=normalise_name(str(header)),
        candidate=canon or "",
        method=method,
        confidence=float(conf or 0.0),
        semantic_alignment_used=bool(canon) and is_semantic_tier(method),
    )
    if not canon:
        res.field_scope_status = "unmapped"
        res.requires_review = True
        return res

    # Field-scope safety (mode-aware): excluded targets are reported, not applied.
    if field_scope is not None and getattr(field_scope, "is_excluded", None) \
            and field_scope.is_excluded(canon):
        res.field_scope_status = "out_of_scope"
    res.requires_review = conf < REVIEW_THRESHOLD
    return res


def build_header_mapper(
    registry: dict | str | Path,
    aliases_dir: str | Path,
    portfolio_type: str = "equity_release",
) -> Tuple[HeaderMapper, List[str]]:
    """Build a :class:`HeaderMapper` from the registry + alias libraries.

    Returns ``(mapper, canonical_fields)``. This is the single place the
    Onboarding Agent and the parity audit construct the Gate 1 mapper, so both
    are provably running the same engine.
    """
    if not isinstance(registry, dict):
        registry = load_field_registry(Path(registry))
    canonical_fields = select_registry_fields(registry, portfolio_type)
    alias_map = load_aliases_from_dir(Path(aliases_dir))
    return HeaderMapper(canonical_fields, alias_map), canonical_fields


def run_semantic_alignment_for_headers(
    headers: List[str],
    registry: dict | str | Path,
    aliases_dir: str | Path,
    portfolio_type: str = "equity_release",
    mode: str = "",
    field_scope: Any = None,
    series_by_header: Optional[Dict[str, Any]] = None,
    source_file: str = "",
    classification: str = "",
) -> List[MappingCandidate]:
    """Map a list of source headers via the deterministic semantic alignment engine.

    Returns onboarding :class:`MappingCandidate` objects (the Onboarding Agent's
    own model) so results can flow straight into the existing pipeline / trace.
    No files are written. Out-of-scope targets are flagged (``requires_review``
    with an explicit reason) so the caller can route them to the out-of-scope
    table rather than promoting them.
    """
    mapper, _ = build_header_mapper(registry, aliases_dir, portfolio_type)
    series_by_header = series_by_header or {}
    out: List[MappingCandidate] = []
    for header in headers:
        res = align_header(mapper, header, field_scope=field_scope)
        samples: List[str] = []
        series = series_by_header.get(header)
        if series is not None:
            try:
                samples = [redact_value(v) for v in
                           series.dropna().drop_duplicates().head(5).tolist()]
            except Exception:
                samples = []
        cand = MappingCandidate(
            source_file=source_file,
            source_file_classification=classification,
            source_column=str(header),
            candidate_canonical_field=res.candidate,
            confidence=res.confidence,
            method=res.method if res.candidate else "unmapped",
            sample_values_redacted=samples,
            requires_review=res.requires_review or res.field_scope_status != "in_scope",
        )
        if res.field_scope_status == "out_of_scope":
            cand.reason = (
                f"Semantic alignment target '{res.candidate}' is out of scope for "
                f"mode '{mode}'."
            )
        elif res.semantic_alignment_used:
            cand.reason = (
                f"Mapped by Gate 1 semantic alignment ({res.method}, "
                f"confidence {res.confidence:.0%})."
            )
        elif not res.candidate:
            cand.reason = "No deterministic/semantic match; needs alias or review."
        out.append(cand)
    return out
