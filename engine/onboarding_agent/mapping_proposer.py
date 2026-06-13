"""
mapping_proposer.py
===================

PART 6 — field mapping candidates.

Reuses the Gate 1 alignment engine (:class:`HeaderMapper`) for header
normalisation, alias matching and confidence scoring. For each source column
across all structured files we propose a candidate canonical field.

File classification is used to disambiguate context-dependent headers (e.g.
``Balance`` means a cashflow amount in a cashflow report but
``current_principal_balance`` in a loan report) and to respect the geography
model (readable region labels must not be mapped to the classification-year
field).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from engine.gate_1_alignment.semantic_alignment import (
    HeaderMapper,
    load_aliases_from_dir,
    load_field_registry,
    select_registry_fields,
    tokenize,
)
from .ambiguity_rule import (
    ambiguity_record,
    classify_candidate,
    load_ambiguity_delta,
    load_min_candidate_confidence,
    resolve_regulatory_preference,
)
from .file_profiler import redact_value
from .onboarding_models import FileInventoryItem, MappingAmbiguity, MappingCandidate

# Confidence below which a mapping is flagged for human review.
REVIEW_THRESHOLD = 0.92

# Classification-aware hints: (classification, header substring) -> canonical field.
# Applied as a high-confidence override before the generic header mapper.
_CONTEXT_HINTS = {
    ("cashflow_report", "balance"): "current_principal_balance",
    ("cashflow_report", "principal_outstanding"): "current_principal_balance",
    ("loan_report", "balance"): "current_principal_balance",
    ("collateral_report", "valuation"): "current_valuation_amount",
    ("collateral_report", "post_code"): "property_post_code",
    ("collateral_report", "postcode"): "property_post_code",
}

# Geography guard — readable region labels must stay as the MI/display field and
# must never be proposed for the ESMA classification-year field.
_READABLE_REGION_HEADERS = {"collateral_region", "obligor_region", "region", "collateral_geography"}
_REGION_CLASSIFICATION_FIELD = "geographic_region_classification"
_REGION_DISPLAY_FIELD = "collateral_geography"


def _classification_to_portfolio_type(classification: str) -> str:
    # Onboarding v2 targets UK equity release pools; the registry superset for
    # equity_release also includes all `common` fields, which is what we want.
    return "equity_release"


class MappingProposer:
    """Wraps the Gate 1 HeaderMapper with onboarding context awareness."""

    def __init__(self, registry_path: Path, aliases_dir: Path, portfolio_type: str = "equity_release"):
        registry = load_field_registry(Path(registry_path))
        self.canonical_fields = select_registry_fields(registry, portfolio_type)
        self.alias_map = load_aliases_from_dir(Path(aliases_dir))
        self.mapper = HeaderMapper(self.canonical_fields, self.alias_map)
        self._canonical_set = set(self.canonical_fields)

    def ranked_candidates(
        self, source_column: str, top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """Return up to ``top_k`` scored canonical candidates for a header.

        Used by the regulatory-preference ambiguity rule to detect when two
        plausible targets (one regulatory, one not) have close confidence. An
        exact / normalized / alias hit is decisive (single candidate); otherwise
        we rank by token-set Jaccard so close competitors surface. This does NOT
        modify the Gate 1 HeaderMapper — it reuses its public token sets only.
        """
        canon, method, conf = self.mapper.map_one(source_column)
        if canon and method in ("exact", "normalized", "alias"):
            return [(canon, method, float(conf))]

        tokens = set(tokenize(str(source_column)))
        scored: List[Tuple[str, str, float]] = []
        if tokens:
            for c, c_tokens in self.mapper.token_sets.items():
                if not c_tokens:
                    continue
                union = tokens | c_tokens
                if not union:
                    continue
                jaccard = len(tokens & c_tokens) / len(union)
                if jaccard > 0:
                    scored.append((c, "token_set", round(jaccard, 4)))
        scored.sort(key=lambda x: x[2], reverse=True)
        # Ensure the deterministic best (which may come from a fuzz tier) is present.
        if canon and not any(c == canon for c, _, _ in scored):
            scored.insert(0, (canon, method, float(conf)))
        return scored[:top_k]

    def _context_override(self, classification: str, source_column: str):
        col_norm = str(source_column).lower().replace(" ", "_")
        # loan_report covers current/historical
        cls = "loan_report" if classification.endswith("loan_report") else classification
        is_date_col = "date" in col_norm
        for (hint_cls, hint_sub), canonical in _CONTEXT_HINTS.items():
            if hint_cls != cls or hint_sub not in col_norm or canonical not in self._canonical_set:
                continue
            # Amount/balance hints must not fire on date columns
            # (e.g. valuation_date must not become current_valuation_amount).
            if is_date_col and canonical.endswith(("amount", "balance")):
                continue
            return canonical
        return None

    def propose_for_column(
        self,
        source_file: str,
        classification: str,
        source_column: str,
        series: Optional[pd.Series] = None,
    ) -> MappingCandidate:
        col_norm = str(source_column).lower().replace(" ", "_")
        samples: List[str] = []
        if series is not None:
            samples = [redact_value(v) for v in series.dropna().drop_duplicates().head(5).tolist()]

        cand = MappingCandidate(
            source_file=source_file,
            source_file_classification=classification,
            source_column=str(source_column),
            sample_values_redacted=samples,
        )

        # Geography guard: readable region labels map to the display field only.
        if col_norm in _READABLE_REGION_HEADERS:
            field = _REGION_DISPLAY_FIELD if _REGION_DISPLAY_FIELD in self._canonical_set else ""
            cand.candidate_canonical_field = field
            cand.confidence = 0.85 if field else 0.0
            cand.method = "geography_display_guard"
            cand.requires_review = True
            cand.reason = (
                "Readable region label kept as display geography; not mapped to "
                f"{_REGION_CLASSIFICATION_FIELD} (classification year)."
            )
            return cand

        # Generic Gate 1 mapping first — an exact/alias hit on a real canonical
        # field name (e.g. original_principal_balance, valuation_date) must win
        # over the broad context hints below.
        canon, method, conf = self.mapper.map_one(source_column)
        if canon and conf >= REVIEW_THRESHOLD:
            cand.candidate_canonical_field = canon
            cand.method = method
            cand.confidence = float(conf)
            cand.requires_review = False
            cand.reason = ""
            return cand

        # Context-aware override for ambiguous headers the generic mapper could
        # not confidently resolve (e.g. bare "balance", "principal_outstanding").
        override = self._context_override(classification, source_column)
        if override:
            cand.candidate_canonical_field = override
            cand.confidence = 0.9
            cand.method = "context_hint"
            cand.requires_review = True
            cand.reason = f"Resolved using {classification} context for ambiguous header."
            return cand

        # Fall back to whatever the generic mapper produced (possibly unmapped).
        cand.candidate_canonical_field = canon or ""
        cand.method = method
        cand.confidence = float(conf)
        cand.requires_review = True
        if not canon:
            cand.reason = "No deterministic canonical match; needs manual mapping or alias."
        else:
            cand.reason = f"Below auto-accept threshold ({REVIEW_THRESHOLD})."
        return cand


def _apply_ambiguity_rule(
    proposer: "MappingProposer",
    cand: MappingCandidate,
    source_column: str,
    field_scope,
    mode_name: str,
    delta_threshold: float,
    regulatory_reporting_enabled: bool,
    min_candidate_confidence: float,
) -> Tuple[MappingCandidate, Optional[MappingAmbiguity], Optional[Dict[str, str]]]:
    """Run the regulatory-preference ambiguity rule on one column.

    Returns ``(candidate, ambiguity_or_None, diverted_oos_or_None)``. The
    candidate is mutated to record the selected target plus alternatives. The
    ambiguity record (when present) is written to 05b; the diverted record (only
    in mi_only when a regulatory non-core candidate must be excluded) is written
    to 05a.
    """
    if field_scope is None:
        return cand, None, None

    ranked = proposer.ranked_candidates(source_column)
    if len(ranked) < 2:
        return cand, None, None

    candidates = [
        classify_candidate(f, score, field_scope, method=method)
        for (f, method, score) in ranked
    ]
    resolution = resolve_regulatory_preference(
        candidates,
        mode=mode_name,
        delta_threshold=delta_threshold,
        regulatory_reporting_enabled=regulatory_reporting_enabled,
        min_candidate_confidence=min_candidate_confidence,
    )
    if resolution is None or resolution.selected is None:
        return cand, None, None

    sel = resolution.selected
    alt = resolution.alternative

    # Record the alternative evidence on the candidate regardless of selection.
    cand.candidate_canonical_field = sel.field
    cand.confidence = float(sel.confidence)
    cand.method = sel.method or cand.method
    cand.requires_review = True
    cand.ambiguity_rule_applied = resolution.rule_applied
    cand.reason = (
        f"Regulatory-preference ambiguity rule applied ({resolution.reason}); "
        f"selected '{sel.field}' over '{alt.field if alt else ''}'."
    )
    cand.alternative_candidates = [
        {
            "field": c.field,
            "category": c.category,
            "core_canonical": c.core_canonical,
            "confidence": round(c.confidence, 4),
        }
        for c in candidates
        if c.field and c.field != sel.field
    ][:4]

    ambiguity = MappingAmbiguity(
        **ambiguity_record(
            resolution,
            source_file=cand.source_file,
            source_column=source_column,
            mode=mode_name,
        )
    )

    diverted = None
    if resolution.divert_regulatory_to_out_of_scope and resolution.diverted_field:
        d = resolution.diverted_field
        diverted = {
            "source_file": cand.source_file,
            "source_column": source_column,
            "candidate_field": d.field,
            "category": d.category or "regulatory",
            "reason": (
                "regulatory non-core candidate not selected because this is "
                "MI-only onboarding (regulatory-preference ambiguity rule)"
            ),
            "mode": mode_name,
        }
    return cand, ambiguity, diverted


def propose_mappings(
    inventory: List[FileInventoryItem],
    dataframes: Dict[str, pd.DataFrame],
    registry_path: Path,
    aliases_dir: Path,
    field_scope=None,
    regulatory_reporting_enabled: bool = False,
    ambiguity_delta_threshold: Optional[float] = None,
    min_candidate_confidence: Optional[float] = None,
) -> Tuple[List[MappingCandidate], List[Dict[str, str]], List[MappingAmbiguity]]:
    """Propose canonical mappings for every structured column.

    Returns ``(mapping_candidates, out_of_scope, mapping_ambiguities)``. When
    ``field_scope`` is provided, any column whose proposed canonical target is
    excluded by the mode (e.g. a regulatory-only field in mi_only) is diverted to
    ``out_of_scope`` instead of the mapping candidates, so the review pack can
    show it was deliberately excluded rather than missed.

    The regulatory-preference ambiguity rule (PART 1) is applied per column: when
    a regulatory and a non-regulatory candidate have close confidence, the mode
    decides which is preferred, the decision is always flagged for review, and
    the evidence is recorded in ``mapping_ambiguities``.
    """
    proposer = MappingProposer(registry_path, aliases_dir)
    out: List[MappingCandidate] = []
    out_of_scope: List[Dict[str, str]] = []
    ambiguities: List[MappingAmbiguity] = []
    mode_name = getattr(field_scope, "mode_name", "") if field_scope else ""
    delta = (
        ambiguity_delta_threshold
        if ambiguity_delta_threshold is not None
        else load_ambiguity_delta()
    )
    min_conf = (
        min_candidate_confidence
        if min_candidate_confidence is not None
        else load_min_candidate_confidence()
    )
    for item in inventory:
        df = dataframes.get(item.file_path)
        if df is None:
            continue
        for col in df.columns:
            cand = proposer.propose_for_column(
                item.file_name, item.classification, str(col), df[col]
            )
            # Regulatory-preference ambiguity rule (mode-aware).
            cand, ambiguity, diverted = _apply_ambiguity_rule(
                proposer, cand, str(col), field_scope, mode_name, delta,
                regulatory_reporting_enabled, min_conf,
            )
            if ambiguity is not None:
                ambiguities.append(ambiguity)
            if diverted is not None:
                out_of_scope.append(diverted)

            target = cand.candidate_canonical_field
            if field_scope is not None and target and field_scope.is_excluded(target):
                out_of_scope.append({
                    "source_file": item.file_name,
                    "source_column": str(col),
                    "candidate_field": target,
                    "category": field_scope.category_of(target) or "regulatory",
                    "reason": field_scope.out_of_scope_reason_by_field.get(
                        target, "excluded by mode field scope"
                    ),
                    "mode": mode_name,
                })
                continue
            out.append(cand)
    return out, out_of_scope, ambiguities
