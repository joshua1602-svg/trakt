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
from typing import Dict, List, Optional

import pandas as pd

from engine.gate_1_alignment.semantic_alignment import (
    HeaderMapper,
    load_aliases_from_dir,
    load_field_registry,
    select_registry_fields,
)
from .file_profiler import redact_value
from .onboarding_models import FileInventoryItem, MappingCandidate

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


def propose_mappings(
    inventory: List[FileInventoryItem],
    dataframes: Dict[str, pd.DataFrame],
    registry_path: Path,
    aliases_dir: Path,
) -> List[MappingCandidate]:
    """Propose canonical mappings for every column of every structured file."""
    proposer = MappingProposer(registry_path, aliases_dir)
    out: List[MappingCandidate] = []
    for item in inventory:
        df = dataframes.get(item.file_path)
        if df is None:
            continue
        for col in df.columns:
            out.append(
                proposer.propose_for_column(
                    item.file_name, item.classification, str(col), df[col]
                )
            )
    return out
