"""
config_suggester.py
==================

PART 7 — infer candidate config values from the onboarding evidence.

Sources of evidence:
  * file inventory + classifications
  * column profiles (detected dates, currencies)
  * warehouse agreement text (facility terms) — light regex extraction on
    TXT/MD only; PDF/DOCX become ``requires_document_extraction`` gaps
  * securitisation summary text

Every suggestion carries provenance (source file / column / document reference)
and a ``review_status`` of suggested / requires_review / missing. Nothing is
written to production config — these are proposals.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from .document_extractor import extract_documents
from .onboarding_models import (
    ColumnProfile,
    ConfigSuggestion,
    DocumentExtraction,
    FileInventoryItem,
)

# Reuse the asset-class signal vocabulary from the v1 config bootstrap agent.
try:
    from agents.config_bootstrap_agent import _ASSET_CLASS_SIGNALS
except Exception:  # pragma: no cover - defensive fallback
    _ASSET_CLASS_SIGNALS = {
        "equity_release": ["equity_release", "no_negative_equity", "lifetime_mortgage"],
    }

import yaml

# Onboarding geography/regime policy file used to source the geographic
# classification year (NOT the reporting date).
_ONBOARDING_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "system" / "onboarding_agent.yaml"
# Final fallback if no policy is configured anywhere (still requires_review).
_FALLBACK_CLASSIFICATION_YEAR = "2021"


def _suggest_classification_year() -> "ConfigSuggestion":
    """Source the geographic classification year from policy config.

    classification_year is the year/version of the geographic (NUTS/ITL)
    classification used for geography fields (ESMA RREL12 /
    geographic_region_classification). It is deliberately NOT derived from the
    reporting date. Resolution order: onboarding geography policy config, then a
    documented fallback default — always flagged requires_review for human sign-off.
    """
    value = ""
    source = ""
    try:
        if _ONBOARDING_POLICY_PATH.exists():
            policy = yaml.safe_load(_ONBOARDING_POLICY_PATH.read_text(encoding="utf-8")) or {}
            geo = policy.get("geography_policy", {}) or {}
            if geo.get("classification_year") is not None:
                value = str(geo["classification_year"])
                source = "config/system/onboarding_agent.yaml:geography_policy.classification_year"
    except Exception:
        pass

    if not value:
        value = _FALLBACK_CLASSIFICATION_YEAR
        source = "onboarding/regime policy default"

    return ConfigSuggestion(
        field="classification_year",
        suggested_value=value,
        confidence=0.6,
        source_file="<policy>",
        source_column_or_document_reference=source,
        evidence=(
            "Geographic (NUTS/ITL) classification year from onboarding/regime "
            "policy; NOT derived from the reporting date."
        ),
        review_status="requires_review",
    )


_CURRENCY_RE = re.compile(r"\b(GBP|EUR|USD|CHF|JPY|AUD)\b")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_MONEY_RE = re.compile(r"(GBP|EUR|USD|£|€|\$)\s*([\d,]+(?:\.\d+)?)")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _warehouse_files(inventory: List[FileInventoryItem]) -> List[FileInventoryItem]:
    return [i for i in inventory if i.classification == "warehouse_agreement"]


def _securitisation_files(inventory: List[FileInventoryItem]) -> List[FileInventoryItem]:
    return [i for i in inventory if i.classification == "securitisation_document"]


def _detect_asset_class(profiles: List[ColumnProfile], texts: str) -> Optional[ConfigSuggestion]:
    haystack = " ".join(p.normalized_column_name for p in profiles) + " " + texts.lower()
    scores: Dict[str, int] = {}
    for ac, signals in _ASSET_CLASS_SIGNALS.items():
        hits = sum(1 for s in signals if s.replace(" ", "").replace("_", "") in haystack.replace(" ", "").replace("_", ""))
        if hits:
            scores[ac] = hits
    if not scores:
        return None
    best = max(scores, key=lambda k: scores[k])
    conf = min(0.5 + 0.15 * scores[best], 0.95)
    return ConfigSuggestion(
        field="asset_class",
        suggested_value=best,
        confidence=round(conf, 2),
        source_file="<multiple>",
        source_column_or_document_reference="column names + document text",
        evidence=f"Matched {scores[best]} '{best}' signal token(s).",
        review_status="suggested" if conf >= 0.7 else "requires_review",
    )


def _detect_currency(profiles: List[ColumnProfile], texts: str) -> Optional[ConfigSuggestion]:
    counter: Counter = Counter()
    src = ""
    for p in profiles:
        for v in p.sample_values_redacted:
            m = _CURRENCY_RE.search(str(v))
            if m:
                counter[m.group(1)] += 1
                src = src or f"{p.file_name}:{p.source_column}"
    for m in _CURRENCY_RE.findall(texts):
        counter[m] += 1
    if not counter:
        return None
    cur, _ = counter.most_common(1)[0]
    return ConfigSuggestion(
        field="currency",
        suggested_value=cur,
        confidence=0.8,
        source_file=src or "<document>",
        source_column_or_document_reference=src or "warehouse/securitisation text",
        evidence=f"Currency token '{cur}' observed in samples / documents.",
        review_status="suggested",
    )


def _detect_reporting_dates(profiles: List[ColumnProfile]) -> List[str]:
    dates = set()
    for p in profiles:
        if p.likely_reporting_date:
            if p.date_max:
                dates.add(p.date_max)
            if p.date_min:
                dates.add(p.date_min)
    return sorted(dates)


def suggest_config(
    client_name: str,
    input_dir: Path,
    inventory: List[FileInventoryItem],
    profiles: List[ColumnProfile],
    document_extractions: Optional[List["DocumentExtraction"]] = None,
) -> List[ConfigSuggestion]:
    """Produce candidate config suggestions from all available evidence.

    Warehouse / securitisation facts come from ``document_extractions`` (the
    policy-bound document extractor); if not supplied they are computed here.
    """
    suggestions: List[ConfigSuggestion] = []

    # client_name straight from the CLI.
    suggestions.append(
        ConfigSuggestion(
            field="client_name",
            suggested_value=client_name,
            confidence=1.0,
            source_file="<cli>",
            source_column_or_document_reference="--client-name",
            evidence="Provided on the command line.",
            review_status="suggested",
        )
    )

    # Gather document text (warehouse + securitisation) for inference.
    doc_text = ""
    for item in inventory:
        if item.classification in ("warehouse_agreement", "securitisation_document"):
            if item.file_type in ("txt", "md"):
                doc_text += "\n" + _read_text(Path(item.file_path))

    # asset_class + portfolio_type.
    ac = _detect_asset_class(profiles, doc_text)
    if ac:
        suggestions.append(ac)
        suggestions.append(
            ConfigSuggestion(
                field="portfolio_type",
                suggested_value=ac.suggested_value,
                confidence=ac.confidence,
                source_file=ac.source_file,
                source_column_or_document_reference=ac.source_column_or_document_reference,
                evidence="Derived from asset_class.",
                review_status=ac.review_status,
            )
        )

    # currency.
    cur = _detect_currency(profiles, doc_text)
    if cur:
        suggestions.append(cur)

    # jurisdiction.
    if "GBP" in doc_text or any(
        kw in doc_text.lower() for kw in ("england", "scotland", "wales", "united kingdom")
    ):
        suggestions.append(
            ConfigSuggestion(
                field="jurisdiction",
                suggested_value="GB",
                confidence=0.7,
                source_file="<documents>",
                source_column_or_document_reference="warehouse/securitisation text",
                evidence="UK currency / nations referenced in documents.",
                review_status="suggested",
            )
        )

    # reporting_date / data_cut_off_date.
    dates = _detect_reporting_dates(profiles)
    if dates:
        suggestions.append(
            ConfigSuggestion(
                field="reporting_date",
                suggested_value=dates[-1],
                confidence=0.6 if len(dates) > 1 else 0.85,
                source_file="<multiple>" if len(dates) > 1 else "<structured>",
                source_column_or_document_reference="reporting-date columns",
                evidence=f"Detected reporting date(s): {', '.join(dates)}.",
                review_status="requires_review" if len(dates) > 1 else "suggested",
            )
        )
        suggestions.append(
            ConfigSuggestion(
                field="data_cut_off_date",
                suggested_value=dates[-1],
                confidence=0.5,
                source_file="<structured>",
                source_column_or_document_reference="reporting-date columns",
                evidence="Defaulted to the latest detected reporting date.",
                review_status="requires_review",
            )
        )

    # classification_year is the YEAR/VERSION of the geographic (NUTS/ITL)
    # classification used for geography fields (ESMA RREL12 /
    # geographic_region_classification). It is NOT the reporting date and must
    # NEVER be derived from year(reporting_date). It is sourced from geography /
    # regime policy config (e.g. nuts_classification_year), or marked
    # requires_review when no policy is available.
    suggestions.append(_suggest_classification_year())

    # regime — driven by asset class (equity release -> ESMA Annex 2).
    regime = "ESMA_Annex2"
    suggestions.append(
        ConfigSuggestion(
            field="regime",
            suggested_value=regime,
            confidence=0.7 if ac else 0.4,
            source_file="<inferred>",
            source_column_or_document_reference="asset class + securitisation summary",
            evidence="Equity release pools disclose under ESMA Annex 2.",
            review_status="suggested" if ac else "requires_review",
        )
    )

    # geography_policy — static reference to existing policy (do not change it).
    suggestions.append(
        ConfigSuggestion(
            field="geography_policy",
            suggested_value="ESMA=GBZZZ; MI/FCA=ITL3",
            confidence=0.9,
            source_file="<policy>",
            source_column_or_document_reference="existing geography projection policy",
            evidence="ESMA RREL11/RREC6 use GBZZZ; ITL3 retained for MI/FCA display.",
            review_status="suggested",
        )
    )

    # Warehouse / securitisation config from document extractions (PART 6).
    # Parsing now lives in document_extractor under the minimisation policy; we
    # only fold the extracted, capped facts into config suggestions here.
    if document_extractions is None:
        document_extractions = extract_documents(inventory)

    extracted_fields = set()
    for ex in document_extractions:
        extracted_fields.add(ex.field)
        suggestions.append(
            ConfigSuggestion(
                field=ex.field,
                suggested_value=ex.value,
                confidence=ex.confidence,
                source_file=ex.source_document,
                source_column_or_document_reference=ex.source_reference,
                evidence=ex.retained_evidence,
                review_status=ex.status,
            )
        )

    # If no warehouse agreement was present at all, record the gap explicitly.
    if not _warehouse_files(inventory) and "warehouse_facility_present" not in extracted_fields:
        suggestions.append(
            ConfigSuggestion(
                field="warehouse_facility_present",
                suggested_value="unknown",
                confidence=0.0,
                source_file="",
                source_column_or_document_reference="",
                evidence="No warehouse funding agreement found in data room.",
                review_status="missing",
            )
        )

    return suggestions
