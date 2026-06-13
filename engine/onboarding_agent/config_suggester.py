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

from .onboarding_models import ColumnProfile, ConfigSuggestion, FileInventoryItem

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


def _extract_warehouse_terms(text: str, ref: str) -> List[ConfigSuggestion]:
    out: List[ConfigSuggestion] = []
    low = text.lower()

    out.append(
        ConfigSuggestion(
            field="warehouse_facility_present",
            suggested_value="true",
            confidence=0.95,
            source_file=ref,
            source_column_or_document_reference=ref,
            evidence="Warehouse funding agreement present in data room.",
            review_status="suggested",
        )
    )

    # Lender name — line containing 'warehouse lender' or 'lender:'.
    m = re.search(r"warehouse lender[:\s\*]*([A-Za-z0-9 ,&.'-]+)", text, re.IGNORECASE)
    if m:
        out.append(
            ConfigSuggestion(
                field="warehouse_lender_name",
                suggested_value=m.group(1).strip().rstrip("|").strip(),
                confidence=0.7,
                source_file=ref,
                source_column_or_document_reference=ref,
                evidence="Parsed 'Warehouse Lender' from agreement text.",
                review_status="requires_review",
            )
        )

    # Advance rate.
    m = re.search(r"advance rate[^\d]*(\d+(?:\.\d+)?)\s*%", low)
    if m:
        out.append(
            ConfigSuggestion(
                field="advance_rate",
                suggested_value=f"{m.group(1)}%",
                confidence=0.7,
                source_file=ref,
                source_column_or_document_reference=ref,
                evidence="Parsed 'advance rate' from agreement text.",
                review_status="requires_review",
            )
        )

    # Margin.
    m = re.search(r"margin[^\d]*(\d+(?:\.\d+)?)\s*%", low)
    if m:
        out.append(
            ConfigSuggestion(
                field="margin",
                suggested_value=f"{m.group(1)}%",
                confidence=0.6,
                source_file=ref,
                source_column_or_document_reference=ref,
                evidence="Parsed 'margin' from agreement text.",
                review_status="requires_review",
            )
        )

    # Interest index.
    idx = None
    for cand in ("SONIA", "EURIBOR", "SOFR", "LIBOR", "BASE RATE"):
        if cand.lower() in low:
            idx = cand
            break
    if idx:
        out.append(
            ConfigSuggestion(
                field="interest_index",
                suggested_value=idx,
                confidence=0.7,
                source_file=ref,
                source_column_or_document_reference=ref,
                evidence=f"Interest index '{idx}' referenced in agreement.",
                review_status="requires_review",
            )
        )

    # Warehouse limit.
    m = _MONEY_RE.search(text)
    if m and ("limit" in low or "facility" in low):
        out.append(
            ConfigSuggestion(
                field="warehouse_limit",
                suggested_value=f"{m.group(1)} {m.group(2)}".strip(),
                confidence=0.6,
                source_file=ref,
                source_column_or_document_reference=ref,
                evidence="Parsed a money amount near 'limit'/'facility'.",
                review_status="requires_review",
            )
        )

    return out


def suggest_config(
    client_name: str,
    input_dir: Path,
    inventory: List[FileInventoryItem],
    profiles: List[ColumnProfile],
) -> List[ConfigSuggestion]:
    """Produce candidate config suggestions from all available evidence."""
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

    # warehouse terms.
    wh = _warehouse_files(inventory)
    if wh:
        for item in wh:
            ref = item.file_name
            if item.file_type in ("txt", "md"):
                suggestions.extend(_extract_warehouse_terms(_read_text(Path(item.file_path)), ref))
            else:
                suggestions.append(
                    ConfigSuggestion(
                        field="warehouse_facility_present",
                        suggested_value="true",
                        confidence=0.6,
                        source_file=ref,
                        source_column_or_document_reference=ref,
                        evidence="Warehouse agreement present but is a non-text document.",
                        review_status="requires_review",
                    )
                )
    else:
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

    # securitisation presence + target pool balance.
    sec = _securitisation_files(inventory)
    if sec:
        suggestions.append(
            ConfigSuggestion(
                field="securitisation_present",
                suggested_value="true",
                confidence=0.7,
                source_file=sec[0].file_name,
                source_column_or_document_reference=sec[0].file_name,
                evidence="Securitisation summary present in data room.",
                review_status="suggested",
            )
        )
        sec_text = "".join(
            _read_text(Path(i.file_path)) for i in sec if i.file_type in ("txt", "md")
        )
        m = re.search(r"target pool balance[:\s\*]*([A-Z]{3}\s*[\d,]+)", sec_text, re.IGNORECASE)
        if m:
            suggestions.append(
                ConfigSuggestion(
                    field="target_pool_balance",
                    suggested_value=m.group(1).strip(),
                    confidence=0.6,
                    source_file=sec[0].file_name,
                    source_column_or_document_reference=sec[0].file_name,
                    evidence="Parsed 'target pool balance' from securitisation summary.",
                    review_status="requires_review",
                )
            )

    return suggestions
