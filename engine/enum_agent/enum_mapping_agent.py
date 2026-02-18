from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_NORMALIZE_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
_ID_RE = re.compile(r"\b\d{6,}\b")
_POSTCODE_RE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.IGNORECASE)


@dataclass
class EnumSuggestion:
    field_name: str
    raw_value: str
    suggested_value: Optional[str]
    confidence: float
    reasoning: str
    alternative_value: Optional[str]
    allowed_values: List[str]
    count: int
    deterministic_method: str = "unmapped"
    deterministic_confidence: float = 0.0
    status: str = "pending"
    confirmed_value: Optional[str] = None
    reviewer_note: str = ""
    namespace: str = "global"
    regime: str = ""
    sent_to_llm: bool = False

    def __post_init__(self) -> None:
        allowed = set(self.allowed_values)
        if self.suggested_value not in allowed:
            self.suggested_value = None
        if self.alternative_value not in allowed:
            self.alternative_value = None
        if self.confirmed_value not in allowed:
            self.confirmed_value = None

    def confirm(self, note: str = "") -> None:
        if self.suggested_value is None:
            raise ValueError("Cannot confirm without a valid suggested_value")
        self.status = "confirmed"
        self.confirmed_value = self.suggested_value
        self.reviewer_note = note

    def remap(self, value: str, note: str = "") -> None:
        if value not in set(self.allowed_values):
            raise ValueError(f"Remap target '{value}' not in allowed_values")
        self.status = "remapped"
        self.confirmed_value = value
        self.reviewer_note = note

    def reject(self, note: str = "") -> None:
        self.status = "rejected"
        self.confirmed_value = None
        self.reviewer_note = note

    def skip(self, note: str = "") -> None:
        self.status = "skipped"
        self.reviewer_note = note

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_token(value: Any, strip_punctuation: bool = True) -> str:
    txt = "" if value is None else str(value)
    txt = txt.strip().lower()
    txt = re.sub(r"\s+", " ", txt)
    if strip_punctuation:
        txt = _NORMALIZE_RE.sub("", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _hash_allowed_values(allowed_values: List[str]) -> str:
    payload = json.dumps(sorted(allowed_values), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _redact_sample(value: str, max_len: int = 120) -> str:
    redacted = _EMAIL_RE.sub("[EMAIL]", value)
    redacted = _PHONE_RE.sub("[PHONE]", redacted)
    redacted = _POSTCODE_RE.sub("[POSTCODE]", redacted)
    redacted = _ID_RE.sub("[ID]", redacted)
    return redacted[:max_len]


class EnumResolutionEngine:
    def __init__(
        self,
        fuzzy_threshold: float = 88.0,
        review_threshold: float = 0.92,
        ambiguity_delta: float = 3.0,
        keep_unresolved_original: bool = True,
    ) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.review_threshold = review_threshold
        self.ambiguity_delta = ambiguity_delta
        self.keep_unresolved_original = keep_unresolved_original

    @staticmethod
    def _fuzzy_top2(query: str, allowed_normalized: List[str]) -> List[Tuple[str, float]]:
        try:
            from rapidfuzz import fuzz, process  # type: ignore

            results = process.extract(query, allowed_normalized, scorer=fuzz.token_set_ratio, limit=2)
            return [(name, float(score)) for name, score, _ in results]
        except Exception:
            import difflib

            ranked = sorted(
                ((name, difflib.SequenceMatcher(None, query, name).ratio() * 100.0) for name in allowed_normalized),
                key=lambda x: x[1],
                reverse=True,
            )
            return ranked[:2]

    def resolve(
        self,
        field_name: str,
        series: pd.Series,
        allowed_values: List[str],
        synonyms_map: Dict[str, str],
        namespace: str = "global",
        regime: str = "",
        rule_hook: Optional[Callable[[str, str, List[str]], Optional[Tuple[str, float, str]]]] = None,
    ) -> Tuple[pd.Series, List[EnumSuggestion], List[EnumSuggestion]]:
        counts = series.fillna("<NULL>").astype(str).value_counts(dropna=False).to_dict()
        allowed_norm = {normalize_token(v): v for v in allowed_values}
        mapped_lookup: Dict[str, Optional[str]] = {}
        report: List[EnumSuggestion] = []
        candidates: List[EnumSuggestion] = []

        for raw_value, count in counts.items():
            if raw_value == "<NULL>":
                mapped_lookup[raw_value] = None
                continue

            normalized = normalize_token(raw_value)
            suggested_value: Optional[str] = None
            method = "unmapped"
            confidence = 0.0
            reasoning = "No deterministic mapping"
            alternative = None
            ambiguous = False

            if normalized in allowed_norm:
                suggested_value = allowed_norm[normalized]
                method = "exact"
                confidence = 1.0
                reasoning = "Tier1 exact normalized match"
            elif normalized in synonyms_map:
                suggested_value = synonyms_map[normalized]
                method = "synonym"
                confidence = 0.98
                reasoning = "Tier2 synonym match"
            else:
                fuzzy = self._fuzzy_top2(normalized, list(allowed_norm.keys()))
                if fuzzy:
                    top_norm, top_score = fuzzy[0]
                    second_score = fuzzy[1][1] if len(fuzzy) > 1 else -1.0
                    if top_score >= self.fuzzy_threshold:
                        suggested_value = allowed_norm[top_norm]
                        method = "fuzzy"
                        confidence = top_score / 100.0
                        reasoning = f"Tier3 fuzzy match ({top_score:.1f})"
                        if second_score >= self.fuzzy_threshold and abs(top_score - second_score) <= self.ambiguity_delta:
                            ambiguous = True
                            confidence = min(confidence, self.review_threshold - 0.01)
                            alternative = allowed_norm[fuzzy[1][0]]
                            reasoning = (
                                f"Tier3 ambiguous fuzzy: top={top_score:.1f}, second={second_score:.1f}; requires review"
                            )

            if rule_hook and suggested_value is None:
                hook_result = rule_hook(field_name, str(raw_value), allowed_values)
                if hook_result:
                    hook_value, hook_conf, hook_reason = hook_result
                    if hook_value in set(allowed_values):
                        suggested_value = hook_value
                        method = "rule"
                        confidence = float(hook_conf)
                        reasoning = hook_reason

            suggestion = EnumSuggestion(
                field_name=field_name,
                raw_value=str(raw_value),
                suggested_value=suggested_value,
                confidence=max(0.0, min(float(confidence), 1.0)),
                reasoning=reasoning,
                alternative_value=alternative,
                allowed_values=allowed_values,
                count=int(count),
                deterministic_method=method,
                deterministic_confidence=max(0.0, min(float(confidence), 1.0)),
                namespace=namespace,
                regime=regime,
            )

            mapped_lookup[str(raw_value)] = suggestion.suggested_value
            report.append(suggestion)

            if (
                suggestion.deterministic_method == "unmapped"
                or suggestion.deterministic_confidence < self.review_threshold
                or ambiguous
            ):
                candidates.append(suggestion)

        def _map_cell(v: Any) -> Any:
            if pd.isna(v):
                return v
            raw = str(v)
            mapped = mapped_lookup.get(raw)
            if mapped is None:
                return raw if self.keep_unresolved_original else None
            return mapped

        return series.map(_map_cell), report, candidates


class LLMEnumMapper:
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        batch_size: int = 20,
        api_key: Optional[str] = None,
    ):
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic  # type: ignore

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def suggest(
        self,
        candidates: List[EnumSuggestion],
        allow_samples: bool = False,
        sample_values_by_raw: Optional[Dict[str, List[str]]] = None,
    ) -> List[EnumSuggestion]:
        if not candidates:
            return candidates
        client = self._get_client()

        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i : i + self.batch_size]
            user_payload = []
            for c in batch:
                item: Dict[str, Any] = {
                    "field_name": c.field_name,
                    "raw_value": c.raw_value,
                    "count": c.count,
                    "allowed_values": c.allowed_values,
                    "deterministic_hint": {
                        "suggested": c.suggested_value,
                        "confidence": c.deterministic_confidence,
                    },
                }
                if allow_samples and sample_values_by_raw:
                    samples = sample_values_by_raw.get(c.raw_value, [])
                    item["sample_values"] = [_redact_sample(str(s)) for s in samples[:5]]
                user_payload.append(item)

            response = client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                system=self._system_prompt(),
                messages=[{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
            )

            parsed = self._extract_json(response.content[0].text)
            if not isinstance(parsed, list):
                parsed = []
            while len(parsed) < len(batch):
                parsed.append({})

            for cand, item in zip(batch, parsed):
                cand.sent_to_llm = True
                suggested = item.get("suggested_value")
                alternative = item.get("alternative_value")
                if suggested not in set(cand.allowed_values):
                    suggested = None
                if alternative not in set(cand.allowed_values):
                    alternative = None
                cand.suggested_value = suggested
                cand.alternative_value = alternative
                cand.confidence = max(0.0, min(float(item.get("confidence", 0.0)), 1.0)) if suggested else 0.0
                cand.reasoning = str(item.get("reasoning", ""))

        return candidates

    @staticmethod
    def _extract_json(text: str) -> Any:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        start = min([p for p in [cleaned.find("["), cleaned.find("{")] if p >= 0], default=-1)
        end = max(cleaned.rfind("]"), cleaned.rfind("}"))
        if start >= 0 and end >= start:
            cleaned = cleaned[start : end + 1]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM enum response as JSON; falling back to empty list")
            return []

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are an enum mapping assistant. Return STRICT JSON only. "
            "For each token choose suggested_value ONLY from allowed_values, otherwise null. "
            "Never invent new enum values. Handle multilingual tokens (e.g. Spanish) semantically. "
            "Map semantically to the target enum set; do not translate freely beyond allowed_values. "
            "Return an array aligned to input order with keys: suggested_value, confidence, reasoning, alternative_value."
        )


class EnumAliasLearner:
    def __init__(self, output_path: Path) -> None:
        self.output_path = Path(output_path)

    def persist_confirmed(self, suggestions: List[EnumSuggestion]) -> int:
        confirmed = [s for s in suggestions if s.status in {"confirmed", "remapped"} and s.confirmed_value]
        if not confirmed:
            return 0

        data = self._load_yaml(self.output_path)
        added = 0
        for s in confirmed:
            ns = data.setdefault(s.namespace or "global", {})
            rg = ns.setdefault(s.regime or "global", {})
            field = rg.setdefault(s.field_name, {})
            key = normalize_token(s.raw_value)
            if field.get(key) != s.confirmed_value:
                field[key] = s.confirmed_value
                added += 1

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True), encoding="utf-8")
        return added

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return loaded if isinstance(loaded, dict) else {}


def load_enum_synonyms(
    base_path: Path,
    confirmed_path: Path,
    field_name: str,
    namespace: str,
    regime: str,
) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    base_data = yaml.safe_load(Path(base_path).read_text(encoding="utf-8")) or {}
    field_node = base_data.get(field_name, {}) if isinstance(base_data, dict) else {}

    for branch in ("manual", "learned"):
        values = field_node.get(branch, {}) if isinstance(field_node, dict) else {}
        for raw, mapped in (values or {}).items():
            merged[normalize_token(raw)] = mapped

    confirmed = yaml.safe_load(Path(confirmed_path).read_text(encoding="utf-8")) if Path(confirmed_path).exists() else {}
    if not isinstance(confirmed, dict):
        confirmed = {}

    for ns in [namespace, "global"]:
        ns_node = confirmed.get(ns, {}) if isinstance(confirmed, dict) else {}
        for rg in [regime, "global"]:
            mappings = (((ns_node or {}).get(rg) or {}).get(field_name) or {})
            for raw, mapped in mappings.items():
                merged[normalize_token(raw)] = mapped

    return merged


def resolve_enums_for_field(
    field_name: str,
    series: pd.Series,
    allowed_values: List[str],
    namespace: str,
    regime: str,
    rule_hook: Optional[Callable[[str, str, List[str]], Optional[Tuple[str, float, str]]]] = None,
    fuzzy_threshold: float = 88.0,
    review_threshold: float = 0.92,
    synonyms_base_path: Path = Path("config/system/enum_synonyms.yaml"),
    synonyms_confirmed_path: Path = Path("config/system/enum_synonyms_confirmed.yaml"),
) -> Tuple[pd.Series, List[EnumSuggestion], List[EnumSuggestion], Dict[str, Any]]:
    synonyms = load_enum_synonyms(synonyms_base_path, synonyms_confirmed_path, field_name, namespace, regime)
    engine = EnumResolutionEngine(fuzzy_threshold=fuzzy_threshold, review_threshold=review_threshold)
    mapped, report, candidates = engine.resolve(
        field_name=field_name,
        series=series,
        allowed_values=allowed_values,
        synonyms_map=synonyms,
        namespace=namespace,
        regime=regime,
        rule_hook=rule_hook,
    )

    meta = {
        "field_name": field_name,
        "allowed_values_hash": _hash_allowed_values(allowed_values),
        "counts": {
            "distinct_tokens": len(report),
            "candidates_for_llm": len(candidates),
            "resolved": len([r for r in report if r.suggested_value is not None]),
        },
    }
    return mapped, report, candidates, meta
