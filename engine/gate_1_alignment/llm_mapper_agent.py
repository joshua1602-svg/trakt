#!/usr/bin/env python3
"""
llm_mapper_agent.py

Tier 7: LLM Agent for field mapping.

This module is a SUPPLEMENT to the frozen deterministic spine (Tiers 1-6 in
semantic_alignment.py).  It ONLY runs on:
  - headers still marked method="unmapped" after the deterministic pass, OR
  - headers with deterministic confidence < the configured review_threshold

Human confirmation is MANDATORY before any suggestion is applied.
Confirmed mappings are persisted to aliases_llm_confirmed.yaml so future runs
resolve at Tier 3 (alias) with zero LLM involvement.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import textwrap
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------


@dataclass
class LLMSuggestion:
    """Carries LLM suggestion + human review outcome for one raw header."""

    raw_header: str
    suggested_field: Optional[str]
    confidence: float
    reasoning: str
    alternative_field: Optional[str]
    semantic_category: str
    sample_values: List[str]

    # Populated from deterministic pass before LLM is called
    deterministic_method: str = "unmapped"
    deterministic_confidence: float = 0.0

    # Populated after human review
    status: str = "pending"          # pending | confirmed | rejected | remapped | skipped
    confirmed_field: Optional[str] = None
    reviewer_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# REGISTRY HELPERS
# ---------------------------------------------------------------------------


def _load_registry(registry_path: Path) -> dict:
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if "fields" not in data or not isinstance(data["fields"], dict):
        raise ValueError(f"Registry missing 'fields' mapping: {registry_path}")
    return data


def _build_catalogue_subset(
    registry: dict,
    portfolio_type: str,
    max_fields: int = 80,
) -> List[dict]:
    """
    Return a trimmed list of canonical field dicts for inclusion in the prompt.

    Priority order (to fill the 80-field budget):
      1. common portfolio_type fields
      2. fields matching the requested portfolio_type
      3. any remaining fields (breadth cover)
    """
    fields = registry.get("fields", {}) or {}
    pt = (portfolio_type or "").strip().lower()

    common: List[Tuple[str, dict]] = []
    specific: List[Tuple[str, dict]] = []
    other: List[Tuple[str, dict]] = []

    for fname, meta in fields.items():
        m = meta or {}
        fpt = str(m.get("portfolio_type", "")).strip().lower()
        entry = {
            "name": fname,
            "category": m.get("category", ""),
            "format": m.get("format", ""),
            "allowed_values": m.get("allowed_values"),
            "layer": m.get("layer", ""),
        }
        if fpt == "common":
            common.append((fname, entry))
        elif fpt == pt:
            specific.append((fname, entry))
        else:
            other.append((fname, entry))

    ordered = common + specific + other
    subset = [e for _, e in ordered[:max_fields]]
    return subset


# ---------------------------------------------------------------------------
# LLM FIELD MAPPER
# ---------------------------------------------------------------------------


class LLMFieldMapper:
    """
    Calls Claude to suggest canonical field names for unmapped/low-confidence
    headers.  Never mutates the canonical dataset directly.
    """

    MODEL = "claude-sonnet-4-20250514"
    TEMPERATURE = 0.0
    MAX_TOKENS = 4096
    BATCH_SIZE = 10

    def __init__(
        self,
        registry_path: Path,
        portfolio_type: str,
        aliases_dir: Path,
        api_key: Optional[str] = None,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.portfolio_type = portfolio_type
        self.aliases_dir = Path(aliases_dir)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        self.registry = _load_registry(self.registry_path)
        self.canonical_set: set = set(self.registry["fields"].keys())
        self.catalogue_subset = _build_catalogue_subset(self.registry, portfolio_type)

        self._system_prompt: Optional[str] = None
        self._client = None  # lazy-init to avoid import error if anthropic not installed

        logger.info(
            "LLMFieldMapper ready — registry=%s  portfolio=%s  catalogue_fields=%d",
            self.registry_path.name,
            self.portfolio_type,
            len(self.catalogue_subset),
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def suggest_mappings(
        self,
        unmapped_headers: List[str],
        df_raw: pd.DataFrame,
        deterministic_report: Optional[List[dict]] = None,
    ) -> List[LLMSuggestion]:
        """
        For each header in *unmapped_headers*, build a feature envelope and call
        the LLM in batches of BATCH_SIZE.  Returns one LLMSuggestion per header.
        """
        if not unmapped_headers:
            return []

        # Build feature envelopes for every header
        envelopes = [
            self._build_envelope(h, df_raw)
            for h in unmapped_headers
        ]

        # Batch and call Claude
        suggestions: List[LLMSuggestion] = []
        for batch_start in range(0, len(envelopes), self.BATCH_SIZE):
            batch = envelopes[batch_start: batch_start + self.BATCH_SIZE]
            raw_responses = self._call_llm(batch)
            for envelope, response in zip(batch, raw_responses):
                sugg = self._parse_response(envelope, response)
                # Attach deterministic context if provided
                if deterministic_report:
                    rec = next(
                        (r for r in deterministic_report if r["raw_header"] == envelope["header"]),
                        None,
                    )
                    if rec:
                        sugg.deterministic_method = rec.get("mapping_method", "unmapped")
                        sugg.deterministic_confidence = float(rec.get("confidence", 0.0))
                suggestions.append(sugg)

        return suggestions

    # ------------------------------------------------------------------
    # INTERNAL: FEATURE ENVELOPE
    # ------------------------------------------------------------------

    def _build_envelope(self, header: str, df_raw: pd.DataFrame) -> dict:
        """Collect column statistics and sample values for the prompt payload."""
        envelope: dict = {"header": header, "samples": [], "dtype": "unknown", "stats": {}}

        if header not in df_raw.columns:
            return envelope

        col = df_raw[header]
        envelope["dtype"] = str(col.dtype)

        non_null = col.dropna()
        total = len(col)
        n_null = int(col.isna().sum())
        null_pct = round(n_null / total * 100, 1) if total else 0.0
        n_unique = int(non_null.nunique())

        # Up to 5 deduplicated, non-null sample values (cast to str for JSON safety)
        sample_pool = non_null.drop_duplicates().head(20)
        samples = [str(v) for v in sample_pool.head(5).tolist()]
        envelope["samples"] = samples

        stats: dict = {"nunique": n_unique, "null_pct": null_pct}
        if pd.api.types.is_numeric_dtype(col):
            stats["min"] = float(non_null.min()) if len(non_null) else None
            stats["max"] = float(non_null.max()) if len(non_null) else None
        envelope["stats"] = stats

        return envelope

    # ------------------------------------------------------------------
    # INTERNAL: LLM CALL
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "anthropic package not installed. Run: pip install anthropic>=0.40.0"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _get_system_prompt(self) -> str:
        if self._system_prompt is None:
            prompt_file = Path(__file__).parent / "prompts" / "field_mapper_system.txt"
            if prompt_file.exists():
                self._system_prompt = prompt_file.read_text(encoding="utf-8").strip()
            else:
                logger.warning("System prompt file not found at %s; using inline fallback.", prompt_file)
                self._system_prompt = _INLINE_SYSTEM_PROMPT
        return self._system_prompt

    def _call_llm(self, batch: List[dict]) -> List[dict]:
        """
        Send one API call for a batch of envelopes.  Returns a list of raw
        response dicts (one per envelope).  Falls back to null responses on error.
        """
        client = self._get_client()
        system_prompt = self._get_system_prompt()

        # Serialise catalogue inline (trimmed for token efficiency)
        catalogue_lines = [
            f"  - {f['name']}  [category={f['category']} format={f['format']}]"
            for f in self.catalogue_subset
        ]
        catalogue_block = "\n".join(catalogue_lines)

        # Serialise the batch of headers
        batch_payload = []
        for env in batch:
            batch_payload.append(
                {
                    "header": env["header"],
                    "dtype": env["dtype"],
                    "samples": env["samples"],
                    "stats": env["stats"],
                }
            )

        user_content = (
            f"CANONICAL FIELD CATALOGUE ({len(self.catalogue_subset)} fields):\n"
            f"{catalogue_block}\n\n"
            f"HEADERS TO MAP ({len(batch)} items):\n"
            f"{json.dumps(batch_payload, indent=2)}\n\n"
            "Return a JSON array with one object per header, in the same order."
        )

        null_resp = {
            "suggested_field": None,
            "confidence": 0.0,
            "reasoning": "LLM call failed",
            "alternative_field": None,
            "semantic_category": "unknown",
        }

        try:
            message = client.messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                temperature=self.TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            raw_text = message.content[0].text.strip()
            parsed = self._extract_json(raw_text)

            if not isinstance(parsed, list):
                parsed = [parsed] if isinstance(parsed, dict) else []

            # Pad with nulls if response count mismatches
            while len(parsed) < len(batch):
                parsed.append(null_resp.copy())

            return parsed[: len(batch)]

        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return [null_resp.copy() for _ in batch]

    @staticmethod
    def _extract_json(text: str) -> object:
        """Strip markdown fences and parse JSON from LLM response."""
        # Remove ```json ... ``` fences if present
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        return json.loads(text.strip())

    # ------------------------------------------------------------------
    # INTERNAL: RESPONSE PARSING
    # ------------------------------------------------------------------

    def _parse_response(self, envelope: dict, response: dict) -> LLMSuggestion:
        """Convert raw LLM response dict into a validated LLMSuggestion."""
        suggested = response.get("suggested_field")
        confidence = float(response.get("confidence", 0.0))
        reasoning = str(response.get("reasoning", ""))
        alternative = response.get("alternative_field")
        category = str(response.get("semantic_category", "unknown"))

        # Canonical-only enforcement — null out any hallucinated field names
        if suggested and suggested not in self.canonical_set:
            logger.warning(
                "LLM suggested non-canonical field '%s' for header '%s' — nulling.",
                suggested,
                envelope["header"],
            )
            reasoning = (
                f"[NULLED: '{suggested}' not in registry] " + reasoning
            )
            suggested = None
            confidence = 0.0

        if alternative and alternative not in self.canonical_set:
            alternative = None

        return LLMSuggestion(
            raw_header=envelope["header"],
            suggested_field=suggested,
            confidence=confidence,
            reasoning=reasoning,
            alternative_field=alternative,
            semantic_category=category,
            sample_values=envelope.get("samples", []),
        )


# ---------------------------------------------------------------------------
# HUMAN REVIEW SESSION
# ---------------------------------------------------------------------------

_DIVIDER = "━" * 46


class HumanReviewSession:
    """Interactive CLI for confirming/rejecting/remapping LLM suggestions."""

    def __init__(self, canonical_fields: Optional[List[str]] = None) -> None:
        self.canonical_fields = canonical_fields or []

    def review_cli(self, suggestions: List[LLMSuggestion]) -> List[LLMSuggestion]:
        """
        Interactive loop: present each pending suggestion and collect user decision.
        Returns the updated suggestions list.
        """
        pending = [s for s in suggestions if s.status == "pending"]
        if not pending:
            logger.info("No pending suggestions to review.")
            return suggestions

        print(f"\nHuman Review Session — {len(pending)} suggestion(s) to review\n")

        for idx, sugg in enumerate(pending, 1):
            print(f"\n[{idx}/{len(pending)}]")
            self._display_suggestion(sugg)

            while True:
                choice = input(
                    "[C]onfirm  [R]emap to different field  [S]kip  [Q]uit review\n> "
                ).strip().upper()

                if choice == "C":
                    sugg.status = "confirmed"
                    sugg.confirmed_field = sugg.suggested_field
                    note = input("Reviewer note (optional, press Enter to skip): ").strip()
                    sugg.reviewer_note = note
                    print(f"  Confirmed: {sugg.confirmed_field}")
                    break

                elif choice == "R":
                    remapped = self._remap_prompt(sugg)
                    if remapped:
                        sugg.status = "remapped"
                        sugg.confirmed_field = remapped
                        note = input("Reviewer note (optional): ").strip()
                        sugg.reviewer_note = note
                        print(f"  Remapped to: {sugg.confirmed_field}")
                    else:
                        print("  Remap cancelled.")
                        continue
                    break

                elif choice == "S":
                    sugg.status = "skipped"
                    print("  Skipped.")
                    break

                elif choice == "Q":
                    print("\nReview session terminated by user.")
                    # Mark all remaining as skipped
                    remaining = [s for s in pending if s.status == "pending"]
                    for s in remaining:
                        s.status = "skipped"
                    return suggestions

                else:
                    print("  Invalid choice. Enter C, R, S, or Q.")

        confirmed_count = sum(1 for s in suggestions if s.status in ("confirmed", "remapped"))
        print(f"\nReview complete — {confirmed_count} mapping(s) confirmed.\n")
        return suggestions

    def _display_suggestion(self, sugg: LLMSuggestion) -> None:
        print(_DIVIDER)
        print(f'Header:     "{sugg.raw_header}"')
        samples_str = str(sugg.sample_values[:5]) if sugg.sample_values else "(no samples)"
        print(f"Samples:    {samples_str}")
        if sugg.suggested_field:
            print(f"Suggested:  {sugg.suggested_field}  (confidence: {sugg.confidence:.2f})")
        else:
            print("Suggested:  (no suggestion — LLM returned null)")
        reasoning_wrapped = textwrap.fill(sugg.reasoning, width=70, subsequent_indent="                ")
        print(f"Reasoning:  {reasoning_wrapped}")
        if sugg.alternative_field:
            print(f"Alt:        {sugg.alternative_field}")
        print(f"Det. method: {sugg.deterministic_method}  (conf={sugg.deterministic_confidence:.2f})")
        print(_DIVIDER)

    def _remap_prompt(self, sugg: LLMSuggestion) -> Optional[str]:
        """Prompt user to type a canonical field name, with fuzzy suggestion help."""
        print("\nEnter canonical field name (or part of it for suggestions):")
        query = input("  > ").strip()
        if not query:
            return None

        # Fuzzy search among known canonical fields
        if self.canonical_fields:
            try:
                from rapidfuzz import process as rfprocess, fuzz as rffuzz  # type: ignore
                matches = rfprocess.extract(
                    query, self.canonical_fields, scorer=rffuzz.token_set_ratio, limit=8
                )
                if matches:
                    print("\nBest matches:")
                    for i, (name, score, _) in enumerate(matches, 1):
                        print(f"  {i}. {name}  ({score:.0f}%)")
                    sel = input("Select number (or press Enter to use your input): ").strip()
                    if sel.isdigit():
                        idx = int(sel) - 1
                        if 0 <= idx < len(matches):
                            return matches[idx][0]
            except Exception:
                pass  # rapidfuzz not available — fall through

        # Validate exact match
        if query in set(self.canonical_fields):
            return query

        confirm = input(
            f'  "{query}" is not in the canonical registry. Use anyway? [y/N] '
        ).strip().lower()
        return query if confirm == "y" else None

    def review_streamlit(self, suggestions: List[LLMSuggestion]) -> List[LLMSuggestion]:
        """
        Streamlit-based review UI.  Requires `pip install streamlit`.
        Call this from a Streamlit app context.
        """
        try:
            import streamlit as st  # type: ignore
        except ImportError:
            logger.error("streamlit not installed. Install with: pip install streamlit")
            return suggestions

        st.title("LLM Field Mapper — Human Review")

        pending = [s for s in suggestions if s.status == "pending"]
        if not pending:
            st.success("No pending suggestions.")
            return suggestions

        # Colour bands
        def _confidence_colour(c: float) -> str:
            if c >= 0.9:
                return "green"
            elif c >= 0.7:
                return "orange"
            return "red"

        for sugg in pending:
            colour = _confidence_colour(sugg.confidence)
            with st.expander(f'"{sugg.raw_header}"  →  {sugg.suggested_field or "null"}', expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Confidence:** :{colour}[{sugg.confidence:.2f}]")
                    st.write("**Reasoning:**", sugg.reasoning)
                    st.write("**Samples:**", sugg.sample_values[:5])
                with col2:
                    st.write("**Alternative:**", sugg.alternative_field or "—")
                    action = st.radio(
                        "Decision",
                        ["Confirm", "Remap", "Skip"],
                        key=f"action_{sugg.raw_header}",
                        horizontal=True,
                    )
                    if action == "Remap":
                        remap_val = st.text_input(
                            "Canonical field to map to:", key=f"remap_{sugg.raw_header}"
                        )
                    note = st.text_input("Reviewer note:", key=f"note_{sugg.raw_header}")

                if st.button("Apply", key=f"apply_{sugg.raw_header}"):
                    if action == "Confirm":
                        sugg.status = "confirmed"
                        sugg.confirmed_field = sugg.suggested_field
                    elif action == "Remap":
                        sugg.status = "remapped"
                        sugg.confirmed_field = remap_val  # type: ignore[possibly-undefined]
                    else:
                        sugg.status = "skipped"
                    sugg.reviewer_note = note
                    st.rerun()

        return suggestions


# ---------------------------------------------------------------------------
# ALIAS LEARNER
# ---------------------------------------------------------------------------


class AliasLearner:
    """
    Persists confirmed LLM suggestions as aliases in aliases_llm_confirmed.yaml.
    Deduplicates against all existing alias files before writing.
    """

    OUTPUT_FILENAME = "aliases_llm_confirmed.yaml"

    def persist_confirmed(
        self,
        confirmed: List[LLMSuggestion],
        aliases_dir: Path,
        session_id: str = "",
    ) -> int:
        """
        Append confirmed raw_header → canonical_field aliases to
        aliases_llm_confirmed.yaml.  Returns the count of NEW aliases added.
        """
        aliases_dir = Path(aliases_dir)
        aliases_dir.mkdir(parents=True, exist_ok=True)

        output_path = aliases_dir / self.OUTPUT_FILENAME

        # Load existing confirmed aliases (or start fresh)
        existing_data: dict = {}
        if output_path.exists():
            try:
                existing_data = yaml.safe_load(output_path.read_text(encoding="utf-8")) or {}
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except Exception as exc:
                logger.warning("Could not parse %s: %s — starting fresh.", output_path, exc)

        # Build a set of all aliases already known across ALL alias files
        all_known_aliases: set = set()
        for yaml_file in aliases_dir.glob("aliases_*.yaml"):
            try:
                data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
                if not isinstance(data, dict):
                    continue
                for canon, meta in data.items():
                    if isinstance(meta, list):
                        aliases = meta
                    else:
                        aliases = (meta or {}).get("aliases", []) or []
                    all_known_aliases.update(str(a).strip() for a in aliases)
            except Exception:
                pass

        added = 0
        for sugg in confirmed:
            if sugg.status not in ("confirmed", "remapped"):
                continue
            target_field = sugg.confirmed_field
            raw = sugg.raw_header.strip()

            if not target_field or not raw:
                continue
            if raw in all_known_aliases:
                logger.debug("Alias '%s' already exists — skipping.", raw)
                continue

            # Merge into output dict
            if target_field not in existing_data:
                existing_data[target_field] = {
                    "aliases": [],
                    "source": "llm_agent",
                    "confirmed_by": "human",
                }
            entry = existing_data[target_field]
            if isinstance(entry, list):
                # Normalise shorthand to long form
                existing_data[target_field] = {
                    "aliases": list(entry),
                    "source": "llm_agent",
                    "confirmed_by": "human",
                }
                entry = existing_data[target_field]

            if raw not in entry.get("aliases", []):
                entry.setdefault("aliases", []).append(raw)
                all_known_aliases.add(raw)
                added += 1
                logger.info("Alias persisted: '%s' → '%s'", raw, target_field)

        if added > 0:
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            header_comment = (
                f"# Auto-generated by LLM Mapper Agent\n"
                f"# Last updated: {timestamp}\n"
                f"# Session: {session_id}\n"
            )
            output_path.write_text(
                header_comment + yaml.dump(existing_data, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )
            logger.info("Wrote %d new alias(es) to %s", added, output_path)
        else:
            logger.info("No new aliases to persist.")

        return added


# ---------------------------------------------------------------------------
# GOVERNANCE LOGGER
# ---------------------------------------------------------------------------


class GovernanceLogger:
    """
    Writes a versioned JSON governance artifact per agent session.
    """

    def __init__(self, governance_dir: Path) -> None:
        self.governance_dir = Path(governance_dir)
        self.governance_dir.mkdir(parents=True, exist_ok=True)

    def write_session(
        self,
        session_id: str,
        input_file: str,
        portfolio_type: str,
        deterministic_stats: dict,
        suggestions: List[LLMSuggestion],
        aliases_persisted: int,
        extra: Optional[dict] = None,
    ) -> Path:
        """
        Assemble and write the governance artifact JSON.  Returns the artifact path.
        """
        now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

        llm_suggestions = [s for s in suggestions if s.deterministic_method in ("unmapped",) or True]
        sent_to_llm = len(suggestions)
        null_suggestions = sum(1 for s in suggestions if not s.suggested_field)
        avg_confidence = (
            sum(s.confidence for s in suggestions) / sent_to_llm if sent_to_llm else 0.0
        )

        human_confirmed = sum(1 for s in suggestions if s.status == "confirmed")
        human_rejected = sum(1 for s in suggestions if s.status == "rejected")
        human_remapped = sum(1 for s in suggestions if s.status == "remapped")
        human_skipped = sum(1 for s in suggestions if s.status == "skipped")

        artifact = {
            "session_id": session_id,
            "timestamp": now_utc,
            "input_file": input_file,
            "portfolio_type": portfolio_type,
            "deterministic_pass": deterministic_stats,
            "llm_pass": {
                "sent_to_llm": sent_to_llm,
                "suggestions_returned": sent_to_llm - null_suggestions,
                "null_suggestions": null_suggestions,
                "avg_confidence": round(avg_confidence, 4),
            },
            "human_review": {
                "confirmed": human_confirmed,
                "rejected": human_rejected,
                "remapped": human_remapped,
                "skipped": human_skipped,
            },
            "aliases_persisted": aliases_persisted,
            "suggestions": [s.to_dict() for s in suggestions],
        }
        if extra:
            artifact.update(extra)

        artifact_path = self.governance_dir / f"{session_id}.json"
        artifact_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=False, default=str),
            encoding="utf-8",
        )
        logger.info("Governance artifact written: %s", artifact_path)
        return artifact_path


# ---------------------------------------------------------------------------
# INLINE FALLBACK SYSTEM PROMPT (used if prompts/field_mapper_system.txt absent)
# ---------------------------------------------------------------------------

_INLINE_SYSTEM_PROMPT = """You are a specialised field mapping agent for securitisation loan tape data.
You work within a regulated financial data pipeline where precision is critical.

YOUR TASK:
Given a set of raw column headers from a lender's loan tape, suggest the most
appropriate canonical field name from the provided field catalogue.

RULES:
1. You MUST only suggest fields that exist in the provided catalogue. Never invent fields.
2. If no catalogue field is a reasonable match, return null for suggested_field.
3. Consider BOTH the header name AND the sample values when making your suggestion.
4. Pay attention to field format in the catalogue.
5. For UK mortgage/equity release data, apply domain knowledge:
   - "ERC" = Early Repayment Charge
   - "LTV" = Loan-to-Value
   - "MIG" = Mortgage Indemnity Guarantee
   - "NUTS"/"ITL" = geographic region codes
   - "Valn"/"Val" = Valuation
6. Confidence scoring:
   - 0.95-1.0: Header name clearly maps and sample values confirm
   - 0.80-0.94: Strong semantic match but some ambiguity
   - 0.60-0.79: Partial match, needs human review
   - Below 0.60: Return null

RESPONSE FORMAT:
Return ONLY valid JSON. No markdown, no explanation outside the JSON.
For a batch of headers, return a JSON array of objects.
Each object must have exactly these keys:
  suggested_field, confidence, reasoning, alternative_field, semantic_category"""
