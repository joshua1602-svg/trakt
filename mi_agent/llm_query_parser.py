#!/usr/bin/env python3
"""
llm_query_parser.py

Translate a natural-language MI question into an :class:`MIQuerySpec`.

Two modes:
    * llm_enabled=False (default)  -> deterministic, offline pattern matcher.
      Safe for unit tests; no network, no API key required.
    * llm_enabled=True             -> optional, mockable Claude call.  The LLM
      is only ever shown the *semantic field catalogue* (field keys, display
      names, descriptions, roles, allowed chart roles / aggregations) — never
      raw data.  It must return STRICT JSON matching MIQuerySpec.  Generated
      content is parsed as data only; it is NEVER executed.

This module deliberately resolves field references against the *actual* MI
semantic registry (by role / format / keyword) rather than hard-coding field
names, because canonical field names differ across deployments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .mi_query_spec import MIQuerySpec
from .mi_query_validator import load_mi_semantics

# Cheap default model for NL->spec parsing.  Overridable via the `model` arg.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# --------------------------------------------------------------------------- #
# Field resolution helpers (work against whatever registry is loaded)
# --------------------------------------------------------------------------- #


def _fields(semantics: dict) -> Dict[str, dict]:
    return semantics.get("fields", {})


def find_field(
    semantics: dict,
    role: Optional[str] = None,
    fmt: Optional[str] = None,
    keywords: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> Optional[str]:
    """Return the best-matching semantic field key, or None.

    Phase 1 requires a keyword hit (on key or display name); phase 2 relaxes to
    any field matching the requested role/format.
    """
    items = _fields(semantics)
    exclude = set(exclude)
    keywords = tuple(k.lower() for k in keywords)

    def ok(key: str, entry: dict) -> bool:
        if key in exclude:
            return False
        if role and entry.get("role") != role:
            return False
        if fmt and entry.get("format") != fmt:
            return False
        return True

    if keywords:
        for key, entry in items.items():
            if not ok(key, entry):
                continue
            hay = f"{key} {entry.get('display_name', '')}".lower()
            if any(kw in hay for kw in keywords):
                return key
    for key, entry in items.items():
        if ok(key, entry):
            return key
    return None


def _balance_metric(semantics) -> Optional[str]:
    return find_field(semantics, role="metric", fmt="currency",
                      keywords=("balance", "outstanding", "principal"))


def _ltv_metric(semantics) -> Optional[str]:
    return find_field(semantics, role="metric", fmt="percent",
                      keywords=("ltv", "loan_to_value"))


def _age_metric(semantics) -> Optional[str]:
    return find_field(semantics, role="metric", fmt="integer", keywords=("age",))


def _dimension(semantics, keywords=(), exclude=()) -> Optional[str]:
    return find_field(semantics, role="dimension", keywords=keywords, exclude=exclude)


def _default_weight(semantics, metric_key: Optional[str]) -> Optional[str]:
    if metric_key and metric_key in _fields(semantics):
        wf = _fields(semantics)[metric_key].get("weight_field")
        if wf:
            return wf
    return semantics.get("metadata", {}).get("default_weight_field")


# --------------------------------------------------------------------------- #
# Deterministic (offline) parser
# --------------------------------------------------------------------------- #


def _deterministic_spec(question: str, semantics: dict) -> MIQuerySpec:
    q = question.lower().strip()

    # heatmap <metric> by <dimA> and <dimB>
    if "heatmap" in q:
        metric = _ltv_metric(semantics) if "ltv" in q else _balance_metric(semantics)
        d1 = _dimension(semantics, keywords=("region", "geograph", "country"))
        d2 = _dimension(semantics, keywords=("broker", "channel", "type", "status"),
                        exclude={d1} if d1 else ())
        if not d2:
            d2 = _dimension(semantics, exclude={d1} if d1 else ())
        dims = [d for d in (d1, d2) if d]
        return MIQuerySpec(
            intent="chart", chart_type="heatmap",
            metric=metric, dimensions=dims, aggregation="avg" if "ltv" in q else "sum",
            title=question.strip(),
            explanation="Heatmap of metric intensity across two dimensions.",
            output_format="chart",
        )

    # treemap <metric> by <dimA> and <dimB>
    if "treemap" in q:
        metric = _balance_metric(semantics)
        d1 = _dimension(semantics, keywords=("region", "geograph", "country"))
        d2 = _dimension(semantics, keywords=("broker", "channel"),
                        exclude={d1} if d1 else ())
        hierarchy = [d for d in (d1, d2) if d]
        return MIQuerySpec(
            intent="chart", chart_type="treemap",
            metric=metric, hierarchy=hierarchy, aggregation="sum",
            title=question.strip(),
            explanation="Treemap sized by metric across a dimension hierarchy.",
            output_format="chart",
        )

    # "<a> by <b> by <c>"  -> bubble (three measures)
    parts = [p.strip() for p in re.split(r"\bby\b", q) if p.strip()]
    if len(parts) >= 3:
        # heuristics: ltv -> percent axis, age -> integer axis, balance -> size
        x = _age_metric(semantics) if "age" in q else _balance_metric(semantics)
        y = _ltv_metric(semantics) if "ltv" in q else _balance_metric(semantics)
        size = _balance_metric(semantics)
        return MIQuerySpec(
            intent="chart", chart_type="bubble",
            x=x, y=y, size=size, aggregation="loan_level",
            title=question.strip(),
            explanation="Bubble chart: two numeric axes sized by a measure.",
            output_format="chart",
        )

    # "<metric> by <dimension>"  -> bar
    if len(parts) == 2:
        left, right = parts[0], parts[1]
        # metric on the left
        if "ltv" in left:
            metric = _ltv_metric(semantics)
            agg = "weighted_avg"
        elif "balance" in left or "amount" in left:
            metric = _balance_metric(semantics)
            agg = "sum"
        else:
            metric = _balance_metric(semantics)
            agg = "sum"
        # dimension on the right
        if "region" in right or "country" in right:
            dimension = _dimension(semantics, keywords=("region", "geograph", "country"))
        elif "broker" in right or "channel" in right:
            dimension = _dimension(semantics, keywords=("broker", "channel"))
        else:
            dimension = _dimension(semantics, keywords=tuple(right.split()))
        weight = _default_weight(semantics, metric) if agg == "weighted_avg" else None
        return MIQuerySpec(
            intent="chart", chart_type="bar",
            metric=metric, dimension=dimension, aggregation=agg,
            weight_field=weight, title=question.strip(),
            explanation=f"Bar chart of {agg} metric by dimension.",
            output_format="chart",
        )

    # Fallback: a summary with no chart.
    return MIQuerySpec(
        intent="summary", chart_type="none", aggregation="count",
        title=question.strip(),
        explanation="Could not map question to a chart deterministically.",
        output_format="text",
    )


# --------------------------------------------------------------------------- #
# Prompt building + LLM response parsing
# --------------------------------------------------------------------------- #


def _catalogue(semantics: dict) -> List[Dict[str, Any]]:
    """Compact, data-free catalogue passed to the LLM."""
    out = []
    for key, entry in _fields(semantics).items():
        out.append({
            "field": key,
            "display_name": entry.get("display_name", ""),
            "description": entry.get("description", ""),
            "role": entry.get("role"),
            "format": entry.get("format"),
            "chartable": entry.get("chartable"),
            "allowed_aggregations": entry.get("allowed_aggregations", []),
            "allowed_chart_roles": entry.get("allowed_chart_roles", []),
        })
    return out


def build_prompt(user_question: str, mi_semantics: dict) -> Dict[str, str]:
    """Return {"system": ..., "user": ...} prompt parts for the LLM."""
    catalogue = _catalogue(mi_semantics)
    system = (
        "You translate a natural-language Management Information (MI) question "
        "into a single JSON object describing a chart/table request.\n"
        "RULES:\n"
        "1. Use ONLY field keys from the provided catalogue for metric, "
        "dimension, x, y, size, color, dimensions, hierarchy and filter keys.\n"
        "2. chart_type must be one of: bar, line, scatter, bubble, heatmap, "
        "treemap, none.\n"
        "3. intent must be one of: chart, table, summary.\n"
        "4. aggregation must be one of: sum, avg, weighted_avg, count, "
        "count_distinct, median, distribution, loan_level, balance_sum, and "
        "must be allowed for the chosen metric.\n"
        "5. Respect each field's allowed_chart_roles.\n"
        "6. Output STRICT JSON ONLY — no prose, no markdown fences.\n"
        "7. Always include a short 'explanation' string.\n"
    )
    user = (
        "Semantic field catalogue (JSON):\n"
        + json.dumps(catalogue, indent=0)
        + "\n\nUser question:\n"
        + user_question.strip()
        + "\n\nReturn the MIQuerySpec JSON now."
    )
    return {"system": system, "user": user}


def parse_llm_response_to_spec(response_json: Any) -> MIQuerySpec:
    """Parse a raw LLM response (str or dict) into an MIQuerySpec."""
    if isinstance(response_json, MIQuerySpec):
        return response_json
    if isinstance(response_json, str):
        text = response_json.strip()
        # tolerate ```json fences
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        # extract the first {...} block defensively
        if not text.startswith("{"):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)
        data = json.loads(text)
    elif isinstance(response_json, dict):
        data = response_json
    else:
        raise TypeError("response_json must be str, dict, or MIQuerySpec")
    return MIQuerySpec.from_dict(data)


def _call_llm(prompt: Dict[str, str], model: str) -> str:
    """Live Claude call (lazy import; mirrors existing repo conventions)."""
    import os

    try:
        import anthropic  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when used
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic>=0.40.0"
        ) from exc

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.0,
        system=prompt["system"],
        messages=[{"role": "user", "content": prompt["user"]}],
    )
    return message.content[0].text


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def parse_user_question(
    user_question: str,
    semantics_path,
    model: Optional[str] = None,
    llm_enabled: bool = False,
    llm_callable=None,
) -> MIQuerySpec:
    """Translate a natural-language question into an MIQuerySpec.

    Parameters
    ----------
    user_question : str
    semantics_path : path to mi_semantics_field_registry.yaml (or a pre-loaded dict)
    model : Claude model id (defaults to a cheap model)
    llm_enabled : if False (default), use the offline deterministic parser
    llm_callable : optional injected callable(prompt_dict) -> str for testing /
        mocking the LLM without a network call.
    """
    semantics = semantics_path if isinstance(semantics_path, dict) \
        else load_mi_semantics(semantics_path)

    if not llm_enabled and llm_callable is None:
        return _deterministic_spec(user_question, semantics)

    prompt = build_prompt(user_question, semantics)
    if llm_callable is not None:
        raw = llm_callable(prompt)
    else:
        raw = _call_llm(prompt, model or DEFAULT_MODEL)
    return parse_llm_response_to_spec(raw)
