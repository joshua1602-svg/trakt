from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.enum_agent.enum_mapping_agent import (
    EnumAliasLearner,
    EnumSuggestion,
    resolve_enums_for_field,
)


def test_invariants_and_review_guards() -> None:
    s = EnumSuggestion(
        field_name="interest_rate_type",
        raw_value="foo",
        suggested_value="OUTSIDE",
        confidence=0.7,
        reasoning="",
        alternative_value="ALSO_OUTSIDE",
        allowed_values=["FIXED", "VARIABLE"],
        count=3,
    )
    assert s.suggested_value is None
    assert s.alternative_value is None

    try:
        s.confirm()
        raise AssertionError("confirm should fail when suggested_value is None")
    except ValueError:
        pass

    try:
        s.remap("OUTSIDE")
        raise AssertionError("remap should fail for value outside allowed_values")
    except ValueError:
        pass


def test_resolution_and_candidate_rules(tmp_path: Path) -> None:
    base = tmp_path / "enum_synonyms.yaml"
    base.write_text("interest_rate_type:\n  manual:\n    fixed rate: FIXED\n", encoding="utf-8")
    confirmed = tmp_path / "enum_synonyms_confirmed.yaml"
    confirmed.write_text("global: {}\n", encoding="utf-8")

    series = pd.Series(["FIXED", "float", "fijo", "fixed rate", "variable", "float"], dtype="object")
    mapped, report, candidates, _ = resolve_enums_for_field(
        field_name="interest_rate_type",
        series=series,
        allowed_values=["FIXED", "VARIABLE"],
        namespace="ERE",
        regime="annex12",
        fuzzy_threshold=95.0,
        review_threshold=0.92,
        synonyms_base_path=base,
        synonyms_confirmed_path=confirmed,
    )

    by_raw = {r.raw_value: r for r in report}
    assert by_raw["FIXED"].deterministic_method == "exact"
    assert by_raw["fixed rate"].deterministic_method == "synonym"
    assert by_raw["float"].deterministic_method == "unmapped"
    assert by_raw["fijo"].deterministic_method == "unmapped"

    candidate_raw = {c.raw_value for c in candidates}
    assert "float" in candidate_raw
    assert "fijo" in candidate_raw
    assert mapped.iloc[0] == "FIXED"


def test_alias_learner_namespace_regime_schema(tmp_path: Path) -> None:
    learner = EnumAliasLearner(tmp_path / "enum_synonyms_confirmed.yaml")
    s1 = EnumSuggestion(
        field_name="interest_rate_type",
        raw_value="fijo",
        suggested_value="FIXED",
        confidence=0.95,
        reasoning="",
        alternative_value=None,
        allowed_values=["FIXED", "VARIABLE"],
        count=4,
        namespace="ERE",
        regime="annex12",
        status="confirmed",
        confirmed_value="FIXED",
    )
    added = learner.persist_confirmed([s1])
    assert added == 1
    content = (tmp_path / "enum_synonyms_confirmed.yaml").read_text(encoding="utf-8")
    assert "ERE:" in content
    assert "annex12:" in content
    assert "interest_rate_type:" in content
    assert "fijo: FIXED" in content


def test_llm_json_extraction_is_robust() -> None:
    from engine.enum_agent.enum_mapping_agent import LLMEnumMapper

    parsed = LLMEnumMapper._extract_json("prefix```json\n[{\"suggested_value\": null}]\n```suffix")
    assert isinstance(parsed, list)

    parsed_bad = LLMEnumMapper._extract_json("not json")
    assert parsed_bad == []


def test_load_synonyms_global_fallback(tmp_path: Path) -> None:
    from engine.enum_agent.enum_mapping_agent import load_enum_synonyms

    base = tmp_path / "enum_synonyms.yaml"
    base.write_text("interest_rate_type:\n  manual:\n    fixed: FIXED\n", encoding="utf-8")
    confirmed = tmp_path / "enum_synonyms_confirmed.yaml"
    confirmed.write_text(
        "global:\n  annex12:\n    interest_rate_type:\n      tipo variable: VARIABLE\n",
        encoding="utf-8",
    )

    merged = load_enum_synonyms(base, confirmed, "interest_rate_type", namespace="ERE", regime="annex12")
    assert merged["fixed"] == "FIXED"
    assert merged["tipo variable"] == "VARIABLE"
