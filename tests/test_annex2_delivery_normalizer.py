import pandas as pd

from engine.gate_4b_delivery.annex2_delivery_normalizer import (
    apply_precision,
    generate_securitisation_id,
    normalize_boolean,
    normalize_delivery,
    validate_lei,
)


def _rules() -> dict:
    return {
        "defaults": {"reporting_year": "2026"},
        "field_rules": {
            "RREL1": {"mandatory": True, "nd_allowed": ["ND1"], "validators": {"lei": True}},
            "RREL2": {
                "mandatory": True,
                "nd_allowed": ["ND1"],
                "validators": {"regex": r"^[A-Z0-9]{18}[0-9]{2}N\d{4}\d{2}$"},
                "generator": {
                    "type": "securitisation_id",
                    "lei_field": "RREL1",
                    "year_field": "reporting_year",
                    "sequence_width": 2,
                },
            },
            "RREC36": {
                "mandatory": False,
                "nd_allowed": ["ND5"],
                "transform": {"boolean": "xsd_lowercase_true_false"},
            },
            "RREC8": {
                "mandatory": False,
                "nd_allowed": ["ND5"],
                "transform": {"enum_map": {"employed": "EMP", "EMP": "EMP"}},
            },
            "RREC25": {
                "mandatory": False,
                "nd_allowed": ["ND5"],
                "precision": {"total_digits": 8, "fraction_digits": 4},
            },
            "RREC26": {
                "mandatory": False,
                "nd_allowed": ["ND5"],
                "precision": {"total_digits": 6, "fraction_digits": 2},
            },
        },
    }


def test_generate_and_validate_identifiers() -> None:
    lei = "5493001KJTIIGC8Y1R12"
    assert validate_lei(lei)
    assert not validate_lei("BADLEI")
    assert generate_securitisation_id(lei, "2026", 1) == "5493001KJTIIGC8Y1R12N202601"


def test_boolean_normalization() -> None:
    assert normalize_boolean("Y") == "true"
    assert normalize_boolean("0") == "false"
    assert normalize_boolean("MAYBE") is None


def test_enum_mapping_nd_and_precision_enforcement() -> None:
    df = pd.DataFrame(
        [
            {
                "RREL1": "5493001KJTIIGC8Y1R12",
                "RREL2": "",
                "reporting_year": "2026",
                "RREC36": "True",
                "RREC8": "employed",
                "RREC25": "3.45678",
                "RREC26": "ND4",
            }
        ]
    )

    out_df, issues, summary = normalize_delivery(df, _rules())

    assert out_df.loc[0, "RREL2"] == "5493001KJTIIGC8Y1R12N202601"
    assert out_df.loc[0, "RREC36"] == "true"
    assert out_df.loc[0, "RREC8"] == "EMP"
    assert out_df.loc[0, "RREC25"] == "3.4568"
    assert summary["preflight"]["status"] == "FAIL"
    assert any(i.issue_type == "nd_not_allowed" and i.field == "RREC26" for i in issues)


def test_preflight_fail_on_missing_mandatory() -> None:
    df = pd.DataFrame([{"RREL1": "", "RREL2": "", "reporting_year": "2026"}])
    _, _, summary = normalize_delivery(df, _rules())
    assert summary["preflight"]["status"] == "FAIL"
    assert summary["errors_total"] >= 1


def test_precision_helper() -> None:
    value, err = apply_precision("123.4567", total_digits=8, fraction_digits=2)
    assert err is None
    assert value == "123.46"

    value, err = apply_precision("1234567.89", total_digits=6, fraction_digits=2)
    assert value is None
    assert err is not None


def test_missing_optional_field_not_hard_fail() -> None:
    df = pd.DataFrame([{"RREL1": "5493001KJTIIGC8Y1R12"}])
    rules = {
        "field_rules": {
            "RREC39": {
                "mandatory": False,
                "enforce_presence": False,
                "nd_allowed": ["ND5"],
                "transform": {"boolean": "xsd_lowercase_true_false"},
            }
        }
    }
    _, issues, summary = normalize_delivery(df, rules)
    assert issues == []
    assert summary["preflight"]["status"] == "PASS"


def test_mandatory_default_nd1_applied_for_income_block() -> None:
    df = pd.DataFrame([{"RREL16": "", "RREL17": ""}])
    rules = {
        "field_rules": {
            "RREL16": {
                "mandatory": True,
                "enforce_presence": True,
                "nd_allowed": ["ND1", "ND2", "ND3", "ND4"],
                "default_allowed": True,
                "default_value": "ND1",
            },
            "RREL17": {
                "mandatory": True,
                "enforce_presence": True,
                "nd_allowed": ["ND1", "ND2", "ND3", "ND4"],
                "default_allowed": True,
                "default_value": "ND1",
            },
        }
    }
    out_df, issues, summary = normalize_delivery(df, rules)
    assert out_df.loc[0, "RREL16"] == "ND1"
    assert out_df.loc[0, "RREL17"] == "ND1"
    assert issues == []
    assert summary["preflight"]["status"] == "PASS"


def test_cohort1_rrel25_derives_from_dates_before_nd_default() -> None:
    df = pd.DataFrame([{"RREL23": "2025-01-15", "RREL24": "2043-01-15", "RREL25": ""}])
    rules = {
        "field_rules": {
            "RREL25": {
                "mandatory": True,
                "enforce_presence": True,
                "nd_allowed": ["ND1", "ND2", "ND3", "ND4"],
                "derive": {"type": "months_between_dates", "start_field": "RREL23", "end_field": "RREL24"},
                "default_allowed": True,
                "default_value": "ND1",
                "validators": {"regex": r"^\d{1,4}$"},
            }
        }
    }
    out_df, issues, summary = normalize_delivery(df, rules)
    assert out_df.loc[0, "RREL25"] == "216"
    assert issues == []
    assert summary["preflight"]["status"] == "PASS"


def test_cohort1_rrel22_boolean_normalization_when_non_nd() -> None:
    df = pd.DataFrame([{"RREL22": "Y"}])
    rules = {
        "field_rules": {
            "RREL22": {
                "mandatory": True,
                "enforce_presence": True,
                "nd_allowed": ["ND1", "ND2", "ND3", "ND4"],
                "default_allowed": True,
                "default_value": "ND1",
                "transform": {"boolean": "xsd_lowercase_true_false"},
            }
        }
    }
    out_df, issues, summary = normalize_delivery(df, rules)
    assert out_df.loc[0, "RREL22"] == "true"
    assert issues == []
    assert summary["preflight"]["status"] == "PASS"


def test_cohort2_defaults_and_type_validations() -> None:
    df = pd.DataFrame(
        [
            {
                "RREL41": "",
                "RREL50": "1.23456",
                "RREL51": "2026-01-31",
                "RREL52": "",
                "RREL53": "2026-02-28",
            }
        ]
    )
    rules = {
        "field_rules": {
            "RREL41": {
                "mandatory": True,
                "enforce_presence": True,
                "nd_allowed": ["ND1", "ND2", "ND3", "ND4"],
                "default_allowed": True,
                "default_value": "ND1",
                "precision": {"total_digits": 18, "fraction_digits": 2},
            },
            "RREL50": {
                "mandatory": True,
                "enforce_presence": True,
                "nd_allowed": ["ND1", "ND2", "ND3", "ND4"],
                "default_allowed": True,
                "default_value": "ND1",
                "precision": {"total_digits": 8, "fraction_digits": 4},
            },
            "RREL51": {
                "mandatory": True,
                "enforce_presence": True,
                "nd_allowed": ["ND1", "ND2", "ND3", "ND4"],
                "default_allowed": True,
                "default_value": "ND1",
                "validators": {"regex": r"^\d{4}-\d{2}-\d{2}$"},
            },
            "RREL52": {
                "mandatory": True,
                "enforce_presence": True,
                "nd_allowed": ["ND1", "ND2", "ND3", "ND4"],
                "default_allowed": True,
                "default_value": "ND1",
                "precision": {"total_digits": 8, "fraction_digits": 4},
            },
            "RREL53": {
                "mandatory": True,
                "enforce_presence": True,
                "nd_allowed": ["ND1", "ND2", "ND3", "ND4"],
                "default_allowed": True,
                "default_value": "ND1",
                "validators": {"regex": r"^\d{4}-\d{2}-\d{2}$"},
            },
        }
    }
    out_df, issues, summary = normalize_delivery(df, rules)
    assert out_df.loc[0, "RREL41"] == "ND1"
    assert out_df.loc[0, "RREL50"] == "1.2346"
    assert out_df.loc[0, "RREL51"] == "2026-01-31"
    assert out_df.loc[0, "RREL52"] == "ND1"
    assert out_df.loc[0, "RREL53"] == "2026-02-28"
    assert issues == []
    assert summary["preflight"]["status"] == "PASS"
