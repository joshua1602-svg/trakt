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


def test_cohort3_defaults_for_xml_mandatory_blank_cluster() -> None:
    df = pd.DataFrame(
        [
            {
                "RREL54": "",
                "RREL55": "",
                "RREL56": "",
                "RREL57": "",
                "RREL58": "",
                "RREL59": "",
                "RREL60": "",
                "RREL9": "",
                "RREC6": "",
                "RREC8": "",
                "RREC10": "",
                "RREC11": "",
                "RREC15": "",
                "RREC17": "",
                "RREC18": "",
                "RREC19": "",
                "RREC20": "",
                "RREL40": "",
            }
        ]
    )
    rules = {
        "field_rules": {
            code: {
                "mandatory": True,
                "enforce_presence": True,
                "nd_allowed": ["ND1", "ND2", "ND3", "ND4"],
                "default_allowed": True,
                "default_value": "ND1",
            }
            for code in [
                "RREL40",
                "RREL54",
                "RREL55",
                "RREL56",
                "RREL57",
                "RREL58",
                "RREL59",
                "RREL60",
                "RREL9",
                "RREC6",
                "RREC8",
                "RREC10",
                "RREC11",
                "RREC15",
                "RREC17",
                "RREC18",
                "RREC19",
                "RREC20",
            ]
        }
    }

    out_df, issues, summary = normalize_delivery(df, rules)

    for code in rules["field_rules"].keys():
        assert out_df.loc[0, code] == "ND1"
    assert issues == []
    assert summary["preflight"]["status"] == "PASS"


def test_cohort4_defaults_for_rrel76_family_blank_cluster() -> None:
    fields = [
        "RREL76",
        "RREL78",
        "RREL65",
        "RREL66",
        "RREL72",
        "RREL70",
        "RREL62",
        "RREL63",
        "RREL64",
        "RREL79",
        "RREL80",
        "RREL81",
        "RREC23",
        "RREC21",
    ]
    df = pd.DataFrame([{field: "" for field in fields}])
    rules = {
        "field_rules": {
            code: {
                "mandatory": True,
                "enforce_presence": True,
                "nd_allowed": ["ND1", "ND2", "ND3", "ND4"],
                "default_allowed": True,
                "default_value": "ND1",
            }
            for code in fields
        }
    }

    out_df, issues, summary = normalize_delivery(df, rules)

    for code in fields:
        assert out_df.loc[0, code] == "ND1"
    assert issues == []
    assert summary["preflight"]["status"] == "PASS"


def test_xsd_alignment_enum_boolean_and_nd5_defaults() -> None:
    df = pd.DataFrame(
        [
            {
                "RREL69": "Active",
                "RREL42": "manual",
                "RREL13": "manual",
                "RREL27": "manual",
                "RREL37": "Quarterly",
                "RREL38": "Monthly",
                "RREL44": "SONIA",
                "RREL45": "1M",
                "RREL35": "Interest roll-up",
                "RREC9": "R1",
                "RREC14": "Full survey",
                "RREL10": "Y",
                "RREL14": "N",
                "RREL75": "Y",
                "RREL54": "",
            }
        ]
    )
    rules = {
        "field_rules": {
            "RREL69": {"transform": {"enum_map": {"Active": "PERF", "PERF": "PERF"}}, "nd_allowed": ["ND5"]},
            "RREL42": {"transform": {"enum_map": {"manual": "OTHR", "OTHR": "OTHR"}}, "nd_allowed": ["ND5"]},
            "RREL13": {"transform": {"enum_map": {"manual": "OTHR", "OTHR": "OTHR"}}, "nd_allowed": ["ND5"]},
            "RREL27": {"transform": {"enum_map": {"manual": "OTHR", "OTHR": "OTHR"}}, "nd_allowed": ["ND5"]},
            "RREL37": {"transform": {"enum_map": {"Quarterly": "QUTR", "QUTR": "QUTR"}}, "nd_allowed": ["ND5"]},
            "RREL38": {"transform": {"enum_map": {"Monthly": "MNTH", "MNTH": "MNTH"}}, "nd_allowed": ["ND5"]},
            "RREL44": {"transform": {"enum_map": {"SONIA": "EONS", "EONS": "EONS"}}, "nd_allowed": ["ND5"]},
            "RREL45": {"transform": {"enum_map": {"1M": "MNTH", "MNTH": "MNTH"}}, "nd_allowed": ["ND5"]},
            "RREL35": {"transform": {"enum_map": {"Interest roll-up": "OTHR", "OTHR": "OTHR"}}, "nd_allowed": ["ND5"]},
            "RREC9": {"transform": {"enum_map": {"R1": "RHOS", "RHOS": "RHOS"}}, "nd_allowed": ["ND5"]},
            "RREC14": {"transform": {"enum_map": {"Full survey": "FOEI", "FOEI": "FOEI"}}, "nd_allowed": ["ND5"]},
            "RREL10": {"transform": {"boolean": "xsd_lowercase_true_false"}, "nd_allowed": ["ND5"]},
            "RREL14": {"transform": {"boolean": "xsd_lowercase_true_false"}, "nd_allowed": ["ND5"]},
            "RREL75": {"transform": {"boolean": "xsd_lowercase_true_false"}, "nd_allowed": ["ND5"]},
            "RREL54": {
                "mandatory": True,
                "enforce_presence": True,
                "nd_allowed": ["ND5"],
                "default_allowed": True,
                "default_value": "ND5",
            },
        }
    }

    out_df, issues, summary = normalize_delivery(df, rules)

    assert out_df.loc[0, "RREL69"] == "PERF"
    assert out_df.loc[0, "RREL42"] == "OTHR"
    assert out_df.loc[0, "RREL13"] == "OTHR"
    assert out_df.loc[0, "RREL27"] == "OTHR"
    assert out_df.loc[0, "RREL37"] == "QUTR"
    assert out_df.loc[0, "RREL38"] == "MNTH"
    assert out_df.loc[0, "RREL44"] == "EONS"
    assert out_df.loc[0, "RREL45"] == "MNTH"
    assert out_df.loc[0, "RREL35"] == "OTHR"
    assert out_df.loc[0, "RREC9"] == "RHOS"
    assert out_df.loc[0, "RREC14"] == "FOEI"
    assert out_df.loc[0, "RREL10"] == "true"
    assert out_df.loc[0, "RREL14"] == "false"
    assert out_df.loc[0, "RREL75"] == "true"
    assert out_df.loc[0, "RREL54"] == "ND5"
    assert issues == []
    assert summary["preflight"]["status"] == "PASS"


def test_xsd_alignment_nd_profile_and_additional_enum_coercions() -> None:
    df = pd.DataFrame(
        [
            {
                "RREL26": "Direct",
                "RREC7": "Owner Occupied",
                "RREL11": "West Midlands",
                "RREL12": "London",
                "RREL83": "213800ABCDE123456701N202501",
                "RREL58": "",
                "RREC17": "",
                "RREC18": "",
                "RREC19": "",
                "RREC23": "",
            }
        ]
    )
    rules = {
        "field_rules": {
            "RREL26": {"transform": {"enum_map": {"Direct": "DRCT", "DRCT": "DRCT"}}, "nd_allowed": ["ND5"]},
            "RREC7": {"transform": {"enum_map": {"Owner Occupied": "POWN", "POWN": "POWN"}}, "nd_allowed": ["ND5"]},
            "RREL11": {"transform": {"geography_map": {"West Midlands": "GBZZZ", "GBZZZ": "GBZZZ"}}, "nd_allowed": ["ND5"]},
            "RREL12": {"transform": {"geography_map": {"London": "GBZZZ", "GBZZZ": "GBZZZ"}}, "nd_allowed": ["ND5"]},
            "RREL83": {
                "transform": {
                    "enum_map": {
                        "213800ABCDE123456701N202501": "213800ABCDE123456701",
                        "213800ABCDE123456701": "213800ABCDE123456701",
                    }
                },
                "nd_allowed": ["ND5"],
            },
            "RREL58": {"mandatory": True, "enforce_presence": True, "nd_allowed": ["ND1", "ND2", "ND3"], "default_allowed": True, "default_value": "ND1"},
            "RREC17": {"mandatory": True, "enforce_presence": True, "nd_allowed": ["ND1", "ND2", "ND3"], "default_allowed": True, "default_value": "ND1"},
            "RREC18": {"mandatory": True, "enforce_presence": True, "nd_allowed": ["ND1", "ND2", "ND3"], "default_allowed": True, "default_value": "ND1"},
            "RREC19": {"mandatory": True, "enforce_presence": True, "nd_allowed": ["ND1", "ND2", "ND3"], "default_allowed": True, "default_value": "ND1"},
            "RREC23": {"mandatory": True, "enforce_presence": True, "nd_allowed": ["ND1", "ND2", "ND3"], "default_allowed": True, "default_value": "ND1"},
        }
    }

    out_df, issues, summary = normalize_delivery(df, rules)

    assert out_df.loc[0, "RREL26"] == "DRCT"
    assert out_df.loc[0, "RREC7"] == "POWN"
    assert out_df.loc[0, "RREL11"] == "GBZZZ"
    assert out_df.loc[0, "RREL12"] == "GBZZZ"
    assert out_df.loc[0, "RREL83"] == "213800ABCDE123456701"
    assert out_df.loc[0, "RREL58"] == "ND1"
    assert out_df.loc[0, "RREC17"] == "ND1"
    assert out_df.loc[0, "RREC18"] == "ND1"
    assert out_df.loc[0, "RREC19"] == "ND1"
    assert out_df.loc[0, "RREC23"] == "ND1"
    assert issues == []
    assert summary["preflight"]["status"] == "PASS"
