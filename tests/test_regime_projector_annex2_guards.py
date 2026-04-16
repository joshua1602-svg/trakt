import pandas as pd

from engine.gate_4_projection.regime_projector import apply_annex2_post_projection_guards


def test_annex2_guards_generate_rrel1_and_backfill_rrec9() -> None:
    df = pd.DataFrame(
        {
            "RREL1": ["SYNTH2026POOL01", "SYNTH2026POOL01"],
            "RREL2": ["DEMO-0001", "DEMO-0002"],
            "RREL3": ["A1", "A2"],
            "RREL4": ["B1", "B2"],
            "RREL5": ["B1", "B2"],
            "RREL6": ["2026-01-31", "2026-01-31"],
            "RREC2": ["A1", "A2"],
            "RREC9": ["manual", "manual"],
        }
    )
    config = {
        "defaults": {
            "originator_legal_entity_identifier": "213800ABCDE123456701N202501",
            "collateral_type": "HOUSE",
        },
        "regime_overrides": {"ESMA_Annex2": {"securitisation_sequence": 1}},
    }
    enum_mapping = {"ESMA_Annex2": {"collateral_type": {"HOUSE": "R1"}}}

    out_df, report = apply_annex2_post_projection_guards(df, config, enum_mapping)

    assert out_df["RREL1"].iloc[0] == "213800ABCDE123456701N202601"
    assert out_df["RREL1"].nunique() == 1
    assert out_df["RREC9"].tolist() == ["R1", "R1"]
    assert report["rrec9_backfilled_from_collateral_default_rows"] == 2
