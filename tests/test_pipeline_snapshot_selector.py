from analytics.pipeline_snapshot_selector import resolve_pipeline_snapshot_selection


def test_selection_prefers_prior_choice():
    idx = resolve_pipeline_snapshot_selection(
        snapshot_names=["new.csv", "old.csv"],
        latest_blob_name="new.csv",
        prior_selection="old.csv",
    )
    assert idx == 1


def test_selection_falls_back_to_latest_then_zero():
    idx_latest = resolve_pipeline_snapshot_selection(
        snapshot_names=["new.csv", "old.csv"],
        latest_blob_name="new.csv",
    )
    assert idx_latest == 0

    idx_zero = resolve_pipeline_snapshot_selection(
        snapshot_names=["only.csv"],
        latest_blob_name="missing.csv",
    )
    assert idx_zero == 0


def test_selection_handles_empty_list():
    assert resolve_pipeline_snapshot_selection([], latest_blob_name=None) == 0
