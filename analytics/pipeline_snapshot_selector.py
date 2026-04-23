"""Pure selection helpers for pipeline snapshot UX."""

from __future__ import annotations


def resolve_pipeline_snapshot_selection(
    snapshot_names: list[str],
    latest_blob_name: str | None,
    prior_selection: str | None = None,
) -> int:
    """Return preferred selectbox index for pipeline snapshots."""
    if not snapshot_names:
        return 0
    if prior_selection and prior_selection in snapshot_names:
        return snapshot_names.index(prior_selection)
    if latest_blob_name and latest_blob_name in snapshot_names:
        return snapshot_names.index(latest_blob_name)
    return 0
