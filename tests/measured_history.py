"""A pipeline history MEASURED at the scale of a test book (owner decision D21).

D21 (2026-09-30): "measured or decline if not enough history". A stage's chance
of completing, and its validity window, are measured from the client's own
history; where the history is not enough, no configured value stands in and
every weighted figure that depends on the stage is not stated.

The test books hold a handful of cases across a few extracts — far below the
production thresholds (`MIN_OBSERVATIONS` cases per stage, the run-off's
`min_events`) — so, read with the production thresholds, every weighted
figure in them is rightly withheld. Tests that check a weighted figure is the
dashboard's own therefore read the SAME book's history measured with the
thresholds set to one: the rates and windows are still measured from the
book's own extracts, never configured. The thresholds are the history owner's
own settings; nothing else differs from production.

Tests of D21 itself read the book without this and assert the withholding.
"""
from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any, Dict

#: The measurement thresholds at test-book scale.
TEST_BOOK_THRESHOLDS = {"min_observations": 1, "runoff_settings": {"min_events": 1}}


@lru_cache(maxsize=8)
def _measured(root: str, client_id: str) -> Dict[str, Any]:
    from mi_agent_api import pipeline_contract as pc
    from mi_agent_api.pipeline_history import build_historical_completion_model
    inventory = pc.weekly_extract_inventory(root, client_id)
    return build_historical_completion_model(
        inventory["extracts"],
        min_observations=TEST_BOOK_THRESHOLDS["min_observations"],
        runoff_settings=dict(TEST_BOOK_THRESHOLDS["runoff_settings"]))


def measured_history(root: str, client_id: str = "client_001") -> Dict[str, Any]:
    """The test book's own history, measured at test-book scale (a copy)."""
    return copy.deepcopy(_measured(str(root), client_id))
