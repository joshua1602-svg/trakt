"""analytics_lib — pure, route-agnostic shared analytics library (Phase 1).

A dependency-light substrate of pure analytical functions consumed by the MI
and M&A routes in later phases. Functions take pandas DataFrames + config
dicts/paths and return DataFrames or plain dict/list results.

Deliberate constraints (Phase 1 scope):
  * no orchestration, no runtime routes, no MI/M&A agent execution;
  * no snapshot/history persistence, no Azure, no Streamlit, no LLM calls;
  * no imports from the legacy ``analytics/`` Streamlit app;
  * no chart output (the MI Agent owns the governed chart factory).
"""

from __future__ import annotations

from .buckets import (
    apply_bucket,
    materialise_buckets,
    normalise_scale,
)
from .cohort import (
    add_cohort_period,
    cohort_period,
    cohort_table,
    months_on_book,
)
from .concentration import (
    group_shares,
    limit_usage,
    rag_status,
    top_n_concentration,
)
from .config_loader import load_bucket_config, load_yaml
from .stratify import stratify

__all__ = [
    "apply_bucket",
    "materialise_buckets",
    "normalise_scale",
    "stratify",
    "group_shares",
    "top_n_concentration",
    "limit_usage",
    "rag_status",
    "add_cohort_period",
    "cohort_period",
    "cohort_table",
    "months_on_book",
    "load_bucket_config",
    "load_yaml",
]
