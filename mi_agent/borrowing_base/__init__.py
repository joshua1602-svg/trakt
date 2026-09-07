"""mi_agent.borrowing_base — governed facility borrowing-base calculation.

Four modules, in dependency order, none of which import Streamlit, FastAPI or
any dashboard code:

* :mod:`models` — typed contracts and the closed status vocabularies;
* :mod:`config` — the governed facility configuration loader (client
  configuration first, platform configuration second, never Python source);
* :mod:`eligibility` — the loan-level governed eligibility derivation that the
  canonical preparation layer materialises onto the frame;
* :mod:`calculator` — the pure deterministic facility calculation and its
  reconciliation invariants;
* :mod:`receipt` — the reproducible calculation receipt;
* :mod:`service` — the stable measure interface a later MI Query Agent sprint
  registers against.

The package is import-safe without pandas installed only insofar as pandas is
already a platform dependency; nothing here reaches the network, the clock
(except through the receipt's explicit timestamp) or an LLM.
"""

from .models import (  # noqa: F401
    ELIGIBILITY_STATUSES,
    ELIGIBLE,
    FIELD_ELIGIBILITY_REASON,
    FIELD_ELIGIBILITY_STATUS,
    FIELD_ELIGIBLE,
    FIELD_FACILITY_ID,
    INELIGIBLE,
    NOT_CALCULABLE,
    TREATMENT_EXCLUDE_EXCESS,
    TREATMENT_MONITOR_ONLY,
    UNDETERMINED,
    EligibilityRule,
    FacilityConfiguration,
)

__all__ = [
    "ELIGIBILITY_STATUSES", "ELIGIBLE", "INELIGIBLE", "UNDETERMINED",
    "NOT_CALCULABLE", "FIELD_ELIGIBLE", "FIELD_ELIGIBILITY_STATUS",
    "FIELD_ELIGIBILITY_REASON", "FIELD_FACILITY_ID",
    "TREATMENT_MONITOR_ONLY", "TREATMENT_EXCLUDE_EXCESS",
    "EligibilityRule", "FacilityConfiguration",
]
