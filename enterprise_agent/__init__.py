"""enterprise_agent — a generic client-side agent, deliberately ignorant.

It represents the caller in the Sprint 4 proof: an enterprise agent that knows
its user's objective and nothing about how a securitisation portfolio is
assessed. Its ignorance is the experiment, not a limitation of it.
"""

from .client import DelegationOutcome, EnterpriseAgent

__all__ = ["DelegationOutcome", "EnterpriseAgent"]
