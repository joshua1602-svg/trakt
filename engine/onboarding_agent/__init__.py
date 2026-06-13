"""
engine.onboarding_agent
=======================

Trakt Onboarding Agent v2.

A thin, auditable, end-to-end lender onboarding workflow that turns a folder of
lender data-room artefacts into a reviewable onboarding pack:

    folder of lender artefacts
      -> file classification          (file_classifier)
      -> file profiling               (file_profiler)
      -> candidate keys + overlap     (source_consolidator)
      -> field mapping proposals      (mapping_proposer)
      -> config suggestions           (config_suggester)
      -> gap questions                (gap_analyzer)
      -> review pack (HTML)           (review_pack_builder)
      -> optional pipeline handoff    (onboarding_orchestrator)

It deliberately reuses the existing Gate 1 alignment engine
(``engine.gate_1_alignment.semantic_alignment``) for header normalisation,
alias matching and confidence scoring rather than reimplementing it.

The agent produces *recommendations only*. It never mutates canonical data,
production config, or makes final unreviewed decisions.
"""

from .onboarding_models import OnboardingProject

__all__ = ["OnboardingProject"]
