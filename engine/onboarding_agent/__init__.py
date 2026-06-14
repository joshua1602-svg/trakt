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

Phase 1 (Azure-ready, domain-based consolidation) adds, on top of the above:

    -> Azure-ready run-folder contract        (storage_paths)
    -> domain detection + coverage            (domain_coverage)
    -> central lender + pipeline tape build   (central_tape_builder)
    -> dry-run handoff manifests / trigger    (promotion_planner)

These reason about DATA DOMAINS, not a fixed set of files (a combined master
tape may cover loan + borrower + collateral at once), produce blob-compatible
paths/URIs, and remain review-first and dry-run: no live Azure upload, no Event
Grid wiring, and Gates 1-5 are never run.
"""

from .onboarding_models import OnboardingProject

__all__ = ["OnboardingProject"]
