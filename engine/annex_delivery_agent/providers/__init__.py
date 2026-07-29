"""
Annex providers — one module per regulatory report.

Adding an annex means adding a module here and one ``register(...)`` line in
:func:`engine.annex_delivery_agent.registry.default_registry`. Nothing in the
shared agent, lifecycle, persistence, evidence or publication layers changes.

Annex 3 and Annex 9 are intentionally absent rather than stubbed: this
repository has no authoritative normaliser, builder or schema for them, and a
placeholder registration would let an operator start a run that cannot finish.
``docs/annex_delivery_agent.md`` sets out exactly what each would need.
"""

from .base import AnnexProvider, BuildOutcome, NormalisationOutcome, ResolvedInput  # noqa: F401

__all__ = ["AnnexProvider", "ResolvedInput", "NormalisationOutcome", "BuildOutcome"]
