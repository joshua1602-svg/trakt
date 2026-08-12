"""readiness_agent — the Securitisation Readiness Agent and its governed door.

Deliberately a separate package from ``trakt_core``, ``analytics_lib`` and
``trakt_tools``. Trakt owns canonical data, calculations, methodologies,
capability discovery, provenance, permissions and audit. This package owns
only the *consumer* side: deciding what to investigate, connecting findings,
weighing materiality and forming a qualitative judgement.

The boundary is enforced by what this package is allowed to import. It reaches
Trakt through ``trakt_tools.execution`` and nothing else — no analytics module,
no canonical reader, no storage. A test asserts that, because the separation is
the entire point: an agent that could import ``analytics_lib`` would eventually
compute something itself, and then Trakt would have two answers to one question.
"""
