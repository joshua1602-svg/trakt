"""
engine.annex_delivery_agent.validators.base
===========================================

The schema-validator seam.

An annex declares *which* validator checks its artefact; the shared lifecycle
runs it and records the outcome. Nothing above this interface knows whether the
schema is an XSD, a JSON Schema or a bespoke rulebook — which is what allows a
future annex delivered as, say, CSV-with-a-control-file to reuse the whole
lifecycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from ..contracts import SchemaValidationResult


class ArtefactValidator(ABC):
    """Validates a produced artefact against its authoritative schema."""

    #: Stable identifier recorded in the governed result.
    name: str = "validator"

    @abstractmethod
    def validate(self, artefact_path: Path, *, schema_path: Path,
                 parsed: Optional[Any] = None) -> SchemaValidationResult:
        """Validate ``artefact_path`` against ``schema_path``.

        ``parsed`` is an already-in-memory representation the builder produced
        (for Annex 2, the ``lxml`` element tree). Passing it lets a validator
        skip a re-parse of a 200 MB file; a validator that cannot use it must
        ignore it and read from disk.

        Implementations must not raise for an *invalid* artefact — invalidity is
        a result, not an error. They may raise if the schema itself cannot be
        loaded, which is an infrastructure fault.
        """


__all__ = ["ArtefactValidator"]
