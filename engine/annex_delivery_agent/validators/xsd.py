"""
engine.annex_delivery_agent.validators.xsd
==========================================

``lxml`` XML-Schema validation — the same validator the proven Gate 5 route
runs, lifted into the shared interface without changing what it checks.

The Gate 5 builder validates with::

    schema = etree.XMLSchema(etree.parse(xsd_path))
    schema.validate(root)

This class does exactly that, and additionally captures the error log as
structured records instead of printing it, so a failure is persisted evidence
rather than console history.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..contracts import SchemaValidationResult, file_sha256
from .base import ArtefactValidator

#: Enough detail to locate and fix a problem; bounded so a systematically broken
#: 11k-record file does not persist a million-line error log.
MAX_RECORDED_ERRORS = 200


class LxmlXsdValidator(ArtefactValidator):
    """Validate an XML artefact against an XSD using ``lxml``.

    Compiled schemas are cached per ``(path, mtime, size)``: an Annex 2 XSD is
    835 KB and compiling it is not free, while a rerun of the same reporting
    period will use the same schema.
    """

    name = "lxml.etree.XMLSchema"

    _cache: Dict[Tuple[str, int, int], Any] = {}

    def validate(self, artefact_path: Path, *, schema_path: Path,
                 parsed: Optional[Any] = None) -> SchemaValidationResult:
        from lxml import etree

        schema_path = Path(schema_path)
        started = time.perf_counter()
        schema = self._schema_for(schema_path)

        # Prefer the in-memory tree the builder just produced. Re-reading a
        # 200 MB artefact to prove what we already hold would roughly double
        # peak memory for no additional assurance: the bytes on disk are
        # serialised from this exact tree, and the artefact hash records them.
        if parsed is not None:
            target = parsed
        else:
            target = etree.parse(str(artefact_path))

        valid = bool(schema.validate(target))
        errors: List[Dict[str, Any]] = []
        if not valid:
            for entry in schema.error_log:
                if len(errors) >= MAX_RECORDED_ERRORS:
                    break
                errors.append({
                    "line": getattr(entry, "line", None),
                    "column": getattr(entry, "column", None),
                    "domain": getattr(entry, "domain_name", None),
                    "type": getattr(entry, "type_name", None),
                    "path": getattr(entry, "path", None),
                    "message": str(getattr(entry, "message", "")),
                })

        total_errors = len(schema.error_log) if not valid else 0
        return SchemaValidationResult(
            validator=self.name,
            schema_reference=str(schema_path),
            schema_sha256=file_sha256(schema_path) if schema_path.is_file() else "",
            valid=valid,
            executed=True,
            error_count=total_errors,
            errors=tuple(errors),
            seconds=time.perf_counter() - started,
        )

    # -- internals ---------------------------------------------------------- #
    @classmethod
    def _schema_for(cls, schema_path: Path):
        from lxml import etree

        stat = schema_path.stat()
        key = (str(schema_path), int(stat.st_mtime), int(stat.st_size))
        schema = cls._cache.get(key)
        if schema is None:
            schema = etree.XMLSchema(etree.parse(str(schema_path)))
            cls._cache[key] = schema
        return schema

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()


__all__ = ["LxmlXsdValidator", "MAX_RECORDED_ERRORS"]
