"""trakt_tools.schema — the JSON Schema subset the tool contract is written in.

A tool's input and output schemas are the *external* contract: they are what
``GET /v1/agent/tools`` publishes, what the generated OpenAPI document contains,
and what an MCP adapter would translate into a tool declaration. They therefore
have to be plain JSON Schema, and they have to be validated identically wherever
they are checked.

Why a validator here rather than pydantic or jsonschema:

* ``trakt_tools`` must stay importable without a web framework, for the same
  reason ``trakt_core`` does — a governed tool is called from a route, a job, a
  test and (later) an MCP server, and none of those should drag in FastAPI;
* the tool contract is small and deliberately closed. The subset below is all a
  typed credit tool needs, and a closed subset is one an agent can be *told*
  about honestly;
* no new runtime dependency, so this module works anywhere ``trakt_core`` does.

Supported keywords: ``type`` (object / string / number / integer / boolean /
array / null, or a list of those), ``properties``, ``required``,
``additionalProperties`` (boolean only), ``enum``, ``pattern``, ``minLength`` /
``maxLength``, ``minimum`` / ``maximum``, ``items``, ``minItems`` / ``maxItems``,
``default``, ``const``. Anything else in a schema is ignored rather than
silently treated as satisfied — see :func:`assert_supported_schema`, which the
registry runs at registration time so an unsupported keyword is a startup
failure rather than a control that quietly does nothing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Keywords this validator understands. A schema using anything else is rejected
#: at registration: a keyword that is present but unenforced is worse than one
#: that is absent, because the published contract would promise a constraint the
#: server does not apply.
SUPPORTED_KEYWORDS = frozenset({
    "type", "title", "description", "properties", "required",
    "additionalProperties", "enum", "const", "pattern", "minLength",
    "maxLength", "minimum", "maximum", "items", "minItems", "maxItems",
    "default", "examples",
})

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, Mapping),
    "array": lambda v: isinstance(v, (list, tuple)),
    "string": lambda v: isinstance(v, str),
    # bool is a subclass of int in Python; a boolean is never a number here.
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


class SchemaError(ValueError):
    """A schema itself is malformed or uses an unsupported keyword."""


def assert_supported_schema(schema: Mapping[str, Any], *, path: str = "") -> None:
    """Raise :class:`SchemaError` if ``schema`` uses anything unenforced.

    Run at tool-registration time. The point is that a tool cannot publish a
    constraint the executor does not check.
    """
    if not isinstance(schema, Mapping):
        raise SchemaError(f"{path or 'schema'} must be an object")
    unsupported = sorted(set(schema) - SUPPORTED_KEYWORDS)
    if unsupported:
        raise SchemaError(
            f"{path or 'schema'} uses unsupported schema keywords "
            f"{unsupported}; trakt_tools.schema would not enforce them")
    declared = schema.get("type")
    types = [declared] if isinstance(declared, str) else list(declared or ())
    for t in types:
        if t not in _TYPE_CHECKS:
            raise SchemaError(f"{path or 'schema'} declares unknown type {t!r}")
    for name, sub in (schema.get("properties") or {}).items():
        assert_supported_schema(sub, path=f"{path}.{name}" if path else name)
    items = schema.get("items")
    if items is not None:
        assert_supported_schema(items, path=f"{path}[]" if path else "[]")
    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool):
        raise SchemaError(
            f"{path or 'schema'} additionalProperties must be true or false; "
            "sub-schemas are not supported")


def _type_ok(value: Any, declared: Any) -> bool:
    if declared is None:
        return True
    types = [declared] if isinstance(declared, str) else list(declared)
    return any(_TYPE_CHECKS[t](value) for t in types)


def _validate(value: Any, schema: Mapping[str, Any], path: str,
              errors: List[str]) -> Any:
    """Validate ``value``, returning it with any object defaults applied."""
    declared = schema.get("type")
    if not _type_ok(value, declared):
        expected = declared if isinstance(declared, str) else "/".join(declared or ())
        errors.append(f"{path or '(root)'}: expected {expected}, "
                      f"got {type(value).__name__}")
        return value

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path or '(root)'}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path or '(root)'}: must be one of "
                      f"{list(schema['enum'])!r}")

    if isinstance(value, str):
        pattern = schema.get("pattern")
        if pattern and not re.search(pattern, value):
            errors.append(f"{path or '(root)'}: does not match {pattern!r}")
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path or '(root)'}: shorter than "
                          f"{schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path or '(root)'}: longer than "
                          f"{schema['maxLength']} characters")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path or '(root)'}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path or '(root)'}: above maximum {schema['maximum']}")

    if isinstance(value, (list, tuple)):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path or '(root)'}: fewer than "
                          f"{schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path or '(root)'}: more than "
                          f"{schema['maxItems']} items")
        item_schema = schema.get("items")
        if item_schema is not None:
            return [
                _validate(item, item_schema, f"{path}[{i}]", errors)
                for i, item in enumerate(value)
            ]
        return list(value)

    if isinstance(value, Mapping):
        properties: Dict[str, Any] = dict(schema.get("properties") or {})
        required: Sequence[str] = schema.get("required") or ()
        out: Dict[str, Any] = {}

        for name in required:
            if name not in value:
                errors.append(f"{path + '.' if path else ''}{name}: required")

        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(
                        f"{path + '.' if path else ''}{name}: not a recognised "
                        "argument")

        for name, sub in properties.items():
            child_path = f"{path}.{name}" if path else name
            if name in value:
                out[name] = _validate(value[name], sub, child_path, errors)
            elif "default" in sub:
                # Applied server-side so an agent may omit optional arguments and
                # still get the documented behaviour, and so the handler never has
                # to re-decide a default the schema already published.
                out[name] = sub["default"]
        for name, item in value.items():
            if name not in properties:
                out[name] = item
        return out

    return value


def validate(value: Any, schema: Mapping[str, Any]) -> Tuple[Any, List[str]]:
    """``(normalised_value, errors)``. Never raises for invalid *data*.

    ``normalised_value`` is the input with schema ``default``\\ s filled in. When
    ``errors`` is non-empty the value must not be used — the caller turns the
    list into a typed ``TOOL_INPUT_INVALID``.
    """
    errors: List[str] = []
    normalised = _validate(value, schema, "", errors)
    return normalised, errors


def object_schema(
    *,
    description: str,
    properties: Dict[str, Any],
    required: Optional[Sequence[str]] = None,
    additional_properties: bool = False,
) -> Dict[str, Any]:
    """A closed object schema. Closed by default on purpose: an unrecognised
    argument from an agent is a mistake worth reporting, not something to
    ignore."""
    schema: Dict[str, Any] = {
        "type": "object",
        "description": description,
        "properties": dict(properties),
        "additionalProperties": bool(additional_properties),
    }
    if required:
        schema["required"] = list(required)
    return schema
