#!/usr/bin/env python3
"""Generate the agent tool API's OpenAPI document from the tool registry.

    python scripts/build_agent_openapi.py            # write the document
    python scripts/build_agent_openapi.py --check    # fail if it is stale

Written to ``deploy/agent-api/trakt-agent-openapi.yaml``.

Why generated rather than hand-authored
---------------------------------------
``deploy/copilot-agent/trakt-copilot-openapi.yaml`` is hand-authored, and a test
keeps it in lock-step with the routes. That works for three fixed actions. The
agent surface grows a route per tool, so the document is derived instead and
``tests/test_agent_openapi_document.py`` asserts it matches the registry — the
same guarantee, maintained by construction rather than by discipline.

FastAPI's own ``/openapi.json`` is deliberately NOT used: it is 3.1, it describes
whichever routes the process happens to have mounted, and it would expose the
entire internal API. This document describes the closed tool surface and nothing
else, at 3.0.3, which is the version most agent tool-loaders consume.

The published `servers` URL is a placeholder. Set it to the deployment's base URL
before handing the document to a client.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import trakt_tools
from trakt_core.envelope import SCHEMA_VERSION as ENVELOPE_SCHEMA_VERSION

OUTPUT_PATH = _REPO_ROOT / "deploy" / "agent-api" / "trakt-agent-openapi.yaml"

#: Bumped when the SURFACE changes (a tool added, removed, or its schema
#: changed). Individual tools carry their own version in the registry.
API_VERSION = "1.0.0"

PLACEHOLDER_SERVER = "https://trakt-mi-api.example.azurewebsites.net"

_ERROR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "A governed failure. Branch on `code`, never on `message`.",
    "properties": {
        "code": {"type": "string",
                 "description": "Stable machine-readable error code."},
        "message": {"type": "string"},
        "category": {"type": "string",
                     "enum": ["authentication", "authorisation", "input",
                              "policy", "data", "capability", "infrastructure"]},
        "retryable": {"type": "boolean",
                      "description": "Whether waiting and retrying could help."},
        "request_id": {"type": "string"},
        "details": {"type": "object"},
    },
    "required": ["code", "message", "category", "retryable"],
}

_GOVERNED_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "The stable envelope every tool returns. `result` carries the tool's own "
        "output; everything else is the governance context needed to trust it."),
    "properties": {
        "capability": {"type": "string", "example": "tool.evaluate_covenants"},
        "schema_version": {"type": "string", "example": ENVELOPE_SCHEMA_VERSION},
        "status": {"type": "string",
                   "enum": ["success", "partial_success", "blocked", "error"],
                   "description": (
                       "`blocked` means governance refused (permission or "
                       "policy). `error` means the capability could not produce "
                       "an answer. The two are different situations for an "
                       "autonomous caller: do not retry a `blocked` call.")},
        "request_id": {"type": "string"},
        "correlation_id": {"type": ["string", "null"],
                           "description": "Echoes the inbound X-Correlation-Id."},
        "tenant_id": {"type": "string"},
        "portfolio_id": {"type": ["string", "null"],
                         "description": "The resource reference this call named."},
        "snapshot": {"type": ["object", "null"],
                     "description": "Which governed dataset answered: label, "
                                    "snapshot id, content hash, reporting date."},
        "result": {"type": ["object", "null"]},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "policy": {"type": "object",
                   "description": "What governance decided."},
        "provenance": {"type": ["object", "null"]},
        "audit": {"type": ["object", "null"]},
        "error": {"oneOf": [_ERROR_SCHEMA, {"type": "null"}]},
    },
    "required": ["capability", "schema_version", "status", "request_id",
                 "tenant_id"],
}


def build_document() -> Dict[str, Any]:
    tools = trakt_tools.all_tools()

    paths: Dict[str, Any] = {
        "/v1/agent/tools": {
            "get": {
                "operationId": "listTraktTools",
                "summary": "List the Trakt tools this identity may call",
                "description": (
                    "Returns the tool catalogue narrowed to the capabilities this "
                    "agent's grants confer, each with its JSON Schema, plus the "
                    "closed set of resource references it may name. Call this "
                    "first: it removes any need to guess an identifier."),
                "responses": {
                    "200": {
                        "description": "The authorised tool catalogue.",
                        "content": {"application/json": {"schema": {
                            "type": "object",
                            "properties": {
                                "tools": {"type": "array",
                                          "items": {"$ref": "#/components/schemas/ToolDeclaration"}},
                                "tool_count": {"type": "integer"},
                                "resources": {
                                    "type": "object",
                                    "description": (
                                        "capability -> the resource references "
                                        "this agent may use it against.")},
                                "organisation_id": {"type": ["string", "null"]},
                                "tenant_id": {"type": "string"},
                                "request_id": {"type": "string"},
                            }}}},
                    },
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "403": {"$ref": "#/components/responses/Forbidden"},
                    "503": {"$ref": "#/components/responses/Unavailable"},
                },
            }
        }
    }

    for spec in tools:
        entry = spec.to_catalogue_entry()
        paths[f"/v1/agent/tools/{spec.name}"] = {
            "post": {
                "operationId": f"invoke_{spec.name}",
                "summary": spec.description.split(".")[0],
                "description": (
                    f"{spec.description}\n\n{spec.agent_guidance}\n\n"
                    f"Tool version {spec.version}. Requires the "
                    f"`{spec.required_capability}` capability against the named "
                    f"resource."
                ).strip(),
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": entry["input_schema"]}},
                },
                "responses": {
                    "200": {
                        "description": "The governed result.",
                        "content": {"application/json": {"schema": {
                            "$ref": "#/components/schemas/GovernedResult"}}},
                    },
                    "400": {"$ref": "#/components/responses/InvalidArguments"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "403": {"$ref": "#/components/responses/Forbidden"},
                    "404": {"$ref": "#/components/responses/UnknownTool"},
                    "503": {"$ref": "#/components/responses/Unavailable"},
                },
            }
        }

    error_response = {
        "content": {"application/json": {"schema": {
            "$ref": "#/components/schemas/GovernedResult"}}}}

    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Trakt Agent Tools",
            "version": API_VERSION,
            "description": (
                "Governed, deterministic credit capabilities for an external AI "
                "agent.\n\n"
                "Every figure these tools return is computed by Trakt from "
                "governed data under an operator-approved definition, and carries "
                "the snapshot and configuration provenance needed to trust it. "
                "Never compute a portfolio figure yourself — call a tool.\n\n"
                "Authorisation is per resource. An agent holds capabilities "
                "against named resources and nothing else; a resource it is not "
                "granted is refused identically to one that does not exist."),
        },
        "servers": [{"url": PLACEHOLDER_SERVER,
                     "description": "Set to your deployment URL before use."}],
        "security": [{"entraBearer": []}],
        "paths": paths,
        "components": {
            "securitySchemes": {
                "entraBearer": {
                    "type": "http", "scheme": "bearer", "bearerFormat": "JWT",
                    "description": (
                        "Microsoft Entra client-credentials token. The calling "
                        "application's service principal must be assigned the "
                        "Trakt agent app role, and its directory must be "
                        "registered as a Trakt organisation."),
                }
            },
            "schemas": {
                "GovernedResult": _GOVERNED_RESULT_SCHEMA,
                "TraktError": _ERROR_SCHEMA,
                "ToolDeclaration": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "version": {"type": "string"},
                        "description": {"type": "string"},
                        "agent_guidance": {"type": "string"},
                        "required_capability": {"type": "string"},
                        "resource_argument": {"type": "string"},
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                    },
                    "required": ["name", "version", "description",
                                 "required_capability", "input_schema"],
                },
            },
            "responses": {
                "InvalidArguments": {
                    "description": ("TOOL_INPUT_INVALID — `error.details.violations` "
                                    "names the failing arguments."),
                    **error_response},
                "Unauthorized": {
                    "description": "The bearer token is missing, invalid or expired.",
                    **error_response},
                "Forbidden": {
                    "description": (
                        "SCOPE_MISSING, RESOURCE_NOT_AUTHORISED, "
                        "CAPABILITY_NOT_GRANTED, TENANT_MISMATCH or "
                        "RESOURCE_NOT_PARTITIONABLE. A permission decision — do "
                        "not retry."),
                    **error_response},
                "UnknownTool": {
                    "description": ("TOOL_NOT_FOUND — `error.details.available_tools` "
                                    "lists what this deployment serves."),
                    **error_response},
                "Unavailable": {
                    "description": ("The agent API or the governed data source is "
                                    "not available. Retryable."),
                    **error_response},
            },
        },
    }


def render(document: Dict[str, Any]) -> str:
    import yaml

    header = (
        "# GENERATED FILE — do not edit by hand.\n"
        "#\n"
        "# Built from the Trakt agent tool registry by\n"
        "# scripts/build_agent_openapi.py. Add or change a tool in\n"
        "# trakt_tools/handlers/, then regenerate.\n"
        "#\n"
        "# tests/test_agent_openapi_document.py fails when this file is stale.\n"
        "#\n"
        "# Set servers[0].url to the deployment base URL before handing this to a\n"
        "# client.\n"
    )
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True,
                          default_flow_style=False, width=100)
    return header + body


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if the written document is stale.")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)

    rendered = render(build_document())
    path = Path(args.output)

    if args.check:
        if not path.exists():
            print(f"{path} does not exist; run scripts/build_agent_openapi.py",
                  file=sys.stderr)
            return 1
        if path.read_text(encoding="utf-8") != rendered:
            print(f"{path} is stale; run scripts/build_agent_openapi.py",
                  file=sys.stderr)
            return 1
        print(f"{path} is up to date.")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {path} ({len(trakt_tools.all_tools())} tool(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
