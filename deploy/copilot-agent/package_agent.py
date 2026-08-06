#!/usr/bin/env python3
"""package_agent.py — build the sideloadable Trakt Copilot agent package.

Zips the declarative-agent artefacts in this directory (Teams app manifest,
declarative agent, API plugin, OpenAPI spec) plus generated placeholder icons
into ``dist/trakt-copilot-agent.zip``, ready for upload to Microsoft 365 admin
center / Teams "Upload a custom app".

Before packaging, edit:
  * ``manifest.json``  — set ``id`` to a fresh GUID and the real host in
    ``developer.*Url`` / ``validDomains``;
  * ``trakt-copilot-openapi.yaml`` — set ``servers[0].url`` to the deployed
    Trakt MI API and the real app id in the OAuth scope;
  * ``ai-plugin.json`` — set the OAuth registration id from the Teams developer
    portal (replaces ``${{OAUTH2_CONFIGURATION_ID}}``), or leave the token if
    your toolchain substitutes it at provisioning time.

Usage:
    python deploy/copilot-agent/package_agent.py [--out DIST_DIR]

No third-party dependencies: icons are written as minimal solid-colour PNGs via
zlib/struct so the package passes manifest validation without shipping binary
assets in the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import zipfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent

PACKAGE_FILES = [
    "manifest.json",
    "declarativeAgent.json",
    "ai-plugin.json",
    "trakt-copilot-openapi.yaml",
]

ACCENT_RGB = (0x1F, 0x3A, 0x5F)  # matches manifest accentColor


def _png(width: int, height: int, rgba: tuple[int, int, int, int]) -> bytes:
    """A minimal valid RGBA PNG of one solid colour."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    row = b"\x00" + bytes(rgba) * width           # filter 0 + pixels
    raw = row * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


#: Per-deployment token, substituted by the provisioning toolchain exactly as
#: ``${{OAUTH2_CONFIGURATION_ID}}`` already is in ``ai-plugin.json``. It is
#: therefore expected in the repository copy of the manifest, and only a
#: RELEASE build (``--require-resolved``) insists it has been replaced — a
#: manifest uploaded with the literal token installs cleanly and then fails
#: every proactive send, days later, in production.
_PLACEHOLDER = "${{TEAMS_BOT_APP_ID}}"


def _validate_manifest(manifest: dict, *, require_resolved: bool = False) -> None:
    """Check the capabilities this package is required to carry.

    The declarative agent and the bot are separate capabilities of ONE app.
    Losing either during an edit is silent — the package still validates and
    installs — so both are asserted here rather than trusted.
    """
    agents = (manifest.get("copilotAgents") or {}).get("declarativeAgents") or []
    if not agents:
        raise SystemExit(
            "manifest.json no longer declares a declarative agent: the "
            "existing Copilot capability must be preserved")

    bots = manifest.get("bots") or []
    if not bots:
        raise SystemExit(
            "manifest.json declares no bot: proactive notifications require "
            "the bot capability in this same package")
    bot = bots[0]
    if "personal" not in (bot.get("scopes") or []):
        raise SystemExit(
            "the bot must declare the 'personal' scope: v1 delivers to 1:1 "
            "chats only")
    for unsupported in ("team", "groupChat", "groupchat"):
        if unsupported in (bot.get("scopes") or []):
            raise SystemExit(
                f"the bot declares the {unsupported!r} scope, which v1 does "
                f"not implement")

    if not require_resolved:
        return
    blob = json.dumps(manifest)
    if _PLACEHOLDER in blob:
        raise SystemExit(
            f"{_PLACEHOLDER} is unresolved in manifest.json. Set the bot app "
            f"id before building a release package.")


#: A bot app id is an Entra application (client) id — a GUID. Validated before
#: substitution because a typo produces a package that installs cleanly and
#: then fails every proactive send against an app id that does not exist.
_GUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

#: Environment fallback, so CI can supply the id without it appearing in a
#: command line (and therefore in a build log).
BOT_APP_ID_ENV = "TEAMS_BOT_APP_ID"


def resolve_bot_app_id(explicit: str | None = None) -> str | None:
    """The bot app id to substitute, from the flag or the environment.

    Returns ``None`` when neither is set, which leaves the token in place — a
    development build stays possible without the real id.
    """
    value = (explicit or os.environ.get(BOT_APP_ID_ENV) or "").strip()
    if not value:
        return None
    if not _GUID_RE.match(value):
        raise SystemExit(
            f"{value!r} is not a valid Entra application (client) id. Expected "
            f"a GUID like 00000000-0000-0000-0000-000000000000.")
    return value


def substitute(text: str, bot_app_id: str | None) -> str:
    """Replace the bot app id token in a package file's TEXT.

    Substitution happens on the way into the archive; the repository copy is
    never rewritten. That is the point — the id belongs to a deployment, not to
    the source tree, and a build must not leave the working copy dirty.
    """
    if not bot_app_id:
        return text
    return text.replace(_PLACEHOLDER, bot_app_id)


def build(out_dir: Path, *, require_resolved: bool = False,
          bot_app_id: str | None = None) -> Path:
    """Build the package, substituting per-deployment tokens on the way in."""
    rendered: dict[str, str] = {}
    for name in PACKAGE_FILES:
        path = HERE / name
        if not path.exists():
            raise SystemExit(f"missing package file: {path}")
        text = substitute(path.read_text(encoding="utf-8"), bot_app_id)
        rendered[name] = text
        if name.endswith(".json"):
            data = json.loads(text)  # bad JSON → fail fast
            if name == "manifest.json":
                # Validated AFTER substitution, so --require-resolved checks
                # what will actually ship rather than what is in the repository.
                _validate_manifest(data, require_resolved=require_resolved)

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "trakt-copilot-agent.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in PACKAGE_FILES:
            zf.writestr(name, rendered[name])
        zf.writestr("color.png", _png(192, 192, (*ACCENT_RGB, 255)))
        zf.writestr("outline.png", _png(32, 32, (255, 255, 255, 255)))
    return zip_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "dist"),
                    help="output directory (default: deploy/copilot-agent/dist)")
    ap.add_argument("--require-resolved", action="store_true",
                    help="fail if a ${{...}} token is still unresolved — use "
                         "for a release build, where the provisioning "
                         "toolchain has already substituted them")
    ap.add_argument("--bot-app-id", default=None,
                    help=f"Entra application (client) id of the Teams bot, "
                         f"substituted into the packaged manifest. Falls back "
                         f"to ${BOT_APP_ID_ENV}. The repository copy of "
                         f"manifest.json is never rewritten.")
    args = ap.parse_args(argv)
    bot_app_id = resolve_bot_app_id(args.bot_app_id)
    zip_path = build(Path(args.out), require_resolved=args.require_resolved,
                     bot_app_id=bot_app_id)
    print(f"wrote {zip_path}")
    if bot_app_id:
        # The bot app id is public (it ships inside the manifest), so echoing it
        # is safe and lets the operator confirm the package carries the id they
        # intended. The client SECRET is never handled here at all.
        print(f"bot app id substituted: {bot_app_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
