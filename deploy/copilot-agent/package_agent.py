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


def build(out_dir: Path) -> Path:
    for name in PACKAGE_FILES:
        path = HERE / name
        if not path.exists():
            raise SystemExit(f"missing package file: {path}")
        if name.endswith(".json"):
            json.loads(path.read_text(encoding="utf-8"))  # fail fast on bad JSON

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "trakt-copilot-agent.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in PACKAGE_FILES:
            zf.write(HERE / name, arcname=name)
        zf.writestr("color.png", _png(192, 192, (*ACCENT_RGB, 255)))
        zf.writestr("outline.png", _png(32, 32, (255, 255, 255, 255)))
    return zip_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "dist"),
                    help="output directory (default: deploy/copilot-agent/dist)")
    args = ap.parse_args(argv)
    zip_path = build(Path(args.out))
    print(f"wrote {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
