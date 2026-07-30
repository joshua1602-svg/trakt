"""Read oryx-manifest.toml and say what build output should exist on the site.

    python3 oryx_manifest.py plan --manifest <file>

Emits `key=value` lines for the shell to consume:

    mode=compressed|virtualenv|unknown
    venv=<VirtualEnvName or empty>
    pkgdir=<packagedir or empty>
    candidate=<path>            (zero or more, in probe order)

WHY THIS EXISTS

The previous gate required `antenv/` or `__oryx_packages__/` to be readable over
Kudu VFS, and failed the job when neither was. That is wrong for the build mode
this service actually uses. The build log ends:

    --compress-destination-dir
    Copied the compressed output to '/home/site/wwwroot'
    Direct compression with zstd done
    Manifest file created
    Total execution done

In that mode the output is a single compressed artefact in /home/site/wwwroot and
the runtime extracts it at startup using oryx-manifest.toml. No expanded
`antenv/` or `__oryx_packages__/` directory is published, so requiring one turns
a successful build — 91 packages installed, including pyyaml==6.0.3, deployment
`complete=true`, `status=4` (Success) — into a reported failure.

So: keep asserting the manifest exists, then ask the manifest what to look for
rather than assuming the uncompressed layout.

KEY NAMES ARE NOT ASSUMED

Oryx writes `key="value"` lines and the exact key that records the compressed
artefact varies by Oryx version. Rather than hard-code one, any value that looks
like an archive is treated as a candidate, and compression is inferred from
either such a value or any `*compress*` key. Well-known names are probed as a
fallback so a manifest that records compression without naming the file still
resolves. Nothing here decides whether the deployment passes — the definitive
gate is the runtime health check.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

#: A value that names a build artefact rather than describing the build.
ARCHIVE = re.compile(r"\.(tar\.zst|tar\.gz|tar\.bz2|tgz|tar|zip|zst)$",
                     re.IGNORECASE)

#: Oryx/App Service names for a compressed destination directory, probed when the
#: manifest indicates compression without naming the file.
DEFAULT_COMPRESSED = ("output.tar.zst", "output.tar.gz", "output.tar")

TRUTHY = {"true", "1", "yes", "on", "enabled"}


def parse(text: str) -> dict[str, str]:
    """Oryx manifests are flat `key="value"` lines. Tolerate unquoted values,
    blank lines, comments and TOML section headers."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        out[key.strip()] = value
    return out


def get(fields: dict[str, str], *names: str) -> str:
    """Case-insensitive lookup; Oryx capitalisation varies between versions."""
    lowered = {k.lower(): v for k, v in fields.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value:
            return value
    return ""


def compression_indicated(fields: dict[str, str]) -> bool:
    for key, value in fields.items():
        if ARCHIVE.search(value or ""):
            return True
        if "compress" in key.lower():
            # A named file, or an explicit flag. An empty value says nothing.
            if value and (value.lower() in TRUTHY or ARCHIVE.search(value)):
                return True
    return False


def plan(text: str) -> tuple[str, str, str, list[str]]:
    """-> (mode, venv, pkgdir, candidates)"""
    fields = parse(text)
    venv = get(fields, "VirtualEnvName", "virtualenvname", "virtual_env_name")
    pkgdir = get(fields, "packagedir", "PackageDir", "package_dir")

    candidates: list[str] = []
    for key in sorted(fields):
        value = fields[key]
        if value and ARCHIVE.search(value) and value not in candidates:
            candidates.append(value)

    if compression_indicated(fields):
        mode = "compressed"
        for name in DEFAULT_COMPRESSED:
            if name not in candidates:
                candidates.append(name)
        if venv:
            for suffix in (".tar.gz", ".tar.zst"):
                name = f"{venv}{suffix}"
                if name not in candidates:
                    candidates.append(name)
    elif venv or pkgdir:
        mode = "virtualenv"
    else:
        mode = "unknown"
    return mode, venv, pkgdir, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan", help="what build output should exist")
    p.add_argument("--manifest", required=True)
    args = parser.parse_args()

    path = pathlib.Path(args.manifest)
    if not path.is_file():
        print("mode=unknown")
        print("venv=")
        print("pkgdir=")
        sys.exit(0)

    mode, venv, pkgdir, candidates = plan(
        path.read_text(encoding="utf-8", errors="replace"))
    print(f"mode={mode}")
    print(f"venv={venv}")
    print(f"pkgdir={pkgdir}")
    for candidate in candidates:
        print(f"candidate={candidate}")


if __name__ == "__main__":
    main()
