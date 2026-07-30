"""Verdicts on Kudu responses for the Function App post-deployment gates.

The bash side does the HTTP; this side does the parsing and the pass/fail
decision, so the decisions are unit-testable without an Azure subscription.

    python3 kudu_probe.py site-packages  --listing <vfs-listing.json>
    python3 kudu_probe.py command-result --json <kudu-command-response.json>

Both exit 0 on pass, 1 on fail, and print human-readable evidence either way.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

#: `import <module>` -> the distribution that provides it. The Function App
#: installs what requirements.txt declares, so both halves are checked: the
#: importable package directory AND the .dist-info that records a real install.
MODULE_DISTRIBUTIONS = {
    "yaml": "PyYAML",
    "pandas": "pandas",
    "numpy": "numpy",
    "openpyxl": "openpyxl",
}

DIST_INFO = re.compile(r"^(?P<name>.+?)-(?P<version>[^-]+)\.dist-info$")


def normalise(name: str) -> str:
    """PEP 503 style comparison: `PyYAML`, `pyyaml` and `py_yaml` are one name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_listing(raw: str) -> tuple[set[str], set[str], dict[str, str]]:
    """-> (directory names, all names, {normalised distribution: version})

    Kudu VFS returns a JSON array of entries; directories carry
    ``mime == "inode/directory"``.
    """
    entries = json.loads(raw)
    if not isinstance(entries, list):
        raise ValueError("expected a JSON array from the Kudu VFS listing")

    dirs, names, versions = set(), set(), {}
    for entry in entries:
        name = (entry.get("name") or "").rstrip("/")
        if not name:
            continue
        names.add(name)
        if entry.get("mime") == "inode/directory":
            dirs.add(name)
        match = DIST_INFO.match(name)
        if match:
            versions[normalise(match.group("name"))] = match.group("version")
    return dirs, names, versions


def check_site_packages(raw: str) -> int:
    try:
        dirs, names, versions = parse_listing(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: could not read the site-packages listing: {exc}")
        print("      first 400 characters of the response:")
        print("      " + raw[:400].replace("\n", "\n      "))
        return 1

    print(f"  entries in site-packages: {len(names)}")
    if not names:
        print()
        print("FAIL: site-packages is EMPTY. The remote build installed nothing.")
        return 1

    missing: list[str] = []
    for module, dist in sorted(MODULE_DISTRIBUTIONS.items()):
        has_dir = module in dirs
        version = versions.get(normalise(dist))
        if has_dir and version:
            print(f"  OK       {module:<10} <- {dist} {version}")
        elif has_dir:
            # Importable but with no install record. Treat as present and say so:
            # the worker only needs the package on sys.path.
            print(f"  OK       {module:<10} <- {dist} (directory present, "
                  f"no .dist-info recorded)")
        else:
            where = f"{dist} {version}" if version else dist
            print(f"  MISSING  {module:<10} <- {where}"
                  f"{' (.dist-info only, no package directory!)' if version else ''}")
            missing.append(module)

    if missing:
        print()
        print("FAIL: not importable on the deployed site: " + ", ".join(missing))
        return 1
    print("  every module the OCC intake imports is present on the deployed site")
    return 0


def check_command_result(raw: str) -> int:
    """Kudu /api/command returns {"Output": str, "Error": str, "ExitCode": int}."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"FAIL: the Kudu command response was not JSON: {exc}")
        print("      first 400 characters:")
        print("      " + raw[:400].replace("\n", "\n      "))
        return 1

    out = (payload.get("Output") or "").rstrip()
    err = (payload.get("Error") or "").rstrip()
    code = payload.get("ExitCode")

    if out:
        print("  --- stdout on the deployed site ---")
        print("\n".join("  " + line for line in out.splitlines()))
    if err:
        print("  --- stderr on the deployed site ---")
        print("\n".join("  " + line for line in err.splitlines()))
    print(f"  exit code: {code!r}")

    if code == 0:
        print()
        print("  PASS: the import chain that failed in production now succeeds "
              "on the deployed site.")
        return 0

    combined = f"{out}\n{err}"
    match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", combined)
    if match:
        print()
        print(f"FAIL: the deployed site still cannot import '{match.group(1)}'.")
        print("      This is the production failure, reproduced on the site "
              "itself — the")
        print("      dependencies are not installed where the worker looks for "
              "them.")
        return 1

    print()
    print("FAIL: the import probe exited non-zero on the deployed site "
          "(see stderr above).")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    sp = sub.add_parser("site-packages",
                        help="verdict on a Kudu VFS listing of site-packages")
    sp.add_argument("--listing", required=True)

    cr = sub.add_parser("command-result",
                        help="verdict on a Kudu /api/command response")
    cr.add_argument("--json", required=True, dest="json_path")

    args = parser.parse_args()
    if args.mode == "site-packages":
        sys.exit(check_site_packages(
            pathlib.Path(args.listing).read_text(encoding="utf-8", errors="replace")))
    sys.exit(check_command_result(
        pathlib.Path(args.json_path).read_text(encoding="utf-8", errors="replace")))


if __name__ == "__main__":
    main()
