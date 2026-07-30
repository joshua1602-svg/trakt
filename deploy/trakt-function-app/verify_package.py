"""Verify the Azure Function App deployment package before it is deployed.

    python deploy/trakt-function-app/verify_package.py deploy.zip

Three checks, each guarding a failure mode that has actually reached production:

1. MANIFEST — every runtime path the entrypoint can import is in the archive,
   *including the ones reached only through lazy imports inside handlers*.

   `operations_control/` was missing from the zip while
   `apps/blob_trigger_app/occ_intake.py` imported it. Both hops are lazy:

       function_app.py:107   from apps.blob_trigger_app import occ_intake
       occ_intake.py:30-31   from operations_control.engine import OpsEngine

   so the host started, the trigger registered, and the omission only appeared on
   the first blob arrival.

2. REQUIREMENTS — every third-party module that closure imports is *declared* in
   the requirements.txt being shipped. The previous check pip-installed a
   hardcoded list, so a dependency missing from requirements.txt was invisible:
   CI installed it by name while the Function App installed only what
   requirements.txt declared.

3. IMPORT — unpack the real artefact and walk the lazy chain, not just
   `import function_app`. The module-level import succeeds even when
   operations_control is absent, which is precisely why this was not caught.

Exits non-zero with a named reason. Prints the pip requirement lines the caller
should install for check 3 to `--emit-requirements <path>` when asked.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import zipfile

#: Paths that must exist in the archive. Ordered entrypoint-first.
REQUIRED_PATHS = (
    "function_app.py",
    "host.json",
    "requirements.txt",
    "apps/blob_trigger_app/__init__.py",
    "apps/blob_trigger_app/occ_intake.py",
    "engine/__init__.py",
    "operations_control/__init__.py",
    "operations_control/engine.py",
    "operations_control/stores.py",
    "mi_agent_api/__init__.py",
)

#: Distribution name -> the module the traced closure imports. Established by
#: importing apps.blob_trigger_app.occ_intake plus operations_control.engine and
#: recording the third-party top-level modules that appear.
REQUIRED_DISTRIBUTIONS = {
    "azure-functions": "azure.functions",
    "PyYAML": "yaml",
    "pandas": "pandas",
    "numpy": "numpy",
    "openpyxl": "openpyxl",
}

#: The lazy chain to exercise inside the unpacked artefact.
IMPORT_PROBE = """
import sys
sys.path.insert(0, ".")
import function_app
print("  import function_app OK")
from apps.blob_trigger_app import occ_intake
assert hasattr(occ_intake, "handle_arrival"), "occ_intake.handle_arrival missing"
from operations_control.engine import OpsEngine
from operations_control.stores import OpsStore
print("  OCC intake chain OK:", OpsEngine.__name__, OpsStore.__name__)
"""


def fail(message: str) -> None:
    print(f"DEPLOY PACKAGE CHECK FAILED: {message}")
    sys.exit(1)


def check_manifest(zip_path: pathlib.Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    print(f"archive entries: {len(names)}")
    missing = [p for p in REQUIRED_PATHS if p not in names]
    for path in REQUIRED_PATHS:
        print(f"  {'PRESENT' if path in names else 'MISSING '}  {path}")
    if missing:
        fail("the archive is missing " + ", ".join(missing))
    return sorted(names)


def check_requirements(zip_path: pathlib.Path) -> list[str]:
    """Assert requirements.txt DECLARES every needed distribution, and return the
    exact requirement lines so the caller installs the declared versions."""
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read("requirements.txt").decode("utf-8", "replace")
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and not line.strip().startswith("#")]

    declared: list[str] = []
    missing: list[str] = []
    for dist, module in REQUIRED_DISTRIBUTIONS.items():
        pattern = rf"^{re.escape(dist)}\s*([<>=!~\[]|$)"
        hit = next((l for l in lines if re.match(pattern, l, re.IGNORECASE)), None)
        if hit:
            declared.append(hit)
            print(f"  declared  {hit}   (for `import {module}`)")
        else:
            missing.append(f"{dist} (needed for `import {module}`)")

    if missing:
        print("requirements.txt DOES NOT DECLARE:")
        for entry in missing:
            print("  -", entry)
        fail("the Function App installs only what requirements.txt declares, so "
             "the trigger would fail at runtime with ModuleNotFoundError")
    return declared


def check_imports(zip_path: pathlib.Path, workdir: pathlib.Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(workdir)
    probe = workdir / "_import_probe.py"
    probe.write_text(IMPORT_PROBE, encoding="utf-8")
    result = subprocess.run([sys.executable, probe.name], cwd=workdir,
                            capture_output=True, text=True)
    print(result.stdout.rstrip())
    if result.returncode != 0:
        print(result.stderr.rstrip())
        fail("the unpacked artefact cannot import the OCC intake chain")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("--emit-requirements",
                        help="write the declared requirement lines here")
    parser.add_argument("--skip-imports", action="store_true",
                        help="manifest and requirements only (no dependencies "
                             "installed yet)")
    parser.add_argument("--workdir", default="_pkgcheck")
    args = parser.parse_args()

    zip_path = pathlib.Path(args.zip_path)
    if not zip_path.is_file():
        fail(f"{zip_path} does not exist")

    print("=== 1. archive manifest ===")
    check_manifest(zip_path)

    print("\n=== 2. requirements declaration ===")
    declared = check_requirements(zip_path)
    if args.emit_requirements:
        pathlib.Path(args.emit_requirements).write_text(
            "\n".join(declared) + "\n", encoding="utf-8")
        print(f"  wrote {args.emit_requirements}")

    if args.skip_imports:
        print("\n(skipping the import check as asked)")
        print("\nPackage checks passed.")
        return

    print("\n=== 3. import the artefact the way the host does ===")
    check_imports(zip_path, pathlib.Path(args.workdir))

    print("\nPackage checks passed.")


if __name__ == "__main__":
    main()
