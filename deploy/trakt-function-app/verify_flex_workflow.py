"""Repo invariants for the trakt-blob-trigger-v2 Flex Consumption deployment.

    python3 verify_flex_workflow.py workflow [--root .]
    python3 verify_flex_workflow.py package  <deploy.zip>

Both checks are static — no Azure, no network — so they run on every push.

WHY THESE TWO CHECKS

The `ModuleNotFoundError: No module named 'yaml'` in production was not an
application fault. Two deployment-architecture faults produced it:

1. TWO workflows deployed the same Function App on every push to main.
   `main_trakt.yml` requested a remote build (`az ... --build-remote true`);
   `main_trakt-blob-trigger-v2.yml` used Azure/functions-action@v1 without
   `remote-build`, whose declared default is 'false'. The two raced —
   "ERROR: Deployment was cancelled and another deployment is in progress." —
   and the local-build one won every time. The `workflow` check makes a second
   deploying workflow a build failure.

2. With `remote-build: false` the platform installs nothing, and the documented
   contract shifts to the package: "Dependencies must be installed into
   ./.python_packages/lib/site-packages". Nothing did that. The `package` check
   asserts the opposite invariant now that remote build is authoritative: the
   archive must NOT carry a half-built local environment, which would put a
   second, stale source of dependencies on the deployed site.

Sources: Azure/functions-action action.yml (`remote-build` default 'false');
learn.microsoft.com/azure/azure-functions/python-build-options;
learn.microsoft.com/azure/azure-functions/functions-how-to-github-actions
(`slot-name` is "Currently not supported in Flex Consumption"; `sku` is only
needed with publish-profile credentials, not RBAC).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import zipfile

import yaml

#: The one Function App this repo deploys, and the one workflow allowed to do it.
APP_NAME = "trakt-blob-trigger-v2"
SOLE_WORKFLOW = "main_trakt-blob-trigger-v2.yml"

#: Inputs that must be present, with the required value.
REQUIRED_INPUTS = {"remote-build": True}

#: Inputs that must be absent, with why. Setting any of these on a Flex
#: Consumption app is either unsupported or actively re-enables the failure.
FORBIDDEN_INPUTS = {
    "slot-name":
        "deployment slots are not supported in Flex Consumption",
    "scm-do-build-during-deployment":
        "a Dedicated/Elastic-Premium control; Microsoft's guidance for Flex is "
        "explicitly \"don't set scm-do-build-during-deployment or "
        "enable-oryx-build\"",
    "enable-oryx-build":
        "a Dedicated/Elastic-Premium control; an Oryx build is always performed "
        "during a remote build in Flex Consumption",
}

#: Paths that must not be in the archive: they are a local dependency build, and
#: remote build is now the single source of installed packages.
LOCAL_BUILD_MARKERS = (
    ".python_packages/",
    "venv/",
    ".venv/",
    "antenv/",
)


class Failure(Exception):
    pass


def load_workflows(root: pathlib.Path) -> dict[str, dict]:
    directory = root / ".github" / "workflows"
    if not directory.is_dir():
        raise Failure(f"{directory} does not exist")
    out = {}
    for path in sorted(directory.glob("*.yml")) + sorted(directory.glob("*.yaml")):
        try:
            out[path.name] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise Failure(f"{path.name} is not valid YAML: {exc}") from exc
    return out


def steps_of(document: dict) -> list[tuple[str, dict]]:
    """-> [(job name, step)] for every step in the document."""
    found = []
    for job_name, job in (document.get("jobs") or {}).items():
        for step in (job or {}).get("steps") or []:
            if isinstance(step, dict):
                found.append((job_name, step))
    return found


def deploying_steps(document: dict) -> list[tuple[str, dict, str]]:
    """Steps that deploy APP_NAME, by either supported mechanism.

    Both are recognised on purpose: the whole failure was one workflow using the
    action and another using the CLI, so a check that only understood one of them
    would have seen a single deployer and passed.
    """
    hits = []
    for job_name, step in steps_of(document):
        uses = str(step.get("uses") or "")
        if uses.lower().startswith("azure/functions-action"):
            if str((step.get("with") or {}).get("app-name") or "") == APP_NAME:
                hits.append((job_name, step, "Azure/functions-action"))
            continue
        run = str(step.get("run") or "")
        if "functionapp deployment" in run and APP_NAME in run:
            hits.append((job_name, step, "az functionapp deployment"))
    return hits


def check_workflow(root: pathlib.Path) -> list[str]:
    problems: list[str] = []
    workflows = load_workflows(root)
    print(f"workflows scanned: {len(workflows)}")

    deployers = {name: deploying_steps(doc) for name, doc in workflows.items()}
    deployers = {name: hits for name, hits in deployers.items() if hits}

    for name, hits in sorted(deployers.items()):
        for _job, _step, mechanism in hits:
            print(f"  deploys {APP_NAME}:  {name}  via {mechanism}")
    if not deployers:
        problems.append(f"no workflow deploys {APP_NAME} at all")
        return problems

    extra = sorted(set(deployers) - {SOLE_WORKFLOW})
    if extra:
        problems.append(
            f"{len(deployers)} workflows deploy {APP_NAME}: "
            + ", ".join(sorted(deployers))
            + f". Exactly one may — {SOLE_WORKFLOW}. Concurrent deployments of "
              "one app race, and the loser reports 'Deployment was cancelled "
              "and another deployment is in progress.' Remove the deploy step "
              f"from: {', '.join(extra)}")
    if SOLE_WORKFLOW not in deployers:
        problems.append(f"{SOLE_WORKFLOW} does not deploy {APP_NAME}")
        return problems

    hits = deployers[SOLE_WORKFLOW]
    if len(hits) != 1:
        problems.append(
            f"{SOLE_WORKFLOW} has {len(hits)} steps deploying {APP_NAME}; "
            "expected exactly 1")
    for _job, step, mechanism in hits:
        if mechanism != "Azure/functions-action":
            problems.append(
                f"{SOLE_WORKFLOW} deploys via {mechanism}. One deploy is the "
                "only deployment technology supported on Flex Consumption, and "
                "the action selects it automatically; keep the action.")
            continue
        inputs = step.get("with") or {}
        for key, want in REQUIRED_INPUTS.items():
            got = inputs.get(key, "<unset>")
            if got is want or str(got).lower() == str(want).lower():
                print(f"  OK       {key}: {got}")
            else:
                problems.append(
                    f"{SOLE_WORKFLOW} sets {key}={got!r}, must be {want!r}. "
                    "Without it the platform installs nothing from "
                    "requirements.txt and the trigger dies on its first lazy "
                    "import — the exact production failure.")
        for key, why in FORBIDDEN_INPUTS.items():
            if key in inputs:
                problems.append(
                    f"{SOLE_WORKFLOW} sets {key}={inputs[key]!r}: {why}")
            else:
                print(f"  OK       {key} not set ({why.split(';')[0]})")
    return problems


def check_package(zip_path: pathlib.Path) -> list[str]:
    if not zip_path.is_file():
        return [f"{zip_path} does not exist"]
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    print(f"archive entries: {len(names)}")

    problems = []
    for marker in LOCAL_BUILD_MARKERS:
        hits = [n for n in names if n.startswith(marker) or f"/{marker}" in n]
        if hits:
            problems.append(
                f"the archive contains {len(hits)} entries under {marker!r} "
                f"(e.g. {hits[0]}). Remote build installs dependencies into "
                "./.python_packages/lib/site-packages on the platform; shipping "
                "a local build puts a second, stale copy on the site. Remove it "
                "from the packaging step.")
        else:
            print(f"  OK       no {marker} in the archive")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    wf = sub.add_parser("workflow", help="exactly one workflow deploys the app, "
                                        "with Flex-correct inputs")
    wf.add_argument("--root", default=".")
    pkg = sub.add_parser("package", help="the archive carries no local build")
    pkg.add_argument("zip_path")

    args = parser.parse_args()
    try:
        if args.mode == "workflow":
            print("=== deployment path (static) ===")
            problems = check_workflow(pathlib.Path(args.root))
        else:
            print("=== package carries no local dependency build ===")
            problems = check_package(pathlib.Path(args.zip_path))
    except Failure as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)

    if problems:
        print()
        for problem in problems:
            print(f"FAIL: {problem}")
        sys.exit(1)
    print("  checks passed")


if __name__ == "__main__":
    main()
