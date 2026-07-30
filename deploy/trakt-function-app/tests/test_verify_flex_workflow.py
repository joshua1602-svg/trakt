"""Tests for the static Flex Consumption deployment invariants.

Runnable with pytest, or standalone (`python3 test_verify_flex_workflow.py`).

The cases are written from the real failure: two workflows deploying one app,
one of them via the CLI and one via the action, with remote-build defaulted to
false. A check that missed any of those would have passed the broken repo.
"""

from __future__ import annotations

import io
import pathlib
import sys
import zipfile
from contextlib import redirect_stdout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import verify_flex_workflow as vfw  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

GOOD_ACTION_STEP = """
      - name: Deploy
        uses: Azure/functions-action@v1
        with:
          app-name: 'trakt-blob-trigger-v2'
          package: 'deploy.zip'
          remote-build: true
"""


def write_workflows(tmp: pathlib.Path, files: dict[str, str]) -> pathlib.Path:
    directory = tmp / ".github" / "workflows"
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (directory / name).write_text(body, encoding="utf-8")
    return tmp


def workflow_yaml(step: str = GOOD_ACTION_STEP) -> str:
    return f"""
name: deploy
on:
  push:
    branches: [main]
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
{step}
"""


def run_workflow(tmp: pathlib.Path) -> tuple[list[str], str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        problems = vfw.check_workflow(tmp)
    return problems, buf.getvalue()


def run_package(zip_path: pathlib.Path) -> tuple[list[str], str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        problems = vfw.check_package(zip_path)
    return problems, buf.getvalue()


def make_zip(path: pathlib.Path, names: list[str]) -> pathlib.Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, b"x")
    return path


# --------------------------------------------------------------------------
# workflow mode
# --------------------------------------------------------------------------

def test_the_real_repository_satisfies_the_invariant():
    """The check must pass on this repo as committed, or it is not a gate."""
    problems, out = run_workflow(REPO_ROOT)
    assert problems == [], f"{problems}\n{out}"
    assert "main_trakt-blob-trigger-v2.yml  via Azure/functions-action" in out


def test_single_action_workflow_with_remote_build_passes(tmp_path):
    root = write_workflows(tmp_path, {vfw.SOLE_WORKFLOW: workflow_yaml()})
    problems, out = run_workflow(root)
    assert problems == [], f"{problems}\n{out}"
    assert "OK       remote-build: True" in out


def test_remote_build_missing_is_the_production_failure(tmp_path):
    step = GOOD_ACTION_STEP.replace("          remote-build: true\n", "")
    root = write_workflows(tmp_path, {vfw.SOLE_WORKFLOW: workflow_yaml(step)})
    problems, _ = run_workflow(root)
    assert len(problems) == 1
    assert "remote-build='<unset>'" in problems[0]
    assert "installs nothing from requirements.txt" in problems[0]


def test_remote_build_false_is_rejected(tmp_path):
    step = GOOD_ACTION_STEP.replace("remote-build: true", "remote-build: false")
    root = write_workflows(tmp_path, {vfw.SOLE_WORKFLOW: workflow_yaml(step)})
    problems, _ = run_workflow(root)
    assert any("must be True" in p for p in problems)


def test_remote_build_as_the_string_true_is_accepted(tmp_path):
    """YAML quoting must not decide whether the gate passes."""
    step = GOOD_ACTION_STEP.replace("remote-build: true", "remote-build: 'true'")
    root = write_workflows(tmp_path, {vfw.SOLE_WORKFLOW: workflow_yaml(step)})
    problems, _ = run_workflow(root)
    assert problems == []


def test_a_second_workflow_deploying_via_the_cli_is_caught(tmp_path):
    """Exactly the retired main_trakt.yml. The two mechanisms differ, so a check
    that understood only the action would have seen one deployer and passed."""
    cli = workflow_yaml("""
      - name: Deploy
        run: |
          az functionapp deployment source config-zip \\
            --resource-group trakt --name trakt-blob-trigger-v2 \\
            --src deploy.zip --build-remote true
""")
    root = write_workflows(tmp_path, {vfw.SOLE_WORKFLOW: workflow_yaml(),
                                      "main_trakt.yml": cli})
    problems, out = run_workflow(root)
    assert any("2 workflows deploy" in p for p in problems)
    assert any("Remove the deploy step from: main_trakt.yml" in p
               for p in problems)
    assert "via az functionapp deployment" in out


def test_a_second_workflow_deploying_via_the_action_is_caught(tmp_path):
    root = write_workflows(tmp_path, {vfw.SOLE_WORKFLOW: workflow_yaml(),
                                      "other.yml": workflow_yaml()})
    problems, _ = run_workflow(root)
    assert any("2 workflows deploy" in p for p in problems)


def test_a_workflow_deploying_a_different_app_is_ignored(tmp_path):
    other = workflow_yaml("""
      - name: Deploy
        uses: Azure/functions-action@v1
        with:
          app-name: 'some-other-app'
          remote-build: true
""")
    root = write_workflows(tmp_path, {vfw.SOLE_WORKFLOW: workflow_yaml(),
                                      "other.yml": other})
    problems, _ = run_workflow(root)
    assert problems == []


def test_slot_name_is_rejected_as_unsupported_on_flex(tmp_path):
    step = GOOD_ACTION_STEP + "          slot-name: 'Production'\n"
    root = write_workflows(tmp_path, {vfw.SOLE_WORKFLOW: workflow_yaml(step)})
    problems, _ = run_workflow(root)
    assert any("slot-name" in p and "not supported in Flex" in p
               for p in problems)


def test_dedicated_plan_build_inputs_are_rejected(tmp_path):
    step = (GOOD_ACTION_STEP
            + "          scm-do-build-during-deployment: true\n"
              "          enable-oryx-build: true\n")
    root = write_workflows(tmp_path, {vfw.SOLE_WORKFLOW: workflow_yaml(step)})
    problems, _ = run_workflow(root)
    assert any("scm-do-build-during-deployment" in p for p in problems)
    assert any("enable-oryx-build" in p for p in problems)


def test_no_workflow_deploying_the_app_fails(tmp_path):
    root = write_workflows(tmp_path, {"unrelated.yml": workflow_yaml("""
      - name: Build
        run: echo hi
""")})
    problems, _ = run_workflow(root)
    assert any("no workflow deploys" in p for p in problems)


def test_deploying_from_the_wrong_workflow_name_fails(tmp_path):
    root = write_workflows(tmp_path, {"renamed.yml": workflow_yaml()})
    problems, _ = run_workflow(root)
    assert any("Exactly one may" in p for p in problems)


def test_reverting_to_the_cli_in_the_sole_workflow_is_rejected(tmp_path):
    cli = workflow_yaml("""
      - name: Deploy
        run: az functionapp deployment source config-zip -g trakt -n trakt-blob-trigger-v2 --src deploy.zip
""")
    root = write_workflows(tmp_path, {vfw.SOLE_WORKFLOW: cli})
    problems, _ = run_workflow(root)
    assert any("One deploy is the only deployment technology" in p
               for p in problems)


def test_a_missing_workflows_directory_fails_cleanly(tmp_path):
    try:
        run_workflow(tmp_path)
    except vfw.Failure as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("expected a Failure")


def test_invalid_yaml_fails_cleanly(tmp_path):
    root = write_workflows(tmp_path, {vfw.SOLE_WORKFLOW: "jobs: [oops\n"})
    try:
        run_workflow(root)
    except vfw.Failure as exc:
        assert "not valid YAML" in str(exc)
    else:
        raise AssertionError("expected a Failure")


# --------------------------------------------------------------------------
# package mode
# --------------------------------------------------------------------------

def test_a_clean_package_passes(tmp_path):
    z = make_zip(tmp_path / "a.zip",
                 ["function_app.py", "host.json", "requirements.txt",
                  "operations_control/engine.py"])
    problems, out = run_package(z)
    assert problems == [], f"{problems}\n{out}"
    assert "no .python_packages/ in the archive" in out


def test_a_shipped_local_build_is_rejected(tmp_path):
    z = make_zip(tmp_path / "b.zip",
                 ["function_app.py",
                  ".python_packages/lib/site-packages/yaml/__init__.py"])
    problems, _ = run_package(z)
    assert len(problems) == 1
    assert ".python_packages/" in problems[0]
    assert "second, stale copy" in problems[0]


def test_a_shipped_virtualenv_is_rejected(tmp_path):
    """`zip release.zip ./* -r` shipped venv/ on every run."""
    z = make_zip(tmp_path / "c.zip", ["function_app.py", "venv/bin/python"])
    problems, _ = run_package(z)
    assert any("'venv/'" in p for p in problems)


def test_a_nested_local_build_is_also_found(tmp_path):
    z = make_zip(tmp_path / "d.zip", ["apps/.venv/lib/x.py"])
    problems, _ = run_package(z)
    assert any(".venv/" in p for p in problems)


def test_a_missing_package_fails(tmp_path):
    problems, _ = run_package(tmp_path / "nope.zip")
    assert any("does not exist" in p for p in problems)


if __name__ == "__main__":
    import tempfile

    failures = 0
    names = sorted(n for n in list(globals()) if n.startswith("test_"))
    for name in names:
        fn = globals()[name]
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as td:
                    fn(pathlib.Path(td))
            else:
                fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL: {name}\n      {exc}")
        else:
            print(f"pass: {name}")
    print(f"\npassed: {len(names) - failures}   failed: {failures}")
    sys.exit(1 if failures else 0)
