"""Tests for the post-deployment verdicts in kudu_probe.py.

Runnable with pytest, or standalone (`python3 test_kudu_probe.py`) so the
deployment workflow can run them before any dependency is installed.

The fixtures are the real response shapes: a Kudu VFS directory listing and a
Kudu /api/command response.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
from contextlib import redirect_stdout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import kudu_probe  # noqa: E402


def _dir(name: str) -> dict:
    return {"name": name, "size": 0, "mime": "inode/directory",
            "href": f"https://x/{name}/"}


def _file(name: str) -> dict:
    return {"name": name, "size": 12, "mime": "application/octet-stream"}


def _healthy() -> list[dict]:
    return [
        _dir("yaml"), _dir("PyYAML-6.0.2.dist-info"),
        _dir("pandas"), _dir("pandas-2.2.2.dist-info"),
        _dir("numpy"), _dir("numpy-1.26.4.dist-info"),
        _dir("openpyxl"), _dir("openpyxl-3.1.5.dist-info"),
        _dir("azure"), _file("six.py"),
    ]


def run(fn, payload) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = fn(json.dumps(payload) if not isinstance(payload, str) else payload)
    return code, buf.getvalue()


# --------------------------------------------------------------------------
# site-packages listing
# --------------------------------------------------------------------------

def test_all_dependencies_present_passes_and_reports_versions():
    code, out = run(kudu_probe.check_site_packages, _healthy())
    assert code == 0, out
    assert "PyYAML 6.0.2" in out
    assert "pandas 2.2.2" in out
    assert "numpy 1.26.4" in out
    assert "openpyxl 3.1.5" in out
    assert "MISSING" not in out


def test_empty_site_packages_is_the_production_failure():
    code, out = run(kudu_probe.check_site_packages, [])
    assert code == 1
    assert "EMPTY" in out
    assert "installed nothing" in out


def test_missing_yaml_is_named():
    """The exact reported failure: engine.py:25 `import yaml`."""
    listing = [e for e in _healthy() if not e["name"].lower().startswith(
        ("yaml", "pyyaml"))]
    code, out = run(kudu_probe.check_site_packages, listing)
    assert code == 1
    assert "MISSING  yaml" in out
    assert "not importable on the deployed site: yaml" in out


def test_several_missing_are_all_named_not_just_the_first():
    listing = [_dir("yaml"), _dir("PyYAML-6.0.2.dist-info")]
    code, out = run(kudu_probe.check_site_packages, listing)
    assert code == 1
    for module in ("numpy", "openpyxl", "pandas"):
        assert f"MISSING  {module}" in out
    assert "numpy, openpyxl, pandas" in out


def test_dist_info_without_a_package_directory_fails_loudly():
    """A half-written install must not read as success."""
    listing = [e for e in _healthy() if e["name"] != "pandas"]
    code, out = run(kudu_probe.check_site_packages, listing)
    assert code == 1
    assert ".dist-info only, no package directory" in out


def test_package_directory_without_dist_info_still_passes():
    """The worker only needs the package on sys.path; say so rather than fail."""
    listing = [e for e in _healthy() if e["name"] != "numpy-1.26.4.dist-info"]
    code, out = run(kudu_probe.check_site_packages, listing)
    assert code == 0, out
    assert "no .dist-info recorded" in out


def test_distribution_name_matching_is_pep503_insensitive():
    listing = [e for e in _healthy() if e["name"] != "PyYAML-6.0.2.dist-info"]
    listing.append(_dir("pyyaml-6.0.2.dist-info"))
    code, out = run(kudu_probe.check_site_packages, listing)
    assert code == 0, out
    assert "PyYAML 6.0.2" in out


def test_a_file_named_like_a_package_is_not_a_package():
    """mime distinguishes a directory from a stray file of the same name."""
    listing = [e for e in _healthy() if e["name"] != "yaml"]
    listing.append(_file("yaml"))
    code, out = run(kudu_probe.check_site_packages, listing)
    assert code == 1
    assert "MISSING  yaml" in out


def test_html_error_page_instead_of_json_fails_with_the_body():
    code, out = run(kudu_probe.check_site_packages,
                    "<html><body>403 Forbidden</body></html>")
    assert code == 1
    assert "could not read the site-packages listing" in out
    assert "403 Forbidden" in out


def test_json_object_instead_of_array_fails():
    code, out = run(kudu_probe.check_site_packages, {"error": "nope"})
    assert code == 1
    assert "could not read the site-packages listing" in out


# --------------------------------------------------------------------------
# /api/command response
# --------------------------------------------------------------------------

def test_successful_import_chain_passes():
    code, out = run(kudu_probe.check_command_result, {
        "Output": "interpreter: /opt/python/3.11.9/bin/python3\n"
                  "Python 3.11.9\n"
                  "yaml 6.0.2 | pandas 2.2.2 | numpy 1.26.4 | openpyxl 3.1.5\n"
                  "OCC intake import chain OK: OpsEngine OpsStore\n",
        "Error": "", "ExitCode": 0})
    assert code == 0, out
    assert "PASS" in out
    assert "OCC intake import chain OK" in out
    assert "Python 3.11.9" in out


def test_module_not_found_is_reported_with_the_module_name():
    code, out = run(kudu_probe.check_command_result, {
        "Output": "interpreter: /opt/python/3.11.9/bin/python3\nPython 3.11.9\n",
        "Error": 'Traceback (most recent call last):\n'
                 '  File "<stdin>", line 3, in <module>\n'
                 "ModuleNotFoundError: No module named 'yaml'\n",
        "ExitCode": 1})
    assert code == 1
    assert "still cannot import 'yaml'" in out
    assert "reproduced on the site" in out


def test_module_not_found_deeper_in_the_chain_is_also_caught():
    code, out = run(kudu_probe.check_command_result, {
        "Output": "", "ExitCode": 1,
        "Error": "ModuleNotFoundError: No module named 'operations_control'\n"})
    assert code == 1
    assert "still cannot import 'operations_control'" in out


def test_non_import_failure_is_not_misreported_as_a_missing_module():
    code, out = run(kudu_probe.check_command_result, {
        "Output": "", "Error": "no python interpreter in the SCM container\n",
        "ExitCode": 127})
    assert code == 1
    assert "exited non-zero" in out
    assert "cannot import" not in out


def test_missing_exit_code_is_not_treated_as_success():
    code, out = run(kudu_probe.check_command_result, {"Output": "", "Error": ""})
    assert code == 1


def test_non_json_command_response_fails_with_the_body():
    code, out = run(kudu_probe.check_command_result, "Bad Gateway")
    assert code == 1
    assert "was not JSON" in out
    assert "Bad Gateway" in out


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL: {name}\n      {exc}")
        else:
            print(f"pass: {name}")
    print(f"\npassed: {sum(1 for n in globals() if n.startswith('test_')) - failures}"
          f"   failed: {failures}")
    sys.exit(1 if failures else 0)
