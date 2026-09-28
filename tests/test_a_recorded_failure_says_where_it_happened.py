"""Two halts in one night, and neither record said which line.

    "error": "TypeError: Messages.create() got an unexpected keyword 'temperature'"
    "error": "ValueError: The truth value of a Series is ambiguous. ..."

Both were one line of code away from obvious. Both took hours, because the one
thing a message of that shape cannot tell you is where it came from: the first
was found by reading four LLM call sites, the second by statically scanning
every boolean test in a package for one that could be holding a Series.

The recording code had the traceback in its hands and kept the ``str()``::

    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"

So a failure now carries its location. This is deliberately NOT operator text —
``operations_control.language`` forbids exception classes, file paths and
tracebacks in anything the UI renders, and that contract is not being loosened
here. It goes to the run artefacts and the event log, which the UI never renders
and an engineer always reads.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trakt_core import fault_report


def _raise_the_live_failure() -> BaseException:
    """The delivery's actual second halt, raised from a named function."""
    try:
        frame = pd.DataFrame([[1, 2]], columns=["dup", "dup"])
        if frame["dup"].isna().any():       # a DataFrame, so .any() is a Series
            pass
    except Exception as exc:                # noqa: BLE001 — the subject
        return exc
    raise AssertionError("expected the ambiguous-Series failure")


class TestTheLocationIsRecorded:

    def test_it_names_the_file_line_and_function(self):
        where = fault_report.fault_location(_raise_the_live_failure())
        assert "test_a_recorded_failure_says_where_it_happened.py:" in where
        assert "in _raise_the_live_failure" in where

    def test_the_path_is_relative_to_the_repository(self):
        """An absolute path would be a machine's detail, not a location."""
        where = fault_report.fault_location(_raise_the_live_failure())
        assert not where.startswith("/")

    def test_the_deepest_frame_we_own_is_the_one_reported(self):
        """The raise is inside pandas; the useful frame is the last one ours."""
        where = fault_report.fault_location(_raise_the_live_failure())
        assert "pandas" not in where
        assert "site-packages" not in where


class TestTheReportCarriesWhatDiagnosisNeeds:

    def test_message_type_location_and_traceback(self):
        report = fault_report.fault_report(_raise_the_live_failure())
        assert report["error_type"] == "ValueError"
        assert "truth value of a Series is ambiguous" in report["error"]
        assert "in _raise_the_live_failure" in report["error_location"]
        assert "Traceback (most recent call last)" in report["error_traceback"]

    def test_the_traceback_can_be_left_out(self):
        report = fault_report.fault_report(_raise_the_live_failure(),
                                           include_traceback=False)
        assert "error_traceback" not in report
        assert report["error_location"]

    def test_one_line_reads_as_message_and_place(self):
        line = fault_report.describe(_raise_the_live_failure())
        assert line.startswith("ValueError: ")
        assert " (at " in line and line.endswith(")")


class TestReportingNeverBecomesTheFailure:
    """A diagnostic that raises while describing a failure is worse than the
    gap it was added to close."""

    def test_an_exception_that_was_never_raised_has_no_location(self):
        assert fault_report.fault_location(ValueError("never raised")) == ""

    def test_it_still_reports_the_message_without_a_traceback(self):
        report = fault_report.fault_report(ValueError("never raised"))
        assert report["error"] == "ValueError: never raised"
        assert report["error_location"] == ""

    @pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(1)])
    def test_it_handles_what_is_not_an_exception_subclass(self, exc):
        assert isinstance(fault_report.fault_report(exc), dict)


class TestThisIsNotOperatorText:
    """The technical record and the operator's sentence are different things,
    and this module is only ever the first."""

    def test_what_is_recorded_here_would_fail_the_operator_contract(self):
        from operations_control import language
        line = fault_report.describe(_raise_the_live_failure())
        assert not language.is_operator_safe(line), (
            "a location is for the event log; the UI gets a sentence")
