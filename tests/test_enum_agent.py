"""
Tests for engine/enum_agent — review_cli and _redact_sample.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from typing import List
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Make the package importable from the repo root without installation
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Provide a minimal stub for pandas so the module loads without the full dep
if "pandas" not in sys.modules:
    pd_stub = types.ModuleType("pandas")
    pd_stub.Series = list  # type: ignore[attr-defined]
    pd_stub.isna = lambda v: v is None  # type: ignore[attr-defined]
    sys.modules["pandas"] = pd_stub

from engine.enum_agent.enum_mapping_agent import EnumSuggestion, _redact_sample
from engine.enum_agent.enum_review import review_cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWED = ["Male", "Female", "Unknown"]


def _make_suggestion(
    raw_value: str = "m",
    suggested_value: str = "Male",
    status: str = "pending",
) -> EnumSuggestion:
    return EnumSuggestion(
        field_name="borrower_gender",
        raw_value=raw_value,
        suggested_value=suggested_value,
        confidence=0.95,
        reasoning="fuzzy match",
        alternative_value=None,
        allowed_values=ALLOWED,
        count=10,
        namespace="global",
        regime="",
        status=status,
    )


def _run_review(inputs: List[str], suggestions: List[EnumSuggestion]) -> List[EnumSuggestion]:
    """Drive review_cli by feeding pre-programmed input strings."""
    input_iter = iter(inputs)
    with patch("builtins.input", side_effect=lambda _="": next(input_iter)):
        return review_cli(suggestions)


# ---------------------------------------------------------------------------
# Tests: action paths
# ---------------------------------------------------------------------------

class TestReviewCliActions(unittest.TestCase):

    def test_confirm_sets_status(self):
        s = _make_suggestion()
        result = _run_review(["C", ""], [s])
        self.assertEqual(result[0].status, "confirmed")
        self.assertEqual(result[0].confirmed_value, "Male")

    def test_skip_sets_status(self):
        s = _make_suggestion()
        result = _run_review(["S", ""], [s])
        self.assertEqual(result[0].status, "skipped")

    def test_reject_sets_status(self):
        s = _make_suggestion()
        result = _run_review(["J", "poor match"], [s])
        self.assertEqual(result[0].status, "rejected")
        self.assertIsNone(result[0].confirmed_value)

    def test_remap_sets_status_and_value(self):
        s = _make_suggestion()
        result = _run_review(["R", "Female", "manual override"], [s])
        self.assertEqual(result[0].status, "remapped")
        self.assertEqual(result[0].confirmed_value, "Female")

    def test_remap_invalid_value_prompts_again(self):
        s = _make_suggestion()
        # First remap attempt uses a value not in allowed list; second is valid
        result = _run_review(["R", "NotAllowed", "", "R", "Unknown", ""], [s])
        self.assertEqual(result[0].status, "remapped")
        self.assertEqual(result[0].confirmed_value, "Unknown")

    def test_confirm_null_suggestion_prompts_again(self):
        s = _make_suggestion(suggested_value=None)
        # suggested_value is None so confirm is blocked; fall through to skip
        result = _run_review(["C", "S", ""], [s])
        self.assertEqual(result[0].status, "skipped")

    def test_invalid_choice_prompts_again(self):
        s = _make_suggestion()
        # "X" is invalid — loop should continue and accept "C" next
        result = _run_review(["X", "C", ""], [s])
        self.assertEqual(result[0].status, "confirmed")

    def test_no_pending_returns_immediately(self):
        already_confirmed = _make_suggestion(status="confirmed")
        already_confirmed.confirmed_value = "Male"
        with patch("builtins.input", side_effect=AssertionError("input called unexpectedly")):
            result = review_cli([already_confirmed])
        self.assertEqual(result[0].status, "confirmed")

    def test_multiple_suggestions_processed_in_order(self):
        s1 = _make_suggestion(raw_value="m")
        s2 = _make_suggestion(raw_value="f", suggested_value="Female")
        result = _run_review(["C", "", "J", "low confidence"], [s1, s2])
        self.assertEqual(result[0].status, "confirmed")
        self.assertEqual(result[1].status, "rejected")


# ---------------------------------------------------------------------------
# Tests: _redact_sample date shielding
# ---------------------------------------------------------------------------

class TestRedactSampleDateShielding(unittest.TestCase):

    def test_iso_date_not_redacted_as_phone(self):
        result = _redact_sample("2024-01-15")
        self.assertEqual(result, "2024-01-15")
        self.assertNotIn("[PHONE]", result)

    def test_european_dot_date_not_redacted(self):
        result = _redact_sample("15.01.2024")
        self.assertEqual(result, "15.01.2024")
        self.assertNotIn("[PHONE]", result)

    def test_european_hyphen_date_not_redacted(self):
        result = _redact_sample("15-01-2024")
        self.assertEqual(result, "15-01-2024")
        self.assertNotIn("[PHONE]", result)

    def test_undelimited_date_not_redacted_as_id(self):
        result = _redact_sample("20240115")
        self.assertEqual(result, "20240115")
        self.assertNotIn("[ID]", result)

    def test_real_phone_still_redacted(self):
        result = _redact_sample("+44 7911 123456")
        self.assertIn("[PHONE]", result)
        self.assertNotIn("7911", result)

    def test_real_id_still_redacted(self):
        result = _redact_sample("account 1234567890")
        self.assertIn("[ID]", result)

    def test_date_and_phone_in_same_string(self):
        result = _redact_sample("DOB 1985-03-22 mob 07700900123")
        self.assertIn("1985-03-22", result)
        self.assertIn("[PHONE]", result)

    def test_email_still_redacted(self):
        result = _redact_sample("contact@example.com")
        self.assertIn("[EMAIL]", result)


if __name__ == "__main__":
    unittest.main()
