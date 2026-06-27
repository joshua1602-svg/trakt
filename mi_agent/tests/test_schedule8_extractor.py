#!/usr/bin/env python3
"""tests/test_schedule8_extractor.py

Schedule 8 concentration-limit extraction (Part 4):
  * the Schedule 8 fixture can be located and read;
  * structured limits are extracted where present (geographic / broker / large
    loan / LTV / WAC / borrower / age);
  * ambiguous / hedged limits are marked needs_review with a null value;
  * unavailable limits are NOT fabricated (controlled unavailable / needs-review);
  * the source snippet + section are retained.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.risk_monitor import schedule8_extractor as ex

_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack" / "schedule_8_concentration.txt"


@pytest.fixture(scope="module")
def extracted():
    return ex.extract_from_file(_FIXTURE, portfolio_id="client_001")


def test_fixture_is_locatable():
    found = ex.locate_schedule8(_REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack")
    assert found is not None and found.name == "schedule_8_concentration.txt"


def test_extracts_structured_limits(extracted):
    assert extracted["available"] is True
    assert extracted["limit_count"] >= 12
    cats = set(extracted["categories"])
    for required in ("geographic_concentration", "broker_concentration",
                     "large_loan_concentration", "ltv_limit",
                     "interest_rate_limit", "borrower_concentration", "age_limit"):
        assert required in cats, required


def test_geographic_london_limit(extracted):
    london = next((l for l in extracted["limits"]
                   if l["category"] == "geographic_concentration" and l["region"] == "London"), None)
    assert london is not None
    assert london["limit_value"] == 30.0
    assert london["unit"] == "percent"
    assert london["direction"] == "max"
    assert london["confidence"] == "high"
    assert london["needs_review"] is False
    assert "London" in london["source_snippet"]
    assert london["source_section"]  # section retained


def test_top_n_broker_limit(extracted):
    topn = [l for l in extracted["limits"] if l["category"] == "broker_concentration"]
    values = {l["limit_value"] for l in topn}
    assert 20.0 in values and 45.0 in values  # single broker + top 3 brokers


def test_count_limit_loans_per_borrower(extracted):
    counts = [l for l in extracted["limits"] if l["unit"] == "count"]
    assert any(l["limit_value"] == 5.0 for l in counts)


def test_ambiguous_limit_marked_needs_review(extracted):
    joint = next((l for l in extracted["limits"]
                  if l["category"] == "joint_borrower_limit"), None)
    assert joint is not None
    assert joint["needs_review"] is True
    assert joint["limit_value"] is None  # not fabricated
    assert extracted["needs_review_count"] >= 1


def test_unavailable_file_is_controlled_not_fabricated(tmp_path):
    out = ex.extract_from_file(tmp_path / "does_not_exist.txt", portfolio_id="client_001")
    assert out["available"] is False
    assert out["status"] == "unavailable"
    assert out["limits"] == []
    assert out["limit_count"] == 0


def test_non_text_file_needs_review(tmp_path):
    binary = tmp_path / "schedule8.bin"
    binary.write_bytes(b"\xff\xfe\x00\x01\x02scanned-pdf-bytes\x80\x81")
    out = ex.extract_from_file(binary)
    assert out["available"] is False
    assert out["status"] == "needs_review"
    assert out["limits"] == []


def test_no_hallucinated_limits_from_prose():
    text = "This schedule describes the portfolio. Limits are agreed separately."
    out = ex.extract_schedule8_limits(text)
    # No numeric limit statements -> no fabricated limits.
    assert out["limit_count"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
