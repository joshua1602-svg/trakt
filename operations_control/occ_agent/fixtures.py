"""operations_control.occ_agent.fixtures — the deterministic synthetic scenarios.

Five reusable onboarding fixtures, each exercising a different control path
(§18). They are entirely synthetic: invented lenders, invented loan references,
invented values. No fixture derives from, approximates, or names a live client.

Everything is deterministic — fixed rows, fixed dates, no randomness and no
dependence on the current date — so a fixture always reaches the same state and
produces the same execution manifest. That is what makes the scenarios usable as
tests as well as demonstrations.

The scenarios are chosen so that each one is blocked (or not) by a DIFFERENT
control:

* **A — clean onboarding**: everything resolves; reaches ``READY_FOR_EXECUTION``.
* **B — ambiguous mapping**: two source columns claim one canonical field, so
  the header mapper cannot choose; a human confirms, the affected controls rerun,
  and the case reaches ``READY_FOR_EXECUTION``.
* **C — missing mandatory artefact**: the required input role is absent, so the
  configured input requirements block the case.
* **D — product configuration gap**: core onboarding is fine, but the selected
  product needs configuration the client has not supplied, so the configuration
  resolver blocks readiness. The gap belongs to whichever product the fixture
  selects — it is not written around any one reporting product.
* **E — material business-rule failure**: canonical transformation succeeds and
  the deterministic business rules then fail at BLOCKING materiality, which
  ordinary conversation cannot clear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from . import states as _states

#: The synthetic clients. Invented names; no relation to any real entity.
CLIENT_A = "Northstar Lending"
CLIENT_B = "Harbour Point Capital"
CLIENT_C = "Kestrel Mutual"
CLIENT_D = "Aldermere Advances"
CLIENT_E = "Brackenfield Finance"


@dataclass(frozen=True)
class SyntheticFile:
    filename: str
    content: str
    declared_type: str = ""


@dataclass(frozen=True)
class Scenario:
    """One reusable synthetic onboarding scenario."""

    fixture_id: str
    label: str
    description: str
    instruction: str
    files: Tuple[SyntheticFile, ...] = ()
    #: The state the scenario is expected to settle in when driven end to end.
    expected_state: str = _states.READY_FOR_EXECUTION
    #: What the human must do part-way through, in the order it is needed.
    expected_human_steps: Tuple[str, ...] = ()
    #: What this scenario is demonstrating.
    demonstrates: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "fixture_id": self.fixture_id, "label": self.label,
            "description": self.description, "instruction": self.instruction,
            "files": [{"filename": f.filename, "artefact_type": f.declared_type,
                       "bytes": len(f.content.encode("utf-8"))}
                      for f in self.files],
            "expected_state": self.expected_state,
            "expected_human_steps": list(self.expected_human_steps),
            "demonstrates": self.demonstrates,
        }


# --------------------------------------------------------------------------- #
# Tape construction
# --------------------------------------------------------------------------- #

#: The columns of a well-formed synthetic loan tape, and the headers a client
#: would actually use. Every canonical field the platform marks
#: ``core_canonical`` for a direct book is present, so a clean fixture has no
#: manufactured gaps.
_LOAN_HEADERS = (
    "Loan Identifier",
    "Account Status",
    "Amortisation Type",
    "Exposure Currency Denomination",
    "Current Interest Rate",
    "Current Principal Balance",
    "Original Principal Balance",
    "Data Cut Off Date",
    "Interest Rate Type",
    "Maturity Date",
    "Origination Date",
    "Originator Legal Entity Identifier",
    "Originator Name",
    "Current Valuation Amount",
)

REPORTING_DATE = "2026-06-30"

#: A fixed, synthetic LEI. Twenty characters in the ISO 17442 shape so the
#: platform's format checks are genuinely exercised rather than skipped.
_SYNTHETIC_LEI = "894500SYNTHETIC00042X"[:20]


def _loan_rows(count: int, *, prefix: str, originator: str,
               maturity_before_origination: bool = False) -> List[List[str]]:
    """``count`` deterministic loan rows. No randomness anywhere."""
    rows: List[List[str]] = []
    for i in range(1, count + 1):
        balance = 100000 + (i * 1373)          # deterministic spread
        original = balance - (i * 97)
        valuation = balance * 3 + (i * 250)
        origination = f"20{10 + (i % 9):02d}-0{1 + (i % 8)}-1{i % 9}"
        maturity = (origination if not maturity_before_origination
                    else f"20{5 + (i % 4):02d}-0{1 + (i % 8)}-1{i % 9}")
        if not maturity_before_origination:
            maturity = f"20{40 + (i % 9):02d}-0{1 + (i % 8)}-1{i % 9}"
        rows.append([
            f"{prefix}{i:05d}",
            "PERF",
            "BULLET",
            "GBP",
            f"{4.5 + (i % 5) * 0.15:.4f}",
            f"{balance}",
            f"{original}",
            REPORTING_DATE,
            "FIXED",
            maturity,
            origination,
            _SYNTHETIC_LEI,
            originator,
            f"{valuation}",
        ])
    return rows


def _csv(headers: List[str], rows: List[List[str]]) -> str:
    lines = [",".join(headers)]
    lines += [",".join(r) for r in rows]
    return "\n".join(lines) + "\n"


def loan_tape(*, count: int = 24, prefix: str, originator: str,
              extra_balance_column: bool = False,
              maturity_before_origination: bool = False) -> str:
    """A synthetic loan tape.

    ``extra_balance_column`` adds a second column the platform's alias table
    also resolves to ``current_principal_balance`` — a real ambiguity the header
    mapper cannot settle on its own, which is exactly what scenario B needs.
    """
    headers = list(_LOAN_HEADERS)
    rows = _loan_rows(count, prefix=prefix, originator=originator,
                      maturity_before_origination=maturity_before_origination)
    if extra_balance_column:
        headers.append("Principal Balance")
        balance_index = headers.index("Current Principal Balance")
        for i, row in enumerate(rows, start=1):
            # Populated for all but the last few rows, so the decision card can
            # show a genuine coverage difference between the two candidates.
            row.append(row[balance_index] if i <= count - 3 else "")
    return _csv(headers, rows)


def property_tape(*, count: int = 24, prefix: str) -> str:
    """A synthetic property/valuation extract."""
    headers = ["Loan Identifier", "Collateral Identifier",
               "Current Valuation Amount", "Current Valuation Date",
               "Property Postcode"]
    rows = []
    for i in range(1, count + 1):
        rows.append([f"{prefix}{i:05d}", f"COL{i:05d}",
                     f"{300000 + i * 4119}", REPORTING_DATE,
                     f"AB{i % 9 + 1} {i % 9 + 1}CD"])
    return _csv(headers, rows)


def cashflow_tape(*, count: int = 24, prefix: str) -> str:
    """A synthetic executed-cashflow extract."""
    headers = ["Loan Identifier", "Payment Date", "Principal Paid",
               "Interest Paid"]
    rows = []
    for i in range(1, count + 1):
        rows.append([f"{prefix}{i:05d}", REPORTING_DATE, f"{i * 11}",
                     f"{i * 37}"])
    return _csv(headers, rows)


# --------------------------------------------------------------------------- #
# The scenarios
# --------------------------------------------------------------------------- #

SCENARIO_A = Scenario(
    fixture_id="scenario_a_clean",
    label="A — Clean onboarding",
    description="Complete artefacts, valid configuration and high-confidence "
                "mappings. Reaches READY_FOR_EXECUTION with no blocking "
                "control.",
    instruction=(
        f"Onboard {CLIENT_A}. It is a UK equity-release portfolio. They "
        "require monthly portfolio MI. Portfolio id direct_101. First "
        f"reporting date {REPORTING_DATE}. Go-live 2026-09-30. The initial "
        "monthly files will be a loan tape and executed cashflows."),
    files=(
        SyntheticFile("northstar_loan_extract_202606.csv",
                      loan_tape(prefix="NSL", originator=CLIENT_A),
                      "loan_extract"),
        SyntheticFile("northstar_cashflow_202606.csv",
                      cashflow_tape(prefix="NSL"), "cashflow_extract"),
    ),
    expected_state=_states.READY_FOR_EXECUTION,
    expected_human_steps=("confirm the interpretation", "approve the pack",
                          "approve the configuration", "approve readiness"),
    demonstrates="The whole operating process end to end with nothing in the "
                 "way.",
)

SCENARIO_B = Scenario(
    fixture_id="scenario_b_ambiguous_mapping",
    label="B — Ambiguous mapping",
    description="Two source columns resolve to the same canonical field. The "
                "run halts for a human decision; once confirmed the affected "
                "controls rerun and the case reaches READY_FOR_EXECUTION.",
    instruction=(
        f"Onboard {CLIENT_B}. UK equity release. Monthly portfolio MI. "
        f"Portfolio id direct_102. First reporting date {REPORTING_DATE}. "
        "Go-live 2026-10-31. They will send a loan tape."),
    files=(
        SyntheticFile("harbourpoint_loan_extract_202606.csv",
                      loan_tape(prefix="HPC", originator=CLIENT_B,
                                extra_balance_column=True),
                      "loan_extract"),
    ),
    expected_state=_states.READY_FOR_EXECUTION,
    expected_human_steps=("confirm the interpretation", "approve the pack",
                          "approve the configuration",
                          "settle the ambiguous mapping", "approve readiness"),
    demonstrates="A control the engine cannot settle alone, and the rerun that "
                 "follows a human decision.",
)

SCENARIO_C = Scenario(
    fixture_id="scenario_c_missing_artefact",
    label="C — Missing mandatory artefact",
    description="A required input role is absent. The agent asks a targeted "
                "question and the case stays blocked.",
    instruction=(
        f"Onboard {CLIENT_C}. UK equity release. Monthly portfolio MI. "
        f"Portfolio id direct_103. First reporting date {REPORTING_DATE}. "
        "Go-live 2026-11-30."),
    files=(
        # A property tape only: the configured required role (the loan tape) is
        # missing, so readiness against the input requirements fails.
        SyntheticFile("kestrel_property_extract_202606.csv",
                      property_tape(prefix="KML"), "property_extract"),
    ),
    expected_state=_states.BLOCKED,
    expected_human_steps=("confirm the interpretation", "approve the pack"),
    demonstrates="The configured input requirements refusing an incomplete "
                 "pack.",
)

SCENARIO_D = Scenario(
    fixture_id="scenario_d_product_config_gap",
    label="D — Product configuration gap",
    description="Core onboarding can proceed, but the selected product needs "
                "configuration the client has not supplied. Readiness stays "
                "blocked until it is provided.",
    instruction=(
        f"Onboard {CLIENT_D}. UK equity release. They need monthly portfolio "
        "MI and regulatory reporting. Portfolio id direct_104. First "
        f"reporting date {REPORTING_DATE}. Go-live 2027-01-31. They will send "
        "a loan tape."),
    files=(
        SyntheticFile("aldermere_loan_extract_202606.csv",
                      loan_tape(prefix="ALD", originator=CLIENT_D),
                      "loan_extract"),
    ),
    expected_state=_states.BLOCKED,
    expected_human_steps=("confirm the interpretation", "approve the pack",
                          "supply the product configuration"),
    demonstrates="A product-specific configuration requirement, loaded through "
                 "the product framework rather than hard-coded to one report.",
)

SCENARIO_E = Scenario(
    fixture_id="scenario_e_business_rule_failure",
    label="E — Material business-rule failure",
    description="Canonical transformation succeeds; the deterministic business "
                "rules then fail, and the materiality logic makes the failure "
                "blocking. Conversation cannot clear it.",
    instruction=(
        f"Onboard {CLIENT_E}. UK equity release. Monthly portfolio MI. "
        f"Portfolio id direct_105. First reporting date {REPORTING_DATE}. "
        "Go-live 2027-03-31. They will send a loan tape."),
    files=(
        SyntheticFile("brackenfield_loan_extract_202606.csv",
                      loan_tape(prefix="BFF", originator=CLIENT_E,
                                maturity_before_origination=True),
                      "loan_extract"),
    ),
    expected_state=_states.BLOCKED,
    expected_human_steps=("confirm the interpretation", "approve the pack",
                          "approve the configuration"),
    demonstrates="A deterministic blocker that natural language cannot bypass.",
)

SCENARIOS: Tuple[Scenario, ...] = (SCENARIO_A, SCENARIO_B, SCENARIO_C,
                                   SCENARIO_D, SCENARIO_E)


def by_id(fixture_id: str) -> Scenario:
    scenario = next((s for s in SCENARIOS if s.fixture_id == fixture_id), None)
    if scenario is None:
        from ..engine import OpsError
        raise OpsError("OCC_AGENT_FIXTURE_NOT_FOUND",
                       "That synthetic scenario could not be found.",
                       http_status=404)
    return scenario


def catalogue() -> List[Dict[str, object]]:
    return [s.to_dict() for s in SCENARIOS]
