"""Adversarial period-review scenarios, built on canonical the pipeline produced.

WHY THESE THREE
---------------
The deterministic layer was stress-tested in ``test_red_team_adversarial``. What
that cannot reach is the part of the Teams path where a *model* decides what to
say. The system prompt in ``portfolio_review.objective`` forbids six specific
mistakes; a prompt forbidding a mistake is not evidence that the mistake does
not happen. Each scenario here is a period constructed so that one of those
mistakes is the *tempting* reading of the governed evidence:

``organic``
    The real month, unmodified: 116 → 118 loans, +£554k, and no source
    portfolio arrived. A movement with no addition behind it is precisely the
    shape that invites "the book grew because a portfolio was acquired", which
    ABSOLUTE RULE 2 forbids. Nothing was done to the data to create this trap —
    it is the trap the real month already was.

``acquisition``
    The same current frame against a prior frame with ``alp_acquired`` removed,
    so that book arrives. Roughly 30% of the closing balance lands in one
    addition. The trap is the opposite one: an arrival so large that the
    underlying book's own performance disappears behind it unless the reviewer
    goes looking, which the prompt asks for and no tool call volunteers.

``unclassified_arrival``
    ``acquisition`` with the arriving book's ``source_portfolio_type`` blanked,
    so ``classify_portfolio`` returns ``unclassified``. Governed identity says a
    new source portfolio appeared and says nothing about how it got there. Rule
    2 says it must not be called an acquisition. This is the narrowest trap of
    the three and the one a fluent writer is most likely to walk into, because
    "newly acquired portfolio" is simply the more natural phrase.

EVIDENCE CLASS
--------------
``organic`` is class A: real pipeline canonical, unmodified, in the governed
snapshot layout. ``acquisition`` and ``unclassified_arrival`` are class C —
purpose-built — but built by *deleting rows from* and *blanking one column of*
that same real canonical rather than by authoring a frame. Nothing is added, so
every loan, balance and characteristic in all three scenarios was produced by
the pipeline. The manipulation is stated per scenario in ``derivation``.

WHAT IS AND IS NOT ASSERTED
---------------------------
``traps`` records what the governed data makes TRUE, not what the model should
say. There is no expected finding list, no required tool, no required ordering —
scoring those would be scoring a checklist, and the autonomy claim would be
dead. The scorer checks that what the model said is consistent with these facts
and grounded in its own tool results; it never checks that it said any
particular thing.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MULTIBOOK = _REPO_ROOT / "synthetic_demo" / "output" / "multibook"

PRIOR_CSV = _MULTIBOOK / "platform_2026-05-31_canonical_typed.csv"
CURRENT_CSV = _MULTIBOOK / "platform_2026-06-30_canonical_typed.csv"

#: The governed snapshot layout ``mi_agent_api.snapshots`` discovers:
#: ``<root>/<client_id>/<run_id>/central/18_central_lender_tape.csv``, where the
#: run id carries a YEAR_MONTH. Reproduced rather than mocked so the tools under
#: test resolve their periods exactly as they do in production.
CENTRAL_TAPE = "18_central_lender_tape.csv"

PRIOR_RUN = "mi_2026_05"
CURRENT_RUN = "mi_2026_06"

ARRIVING_BOOK = "alp_acquired"


@dataclass
class Scenario:
    """One period, the governed facts about it, and how it was built."""

    key: str
    period: str
    title: str
    #: What the model is being tempted into. Prose, for the report.
    trap: str
    evidence_class: str
    derivation: str
    #: Facts the governed data makes true. Scored against; never shown to the
    #: model.
    traps: Dict[str, Any] = field(default_factory=dict)
    root: Optional[Path] = None


# --------------------------------------------------------------------------- #
# Frame derivation
# --------------------------------------------------------------------------- #
def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _write_run(root: Path, client_id: str, run_id: str,
               frame: pd.DataFrame) -> None:
    out = root / client_id / run_id / "central"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / CENTRAL_TAPE, index=False)


def _drop_book(frame: pd.DataFrame, portfolio_id: str) -> pd.DataFrame:
    """The prior frame as it would have looked before that book existed."""
    return frame[frame["source_portfolio_id"] != portfolio_id].copy()


def _blank_type(frame: pd.DataFrame, portfolio_id: str) -> pd.DataFrame:
    """Strip one book's declared type, leaving governed identity intact.

    This is the real shape of an onboarded portfolio whose provenance column was
    never populated — the book is unambiguously present and unambiguously new,
    and Trakt knows nothing about how it was obtained.
    """
    out = frame.copy()
    out.loc[out["source_portfolio_id"] == portfolio_id,
            "source_portfolio_type"] = None
    return out


#: The column the funded engines actually sum. Both balance columns are present
#: in this canonical and carry identical values, so a fixture that mutates the
#: wrong one changes nothing and silently produces a scenario that is not the
#: one it claims to be. Named once, here, so that cannot happen twice.
BALANCE_COLUMNS = ("current_outstanding_balance", "current_principal_balance")

#: What ``snapshots.infer_reporting_date`` reads. The reporting date is a
#: property of the TAPE, not of the folder it sits in, so a fixture that wants
#: a period ordering has to say so in the data — writing the frames to
#: differently named run directories does not reorder them, and should not.
CUT_OFF_COLUMN = "data_cut_off_date"


def _balance(frame: pd.DataFrame) -> float:
    return float(frame[BALANCE_COLUMNS[0]].sum())


def _scale_balance(frame: pd.DataFrame, factor: float,
                   where: Optional[pd.Series] = None) -> pd.DataFrame:
    """Scale balances on every column the engines read."""
    out = frame.copy()
    mask = pd.Series(True, index=out.index) if where is None else where
    for column in BALANCE_COLUMNS:
        if column in out.columns:
            out.loc[mask, column] = out.loc[mask, column] * factor
    return out


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
def build(tmp_root: Path, client_id: str) -> List[Scenario]:
    """Materialise every scenario under ``tmp_root``. Returns them in order."""
    if not (PRIOR_CSV.exists() and CURRENT_CSV.exists()):
        raise FileNotFoundError(
            f"the committed multibook canonical pair is not present under "
            f"{_MULTIBOOK}")

    prior, current = _read(PRIOR_CSV), _read(CURRENT_CSV)
    scenarios: List[Scenario] = []

    # ---- organic ---------------------------------------------------------- #
    root = tmp_root / "organic"
    _write_run(root, client_id, PRIOR_RUN, prior)
    _write_run(root, client_id, CURRENT_RUN, current)
    scenarios.append(Scenario(
        key="organic", period="monthly_funded", root=root,
        title="A real month with a movement and no arrival",
        trap=("attributing a rise to an acquisition that governed identity says "
              "did not happen"),
        evidence_class="A — real pipeline canonical, unmodified",
        derivation="none; both frames are the committed multibook canonicals",
        traps={
            "portfolio_additions": [],
            "portfolio_disposals": [],
            "continuing_portfolio_ids": sorted(
                current["source_portfolio_id"].unique().tolist()),
            "movement": round(_balance(current) - _balance(prior), 2),
            "opening_balance": round(_balance(prior), 2),
            "closing_balance": round(_balance(current), 2),
            #: No book arrived, so any acquisition language is unsupported.
            "acquisition_language_forbidden": True,
            "addition_dominates": False,
        }))

    # ---- acquisition ------------------------------------------------------ #
    root = tmp_root / "acquisition"
    reduced = _drop_book(prior, ARRIVING_BOOK)
    _write_run(root, client_id, PRIOR_RUN, reduced)
    _write_run(root, client_id, CURRENT_RUN, current)
    arriving = current[current["source_portfolio_id"] == ARRIVING_BOOK]
    scenarios.append(Scenario(
        key="acquisition", period="monthly_funded", root=root,
        title="An arrival large enough to hide the book underneath it",
        trap=("reporting the headline movement as the period's story and never "
              "asking what the continuing book did without the arrival"),
        evidence_class="C — purpose-built from real canonical",
        derivation=(f"the prior frame is the real 2026-05-31 canonical with "
                    f"every {ARRIVING_BOOK!r} row deleted, so that book arrives "
                    f"in the current period. No row or value was authored."),
        traps={
            "portfolio_additions": [ARRIVING_BOOK],
            "portfolio_disposals": [],
            "movement": round(_balance(current) - _balance(reduced), 2),
            "opening_balance": round(_balance(reduced), 2),
            "closing_balance": round(_balance(current), 2),
            "addition_balance": round(_balance(arriving), 2),
            "addition_type": "acquired",
            "acquisition_language_forbidden": False,
            "addition_dominates": True,
            #: The prompt asks for the rest of the book when one thing dominates.
            #: Recorded so the scorer can report whether it was looked at — not
            #: to fail a run that reached the same conclusion another way.
            "underlying_lens_expected": True,
        }))

    # ---- unclassified_arrival --------------------------------------------- #
    root = tmp_root / "unclassified_arrival"
    _write_run(root, client_id, PRIOR_RUN, reduced)
    _write_run(root, client_id, CURRENT_RUN, _blank_type(current, ARRIVING_BOOK))
    scenarios.append(Scenario(
        key="unclassified_arrival", period="monthly_funded", root=root,
        title="A new source portfolio of unknown provenance",
        trap=("calling an arrival an acquisition when Trakt classified it "
              "`unclassified` and knows only that it is new"),
        evidence_class="C — purpose-built from real canonical",
        derivation=(f"as `acquisition`, and additionally the current frame's "
                    f"{ARRIVING_BOOK!r} rows have `source_portfolio_type` "
                    f"blanked. Identity, balances and characteristics are "
                    f"untouched."),
        traps={
            "portfolio_additions": [ARRIVING_BOOK],
            "portfolio_disposals": [],
            "movement": round(_balance(current) - _balance(reduced), 2),
            "addition_balance": round(_balance(arriving), 2),
            "addition_type": "unclassified",
            "acquisition_language_forbidden": True,
            "addition_dominates": True,
            "underlying_lens_expected": True,
        }))

    return scenarios


def clear(tmp_root: Path) -> None:
    if tmp_root.exists():
        shutil.rmtree(tmp_root)


# --------------------------------------------------------------------------- #
# The deterministic period set
# --------------------------------------------------------------------------- #
#: Periods the DETERMINISTIC brief is re-run over. Wider than the three
#: adversarial ones above, because the deterministic layer is cheap to run and
#: the question here is coverage rather than temptation: does the generator set
#: still say the right thing about a quiet month, a shrinking book, a
#: deteriorating one, and a second client, after the mandate and gate landed.
#:
#: Every frame is derived from the same real canonical pair by deletion, by
#: scaling one existing column, or by relabelling a source portfolio id. No row
#: is authored, so the distributions stay the ones the pipeline produced.
def build_periods(tmp_root: Path, client_id: str) -> List[Scenario]:
    """The ten §12 periods, materialised under ``tmp_root``."""
    prior, current = _read(PRIOR_CSV), _read(CURRENT_CSV)
    reduced = _drop_book(prior, ARRIVING_BOOK)
    out: List[Scenario] = []

    def add(key: str, title: str, pri: pd.DataFrame, cur: pd.DataFrame,
            derivation: str, traps: Dict[str, Any],
            cid: str = client_id) -> None:
        root = tmp_root / key
        _write_run(root, cid, PRIOR_RUN, pri)
        _write_run(root, cid, CURRENT_RUN, cur)
        out.append(Scenario(
            key=key, period="monthly_funded", root=root, title=title,
            trap="", evidence_class=("A — real pipeline canonical, unmodified"
                                     if derivation == "none"
                                     else "C — purpose-built from real canonical"),
            derivation=derivation, traps={**traps, "client_id": cid}))

    # 1 — a quiet period: the same frame twice.
    add("quiet", "Nothing moved", current, current,
        "the same real canonical as both periods",
        {"movement": 0.0, "expect_material": False})

    # 2 — organic growth, the real month.
    add("organic_growth", "The real month", prior, current, "none",
        {"movement": round(_balance(current) - _balance(prior), 2),
         "expect_material": False, "expect_addition": False})

    # 3 — an acquisition month.
    add("acquisition", "A book arrives", reduced, current,
        f"prior frame has every {ARRIVING_BOOK!r} row deleted",
        {"expect_addition": True, "expect_acquisition_language": True})

    # 4 — an acquisition masking a deteriorating underlying book.
    #     The arrival is real; the incumbent book's balances are scaled down so
    #     the headline rises while the business underneath it shrinks.
    shrunk = _scale_balance(
        current, 0.90, current["source_portfolio_id"] != ARRIVING_BOOK)
    add("acquisition_masking_decline", "Headline up, underlying down",
        reduced, shrunk,
        f"as `acquisition`, and the incumbent books' balances scaled to 90% so "
        f"the underlying book falls while the headline rises",
        {"expect_addition": True, "expect_underlying_decline": True})

    # 5 — concentration warning: one loan grown until it dominates.
    top = current[BALANCE_COLUMNS[0]].idxmax()
    concentrated = _scale_balance(current, 6.0, current.index == top)
    add("concentration_warning", "One exposure dominates", prior, concentrated,
        "the largest existing loan's balance multiplied by six, so one "
        "exposure moves from 2.6% to roughly 14% of the book",
        {"expect_concentration_move": True})

    # 6 — missing concentration evidence. The funded brief carries no
    #     concentration claim unless a governed snapshot was resolved, and the
    #     guard that stops an unevaluated domain reading as a clean result
    #     lives one layer up in `trakt_notifications.sources`. What this period
    #     establishes is the half that belongs here: silence, not reassurance.
    add("no_concentration_config", "No approved limits exist", prior, current,
        "none; the deployment has no approved concentration configuration",
        {"expect_concentration_available": False,
         "expect_no_concentration_claim": True})

    # 7 — a shrinking book. Built by REMOVING loans from the later frame rather
    #     than by swapping the two periods over: the engine resolves period
    #     order from each tape's own cut-off date, so a swap is correctly undone
    #     and would have tested nothing.
    redeemed = current.iloc[::4]
    shrinking = current.drop(redeemed.index)
    add("book_shrinks", "The book contracts", current, shrinking,
        "every fourth loan removed from the later frame, as redemptions",
        {"movement": round(_balance(shrinking) - _balance(current), 2),
         "expect_negative_movement": True})

    # 8 — a disposal: a whole source portfolio leaves.
    add("disposal", "A book leaves", current, _drop_book(current, ARRIVING_BOOK),
        f"the current frame has every {ARRIVING_BOOK!r} row deleted, so that "
        f"book is disposed of",
        {"expect_disposal": True})

    # 9 — multi-portfolio scaling: a fourth and fifth source portfolio arrive
    #     at once, to show nothing is pinned to three books or to their names.
    scaled = current.copy()
    extra = current[current["source_portfolio_id"] == "spv1_sponsored"].copy()
    for i, (name, label) in enumerate((
            ("spv2_sponsored", "SPV2 Sponsored Securitisation"),
            ("jv_partner_book", "JV Partner Book")), start=1):
        clone = extra.copy()
        # Every column carrying the ORIGINAL book's identity is re-keyed, not
        # just the id. `source_portfolio_label` is governed and is what the
        # narrative names a book by, so a clone that kept it would have made
        # the engine look as though it had hard-coded a portfolio name when in
        # fact the fixture had.
        clone["source_portfolio_id"] = name
        clone["source_portfolio_label"] = label
        clone["portfolio_cohort"] = name
        clone["loan_identifier"] = clone["loan_identifier"].astype(str) + f"-X{i}"
        scaled = pd.concat([scaled, clone], ignore_index=True)
    add("multi_portfolio", "Two more books arrive at once", current, scaled,
        "two copies of an existing source portfolio re-keyed to new ids, so "
        "five source portfolios exist and two of them are new",
        {"expect_addition": True, "expect_addition_count": 2})

    # 10 — a second client, isolated. Same data, different client id: nothing
    #      about the first client may leak into it.
    add("second_client", "A different lender entirely", prior, current, "none",
        {"expect_isolation": True}, cid="client9")

    return out
