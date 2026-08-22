"""tests/analytical/client_fixture.py — a CLIENT-SHAPED weekly pipeline pack.

Why this is a second generator rather than a change to the first
---------------------------------------------------------------
``pipeline_fixture.py`` builds the twelve-week pack the V1 evidence is hashed
against. Every artefact in ``MANIFEST.json`` that names those extracts depends on
them staying byte-identical, so that generator is not touched. This one is
additive.

The defect it exists to fix
---------------------------
The twelve-week pack carries no COMPLETED case in ten of its twelve extracts,
then ten and thirty-two in the last two. That is not a book converting. Every
case in that pack enters INSIDE the published window — cohort 0 arrives at the
first extract — and a case needs ten weeks to complete, so only the first two
cohorts can ever reach COMPLETED before the window closes. There is no
population already in flight when the observation starts.

Three consequences followed from that one cause: completions bunched at the end,
**no withdrawal anywhere in the pack**, and dwell times compressed against the
governed lags. The empirical conversion rates Tranche B made the sole basis of
the forward forecast were derived from it.

What this generator does differently
------------------------------------
* **A warm-up.** Cases begin entering ``WARMUP_WEEKS`` before the first
  PUBLISHED extract, so a full funnel is already in flight when the window opens
  and completions occur from the first week onward.
* **Dwell times taken from the governed config.**
  ``config/client/pipeline_expected_funding.yaml`` says a case at KFI funds in 90
  days, at APPLICATION in 60, at OFFER in 30. So the funnel is KFI →(30d)→
  APPLICATION →(30d)→ OFFER →(30d)→ COMPLETED, with a per-case spread around it.
* **A funnel that reproduces the configured probabilities as GROUND TRUTH.**
  With ``P(complete | OFFER) = 0.75``, ``P(reach OFFER | APPLICATION) = 0.60`` and
  ``P(reach APPLICATION | KFI) = 4/9``, the implied stage-to-funding rates are
  0.75, 0.45 and 0.20 — the configured values exactly. A case that does not
  progress WITHDRAWS. That gives Tranche F something real to recover: an
  empirical estimator either finds the rate that generated the data or it does
  not.
* **Withdrawals throughout**, as the complement of progression, so the governed
  ``exclude_stages`` contract has something to exclude at every point in the
  window rather than only at the end.
* **A retention window on terminal cases.** An operational extract does not
  carry every case ever withdrawn since inception. Without a retention window
  the live pipeline stays stationary (250 cases in week 1, 249 in week 52) while
  the extract grows 882 -> 3,024 rows, so the live share falls 28.3% -> 8.2%
  purely from stock accumulation — a monotone drift any time-series question
  over the pipeline dataset would read as a real trend. Terminal cases are
  therefore carried ``TERMINAL_RETENTION_WEEKS`` after they terminate and then
  drop out. This is safe for the historical model, which unions case timelines
  across all snapshots rather than reading the last one: a case's terminal state
  is still observed in the snapshots inside its retention window.

Deterministic by construction: a fixed seed per book, no wall-clock, no
randomness that is not reproducible from ``(seed, case index)``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .pipeline_fixture import HEADER, BookProfile, ALDERBRIDGE, KESTRELMOOR

#: Weeks of intake BEFORE the first published extract. Must cover the longest
#: KFI-to-completion dwell (90 days, ~13 weeks) PLUS the terminal retention
#: window below, or the first published extract opens with a partially
#: accumulated terminal stock and the composition drifts to its plateau inside
#: the observation window — a smaller version of the defect this pack corrects.
WARMUP_WEEKS = 40

#: config/client/pipeline_expected_funding.yaml :: stage_days_to_fund.
#: Days from being AT a stage to funding.
STAGE_DAYS_TO_FUND = {"KFI": 90, "APPLICATION": 60, "OFFER": 30}
#: The dwell in each stage, derived from the lags above: 90-60, 60-30, 30-0.
STAGE_DWELL_DAYS = {"KFI": 30, "APPLICATION": 30, "OFFER": 30}

#: How long a COMPLETED or WITHDRAWN case is carried in the extract after it
#: terminates, before the operational feed drops it. ``None`` carries everything
#: forever, which is the un-retained shape this default exists to correct.
TERMINAL_RETENTION_WEEKS = 26

#: config/client/pipeline_expected_funding.yaml :: stage_probabilities.
#: These are the rates the generated data must EXHIBIT, not assumptions it makes.
TARGET_STAGE_RATE = {"KFI": 0.20, "APPLICATION": 0.45, "OFFER": 0.75}
#: The per-transition probabilities that produce them:
#:   P(complete|OFFER)                  = 0.75          = TARGET["OFFER"]
#:   P(reach OFFER|APPLICATION)  * 0.75 = 0.45  -> 0.60
#:   P(reach APPLICATION|KFI) * 0.60 * 0.75 = 0.20 -> 4/9
ADVANCE_PROBABILITY = {
    "KFI": TARGET_STAGE_RATE["KFI"] / TARGET_STAGE_RATE["APPLICATION"],        # 4/9
    "APPLICATION": TARGET_STAGE_RATE["APPLICATION"] / TARGET_STAGE_RATE["OFFER"],  # 0.60
    "OFFER": TARGET_STAGE_RATE["OFFER"],                                       # 0.75
}

_STATUS_TEXT = {"KFI": "KFI Issued", "APPLICATION": "Application",
                "OFFER": "Offer Issued", "COMPLETED": "Completed",
                "WITHDRAWN": "Withdrawn"}


@dataclass(frozen=True)
class DeclineProfile:
    """A material fall in NEW KFI intake over the final weeks of the window.

    The stress case for Tranche F: a forward forecast must reflect a falling
    intake rather than extrapolating an older run-rate, and the maturity logic
    must not read a thin recent cohort as a wave of non-conversions. Applied to
    NEW intake only — cases already in the funnel progress normally, because a
    lender's existing pipeline does not evaporate when origination slows.
    """

    #: Weeks before the end of the window over which the decline runs.
    weeks: int = 13
    #: Intake at the end of the decline, as a fraction of the stationary rate.
    final_fraction: float = 0.35


def _lcg(state: int) -> Tuple[int, float]:
    """The same tiny deterministic generator the V1 pack uses."""
    state = (1103515245 * state + 12345) % (2 ** 31)
    return state, state / float(2 ** 31)


@dataclass
class _Case:
    """One case's whole life, decided once, at creation."""

    number: int
    entered: date
    #: The furthest stage this case ever reaches.
    terminal: str
    #: Day offsets from ``entered`` at which each stage is reached.
    reached: Dict[str, int]
    #: The day the case leaves the funnel, for a terminal stage that ends it.
    exits_on: date
    advance: float
    value: float
    product: str
    rate: float
    region: str
    broker: str
    dob: str

    def stage_at(self, when: date, retention_weeks: int = None) -> str:
        """The stage an extract taken on ``when`` observes, or "" if absent.

        A terminal state is STICKY while the case is retained: one that completed
        in week 3 is still reported COMPLETED in week 12, exactly as an
        operational extract carries it. After ``retention_weeks`` past the
        terminal transition the feed drops the case and this returns "".
        """
        elapsed = (when - self.entered).days
        if elapsed < 0:
            return ""                               # not yet in the funnel
        stage = "KFI"
        terminal_on = None
        for candidate in ("APPLICATION", "OFFER", "COMPLETED", "WITHDRAWN"):
            offset = self.reached.get(candidate)
            if offset is not None and elapsed >= offset:
                stage = candidate
                if candidate in ("COMPLETED", "WITHDRAWN"):
                    terminal_on = offset
        if (retention_weeks is not None and terminal_on is not None
                and elapsed > terminal_on + retention_weeks * 7):
            return ""                               # dropped by the feed
        return stage


def _build_cases(profile: BookProfile, weeks: Sequence[date],
                 decline: DeclineProfile = None) -> List[_Case]:
    """Every case the pack contains, with its whole life decided up front."""
    state = profile.seed ^ 0x5EED
    cases: List[_Case] = []
    number = 1
    last_week = weeks[-1]
    for week in weeks:
        intake = profile.weekly_cases
        if decline is not None:
            weeks_left = (last_week - week).days / 7.0
            if 0 <= weeks_left < decline.weeks:
                # Linear from full intake at the start of the decline to
                # ``final_fraction`` at the last week.
                progress = 1.0 - (weeks_left / decline.weeks)
                factor = 1.0 - progress * (1.0 - decline.final_fraction)
                intake = max(1, int(round(profile.weekly_cases * factor)))
        for _ in range(intake):
            state, r1 = _lcg(state)
            state, r2 = _lcg(state)
            state, r3 = _lcg(state)
            state, r4 = _lcg(state)
            state, r5 = _lcg(state)
            state, adv_r = _lcg(state)
            state, app_r = _lcg(state)
            state, off_r = _lcg(state)
            state, jitter_r = _lcg(state)

            advance = max(28_000.0, round(profile.mean_advance
                                          + (r1 - 0.5) * 2 * profile.advance_spread, 2))
            value = round(advance * (profile.value_multiple * (0.85 + r2 * 0.4)), 2)
            product, base_rate = profile.products[int(r3 * len(profile.products))
                                                  % len(profile.products)]

            # The funnel. Each transition is an independent draw at the
            # configured probability; a case that fails one WITHDRAWS there.
            reached: Dict[str, int] = {}
            day = 0
            terminal = "KFI"
            # A per-case pace, +/- a fifth of a stage dwell, so completions are
            # spread across weeks rather than landing in lockstep.
            jitter = int((jitter_r - 0.5) * 2 * (STAGE_DWELL_DAYS["KFI"] // 5))
            for stage, draw in (("KFI", adv_r), ("APPLICATION", app_r), ("OFFER", off_r)):
                if draw > ADVANCE_PROBABILITY[stage]:
                    reached["WITHDRAWN"] = max(1, day + STAGE_DWELL_DAYS[stage] // 2 + jitter)
                    terminal = "WITHDRAWN"
                    break
                day += STAGE_DWELL_DAYS[stage] + jitter
                nxt = {"KFI": "APPLICATION", "APPLICATION": "OFFER",
                       "OFFER": "COMPLETED"}[stage]
                reached[nxt] = max(1, day)
                terminal = nxt
            exits = week + timedelta(days=max(reached.values()) if reached else 0)
            cases.append(_Case(
                number=number, entered=week, terminal=terminal, reached=reached,
                exits_on=exits, advance=advance, value=value, product=product,
                rate=round(base_rate + (r4 - 0.5) * 0.6, 3),
                region=profile.regions[int(r5 * len(profile.regions)) % len(profile.regions)],
                broker=profile.brokers[number % len(profile.brokers)],
                dob=f"{1946 + (number % 18)}-{1 + (number % 12):02d}-"
                    f"{1 + (number % 27):02d}"))
            number += 1
    return cases


def write_pack(root: Path, profile: BookProfile, *, weeks: int = 52,
               last_week: date, decline: DeclineProfile = None,
               retention_weeks: int = TERMINAL_RETENTION_WEEKS) -> List[Path]:
    """Write ``weeks`` PUBLISHED weekly extracts ending at ``last_week``.

    Intake begins ``WARMUP_WEEKS`` before the first published extract. Those
    earlier weeks are generated but never written: their purpose is to put a
    full funnel in flight before the observation window opens.

    ``retention_weeks`` is how long a terminated case is carried before the feed
    drops it; ``None`` carries every case forever.
    """
    published = [last_week - timedelta(weeks=(weeks - 1 - i)) for i in range(weeks)]
    warmup = [published[0] - timedelta(weeks=(WARMUP_WEEKS - i))
              for i in range(WARMUP_WEEKS)]
    cases = _build_cases(profile, list(warmup) + published, decline)

    written: List[Path] = []
    for week in published:
        folder = root / profile.client / "pipeline" / week.isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / ("M2L_KFI_and_Pipeline_"
                         f"{week.strftime('%Y_%m_%d')}.csv")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            for case in cases:
                stage = case.stage_at(week, retention_weeks)
                if not stage:
                    continue                        # not yet in the funnel
                entered = case.entered
                app = case.reached.get("APPLICATION")
                offer = case.reached.get("OFFER")
                done = case.reached.get("COMPLETED")
                app_date = (entered + timedelta(days=app)
                            if app is not None and stage in
                            ("APPLICATION", "OFFER", "COMPLETED") else "")
                offer_date = (entered + timedelta(days=offer)
                              if offer is not None and stage in ("OFFER", "COMPLETED") else "")
                funds_date = (entered + timedelta(days=done)
                              if done is not None and stage == "COMPLETED" else "")
                writer.writerow([
                    profile.company, "POOL_A",
                    f"{profile.client[:3].upper()}C{case.number:06d}",
                    f"KFI{case.number:07d}", case.broker, entered.isoformat(),
                    case.dob, "F", "", "",
                    f"{case.advance:.2f}", f"{case.value:.2f}", case.product,
                    f"{case.rate:.3f}", "Plan A", f"{case.advance:.2f}",
                    f"{case.advance * 1.3:.2f}", f"{case.advance * 1.2:.2f}",
                    case.region, "25.0", "1195.00", f"{case.value:.2f}",
                    "Home Improvements", "Property works",
                    _STATUS_TEXT[stage], "Approved",
                    app_date.isoformat() if app_date else "",
                    offer_date.isoformat() if offer_date else "",
                    funds_date.isoformat() if funds_date else "",
                    "", "", "Yes", "12", "0",
                ])
        written.append(path)
    return written


__all__ = ["write_pack", "DeclineProfile", "WARMUP_WEEKS", "STAGE_DAYS_TO_FUND",
           "STAGE_DWELL_DAYS", "TARGET_STAGE_RATE", "ADVANCE_PROBABILITY",
           "TERMINAL_RETENTION_WEEKS", "ALDERBRIDGE", "KESTRELMOOR"]
