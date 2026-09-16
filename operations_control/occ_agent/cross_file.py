"""operations_control.occ_agent.cross_file — where a pack's files disagree.

WHAT THIS EXISTS FOR. A delivery is several files, and the same fact usually
appears in more than one of them under different names. In one real equity
release pack the current interest rate arrives three times — ``Loan Interest
Rate`` in the loan tape, ``Current Interest Rate`` in the payments tape,
``Interest Rate`` in the property tape — and the origination date, the
redemption date, the outstanding balance and the pool identifier all arrive
twice.

That is not a problem. It is free reconciliation: when the files agree, the
second one corroborates the first, and the central tape builder records the
field as ``validated``. It becomes a problem only when they DISAGREE, and then
the builder raises a blocking ``value_conflict`` gap whose remedy is an
approved source-precedence rule — "for this field, the loan tape wins".

The gap this closes is WHEN you find out. The rehearsal maps every file in the
pack and reports what it made of each, but it compares nothing ACROSS them:
duplicates were only ever detected within the one file the canonical tape is
built from. So a pack whose three files disagree about an interest rate looked
clean in rehearsal and raised a blocking gap after activation — the one place
the whole point of a rehearsal is to have looked first.

WHAT IT DELIBERATELY DOES NOT DO. It does not block, and it does not choose.
The remedy for a real conflict is a source-precedence rule, which is a governed
artefact the rehearsal has no machinery to record — so blocking here would trap
an operator in front of a question they cannot answer from this screen. It
reports, name by name, with examples and with the remedy named, and lets the
run continue.

Comparison semantics are the platform's own —
``engine.onboarding_agent.central_tape_builder`` — not a second opinion. If the
rehearsal called two values equal and the builder later called them different,
the rehearsal would be worse than useless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: The canonical field whose value joins one file's rows to another's. Without
#: it in both files there is nothing to compare row-wise, and the pair is
#: reported as present-in-both rather than as agreeing or disagreeing.
JOIN_FIELD = "loan_identifier"

#: How many disagreeing loans to quote. Enough to recognise the shape of the
#: problem — a systematic offset reads differently from three bad rows — and
#: not so many that the finding becomes a data dump.
EXAMPLE_LIMIT = 3

#: Tiers whose mapping is trusted enough to compare on. A column matched below
#: the confidence threshold is not yet claimed to BE the field, so comparing it
#: would manufacture disagreements out of the mapper's own uncertainty.
def _claimed(row: Dict[str, Any]) -> bool:
    return bool(row.get("canonical_field")) and not row.get("note")


@dataclass
class FieldSources:
    """One canonical field, and every file claiming to carry it."""

    canonical_field: str
    #: ``[(file, column)]`` in the order the report lists them.
    sources: List[Tuple[str, str]] = field(default_factory=list)
    #: Loans where every source agreed.
    agreed: int = 0
    #: Loans where at least two sources differed.
    differed: int = 0
    #: ``[(loan, [(file, column, value)])]`` for the first few disagreements.
    examples: List[Tuple[str, List[Tuple[str, str, str]]]] = field(
        default_factory=list)
    #: Why no row-wise comparison was possible, when none was.
    not_compared: str = ""
    #: Sources that were LEFT OUT of the comparison — unreadable, or with no
    #: loan identifier read in them. Held separately because saying "these
    #: three agree" when only two were checked is the kind of quiet
    #: overstatement this module exists to prevent.
    uncompared_sources: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def files(self) -> List[str]:
        out: List[str] = []
        for file_name, _column in self.sources:
            if file_name not in out:
                out.append(file_name)
        return out

    @property
    def conflicted(self) -> bool:
        return self.differed > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_field": self.canonical_field,
            "sources": [{"source_file": f, "source_column": c}
                        for f, c in self.sources],
            "files": self.files,
            "agreed": self.agreed,
            "differed": self.differed,
            "conflicted": self.conflicted,
            "not_compared": self.not_compared,
            "uncompared_sources": [{"source_file": f, "source_column": c}
                                   for f, c in self.uncompared_sources],
            "examples": [
                {"loan": loan,
                 "values": [{"source_file": f, "source_column": c,
                             "value": v} for f, c, v in values]}
                for loan, values in self.examples],
        }

    def sentence(self) -> str:
        """What an operator needs to read, in one line."""
        where = " and ".join(f"'{c}' in {f}" for f, c in self.sources)
        name = self.canonical_field.replace("_", " ")
        if self.not_compared:
            return (f"{name} is carried by {where}. {self.not_compared} "
                    "Whether they agree is settled when the delivery is built.")
        # Never claim more was checked than was. A field carried by three
        # files, two of which could be compared, must not read as three
        # agreeing.
        aside = ""
        if self.uncompared_sources:
            left = " and ".join(f"'{c}' in {f}"
                                for f, c in self.uncompared_sources)
            aside = (f" {left} could not be checked against them, because the "
                     "loan identifier was not read there.")
        if self.conflicted:
            return (f"{name} DIFFERS between {where}: {self.differed} of "
                    f"{self.agreed + self.differed} loans disagree. Until a "
                    "source is named as the one to believe, the delivery will "
                    f"stop on this.{aside}")
        checked = " and ".join(f"'{c}' in {f}" for f, c in self.compared_sources) \
            if self.uncompared_sources else where
        return (f"{name} is carried by {where}, and {checked} agree on all "
                f"{self.agreed} loans.{aside}")

    @property
    def compared_sources(self) -> List[Tuple[str, str]]:
        left_out = set(self.uncompared_sources)
        return [s for s in self.sources if s not in left_out]


def _values_match(a: Any, b: Any, tol: float = 0.01) -> bool:
    """The platform's own comparison, kept identical on purpose.

    Mirrors ``central_tape_builder._values_match``: numbers within a relative
    tolerance, everything else as trimmed strings, and a blank never matches.
    A rehearsal that judged equality differently from the builder would pass
    packs the builder then stops — which is worse than not looking at all.
    """
    if a is None or b is None:
        return False
    sa, sb = str(a).strip(), str(b).strip()
    if sa == "" or sb == "":
        return False
    try:
        fa, fb = float(sa), float(sb)
        denom = max(abs(fa), abs(fb), 1.0)
        return abs(fa - fb) / denom <= tol
    except (ValueError, TypeError):
        return sa == sb


def _blank(value: Any) -> bool:
    text = "" if value is None else str(value).strip()
    return text == "" or text.lower() in ("nan", "nat", "none", "<na>")


def claimed_by_file(report: List[Dict[str, Any]]
                    ) -> Dict[str, List[Tuple[str, str]]]:
    """``{canonical field: [(file, column)]}`` for every field a file claims.

    Read from the mapping report rather than from the frames, so the rehearsal
    compares exactly what it told the operator it had read.
    """
    out: Dict[str, List[Tuple[str, str]]] = {}
    for row in report or []:
        if not _claimed(row):
            continue
        canonical = str(row["canonical_field"])
        pair = (str(row.get("source_file") or ""),
                str(row.get("source_column") or ""))
        if pair not in out.setdefault(canonical, []):
            out[canonical].append(pair)
    return out


def shared_fields(report: List[Dict[str, Any]]
                  ) -> Dict[str, List[Tuple[str, str]]]:
    """Only the fields more than one FILE claims.

    Two columns of the same file claiming one field is a different problem —
    an ambiguity the rehearsal already raises as a decision, because the
    operator can settle it there and then.
    """
    out: Dict[str, List[Tuple[str, str]]] = {}
    for canonical, sources in claimed_by_file(report).items():
        if len({f for f, _c in sources}) > 1:
            out[canonical] = sources
    return out


def _key_column(report: List[Dict[str, Any]], file_name: str) -> str:
    """The column carrying the loan identifier in one file, if any."""
    for row in report or []:
        if (str(row.get("source_file") or "") == file_name
                and str(row.get("canonical_field") or "") == JOIN_FIELD
                and _claimed(row)):
            return str(row.get("source_column") or "")
    return ""


def _indexed(frame: Any, key_column: str, value_column: str
             ) -> Dict[str, Any]:
    """``{loan: value}`` for one file's column. Blanks are left out."""
    out: Dict[str, Any] = {}
    if frame is None or key_column not in frame.columns \
            or value_column not in frame.columns:
        return out
    for key, value in zip(frame[key_column], frame[value_column]):
        if _blank(key) or _blank(value):
            continue
        out.setdefault(str(key).strip(), value)
    return out


def compare(report: List[Dict[str, Any]],
            frames: Dict[str, Any]) -> List[FieldSources]:
    """Every field more than one file carries, and whether they agree.

    ``frames`` is ``{file name: DataFrame}``. A file with no frame — one that
    could not be read — is simply not compared; it was already reported as
    unreadable where that happened.

    Ordered so the disagreements come first: an operator reading this is
    looking for what will stop the delivery, not for a census.
    """
    out: List[FieldSources] = []
    for canonical, sources in shared_fields(report).items():
        found = FieldSources(canonical_field=canonical, sources=list(sources))
        usable = [(f, c) for f, c in sources
                  if frames.get(f) is not None and _key_column(report, f)]
        found.uncompared_sources = [s for s in sources if s not in set(usable)]
        if len({f for f, _c in usable}) < 2:
            found.uncompared_sources = []
            found.not_compared = (
                "They could not be compared row by row, because the loan "
                "identifier was not read in every one of them.")
            out.append(found)
            continue

        indexes = {(f, c): _indexed(frames[f], _key_column(report, f), c)
                   for f, c in usable}
        loans: List[str] = []
        for values in indexes.values():
            for loan in values:
                if loan not in loans:
                    loans.append(loan)
        for loan in loans:
            present = [(f, c, values[loan]) for (f, c), values in
                       indexes.items() if loan in values]
            if len(present) < 2:
                continue        # only one file has this loan; nothing to check
            first = present[0][2]
            if all(_values_match(first, value) for _f, _c, value in present[1:]):
                found.agreed += 1
                continue
            found.differed += 1
            if len(found.examples) < EXAMPLE_LIMIT:
                found.examples.append(
                    (loan, [(f, c, str(v)) for f, c, v in present]))
        out.append(found)
    return sorted(out, key=lambda r: (not r.conflicted, r.canonical_field))


def findings(comparisons: List[FieldSources]) -> List[Dict[str, Any]]:
    """The conflicts alone, in the shape the run records a finding in.

    Not decisions. The remedy is a source-precedence rule — a governed
    artefact this surface cannot write — so presenting these as answerable
    questions would be asking an operator for something they cannot give here.
    They are findings, with the remedy named.
    """
    return [{
        "canonical_field": row.canonical_field,
        "severity": "conflict",
        "summary": row.sentence(),
        "remedy": ("Name which file is the one to believe for this field, "
                   "before the delivery is built."),
        **row.to_dict(),
    } for row in comparisons if row.conflicted]
