"""One legal entity, every role the sentence gave it.

THE HALF OF THE DEFECT THAT SURVIVED THE FIRST FIX

"A and B" silently becoming "B" was fixed once, in the assembly of blanket
field hits. It looked fixed, because the sentence used to prove it —

    "ERE Funding Limited is the originator and the reporting entity."

— reads both roles. But an OPENING INSTRUCTION never looks like that. It names
the client first, and the moment it does, a different code path builds the
entity rows: ``_entities`` matches each role against the text and keys the rows
it makes by legal name. Its de-duplication was by NAME, so the first role an
entity matched claimed it and every later role for the same entity was skipped:

    "Onboard ERE Funding Limited. ERE Funding Limited is the originator
     and the reporting entity."
        -> roles: ["originator"]

One entity could hold exactly one role. That is not a rule anyone wrote down —
``entities.roles`` is a multi-valued field precisely because an originator is
routinely also the reporting entity, and the structural rule that requires an
originator exists alongside one that requires a reporting entity.

The consequence was the same as before and just as quiet: the reply confirmed
the role it kept, the operator read their own sentence back in it, and the
missing one surfaced later as a field that had gone optional.

De-duplication by name was right about the ROW — one entity is one row — and
wrong about the ROLES. Both now hold: a repeated name merges into the row that
already exists rather than being dropped.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent.interpretation import DeterministicInterpreter


def entities(text: str):
    steps = DeterministicInterpreter().interpret_instruction(text).steps
    return (steps.get("entities") or {}).get("entities") or []


def roles_of(text: str, name: str = "ERE Funding Limited"):
    for row in entities(text):
        if str(row.get("legal_name", "")).lower() == name.lower():
            return set(row.get("roles") or [])
    return set()


class TestAnOpeningInstructionKeepsEveryRole:
    """Role-first phrasing — the shape ``_ROLE_RE`` reads.

    "The originator is X" rather than "X is the originator". See
    :class:`TestNameFirstPhrasingIsStillOneRole` for why that distinction is
    load-bearing rather than pedantic.
    """

    @pytest.mark.parametrize("instruction", [
        "Onboard ERE Funding Limited. The originator is ERE Funding Limited. "
        "The reporting entity is ERE Funding Limited.",
        # Order reversed.
        "Onboard ERE Funding Limited. The reporting entity is ERE Funding "
        "Limited. The originator is ERE Funding Limited.",
        # With everything else an opening instruction carries around it.
        "Onboard ERE Funding Limited. It is a UK equity release lender. They "
        "need monthly MI and ESMA Annex 2 regime reporting. The originator is "
        "ERE Funding Limited. The reporting entity is ERE Funding Limited.",
    ])
    def test_both_roles_survive(self, instruction):
        assert roles_of(instruction) == {"originator", "reporting_entity"}

    def test_the_entity_is_still_one_row(self):
        """Merging roles must not split one company into two entities."""
        rows = entities(
            "Onboard ERE Funding Limited. The originator is ERE Funding "
            "Limited. The reporting entity is ERE Funding Limited.")
        names = [r.get("legal_name") for r in rows]
        assert names.count("ERE Funding Limited") == 1

    def test_a_third_role_is_kept_too(self):
        assert roles_of(
            "Onboard ERE Funding Limited. The originator is ERE Funding "
            "Limited. The servicer is ERE Funding Limited. The reporting "
            "entity is ERE Funding Limited."
        ) == {"originator", "servicer", "reporting_entity"}

    def test_a_single_role_is_still_a_single_role(self):
        assert roles_of(
            "Onboard ERE Funding Limited. The originator is ERE Funding "
            "Limited.") == {"originator"}


class TestSeparateCompaniesStaySeparate:
    """De-duplicating by name was right about ROWS. That must not regress.

    Merging roles into an existing row must never merge two companies, and
    must never move one company's role onto another's.
    """

    def test_two_named_entities_are_two_rows(self):
        rows = entities(
            "Onboard ERE Funding Limited. The originator is ERE Funding "
            "Limited. The servicer is Halewood Servicing Limited.")
        by_name = {r["legal_name"]: set(r["roles"]) for r in rows}
        assert by_name.get("ERE Funding Limited") == {"originator"}
        assert by_name.get("Halewood Servicing Limited") == {"servicer"}

    def test_a_shared_role_does_not_collapse_two_companies(self):
        rows = entities(
            "Onboard ERE Funding Limited. The originator is ERE Funding "
            "Limited. The reporting entity is ERE Funding Limited. The "
            "servicer is Halewood Servicing Limited.")
        by_name = {r["legal_name"]: set(r["roles"]) for r in rows}
        assert by_name["ERE Funding Limited"] == {"originator",
                                                  "reporting_entity"}
        assert by_name["Halewood Servicing Limited"] == {"servicer"}

    def test_the_client_is_still_proposed_as_originator(self):
        """Unchanged: naming no role at all still proposes the client's."""
        assert roles_of("Onboard ERE Funding Limited. UK equity release.") \
            == {"originator"}


class TestNameFirstPhrasingIsStillOneRole:
    """A KNOWN GAP, recorded rather than hidden.

    ``_ROLE_RE`` matches "the originator is X", not "X is the originator". With
    name-first phrasing no entity is matched at all, the originator fallback
    proposes the client's row, and the roles read from the instruction as a
    whole are discarded as already answered — so the second role is lost.

    The obvious repair — letting those roles override the proposal — was tried
    and reverted. Roles read that way carry no attribution, so
    "…the originator. Halewood Servicing Limited is the servicer." put
    ``servicer`` on the CLIENT's row: a false statement about a legal entity in
    place of an incomplete one. Closing this properly needs per-clause
    attribution.

    Until then the supported phrasing is role-first, and this test fails the
    day that stops being true so the gap is reviewed rather than rediscovered.
    """

    @pytest.mark.xfail(reason="name-first phrasing needs per-clause role "
                              "attribution; role-first reads both roles",
                       strict=True)
    def test_name_first_keeps_both_roles(self):
        assert roles_of(
            "Onboard ERE Funding Limited. ERE Funding Limited is the "
            "originator and the reporting entity."
        ) == {"originator", "reporting_entity"}

    def test_but_it_never_attributes_another_companys_role_to_the_client(self):
        """The reason the repair was reverted. This is the line that matters."""
        assert "servicer" not in roles_of(
            "Onboard ERE Funding Limited. ERE Funding Limited is the "
            "originator. Halewood Servicing Limited is the servicer.")
