"""trakt_core.entity — assembling a credit object from the flat canonical row.

The canonical tape stays the analytical truth. This module is a *reading* of it:
it knows that ``current_valuation_amount`` belongs to a valuation of a
collateral of a loan, so Trakt can hand an agent a structured credit object
instead of 130 loose columns.

    loan
     ├── borrower       1..1 on the current tape shape
     ├── collateral     1..1
     │    └── valuation 0..n   <- the only repeating entity
     ├── contract       attribute group
     └── performance    attribute group

Three things this deliberately is not:

* **not a second store.** Assembly happens per request from the row already in
  memory. There is no materialised object database to keep in step, and no
  second answer to "what is the balance".
* **not a normalisation.** No canonical field moves, no pipeline changes, and a
  field claimed by no entity simply does not appear — it is still on the tape
  and still reachable through the field projection on ``get_loans``.
* **not an ontology.** Six entities, declared in
  ``config/system/entity_model.yaml`` because the canonical fields genuinely
  support them, and no more.

Valuation history comes from the narrow sidecar
(:mod:`trakt_core.valuation`); where a deployment has no sidecar the assembler
still emits the observations the tape itself carries — the current selected
valuation and the original — so a loan object always has a truthful, if short,
valuation history rather than an empty one.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import config_cache
from .errors import ErrorCode, TraktError
from .valuation import (
    TYPE_FULL,
    TYPE_INDEXED,
    TYPE_PURCHASE_PRICE,
    ValuationObservation,
    make_valuation_id,
)

DEFAULT_ENTITY_MODEL_CONFIG = "config/system/entity_model.yaml"
ENTITY_MODEL_CONFIG_ENV = "TRAKT_ENTITY_MODEL_CONFIG"

#: Emitted as sections of the assembled object, in this order.
SECTION_ORDER = ("loan", "borrower", "collateral", "contract", "performance")


def _present(value: Any) -> bool:
    """Is there actually a value here?

    ``NaN`` is the trap: a pandas frame fills an absent column with it, and it is
    neither ``None`` nor falsy — ``bool(float('nan'))`` is ``True``. Left
    unguarded, a loan with no collateral column resolves a collateral id of
    ``"nan"`` and a loan with no valuation grows an observation whose amount is
    ``NaN``. Both are worse than the absence they came from, because they look
    like data.
    """
    if value is None or value == "":
        return False
    return not (isinstance(value, float) and value != value)


@dataclass(frozen=True)
class EntitySpec:
    """One declared entity or attribute group."""

    name: str
    key: Optional[str] = None
    key_role: str = "key"
    key_fallback: Optional[str] = None
    parent: Optional[str] = None
    parent_key: Optional[str] = None
    cardinality: str = "1..1"
    repeating: bool = False
    attribute_group: bool = False
    description: str = ""
    fields: Tuple[str, ...] = ()
    patterns: Tuple[str, ...] = ()
    #: Valuation only: tape columns describing an observation.
    observation_fields: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def claims(self, canonical_field: str) -> bool:
        if canonical_field in self.fields:
            return True
        return any(re.match(p, canonical_field) for p in self.patterns)


@dataclass(frozen=True)
class EntityModel:
    """The declared model, with a field -> entity index resolved once."""

    entities: Dict[str, EntitySpec] = field(default_factory=dict)
    field_owner: Dict[str, str] = field(default_factory=dict)
    configured: bool = False

    def entity(self, name: str) -> Optional[EntitySpec]:
        return self.entities.get(name)

    def owner_of(self, canonical_field: str) -> Optional[str]:
        owner = self.field_owner.get(canonical_field)
        if owner:
            return owner
        for name, spec in self.entities.items():
            if spec.patterns and spec.claims(canonical_field):
                return name
        return None

    def describe(self) -> Dict[str, Any]:
        """The published shape, for an agent asking what objects exist."""
        return {
            "configured": self.configured,
            "entities": {
                name: {
                    "key": spec.key,
                    "key_role": spec.key_role,
                    "parent": spec.parent,
                    "cardinality": spec.cardinality,
                    "repeating": spec.repeating,
                    "attribute_group": spec.attribute_group,
                    "description": spec.description,
                    "field_count": len(spec.fields),
                }
                for name, spec in sorted(self.entities.items())
            },
        }


def load_entity_model(config_path: Optional[str | Path] = None) -> EntityModel:
    """Load the entity model, cached on content identity.

    An absent file yields an unconfigured model and the assembler falls back to
    a flat record — the pre-Sprint-2 behaviour — rather than guessing a shape.

    A field claimed by two entities is a **hard failure**: an object where the
    same value appears in two sections is ambiguous about which is authoritative,
    and that is worth failing config review over.
    """
    path = Path(config_path
                or os.environ.get(ENTITY_MODEL_CONFIG_ENV)
                or DEFAULT_ENTITY_MODEL_CONFIG)

    def _load() -> EntityModel:
        if not path.exists():
            return EntityModel(configured=False)
        try:
            import yaml
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 - a bad model file falls back to flat
            return EntityModel(configured=False)

        entities: Dict[str, EntitySpec] = {}
        owner: Dict[str, str] = {}
        for name, spec in (raw.get("entities") or {}).items():
            spec = spec or {}
            fields = tuple(str(f) for f in (spec.get("fields") or ()))
            observation_fields = {
                k: dict(v) for k, v in spec.items()
                if k.endswith("_observation_fields") and isinstance(v, Mapping)}
            entities[str(name)] = EntitySpec(
                name=str(name),
                key=spec.get("key"),
                key_role=str(spec.get("key_role") or "key"),
                key_fallback=spec.get("key_fallback"),
                parent=spec.get("parent"),
                parent_key=spec.get("parent_key"),
                cardinality=str(spec.get("cardinality") or "1..1"),
                repeating=bool(spec.get("repeating", False)),
                attribute_group=bool(spec.get("attribute_group", False)),
                description=str(spec.get("description") or "").strip(),
                fields=fields,
                patterns=tuple(str(p) for p in (spec.get("patterns") or ())),
                observation_fields=observation_fields)
            for f in fields:
                if f in owner and owner[f] != str(name):
                    return EntityModel(configured=False)   # ambiguous: refuse
                owner[f] = str(name)

        return EntityModel(entities=entities, field_owner=owner, configured=True)

    return config_cache.get_or_load("entity_model", path, _load)


# --------------------------------------------------------------------------- #
# Collateral identity — resolved ONCE, for every reader
# --------------------------------------------------------------------------- #
def resolve_collateral_id(row: Mapping[str, Any],
                          model: EntityModel) -> Optional[str]:
    """The collateral identity for one canonical row.

    There is exactly one resolver because ``valuation_id`` is derived from this
    value: the assembled loan object and the ``explain_values`` derivation both
    address observations by that id, and if the two resolved the collateral
    differently the same valuation would carry two identities — an explanation
    citing ``val_a1b2`` for a valuation the loan object showed as ``val_c3d4``.

    Order: the declared key, then ``collateral_id`` (what the servicer extracts
    and the demo generator emit before the consolidator renames it), then the
    declared fallback. A tape carrying either column resolves the same way.
    """
    spec = model.entity("collateral") if model.configured else None
    if spec is None:
        candidates = ("collateral_identifier", "collateral_id", "loan_identifier")
    else:
        candidates = tuple(c for c in (spec.key, "collateral_id", spec.key_fallback)
                           if c)
    seen: set = set()
    for column in candidates:
        if column in seen:
            continue
        seen.add(column)
        value = row.get(column)
        if _present(value):
            return str(value)
    return None


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _observations_from_row(row: Mapping[str, Any], collateral_id: str,
                           spec: EntitySpec) -> List[ValuationObservation]:
    """The valuation observations the TAPE itself carries.

    A deployment with no sidecar still gets a truthful history — typically the
    current selected valuation plus the original, and the indexed value where
    one is supplied. Short, but real: better than an empty list, and better than
    pretending the tape's single current value is the whole story.
    """
    out: List[ValuationObservation] = []
    mapping = (
        (TYPE_FULL, spec.observation_fields.get("current_observation_fields") or {}),
        (TYPE_PURCHASE_PRICE, spec.observation_fields.get("original_observation_fields") or {}),
        (TYPE_INDEXED, spec.observation_fields.get("indexed_observation_fields") or {}),
    )
    for valuation_type, columns in mapping:
        amount_col = columns.get("amount")
        if not amount_col:
            continue
        amount = row.get(amount_col)
        if not _present(amount):
            continue
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            continue
        if amount != amount:                       # NaN that survived the cast
            continue
        date_col = columns.get("valuation_date")
        date_value = row.get(date_col) if date_col else None
        valuation_date = str(date_value)[:10] if _present(date_value) else None
        method_col = columns.get("methodology")
        method_value = row.get(method_col) if method_col else None
        source = str(method_value) if _present(method_value) else "canonical tape"
        currency = row.get("collateral_currency")
        if not _present(currency):
            currency = row.get("exposure_currency_denomination")
        out.append(ValuationObservation(
            valuation_id=make_valuation_id(collateral_id, valuation_type,
                                           valuation_date, source),
            collateral_id=collateral_id,
            valuation_type=valuation_type,
            amount=amount,
            currency=currency if _present(currency) else None,
            valuation_date=valuation_date,
            effective_date=valuation_date,
            source=source,
            provenance_ref=amount_col))
    return out


def assemble_loan(row: Mapping[str, Any], model: EntityModel, *,
                  observations: Optional[Sequence[ValuationObservation]] = None,
                  include: Sequence[str] = (),
                  ) -> Dict[str, Any]:
    """Assemble ONE canonical row into a structured credit object.

    ``observations`` are the sidecar's valuation history for this collateral.
    When none are supplied the tape's own observations are used, so the object
    is never silently empty.

    ``include`` opts in to repeating children — ``["valuations"]``. Absent, the
    object carries scalar sections only, because a bounded response matters more
    than completeness an agent did not ask for.
    """
    if not model.configured:
        # No model: the flat record, exactly as before. Never a guessed shape.
        return {"loan": dict(row)}

    sections: Dict[str, Dict[str, Any]] = {name: {} for name in SECTION_ORDER}
    unassigned: Dict[str, Any] = {}
    for canonical_field, value in row.items():
        owner = model.owner_of(canonical_field)
        if owner in sections:
            sections[owner][canonical_field] = value
        elif owner == "valuation":
            continue                      # surfaced as observations, not scalars
        else:
            unassigned[canonical_field] = value

    loan_id = row.get("loan_identifier")
    collateral_id = resolve_collateral_id(row, model)

    out: Dict[str, Any] = {}
    for name in SECTION_ORDER:
        spec = model.entity(name)
        if spec is None:
            continue
        block = sections[name]
        if name == "collateral" and collateral_id is not None:
            block = {"collateral_id": collateral_id, **block}
        out[name] = block

    if "valuations" in include:
        valuation_spec = model.entity("valuation")
        history = list(observations or ())
        if not history and valuation_spec is not None and collateral_id is not None:
            history = _observations_from_row(row, str(collateral_id), valuation_spec)
        out.setdefault("collateral", {})["valuations"] = [
            o.to_dict() for o in sorted(
                history, key=lambda o: (o.valuation_date or "", o.valuation_id),
                reverse=True)]

    if unassigned:
        # Never dropped silently: a field the model does not claim is still a
        # governed value, and an agent that asked for it should receive it.
        out["other"] = unassigned
    out["_entity_model"] = {"configured": True,
                            "loan_id": loan_id,
                            "collateral_id": collateral_id}
    return out
