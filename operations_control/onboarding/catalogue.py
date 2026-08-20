"""operations_control.onboarding.catalogue — the onboarding information model.

Loads ``config/onboarding/field_catalogue.yaml`` and turns it into something the
wizard, the validator, the client information request and the generator all read
from. Nothing in this module knows about any particular client, asset class or
regime: it resolves whatever the catalogue declares.

Two things are resolved at load time:

* **Vocabularies with an ``owner``** are read from the module that actually owns
  them, so the catalogue can never drift from the platform. A catalogue that
  restated the dataset list would eventually disagree with the intake.
* **Regime sections** (``fields_from: regime_products``) are filled from
  ``config/regime/onboarding_standing_fields.yaml``, so a future regime becomes
  available to onboarding as a configuration change.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]

CATALOGUE_PATH = REPO / "config/onboarding/field_catalogue.yaml"
STANDING_FIELDS_PATH = REPO / "config/regime/onboarding_standing_fields.yaml"
ANNEX12_TEMPLATE_PATH = REPO / "config/regime/annex12_template.yaml"

#: Who supplies a value. Only ``client_supplied`` fields can appear on a client
#: information request — asking a client for an identifier Trakt mints, or for a
#: value inferred from their own upload, wastes their time and ours.
SOURCE_CLIENT = "client_supplied"
SOURCE_OPERATOR = "operator_supplied"
SOURCE_INFERRED = "inferred"
SOURCE_DERIVED = "derived"
SOURCE_DEFAULT = "trakt_default"
SOURCE_SYSTEM = "system_generated"
SOURCE_DELIVERY = "delivery_specific"

LEI_RE = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")
CCY_RE = re.compile(r"^[A-Z]{3}$")
COUNTRY_RE = re.compile(r"^[A-Z]{2}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
IDENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")
COLOUR_RE = re.compile(r"^#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class Field:
    """One question, or one value Trakt supplies."""

    key: str
    label: str
    scope: str = ""
    source: str = SOURCE_CLIENT
    type: str = "text"
    help: str = ""
    required: bool = False
    required_when: str = ""
    #: When this field is worth ASKING at all, as an expression over the other
    #: answers. Distinct from ``required_when``, which decides whether an asked
    #: question must be answered: a brand colour is never required, but it is
    #: only worth asking of a client who receives a branded report. Empty means
    #: "always worth asking".
    asked_when: str = ""
    validation: str = ""
    options: List[Dict[str, str]] = field(default_factory=list)
    options_from: str = ""
    default: Any = None
    derived_default: str = ""
    max_length: Optional[int] = None
    unique_across: str = ""
    consumers: List[str] = field(default_factory=list)
    writes_to: str = ""
    evidence_required: bool = False
    sensitive: bool = False
    amendable: bool = True
    regime_code: str = ""
    product: str = ""            # set for regime fields

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def asked_of_client(self) -> bool:
        return self.source == SOURCE_CLIENT

    @property
    def collected(self) -> bool:
        """Whether this field is validated and shown.

        Includes ``inferred``: inference DISCHARGES a requirement, it does not
        remove one. A field Trakt normally works out but could not — an asset
        class with no sample pack, a currency for an unlisted jurisdiction —
        must still be answered before approval, or the generated configuration
        would be quietly incomplete.
        """
        return self.source in (SOURCE_CLIENT, SOURCE_OPERATOR, SOURCE_DEFAULT,
                               SOURCE_INFERRED)

    @property
    def answered_by_trakt(self) -> bool:
        """Whether Trakt supplies this rather than a human. Used to count what
        onboarding actually asks."""
        return self.source in (SOURCE_INFERRED, SOURCE_DEFAULT, SOURCE_DERIVED,
                               SOURCE_SYSTEM)


@dataclass
class Section:
    key: str
    label: str
    help: str = ""
    repeatable: bool = False
    repeatable_key: str = ""
    item_label_field: str = ""
    min_items: int = 0
    derived_from: str = ""
    #: When a whole section's questions only make sense once something else
    #: exists, stated here rather than in code. Non-empty means the section is
    #: not part of the initial pack, and the text says what has to happen first.
    deferred_until: str = ""
    #: True when a client may leave the whole section blank. Declared here
    #: rather than inferred from every field being optional: "no field in this
    #: section is required" and "this section is offered, not asked" are
    #: different statements, and only the second should change how a client is
    #: addressed.
    optional_context: bool = False
    fields: List[Field] = field(default_factory=list)
    #: True when the section's fields come from the regime products rather than
    #: from the catalogue itself.
    from_regime: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {**{k: v for k, v in asdict(self).items() if k != "fields"},
                "fields": [f.to_dict() for f in self.fields]}

    def field(self, key: str) -> Optional[Field]:
        return next((f for f in self.fields if f.key == key), None)


class Catalogue:
    """The resolved information model."""

    def __init__(self, sections: List[Section], vocabularies: Dict[str, Any],
                 not_collected: List[Dict[str, Any]],
                 regime_products: Dict[str, Any], version: int = 1):
        self.sections = sections
        self.vocabularies = vocabularies
        self.not_collected = not_collected
        self.regime_products = regime_products
        self.version = version

    # -- lookup ------------------------------------------------------------ #
    def section(self, key: str) -> Optional[Section]:
        return next((s for s in self.sections if s.key == key), None)

    def field(self, section_key: str, field_key: str) -> Optional[Field]:
        s = self.section(section_key)
        return s.field(field_key) if s else None

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version,
                "sections": [s.to_dict() for s in self.sections],
                "vocabularies": self.vocabularies,
                "not_collected": self.not_collected,
                "regime_products": self.regime_products}

    # -- conditional requirement ------------------------------------------- #
    def is_asked(self, f: Field, answers: Dict[str, Any],
                 item: Optional[Dict[str, Any]] = None) -> bool:
        """Whether this field is worth putting in front of anyone at all.

        A field with no ``asked_when`` is always asked. One that declares a
        condition is asked only when the condition holds, so a follow-up
        question appears when its trigger is answered and not before.
        """
        if not f.asked_when:
            return True
        return evaluate(f.asked_when, answers, item)

    def is_required(self, f: Field, answers: Dict[str, Any],
                    item: Optional[Dict[str, Any]] = None) -> bool:
        """Whether ``f`` must be answered given what has been answered so far."""
        if f.required:
            return True
        if not f.required_when:
            return False
        return evaluate(f.required_when, answers, item)

    # -- the client information request ------------------------------------ #
    def outstanding_for_client(self, answers: Dict[str, Any]
                               ) -> List[Dict[str, Any]]:
        """Every field the CLIENT still needs to answer.

        Operator-supplied values, Trakt defaults, derived and inferred values are
        excluded: a checklist sent to a client should contain only things the
        client can actually tell us.
        """
        chosen = set((answers.get("reporting") or {}).get("products") or [])
        out: List[Dict[str, Any]] = []
        for section in self.sections:
            if section.repeatable:
                items = answers.get(section.key) or []
                for index, item in enumerate(items):
                    for f in section.fields:
                        if not f.asked_of_client:
                            continue
                        if not self.is_required(f, answers, item):
                            continue
                        if _present(item.get(f.key)):
                            continue
                        out.append(_request_row(section, f, index, item))
                continue

            block = answers.get(section.key) or {}
            for f in section.fields:
                if not f.asked_of_client:
                    continue
                # A regime the client does not receive asks them nothing. Its
                # fields are held per product, so the answer lives one level in.
                if section.from_regime:
                    if f.product and f.product not in chosen:
                        continue
                    value = (block.get(f.product) or {}).get(f.key)
                else:
                    value = block.get(f.key)
                if not self.is_required(f, answers, block):
                    continue
                if _present(value):
                    continue
                out.append(_request_row(section, f, None, block))
        return out


def _request_row(section: Section, f: Field, index: Optional[int],
                 item: Dict[str, Any]) -> Dict[str, Any]:
    label = f.label
    if index is not None:
        name = str(item.get(section.item_label_field) or "").strip()
        label = f"{label} — {name or f'item {index + 1}'}"
    return {"section": section.key, "section_label": section.label,
            "field": f.key, "label": label, "help": f.help,
            "index": index, "scope": f.scope,
            "evidence_required": f.evidence_required,
            "sensitive": f.sensitive}


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    if isinstance(value, bool):
        return True
    return bool(str(value).strip())


# --------------------------------------------------------------------------- #
# Conditional expressions
# --------------------------------------------------------------------------- #

_COND = re.compile(
    r"^\s*(?P<path>[\w.]+)\s+(?P<op>==|!=|in|contains)\s+(?P<value>.+?)\s*$")


def evaluate(expression: str, answers: Dict[str, Any],
             item: Optional[Dict[str, Any]] = None) -> bool:
    """Evaluate one ``required_when`` / ``derived_default`` expression.

    Deliberately tiny and total: an expression Trakt cannot parse is treated as
    NOT satisfied, so a malformed catalogue makes a field optional rather than
    making onboarding impossible.
    """
    expression = (expression or "").strip()
    if not expression:
        return False
    if expression == "always":
        return True
    m = _COND.match(expression)
    if not m:
        return False
    path, op, raw = m.group("path"), m.group("op"), m.group("value").strip()
    actual = _resolve(path, answers, item)
    expected = _literal(raw)

    if op == "==":
        return _scalar(actual) == _scalar(expected)
    if op == "!=":
        return _scalar(actual) != _scalar(expected)
    if op == "in":
        values = expected if isinstance(expected, list) else [expected]
        return _scalar(actual) in [_scalar(v) for v in values]
    if op == "contains":
        if isinstance(actual, (list, tuple, set)):
            return _scalar(expected) in [_scalar(v) for v in actual]
        return _scalar(expected) in str(actual or "")
    return False


def _resolve(path: str, answers: Dict[str, Any],
             item: Optional[Dict[str, Any]]) -> Any:
    """A bare name resolves against the current item first, then the answers."""
    if "." not in path and item is not None and path in item:
        return item[path]
    node: Any = answers
    for part in path.split("."):
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node


def _literal(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        return [p.strip().strip("'\"") for p in raw[1:-1].split(",") if p.strip()]
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw.strip("'\"")


def _scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    return str(value).strip() if value is not None else ""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _yaml(path: Path) -> Dict[str, Any]:
    if not Path(path).exists():
        return {}
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}


def _resolve_owner(owner: str) -> List[Dict[str, str]]:
    """Read a vocabulary from the module that owns it.

    Supports a dotted path to a dict (values become options, with ``label`` used
    when present) or to a sequence.
    """
    module_path, _, attr = owner.rpartition(".")
    try:
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
    except (ImportError, AttributeError):
        return []
    if isinstance(value, dict):
        return [{"value": k, "label": (v.get("label", k)
                                       if isinstance(v, dict) else str(v))}
                for k, v in sorted(value.items())]
    return [{"value": str(v), "label": str(v).replace("_", " ").capitalize()}
            for v in dict.fromkeys(value)]


def _resolve_vocabularies(raw: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, spec in (raw or {}).items():
        spec = spec or {}
        values = spec.get("values")
        if spec.get("owner"):
            values = _resolve_owner(str(spec["owner"]))
        out[name] = {"label": spec.get("label", name),
                     "help": spec.get("help", ""),
                     "owner": spec.get("owner", ""),
                     "values": values or []}
    return out


def _regime_products(path: Path = STANDING_FIELDS_PATH) -> Dict[str, Any]:
    """The per-product standing-field declaration, with enumerations resolved
    from the regime template so the wizard offers only accepted values."""
    spec = _yaml(path)
    products = spec.get("products") or {}
    template = _yaml(ANNEX12_TEMPLATE_PATH)
    for product in products.values():
        for f in product.get("fields") or []:
            ref = f.get("options_from")
            if not ref or not str(ref).startswith("annex12_template:"):
                continue
            node: Any = template
            for part in str(ref).split(":", 1)[1].split("."):
                node = node.get(part) if isinstance(node, dict) else None
            if isinstance(node, dict):
                f["options"] = [{"value": k, "label": v}
                                for k, v in node.items()]
    return products


#: The regime declaration predates the catalogue and calls a field's shape
#: ``format`` where the catalogue calls it ``type``. Mapped rather than renamed,
#: because ``format`` is the word the delivery rules use throughout and changing
#: it there would be a regime-package change for a naming preference.
_REGIME_FORMAT_TO_TYPE = {
    "lei": "lei", "country": "country", "email": "email", "text": "text",
    "enum": "enum", "boolean": "boolean", "yes_no": "yes_no",
    "date_or_nd": "date_or_nd", "multiline": "multiline",
}


def _build_field(raw: Dict[str, Any], vocabularies: Dict[str, Any],
                 product: str = "") -> Field:
    known = {f for f in Field.__dataclass_fields__}   # type: ignore[attr-defined]
    data = {k: v for k, v in (raw or {}).items() if k in known}
    fmt = str((raw or {}).get("format") or "")
    if fmt and "type" not in data:
        data["type"] = _REGIME_FORMAT_TO_TYPE.get(fmt, "text")
        data.setdefault("validation", data["type"])
    f = Field(**data)
    f.product = product
    # A declaration may list options as bare values (``["2021", "2024"]``) where
    # the value is its own label. Normalise, so everything downstream — the
    # renderer and the enum validator alike — sees one shape.
    f.options = [o if isinstance(o, dict)
                 else {"value": str(o), "label": str(o)}
                 for o in (f.options or [])]
    if f.options_from and not f.options:
        f.options = list((vocabularies.get(f.options_from) or {}).get("values")
                         or [])
    # A field with an option list validates against it unless told otherwise.
    if f.options and not f.validation and f.type in ("enum", "multi_enum"):
        f.validation = "enum"
    return f


def load(path: Path = CATALOGUE_PATH,
         standing_fields_path: Path = STANDING_FIELDS_PATH) -> Catalogue:
    doc = _yaml(path)
    vocabularies = _resolve_vocabularies(doc.get("vocabularies") or {})
    products = _regime_products(standing_fields_path)

    sections: List[Section] = []
    for raw in doc.get("sections") or []:
        section = Section(
            key=raw["key"], label=raw.get("label", raw["key"]),
            help=raw.get("help", ""),
            repeatable=bool(raw.get("repeatable")),
            deferred_until=str(raw.get("deferred_until") or "").strip(),
            optional_context=bool(raw.get("optional_context")),
            repeatable_key=raw.get("repeatable_key", ""),
            item_label_field=raw.get("item_label_field", ""),
            min_items=int(raw.get("min_items") or 0),
            derived_from=raw.get("derived_from", ""))
        if raw.get("fields_from") == "regime_products":
            section.from_regime = True
            for product_key, product in products.items():
                for spec in product.get("fields") or []:
                    section.fields.append(
                        _build_field({**spec, "scope": spec.get("scope")
                                      or "regime"},
                                     vocabularies, product=product_key))
        else:
            for spec in raw.get("fields") or []:
                section.fields.append(_build_field(spec, vocabularies))
        sections.append(section)

    return Catalogue(sections=sections, vocabularies=vocabularies,
                     not_collected=list(doc.get("not_collected") or []),
                     regime_products=products,
                     version=int(doc.get("version") or 1))


_CACHE: Optional[Catalogue] = None


def catalogue() -> Catalogue:
    """The process-wide catalogue. Loaded once; the file is governed, not hot."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load()
    return _CACHE


def reset_cache() -> None:
    """Test seam."""
    global _CACHE
    _CACHE = None


# --------------------------------------------------------------------------- #
# Value validation, by declared type / validation rule
# --------------------------------------------------------------------------- #

def validate_value(f: Field, value: Any) -> str:
    """Return an operator-safe problem, or "" when the value is acceptable."""
    if not _present(value):
        return ""
    text = str(value).strip()
    rule = f.validation or f.type

    if rule == "lei" and not LEI_RE.fullmatch(text.upper()):
        return f"{f.label} is not a valid 20-character identifier."
    if rule == "country" and not COUNTRY_RE.fullmatch(text.upper()):
        return f"{f.label} must be a two-letter country code."
    if rule == "currency" and not CCY_RE.fullmatch(text.upper()):
        return f"{f.label} must be a three-letter currency code."
    if rule == "email" and not EMAIL_RE.fullmatch(text):
        return f"{f.label} does not look like an email address."
    if rule == "identifier" and not IDENT_RE.fullmatch(text):
        return (f"{f.label} may use letters, numbers, hyphens and underscores "
                "only, and must start with a letter or number.")
    if rule == "colour" and not COLOUR_RE.fullmatch(text):
        return f"{f.label} must be a colour such as #1F3B5C."
    if rule == "date_or_nd" and not (DATE_RE.fullmatch(text)
                                     or text.upper().startswith("ND")):
        return f"{f.label} must be a date, or ND where none applies."
    if rule == "yes_no" and text.upper() not in ("Y", "N"):
        return f"{f.label} must be yes or no."
    if f.max_length and len(text) > int(f.max_length):
        return f"{f.label} is longer than {f.max_length} characters."
    if rule == "enum" and f.options:
        allowed = {str(o.get("value")) for o in f.options}
        chosen = value if isinstance(value, (list, tuple)) else [value]
        for item in chosen:
            if str(item).strip() and str(item).strip() not in allowed:
                return f"{f.label} is not one of the accepted values."
    return ""
