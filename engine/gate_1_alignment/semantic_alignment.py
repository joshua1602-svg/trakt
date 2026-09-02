#!/usr/bin/env python3
"""
semantic_alignment.py

Purpose (locked contract):
- Ingest a raw loan tape (CSV/XLSX)
- Map columns to the canonical field registry (core + extensions)
- Emit a FULL canonical dataset (truth set; no ND padding)
- Emit mapping diagnostics (mapping report + unmapped headers)

Optional:
- If --regimes is provided, also emit regime projections (schema subset only).
  NOTE: Regime padding (ND insertion) belongs in the downstream ESMA projection step,
  not in this mapper.
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz, process
import yaml
import json

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Matching thresholds (conservative; avoid false positives)
JACCARD_THRESHOLD = 0.85
FUZZ_TOKEN_SET_THRESHOLD = 88
FUZZ_NORM_THRESHOLD = 92

# Mapping method precedence (higher wins) to resolve duplicate collisions deterministically
METHOD_RANK = {
    # An operator-approved decision outranks every automated method. It is the
    # only entry here that is not a matching technique: it is a governance fact.
    "operator_approved": 7,
    "exact": 6,
    "normalized": 5,
    "alias": 4,
    "token_set": 3,
    "fuzz_token_set": 2,
    "fuzz_ratio_norm": 1,
    "unmapped": 0,
    "empty": 0,
}
LOW_CONFIDENCE_THRESHOLD = 0.95


# ------------------------------------------------------------------
# NORMALISATION & TOKENISATION (shared by aliases + headers)
# ------------------------------------------------------------------

ABBREV_MAP = {
    "valn": "valuation",
    "val": "valuation",
    "prin": "principal",
    "princ": "principal",
    "bal": "balance",
    "baln": "balance",
    "curr": "current",
    "cur": "current",
    "orig": "original",
    "geo": "geographic",
    "ctry": "country",
    "verif": "verification",
    "estab": "establishment",
    "mgn": "margin",
    "arrs": "arrears",
    "dt": "date",
    "amt": "amount",
    "amnt": "amount",
    "no": "number",
    "mat": "maturity",
}

STOPWORDS = {"loan", "account", "field", "info", "information", "data"}


def tokenize(text: str) -> List[str]:
    s = str(text).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    raw_tokens = s.split()

    tokens: List[str] = []
    for tok in raw_tokens:
        if tok in ABBREV_MAP:
            tokens.append(ABBREV_MAP[tok])
            continue

        expanded_split = False
        for abbr, full in ABBREV_MAP.items():
            if tok.startswith(abbr) and len(tok) > len(abbr):
                tail = tok[len(abbr) :]
                tokens.append(full)
                tokens.append(ABBREV_MAP.get(tail, tail))
                expanded_split = True
                break
        if expanded_split:
            continue

        tokens.append(tok)

    tokens = [t for t in tokens if t and t not in STOPWORDS]
    return tokens


def normalise_name(text: str) -> str:
    tokens = tokenize(text)
    return "".join(sorted(tokens))


# ------------------------------------------------------------------
# REGISTRY + ALIASES
# ------------------------------------------------------------------


def load_field_registry(registry_path: Path) -> dict:
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if "fields" not in data or not isinstance(data["fields"], dict):
        raise ValueError(f"Registry file does not contain a 'fields:' mapping: {registry_path}")
    return data


def select_registry_fields(registry: dict, portfolio_type: str) -> List[str]:
    """
    Return the full canonical superset for mapping:
    - all fields where portfolio_type is 'common'
    - plus fields matching the requested portfolio_type (case-insensitive)
    """
    pt = (portfolio_type or "").strip().lower()
    selected: List[str] = []
    for fname, meta in registry["fields"].items():
        fpt = str(meta.get("portfolio_type", "")).strip().lower()
        if fpt == "common" or fpt == pt:
            selected.append(fname)

    # deterministic ordering: keep registry insertion order if possible
    return selected


def select_regime_fields(registry: dict, regime: str, portfolio_type: str) -> List[str]:
    """
    Return the field subset that has a regime mapping for the given regime,
    constrained to common + portfolio_type.
    """
    regime_key = str(regime).strip()
    base = set(select_registry_fields(registry, portfolio_type))
    selected: List[str] = []
    for fname in registry["fields"].keys():
        if fname not in base:
            continue
        meta = registry["fields"][fname] or {}
        rm = meta.get("regime_mapping", {}) or {}
        if regime_key in rm:
            selected.append(fname)
    return selected


def load_aliases_from_dir(aliases_dir: Path) -> Dict[str, str]:
    """
    Load aliases_*.yaml into a single map: normalised_alias -> canonical_field.

    Supports BOTH YAML shapes:
      1) canonical_field: { aliases: [..] }
      2) canonical_field: [..]   (shorthand list of aliases)
    """
    alias_map: Dict[str, str] = {}
    if not aliases_dir.exists():
        return alias_map

    for yaml_file in aliases_dir.glob("aliases_*.yaml"):
        try:
            data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                logging.warning(f"Skipping {yaml_file}: expected mapping at top-level.")
                continue

            for canon, meta in data.items():
                # meta can be dict (with key 'aliases') or a list (shorthand)
                if isinstance(meta, list):
                    aliases = meta
                else:
                    aliases = (meta or {}).get("aliases", []) or []

                for a in aliases:
                    norm = normalise_name(a)
                    if norm and norm not in alias_map:
                        alias_map[norm] = canon
        except Exception as e:
            logging.warning(f"Failed to load {yaml_file}: {e}")

    logging.info(f"Loaded {len(alias_map)} alias entries from {aliases_dir}")
    return alias_map


# ------------------------------------------------------------------
# APPROVED MAPPING CONTRACT (tier 0)
# ------------------------------------------------------------------
#
# The mapping semantics of a client's source columns are decided ONCE, during
# onboarding, by the mapping model and a human operator. The Operations Control
# Centre records each approval as a versioned rule and materialises the decisions
# as ``12_approved_mapping_overrides.yaml``. Until this layer existed the mapper
# had no way to be told any of that: the only client-specific channel was the
# alias overlay, which sits BELOW exact and normalised canonical-name matching,
# so an approved decision was silently overridable by automated name matching.
#
# An approved decision is not a matching technique competing with the others. It
# is the answer. It is therefore consulted first, and nothing below it can win.

#: Contract document versions this loader understands.
SUPPORTED_APPROVED_CONTRACT_VERSIONS = (1,)

#: Groups inside the contract, MOST authoritative first. ``user_overrides`` are
#: the operator's own corrections; ``approved_high_confidence_mappings`` are the
#: machine proposals approved wholesale alongside them.
_APPROVED_GROUPS = ("user_overrides", "approved_high_confidence_mappings")


def approved_column_key(column: str) -> str:
    """Match key for an approved decision: case and separators only.

    Deliberately NOT :func:`normalise_name`, which is built for *fuzzy* matching:
    it drops stopwords and sorts tokens, so "Loan Type", "Type" and "Account
    Type" all collapse to the same key. That tolerance is right when guessing
    what a header means and wrong when applying a decision an operator made
    about one specific column — a decision must land on the column it was made
    about and no other. This is the same rule the client mapping memory uses
    (``mapping_memory.normalize_column``), so a decision recorded in one place
    matches the same header in the other.
    """
    return re.sub(r"[\s_]+", " ", str(column or "").strip().lower()).strip()


class ApprovedMappingError(ValueError):
    """The approved mapping contract is missing, malformed, mis-scoped or of an
    unsupported version. Always fatal: a run that was told to use an approved
    contract must never quietly fall back to alias and fuzzy matching."""


class ApprovedMapping(NamedTuple):
    """One approved source-column -> canonical-field decision."""

    source_column: str
    canonical_field: str
    source_file: str
    group: str
    rule_id: str
    rule_version: str
    confidence: float
    contract_path: str


class ApprovedContract:
    """The approved decisions in force for one run, indexed for lookup."""

    def __init__(self, mappings: List[ApprovedMapping],
                 sources: Optional[List[str]] = None,
                 scopes: Optional[List[Dict[str, str]]] = None):
        self.mappings = list(mappings)
        self.sources = list(sources or [])
        self.scopes = list(scopes or [])
        self._by_column: Dict[str, ApprovedMapping] = {}
        # Most authoritative group first, and within a group first-declared wins,
        # so the index is built by walking groups in order and never overwriting.
        for group in _APPROVED_GROUPS:
            for m in self.mappings:
                if m.group != group:
                    continue
                self._by_column.setdefault(approved_column_key(m.source_column), m)

    def __len__(self) -> int:
        return len(self.mappings)

    def __bool__(self) -> bool:
        return bool(self.mappings)

    def lookup(self, raw_header: str) -> Optional[ApprovedMapping]:
        return self._by_column.get(approved_column_key(raw_header))

    def as_report(self) -> Dict[str, Any]:
        return {
            "contract_files": list(self.sources),
            "contract_scopes": list(self.scopes),
            "approved_mapping_count": len(self.mappings),
            "approved_mappings": [
                {"source_column": m.source_column,
                 "canonical_field": m.canonical_field,
                 "source_file": m.source_file,
                 "group": m.group,
                 "rule_id": m.rule_id,
                 "rule_version": m.rule_version,
                 "contract_file": m.contract_path}
                for m in self.mappings
            ],
        }


def _scope_mismatch(scope: Dict[str, str], expected: Dict[str, str]) -> List[str]:
    """Names of the scope dimensions on which a contract does not match the run.

    A dimension the run does not state is not checked; a dimension the run DOES
    state must be present and equal in the contract. A contract that cannot prove
    its scope is treated as mismatched on every stated dimension — an unscoped
    document is exactly what a mis-filed one looks like.
    """
    bad: List[str] = []
    for key, want in (expected or {}).items():
        want = str(want or "").strip()
        if not want:
            continue
        got = str((scope or {}).get(key, "") or "").strip()
        if got.lower() != want.lower():
            bad.append(f"{key}: contract={got or '<absent>'!s} run={want}")
    return bad


def load_approved_mappings(
    paths: List[str | Path],
    canonical_fields: Optional[List[str]] = None,
    expected_scope: Optional[Dict[str, str]] = None,
) -> ApprovedContract:
    """Load and validate approved mapping contracts. Fails closed on any doubt.

    ``expected_scope`` names the run's client / portfolio / dataset. When given,
    every contract must carry a matching ``contract_scope`` block; a contract that
    is silent about its scope, or that names a different one, is refused rather
    than applied to the wrong book.
    """
    mappings: List[ApprovedMapping] = []
    sources: List[str] = []
    scopes: List[Dict[str, str]] = []
    allowed = set(canonical_fields or [])

    for raw_path in paths or []:
        path = Path(raw_path)
        if not path.exists():
            raise ApprovedMappingError(
                f"approved mapping contract not found: {path}. Refusing to run "
                f"with automated matching in place of an approved decision.")
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — reported, never swallowed
            raise ApprovedMappingError(
                f"approved mapping contract {path} is not readable YAML: {exc}")
        if not isinstance(doc, dict):
            raise ApprovedMappingError(
                f"approved mapping contract {path} is malformed: expected a "
                f"mapping at the top level, got {type(doc).__name__}.")

        version = doc.get("version", None)
        try:
            version_i = int(version)
        except (TypeError, ValueError):
            raise ApprovedMappingError(
                f"approved mapping contract {path} declares no usable version "
                f"({version!r}); supported: {list(SUPPORTED_APPROVED_CONTRACT_VERSIONS)}.")
        if version_i not in SUPPORTED_APPROVED_CONTRACT_VERSIONS:
            raise ApprovedMappingError(
                f"approved mapping contract {path} is version {version_i}; this "
                f"build understands {list(SUPPORTED_APPROVED_CONTRACT_VERSIONS)}. "
                f"Refusing to interpret a contract written to another schema.")

        scope = doc.get("contract_scope") or {}
        if not isinstance(scope, dict):
            raise ApprovedMappingError(
                f"approved mapping contract {path} has a malformed contract_scope.")
        bad = _scope_mismatch(scope, expected_scope or {})
        if bad:
            raise ApprovedMappingError(
                f"approved mapping contract {path} is scoped to a different "
                f"source ({'; '.join(bad)}). Refusing to apply one book's "
                f"approved decisions to another.")
        scopes.append({k: str(v) for k, v in scope.items()})
        sources.append(str(path))

        found_group = False
        for group in _APPROVED_GROUPS:
            if group not in doc:
                continue
            entries = doc.get(group)
            if entries is None:
                entries = []
            if not isinstance(entries, list):
                raise ApprovedMappingError(
                    f"approved mapping contract {path}: '{group}' must be a list, "
                    f"got {type(entries).__name__}.")
            found_group = True
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ApprovedMappingError(
                        f"approved mapping contract {path}: '{group}' contains a "
                        f"non-mapping entry ({entry!r}).")
                column = str(entry.get("source_column", "") or "").strip()
                target = str(entry.get("canonical_field", "") or "").strip()
                if not column or not target:
                    raise ApprovedMappingError(
                        f"approved mapping contract {path}: every '{group}' entry "
                        f"needs a source_column and a canonical_field; got "
                        f"{entry!r}.")
                if allowed and target not in allowed:
                    raise ApprovedMappingError(
                        f"approved mapping contract {path}: approved target "
                        f"'{target}' (for column '{column}') is not a canonical "
                        f"field of this run's registry scope.")
                mappings.append(ApprovedMapping(
                    source_column=column,
                    canonical_field=target,
                    source_file=str(entry.get("source_file", "") or "").strip(),
                    group=group,
                    rule_id=str(entry.get("rule_id", "") or ""),
                    rule_version=str(entry.get("rule_version", "") or ""),
                    confidence=float(entry.get("confidence", 1.0) or 1.0),
                    contract_path=str(path),
                ))
        if not found_group:
            raise ApprovedMappingError(
                f"approved mapping contract {path} declares none of "
                f"{list(_APPROVED_GROUPS)} — there is no approved decision in it.")

    return ApprovedContract(mappings, sources, scopes)


def apply_alias_overlay(
    alias_map: Dict[str, str], overlay_dir: Path
) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """Layer a CLIENT-SPECIFIC alias directory on top of the global alias map.

    The managed service onboards portfolios whose source headers are the client's
    own vocabulary ("Principal Outstanding", "Youngest Borrower"). Those belong to
    that client's approved onboarding contract, not to the global mapper — adding
    them globally would broaden every other client's matching surface.

    An overlay entry **wins** over the global map for the same normalised header,
    which is the point: a client contract may legitimately resolve a header
    differently from the global default. Returns the merged map plus the list of
    applied overrides so the run report can record exactly what the client
    contract changed.
    """
    overlay = load_aliases_from_dir(Path(overlay_dir))
    merged = dict(alias_map)
    applied: List[Dict[str, str]] = []
    for norm, canon in overlay.items():
        previous = merged.get(norm)
        merged[norm] = canon
        applied.append({
            "normalised_alias": norm,
            "canonical_field": canon,
            "overrode_global": previous or "",
            "overlay_dir": str(overlay_dir),
        })
    if applied:
        logging.info(
            "Applied %d client alias overlay entries from %s (%d overrode a global alias)",
            len(applied), overlay_dir,
            sum(1 for a in applied if a["overrode_global"]),
        )
    return merged, applied

# ------------------------------------------------------------------
# MAPPER
# ------------------------------------------------------------------


class HeaderMapper:
    def __init__(self, canonical_fields: List[str], alias_map: Dict[str, str],
                 approved: Optional[ApprovedContract] = None):
        self.canonical = canonical_fields
        self.alias_map = alias_map
        #: The operator-approved decisions in force for this run, consulted
        #: before any automated matching. ``None`` on a run with no contract,
        #: which leaves every tier below exactly as it was.
        self.approved = approved

        self.norm_map = {normalise_name(c): c for c in canonical_fields}
        self.token_sets = {c: set(tokenize(c)) for c in canonical_fields}

    def approved_for(self, raw_header: str) -> Optional[ApprovedMapping]:
        """The approved decision governing this header, if there is one."""
        if self.approved is None or pd.isna(raw_header):
            return None
        return self.approved.lookup(str(raw_header).strip())

    def map_one(self, raw_header: str) -> Tuple[Optional[str], str, float]:
        if pd.isna(raw_header):
            return None, "empty", 0.0

        h = str(raw_header).strip()
        if not h:
            return None, "empty", 0.0

        # Tier 0: the operator-approved decision. Checked before every automated
        # method, including exact and normalised canonical-name matching, because
        # it is not a guess about what the column means — it is the answer.
        decision = self.approved_for(h)
        if decision is not None:
            return decision.canonical_field, "operator_approved", 1.0

        lowered = h.lower()
        norm = normalise_name(h)
        tokens = set(tokenize(h))

        # Tier 1: exact match (case-insensitive)
        for c in self.canonical:
            if lowered == c.lower():
                return c, "exact", 1.0

        # Tier 2: normalised canonical name match
        if norm in self.norm_map:
            return self.norm_map[norm], "normalized", 1.0

        # Tier 3: alias match
        if norm in self.alias_map and self.alias_map[norm] in self.canonical:
            return self.alias_map[norm], "alias", 1.0

        # Tier 4: token-set Jaccard
        if tokens:
            best, score = None, 0.0
            for c, c_tokens in self.token_sets.items():
                if not c_tokens:
                    continue
                jaccard = len(tokens & c_tokens) / len(tokens | c_tokens)
                if jaccard > score:
                    score, best = jaccard, c
            if best and score >= JACCARD_THRESHOLD:
                if score < LOW_CONFIDENCE_THRESHOLD:
                    logging.warning(f"LOW CONFIDENCE: '{h}' → '{best}' (token_set={score:.4f})")
                return best, "token_set", round(score, 4)

        # Tier 5: RapidFuzz token_set_ratio
        match = process.extractOne(h, self.canonical, scorer=fuzz.token_set_ratio)
        if match and match[1] >= FUZZ_TOKEN_SET_THRESHOLD:
            return match[0], "fuzz_token_set", match[1] / 100.0

        # Tier 6: RapidFuzz on normalised forms
        norm_keys = list(self.norm_map.keys())
        norm_match = process.extractOne(norm, norm_keys, scorer=fuzz.ratio)
        if norm_match and norm_match[1] >= FUZZ_NORM_THRESHOLD:
            return self.norm_map[norm_match[0]], "fuzz_ratio_norm", norm_match[1] / 100.0

        return None, "unmapped", 0.0


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------


def read_input_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)



def _clean_numeric_series(s: pd.Series) -> pd.Series:
    # Remove currency symbols, spaces, and thousands separators; keep digits, sign, dot
    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace(r"[^0-9\-\.]", "", regex=True)
    s2 = s2.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(s2, errors="coerce")


def _clean_boolean_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip().str.lower()
    true_set = {"y", "yes", "true", "1", "t"}
    false_set = {"n", "no", "false", "0", "f"}
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    out[s2.isin(true_set)] = "Y"
    out[s2.isin(false_set)] = "N"
    return out


def apply_types(df: pd.DataFrame, registry: dict) -> pd.DataFrame:
    """Light, deterministic typing/normalisation driven by registry 'format'.

    This is not a full transformation engine; it standardises common representations so
    downstream validation and MI do not depend on lender-specific formatting.
    """
    fields = registry.get("fields", {}) or {}
    for col in df.columns:
        meta = fields.get(col, {}) or {}
        fmt = str(meta.get("format", "")).strip().lower()

        if fmt in ("", "string", "text"):
            continue

        if fmt == "date":
            dt = pd.to_datetime(df[col], errors="coerce", dayfirst=False, utc=False)
            df[col] = dt.dt.strftime("%Y-%m-%d")
            continue

        if fmt in ("decimal", "number", "numeric", "float"):
            df[col] = _clean_numeric_series(df[col])
            continue

        if fmt in ("integer", "int"):
            df[col] = _clean_numeric_series(df[col]).astype("Int64")
            continue

        if fmt in ("boolean", "bool", "y/n", "yn"):
            df[col] = _clean_boolean_series(df[col])
            continue

        # list/enums: keep as-is; enum normalisation handled upstream via synonym libs
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Messy → Canonical (Frozen Spine v1.0)")
    parser.add_argument("--input", required=True, help="Messy CSV/XLSX loan tape")
    parser.add_argument("--portfolio-type", default="equity_release", help="e.g. equity_release, sme, cre")
    parser.add_argument(
        "--registry",
        default="fields_registry.yaml",
        help="Field registry YAML (source of truth)",
    )
    parser.add_argument(
        "--aliases-dir",
        default="aliases",
        help="Directory containing aliases_*.yaml",
    )
    parser.add_argument(
        "--extra-aliases-dir",
        action="append",
        default=[],
        help="Optional CLIENT-SPECIFIC aliases_*.yaml directory layered on top of "
             "--aliases-dir. Repeatable; later directories win. Use this for a "
             "client's approved onboarding contract instead of widening the "
             "global alias files.",
    )
    parser.add_argument(
        "--approved-mappings",
        action="append",
        default=[],
        help="Path to an APPROVED mapping contract "
             "(12_approved_mapping_overrides.yaml) produced by the Operations "
             "Control Centre. Repeatable. Approved decisions are tier 0: they "
             "beat exact, normalised, alias and fuzzy matching. This is NOT an "
             "alias overlay and must not be supplied through --extra-aliases-dir.",
    )
    parser.add_argument(
        "--require-approved-mappings",
        action="store_true",
        default=False,
        help="Fail the run when no usable approved mapping contract is supplied. "
             "For a client/portfolio governed by an approved onboarding contract, "
             "reverting to alias and fuzzy matching is a governed failure, not a "
             "fallback.",
    )
    parser.add_argument("--approved-scope-client", default="",
                        help="Client the approved contract must be scoped to.")
    parser.add_argument("--approved-scope-portfolio", default="",
                        help="source_portfolio_id the contract must be scoped to.")
    parser.add_argument("--approved-scope-dataset", default="",
                        help="Dataset (funded|pipeline) the contract must be scoped to.")
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=[],
        help="Optional regimes to also output projections for (e.g. ESMA_Annex2)",
    )
    parser.add_argument("--output-dir", default="out", help="Output directory")
    parser.add_argument("--output-prefix", help="Override output stem")
    parser.add_argument(
        "--output-schema",
        choices=["active", "full"],
        default="active",
        help="Schema for canonical_full.csv: 'active' = core:true + mapped fields; 'full' = all registry fields for the portfolio_type.",
    )
    parser.add_argument(
        "--apply-types",
        action="store_true",
        default=False,
        help="Typing happens in transformation layer.",
    )
    parser.add_argument(
        "--no-apply-types",
        dest="apply_types",
        action="store_false",
        help="Disable type/format normalisation.",
    )
    args = parser.parse_args()
    
    hq_recommendations: List[dict] = []
    hq_recommendations_count: int = 0

    base_dir = Path(__file__).resolve().parent
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = base_dir / registry_path
    aliases_dir = Path(args.aliases_dir)
    if not aliases_dir.is_absolute():
        aliases_dir = base_dir / aliases_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = load_field_registry(registry_path)
    canonical_fields = select_registry_fields(registry, args.portfolio_type)
    if not canonical_fields:
        raise ValueError(f"No registry fields selected for portfolio_type='{args.portfolio_type}'")

    df_raw = read_input_table(input_path)
    alias_map = load_aliases_from_dir(aliases_dir)
    # Client-specific onboarding-contract aliases, layered on top (never global).
    alias_overlay_applied: List[dict] = []
    for extra in (args.extra_aliases_dir or []):
        extra_dir = Path(extra)
        if not extra_dir.is_absolute():
            # A relative overlay path is what the operator typed, so resolve it
            # against the working directory first; fall back to the script dir
            # (which is how --aliases-dir resolves) only if that does not exist.
            cwd_candidate = Path.cwd() / extra_dir
            extra_dir = cwd_candidate if cwd_candidate.exists() else base_dir / extra_dir
        if not extra_dir.exists():
            raise FileNotFoundError(
                f"--extra-aliases-dir {extra!r} does not exist (looked in "
                f"{Path.cwd()} and {base_dir}). Refusing to run with a silently "
                f"empty client alias contract."
            )
        alias_map, applied = apply_alias_overlay(alias_map, extra_dir)
        alias_overlay_applied.extend(applied)

    # Operator-approved decisions, loaded and validated BEFORE any header is
    # looked at. A required-but-unusable contract stops the run here, so no
    # canonical is ever produced from automated matching in its place.
    expected_scope = {
        "client_id": args.approved_scope_client,
        "portfolio_id": args.approved_scope_portfolio,
        "dataset": args.approved_scope_dataset,
    }
    approved = load_approved_mappings(
        [Path(a) for a in (args.approved_mappings or [])],
        canonical_fields=canonical_fields,
        expected_scope=expected_scope,
    )
    if args.require_approved_mappings and not approved:
        raise ApprovedMappingError(
            "This client/portfolio requires an approved mapping contract and "
            "none was supplied (--approved-mappings). Refusing to map headers "
            "by alias or fuzzy matching: the approved decision is the mapping "
            "authority, and running without it is what silently remapped "
            "columns before.")
    if approved:
        logging.info("Loaded %d approved mapping decision(s) from %s",
                     len(approved), ", ".join(approved.sources))

    mapper = HeaderMapper(canonical_fields, alias_map, approved=approved or None)

    header_map: Dict[str, Optional[str]] = {}
    report_rows = []
    for col in df_raw.columns:
        canon, method, conf = mapper.map_one(col)
        header_map[col] = canon
        decision = mapper.approved_for(col)
        report_rows.append(
            {
                "raw_header": col,
                "canonical_field": canon or "",
                # The method that ACTUALLY decided this column, so a reader can
                # never mistake "a contract was loaded" for "the contract won".
                "mapping_method": method,
                "confidence": conf,
                "approved_rule_id": decision.rule_id if decision else "",
                "approved_rule_version": decision.rule_version if decision else "",
                "approved_contract_file": decision.contract_path if decision else "",
                "approved_source_file": decision.source_file if decision else "",
            }
        )

    
    # Build candidate lists per canonical field to detect duplicates and resolve deterministically
    canon_candidates: Dict[str, List[dict]] = {}
    for raw_col, canon in header_map.items():
        if not canon:
            continue
        # Find mapping record for this raw header
        rec = next((r for r in report_rows if r["raw_header"] == raw_col), None)
        if rec is None:
            continue
        canon_candidates.setdefault(canon, []).append(rec)

    # Determine which output schema to emit
    core_required = {fname for fname, meta in registry["fields"].items() if (meta or {}).get("core_canonical") is True}
    mapped_fields = set(canon_candidates.keys())

    # ------------------------------------------------------------------
    # FIXED OUTPUT SCHEMA LOGIC (Portfolio + Regime Filter)
    # ------------------------------------------------------------------
    schema_mode = getattr(args, "output_schema", "active")

    if schema_mode == "full":
        # 1. Start with everything valid for this Asset Class (e.g. Equity Release)
        #    (canonical_fields is already filtered by select_registry_fields earlier in the script)
        candidates = canonical_fields
        
        # 2. OPTIONAL: If a specific Regime is requested (e.g. ESMA_annex_2), filter further.
        if args.regimes:
            regime_whitelist = set()
            for r in args.regimes:
                # Use the existing helper function to find fields mapped to this regime
                # This looks for 'regime_mapping: {ESMA_annex_2: ...}' in the YAML
                r_fields = select_regime_fields(registry, r, args.portfolio_type)
                regime_whitelist.update(r_fields)
            
            # Intersection: Keep field IF (It is Core) OR (It is in the Regime)
            # We always keep Core (like Loan ID) to ensure the file is usable.
            candidates = [f for f in candidates if f in regime_whitelist or f in core_required]
            
            logging.info(f"Schema filtered by regime(s) {args.regimes}. Reduced to {len(candidates)} columns.")

        active_fields = candidates

    else:
        # Default 'active' mode: Core + whatever was found in the input file
        active_fields = sorted(core_required | mapped_fields)

    df_canon = pd.DataFrame(columns=active_fields)

    unmapped = [c for c, v in header_map.items() if v is None]

    # Resolve duplicates: choose the highest-ranked method, then highest confidence
    for canon_field, recs in canon_candidates.items():
        if canon_field not in df_canon.columns:
            continue
        if len(recs) > 1:
            recs_sorted = sorted(
                recs,
                key=lambda r: (METHOD_RANK.get(r["mapping_method"], 0), float(r["confidence"])),
                reverse=True,
            )
            chosen = recs_sorted[0]
            others = [r["raw_header"] for r in recs_sorted[1:]]
            logging.warning(
                f"DUPLICATE MAPPING: {canon_field} mapped by multiple headers. "
                f"Chosen='{chosen['raw_header']}' ({chosen['mapping_method']}, conf={chosen['confidence']}); "
                f"Ignored={others}"
            )
        else:
            chosen = recs[0]

        df_canon[canon_field] = df_raw[chosen["raw_header"]].copy()

    if getattr(args, "apply_types", False):
        df_canon = apply_types(df_canon, registry)

    stem = args.output_prefix or input_path.stem
    full_path = out_dir / f"{stem}_canonical_full.csv"
    report_path = out_dir / f"{stem}_mapping_report.csv"
    unmapped_path = out_dir / f"{stem}_unmapped_headers.csv"

    df_canon.to_csv(full_path, index=False)
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    if unmapped:
        pd.DataFrame({"raw_header": unmapped}).to_csv(unmapped_path, index=False)

    mapped = len(df_raw.columns) - len(unmapped)
    logging.info(f"Mapped {mapped}/{len(df_raw.columns)} headers ({mapped/len(df_raw.columns):.1%}).")
    logging.info(f"Wrote: {full_path}")
    logging.info(f"Wrote: {report_path}")
    if unmapped:
        logging.info(f"Wrote: {unmapped_path} ({len(unmapped)} unmapped)")
    
    # ---------------------------------------------------------
    # HQ recommendations (Near-misses) - compute BEFORE JSON write
    # ---------------------------------------------------------
    field_defs = registry.get("fields", {}) or {}
    if unmapped:
        candidate_keys = list(field_defs.keys())
        for raw_header in unmapped:
            match = process.extractOne(raw_header, candidate_keys, scorer=fuzz.token_set_ratio)
            if not match:
                continue
            best_field, score, _ = match
            if 60 <= score < 88:
                hq_recommendations.append({
                    "raw_header": raw_header,
                    "suggested_canonical": best_field,
                    "confidence": round(float(score) / 100.0, 4),
                    "confidence_pct": float(score),
                    "method": "token_set_ratio",
                })

    hq_recommendations_count = len(hq_recommendations)

    # Write header mapping report JSON (for delta + lineage tooling)
    header_map_json_path = out_dir / f"{stem}_header_mapping_report.json"
    header_map_payload = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "input_file": str(input_path),
        "portfolio_type": args.portfolio_type,
        "output_schema": args.output_schema,
        "regimes": list(args.regimes or []),
        "registry_path": str(registry_path),
        "aliases_dir": str(aliases_dir),
        # Client onboarding-contract alias overlay (empty on a standard run).
        "extra_aliases_dirs": [str(d) for d in (args.extra_aliases_dir or [])],
        "alias_overlay_applied": alias_overlay_applied,
        "alias_overlay_entry_count": len(alias_overlay_applied),
        # The approved mapping contract in force, what it said, and how many
        # columns it actually decided.
        "approved_mapping_contract": approved.as_report(),
        "approved_mapping_required": bool(args.require_approved_mappings),
        "approved_mapping_scope_expected": {k: v for k, v in expected_scope.items() if v},
        "columns_decided_by_operator_approval": sorted(
            r["raw_header"] for r in report_rows
            if r["mapping_method"] == "operator_approved"),
        "thresholds": {
            "JACCARD_THRESHOLD": JACCARD_THRESHOLD,
            "FUZZ_TOKEN_SET_THRESHOLD": FUZZ_TOKEN_SET_THRESHOLD,
            "FUZZ_NORM_THRESHOLD": FUZZ_NORM_THRESHOLD,
            "LOW_CONFIDENCE_THRESHOLD": LOW_CONFIDENCE_THRESHOLD,
        },
        "hq_recommendations": hq_recommendations,
        "hq_recommendations_count": hq_recommendations_count,
    
        # full per-column record (best for audit)
        "mappings": report_rows,
        # convenience map raw->canonical (best for quick diffs)
        "raw_to_canonical": {r["raw_header"]: r["canonical_field"] for r in report_rows if r.get("canonical_field")},
        "unmapped_headers": unmapped,
    }
    header_map_json_path.write_text(json.dumps(header_map_payload, indent=2, sort_keys=True), encoding="utf-8")
    logging.info(f"Wrote: {header_map_json_path}")

    # ---------------------------------------------------------
    # FINAL REPORTING BLOCK (Fixed: Aligned to Active Schema)
    # ---------------------------------------------------------
 
    # Define Groups based on the OUTPUT DATAFRAME columns
    # (This ensures the report reflects the Regime/Portfolio filter you just ran)
    active_schema_cols = set(df_canon.columns)
    
    core_fields = []
    enhancement_fields = []

    for f_name in active_schema_cols:
        meta = field_defs.get(f_name, {})
        # Safety check if field missing from registry
        if not meta: continue 
        
        is_core_layer = meta.get('layer') == 'core'
        is_canonical = meta.get('core_canonical') is True
        
        if is_core_layer and is_canonical:
            core_fields.append(f_name)
        else:
            enhancement_fields.append(f_name)

    # Calculate Coverage
    # We check which of the active schema columns were actually found in the input tape
    # 'mapped_fields' comes from the mapping logic earlier in the script
    mapped_core = [f for f in core_fields if f in mapped_fields]
    mapped_enhancements = [f for f in enhancement_fields if f in mapped_fields]

    # Generate Report
    print("\n" + "="*60)
    print(f"MAPPING PERFORMANCE REPORT")
    print("="*60)
    print(f"TOTAL FIELDS IN REGIME/SCHEMA: {len(active_schema_cols)}")
    
    # Core Metrics
    core_pct = len(mapped_core) / len(core_fields) if core_fields else 0
    print(f"CRITICAL CORE FIELDS: {len(mapped_core)}/{len(core_fields)} ({core_pct:.1%})")
    
    if len(mapped_core) < len(core_fields):
        missing_core = sorted(list(set(core_fields) - set(mapped_core)))
        print(f"  Missing Critical Fields: {', '.join(missing_core[:5])}...")

    # Enhancement Metrics
    enh_pct = len(mapped_enhancements) / len(enhancement_fields) if enhancement_fields else 0
    print(f"\nENHANCEMENT FIELDS MAPPED: {len(mapped_enhancements)}/{len(enhancement_fields)} ({enh_pct:.1%})")
    
    layer_counts = {}
    for field in mapped_enhancements:
        layer = field_defs.get(field, {}).get('layer', 'uncategorized')
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    if not layer_counts and mapped_enhancements:
        print("  (All Uncategorized)")
    elif not mapped_enhancements:
        print("  (None found in tape)")
        
    for layer, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  > {layer.title()}: {count} fields")

    # Unmapped Opportunities (The 'Near Misses')
    print("\nUNMAPPED OPPORTUNITIES (>60% Confidence match):")

    if hq_recommendations_count == 0:
        print("  (No high-probability candidates found)")
    else:
        for rec in hq_recommendations[:10]:
            print(f"  ? '{rec['raw_header']}' matches '{rec['suggested_canonical']}' ({rec['confidence_pct']:.0f}%)")
        if hq_recommendations_count > 10:
            print(f"  ... and {hq_recommendations_count - 10} more.")

    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()