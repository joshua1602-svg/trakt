from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

BASE_DIR = Path(__file__).parent

CANONICAL_FILE = BASE_DIR / "canonical_output" / "canonical_v1.3_auto_extended.xlsx"
FIELD_ROLES_CSV = BASE_DIR / "field_roles.csv"
ALIASES_DIR = BASE_DIR / "aliases"
MIN_SCORE = 0.2


def normalize_header(s: str) -> str:
    """Stable normalisation used only for matching / dedupe."""
    if s is None:
        return ""
    s = str(s).strip()
    if s == "":
        return ""
    s = s.lower()
    # camelCase → snake_case
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # non-alphanumeric → underscore
    s = re.sub(r"[^a-z0-9]+", "_", s)
    # collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def load_canonical(path: Path):
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    names = df["canonical_name"].str.strip().tolist()

    if "description" in df.columns:
        descs = df["description"].fillna("").astype(str).tolist()
    else:
        descs = [""] * len(df)

    return names, descs


def load_field_roles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["canonical_name"] = df["canonical_name"].astype(str).str.strip()
    df["role"] = df["role"].astype(str).str.lower()
    return df


def load_existing_aliases() -> dict:
    aliases = {}
    if not ALIASES_DIR.exists():
        return aliases

    for file in ALIASES_DIR.glob("*.yaml"):
        with open(file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # If the YAML file is not a dict at top level, skip it
        if not isinstance(data, dict):
            print(f"Warning: {file} has unexpected top-level type {type(data).__name__}, skipping.")
            continue

        for canon, meta in data.items():
            # meta can be either:
            # - dict like {"aliases": ["a", "b"]}
            # - list like ["a", "b"] from older/incorrect format
            if isinstance(meta, dict):
                alias_list = meta.get("aliases", []) or []
            elif isinstance(meta, list):
                alias_list = meta
            else:
                # unknown structure, ignore
                print(
                    f"Warning: {file} -> {canon} has unexpected type {type(meta).__name__}, skipping."
                )
                continue

            aliases.setdefault(canon, set()).update(alias_list)

    return aliases

def build_alias_suggestions(
    tape_path: Path,
    apply_changes: bool = False,
    min_score: float = MIN_SCORE,
):
    print(f"\n=== Running alias builder on: {tape_path} ===")

    # Load inputs
    canon_names, canon_descs = load_canonical(CANONICAL_FILE)
    field_roles = load_field_roles(FIELD_ROLES_CSV)
    existing_aliases = load_existing_aliases()

    # Read headers from tape
    if tape_path.suffix.lower() in [".xlsx", ".xls"]:
        df_tape = pd.read_excel(tape_path, nrows=1)
    else:
        df_tape = pd.read_csv(tape_path, nrows=1)
    headers = [str(h) for h in df_tape.columns]

    # Build TF-IDF model (character n-grams – more robust to messy headers)
    corpus = [f"{name} {desc}" for name, desc in zip(canon_names, canon_descs)]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",      # character n-grams within word boundaries
        ngram_range=(3, 5),      # 3–5 character chunks
        min_df=1
    )
    X_canon = vectorizer.fit_transform(corpus)
    X_tape = vectorizer.transform(headers)
    sim_matrix = cosine_similarity(X_tape, X_canon)

    # Build suggestions list
    suggestions = []
    for i, header in enumerate(headers):
        norm_header = normalize_header(header)
        sim_row = sim_matrix[i]

        # Take top 3 matches, descending score
        top_idx = np.argsort(sim_row)[-3:][::-1]
        for idx in top_idx:
            score = float(sim_row[idx])
            if score < min_score:
                break

            canon = canon_names[idx]
            role_series = field_roles.loc[field_roles["canonical_name"] == canon, "role"]
            role = role_series.iloc[0] if not role_series.empty else "unknown"

            # Check if an equivalent alias already exists (by normalised form)
            existing_for_canon = existing_aliases.get(canon, set())
            existing_norms = {normalize_header(a) for a in existing_for_canon}
            already_exists = norm_header in existing_norms

            band = "high" if score >= 0.725 else "medium" if score >= 0.6 else "low"

            suggestions.append(
                {
                    "raw_header": header,
                    "normalized_header": norm_header,
                    "canonical_candidate": canon,
                    "score": round(score, 3),
                    "band": band,
                    "role": role,
                    "already_exists": already_exists,
                }
            )

    suggestions_df = pd.DataFrame(suggestions)
    out_csv = tape_path.parent / f"{tape_path.stem}_alias_suggestions.csv"
    suggestions_df.to_csv(out_csv, index=False)
    print(f"Suggestions written → {out_csv} ({len(suggestions_df)} rows)")

    # Simple summary
    if not suggestions_df.empty:
        high_conf = (suggestions_df["band"] == "high").sum()
        med = (suggestions_df["band"] == "medium").sum()
        low = (suggestions_df["band"] == "low").sum()

        covered_headers = (
            suggestions_df
            .loc[suggestions_df["band"].isin(["high", "medium"]), "raw_header"]
            .nunique()
        )
        coverage_pct = covered_headers / len(headers) * 100

        print(
            f"High-confidence rows: {high_conf} | Medium rows: {med} | Low rows: {low} | "
            f"Header coverage (high+med): {coverage_pct:.1f}%"
        )

    if not apply_changes:
        print("APPLY_CHANGES is False – no YAML alias updates performed.")
        return

    # ----- Apply changes to YAMLs (HIGH BAND ONLY) -----
    print("\nApplying alias updates to YAMLs...")
    updated_aliases = {k: set(v) for k, v in existing_aliases.items()}
    added_count = 0

    for _, row in suggestions_df.iterrows():
        if row["band"] != "high":
            continue

        canon = row["canonical_candidate"]
        raw_header = row["raw_header"]
        norm_header = row["normalized_header"]

        # --- extra safety: avoid mixing "date"/"id"/"ref"/"bal" into name fields ---
        raw_tokens = set(str(raw_header).lower().split())
        canon_tokens = set(str(canon).lower().split("_"))

        suspicious_tokens = {"date", "dt", "id", "ref", "bal", "amount"}
        if suspicious_tokens & raw_tokens and not (suspicious_tokens & canon_tokens):
        # e.g. "servicer name Date" vs "servicer_name" -> SKIP
            continue

        existing_for_canon = updated_aliases.get(canon, set())
        existing_norms = {normalize_header(a) for a in existing_for_canon}

        if norm_header not in existing_norms:
            updated_aliases.setdefault(canon, set()).add(raw_header)
            added_count += 1
            print(f"Adding alias: '{raw_header}' (norm='{norm_header}') -> {canon}")

    print(f"\nTotal new aliases added: {added_count}")

    print(f"\nTotal new aliases added before deduplication: {added_count}")

    # ==================================================================
    # DEDUPLICATION + CLEAN-UP (THIS IS THE CRITICAL FIX)
    # ==================================================================
    def dedupe_aliases(alias_set: set) -> list:
        """Remove normalised duplicates but keep the original human-readable form."""
        seen = set()
        clean = []
        for alias in sorted(alias_set):                    # deterministic order
            norm = normalize_header(alias)
            if not norm or norm in seen:
                continue
            if norm:
                seen.add(norm)
                clean.append(alias)                        # keep original casing/spelling
        return clean

    # Apply deduplication
    final_aliases = {}
    total_after = 0
    for canon, alias_set in updated_aliases.items():
        clean_list = dedupe_aliases(alias_set)
        final_aliases[canon] = clean_list
        total_after += len(clean_list)

    print(f"After deduplication: {total_after} aliases remaining "
          f"({len(updated_aliases)} fields)")

    # ==================================================================
    # WRITE FINAL YAML FILES
    # ==================================================================
    roles_map = dict(zip(field_roles["canonical_name"], field_roles["role"]))
    ALIASES_DIR.mkdir(parents=True, exist_ok=True)

    for category in ["mandatory", "analytics", "optional"]:
        allowed_canons = {
            cn for cn, r in roles_map.items() if r.lower() == category
        }
        category_data = {
            canon: {"aliases": aliases}
            for canon, aliases in final_aliases.items()
            if canon in allowed_canons
        }

        out_yaml = ALIASES_DIR / f"aliases_{category}.yaml"
        with open(out_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                category_data,
                f,
                sort_keys=True,
                allow_unicode=True,
                width=4096,
                default_flow_style=False
            )
        count_fields = len(category_data)
        count_aliases = sum(len(v["aliases"]) for v in category_data.values())
        print(f"Wrote {out_yaml} → {count_fields} fields, {count_aliases} unique aliases")

    print("\nDeduplication complete. Alias libraries are now clean and production-safe.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build alias suggestions from a messy tape.")
    parser.add_argument(
        "tape_path",
        help="Path to messy lender tape (CSV or Excel)",
    )
    parser.add_argument(
        "--apply-changes",
        action="store_true",
        help="If set, update aliases_*.yaml with HIGH band suggestions",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_SCORE,
        help=f"Minimum similarity score for suggestions (default {MIN_SCORE})",
    )

    args = parser.parse_args()

    build_alias_suggestions(
        tape_path=Path(args.tape_path),
        apply_changes=bool(args.apply_changes),
        min_score=float(args.min_score),
    )
