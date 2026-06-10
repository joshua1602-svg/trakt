"""
agents/config_bootstrap_agent.py

Profiles a raw tape and bootstraps a merged run config by:
  1. Profiling headers, row counts, dtypes, null rates, enum-like columns.
  2. Detecting asset class from deterministic signals (known templates, field
     names, aliases, regime-specific required fields).
  3. Selecting the appropriate regime template.
  4. Merging client / asset / regime config templates into a draft config.
  5. Generating structured config questions for any missing critical values.
     LLM is called optionally to produce user-friendly question wording.
  6. Emitting ConfigBootstrapResult.

LLM use is strictly limited to question wording and narrative — it never
touches financial data or makes config decisions.

Usage:
  from agents.config_bootstrap_agent import ConfigBootstrapAgent
  result = ConfigBootstrapAgent(...).run()
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from agents.onboarding_schemas import ConfigBootstrapResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fields that are critical — if missing from config, we cannot proceed safely.
_CRITICAL_CONFIG_FIELDS: Dict[str, Dict[str, Any]] = {
    "portfolio.asset_class": {
        "category": "client_config",
        "why_needed": "Asset class determines which canonical fields and ESMA codes apply.",
        "blocking": True,
        "options": ["equity_release", "rre", "sme", "cre", "auto", "consumer"],
    },
    "default_regime": {
        "category": "regime_config",
        "why_needed": "Determines which ESMA Annex template is applied for regulatory output.",
        "blocking": True,
        "options": ["ESMA_Annex2", "ESMA_Annex3", "ESMA_Annex4", "ESMA_Annex8", "ESMA_Annex9",
                    "ESMA_Annex12"],
    },
    "portfolio.base_currency": {
        "category": "asset_config",
        "why_needed": "Currency is required for monetary field normalisation.",
        "blocking": True,
        "options": ["GBP", "EUR", "USD"],
    },
    "portfolio.country": {
        "category": "asset_config",
        "why_needed": "Country determines geographic enrichment and jurisdiction-specific rules.",
        "blocking": False,
        "options": ["GB", "DE", "FR", "IE", "NL", "ES", "IT"],
    },
    "defaults.originator_legal_entity_identifier": {
        "category": "client_config",
        "why_needed": "ESMA submissions require the originator LEI (20-char alphanumeric).",
        "blocking": True,
        "options": [],
    },
    "portfolio.static_reporting_date": {
        "category": "client_config",
        "why_needed": "Reporting cut-off date for the tape.",
        "blocking": False,
        "options": [],
    },
}

# Signals that suggest specific asset classes based on column names
_ASSET_CLASS_SIGNALS: Dict[str, List[str]] = {
    "equity_release": [
        "no_negative_equity", "nneg", "equity_release", "ere", "lifetime_mortgage",
        "rolled_interest", "roll_up", "drawdown", "no negative equity", "drawdown_facility",
    ],
    "rre": [
        "repayment_type", "ltv", "lti", "loan_to_income", "btl", "buy_to_let",
        "residential", "rre", "owner_occupied",
    ],
    "sme": [
        "sme", "small_medium", "company_number", "turnover", "employees",
        "corporate", "enterprise",
    ],
    "cre": [
        "commercial_real_estate", "cre", "tenant", "noi", "dscr",
        "net_operating_income", "cap_rate",
    ],
    "auto": [
        "vehicle", "vin", "make", "model_year", "odometer", "auto",
        "motor", "car_loan",
    ],
    "consumer": [
        "personal_loan", "consumer", "unsecured", "credit_card",
    ],
}

# Max sample values to inspect per field during profiling
_MAX_SAMPLE_VALUES = 5
# Max enum cardinality to treat a column as enum-like
_ENUM_CARDINALITY_THRESHOLD = 30
# Max fields whose names are sent to LLM for question wording
_MAX_FIELDS_TO_LLM = 20


# ---------------------------------------------------------------------------
# Tape profiler
# ---------------------------------------------------------------------------

def _profile_tape(tape_path: Path, max_sample_rows: int = 500) -> Dict[str, Any]:
    """
    Return a lightweight profile: headers, row count, dtypes, null rates,
    sample values, distinct counts for enum-like columns.
    No raw financial values leave this function.
    """
    profile: Dict[str, Any] = {
        "file_name": tape_path.name,
        "row_count": 0,
        "field_count": 0,
        "columns": {},
    }

    try:
        if tape_path.suffix.lower() in (".xlsx", ".xls"):
            import pandas as pd
            df = pd.read_excel(tape_path, nrows=max_sample_rows)
        else:
            import pandas as pd
            df = pd.read_csv(tape_path, nrows=max_sample_rows, low_memory=False)

        profile["row_count"] = len(df)
        profile["field_count"] = len(df.columns)
        headers = list(df.columns)

        for col in headers:
            s = df[col]
            n_null = int(s.isna().sum())
            n_total = len(s)
            null_rate = round(n_null / n_total, 3) if n_total else 0.0
            non_null = s.dropna()
            n_unique = int(non_null.nunique())
            dtype_str = str(s.dtype)

            col_info: Dict[str, Any] = {
                "dtype": dtype_str,
                "null_rate": null_rate,
                "n_unique": n_unique,
                "is_enum_like": n_unique <= _ENUM_CARDINALITY_THRESHOLD and n_unique > 0,
            }

            # Sample values — no raw numeric/date detail sent further
            if col_info["is_enum_like"]:
                col_info["sample_values"] = [
                    str(v)[:40] for v in non_null.value_counts().head(_MAX_SAMPLE_VALUES).index.tolist()
                ]
            else:
                col_info["sample_values"] = []

            profile["columns"][col] = col_info

        profile["headers"] = headers

    except Exception as exc:
        logger.warning("Tape profiling failed for %s: %s", tape_path, exc)
        profile["errors"] = [str(exc)]

    return profile


# ---------------------------------------------------------------------------
# Asset class detection
# ---------------------------------------------------------------------------

def _detect_asset_class(
    profile: Dict[str, Any],
    existing_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float]:
    """
    Detect asset class from config (highest priority) then column names.
    Returns (asset_class, confidence).
    """
    # If config already specifies it, accept with high confidence
    if existing_config:
        ac = (
            existing_config.get("portfolio", {}).get("asset_class", "")
            or existing_config.get("asset_class", "")
        )
        if ac:
            return str(ac).strip().lower(), 1.0

    headers_lower = {str(h).lower().replace(" ", "_") for h in profile.get("headers", [])}

    scores: Dict[str, float] = {}
    for asset_class, signals in _ASSET_CLASS_SIGNALS.items():
        hits = sum(
            1 for sig in signals
            if any(sig.replace(" ", "_") in h for h in headers_lower)
        )
        if hits > 0:
            scores[asset_class] = hits / len(signals)

    if not scores:
        return "unknown", 0.0

    best = max(scores, key=lambda k: scores[k])
    confidence = min(scores[best] * 10, 1.0)   # scale: >10% hit rate → 1.0
    return best, round(confidence, 2)


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------

def _load_yaml_safe(path: Path) -> Optional[Dict[str, Any]]:
    if not path or not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.warning("Could not load YAML %s: %s", path, exc)
        return None


def _find_client_config(
    client_config_dir: Path,
    client_id: str,
) -> Optional[Path]:
    """Search for a client config YAML by client_id."""
    if not client_config_dir.exists():
        return None
    for f in client_config_dir.glob("config_client_*.yaml"):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            cid = (
                data.get("client", {}).get("client_id", "")
                or data.get("client_id", "")
            )
            if cid and cid.lower() == client_id.lower():
                return f
        except Exception:
            continue
    return None


def _find_asset_config(
    asset_config_dir: Path,
    asset_class: str,
) -> Optional[Path]:
    """Search for an asset defaults YAML for a given asset class."""
    if not asset_config_dir.exists():
        return None
    patterns = [
        f"product_defaults_{asset_class.upper()}.yaml",
        f"product_defaults_{asset_class}.yaml",
        f"*{asset_class}*.yaml",
    ]
    for pat in patterns:
        matches = list(asset_config_dir.glob(pat))
        if matches:
            return matches[0]
    return None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (override wins on scalar conflicts)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Config question generation
# ---------------------------------------------------------------------------

def _build_deterministic_questions(
    merged_config: Dict[str, Any],
    profile: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return (questions, missing_critical_config).

    Walks _CRITICAL_CONFIG_FIELDS and emits a question for each value
    that is absent from the merged config.
    """
    questions: List[Dict[str, Any]] = []
    missing_critical: List[Dict[str, Any]] = []

    def _nested_get(d: Dict[str, Any], dotted_key: str) -> Any:
        parts = dotted_key.split(".")
        cur: Any = d
        for p in parts:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        return cur

    for config_key, meta in _CRITICAL_CONFIG_FIELDS.items():
        val = _nested_get(merged_config, config_key)
        if val:
            continue   # already configured

        q: Dict[str, Any] = {
            "question_id": f"q_{config_key.replace('.', '_')}",
            "category": meta["category"],
            "field": config_key,
            "question": f"What is the value for '{config_key}'?",
            "why_needed": meta["why_needed"],
            "blocking": meta["blocking"],
            "suggested_answer": "",
            "options": meta.get("options", []),
            "source": "template",
            "confidence": 0.0,
        }

        # Try to infer a suggested answer from profile signals
        if config_key == "portfolio.asset_class" and profile.get("detected_asset_class"):
            q["suggested_answer"] = profile["detected_asset_class"]
            q["confidence"] = profile.get("detected_asset_confidence", 0.0)
            q["source"] = "detected_from_tape"

        questions.append(q)
        if meta["blocking"]:
            missing_critical.append({
                "field": config_key,
                "category": meta["category"],
                "why_needed": meta["why_needed"],
                "blocking": True,
            })

    return questions, missing_critical


def _apply_llm_question_wording(
    questions: List[Dict[str, Any]],
    context: Dict[str, Any],
    llm_model: str,
    api_key: str,
) -> List[Dict[str, Any]]:
    """
    Optionally improve question wording via a cheap LLM call.
    Only question wording is updated — no config decisions are made.
    Fields in questions list are mutated in place.
    """
    if not questions:
        return questions

    try:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as exc:
        logger.warning("LLM question wording skipped (import/auth error): %s", exc)
        return questions

    # Trim to budget
    batch = questions[:_MAX_FIELDS_TO_LLM]
    tape_name = context.get("tape_file_name", "unknown")
    asset_class = context.get("detected_asset_class", "unknown")
    regime = context.get("selected_regime", "unknown")

    payload = [
        {
            "question_id": q["question_id"],
            "field": q["field"],
            "why_needed": q["why_needed"],
            "blocking": q["blocking"],
            "options": q.get("options", []),
        }
        for q in batch
    ]

    system_prompt = (
        "You are an expert in ESMA securitisation data pipelines. "
        "Rewrite each config question to be clear and user-friendly for a data operations analyst. "
        "Keep the question concise (max 20 words). "
        "Do not change question_id, field, why_needed, blocking, or options. "
        "Return a JSON array with the same objects, adding/updating only the 'question' key."
    )
    user_content = (
        f"Tape: {tape_name}  Asset class: {asset_class}  Regime: {regime}\n\n"
        f"Questions to reword:\n{json.dumps(payload, indent=2)}\n\n"
        "Return ONLY a valid JSON array. No markdown."
    )

    try:
        msg = client.messages.create(
            model=llm_model,
            max_tokens=1024,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown fences
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().strip("`").strip()
        improved = json.loads(raw)
        if isinstance(improved, list):
            id_to_improved = {item["question_id"]: item for item in improved if "question_id" in item}
            for q in questions:
                if q["question_id"] in id_to_improved:
                    improved_q = id_to_improved[q["question_id"]]
                    if improved_q.get("question"):
                        q["question"] = improved_q["question"]
                        q["source"] = "llm_wording"
    except Exception as exc:
        logger.warning("LLM question wording failed: %s — using deterministic questions.", exc)

    return questions


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class ConfigBootstrapAgent:
    """
    Profiles the tape, detects asset class, merges config templates,
    generates config questions, and emits ConfigBootstrapResult.
    """

    def __init__(
        self,
        raw_tape_path: str | Path,
        run_id: str,
        output_dir: str | Path,
        client_id: Optional[str] = None,
        existing_client_config_path: Optional[str | Path] = None,
        config_template_dir: Optional[str | Path] = None,
        asset_template_dir: Optional[str | Path] = None,
        questionnaire_answers_path: Optional[str | Path] = None,
        llm_model_questionnaire: str = "claude-haiku-4-5-20251001",
        llm_enabled: bool = True,
        api_key: Optional[str] = None,
    ) -> None:
        self.tape_path = Path(raw_tape_path)
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client_id = client_id or ""
        self.existing_client_config_path = (
            Path(existing_client_config_path) if existing_client_config_path else None
        )

        # Derive project root from this file's location (agents/ is one level below root)
        self._project_root = Path(__file__).resolve().parent.parent
        self.config_template_dir = (
            Path(config_template_dir)
            if config_template_dir
            else self._project_root / "config" / "client"
        )
        self.asset_template_dir = (
            Path(asset_template_dir)
            if asset_template_dir
            else self._project_root / "config" / "asset"
        )

        self.questionnaire_answers_path = (
            Path(questionnaire_answers_path) if questionnaire_answers_path else None
        )
        self.llm_model = llm_model_questionnaire
        self.llm_enabled = llm_enabled
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    # ------------------------------------------------------------------

    def run(self) -> ConfigBootstrapResult:
        result = ConfigBootstrapResult(run_id=self.run_id)
        result.tape_file_name = self.tape_path.name

        # 1. Profile tape
        logger.info("[ConfigBootstrap] Profiling tape: %s", self.tape_path)
        profile = _profile_tape(self.tape_path)
        result.tape_row_count = profile.get("row_count", 0)
        result.tape_field_count = profile.get("field_count", 0)

        # 2. Load any existing approved client config
        existing_config: Optional[Dict[str, Any]] = None
        if self.existing_client_config_path:
            existing_config = _load_yaml_safe(self.existing_client_config_path)
            if existing_config:
                result.client_config_path = str(self.existing_client_config_path)
                logger.info("[ConfigBootstrap] Loaded existing client config: %s",
                            self.existing_client_config_path)
        elif self.client_id:
            found = _find_client_config(self.config_template_dir, self.client_id)
            if found:
                existing_config = _load_yaml_safe(found)
                result.client_config_path = str(found)

        # 3. Detect asset class
        profile_with_signals = dict(profile)
        detected_ac, detected_conf = _detect_asset_class(profile, existing_config)
        result.detected_asset_class = detected_ac
        result.detected_asset_confidence = detected_conf
        profile_with_signals["detected_asset_class"] = detected_ac
        profile_with_signals["detected_asset_confidence"] = detected_conf

        logger.info("[ConfigBootstrap] Asset class: %s (conf=%.2f)", detected_ac, detected_conf)

        # 4. Load asset config template
        asset_config: Optional[Dict[str, Any]] = None
        if detected_ac and detected_ac != "unknown":
            asset_conf_path = _find_asset_config(self.asset_template_dir, detected_ac)
            if asset_conf_path:
                asset_config = _load_yaml_safe(asset_conf_path)
                result.asset_config_path = str(asset_conf_path)

        # 5. Determine regime
        selected_regime, regime_conf = self._detect_regime(existing_config, detected_ac)
        result.selected_regime = selected_regime
        result.selected_regime_confidence = regime_conf

        # 6. Merge configs: asset defaults < client config < detected signals
        merged: Dict[str, Any] = {}
        if asset_config:
            merged = _deep_merge(merged, asset_config)
            for k, v in _flatten_config(asset_config).items():
                result.default_values_applied.append({
                    "field": k, "value": v, "source": "asset_template", "confidence": 0.9
                })
        if existing_config:
            merged = _deep_merge(merged, existing_config)
        if detected_ac and detected_ac != "unknown":
            merged.setdefault("portfolio", {})["asset_class"] = detected_ac
        if selected_regime:
            merged.setdefault("default_regime", selected_regime)
            merged.setdefault("regime", selected_regime)

        # 7. Apply questionnaire answers if provided
        if self.questionnaire_answers_path and self.questionnaire_answers_path.exists():
            answers = self._load_answers(self.questionnaire_answers_path)
            merged = self._apply_answers(merged, answers)
            result.user_answers_path = str(self.questionnaire_answers_path)
            logger.info("[ConfigBootstrap] Applied %d questionnaire answers.", len(answers))

        # 8. Generate config questions for missing values
        questions, missing_critical = _build_deterministic_questions(merged, profile_with_signals)

        # Optionally improve wording via cheap LLM
        if self.llm_enabled and questions and self.api_key:
            context = {
                "tape_file_name": self.tape_path.name,
                "detected_asset_class": detected_ac,
                "selected_regime": selected_regime,
            }
            questions = _apply_llm_question_wording(
                questions, context, self.llm_model, self.api_key
            )

        result.config_questions = questions
        result.missing_critical_config = missing_critical

        # 9. Write draft config
        draft_path = self.output_dir / f"{self.run_id}_draft_config.yaml"
        draft_path.write_text(
            yaml.dump(merged, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        result.draft_config_path = str(draft_path)

        # 10. Determine status
        blocking_missing = [q for q in questions if q.get("blocking")]
        result.approval_required = bool(questions)

        if blocking_missing:
            result.status = "blocked"
            result.proceed = False
        elif questions:
            result.status = "review_required"
            result.proceed = False
            # If an existing approved config covers everything, mark approved
            if existing_config and not missing_critical:
                result.status = "approved"
                result.approved_config_path = str(draft_path)
                result.proceed = True
                result.approval_required = False
        else:
            # All critical values present
            result.status = "approved"
            result.approved_config_path = str(draft_path)
            result.proceed = True
            result.approval_required = False

        # 11. Write result JSON
        result_path = self.output_dir / f"{self.run_id}_config_bootstrap.json"
        result.to_json(result_path)
        logger.info("[ConfigBootstrap] Result: %s  proceed=%s", result.status, result.proceed)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_regime(
        existing_config: Optional[Dict[str, Any]],
        asset_class: str,
    ) -> Tuple[str, float]:
        """Select regime from config, then asset-class heuristic."""
        if existing_config:
            r = (
                existing_config.get("default_regime")
                or existing_config.get("regime")
                or ""
            )
            if r:
                return str(r).strip(), 1.0

        # Asset-class heuristic defaults
        _AC_REGIME: Dict[str, str] = {
            "equity_release": "ESMA_Annex2",
            "rre": "ESMA_Annex2",
            "sme": "ESMA_Annex3",
            "cre": "ESMA_Annex3",
            "auto": "ESMA_Annex8",
            "consumer": "ESMA_Annex8",
        }
        if asset_class in _AC_REGIME:
            return _AC_REGIME[asset_class], 0.7

        return "ESMA_Annex2", 0.3   # safe default

    @staticmethod
    def _load_answers(path: Path) -> Dict[str, Any]:
        """Load questionnaire answers JSON → {question_id: answer_value}."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return {item["question_id"]: item.get("answer") for item in data
                        if "question_id" in item}
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Could not load questionnaire answers: %s", exc)
        return {}

    @staticmethod
    def _apply_answers(
        merged: Dict[str, Any],
        answers: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply questionnaire answers to the merged config dict.
        question_id format: q_portfolio_asset_class → portfolio.asset_class
        """
        def _nested_set(d: Dict[str, Any], dotted_key: str, val: Any) -> None:
            parts = dotted_key.split(".")
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = val

        for question_id, answer in answers.items():
            if answer is None:
                continue
            # Convert q_portfolio_asset_class → portfolio.asset_class
            config_key = question_id.removeprefix("q_").replace("_", ".", 1)
            # try exact match against known keys first
            if config_key in _CRITICAL_CONFIG_FIELDS:
                _nested_set(merged, config_key, answer)
            else:
                # fallback: best-effort dot-path
                _nested_set(merged, question_id.removeprefix("q_").replace("_", "."), answer)
        return merged


def _flatten_config(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict to dot-key pairs."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_config(v, full))
        else:
            out[full] = v
    return out
