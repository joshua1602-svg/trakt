"""operations_control.occ_agent.base_mi_gate — what stops a management-
information run, as the product profile already decides it.

THE DEFECT THIS CLOSES. A live equity release onboarding was blocked on eight
required fields, among them ``maturity_date`` — and a lifetime mortgage has no
contractual maturity date. The client's own files, loaded through the
platform's ingestion route, were not blocked at all, because nothing there
gates on materiality: that gate belongs to the Agent alone, and the Agent
stopped on every BLOCKING finding without ever asking what the product needs.

``config/asset/product_profiles.yaml`` has always answered this, per field, per
product. For the equity release lifetime mortgage profile:

    maturity_date                       base_mi: not_applicable
    amortisation_type                   base_mi: defaulted
    interest_rate_type                  base_mi: defaulted
    originator_legal_entity_identifier  base_mi: optional
    originator_name                     base_mi: optional
    current_principal_balance           base_mi: required
    exposure_currency_denomination      base_mi: required
    data_cut_off_date                   base_mi: required

Five excused, three required. Which is the right answer, and not one this
module is entitled to reach on its own — it reads the profile and applies it.

TWO THINGS IT DELIBERATELY DOES NOT DO.

It does not excuse anything on a REGULATORY run. ``base_mi`` says what
management information needs; the Annex 2 return needs more, and a field the
profile marks optional for MI can still be mandatory for the regulator. So the
excuse applies only when no regime is being prepared.

It does not hide the finding. An excused field is still reported, still on the
record, and still visible to whoever approves the run — it simply stops being
a reason to refuse the delivery. "Not applicable to this product" is an answer,
not an absence, and an operator who cannot see it cannot question it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: Declared here rather than imported from :mod:`.execution`, which imports
#: THIS module — the import would be circular.
_REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"

#: Reported against a finding the profile excuses, so the reason travels with
#: it rather than being inferable only from the absence of a blocker.
EXCUSED_NOTE = "not required for management information on this product"


def resolve(asset_class: str, confirmed_profile_id: str = ""):
    """The platform's own profile resolution for this run.

    Returns the ``ResolvedProfile``, or None where the configuration could not
    be read at all. Resolved through ``resolve_product_profile`` rather than by
    looking a profile id up directly: which profile applies is a governed
    decision with its own evidence and confidence thresholds, and a second way
    of answering it is a second answer waiting to disagree.

    ON ASSET CLASS ALONE THE PLATFORM PROPOSES RATHER THAN APPLIES. Equity
    release scores 0.6, inside the confirm band, so the profile is offered for
    confirmation and nothing is relaxed on the strength of it. That guard is
    correct and is not worked around here: an operator CONFIRMS the product,
    and the confirmed id comes back as ``confirmed_profile_id``, which the
    resolver then treats as explicit and trusted.
    """
    try:
        from engine.onboarding_agent.product_profile import (
            resolve_product_profile,
        )
    except Exception:                      # pragma: no cover — import guard
        return None
    try:
        return resolve_product_profile(
            {"asset_class": asset_class or ""},
            explicit_profile_id=confirmed_profile_id or "")
    except Exception:                      # pragma: no cover — config guard
        return None


def _profile_for(asset_class: str, confirmed_profile_id: str = ""):
    """The applied profile, or None when nothing has been confirmed.

    Only an APPLIED profile excuses anything. A profile merely proposed for
    confirmation has not been accepted by anyone, and acting on it would be
    relaxing a check on the strength of a guess.
    """
    resolved = resolve(asset_class, confirmed_profile_id)
    if resolved is None or not getattr(resolved, "applied", False):
        return None
    return getattr(resolved, "profile", None)


def needs_confirmation(asset_class: str, confirmed_profile_id: str = ""):
    """The profile awaiting an operator's confirmation, or None.

    This is the question that has to be put BEFORE anything can be excused:
    "is this client's book a lifetime mortgage?" Until it is answered, every
    required field keeps blocking — which is the safe direction, and is what
    the platform's own confidence bands ask for.
    """
    resolved = resolve(asset_class, confirmed_profile_id)
    if resolved is None or getattr(resolved, "applied", False):
        return None
    if not getattr(resolved, "needs_confirmation", False):
        return None
    return resolved


def asset_supplies(field: str, *, asset_class: str) -> str:
    """Where the ASSET PACK would get this field from, or ``""`` if nowhere.

    ``config/asset/product_defaults_ERM.yaml`` answers for a lifetime mortgage
    what the tape does not have to: ``maturity_date: ND5`` because there is no
    fixed term, ``amortisation_type: Bullet`` because an ERM rolls up,
    ``exposure_currency_denomination: GBP`` because the denomination is the
    client's reporting currency and not a per-loan fact. The projection agent
    reads exactly these at Gate 4.

    So a field the asset pack answers is NOT outstanding for the regime, and
    reporting it as though the lender still owed us the data would be asking
    them for something the platform already knows.
    """
    try:
        import yaml as _yaml
        from pathlib import Path as _Path
        root = _Path(__file__).resolve().parents[2]
        name = str(asset_class or "").strip().lower()
        pack = root / "config" / "asset" / "product_defaults_ERM.yaml"
        if name not in ("equity_release", "erm", "rre", "equity release"):
            return ""
        if not pack.exists():
            return ""
        cfg = _yaml.safe_load(pack.read_text(encoding="utf-8")) or {}
    except Exception:                        # pragma: no cover — config guard
        return ""
    nd = (cfg.get("nd_defaults") or {})
    if field in nd:
        return f"the asset pack answers it as {nd[field]}"
    static = (cfg.get("defaults") or {})
    if field in static:
        return f"the asset pack defaults it to {static[field]}"
    return ""


def client_supplies(field: str, *, client_defaults: Optional[Dict[str, Any]]
                    ) -> str:
    """Where the CLIENT CONFIGURATION would get this field from.

    ``config/regime/onboarding_standing_fields.yaml`` declares the originator's
    name, LEI and country of establishment as ``standing_client`` fields,
    captured once at onboarding and written to client config. They are not
    columns in a monthly extract, and blocking a loan tape for want of them
    asks the client to restate per loan what they told us once.
    """
    value = (client_defaults or {}).get(field)
    if value in (None, "", []):
        return ""
    return "the client configuration holds it"


def excused_fields(fields: List[str], *, asset_class: str,
                   regime: str = "",
                   confirmed_profile_id: str = "") -> Dict[str, str]:
    """``{field: why}`` for fields this product does not need for base MI.

    THE REGIME NO LONGER EMPTIES THIS. It used to: ``base_mi`` speaks for
    management information, so on a regulatory run nothing was excused and the
    delivery stopped on every field Annex 2 wants. That conflated two verdicts.

        "It should be the case the Operator invokes an MI + Regime run AND
         that MI can run without Regime being fully validated."

    Which is right, and is how the pipeline is already built: the handoff
    manifest carries ``ready_for_transformation_validation`` and
    ``ready_for_projection`` as two separate flags. A field Annex 2 needs and
    base MI does not has nothing to do with whether the MI is sound — and
    holding the MI for it means a lender waiting on one LEI cannot see their
    own book. What the regime still wants is reported separately, by
    :func:`regime_outstanding`, and holds the REGIME back rather than the run.
    """
    profile = _profile_for(asset_class, confirmed_profile_id)
    if profile is None:
        return {}
    out: Dict[str, str] = {}
    for field in fields:
        try:
            if profile.is_non_blocking_for_base_mi(field):
                out[field] = str(profile.base_mi_policy(field))
        except Exception:                  # pragma: no cover — profile guard
            continue
    return out


def split(findings: List[Dict[str, Any]], *, asset_class: str,
          regime: str = "", confirmed_profile_id: str = ""
          ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(blocking, excused)`` — what actually stops this run, and what does not.

    ``findings`` are the aggregated validation rows. Only rows the aggregator
    already called BLOCKING are considered; nothing here promotes a finding.
    Each excused row is annotated with the policy that excused it, so the
    reason is carried on the record rather than left to be inferred.
    """
    blocking = [r for r in (findings or [])
                if str(r.get("materiality", "")).upper() == "BLOCKING"]
    if not blocking:
        return [], []

    excuses = excused_fields([str(r.get("field_name") or "") for r in blocking],
                             asset_class=asset_class, regime=regime,
                             confirmed_profile_id=confirmed_profile_id)
    if not excuses:
        return blocking, []

    kept: List[Dict[str, Any]] = []
    let_through: List[Dict[str, Any]] = []
    for row in blocking:
        policy = excuses.get(str(row.get("field_name") or ""))
        if policy:
            annotated = dict(row)
            annotated["base_mi_policy"] = policy
            annotated["excused_reason"] = EXCUSED_NOTE
            let_through.append(annotated)
        else:
            kept.append(row)
    return kept, let_through


def regime_outstanding(excused: List[Dict[str, Any]], *, regime: str,
                       asset_class: str,
                       client_defaults: Optional[Dict[str, Any]] = None,
                       registry_path: Path = REGISTRY_PATH
                       ) -> List[Dict[str, Any]]:
    """What the REGULATORY return still wants, once the config has been asked.

    Excused for management information is not the same as answered for the
    regulator, and the difference is the whole reason the two verdicts are
    separate. But a field is only outstanding when NOBODY can supply it:

        "Any core_canonical: true fields that are not met for MI purposes must
         first consult the asset and client configuration to assess whether
         there are any rules. For example, maturity date is not relevant for an
         equity release portfolio."

    So each field is put to the asset pack and then to the client
    configuration before it is called outstanding. ``maturity_date`` is
    answered — ``ND5``, no fixed term — and never reaches the operator as
    something the lender owes us. An originator LEI is not: RREL83 permits no
    ND code and must match GLEIF, so it is a real ask, and it is ONE value in
    client config rather than a column in every monthly extract.

    Returns the rows that genuinely hold the regime back, each carrying the
    regime code it answers so the ask can be put to the lender in their terms.
    """
    if not regime:
        return []
    fields = _registry_fields(registry_path)
    out: List[Dict[str, Any]] = []
    for row in excused:
        field = str(row.get("field_name") or "")
        codes = ((fields.get(field) or {}).get("regime_mapping") or {})
        mapping = codes.get(regime) or {}
        if str(mapping.get("priority") or "").lower() not in ("mandatory",):
            continue
        supplied = (asset_supplies(field, asset_class=asset_class)
                    or client_supplies(field, client_defaults=client_defaults))
        if supplied:
            continue
        pending = dict(row)
        pending["regime_code"] = str(mapping.get("code") or "")
        pending["regime"] = regime
        out.append(pending)
    return out


def _registry_fields(registry_path: Path) -> Dict[str, Any]:
    try:
        from engine.gate_1_alignment.semantic_alignment import (
            load_field_registry,
        )
        return (load_field_registry(Path(registry_path)).get("fields") or {})
    except Exception:                        # pragma: no cover — config guard
        return {}


def regime_sentence(row: Dict[str, Any]) -> str:
    """One outstanding regulatory field, in words an operator can pass on."""
    field = str(row.get("field_name") or "").replace("_", " ")
    code = str(row.get("regime_code") or "")
    return (f"{field}{f' ({code})' if code else ''}: needed for "
            f"{str(row.get('regime') or 'the regulatory return')}, and neither "
            "the asset pack nor the client configuration supplies it.")


def sentence(row: Dict[str, Any]) -> str:
    """One excused finding, in words an approver can weigh.

    Deliberately says what the product profile decided, not merely that
    something was skipped: an approver signing off a run is entitled to see
    which governed answer let a required field through.
    """
    field = str(row.get("field_name") or "").replace("_", " ")
    policy = str(row.get("base_mi_policy") or "").replace("_", " ")
    return (f"{field}: {policy} for this product, so it does not hold up "
            "management information. It is still required for the "
            "regulatory return.")


def confirmation_decision(resolved: Any, blocked_fields: List[str]
                          ) -> Dict[str, Any]:
    """The product question, as a CARD an operator can actually answer.

    Answering it is what lets the profile excuse anything, so it is put in
    front of the blockers it would clear rather than buried beside them. The
    fields it WOULD excuse are named, because "confirm the product" with no
    consequence attached is a question nobody can weigh.

    WHY THIS IS A CARD AND NOT AN ARTEFACT ROW. There are two decision shapes
    in this system. The RAW row — ``decision_type`` and ``issue`` at the top
    level, ``status: "pending"`` — is what a stage writes into
    ``34_target_first_decisions.yaml``; ``_decisions_from_run`` then reads that
    artefact and converts it into the operator-facing CARD that
    ``run.open_decisions`` holds. This decision is raised by the gate rather
    than by the artefact, and was prepended to the list AFTER that conversion:

        decisions = self._decisions_from_run(run, facts, run_root)
        if adapters.product_profile_decision is not None:
            decisions = [adapters.product_profile_decision, *decisions]

    So it landed in a list of cards wearing the wrong shape, and was invisible
    twice over. Every reader treats a MISSING status as open — ``d.get("status",
    "open")`` — and this was the one decision that set one explicitly, to
    ``"pending"``, which is not ``"open"``. So the screen's ``d.status ===
    "open"`` filtered it out, ``open_now`` did not count it, and
    ``blocking_decisions`` did not see it. And had it rendered, the card does
    ``decision.evidence.map(...)`` and ``decision.options.map(...)`` on arrays
    this dict did not carry, which throws and takes the whole panel down.

    The result was a live case showing seven blocking fields, no question to
    answer, and no way to confirm the product that would have cleared six of
    them. The raw keys are kept beside the card ones because
    ``resolve_decision`` reads ``decision_type`` and ``profile_id`` from the
    top level to record the confirmation.
    """
    profile_id = str(getattr(resolved, "profile_id", ""))
    profile = getattr(resolved, "profile", None)
    would_clear = sorted(
        f for f in blocked_fields
        if profile is not None and profile.is_non_blocking_for_base_mi(f))
    label = profile_id.replace("_", " ")
    article = "an" if label[:1].lower() in "aeiou" else "a"
    consequence = (
        f"Confirming it means {len(would_clear)} required field"
        f"{'s' if len(would_clear) != 1 else ''} "
        f"({', '.join(f.replace('_', ' ') for f in would_clear)}) "
        "are not needed for management information on this product. They "
        "remain required for the regulatory return."
        if would_clear else
        f"Confirming it records the product as {article} {label}.")
    rationale = str(getattr(resolved, "rationale", ""))
    return {
        "decision_id": "product_profile",
        "decision_type": "product_confirmation",
        "target_field": "",
        "source_column": "",
        # NO EXPLICIT STATUS. Every reader in the system treats a missing one
        # as open — `d.get("status", "open")` — and the one decision that set
        # it explicitly was the one nobody could see.
        "blocking": True,
        "kind": "product_confirmation",
        "title": f"Is this book {article} {label}?",
        "question": ("Confirm the product so Trakt can apply what the asset "
                     "pack already knows about it."),
        "issue": (f"{len(blocked_fields)} required field"
                  f"{'s are' if len(blocked_fields) != 1 else ' is'} missing, "
                  "and nothing can be excused until the product is confirmed. "
                  "On the asset class alone Trakt proposes a profile rather "
                  "than applying it."),
        "evidence": [{"label": "Why Trakt thinks so", "kind": "text",
                      "data": {"issue": rationale or
                               f"the asset class resolves to {label}"}}],
        "recommendation": profile_id,
        "recommendation_source": "engine.onboarding_agent.product_profile",
        "materiality": "BLOCKING",
        "downstream_consequence": consequence,
        # The alternative is named rather than free-typed: a profile id the
        # resolver does not know is not an answer to this question.
        "options": [{"value": profile_id,
                     "label": f"Yes — {label}"}],
        "recommended_action": "confirm_product",
        "available_actions": ["confirm_product", "choose_alternative"],
        "confidence": round(float(getattr(resolved, "confidence", 0.0) or 0.0),
                            4),
        "profile_id": profile_id,
        "evidence_summary": rationale,
        "would_clear": would_clear,
        "proposed_mapping": consequence,
        "subject": {
            "artefact": "product_profile",
            "decision_id": "product_profile",
            "decision_type": "product_confirmation",
            "profile_id": profile_id,
        },
    }
