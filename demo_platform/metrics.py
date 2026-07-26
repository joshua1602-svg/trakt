"""demo_platform.metrics — export the ONE set of figures the film may show.

SYNTHETIC DEMONSTRATION DATA — NOT A REAL CUSTOMER.

Every number in the video comes from here, and everything here comes from the
product's own deterministic execution against the assembled platform canonicals:

  * ``mi_agent_api.evolution.funded_evolution``  — per-period governed metrics;
  * ``mi_agent_api.evolution.funded_bridge``     — regional attribution whose
    per-category deltas sum exactly to the net change;
  * ``mi_agent_api.movement_summary``            — the composed summary and
    month-on-month movement services;
  * ``POST /mi/query``                           — the exact React MI Agent
    envelopes, captured verbatim;
  * ``POST /v1/copilot/mi/query`` and
    ``GET /v1/copilot/artifacts/latest/investor-deck`` — the exact Microsoft 365
    Copilot action responses, captured verbatim.

The Remotion project reads the exported JSON. It never recomputes anything, so
the dashboard tiles, the MI Agent answer, the Copilot answer and the investor
artefact cannot disagree: there is one source.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from . import config as cfg

warnings.simplefilter("ignore")


class MetricsError(RuntimeError):
    """Raised when the demo metrics cannot be exported."""


def _apply_env() -> None:
    """Point this process's MI Agent API at the demo platform."""
    os.environ.update(cfg.mi_env())


def _client():
    """A TestClient over the real FastAPI app — the same handlers production serves."""
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    return TestClient(app)


def _portfolio_id(period_role: str = "current") -> str:
    """The React portfolio selector value: ``{client_id}/{run_id}``."""
    return f"{cfg.CLIENT_ID}/{cfg.period(period_role).reporting_date}"


# --------------------------------------------------------------------------- #
# Governed metric export (direct service calls)
# --------------------------------------------------------------------------- #
def _evolution_series(scope: str) -> List[Dict[str, Any]]:
    from mi_agent_api import evolution as evolution_mod
    from mi_agent_api import movement_summary as movement_mod
    evo = evolution_mod.funded_evolution(cfg.platform_prefix(), scope)
    out: List[Dict[str, Any]] = []
    for p in evo.get("periods", []):
        m = p.get("metrics") or {}
        out.append({
            "period": p.get("period"),
            "reportingDate": p.get("reporting_date"),
            "loanCount": m.get("loan_count"),
            "fundedBalance": m.get("funded_balance"),
            "waCurrentLtvPoints": movement_mod._ltv_to_points(m.get("wa_ltv")),
            "waInterestRate": m.get("wa_interest_rate"),
            "avgBorrowerAge": m.get("avg_borrower_age"),
        })
    return out


def _sponsored_scope() -> Dict[str, Any]:
    """SPV1's own figures, from its own governed canonical.

    Read from the portfolio's published Gate 2 output rather than the platform
    canonical, because SPV1 is deliberately not in the platform canonical.
    """
    spec = cfg.PORTFOLIO_S
    canonical = (cfg.run_output_dir(spec.source_portfolio_id, cfg.CURRENT_PERIOD.run_id)
                 / f"{cfg.source_file(spec, cfg.CURRENT_PERIOD).stem}_canonical_typed.csv")
    if not canonical.exists():
        raise MetricsError(
            f"SPV1 has no governed canonical at {canonical} — run the orchestration first."
        )
    df = pd.read_csv(canonical, low_memory=False)
    balances = pd.to_numeric(df["current_outstanding_balance"], errors="coerce").fillna(0.0)
    return {
        "sourcePortfolioId": spec.source_portfolio_id,
        "displayId": spec.display_id,
        "label": spec.label,
        "status": spec.description,
        "fundedBalance": round(float(balances.sum()), 2),
        "funded_balance": round(float(balances.sum()), 2),
        "loanCount": int((balances > 0).sum()),
        "loan_count": int((balances > 0).sum()),
        "reportingDate": cfg.CURRENT_PERIOD.reporting_date,
        "producedBy": "engine/gate_2_transform/canonical_transform.py (SPV1 run)",
    }


def governed_metrics() -> Dict[str, Any]:
    """The full governed metric set, per lens, straight from the services."""
    from mi_agent_api import movement_summary as movement_mod

    root = cfg.platform_prefix()
    current_run = cfg.CURRENT_PERIOD.reporting_date

    lenses: Dict[str, Dict[str, Any]] = {}
    scopes = [("total", None, "Total", cfg.CLIENT_ID)]
    for p in cfg.PORTFOLIOS:
        scopes.append((p.source_portfolio_id,
                       {"source_portfolio_id": p.source_portfolio_id},
                       p.label, p.source_portfolio_id))

    for key, filters, label, scope in scopes:
        summary = movement_mod.portfolio_summary(
            root, scope, to_run_id=current_run,
            lens_filters=filters, lens_label=label)
        movement = movement_mod.period_movement(
            root, scope, to_run_id=current_run,
            lens_filters=filters, lens_label=label)
        if not summary.get("available") or not movement.get("available"):
            raise MetricsError(
                f"lens {key!r}: governed metrics unavailable "
                f"({summary.get('reason') or movement.get('reason')})."
            )
        lenses[key] = {
            "label": label,
            "summary": summary,
            "movement": movement,
            "series": _evolution_series(scope),
        }

    # --- Sponsor scope -----------------------------------------------------
    # SPV1 is governed through the same gates but is NOT in the platform canonical, so
    # its figures are read from its own published canonical and aggregated here — the
    # one place in the demonstration where the two scopes are added together.
    sponsored = _sponsored_scope()
    platform_summary = lenses["total"]["summary"]["metrics"]
    sponsor_balance = (platform_summary.get("funded_balance") or 0.0) + sponsored["funded_balance"]
    sponsor_loans = (platform_summary.get("loan_count") or 0) + sponsored["loan_count"]

    return {
        "synthetic_notice": cfg.SYNTHETIC_BANNER,
        "client": {"id": cfg.CLIENT_ID, "displayName": cfg.CLIENT_DISPLAY_NAME,
                   "shortName": cfg.CLIENT_SHORT_NAME},
        # Two scopes, and the film must never confuse them.
        #   PLATFORM — the warehoused books the assembler consolidates. Every figure in
        #              the film except the sponsor consolidation is this scope.
        #   SPONSOR  — the platform plus the deal that was sold. Used once, in S3.
        "scopes": {
            "platform": {
                "label": "Platform (warehoused)",
                "portfolios": [p.display_id for p in cfg.PORTFOLIOS],
                "fundedBalance": platform_summary.get("funded_balance"),
                "loanCount": platform_summary.get("loan_count"),
            },
            "sponsor": {
                "label": "Sponsor AUM",
                "portfolios": [p.display_id for p in cfg.ALL_PORTFOLIOS],
                "fundedBalance": round(sponsor_balance, 2),
                "loanCount": int(sponsor_loans),
            },
            "sponsored": sponsored,
        },
        "currentPeriod": {
            "period": cfg.CURRENT_PERIOD.period,
            "reportingDate": cfg.CURRENT_PERIOD.reporting_date,
            "label": cfg.CURRENT_PERIOD.label,
        },
        "priorPeriod": {
            "period": cfg.PRIOR_PERIOD.period,
            "reportingDate": cfg.PRIOR_PERIOD.reporting_date,
            "label": cfg.PRIOR_PERIOD.label,
        },
        "lenses": lenses,
    }


# --------------------------------------------------------------------------- #
# Surface capture: the exact envelopes each interface returns
# --------------------------------------------------------------------------- #
def _strip_internals(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Remove implementation detail the CLIENT-FACING UI must not display.

    The film shows a client surface, so the captured envelope keeps only what a
    client sees: the answer, the artifacts, the reconciliation and the source
    notes. The parsed query spec, validation internals, resolved-field maps and
    engine metadata are dropped here rather than being hidden in the video layer —
    if it is not in the fixture, it cannot leak onto screen.
    """
    keep = ("ok", "question", "answer", "artifacts", "warnings", "assumptions")
    out = {k: envelope.get(k) for k in keep if k in envelope}
    for art in out.get("artifacts") or []:
        if isinstance(art, dict):
            art.pop("source", None)      # carries the raw spec
            art.pop("id", None)          # internal artefact uid
            art.pop("createdAt", None)   # non-deterministic
            art.pop("mock", None)
    meta = envelope.get("metadata") or {}
    if isinstance(meta, dict):
        out["route"] = meta.get("route")
        out["datasetContext"] = meta.get("datasetContext")
        out["portfolioLens"] = meta.get("portfolioLens")
    recon = envelope.get("reconciliation")
    if recon:
        out["reconciliation"] = recon
    notes = envelope.get("sourceNotes") or envelope.get("source_notes")
    if notes:
        out["sourceNotes"] = notes
    return out


def capture_mi_agent(client) -> List[Dict[str, Any]]:
    """The React MI Agent conversation: both required questions, in order."""
    pid = _portfolio_id("current")
    captured: List[Dict[str, Any]] = []
    for question in cfg.MI_QUESTIONS:
        resp = client.post("/mi/query", json={
            "question": question,
            "portfolioId": pid,
            "asOfDate": cfg.CURRENT_PERIOD.reporting_date,
        })
        if resp.status_code != 200:
            raise MetricsError(
                f"MI Agent query {question!r} returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}")
        body = resp.json()
        if not body.get("ok"):
            raise MetricsError(f"MI Agent query {question!r} failed: {body.get('error')}")
        captured.append(_strip_internals(body))
    return captured


def capture_dashboard(client) -> Dict[str, Any]:
    """The deterministic dashboard state the MI Agent scene renders.

    Snapshot discovery (the portfolio + reporting-date selectors), the funded KPI
    snapshot, and the geographic exposure view — each from its own production
    endpoint, so the scene is a reconstruction of real state rather than a mock-up.
    """
    pid = _portfolio_id("current")
    out: Dict[str, Any] = {}

    snaps = client.get("/mi/snapshots")
    if snaps.status_code != 200:
        raise MetricsError(f"/mi/snapshots returned HTTP {snaps.status_code}.")
    out["snapshots"] = snaps.json()

    lenses = client.get("/mi/source-portfolios")
    out["sourcePortfolios"] = lenses.json() if lenses.status_code == 200 else {}

    snap = client.get("/mi/snapshot", params={"portfolioId": pid})
    if snap.status_code != 200:
        raise MetricsError(f"/mi/snapshot returned HTTP {snap.status_code}.")
    out["fundedSnapshot"] = snap.json()

    geo = client.get("/mi/geo/exposure", params={"portfolioId": pid})
    out["geoExposure"] = geo.json() if geo.status_code == 200 else {}

    evo = client.get("/mi/evolution/funded", params={"portfolioId": pid})
    out["fundedEvolution"] = evo.json() if evo.status_code == 200 else {}

    health = client.get("/health")
    if health.status_code == 200:
        h = health.json()
        # /health carries operator diagnostics; keep only what the client UI shows.
        out["dataSource"] = {
            "kind": h.get("dataSourceKind"),
            "name": h.get("dataSource"),
            "dimensionsAvailable": h.get("dimensionsAvailable"),
        }
    return out


def capture_copilot(client) -> Dict[str, Any]:
    """The Microsoft 365 Copilot action responses.

    ``askTraktMi`` delegates to the same ``/mi/query`` handler the React agent
    uses, so the values agree by construction. The Copilot action layer requires a
    validated Entra ID bearer token in production; there is no tenant in the demo
    environment, so the underlying handler is invoked directly here and the
    response is shaped exactly as ``copilot_actions`` shapes it. What the film
    shows is therefore the real action contract, captured without a live tenant.
    """
    from mi_agent_api import copilot_actions as copilot_mod

    answers: List[Dict[str, Any]] = []
    for question in cfg.MI_QUESTIONS:
        req = copilot_mod.CopilotMiQueryRequest(
            question=question,
            portfolioId=_portfolio_id("current"),
            asOfDate=cfg.CURRENT_PERIOD.reporting_date,
        )
        result = copilot_mod.ask_trakt_mi(req)
        payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        answers.append(payload)

    deck = _copilot_deck_action()
    return {
        "agentName": "Trakt",
        "actions": ["askTraktMi", "getLatestInvestorDeck", "getLatestCanonicalTape"],
        "answers": answers,
        "artefactAction": deck,
        "artefactQuestion": cfg.COPILOT_ARTEFACT_QUESTION,
    }


def _copilot_deck_action() -> Optional[Dict[str, Any]]:
    """``getLatestInvestorDeck``, with the signed download URL redacted.

    The action returns a short-lived HMAC-signed download link. A signed URL must
    never reach a video frame, so the token is replaced with a fixed placeholder
    while the metadata the film actually shows (file name, reporting period, size)
    is kept exactly as the action returns it.
    """
    from mi_agent_api import decks as decks_mod
    listing = decks_mod.list_decks(cfg.CLIENT_ID) if hasattr(decks_mod, "list_decks") else None
    if not listing:
        return None
    resolved = listing if isinstance(listing, dict) else {}
    decks = resolved.get("decks") or resolved.get("periods") or []
    latest = resolved.get("latest") or (decks[0] if decks else None)
    if not latest:
        return None
    entry = latest if isinstance(latest, dict) else {"period": str(latest)}
    return {
        "fileName": entry.get("fileName") or decks_mod.DECK_NAME,
        "reportingPeriod": entry.get("period") or cfg.CURRENT_PERIOD.period,
        "sizeBytes": entry.get("sizeBytes"),
        "downloadUrl": "https://<trakt-api>/v1/copilot/artifacts/download?token=<redacted>",
        "downloadUrlNote": "Short-lived HMAC-signed link. Redacted in the "
                           "demonstration fixture — a signed URL must never appear "
                           "on screen.",
        "expiresInSeconds": 300,
        "listing": resolved,
    }


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export(*, verbose: bool = True) -> Dict[str, Any]:
    """Compute and write the complete demo metric + surface export."""
    _apply_env()
    # The data source caches the loaded canonical; clear it so an export always
    # reflects the platform canonical this run just published.
    try:
        from mi_agent_api import data_source
        data_source.get_dataframe.cache_clear()
    except Exception:  # noqa: BLE001 - a missing cache is not an error
        pass

    client = _client()
    metrics = governed_metrics()
    dashboard = capture_dashboard(client)
    mi_agent = capture_mi_agent(client)
    copilot = capture_copilot(client)

    export_payload = {
        "synthetic_notice": cfg.SYNTHETIC_BANNER,
        "narrative": cfg.as_dict(),
        "schemas": None,   # filled by run_demo (avoids importing schemas here)
        "metrics": metrics,
        "dashboard": dashboard,
        "miAgent": {"questions": list(cfg.MI_QUESTIONS), "turns": mi_agent},
        "copilot": copilot,
    }

    out_dir = cfg.export_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "demo_metrics.json").write_text(
        json.dumps(export_payload, indent=2, default=str), encoding="utf-8")

    if verbose:
        total = metrics["lenses"]["total"]
        mv = total["movement"]
        print(f"  [metrics] consolidated movement "
              f"£{(mv['delta']['funded_balance'] or 0) / 1e6:,.3f}m; "
              f"WA LTV {mv['current']['wa_ltv_points']:.3f}pp; "
              f"age {mv['current']['avg_borrower_age']:.3f}y; "
              f"primary region {(mv.get('primaryRegion') or {}).get('region')}")
        for turn in mi_agent:
            print(f"  [mi-agent] {turn['question']}\n             {turn['answer'][:160]}…")

    return export_payload


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse
    ap = argparse.ArgumentParser(description="Export the demo metrics + surfaces.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    export(verbose=not args.quiet)
    print(f"Wrote {cfg.export_dir() / 'demo_metrics.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
