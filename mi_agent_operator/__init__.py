"""mi_agent_operator — the standalone Operator approval console.

A SEPARATE surface from the client-facing MI dashboard (different app, different
deployment, its own auth). It exposes the human one-click approval queue — new
sources and material schema changes wait here with the AI-proposed mapping
pre-filled — plus an audit log of the auto-approvals that did NOT need a human.

Nothing here changes canonical / MI / ESMA logic; it wraps the existing approval
workflow (apps.blob_trigger_app.approvals / ops / persistence) behind a small,
authenticated API and a self-contained UI.
"""

from .service import OperatorService  # noqa: F401
