"""MI Agent API — a thin FastAPI layer over the existing MI Agent.

This package does NOT introduce any new analytics or semantic model. It wraps
``mi_agent.mi_agent_workflow.run_mi_agent_query`` and projects the real
``mi_semantics_field_registry.yaml`` + ``MIQuerySpec`` enums to the React UI.
"""
