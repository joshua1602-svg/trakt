"""Trakt Blob Trigger app — routes raw uploads to the Orchestrator Agent.

Thin Azure Functions app: parse path → schema fingerprint → source registry
inference → source-onboarding vs deterministic decision → invoke Orchestrator →
event manifest. Importing this package does not require azure-functions (the
Azure binding lives in function_app.py).
"""
from .router import handle_blob_event  # noqa: F401
from .source_registry import SourceRegistry, SourceRecord  # noqa: F401
