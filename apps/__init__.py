"""Top-level ``apps`` package.

Present so the root Azure Functions entrypoint can import the blob-trigger app
via ``from apps.blob_trigger_app.function_app import app`` (regular package
import, not namespace-package resolution).
"""
