"""Top-level ``apps`` package.

Present so the root Azure Functions entrypoint can import the blob-trigger
modules (router, eventgrid, azure_io, …) via ``apps.blob_trigger_app.*`` as a
regular package, not namespace-package resolution.
"""
