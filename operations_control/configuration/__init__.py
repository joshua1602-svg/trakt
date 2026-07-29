"""operations_control.configuration — OCC-owned configuration governance.

The OCC is the authoritative resolver of system, regime, asset, client and
portfolio configuration. Repository YAML files remain the storage format and
the seed of version 1; this package versions them (config packages), resolves
them with explicit precedence into one immutable EffectiveConfiguration per
run, and materialises hash-verified snapshots for the Onboarding Agent.
"""
