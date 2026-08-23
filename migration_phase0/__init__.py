"""MI compositional migration — Phase 0 instruments and baseline.

READ-ONLY / SHADOW ONLY. Nothing in this package is imported by production code,
wired into a route, or placed behind a flag. Nothing here ships to a client.

    freeze_baseline.py            the attribution baseline -> BASELINE.json
    route_identity_inventory.py   where route IDENTITY decides governance
    shadow_portfolio_summary.py   the first migration slice, shadow only
    equivalence_portfolio_summary.py   shipped vs shadow, field by field
    probe_arity.py                which grouped paths disclose at which arity
    probe_arity2_defect.py        how many leaf groups the arity-2 case hides
    probe_interpretation_gap.py   what the contract supplies, and what it cannot

See docs/mi_migration_abort_conditions.md and docs/mi_phase0_report.md.
"""
