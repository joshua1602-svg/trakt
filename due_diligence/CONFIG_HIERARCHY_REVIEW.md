# Configuration Hierarchy & Architecture Review (Pre-Client 1)

## Executive summary

The repository has a strong base (registry-led pipeline, asset/regime/client config folders), but is currently in a mixed state: several ERM/ERE assumptions are still embedded in runtime defaults and dashboard wiring. It is not a full rebuild away from the intended hierarchy; this is a boundary-hardening and config-composition refactor.

## Key ERM/ERE-specific code shape

1. **Core defaults are ERM-biased**
   - `--portfolio-type` defaults to `equity_release`.
   - `--master-config` defaults to `config/client/config_client_ERM_UK.yaml`.

2. **Dashboard is ERM-specific entrypoint**
   - App file is `analytics/streamlit_app_erm.py`.
   - Client config loading is hardwired to `config_client_ERM_UK.yaml` candidate paths.

3. **Chart layering is partial**
   - Static pools uses `config/asset/static_pools_config_erm.yaml`.
   - Other chart sections remain mostly script-defined; no client chart override layer.

4. **Risk logic is coupled to current client/region assumptions**
   - `risk_monitor.py` imports `config.client.risk_limits_config` directly.
   - UK region mappings are code constants.

5. **Client constants/date drift risk**
   - Master config static reporting date (`2025-11-30`) and annex12 period date (`2025-10-31`) differ.

## Alignment to intended hierarchy (status)

- Platform config: **Partial**
- Field registry: **Present / strong**
- Standard chart pack: **Missing/partial**
- Asset config: **Present**
- Asset chart pack: **Partial**
- Regime config: **Present**
- Client config: **Present**
- Client chart overrides: **Missing**
- Resolved runtime config artifact: **Missing (manifest exists, not full merged config)**

## Is onboarding Client 1 safe yet?

**Borderline**. The platform can run, but anti-drift boundaries should be tightened before onboarding to avoid bespoke forks.

## Must-fix before Client 1

1. Introduce deterministic config composition order: `platform -> asset -> regime -> client`.
2. Remove ERM/client hardcoded defaults from core entrypoints.
3. Define chart layering: `standard_chart_pack -> asset_chart_pack -> client_chart_overrides`.
4. Consolidate deterministic constants/date ownership (remove duplicated conflicting values).

## Should fix soon after

1. Move client risk limits from Python module to declarative client config.
2. Add ownership-lint checks so canonical field semantics are only defined in field registry.
3. Add resolved config artifact (`out/resolved_config.yaml`) and include its hash/path in run manifest.

## Effort estimate

This is **not a material reconstruction**.

Estimated effort for a clean baseline hierarchy:
- **2–4 engineering weeks** (single team, focused) for resolver + boundary cleanup + chart layering model.
- Additional time depends on how many legacy dashboard paths must be brought under config.

