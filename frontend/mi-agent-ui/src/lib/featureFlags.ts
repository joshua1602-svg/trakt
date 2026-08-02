/**
 * Phase 2A — build-time gate for the optional hover / drill-down layer.
 *
 * Two independent gates gate this feature: this one decides whether the UI
 * offers the affordance at all, and `TRAKT_MI_ENHANCED_HOVERS` on the API
 * decides whether the detail endpoint exists. Both default OFF.
 *
 * They are deliberately not coupled. If the UI is enabled against an API that
 * is not, the detail fetch 404s, the hook reports "unavailable" and the chart
 * keeps its existing tooltip — a mismatch degrades to current behaviour rather
 * than to an error. That is the property that makes the feature safe to enable
 * per environment.
 *
 * This is a build-time flag, so a bundle built without it contains no way to
 * reach the detail layer at all.
 */

/** The Vite env fields this module reads (injectable so tests need no bundler). */
export interface FeatureEnv {
  VITE_MI_ENHANCED_HOVERS?: string;
}

const TRUTHY = new Set(["1", "true", "on", "yes", "enabled"]);

/**
 * True only when this build explicitly opted in.
 *
 * Anything unrecognised — absent, empty, "maybe", "0" — leaves the feature off,
 * so a typo in a deployment variable can never half-enable it.
 */
export function enhancedHoversEnabled(
  env: FeatureEnv = import.meta.env as unknown as FeatureEnv,
): boolean {
  return TRUTHY.has(String(env?.VITE_MI_ENHANCED_HOVERS ?? "").trim().toLowerCase());
}
