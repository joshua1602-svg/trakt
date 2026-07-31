/**
 * Build-time feature flags.
 *
 * The OCC Agent tab is a pre-scale capability, so it is off unless the
 * environment explicitly turns it on. The flag controls VISIBILITY only — it is
 * not a security boundary. The backend reads the same flag name from its own
 * environment and does not mount the routes without it, so hiding the tab and
 * refusing the API are two independent decisions and neither depends on the
 * browser.
 *
 * Vite inlines `import.meta.env.*` at build time, so the deployed bundle
 * contains the literal value; nothing is read from the network at runtime.
 */

/** The environment variable name, shared with the backend. */
export const OCC_AGENT_FLAG = "OCC_AGENT_SYNTHETIC_ENABLED";

const TRUTHY = new Set(["1", "true", "yes", "on", "enabled"]);

/**
 * Whether the OCC Agent tab is offered.
 *
 * Fails closed: an unset, empty or unrecognised value is off.
 */
export function occAgentEnabled(): boolean {
  const raw = import.meta.env.VITE_OCC_AGENT_SYNTHETIC_ENABLED;
  return TRUTHY.has(String(raw ?? "").trim().toLowerCase());
}
