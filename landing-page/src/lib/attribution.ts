/**
 * Campaign attribution.
 *
 * Captured from the landing URL, held for the current browsing session only,
 * and attached to a lead notification server-side. It is deliberately **not**
 * echoed back to the browser after submission and never persisted beyond the
 * session, so it cannot become a cross-site identifier.
 *
 * Shared by client and server: the browser reads the query string into
 * `sessionStorage`, the API re-sanitises whatever it receives. The server is
 * the authority — a caller can put anything in the request body, so every
 * field is re-validated here before it reaches a notification.
 */

export const ATTRIBUTION_FIELDS = [
  "utm_source",
  "utm_medium",
  "utm_campaign",
  "utm_content",
  "utm_term",
  "persona",
  "use_case",
  "source",
] as const;

export type AttributionField = (typeof ATTRIBUTION_FIELDS)[number];
export type Attribution = Partial<Record<AttributionField, string>>;

/** Per-field ceiling. Campaign codes are short; anything longer is noise. */
export const MAX_ATTRIBUTION_LENGTH = 64;

/**
 * Allow-list rather than deny-list: letters, digits and a small set of
 * separators that real campaign codes use. Everything else is dropped, so no
 * markup, control character or delimiter can ride through into an email.
 */
const ALLOWED = /[^a-zA-Z0-9 ._\-+/:]/g;

export function sanitiseAttributionValue(raw: unknown): string | null {
  if (typeof raw !== "string") return null;
  // Disallowed characters become a space rather than vanishing, so a newline
  // injection collapses to separated words instead of running them together.
  const value = raw.replace(ALLOWED, " ").replace(/\s+/g, " ").trim();
  if (!value) return null;
  return value.slice(0, MAX_ATTRIBUTION_LENGTH);
}

/** Sanitise an arbitrary object into the allow-listed attribution shape. */
export function sanitiseAttribution(raw: unknown): Attribution {
  if (!raw || typeof raw !== "object") return {};
  const input = raw as Record<string, unknown>;
  const out: Attribution = {};
  for (const field of ATTRIBUTION_FIELDS) {
    const value = sanitiseAttributionValue(input[field]);
    if (value) out[field] = value;
  }
  return out;
}

/** Read attribution from a URL's query string. */
export function attributionFromSearch(search: string): Attribution {
  const params = new URLSearchParams(search);
  const out: Attribution = {};
  for (const field of ATTRIBUTION_FIELDS) {
    const value = sanitiseAttributionValue(params.get(field));
    if (value) out[field] = value;
  }
  return out;
}

export function hasAttribution(value: Attribution): boolean {
  return Object.keys(value).length > 0;
}

export const ATTRIBUTION_STORAGE_KEY = "trakt_attribution";
