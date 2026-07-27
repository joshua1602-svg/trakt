/**
 * The analytics event contract, shared by the browser and the collector.
 *
 * The endpoint is an allow-list, not an open event sink: an unknown event name
 * is rejected rather than recorded. That keeps the collector from becoming a
 * general-purpose logging endpoint that anyone on the internet can write to.
 *
 * What is never carried, by construction rather than by policy: the text a
 * visitor typed, any answer content, any name or email address, and any
 * identifier that survives the session.
 */

export const ANALYTICS_EVENTS = [
  "hero_demo_click",
  "demo_open",
  "suggested_question_click",
  "typed_question_submit",
  "demo_answer_returned",
  "demo_refusal_returned",
  "report_preview_opened",
  "book_demo_click",
  "lead_submit_success",
  "lead_submit_failure",
] as const;

export type AnalyticsEvent = (typeof ANALYTICS_EVENTS)[number];

const EVENT_SET: ReadonlySet<string> = new Set(ANALYTICS_EVENTS);

export function isAnalyticsEvent(value: unknown): value is AnalyticsEvent {
  return typeof value === "string" && EVENT_SET.has(value);
}

/**
 * The only property keys that may be transmitted, each low-cardinality.
 *
 * `intentId` is an allow-listed identifier from the demo pack, and
 * `refusalCategory` is one of a fixed set — neither can carry free text,
 * because the browser never has anything else to put in them.
 */
export const ANALYTICS_PROPERTY_KEYS = [
  "intentId",
  "reportId",
  "refusalCategory",
  "section",
  "source",
  "outcome",
] as const;

export type AnalyticsPropertyKey = (typeof ANALYTICS_PROPERTY_KEYS)[number];
export type AnalyticsProperties = Partial<Record<AnalyticsPropertyKey, string>>;

/** Property values are identifiers, so this is deliberately tight. */
export const MAX_PROPERTY_LENGTH = 64;
const PROPERTY_ALLOWED = /[^a-zA-Z0-9_\-.]/g;

export function sanitiseProperties(raw: unknown): AnalyticsProperties {
  if (!raw || typeof raw !== "object") return {};
  const input = raw as Record<string, unknown>;
  const out: AnalyticsProperties = {};
  for (const key of ANALYTICS_PROPERTY_KEYS) {
    const value = input[key];
    if (typeof value !== "string") continue;
    const clean = value.replace(PROPERTY_ALLOWED, "").slice(0, MAX_PROPERTY_LENGTH);
    if (clean) out[key] = clean;
  }
  return out;
}

export interface AnalyticsPayload {
  event: AnalyticsEvent;
  properties: AnalyticsProperties;
}

/** Body ceiling for the collector. An event is a few dozen bytes. */
export const MAX_ANALYTICS_BODY_BYTES = 1024;
