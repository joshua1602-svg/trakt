"use client";

import { publicConfig } from "@/lib/public-config";

/**
 * Privacy-conscious analytics.
 *
 * What is captured: an event name, and a small set of low-cardinality
 * properties drawn from a fixed vocabulary (intent ids, capability ids, section
 * ids). What is never captured: the text a visitor types into the demo, their
 * lead-form contents, any identifier that persists across sessions, and any
 * third-party cookie. Demo questions are recorded as intent ids only — the
 * question string is not sent anywhere by this module.
 *
 * No third-party tracker is loaded by default. The active adapter is chosen by
 * `NEXT_PUBLIC_ANALYTICS_PROVIDER`; the default (`none`) is a no-op.
 */

export type AnalyticsEvent =
  | "hero_demo_click"
  | "demo_video_play"
  | "suggested_question_click"
  | "typed_question_submit"
  | "answer_supported"
  | "answer_unsupported"
  | "report_preview_request"
  | "session_limit_reached"
  | "demo_reset"
  | "capability_card_open"
  | "book_demo_click"
  | "lead_form_submit"
  | "lead_form_success"
  | "lead_form_error";

/** Only these property keys are ever transmitted. */
export interface AnalyticsProperties {
  intentId?: string;
  reportId?: string;
  capabilityId?: string;
  section?: string;
  source?: string;
  outcome?: string;
}

interface Adapter {
  track(event: AnalyticsEvent, properties?: AnalyticsProperties): void;
}

/** Development / default. Records nothing, anywhere. */
const noopAdapter: Adapter = {
  track() {
    /* intentionally empty */
  },
};

/**
 * Azure Application Insights. Uses the global the App Insights snippet defines
 * if — and only if — an operator has loaded it; this module never injects a
 * script tag, so enabling it is an explicit deployment decision.
 */
const appInsightsAdapter: Adapter = {
  track(event, properties) {
    const insights = (
      globalThis as unknown as {
        appInsights?: { trackEvent?: (item: unknown) => void };
      }
    ).appInsights;
    insights?.trackEvent?.({ name: `trakt.landing.${event}`, properties: properties ?? {} });
  },
};

/** First-party collector on this origin. Fire-and-forget, never blocking. */
const firstPartyAdapter: Adapter = {
  track(event, properties) {
    try {
      const body = JSON.stringify({ event, properties: properties ?? {} });
      if (typeof navigator !== "undefined" && "sendBeacon" in navigator) {
        navigator.sendBeacon("/api/analytics", new Blob([body], { type: "application/json" }));
        return;
      }
      void fetch("/api/analytics", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
        keepalive: true,
      }).catch(() => undefined);
    } catch {
      /* analytics must never break the page */
    }
  },
};

function adapter(): Adapter {
  switch (publicConfig.analyticsProvider) {
    case "appinsights":
      return appInsightsAdapter;
    case "firstparty":
      return firstPartyAdapter;
    default:
      return noopAdapter;
  }
}

export function track(event: AnalyticsEvent, properties?: AnalyticsProperties): void {
  if (typeof window === "undefined") return;
  adapter().track(event, properties);
}
